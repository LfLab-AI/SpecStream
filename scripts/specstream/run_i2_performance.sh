#!/usr/bin/env bash
set -euo pipefail

# Minimal performance test for in-process SpecStream innovation 2.
# One fixed workload, four methods, repeated runs:
#   P0 native, P1 serial control, P2 h=4, P3 Auto.

TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL}"
DRAFT_MODEL="${DRAFT_MODEL:?Set DRAFT_MODEL}"
SHAREGPT_JSON="${SHAREGPT_JSON:?Set SHAREGPT_JSON}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-200000}"
INPUT_LEN="${INPUT_LEN:-16384}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
REPETITIONS="${REPETITIONS:-3}"
METHODS="${METHODS:-P0_NATIVE P1_SERIAL P2_H4 P3_AUTO}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/innovation2_inproc_minimal/performance_${RUN_STAMP}}"

export PYTHONPATH="${PYTHONPATH:-$PWD/python}"
export SGLANG_ENABLE_SPEC_V2=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

test -s "$SHAREGPT_JSON"
mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/bench" "$RESULT_ROOT/profiles"

echo "[precheck] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python - <<'PY'
import torch

print("torch =", torch.__version__)
print("torch CUDA =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("visible device count =", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        "No CUDA device is visible. Set GPU_ID to a valid logical GPU index "
        "(normally GPU_ID=0 inside the AutoDL container)."
    )
print("device 0 =", torch.cuda.get_device_name(0))
PY

BASE_ARGS=(
  python -m sglang.launch_server
  --model-path "$TARGET_MODEL"
  --port "$SERVER_PORT"
  --context-length "$CONTEXT_LENGTH"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --skip-server-warmup
  --tp-size 1
  --dp-size 1
  --page-size 1
  --attention-backend "$ATTENTION_BACKEND"
  --disable-radix-cache
  --speculative-algorithm STANDALONE
  --speculative-draft-model-path "$DRAFT_MODEL"
  --speculative-num-steps 4
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 5
)

SERVER_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

trap stop_server EXIT INT TERM

start_server() {
  local method="$1"
  local repetition="$2"
  local tag="${method}_16k_c${MAX_CONCURRENCY}_rep${repetition}"
  local log_dir="$RESULT_ROOT/logs/$tag"
  local profile="$RESULT_ROOT/profiles/${tag}.jsonl"
  local extra=()

  mkdir -p "$log_dir" "$RESULT_ROOT/bench/$method"
  case "$method" in
    P0_NATIVE)
      ;;
    P1_SERIAL)
      extra+=(
        --specstream-inproc-enabled
        --specstream-inproc-mode serial
        --specstream-inproc-profile-path "$profile"
      )
      ;;
    P2_H4)
      extra+=(
        --specstream-inproc-enabled
        --specstream-inproc-mode ahead-free
        --specstream-inproc-ahead-depth 4
        --specstream-inproc-profile-path "$profile"
        --specstream-inproc-profile-interval 128
      )
      ;;
    P3_AUTO)
      extra+=(
        --specstream-inproc-enabled
        --specstream-inproc-mode auto
        --specstream-inproc-ahead-depth 4
        --specstream-inproc-min-reuse-ratio 0.25
        --specstream-inproc-profile-path "$profile"
        --specstream-inproc-profile-interval 128
      )
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 2
      ;;
  esac

  printf '%q ' "${BASE_ARGS[@]}" "${extra[@]}" > "$log_dir/launch_command.txt"
  printf '\n' >> "$log_dir/launch_command.txt"
  "${BASE_ARGS[@]}" "${extra[@]}" > "$log_dir/server.log" 2>&1 &
  SERVER_PID=$!

  local ready=0
  for _ in $(seq 1 600); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$tag exited before readiness" >&2
      tail -200 "$log_dir/server.log" >&2
      return 3
    fi
    if curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "$tag readiness timeout" >&2
    tail -200 "$log_dir/server.log" >&2
    return 4
  fi
  curl -fsS "http://127.0.0.1:${SERVER_PORT}/server_info" \
    > "$log_dir/server_info.json"
}

for repetition in $(seq 1 "$REPETITIONS"); do
  for method in $METHODS; do
    tag="${method}_16k_c${MAX_CONCURRENCY}_rep${repetition}"
    log_dir="$RESULT_ROOT/logs/$tag"
    echo "[performance] $tag"
    start_server "$method" "$repetition"

    set +e
    set -o pipefail
    BASE_URL="http://127.0.0.1:${SERVER_PORT}" \
    TARGET_MODEL="$TARGET_MODEL" \
    CASE_TAG="$tag" \
    DATASET_NAME=random \
    DATASET_PATH="$SHAREGPT_JSON" \
    INPUT_LEN="$INPUT_LEN" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    REQUEST_RATE=inf \
    MAX_CONCURRENCY="$MAX_CONCURRENCY" \
    RANGE_RATIO=1 \
    WARMUP_REQUESTS=16 \
    SEED="$repetition" \
    CONTEXT_LEN="$CONTEXT_LENGTH" \
    RUN_ID="rep${repetition}" \
    OUTPUT_DIR="$RESULT_ROOT/bench/$method" \
    bash scripts/specstream/run_benchmark_case.sh \
      2>&1 | tee "$log_dir/benchmark.log"
    status=${PIPESTATUS[0]}
    set -e

    stop_server
    if [[ "$status" -ne 0 ]]; then
      echo "Benchmark failed: $tag" >&2
      exit "$status"
    fi
  done
done

python scripts/specstream/summarize_benchmarks.py \
  "$RESULT_ROOT"/bench/*/*.jsonl \
  > "$RESULT_ROOT/benchmark_summary.tsv"

python - "$RESULT_ROOT" "$METHODS" <<'PY'
import glob
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = sys.argv[2].split()
values = {}
for method in methods:
    rows = []
    for filename in sorted(glob.glob(str(root / "bench" / method / "*.jsonl"))):
        with open(filename, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise SystemExit(f"No benchmark rows for {method}")
    if any(row.get("errors") for row in rows):
        raise SystemExit(f"Request errors found for {method}")
    values[method] = {
        "runs": len(rows),
        "output_throughput": statistics.mean(float(row["output_throughput"]) for row in rows),
        "p99_ttft_ms": statistics.mean(float(row["p99_ttft_ms"]) for row in rows),
        "p99_tpot_ms": statistics.mean(float(row["p99_tpot_ms"]) for row in rows),
        "p99_e2e_latency_ms": statistics.mean(float(row["p99_e2e_latency_ms"]) for row in rows),
    }

baseline = values["P0_NATIVE"]["output_throughput"]
output_path = root / "performance_summary.tsv"
with output_path.open("w", encoding="utf-8") as output:
    output.write("method\truns\toutput_tok_s\tspeedup_vs_P0\tp99_ttft_ms\tp99_tpot_ms\tp99_e2e_ms\n")
    for method in methods:
        row = values[method]
        output.write(
            f"{method}\t{row['runs']}\t{row['output_throughput']:.4f}\t"
            f"{row['output_throughput'] / baseline:.4f}\t"
            f"{row['p99_ttft_ms']:.4f}\t{row['p99_tpot_ms']:.4f}\t"
            f"{row['p99_e2e_latency_ms']:.4f}\n"
        )

print(output_path.read_text(encoding="utf-8"))
print("PERFORMANCE TEST: PASS")
PY

echo "Results: $RESULT_ROOT"
