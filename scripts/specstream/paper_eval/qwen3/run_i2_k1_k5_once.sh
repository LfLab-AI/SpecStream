#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/root/lifei/SpecStream}"
SPECSTREAM_PYTHON="${SPECSTREAM_PYTHON:-/root/miniconda3/envs/spectre/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/model/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-/root/autodl-tmp/model/Qwen3-0.6B}"
TARGET_GPU="${TARGET_GPU:-1}"
TARGET_GPUS="${TARGET_GPUS:-$TARGET_GPU}"
TARGET_TP_SIZE="${TARGET_TP_SIZE:-1}"
DRAFT_GPU="${DRAFT_GPU:-0}"
COLOCATED_GPU="${COLOCATED_GPU:-1}"
COLOCATED_TP_RANK="${COLOCATED_TP_RANK:-0}"
MODEL_TAG="${MODEL_TAG:-qwen3_0p6b_8b}"

cd "$REPO"
if [[ -s /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate /root/miniconda3/envs/spectre
fi

export REPO SPECSTREAM_PYTHON TARGET_MODEL DRAFT_MODEL
export TARGET_GPU TARGET_GPUS TARGET_TP_SIZE DRAFT_GPU COLOCATED_GPU
export COLOCATED_TP_RANK MODEL_TAG
# I2 changes H2D/q/cohort policy, not Draft placement or Target/Draft overlap.
# Override a stale TP1 environment inherited from the old manual.
export SPECSTREAM_DRAFT_TP_SIZE="$TARGET_TP_SIZE"
export SPECSTREAM_OVERLAP_MODE=serial
export SPECSTREAM_FIXED_Q=0
export I2_DRY_RUN="${I2_DRY_RUN:-0}"
[[ "$I2_DRY_RUN" == 0 || "$I2_DRY_RUN" == 1 ]] || { echo "ERROR: I2_DRY_RUN must be 0 or 1" >&2; exit 2; }
export SPECSTREAM_DRY_RUN="$I2_DRY_RUN"
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$(dirname "$SPECSTREAM_PYTHON"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
export TARGET_PORT="${TARGET_PORT:-30000}"
export DRAFT_PORT="${DRAFT_PORT:-30001}"
export ZMQ_PORT="${ZMQ_PORT:-5557}"
export SERVER_CONTEXT_LEN=40960
export FINAL_DRAFT_TPCS="${FINAL_DRAFT_TPCS:-34}"

export SPECSTREAM_TARGET_MEM_FRACTION="${SPECSTREAM_TARGET_MEM_FRACTION:-0.50}"
export SPECSTREAM_DRAFT_MEM_FRACTION="${SPECSTREAM_DRAFT_MEM_FRACTION:-0.72}"
export SPECSTREAM_TARGET_MAX_TOTAL_TOKENS="${SPECSTREAM_TARGET_MAX_TOTAL_TOKENS:-98304}"
export SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS="${SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS:-516224}" ##270336
export SPECSTREAM_TARGET_MIN_KV_TOKENS="${SPECSTREAM_TARGET_MIN_KV_TOKENS:-$SPECSTREAM_TARGET_MAX_TOTAL_TOKENS}"
export SPECSTREAM_DRAFT_MIN_KV_TOKENS="${SPECSTREAM_DRAFT_MIN_KV_TOKENS:-$SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS}"
export SPECSTREAM_PREFILL_MAX_REQUESTS="${SPECSTREAM_PREFILL_MAX_REQUESTS:-1}"
export SPECSTREAM_GPU_HISTORY_CACHE_TOKENS="${SPECSTREAM_GPU_HISTORY_CACHE_TOKENS:-0}" ##8192
export SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS="${SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS:-0}"
export SPECSTREAM_REQUIRE_SLACK_FILL="${SPECSTREAM_REQUIRE_SLACK_FILL:-0}"
export I2_INPUT_LENGTHS="${I2_INPUT_LENGTHS:-30720}" ##16384
export I2_CONCURRENCIES="${I2_CONCURRENCIES:-4 8 16}" ## 1
export I2_METHODS="${I2_METHODS:-K1 K2 K3 K4 K5}"

I2_ROOT="${I2_ROOT:-$REPO/results/${MODEL_TAG}_i2_k1_k5_once_$(date +%Y%m%d_%H%M%S)}"
export I2_ROOT
if [[ -e "$I2_ROOT" ]] && [[ -n "$(find "$I2_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: I2_ROOT is not empty; formal matrices never append or resume: $I2_ROOT" >&2
  echo "Unset I2_ROOT or choose a new directory." >&2
  exit 2
fi
mkdir -p "$I2_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
"$SPECSTREAM_PYTHON" - "$I2_ROOT/summary/matrix_plan.json" <<'PY'
import itertools
import json
import os
import sys

inputs = [int(value) for value in os.environ["I2_INPUT_LENGTHS"].split()]
concurrencies = [int(value) for value in os.environ["I2_CONCURRENCIES"].split()]
methods = os.environ["I2_METHODS"].split()
for name, values in (("input lengths", inputs), ("concurrencies", concurrencies), ("methods", methods)):
    if not values or len(values) != len(set(values)):
        raise SystemExit(f"ERROR: {name} must be nonempty and unique")
if any(value < 1 for value in inputs + concurrencies):
    raise SystemExit("ERROR: input lengths and concurrencies must be positive")
if not set(methods) <= {"K1", "K2", "K3", "K4", "K5"}:
    raise SystemExit("ERROR: I2_METHODS supports only K1 through K5")
cache_tokens = int(os.environ["SPECSTREAM_GPU_HISTORY_CACHE_TOKENS"])
if cache_tokens < -1:
    raise SystemExit("ERROR: GPU History budget must be -1, zero, or positive")
draft_cap = int(os.environ["SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS"])
required_draft_tokens = min(max(concurrencies), 64) * (max(inputs) + 256 + 8)
if draft_cap > 0 and draft_cap < required_draft_tokens:
    raise SystemExit(
        f"ERROR: Draft KV cap {draft_cap} cannot cover requested concurrency; "
        f"need at least {required_draft_tokens} tokens for full-context Draft. "
        "Reduce concurrency or recalibrate the same capacity contract for all cells."
    )
cells = [
    {"method": method, "input_len": length, "concurrency": concurrency,
     "case_tag": f"{method}_i2_{length}_c{concurrency}"}
    for length, concurrency, method in itertools.product(inputs, concurrencies, methods)
]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({"cells": cells, "expected_cells": len(cells), "gpu_history_cache_tokens": cache_tokens, "required_draft_tokens": required_draft_tokens, "target_tp_size": int(os.environ["TARGET_TP_SIZE"]), "draft_tp_size": int(os.environ["SPECSTREAM_DRAFT_TP_SIZE"]), "dry_run": os.environ["I2_DRY_RUN"] == "1"}, handle, indent=2)
print(f"I2_MATRIX_PLANNED_CELLS={len(cells)}")
PY
read -r -a i2_inputs <<< "$I2_INPUT_LENGTHS"
read -r -a i2_concurrencies <<< "$I2_CONCURRENCIES"
read -r -a i2_methods <<< "$I2_METHODS"

for path in \
  "$SPECSTREAM_PYTHON" \
  "$TARGET_MODEL/config.json" \
  "$DRAFT_MODEL/config.json" \
  scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  scripts/specstream/paper_eval/qwen3/run_public_once.sh; do
  [[ -s "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
bash -n scripts/specstream/paper_eval/qwen3/run_public_once.sh
"$SPECSTREAM_PYTHON" - <<'PY'
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spectre.specstream.config import SpecStreamConfig

pairs = {
    "specstream_pcie_grant_poll_us": "pcie_grant_poll_us",
    "specstream_serialize_h2d": "serialize_h2d",
}
for server_name, config_name in pairs.items():
    server_default = getattr(ServerArgs, server_name, None)
    config_default = getattr(SpecStreamConfig, config_name)
    if server_default != config_default:
        raise SystemExit(
            "ERROR: ServerArgs/SpecStreamConfig mismatch for "
            f"{server_name}: server={server_default!r} config={config_default!r}"
        )
print("I2_SERVER_ARGS_CONFIG_GATE=PASS")
PY

if [[ "$I2_DRY_RUN" != 1 ]]; then
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -Eq '^[[:space:]]*[0-9]+'; then
  echo "ERROR: a GPU compute process is already running; refusing a formal matrix" >&2
  exit 2
fi

export PREFLIGHT_ROOT="$I2_ROOT/preflight"
set +e
bash scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  2>&1 | tee "$I2_ROOT/summary/preflight.log"
preflight_rc=${PIPESTATUS[0]}
set -e
if (( preflight_rc != 0 )) || \
   ! grep -q 'QWEN3_PUBLIC_PREFLIGHT=PASS' "$I2_ROOT/summary/preflight.log" || \
   [[ ! -s "$PREFLIGHT_ROOT/runtime_env.sh" ]]; then
  echo "ERROR: preflight failed; inspect $I2_ROOT/summary/preflight.log" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$PREFLIGHT_ROOT/runtime_env.sh"

else
  # Command generation only: require an existing current-machine UUID mapping.
  : "${TARGET_UUID:?Source the current preflight runtime_env.sh for I2 dry-run}"
  : "${TARGET_UUIDS:?Source the current preflight runtime_env.sh for I2 dry-run}"
fi

echo "I2_ROOT=$I2_ROOT"
echo "I2_TP_CONTRACT=target_tp:${TARGET_TP_SIZE},draft_tp:${SPECSTREAM_DRAFT_TP_SIZE},draft_visible:${TARGET_UUIDS},overlap:serial"
echo "I2_MEMORY_CONTRACT=target_fraction:${SPECSTREAM_TARGET_MEM_FRACTION},draft_fraction:${SPECSTREAM_DRAFT_MEM_FRACTION},target_kv:${SPECSTREAM_TARGET_MAX_TOTAL_TOKENS},draft_kv:${SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS},gpu_history_cache:${SPECSTREAM_GPU_HISTORY_CACHE_TOKENS},target_allocation_guard:${SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS}"

i2_failed=0
for input in "${i2_inputs[@]}"; do
  for conc in "${i2_concurrencies[@]}"; do
    for method in "${i2_methods[@]}"; do
      echo "RUN innovation=2 method=$method input=$input concurrency=$conc"
      if ! METHOD=$method \
        DATASET_TAG="i2_${input}" DATASET_NAME=random-ids \
        INPUT_LEN=$input NUM_PROMPTS=32 OUTPUT_LEN=256 \
        MAX_CONCURRENCY=$conc WARMUP_REQUESTS=4 \
        REQUEST_RATE=inf SEED=1 RESULT_ROOT="$I2_ROOT" \
        CASE_TIMEOUT_S=21600 \
        bash scripts/specstream/paper_eval/qwen3/run_public_once.sh; then
        echo "ERROR: formal I2 cell failed method=$method input=$input c=$conc" >&2
        i2_failed=1
        break 3
      fi
    done
  done
done

if [[ "$I2_DRY_RUN" == 1 ]]; then
  (( i2_failed == 0 )) || exit 1
  "$SPECSTREAM_PYTHON" - "$I2_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
plan = json.loads((root / "summary/matrix_plan.json").read_text())
for cell in plan["cells"]:
    path = root / "logs" / cell["case_tag"] / "config.env"
    c = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    assert c["TARGET_TP_SIZE"] == c["DRAFT_TP_SIZE"] == str(plan["target_tp_size"]), path
    assert c["DRAFT_VISIBLE"] == c["TARGET_VISIBLE"], path
    assert c["FIXED_Q_MODE"] == "ordinary", path
print("I2_DRY_RUN=PASS", len(plan["cells"]))
PY
  exit 0
fi

if (( i2_failed == 0 )); then
  set +e
  "$SPECSTREAM_PYTHON" - "$I2_ROOT" <<'PY'
import csv
import glob
import json
import os
import sys

root = sys.argv[1]
with open(os.path.join(root, "summary", "matrix_plan.json"), encoding="utf-8") as handle:
    plan = json.load(handle)
expected_tags = {cell["case_tag"] for cell in plan["cells"]}
expected_cache = int(plan["gpu_history_cache_tokens"])
paths = sorted(glob.glob(os.path.join(root, "profiles", "K*.csv")))
profile_tags = {os.path.splitext(os.path.basename(path))[0] for path in paths}
if profile_tags != expected_tags:
    raise SystemExit(f"ERROR: profile cells mismatch: missing={sorted(expected_tags-profile_tags)}, unexpected={sorted(profile_tags-expected_tags)}")
markers = sorted(glob.glob(os.path.join(root, "logs", "K*", "case_complete.marker")))
marker_tags = {os.path.basename(os.path.dirname(path)) for path in markers}
if marker_tags != expected_tags:
    raise SystemExit(f"ERROR: completion cells mismatch: missing={sorted(expected_tags-marker_tags)}, unexpected={sorted(marker_tags-expected_tags)}")
for path in paths:
    case_tag = os.path.splitext(os.path.basename(path))[0]
    method = case_tag.split("_", 1)[0]
    config_path = os.path.join(root, "logs", case_tag, "config.env")
    with open(config_path, encoding="utf-8") as handle:
        config = dict(
            line.rstrip("\n").split("=", 1)
            for line in handle
            if "=" in line
        )
    if (
        config.get("TARGET_TP_SIZE") != str(plan["target_tp_size"])
        or config.get("DRAFT_TP_SIZE") != str(plan["draft_tp_size"])
        or config.get("TARGET_VISIBLE") != config.get("DRAFT_VISIBLE")
        or config.get("FIXED_Q_MODE") != "ordinary"
    ):
        raise SystemExit(f"ERROR: invalid I2 Draft TP placement/mode: {config_path}")
    expected_serial = method in {"K1", "K2"}
    actual_buffers = int(config.get("NUM_STAGING_BUFFERS", "0"))
    expected_prefetch = "0" if expected_serial else "1"
    expected_execution = "serialized" if expected_serial else "async_copy_stream"
    if (
        config.get("SERIALIZE_H2D") != ("1" if expected_serial else "0")
        or config.get("H2D_EXECUTION") != expected_execution
        or (actual_buffers != 1 if expected_serial else actual_buffers < 2)
        or config.get("LAYER_PREFETCH") != expected_prefetch
        or config.get("GPU_HISTORY_CACHE_TOKENS") != str(expected_cache)
    ):
        raise SystemExit(f"ERROR: invalid K1-K5 execution contract: {config_path}")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hit = max((int(row["gpu_history_hit_tokens"]) for row in rows), default=0)
    miss = max((int(row["cpu_history_miss_tokens"]) for row in rows), default=0)
    input_length = next(cell["input_len"] for cell in plan["cells"] if cell["case_tag"] == case_tag)
    requires_offload = input_length > int(config.get("MIN_HISTORY_TOKENS", "8192"))
    invalid_hit = (hit != 0 if expected_cache == 0 else expected_cache > 0 and hit <= 0 and requires_offload)
    chunk_tokens = int(config.get("CHUNK_TOKENS", "2048"))
    initial_sealed = max(0, input_length - int(config.get("ACTIVE_TAIL_TOKENS", "512"))) // chunk_tokens * chunk_tokens
    requires_miss = requires_offload and 0 <= expected_cache < initial_sealed
    if invalid_hit or (requires_miss and miss <= 0) or (requires_offload and hit + miss <= 0):
        raise SystemExit(
            f"ERROR: GPU/CPU History path inactive: {path} hit={hit} miss={miss}"
        )
    for row in rows:
        ops = int(row["h2d_ops"])
        if ops <= 0:
            continue
        copy_events = int(row["h2d_event_ops"])
        wait_events = int(row["h2d_wait_event_ops"])
        if copy_events != ops or wait_events != ops:
            raise SystemExit(
                f"ERROR: incomplete CUDA H2D/stall coverage: {path} "
                f"round={row['round_id']} ops={ops} copy_events={copy_events} "
                f"wait_events={wait_events}"
            )
    if method in {"K4", "K5"}:
        rtt_samples = {}
        candidate_costs_complete = False
        for row in rows:
            try:
                samples = json.loads(row.get("draft_rtt_samples_by_q") or "{}")
                costs = json.loads(row.get("controller_candidate_costs") or "[]")
            except json.JSONDecodeError as exc:
                raise SystemExit(f"ERROR: malformed controller telemetry: {path}: {exc}")
            for q, count in samples.items():
                rtt_samples[str(q)] = max(rtt_samples.get(str(q), 0), int(count))
            if {int(item["q"]) for item in costs} == {2, 4, 6, 8}:
                candidate_costs_complete = candidate_costs_complete or all(
                    float(item["ordinary"]) > 0.0 for item in costs
                )
        if any(rtt_samples.get(str(q), 0) < 4 for q in (2, 4, 6, 8)):
            raise SystemExit(
                f"ERROR: incomplete per-q Draft RTT warmup: {path} {rtt_samples}"
            )
        if not candidate_costs_complete:
            raise SystemExit(f"ERROR: candidate controller costs missing: {path}")
print("I2_RUNTIME_MEASUREMENT_GATE=PASS")
PY
  measurement_rc=$?
  set -e
  if (( measurement_rc != 0 )); then
    i2_failed=1
  fi
fi

if (( i2_failed == 0 )); then
  printf 'I2_ROOT=%s\nCOMPLETED_AT_UTC=%s\n' \
    "$I2_ROOT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$I2_ROOT/i2_matrix_complete.marker"
fi
echo "I2_FORMAL_MATRIX_FAILED=$i2_failed"
(( i2_failed == 0 ))
