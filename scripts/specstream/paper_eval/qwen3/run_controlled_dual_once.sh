#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

: "${VARIANT:?NATIVE/RESTORE/REFERENCE/FUSED/FULL_IO}"
: "${CASE_TAG:?}"
: "${INPUT_LEN:?}"
: "${OUTPUT_LEN:?}"
: "${NUM_PROMPTS:?}"
: "${MAX_CONCURRENCY:?}"
: "${RESULT_ROOT:?}"
: "${SPECSTREAM_PYTHON:?}"
: "${TARGET_MODEL:?}"
: "${DRAFT_MODEL:?}"
: "${TARGET_UUID:?}"
: "${DRAFT_UUID:?}"

TARGET_PORT="${TARGET_PORT:-30000}"
DRAFT_PORT="${DRAFT_PORT:-30001}"
ZMQ_PORT="${ZMQ_PORT:-5557}"
SERVER_CONTEXT_LEN="${SERVER_CONTEXT_LEN:-32768}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-1}"
WARMUP_REQUESTS=4
(( MAX_CONCURRENCY <= 1 )) && WARMUP_REQUESTS=1
CASE_ROOT="$RESULT_ROOT/logs/$CASE_TAG"
PROFILE_CSV="$RESULT_ROOT/profiles/$CASE_TAG.csv"
mkdir -p "$CASE_ROOT" "$RESULT_ROOT"/{bench,profiles,gpu_monitor,env}
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$(dirname "$SPECSTREAM_PYTHON"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v ninja >/dev/null || {
  echo "ERROR: ninja is required for SGLang Qwen3 JIT kernels" >&2
  return 2
}

COMMON_TARGET=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
  --context-length "$SERVER_CONTEXT_LEN" --mem-fraction-static 0.85
  --skip-server-warmup --attention-backend fa3 --page-size 1
  --speculative-algorithm SPECTRE --spectre-role target
  --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --spectre-fixed-q-mode parallel
  --spectre-require-draft --spectre-draft-timeout-action fallback
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
  --disable-radix-cache --disable-cuda-graph --disable-piecewise-cuda-graph
  --disable-overlap-schedule
)
DRAFT_ARGS=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
  --context-length "$SERVER_CONTEXT_LEN" --mem-fraction-static 0.85
  --skip-server-warmup --attention-backend fa3
  --speculative-algorithm SPECTRE --spectre-role draft
  --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --spectre-draft-priority
  --spectre-max-draft-priority-steps 8
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
  --disable-overlap-schedule
)
TARGET_ARGS=("${COMMON_TARGET[@]}")

case "$VARIANT" in
  NATIVE) ;;
  RESTORE)
    TARGET_ARGS+=(
      --specstream-enabled --specstream-full-restore-baseline
      --specstream-chunk-tokens 2048 --specstream-num-buffers 2
      --specstream-chunks-per-transfer 1 --specstream-active-tail-tokens 512
      --specstream-min-history-tokens 8192 --specstream-cpu-memory-gb 128
      --specstream-gpu-reserve-mb 1024 --specstream-profile-path "$PROFILE_CSV"
    )
    ;;
  REFERENCE)
    TARGET_ARGS+=(
      --specstream-enabled --specstream-reference-attention
      --specstream-chunk-tokens 2048 --specstream-num-buffers 2
      --specstream-chunks-per-transfer 1 --no-specstream-layer-prefetch
      --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
      --specstream-cpu-memory-gb 128 --specstream-gpu-reserve-mb 1024
      --specstream-profile-path "$PROFILE_CSV"
    )
    ;;
  FUSED)
    TARGET_ARGS+=(
      --specstream-enabled --no-specstream-reference-attention
      --specstream-chunk-tokens 2048 --specstream-num-buffers 2
      --specstream-chunks-per-transfer 4 --no-specstream-layer-prefetch
      --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
      --specstream-cpu-memory-gb 128 --specstream-gpu-reserve-mb 1024
      --specstream-profile-path "$PROFILE_CSV"
    )
    ;;
  FULL_IO)
    TARGET_ARGS+=(
      --specstream-enabled --no-specstream-reference-attention
      --specstream-chunk-tokens 2048 --specstream-num-buffers 2
      --specstream-chunks-per-transfer 4 --specstream-layer-prefetch
      --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
      --specstream-cpu-memory-gb 128 --specstream-gpu-reserve-mb 1024
      --specstream-dynamic-q --specstream-q-candidates 1,2,4
      --specstream-q-switch-threshold 0.08 --specstream-cohort-enabled
      --specstream-max-cohort-size 8 --specstream-max-cohort-delay-us 200
      --specstream-profile-path "$PROFILE_CSV"
    )
    ;;
  *) echo "ERROR: unsupported VARIANT=$VARIANT" >&2; return 2 ;;
esac

printf -v TARGET_CMD '%q ' "${TARGET_ARGS[@]}"
printf -v DRAFT_CMD '%q ' "${DRAFT_ARGS[@]}"
printf '%s\n' "$TARGET_CMD" > "$CASE_ROOT/target_command.txt"
printf '%s\n' "$DRAFT_CMD" > "$CASE_ROOT/draft_command.txt"
cat > "$CASE_ROOT/config.env" <<EOF
VARIANT=$VARIANT
CASE_TAG=$CASE_TAG
TARGET_MODEL=$TARGET_MODEL
DRAFT_MODEL=$DRAFT_MODEL
TARGET_UUID=$TARGET_UUID
DRAFT_UUID=$DRAFT_UUID
INPUT_LEN=$INPUT_LEN
OUTPUT_LEN=$OUTPUT_LEN
NUM_PROMPTS=$NUM_PROMPTS
MAX_CONCURRENCY=$MAX_CONCURRENCY
REQUEST_RATE=$REQUEST_RATE
WARMUP_REQUESTS=$WARMUP_REQUESTS
SEED=$SEED
SERVER_CONTEXT_LEN=$SERVER_CONTEXT_LEN
VERIFY_Q=4
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_NUM_DRAFT_TOKENS=4
FIXED_Q_MODE=parallel
DYNAMIC_Q=$([[ "$VARIANT" == FULL_IO ]] && echo 1 || echo 0)
Q_CANDIDATES=$([[ "$VARIANT" == FULL_IO ]] && echo 1,2,4 || echo disabled)
COHORT_ENABLED=$([[ "$VARIANT" == FULL_IO ]] && echo 1 || echo 0)
MAX_COHORT_SIZE=$([[ "$VARIANT" == FULL_IO ]] && echo 8 || echo disabled)
MAX_COHORT_DELAY_US=$([[ "$VARIANT" == FULL_IO ]] && echo 200 || echo disabled)
EOF
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$TARGET_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$DRAFT_UUID"
export SPECSTREAM_TARGET_CMD="$TARGET_CMD"
export SPECSTREAM_DRAFT_CMD="$DRAFT_CMD"
export SPECSTREAM_TARGET_READY_CMD="curl -fsS http://127.0.0.1:${TARGET_PORT}/health"
export SPECSTREAM_DRAFT_READY_CMD="curl -fsS http://127.0.0.1:${DRAFT_PORT}/health"
export SPECSTREAM_RESULT_ROOT="$CASE_ROOT"
export SPECSTREAM_BENCH_CMD="PYTHONUNBUFFERED=1 TQDM_MININTERVAL=0.5 \
SPECSTREAM_PYTHON='$SPECSTREAM_PYTHON' TARGET_MODEL='$TARGET_MODEL' \
DATASET_NAME=random-ids INPUT_LEN='$INPUT_LEN' OUTPUT_LEN='$OUTPUT_LEN' \
NUM_PROMPTS='$NUM_PROMPTS' REQUEST_RATE='$REQUEST_RATE' \
MAX_CONCURRENCY='$MAX_CONCURRENCY' RANGE_RATIO=1 \
WARMUP_REQUESTS='$WARMUP_REQUESTS' SEED='$SEED' \
CONTEXT_LEN='$SERVER_CONTEXT_LEN' CASE_TAG='$CASE_TAG' \
OUTPUT_DIR='$RESULT_ROOT/bench' RUN_ID=once \
bash scripts/specstream/run_benchmark_case.sh"

monitor_pid=""
cleanup(){ [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw \
  --format=csv -lms 200 > "$RESULT_ROOT/gpu_monitor/$CASE_TAG.csv" 2>&1 &
monitor_pid=$!

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
[[ "$VARIANT" == NATIVE ]] || test -s "$PROFILE_CSV"
echo "PASS: $CASE_TAG"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
