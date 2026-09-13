#!/usr/bin/env bash
set -euo pipefail

: "${SPECSTREAM_PYTHON:?}"
: "${TARGET_MODEL:?}"

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
DATASET_NAME="${DATASET_NAME:-random-ids}"
DATASET_PATH="${DATASET_PATH:-}"
CASE_TAG="${CASE_TAG:-smoke}"
INPUT_LEN="${INPUT_LEN:-16384}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
RANGE_RATIO="${RANGE_RATIO:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
SEED="${SEED:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
OUTPUT_DIR="${OUTPUT_DIR:-results/bench}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"
curl -fsS "$BASE_URL/health" >/dev/null

args=(
  "$SPECSTREAM_PYTHON" -m sglang.bench_serving
  --backend sglang
  --base-url "$BASE_URL"
  --model "$TARGET_MODEL"
  --tokenizer "$TARGET_MODEL"
  --dataset-name "$DATASET_NAME"
  --request-rate "$REQUEST_RATE"
  --max-concurrency "$MAX_CONCURRENCY"
  --warmup-requests "$WARMUP_REQUESTS"
  --seed "$SEED"
  --flush-cache
  --output-details
  --tag "$CASE_TAG"
  --output-file "$OUTPUT_DIR/${CASE_TAG}_${RUN_ID}.jsonl"
)

case "$DATASET_NAME" in
  random-ids)
    args+=(
      --tokenize-prompt
      --num-prompts "$NUM_PROMPTS"
      --random-input-len "$INPUT_LEN"
      --random-output-len "$OUTPUT_LEN"
      --random-range-ratio "$RANGE_RATIO"
    )
    ;;
  random)
    args+=(
      --num-prompts "$NUM_PROMPTS"
      --random-input-len "$INPUT_LEN"
      --random-output-len "$OUTPUT_LEN"
      --random-range-ratio "$RANGE_RATIO"
    )
    [[ -n "$DATASET_PATH" ]] && args+=(--dataset-path "$DATASET_PATH")
    ;;
  sharegpt|custom|longbench_v2)
    args+=(
      --num-prompts "$NUM_PROMPTS"
      --sharegpt-output-len "$OUTPUT_LEN"
      --sharegpt-context-len "$CONTEXT_LEN"
    )
    [[ -n "$DATASET_PATH" ]] && args+=(--dataset-path "$DATASET_PATH")
    ;;
  *)
    echo "ERROR: unsupported DATASET_NAME=$DATASET_NAME" >&2
    exit 3
    ;;
esac

echo "[BENCH] python=$SPECSTREAM_PYTHON"
echo "[BENCH] case=$CASE_TAG dataset=$DATASET_NAME input=$INPUT_LEN output=$OUTPUT_LEN concurrency=$MAX_CONCURRENCY"
"${args[@]}"
