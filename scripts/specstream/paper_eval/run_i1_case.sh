#!/usr/bin/env bash
set -euo pipefail

: "${METHOD:?Set METHOD=E0/S0...S5/P0...P6}"
: "${CASE_TAG:?Set CASE_TAG}"
: "${INPUT_LEN:?Set INPUT_LEN}"
: "${OUTPUT_LEN:?Set OUTPUT_LEN}"
: "${NUM_PROMPTS:?Set NUM_PROMPTS}"
: "${MAX_CONCURRENCY:?Set MAX_CONCURRENCY}"
: "${REQUEST_RATE:?Set REQUEST_RATE}"
: "${WARMUP_REQUESTS:?Set WARMUP_REQUESTS}"

: "${REPO:?Set REPO}"
: "${TARGET_MODEL:?Set TARGET_MODEL}"
: "${SHAREGPT_JSON:?Set SHAREGPT_JSON}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"

DATASET_NAME="${DATASET_NAME:-random}"
DATASET_PATH="${DATASET_PATH:-$SHAREGPT_JSON}"
RANGE_RATIO="${RANGE_RATIO:-1}"
SEED="${SEED:-1}"

cleanup() {
  bash "$REPO/scripts/specstream/paper_eval/stop_i1_system.sh" || true
}
trap cleanup EXIT INT TERM

bash "$REPO/scripts/specstream/paper_eval/start_i1_system.sh" "$METHOD"

echo "===== Target health ====="
curl -fsS "http://127.0.0.1:${TARGET_PORT:-30000}/health"

if [[ "$METHOD" != "E0" ]]; then
  echo "===== Draft health ====="
  curl -fsS "http://127.0.0.1:${DRAFT_PORT:-30001}/health"
fi

echo "===== Running benchmark: $CASE_TAG ====="

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}" \
TARGET_MODEL="$TARGET_MODEL" \
CASE_TAG="$CASE_TAG" \
DATASET_NAME="$DATASET_NAME" \
DATASET_PATH="$DATASET_PATH" \
INPUT_LEN="$INPUT_LEN" \
OUTPUT_LEN="$OUTPUT_LEN" \
NUM_PROMPTS="$NUM_PROMPTS" \
REQUEST_RATE="$REQUEST_RATE" \
MAX_CONCURRENCY="$MAX_CONCURRENCY" \
RANGE_RATIO="$RANGE_RATIO" \
WARMUP_REQUESTS="$WARMUP_REQUESTS" \
SEED="$SEED" \
CONTEXT_LEN="${SERVER_CONTEXT_LEN:-32768}" \
OUTPUT_DIR="$RESULT_ROOT/bench" \
bash "$REPO/scripts/specstream/run_benchmark_case.sh"

echo "===== Server error scan ====="

LOG_DIR="$RESULT_ROOT/logs/$CASE_TAG"

grep -Eini \
  'out of memory|OOM|timeout|missing|error|traceback|assert|nan|inf' \
  "$LOG_DIR"/*.log \
  | tail -n 100 || true

echo "Completed: $CASE_TAG"
