#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL to the Target model path}"
DATASET_NAME="${DATASET_NAME:-random-ids}"
DATASET_PATH="${DATASET_PATH:-}"
CASE_TAG="${CASE_TAG:-B0_smoke}"
INPUT_LEN="${INPUT_LEN:-4096}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
REQUEST_RATE="${REQUEST_RATE:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
# In SGLang compute_random_lens(), 1 means exactly INPUT_LEN/OUTPUT_LEN;
# 0 samples uniformly from 1..full_len.  Controlled baseline sweeps need 1.
RANGE_RATIO="${RANGE_RATIO:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
SEED="${SEED:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
OUTPUT_DIR="${OUTPUT_DIR:-results/bench}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"

if ! curl -fsS "${BASE_URL}/health" >/dev/null; then
  echo "Target health check failed: ${BASE_URL}/health" >&2
  exit 2
fi

args=(
  python -m sglang.bench_serving
  --backend sglang
  --base-url "${BASE_URL}"
  --model "${TARGET_MODEL}"
  --tokenizer "${TARGET_MODEL}"
  --dataset-name "${DATASET_NAME}"
  --request-rate "${REQUEST_RATE}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --seed "${SEED}"
  --flush-cache
  --output-details
  --tag "${CASE_TAG}"
  --output-file "${OUTPUT_DIR}/${CASE_TAG}_${RUN_ID}.jsonl"
)

case "${DATASET_NAME}" in
  random-ids)
    args+=(
      --tokenize-prompt
      --num-prompts "${NUM_PROMPTS}"
      --random-input-len "${INPUT_LEN}"
      --random-output-len "${OUTPUT_LEN}"
      --random-range-ratio "${RANGE_RATIO}"
    )
    ;;
  random)
    args+=(
      --num-prompts "${NUM_PROMPTS}"
      --random-input-len "${INPUT_LEN}"
      --random-output-len "${OUTPUT_LEN}"
      --random-range-ratio "${RANGE_RATIO}"
    )
    if [[ -n "${DATASET_PATH}" ]]; then
      args+=(--dataset-path "${DATASET_PATH}")
    fi
    ;;
  sharegpt|custom|longbench_v2)
    args+=(
      --num-prompts "${NUM_PROMPTS}"
      --sharegpt-output-len "${OUTPUT_LEN}"
      --sharegpt-context-len "${CONTEXT_LEN}"
    )
    if [[ -n "${DATASET_PATH}" ]]; then
      args+=(--dataset-path "${DATASET_PATH}")
    fi
    ;;
  generated-shared-prefix)
    GSP_NUM_GROUPS="${GSP_NUM_GROUPS:-8}"
    GSP_PROMPTS_PER_GROUP="${GSP_PROMPTS_PER_GROUP:-8}"
    GSP_QUESTION_LEN="${GSP_QUESTION_LEN:-128}"
    args+=(
      --gsp-num-groups "${GSP_NUM_GROUPS}"
      --gsp-prompts-per-group "${GSP_PROMPTS_PER_GROUP}"
      --gsp-system-prompt-len "${INPUT_LEN}"
      --gsp-question-len "${GSP_QUESTION_LEN}"
      --gsp-output-len "${OUTPUT_LEN}"
      --gsp-range-ratio "${RANGE_RATIO}"
    )
    ;;
  *)
    echo "Unsupported DATASET_NAME: ${DATASET_NAME}" >&2
    exit 3
    ;;
esac

printf 'Running case=%s dataset=%s input=%s output=%s rate=%s concurrency=%s\n' \
  "${CASE_TAG}" "${DATASET_NAME}" "${INPUT_LEN}" "${OUTPUT_LEN}" \
  "${REQUEST_RATE}" "${MAX_CONCURRENCY}"

"${args[@]}"
