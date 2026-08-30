#!/usr/bin/env bash
set -euo pipefail

: "${CASE_TAG:?Set CASE_TAG}"
: "${PUBLIC_DATASET_PATH:?Set PUBLIC_DATASET_PATH}"
: "${TARGET_MODEL:?Set TARGET_MODEL}"
: "${TARGET_PORT:?Set TARGET_PORT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
SEED="${SEED:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"

# NUM_PROMPTS=auto 时取 min(dataset size, MAX_PROMPTS)
NUM_PROMPTS="${NUM_PROMPTS:-auto}"
MAX_PROMPTS="${MAX_PROMPTS:-256}"

if [[ "$NUM_PROMPTS" == "auto" ]]; then
  DATASET_N=$(python - "$PUBLIC_DATASET_PATH" <<'PY'
import json, sys
p = sys.argv[1]
obj = json.load(open(p, encoding="utf-8"))
print(len(obj))
PY
)

  if (( DATASET_N < MAX_PROMPTS )); then
    NUM_PROMPTS="$DATASET_N"
  else
    NUM_PROMPTS="$MAX_PROMPTS"
  fi
fi

mkdir -p "$OUTPUT_DIR"

OUT="$OUTPUT_DIR/${CASE_TAG}.jsonl"

echo
echo "============================================================"
echo "PUBLIC E2E BENCHMARK"
echo "CASE_TAG        = $CASE_TAG"
echo "DATASET_PATH    = $PUBLIC_DATASET_PATH"
echo "NUM_PROMPTS     = $NUM_PROMPTS"
echo "OUTPUT_LEN      = $OUTPUT_LEN"
echo "REQUEST_RATE    = $REQUEST_RATE"
echo "MAX_CONCURRENCY = $MAX_CONCURRENCY"
echo "WARMUP_REQUESTS = $WARMUP_REQUESTS"
echo "OUTPUT          = $OUT"
echo "============================================================"
echo

python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port "$TARGET_PORT" \
  --model "$TARGET_MODEL" \
  --dataset-name sharegpt \
  --dataset-path "$PUBLIC_DATASET_PATH" \
  --sharegpt-output-len "$OUTPUT_LEN" \
  --sharegpt-context-len "$CONTEXT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --warmup-requests "$WARMUP_REQUESTS" \
  --seed "$SEED" \
  --extra-request-body '{"temperature":0,"top_p":1}' \
  --output-file "$OUT" \
  --output-details

echo
echo "WROTE $OUT"
