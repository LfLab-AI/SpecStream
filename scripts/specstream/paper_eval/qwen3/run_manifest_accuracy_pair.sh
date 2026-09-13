#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

for required_var in REPO SPECSTREAM_PYTHON ACCURACY_DATASET DATASET_TAG \
  DATASET_PATH OUTPUT_LEN MAX_CONCURRENCY ACC_ROOT; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "ERROR: required environment variable is unset: $required_var" >&2
    return 2
  fi
done

case "$ACCURACY_DATASET" in
  longbench_v2|mrcr) ;;
  *) echo "ERROR: unsupported ACCURACY_DATASET=$ACCURACY_DATASET" >&2; return 2 ;;
esac
if [[ "$ACCURACY_DATASET" == longbench_v2 ]] && (( OUTPUT_LEN < 256 )); then
  echo "ERROR: LongBench-v2 accuracy requires OUTPUT_LEN>=256; got $OUTPUT_LEN" >&2
  return 2
fi
[[ -s "$DATASET_PATH" ]] || { echo "ERROR: accuracy manifest missing: $DATASET_PATH" >&2; return 2; }
expected_rows=$(DATASET_PATH="$DATASET_PATH" "$SPECSTREAM_PYTHON" - <<'PY'
import json, os
rows=json.load(open(os.environ["DATASET_PATH"],encoding="utf-8"))
assert isinstance(rows,list) and rows
print(len(rows))
PY
)
mkdir -p "$ACC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}

for method in SGLANG_SD SPECSTREAM_1GPU; do
  METHOD="$method" \
  CLIENT_MODE=accuracy \
  ACCURACY_DATASET="$ACCURACY_DATASET" \
  DATASET_NAME=sharegpt \
  DATASET_TAG="$DATASET_TAG" \
  DATASET_PATH="$DATASET_PATH" \
  NUM_PROMPTS=0 \
  OUTPUT_LEN="$OUTPUT_LEN" \
  MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  WARMUP_REQUESTS=0 \
  REQUEST_RATE=inf \
  SEED=1 \
  RESULT_ROOT="$ACC_ROOT" \
  CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-21600}" \
  bash "$REPO/scripts/specstream/paper_eval/qwen3/run_public_once.sh"
done

"$SPECSTREAM_PYTHON" "$REPO/scripts/specstream/paper_eval/qwen3/score_public_accuracy.py" \
  --dataset "$ACCURACY_DATASET" \
  --input "SGLANG_SD=$ACC_ROOT/bench/SGLANG_SD_${DATASET_TAG}_c${MAX_CONCURRENCY}.jsonl" \
  --input "SPECSTREAM=$ACC_ROOT/bench/SPECSTREAM_1GPU_${DATASET_TAG}_c${MAX_CONCURRENCY}.jsonl" \
  --output "$ACC_ROOT/summary/${ACCURACY_DATASET}_accuracy_comparison.json" \
  | tee "$ACC_ROOT/summary/${ACCURACY_DATASET}_accuracy_comparison.txt"

EXPECTED_ROWS="$expected_rows" REPORT="$ACC_ROOT/summary/${ACCURACY_DATASET}_accuracy_comparison.json" \
  "$SPECSTREAM_PYTHON" - <<'PY'
import json, os
r=json.load(open(os.environ["REPORT"],encoding="utf-8"))
n=int(os.environ["EXPECTED_ROWS"])
assert r["methods"]["SGLANG_SD"]["samples"] == n
assert r["methods"]["SPECSTREAM"]["samples"] == n
assert r["methods"]["SGLANG_SD"]["failures"] == 0
assert r["methods"]["SPECSTREAM"]["failures"] == 0
print("MANIFEST_ACCURACY_PAIR_GATE=PASS", n)
PY

printf 'FINAL_RESULT=%s\n' "$ACC_ROOT/summary/${ACCURACY_DATASET}_accuracy_comparison.json"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
