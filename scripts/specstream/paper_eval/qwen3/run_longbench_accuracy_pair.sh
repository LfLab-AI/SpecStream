#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

for required_var in REPO SPECSTREAM_PYTHON DATASET_PATH MAX_CONCURRENCY ACC_ROOT; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "ERROR: required environment variable is unset: $required_var" >&2
    return 2
  fi
done

OUTPUT_LEN="${OUTPUT_LEN:-256}"
DATASET_TAG="${DATASET_TAG:-accuracy_longbench_v2}"
LONGBENCH_LIMIT="${LONGBENCH_LIMIT:-0}"
if (( OUTPUT_LEN < 256 )); then
  echo "ERROR: LongBench-v2 accuracy requires OUTPUT_LEN>=256; got $OUTPUT_LEN" >&2
  return 2
fi
if ! [[ "$LONGBENCH_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: LONGBENCH_LIMIT must be a non-negative integer" >&2
  return 2
fi
[[ -s "$DATASET_PATH" ]] || {
  echo "ERROR: LongBench-v2 manifest missing: $DATASET_PATH" >&2
  return 2
}

expected_rows=$(DATASET_PATH="$DATASET_PATH" "$SPECSTREAM_PYTHON" - <<'PY'
import json, os
rows = json.load(open(os.environ["DATASET_PATH"], encoding="utf-8"))
assert isinstance(rows, list) and rows
print(len(rows))
PY
)
if (( LONGBENCH_LIMIT > 0 && LONGBENCH_LIMIT < expected_rows )); then
  expected_rows="$LONGBENCH_LIMIT"
fi
mkdir -p "$ACC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}

for method in SGLANG_SD SPECSTREAM_1GPU; do
  METHOD="$method" \
  CLIENT_MODE=accuracy \
  ACCURACY_DATASET=longbench_v2 \
  DATASET_NAME=sharegpt \
  DATASET_TAG="$DATASET_TAG" \
  DATASET_PATH="$DATASET_PATH" \
  NUM_PROMPTS="$LONGBENCH_LIMIT" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  WARMUP_REQUESTS=0 \
  REQUEST_RATE=inf \
  SEED=1 \
  RESULT_ROOT="$ACC_ROOT" \
  CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-21600}" \
  bash "$REPO/scripts/specstream/paper_eval/qwen3/run_public_once.sh"

  result_json="$ACC_ROOT/bench/${method}_${DATASET_TAG}_c${MAX_CONCURRENCY}.jsonl"
  summary_json="$ACC_ROOT/summary/${method}_${DATASET_TAG}_c${MAX_CONCURRENCY}.json"
  "$SPECSTREAM_PYTHON" \
    "$REPO/scripts/specstream/paper_eval/qwen3/score_public_accuracy.py" \
    --dataset longbench_v2 \
    --input "$method=$result_json" \
    --output "$summary_json" \
    | tee "$ACC_ROOT/summary/${method}_${DATASET_TAG}_c${MAX_CONCURRENCY}.txt"
  printf 'LONGBENCH_METHOD_RESULT=%s\n' "$summary_json"
done

"$SPECSTREAM_PYTHON" \
  "$REPO/scripts/specstream/paper_eval/qwen3/score_public_accuracy.py" \
  --dataset longbench_v2 \
  --input "SGLANG_SD=$ACC_ROOT/bench/SGLANG_SD_${DATASET_TAG}_c${MAX_CONCURRENCY}.jsonl" \
  --input "SPECSTREAM_1GPU=$ACC_ROOT/bench/SPECSTREAM_1GPU_${DATASET_TAG}_c${MAX_CONCURRENCY}.jsonl" \
  --output "$ACC_ROOT/summary/longbench_v2_accuracy_comparison.json" \
  | tee "$ACC_ROOT/summary/longbench_v2_accuracy_comparison.txt"

EXPECTED_ROWS="$expected_rows" \
REPORT="$ACC_ROOT/summary/longbench_v2_accuracy_comparison.json" \
"$SPECSTREAM_PYTHON" - <<'PY'
import json, os
r = json.load(open(os.environ["REPORT"], encoding="utf-8"))
n = int(os.environ["EXPECTED_ROWS"])
for method in ("SGLANG_SD", "SPECSTREAM_1GPU"):
    assert r["methods"][method]["samples"] == n
    assert r["methods"][method]["failures"] == 0
    assert r["methods"][method]["scored_samples"] == n
print("LONGBENCH_ACCURACY_PAIR_GATE=PASS", n)
PY

printf 'LONGBENCH_FINAL_RESULT=%s\n' \
  "$ACC_ROOT/summary/longbench_v2_accuracy_comparison.json"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
