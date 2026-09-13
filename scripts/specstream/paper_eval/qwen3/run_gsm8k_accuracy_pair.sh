#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

for required_var in REPO SPECSTREAM_PYTHON TARGET_MODEL DRAFT_MODEL \
  TARGET_UUID DRAFT_UUID COLOCATED_UUID GSM8K_TEST GSM8K_FEWSHOT; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "ERROR: required environment variable is unset: $required_var" >&2
    return 2
  fi
done

[[ -s "$GSM8K_TEST" ]] || { echo "ERROR: GSM8K test JSONL missing: $GSM8K_TEST" >&2; return 2; }
[[ -s "$GSM8K_FEWSHOT" ]] || { echo "ERROR: GSM8K few-shot JSONL missing: $GSM8K_FEWSHOT" >&2; return 2; }
expected_rows=$(grep -cve '^$' "$GSM8K_TEST")
[[ "$expected_rows" == 1319 ]] || {
  echo "ERROR: expected 1319 GSM8K main/test rows, got $expected_rows" >&2
  return 2
}

ACC_ROOT="${ACC_ROOT:-$REPO/results/qwen3_gsm8k_accuracy_native_$(date +%Y%m%d_%H%M%S)}"
export ACC_ROOT
mkdir -p "$ACC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
git -C "$REPO" rev-parse HEAD > "$ACC_ROOT/env/git_commit.txt"
git -C "$REPO" status --short > "$ACC_ROOT/env/git_status.txt"
cp "$REPO/specstream_prepared/qwen3_offline/gsm8k_native_eval_manifest.json" \
  "$ACC_ROOT/env/" 2>/dev/null || true

for method in SGLANG_SD SPECSTREAM_1GPU; do
  echo "===== GSM8K method=$method rows=$expected_rows parallel=32 ====="
  METHOD="$method" \
  CLIENT_MODE=gsm8k_native \
  DATASET_NAME=gsm8k-native \
  DATASET_TAG=accuracy_gsm8k \
  DATASET_PATH="$GSM8K_TEST" \
  GSM8K_FEWSHOT_PATH="$GSM8K_FEWSHOT" \
  GSM8K_NUM_SHOTS=5 \
  NUM_PROMPTS=0 \
  OUTPUT_LEN=512 \
  MAX_CONCURRENCY=32 \
  WARMUP_REQUESTS=0 \
  REQUEST_RATE=inf \
  SEED=1 \
  RESULT_ROOT="$ACC_ROOT" \
  CASE_TIMEOUT_S=14400 \
  bash "$REPO/scripts/specstream/paper_eval/qwen3/run_public_once.sh"
done

"$SPECSTREAM_PYTHON" \
  "$REPO/scripts/specstream/paper_eval/qwen3/score_public_accuracy.py" \
  --dataset gsm8k \
  --input "SGLANG_SD=$ACC_ROOT/bench/SGLANG_SD_accuracy_gsm8k_c32.jsonl" \
  --input "SPECSTREAM=$ACC_ROOT/bench/SPECSTREAM_1GPU_accuracy_gsm8k_c32.jsonl" \
  --output "$ACC_ROOT/summary/gsm8k_accuracy_comparison.json" \
  | tee "$ACC_ROOT/summary/gsm8k_accuracy_comparison.txt"

EXPECTED_ROWS="$expected_rows" ACC_ROOT="$ACC_ROOT" "$SPECSTREAM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ACC_ROOT"])
expected = int(os.environ["EXPECTED_ROWS"])
raw = {
    "SGLANG_SD": root / "bench/SGLANG_SD_accuracy_gsm8k_c32.jsonl",
    "SPECSTREAM_1GPU": root / "bench/SPECSTREAM_1GPU_accuracy_gsm8k_c32.jsonl",
}
for method, path in raw.items():
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == expected, (method, len(rows), expected)
    assert len({row["sample_id"] for row in rows}) == expected
    assert sum(bool(row.get("error")) for row in rows) == 0
report = json.loads((root / "summary/gsm8k_accuracy_comparison.json").read_text(encoding="utf-8"))
assert report["methods"]["SGLANG_SD"]["samples"] == expected
assert report["methods"]["SPECSTREAM"]["samples"] == expected
assert report["methods"]["SGLANG_SD"]["failures"] == 0
assert report["methods"]["SPECSTREAM"]["failures"] == 0
print("GSM8K_ACCURACY_PAIR_GATE=PASS")
PY

printf 'GSM8K_RESULT_ROOT=%s\n' "$ACC_ROOT"
printf 'GSM8K_FINAL_RESULT=%s\n' "$ACC_ROOT/summary/gsm8k_accuracy_comparison.json"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
