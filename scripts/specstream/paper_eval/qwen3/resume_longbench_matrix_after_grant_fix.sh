#!/usr/bin/env bash
set -euo pipefail

REPO=/root/lifei/SpecStream
cd "$REPO"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate spectre
source results/qwen3_0p6b_32b_preflight_20260906_170952/runtime_env.sh

export MODEL_TAG=qwen3_0p6b_32b
export SPECSTREAM_TARGET_MEM_FRACTION=0.62
export SPECSTREAM_DRAFT_MEM_FRACTION=0.80
export SPECSTREAM_TARGET_MAX_TOTAL_TOKENS=131072
export SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=196608
export SPECSTREAM_TARGET_MIN_KV_TOKENS=131072
export SPECSTREAM_DRAFT_MIN_KV_TOKENS=196608
export SPECSTREAM_PREFILL_MAX_REQUESTS=1
export SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192
export SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0
export SPECSTREAM_REQUIRE_SLACK_FILL=0

DATASET_PATH="$REPO/specstream_prepared/qwen3_offline/longbench_v2_qwen3_8b_8k32k_sharegpt.json"
test -s "$DATASET_PATH"
NUM_FORMAL=$("$SPECSTREAM_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$DATASET_PATH")
test "$NUM_FORMAL" -eq 131

RESULT_ROOT="$REPO/results/qwen3_0p6b_32b_longbench_matrix_resume_$(date +%Y%m%d_%H%M%S)"
export RESULT_ROOT
mkdir -p "$RESULT_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
printf '%s\n' "$RESULT_ROOT" > "$REPO/results/latest_longbench_resume.path"
exec > >(tee -a "$RESULT_ROOT/console.log") 2>&1
trap 'rc=$?; printf "LONG_BENCH_MATRIX_EXIT=%s\n" "$rc"; date -u +FINISHED_AT_UTC=%Y-%m-%dT%H:%M:%SZ' EXIT

echo "LONG_BENCH_MATRIX_RESULT_ROOT=$RESULT_ROOT"
echo "LONG_BENCH_MATRIX_STARTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' SGLANG_SD SPECSTREAM_1GPU AR SGLANG_SD_KV_OFFLOAD > "$RESULT_ROOT/env/matrix_methods.txt"
sha256sum "$DATASET_PATH" > "$RESULT_ROOT/env/dataset_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/env/git_commit.txt"
git status --short > "$RESULT_ROOT/env/git_status.txt"
printf 'phase\tmethod\tstatus\n' > "$RESULT_ROOT/summary/execution_plan.tsv"

METHODS=(SGLANG_SD SPECSTREAM_1GPU AR SGLANG_SD_KV_OFFLOAD)
for method in "${METHODS[@]}"; do
  echo "===== SMOKE START method=$method ====="
  METHOD="$method" DATASET_TAG=smoke_longbench_v2 DATASET_NAME=sharegpt \
    DATASET_PATH="$DATASET_PATH" NUM_PROMPTS=8 OUTPUT_LEN=32 \
    MAX_CONCURRENCY=4 WARMUP_REQUESTS=1 REQUEST_RATE=inf SEED=1 \
    CASE_TIMEOUT_S=1200 bash scripts/specstream/paper_eval/qwen3/run_public_once.sh
  marker="$RESULT_ROOT/logs/${method}_smoke_longbench_v2_c4/case_complete.marker"
  fatal="$RESULT_ROOT/logs/${method}_smoke_longbench_v2_c4/fatal_errors.txt"
  test -s "$marker"
  test -e "$fatal" && test ! -s "$fatal"
  printf 'smoke\t%s\tPASS\n' "$method" | tee -a "$RESULT_ROOT/summary/execution_plan.tsv"
done

FORMAL_RESULTS=()
for method in "${METHODS[@]}"; do
  echo "===== FORMAL START method=$method prompts=$NUM_FORMAL ====="
  METHOD="$method" DATASET_TAG=longbench_v2 DATASET_NAME=sharegpt \
    DATASET_PATH="$DATASET_PATH" NUM_PROMPTS="$NUM_FORMAL" OUTPUT_LEN=256 \
    MAX_CONCURRENCY=4 WARMUP_REQUESTS=4 REQUEST_RATE=inf SEED=1 \
    CASE_TIMEOUT_S=21600 bash scripts/specstream/paper_eval/qwen3/run_public_once.sh
  marker="$RESULT_ROOT/logs/${method}_longbench_v2_c4/case_complete.marker"
  fatal="$RESULT_ROOT/logs/${method}_longbench_v2_c4/fatal_errors.txt"
  bench="$RESULT_ROOT/bench/${method}_longbench_v2_c4.jsonl"
  test -s "$marker"
  test -s "$bench"
  test -e "$fatal" && test ! -s "$fatal"
  FORMAL_RESULTS+=("$bench")
  printf 'formal\t%s\tPASS\n' "$method" | tee -a "$RESULT_ROOT/summary/execution_plan.tsv"
done

"$SPECSTREAM_PYTHON" scripts/specstream/summarize_benchmarks.py \
  "${FORMAL_RESULTS[@]}" | tee "$RESULT_ROOT/summary/benchmark_summary.tsv"
"$SPECSTREAM_PYTHON" scripts/specstream/summarize_specstream_profile.py \
  "$RESULT_ROOT"/profiles/*.csv | tee "$RESULT_ROOT/summary/specstream_profile_summary.tsv"
date -u +COMPLETED_AT_UTC=%Y-%m-%dT%H:%M:%SZ > "$RESULT_ROOT/matrix_complete.marker"
echo "QWEN3_LONGBENCH_MATRIX_RESUME=PASS"
echo "RESULT_ROOT=$RESULT_ROOT"
