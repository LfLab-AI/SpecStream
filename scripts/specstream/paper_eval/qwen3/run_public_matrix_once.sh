#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

: "${REPO:=/root/lifei/SpecStream}"
: "${QWEN3_DATA_ROOT:=$REPO/specstream_prepared/qwen3_offline}"
: "${MODEL_TAG:=qwen3_0p6b_8b}"
: "${RESULT_ROOT:=$REPO/results/${MODEL_TAG}_public_once_$(date +%Y%m%d_%H%M%S)}"
: "${CASE_TIMEOUT_S:=3600}"
: "${ALLOW_LONG_CASE:=0}"
: "${SPECSTREAM_PYTHON:?Source the preflight runtime_env.sh first}"
export MODEL_TAG

cd "$REPO"
mkdir -p "$RESULT_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}

GSM8K_QWEN3="$QWEN3_DATA_ROOT/gsm8k_qwen3_nothink_sharegpt.json"
LONGBENCH_QWEN3="$QWEN3_DATA_ROOT/longbench_v2_qwen3_8b_8k32k_sharegpt.json"
MRCR_QWEN3="$QWEN3_DATA_ROOT/mrcr_qwen3_16k32k_sharegpt.json"
for path in "$GSM8K_QWEN3" "$LONGBENCH_QWEN3" "$MRCR_QWEN3"; do
  test -s "$path" || { echo "ERROR: missing manifest $path" >&2; return 2; }
done

cp "$QWEN3_DATA_ROOT/dataset_sha256.txt" "$RESULT_ROOT/env/dataset_sha256.txt"
git rev-parse HEAD > "$RESULT_ROOT/env/git_commit.txt"
git status --short > "$RESULT_ROOT/env/git_status.txt"

# Four-method paper matrix.  HiCache is intentionally excluded: it is a
# reusable-prefix hierarchy, not guaranteed active-History KV offload.
METHODS=(SPECSTREAM_1GPU SGLANG_SD AR SGLANG_SD_KV_OFFLOAD)
DATASETS=(longbench_v2 mrcr16_32) ##gsm8k
printf '%s\n' "${METHODS[@]}" > "$RESULT_ROOT/env/matrix_methods.txt"
target_gpu_count="${TARGET_TP_SIZE:-1}"
cat > "$RESULT_ROOT/env/matrix_design.tsv" <<EOF
order	method	engine	kv_management	q	chunks_per_transfer	layer_prefetch	buffers	dynamic_q	cross_query_cohort	pcie_slack	target_tp	draft_placement	unique_gpu_count
1	AR	autoregressive	disabled	disabled	disabled	disabled	disabled	disabled	disabled	disabled	${target_gpu_count}	disabled	${target_gpu_count}
2	SGLANG_SD	STANDALONE	disabled	4	disabled	disabled	disabled	disabled	disabled	disabled	${target_gpu_count}	in_process	${target_gpu_count}
3	SGLANG_SD_KV_OFFLOAD	SPECTRE_ordinary	cpu_history_independent_streaming	4	1	enabled	2	disabled	disabled	disabled	${target_gpu_count}	colocated_rank_${COLOCATED_TP_RANK:-0}	${target_gpu_count}
4	SPECSTREAM_1GPU	SPECTRE_parallel	cpu_history_grouped_streaming	2,4,6,8	4	enabled	2	enabled	enabled	enabled	${target_gpu_count}	colocated_rank_${COLOCATED_TP_RANK:-0}	${target_gpu_count}
EOF
declare -A PATHS=(
  [gsm8k]="$GSM8K_QWEN3"
  [longbench_v2]="$LONGBENCH_QWEN3"
  [mrcr16_32]="$MRCR_QWEN3"
)
declare -A COUNTS
COUNTS[gsm8k]=$("$SPECSTREAM_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$GSM8K_QWEN3")
COUNTS[longbench_v2]=$("$SPECSTREAM_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$LONGBENCH_QWEN3")
COUNTS[mrcr16_32]=$("$SPECSTREAM_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$MRCR_QWEN3")
for dataset in gsm8k longbench_v2 mrcr16_32; do
  (( COUNTS[$dataset] > 0 )) || { echo "ERROR: empty dataset $dataset" >&2; return 2; }
done
declare -A OUTPUTS=([gsm8k]=256 [longbench_v2]=256 [mrcr16_32]=256)
declare -A CONCURRENCIES=([gsm8k]=8 [longbench_v2]=8 [mrcr16_32]=4)

printf 'dataset\tmethod\tsmoke_seconds\tprojected_case_seconds\tformal_status\n' \
  > "$RESULT_ROOT/summary/execution_plan.tsv"
FORMAL_RESULTS=()

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    smoke_tag="smoke_${dataset}"
    METHOD="$method" DATASET_TAG="$smoke_tag" DATASET_PATH="${PATHS[$dataset]}" \
      NUM_PROMPTS=8 OUTPUT_LEN=32 MAX_CONCURRENCY="${CONCURRENCIES[$dataset]}" \
      WARMUP_REQUESTS=1 \
      REQUEST_RATE=inf SEED=1 RESULT_ROOT="$RESULT_ROOT" CASE_TIMEOUT_S=1200 \
      bash scripts/specstream/paper_eval/qwen3/run_public_once.sh

    timing="$RESULT_ROOT/logs/${method}_${smoke_tag}_c${CONCURRENCIES[$dataset]}/timing.env"
    test -s "$timing"
    # shellcheck disable=SC1090
    source "$timing"
    setup_s=$((total_elapsed_s - bench_elapsed_s))
    work_ratio=$(( COUNTS[$dataset] * OUTPUTS[$dataset] / (8 * 32) ))
    (( work_ratio > 0 )) || work_ratio=1
    projected=$((setup_s + bench_elapsed_s * work_ratio))

    if (( projected > 3600 )) && [[ "$ALLOW_LONG_CASE" != 1 ]]; then
      printf '%s\t%s\t%s\t%s\tPAUSED_ESTIMATE_OVER_1H\n' \
        "$dataset" "$method" "$bench_elapsed_s" "$projected" \
        | tee -a "$RESULT_ROOT/summary/execution_plan.tsv"
      echo "LONG_CASE_APPROVAL_REQUIRED dataset=$dataset method=$method projected_seconds=$projected"
      echo "Re-run with ALLOW_LONG_CASE=1 only after the user approves this case."
      return 42
    fi

    printf '%s\t%s\t%s\t%s\tSTARTED\n' \
      "$dataset" "$method" "$bench_elapsed_s" "$projected" \
      | tee -a "$RESULT_ROOT/summary/execution_plan.tsv"
    METHOD="$method" DATASET_TAG="$dataset" DATASET_PATH="${PATHS[$dataset]}" \
      NUM_PROMPTS="${COUNTS[$dataset]}" OUTPUT_LEN="${OUTPUTS[$dataset]}" \
      MAX_CONCURRENCY="${CONCURRENCIES[$dataset]}" WARMUP_REQUESTS=4 \
      REQUEST_RATE=inf SEED=1 \
      RESULT_ROOT="$RESULT_ROOT" CASE_TIMEOUT_S="$CASE_TIMEOUT_S" \
      bash scripts/specstream/paper_eval/qwen3/run_public_once.sh
    FORMAL_RESULTS+=(
      "$RESULT_ROOT/bench/${method}_${dataset}_c${CONCURRENCIES[$dataset]}.jsonl"
    )
  done
done

"$SPECSTREAM_PYTHON" scripts/specstream/summarize_benchmarks.py \
  "${FORMAL_RESULTS[@]}" | tee "$RESULT_ROOT/summary/benchmark_summary.tsv"
"$SPECSTREAM_PYTHON" scripts/specstream/summarize_specstream_profile.py \
  "$RESULT_ROOT"/profiles/*.csv | tee "$RESULT_ROOT/summary/specstream_profile_summary.tsv"

echo "QWEN3_PUBLIC_MATRIX_ONCE=PASS"
echo "RESULT_ROOT=$RESULT_ROOT"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
