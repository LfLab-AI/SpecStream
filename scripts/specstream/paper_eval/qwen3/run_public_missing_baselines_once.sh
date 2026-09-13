#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

: "${REPO:=/root/lifei/SpecStream}"
: "${QWEN3_DATA_ROOT:=$REPO/specstream_prepared/qwen3_offline}"
: "${RESULT_ROOT:?Set RESULT_ROOT to the completed legacy result directory}"
: "${CASE_TIMEOUT_S:=21600}"
: "${SPECSTREAM_PYTHON:?Source the preflight runtime_env.sh first}"

cd "$REPO"
mkdir -p "$RESULT_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}

METHODS=(AR SGLANG_SD_KV_OFFLOAD)
DATASETS=(gsm8k longbench_v2 mrcr16_32)
declare -A PATHS=(
  [gsm8k]="$QWEN3_DATA_ROOT/gsm8k_qwen3_nothink_sharegpt.json"
  [longbench_v2]="$QWEN3_DATA_ROOT/longbench_v2_qwen3_8b_8k32k_sharegpt.json"
  [mrcr16_32]="$QWEN3_DATA_ROOT/mrcr_qwen3_16k32k_sharegpt.json"
)
declare -A OUTPUTS=([gsm8k]=256 [longbench_v2]=256 [mrcr16_32]=256)
declare -A CONCURRENCIES=([gsm8k]=8 [longbench_v2]=4 [mrcr16_32]=4)

for dataset in "${DATASETS[@]}"; do
  test -s "${PATHS[$dataset]}" || {
    echo "ERROR: missing manifest ${PATHS[$dataset]}" >&2
    return 2
  }
done

printf '%s\n' AR SGLANG_SD SGLANG_SD_KV_OFFLOAD SPECSTREAM_1GPU \
  > "$RESULT_ROOT/env/matrix_methods.txt"
cat > "$RESULT_ROOT/env/matrix_design.tsv" <<'EOF'
order	method	engine	kv_management	q	chunks_per_transfer	layer_prefetch	buffers	dynamic_q	cross_query_cohort	pcie_slack	gpu_count
1	AR	autoregressive	disabled	disabled	disabled	disabled	disabled	disabled	disabled	disabled	1
2	SGLANG_SD	STANDALONE	disabled	4	disabled	disabled	disabled	disabled	disabled	disabled	1
3	SGLANG_SD_KV_OFFLOAD	SPECTRE_ordinary	cpu_history_independent_streaming	4	1	enabled	2	disabled	disabled	disabled	1
4	SPECSTREAM_1GPU	SPECTRE_parallel	cpu_history_grouped_streaming	2,4,6,8	4	enabled	2	enabled	enabled	enabled	1
EOF

for dataset in "${DATASETS[@]}"; do
  path="${PATHS[$dataset]}"
  concurrency="${CONCURRENCIES[$dataset]}"
  count=$("$SPECSTREAM_PYTHON" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "$path")
  (( count > 0 )) || { echo "ERROR: empty dataset $dataset" >&2; return 2; }

  for method in "${METHODS[@]}"; do
    formal_out="$RESULT_ROOT/bench/${method}_${dataset}_c${concurrency}.jsonl"
    if [[ -s "$formal_out" ]]; then
      echo "INFO: keep existing result $formal_out"
      continue
    fi

    METHOD="$method" DATASET_TAG="smoke_${dataset}" DATASET_NAME=sharegpt \
      DATASET_PATH="$path" NUM_PROMPTS=8 OUTPUT_LEN=32 \
      MAX_CONCURRENCY="$concurrency" WARMUP_REQUESTS=1 REQUEST_RATE=inf SEED=1 \
      RESULT_ROOT="$RESULT_ROOT" CASE_TIMEOUT_S=1200 \
      bash scripts/specstream/paper_eval/qwen3/run_public_once.sh

    METHOD="$method" DATASET_TAG="$dataset" DATASET_NAME=sharegpt \
      DATASET_PATH="$path" NUM_PROMPTS="$count" OUTPUT_LEN="${OUTPUTS[$dataset]}" \
      MAX_CONCURRENCY="$concurrency" WARMUP_REQUESTS=4 REQUEST_RATE=inf SEED=1 \
      RESULT_ROOT="$RESULT_ROOT" CASE_TIMEOUT_S="$CASE_TIMEOUT_S" \
      bash scripts/specstream/paper_eval/qwen3/run_public_once.sh
  done
done

FORMAL_RESULTS=()
for dataset in "${DATASETS[@]}"; do
  concurrency="${CONCURRENCIES[$dataset]}"
  for method in AR SGLANG_SD SGLANG_SD_KV_OFFLOAD SPECSTREAM_1GPU; do
    path="$RESULT_ROOT/bench/${method}_${dataset}_c${concurrency}.jsonl"
    test -s "$path" || { echo "ERROR: missing formal result $path" >&2; return 2; }
    FORMAL_RESULTS+=("$path")
  done
done
"$SPECSTREAM_PYTHON" scripts/specstream/summarize_benchmarks.py \
  "${FORMAL_RESULTS[@]}" | tee "$RESULT_ROOT/summary/benchmark_summary_new_matrix.tsv"

echo "QWEN3_PUBLIC_MISSING_BASELINES_ONCE=PASS"
echo "RESULT_ROOT=$RESULT_ROOT"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
