#!/usr/bin/env bash
set -euo pipefail
: "${TEST_ROOT:?}"
: "${RESOURCE_PROFILE:?}"
: "${SPECSTREAM_SMCTRL_VALIDATED:?}"

orders=("A B C" "B C A" "C A B" "A C B" "C B A")
export REFERENCE_ATTENTION=0
export SHADOW_ATTENTION=0

for rep in 1 2 3 4 5; do
  read -r -a methods <<< "${orders[$((rep-1))]}"
  for method in "${methods[@]}"; do
    export METHOD="$method"
    export CASE_ROOT="$TEST_ROOT/performance/$method/run_r${rep}"
    export PROFILE_CSV="$TEST_ROOT/profiles/performance_${method}_r${rep}.csv"
    export BENCH_CMD="TARGET_MODEL='$TARGET_MODEL' DATASET_NAME=random-ids INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=64 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 CONTEXT_LEN=$CONTEXT_LENGTH CASE_TAG=${method}_ctx16k_c8_r${rep} OUTPUT_DIR='$TEST_ROOT/performance/$method/results' RUN_ID=r${rep} bash scripts/specstream/paper_eval/pcie_slack_v2/run_benchmark_v2.sh"
    echo "RUN rep=$rep method=$method"
    bash scripts/specstream/paper_eval/pcie_slack_v2/run_case_v2.sh
  done
done
