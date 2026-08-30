#!/usr/bin/env bash
set -euo pipefail

: "${TEST_ROOT:?Source env_pcie_slack_isolated.sh first}"
: "${RESOURCE_PROFILE:?}"
: "${SPECSTREAM_SMCTRL_VALIDATED:?}"
: "${CUDA_MPS_PIPE_DIRECTORY:?}"
: "${CUDA_MPS_LOG_DIRECTORY:?}"

if [[ ! -s "$RESOURCE_PROFILE" ]]; then
  echo "ERROR: missing RESOURCE_PROFILE=$RESOURCE_PROFILE" >&2
  exit 2
fi
if ! echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "ERROR: MPS daemon unavailable" >&2
  exit 2
fi

orders=(
  "A B C"
  "B C A"
  "C A B"
  "A C B"
  "C B A"
)

for rep in 1 2 3; do
  read -r -a methods <<< "${orders[$((rep - 1))]}"
  for method in "${methods[@]}"; do
    export METHOD="$method"
    export CASE_ROOT="$TEST_ROOT/performance/$method/run_r${rep}"
    export PROFILE_CSV="$TEST_ROOT/profiles/performance_${method}_r${rep}.csv"
    export BENCH_CMD="TARGET_MODEL='$TARGET_MODEL' \
      DATASET_NAME=random-ids \
      INPUT_LEN=16384 \
      OUTPUT_LEN=128 \
      NUM_PROMPTS=64 \
      REQUEST_RATE=inf \
      MAX_CONCURRENCY=8 \
      RANGE_RATIO=1 \
      WARMUP_REQUESTS=4 \
      SEED=1 \
      CONTEXT_LEN=$CONTEXT_LENGTH \
      CASE_TAG=${method}_ctx16k_c8_r${rep} \
      OUTPUT_DIR='$TEST_ROOT/performance/$method/results' \
      RUN_ID=r${rep} \
      bash scripts/specstream/run_benchmark_case.sh"

    echo "RUN rep=$rep method=$method"
    bash scripts/specstream/paper_eval/run_pcie_slack_case.sh

    if [[ "$method" == C ]]; then
      python scripts/specstream/paper_eval/analyze_slack_fill_acks.py \
        --draft-log "$CASE_ROOT/draft.log" \
        --q "$VERIFY_Q" \
        --output "$TEST_ROOT/summary/performance_C_r${rep}_slack_ack.json" \
        --require-complete
    fi
  done
done
