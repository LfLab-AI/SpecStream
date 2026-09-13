#!/usr/bin/env bash
set -euo pipefail

: "${TEST_ROOT:?Source env_fixed27.sh first}"
: "${SPECSTREAM_SMCTRL_VALIDATED:?}"
: "${SPECSTREAM_MPS_PIPE:?}"
: "${SPECSTREAM_MPS_LOG:?}"
: "${SMCTRL_MASK_SCOPE:?}"
: "${TOTAL_TPCS:?}"
: "${FIXED_DRAFT_TPCS:?Run: export FIXED_DRAFT_TPCS=<positive integer>}"
: "${TARGET_MODEL:?}"
: "${CONTEXT_LENGTH:?}"
: "${VERIFY_Q:?}"

[[ "$SPECSTREAM_SMCTRL_VALIDATED" == 1 ]] || {
  echo "ERROR: SMCTRL was not validated" >&2
  exit 2
}

[[ "$FIXED_DRAFT_TPCS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: invalid FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS" >&2
  exit 2
}

if (( FIXED_DRAFT_TPCS > TOTAL_TPCS )); then
  echo \
    "ERROR: FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS exceeds TOTAL_TPCS=$TOTAL_TPCS" \
    >&2
  exit 2
fi

CASE_RUNNER="scripts/specstream/paper_eval/pcie_slack_fixed27/run_case_fixed27.sh"
BENCH_RUNNER="scripts/specstream/paper_eval/pcie_slack_fixed27/run_benchmark_fixed27.sh"
ACK_ANALYZER="scripts/specstream/paper_eval/pcie_slack_fixed27/analyze_fixed27_acks.py"

test -x "$CASE_RUNNER"
test -x "$BENCH_RUNNER"
test -s "$ACK_ANALYZER"

env \
  CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
  CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
  bash -c 'echo get_server_list | nvidia-cuda-mps-control >/dev/null' || {
    echo "ERROR: session-local MPS daemon unavailable" >&2
    exit 2
  }

TPC_TAG="tpc${FIXED_DRAFT_TPCS}"

mkdir -p \
  "$TEST_ROOT/performance/A/results" \
  "$TEST_ROOT/performance/B/results" \
  "$TEST_ROOT/performance/C/results" \
  "$TEST_ROOT/profiles" \
  "$TEST_ROOT/logs" \
  "$TEST_ROOT/summary"

orders=(
  "A B C"
  "B C A"
  "C A B"
  "A C B"
  "C B A"
)

export REFERENCE_ATTENTION=0
export SHADOW_ATTENTION=0

for rep in 1 2 3 4 5; do
  read -r -a methods <<< "${orders[$((rep - 1))]}"

  for method in "${methods[@]}"; do
    export METHOD="$method"
    export CASE_ROOT="$TEST_ROOT/performance/$method/run_r${rep}"
    export PROFILE_CSV="$TEST_ROOT/profiles/performance_${method}_${TPC_TAG}_r${rep}.csv"
    export BENCH_CMD="TARGET_MODEL='$TARGET_MODEL' DATASET_NAME=random-ids INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=64 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 CONTEXT_LEN=$CONTEXT_LENGTH CASE_TAG=${method}_ctx16k_c8_${TPC_TAG}_r${rep} OUTPUT_DIR='$TEST_ROOT/performance/$method/results' RUN_ID=r${rep} bash '$BENCH_RUNNER'"

    echo \
      "RUN rep=$rep method=$method fixed_tpcs=$FIXED_DRAFT_TPCS test_root=$TEST_ROOT"

    bash "$CASE_RUNNER"

    if [[ "$method" == C ]]; then
      grep -F -- \
        "--specstream-smctrl-calibration-tpcs $FIXED_DRAFT_TPCS" \
        "$CASE_ROOT/target_command.txt" >/dev/null || {
          echo \
            "ERROR: C did not use FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS" \
            >&2
          exit 2
        }

      grep -F -- \
        '--specstream-smctrl-calibration-allow-overlap' \
        "$CASE_ROOT/target_command.txt" >/dev/null || {
          echo "ERROR: C did not enable overlap" >&2
          exit 2
        }

      "$SPECSTREAM_PYTHON" "$ACK_ANALYZER" \
        --draft-log "$CASE_ROOT/draft.log" \
        --q "$VERIFY_Q" \
        --fixed-tpcs "$FIXED_DRAFT_TPCS" \
        --output "$TEST_ROOT/summary/performance_C_r${rep}_${TPC_TAG}_ack.json" \
        --require-complete
    fi

    echo "PASS rep=$rep method=$method fixed_tpcs=$FIXED_DRAFT_TPCS"
  done
done

echo "ALL_15_VARIABLE_TPC_CASES=PASS tpcs=$FIXED_DRAFT_TPCS"
