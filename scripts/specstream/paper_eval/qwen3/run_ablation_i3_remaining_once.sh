#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

: "${ABLATION_ROOT:?Set the formal ablation result root}"

REPO="${REPO:-/root/lifei/SpecStream}"
PREFLIGHT_RUNTIME="${PREFLIGHT_RUNTIME:-$ABLATION_ROOT/preflight/runtime_env.sh}"
CURRENT_PID="${CURRENT_PID:-}"
TPC_LIST="${TPC_LIST:-34 40}"

cd "$REPO"
source "$PREFLIGHT_RUNTIME"
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"

wait_gpu_idle() {
  local deadline=$((SECONDS + 600))
  local stable=0
  local process_count max_memory
  while (( SECONDS < deadline )); do
    process_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
    max_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} $1>m{m=$1} END{print m}')
    if (( process_count == 0 && max_memory <= 256 )); then
      stable=$((stable + 1))
      (( stable >= 3 )) && return 0
    else
      stable=0
    fi
    sleep 5
  done
  echo "ERROR: GPUs did not become idle within 600 seconds" >&2
  nvidia-smi >&2 || true
  return 2
}

if [[ -n "$CURRENT_PID" ]]; then
  echo "Waiting for existing case pid=$CURRENT_PID"
  while kill -0 "$CURRENT_PID" 2>/dev/null; do
    sleep 30
  done
fi
wait_gpu_idle

TPC_ROOT="$ABLATION_ROOT/i3_tpc"
mkdir -p "$TPC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
for tpc in $TPC_LIST; do
  wait_gpu_idle
  out="$TPC_ROOT/bench/C_I3_C_tpc${tpc}_c8.jsonl"
  [[ ! -e "$out" ]] || {
    echo "ERROR: refusing to repeat existing case: $out" >&2
    return 2
  }
  echo "RUN_I3_TPC method=C tpc=$tpc input_len=16384 output_len=128 prompts=64 concurrency=8 q_max=4 dynamic_q=on cohort=on cohort_max=8 cohort_delay_us=200 coexec=on seed=1" \
    | tee -a "$TPC_ROOT/env/execution_order.log"
  METHOD=C \
  FINAL_DRAFT_TPCS="$tpc" \
  DATASET_NAME=random-ids \
  DATASET_TAG="I3_C_tpc${tpc}" \
  INPUT_LEN=16384 \
  OUTPUT_LEN=128 \
  NUM_PROMPTS=64 \
  MAX_CONCURRENCY=8 \
  WARMUP_REQUESTS=4 \
  REQUEST_RATE=inf \
  SEED=1 \
  RESULT_ROOT="$TPC_ROOT" \
  CASE_TIMEOUT_S=3600 \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh \
    2>&1 | tee "$TPC_ROOT/logs/I3_C_tpc${tpc}.console.log"
done

ABC_ROOT="$ABLATION_ROOT/i3_abc"
mkdir -p "$ABC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
for method in A B; do
  wait_gpu_idle
  out="$ABC_ROOT/bench/${method}_I3_${method}_16k_c8_c8.jsonl"
  [[ ! -e "$out" ]] || {
    echo "ERROR: refusing to repeat existing case: $out" >&2
    return 2
  }
  echo "RUN_I3_ABC method=$method tpc=NA input_len=16384 output_len=128 prompts=64 concurrency=8 q_max=4 dynamic_q=on cohort=on cohort_max=8 cohort_delay_us=200 seed=1" \
    | tee -a "$ABC_ROOT/env/execution_order.log"
  METHOD="$method" \
  FINAL_DRAFT_TPCS=34 \
  DATASET_NAME=random-ids \
  DATASET_TAG="I3_${method}_16k_c8" \
  INPUT_LEN=16384 \
  OUTPUT_LEN=128 \
  NUM_PROMPTS=64 \
  MAX_CONCURRENCY=8 \
  WARMUP_REQUESTS=4 \
  REQUEST_RATE=inf \
  SEED=1 \
  RESULT_ROOT="$ABC_ROOT" \
  CASE_TIMEOUT_S=3600 \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh \
    2>&1 | tee "$ABC_ROOT/logs/I3_${method}.console.log"
done

printf '%s\n' "$TPC_ROOT/bench/C_I3_C_tpc34_c8.jsonl" \
  > "$ABC_ROOT/env/C_REUSED_FROM.txt"
touch "$ABLATION_ROOT/I3_FORMAL_COMPLETE"
echo "I3_FORMAL_COMPLETE=1"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
