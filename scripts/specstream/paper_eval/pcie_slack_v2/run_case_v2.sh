#!/usr/bin/env bash
set -euo pipefail

: "${METHOD:?}"
: "${BENCH_CMD:?}"
: "${CASE_ROOT:?}"
: "${PROFILE_CSV:?}"
: "${SPECSTREAM_PYTHON:?}"
: "${TARGET_MODEL:?}"
: "${DRAFT_MODEL:?}"
: "${COLOCATED_UUID:?}"
: "${TARGET_UUID:?}"
: "${DRAFT_UUID:?}"

mkdir -p "$CASE_ROOT" "$(dirname "$PROFILE_CSV")"
SPEC_STEPS=$((VERIFY_Q-1))
MODE=parallel
NEEDS_SMCTRL=0
ALLOW_CAL_OVERLAP=0

case "$METHOD" in
  A) TARGET_VISIBLE="$COLOCATED_UUID"; DRAFT_VISIBLE="$COLOCATED_UUID"; MODE=ordinary ;;
  B) TARGET_VISIBLE="$TARGET_UUID"; DRAFT_VISIBLE="$DRAFT_UUID"; MODE=parallel ;;
  C) TARGET_VISIBLE="$COLOCATED_UUID"; DRAFT_VISIBLE="$COLOCATED_UUID"; NEEDS_SMCTRL=1; test -s "$RESOURCE_PROFILE" ;;
  CAL_BASE) TARGET_VISIBLE="$COLOCATED_UUID"; DRAFT_VISIBLE="$COLOCATED_UUID"; NEEDS_SMCTRL=1 ;;
  CAL_OVERLAP) TARGET_VISIBLE="$COLOCATED_UUID"; DRAFT_VISIBLE="$COLOCATED_UUID"; NEEDS_SMCTRL=1; ALLOW_CAL_OVERLAP=1 ;;
  *) echo "ERROR: unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

if (( NEEDS_SMCTRL )); then
  : "${SPECSTREAM_SMCTRL_VALIDATED:?}"
  : "${SMCTRL_MASK_SCOPE:?}"
  : "${SPECSTREAM_MPS_PIPE:?}"
  : "${SPECSTREAM_MPS_LOG:?}"
  [[ "$SPECSTREAM_SMCTRL_VALIDATED" == 1 ]]
  [[ "$SMCTRL_MASK_SCOPE" == stream || "$SMCTRL_MASK_SCOPE" == global ]]
  env CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
      bash -c 'echo get_server_list | nvidia-cuda-mps-control >/dev/null'
fi

port_open(){ (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
for p in "$TARGET_PORT" "$DRAFT_PORT"; do
  port_open "$p" && { echo "ERROR: stale server on $p" >&2; exit 2; } || true
done

TARGET_ARGS=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --model-path "$TARGET_MODEL" --port "$TARGET_PORT" --tp-size 1
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$TARGET_MEM_FRACTION"
  --skip-server-warmup --attention-backend fa3
  --speculative-algorithm SPECTRE --spectre-role target
  --speculative-num-steps "$SPEC_STEPS" --speculative-eagle-topk 1
  --speculative-num-draft-tokens "$VERIFY_Q" --page-size 1
  --spectre-fixed-q-mode "$MODE" --spectre-require-draft
  --spectre-draft-timeout-action fallback --spectre-recv-timeout-ms 5000
  --spectre-initial-recv-timeout-ms 15000
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule
  --specstream-enabled --specstream-chunk-tokens 2048 --specstream-num-buffers 2
  --specstream-chunks-per-transfer 4 --specstream-layer-prefetch
  --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
  --specstream-cpu-memory-gb 64 --specstream-gpu-reserve-mb 1024
  --specstream-profile-path "$PROFILE_CSV" --log-level debug
)

if [[ "${REFERENCE_ATTENTION:-0}" == 1 ]]; then
  TARGET_ARGS+=(--specstream-reference-attention)
else
  TARGET_ARGS+=(--no-specstream-reference-attention)
fi
[[ "${SHADOW_ATTENTION:-0}" == 1 ]] && TARGET_ARGS+=(--specstream-shadow-attention)

DRAFT_ARGS=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT" --tp-size 1
  --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$DRAFT_MEM_FRACTION"
  --skip-server-warmup --attention-backend fa3
  --speculative-algorithm SPECTRE --spectre-role draft
  --speculative-num-steps "$SPEC_STEPS" --speculative-eagle-topk 1
  --speculative-num-draft-tokens "$VERIFY_Q"
  --spectre-draft-priority --spectre-max-draft-priority-steps "$((VERIFY_Q*2))"
  --disable-overlap-schedule --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
  --log-level debug
)

if (( NEEDS_SMCTRL )); then
  TARGET_ARGS+=(--specstream-pcie-slack-coexec --specstream-smctrl-enabled
    --specstream-smctrl-library "$SMCTRL_LIB" --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE"
    --specstream-coexec-require-mps --specstream-grant-token-quantum 1
    --specstream-coexec-target-slowdown-budget 0.05 --specstream-coexec-guard-us 200)
  DRAFT_ARGS+=(--specstream-smctrl-enabled --specstream-smctrl-library "$SMCTRL_LIB"
    --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE")
  if [[ "$METHOD" == C ]]; then
    TARGET_ARGS+=(--specstream-coexec-resource-profile-path "$RESOURCE_PROFILE")
  else
    TARGET_ARGS+=(--specstream-smctrl-calibration-tpcs "$CALIBRATION_TPCS")
    (( ALLOW_CAL_OVERLAP )) && TARGET_ARGS+=(--specstream-smctrl-calibration-allow-overlap)
  fi
fi

printf '%q ' "${TARGET_ARGS[@]}" > "$CASE_ROOT/target_command.txt"; printf '\n' >> "$CASE_ROOT/target_command.txt"
printf '%q ' "${DRAFT_ARGS[@]}" > "$CASE_ROOT/draft_command.txt"; printf '\n' >> "$CASE_ROOT/draft_command.txt"

target_pid=''; draft_pid=''; monitor_pid=''
cleanup(){
  [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true
  for pid in "$draft_pid" "$target_pid"; do [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 2
  for pid in "$draft_pid" "$target_pid"; do
    [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null || true
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

STARTED_PID=""
start_role(){
  local role="$1" visible="$2"; shift 2; local -a cmd=("$@")
  if (( NEEDS_SMCTRL )); then
    setsid env CUDA_VISIBLE_DEVICES="$visible" \
      CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
      "${cmd[@]}" > "$CASE_ROOT/$role.log" 2>&1 &
  else
    setsid env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
      -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u CUDA_MPS_CLIENT_PRIORITY \
      CUDA_VISIBLE_DEVICES="$visible" "${cmd[@]}" > "$CASE_ROOT/$role.log" 2>&1 &
  fi
  STARTED_PID=$!
}

wait_ready(){
  local role="$1"
  local port="$2"
  local pid="$3"
  log="$CASE_ROOT/$role.log" deadline=$((SECONDS+300))
  until grep -q 'The server is fired up and ready to roll!' "$log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || { echo "ERROR: $role exited"; tail -n 180 "$log"; return 1; }
    (( SECONDS < deadline )) || { echo "ERROR: $role timeout"; tail -n 180 "$log"; return 1; }
    sleep 1
  done
  curl -fsS "http://127.0.0.1:$port/health" >/dev/null
}

echo "[$METHOD] target=$TARGET_VISIBLE draft=$DRAFT_VISIBLE"
start_role target "$TARGET_VISIBLE" "${TARGET_ARGS[@]}"
target_pid="$STARTED_PID"
wait_ready target "$TARGET_PORT" "$target_pid"

start_role draft "$DRAFT_VISIBLE" "${DRAFT_ARGS[@]}"
draft_pid="$STARTED_PID"
wait_ready draft "$DRAFT_PORT" "$draft_pid"
sleep "${DRAFT_REGISTRATION_WAIT_S:-5}"

nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory --format=csv > "$CASE_ROOT/process_placement.csv"
nvidia-smi dmon -s pucvmet -d 1 -o DT > "$CASE_ROOT/gpu_dmon.log" 2>&1 & monitor_pid=$!

set +e
env -u CUDA_VISIBLE_DEVICES -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
    -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u CUDA_MPS_CLIENT_PRIORITY \
    SPECSTREAM_PYTHON="$SPECSTREAM_PYTHON" PYTHONUNBUFFERED=1 TQDM_MININTERVAL=0.5 \
    bash -c "$BENCH_CMD" 2>&1 | tee "$CASE_ROOT/benchmark.log"
bench_status=${PIPESTATUS[0]}
set -e

(( bench_status == 0 )) || { echo "ERROR: benchmark status=$bench_status"; tail -n 160 "$CASE_ROOT/target.log"; tail -n 160 "$CASE_ROOT/draft.log"; exit "$bench_status"; }

grep -E 'Scheduler hit an exception|CUDA out of memory|Traceback \(most recent call last\)' \
  "$CASE_ROOT/target.log" "$CASE_ROOT/draft.log" > "$CASE_ROOT/fatal_errors.txt" && {
    cat "$CASE_ROOT/fatal_errors.txt"; exit 3;
} || true

test -s "$PROFILE_CSV" || { echo "ERROR: missing profile $PROFILE_CSV"; exit 4; }
echo "PASS case=$METHOD profile=$PROFILE_CSV"
