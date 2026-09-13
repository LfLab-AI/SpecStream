#!/usr/bin/env bash
set -euo pipefail

: "${METHOD:?Set METHOD=A, B, or C}"
: "${BENCH_CMD:?Set the complete benchmark command}"
: "${CASE_ROOT:?}"
: "${PROFILE_CSV:?}"
: "${SPECSTREAM_PYTHON:?}"
: "${TARGET_MODEL:?}"
: "${DRAFT_MODEL:?}"
: "${TARGET_PORT:?}"
: "${DRAFT_PORT:?}"
: "${ZMQ_PORT:?}"
: "${CONTEXT_LENGTH:?}"
: "${VERIFY_Q:?}"
: "${TARGET_MEM_FRACTION:?}"
: "${DRAFT_MEM_FRACTION:?}"
: "${COLOCATED_UUID:?}"
: "${TARGET_UUID:?}"
: "${DRAFT_UUID:?}"
: "${FIXED_DRAFT_TPCS:?}"
: "${SMCTRL_LIB:?}"

mkdir -p "$CASE_ROOT" "$(dirname "$PROFILE_CSV")"

[[ "$METHOD" == A || "$METHOD" == B || "$METHOD" == C ]] || {
  echo "ERROR: METHOD must be A, B, or C; got $METHOD" >&2
  exit 2
}
[[ "$FIXED_DRAFT_TPCS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: invalid FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS" >&2
  exit 2
}
(( VERIFY_Q >= 2 )) || {
  echo "ERROR: VERIFY_Q must be >=2" >&2
  exit 2
}

# Per-token GRANT/ACK debug formatting produces multi-megabyte logs and is not
# part of the serving algorithm.  Keep formal performance runs at INFO by
# default; correctness investigations can opt back in with
# SPECSTREAM_SERVER_LOG_LEVEL=debug.
SPECSTREAM_SERVER_LOG_LEVEL="${SPECSTREAM_SERVER_LOG_LEVEL:-info}"

SPEC_STEPS=$((VERIFY_Q - 1))
MODE=parallel
NEEDS_SMCTRL=0

case "$METHOD" in
  A)
    TARGET_VISIBLE="$COLOCATED_UUID"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    MODE=ordinary
    ;;
  B)
    TARGET_VISIBLE="$TARGET_UUID"
    DRAFT_VISIBLE="$DRAFT_UUID"
    MODE=parallel
    [[ "$TARGET_VISIBLE" != "$DRAFT_VISIBLE" ]] || {
      echo "ERROR: method B requires different Target/Draft GPUs" >&2
      exit 2
    }
    ;;
  C)
    TARGET_VISIBLE="$COLOCATED_UUID"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    MODE=parallel
    NEEDS_SMCTRL=1
    ;;
esac

if (( NEEDS_SMCTRL )); then
  : "${SPECSTREAM_SMCTRL_VALIDATED:?}"
  : "${SMCTRL_MASK_SCOPE:?}"
  : "${SPECSTREAM_MPS_PIPE:?}"
  : "${SPECSTREAM_MPS_LOG:?}"
  : "${TOTAL_TPCS:?}"

  [[ "$SPECSTREAM_SMCTRL_VALIDATED" == 1 ]] || {
    echo "ERROR: SMCTRL validator gate did not pass" >&2
    exit 2
  }
  [[ "$SMCTRL_MASK_SCOPE" == stream || "$SMCTRL_MASK_SCOPE" == global ]] || {
    echo "ERROR: invalid SMCTRL_MASK_SCOPE=$SMCTRL_MASK_SCOPE" >&2
    exit 2
  }
  (( FIXED_DRAFT_TPCS <= TOTAL_TPCS )) || {
    echo "ERROR: fixed TPCs exceed total TPCs" >&2
    exit 2
  }

  env CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
      CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
      bash -c 'echo get_server_list | nvidia-cuda-mps-control >/dev/null'
fi

port_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

for port in "$TARGET_PORT" "$DRAFT_PORT"; do
  port_open "$port" && {
    echo "ERROR: stale server is listening on port $port" >&2
    exit 2
  } || true
done

TARGET_ARGS=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --model-path "$TARGET_MODEL"
  --port "$TARGET_PORT"
  --tp-size 1
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$TARGET_MEM_FRACTION"
  --skip-server-warmup
  --attention-backend fa3
  --speculative-algorithm SPECTRE
  --spectre-role target
  --speculative-num-steps "$SPEC_STEPS"
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens "$VERIFY_Q"
  --page-size 1
  --spectre-fixed-q-mode "$MODE"
  --spectre-require-draft
  --spectre-draft-timeout-action fallback
  --spectre-recv-timeout-ms 5000
  --spectre-initial-recv-timeout-ms 15000
  --spectre-zmq-addr 127.0.0.1
  --spectre-zmq-port "$ZMQ_PORT"
  --disable-radix-cache
  --disable-cuda-graph
  --disable-overlap-schedule
  --specstream-enabled
  --specstream-chunk-tokens 2048
  --specstream-num-buffers 2
  --specstream-chunks-per-transfer 4
  --specstream-layer-prefetch
  --specstream-active-tail-tokens 512
  --specstream-min-history-tokens 8192
  --specstream-cpu-memory-gb 64
  --specstream-gpu-reserve-mb 1024
  --specstream-profile-path "$PROFILE_CSV"
  --log-level "$SPECSTREAM_SERVER_LOG_LEVEL"
)

if [[ "${REFERENCE_ATTENTION:-0}" == 1 ]]; then
  TARGET_ARGS+=(--specstream-reference-attention)
else
  TARGET_ARGS+=(--no-specstream-reference-attention)
fi

if [[ "${SHADOW_ATTENTION:-0}" == 1 ]]; then
  TARGET_ARGS+=(--specstream-shadow-attention)
fi

DRAFT_ARGS=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --model-path "$DRAFT_MODEL"
  --port "$DRAFT_PORT"
  --tp-size 1
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$DRAFT_MEM_FRACTION"
  --skip-server-warmup
  --attention-backend fa3
  --speculative-algorithm SPECTRE
  --spectre-role draft
  --speculative-num-steps "$SPEC_STEPS"
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens "$VERIFY_Q"
  --spectre-draft-priority
  --spectre-max-draft-priority-steps "$((VERIFY_Q * 2))"
  --disable-overlap-schedule
  --spectre-zmq-addr 127.0.0.1
  --spectre-zmq-port "$ZMQ_PORT"
  --log-level "$SPECSTREAM_SERVER_LOG_LEVEL"
)

if (( NEEDS_SMCTRL )); then
  TARGET_ARGS+=(
    --specstream-pcie-slack-coexec
    --specstream-smctrl-enabled
    --specstream-smctrl-library "$SMCTRL_LIB"
    --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE"
    --specstream-coexec-require-mps
    --specstream-grant-token-quantum 1
    --specstream-coexec-target-slowdown-budget 0.05
    --specstream-coexec-guard-us 200
    --specstream-smctrl-calibration-tpcs "$FIXED_DRAFT_TPCS"
    --specstream-smctrl-calibration-allow-overlap
  )

  DRAFT_ARGS+=(
    --specstream-smctrl-enabled
    --specstream-smctrl-library "$SMCTRL_LIB"
    --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE"
  )
fi

printf '%q ' "${TARGET_ARGS[@]}" > "$CASE_ROOT/target_command.txt"
printf '\n' >> "$CASE_ROOT/target_command.txt"
printf '%q ' "${DRAFT_ARGS[@]}" > "$CASE_ROOT/draft_command.txt"
printf '\n' >> "$CASE_ROOT/draft_command.txt"

cat > "$CASE_ROOT/method_metadata.txt" <<EOF2
method=$METHOD
fixed_draft_tpcs=$FIXED_DRAFT_TPCS
target_visible=$TARGET_VISIBLE
draft_visible=$DRAFT_VISIBLE
smctrl_enabled=$NEEDS_SMCTRL
server_log_level=$SPECSTREAM_SERVER_LOG_LEVEL
EOF2

target_pid=''
draft_pid=''
monitor_pid=''

cleanup() {
  [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true
  for pid in "$draft_pid" "$target_pid"; do
    [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "$draft_pid" "$target_pid"; do
    [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null || true
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

STARTED_PID=''
start_role() {
  local role="$1"
  local visible="$2"
  shift 2
  local -a cmd=("$@")

  if (( NEEDS_SMCTRL )); then
    setsid env \
      -u MASK_OFF \
      CUDA_VISIBLE_DEVICES="$visible" \
      CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
      CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
      "${cmd[@]}" > "$CASE_ROOT/$role.log" 2>&1 &
  else
    setsid env \
      -u CUDA_MPS_PIPE_DIRECTORY \
      -u CUDA_MPS_LOG_DIRECTORY \
      -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
      -u CUDA_MPS_CLIENT_PRIORITY \
      CUDA_VISIBLE_DEVICES="$visible" \
      "${cmd[@]}" > "$CASE_ROOT/$role.log" 2>&1 &
  fi
  STARTED_PID=$!
}

wait_ready() {
  local role="$1"
  local port="$2"
  local pid="$3"
  local log="$CASE_ROOT/$role.log"
  local deadline=$((SECONDS + 300))

  until grep -q 'The server is fired up and ready to roll!' "$log" 2>/dev/null; do
    kill -0 "$pid" 2>/dev/null || {
      echo "ERROR: $role exited before readiness" >&2
      tail -n 180 "$log"
      return 1
    }
    (( SECONDS < deadline )) || {
      echo "ERROR: $role readiness timeout" >&2
      tail -n 180 "$log"
      return 1
    }
    sleep 1
  done

  curl -fsS "http://127.0.0.1:$port/health" >/dev/null
}

echo "[$METHOD] target=$TARGET_VISIBLE draft=$DRAFT_VISIBLE fixed_tpcs=$FIXED_DRAFT_TPCS"

start_role target "$TARGET_VISIBLE" "${TARGET_ARGS[@]}"
target_pid="$STARTED_PID"
wait_ready target "$TARGET_PORT" "$target_pid"

start_role draft "$DRAFT_VISIBLE" "${DRAFT_ARGS[@]}"
draft_pid="$STARTED_PID"
wait_ready draft "$DRAFT_PORT" "$draft_pid"
sleep "${DRAFT_REGISTRATION_WAIT_S:-5}"

nvidia-smi \
  --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
  --format=csv > "$CASE_ROOT/process_placement.csv"

nvidia-smi dmon -s pucvmet -d 1 -o DT \
  > "$CASE_ROOT/gpu_dmon.log" 2>&1 &
monitor_pid=$!

set +e
env \
  -u CUDA_VISIBLE_DEVICES \
  -u CUDA_MPS_PIPE_DIRECTORY \
  -u CUDA_MPS_LOG_DIRECTORY \
  -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
  -u CUDA_MPS_CLIENT_PRIORITY \
  SPECSTREAM_PYTHON="$SPECSTREAM_PYTHON" \
  PYTHONUNBUFFERED=1 \
  TQDM_MININTERVAL=0.5 \
  bash -c "$BENCH_CMD" 2>&1 | tee "$CASE_ROOT/benchmark.log"
bench_status=${PIPESTATUS[0]}
set -e

if (( bench_status != 0 )); then
  echo "ERROR: benchmark status=$bench_status" >&2
  tail -n 160 "$CASE_ROOT/target.log"
  tail -n 160 "$CASE_ROOT/draft.log"
  exit "$bench_status"
fi

if grep -E \
  'Scheduler hit an exception|CUDA out of memory|Traceback \(most recent call last\)' \
  "$CASE_ROOT/target.log" "$CASE_ROOT/draft.log" \
  > "$CASE_ROOT/fatal_errors.txt"; then
  cat "$CASE_ROOT/fatal_errors.txt"
  exit 3
fi

test -s "$PROFILE_CSV" || {
  echo "ERROR: missing runtime profile $PROFILE_CSV" >&2
  exit 4
}

date -Is > "$CASE_ROOT/case_complete.marker"
echo "PASS case=$METHOD profile=$PROFILE_CSV"
