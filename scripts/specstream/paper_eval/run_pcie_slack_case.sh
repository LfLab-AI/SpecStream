#!/usr/bin/env bash
set -euo pipefail

: "${METHOD:?Set METHOD=A, B, C, CAL_BASE, or CAL_OVERLAP}"
: "${BENCH_CMD:?Set the complete benchmark/client command before launching}"
: "${CASE_ROOT:?Set CASE_ROOT}"
: "${PROFILE_CSV:?Set PROFILE_CSV}"
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
: "${SMCTRL_LIB:?}"

mkdir -p "$CASE_ROOT" "$(dirname "$PROFILE_CSV")"

if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required for process-group cleanup" >&2
  exit 2
fi

# 必须在启动前确认端口空闲。使用 Bash 自带的 /dev/tcp，避免依赖
# 精简容器中可能不存在的 ss、netstat 或 lsof。
port_is_listening() {
  local port="$1"
  (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null
}

# 否则健康检查可能命中上一轮残留服务，benchmark 也会误连旧 Target，
# 而本轮新进程会因重复占用 GPU/端口失败。
for port in "$TARGET_PORT" "$DRAFT_PORT"; do
  if port_is_listening "$port"; then
    echo "ERROR: port $port is already listening; stale server is still alive" >&2
    echo "Run: bash scripts/killall_sglang.sh" >&2
    exit 2
  fi
done

if (( VERIFY_Q < 2 )); then
  echo "ERROR: VERIFY_Q must be >= 2" >&2
  exit 2
fi

SPEC_STEPS=$((VERIFY_Q - 1))
NEEDS_SMCTRL=0
ALLOW_CAL_OVERLAP=0
MODE=parallel

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
    if [[ "$TARGET_VISIBLE" == "$DRAFT_VISIBLE" ]]; then
      echo "ERROR: method B requires different Target/Draft UUIDs" >&2
      exit 2
    fi
    ;;
  C)
    TARGET_VISIBLE="$COLOCATED_UUID"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    MODE=parallel
    NEEDS_SMCTRL=1
    : "${RESOURCE_PROFILE:?Set RESOURCE_PROFILE for method C}"
    if [[ ! -s "$RESOURCE_PROFILE" ]]; then
      echo "ERROR: resource profile missing: $RESOURCE_PROFILE" >&2
      exit 2
    fi
    if ! grep -q 'history_h2d' "$RESOURCE_PROFILE"; then
      echo "ERROR: profile has no history_h2d entry" >&2
      exit 2
    fi
    ;;
  CAL_BASE)
    TARGET_VISIBLE="$COLOCATED_UUID"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    MODE=parallel
    NEEDS_SMCTRL=1
    ;;
  CAL_OVERLAP)
    TARGET_VISIBLE="$COLOCATED_UUID"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    MODE=parallel
    NEEDS_SMCTRL=1
    ALLOW_CAL_OVERLAP=1
    ;;
  *)
    echo "ERROR: unknown METHOD=$METHOD" >&2
    exit 2
    ;;
esac

if (( NEEDS_SMCTRL )); then
  : "${SPECSTREAM_SMCTRL_VALIDATED:?Set only after functional validator passes}"
  : "${CALIBRATION_TPCS:?}"
  : "${SMCTRL_MASK_SCOPE:?}"
  if [[ "$SPECSTREAM_SMCTRL_VALIDATED" != 1 ]]; then
    echo "ERROR: SPECSTREAM_SMCTRL_VALIDATED must equal 1" >&2
    exit 2
  fi
  if [[ ! -f "$SMCTRL_LIB" ]]; then
    echo "ERROR: libsmctrl not found: $SMCTRL_LIB" >&2
    exit 2
  fi
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "ERROR: nvidia-cuda-mps-control not found" >&2
    exit 2
  fi
  if ! echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "ERROR: MPS daemon is not reachable through CUDA_MPS_PIPE_DIRECTORY" >&2
    exit 2
  fi
fi

TARGET_ARGS=(
  python -m sglang.launch_server
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
  --specstream-reference-attention
  --specstream-chunk-tokens 2048
  --specstream-num-buffers 2
  --specstream-chunks-per-transfer 4
  --specstream-layer-prefetch
  --specstream-active-tail-tokens 512
  --specstream-min-history-tokens 8192
  --specstream-cpu-memory-gb 64
  --specstream-gpu-reserve-mb 1024
  --specstream-profile-path "$PROFILE_CSV"
  --log-level debug
)

DRAFT_ARGS=(
  python -m sglang.launch_server
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
  --log-level debug
)

if (( NEEDS_SMCTRL )); then
  TARGET_ARGS+=(
    --specstream-pcie-slack-coexec
    --specstream-smctrl-enabled
    --specstream-coexec-require-mps
    --specstream-grant-token-quantum 1
    --specstream-coexec-target-slowdown-budget 0.05
    --specstream-coexec-guard-us 200
  )
  TARGET_ARGS+=(
  --specstream-smctrl-enabled
  --specstream-smctrl-library "$SMCTRL_LIB"
  --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE"
  )
  DRAFT_ARGS+=(
    --specstream-smctrl-enabled
    --specstream-smctrl-library "$SMCTRL_LIB"
    --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE"
  )

  if [[ "$METHOD" == C ]]; then
    TARGET_ARGS+=(--specstream-coexec-resource-profile-path "$RESOURCE_PROFILE")
  else
    TARGET_ARGS+=(--specstream-smctrl-calibration-tpcs "$CALIBRATION_TPCS")
    if (( ALLOW_CAL_OVERLAP )); then
      TARGET_ARGS+=(--specstream-smctrl-calibration-allow-overlap)
    fi
  fi
fi

printf '%q ' "${TARGET_ARGS[@]}" > "$CASE_ROOT/target_command.txt"
printf '\n' >> "$CASE_ROOT/target_command.txt"
printf '%q ' "${DRAFT_ARGS[@]}" > "$CASE_ROOT/draft_command.txt"
printf '\n' >> "$CASE_ROOT/draft_command.txt"

target_pid=""
draft_pid=""
monitor_pid=""

cleanup() {
  local pid
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  # Target/Draft 均由 setsid 启动。负 PID 会终止整个进程组，
  # 包括 launch_server 派生的 scheduler/tokenizer/detokenizer 子进程。
  for pid in "$draft_pid" "$target_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "$draft_pid" "$target_pid"; do
    if [[ -n "$pid" ]]; then
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_target() {
  if (( NEEDS_SMCTRL )); then
    setsid env CUDA_VISIBLE_DEVICES="$TARGET_VISIBLE" \
      "${TARGET_ARGS[@]}" > "$CASE_ROOT/target.log" 2>&1 &
  else
    setsid env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
      CUDA_VISIBLE_DEVICES="$TARGET_VISIBLE" \
      "${TARGET_ARGS[@]}" > "$CASE_ROOT/target.log" 2>&1 &
  fi
  target_pid=$!
}

start_draft() {
  if (( NEEDS_SMCTRL )); then
    setsid env CUDA_VISIBLE_DEVICES="$DRAFT_VISIBLE" \
      "${DRAFT_ARGS[@]}" > "$CASE_ROOT/draft.log" 2>&1 &
  else
    setsid env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
      CUDA_VISIBLE_DEVICES="$DRAFT_VISIBLE" \
      "${DRAFT_ARGS[@]}" > "$CASE_ROOT/draft.log" 2>&1 &
  fi
  draft_pid=$!
}

wait_ready() {
  local port="$1"
  local pid="$2"
  local label="$3"
  local log_file="$CASE_ROOT/${label,,}.log"
  local deadline=$((SECONDS + 300))
  until grep -q 'The server is fired up and ready to roll!' "$log_file" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $label exited before readiness" >&2
      tail -n 160 "$log_file" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: $label readiness timed out" >&2
      tail -n 160 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
  curl -fsS "http://127.0.0.1:$port/health" >/dev/null
}

echo "[$METHOD] Target GPU=$TARGET_VISIBLE Draft GPU=$DRAFT_VISIBLE"
start_target
wait_ready "$TARGET_PORT" "$target_pid" Target

start_draft
wait_ready "$DRAFT_PORT" "$draft_pid" Draft

# HTTP 进程与模型引擎均已就绪；再留出时间让 Drafter 完成向 Target 的注册。
sleep "${DRAFT_REGISTRATION_WAIT_S:-5}"

nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
  --format=csv > "$CASE_ROOT/process_placement.csv"

nvidia-smi dmon -s pucvmet -d 1 -o DT > "$CASE_ROOT/gpu_dmon.log" &
monitor_pid=$!

set +e
PYTHONUNBUFFERED=1 TQDM_MININTERVAL=0.5 \
  bash -lc "$BENCH_CMD" 2>&1 | tee "$CASE_ROOT/benchmark.log"
bench_status=${PIPESTATUS[0]}
set -e

if (( bench_status != 0 )); then
  echo "ERROR: benchmark/client failed with status $bench_status" >&2
  echo "========== TARGET LOG (last 160 lines) ==========" >&2
  tail -n 160 "$CASE_ROOT/target.log" >&2 || true
  echo "========== DRAFT LOG (last 160 lines) ==========" >&2
  tail -n 160 "$CASE_ROOT/draft.log" >&2 || true
  exit "$bench_status"
fi

if grep -E 'Traceback|RuntimeError:|Scheduler hit an exception|CUDA out of memory' \
  "$CASE_ROOT/target.log" "$CASE_ROOT/draft.log" > "$CASE_ROOT/fatal_errors.txt"; then
  echo "ERROR: fatal server error detected" >&2
  cat "$CASE_ROOT/fatal_errors.txt" >&2
  exit 3
fi

echo "PASS case=$METHOD root=$CASE_ROOT profile=$PROFILE_CSV"
