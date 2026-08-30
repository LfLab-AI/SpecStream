#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?Usage: start_i1_system.sh METHOD}"

: "${REPO:?Set REPO}"
: "${TARGET_MODEL:?Set TARGET_MODEL}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"
: "${TARGET_GPU:?Set TARGET_GPU}"
: "${DRAFT_GPU:?Set DRAFT_GPU}"

TARGET_PORT="${TARGET_PORT:-30000}"
DRAFT_PORT="${DRAFT_PORT:-30001}"
ZMQ_PORT="${ZMQ_PORT:-29000}"
SERVER_CONTEXT_LEN="${SERVER_CONTEXT_LEN:-32768}"

# fixed-q 默认 q=4；q sensitivity 可在启动前 export I1_Q=2/4/6/8。
I1_Q="${I1_Q:-4}"
if (( I1_Q < 2 )); then
  echo "ERROR: fixed speculative q must be >=2 for this launcher" >&2
  exit 2
fi
I1_STEPS=$((I1_Q - 1))
I1_DRAFT_TOKENS="$I1_Q"

# SpecStream tunables，可在 sensitivity 实验前覆盖。
CHUNK_TOKENS="${SPECSTREAM_CHUNK_TOKENS:-2048}"
NUM_BUFFERS="${SPECSTREAM_NUM_BUFFERS:-2}"
CHUNKS_PER_TRANSFER="${SPECSTREAM_CHUNKS_PER_TRANSFER:-4}"
ACTIVE_TAIL_TOKENS="${SPECSTREAM_ACTIVE_TAIL_TOKENS:-512}"
MIN_HISTORY_TOKENS="${SPECSTREAM_MIN_HISTORY_TOKENS:-8192}"
CPU_MEMORY_GB="${SPECSTREAM_CPU_MEMORY_GB:-128}"
MAX_COHORT_SIZE="${SPECSTREAM_MAX_COHORT_SIZE:-8}"
MAX_COHORT_DELAY_US="${SPECSTREAM_MAX_COHORT_DELAY_US:-200}"
Q_CANDIDATES="${SPECSTREAM_Q_CANDIDATES:-1,2,4,6,8}"
Q_SWITCH_THRESHOLD="${SPECSTREAM_Q_SWITCH_THRESHOLD:-0.08}"
SHADOW_ATTENTION="${SPECSTREAM_SHADOW_ATTENTION:-0}"
FORCE_COHORT="${SPECSTREAM_FORCE_COHORT:-0}"

CASE_TAG="${CASE_TAG:-${METHOD}}"
LOG_DIR="$RESULT_ROOT/logs/$CASE_TAG"
PROFILE_PATH="$RESULT_ROOT/profiles/${CASE_TAG}.csv"
PID_DIR="$RESULT_ROOT/pids"

mkdir -p \
  "$LOG_DIR" \
  "$RESULT_ROOT/profiles" \
  "$PID_DIR"

TARGET_PID_FILE="$PID_DIR/target.pid"
DRAFT_PID_FILE="$PID_DIR/draft.pid"

wait_health() {
  local url="$1"
  local pid="$2"
  local name="$3"
  local timeout="${4:-300}"
  local start
  start=$(date +%s)

  while true; do
    if curl -fsS "$url/health" >/dev/null 2>&1; then
      echo "$name ready: $url"
      return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name exited before health became ready" >&2
      return 1
    fi

    if (( $(date +%s) - start >= timeout )); then
      echo "ERROR: $name readiness timeout (${timeout}s)" >&2
      return 1
    fi

    sleep 1
  done
}

stop_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid=$(cat "$file" || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 0.5
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

# 防止端口仍被上一次实验占用。
stop_pid_file "$DRAFT_PID_FILE"
stop_pid_file "$TARGET_PID_FILE"

if curl -fsS "http://127.0.0.1:${TARGET_PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: Target port ${TARGET_PORT} already has a healthy server." >&2
  echo "Stop the old server before continuing." >&2
  exit 3
fi

# ------------------------------------------------------------
# E0: native single-process SGLang STANDALONE.
# ------------------------------------------------------------
if [[ "$METHOD" == "E0" ]]; then
  echo "Starting E0: SGLang STANDALONE, q=${I1_Q}, GPU=${TARGET_GPU}"

  CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
  python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --port "$TARGET_PORT" \
    --context-length "$SERVER_CONTEXT_LEN" \
    --skip-server-warmup \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-num-steps "$I1_STEPS" \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens "$I1_DRAFT_TOKENS" \
    --page-size 1 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --disable-overlap-schedule \
    >"$LOG_DIR/server.log" 2>&1 &

  TARGET_PID=$!
  echo "$TARGET_PID" > "$TARGET_PID_FILE"

  if ! wait_health "http://127.0.0.1:${TARGET_PORT}" "$TARGET_PID" "E0 Target"; then
    tail -n 200 "$LOG_DIR/server.log" >&2 || true
    exit 4
  fi

  curl -fsS "http://127.0.0.1:${TARGET_PORT}/v1/models" \
    >"$LOG_DIR/models.json" || true

  echo "E0 started successfully."
  echo "Target PID=$TARGET_PID"
  exit 0
fi

# ------------------------------------------------------------
# S0-S5: SPECTRE ordinary.
# P0-P6: SPECTRE parallel / adaptive.
# ------------------------------------------------------------
if [[ "$METHOD" =~ ^S[0-5]$ ]]; then
  FIXED_MODE="ordinary"
elif [[ "$METHOD" =~ ^P[0-6]$ ]]; then
  FIXED_MODE="parallel"
else
  echo "ERROR: unsupported METHOD=$METHOD" >&2
  echo "Supported: E0 S0 S1 S2 S3 S4 S5 P0 P1 P2 P3 P4 P5 P6" >&2
  exit 5
fi

TARGET_ARGS=(
  python -m sglang.launch_server
  --model-path "$TARGET_MODEL"
  --port "$TARGET_PORT"
  --context-length "$SERVER_CONTEXT_LEN"
  --skip-server-warmup
  --speculative-algorithm SPECTRE
  --spectre-role target
  --speculative-num-steps "$I1_STEPS"
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens "$I1_DRAFT_TOKENS"
  --page-size 1
  --attention-backend fa3
  --spectre-fixed-q-mode "$FIXED_MODE"
  --spectre-require-draft
  --spectre-draft-timeout-action fallback
  --spectre-recv-timeout-ms 5000
  --spectre-initial-recv-timeout-ms 15000
  --spectre-failure-threshold 3
  --spectre-cooldown-rounds 32
  --spectre-retry-min-count 1
  --spectre-retry-fail-ratio 0
  --spectre-reject-interval 1
  --spectre-zmq-addr 127.0.0.1
  --spectre-zmq-port "$ZMQ_PORT"
  --disable-radix-cache
  --disable-cuda-graph
  --disable-overlap-schedule
)

# S0/P0: GPU-resident SPECTRE baseline，不追加 SpecStream KV 参数。
case "$METHOD" in
  S0|P0)
    ;;

  # Full-Restore-per-round.
  S1|P1)
    TARGET_ARGS+=(
      --specstream-enabled
      --specstream-full-restore-baseline
      --specstream-chunk-tokens "$CHUNK_TOKENS"
      --specstream-num-buffers "$NUM_BUFFERS"
      --specstream-chunks-per-transfer 1
      --specstream-active-tail-tokens "$ACTIVE_TAIL_TOKENS"
      --specstream-min-history-tokens "$MIN_HISTORY_TOKENS"
      --specstream-cpu-memory-gb "$CPU_MEMORY_GB"
      --specstream-profile-path "$PROFILE_PATH"
    )
    ;;

  # Bounded Torch/reference streaming.
  S2|P2)
    TARGET_ARGS+=(
      --specstream-enabled
      --specstream-reference-attention
      --specstream-chunk-tokens "$CHUNK_TOKENS"
      --specstream-num-buffers "$NUM_BUFFERS"
      --specstream-chunks-per-transfer 1
      --no-specstream-layer-prefetch
      --specstream-active-tail-tokens "$ACTIVE_TAIL_TOKENS"
      --specstream-min-history-tokens "$MIN_HISTORY_TOKENS"
      --specstream-cpu-memory-gb "$CPU_MEMORY_GB"
      --specstream-profile-path "$PROFILE_PATH"
    )
    ;;

  # Fused/grouped streaming, no cross-layer prefetch.
  S3|P3)
    TARGET_ARGS+=(
      --specstream-enabled
      --no-specstream-reference-attention
      --specstream-chunk-tokens "$CHUNK_TOKENS"
      --specstream-num-buffers "$NUM_BUFFERS"
      --specstream-chunks-per-transfer "$CHUNKS_PER_TRANSFER"
      --no-specstream-layer-prefetch
      --specstream-active-tail-tokens "$ACTIVE_TAIL_TOKENS"
      --specstream-min-history-tokens "$MIN_HISTORY_TOKENS"
      --specstream-cpu-memory-gb "$CPU_MEMORY_GB"
      --specstream-profile-path "$PROFILE_PATH"
    )
    ;;

  # Fused/grouped + layer prefetch.
  S4|P4)
    TARGET_ARGS+=(
      --specstream-enabled
      --no-specstream-reference-attention
      --specstream-chunk-tokens "$CHUNK_TOKENS"
      --specstream-num-buffers "$NUM_BUFFERS"
      --specstream-chunks-per-transfer "$CHUNKS_PER_TRANSFER"
      --specstream-layer-prefetch
      --specstream-active-tail-tokens "$ACTIVE_TAIL_TOKENS"
      --specstream-min-history-tokens "$MIN_HISTORY_TOKENS"
      --specstream-cpu-memory-gb "$CPU_MEMORY_GB"
      --specstream-profile-path "$PROFILE_PATH"
    )
    ;;

  # Serial full I1: streaming + Cohort, fixed q.
  S5)
    TARGET_ARGS+=(
      --specstream-enabled
      --no-specstream-reference-attention
      --specstream-chunk-tokens "$CHUNK_TOKENS"
      --specstream-num-buffers "$NUM_BUFFERS"
      --specstream-chunks-per-transfer "$CHUNKS_PER_TRANSFER"
      --specstream-layer-prefetch
      --specstream-active-tail-tokens "$ACTIVE_TAIL_TOKENS"
      --specstream-min-history-tokens "$MIN_HISTORY_TOKENS"
      --specstream-cpu-memory-gb "$CPU_MEMORY_GB"
      --specstream-cohort-enabled
      --specstream-max-cohort-size "$MAX_COHORT_SIZE"
      --specstream-max-cohort-delay-us "$MAX_COHORT_DELAY_US"
      --specstream-profile-path "$PROFILE_PATH"
    )
    ;;

  # Parallel streaming + dynamic (q, mode), no Cohort.
  P5)
    TARGET_ARGS+=(
      --specstream-enabled
      --no-specstream-reference-attention
      --specstream-chunk-tokens "$CHUNK_TOKENS"
      --specstream-num-buffers "$NUM_BUFFERS"
      --specstream-chunks-per-transfer "$CHUNKS_PER_TRANSFER"
      --specstream-layer-prefetch
      --specstream-active-tail-tokens "$ACTIVE_TAIL_TOKENS"
      --specstream-min-history-tokens "$MIN_HISTORY_TOKENS"
      --specstream-cpu-memory-gb "$CPU_MEMORY_GB"
      --specstream-dynamic-q
      --specstream-q-candidates "$Q_CANDIDATES"
      --specstream-q-switch-threshold "$Q_SWITCH_THRESHOLD"
      --specstream-profile-path "$PROFILE_PATH"
    )
    ;;

  # Parallel full I1: dynamic (q, mode) + Cohort.
  P6)
    TARGET_ARGS+=(
      --specstream-enabled
      --no-specstream-reference-attention
      --specstream-chunk-tokens "$CHUNK_TOKENS"
      --specstream-num-buffers "$NUM_BUFFERS"
      --specstream-chunks-per-transfer "$CHUNKS_PER_TRANSFER"
      --specstream-layer-prefetch
      --specstream-active-tail-tokens "$ACTIVE_TAIL_TOKENS"
      --specstream-min-history-tokens "$MIN_HISTORY_TOKENS"
      --specstream-cpu-memory-gb "$CPU_MEMORY_GB"
      --specstream-dynamic-q
      --specstream-q-candidates "$Q_CANDIDATES"
      --specstream-q-switch-threshold "$Q_SWITCH_THRESHOLD"
      --specstream-cohort-enabled
      --specstream-max-cohort-size "$MAX_COHORT_SIZE"
      --specstream-max-cohort-delay-us "$MAX_COHORT_DELAY_US"
      --specstream-profile-path "$PROFILE_PATH"
    )
    ;;
esac

# Cohort ablation override:
# e.g. METHOD=P4 + SPECSTREAM_FORCE_COHORT=1 gives fixed-q P4-C.
if [[ "$FORCE_COHORT" == "1" ]]; then
  if [[ "$METHOD" =~ ^(S3|S4|P3|P4|P5)$ ]]; then
    TARGET_ARGS+=(
      --specstream-cohort-enabled
      --specstream-max-cohort-size "$MAX_COHORT_SIZE"
      --specstream-max-cohort-delay-us "$MAX_COHORT_DELAY_US"
    )
  fi
fi

if [[ "$SHADOW_ATTENTION" == "1" ]]; then
  TARGET_ARGS+=(--specstream-shadow-attention)
fi

echo "Starting $METHOD Target on GPU=$TARGET_GPU, mode=$FIXED_MODE, q=$I1_Q"

CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
"${TARGET_ARGS[@]}" \
  >"$LOG_DIR/target.log" 2>&1 &

TARGET_PID=$!
echo "$TARGET_PID" > "$TARGET_PID_FILE"

if ! wait_health "http://127.0.0.1:${TARGET_PORT}" "$TARGET_PID" "$METHOD Target"; then
  tail -n 250 "$LOG_DIR/target.log" >&2 || true
  exit 6
fi

# Target ready 后再启动 remote Drafter，避免初始化阶段大量 timeout。
DRAFT_ARGS=(
  python -m sglang.launch_server
  --model-path "$DRAFT_MODEL"
  --port "$DRAFT_PORT"
  --context-length "$SERVER_CONTEXT_LEN"
  --skip-server-warmup
  --speculative-algorithm SPECTRE
  --spectre-role draft
  --speculative-num-steps "$I1_STEPS"
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens "$I1_DRAFT_TOKENS"
  --spectre-draft-priority
  --spectre-max-draft-priority-steps 8
  --disable-overlap-schedule
  --spectre-zmq-addr 127.0.0.1
  --spectre-zmq-port "$ZMQ_PORT"
)

echo "Starting $METHOD Drafter on GPU=$DRAFT_GPU, q=$I1_Q"

CUDA_VISIBLE_DEVICES="$DRAFT_GPU" \
"${DRAFT_ARGS[@]}" \
  >"$LOG_DIR/draft.log" 2>&1 &

DRAFT_PID=$!
echo "$DRAFT_PID" > "$DRAFT_PID_FILE"

if ! wait_health "http://127.0.0.1:${DRAFT_PORT}" "$DRAFT_PID" "$METHOD Drafter"; then
  tail -n 250 "$LOG_DIR/draft.log" >&2 || true
  exit 7
fi

curl -fsS "http://127.0.0.1:${TARGET_PORT}/v1/models" \
  >"$LOG_DIR/target_models.json" || true

echo "$METHOD started successfully."
echo "Target PID=$TARGET_PID"
echo "Draft PID=$DRAFT_PID"
echo "Profile=$PROFILE_PATH"
