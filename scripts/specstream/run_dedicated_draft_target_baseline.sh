#!/usr/bin/env bash
set -euo pipefail

: "${SPECSTREAM_TARGET_VISIBLE_DEVICES:?Set Target GPU}"
: "${SPECSTREAM_DRAFT_VISIBLE_DEVICES:?Set Draft GPU}"
: "${SPECSTREAM_TARGET_CMD:?Set Target command}"
: "${SPECSTREAM_DRAFT_CMD:?Set Draft command}"
: "${SPECSTREAM_BENCH_CMD:?Set benchmark command}"

RESULT_ROOT="${SPECSTREAM_RESULT_ROOT:-results/specstream_public}"
TARGET_READY_CMD="${SPECSTREAM_TARGET_READY_CMD:-curl -fsS http://127.0.0.1:30000/health}"
DRAFT_READY_CMD="${SPECSTREAM_DRAFT_READY_CMD:-}"
READY_TIMEOUT_S="${SPECSTREAM_READY_TIMEOUT_S:-300}"
DRAFT_WARMUP_S="${SPECSTREAM_DRAFT_WARMUP_S:-5}"

mkdir -p "$RESULT_ROOT"

target_pid=""
draft_pid=""

cleanup() {
  for pid in "$draft_pid" "$target_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo
echo "============================================================"
echo "[1/5] Starting Target"
echo "GPU: $SPECSTREAM_TARGET_VISIBLE_DEVICES"
echo "Log: $RESULT_ROOT/target.log"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$SPECSTREAM_TARGET_VISIBLE_DEVICES" \
  bash -c "$SPECSTREAM_TARGET_CMD" \
  >"$RESULT_ROOT/target.log" 2>&1 &
target_pid=$!

deadline=$((SECONDS + READY_TIMEOUT_S))

until bash -c "$TARGET_READY_CMD" >/dev/null 2>&1; do
  if ! kill -0 "$target_pid" 2>/dev/null; then
    echo "ERROR: Target exited before ready"
    tail -n 120 "$RESULT_ROOT/target.log" || true
    exit 1
  fi

  if (( SECONDS >= deadline )); then
    echo "ERROR: Target readiness timeout"
    tail -n 120 "$RESULT_ROOT/target.log" || true
    exit 1
  fi

  sleep 1
done

echo "[2/5] Target ready"

echo
echo "============================================================"
echo "[3/5] Starting Drafter"
echo "GPU: $SPECSTREAM_DRAFT_VISIBLE_DEVICES"
echo "Log: $RESULT_ROOT/draft.log"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$SPECSTREAM_DRAFT_VISIBLE_DEVICES" \
  bash -c "$SPECSTREAM_DRAFT_CMD" \
  >"$RESULT_ROOT/draft.log" 2>&1 &
draft_pid=$!

if [[ -n "$DRAFT_READY_CMD" ]]; then
  deadline=$((SECONDS + READY_TIMEOUT_S))

  until bash -c "$DRAFT_READY_CMD" >/dev/null 2>&1; do
    if ! kill -0 "$draft_pid" 2>/dev/null; then
      echo "ERROR: Drafter exited before ready"
      tail -n 120 "$RESULT_ROOT/draft.log" || true
      exit 1
    fi

    if (( SECONDS >= deadline )); then
      echo "ERROR: Drafter readiness timeout"
      tail -n 120 "$RESULT_ROOT/draft.log" || true
      exit 1
    fi

    sleep 1
  done
else
  sleep "$DRAFT_WARMUP_S"
fi

echo "[4/5] Drafter ready"

echo
echo "============================================================"
echo "[5/5] Running benchmark"
echo "The tqdm progress is shown LIVE below."
echo "A copy is saved to: $RESULT_ROOT/benchmark.log"
echo "============================================================"
echo

# 不要把 benchmark 只重定向到文件。
# stderr 中的 tqdm 也通过 2>&1 送给 tee。
PYTHONUNBUFFERED=1 \
TQDM_MININTERVAL=0.5 \
  bash -c "$SPECSTREAM_BENCH_CMD" \
  2>&1 | tee "$RESULT_ROOT/benchmark.log"

echo
echo "============================================================"
echo "Benchmark completed"
echo "Result root: $RESULT_ROOT"
echo "============================================================"
