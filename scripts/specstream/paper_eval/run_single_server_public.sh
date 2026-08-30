#!/usr/bin/env bash
set -euo pipefail

: "${PUBLIC_SERVER_CMD:?Set PUBLIC_SERVER_CMD}"
: "${PUBLIC_BENCH_CMD:?Set PUBLIC_BENCH_CMD}"
: "${TARGET_GPU:?Set TARGET_GPU}"
: "${TARGET_PORT:?Set TARGET_PORT}"
: "${SPECSTREAM_RESULT_ROOT:?Set result root}"

mkdir -p "$SPECSTREAM_RESULT_ROOT"

pid=""

cleanup() {
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo
echo "============================================================"
echo "[1/3] Starting server"
echo "Log: $SPECSTREAM_RESULT_ROOT/server.log"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$TARGET_GPU" \
  bash -c "$PUBLIC_SERVER_CMD" \
  >"$SPECSTREAM_RESULT_ROOT/server.log" 2>&1 &
pid=$!

deadline=$((SECONDS + 300))

until curl -fsS \
  "http://127.0.0.1:${TARGET_PORT}/health" \
  >/dev/null 2>&1; do

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: server exited"
    tail -n 120 "$SPECSTREAM_RESULT_ROOT/server.log" || true
    exit 1
  fi

  if (( SECONDS >= deadline )); then
    echo "ERROR: health timeout"
    tail -n 120 "$SPECSTREAM_RESULT_ROOT/server.log" || true
    exit 1
  fi

  sleep 1
done

echo "[2/3] Server ready"

echo
echo "============================================================"
echo "[3/3] Running benchmark"
echo "tqdm progress is shown LIVE below."
echo "============================================================"
echo

PYTHONUNBUFFERED=1 \
TQDM_MININTERVAL=0.5 \
  bash -c "$PUBLIC_BENCH_CMD" \
  2>&1 | tee "$SPECSTREAM_RESULT_ROOT/benchmark.log"
