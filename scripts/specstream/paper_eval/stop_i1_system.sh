#!/usr/bin/env bash
set -euo pipefail

: "${RESULT_ROOT:?Set RESULT_ROOT}"

PID_DIR="$RESULT_ROOT/pids"

stop_one() {
  local file="$1"
  local name="$2"

  if [[ ! -f "$file" ]]; then
    return 0
  fi

  local pid
  pid=$(cat "$file" || true)

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $name PID=$pid"
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
}

stop_one "$PID_DIR/draft.pid" "Drafter"
stop_one "$PID_DIR/target.pid" "Target"

echo "SpecStream experiment servers stopped."
