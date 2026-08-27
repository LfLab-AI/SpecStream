#!/usr/bin/env bash
set -euo pipefail

# Dedicated baseline used by both experiments:
#   Step 2: Target on GPU 0, Draft on GPU 1.
#   Step 3: Target TP ranks on GPU 0..N-1, Draft on one extra GPU.
: "${SPECSTREAM_TARGET_VISIBLE_DEVICES:?Set Target GPU UUID(s), comma separated}"
: "${SPECSTREAM_DRAFT_VISIBLE_DEVICES:?Set the dedicated Draft GPU UUID}"
: "${SPECSTREAM_TARGET_CMD:?Set the complete Target launch command}"
: "${SPECSTREAM_DRAFT_CMD:?Set the complete Drafter launch command}"
: "${SPECSTREAM_BENCH_CMD:?Set the benchmark command}"

RESULT_ROOT="${SPECSTREAM_RESULT_ROOT:-results/specstream_dedicated_baseline}"
TARGET_READY_CMD="${SPECSTREAM_TARGET_READY_CMD:-curl -fsS http://127.0.0.1:30000/health}"
DRAFT_READY_CMD="${SPECSTREAM_DRAFT_READY_CMD:-}"
READY_TIMEOUT_S="${SPECSTREAM_READY_TIMEOUT_S:-240}"
DRAFT_WARMUP_S="${SPECSTREAM_DRAFT_WARMUP_S:-5}"
mkdir -p "${RESULT_ROOT}"

target_pid=""
draft_pid=""
cleanup() {
  for pid in "${draft_pid}" "${target_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${SPECSTREAM_TARGET_VISIBLE_DEVICES}" \
  bash -lc "${SPECSTREAM_TARGET_CMD}" >"${RESULT_ROOT}/target.log" 2>&1 &
target_pid=$!

deadline=$((SECONDS + READY_TIMEOUT_S))
until bash -lc "${TARGET_READY_CMD}" >/dev/null 2>&1; do
  if ! kill -0 "${target_pid}" 2>/dev/null; then
    printf 'Target exited before readiness; see %s\n' "${RESULT_ROOT}/target.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    printf 'Target readiness timed out\n' >&2
    exit 1
  fi
  sleep 1
done

CUDA_VISIBLE_DEVICES="${SPECSTREAM_DRAFT_VISIBLE_DEVICES}" \
  bash -lc "${SPECSTREAM_DRAFT_CMD}" >"${RESULT_ROOT}/draft.log" 2>&1 &
draft_pid=$!

if [[ -n "${DRAFT_READY_CMD}" ]]; then
  deadline=$((SECONDS + READY_TIMEOUT_S))
  until bash -lc "${DRAFT_READY_CMD}" >/dev/null 2>&1; do
    if ! kill -0 "${draft_pid}" 2>/dev/null; then
      printf 'Draft exited before readiness; see %s\n' "${RESULT_ROOT}/draft.log" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      printf 'Draft readiness timed out\n' >&2
      exit 1
    fi
    sleep 1
  done
else
  sleep "${DRAFT_WARMUP_S}"
fi

bash -lc "${SPECSTREAM_BENCH_CMD}" >"${RESULT_ROOT}/benchmark.log" 2>&1
printf 'Dedicated Draft/Target baseline completed: %s\n' "${RESULT_ROOT}"
