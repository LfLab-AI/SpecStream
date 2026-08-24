#!/usr/bin/env bash
set -euo pipefail

: "${SPECSTREAM_TARGET_VISIBLE_DEVICES:?Set the comma-separated Target TP GPU UUIDs}"
: "${SPECSTREAM_DRAFT_GPU_UUID:?Set the UUID colocated with --specstream-colocated-tp-rank}"
: "${SPECSTREAM_TARGET_CMD:?Set the complete TP Target launch command}"
: "${SPECSTREAM_DRAFT_CMD:?Set the complete Drafter launch command}"
: "${SPECSTREAM_BENCH_CMD:?Set the benchmark command}"

TARGET_SHARE="${SPECSTREAM_TARGET_SHARE:-90}"
DRAFT_SHARE="${SPECSTREAM_DRAFT_SHARE:-10}"
RESULT_ROOT="${SPECSTREAM_RESULT_ROOT:-results/specstream_h1_tp}"
READY_CMD="${SPECSTREAM_TARGET_READY_CMD:-curl -fsS http://127.0.0.1:30000/health}"
READY_TIMEOUT_S="${SPECSTREAM_READY_TIMEOUT_S:-240}"
DRAFT_READY_CMD="${SPECSTREAM_DRAFT_READY_CMD:-}"
DRAFT_WARMUP_S="${SPECSTREAM_DRAFT_WARMUP_S:-5}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/specstream-mps-${USER}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/specstream-mps-log-${USER}}"

mkdir -p "${RESULT_ROOT}" "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

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

# MPS active-thread percentage is inherited by every Target worker spawned by
# this command.  This keeps Target rank quotas symmetric; the selected rank is
# still the only one sharing a physical GPU with the Drafter.  Use per-device
# static SM partitions when the deployment requires asymmetric Target quotas.
CUDA_VISIBLE_DEVICES="${SPECSTREAM_TARGET_VISIBLE_DEVICES}" \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${TARGET_SHARE}" \
CUDA_MPS_CLIENT_PRIORITY=0 \
  bash -lc "${SPECSTREAM_TARGET_CMD}" >"${RESULT_ROOT}/target.log" 2>&1 &
target_pid=$!

deadline=$((SECONDS + READY_TIMEOUT_S))
until bash -lc "${READY_CMD}" >/dev/null 2>&1; do
  if ! kill -0 "${target_pid}" 2>/dev/null; then
    printf 'TP Target exited before readiness; see %s\n' "${RESULT_ROOT}/target.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    printf 'TP Target readiness timed out\n' >&2
    exit 1
  fi
  sleep 1
done

CUDA_VISIBLE_DEVICES="${SPECSTREAM_DRAFT_GPU_UUID}" \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${DRAFT_SHARE}" \
CUDA_MPS_CLIENT_PRIORITY=1 \
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
printf 'H1 colocated TP benchmark completed: %s\n' "${RESULT_ROOT}"
