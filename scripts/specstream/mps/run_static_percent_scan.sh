#!/usr/bin/env bash
set -euo pipefail

: "${SPECSTREAM_GPU_UUID:?Set the shared physical GPU UUID}"
: "${SPECSTREAM_TARGET_CMD:?Set the complete Target launch command}"
: "${SPECSTREAM_DRAFT_CMD:?Set the complete Drafter launch command}"
: "${SPECSTREAM_BENCH_CMD:?Set the benchmark command run for each quota pair}"

QUOTA_PAIRS="${SPECSTREAM_QUOTA_PAIRS:-90:10 80:20 70:30 60:40}"
RESULT_ROOT="${SPECSTREAM_RESULT_ROOT:-results/specstream_mps_scan}"
READY_CMD="${SPECSTREAM_TARGET_READY_CMD:-curl -fsS http://127.0.0.1:30000/health}"
READY_TIMEOUT_S="${SPECSTREAM_READY_TIMEOUT_S:-180}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/specstream-mps-${USER}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/specstream-mps-log-${USER}}"

mkdir -p "${RESULT_ROOT}" "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

target_pid=""
draft_pid=""
cleanup_pair() {
  if [[ -n "${draft_pid}" ]] && kill -0 "${draft_pid}" 2>/dev/null; then
    kill "${draft_pid}"
    wait "${draft_pid}" 2>/dev/null || true
  fi
  if [[ -n "${target_pid}" ]] && kill -0 "${target_pid}" 2>/dev/null; then
    kill "${target_pid}"
    wait "${target_pid}" 2>/dev/null || true
  fi
  target_pid=""
  draft_pid=""
}
trap cleanup_pair EXIT INT TERM

for pair in ${QUOTA_PAIRS}; do
  target_share="${pair%%:*}"
  draft_share="${pair##*:}"
  if (( target_share < 1 || target_share > 100 || draft_share < 1 || draft_share > 100 )); then
    printf 'Invalid quota pair: %s\n' "${pair}" >&2
    exit 2
  fi

  tag="target${target_share}_draft${draft_share}"
  run_dir="${RESULT_ROOT}/${tag}"
  mkdir -p "${run_dir}"
  printf 'Running %s on %s\n' "${tag}" "${SPECSTREAM_GPU_UUID}"

  CUDA_VISIBLE_DEVICES="${SPECSTREAM_GPU_UUID}" \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${target_share}" \
  CUDA_MPS_CLIENT_PRIORITY=0 \
  SPECSTREAM_MPS_SCAN_TAG="${tag}" \
    bash -lc "${SPECSTREAM_TARGET_CMD}" >"${run_dir}/target.log" 2>&1 &
  target_pid=$!

  deadline=$((SECONDS + READY_TIMEOUT_S))
  until bash -lc "${READY_CMD}" >/dev/null 2>&1; do
    if ! kill -0 "${target_pid}" 2>/dev/null; then
      printf 'Target exited before readiness; see %s\n' "${run_dir}/target.log" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      printf 'Target readiness timed out; see %s\n' "${run_dir}/target.log" >&2
      exit 1
    fi
    sleep 1
  done

  CUDA_VISIBLE_DEVICES="${SPECSTREAM_GPU_UUID}" \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${draft_share}" \
  CUDA_MPS_CLIENT_PRIORITY=1 \
  SPECSTREAM_MPS_SCAN_TAG="${tag}" \
    bash -lc "${SPECSTREAM_DRAFT_CMD}" >"${run_dir}/draft.log" 2>&1 &
  draft_pid=$!

  SPECSTREAM_MPS_SCAN_TAG="${tag}" \
  SPECSTREAM_MPS_SCAN_OUTPUT="${run_dir}" \
    bash -lc "${SPECSTREAM_BENCH_CMD}" >"${run_dir}/benchmark.log" 2>&1

  cleanup_pair
done

trap - EXIT INT TERM
printf 'Static MPS scan completed: %s\n' "${RESULT_ROOT}"

