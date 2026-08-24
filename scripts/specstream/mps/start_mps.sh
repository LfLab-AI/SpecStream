#!/usr/bin/env bash
set -euo pipefail

: "${SPECSTREAM_GPU_UUID:?Set SPECSTREAM_GPU_UUID to the physical GPU UUID from nvidia-smi -L}"

export CUDA_VISIBLE_DEVICES="${SPECSTREAM_GPU_UUID}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/specstream-mps-${USER}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/specstream-mps-log-${USER}}"

mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d

printf 'MPS started for %s\n' "${SPECSTREAM_GPU_UUID}"
printf 'export CUDA_MPS_PIPE_DIRECTORY=%q\n' "${CUDA_MPS_PIPE_DIRECTORY}"
printf 'export CUDA_MPS_LOG_DIRECTORY=%q\n' "${CUDA_MPS_LOG_DIRECTORY}"

