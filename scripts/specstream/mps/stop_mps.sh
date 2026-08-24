#!/usr/bin/env bash
set -euo pipefail

export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/specstream-mps-${USER}}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/specstream-mps-log-${USER}}"

printf 'quit\n' | nvidia-cuda-mps-control
printf 'MPS stopped (pipe=%s)\n' "${CUDA_MPS_PIPE_DIRECTORY}"

