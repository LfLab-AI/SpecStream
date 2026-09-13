#!/usr/bin/env bash
set -euo pipefail

: "${COLOCATED_UUID:?}"
: "${SPECSTREAM_MPS_PIPE:?}"
: "${SPECSTREAM_MPS_LOG:?}"

mkdir -p "$SPECSTREAM_MPS_PIPE" "$SPECSTREAM_MPS_LOG"

if env CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
       CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
       bash -c 'echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1'; then
  echo "MPS already ready: $SPECSTREAM_MPS_PIPE"
  exit 0
fi

env CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
    CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
    CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
    nvidia-cuda-mps-control -d

for _ in $(seq 1 20); do
  if env CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
         CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
         bash -c 'echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1'; then
    echo "MPS_GATE=PASS"
    exit 0
  fi
  sleep 1
done

echo "ERROR: MPS daemon not ready" >&2
exit 2
