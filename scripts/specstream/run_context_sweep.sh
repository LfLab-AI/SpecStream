#!/usr/bin/env bash
set -euo pipefail

BASELINE="${BASELINE:-B0}"
CONTEXT_LENS="${CONTEXT_LENS:-4096 16384 30000}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

for input_len in ${CONTEXT_LENS}; do
  CASE_TAG="${BASELINE}_${DATASET_NAME:-random}_${input_len}_c${MAX_CONCURRENCY:-1}" \
  INPUT_LEN="${input_len}" \
  OUTPUT_LEN="${OUTPUT_LEN}" \
  bash scripts/specstream/run_benchmark_case.sh
done

