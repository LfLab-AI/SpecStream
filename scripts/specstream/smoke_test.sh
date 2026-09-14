#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
TEST_DIR=python/sglang/test/spectre_specstream
case "${1:-}" in
  '')
    TESTS=(test_online_softmax.py test_resource_profile.py test_gpu_grant_controller.py test_gpu_history_budget.py)
    ;;
  --gpu)
    "$PYTHON" scripts/specstream/check_environment.py --gpu
    "$PYTHON" -c 'from sglang.srt.speculative.spectre.specstream.triton_stream_attn import triton_fused_available; assert triton_fused_available(), "CUDA attention kernels are unavailable"'
    TESTS=(test_split_kv_attention.py test_verifier_gpu_integration.py)
    ;;
  *) echo 'Usage: bash scripts/specstream/smoke_test.sh [--gpu]' >&2; exit 2 ;;
esac
(( $# <= 1 )) || { echo 'Unexpected arguments' >&2; exit 2; }
ARGS=()
for name in "${TESTS[@]}"; do ARGS+=("$TEST_DIR/$name"); done
"$PYTHON" -m pytest -q -p no:cacheprovider "${ARGS[@]}"
echo SPECSTREAM_SMOKE=PASS
