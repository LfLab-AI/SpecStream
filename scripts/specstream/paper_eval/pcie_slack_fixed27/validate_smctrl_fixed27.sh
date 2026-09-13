#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${TEST_ROOT:?}"
: "${SPECSTREAM_PYTHON:?}"
: "${COLOCATED_UUID:?}"
: "${FIXED_DRAFT_TPCS:?}"
: "${SMCTRL_LIB:?}"

[[ "$FIXED_DRAFT_TPCS" == 27 ]] || {
  echo "ERROR: this validator requires FIXED_DRAFT_TPCS=27" >&2
  exit 2
}

mkdir -p "$TEST_ROOT/logs" "$TEST_ROOT/env"
cd "$REPO_ROOT/csrc/specstream_smctrl"

SMCTRL_MASK_SCOPE=''

set +e
env -u CUDA_MPS_PIPE_DIRECTORY \
    -u CUDA_MPS_LOG_DIRECTORY \
    -u MASK_OFF \
    CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
    make validate TPC_LOW=0 TPC_HIGH="$FIXED_DRAFT_TPCS" \
    2>&1 | tee "$TEST_ROOT/logs/smctrl_validate_fixed27_stream.log"
STREAM_STATUS=${PIPESTATUS[0]}
set -e

echo "STREAM_STATUS=$STREAM_STATUS"

if (( STREAM_STATUS == 0 )) && \
   grep -q 'test passed' "$TEST_ROOT/logs/smctrl_validate_fixed27_stream.log" && \
   ! grep -q 'unsupported' "$TEST_ROOT/logs/smctrl_validate_fixed27_stream.log"; then
  SMCTRL_MASK_SCOPE=stream
else
  set +e
  env -u CUDA_MPS_PIPE_DIRECTORY \
      -u CUDA_MPS_LOG_DIRECTORY \
      -u MASK_OFF \
      CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
      make validate-global TPC_LOW=0 TPC_HIGH="$FIXED_DRAFT_TPCS" \
      2>&1 | tee "$TEST_ROOT/logs/smctrl_validate_fixed27_global.log"
  GLOBAL_STATUS=${PIPESTATUS[0]}
  set -e

  echo "GLOBAL_STATUS=$GLOBAL_STATUS"

  if (( GLOBAL_STATUS == 0 )) && \
     grep -q 'using process-global QMD/TMD mask backend' "$TEST_ROOT/logs/smctrl_validate_fixed27_global.log" && \
     grep -q 'test passed' "$TEST_ROOT/logs/smctrl_validate_fixed27_global.log"; then
    SMCTRL_MASK_SCOPE=global
  else
    echo 'ERROR: both stream and global fixed-27 validators failed' >&2
    exit 2
  fi
fi

TOTAL_TPCS="$(
  env -u CUDA_MPS_PIPE_DIRECTORY \
      -u CUDA_MPS_LOG_DIRECTORY \
      -u MASK_OFF \
      CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
      "$SPECSTREAM_PYTHON" - "$SMCTRL_LIB" <<'PY'
import ctypes
import sys

lib = ctypes.CDLL(sys.argv[1])
fn = lib.libsmctrl_get_tpc_info_cuda
fn.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
fn.restype = ctypes.c_int
value = ctypes.c_uint32()
status = fn(ctypes.byref(value), 0)
if status != 0 or value.value <= 0:
    raise SystemExit('TPC query failed')
print(value.value)
PY
)"

echo "TOTAL_TPCS=$TOTAL_TPCS"

[[ "$TOTAL_TPCS" =~ ^[1-9][0-9]*$ ]]
(( FIXED_DRAFT_TPCS <= TOTAL_TPCS )) || {
  echo "ERROR: fixed TPCs $FIXED_DRAFT_TPCS exceed total $TOTAL_TPCS" >&2
  exit 2
}

cat > "$TEST_ROOT/env/smctrl_runtime_env.sh" <<EOF2
export SMCTRL_MASK_SCOPE=$SMCTRL_MASK_SCOPE
export SPECSTREAM_SMCTRL_VALIDATED=1
export TOTAL_TPCS=$TOTAL_TPCS
export FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS
EOF2

cat "$TEST_ROOT/env/smctrl_runtime_env.sh"
echo FIXED27_SMCTRL_GATE=PASS
