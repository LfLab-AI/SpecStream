#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?}"
: "${TEST_ROOT:?}"
: "${SPECSTREAM_PYTHON:?}"
: "${COLOCATED_UUID:?}"
: "${FIXED_DRAFT_TPCS:?Run: export FIXED_DRAFT_TPCS=<positive integer>}"
: "${SMCTRL_LIB:?}"

[[ "$FIXED_DRAFT_TPCS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: FIXED_DRAFT_TPCS must be a positive integer; got $FIXED_DRAFT_TPCS" >&2
  exit 2
}

mkdir -p "$TEST_ROOT/logs" "$TEST_ROOT/env"
cd "$REPO_ROOT/csrc/specstream_smctrl"

test -x build/specstream_smctrl_validator || {
  echo "ERROR: missing build/specstream_smctrl_validator" >&2
  exit 2
}

test -f "$SMCTRL_LIB" || {
  echo "ERROR: missing SMCTRL_LIB=$SMCTRL_LIB" >&2
  exit 2
}

TOTAL_TPCS="$(
  env \
    -u CUDA_MPS_PIPE_DIRECTORY \
    -u CUDA_MPS_LOG_DIRECTORY \
    -u MASK_OFF \
    CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
    "$SPECSTREAM_PYTHON" - "$SMCTRL_LIB" <<'PY'
import ctypes
import sys

library_path = sys.argv[1]
lib = ctypes.CDLL(library_path)

fn = lib.libsmctrl_get_tpc_info_cuda
fn.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int,
]
fn.restype = ctypes.c_int

value = ctypes.c_uint32()
status = fn(ctypes.byref(value), 0)

if status != 0:
    raise SystemExit(
        f"libsmctrl_get_tpc_info_cuda failed with status={status}"
    )
if value.value <= 0:
    raise SystemExit(f"invalid total TPC count: {value.value}")

print(value.value)
PY
)"

echo "TOTAL_TPCS=$TOTAL_TPCS"
echo "REQUESTED_TPCS=$FIXED_DRAFT_TPCS"

[[ "$TOTAL_TPCS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: invalid TOTAL_TPCS=$TOTAL_TPCS" >&2
  exit 2
}

if (( FIXED_DRAFT_TPCS > TOTAL_TPCS )); then
  echo \
    "ERROR: FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS exceeds TOTAL_TPCS=$TOTAL_TPCS" \
    >&2
  exit 2
fi

TPC_TAG="tpc${FIXED_DRAFT_TPCS}"
STREAM_LOG="$TEST_ROOT/logs/smctrl_validate_${TPC_TAG}_stream.log"
GLOBAL_LOG="$TEST_ROOT/logs/smctrl_validate_${TPC_TAG}_global.log"

SMCTRL_MASK_SCOPE=""

set +e
env \
  -u CUDA_MPS_PIPE_DIRECTORY \
  -u CUDA_MPS_LOG_DIRECTORY \
  -u MASK_OFF \
  CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
  make validate \
    TPC_LOW=0 \
    TPC_HIGH="$FIXED_DRAFT_TPCS" \
  2>&1 | tee "$STREAM_LOG"

STREAM_STATUS=${PIPESTATUS[0]}
set -e

echo "STREAM_STATUS=$STREAM_STATUS"

if (( STREAM_STATUS == 0 )) &&
   grep -q 'test passed' "$STREAM_LOG" &&
   ! grep -qi 'unsupported' "$STREAM_LOG"; then

  SMCTRL_MASK_SCOPE="stream"
  echo "STREAM_VALIDATOR=PASS"

else
  echo "STREAM_VALIDATOR=UNAVAILABLE_OR_FAILED"
  echo "Trying process-global QMD/TMD backend"

  set +e
  env \
    -u CUDA_MPS_PIPE_DIRECTORY \
    -u CUDA_MPS_LOG_DIRECTORY \
    -u MASK_OFF \
    CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
    make validate-global \
      TPC_LOW=0 \
      TPC_HIGH="$FIXED_DRAFT_TPCS" \
    2>&1 | tee "$GLOBAL_LOG"

  GLOBAL_STATUS=${PIPESTATUS[0]}
  set -e

  echo "GLOBAL_STATUS=$GLOBAL_STATUS"

  if (( GLOBAL_STATUS == 0 )) &&
     grep -q 'using process-global QMD/TMD mask backend' "$GLOBAL_LOG" &&
     grep -q 'test passed' "$GLOBAL_LOG"; then

    SMCTRL_MASK_SCOPE="global"
    echo "GLOBAL_VALIDATOR=PASS"

  else
    echo \
      "ERROR: stream and global validators both failed for TPC=$FIXED_DRAFT_TPCS" \
      >&2
    exit 2
  fi
fi

cat > "$TEST_ROOT/env/smctrl_runtime_env.sh" <<EOF
export SMCTRL_MASK_SCOPE=$SMCTRL_MASK_SCOPE
export SPECSTREAM_SMCTRL_VALIDATED=1
export TOTAL_TPCS=$TOTAL_TPCS
export FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS
EOF

cat > "$TEST_ROOT/env/tpc_experiment.txt" <<EOF
requested_tpcs=$FIXED_DRAFT_TPCS
total_tpcs=$TOTAL_TPCS
mask_scope=$SMCTRL_MASK_SCOPE
EOF

echo "========== saved runtime environment =========="
cat "$TEST_ROOT/env/smctrl_runtime_env.sh"

echo "========== validation result =========="
echo "SMCTRL_MASK_SCOPE=$SMCTRL_MASK_SCOPE"
echo "TOTAL_TPCS=$TOTAL_TPCS"
echo "FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS"
echo "SMCTRL_VARIABLE_TPC_GATE=PASS"
