#!/usr/bin/env bash

# 此文件需要 source，不在其中开启 errexit/nounset。
export REPO_ROOT="${REPO_ROOT:-/root/lifei/SpecStream}"
export TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}"
export DRAFT_MODEL="${DRAFT_MODEL:-/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct}"

export TARGET_PORT="${TARGET_PORT:-30000}"
export DRAFT_PORT="${DRAFT_PORT:-30001}"
export ZMQ_PORT="${ZMQ_PORT:-5557}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export VERIFY_Q="${VERIFY_Q:-4}"
export TARGET_MEM_FRACTION="${TARGET_MEM_FRACTION:-0.70}"
export DRAFT_MEM_FRACTION="${DRAFT_MEM_FRACTION:-0.18}"

# B：Target GPU 1、Draft GPU 0；A/C：共同使用 GPU 0。
export COLOCATED_GPU="${COLOCATED_GPU:-0}"
export TARGET_GPU="${TARGET_GPU:-1}"
export DRAFT_GPU="${DRAFT_GPU:-0}"

# 每次测试使用新目录，避免把旧失败结果混入新结果。
export SPECSTREAM_SESSION_ID="${SPECSTREAM_SESSION_ID:-$(date +%Y%m%d_%H%M%S)}"
export TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/results/specstream_pcie_slack_isolated/$SPECSTREAM_SESSION_ID}"
export RESOURCE_PROFILE="$TEST_ROOT/resource/qwen25_history_h2d.json"
export SMCTRL_LIB="$REPO_ROOT/csrc/specstream_smctrl/build/libsmctrl.so"
export SMCTRL_MASK_SCOPE="${SMCTRL_MASK_SCOPE:-global}" ####lifei##-stream
export CALIBRATION_TPCS="${CALIBRATION_TPCS:-4}"

# 修复 libgomp: Invalid value for OMP_NUM_THREADS。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "$TEST_ROOT"/{env,clients,correctness,calibration,resource,performance,profiles,monitor,logs,summary,runs}

if [[ ! -s "$TARGET_MODEL/config.json" ]]; then
  echo "ERROR: Target model missing: $TARGET_MODEL/config.json" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -s "$DRAFT_MODEL/config.json" ]]; then
  echo "ERROR: Draft model missing: $DRAFT_MODEL/config.json" >&2
  return 2 2>/dev/null || exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found" >&2
  return 2 2>/dev/null || exit 2
fi

export COLOCATED_UUID="$(
  nvidia-smi -i "$COLOCATED_GPU" --query-gpu=uuid --format=csv,noheader |
    tr -d '[:space:]'
)"
export TARGET_UUID="$(
  nvidia-smi -i "$TARGET_GPU" --query-gpu=uuid --format=csv,noheader |
    tr -d '[:space:]'
)"
export DRAFT_UUID="$(
  nvidia-smi -i "$DRAFT_GPU" --query-gpu=uuid --format=csv,noheader |
    tr -d '[:space:]'
)"

if [[ "$COLOCATED_UUID" != GPU-* || "$TARGET_UUID" != GPU-* || "$DRAFT_UUID" != GPU-* ]]; then
  echo "ERROR: failed to resolve GPU UUIDs" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ "$TARGET_UUID" == "$DRAFT_UUID" ]]; then
  echo "ERROR: B requires two different physical GPU UUIDs" >&2
  return 2 2>/dev/null || exit 2
fi

cat > "$TEST_ROOT/env/gpu_placement.txt" <<EOF
colocated=$COLOCATED_UUID
target=$TARGET_UUID
draft=$DRAFT_UUID
EOF

printf 'REPO_ROOT=%s\nTEST_ROOT=%s\n' "$REPO_ROOT" "$TEST_ROOT"
cat "$TEST_ROOT/env/gpu_placement.txt"
