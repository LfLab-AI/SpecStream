#!/usr/bin/env bash

export REPO_ROOT="${REPO_ROOT:-/root/lifei/SpecStream}"
export SPECSTREAM_PYTHON="${SPECSTREAM_PYTHON:-/root/miniconda3/envs/spectre/bin/python}"
export TARGET_MODEL="${TARGET_MODEL:-/root/autodl-tmp/model/Qwen2.5-7B-Instruct}"
export DRAFT_MODEL="${DRAFT_MODEL:-/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct}"

export TARGET_PORT="${TARGET_PORT:-30000}"
export DRAFT_PORT="${DRAFT_PORT:-30001}"
export ZMQ_PORT="${ZMQ_PORT:-5557}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export VERIFY_Q="${VERIFY_Q:-4}"
export TARGET_MEM_FRACTION="${TARGET_MEM_FRACTION:-0.6875}"
export DRAFT_MEM_FRACTION="${DRAFT_MEM_FRACTION:-0.18}"

export COLOCATED_GPU="${COLOCATED_GPU:-0}"
export TARGET_GPU="${TARGET_GPU:-1}"
export DRAFT_GPU="${DRAFT_GPU:-0}"

export FIXED_DRAFT_TPCS="${FIXED_DRAFT_TPCS:-27}"
export SMCTRL_LIB="$REPO_ROOT/csrc/specstream_smctrl/build/libsmctrl.so"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export SPECSTREAM_SESSION_ID="${SPECSTREAM_SESSION_ID:-$(date +%Y%m%d_%H%M%S)}"
export TEST_ROOT="${TEST_ROOT:-$REPO_ROOT/results/specstream_pcie_slack_fixed27/$SPECSTREAM_SESSION_ID}"

# 只保存 session-local MPS 路径；不要 export CUDA_MPS_* 到登录 Shell。
export SPECSTREAM_MPS_PIPE="${SPECSTREAM_MPS_PIPE:-/tmp/specstream-mps-${USER}-${SPECSTREAM_SESSION_ID}}"
export SPECSTREAM_MPS_LOG="${SPECSTREAM_MPS_LOG:-/tmp/specstream-mps-log-${USER}-${SPECSTREAM_SESSION_ID}}"

[[ -x "$SPECSTREAM_PYTHON" ]] || {
  echo "ERROR: invalid SPECSTREAM_PYTHON=$SPECSTREAM_PYTHON" >&2
  return 2 2>/dev/null || exit 2
}
[[ -s "$TARGET_MODEL/config.json" ]] || {
  echo "ERROR: invalid TARGET_MODEL=$TARGET_MODEL" >&2
  return 2 2>/dev/null || exit 2
}
[[ -s "$DRAFT_MODEL/config.json" ]] || {
  echo "ERROR: invalid DRAFT_MODEL=$DRAFT_MODEL" >&2
  return 2 2>/dev/null || exit 2
}
[[ "$FIXED_DRAFT_TPCS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: invalid FIXED_DRAFT_TPCS=$FIXED_DRAFT_TPCS" >&2
  return 2 2>/dev/null || exit 2
}
(( VERIFY_Q >= 2 )) || {
  echo "ERROR: VERIFY_Q must be >=2" >&2
  return 2 2>/dev/null || exit 2
}

export COLOCATED_UUID="$(nvidia-smi -i "$COLOCATED_GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
export TARGET_UUID="$(nvidia-smi -i "$TARGET_GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
export DRAFT_UUID="$(nvidia-smi -i "$DRAFT_GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"

[[ "$COLOCATED_UUID" == GPU-* && "$TARGET_UUID" == GPU-* && "$DRAFT_UUID" == GPU-* ]] || {
  echo "ERROR: GPU UUID resolution failed" >&2
  return 2 2>/dev/null || exit 2
}
[[ "$COLOCATED_UUID" == "$DRAFT_UUID" ]] || {
  echo "ERROR: COLOCATED_GPU and DRAFT_GPU must identify the same GPU" >&2
  return 2 2>/dev/null || exit 2
}
[[ "$TARGET_UUID" != "$DRAFT_UUID" ]] || {
  echo "ERROR: method B requires two different GPUs" >&2
  return 2 2>/dev/null || exit 2
}

mkdir -p "$TEST_ROOT"/{env,logs,runs,correctness,profiles,performance,summary,monitor,provenance}

# 只有 validator 成功后该文件才存在。
[[ -s "$TEST_ROOT/env/smctrl_runtime_env.sh" ]] && source "$TEST_ROOT/env/smctrl_runtime_env.sh"

cat > "$TEST_ROOT/env/gpu_placement.txt" <<EOF2
colocated=$COLOCATED_UUID
target=$TARGET_UUID
draft=$DRAFT_UUID
fixed_draft_tpcs=$FIXED_DRAFT_TPCS
EOF2

echo "REPO_ROOT=$REPO_ROOT"
echo "TEST_ROOT=$TEST_ROOT"
cat "$TEST_ROOT/env/gpu_placement.txt"
