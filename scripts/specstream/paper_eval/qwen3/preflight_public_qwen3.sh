#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

: "${REPO:=/root/lifei/SpecStream}"
: "${SPECSTREAM_PYTHON:=/root/miniconda3/envs/spectre/bin/python}"
: "${TARGET_MODEL:=/root/autodl-tmp/model/Qwen3-8B}"
: "${DRAFT_MODEL:=/root/autodl-tmp/model/Qwen3-0.6B}"
: "${TARGET_GPU:=1}"
: "${TARGET_GPUS:=$TARGET_GPU}"
: "${TARGET_TP_SIZE:=1}"
: "${DRAFT_GPU:=0}"
: "${COLOCATED_GPU:=$TARGET_GPU}"
: "${COLOCATED_TP_RANK:=0}"
: "${REQUIRE_SEPARATE_DRAFT_GPU:=0}"
: "${FINAL_DRAFT_TPCS:=34}"
: "${PREFLIGHT_ROOT:=$REPO/results/qwen3_public_preflight_$(date +%Y%m%d_%H%M%S)}"

cd "$REPO" || { echo "ERROR: repository is unavailable: $REPO" >&2; return 2; }
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"
mkdir -p "$PREFLIGHT_ROOT"/{env,logs}

[[ -x "$SPECSTREAM_PYTHON" ]] || { echo "ERROR: Python is not executable: $SPECSTREAM_PYTHON" >&2; return 2; }
[[ -s "$TARGET_MODEL/config.json" ]] || { echo "ERROR: Target config missing: $TARGET_MODEL/config.json" >&2; return 2; }
[[ -s "$DRAFT_MODEL/config.json" ]] || { echo "ERROR: Draft config missing: $DRAFT_MODEL/config.json" >&2; return 2; }
[[ "$FINAL_DRAFT_TPCS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: FINAL_DRAFT_TPCS must be positive: $FINAL_DRAFT_TPCS" >&2; return 2; }
[[ "$TARGET_TP_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: TARGET_TP_SIZE must be positive: $TARGET_TP_SIZE" >&2; return 2; }
[[ "$COLOCATED_TP_RANK" =~ ^[0-9]+$ ]] || { echo "ERROR: COLOCATED_TP_RANK must be non-negative" >&2; return 2; }

for port in 30000 30001 5557; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "ERROR: port $port is already in use" >&2
    return 2
  fi
done

git rev-parse HEAD | tee "$PREFLIGHT_ROOT/env/git_commit.txt"
git status --short | tee "$PREFLIGHT_ROOT/env/git_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used \
  --format=csv | tee "$PREFLIGHT_ROOT/env/gpus.csv"

IFS=',' read -r -a target_gpu_array <<<"$TARGET_GPUS"
if (( ${#target_gpu_array[@]} != TARGET_TP_SIZE )); then
  echo "ERROR: TARGET_GPUS=$TARGET_GPUS exposes ${#target_gpu_array[@]} GPU(s), but TARGET_TP_SIZE=$TARGET_TP_SIZE" >&2
  return 2
fi
[[ "${target_gpu_array[0]}" == "$TARGET_GPU" ]] || {
  echo "ERROR: TARGET_GPU must equal TARGET_GPUS rank 0 (${target_gpu_array[0]})" >&2
  return 2
}
target_uuid_array=()
for gpu in "${target_gpu_array[@]}"; do
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
  [[ -n "$uuid" ]] || { echo "ERROR: cannot resolve Target GPU $gpu" >&2; return 2; }
  target_uuid_array+=("$uuid")
done
if (( $(printf '%s\n' "${target_uuid_array[@]}" | sort -u | wc -l) != TARGET_TP_SIZE )); then
  echo "ERROR: TARGET_GPUS contains duplicate physical GPUs" >&2
  return 2
fi
export TARGET_UUIDS=$(IFS=,; echo "${target_uuid_array[*]}")
export TARGET_UUID="${target_uuid_array[0]}"
export DRAFT_UUID=$(nvidia-smi -i "$DRAFT_GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
export COLOCATED_UUID=$(nvidia-smi -i "$COLOCATED_GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
(( COLOCATED_TP_RANK < TARGET_TP_SIZE )) || { echo "ERROR: COLOCATED_TP_RANK is outside the Target TP group" >&2; return 2; }
[[ "${target_uuid_array[$COLOCATED_TP_RANK]}" == "$COLOCATED_UUID" ]] || {
  echo "ERROR: COLOCATED_GPU must be TARGET_GPUS rank $COLOCATED_TP_RANK" >&2
  return 2
}
if [[ "$REQUIRE_SEPARATE_DRAFT_GPU" == 1 ]]; then
  for uuid in "${target_uuid_array[@]}"; do
    [[ "$uuid" != "$DRAFT_UUID" ]] || {
      echo "ERROR: a separate Draft GPU was requested, but DRAFT_GPU belongs to TARGET_GPUS" >&2
      return 2
    }
  done
fi

TARGET_MODEL="$TARGET_MODEL" DRAFT_MODEL="$DRAFT_MODEL" \
"$SPECSTREAM_PYTHON" - <<'PY' | tee "$PREFLIGHT_ROOT/env/python_model_gate.txt"
import hashlib
import json
import os
import sys

import torch
import transformers
from transformers import AutoTokenizer

target = AutoTokenizer.from_pretrained(
    os.environ["TARGET_MODEL"], trust_remote_code=True, local_files_only=True
)
draft = AutoTokenizer.from_pretrained(
    os.environ["DRAFT_MODEL"], trust_remote_code=True, local_files_only=True
)
assert target.get_vocab() == draft.get_vocab(), "token-to-id mapping mismatch"
for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
    assert getattr(target, key) == getattr(draft, key), key
probe = target.apply_chat_template(
    [{"role": "user", "content": "offline gate"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
vocab = json.dumps(sorted(target.get_vocab().items()), ensure_ascii=False).encode()
print("python=", sys.executable)
print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("cuda=", torch.cuda.is_available())
print("vocab_sha256=", hashlib.sha256(vocab).hexdigest())
print("template_probe=", probe[:120].replace("\n", "\\n"))
assert torch.cuda.is_available()
print("PYTHON_MODEL_GATE=PASS")
PY

"$SPECSTREAM_PYTHON" - <<'PY' | tee "$PREFLIGHT_ROOT/env/code_gate.txt"
import inspect
from sglang.srt.speculative.spectre.drafter.spectre_draft_scheduler_mixin import SpectreDraftSchedulerMixin
from sglang.srt.server_args import ServerArgs

priority = inspect.getsource(SpectreDraftSchedulerMixin._run_draft_priority_phase)
assert "_run_granted_draft_step()" in priority, "Target-issued grant path is not wired"
required = (
    "specstream_pcie_slack_coexec",
    "specstream_smctrl_enabled",
    "specstream_smctrl_calibration_tpcs",
    "specstream_smctrl_calibration_allow_overlap",
)
for name in required:
    assert hasattr(ServerArgs, name), name
print("CODE_GATE=PASS")
PY

export SMCTRL_LIB="$REPO/csrc/specstream_smctrl/build/libsmctrl.so"
if [[ ! -s "$SMCTRL_LIB" ]]; then
  command -v nvcc >/dev/null
  export CUDACXX=$(command -v nvcc)
  make -C "$REPO/csrc/specstream_smctrl" config
  make -C "$REPO/csrc/specstream_smctrl" build
fi
[[ -s "$SMCTRL_LIB" ]] || { echo "ERROR: SM control library is missing: $SMCTRL_LIB" >&2; return 2; }

unset SMCTRL_MASK_SCOPE
set +e
env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY -u MASK_OFF \
  CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
  make -C "$REPO/csrc/specstream_smctrl" validate \
    TPC_LOW=0 TPC_HIGH="$FINAL_DRAFT_TPCS" \
  2>&1 | tee "$PREFLIGHT_ROOT/logs/smctrl_stream.log"
stream_rc=${PIPESTATUS[0]}
set -e

if (( stream_rc == 0 )) && \
   grep -q 'test passed' "$PREFLIGHT_ROOT/logs/smctrl_stream.log" && \
   ! grep -qi 'unsupported' "$PREFLIGHT_ROOT/logs/smctrl_stream.log"; then
  export SMCTRL_MASK_SCOPE=stream
else
  set +e
  env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY -u MASK_OFF \
    CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" \
    make -C "$REPO/csrc/specstream_smctrl" validate-global \
      TPC_LOW=0 TPC_HIGH="$FINAL_DRAFT_TPCS" \
    2>&1 | tee "$PREFLIGHT_ROOT/logs/smctrl_global.log"
  global_rc=${PIPESTATUS[0]}
  set -e
  if (( global_rc != 0 )) || ! grep -q 'test passed' "$PREFLIGHT_ROOT/logs/smctrl_global.log"; then
    echo "ERROR: both stream and global SM/TPC validators failed; see $PREFLIGHT_ROOT/logs" >&2
    return 2
  fi
  export SMCTRL_MASK_SCOPE=global
fi

cat > "$PREFLIGHT_ROOT/runtime_env.sh" <<EOF
export REPO=$REPO
export SPECSTREAM_PYTHON=$SPECSTREAM_PYTHON
export TARGET_MODEL=$TARGET_MODEL
export DRAFT_MODEL=$DRAFT_MODEL
export TARGET_GPU=$TARGET_GPU
export TARGET_GPUS=$TARGET_GPUS
export TARGET_TP_SIZE=$TARGET_TP_SIZE
export TARGET_UUIDS=$TARGET_UUIDS
export DRAFT_GPU=$DRAFT_GPU
export COLOCATED_GPU=$COLOCATED_GPU
export COLOCATED_TP_RANK=$COLOCATED_TP_RANK
export TARGET_UUID=$TARGET_UUID
export DRAFT_UUID=$DRAFT_UUID
export COLOCATED_UUID=$COLOCATED_UUID
export FINAL_DRAFT_TPCS=$FINAL_DRAFT_TPCS
export SMCTRL_LIB=$SMCTRL_LIB
export SMCTRL_MASK_SCOPE=$SMCTRL_MASK_SCOPE
export SPECSTREAM_SMCTRL_VALIDATED=1
export PREFLIGHT_ROOT=$PREFLIGHT_ROOT
EOF

echo "PREFLIGHT_ROOT=$PREFLIGHT_ROOT"
echo "SMCTRL_MASK_SCOPE=$SMCTRL_MASK_SCOPE"
echo "QWEN3_PUBLIC_PREFLIGHT=PASS"
)

# Sourcing this file must never terminate or alter the caller's interactive shell.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
