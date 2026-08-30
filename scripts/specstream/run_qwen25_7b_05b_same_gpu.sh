#!/usr/bin/env bash
set -euo pipefail

# Launch Qwen2.5-7B Target and Qwen2.5-0.5B remote SPECTRE Drafter on the
# exact same physical GPU.  Passing a GPU UUID (rather than two independently
# remapped cuda:0 ordinals) makes placement explicit and reproducible.

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
TARGET_PORT="${TARGET_PORT:-30000}"
DRAFT_PORT="${DRAFT_PORT:-30001}"
ZMQ_PORT="${ZMQ_PORT:-5557}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
VERIFY_Q="${VERIFY_Q:-4}"
TARGET_MEM_FRACTION="${TARGET_MEM_FRACTION:-0.70}"
DRAFT_MEM_FRACTION="${DRAFT_MEM_FRACTION:-0.18}"
RESULT_ROOT="${RESULT_ROOT:-results/specstream_qwen25_same_gpu}"
SMCTRL_MASK_SCOPE="${SMCTRL_MASK_SCOPE:-stream}"

: "${RESOURCE_PROFILE:?Set RESOURCE_PROFILE to a measured history_h2d profile}"
: "${SMCTRL_LIB:?Set SMCTRL_LIB to the validated libsmctrl.so path}"

if [[ "${SPECSTREAM_SMCTRL_VALIDATED:-0}" != "1" ]]; then
  echo "Refusing to launch: run the SM/TPC functional validator, then set SPECSTREAM_SMCTRL_VALIDATED=1." >&2
  exit 2
fi
if (( VERIFY_Q < 2 )); then
  echo "VERIFY_Q must be at least 2 for parallel SPECTRE." >&2
  exit 2
fi
if [[ ! -s "$RESOURCE_PROFILE" ]]; then
  echo "RESOURCE_PROFILE is missing or empty: $RESOURCE_PROFILE" >&2
  exit 2
fi
if [[ ! -f "$SMCTRL_LIB" ]]; then
  echo "SMCTRL_LIB does not exist: $SMCTRL_LIB" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to resolve an immutable physical GPU UUID." >&2
  exit 2
fi
if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "CUDA MPS is required for inter-process Target/Draft overlap." >&2
  exit 2
fi
if ! echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "CUDA MPS control daemon is not reachable in the current MPS pipe directory." >&2
  exit 2
fi

GPU_UUID="$(
  nvidia-smi -i "$PHYSICAL_GPU" --query-gpu=uuid --format=csv,noheader \
    | tr -d '[:space:]'
)"
if [[ "$GPU_UUID" != GPU-* ]]; then
  echo "Could not resolve PHYSICAL_GPU=$PHYSICAL_GPU to a GPU UUID." >&2
  exit 2
fi

python - "$RESOURCE_PROFILE" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
entries = payload.get("entries", ())
if not any(entry.get("slack_source") in ("history_h2d", "*") for entry in entries):
    raise SystemExit(
        f"{path} has no history_h2d resource-profile entry; refusing unsafe reuse"
    )
PY

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/profiles"
SPECULATIVE_STEPS=$((VERIFY_Q - 1))

echo "Target model : $TARGET_MODEL"
echo "Draft model  : $DRAFT_MODEL"
echo "Physical GPU : $GPU_UUID"
echo "Placement    : Target cuda:0 and Drafter cuda:0 both map to $GPU_UUID"

CUDA_VISIBLE_DEVICES="$GPU_UUID" \
python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --port "$TARGET_PORT" \
  --tp-size 1 \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$TARGET_MEM_FRACTION" \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role target \
  --speculative-num-steps "$SPECULATIVE_STEPS" \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens "$VERIFY_Q" \
  --page-size 1 \
  --spectre-fixed-q-mode parallel \
  --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port "$ZMQ_PORT" \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --specstream-enabled \
  --specstream-pcie-slack-coexec \
  --specstream-smctrl-enabled \
  --specstream-coexec-require-mps \
  --specstream-grant-token-quantum 1 \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --specstream-coexec-resource-profile-path "$RESOURCE_PROFILE" \
  --specstream-profile-path "$RESULT_ROOT/profiles/target.csv" \
  >"$RESULT_ROOT/logs/target.log" 2>&1 &
TARGET_PID=$!

# Target owns the ZMQ server side.  It is intentionally started first; the
# explicit SPECTRE initial timeout covers Drafter model startup.
sleep 2

CUDA_VISIBLE_DEVICES="$GPU_UUID" \
python -m sglang.launch_server \
  --model-path "$DRAFT_MODEL" \
  --port "$DRAFT_PORT" \
  --tp-size 1 \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$DRAFT_MEM_FRACTION" \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role draft \
  --speculative-num-steps "$SPECULATIVE_STEPS" \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens "$VERIFY_Q" \
  --spectre-draft-priority \
  --spectre-max-draft-priority-steps "$((VERIFY_Q * 2))" \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port "$ZMQ_PORT" \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library "$SMCTRL_LIB" \
  --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE" \
  >"$RESULT_ROOT/logs/draft.log" 2>&1 &
DRAFT_PID=$!

cleanup() {
  kill "$TARGET_PID" "$DRAFT_PID" 2>/dev/null || true
  wait "$TARGET_PID" "$DRAFT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Target PID=$TARGET_PID, Drafter PID=$DRAFT_PID"
echo "Logs: $RESULT_ROOT/logs"
set +e
wait -n "$TARGET_PID" "$DRAFT_PID"
STATUS=$?
set -e
exit "$STATUS"
