#!/usr/bin/env bash
# Run a Target and Draft pair on the same GPUs. Stop both with Ctrl-C.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-python}
: "${TARGET_MODEL:?Set TARGET_MODEL to a local model directory}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to a compatible local model directory}"
: "${DRAFT_TPCS:?Set DRAFT_TPCS to the Draft TPC quota for this deployment}"
GPU_IDS=${GPU_IDS:-0,1}
TARGET_PORT=${TARGET_PORT:-30000}
DRAFT_PORT=${DRAFT_PORT:-30001}
ZMQ_PORT=${ZMQ_PORT:-5557}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
TARGET_KV_TOKENS=${TARGET_KV_TOKENS:-65536}
DRAFT_KV_TOKENS=${DRAFT_KV_TOKENS:-131072}
CPU_MEMORY_GB=${CPU_MEMORY_GB:-32}
RUN_DIR=${RUN_DIR:-"$REPO/outputs/server_$(date +%Y%m%d_%H%M%S)_$$"}
[[ ! -e "$RUN_DIR" ]] || { echo "RUN_DIR already exists: $RUN_DIR" >&2; exit 1; }
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export TARGET_MODEL DRAFT_MODEL
export SPECSTREAM_OVERLAP_MODE=auto
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY
GPU_UUIDS=$(nvidia-smi -i "$GPU_IDS" --query-gpu=uuid --format=csv,noheader | paste -sd, -)
IFS=, read -ra GPU_ARRAY <<< "$GPU_UUIDS"
TP_SIZE=${#GPU_ARRAY[@]}
export SPECSTREAM_DRAFT_TP_SIZE=$TP_SIZE
export SPECSTREAM_TP_WINDOW_DIR="$RUN_DIR/tp_windows"
mkdir -p "$SPECSTREAM_TP_WINDOW_DIR"
SMCTRL_LIB="$REPO/csrc/specstream_smctrl/build/libsmctrl.so"
VALIDATOR="$REPO/csrc/specstream_smctrl/build/specstream_smctrl_validator"
test -s "$SMCTRL_LIB"
test -x "$VALIDATOR"
command -v nvidia-cuda-mps-control >/dev/null
command -v setsid >/dev/null
"$PYTHON" - "$TARGET_PORT" "$DRAFT_PORT" "$ZMQ_PORT" "$DRAFT_TPCS" <<'PY'
import os, socket, sys
from transformers import AutoTokenizer, AutoConfig
ports = [int(port) for port in sys.argv[1:4]]
assert len(set(ports)) == 3, 'Use three distinct ports'
assert int(sys.argv[4]) > 0, 'DRAFT_TPCS must be positive'
for port in ports:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
paths = [os.environ['TARGET_MODEL'], os.environ['DRAFT_MODEL']]
tokenizers = [AutoTokenizer.from_pretrained(p, trust_remote_code=True) for p in paths]
assert tokenizers[0].get_vocab() == tokenizers[1].get_vocab(), 'Target and Draft token mappings differ'
assert tokenizers[0].special_tokens_map == tokenizers[1].special_tokens_map, 'Special tokens differ'
assert tokenizers[0].chat_template == tokenizers[1].chat_template, 'Chat templates differ'
messages = [{'role': 'user', 'content': 'Summarize the document. 你好 123'}]
ids = [t.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False) for t in tokenizers]
assert ids[0] == ids[1], 'Rendered prompt IDs differ'
for path, tokenizer in zip(paths, tokenizers):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    assert max(tokenizer.get_vocab().values()) < config.vocab_size, 'Token IDs exceed model vocabulary'
print('Model pair and ports checked', flush=True)
PY
for uuid in "${GPU_ARRAY[@]}"; do
  CUDA_VISIBLE_DEVICES="$uuid" "$VALIDATOR" 0 "$DRAFT_TPCS" global > "$RUN_DIR/mask_${uuid}.log" 2>&1
done
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/specstream-mps-$$"
export CUDA_MPS_LOG_DIRECTORY="$RUN_DIR/mps"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
target_pid= draft_pid= mps_started=0
cleanup() {
  trap - EXIT INT TERM
  for pid in "$draft_pid" "$target_pid"; do
    if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
  done
  sleep 3
  for pid in "$draft_pid" "$target_pid"; do
    if [[ -n "$pid" ]]; then kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
  if (( mps_started )); then echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nvidia-cuda-mps-control -d
mps_started=1
COMMON=("$PYTHON" -m sglang.launch_server
  --host 127.0.0.1 --tp-size "$TP_SIZE" --context-length "$CONTEXT_LENGTH"
  --trust-remote-code --skip-server-warmup --page-size 1
  --attention-backend "${ATTENTION_BACKEND:-fa3}"
  --disable-radix-cache --disable-overlap-schedule --log-level info
  --speculative-algorithm SPECTRE --speculative-num-steps 7
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 8
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
  --specstream-smctrl-enabled --specstream-smctrl-library "$SMCTRL_LIB"
  --specstream-smctrl-mask-scope global)
TARGET=("${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
  --spectre-role target --mem-fraction-static "${TARGET_MEM_FRACTION:-0.55}"
  --max-total-tokens "$TARGET_KV_TOKENS" --prefill-max-requests 1
  --disable-cuda-graph --disable-piecewise-cuda-graph
  --spectre-fixed-q-mode parallel --spectre-require-draft
  --spectre-draft-timeout-action fallback
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
  --specstream-enabled --no-specstream-reference-attention
  --specstream-chunk-tokens 2048 --specstream-num-buffers 2
  --specstream-chunks-per-transfer 4 --specstream-layer-prefetch
  --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
  --specstream-gpu-history-cache-tokens 0 --specstream-gpu-history-min-free-tokens 0
  --specstream-cpu-memory-gb "$CPU_MEMORY_GB" --specstream-gpu-reserve-mb 1024
  --specstream-dynamic-q --specstream-q-candidates 2,4,6,8
  --specstream-q-switch-threshold 0.08 --specstream-cohort-enabled
  --specstream-max-cohort-size 8 --specstream-max-cohort-delay-us 200
  --specstream-profile-path "$RUN_DIR/profile.csv"
  --specstream-pcie-slack-coexec --specstream-coexec-require-mps
  --specstream-grant-token-quantum 1 --specstream-coexec-target-slowdown-budget 0.05
  --specstream-coexec-guard-us 200 --specstream-smctrl-calibration-tpcs "$DRAFT_TPCS"
  --specstream-smctrl-calibration-allow-overlap)
if (( TP_SIZE > 1 )); then
  TARGET+=(--specstream-tp-straggler-control --specstream-colocated-tp-rank 0)
fi
DRAFT=("${COMMON[@]}" --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
  --spectre-role draft --mem-fraction-static "${DRAFT_MEM_FRACTION:-0.80}"
  --max-total-tokens "$DRAFT_KV_TOKENS" --spectre-draft-priority
  --spectre-max-draft-priority-steps 16)
printf '%q ' "${TARGET[@]}" > "$RUN_DIR/target_command.txt"
printf '%q ' "${DRAFT[@]}" > "$RUN_DIR/draft_command.txt"
printf 'GPU_UUIDS=%s\nTP_SIZE=%s\nDRAFT_TPCS=%s\n' "$GPU_UUIDS" "$TP_SIZE" "$DRAFT_TPCS" > "$RUN_DIR/config.txt"
wait_ready() {
  "$PYTHON" - "$1" "$2" <<'PY'
import os, sys, time, urllib.request
port, pid = map(int, sys.argv[1:])
deadline = time.monotonic() + 600
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise TimeoutError(f'Server on port {port} did not become ready')
PY
}
echo "Loading Target; logs: $RUN_DIR"
setsid "${TARGET[@]}" > "$RUN_DIR/target.log" 2>&1 &
target_pid=$!
wait_ready "$TARGET_PORT" "$target_pid" || { tail -n 60 "$RUN_DIR/target.log"; exit 1; }
echo "Loading Draft"
setsid "${DRAFT[@]}" > "$RUN_DIR/draft.log" 2>&1 &
draft_pid=$!
wait_ready "$DRAFT_PORT" "$draft_pid" || { tail -n 60 "$RUN_DIR/draft.log"; exit 1; }
sleep 5
echo "SPECSTREAM_SERVER_READY=http://127.0.0.1:$TARGET_PORT"
touch "$RUN_DIR/ready.marker"
wait -n "$target_pid" "$draft_pid" || true
echo "A server process exited; see $RUN_DIR/target.log and draft.log" >&2
exit 1
