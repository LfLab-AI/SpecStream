#!/usr/bin/env bash
set -euo pipefail

# Minimal correctness test for in-process SpecStream innovation 2.
# Compares greedy output token IDs from:
#   P0 native STANDALONE Spec V2
#   P2 fixed h=4 Draft-ahead
#   P3 Auto Draft-ahead

TARGET_MODEL="${TARGET_MODEL:?Set TARGET_MODEL}"
DRAFT_MODEL="${DRAFT_MODEL:?Set DRAFT_MODEL}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
NUM_PROMPTS="${NUM_PROMPTS:-24}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-results/innovation2_inproc_minimal/correctness_${RUN_STAMP}}"

export PYTHONPATH="${PYTHONPATH:-$PWD/python}"
export SGLANG_ENABLE_SPEC_V2=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/outputs" "$RESULT_ROOT/profiles"

echo "[precheck] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python - <<'PY'
import torch

print("torch =", torch.__version__)
print("torch CUDA =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("visible device count =", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(
        "No CUDA device is visible. Set GPU_ID to a valid logical GPU index "
        "(normally GPU_ID=0 inside the AutoDL container)."
    )
print("device 0 =", torch.cuda.get_device_name(0))
PY

BASE_ARGS=(
  python -m sglang.launch_server
  --model-path "$TARGET_MODEL"
  --port "$SERVER_PORT"
  --context-length "$CONTEXT_LENGTH"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --skip-server-warmup
  --tp-size 1
  --dp-size 1
  --page-size 1
  --attention-backend "$ATTENTION_BACKEND"
  --disable-radix-cache
  --speculative-algorithm STANDALONE
  --speculative-draft-model-path "$DRAFT_MODEL"
  --speculative-num-steps 4
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 5
)

SERVER_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

trap stop_server EXIT INT TERM

start_server() {
  local method="$1"
  local log_dir="$RESULT_ROOT/logs/$method"
  local profile="$RESULT_ROOT/profiles/${method}.jsonl"
  local extra=()

  mkdir -p "$log_dir"
  case "$method" in
    P0_NATIVE)
      ;;
    P2_H4)
      extra+=(
        --specstream-inproc-enabled
        --specstream-inproc-mode ahead-free
        --specstream-inproc-ahead-depth 4
        --specstream-inproc-profile-path "$profile"
        --specstream-inproc-profile-interval 1
      )
      ;;
    P3_AUTO)
      extra+=(
        --specstream-inproc-enabled
        --specstream-inproc-mode auto
        --specstream-inproc-ahead-depth 4
        --specstream-inproc-min-reuse-ratio 0.25
        --specstream-inproc-profile-path "$profile"
        --specstream-inproc-profile-interval 1
      )
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 2
      ;;
  esac

  printf '%q ' "${BASE_ARGS[@]}" "${extra[@]}" > "$log_dir/launch_command.txt"
  printf '\n' >> "$log_dir/launch_command.txt"

  "${BASE_ARGS[@]}" "${extra[@]}" > "$log_dir/server.log" 2>&1 &
  SERVER_PID=$!

  local ready=0
  for _ in $(seq 1 600); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$method exited before readiness" >&2
      tail -200 "$log_dir/server.log" >&2
      return 3
    fi
    if curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "$method readiness timeout" >&2
    tail -200 "$log_dir/server.log" >&2
    return 4
  fi

  curl -fsS "http://127.0.0.1:${SERVER_PORT}/server_info" \
    > "$log_dir/server_info.json"
}

PROMPTS="$RESULT_ROOT/prompts.jsonl"
python - "$PROMPTS" "$NUM_PROMPTS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
limit = int(sys.argv[2])
base = [
    "Explain speculative decoding in two concise sentences.",
    "What is 37 multiplied by 19? Show the final number only.",
    "Write a Python function that reverses a list without calling reverse().",
    "Summarize why KV cache matters during autoregressive decoding.",
    "Translate into English: 推测性解码必须保持目标模型输出不变。",
    "Name three causes of high P99 latency in an online inference server.",
    "Given the sequence 2, 6, 12, 20, 30, give the next two values.",
    "Compare throughput and latency in one paragraph.",
    "Return a JSON object with keys name and purpose for a GPU stream.",
    "Explain the difference between a CUDA event and device synchronization.",
    "A server completes 240 requests in 60 seconds. What is its throughput?",
    "Write a short test case for longest common prefix.",
]
prompts = []
while len(prompts) < limit:
    index = len(prompts)
    prompt = base[index % len(base)] + f"\nCase identifier: {index}."
    if index % 6 == 5:
        prompt += "\nContext: " + ("GPU scheduling and KV cache. " * 256)
    prompts.append(prompt)

with path.open("w", encoding="utf-8") as output:
    for index, prompt in enumerate(prompts):
        output.write(json.dumps({
            "id": f"prompt-{index:04d}",
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }, ensure_ascii=False) + "\n")
PY

collect_outputs() {
  local method="$1"
  local output="$RESULT_ROOT/outputs/${method}.jsonl"
  python - "http://127.0.0.1:${SERVER_PORT}" "$PROMPTS" "$output" "$MAX_NEW_TOKENS" <<'PY'
import json
import sys
from pathlib import Path

import requests

base_url = sys.argv[1].rstrip("/")
prompt_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
max_new_tokens = int(sys.argv[4])

def get_token_id(item):
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return int(item[1])
    if isinstance(item, dict):
        if "token_id" in item:
            return int(item["token_id"])
        if "id" in item:
            return int(item["id"])
    raise ValueError(f"Unsupported output token record: {item!r}")

rows = [json.loads(line) for line in prompt_path.open(encoding="utf-8") if line.strip()]
with output_path.open("w", encoding="utf-8") as output:
    for index, row in enumerate(rows, 1):
        response = requests.post(
            base_url + "/generate",
            json={
                "text": row["prompt"],
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1.0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": 0,
                "top_logprobs_num": 0,
                "stream": False,
            },
            timeout=1800,
        )
        response.raise_for_status()
        obj = response.json()
        obj = obj[0] if isinstance(obj, list) else obj
        token_rows = obj["meta_info"]["output_token_logprobs"]
        output.write(json.dumps({
            "id": row["id"],
            "prompt_sha256": row["prompt_sha256"],
            "output_token_ids": [get_token_id(item) for item in token_rows],
            "text": obj.get("text", ""),
        }, ensure_ascii=False) + "\n")
        output.flush()
        print(f"{index}/{len(rows)} {row['id']}", flush=True)
PY
}

for method in P0_NATIVE P2_H4 P3_AUTO; do
  echo "[correctness] running $method"
  start_server "$method"
  collect_outputs "$method"
  stop_server
done

python - \
  "$RESULT_ROOT/outputs/P0_NATIVE.jsonl" \
  "$RESULT_ROOT/outputs/P2_H4.jsonl" \
  "$RESULT_ROOT/outputs/P3_AUTO.jsonl" \
  "$RESULT_ROOT/profiles/P2_H4.jsonl" \
  "$RESULT_ROOT/profiles/P3_AUTO.jsonl" <<'PY'
import json
import sys
from pathlib import Path

def load_outputs(path):
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in Path(path).open(encoding="utf-8")
            if line.strip()
        )
    }

reference = load_outputs(sys.argv[1])
failed = False
for candidate_path in sys.argv[2:4]:
    candidate = load_outputs(candidate_path)
    mismatches = []
    for rid in sorted(set(reference) | set(candidate)):
        if rid not in reference or rid not in candidate:
            mismatches.append((rid, "missing"))
            continue
        if reference[rid]["prompt_sha256"] != candidate[rid]["prompt_sha256"]:
            mismatches.append((rid, "prompt"))
        elif reference[rid]["output_token_ids"] != candidate[rid]["output_token_ids"]:
            mismatches.append((rid, "tokens"))
    print(Path(candidate_path).stem, "mismatches =", len(mismatches))
    for item in mismatches[:10]:
        print(" ", item)
    failed |= bool(mismatches)

for profile_path in sys.argv[4:]:
    rows = [
        json.loads(line)
        for line in Path(profile_path).open(encoding="utf-8")
        if line.strip()
    ]
    generated = sum(row["ahead_tokens_generated"] for row in rows)
    print(Path(profile_path).stem, "profile_rows =", len(rows), "ahead_generated =", generated)
    if not rows or generated <= 0:
        print("ERROR: Draft-ahead was not exercised")
        failed = True
    for row in rows:
        if row["ahead_tokens_generated"] != row["ahead_tokens_reused"] + row["ahead_tokens_discarded"]:
            print("ERROR: token accounting violation in", profile_path)
            failed = True
            break

if failed:
    print("CORRECTNESS TEST: FAIL")
    raise SystemExit(1)
print("CORRECTNESS TEST: PASS")
PY

echo "Results: $RESULT_ROOT"
