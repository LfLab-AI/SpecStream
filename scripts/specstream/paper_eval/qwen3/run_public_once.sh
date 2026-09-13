#!/usr/bin/env bash
_specstream_main() (
set -euo pipefail

for required_var in METHOD DATASET_TAG NUM_PROMPTS OUTPUT_LEN MAX_CONCURRENCY \
  RESULT_ROOT SPECSTREAM_PYTHON TARGET_MODEL DRAFT_MODEL TARGET_UUID DRAFT_UUID \
  COLOCATED_UUID; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "ERROR: required environment variable is unset: $required_var" >&2
    return 2
  fi
done

TARGET_TP_SIZE="${TARGET_TP_SIZE:-1}"
SPECSTREAM_DRAFT_TP_SIZE="${SPECSTREAM_DRAFT_TP_SIZE:-$TARGET_TP_SIZE}"
SPECSTREAM_OVERLAP_MODE="${SPECSTREAM_OVERLAP_MODE:-auto}"
SPECSTREAM_FIXED_Q="${SPECSTREAM_FIXED_Q:-0}"
TARGET_UUIDS="${TARGET_UUIDS:-$TARGET_UUID}"
COLOCATED_TP_RANK="${COLOCATED_TP_RANK:-0}"
[[ "$TARGET_TP_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: TARGET_TP_SIZE must be positive: $TARGET_TP_SIZE" >&2
  return 2
}
IFS=',' read -r -a target_uuid_array <<<"$TARGET_UUIDS"
if (( ${#target_uuid_array[@]} != TARGET_TP_SIZE )); then
  echo "ERROR: TARGET_UUIDS exposes ${#target_uuid_array[@]} GPU(s), but TARGET_TP_SIZE=$TARGET_TP_SIZE" >&2
  return 2
fi
(( COLOCATED_TP_RANK >= 0 && COLOCATED_TP_RANK < TARGET_TP_SIZE )) || {
  echo "ERROR: COLOCATED_TP_RANK=$COLOCATED_TP_RANK is outside TP=$TARGET_TP_SIZE" >&2
  return 2
}
[[ "${target_uuid_array[$COLOCATED_TP_RANK]}" == "$COLOCATED_UUID" ]] || {
  echo "ERROR: COLOCATED_UUID is not TARGET_UUIDS rank $COLOCATED_TP_RANK" >&2
  return 2
}

# Non-interactive SSH does not inherit `conda activate spectre`. Qwen3's
# first forward compiles local JIT kernels, so expose the environment's ninja
# and the CUDA toolkit used to build this checkout.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$(dirname "$SPECSTREAM_PYTHON"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
command -v ninja >/dev/null || {
  echo "ERROR: ninja is required for SGLang Qwen3 JIT kernels" >&2
  return 2
}

TARGET_PORT="${TARGET_PORT:-30000}"
DRAFT_PORT="${DRAFT_PORT:-30001}"
ZMQ_PORT="${ZMQ_PORT:-5557}"
SERVER_CONTEXT_LEN="${SERVER_CONTEXT_LEN:-40960}"
FINAL_DRAFT_TPCS="${FINAL_DRAFT_TPCS:-34}"
# Controlled-ablation defaults.  Every controlled method uses the same
# role-local upper bound; exact KV comparability is enforced separately by
# --max-total-tokens and the post-start capacity gate below.
SPECSTREAM_TARGET_MEM_FRACTION="${SPECSTREAM_TARGET_MEM_FRACTION:-0.50}"
SPECSTREAM_DRAFT_MEM_FRACTION="${SPECSTREAM_DRAFT_MEM_FRACTION:-0.72}"
SPECSTREAM_TARGET_MIN_KV_TOKENS="${SPECSTREAM_TARGET_MIN_KV_TOKENS:-${SPECSTREAM_MIN_COLOCATED_KV_TOKENS:-0}}"
SPECSTREAM_DRAFT_MIN_KV_TOKENS="${SPECSTREAM_DRAFT_MIN_KV_TOKENS:-${SPECSTREAM_MIN_COLOCATED_KV_TOKENS:-0}}"
SPECSTREAM_TARGET_MAX_TOTAL_TOKENS="${SPECSTREAM_TARGET_MAX_TOTAL_TOKENS:-0}"
SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS="${SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS:-0}"
SPECSTREAM_PREFILL_MAX_REQUESTS="${SPECSTREAM_PREFILL_MAX_REQUESTS:-0}"
SPECSTREAM_GPU_HISTORY_CACHE_TOKENS="${SPECSTREAM_GPU_HISTORY_CACHE_TOKENS:-49152}"
SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS="${SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS:-0}"
SPECSTREAM_REQUIRE_SLACK_FILL="${SPECSTREAM_REQUIRE_SLACK_FILL:-0}"
SPECSTREAM_GRANT_TOKEN_QUANTUM="${SPECSTREAM_GRANT_TOKEN_QUANTUM:-1}"
SPECSTREAM_NUM_BUFFERS="${SPECSTREAM_NUM_BUFFERS:-2}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
SEED="${SEED:-1}"
CASE_TIMEOUT_S="${CASE_TIMEOUT_S:-3600}"
SPECSTREAM_DRY_RUN="${SPECSTREAM_DRY_RUN:-0}"
CLIENT_MODE="${CLIENT_MODE:-performance}"
SERVER_LOG_LEVEL="${SPECSTREAM_SERVER_LOG_LEVEL:-info}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-}"
INPUT_LEN="${INPUT_LEN:-16384}"
CASE_TAG="${METHOD}_${DATASET_TAG}_c${MAX_CONCURRENCY}"
CASE_ROOT="$RESULT_ROOT/logs/$CASE_TAG"
PROFILE_CSV="$RESULT_ROOT/profiles/$CASE_TAG.csv"
OUT_JSONL="$RESULT_ROOT/bench/$CASE_TAG.jsonl"
CASE_COMPLETE_MARKER="$CASE_ROOT/case_complete.marker"

if [[ "$CLIENT_MODE" == accuracy && "${ACCURACY_DATASET:-}" == longbench_v2 ]] \
  && (( OUTPUT_LEN < 256 )); then
  echo "ERROR: LongBench-v2 accuracy requires OUTPUT_LEN>=256; got $OUTPUT_LEN" >&2
  return 2
fi

if [[ "$DATASET_NAME" == sharegpt ]]; then
  [[ -s "$DATASET_PATH" ]] || { echo "ERROR: ShareGPT manifest missing: $DATASET_PATH" >&2; return 2; }
fi
if [[ -e "$CASE_COMPLETE_MARKER" || -s "$PROFILE_CSV" || -s "$OUT_JSONL" ]]; then
  echo "ERROR: case artifacts already exist for $CASE_TAG; use a fresh RESULT_ROOT" >&2
  return 2
fi
mkdir -p "$CASE_ROOT" "$RESULT_ROOT"/{bench,profiles,gpu_monitor,summary,env}

port_open(){ (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if [[ "$SPECSTREAM_DRY_RUN" != 1 ]]; then
  for port in "$TARGET_PORT" "$DRAFT_PORT"; do
    port_open "$port" && { echo "ERROR: stale server on port $port" >&2; return 2; } || true
  done
fi

COMMON=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --context-length "$SERVER_CONTEXT_LEN"
  --skip-server-warmup
  --page-size 1
  --attention-backend fa3
  --disable-radix-cache
  --disable-cuda-graph
  --disable-piecewise-cuda-graph
  --disable-overlap-schedule
  --log-level "$SERVER_LOG_LEVEL"
)

# The remote Drafter only runs ordinary decode kernels. Keep the Target's
# conservative graph settings, but let the final SpecStream method use the
# normal SGLang CUDA-graph fast path for its repeated q-step draft loop.
DRAFT_CUDA_COMMON=()
for common_arg in "${COMMON[@]}"; do
  case "$common_arg" in
    --disable-cuda-graph|--disable-piecewise-cuda-graph) ;;
    *) DRAFT_CUDA_COMMON+=("$common_arg") ;;
  esac
done

# HiCache requires radix cache and is incompatible with COMMON's
# --disable-radix-cache. Keep a separate, explicit common vector so the exact
# cache configuration is visible in target_command.txt.
HICACHE_COMMON=(
  "$SPECSTREAM_PYTHON" -m sglang.launch_server
  --context-length "$SERVER_CONTEXT_LEN"
  --skip-server-warmup
  --page-size 64
  --attention-backend fa3
  --decode-attention-backend flashinfer
  --disable-cuda-graph
  --disable-piecewise-cuda-graph
  --disable-overlap-schedule
  --log-level "$SERVER_LOG_LEVEL"
  --enable-hierarchical-cache
  --hicache-size "${HICACHE_SIZE_GB:-64}"
  --hicache-io-backend kernel
  --hicache-write-policy write_through
  --hicache-mem-layout layer_first
)

TARGET_ARGS=()
DRAFT_ARGS=()
TARGET_VISIBLE="$TARGET_UUIDS"
DRAFT_VISIBLE="$DRAFT_UUID"
START_DRAFT=0
USE_MPS=0

case "$METHOD" in
  AR_HICACHE)
    TARGET_ARGS=("${HICACHE_COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT" --mem-fraction-static 0.72)
    ;;
  AR)
    TARGET_ARGS=("${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT" --mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION")
    ;;
  SGLANG_SD)
    TARGET_ARGS=(
      "${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      # Both models live in this process.  Use the matrix-wide calibrated
      # fraction instead of silently overriding it with a legacy value.
      --mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION"
      --speculative-algorithm STANDALONE
      --speculative-draft-model-path "$DRAFT_MODEL"
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
    )
    ;;
  SGLANG_SD_HICACHE)
    TARGET_ARGS=(
      "${HICACHE_COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      --mem-fraction-static 0.54
      --speculative-algorithm STANDALONE
      --speculative-draft-model-path "$DRAFT_MODEL"
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
    )
    ;;
  SPECTRE_2GPU)
    START_DRAFT=1
    TARGET_ARGS=(
      "${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      --mem-fraction-static 0.85
      --speculative-algorithm SPECTRE --spectre-role target
      --speculative-num-steps 3 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4 --spectre-fixed-q-mode parallel
      --spectre-require-draft --spectre-draft-timeout-action fallback
      --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
    )
    DRAFT_ARGS=(
      "${COMMON[@]}" --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
      --mem-fraction-static 0.85
      --speculative-algorithm SPECTRE --spectre-role draft
      --speculative-num-steps 3 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4 --spectre-draft-priority
      --spectre-max-draft-priority-steps 8
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
    )
    ;;
  I1_GPU_ONLY)
    START_DRAFT=1
    USE_MPS=1
    TARGET_VISIBLE="$TARGET_UUIDS"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    TARGET_ARGS=(
      "${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      --mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role target
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-fixed-q-mode ordinary
      --spectre-require-draft --spectre-draft-timeout-action fallback
      --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
    )
    DRAFT_ARGS=(
      "${COMMON[@]}" --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
      --mem-fraction-static "$SPECSTREAM_DRAFT_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role draft
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-draft-priority
      --spectre-max-draft-priority-steps 16
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
    )
    ;;
  I1_REF|I1_FUSED|I1_SHADOW|I1_SEALED_HISTORY|K1|K2|K3|K4|K5|SGLANG_SD_KV_OFFLOAD)
    START_DRAFT=1
    USE_MPS=1
    TARGET_VISIBLE="$TARGET_UUIDS"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    if [[ "$METHOD" == K4 || "$METHOD" == K5 || "$METHOD" == I1_SEALED_HISTORY ]]; then
      K_STEPS=7
      K_TOKENS=8
    else
      K_STEPS=3
      K_TOKENS=4
    fi
    if [[ "$METHOD" == SGLANG_SD_KV_OFFLOAD || "$METHOD" == I1_SEALED_HISTORY ]]; then
      K_CHUNKS_PER_TRANSFER=1
    else
      K_CHUNKS_PER_TRANSFER=4
    fi
    TARGET_ARGS=(
      "${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      --mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role target
      --speculative-num-steps "$K_STEPS" --speculative-eagle-topk 1
      --speculative-num-draft-tokens "$K_TOKENS" --spectre-fixed-q-mode ordinary
      --spectre-require-draft --spectre-draft-timeout-action fallback
      --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
      --specstream-enabled
      --specstream-chunk-tokens 2048
      --specstream-chunks-per-transfer "$K_CHUNKS_PER_TRANSFER"
      --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
      --specstream-gpu-history-cache-tokens "$SPECSTREAM_GPU_HISTORY_CACHE_TOKENS"
      --specstream-gpu-history-min-free-tokens "$SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS"
      --specstream-cpu-memory-gb 128 --specstream-gpu-reserve-mb 1024
      --specstream-profile-path "$PROFILE_CSV"
    )
    case "$METHOD" in
      I1_REF) TARGET_ARGS+=(--specstream-reference-attention --specstream-num-buffers 1 --no-specstream-layer-prefetch) ;;
      I1_FUSED) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-num-buffers 1 --no-specstream-layer-prefetch) ;;
      I1_SHADOW) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-shadow-attention --specstream-num-buffers 1 --no-specstream-layer-prefetch) ;;
      I1_SEALED_HISTORY) TARGET_ARGS+=(--specstream-reference-attention --specstream-full-restore-baseline --specstream-num-buffers 1 --no-specstream-layer-prefetch --specstream-strict-invariants) ;;
      K1) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-full-restore-baseline --specstream-num-buffers 1 --no-specstream-layer-prefetch --specstream-serialize-h2d) ;;
      K2) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-num-buffers 1 --no-specstream-layer-prefetch --specstream-serialize-h2d) ;;
      K3) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-num-buffers 2 --specstream-layer-prefetch --no-specstream-serialize-h2d) ;;
      SGLANG_SD_KV_OFFLOAD) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-num-buffers 2 --specstream-layer-prefetch --specstream-force-ordinary-mode) ;;
      K4) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-num-buffers 2 --specstream-layer-prefetch --no-specstream-serialize-h2d --specstream-dynamic-q --specstream-force-ordinary-mode --specstream-q-candidates 2,4,6,8 --specstream-q-switch-threshold 0.08) ;;
      K5) TARGET_ARGS+=(--no-specstream-reference-attention --specstream-num-buffers 2 --specstream-layer-prefetch --no-specstream-serialize-h2d --specstream-dynamic-q --specstream-force-ordinary-mode --specstream-q-candidates 2,4,6,8 --specstream-q-switch-threshold 0.08 --specstream-cohort-enabled --specstream-max-cohort-size 8 --specstream-max-cohort-delay-us 200) ;;
    esac
    DRAFT_ARGS=(
      "${COMMON[@]}" --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
      --mem-fraction-static "$SPECSTREAM_DRAFT_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role draft
      --speculative-num-steps "$K_STEPS" --speculative-eagle-topk 1
      --speculative-num-draft-tokens "$K_TOKENS" --spectre-draft-priority
      --spectre-max-draft-priority-steps 16
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
    )
    ;;
  A)
    START_DRAFT=1
    USE_MPS=1
    TARGET_VISIBLE="$TARGET_UUIDS"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    TARGET_ARGS=(
      "${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      --mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role target
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-fixed-q-mode ordinary
      --spectre-require-draft --spectre-draft-timeout-action fallback
      --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
      --specstream-enabled --no-specstream-reference-attention
      --specstream-chunk-tokens 2048 --specstream-num-buffers 2
      --specstream-chunks-per-transfer 4 --specstream-layer-prefetch
      --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
      --specstream-gpu-history-cache-tokens "$SPECSTREAM_GPU_HISTORY_CACHE_TOKENS"
      --specstream-gpu-history-min-free-tokens "$SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS"
      --specstream-cpu-memory-gb 128 --specstream-gpu-reserve-mb 1024
      --specstream-dynamic-q --specstream-force-ordinary-mode
      --specstream-q-candidates 2,4,6,8
      --specstream-q-switch-threshold 0.08
      --specstream-cohort-enabled --specstream-max-cohort-size 8
      --specstream-max-cohort-delay-us 200 --specstream-profile-path "$PROFILE_CSV"
    )
    DRAFT_ARGS=(
      "${COMMON[@]}" --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
      --mem-fraction-static "$SPECSTREAM_DRAFT_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role draft
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-draft-priority
      --spectre-max-draft-priority-steps 16
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
    )
    ;;
  B)
    START_DRAFT=1
    TARGET_ARGS=(
      "${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      --mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role target
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-fixed-q-mode parallel
      --spectre-require-draft --spectre-draft-timeout-action fallback
      --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
      --specstream-enabled --no-specstream-reference-attention
      --specstream-chunk-tokens 2048 --specstream-num-buffers 2
      --specstream-chunks-per-transfer 4 --specstream-layer-prefetch
      --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
      --specstream-gpu-history-cache-tokens "$SPECSTREAM_GPU_HISTORY_CACHE_TOKENS"
      --specstream-gpu-history-min-free-tokens "$SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS"
      --specstream-cpu-memory-gb 128 --specstream-gpu-reserve-mb 1024
      --specstream-dynamic-q --specstream-q-candidates 2,4,6,8
      --specstream-q-switch-threshold 0.08
      --specstream-cohort-enabled --specstream-max-cohort-size 8
      --specstream-max-cohort-delay-us 200 --specstream-profile-path "$PROFILE_CSV"
    )
    DRAFT_ARGS=(
      "${COMMON[@]}" --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
      --mem-fraction-static "$SPECSTREAM_DRAFT_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role draft
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-draft-priority
      --spectre-max-draft-priority-steps 16
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
    )
    ;;
  C|SPECSTREAM_1GPU)
    START_DRAFT=1
    USE_MPS=1
    TARGET_VISIBLE="$TARGET_UUIDS"
    DRAFT_VISIBLE="$COLOCATED_UUID"
    : "${SMCTRL_LIB:?}"
    : "${SMCTRL_MASK_SCOPE:?}"
    : "${SPECSTREAM_SMCTRL_VALIDATED:?}"
    test -s "$SMCTRL_LIB"

    TARGET_ARGS=(
      "${COMMON[@]}" --model-path "$TARGET_MODEL" --port "$TARGET_PORT"
      --mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role target
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-fixed-q-mode parallel
      --spectre-require-draft --spectre-draft-timeout-action fallback
      --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
      --specstream-enabled --no-specstream-reference-attention
      --specstream-chunk-tokens 2048 --specstream-num-buffers "$SPECSTREAM_NUM_BUFFERS"
      --specstream-chunks-per-transfer 4 --specstream-layer-prefetch
      --specstream-active-tail-tokens 512 --specstream-min-history-tokens 8192
      --specstream-gpu-history-cache-tokens "$SPECSTREAM_GPU_HISTORY_CACHE_TOKENS"
      --specstream-gpu-history-min-free-tokens "$SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS"
      --specstream-cpu-memory-gb 128 --specstream-gpu-reserve-mb 1024
      --specstream-dynamic-q --specstream-q-candidates 2,4,6,8
      --specstream-q-switch-threshold 0.08
      --specstream-cohort-enabled --specstream-max-cohort-size 8
      --specstream-max-cohort-delay-us 200 --specstream-profile-path "$PROFILE_CSV"
      --specstream-pcie-slack-coexec --specstream-smctrl-enabled
      --specstream-smctrl-library "$SMCTRL_LIB"
      --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE"
      --specstream-coexec-require-mps --specstream-grant-token-quantum "$SPECSTREAM_GRANT_TOKEN_QUANTUM"
      --specstream-coexec-target-slowdown-budget 0.05
      --specstream-coexec-guard-us 200
      --specstream-smctrl-calibration-tpcs "$FINAL_DRAFT_TPCS"
      --specstream-smctrl-calibration-allow-overlap
    )
    DRAFT_ARGS=(
      "${DRAFT_CUDA_COMMON[@]}" --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT"
      --mem-fraction-static "$SPECSTREAM_DRAFT_MEM_FRACTION"
      --speculative-algorithm SPECTRE --spectre-role draft
      --speculative-num-steps 7 --speculative-eagle-topk 1
      --speculative-num-draft-tokens 8 --spectre-draft-priority
      --spectre-max-draft-priority-steps 16
      --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT"
      --specstream-smctrl-enabled --specstream-smctrl-library "$SMCTRL_LIB"
      --specstream-smctrl-mask-scope "$SMCTRL_MASK_SCOPE"
    )
    ;;
  *) echo "ERROR: unsupported METHOD=$METHOD" >&2; return 2 ;;
esac

# Colocated external Drafts follow Target TP by default. TP1 remains an
# explicit placement ablation; dedicated-GPU baselines retain their topology.
TARGET_ARGS+=(--tp-size "$TARGET_TP_SIZE")
RECORDED_DRAFT_TP_SIZE=disabled
unset SPECSTREAM_TP_WINDOW_DIR
case "$METHOD" in
  I1_GPU_ONLY|I1_REF|I1_FUSED|I1_SHADOW|I1_SEALED_HISTORY|K1|K2|K3|K4|K5|SGLANG_SD_KV_OFFLOAD|A|C|SPECSTREAM_1GPU)
    [[ "$SPECSTREAM_DRAFT_TP_SIZE" == 1 || "$SPECSTREAM_DRAFT_TP_SIZE" == "$TARGET_TP_SIZE" ]] || {
      echo "ERROR: Draft TP must be 1 or TARGET_TP_SIZE" >&2; return 2;
    }
    DRAFT_ARGS+=(--tp-size "$SPECSTREAM_DRAFT_TP_SIZE")
    DRAFT_VISIBLE="$COLOCATED_UUID"
    if (( SPECSTREAM_DRAFT_TP_SIZE > 1 )); then
      DRAFT_VISIBLE="$TARGET_UUIDS"
    fi
    RECORDED_DRAFT_TP_SIZE="$SPECSTREAM_DRAFT_TP_SIZE"
    export SPECSTREAM_DRAFT_TP_SIZE
    if [[ "$METHOD" != C && "$METHOD" != SPECSTREAM_1GPU ]]; then
      SPECSTREAM_OVERLAP_MODE=serial
    fi
    ;;
  B|SPECTRE_2GPU)
    DRAFT_ARGS+=(--tp-size 1)
    RECORDED_DRAFT_TP_SIZE=1
    export SPECSTREAM_DRAFT_TP_SIZE=1
    ;;
  SGLANG_SD|SGLANG_SD_HICACHE)
    RECORDED_DRAFT_TP_SIZE="$TARGET_TP_SIZE"
    ;;
esac
if [[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]]; then
  [[ "$SPECSTREAM_OVERLAP_MODE" == auto || "$SPECSTREAM_OVERLAP_MODE" == serial ]] || {
    echo "ERROR: SPECSTREAM_OVERLAP_MODE must be auto or serial" >&2; return 2;
  }
  [[ "$SPECSTREAM_FIXED_Q" =~ ^(0|[2-8])$ ]] || {
    echo "ERROR: SPECSTREAM_FIXED_Q must be 0 (dynamic) or 2..8" >&2; return 2;
  }
  if (( SPECSTREAM_DRAFT_TP_SIZE > 1 )); then
    export SPECSTREAM_TP_WINDOW_DIR="$CASE_ROOT/tp_windows"
    mkdir -p "$SPECSTREAM_TP_WINDOW_DIR"
  fi
  if [[ "$SPECSTREAM_OVERLAP_MODE" == serial ]]; then
    TARGET_ARGS+=(--specstream-force-ordinary-mode)
  fi
  if (( SPECSTREAM_FIXED_Q > 0 )); then
    # Singleton controller candidate preserves slowdown protection and mode
    # selection while holding q equal across placement/overlap comparisons.
    TARGET_ARGS+=(--specstream-q-candidates "$SPECSTREAM_FIXED_Q"
      --speculative-num-steps "$((SPECSTREAM_FIXED_Q - 1))"
      --speculative-num-draft-tokens "$SPECSTREAM_FIXED_Q")
    DRAFT_ARGS+=(--speculative-num-steps "$((SPECSTREAM_FIXED_Q - 1))"
      --speculative-num-draft-tokens "$SPECSTREAM_FIXED_Q")
  fi
fi
if [[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]] && (( TARGET_TP_SIZE > 1 )); then
  TARGET_ARGS+=(
    --specstream-tp-straggler-control
    --specstream-colocated-tp-rank "$COLOCATED_TP_RANK"
  )
fi

# Exact role-local KV caps make colocated variants reproducible.  The memory
# fractions remain upper ceilings; max-total-tokens fixes the actual pool when
# the profiled capacity is large enough.
if (( SPECSTREAM_TARGET_MAX_TOTAL_TOKENS > 0 )); then
  TARGET_ARGS+=(--max-total-tokens "$SPECSTREAM_TARGET_MAX_TOTAL_TOKENS")
fi
if (( START_DRAFT && SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS > 0 )); then
  DRAFT_ARGS+=(--max-total-tokens "$SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS")
fi

# Serializing Target prefill bounds the pre-seal peak without changing decode
# concurrency.  Apply the same setting to every controlled SPECTRE variant so
# it cannot become an ablation variable.
case "$METHOD" in
  I1_GPU_ONLY|I1_REF|I1_FUSED|I1_SHADOW|I1_SEALED_HISTORY|K1|K2|K3|K4|K5|A|B|C|SPECSTREAM_1GPU|SGLANG_SD_KV_OFFLOAD)
    if (( SPECSTREAM_PREFILL_MAX_REQUESTS > 0 )); then
      TARGET_ARGS+=(--prefill-max-requests "$SPECSTREAM_PREFILL_MAX_REQUESTS")
    fi
    ;;
esac

printf '%q ' "${TARGET_ARGS[@]}" > "$CASE_ROOT/target_command.txt"; printf '\n' >> "$CASE_ROOT/target_command.txt"
if (( START_DRAFT )); then
  printf '%q ' "${DRAFT_ARGS[@]}" > "$CASE_ROOT/draft_command.txt"; printf '\n' >> "$CASE_ROOT/draft_command.txt"
fi
if [[ "$METHOD" == SGLANG_SD_KV_OFFLOAD ]]; then
  for required_flag in \
    '--spectre-fixed-q-mode ordinary' \
    '--speculative-num-draft-tokens 4' \
    '--specstream-enabled' \
    '--specstream-chunks-per-transfer 1' \
    '--specstream-num-buffers 2' \
    '--specstream-layer-prefetch' \
    '--specstream-force-ordinary-mode'; do
    grep -Fq -- "$required_flag" "$CASE_ROOT/target_command.txt" || {
      echo "ERROR: SGLANG_SD_KV_OFFLOAD missing required setting: $required_flag" >&2
      return 2
    }
  done
  if grep -Eq -- \
    '--enable-hierarchical-cache|--specstream-dynamic-q|--specstream-cohort-enabled|--specstream-pcie-slack-coexec|--specstream-smctrl-enabled' \
    "$CASE_ROOT/target_command.txt"; then
    echo "ERROR: SGLANG_SD_KV_OFFLOAD enabled a forbidden HiCache or SpecStream core feature" >&2
    return 2
  fi
fi
case "$METHOD" in
  AR|AR_HICACHE)
    RECORDED_VERIFY_Q=disabled
    RECORDED_SPEC_STEPS=disabled
    RECORDED_DRAFT_TOKENS=disabled
    RECORDED_Q_CANDIDATES=disabled
    ;;
  I1_GPU_ONLY|I1_SEALED_HISTORY)
    RECORDED_VERIFY_Q=8
    RECORDED_SPEC_STEPS=7
    RECORDED_DRAFT_TOKENS=8
    RECORDED_Q_CANDIDATES=disabled
    ;;
  C|SPECSTREAM_1GPU)
    RECORDED_VERIFY_Q=dynamic_max_8
    RECORDED_SPEC_STEPS=7
    RECORDED_DRAFT_TOKENS=8
    RECORDED_Q_CANDIDATES=2,4,6,8
    ;;
  A|B|K4|K5)
    RECORDED_VERIFY_Q=dynamic_max_8
    RECORDED_SPEC_STEPS=7
    RECORDED_DRAFT_TOKENS=8
    RECORDED_Q_CANDIDATES=2,4,6,8
    ;;
  *)
    RECORDED_VERIFY_Q=4
    RECORDED_SPEC_STEPS=3
    RECORDED_DRAFT_TOKENS=4
    RECORDED_Q_CANDIDATES=disabled
    ;;
esac
case "$METHOD" in
  AR|AR_HICACHE) RECORDED_FIXED_Q_MODE=disabled ;;
  SGLANG_SD|SGLANG_SD_HICACHE) RECORDED_FIXED_Q_MODE=standalone ;;
  A|I1_GPU_ONLY|I1_REF|I1_FUSED|I1_SHADOW|I1_SEALED_HISTORY|K1|K2|K3|K4|K5|SGLANG_SD_KV_OFFLOAD) RECORDED_FIXED_Q_MODE=ordinary ;;
  *) RECORDED_FIXED_Q_MODE=parallel ;;
esac
case "$METHOD" in
  AR|AR_HICACHE) RECORDED_SPECULATIVE_ENGINE=disabled ;;
  SGLANG_SD|SGLANG_SD_HICACHE) RECORDED_SPECULATIVE_ENGINE=STANDALONE ;;
  *) RECORDED_SPECULATIVE_ENGINE=SPECTRE ;;
esac
case "$METHOD" in
  I1_GPU_ONLY)
    RECORDED_KV_MANAGEMENT=gpu_only_no_offload
    RECORDED_CHUNKS_PER_TRANSFER=disabled
    RECORDED_NUM_BUFFERS=disabled
    RECORDED_LAYER_PREFETCH=0
    RECORDED_CROSS_QUERY_COHORT=0
    ;;
  I1_SEALED_HISTORY)
    RECORDED_KV_MANAGEMENT=cpu_sealed_history_full_restore
    RECORDED_CHUNKS_PER_TRANSFER=1
    RECORDED_NUM_BUFFERS=1
    RECORDED_LAYER_PREFETCH=0
    RECORDED_CROSS_QUERY_COHORT=0
    ;;
  SGLANG_SD_KV_OFFLOAD)
    RECORDED_KV_MANAGEMENT=cpu_history_independent_streaming
    RECORDED_CHUNKS_PER_TRANSFER=1
    RECORDED_NUM_BUFFERS=2
    RECORDED_LAYER_PREFETCH=1
    RECORDED_CROSS_QUERY_COHORT=0
    ;;
  K3)
    RECORDED_KV_MANAGEMENT=cpu_history_grouped_streaming
    RECORDED_CHUNKS_PER_TRANSFER=4
    RECORDED_NUM_BUFFERS=2
    RECORDED_LAYER_PREFETCH=1
    RECORDED_CROSS_QUERY_COHORT=0
    ;;
  C|SPECSTREAM_1GPU|A|B|K4|K5)
    RECORDED_KV_MANAGEMENT=cpu_history_grouped_streaming
    RECORDED_CHUNKS_PER_TRANSFER=4
    RECORDED_NUM_BUFFERS=2
    RECORDED_LAYER_PREFETCH=1
    RECORDED_CROSS_QUERY_COHORT=$([[ "$METHOD" == K4 ]] && echo 0 || echo 1)
    if [[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]]; then RECORDED_NUM_BUFFERS="$SPECSTREAM_NUM_BUFFERS"; fi
    ;;
  K1)
    RECORDED_KV_MANAGEMENT=cpu_history_full_restore
    RECORDED_CHUNKS_PER_TRANSFER=4
    RECORDED_NUM_BUFFERS=1
    RECORDED_LAYER_PREFETCH=0
    RECORDED_CROSS_QUERY_COHORT=0
    ;;
  K2|I1_REF|I1_FUSED|I1_SHADOW)
    RECORDED_KV_MANAGEMENT=cpu_history_grouped_streaming
    RECORDED_CHUNKS_PER_TRANSFER=4
    RECORDED_NUM_BUFFERS=1
    RECORDED_LAYER_PREFETCH=0
    RECORDED_CROSS_QUERY_COHORT=0
    ;;
  *)
    RECORDED_KV_MANAGEMENT=disabled
    RECORDED_CHUNKS_PER_TRANSFER=disabled
    RECORDED_NUM_BUFFERS=disabled
    RECORDED_LAYER_PREFETCH=0
    RECORDED_CROSS_QUERY_COHORT=0
    ;;
esac

case "$METHOD" in
  K1|K2)
    RECORDED_SERIALIZE_H2D=1
    RECORDED_H2D_EXECUTION=serialized
    ;;
  K3|K4|K5)
    RECORDED_SERIALIZE_H2D=0
    RECORDED_H2D_EXECUTION=async_copy_stream
    ;;
  *)
    RECORDED_SERIALIZE_H2D=disabled
    RECORDED_H2D_EXECUTION=disabled
    ;;
esac

arg_value_from_array() {
  local needle="$1"
  shift
  local previous="" value
  for value in "$@"; do
    if [[ "$previous" == "$needle" ]]; then
      printf '%s' "$value"
      return 0
    fi
    previous="$value"
  done
  printf 'disabled'
}

RECORDED_TARGET_MEM_FRACTION=$(arg_value_from_array --mem-fraction-static "${TARGET_ARGS[@]}")
if [[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]]; then
  if (( SPECSTREAM_FIXED_Q > 0 )); then
    RECORDED_VERIFY_Q="$SPECSTREAM_FIXED_Q"
    RECORDED_SPEC_STEPS="$((SPECSTREAM_FIXED_Q - 1))"
    RECORDED_DRAFT_TOKENS="$SPECSTREAM_FIXED_Q"
    RECORDED_Q_CANDIDATES="$SPECSTREAM_FIXED_Q"
  fi
  if [[ "$SPECSTREAM_OVERLAP_MODE" == serial ]]; then
    RECORDED_FIXED_Q_MODE=ordinary
  fi
fi
RECORDED_DRAFT_MEM_FRACTION=disabled
if (( START_DRAFT )); then
  RECORDED_DRAFT_MEM_FRACTION=$(arg_value_from_array --mem-fraction-static "${DRAFT_ARGS[@]}")
fi
cat > "$CASE_ROOT/config.env" <<EOF
METHOD=$METHOD
CASE_TAG=$CASE_TAG
MODEL_TAG=${MODEL_TAG:-qwen3_unspecified}
TARGET_MODEL=$TARGET_MODEL
DRAFT_MODEL=$DRAFT_MODEL
TARGET_VISIBLE=$TARGET_VISIBLE
DRAFT_VISIBLE=$DRAFT_VISIBLE
TARGET_TP_SIZE=$TARGET_TP_SIZE
DRAFT_TP_SIZE=$RECORDED_DRAFT_TP_SIZE
SPECSTREAM_OVERLAP_MODE=$SPECSTREAM_OVERLAP_MODE
SPECSTREAM_FIXED_Q=$SPECSTREAM_FIXED_Q
COLOCATED_TP_RANK=$COLOCATED_TP_RANK
DATASET_NAME=$DATASET_NAME
DATASET_TAG=$DATASET_TAG
INPUT_LEN=$INPUT_LEN
OUTPUT_LEN=$OUTPUT_LEN
NUM_PROMPTS=$NUM_PROMPTS
MAX_CONCURRENCY=$MAX_CONCURRENCY
REQUEST_RATE=$REQUEST_RATE
WARMUP_REQUESTS=$WARMUP_REQUESTS
SEED=$SEED
SERVER_CONTEXT_LEN=$SERVER_CONTEXT_LEN
DECODING_MODE=$([[ "$METHOD" == AR || "$METHOD" == AR_HICACHE ]] && echo autoregressive || echo speculative)
SPECULATIVE_ENGINE=$RECORDED_SPECULATIVE_ENGINE
VERIFY_Q=$RECORDED_VERIFY_Q
SPECULATIVE_NUM_STEPS=$RECORDED_SPEC_STEPS
SPECULATIVE_NUM_DRAFT_TOKENS=$RECORDED_DRAFT_TOKENS
FIXED_Q_MODE=$RECORDED_FIXED_Q_MODE
KV_MANAGEMENT=$RECORDED_KV_MANAGEMENT
CHUNKS_PER_TRANSFER=$RECORDED_CHUNKS_PER_TRANSFER
NUM_STAGING_BUFFERS=$RECORDED_NUM_BUFFERS
LAYER_PREFETCH=$RECORDED_LAYER_PREFETCH
SERIALIZE_H2D=$RECORDED_SERIALIZE_H2D
H2D_EXECUTION=$RECORDED_H2D_EXECUTION
CROSS_QUERY_COHORT=$RECORDED_CROSS_QUERY_COHORT
DYNAMIC_Q=$([[ "$METHOD" == A || "$METHOD" == B || "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU || "$METHOD" == K4 || "$METHOD" == K5 ]] && echo 1 || echo 0)
Q_CANDIDATES=$RECORDED_Q_CANDIDATES
COHORT_ENABLED=$([[ "$METHOD" == A || "$METHOD" == B || "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU || "$METHOD" == K5 ]] && echo 1 || echo 0)
MAX_COHORT_SIZE=$([[ "$METHOD" == A || "$METHOD" == B || "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU || "$METHOD" == K5 ]] && echo 8 || echo disabled)
MAX_COHORT_DELAY_US=$([[ "$METHOD" == A || "$METHOD" == B || "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU || "$METHOD" == K5 ]] && echo 200 || echo disabled)
PCIE_SLACK_COEXEC=$([[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]] && echo 1 || echo 0)
REQUIRE_SLACK_FILL=$([[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]] && echo "$SPECSTREAM_REQUIRE_SLACK_FILL" || echo disabled)
SHORT_CONTEXT_SERIAL_GATE=$([[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]] && echo history_tokens_zero || echo disabled)
SMCTRL_MASK_SCOPE=$([[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]] && echo "$SMCTRL_MASK_SCOPE" || echo disabled)
FINAL_DRAFT_TPCS=$([[ "$METHOD" == C || "$METHOD" == SPECSTREAM_1GPU ]] && echo "$FINAL_DRAFT_TPCS" || echo disabled)
TARGET_MEM_FRACTION=$RECORDED_TARGET_MEM_FRACTION
DRAFT_MEM_FRACTION=$RECORDED_DRAFT_MEM_FRACTION
TARGET_MAX_TOTAL_TOKENS=$SPECSTREAM_TARGET_MAX_TOTAL_TOKENS
DRAFT_MAX_TOTAL_TOKENS=$SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS
TARGET_MIN_KV_TOKENS=$SPECSTREAM_TARGET_MIN_KV_TOKENS
DRAFT_MIN_KV_TOKENS=$SPECSTREAM_DRAFT_MIN_KV_TOKENS
PREFILL_MAX_REQUESTS=$SPECSTREAM_PREFILL_MAX_REQUESTS
CHUNK_TOKENS=$([[ "$RECORDED_KV_MANAGEMENT" == disabled || "$RECORDED_KV_MANAGEMENT" == gpu_only_no_offload ]] && echo disabled || echo 2048)
ACTIVE_TAIL_TOKENS=$([[ "$RECORDED_KV_MANAGEMENT" == disabled || "$RECORDED_KV_MANAGEMENT" == gpu_only_no_offload ]] && echo disabled || echo 512)
MIN_HISTORY_TOKENS=$([[ "$RECORDED_KV_MANAGEMENT" == disabled || "$RECORDED_KV_MANAGEMENT" == gpu_only_no_offload ]] && echo disabled || echo 8192)
GPU_RESERVE_MB=$([[ "$RECORDED_KV_MANAGEMENT" == disabled || "$RECORDED_KV_MANAGEMENT" == gpu_only_no_offload ]] && echo disabled || echo 1024)
GPU_HISTORY_CACHE_TOKENS=$([[ "$RECORDED_KV_MANAGEMENT" == disabled || "$RECORDED_KV_MANAGEMENT" == gpu_only_no_offload ]] && echo disabled || echo "$SPECSTREAM_GPU_HISTORY_CACHE_TOKENS")
GPU_HISTORY_MIN_FREE_TOKENS=$([[ "$RECORDED_KV_MANAGEMENT" == disabled || "$RECORDED_KV_MANAGEMENT" == gpu_only_no_offload ]] && echo disabled || echo "$SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS")
CPU_MEMORY_GB=$([[ "$RECORDED_KV_MANAGEMENT" == disabled || "$RECORDED_KV_MANAGEMENT" == gpu_only_no_offload ]] && echo disabled || echo 128)
EOF

if [[ "$SPECSTREAM_DRY_RUN" == 1 ]]; then
  echo "SPECSTREAM_DRY_RUN=PASS"
  echo "CASE_ROOT=$CASE_ROOT"
  return 0
fi

target_pid=""; draft_pid=""; monitor_pid=""; mps_started=0
cleanup(){
  [[ -n "$monitor_pid" ]] && kill "$monitor_pid" 2>/dev/null || true
  for pid in "$draft_pid" "$target_pid"; do
    [[ -n "$pid" ]] && kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "$draft_pid" "$target_pid"; do
    [[ -n "$pid" ]] && kill -KILL -- "-$pid" 2>/dev/null || true
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  if (( mps_started )); then
    env CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
      CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
      bash -c "echo quit | nvidia-cuda-mps-control" >/dev/null 2>&1 || true
  fi
}
# SGLang workers can require SIGKILL after the benchmark has already passed.
# Bash otherwise prints a misleading ``Killed ... launch_server`` job-status
# line to the experiment console.  Cleanup diagnostics are not result
# evidence (the role logs are), so keep the trap silent.
trap 'cleanup >/dev/null 2>&1' EXIT INT TERM

if (( USE_MPS )); then
  SPECSTREAM_SESSION_ID="${CASE_TAG}_$$"
  SPECSTREAM_MPS_PIPE="/tmp/specstream-mps-${USER}-${SPECSTREAM_SESSION_ID}"
  SPECSTREAM_MPS_LOG="/tmp/specstream-mps-log-${USER}-${SPECSTREAM_SESSION_ID}"
  mkdir -p "$SPECSTREAM_MPS_PIPE" "$SPECSTREAM_MPS_LOG"
  env CUDA_VISIBLE_DEVICES="$TARGET_UUIDS" \
    CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
    CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
    nvidia-cuda-mps-control -d
  mps_started=1
fi

STARTED_PID=""
start_role(){
  local role="$1" visible="$2"; shift 2
  local -a cmd=("$@")
  if (( USE_MPS )); then
    setsid env CUDA_VISIBLE_DEVICES="$visible" \
      CUDA_MPS_PIPE_DIRECTORY="$SPECSTREAM_MPS_PIPE" \
      CUDA_MPS_LOG_DIRECTORY="$SPECSTREAM_MPS_LOG" \
      "${cmd[@]}" > "$CASE_ROOT/$role.log" 2>&1 &
  else
    setsid env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
      -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u CUDA_MPS_CLIENT_PRIORITY \
      CUDA_VISIBLE_DEVICES="$visible" \
      "${cmd[@]}" > "$CASE_ROOT/$role.log" 2>&1 &
  fi
  STARTED_PID=$!
}

wait_ready(){
  local role="$1" port="$2" pid="$3"
  local log="$CASE_ROOT/$role.log"
  local deadline=$((SECONDS + 600))
  until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { tail -n 200 "$log"; return 1; }
    (( SECONDS < deadline )) || { echo "ERROR: $role readiness timeout"; tail -n 200 "$log"; return 1; }
    sleep 1
  done
}

case_started=$SECONDS
echo "[$CASE_TAG] Target GPU=$TARGET_VISIBLE"
start_role target "$TARGET_VISIBLE" "${TARGET_ARGS[@]}"; target_pid="$STARTED_PID"
wait_ready target "$TARGET_PORT" "$target_pid"
if (( START_DRAFT )); then
  echo "[$CASE_TAG] Draft GPU=$DRAFT_VISIBLE"
  start_role draft "$DRAFT_VISIBLE" "${DRAFT_ARGS[@]}"; draft_pid="$STARTED_PID"
  wait_ready draft "$DRAFT_PORT" "$draft_pid"
  sleep 5
fi

target_kv_tokens=$(grep -m1 -oE 'max_total_num_tokens=[0-9]+' "$CASE_ROOT/target.log" | cut -d= -f2 || true)
draft_kv_tokens=disabled
if (( START_DRAFT )); then
  draft_kv_tokens=$(grep -m1 -oE 'max_total_num_tokens=[0-9]+' "$CASE_ROOT/draft.log" | cut -d= -f2 || true)
fi
{
  echo "target_required_min_tokens=$SPECSTREAM_TARGET_MIN_KV_TOKENS"
  echo "draft_required_min_tokens=$([[ $START_DRAFT == 1 ]] && echo "$SPECSTREAM_DRAFT_MIN_KV_TOKENS" || echo disabled)"
  echo "target_requested_max_total_tokens=$SPECSTREAM_TARGET_MAX_TOTAL_TOKENS"
  echo "draft_requested_max_total_tokens=$([[ $START_DRAFT == 1 ]] && echo "$SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS" || echo disabled)"
  echo "target_max_total_num_tokens=${target_kv_tokens:-missing}"
  echo "draft_max_total_num_tokens=${draft_kv_tokens:-missing}"
} | tee "$CASE_ROOT/kv_capacity_gate.txt"
if [[ ! "$target_kv_tokens" =~ ^[0-9]+$ ]]; then
  echo "ERROR: unable to read Target KV capacity; formal case not started" >&2
  return 2
fi
if (( START_DRAFT )); then
  if [[ ! "$draft_kv_tokens" =~ ^[0-9]+$ ]]; then
    echo "ERROR: unable to read Target/Draft KV capacities; formal case not started" >&2
    return 2
  fi
  if (( draft_kv_tokens < SPECSTREAM_DRAFT_MIN_KV_TOKENS )); then
    echo "ERROR: Draft KV capacity below required minimum; formal case not started" >&2
    return 2
  fi
  if (( SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS > 0 && draft_kv_tokens != SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS )); then
    echo "ERROR: Draft could not allocate the exact requested KV token cap" >&2
    return 2
  fi
fi
if (( target_kv_tokens < SPECSTREAM_TARGET_MIN_KV_TOKENS )); then
  echo "ERROR: Target KV capacity below required minimum; formal case not started" >&2
  return 2
fi
if (( SPECSTREAM_TARGET_MAX_TOTAL_TOKENS > 0 && target_kv_tokens != SPECSTREAM_TARGET_MAX_TOTAL_TOKENS )); then
  echo "ERROR: Target could not allocate the exact requested KV token cap" >&2
  return 2
fi
{
  echo "TARGET_ACTUAL_KV_TOKENS=$target_kv_tokens"
  echo "DRAFT_ACTUAL_KV_TOKENS=$draft_kv_tokens"
} >> "$CASE_ROOT/config.env"

nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw \
  --format=csv -lms 200 > "$RESULT_ROOT/gpu_monitor/$CASE_TAG.csv" 2>&1 &
monitor_pid=$!
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory \
  --format=csv > "$CASE_ROOT/process_placement.csv"

remaining=$((CASE_TIMEOUT_S - (SECONDS - case_started)))
(( remaining > 60 )) || { echo "ERROR: server startup consumed the case budget" >&2; return 124; }
if [[ "$CLIENT_MODE" == gsm8k_native ]]; then
  if [[ -z "${GSM8K_FEWSHOT_PATH:-}" ]]; then
    echo "ERROR: required environment variable is unset: GSM8K_FEWSHOT_PATH" >&2
    return 2
  fi
  [[ -s "$DATASET_PATH" ]] || { echo "ERROR: GSM8K test JSONL missing: $DATASET_PATH" >&2; return 2; }
  [[ -s "$GSM8K_FEWSHOT_PATH" ]] || { echo "ERROR: GSM8K few-shot JSONL missing: $GSM8K_FEWSHOT_PATH" >&2; return 2; }
  gsm8k_questions="$NUM_PROMPTS"
  if [[ "$gsm8k_questions" == 0 ]]; then
    gsm8k_questions=$(grep -cve '^$' "$DATASET_PATH")
  fi
  (( gsm8k_questions > 0 )) || { echo "ERROR: empty GSM8K test set" >&2; return 2; }
  BENCH=(
    "$SPECSTREAM_PYTHON" benchmark/gsm8k/bench_sglang.py
    --host 127.0.0.1 --port "$TARGET_PORT" --backend srt
    --data-path "$DATASET_PATH"
    --few-shot-data-path "$GSM8K_FEWSHOT_PATH"
    --num-questions "$gsm8k_questions"
    --num-shots "${GSM8K_NUM_SHOTS:-5}"
    --parallel "$MAX_CONCURRENCY"
    --max-new-tokens "$OUTPUT_LEN"
    --temperature 0 --top-p 1
    --disable-thinking --tokenizer-path "$TARGET_MODEL"
    --method "$METHOD"
    --result-file "$RESULT_ROOT/summary/${CASE_TAG}.json"
    --raw-result-file "$OUT_JSONL"
  )
elif [[ "$CLIENT_MODE" == accuracy ]]; then
  if [[ -z "${ACCURACY_DATASET:-}" ]]; then
    echo "ERROR: required environment variable is unset: ACCURACY_DATASET" >&2
    return 2
  fi
  BENCH=(
    "$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/run_public_accuracy.py
    --manifest "$DATASET_PATH" --dataset "$ACCURACY_DATASET"
    --output "$OUT_JSONL" --max-new-tokens "$OUTPUT_LEN"
    --limit "$NUM_PROMPTS" --seed "$SEED"
    --max-concurrency "$MAX_CONCURRENCY"
  )
  if [[ "${ACCURACY_RETURN_TOKEN_IDS:-0}" == 1 ]]; then
    BENCH+=(--return-token-ids)
  fi
else
  BENCH=(
    "$SPECSTREAM_PYTHON" -m sglang.bench_serving
    --backend sglang --base-url "http://127.0.0.1:$TARGET_PORT"
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL"
    --dataset-name "$DATASET_NAME"
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE"
    --max-concurrency "$MAX_CONCURRENCY" --warmup-requests "$WARMUP_REQUESTS"
    --seed "$SEED" --flush-cache
    --extra-request-body '{"temperature":0,"top_p":1}'
    --output-details --tag "$CASE_TAG" --output-file "$OUT_JSONL"
  )
  case "$DATASET_NAME" in
    sharegpt)
      BENCH+=(
        --dataset-path "$DATASET_PATH"
        --sharegpt-output-len "$OUTPUT_LEN"
        --sharegpt-context-len "$SERVER_CONTEXT_LEN"
      )
      ;;
    random-ids)
      BENCH+=(
        --tokenize-prompt
        --random-input-len "$INPUT_LEN"
        --random-output-len "$OUTPUT_LEN"
        --random-range-ratio 1
      )
      ;;
    *) echo "ERROR: unsupported DATASET_NAME=$DATASET_NAME" >&2; return 2 ;;
  esac
fi

bench_started=$SECONDS
set +e
env -u CUDA_VISIBLE_DEVICES -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY \
  -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u CUDA_MPS_CLIENT_PRIORITY \
  PYTHONUNBUFFERED=1 TQDM_MININTERVAL=0.5 \
  timeout --signal=TERM --kill-after=60s "$remaining" \
  "${BENCH[@]}" 2>&1 | tee "$CASE_ROOT/benchmark.log"
bench_rc=${PIPESTATUS[0]}
set -e
bench_elapsed=$((SECONDS - bench_started))
total_elapsed=$((SECONDS - case_started))
printf 'bench_elapsed_s=%s\ntotal_elapsed_s=%s\nbench_rc=%s\n' \
  "$bench_elapsed" "$total_elapsed" "$bench_rc" | tee "$CASE_ROOT/timing.env"

(( bench_rc == 0 )) || {
  [[ "$bench_rc" == 124 ]] && echo "CASE_TIME_LIMIT_REACHED=1" | tee -a "$CASE_ROOT/timing.env"
  tail -n 200 "$CASE_ROOT/target.log" || true
  [[ -f "$CASE_ROOT/draft.log" ]] && tail -n 200 "$CASE_ROOT/draft.log" || true
  return "$bench_rc"
}

grep -E 'CUDA out of memory|Scheduler hit an exception|Traceback \(most recent call last\)' \
  "$CASE_ROOT"/*.log > "$CASE_ROOT/fatal_errors.txt" && {
    cat "$CASE_ROOT/fatal_errors.txt"; return 3;
  } || true

# The profiler writes one file per Target rank for TP>1.  Rank 0 owns the
# SPECTRE ZMQ/control path, so publish it at the historical canonical path used
# by analyzers and move every raw rank shard below the case directory.  This
# preserves all evidence without making profiles/*.csv double-count a cell.
if (( TARGET_TP_SIZE > 1 )); then
  rank0_profile="${PROFILE_CSV%.csv}.tp0.csv"
  rank0_grants="${PROFILE_CSV%.csv}.tp0.grants.csv"
  if [[ -s "$rank0_profile" ]]; then
    cp "$rank0_profile" "$PROFILE_CSV"
  fi
  if [[ -s "$rank0_grants" ]]; then
    cp "$rank0_grants" "${PROFILE_CSV%.csv}.grants.csv"
  fi
  mkdir -p "$CASE_ROOT/profile_shards"
  for shard in "${PROFILE_CSV%.csv}".tp*.csv; do
    [[ -e "$shard" ]] || continue
    mv "$shard" "$CASE_ROOT/profile_shards/"
  done
  printf 'canonical_profile_source=tp0\ntarget_tp_size=%s\n' "$TARGET_TP_SIZE" \
    > "$CASE_ROOT/profile_layout.env"
fi

if [[ "$METHOD" == SPECSTREAM_1GPU || "$METHOD" == C ]]; then
  test -s "$PROFILE_CSV"
  GRANT_EVENTS_CSV="${PROFILE_CSV%.csv}.grants.csv"
  FINAL_DRAFT_TPCS="$FINAL_DRAFT_TPCS" PROFILE_CSV="$PROFILE_CSV" \
    "$SPECSTREAM_PYTHON" - <<'PY' | tee "$CASE_ROOT/tpc_gate.txt"
import csv, os
rows = list(csv.DictReader(open(os.environ["PROFILE_CSV"], encoding="utf-8")))
used = []
for row in rows:
    if row.get("grant_state") != "SLACK_FILL":
        continue
    low = int(float(row.get("draft_tpc_low") or 0))
    high = int(float(row.get("draft_tpc_high") or 0))
    if high > low:
        used.append(high - low)
print("SLACK_FILL_ROWS=", len(used), "TPC_VALUES=", sorted(set(used)))
if used:
    assert set(used) == {int(os.environ["FINAL_DRAFT_TPCS"])}
PY
  if [[ -s "$GRANT_EVENTS_CSV" ]]; then
    GRANT_ANALYZER=(
      "$SPECSTREAM_PYTHON"
      scripts/specstream/paper_eval/qwen3/analyze_grant_events.py
      --events "$GRANT_EVENTS_CSV"
      --expected-tpcs "$FINAL_DRAFT_TPCS"
      --output "$CASE_ROOT/grant_event_gate.json"
    )
    if [[ "$SPECSTREAM_REQUIRE_SLACK_FILL" == 1 ]]; then
      GRANT_ANALYZER+=(--require-slack-fill)
    fi
    "${GRANT_ANALYZER[@]}"
  elif [[ "$SPECSTREAM_REQUIRE_SLACK_FILL" == 1 ]]; then
    echo "ERROR: formal PCIe-Slack case produced no structured grant events" >&2
    return 3
  fi
fi

[[ -s "$OUT_JSONL" ]] || {
  echo "ERROR: benchmark succeeded but produced no result file: $OUT_JSONL" >&2
  return 3
}
if [[ "$METHOD" =~ ^K[1-5]$ ]]; then
  [[ -s "$PROFILE_CSV" ]] || {
    echo "ERROR: SpecStream case produced no profile: $PROFILE_CSV" >&2
    return 3
  }
fi

# A completion marker represents a fully retired cell, not merely a client
# process that returned zero. Stop both role process groups and MPS first.
cleanup >/dev/null 2>&1
trap - EXIT INT TERM
for port in "$TARGET_PORT" "$DRAFT_PORT"; do
  if port_open "$port"; then
    echo "ERROR: server still owns port $port after case cleanup" >&2
    return 3
  fi
done
{
  echo "CASE_TAG=$CASE_TAG"
  echo "METHOD=$METHOD"
  echo "COMPLETED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$CASE_COMPLETE_MARKER.tmp"
mv "$CASE_COMPLETE_MARKER.tmp" "$CASE_COMPLETE_MARKER"
echo "PASS: $CASE_TAG"
)

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  bash "${BASH_SOURCE[0]}" "$@" || true
  return 0
else
  _specstream_main "$@"
fi
