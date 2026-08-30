#!/usr/bin/env bash
set -euo pipefail

: "${VARIANT:?Set VARIANT, e.g. P4}"
: "${RESULT_ROOT:?Set RESULT_ROOT}"
: "${BASE_URL:?Set BASE_URL}"
: "${TARGET_MODEL:?Set TARGET_MODEL}"
: "${SHAREGPT_JSON:?Set SHAREGPT_JSON}"
: "${DRAFT_CMD_Q4:?Define DRAFT_CMD_Q4 first}"
: "${TARGET_GPU:?Set TARGET_GPU}"
: "${DRAFT_GPU:?Set DRAFT_GPU}"
: "${TARGET_PORT:?Set TARGET_PORT}"
: "${DRAFT_PORT:?Set DRAFT_PORT}"

SERVER_CONTEXT_LEN="${SERVER_CONTEXT_LEN:-32768}"

# ------------------------------------------------------------
# 两种 workload 输入方式
# ------------------------------------------------------------
WORKLOADS="${WORKLOADS:-}"

INPUT_LENS="${INPUT_LENS:-4096 8192 16384 24576 30000}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8 16 32}"

# ------------------------------------------------------------
# benchmark 参数
# ------------------------------------------------------------
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
RANGE_RATIO="${RANGE_RATIO:-1}"
SEED="${SEED:-1}"

# CASE_PREFIX 用来区分 screening/memory/capacity/confirm 等实验。
CASE_PREFIX="${CASE_PREFIX:-$VARIANT}"

# 正式 benchmark 默认遇错即停。
# Capacity/OOM mapping 时可显式 CONTINUE_ON_ERROR=1。
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

case "$VARIANT" in
  S0) TARGET_VAR=S0_TARGET_CMD ;;
  S1) TARGET_VAR=S1_TARGET_CMD ;;
  S2) TARGET_VAR=S2_TARGET_CMD ;;
  S3) TARGET_VAR=S3_TARGET_CMD ;;
  S4) TARGET_VAR=S4_TARGET_CMD ;;
  S5) TARGET_VAR=S5_TARGET_CMD ;;

  P0) TARGET_VAR=P0_TARGET_CMD ;;
  P1) TARGET_VAR=P1_TARGET_CMD ;;
  P2) TARGET_VAR=P2_TARGET_CMD ;;
  P3) TARGET_VAR=P3_TARGET_CMD ;;
  P4) TARGET_VAR=P4_TARGET_CMD ;;
  P5) TARGET_VAR=P5_TARGET_CMD ;;
  P6) TARGET_VAR=P6_TARGET_CMD ;;

  P4C) TARGET_VAR=P4C_TARGET_CMD ;;

  *)
    echo "ERROR: unsupported VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

TARGET_TEMPLATE="${!TARGET_VAR:-}"

if [[ -z "$TARGET_TEMPLATE" ]]; then
  echo "ERROR: $TARGET_VAR is empty." >&2
  echo "Define the Variant commands from Sections 9-11 first." >&2
  exit 3
fi

mkdir -p \
  "$RESULT_ROOT/bench" \
  "$RESULT_ROOT/profiles" \
  "$RESULT_ROOT/logs" \
  "$RESULT_ROOT/source_data"

# ------------------------------------------------------------
# 构造 workload 列表
# ------------------------------------------------------------
declare -a PAIRS=()

if [[ -n "$WORKLOADS" ]]; then

  for pair in $WORKLOADS; do
    if [[ "$pair" != *:* ]]; then
      echo "ERROR: invalid WORKLOADS item: $pair" >&2
      echo "Expected format: INPUT_LEN:CONCURRENCY" >&2
      exit 4
    fi
    PAIRS+=("$pair")
  done

else

  for input_len in $INPUT_LENS; do
    for c in $CONCURRENCIES; do
      PAIRS+=("${input_len}:${c}")
    done
  done

fi

TOTAL="${#PAIRS[@]}"
INDEX=0

FAIL_FILE="$RESULT_ROOT/source_data/grid_failures.tsv"

for pair in "${PAIRS[@]}"; do

  INDEX=$((INDEX + 1))

  INPUT_LEN="${pair%%:*}"
  C="${pair##*:}"

  if (( INPUT_LEN + OUTPUT_LEN > SERVER_CONTEXT_LEN )); then
    echo
    echo "ERROR: context window overflow"
    echo "INPUT_LEN=$INPUT_LEN OUTPUT_LEN=$OUTPUT_LEN"
    echo "SERVER_CONTEXT_LEN=$SERVER_CONTEXT_LEN"
    exit 5
  fi

  if (( C <= 1 )); then
    WARMUP=1
  elif (( C <= 8 )); then
    WARMUP=4
  else
    WARMUP=8
  fi

  CASE_TAG="${CASE_PREFIX}_${INPUT_LEN}_c${C}"

  # ----------------------------------------------------------
  # 每个 case 使用独立 profile，避免不同 workload 的 CSV 混在一起。
  # 原 Variant 命令里若已有 profile-path，先删掉再追加 case-specific path。
  # ----------------------------------------------------------
  TARGET_CMD="$(printf '%s\n' "$TARGET_TEMPLATE" \
    | sed -E "s@--specstream-profile-path '[^']*'@@g")"

  if [[ "$TARGET_CMD" == *"--specstream-"* ]]; then
    TARGET_CMD="$TARGET_CMD \
      --specstream-profile-path '$RESULT_ROOT/profiles/${CASE_TAG}.csv'"
  fi

  export SPECSTREAM_TARGET_VISIBLE_DEVICES="$TARGET_GPU"
  export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$DRAFT_GPU"

  export SPECSTREAM_TARGET_CMD="$TARGET_CMD"
  export SPECSTREAM_DRAFT_CMD="$DRAFT_CMD_Q4"

  export SPECSTREAM_TARGET_READY_CMD="curl -fsS http://127.0.0.1:${TARGET_PORT}/health"
  export SPECSTREAM_DRAFT_READY_CMD="curl -fsS http://127.0.0.1:${DRAFT_PORT}/health"
  export SPECSTREAM_READY_TIMEOUT_S=300

  export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/${CASE_TAG}"

  export SPECSTREAM_BENCH_CMD="BASE_URL='$BASE_URL' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG='$CASE_TAG' \
DATASET_NAME=random \
DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=$INPUT_LEN \
OUTPUT_LEN=$OUTPUT_LEN \
NUM_PROMPTS=$NUM_PROMPTS \
REQUEST_RATE=$REQUEST_RATE \
MAX_CONCURRENCY=$C \
RANGE_RATIO=$RANGE_RATIO \
WARMUP_REQUESTS=$WARMUP \
SEED=$SEED \
CONTEXT_LEN=$SERVER_CONTEXT_LEN \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

  echo
  echo "################################################################"
  echo "GRID [$INDEX/$TOTAL]"
  echo "VARIANT      = $VARIANT"
  echo "CASE_TAG     = $CASE_TAG"
  echo "INPUT_LEN    = $INPUT_LEN"
  echo "OUTPUT_LEN   = $OUTPUT_LEN"
  echo "CONCURRENCY  = $C"
  echo "NUM_PROMPTS  = $NUM_PROMPTS"
  echo "REQUEST_RATE = $REQUEST_RATE"
  echo "WARMUP       = $WARMUP"
  echo "################################################################"
  echo

  if bash scripts/specstream/run_dedicated_draft_target_baseline.sh; then

    echo
    echo "GRID [$INDEX/$TOTAL] PASS: $CASE_TAG"

  else

    rc=$?

    printf '%s\t%s\t%s\t%s\n' \
      "$CASE_TAG" "$INPUT_LEN" "$C" "$rc" \
      >> "$FAIL_FILE"

    echo
    echo "GRID [$INDEX/$TOTAL] FAIL: $CASE_TAG rc=$rc"
    echo "Failure recorded in: $FAIL_FILE"

    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$rc"
    fi

  fi

done

echo
echo "################################################################"
echo "GRID FINISHED"
echo "VARIANT = $VARIANT"
echo "TOTAL   = $TOTAL"
echo "################################################################"
