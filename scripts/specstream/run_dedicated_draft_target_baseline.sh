#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Dedicated Draft / Target launcher
#
# Target: one GPU (or TP GPU set)
# Draft : one dedicated GPU
#
# IMPORTANT:
# Use `bash -c`, NOT `bash -lc`.
#
# `bash -lc` creates a login shell and may reset PATH, causing
# the activated Conda environment (e.g. spectre) to be lost.
# ============================================================

: "${SPECSTREAM_TARGET_VISIBLE_DEVICES:?Set Target GPU UUID(s), comma separated}"
: "${SPECSTREAM_DRAFT_VISIBLE_DEVICES:?Set the dedicated Draft GPU UUID}"
: "${SPECSTREAM_TARGET_CMD:?Set the complete Target launch command}"
: "${SPECSTREAM_DRAFT_CMD:?Set the complete Drafter launch command}"
: "${SPECSTREAM_BENCH_CMD:?Set the benchmark command}"

RESULT_ROOT="${SPECSTREAM_RESULT_ROOT:-results/specstream_dedicated_baseline}"

TARGET_READY_CMD="${SPECSTREAM_TARGET_READY_CMD:-curl -fsS http://127.0.0.1:30000/health}"

DRAFT_READY_CMD="${SPECSTREAM_DRAFT_READY_CMD:-}"

READY_TIMEOUT_S="${SPECSTREAM_READY_TIMEOUT_S:-240}"
DRAFT_WARMUP_S="${SPECSTREAM_DRAFT_WARMUP_S:-5}"

mkdir -p "${RESULT_ROOT}"

# ------------------------------------------------------------
# Record environment
# ------------------------------------------------------------

ENV_LOG="${RESULT_ROOT}/environment.log"

{
    echo "===== SpecStream launcher environment ====="
    echo "date=$(date)"
    echo "pwd=$(pwd)"
    echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
    echo "PATH=${PATH}"
    echo "python=$(command -v python || true)"

    if command -v python >/dev/null 2>&1; then
        python - <<'PY'
import sys

print("sys.executable =", sys.executable)
print("sys.version =", sys.version.replace("\n", " "))

try:
    import sglang
    print("sglang =", sglang.__file__)
except Exception as exc:
    print("sglang import failed =", repr(exc))
PY
    fi

    echo
    echo "TARGET_VISIBLE_DEVICES=${SPECSTREAM_TARGET_VISIBLE_DEVICES}"
    echo "DRAFT_VISIBLE_DEVICES=${SPECSTREAM_DRAFT_VISIBLE_DEVICES}"

    echo
    echo "TARGET_CMD:"
    echo "${SPECSTREAM_TARGET_CMD}"

    echo
    echo "DRAFT_CMD:"
    echo "${SPECSTREAM_DRAFT_CMD}"

    echo
    echo "BENCH_CMD:"
    echo "${SPECSTREAM_BENCH_CMD}"
} > "${ENV_LOG}" 2>&1


# ------------------------------------------------------------
# Fail early if wrong Python environment is active
# ------------------------------------------------------------

if ! python -c "import sglang" >/dev/null 2>&1; then
    echo "ERROR: current Python cannot import sglang." >&2
    echo "Python: $(command -v python || true)" >&2
    echo "Activate the correct spectre environment first." >&2
    echo "See: ${ENV_LOG}" >&2
    exit 10
fi


target_pid=""
draft_pid=""


cleanup() {
    set +e

    for pid in "${draft_pid}" "${target_pid}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done

    for pid in "${draft_pid}" "${target_pid}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
}


trap cleanup EXIT INT TERM


# ============================================================
# 1. Start Target
# ============================================================

echo "Starting Target..."
echo "Target Python: $(command -v python)"

CUDA_VISIBLE_DEVICES="${SPECSTREAM_TARGET_VISIBLE_DEVICES}" \
    bash -c "${SPECSTREAM_TARGET_CMD}" \
    > "${RESULT_ROOT}/target.log" 2>&1 &

target_pid=$!

echo "Target PID=${target_pid}"


# ============================================================
# 2. Wait for Target
# ============================================================

deadline=$((SECONDS + READY_TIMEOUT_S))

until bash -c "${TARGET_READY_CMD}" >/dev/null 2>&1; do

    if ! kill -0 "${target_pid}" 2>/dev/null; then
        printf \
            'Target exited before readiness; see %s\n' \
            "${RESULT_ROOT}/target.log" >&2

        echo "----- target.log tail -----" >&2
        tail -n 80 "${RESULT_ROOT}/target.log" >&2 || true

        exit 1
    fi

    if (( SECONDS >= deadline )); then
        printf \
            'Target readiness timed out after %ss; see %s\n' \
            "${READY_TIMEOUT_S}" \
            "${RESULT_ROOT}/target.log" >&2

        tail -n 80 "${RESULT_ROOT}/target.log" >&2 || true

        exit 1
    fi

    sleep 1
done

echo "Target is ready."


# ============================================================
# 3. Start Drafter
# ============================================================

echo "Starting Drafter..."

CUDA_VISIBLE_DEVICES="${SPECSTREAM_DRAFT_VISIBLE_DEVICES}" \
    bash -c "${SPECSTREAM_DRAFT_CMD}" \
    > "${RESULT_ROOT}/draft.log" 2>&1 &

draft_pid=$!

echo "Draft PID=${draft_pid}"


# ============================================================
# 4. Wait for Drafter
# ============================================================

if [[ -n "${DRAFT_READY_CMD}" ]]; then

    deadline=$((SECONDS + READY_TIMEOUT_S))

    until bash -c "${DRAFT_READY_CMD}" >/dev/null 2>&1; do

        if ! kill -0 "${draft_pid}" 2>/dev/null; then
            printf \
                'Draft exited before readiness; see %s\n' \
                "${RESULT_ROOT}/draft.log" >&2

            echo "----- draft.log tail -----" >&2
            tail -n 80 "${RESULT_ROOT}/draft.log" >&2 || true

            exit 1
        fi

        if (( SECONDS >= deadline )); then
            printf \
                'Draft readiness timed out after %ss; see %s\n' \
                "${READY_TIMEOUT_S}" \
                "${RESULT_ROOT}/draft.log" >&2

            tail -n 80 "${RESULT_ROOT}/draft.log" >&2 || true

            exit 1
        fi

        sleep 1
    done

else
    echo \
        "No Draft health endpoint configured; waiting ${DRAFT_WARMUP_S}s."

    sleep "${DRAFT_WARMUP_S}"

    if ! kill -0 "${draft_pid}" 2>/dev/null; then
        echo "Draft exited during warmup." >&2
        echo "----- draft.log tail -----" >&2
        tail -n 80 "${RESULT_ROOT}/draft.log" >&2 || true
        exit 1
    fi
fi

echo "Drafter is ready/running."


# ============================================================
# 5. Run benchmark
# ============================================================

echo "Starting benchmark..."

bash -c "${SPECSTREAM_BENCH_CMD}" \
    > "${RESULT_ROOT}/benchmark.log" 2>&1

echo "Benchmark finished."


# ============================================================
# 6. Finish
# ============================================================

printf \
    'Dedicated Draft/Target baseline completed: %s\n' \
    "${RESULT_ROOT}"
