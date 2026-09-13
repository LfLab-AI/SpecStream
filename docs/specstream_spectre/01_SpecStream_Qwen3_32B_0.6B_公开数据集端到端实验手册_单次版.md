# 01 · Qwen3-32B / Qwen3-0.6B 公开数据集端到端测试（最终双 TP2 版）

> 更新：2026-09-09；服务器 `/root/lifei/SpecStream`；2 × A800 80 GiB。  
> **正式 SpecStream：Target TP=2 + Draft TP=2，`auto` 并行准入策略，动态 q。**  
> 先按 2–5 节准备并 smoke；第 6 节测准确性，第 7 节测性能。两张 GPU 上一次只运行一个实验。

## 0. 本版测试口径

本手册已合并 TP2 Draft、并行准入退避、P0/P1/P2 和此前 LongBench 线程设备/TP 通信顺序修复的启动口径。01 的旧 K3 smoke 和“Draft 只放 rank 0”的 SpecStream 正式命令已替换；不要再用旧命令补跑本版矩阵。

`auto` 是最终的可并行策略：流水线与各 rank 就绪、窗口和干扰判定允许时才并行；否则保留多 token ordinary 验证。它不是强制每轮并行。`serial` 则禁用 Target/Draft 计算重叠；两者都可以保留 Target 内部 H2D/attention 的跨层预取。吞吐提升、成功租约执行和 GPU 时间上的实际重叠是不同证据，不能混写。

| 本手册固定开关 | 值 | 含义 |
|---|---|---|
| `SPECSTREAM_DRAFT_TP_SIZE` | `2` | 外置 Draft 在 Target 的两张 GPU 上分片 |
| `SPECSTREAM_OVERLAP_MODE` | `auto` | 最终准入策略，允许同卡 Target/Draft 重叠 |
| `SPECSTREAM_FIXED_Q` | `0` | 动态候选 2/4/6/8；保留安全回退 |
| GPU History / buffers | `8192 / 2` | 全局 History token 槽；双缓冲跨层预取 |
| split-KV / 后台泵 | `auto / 1` | 当前优化实现 |
| catchup quantum | `1` | 保持本次比较配置；SLACK_FILL 仍最多一 token |
| `SPECSTREAM_REQUIRE_SLACK_FILL` | `0` | 允许记录 auto 未激活的有效性能观察；机制证据另查 |

2026-09-09 更新：所有同卡外置 Draft（含 I1、K1–K5、A、offload 基线、C/SpecStream）默认随 Target 的 GPU 做 TP 分布；Target TP2 时 Draft 默认 TP2。I2 入口强制两者 TP 相同，避免继承旧 TP1 环境。`SPECSTREAM_OVERLAP_MODE` / `SPECSTREAM_FIXED_Q` 的可切换策略仍仅用于 C/SpecStream；I1/K1–K5/offload 保持 ordinary 和各自 q 定义。I3 的显式 Draft TP1 对照保留。性能表中的模型、内存容量、数据集、并发、输出长度均以实际 `config.env` 为准。

## 1. 方法与负载

| 运行器 ID | Target | Draft / 执行 | 物理 GPU 数 |
|---|---|---|---:|
| `AR` | TP2 | 无 | 2 |
| `SGLANG_SD` | TP2 | 原生进程内 STANDALONE，随 Target TP2 | 2 |
| `SGLANG_SD_KV_OFFLOAD` | TP2 | 外置 Draft TP2、同一对 GPU、固定 q=4 ordinary | 2 |
| `SPECSTREAM_1GPU` | TP2 | **外置 Draft TP2、同一对 GPU、auto、动态 q** | 2 |

`SPECSTREAM_1GPU` 是历史脚本 ID，表示 Draft 不增加专用卡，**不表示单卡运行**。论文名称写“SpecStream（Target TP2 + Draft TP2, auto）”。原生 SD 与 SpecStream 的 Draft 都做 TP2 分片，但前者进程内执行、后者独立进程组，调度和通信实现不同。本版 offload 基线也使用 Draft TP2；这张表仍是完整系统比较；分离 TP 分片与重叠的因果效果请用 02 第 7 节同一 SpecStream 实现的对照。

| 数据集 | 正式样本数 | 并发 | 性能输出上限 |
|---|---:|---:|---:|
| GSM8K | 当前冻结 manifest 全集 | 8 | 256 |
| LongBench-v2 8K–32K | 当前冻结 manifest 全部合格样本（此前 131） | 4 | 256 |
| MRCR 16K–32K | 当前冻结 manifest 全部合格样本 | 4 | 256 |

均为 greedy、non-thinking、seed=1、request-rate=inf、warmup=4。样本数运行时读取文件并冻结 SHA256；不把历史 131 写死。性能与准确性客户端不同，不把性能输出上限当成任务评分配置。

## 2. 一次性环境

长期测试建议先在 XShell 运行 `screen -S specstream_tp2`，进入后执行以下环境块。离开而保留任务：按 Ctrl+A 再按 D；重新查看：`screen -r specstream_tp2`。下面的命令直接输出实时数据集进度条，不要后台重定向到 `/dev/null`。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/spectre
cd /root/lifei/SpecStream

export REPO=$PWD
export SPECSTREAM_PYTHON=/root/miniconda3/envs/spectre/bin/python
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$(dirname "$SPECSTREAM_PYTHON"):$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=1; fi

export MODEL_TAG=qwen3_0p6b_32b
export TARGET_MODEL=/root/autodl-tmp/model/models/Qwen--Qwen3-32B/snapshots/master
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen3-0.6B

# Target 与 TP2 Draft 使用同一 UUID 顺序；COLOCATED_* 保留兼容入口。
export TARGET_GPUS=0,1
export TARGET_TP_SIZE=2
export TARGET_GPU=0
export COLOCATED_GPU=0
export COLOCATED_TP_RANK=0
export DRAFT_GPU=0

export TARGET_PORT=30000 DRAFT_PORT=30001 ZMQ_PORT=5557
export SERVER_CONTEXT_LEN=40960 FINAL_DRAFT_TPCS=34

export QWEN3_DATA_ROOT=$REPO/specstream_prepared/qwen3_offline
export GSM8K_TEST_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/test-00000-of-00001.parquet
export GSM8K_TRAIN_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/train-00000-of-00001.parquet
export GSM8K_TEST=$QWEN3_DATA_ROOT/gsm8k_main_test.jsonl
export GSM8K_FEWSHOT=$QWEN3_DATA_ROOT/gsm8k_main_train_fewshot5.jsonl
export GSM8K_QWEN3=$QWEN3_DATA_ROOT/gsm8k_qwen3_nothink_sharegpt.json
export LONGBENCH_QWEN3=$QWEN3_DATA_ROOT/longbench_v2_qwen3_8b_8k32k_sharegpt.json
export MRCR_QWEN3=$QWEN3_DATA_ROOT/mrcr_qwen3_16k32k_sharegpt.json

# 32B/0.6B 公共矩阵固定容量；所有四种方法不得在 cell 间改动。
export SPECSTREAM_TARGET_MEM_FRACTION=0.62
export SPECSTREAM_DRAFT_MEM_FRACTION=0.80
export SPECSTREAM_TARGET_MAX_TOTAL_TOKENS=131072
export SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=196608
export SPECSTREAM_TARGET_MIN_KV_TOKENS=131072
export SPECSTREAM_DRAFT_MIN_KV_TOKENS=196608
export SPECSTREAM_PREFILL_MAX_REQUESTS=1
export SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192
export SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0
export SPECSTREAM_REQUIRE_SLACK_FILL=0
export SPECSTREAM_SPLIT_KV=auto SPECSTREAM_BACKGROUND_GRANT_PUMP=1
export SPECSTREAM_NUM_BUFFERS=2 SPECSTREAM_GRANT_TOKEN_QUANTUM=1
export SPECSTREAM_DRAFT_TP_SIZE=2 SPECSTREAM_OVERLAP_MODE=auto SPECSTREAM_FIXED_Q=0
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=0.5
unset SPECSTREAM_DRY_RUN CUDA_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY REQUIRE_SEPARATE_DRAFT_GPU
unset LONGBENCH_LIMIT
```

不要全局设置 `CUDA_VISIBLE_DEVICES` 或 `CUDA_MPS_*`；运行器按进程设置并在退出时清理。

## 3. 数据集是否需要重处理

### 3.1 本次结论

此前预检支持复用已有数据；本次仍以随后指纹 Gate 为准。Qwen3-32B、Qwen3-8B 和 Qwen3-0.6B 使用相同 token-to-id 映射、特殊 token 与 non-thinking chat template；当前服务器已经实测 32B/0.6B 的 `VOCAB_EQUAL=True`、special token 一致、template 一致。因此原来由 Qwen3-8B Target tokenizer 生成的冻结 manifest 可以逐 token 复用。文件名中的 `qwen3_8b` 只是历史名称，不代表其中保存了 8B 模型状态。

正式运行前仍须执行下面的指纹 Gate；它失败时才重生成：

```bash
(
set -euo pipefail
TARGET_MODEL="$TARGET_MODEL" DRAFT_MODEL="$DRAFT_MODEL" QWEN3_DATA_ROOT="$QWEN3_DATA_ROOT" \
"$SPECSTREAM_PYTHON" - <<'PY'
import hashlib, json, os
from transformers import AutoTokenizer

def load(path):
    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )

target = load(os.environ["TARGET_MODEL"])
draft = load(os.environ["DRAFT_MODEL"])
assert target.get_vocab() == draft.get_vocab()
for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
    assert getattr(target, key) == getattr(draft, key), key
probe_t = target.apply_chat_template(
    [{"role": "user", "content": "fingerprint"}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False,
)
probe_d = draft.apply_chat_template(
    [{"role": "user", "content": "fingerprint"}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False,
)
assert probe_t == probe_d
vocab = json.dumps(sorted(target.get_vocab().items()), ensure_ascii=False).encode()
actual = hashlib.sha256(vocab).hexdigest()
manifest = json.load(open(
    os.path.join(os.environ["QWEN3_DATA_ROOT"], "tokenizer_manifest.json"),
    encoding="utf-8",
))
assert actual == manifest["vocab_sha256"], (actual, manifest["vocab_sha256"])
assert manifest["context_length"] == 40960
assert manifest["reserved_output_length"] == 1024
assert manifest["enable_thinking"] is False
print("QWEN3_32B_DATA_REUSE_GATE=PASS", actual)
PY

(cd "$QWEN3_DATA_ROOT" && sha256sum -c dataset_sha256.txt)
)
```

### 3.2 仅当指纹 Gate 失败时重生成

不能覆盖冻结目录。先写入新目录，完成检查后再令 `QWEN3_DATA_ROOT` 指向它：

```bash
export GSM8K_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/test-00000-of-00001.parquet
export LONGBENCH_SOURCE=/root/autodl-tmp/dataset/LongBench-v2
export MRCR_SOURCE=/root/autodl-tmp/dataset/mrcr
export NEW_DATA_ROOT=$REPO/specstream_prepared/qwen3_32b_0p6b_$(date +%Y%m%d_%H%M%S)
mkdir -p "$NEW_DATA_ROOT"

(
set -euo pipefail
"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/prepare_qwen3_offline.py \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --gsm8k-source "$GSM8K_SOURCE" \
  --longbench-source "$LONGBENCH_SOURCE" \
  --mrcr-source "$MRCR_SOURCE" \
  --output-root "$NEW_DATA_ROOT" \
  --context-length 40960 --reserved-output-length 1024 --minimum-rows 64 \
  2>&1 | tee "$NEW_DATA_ROOT/prepare_console.log"

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/prepare_gsm8k_native_eval.py \
  --test-source "$GSM8K_TEST_SOURCE" --train-source "$GSM8K_TRAIN_SOURCE" \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --output-root "$NEW_DATA_ROOT" --num-shots 5 --expected-test-rows 1319

grep -q 'QWEN3_OFFLINE_DATA_GATE=PASS' "$NEW_DATA_ROOT/prepare_console.log"
(cd "$NEW_DATA_ROOT" && sha256sum -c dataset_sha256.txt)
(cd "$NEW_DATA_ROOT" && sha256sum -c gsm8k_native_eval_sha256.txt)
)
# 仅在准备和 SHA256 Gate 全部通过后切换数据目录。
export QWEN3_DATA_ROOT=$NEW_DATA_ROOT
export GSM8K_TEST=$QWEN3_DATA_ROOT/gsm8k_main_test.jsonl
export GSM8K_FEWSHOT=$QWEN3_DATA_ROOT/gsm8k_main_train_fewshot5.jsonl
export GSM8K_QWEN3=$QWEN3_DATA_ROOT/gsm8k_qwen3_nothink_sharegpt.json
export LONGBENCH_QWEN3=$QWEN3_DATA_ROOT/longbench_v2_qwen3_8b_8k32k_sharegpt.json
export MRCR_QWEN3=$QWEN3_DATA_ROOT/mrcr_qwen3_16k32k_sharegpt.json
```

脚本当前仍输出 `longbench_v2_qwen3_8b_8k32k_*.json*` 这一兼容文件名；以 `tokenizer_manifest.json` 的实际 SHA256 为准，不要手工改名后混用旧 checksum。

## 4. 当前机器 preflight 与命令核对

同一台机器、同一次会话已通过当前 preflight 可复用该次环境；换卡、重启或更换驱动后重做。不能仅手工设置 `SPECSTREAM_SMCTRL_VALIDATED=1`，也不要 source 其他 GPU UUID 的历史文件。以下 preflight 会做硬件预检；后面的 dry-run 不启动模型。

```bash
export PREFLIGHT_ROOT=$REPO/results/${MODEL_TAG}_preflight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PREFLIGHT_ROOT"
(
  set -euo pipefail
  bash scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
    2>&1 | tee "$PREFLIGHT_ROOT/console.log"
  grep -q 'QWEN3_PUBLIC_PREFLIGHT=PASS' "$PREFLIGHT_ROOT/console.log"
)
# 上面非零即停止；只在 PASS 后 source。
source "$PREFLIGHT_ROOT/runtime_env.sh"
export SPECSTREAM_DRAFT_TP_SIZE=2 SPECSTREAM_OVERLAP_MODE=auto SPECSTREAM_FIXED_Q=0

export DRY_ROOT=$REPO/results/${MODEL_TAG}_public_tp2_dry_$(date +%Y%m%d_%H%M%S)
"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/run_tp2_final_matrix.py \
  --kind public --datasets longbench_v2 --root "$DRY_ROOT" --dry-run
```

必须出现 `TP2_MATRIX_DRY_RUN=PASS cells=4`。查看 `$DRY_ROOT/logs/SPECSTREAM_1GPU_longbench_v2_c4/config.env`：`TARGET_TP_SIZE=2`、`DRAFT_TP_SIZE=2`、`SPECSTREAM_OVERLAP_MODE=auto`、`SPECSTREAM_FIXED_Q=0`，且 `TARGET_VISIBLE` 与 `DRAFT_VISIBLE` 是相同的两个 UUID；两个启动命令都应有 `--tp-size 2`。原生 SD 没有独立 Draft 启动命令，不能用它的 `DRAFT_VISIBLE` 占位字段推断原生 Draft 的卡数。

## 5. 一次最小 smoke

smoke 使用最终双 TP2 auto、动态 q 和正式 LongBench 数据；只缩小为 8 请求、输出 64，并发仍为 4。成功不能代替完整 256-token 正式测试。

```bash
export SMOKE_ROOT=$REPO/results/${MODEL_TAG}_final_tp2_smoke_$(date +%Y%m%d_%H%M%S)
(
  set -euo pipefail
  SPECSTREAM_DRAFT_TP_SIZE=2 SPECSTREAM_OVERLAP_MODE=auto SPECSTREAM_FIXED_Q=0 \
  METHOD=SPECSTREAM_1GPU CLIENT_MODE=performance DATASET_NAME=sharegpt \
  DATASET_TAG=final_tp2_smoke DATASET_PATH="$LONGBENCH_QWEN3" \
  NUM_PROMPTS=8 OUTPUT_LEN=64 MAX_CONCURRENCY=4 WARMUP_REQUESTS=4 \
  REQUEST_RATE=inf SEED=1 CASE_TIMEOUT_S=1800 RESULT_ROOT="$SMOKE_ROOT" \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh
  test -s "$SMOKE_ROOT/logs/SPECSTREAM_1GPU_final_tp2_smoke_c4/case_complete.marker"
  SMOKE_ROOT="$SMOKE_ROOT" "$SPECSTREAM_PYTHON" - <<'PY'
import json, os
from pathlib import Path
r=Path(os.environ['SMOKE_ROOT']); c='SPECSTREAM_1GPU_final_tp2_smoke_c4'
b=json.loads((r/'bench'/f'{c}.jsonl').read_text().splitlines()[-1])
g=json.loads((r/'logs'/c/'grant_event_gate.json').read_text())
assert b['completed']==8 and b['total_output_tokens']>0 and not any(b.get('errors',[]))
assert g['status']=='PASS'
assert not (r/'logs'/c/'fatal_errors.txt').read_text().strip()
for name in ['target.log','draft.log']:
    text=(r/'logs'/c/name).read_text(errors='replace')
    assert 'DraftFallback' not in text and 'RecvTimeout' not in text
print('FINAL_TP2_SMOKE=PASS', 'slack_fill_success=',g['slack_fill_success'])
PY
)
```

失败即停止，保留 `target.log`、`draft.log`、`kv_capacity_gate.txt`。`slack_fill_success=0` 表示本次未成功执行气泡租约；即使 smoke 通过，也不能声称已经验证实际重叠。

## 6. 准确性正式实验

每个 pair runner 会依次运行原生 SD 和 SpecStream、评分并检查完整样本数与请求错误。使用新的结果根：

```bash
# 准确性也固定最终双 TP2 auto；避免继承 02 串行或固定 q。
export SPECSTREAM_DRAFT_TP_SIZE=2 SPECSTREAM_OVERLAP_MODE=auto SPECSTREAM_FIXED_Q=0
export ACCURACY_RETURN_TOKEN_IDS=1
unset LONGBENCH_LIMIT
export ACC_ROOT=$REPO/results/${MODEL_TAG}_public_accuracy_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ACC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
cp "$QWEN3_DATA_ROOT/dataset_sha256.txt" "$ACC_ROOT/env/"
git rev-parse HEAD > "$ACC_ROOT/env/git_commit.txt"
git status --short > "$ACC_ROOT/env/git_status.txt"
```

GSM8K：

```bash
(
set -euo pipefail
GSM8K_TEST="$GSM8K_TEST" GSM8K_FEWSHOT="$GSM8K_FEWSHOT" ACC_ROOT="$ACC_ROOT" \
bash scripts/specstream/paper_eval/qwen3/run_gsm8k_accuracy_pair.sh \
  2>&1 | tee "$ACC_ROOT/gsm8k_console.log"
)
```

LongBench-v2：

```bash
(
set -euo pipefail
DATASET_TAG=accuracy_longbench_v2 DATASET_PATH="$LONGBENCH_QWEN3" \
OUTPUT_LEN=256 MAX_CONCURRENCY=4 CASE_TIMEOUT_S=43200 ACC_ROOT="$ACC_ROOT" \
bash scripts/specstream/paper_eval/qwen3/run_longbench_accuracy_pair.sh \
  2>&1 | tee "$ACC_ROOT/longbench_console.log"
)
```

MRCR：

```bash
(
set -euo pipefail
ACCURACY_DATASET=mrcr DATASET_TAG=accuracy_mrcr_all \
DATASET_PATH="$MRCR_QWEN3" OUTPUT_LEN=1024 MAX_CONCURRENCY=4 \
CASE_TIMEOUT_S=43200 ACC_ROOT="$ACC_ROOT" \
bash scripts/specstream/paper_eval/qwen3/run_manifest_accuracy_pair.sh \
  2>&1 | tee "$ACC_ROOT/mrcr_console.log"
)
```

必须分别出现 `GSM8K_ACCURACY_PAIR_GATE=PASS`、`LONGBENCH_ACCURACY_PAIR_GATE=PASS` 和 `MANIFEST_ACCURACY_PAIR_GATE=PASS`；两方法 `failures=0`。只有 HTTP `error=False` 不能代替任务准确率。

这些 pair Gate 只检查样本完整、请求/评分无错误，**不保证两方法分数相等或逐 token 相等**。报告评分差异与不一致样本；出现差异时保留 token IDs/文本和请求配置，定位首个分歧，不能把所有差异直接归为浮点误差。GSM8K 现有准确性入口固定 5-shot、输出 512、并发 32；这是独立评分负载，不是第 7 节并发 8 的性能负载。

## 7. 正式性能：完整三数据集，或仅重测 LongBench

本版使用新增的轻量编排入口 `run_tp2_final_matrix.py`，仍调用服务器现有 `run_public_once.sh`，不修改模型热路径。它固定第 0/2 节参数，逐 cell 串行启动/清理服务，直接保留实时 tqdm；已经做过第 5 节 smoke 后，不再给每个 cell 重复 smoke。旧 `run_public_matrix_once.sh` 的当前数据集列表与旧文档“12+12”计数不同，**本版不使用旧入口**。

完整 GSM8K + LongBench + MRCR，共 **12 个正式 cell**：

```bash
export RESULT_ROOT=$REPO/results/${MODEL_TAG}_public_final_tp2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_ROOT"
(
  set -euo pipefail
  "$SPECSTREAM_PYTHON" -u scripts/specstream/paper_eval/qwen3/run_tp2_final_matrix.py \
    --kind public --root "$RESULT_ROOT" \
    2>&1 | tee "$RESULT_ROOT/console.log"
)
```

若目前只想重测 LongBench，用下面的**替代命令**，共 **4 个正式 cell**；不要再执行上面的完整矩阵：

```bash
export RESULT_ROOT=$REPO/results/${MODEL_TAG}_longbench_final_tp2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULT_ROOT"
(
  set -euo pipefail
  "$SPECSTREAM_PYTHON" -u scripts/specstream/paper_eval/qwen3/run_tp2_final_matrix.py \
    --kind public --datasets longbench_v2 --root "$RESULT_ROOT" \
    2>&1 | tee "$RESULT_ROOT/console.log"
)
```

另可用 `--datasets longbench_v2 mrcr16_32` 得到 8 个 cell。每 cell 总超时 43200 秒；不会自动续跑或覆盖失败目录，失败后修复并换新根目录重跑所需矩阵。所有方法使用相同 Target 131072 / 外置 Draft 196608 token cap；若某方法容量不足，停止比较并统一重新标定，不能只放宽该方法。原生 SD 的 Draft 池分配遵循其自身实现，需同时保留容量日志，不能声称所有方法 GPU KV 字节数相同。

完成后：

```bash
(
  set -euo pipefail
  test -s "$RESULT_ROOT/matrix_complete.marker"
  grep -q 'TP2_MATRIX_COMPLETE=PASS' "$RESULT_ROOT/console.log"
  cat "$RESULT_ROOT/matrix_complete.marker"
  cat "$RESULT_ROOT/summary/results.json"
  "$SPECSTREAM_PYTHON" scripts/specstream/summarize_benchmarks.py \
    "$RESULT_ROOT"/bench/*.jsonl | tee "$RESULT_ROOT/summary/benchmark.tsv"
)
```

新入口会检查计划中每个 cell 的实际配置、完成 marker、非空结果、全部请求成功、fatal/fallback 和 SpecStream Grant Gate，并核对正式 marker/JSONL 的**精确集合**后才写 `matrix_complete.marker`。这里 12/8/4 都只数正式 cell，不包含另一个目录下的 smoke。`env/matrix_plan.json` 保存样本数与数据 SHA256，`env/frozen_config.json`、git 信息、源码 diff 和每 cell 启动命令用于复现；代码有未提交修改时不能只记录 commit。

## 8. 结果解释与下一步

- 主指标：`output_throughput`（服务端生成 token，非 retokenized token）、request throughput、TTFT、TPOT、P99，以及准确性分数。比较相同样本、并发和输出设置；不要跨不同根的旧配置计算加速比。
- `accept_length` 是每次验证平均接受长度，不能直接当作接受率。动态 q 的接受长度必须结合 `q_histogram` 解读；fixed-q 机制比较在 02 第 7 节。
- `summary/results.json` 中 `overlap_evidence=NOT_ACTIVATED` 表示 Grant Gate 虽通过但没有成功 SLACK_FILL；这是 auto 策略的性能观察，不能当作并行机制的成功证据。`ACTIVATED` 只证明至少一次成功租约，覆盖率、隐藏的传输时间与 Target 干扰仍需 profile 支撑。
- TP0 是规范 profile；TP0/TP1 原始分片位于 `logs/<case>/profile_shards/`，不要把规范 profile 与原始分片重复加总。TP2 Grant ACK 的联合完成不能替代两卡逐时间轴的覆盖率分析。
- 单次结果不报告标准差或显著性。需要论文误差条时，用新目录重复相同矩阵并平衡运行顺序。创新点三具体分组、判断负收益的方法见 [02 手册](02_SpecStream_Qwen3_32B_0.6B_三创新点受控消融实验手册_单次版.md)。

本次仅更新测试手册和编排入口，不在交付过程中启动 GPU 测试；操作者按上述顺序执行。历史修复记录见 [并行准入修复](05_Parallel_admission_fix.md)，其中的旧测量不替代本版正式矩阵。

本次分布修改之后，新的 offload TP2 结果必须写入新目录。此前已经完成的 offload TP1 结果保留其原始标签，不能与新矩阵拼成同配置结果。
