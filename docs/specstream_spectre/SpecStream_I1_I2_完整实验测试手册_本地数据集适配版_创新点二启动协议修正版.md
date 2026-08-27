# SpecStream 创新点一 + 创新点二完整实验测试手册

> **2026-08-26 创新点二启动协议二次修订**：根据服务器真实运行日志，确认当前 I2 calibration/online-smctrl 路径存在 `first-grant-before-first-forward` 启动协议缺口。`--spectre-draft-priority` 为强制条件；仅把 `--specstream-smctrl-calibration-tpcs` 传给 Drafter 不会自动建立 active TPC mask。本版新增 Gate 3：在修复并通过首个 Grant/TPC mask smoke 之前，只允许继续 D0/D1/D2，禁止把 TPC calibration、D3、D4、D5 结果用于正式实验。

> **本地数据集适配修订版**：已按服务器实际目录
> `~/autodl-tmp/dataset/gsm8k/{main,socratic}` 与
> `~/autodl-tmp/dataset/LongBench-v2/data.json` 修正数据准备流程。
> 原始数据集保持不变，实验统一读取 `$PREPARED_ROOT` 下生成的兼容 JSONL。

> **2026-08-26 创新点二启动与公平性修订**：已统一 D0–D4 为固定 q=4，为同卡 Target/Drafter 显式限制 `--max-total-tokens`，增加 Target/Draft 双 `/health` gate，修正 MPS/Conda 启动要求，并将 D4/D5 resource profile 改为按真实 `(batch,q,context bucket)` 分布校准；D5 启动默认 q 也改为 4，正式 dynamic-q 结果要求主要运行 shape 的实测 profile 覆盖。



> **适用代码版本**：`LfLab-AI/SpecStream`，`main` 分支；本文档按 2026-08-25 的主分支实现编写。正式实验开始前请把 `git rev-parse HEAD` 写入结果目录，避免后续代码更新导致结果不可追溯。
>
> **实验范围**：仅覆盖当前论文的创新点一（verification-native bounded KV streaming + dynamic q + Chunk-Cohort）与创新点二（measurement-backed Target-priority single-GPU Draft–Verify co-execution）。不包含后续 TP/PP 创新点三。
>
> **正确性验证按当前需求精简**：只做两类证据：
> 1. 在至少两套数据集上比较最终模型任务准确率；
> 2. 使用 SpecStream 自带 `--specstream-shadow-attention` 生成 fused streaming 与 Torch reference attention 的一致性热力图。
>
> 不把 token-by-token parity、first-divergence 等作为论文主实验；这些诊断仍保留在代码中，仅在准确率或 attention 一致性异常时使用。

---

## 0. 最终需要得到哪些结果

建议最终论文实验只围绕以下结果组织。

### 0.1 正确性结果

| 编号 | 结果 | 主比较 |
|---|---|---|
| C1 | GSM8K 最终准确率 | 原生 SGLang speculative / SPECTRE parallel / 完整 SpecStream-I1 |
| C2 | LongBench v2 最终准确率 | 原生 SGLang speculative / SPECTRE parallel / 完整 SpecStream-I1 |
| C3 | Attention 一致性热力图 | SpecStream fused streaming vs Torch FP32 online-softmax reference |

### 0.2 创新点一性能结果

| 编号 | 系统 | 目的 |
|---|---|---|
| K0 | GPU-resident SPECTRE parallel | 原始并行推测强基线 |
| K1 | CPU History + Full-Restore-per-round | 简单 CPU offload 对照 |
| K2 | bounded reference streaming | 参考实现；主要用于机制与正确性，不作为最终最快实现 |
| K3 | fused/grouped bounded streaming，无跨层预取 | 验证 fused/grouped streaming 本身 |
| K4 | K3 + 双缓冲 + layer prefetch | 验证异步流水收益 |
| K5 | K4 + dynamic q / safety-aware mode control | 验证动态 horizon 控制 |
| K6 | K5 + Chunk-Cohort | 完整创新点一 |

> **重要命名规则**：当前 K1 只称为 **Full-Restore-per-round**。不要在论文中预先称为“异步重叠优化的全量恢复”，除非 Nsight Systems 明确证明完整 History 恢复被独立计算有效隐藏。

### 0.3 创新点二性能结果

| 编号 | GPU 数 | 系统 | 目的 |
|---|---:|---|---|
| D0 | 1 | 原生 SGLang STANDALONE speculative | 原生单卡推测解码参考 |
| D1 | 2 | SPECTRE parallel，Target/Draft 分卡 | 绝对性能强基线 |
| D2 | 1 | SPECTRE parallel，同卡、无 TPC 控制 | 证明“直接塞到一张卡”存在干扰 |
| D3 | 1 | same-GPU + 最佳固定 TPC | 静态资源隔离基线 |
| D4 | 1 | GPU-resident KV + online Target-priority TPC grant | 隔离创新点二；Target 使用 `--specstream-profile-only` |
| D5 | 1 | bounded KV streaming + online TPC grant + dynamic q + Cohort | 创新点一 + 创新点二完整 SpecStream |

创新点二最终实现不是“动态修改 MPS 百分比”。MPS 只用于让两个 CUDA 进程稳定共享同一张 GPU；真正的运行时控制是：**实测 resource profile + libsmctrl TPC mask + one-token Target grant/ACK**。

---

# 1. 一次性环境配置

## 1.1 进入代码仓库

```bash
cd ~/lifei/SpecStream
conda activate spectre

export REPO=$PWD
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
```

记录当前代码版本：

```bash
mkdir -p results/paper/logs

git rev-parse HEAD | tee results/paper/logs/git_commit.txt
git status --short | tee results/paper/logs/git_status.txt
python -c "import sglang; print(sglang.__file__)" | tee results/paper/logs/sglang_path.txt
```

正式论文结果必须固定一个 commit。代码发生变化后，不要把新旧 commit 的结果混在同一个图中。

---

## 1.2 修改下面这一组路径

只需要在第一次实验时修改。

```bash
# ======================== 模型 ========================
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct
export TARGET_MODEL=/root/autodl-tmp/model/Qwen2.5-7B-Instruct
export BASE_URL=http://127.0.0.1:30000
##export SHAREGPT_V3_ROOT=/common_data/dataset/ShareGPT_V3
export PREPARED_ROOT=/root/lifei/SpecStream/specstream_prepared ##/home/lifei/lifei/specdecode/baseline/sglang/specstream_prepared
export DATASET=$PREPARED_ROOT/sharegpt_v3_merged.json
export RUN_ROOT=$PWD/results/innovation2_step3_$(date +%Y%m%d_%H%M%S)

export SMCTRL_LIB=$PWD/csrc/specstream_smctrl/build/libsmctrl.so
export MPS_PIPE=/tmp/specstream-mps-$USER
export MPS_LOG=/tmp/specstream-mps-log-$USER


# ======================== 数据集 ========================
# 你的服务器当前数据集根目录：
export DATA_ROOT=$HOME/autodl-tmp/dataset

# 原始 GSM8K 目录（Hugging Face 仓库布局：README.md / eval.yaml / main / socratic）
export GSM8K_ROOT=$DATA_ROOT/gsm8k

# 原始 LongBench v2 文件（你当前下载的是 data.json）
export LONGBENCH_V2_RAW=$DATA_ROOT/LongBench-v2/data.json

# 统一把论文实验实际读取的数据放到 prepared 目录，避免改动原始数据集。


# 下面两个文件由第 3 节的“本地数据适配”命令生成。
export GSM8K_TEST=$PREPARED_ROOT/gsm8k_main_test.jsonl
export LONGBENCH_V2=$PREPARED_ROOT/longbench_v2.jsonl

# ShareGPT V3：仅用于性能测试
export SHAREGPT_V3_ROOT=$DATA_ROOT/ShareGPT_V3
export SHAREGPT_JSON=$PREPARED_ROOT/sharegpt_v3_merged.json

# ======================== 服务端口 ========================
export BASE_URL=http://127.0.0.1:30000
export TARGET_PORT=30000
export DRAFT_PORT=30001
export ZMQ_PORT=29000

# ======================== GPU ========================
# 创新点一双卡实验：Target 与 Draft 分卡
export DRAFT_GPU=0
export TARGET_GPU=1

# 创新点二同卡实验必须使用物理 GPU UUID，而不是仅写逻辑编号。
# 先执行 nvidia-smi -L 后修改：
export SINGLE_GPU_UUID=GPU-342220f7-2293-1a7e-08df-73cec29f44f5

# ======================== 结果目录 ========================
export RESULT_ROOT=$REPO/results/paper
mkdir -p \
  $RESULT_ROOT/{accuracy,attention,bench,profiles,resource_profiles,smctrl,nsys,gpu_monitor,logs,source_data}
```

检查模型和数据存在：

```bash
test -d "$TARGET_MODEL" || echo "ERROR: TARGET_MODEL not found"
test -d "$DRAFT_MODEL" || echo "ERROR: DRAFT_MODEL not found"
test -d "$GSM8K_ROOT" || echo "ERROR: GSM8K_ROOT not found"
test -f "$LONGBENCH_V2_RAW" || echo "ERROR: LongBench-v2/data.json not found"

echo "GSM8K local files:"
find "$GSM8K_ROOT" -maxdepth 2 -type f | sort

echo "LongBench-v2 local files:"
find "$DATA_ROOT/LongBench-v2" -maxdepth 1 -type f | sort
```

---

# 2. Gate 0：正式实验前必须通过的代码与协议检查

## 2.1 重编译 SPECTRE C++ ZMQ 扩展

只要修改过或同步过 SPECTRE 协议相关代码，就执行：

```bash
cd "$REPO/python/sglang/srt/speculative/spectre/cpp_zmq"
python setup.py build_ext --inplace --force

cd "$REPO"
python -m pip install -e ./python --no-deps
```

检查实际加载位置：

```bash
python - <<'PY'
from sglang.srt.speculative.spectre import cpp_zmq
from sglang.srt.speculative.spectre.cpp_zmq import spectre_zmq
print("cpp_zmq:", cpp_zmq.__file__)
print("spectre_zmq:", spectre_zmq.__file__)
PY
```

路径必须指向当前 `~/lifei/SpecStream` 工作树。

---

## 2.2 检查当前 CLI 中的 SpecStream 参数

```bash
python -m sglang.launch_server --help | grep -E \
  'spectre-fixed-q-mode|specstream-enabled|specstream-profile-only|specstream-dynamic-q|specstream-cohort|specstream-smctrl'
```

至少应看到：

```text
--specstream-enabled
--specstream-profile-only
--specstream-full-restore-baseline
--specstream-dynamic-q
--specstream-cohort-enabled
--specstream-smctrl-enabled
--specstream-coexec-resource-profile-path
--specstream-shadow-attention
```

---

## 2.3 跑当前 SpecStream 全部单元测试

```bash
cd "$REPO"
PYTHONPATH=python pytest -q python/sglang/test/spectre_specstream \
  | tee "$RESULT_ROOT/logs/pytest_specstream.txt"
```

**只有全部通过才继续。**

---

# 3. 本地数据集适配与预检查

你当前服务器的数据布局是：

```text
~/autodl-tmp/dataset/
├── gsm8k/
│   ├── README.md
│   ├── eval.yaml
│   ├── main/
│   └── socratic/
└── LongBench-v2/
    ├── README.md
    └── data.json
```

这与前一版手册假定的 `gsm8k/test.jsonl` 和 `LongBench-v2/longbench_v2.jsonl`
不同。**不要移动、重命名或覆盖原始数据集。** 统一在
`$PREPARED_ROOT` 下生成 SpecStream 实验使用的兼容副本。

---

## 3.1 GSM8K：从 `main/` 中自动定位 test split 并转为 JSONL

首先查看实际文件：

```bash
find "$GSM8K_ROOT/main" -maxdepth 1 -type f -printf '%f\n' | sort
find "$GSM8K_ROOT/socratic" -maxdepth 1 -type f -printf '%f\n' | sort
```

论文主准确率使用 **`main` 配置的 test split**，不使用 `socratic`。

执行下面的适配脚本。它会优先读取：

1. `main/test*.parquet`
2. `main/test*.jsonl`
3. `main/test*.json`

并统一写成仓库 `benchmark/gsm8k/bench_sglang.py` 可直接读取的
`question/answer` JSONL。

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["GSM8K_ROOT"]) / "main"
out = Path(os.environ["GSM8K_TEST"])
out.parent.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(root.glob("test*.parquet"))
jsonl_files = sorted(root.glob("test*.jsonl"))
json_files = sorted(root.glob("test*.json"))

rows = []

if parquet_files:
    import pandas as pd
    for p in parquet_files:
        rows.extend(pd.read_parquet(p).to_dict(orient="records"))
    source = ", ".join(str(p) for p in parquet_files)

elif jsonl_files:
    for p in jsonl_files:
        with p.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    source = ", ".join(str(p) for p in jsonl_files)

elif json_files:
    for p in json_files:
        with p.open(encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            rows.extend(obj)
        elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
            rows.extend(obj["data"])
        else:
            raise ValueError(f"Unsupported GSM8K JSON structure: {p}")
    source = ", ".join(str(p) for p in json_files)

else:
    raise FileNotFoundError(
        f"No test split found under {root}. "
        "Run `find $GSM8K_ROOT/main -maxdepth 1 -type f` and inspect the filenames."
    )

if not rows:
    raise ValueError("GSM8K test split is empty")

required = {"question", "answer"}
missing = required - set(rows[0])
if missing:
    raise ValueError(f"GSM8K fields missing: {sorted(missing)}; keys={sorted(rows[0])}")

with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(
            {"question": row["question"], "answer": row["answer"]},
            ensure_ascii=False
        ) + "\n")

print("source =", source)
print("output =", out)
print("examples =", len(rows))
print("keys =", sorted(rows[0].keys()))
print("question =", str(rows[0]["question"])[:120])
print("answer =", str(rows[0]["answer"])[-120:])
PY
```

检查：

```bash
wc -l "$GSM8K_TEST"
head -n 1 "$GSM8K_TEST" | python -m json.tool
```

此后正文所有 GSM8K 命令仍然使用：

```bash
--data-path "$GSM8K_TEST"
```

无需修改 `benchmark/gsm8k/bench_sglang.py`。该 evaluator 本身按最终数值答案
计算 accuracy，并支持本地 `--data-path`。

---

## 3.2 LongBench v2：将 `data.json` 统一转换为 JSONL

你下载的官方目录只有：

```text
LongBench-v2/
├── README.md
└── data.json
```

当前 SpecStream 仓库内置的 `longbench_v2` serving loader 对非 parquet 文件按
**JSONL（一行一个 JSON 对象）**读取，因此不能直接假设 `data.json` 就是 JSONL。
先检查其顶层结构：

```bash
python - <<'PY'
import json, os
p = os.environ["LONGBENCH_V2_RAW"]
with open(p, encoding="utf-8") as f:
    obj = json.load(f)
print("top-level type =", type(obj).__name__)
if isinstance(obj, list):
    print("examples =", len(obj))
    print("keys =", sorted(obj[0].keys()) if obj else [])
elif isinstance(obj, dict):
    print("top-level keys =", sorted(obj.keys()))
PY
```

然后统一转成 `$LONGBENCH_V2`：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["LONGBENCH_V2_RAW"])
dst = Path(os.environ["LONGBENCH_V2"])
dst.parent.mkdir(parents=True, exist_ok=True)

with src.open(encoding="utf-8") as f:
    obj = json.load(f)

if isinstance(obj, list):
    rows = obj
elif isinstance(obj, dict):
    # 兼容少数镜像把样本包装在 data/examples 字段中的情况。
    if isinstance(obj.get("data"), list):
        rows = obj["data"]
    elif isinstance(obj.get("examples"), list):
        rows = obj["examples"]
    else:
        raise ValueError(
            "Unsupported LongBench-v2 JSON object. "
            f"Top-level keys: {sorted(obj.keys())}"
        )
else:
    raise ValueError(f"Unsupported LongBench-v2 top-level type: {type(obj).__name__}")

if not rows:
    raise ValueError("LongBench-v2 is empty")

required = {
    "context", "question",
    "choice_A", "choice_B", "choice_C", "choice_D",
    "answer",
}
missing = required - set(rows[0])
if missing:
    raise ValueError(
        f"LongBench-v2 fields missing: {sorted(missing)}; "
        f"keys={sorted(rows[0].keys())}"
    )

with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("source =", src)
print("output =", dst)
print("examples =", len(rows))
print("keys =", sorted(rows[0].keys()))
print("answer =", rows[0].get("answer"))
PY
```

检查：

```bash
wc -l "$LONGBENCH_V2"
head -n 1 "$LONGBENCH_V2" | python -m json.tool
```

从这里开始：

- LongBench v2 最终 accuracy helper 使用 `$LONGBENCH_V2`；
- Attention 一致性采样也使用 `$LONGBENCH_V2`；
- 如果后面调用 SGLang 内置 `--dataset-name longbench_v2`，同样传入
  `--dataset-path "$LONGBENCH_V2"`。

这样可以完全离线运行，也不会修改原始 `data.json`。

---

## 3.3 ShareGPT V3：只用于性能实验

先检查三个 split：

```bash
python scripts/specstream/prepare_datasets.py inspect \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split1.json"
python scripts/specstream/prepare_datasets.py inspect \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split2.json"
python scripts/specstream/prepare_datasets.py inspect \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split3.json"
```

合并：

```bash
mkdir -p "$PREPARED_ROOT"

python scripts/specstream/prepare_datasets.py merge-sharegpt \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split1.json" \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split2.json" \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split3.json" \
  --output "$SHAREGPT_JSON" \
  | tee "$RESULT_ROOT/logs/prepare_sharegpt.txt"

python scripts/specstream/prepare_datasets.py inspect "$SHAREGPT_JSON"
```

性能实验的受控长上下文使用：

```text
--dataset-name random
--dataset-path $SHAREGPT_JSON
--random-range-ratio 1
```

`random-range-ratio=1` 很重要：它才表示严格固定输入/输出长度。

---

# 4. 共用服务健康检查与 smoke test

任何 Target 启动后都先执行：

```bash
curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/v1/models" | python -m json.tool
```

然后运行 8 请求 smoke：

```bash
CASE_TAG=smoke \
DATASET_NAME=random-ids \
INPUT_LEN=4096 OUTPUT_LEN=64 \
NUM_PROMPTS=8 REQUEST_RATE=1 MAX_CONCURRENCY=1 \
RANGE_RATIO=1 WARMUP_REQUESTS=1 SEED=1 \
OUTPUT_DIR="$RESULT_ROOT/bench" \
bash scripts/specstream/run_benchmark_case.sh
```

要求：

- 8/8 成功；
- Failed requests = 0；
- Target/Draft 无 traceback；
- 无 CUDA assert、NaN/Inf；
- 之后再进入正式数据集。

---

# 5. 正确性实验：只比较最终数据集准确率

## 5.1 主比较配置

只保留三个方法：

| 标签 | 方法 |
|---|---|
| ACC-SGL | 原生 SGLang `STANDALONE` 单卡 speculative decoding |
| ACC-SP | SPECTRE parallel，GPU-resident KV，Target/Draft 分卡 |
| ACC-SS | 完整 SpecStream 创新点一：fused streaming + prefetch + dynamic q + Cohort |

所有准确率实验：

```text
temperature = 0
top_p = 1
```

同一数据集三个系统必须使用完全相同的输入、prompt 格式和最大输出长度。

---

## 5.2 ACC-SGL：原生 SGLang 单卡 speculative baseline

启动：

```bash
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port "$TARGET_PORT" \
  --skip-server-warmup \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  2>&1 | tee "$RESULT_ROOT/logs/ACC_SGL_server.log"
```

新终端做 health check 后跑数据集。

---

## 5.3 ACC-SP：SPECTRE parallel GPU-resident baseline

### Target：先启动

```bash
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port "$TARGET_PORT" \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode parallel --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --spectre-failure-threshold 3 --spectre-cooldown-rounds 32 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT" \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  2>&1 | tee "$RESULT_ROOT/logs/ACC_SP_target.log"
```

### Drafter：Target ready 后再启动

```bash
CUDA_VISIBLE_DEVICES="$DRAFT_GPU" python -m sglang.launch_server \
  --model-path "$DRAFT_MODEL" --port "$DRAFT_PORT" \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role draft \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --spectre-draft-priority --spectre-max-draft-priority-steps 8 \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT" \
  2>&1 | tee "$RESULT_ROOT/logs/ACC_SP_draft.log"
```

---

## 5.4 ACC-SS：完整创新点一

Target：

```bash
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port "$TARGET_PORT" \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --spectre-failure-threshold 3 --spectre-cooldown-rounds 32 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --specstream-enabled \
  --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-dynamic-q \
  --specstream-q-candidates 1,2,4,6,8 \
  --specstream-q-switch-threshold 0.08 \
  --specstream-cohort-enabled \
  --specstream-max-cohort-size 8 \
  --specstream-profile-path "$RESULT_ROOT/profiles/ACC_SS.csv" \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT" \
  2>&1 | tee "$RESULT_ROOT/logs/ACC_SS_target.log"
```

Drafter 使用与 ACC-SP 相同命令即可。

---

# 6. GSM8K 准确率

## 6.1 计算本地测试集大小

```bash
export GSM8K_N=$(grep -cve '^$' "$GSM8K_TEST")
echo "GSM8K_N=$GSM8K_N"
```

## 6.2 对当前正在运行的系统执行

把 `METHOD` 分别设为 `SGL`、`SP`、`SS`：

```bash
export METHOD=SS

python benchmark/gsm8k/bench_sglang.py \
  --host 127.0.0.1 --port "$TARGET_PORT" --backend srt \
  --data-path "$GSM8K_TEST" \
  --num-questions "$GSM8K_N" \
  --num-shots 5 \
  --parallel 32 \
  --max-new-tokens 512 \
  --temperature 0 --top-p 1 \
  --result-file "$RESULT_ROOT/accuracy/gsm8k_summary.jsonl" \
  --raw-result-file "$RESULT_ROOT/accuracy/gsm8k_${METHOD}_raw.jsonl" \
  | tee "$RESULT_ROOT/accuracy/gsm8k_${METHOD}.log"
```

依次执行：

```text
ACC-SGL -> METHOD=SGL
ACC-SP  -> METHOD=SP
ACC-SS  -> METHOD=SS
```

每换方法必须重启对应服务。

最终主表只需要记录：

```text
SGLang STANDALONE   accuracy = ?
SPECTRE parallel    accuracy = ?
SpecStream-I1       accuracy = ?
```

---

# 7. LongBench v2 准确率辅助脚本

仓库当前有 `longbench_v2` 的 serving dataset loader，但没有直接把 A/B/C/D 输出计算成最终 accuracy 的论文 evaluator。因此这里增加一个**独立测试辅助脚本**；它不修改运行时，也不参与方法实现。

创建目录：

```bash
mkdir -p scripts/specstream/paper_eval
```

创建：

```bash
cat > scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py <<'PY'
#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import AutoTokenizer


def load_rows(path):
    if path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient='records')

    # Preferred paper path is the prepared JSONL file.  Keep raw data.json
    # compatibility as a safeguard.
    with open(path, encoding='utf-8') as f:
        if path.endswith('.json'):
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get('data'), list):
                return obj['data']
            if isinstance(obj, dict) and isinstance(obj.get('examples'), list):
                return obj['examples']
            raise ValueError(f'Unsupported JSON structure: {type(obj).__name__}')
        return [json.loads(x) for x in f if x.strip()]


def prompt_of(x):
    return (
        f"{x['context']}\n\n"
        f"Question: {x['question']}\n"
        f"A. {x['choice_A']}\n"
        f"B. {x['choice_B']}\n"
        f"C. {x['choice_C']}\n"
        f"D. {x['choice_D']}\n"
        "Answer:"
    )


def normalize_label(x):
    s=str(x).strip().upper()
    m=re.search(r'\b([ABCD])\b', s)
    return m.group(1) if m else ''


def infer(base_url, prompt, max_new_tokens):
    payload={
        'text': prompt,
        'sampling_params': {
            'temperature': 0,
            'top_p': 1,
            'max_new_tokens': max_new_tokens,
            'ignore_eos': False,
        },
        'stream': False,
    }
    r=requests.post(base_url.rstrip('/') + '/generate', json=payload, timeout=1800)
    r.raise_for_status()
    obj=r.json()
    text=obj.get('text','')
    return text, normalize_label(text)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:30000')
    ap.add_argument('--answer-field', default='answer')
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-prompt-tokens', type=int, default=0)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    src=load_rows(args.dataset)
    selected=[]
    for i,x in enumerate(src):
        p=prompt_of(x)
        n=len(tok(p).input_ids)
        if n < args.min_prompt_tokens:
            continue
        selected.append((i,x,p,n))
        if args.limit and len(selected)>=args.limit:
            break

    print('selected examples =', len(selected))
    out=[None]*len(selected)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(infer,args.base_url,p,args.max_new_tokens):j
              for j,(_,_,p,_) in enumerate(selected)}
        for fut in as_completed(futs):
            j=futs[fut]
            i,x,p,n=selected[j]
            try:
                text,pred=fut.result()
                gold=normalize_label(x[args.answer_field])
                out[j]={
                    'source_index': i,
                    'prompt_tokens': n,
                    'gold': gold,
                    'pred': pred,
                    'correct': pred==gold,
                    'output': text,
                    'error': '',
                }
            except Exception as e:
                out[j]={
                    'source_index': i,
                    'prompt_tokens': n,
                    'gold': normalize_label(x.get(args.answer_field,'')),
                    'pred': '',
                    'correct': False,
                    'output': '',
                    'error': repr(e),
                }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as f:
        for x in out:
            f.write(json.dumps(x,ensure_ascii=False)+'\n')

    ok=[x for x in out if not x['error']]
    acc=sum(x['correct'] for x in ok)/len(ok) if ok else 0.0
    print(f'total={len(out)} success={len(ok)} errors={len(out)-len(ok)} accuracy={acc:.6f}')

if __name__=='__main__':
    main()
PY

chmod +x scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py
```

---

## 7.1 对三个系统分别运行 LongBench v2

完整数据集：

```bash
export METHOD=SS

python scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py \
  --dataset "$LONGBENCH_V2" \
  --model "$TARGET_MODEL" \
  --base-url "$BASE_URL" \
  --workers 16 \
  --max-new-tokens 32 \
  --output "$RESULT_ROOT/accuracy/longbench_v2_${METHOD}.jsonl" \
  | tee "$RESULT_ROOT/accuracy/longbench_v2_${METHOD}.log"

export METHOD=SGL

python \
  scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py \
  --dataset "$LONGBENCH_PREPARED" \
  --base-url "$BASE_URL" \
  --workers 2 \
  --max-new-tokens 128 \
  --output \
  "$RESULT_ROOT/accuracy/longbench_v2_${METHOD}.jsonl" \
  | tee \
  "$RESULT_ROOT/accuracy/longbench_v2_${METHOD}.log"
```

依次执行：

```text
METHOD=SGL
METHOD=SP
METHOD=SS
```

如果你下载的 LongBench v2 官方评估规则要求更严格的答案提取方式，则保持三个系统输出不变，仅统一替换 evaluator；**不能三个系统使用不同 parser**。

---

# 8. Attention 一致性热力图

用户当前的正确性主图只需要一个 attention 一致性热力图即可。

## 8.1 推荐的热力图定义

不要只挑一个“看起来很好”的请求。

推荐选择 LongBench v2 中 **32 条 prompt token 数 ≥8192 的长上下文样本**。

每个热力图单元：

```text
request × transformer layer
```

数值为该请求在这一层所有 verification round 中：

```text
max(max_abs(fused_output - Torch_reference_output))
```

绘图时使用：

```text
log10(max_abs + 1e-12)
```

这样既能看到不同层的数值误差，也能避免几个极小值挤在同一颜色区间。

---

## 8.2 启动 shadow-attention Target

为避免 dynamic q/Cohort 让诊断图过于复杂，attention 一致性诊断建议用固定 q=5、单请求或低并发，只验证 fused streaming 数值路径。

Target：

```bash
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port "$TARGET_PORT" \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode ordinary --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --specstream-enabled \
  --no-specstream-reference-attention \
  --specstream-shadow-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path "$RESULT_ROOT/attention/attention_shadow.csv" \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port "$ZMQ_PORT" \
  2>&1 | tee "$RESULT_ROOT/logs/ATTN_target.log"
```

然后启动普通 SPECTRE Drafter。

---

## 8.3 发送 32 条真正进入 CPU History 的 LongBench 样本

```bash
python scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py \
  --dataset "$LONGBENCH_V2" \
  --model "$TARGET_MODEL" \
  --base-url "$BASE_URL" \
  --workers 1 \
  --limit 32 \
  --min-prompt-tokens 8192 \
  --max-new-tokens 64 \
  --output "$RESULT_ROOT/attention/attention_longbench_requests.jsonl"
```

SpecStream 会自动生成：

```text
$RESULT_ROOT/attention/attention_shadow.diagnostics.jsonl
```

其中 `kind=attention_shadow` 行包含：

```text
rid
round_id
layer_id
max_abs
relative_l2
```

先检查：

```bash
head "$RESULT_ROOT/attention/attention_shadow.diagnostics.jsonl"
```

---

## 8.4 生成热力图 Source Data 和图

创建：

```bash
cat > scripts/specstream/paper_eval/plot_attention_heatmap.py <<'PY'
#!/usr/bin/env python3
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-prefix', required=True)
    args=ap.parse_args()

    # (rid, layer) -> all round max_abs values
    vals=defaultdict(list)
    with open(args.input,encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            x=json.loads(line)
            if x.get('kind')!='attention_shadow':
                continue
            vals[(str(x['rid']),int(x['layer_id']))].append(float(x['max_abs']))

    rids=sorted({k[0] for k in vals})
    layers=sorted({k[1] for k in vals})
    if not rids or not layers:
        raise RuntimeError('No attention_shadow records found')

    # conservative cell definition: max over all verification rounds
    mat=np.full((len(rids),len(layers)),np.nan,dtype=float)
    for i,rid in enumerate(rids):
        for j,layer in enumerate(layers):
            v=vals.get((rid,layer))
            if v:
                mat[i,j]=max(v)

    prefix=Path(args.out_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)

    # Source Data: raw max_abs, not log-transformed
    df=pd.DataFrame(mat,index=[f'R{i+1}' for i in range(len(rids))],columns=layers)
    df.to_csv(str(prefix)+'_source_data.csv',index_label='request')

    z=np.log10(np.maximum(mat,1e-12))
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    im=ax.imshow(z,aspect='auto',interpolation='nearest',cmap='viridis')
    ax.set_xlabel('Transformer layer')
    ax.set_ylabel('Long-context request')
    ax.set_xticks(range(0,len(layers),max(1,len(layers)//8)))
    ax.set_xticklabels([layers[i] for i in range(0,len(layers),max(1,len(layers)//8))])
    step=max(1,len(rids)//8)
    ax.set_yticks(range(0,len(rids),step))
    ax.set_yticklabels([f'R{i+1}' for i in range(0,len(rids),step)])
    cbar=fig.colorbar(im,ax=ax,pad=0.02)
    cbar.set_label(r'$\log_{10}(\mathrm{max\ absolute\ error})$')
    fig.tight_layout()
    fig.savefig(str(prefix)+'.pdf',bbox_inches='tight')
    fig.savefig(str(prefix)+'.png',dpi=300,bbox_inches='tight')
    plt.close(fig)

    finite=mat[np.isfinite(mat)]
    print('requests=',len(rids),'layers=',len(layers))
    print('global max_abs=',float(np.max(finite)))
    print('median max_abs=',float(np.median(finite)))

if __name__=='__main__':
    main()
PY

chmod +x scripts/specstream/paper_eval/plot_attention_heatmap.py
```

运行：

```bash
python scripts/specstream/paper_eval/plot_attention_heatmap.py \
  --input "$RESULT_ROOT/attention/attention_shadow.diagnostics.jsonl" \
  --out-prefix "$RESULT_ROOT/attention/fig_attention_consistency"
```

最终保留：

```text
fig_attention_consistency.pdf
fig_attention_consistency.png
fig_attention_consistency_source_data.csv
```

论文正文只需要报告：热力图整体误差范围 + `global max_abs`，无需再增加多张正确性图。

---

# 9. 创新点一：共用 Drafter 命令

后续 K0–K6 都使用同一个 Remote Drafter，除非 fixed-q ablation 特别修改 q。

```bash
export DRAFT_CMD_COMMON="python -m sglang.launch_server \
  --model-path '$DRAFT_MODEL' --port $DRAFT_PORT \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role draft \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --spectre-draft-priority --spectre-max-draft-priority-steps 8 \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port $ZMQ_PORT"
```

后续使用仓库现有自动启动器：

```text
scripts/specstream/run_dedicated_draft_target_baseline.sh
```

该脚本会：

1. 先启动 Target；
2. 等待 `/health`；
3. 再启动 Drafter；
4. 执行 benchmark；
5. 自动清理两个进程。

---

# 10. 创新点一：K0–K6 Target 参数

下面先定义公共 SPECTRE Target 参数：

```bash
export TARGET_COMMON="python -m sglang.launch_server \
  --model-path '$TARGET_MODEL' --port $TARGET_PORT \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode parallel --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-failure-threshold 3 --spectre-cooldown-rounds 32 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port $ZMQ_PORT \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule"
```

## K0：GPU-resident SPECTRE parallel

```bash
export K0_TARGET_CMD="$TARGET_COMMON"
```

## K1：Full-Restore-per-round

```bash
export K1_TARGET_CMD="$TARGET_COMMON \
  --specstream-enabled \
  --specstream-full-restore-baseline \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 1 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path '$RESULT_ROOT/profiles/K1.csv'"
```

## K2：bounded reference streaming

```bash
export K2_TARGET_CMD="$TARGET_COMMON \
  --specstream-enabled \
  --specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 1 \
  --no-specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path '$RESULT_ROOT/profiles/K2.csv'"
```

## K3：fused/grouped streaming，无 layer prefetch

```bash
export K3_TARGET_CMD="$TARGET_COMMON \
  --specstream-enabled \
  --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --no-specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path '$RESULT_ROOT/profiles/K3.csv'"
```

## K4：K3 + 双缓冲 + 跨层预取

```bash
export K4_TARGET_CMD="$TARGET_COMMON \
  --specstream-enabled \
  --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path '$RESULT_ROOT/profiles/K4.csv'"
```

## K5：K4 + dynamic q

动态 q 时不要写死 `--spectre-fixed-q-mode parallel` 的含义为“强制 parallel”；当前 dynamic controller 会接管 `(mode,q)`。保留该参数只作为 dynamic disabled 时默认值。

```bash
export K5_TARGET_CMD="$TARGET_COMMON \
  --specstream-enabled \
  --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-dynamic-q \
  --specstream-q-candidates 1,2,4,6,8 \
  --specstream-q-switch-threshold 0.08 \
  --specstream-profile-path '$RESULT_ROOT/profiles/K5.csv'"
```

## K6：完整创新点一

```bash
export K6_TARGET_CMD="$K5_TARGET_CMD \
  --specstream-cohort-enabled \
  --specstream-max-cohort-size 8 \
  --specstream-max-cohort-delay-us 200 \
  --specstream-profile-path '$RESULT_ROOT/profiles/K6.csv'"
```

---

# 11. 创新点一：第一阶段 screening

不要一开始就跑完整大网格。先跑最有诊断价值的三个点：

```text
16K / C1
16K / C8
30K / C8
```

每个点先 200 requests。

## 11.1 示例：运行 K4 的 16K/C8

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$TARGET_GPU"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$DRAFT_GPU"
export SPECSTREAM_TARGET_CMD="$K4_TARGET_CMD"
export SPECSTREAM_DRAFT_CMD="$DRAFT_CMD_COMMON"
export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/K4_16k_c8"

export SPECSTREAM_BENCH_CMD="CASE_TAG=K4_16k_c8 \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=4 SEED=1 OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

对 K0–K6 依次重复，只替换：

```text
SPECSTREAM_TARGET_CMD
CASE_TAG
SPECSTREAM_RESULT_ROOT
```

---

## 11.2 推荐 screening 表

| Variant | 16K/C1 | 16K/C8 | 30K/C8 |
|---|---:|---:|---:|
| K0 | ✓ | ✓ | ✓ |
| K1 | ✓ | ✓ | ✓ |
| K2 | ✓ | ✓ | 可选，reference 很慢时跳过 |
| K3 | ✓ | ✓ | ✓ |
| K4 | ✓ | ✓ | ✓ |
| K5 | ✓ | ✓ | ✓ |
| K6 | ✓ | ✓ | ✓ |

筛选后先汇总：

```bash
python scripts/specstream/summarize_benchmarks.py \
  "$RESULT_ROOT/bench/K*.jsonl" \
  > "$RESULT_ROOT/source_data/screening_bench.tsv"

python scripts/specstream/summarize_specstream_profile.py \
  "$RESULT_ROOT/profiles/K*.csv" \
  > "$RESULT_ROOT/source_data/screening_profile.tsv"
```

重点检查：

```text
output_throughput
p99_ttft_ms
p99_tpot_ms
p99_e2e_latency_ms
mean_accept_length
error_count
h2d_ops
h2d_mib_per_accepted
stream_attn_ops_per_accepted
accepted_tokens
q_dist
mode_dist
mean_cohort / max_cohort
fallback_rows
```

---

# 12. 创新点一：正式 Context × Concurrency 网格

正式输入长度：

```text
4K：短上下文/fallback 边界
16K：主 long-context 点
30K：stress / capacity 点
```

并发：

```text
1, 4, 8, 16, 32
```

饱和吞吐采用：

```text
request-rate = inf
```

## 12.1 当前已启动某个 Variant 后执行

```bash
export VARIANT=K6

for INPUT_LEN in 4096 16384 30000; do
  for C in 1 4 8 16 32; do
    CASE_TAG="${VARIANT}_${INPUT_LEN}_c${C}" \
    DATASET_NAME=random \
    DATASET_PATH="$SHAREGPT_JSON" \
    INPUT_LEN="$INPUT_LEN" OUTPUT_LEN=128 \
    NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY="$C" \
    RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 \
    OUTPUT_DIR="$RESULT_ROOT/bench" \
    bash scripts/specstream/run_benchmark_case.sh
  done
done
```

**Screening 可以同一个 server 连续跑多个点；最终论文 confirmatory run 要求每个 rep 重新启动 Target/Drafter。**

---

# 13. 创新点一：P99 的 Open-loop 到达率实验

P99 不能只用 `request-rate=inf`。

固定主 workload：

```text
context = 16K
max concurrency = 32
request rate = 0.5, 1, 2, 4, 8 requests/s
```

```bash
export VARIANT=K6

for RATE in 0.5 1 2 4 8; do
  CASE_TAG="${VARIANT}_16k_r${RATE}_c32" \
  DATASET_NAME=random DATASET_PATH="$SHAREGPT_JSON" \
  INPUT_LEN=16384 OUTPUT_LEN=128 \
  NUM_PROMPTS=1000 REQUEST_RATE="$RATE" MAX_CONCURRENCY=32 \
  RANGE_RATIO=1 WARMUP_REQUESTS=8 SEED=1 \
  OUTPUT_DIR="$RESULT_ROOT/bench" \
  bash scripts/specstream/run_benchmark_case.sh
done
```

论文至少比较：

```text
K0 vs K1 vs K4 vs K5 vs K6
```

K2/K3 可放 Extended Data。

---

# 14. 创新点一：fixed-q vs dynamic-q

固定 q 的约束：

```text
q = speculative_num_steps + 1 = speculative_num_draft_tokens
```

对应：

| q | num_steps | draft_tokens |
|---:|---:|---:|
| 2 | 1 | 2 |
| 4 | 3 | 4 |
| 6 | 5 | 6 |
| 8 | 7 | 8 |

不要用 `num_steps=0` 构造 fixed q=1 baseline。

建议在：

```text
16K/C1
16K/C8
30K/C8
```

分别测 fixed q=2/4/6/8，然后与 K5 dynamic q 比较。

固定 q 每次必须让 Target 和 Drafter 使用一致的 steps/tokens，并完整重启两端。

最终报告：

```text
q_dist
mode_dist
mean_accept_length
h2d_mib_per_accepted
output_throughput
p99_tpot_ms
fallback_rows
```

如果 dynamic 实验中健康负载几乎总是 `parallel`，论文不要硬写“频繁主动串并行切换”，应写成：

```text
dynamic speculation horizon + safety-aware serialization/fallback
```

---

# 15. 创新点一：Chunk-Cohort 消融

保持 K5 不变，只扫描：

```text
max cohort size = 1, 2, 4, 8
```

主 workload：

```text
16K/C8
16K/C16
16K/C32
```

另外补一个自然 ShareGPT：

```bash
CASE_TAG=K6_sharegpt_r4_c16 \
DATASET_NAME=sharegpt DATASET_PATH="$SHAREGPT_JSON" \
OUTPUT_LEN=256 CONTEXT_LEN=32768 \
NUM_PROMPTS=1000 REQUEST_RATE=4 MAX_CONCURRENCY=16 \
WARMUP_REQUESTS=8 SEED=1 \
OUTPUT_DIR="$RESULT_ROOT/bench" \
bash scripts/specstream/run_benchmark_case.sh
```

Cohort 主要看：

```text
h2d_ops
stream_attn_ops
mean_cohort
max_cohort
round_ms
throughput
P99
```

**不要把 Cohort 写成跨请求 KV 内容去重。** 当前实现会打包兼容请求，但仍传输各请求自己的 KV bytes；其主要收益应体现在 event/launch 数和 fused execution efficiency。

---

# 16. 创新点一：Nsight Systems 机制验证

只选择 3 个代表点：

```text
16K/C1
16K/C8
30K/C8
```

重点比较：

```text
K1 Full-Restore
K3 fused no-prefetch
K4 fused + layer-prefetch
K6 full I1
```

示例：

```bash
mkdir -p "$RESULT_ROOT/nsys"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o "$RESULT_ROOT/nsys/K4_16k_c8" \
  bash -lc "CASE_TAG=K4_nsys_16k_c8 \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=32 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=2 OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"
```

注意：客户端 nsys 不能捕获服务器 GPU kernel。正式 timeline 应把 `nsys profile` 包在 **Target server 进程** 外层；上面的命令仅演示参数格式。推荐实际做法：

```text
nsys profile ... python -m sglang.launch_server [K4 Target args]
```

然后由另一个终端发送 32 个 benchmark 请求。

观察：

```text
H2D memcpy
copy stream
streaming attention kernel
Tail attention
MLP
next-layer prefetch
```

K1 若没有明确 copy/compute overlap，不要将它命名为“异步全量恢复”。

---

# 17. 创新点二 Gate 1：构建并验证 libsmctrl

创新点二正式实验必须在**实际测试 GPU + 实际 Driver/CUDA** 上重新验证。当前创新点二不再使用 MPS Active Thread Percentage 作为资源调度器；MPS 仅提供多进程同卡并发基础设施，真正的资源控制由 **libsmctrl TPC mask + measurement-backed resource profile + one-token Target-priority grant/ACK** 完成。

```bash
cd "$REPO/csrc/specstream_smctrl"
make config
make build
```

先查看物理 GPU：

```bash
nvidia-smi -L
```

使用 dedicated Drafter 的 global mask 后端时：

```bash
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
make validate-global TPC_LOW=0 TPC_HIGH=4
```

如果失败：

```text
STOP
```

不要通过关闭 validator、硬编码 MASK_OFF 或改回 MPS 百分比来绕过。

记录库：

```bash
export SMCTRL_LIB=$REPO/csrc/specstream_smctrl/build/libsmctrl.so

ls -lh "$SMCTRL_LIB"

sha256sum "$SMCTRL_LIB" \
  | tee "$RESULT_ROOT/smctrl/libsmctrl.sha256"
```

读取实际 TPC 数：

```bash
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
SGLANG_SPECSTREAM_SMCTRL_LIBRARY="$SMCTRL_LIB" \
python - <<'PY'
from sglang.srt.speculative.spectre.specstream.sm_controller import SMController

c = SMController(mask_scope="global")
print("total_tpcs =", c.total_tpcs)
PY
```

将输出写入：

```bash
export TOTAL_TPCS=<实际值>
```

例如当前 A800 若实际返回 54，则：

```bash
export TOTAL_TPCS=54
```

---

## 17.1 创新点二正式实验统一 q=4

旧版手册中 D1/D2 使用：

```text
num_steps = 4
draft_tokens = 5
```

而 D3/D4 calibration 使用：

```text
num_steps = 3
draft_tokens = 4
```

这会导致不同 Variant 的 Draft 计算量、Verify 计算量、acceptance length 和 round 数不一致，不能用于严格因果比较。

因此，从本节开始，**D0–D4 的性能隔离实验全部固定 q=4**：

```text
q = 4
speculative_num_steps = 3
speculative_num_draft_tokens = 4
```

只有 D5 作为完整系统实验打开 dynamic q；D5 的启动/default q 也仍设为 q=4，再由动态控制器选择 q=1/2/4/6/8。

统一变量：

```bash
export I2_NUM_STEPS=3
export I2_NUM_DRAFT_TOKENS=4
```

---

## 17.2 同卡实验必须显式限制两个 SGLang 实例的 KV Cache token pool

旧版 D2 直接把 Target 和 Drafter 都启动在同一张 GPU 上，但两端都没有显式限制：

```text
--max-total-tokens
```

这会导致两个独立 SGLang runtime 都按照“自己独占整卡”的假设规划 KV Cache pool。D2 的“无资源控制”应只表示**没有 TPC/计算资源控制**，而不是允许两边无限制重复预留显存。

因此 D1–D5 的性能对比统一显式设置 token pool。

对于第一阶段：

```text
16K / C8 / output=128
```

先使用：

```bash
export I2_TARGET_MAX_TOTAL_TOKENS=200000
export I2_DRAFT_MAX_TOTAL_TOKENS=200000
```

这是 **16K/C8 pilot 的初始诊断值**，不是所有实验固定值。

正式实验前按下面的保守公式估算：

```text
required_tokens
≈ 1.25 × concurrency × (input_len + output_len + 512)
```

建议初始值：

| Workload | 建议 pilot `max-total-tokens` |
|---|---:|
| 4K/C8 | 80K |
| 16K/C8 | 200K |
| 16K/C16 | 400K |
| 30K/C8 | 320K |
| 30K/C16 | 640K |

这些是启动/诊断值，不是论文中的固定系统参数；正式值仍要结合 `/server_info`、实际 admission 和显存监控确认。

为了保证 D1 → D2 → D3 → D4 的比较只改变 GPU topology / TPC policy，**同一 workload 下 D1–D4 使用相同的 `max-total-tokens`**。

---

# 18. 创新点二 Gate 2：MPS 与自动启动器必须先修正

MPS 不是动态调度器，只用于让两个独立 CUDA 进程在同一张物理 GPU 上稳定共执行。

## 18.1 启动 MPS

```bash
export SPECSTREAM_GPU_UUID="$SINGLE_GPU_UUID"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/specstream-mps-${USER}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/specstream-mps-log-${USER}"

SPECSTREAM_GPU_UUID="$SINGLE_GPU_UUID" \
CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
bash scripts/specstream/mps/start_mps.sh
```

检查 control daemon：

```bash
pgrep -af nvidia-cuda-mps
```

在尚未启动 CUDA client 时：

```bash
echo get_server_list | nvidia-cuda-mps-control
```

可能为空，这是正常现象。

Target/Drafter 启动以后再次执行：

```bash
echo get_server_list | nvidia-cuda-mps-control
pgrep -af nvidia-cuda-mps
nvidia-smi
```

实验结束后：

```bash
bash scripts/specstream/mps/stop_mps.sh
```

---

## 18.2 自动启动器必须继承当前 `spectre` Conda 环境

检查：

```bash
which python

bash -c \
  'which python; python -c "import sys,sglang; print(sys.executable); print(sglang.__file__)"'
```

必须指向当前：

```text
/root/miniconda3/envs/spectre/bin/python
```

`run_dedicated_draft_target_baseline.sh` 内部必须使用：

```text
bash -c
```

不能重新使用：

```text
bash -lc
```

否则 login shell 可能把 Python 重新切回 `/root/miniconda3/bin/python`，导致：

```text
ModuleNotFoundError: No module named 'sglang'
```

检查：

```bash
grep -n 'bash -lc' \
  scripts/specstream/run_dedicated_draft_target_baseline.sh
```

正式版本应无匹配。

并执行 Bash 语法检查：

```bash
bash -n \
  scripts/specstream/run_dedicated_draft_target_baseline.sh
```

无输出才继续。

---

## 18.3 Target 和 Drafter 必须分别通过 `/health`

创新点二 D1–D5 统一设置：

```bash
export SPECSTREAM_TARGET_READY_CMD="curl -fsS http://127.0.0.1:${TARGET_PORT}/health"

export SPECSTREAM_DRAFT_READY_CMD="curl -fsS http://127.0.0.1:${DRAFT_PORT}/health"

export SPECSTREAM_READY_TIMEOUT_S=300
```

不能只：

```text
sleep 5
```

然后假定 Drafter 已经 ready。

原因是 Target 当前使用：

```text
--spectre-recv-timeout-ms 5000
```

如果 Drafter 因 OOM、初始化失败、ZMQ 异常或显存池冲突而不能提供 draft，Target 可能反复等待接近 5 秒再 fallback，导致 P99 出现明显的：

```text
5s / 10s / 15s ...
```

阶梯。

因此，任何存在以下现象的 run 均视为**无效实验**：

```text
Drafter /health 失败
Drafter 进程退出
remote draft timeout 大量出现
missing_drafts 明显大于 0
P99 出现接近 5000 ms 整数倍的阶梯
```

---

## 18.4 benchmark 必须同时显示进度并保存日志

当前自动启动器的 benchmark 部分应使用：

```bash
PYTHONUNBUFFERED=1 \
bash -c "${SPECSTREAM_BENCH_CMD}" \
  2>&1 | tee "${RESULT_ROOT}/benchmark.log"
```

而不是：

```bash
>"${RESULT_ROOT}/benchmark.log" 2>&1
```

这样终端能实时看到 `bench_serving` 的进度，同时保留完整日志。

---

# 19. 创新点二：统一的 q=4 公共启动命令

第一阶段只验证：

```text
16K / C8 / q=4
```

统一：

```bash
export I2_NUM_STEPS=3
export I2_NUM_DRAFT_TOKENS=4

export I2_TARGET_MAX_TOTAL_TOKENS=200000
export I2_DRAFT_MAX_TOTAL_TOKENS=200000

export I2_CONTEXT_LENGTH=32768
export I2_MAX_PREFILL_TOKENS=16384
```

---

## 19.1 q=4 Drafter 公共命令

```bash
export I2_DRAFT_COMMON="python -m sglang.launch_server \
  --model-path '$DRAFT_MODEL' \
  --port $DRAFT_PORT \
  --context-length $I2_CONTEXT_LENGTH \
  --max-total-tokens $I2_DRAFT_MAX_TOTAL_TOKENS \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role draft \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --spectre-draft-priority \
  --spectre-max-draft-priority-steps 8 \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port $ZMQ_PORT"
```

---

## 19.2 q=4 Target 公共命令

```bash
export I2_TARGET_COMMON="python -m sglang.launch_server \
  --model-path '$TARGET_MODEL' \
  --port $TARGET_PORT \
  --context-length $I2_CONTEXT_LENGTH \
  --max-prefill-tokens $I2_MAX_PREFILL_TOKENS \
  --max-total-tokens $I2_TARGET_MAX_TOTAL_TOKENS \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role target \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --page-size 1 \
  --attention-backend fa3 \
  --spectre-fixed-q-mode parallel \
  --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --spectre-failure-threshold 3 \
  --spectre-cooldown-rounds 32 \
  --spectre-retry-min-count 1 \
  --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port $ZMQ_PORT \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule"
```

从此以后：

```text
D1 = I2_TARGET_COMMON + I2_DRAFT_COMMON，双卡
D2 = I2_TARGET_COMMON + I2_DRAFT_COMMON，同卡，无 TPC 控制
D3 = q4 + fixed TPC
D4 = q4 + online grant
D5 = q4 startup + dynamic q + I1 + I2
```

---

# 20. 创新点二 Gate 3：先修复并验证“首个 Grant 必须先于首个 Draft Forward”

> **这是 2026-08-26 根据服务器真实运行日志新增的强制 Gate。**
>
> 当前代码在进入 D3/D4/D5 与 TPC calibration 之前存在一个明确的启动协议问题：Drafter 开启 `--specstream-smctrl-enabled` 后，会拒绝任何没有 active TPC mask 的 Draft forward；但现有 calibration/online 启动流程没有保证首个 Draft forward 之前已经收到并安装 Grant/TPC mask。
>
> 因此，**当前代码状态下 D1/D2 可以继续测试，但 TPC calibration、D3、D4、D5 必须先通过本节 Gate。** 不能继续堆正式性能数据。

## 20.1 这次真实报错说明了什么

第一次 smoke 中，Drafter 已经完成 CUDA Graph、libsmctrl、ZMQ 和 HTTP server 初始化，显存仍有约 45.96 GB 可用，并不是 OOM；随后第一次 Draft forward 直接触发：

```text
RuntimeError: SpecStream refused an ungranted Draft forward:
no TPC mask is active
```

这说明执行顺序变成了：

```text
Drafter ready
    ↓
Scheduler 得到可运行 batch
    ↓
run_batch()
    ↓
真正 Draft forward
    ↓
发现 active TPC mask = NONE
    ↓
安全检查主动拒绝 forward
```

随后尝试把：

```text
--specstream-smctrl-calibration-tpcs 4
```

也传给 Drafter，错误仍然存在，因此**该参数当前不会自动为 Drafter 安装首个 active mask**。

又尝试去掉：

```text
--spectre-draft-priority
```

则代码在 Drafter 初始化阶段明确拒绝：

```text
RuntimeError: SpecStream SM control requires --spectre-draft-priority
so one-token grant boundaries cannot be bypassed
```

因此可以确定：

```text
1. --spectre-draft-priority 是 I2 的强制条件，不能删除；
2. 仅把 calibration-tpcs 参数传给 Drafter 不能解决问题；
3. 真正缺失的是 first-grant-before-first-forward 的启动/调度协议。
```

---

## 20.2 正确的运行时不变量

创新点二必须满足：

```text
任何受 smctrl 控制的 Draft forward
        ↓
必须已经存在 active grant
        ↓
必须已经把 grant 对应的 TPC mask 应用到真实 Draft CUDA stream/process
        ↓
只执行一个 token step
        ↓
发送 ACK / 消耗 grant token
        ↓
grant 归零后再次暂停
```

形式化地：

```text
DraftForwardAllowed(rid) => ActiveGrant(rid) && ActiveTPCMask(rid)
```

而不是：

```text
先 run_batch，再在 tp_worker 中发现没有 mask 后报错
```

`tp_worker.py` 中“没有 mask 就拒绝 forward”的安全检查必须保留，**不能为了跑通实验而删除或绕过**。

---

## 20.3 当前代码必须修复的两个点

### A. Drafter scheduler 必须在 `run_batch()` 前做 grant gating

设计要求：

```text
收到/维护 active grant table
        ↓
_filter_by_active_grant(draft_batch)
        ↓
没有 active grant：不执行 forward，只等待控制消息
        ↓
有 active grant：先应用 TPC mask
        ↓
run_batch() exactly one token
```

也就是说，`--spectre-draft-priority` 可以保留，但其 priority phase 不能绕过 grant gate。

### B. calibration 模式必须有 bootstrap/static calibration grant

对于固定 TPC calibration，例如：

```text
TPC=4
```

Target calibration runtime 必须在 Drafter 首个受控 forward 之前建立：

```text
Grant(TPC range = 4 TPCs, token_budget = 1 or calibration-defined budget)
```

或者通过等价的 calibration bootstrap 路径让 Drafter 首个 forward 前已经有 active mask。

**测试文档不再假设“把 `--specstream-smctrl-calibration-tpcs` 传给 Drafter即可自动安装 mask”**；实际日志已经证明这一假设不成立。

---

## 20.4 修复代码以后必须先跑 4-request Gate smoke

不要直接恢复 200-request calibration。

统一：

```bash
export TPC=4
export REP=1
export GATE_PROFILE="$RESULT_ROOT/profiles/gate_q4_tpc4.csv"

export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs 4 \
  --specstream-profile-path '$GATE_PROFILE'"

export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_CMD"

export SPECSTREAM_TARGET_READY_CMD="curl -fsS http://127.0.0.1:${TARGET_PORT}/health"
export SPECSTREAM_DRAFT_READY_CMD="curl -fsS http://127.0.0.1:${DRAFT_PORT}/health"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/gate_q4_tpc4"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=gate_q4_tpc4 \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=32 NUM_PROMPTS=4 \
REQUEST_RATE=inf MAX_CONCURRENCY=1 RANGE_RATIO=1 \
WARMUP_REQUESTS=1 SEED=1 CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

Gate 必须同时满足：

```text
Target /health = OK
Draft /health = OK
4/4 请求成功
benchmark error_count = 0
无 “ungranted Draft forward”
无 “no TPC mask is active”
无 timeout 风暴
Draft forward 确实存在非空 TPC mask
profile 中存在有效 grant/TPC/draft_step 记录
```

检查：

```bash
grep -Eini \
  'ungranted Draft forward|no TPC mask|requires --spectre-draft-priority|timeout|missing|traceback|error' \
  "$SPECSTREAM_RESULT_ROOT/target.log" \
  "$SPECSTREAM_RESULT_ROOT/draft.log" \
  "$SPECSTREAM_RESULT_ROOT/benchmark.log" \
  | tail -n 120
```

如果仍出现：

```text
SpecStream refused an ungranted Draft forward
```

则：

```text
STOP
```

说明 I2 runtime 代码尚未修复，**不要进入第 21 节 calibration**。

---

# 21. TPC calibration：仅在 Gate 3 通过后执行

第一阶段仍固定：

```text
context = 16K
max concurrency = 8
q = 4
TPC = 2,4,6,8,12
每点 3 次
```

目标：找到：

```text
Target slowdown <= 5%
且 Draft step 稳定前进
```

## 21.1 Drafter calibration 公共命令

**必须保留 `--spectre-draft-priority`。** 当前代码已经明确要求 smctrl 与 one-token grant 边界依赖该模式。

```bash
export CAL_DRAFT_CMD="$I2_DRAFT_COMMON \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library '$SMCTRL_LIB' \
  --specstream-smctrl-mask-scope global"
```

不要再通过删除：

```text
--spectre-draft-priority
```

来规避问题。

也不要依赖：

```text
--specstream-smctrl-calibration-tpcs $TPC
```

在 Drafter 启动时自动建立 mask；当前实测已经证明它没有解决首个 mask 问题。

---

## 21.2 Target calibration 公共命令

```bash
export CAL_TARGET_COMMON="$I2_TARGET_COMMON \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200"
```

固定 TPC 数仍由 Target calibration 路径指定：

```text
--specstream-smctrl-calibration-tpcs $TPC
```

但代码必须保证该 calibration TPC 配置会在首个 Draft forward 之前转化成有效 grant/mask。

---

# 22. 每个 TPC 运行 Baseline 与 Overlap 两类 calibration

以：

```text
TPC=4
REP=1
```

为例。

每次运行前：

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_CMD"

export SPECSTREAM_TARGET_READY_CMD="curl -fsS http://127.0.0.1:${TARGET_PORT}/health"
export SPECSTREAM_DRAFT_READY_CMD="curl -fsS http://127.0.0.1:${DRAFT_PORT}/health"
```

## 22.1 Baseline：Target forward 不允许 overlap

```bash
export TPC=4
export REP=1
export BASE_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_base.csv"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC \
  --specstream-profile-path '$BASE_PROFILE'"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_base"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=cal_q4_tpc${TPC}_rep${REP}_base \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=4 SEED=$REP CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

## 22.2 Overlap：允许 fixed-TPC Draft 与 Target forward 重叠

```bash
export OVER_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_overlap.csv"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-profile-path '$OVER_PROFILE'"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_overlap"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=cal_q4_tpc${TPC}_rep${REP}_overlap \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=4 SEED=$REP CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

每个 calibration run 必须保证：

```text
ungranted Draft forward = 0
missing_drafts ≈ 0
timeout = 0 或极低
Draft 结束时仍健康
TPC mask/grant 记录非空
```

任何一点不满足，该 TPC run 无效。

---

# 23. 先根据 calibration CSV 统计真实运行 shape

不再预设：

```text
verify_bs8_q4_ctx16k
```

因为：

```text
MAX_CONCURRENCY=8 ≠ 每轮 verification batch_size=8
```

并且输入正好 16384 token 时，进入 decode/verification 后绝大多数 round 很可能已经落入下一 context bucket。

执行：

```bash
python - <<'PY'
import csv, os
from collections import Counter

p=os.environ['OVER_PROFILE']
rows=list(csv.DictReader(open(p)))

print('rows =', len(rows))
print('batch_size =', Counter(r.get('batch_size') for r in rows).most_common())
print('q =', Counter(r.get('q') for r in rows).most_common())
print('context_tokens =', Counter(r.get('context_tokens') for r in rows).most_common(20))

def bucket(v):
    try:
        n=int(float(v))
    except Exception:
        return 'invalid'
    if n <= 4096: return '4k'
    if n <= 8192: return '8k'
    if n <= 16384: return '16k'
    if n <= 32768: return '32k'
    if n <= 65536: return '64k'
    return '>64k'

print('context_bucket =', Counter(
    bucket(r.get('context_tokens'))
    for r in rows if r.get('context_tokens')
).most_common())
PY
```

正式 profile 按真实 `(batch_size,q,context_bucket)` round 频率排序，优先覆盖累计 ≥95%。

---

# 24. 提取 interference sample 并构建 resource profile

对每个真实高频 shape 提取样本，而不是把 shape 名字手写成不存在的组合。

示例：若真实高频 shape 是：

```text
bs=8, q=4, ctx32k
```

则：

```bash
export SAMPLE_FILE="$RESULT_ROOT/resource_profiles/interference_samples.jsonl"

python scripts/specstream/smctrl/extract_interference_sample.py \
  --baseline-profile "$BASE_PROFILE" \
  --overlap-profile "$OVER_PROFILE" \
  --target-shape verify_bs8_q4_ctx32k \
  --draft-bs 8 \
  --draft-ctx-bucket 32k \
  --draft-tpcs "$TPC" \
  --output "$SAMPLE_FILE"
```

当每个 `(shape,bs,ctx,tpc)` 至少有 3 条独立重复后：

```bash
export RESOURCE_PROFILE="$RESULT_ROOT/resource_profiles/specstream_q4_16k_real_shapes.json"

python scripts/specstream/smctrl/build_resource_profile.py \
  "$SAMPLE_FILE" \
  --output "$RESOURCE_PROFILE" \
  --gpu "$SINGLE_GPU_UUID" \
  --draft-model "$DRAFT_MODEL" \
  --target-model "$TARGET_MODEL" \
  --total-tpcs "$TOTAL_TPCS" \
  --min-repetitions 3

cat "$RESOURCE_PROFILE"
```

筛选：

```text
Target slowdown <= 5%
error_count = 0
Draft step latency > 0
TPC/grant 数据非空
```

未校准 shape 继续 fail closed，不允许 nearest-neighbor 猜测。

---

# 25. 创新点二 D0：原生 SGLang 单卡 speculative

D0 作为原生单卡参考，启动与准确率实验中的 SGLang STANDALONE 类似。

为了与 I2 主链的 fixed q=4 对齐，性能隔离实验建议使用：

```text
num_steps=3
draft_tokens=4
```

性能点先只跑：

```text
16K/C8
```

通过后再扩展网格。

---

# 26. D1：SPECTRE 双卡 q=4 强基线

D1 与 D2/D3/D4 使用完全相同的：

```text
q=4
input/output
max-total-tokens
SPECTRE 参数
```

唯一核心差异是：

```text
Target → GPU 1
Draft  → GPU 0
```

使用：

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$TARGET_GPU"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$DRAFT_GPU"
export SPECSTREAM_TARGET_CMD="$I2_TARGET_COMMON"
export SPECSTREAM_DRAFT_CMD="$I2_DRAFT_COMMON"

export SPECSTREAM_TARGET_READY_CMD="curl -fsS http://127.0.0.1:${TARGET_PORT}/health"
export SPECSTREAM_DRAFT_READY_CMD="curl -fsS http://127.0.0.1:${DRAFT_PORT}/health"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D1_16k_c8"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' CASE_TAG=D1_16k_c8 \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=4 SEED=1 CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

D1 不启用 smctrl，因此不受 Gate 3 的 mask/grant 启动问题影响。

---

# 27. D2：同卡 SPECTRE parallel，无 TPC 控制

D2 用于测“直接把 Target/Draft 放同卡”的纯计算资源竞争。

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_TARGET_CMD="$I2_TARGET_COMMON"
export SPECSTREAM_DRAFT_CMD="$I2_DRAFT_COMMON"
```

D2 不启用：

```text
--specstream-smctrl-enabled
```

但 Target/Draft 的 KV token pool 仍显式限制，因此 D2 的“无控制”仅表示：

```text
无 TPC/计算资源控制
```

而不是让两个 runtime 无限制重复预留显存。

D2 仍可在当前代码版本上测试，是定位“单卡天然资源竞争损失”的有效负面对照。

---

# 28. D3：same-GPU + 最佳固定 TPC

**只有 Gate 3 通过并完成有效 calibration 后才能执行。**

从安全点中选择：

```text
Target slowdown <= 5%
且 Draft step latency 最低
```

例如：

```bash
export BEST_TPC=4
```

Target：

```bash
export D3_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $BEST_TPC \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-profile-path '$RESULT_ROOT/profiles/D3_16k_c8.csv'"
```

Drafter：

```bash
export D3_DRAFT_CMD="$CAL_DRAFT_CMD"
```

如果 D3 再次出现：

```text
no TPC mask is active
```

说明 Gate 3 实际没有通过，D3 结果无效。

---

# 29. D4：GPU-resident KV + online Target-priority TPC grant

D4 用于隔离创新点二，Target KV 保持 GPU-resident，因此：

```text
--specstream-profile-only
```

而不是：

```text
--specstream-enabled
```

Target：

```bash
export RESOURCE_PROFILE="$RESULT_ROOT/resource_profiles/specstream_q4_16k_real_shapes.json"

export D4_TARGET_CMD="$I2_TARGET_COMMON \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-grant-token-quantum 1 \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --specstream-coexec-resource-profile-path '$RESOURCE_PROFILE' \
  --specstream-profile-path '$RESULT_ROOT/profiles/D4_16k_c8.csv'"
```

Drafter：

```bash
export D4_DRAFT_CMD="$CAL_DRAFT_CMD"
```

D4 必须验证：

```text
首个 Draft forward 前已有 grant/mask
TARGET_EXCLUSIVE / SLACK_FILL / DRAFT_CATCHUP 有效记录
grant_epoch 单调
每个 grant token quantum = 1
ACK 前不连续发送下一 grant
draft_step_ms > 0
TPC range 非空
missing_drafts ≈ 0
```

正式 D4 主结果建议 resource-profile coverage ≥95%。

---

# 30. D5：创新点一 + 创新点二完整系统

D5 只能在 D4 online grant 已经稳定工作以后执行。

启动/default q 仍固定为：

```text
q=4
```

然后打开：

```text
dynamic q = 1,2,4,6,8
bounded KV streaming
layer prefetch
Chunk-Cohort
online Target-priority TPC grant
```

Target：

```bash
export D5_TARGET_CMD="$I2_TARGET_COMMON \
  --specstream-enabled \
  --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-dynamic-q \
  --specstream-q-candidates 1,2,4,6,8 \
  --specstream-q-switch-threshold 0.08 \
  --specstream-cohort-enabled \
  --specstream-max-cohort-size 8 \
  --specstream-max-cohort-delay-us 200 \
  --specstream-smctrl-enabled \
  --specstream-grant-token-quantum 1 \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --specstream-coexec-resource-profile-path '$RESOURCE_PROFILE' \
  --specstream-profile-path '$RESULT_ROOT/profiles/D5_16k_c8.csv'"
```

Drafter：

```bash
export D5_DRAFT_CMD="$CAL_DRAFT_CMD"
```

若 resource profile 只覆盖 q=4，则其它 q 会 fail closed；这样的 D5 只能作为 debug run。正式 D5 需要按真实 q 分布补齐高频 q/bs/context shape，累计覆盖 ≥95%。

---

# 31. 创新点二最终正确执行顺序

根据 2026-08-26 的真实启动日志，执行顺序更新为：

```text
[1] D1 双卡 q4
    ↓
[2] D2 同卡 q4、无 smctrl
    ↓
[3] libsmctrl validate-global
    ↓
[4] 修复并验证 first-grant-before-first-forward
    ↓
[5] Gate 3：TPC=4 / 4 requests / C1 smoke
    ↓
[6] Gate 通过后才做 TPC calibration
    ↓
[7] 统计真实 (batch_size,q,ctx) shape
    ↓
[8] 构建 >=95% 高频 round coverage 的 measured resource profile
    ↓
[9] D3 fixed TPC
    ↓
[10] D4 online Target-priority grant
    ↓
[11] 确认 D4 grant/ACK/mask 全链条正确
    ↓
[12] 补 dynamic-q 高频 profile
    ↓
[13] D5 完整 I1+I2
```

## 31.1 当前代码版本哪些实验可以跑

在出现以下错误的当前版本上：

```text
SpecStream refused an ungranted Draft forward: no TPC mask is active
```

允许继续：

```text
D0
D1
D2
创新点一 K0-K6（不走 I2 smctrl grant 的部分）
```

暂时停止：

```text
TPC calibration
D3
D4
D5
```

直到 Gate 3 通过。

## 31.2 D1 vs D2 的解释

D2 raw throughput 比 D1 低约 30%~50% 甚至更多，在同卡资源竞争下可能是正常的；但同时必须报告：

```text
D1 throughput/GPU = raw throughput / 2
D2 throughput/GPU = raw throughput / 1
```

以下现象不是“单卡天然慢”：

```text
P99 接近 5000/10000 ms
missing_drafts > 0
remote_draft_timeout 大量出现
Drafter 退出
ungranted Draft forward
D4/D5 TARGET_EXCLUSIVE≈100%
draft_step_ms=0
无 grant ACK
```

## 31.3 每个 I2 run 后统一故障扫描

```bash
grep -Eini \
  'ungranted Draft forward|no TPC mask|requires --spectre-draft-priority|out of memory|OOM|timeout|missing|fallback|no_measured_safe_profile|error|traceback' \
  "$SPECSTREAM_RESULT_ROOT/target.log" \
  "$SPECSTREAM_RESULT_ROOT/draft.log" \
  "$SPECSTREAM_RESULT_ROOT/benchmark.log" \
  | tail -n 160
```

D3-D5 还必须确认：

```text
grant_state 有效
grant_epoch 有效
draft_tpc_high > draft_tpc_low
draft_step_ms > 0
ACK 正常
无未授权 forward
```

这些条件不满足时，性能数据不得进入论文图。

---

# 32. GPU 显存/利用率监控

在正式性能 run 前另开一个终端：

```bash
nvidia-smi \
  --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw \
  --format=csv \
  -lms 200 \
  > "$RESULT_ROOT/gpu_monitor/current_run.csv"
```

实验结束 `Ctrl+C`。

正式文件名必须包含 variant/context/concurrency/rep，例如：

```text
D5_30k_c16_rep3.csv
```

同时 SpecStream profile 已经提供：

```text
gpu_kv_bytes
staging_bytes
cpu_history_bytes
```

因此论文显存结果最好同时展示：

```text
total GPU memory used
GPU KV bytes
staging bytes
CPU History bytes
```

---

# 33. 最终 confirmatory run：重复次数

screening 完成以后，只对主图用的配置做正式重复。

推荐：

```text
5 个独立 run
每个 run >= 1000 requests
```

每一个 rep：

1. 停止旧 Target/Draft；
2. 重新启动；
3. health check；
4. warmup；
5. 开启 GPU monitor；
6. 跑 1000 requests；
7. 保存 benchmark/profile/log/monitor；
8. 停止服务器；
9. 下一 rep。

示例标签：

```text
K6_16k_c16_rep1
K6_16k_c16_rep2
...
D5_30k_c16_rep5
```

不要把同一个 server 上连续跑 5 次当成 5 个独立系统重复。

---

# 34. 统一汇总全部 benchmark

```bash
python scripts/specstream/summarize_benchmarks.py \
  "$RESULT_ROOT/bench/*.jsonl" \
  > "$RESULT_ROOT/source_data/all_benchmarks.tsv"
```

主要列已经包含：

```text
request_throughput
output_throughput
mean_ttft_ms
p99_ttft_ms
mean_tpot_ms
p99_tpot_ms
mean_e2e_latency_ms
p99_e2e_latency_ms
mean_accept_length
error_count
```

---

# 35. 统一汇总全部 SpecStream profile

```bash
python scripts/specstream/summarize_specstream_profile.py \
  "$RESULT_ROOT/profiles/*.csv" \
  > "$RESULT_ROOT/source_data/all_profiles.tsv"
```

主要列：

```text
q_dist
mode_dist
coexec_dist
max_history
h2d_gib
h2d_ops
h2d_ops_per_accepted
h2d_mib_per_accepted
stream_attn_ops_per_accepted
mean_round_ms
p95_round_ms
max_draft_rtt_p95_ms
max_draft_timeout_rate
accepted_tokens
mean_cohort
max_cohort
fallback_rows
missing_drafts
```

---

# 36. 每跑完一个 Variant 必须做的完整性检查

```bash
python scripts/specstream/summarize_benchmarks.py "$RESULT_ROOT/bench/${VARIANT}*.jsonl"
python scripts/specstream/summarize_specstream_profile.py "$RESULT_ROOT/profiles/${VARIANT}*.csv"
```

检查：

### 对 K1–K6

```text
16K/30K 时 stream_rows > 0
cpu_history_bytes > 0
K3–K6 h2d_ops > 0
staging_bytes > 0
```

### 对 K5/K6

```text
q_dist 不能无解释地全为 1
fallback_rows 不应占绝大多数
```

### 对 K6

高并发时：

```text
mean_cohort > 1（至少在 cohort-friendly workload）
或明确解释为什么兼容率低
```

### 对 D4/D5

```text
grant_state 有有效记录
draft_tpc_high > draft_tpc_low
draft_step_ms > 0
无连续 error/timeout 风暴
```

如果这些条件不满足，不要继续画论文图，先排查机制是否真的执行。

---

# 37. 推荐论文主表最终只保留这些指标

## 正确性表

| Method | GSM8K Accuracy | LongBench v2 Accuracy |
|---|---:|---:|
| SGLang speculative | | |
| SPECTRE parallel | | |
| SpecStream-I1 | | |

旁边配一张 attention consistency heatmap 即可。

---

## 创新点一主性能表

建议主点：

```text
16K/C8
30K/C8
```

列：

```text
Output throughput
P99 TTFT
P99 TPOT
P99 E2E
Peak GPU memory
H2D MiB / accepted token
Mean accept length
```

---

## 创新点二主性能表

列：

```text
Physical GPUs
Output throughput
Output throughput / GPU
P99 TTFT
P99 TPOT
Peak GPU memory
Target slowdown
```

重点比较：

```text
D1 two-GPU SPECTRE
D2 uncontrolled same-GPU
D3 static TPC
D4 online I2
D5 full SpecStream
```

---

# 38. 推荐论文图的最小集合

为了避免实验结果变成“指标大拼盘”，每张图只回答一个问题。

### Fig. Accuracy

**主张：SpecStream 不牺牲模型质量。**

- a：GSM8K + LongBench v2 accuracy；
- b：attention consistency heatmap。

不再增加 token parity 等主图。

### Fig. KV streaming mechanism

**主张：多 Query 共用 History，并保持 bounded GPU staging。**

- q vs H2D MiB/accepted token；
- context vs GPU KV/staging memory；
- Nsight timeline。

### Fig. I1 end-to-end

**主张：各机制最终扩大长上下文 serving region。**

- K0–K6 throughput/P99；
- context×concurrency operating region。

### Fig. I2 safe coexecution

**主张：存在 Target slowdown 受控的 TPC Pareto 区。**

- Draft step latency vs Target slowdown；
- 5% slowdown threshold；
- online grant state timeline。

### Fig. Full system

**主张：SpecStream 降低 dedicated Drafter GPU resource tax。**

- D1–D5 raw throughput；
- throughput/GPU；
- P99；
- D4 vs D5 operating region。

---

# 39. 推荐严格执行顺序

不要跳步骤。

```text
[0] 固定代码 commit
    ↓
[1] 编译 cpp_zmq + pip editable install
    ↓
[2] pytest python/sglang/test/spectre_specstream
    ↓
[3] 检查 GSM8K / LongBench v2 / ShareGPT 本地数据
    ↓
[4] 正确性：ACC-SGL / ACC-SP / ACC-SS
    ├─ GSM8K Accuracy
    ├─ LongBench v2 Accuracy
    └─ Attention Heatmap
    ↓
[5] I1 screening：K0–K6 × {16K/C1,16K/C8,30K/C8}
    ↓
[6] I1 mechanism：fixed-q / prefetch / Cohort / Nsight
    ↓
[7] I1 full context×concurrency + P99 rate sweep
    ↓
[8] build + validate libsmctrl
    ↓
[9] I2 统一 q=4 + 显式 KV token pool，先跑 D1 / D2 的 16K/C8
    ↓
[10] 修复并验证 first-grant-before-first-forward 启动协议
    ↓
[11] Gate 3：TPC=4 / C1 / 4-request smoke，确认首个 Draft forward 前已有 grant/mask
    ↓
[12] Gate 3 通过后才做 q=4 / 16K/C8 TPC feasibility scan
    ↓
[13] 统计真实 (batch,q,context bucket) 分布并构建 measured resource profile
    ↓
[14] D2 → D3 → D4 验证同卡控制链；D4 高频 shape profile 覆盖目标 ≥95%
    ↓
[15] D1 vs D4 比较 raw throughput 与 throughput/GPU
    ↓
[16] 补齐 dynamic-q 高频 shape profile 后再跑 D5
    ↓
[17] D4 vs D5 验证 I1+I2 协同
    ↓
[18] 只对最终主点做 5×1000-request confirmatory runs
    ↓
[19] summarize_benchmarks + summarize_specstream_profile
    ↓
[20] 固化 source_data，再开始论文绘图
```

---

# 40. 最重要的 Stop 条件

出现下面任一情况，不要继续堆实验数据。

### 正确性

```text
SpecStream-I1 在两个数据集之一出现明显准确率下降
或 attention shadow 出现系统性大误差
```

先修 correctness。

### 创新点一

```text
16K/30K profile 没有 cpu_history/h2d
```

说明 streaming 没真正激活。

```text
K5/K6 q_dist 长期全部 q=1
```

说明结果不能归因于 multi-query speculative streaming。

### 创新点二

```text
libsmctrl validate-global 失败
```

停止。

```text
SpecStream refused an ungranted Draft forward: no TPC mask is active
```

停止。说明 first-grant-before-first-forward 尚未建立，不能继续 calibration / D3 / D4 / D5。

```text
SpecStream SM control requires --spectre-draft-priority
```

停止。不要通过删除 `--spectre-draft-priority` 绕过；该模式是 one-token grant 边界的强制前提。

```text
Gate 3 的 4-request smoke 未同时满足：
Target/Draft health 正常、无未授权 forward、TPC mask 非空、grant/TPC/draft_step 记录非空
```

停止。

```text
所有 TPC 点 Target slowdown >5%
```

先重新评估单卡共执行可行性，不要事后把阈值改成 10% 来“救”结果。

```text
D4/D5 没有 grant_state/draft_step/TPC 记录
```

说明实际 online grant 没运行，不能拿吞吐结果声称创新点二成立。

---

# 41. 最终结果目录建议

```text
results/paper/
├── accuracy/
│   ├── gsm8k_SGL_raw.jsonl
│   ├── gsm8k_SP_raw.jsonl
│   ├── gsm8k_SS_raw.jsonl
│   ├── longbench_v2_SGL.jsonl
│   ├── longbench_v2_SP.jsonl
│   └── longbench_v2_SS.jsonl
├── attention/
│   ├── attention_shadow.csv
│   ├── attention_shadow.diagnostics.jsonl
│   ├── fig_attention_consistency.pdf
│   └── fig_attention_consistency_source_data.csv
├── bench/
├── profiles/
├── resource_profiles/
│   ├── interference_samples.jsonl
│   └── specstream_*.json
├── smctrl/
├── nsys/
├── gpu_monitor/
├── logs/
└── source_data/
    ├── all_benchmarks.tsv
    ├── all_profiles.tsv
    └── ...
```

---

# 42. 最终实验叙事

如果结果支持预期，论文实验不要写成：

```text
K0 比 K1 快多少
K2 比 K1 快多少
K3 比 K2 快多少
```

而应形成如下证据链：

```text
模型准确率不下降
    ↓
Fused streaming attention 与 reference attention 高度一致
    ↓
一次 History streaming 可以服务一轮多个 Query
    ↓
GPU staging 不随 History 长度线性增长
    ↓
双缓冲/预取/dynamic q/Cohort 将机制收益转化为 serving 性能
    ↓
KV streaming 释放的显存与阶段 slack 使 Draft 可以与 Target 同卡
    ↓
Target-priority TPC grant 将 Target slowdown 控制在预算内
    ↓
完整 SpecStream 用更少 GPU 获得更高的单位 GPU 服务效率，
或在相同 GPU 预算下支持更长上下文/更高并发
```

这才是创新点一与创新点二完整的实验闭环。
