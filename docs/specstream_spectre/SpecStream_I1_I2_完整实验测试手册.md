# SpecStream 创新点一 + 创新点二完整实验测试手册

> **适用代码版本**：`LfLab-AI/SpecStream`，`main` 分支；本文档按 2026-08-25 的主分支实现编写。正式实验开始前请把 `git rev-parse HEAD` 写入结果目录，避免后续代码更新导致结果不可追溯。
>
> **实验范围**：仅覆盖当前论文的创新点一（verification-native bounded KV streaming + dynamic q + Chunk-Cohort）与创新点二（measurement-backed Target-priority single-GPU Draft–Verify co-execution）。不包含后续 TP/PP 创新点三。
>
> **正确性验证分三层**：
> 1. 以 Target-only autoregressive decoding（AR）作为语义 oracle，执行小规模逐 token 一致性门禁；
> 2. 在至少两套数据集上比较最终任务准确率，并做配对非劣效检验；
> 3. 使用 SpecStream 自带 `--specstream-shadow-attention` 生成 fused streaming 与 Torch FP32 reference attention 的一致性热力图。
>
> token parity/first-divergence 是低成本 Gate，不必占用论文主图，但不能省略：任务准确率相同不能证明推测执行没有实现错误。

## 审查结论（2026-08-25）

方案总体可行，但原稿不能直接作为正式实验 SOP。已修正的关键问题如下：

1. **正确性证据不足**：补入 Target-only AR oracle、逐 token 门禁、配对 bootstrap 非劣效检验；准确率只证明任务质量，不单独证明实现等价。
2. **LongBench v2 evaluator 与 prompt 不规范**：不再使用手写的宽松 A/B/C/D parser；统一复用仓库 `simple_eval_longbench_v2.py` 的 official-template formatter 与 answer extractor。超过模型实验上下文的样本不得静默截断，主文明确标为预先冻结的 `LongBench-v2-30K` 子集，而不是完整官方榜单成绩。
3. **K3/K4 消融混杂**：原命令中 K3 已经使用双缓冲，却把 K4 写成“增加双缓冲”。现将 K3 设为单缓冲；K3→K4 解释为“双缓冲 + 跨层预取”的流水化组合收益，若要分别归因则增加 2×2 微消融。
4. **profile 文件污染**：原稿让不同 context/concurrency/rep 追加到同一 `K*.csv`，无法恢复实验单元。现规定每次服务器重启使用唯一的 `variant_context_concurrency_rate_rep.csv`。
5. **P99 设计不足**：固定 `0.5–8 req/s` 可能全部处于欠载或过载；改为先测容量，再冻结共同 absolute offered-load 网格。正式 P99 每个独立 run 至少 2,000 个完成请求，5 个 run，并用分层 bootstrap 给出 95% CI。
6. **I2 slowdown 定义缺失且有选择偏倚**：区分校准 slowdown 与独立 confirmatory slowdown；TPC/profile 只用 calibration runs 选择，正式评价使用未参与选择的新 seed/run。
7. **Cohort 叙事越过代码边界**：当前实现只在已经进入同一 verify batch 的兼容请求间打包，不做跨请求 KV 内容去重，也没有真正的跨 batch admission wait；因此不把 delay 或 H2D bytes 减少作为主张。
8. **可复现性控制不足**：补入硬件/软件/NUMA/PCIe/模型与数据哈希、运行顺序随机化、重启和热稳态要求。

### 术语锁定

| 规范术语 | 本手册含义 | 禁止混用 |
|---|---|---|
| Target-only AR | 仅 Target 模型的普通自回归解码；正确性 oracle | “原生 speculative” |
| SGLang STANDALONE | SGLang 原生单进程、单 GPU speculative baseline | SPECTRE ordinary |
| SPECTRE parallel | Target/Drafter 跨进程流水并行 | “双卡串行” |
| Full-Restore-per-round | 每轮物化完整 CPU History 的 K1 对照 | “异步全量恢复”（除非 Nsight 证实） |
| bounded KV streaming | 有界 GPU staging 的 CPU History 分块流式验证 | “KV 压缩”或“KV 去重” |
| Chunk-Cohort | 同一 verify batch 内兼容请求的打包/融合执行 | 跨请求 KV 内容去重 |
| Target-priority TPC grant | measured profile + libsmctrl + one-token grant/ACK | 动态 MPS 百分比 |
| Full SpecStream | 创新点一 + 创新点二的 D5 | 单独的 I1 或 I2 |

---

## 0. 最终需要得到哪些结果

建议最终论文实验只围绕以下结果组织。

### 0.1 正确性结果

| 编号 | 结果 | 主比较 |
|---|---|---|
| C0 | 确定性逐 token Gate | Target-only AR vs SGLang STANDALONE / SPECTRE parallel / SpecStream-I1 |
| C1 | GSM8K 最终准确率 | Target-only AR / 原生 SGLang speculative / SPECTRE parallel / 完整 SpecStream-I1 |
| C2 | LongBench-v2-30K 最终准确率 | 同上；仅使用预注册的 8K–30K token 子集 |
| C3 | Attention 一致性热力图 | SpecStream fused streaming vs Torch FP32 online-softmax reference |

### 0.2 创新点一性能结果

| 编号 | 系统 | 目的 |
|---|---|---|
| K0 | GPU-resident SPECTRE parallel | 原始并行推测强基线 |
| K1 | CPU History + Full-Restore-per-round | 简单 CPU offload 对照 |
| K2 | bounded reference streaming | 参考实现；主要用于机制与正确性，不作为最终最快实现 |
| K3 | fused/grouped bounded streaming，单缓冲、无跨层预取 | 验证 fused/grouped streaming 本身 |
| K4 | K3 + 双缓冲 + layer prefetch | 验证流水化组合收益；单项归因使用 2×2 微消融 |
| K5 | K4 + dynamic q / safety-aware mode control | 验证动态 horizon 控制 |
| K6 | K5 + Chunk-Cohort | 完整创新点一 |

> **重要命名规则**：当前 K1 只称为 **Full-Restore-per-round**。不要在论文中预先称为“异步重叠优化的全量恢复”，除非 Nsight Systems 明确证明完整 History 恢复被独立计算有效隐藏。

### 0.3 创新点二性能结果

| 编号 | GPU 数 | 系统 | 目的 |
|---|---:|---|---|
| D-AR | 1 | Target-only AR | 非推测服务参考；量化 speculation 的净收益/代价 |
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

同时记录不可由 commit 恢复的实验环境：

```bash
{
  date -Iseconds
  uname -a
  lscpu
  free -h
  ulimit -l
  nvidia-smi -L
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
  nvidia-smi topo -m
  nvcc --version
  python --version
  python -m pip freeze
} > "results/paper/logs/environment_manifest.txt"
```

CPU History/H2D 对 NUMA 和 PCIe 拓扑敏感。先用 `nvidia-smi topo -m` 找到 Target GPU 的近端 NUMA node；K0–K6 的 Target 必须采用相同 CPU affinity/memory policy。若使用 `numactl`，把同一前缀加入所有 Target 命令并记录，例如：

```bash
export TARGET_NUMA_NODE=<按拓扑填写>
export TARGET_LAUNCH="numactl --cpunodebind=$TARGET_NUMA_NODE --membind=$TARGET_NUMA_NODE"
```

不得只对 SpecStream 绑定 NUMA 而不对 baseline 绑定。正式运行期间固定 persistence/power policy（若无权限则至少记录），避免与其他 GPU/CPU/PCIe 任务共享机器；每个 run 记录起止温度和功耗，出现明显 thermal throttling 的 run 作废并重跑。

`--specstream-cpu-memory-gb 128` 是预算而不是通用安全默认值。正式启动前确认物理 RAM、NUMA node 可用内存和 page-lock 限制足够，并监控 swap/major faults；若机器不满足，统一降低该预算并重新计算 capacity region。发生 swap 的 run 无效，因为它测到的是存储系统而非 CPU↔GPU KV streaming。

---

## 1.2 修改下面这一组路径

只需要在第一次实验时修改。

```bash
# ======================== 模型 ========================
export TARGET_MODEL=/common_data/model/Qwen2.5-7B-Instruct
export DRAFT_MODEL=/common_data/model/Qwen2.5-0.5B-Instruct
export MODEL_CONTEXT_LIMIT=32768

# ======================== 数据集 ========================
export DATA_ROOT=/common_data/dataset
export PREPARED_ROOT=$DATA_ROOT/specstream_prepared

# GSM8K 本地 test.jsonl
export GSM8K_TEST=$DATA_ROOT/gsm8k/test.jsonl

# LongBench v2：推荐本地 JSONL；也支持 parquet
export LONGBENCH_V2=$DATA_ROOT/LongBench-v2/longbench_v2.jsonl
export LONGBENCH_V2_30K=$PREPARED_ROOT/longbench_v2_8k_30k.jsonl

# ShareGPT V3：性能测试使用
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
export SINGLE_GPU_UUID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# ======================== 结果目录 ========================
export RESULT_ROOT=$REPO/results/paper
mkdir -p \
  $RESULT_ROOT/{accuracy,attention,bench,profiles,resource_profiles,smctrl,nsys,gpu_monitor,logs,source_data}

sha256sum "$TARGET_MODEL/config.json" "$DRAFT_MODEL/config.json" \
  "$GSM8K_TEST" "$LONGBENCH_V2" \
  > "$RESULT_ROOT/logs/input_sha256.txt"
```

检查模型和数据存在：

```bash
test -d "$TARGET_MODEL" || echo "ERROR: TARGET_MODEL not found"
test -d "$DRAFT_MODEL" || echo "ERROR: DRAFT_MODEL not found"
test -f "$GSM8K_TEST" || echo "ERROR: GSM8K_TEST not found"
test -f "$LONGBENCH_V2" || echo "ERROR: LONGBENCH_V2 not found"

python - <<'PY'
import os
from transformers import AutoConfig
need=int(os.environ['MODEL_CONTEXT_LIMIT'])
for key in ['TARGET_MODEL','DRAFT_MODEL']:
    cfg=AutoConfig.from_pretrained(os.environ[key],trust_remote_code=True)
    got=int(getattr(cfg,'max_position_embeddings',0) or 0)
    print(key,'max_position_embeddings=',got,'required=',need)
    if got and got < need:
        raise SystemExit(f'{key} context limit is insufficient')
PY
```

若启动参数覆盖模型配置，所有 Target/Drafter/baseline 必须统一追加 `--context-length "$MODEL_CONTEXT_LIMIT"`。30K 输入还需为 chat template 与 128 个输出 token 留余量；实测 prompt token 数超过上限的请求在进入实验前排除或降低统一上限，不能由不同服务各自截断。

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

# 3. 本地数据集预检查

## 3.1 GSM8K

```bash
python - <<'PY'
import json, os
p = os.environ["GSM8K_TEST"]
with open(p, encoding="utf-8") as f:
    rows=[json.loads(x) for x in f if x.strip()]
print("GSM8K examples =", len(rows))
print("keys =", rows[0].keys())
print("question =", rows[0]["question"][:100])
print("answer =", rows[0]["answer"][-100:])
PY
```

GSM8K 的仓库原生 evaluator `benchmark/gsm8k/bench_sglang.py` 可以直接读取本地 `--data-path`，不需要联网。

---

## 3.2 LongBench v2

先检查本地文件格式和字段：

```bash
python - <<'PY'
import os, json
p=os.environ["LONGBENCH_V2"]
if p.endswith(".parquet"):
    import pandas as pd
    row=pd.read_parquet(p).iloc[0].to_dict()
else:
    with open(p, encoding="utf-8") as f:
        row=json.loads(next(x for x in f if x.strip()))
print("keys =", sorted(row.keys()))
for k in ["context","question","choice_A","choice_B","choice_C","choice_D","answer"]:
    print(k, "=>", str(row.get(k, "<MISSING>"))[:160])
PY
```

本文后面的 LongBench v2 accuracy helper 假定标准字段包含：

```text
context
question
choice_A
choice_B
choice_C
choice_D
answer
```

如果你下载的数据字段名不同，只修改辅助脚本中的 `ANSWER_FIELD`/字段映射，不修改 SpecStream runtime。

LongBench v2 原始样本可远超本实验的 30K operating region。**不允许让服务端静默截断后仍称为完整 LongBench v2**。在查看任何方法结果之前，使用仓库官方 formatter 冻结同一个 8K–30K-token 子集：

```bash
mkdir -p "$PREPARED_ROOT"
python - <<'PY'
import json, os
from pathlib import Path
from transformers import AutoTokenizer
from sglang.test.simple_eval_longbench_v2 import format_longbench_v2_question

src=os.environ['LONGBENCH_V2']
if src.endswith('.parquet'):
    import pandas as pd
    rows=pd.read_parquet(src).to_dict(orient='records')
else:
    with open(src,encoding='utf-8') as f:
        rows=[json.loads(x) for x in f if x.strip()]
tok=AutoTokenizer.from_pretrained(os.environ['TARGET_MODEL'],trust_remote_code=True)
kept=[]
for source_index,row in enumerate(rows):
    prompt=format_longbench_v2_question(row)
    n=len(tok(prompt,add_special_tokens=True).input_ids)
    if 8192 <= n <= 30000:
        row=dict(row)
        row['_specstream_source_index']=source_index
        row['_specstream_prompt_tokens']=n
        kept.append(row)
out=Path(os.environ['LONGBENCH_V2_30K'])
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8') as f:
    for row in kept:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
print('original=',len(rows),'kept=',len(kept),'output=',out)
if not kept:
    raise SystemExit('No LongBench-v2 examples in the preregistered token range')
PY

sha256sum "$LONGBENCH_V2_30K" | tee "$RESULT_ROOT/logs/longbench_v2_30k.sha256"
```

论文和图表统一写作 **LongBench-v2-30K subset (8,192–30,000 prompt tokens)**，并报告保留样本数与各 domain/difficulty 构成。四个系统必须使用同一个冻结文件及其 SHA-256；不得按某个方法的成功请求重新筛选。

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
sha256sum "$SHAREGPT_JSON" | tee "$RESULT_ROOT/logs/sharegpt_merged.sha256"
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

# 5. 正确性实验：oracle、实现门禁与任务质量

## 5.1 主比较配置

正式准确率表保留四个方法；其中 ACC-AR 是语义 oracle，不是性能 baseline：

| 标签 | 方法 |
|---|---|
| ACC-AR | Target-only autoregressive decoding；关闭 speculative decoding |
| ACC-SGL | 原生 SGLang `STANDALONE` 单卡 speculative decoding |
| ACC-SP | SPECTRE parallel，GPU-resident KV，Target/Draft 分卡 |
| ACC-SS | 完整 SpecStream 创新点一：fused streaming + prefetch + dynamic q + Cohort |

所有准确率实验：

```text
temperature = 0
top_p = 1
```

同一数据集四个系统必须使用完全相同的样本 ID、prompt formatter、最大输出长度与 parser。所有请求保留逐样本输入 ID、原始输出、解析答案、错误状态和 token IDs。错误请求计为错误，不得从准确率分母删除。

预注册判据：

- **实现 Gate**：从两个数据集各冻结 256 个样本，ACC-SGL/ACC-SP/ACC-SS 与 ACC-AR 的 greedy token sequence 必须 100% 一致；任何 first divergence 均停止正式性能实验并诊断。若经确认差异仅来自已知后端数值非确定性，必须报告差异比例、logit margin 和后端，而不是悄悄放宽条件。
- **任务质量**：对每个 speculative 方法相对 ACC-AR 的逐样本正确性差值做 10,000 次 paired bootstrap，报告 accuracy difference 的 95% CI。非劣效 margin 预先固定为 **−1.0 percentage point**；只有 CI 下界高于 −1.0 pp 才可写“未降低任务准确率”。McNemar exact test 作为补充，不替代效应量和 CI。
- 不比较“哪个 speculative 方法准确率更高”，因为正确实现的推测解码不应改变 Target 分布；这里检验的是等价/非劣效，而非质量提升。

---

## 5.2 ACC-AR：Target-only autoregressive oracle

```bash
CUDA_VISIBLE_DEVICES="$TARGET_GPU" python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port "$TARGET_PORT" \
  --skip-server-warmup \
  --page-size 1 --attention-backend fa3 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  2>&1 | tee "$RESULT_ROOT/logs/ACC_AR_server.log"
```

## 5.3 ACC-SGL：原生 SGLang 单卡 speculative baseline

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

## 5.4 ACC-SP：SPECTRE parallel GPU-resident baseline

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

## 5.5 ACC-SS：完整创新点一

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

准确率完成后必须检查 ACC-SP/ACC-SS 的 missing Draft、timeout、fallback 和 error；若 speculative round 大量退化为 AR，准确率即使相同也不能作为所测试执行路径的证据。ACC-SS 的 `stream_rows>0`、`cpu_history_bytes>0`、`h2d_ops>0` 且 shadow/diagnostic 无异常后，正确性结果才有效。

---

# 6. GSM8K 准确率

## 6.1 计算本地测试集大小

```bash
export GSM8K_N=$(grep -cve '^$' "$GSM8K_TEST")
echo "GSM8K_N=$GSM8K_N"
```

## 6.2 对当前正在运行的系统执行

把 `METHOD` 分别设为 `AR`、`SGL`、`SP`、`SS`。这里固定 **0-shot**：仓库 `benchmark/gsm8k/bench_sglang.py` 若使用 `--num-shots 5`，会从同一 test 文件前五条构造 demonstrations；虽然可排除这五条后再评估，但与本文的实现等价性目标无关，0-shot 更干净且四个系统完全一致。

```bash
export METHOD=SS

python benchmark/gsm8k/bench_sglang.py \
  --host 127.0.0.1 --port "$TARGET_PORT" --backend srt \
  --data-path "$GSM8K_TEST" \
  --num-questions "$GSM8K_N" \
  --num-shots 0 \
  --parallel 32 \
  --max-new-tokens 512 \
  --temperature 0 --top-p 1 \
  --result-file "$RESULT_ROOT/accuracy/gsm8k_summary.jsonl" \
  --raw-result-file "$RESULT_ROOT/accuracy/gsm8k_${METHOD}_raw.jsonl" \
  | tee "$RESULT_ROOT/accuracy/gsm8k_${METHOD}.log"
```

依次执行：

```text
ACC-AR  -> METHOD=AR
ACC-SGL -> METHOD=SGL
ACC-SP  -> METHOD=SP
ACC-SS  -> METHOD=SS
```

每换方法必须重启对应服务。

最终主表只需要记录：

```text
Target-only AR      accuracy = ?
SGLang STANDALONE   accuracy = ?; difference vs AR [95% CI] = ?
SPECTRE parallel    accuracy = ?; difference vs AR [95% CI] = ?
SpecStream-I1       accuracy = ?; difference vs AR [95% CI] = ?
```

注意：该 evaluator 的数值提取属于仓库既有实现，四个方法必须共用；正式汇总前检查 invalid rate，并将请求异常和无法解析的答案计为错误。不得用每种方法各自“最有利”的答案抽取正则。

---

# 7. LongBench v2 准确率辅助脚本

仓库已经提供 `simple_eval_longbench_v2.py`。辅助脚本只能负责逐样本落盘，**必须导入该文件的 official-template formatter 和 answer extractor**，不能再实现一套宽松 parser。

创建目录：

```bash
mkdir -p scripts/specstream/paper_eval
```

创建：

```bash
cat > scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py <<'PY'
#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from transformers import AutoTokenizer
from sglang.test.simple_eval_longbench_v2 import (
    extract_longbench_v2_answer,
    format_longbench_v2_question,
)


def load_rows(path):
    if path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient='records')
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def prompt_of(x):
    return format_longbench_v2_question(x)


def normalize_label(x):
    s=str(x).strip().upper()
    return s if s in {'A','B','C','D'} else ''


def infer(base_url, model, prompt, max_new_tokens):
    payload={
        'model': model,
        'messages': [{'role':'user','content':prompt}],
        'temperature': 0,
        'top_p': 1,
        'max_tokens': max_new_tokens,
    }
    r=requests.post(base_url.rstrip('/') + '/v1/chat/completions', json=payload, timeout=1800)
    r.raise_for_status()
    obj=r.json()
    text=obj['choices'][0]['message']['content'] or ''
    usage=obj.get('usage') or {}
    return text, extract_longbench_v2_answer(text) or '', usage


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
        source_id=x.get('_specstream_source_index',i)
        selected.append((source_id,x,p,n))
        if args.limit and len(selected)>=args.limit:
            break

    print('selected examples =', len(selected))
    out=[None]*len(selected)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(infer,args.base_url,args.model,p,args.max_new_tokens):j
              for j,(_,_,p,_) in enumerate(selected)}
        for fut in as_completed(futs):
            j=futs[fut]
            i,x,p,n=selected[j]
            try:
                text,pred,usage=fut.result()
                gold=normalize_label(x[args.answer_field])
                out[j]={
                    'source_index': i,
                    'prompt_tokens': n,
                    'gold': gold,
                    'pred': pred,
                    'correct': pred==gold,
                    'prompt_sha256': hashlib.sha256(p.encode('utf-8')).hexdigest(),
                    'output': text,
                    'output_token_ids': tok(text,add_special_tokens=False).input_ids,
                    'usage': usage,
                    'error': '',
                }
            except Exception as e:
                out[j]={
                    'source_index': i,
                    'prompt_tokens': n,
                    'gold': normalize_label(x.get(args.answer_field,'')),
                    'pred': '',
                    'correct': False,
                    'prompt_sha256': hashlib.sha256(p.encode('utf-8')).hexdigest(),
                    'output': '',
                    'output_token_ids': [],
                    'error': repr(e),
                }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as f:
        for x in out:
            f.write(json.dumps(x,ensure_ascii=False)+'\n')

    ok=[x for x in out if not x['error']]
    # Errors remain in the denominator and are incorrect.
    acc=sum(x['correct'] for x in out)/len(out) if out else 0.0
    print(f'total={len(out)} success={len(ok)} errors={len(out)-len(ok)} accuracy={acc:.6f}')

if __name__=='__main__':
    main()
PY

chmod +x scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py
```

---

## 7.1 对四个系统分别运行 LongBench-v2-30K

先验证仓库 evaluator：

```bash
PYTHONPATH=python pytest -q python/sglang/test/longbench_v2/test_longbench_v2_eval.py
```

然后运行冻结子集：

```bash
export METHOD=SS

python scripts/specstream/paper_eval/eval_longbench_v2_accuracy.py \
  --dataset "$LONGBENCH_V2_30K" \
  --model "$TARGET_MODEL" \
  --base-url "$BASE_URL" \
  --workers 16 \
  --max-new-tokens 64 \
  --output "$RESULT_ROOT/accuracy/longbench_v2_${METHOD}.jsonl" \
  | tee "$RESULT_ROOT/accuracy/longbench_v2_${METHOD}.log"
```

依次执行：

```text
METHOD=AR
METHOD=SGL
METHOD=SP
METHOD=SS
```

必须检查四个输出文件具有相同 `source_index` 集合和相同行数；任何 API error 都保留在分母。若官方规则或仓库 formatter 后续更新，四个系统全部重跑；**不能四个系统使用不同 prompt/parser**。该结果是冻结子集的内部系统比较，不冒充完整 LongBench v2 leaderboard score。

---

# 8. Attention 一致性热力图

用户当前的正确性主图只需要一个 attention 一致性热力图即可。

## 8.1 推荐的热力图定义

不要只挑一个“看起来很好”的请求。

推荐从冻结的 LongBench-v2-30K manifest 中按 `source_index` 固定选择 **32 条 prompt token 数 ≥8192 的长上下文样本**，而不是按结果好坏挑选。

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
  --dataset "$LONGBENCH_V2_30K" \
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
    df=pd.DataFrame(mat,index=rids,columns=layers)
    df.to_csv(str(prefix)+'_source_data.csv',index_label='request_id')

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

论文正文报告热力图整体误差范围、`global max_abs`、relative-L2 分布和样本/轮次/层数。运行前冻结警戒线：live shadow 的 `global max_abs` 不应超过当前 fused-kernel 单元测试使用的 `atol=3e-2` 数量级；超过时停止并分析，不能靠调整色轴掩盖。最终正确性仍由逐 token Gate 决定，heatmap 只说明数值误差的层间/样本间结构。

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

该 `TARGET_COMMON` 是因果消融的 matched-runtime 配置。另做一次 K0 smoke，逐项恢复原生 SPECTRE 当前版本支持的 CUDA Graph/overlap/radix 优化；若稳定可用，则将最强配置记为 `K0-opt` 并在主性能比较中报告，同时保留 matched K0 用于 K1–K6 机制归因。不能为了“公平”而只展示被统一关闭生产优化的弱 baseline；也不能给 K0 开启会改变 prompt reuse 语义的 radix cache，却不给其他方法相同请求 trace。所有差异明确列在 configuration table。

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
  --specstream-gpu-reserve-mb 1024"
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
  --specstream-gpu-reserve-mb 1024"
```

## K3：fused/grouped streaming，无 layer prefetch

```bash
export K3_TARGET_CMD="$TARGET_COMMON \
  --specstream-enabled \
  --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-num-buffers 1 \
  --specstream-chunks-per-transfer 4 \
  --no-specstream-layer-prefetch \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-gpu-reserve-mb 1024"
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
  --specstream-gpu-reserve-mb 1024"
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
  --specstream-q-switch-threshold 0.08"
```

## K6：完整创新点一

```bash
export K6_TARGET_CMD="$K5_TARGET_CMD \
  --specstream-cohort-enabled \
  --specstream-max-cohort-size 8 \
  --specstream-max-cohort-delay-us 200"
```

K3 与 K4 的主比较把双缓冲和 layer prefetch 视为一个“异步流水化 bundle”。若论文要分别声称两者各自贡献，额外执行固定 K3 参数的 2×2 微消融：`num_buffers ∈ {1,2}` × `layer_prefetch ∈ {off,on}`；否则只能表述 bundle 的整体收益。

上述命令故意不写 `--specstream-profile-path`。每次 server 启动时追加唯一文件名：

```bash
export PROFILE_PATH="$RESULT_ROOT/profiles/${VARIANT}_${INPUT_LEN}_c${C}_rep${REP}.csv"
test ! -e "$PROFILE_PATH" || { echo "Refuse to append to existing profile: $PROFILE_PATH"; exit 2; }
TARGET_CMD_VAR="${VARIANT}_TARGET_CMD"
VARIANT_TARGET_CMD="${!TARGET_CMD_VAR}"
export SPECSTREAM_TARGET_CMD="$VARIANT_TARGET_CMD --specstream-profile-path '$PROFILE_PATH'"
```

`context/concurrency/rate/rep` 中任一项变化都必须更换 profile 文件。SpecStream profiler 使用 append 模式；复用文件会把多个实验单元混在一起，后续无法可靠恢复。

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
export PROFILE_PATH="$RESULT_ROOT/profiles/K4_16k_c8_rep1.csv"
test ! -e "$PROFILE_PATH" || { echo "Refuse to append: $PROFILE_PATH"; exit 2; }
export SPECSTREAM_TARGET_CMD="$K4_TARGET_CMD --specstream-profile-path '$PROFILE_PATH'"
export SPECSTREAM_DRAFT_CMD="$DRAFT_CMD_COMMON"
export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/K4_16k_c8_rep1"

export SPECSTREAM_BENCH_CMD="CASE_TAG=K4_16k_c8_rep1 \
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

**上面的循环只允许 screening。** 最终论文 confirmatory run 必须把每个 `(variant, context, concurrency, offered-load, rep)` 当作独立实验单元：使用唯一 profile/benchmark/monitor/log 文件，重新启动 Target/Drafter，并令 `SEED=REP`（或使用预先冻结的 seed 表）。

在每个 workload block 内随机化 variant 的执行顺序；推荐生成并保存 Latin-square/randomized schedule 到 `source_data/run_order.tsv`。不要固定按 K0→K6 或 D0→D5 执行，否则温度、时钟和后台系统漂移会与方法标签混杂。容量失败/OOM/timeout 记录为 `unsupported` 或 `failed` 并保留日志，不能从 operating-region 图中删除。

---

# 13. 创新点一：P99 的 Open-loop 到达率实验

P99 不能只用 `request-rate=inf`，也不能预先硬写 `0.5,1,2,4,8 req/s`：该网格可能对所有系统都欠载，或对较慢系统全部过载。

固定主 workload `context=16K, max_concurrency=32`。先对每个方法做短饱和 run 得到 sustainable request throughput；再以**该比较组最慢方法**的容量为锚点，冻结共同 absolute offered-load 网格，例如 `0.50×, 0.70×, 0.85×, 0.95×` 最慢容量，并可增加一个超过其容量的 overload 点。所有方法必须在相同 absolute req/s 下比较，不能各自在不同百分比负载下直接比较 P99。

示例（`COMMON_CAPACITY_RPS` 来自只用于定网格的 pilot，不进入正式结果）：

```bash
export VARIANT=K6
export COMMON_CAPACITY_RPS=<pilot 后冻结>

for FRACTION in 0.50 0.70 0.85 0.95 1.10; do
  RATE=$(python -c "print(float('$COMMON_CAPACITY_RPS')*float('$FRACTION'))")
  CASE_TAG="${VARIANT}_16k_r${RATE}_c32" \
  DATASET_NAME=random DATASET_PATH="$SHAREGPT_JSON" \
  INPUT_LEN=16384 OUTPUT_LEN=128 \
  NUM_PROMPTS=2000 REQUEST_RATE="$RATE" MAX_CONCURRENCY=32 \
  RANGE_RATIO=1 WARMUP_REQUESTS=8 SEED=1 \
  OUTPUT_DIR="$RESULT_ROOT/bench" \
  bash scripts/specstream/run_benchmark_case.sh
done
```

主延迟图报告 TTFT/TPOT/E2E 的 P50、P95、P99 和完成率，并增加 **goodput**：在预先冻结的 TTFT 与 TPOT SLO 下，每秒同时满足两项 SLO 的完成请求数。SLO 数值在 pilot 后、查看方法标签结果前冻结到 `source_data/preregistered_endpoints.yaml`；不得看到结果后移动阈值。可在 Extended Data 给出 SLO sensitivity grid。

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

这是当前 cost model 的结构性结果：`parallel=max(draft,verify)+rollback`，`ordinary=draft+verify`，在无额外约束时 parallel 通常不劣于 ordinary。因此 K5 的可检验主贡献是 q 分布随 I/O/acceptance/load 改变；ordinary 主要由 safety/backoff/TP constraint 触发。若需要声称正常运行中的主动 serial↔parallel 优化，必须先修改策略使两种 mode 存在真实 trade-off，再重新设计实验，不能从现代码的少量 fallback 行推出该结论。

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

当前 verifier 对**已经进入同一 scheduled verify batch** 的请求构造 cohort，并以 `now_ns=deadline_ns-1` 规划；`--specstream-max-cohort-delay-us` 目前不是跨 batch admission wait。正式消融固定该值，不把 delay sweep 或“等待更多请求形成 cohort”写成主机制。预期 H2D bytes 基本不因 cohort size 下降；应优先检验 `h2d_ops`、`stream_attn_ops`、kernel/event 数、round latency 和高并发吞吐。

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

创新点二正式实验必须在**实际测试 GPU + 实际 Driver/CUDA** 上重新验证。

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
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" make validate-global TPC_LOW=0 TPC_HIGH=4
```

如果失败：

```text
STOP
```

不要通过关掉 validator、硬编码 MASK_OFF 或改成 MPS 百分比来“绕过”。

记录库：

```bash
export SMCTRL_LIB=$REPO/csrc/specstream_smctrl/build/libsmctrl.so
ls -lh "$SMCTRL_LIB"
sha256sum "$SMCTRL_LIB" | tee "$RESULT_ROOT/smctrl/libsmctrl.sha256"
```

读取实际 TPC 数：

```bash
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
SGLANG_SPECSTREAM_SMCTRL_LIBRARY="$SMCTRL_LIB" \
python - <<'PY'
from sglang.srt.speculative.spectre.specstream.sm_controller import SMController
c=SMController(mask_scope='global')
print(c.total_tpcs)
PY
```

把结果写入：

```bash
export TOTAL_TPCS=<实际数值>
```

---

# 18. 创新点二 Gate 2：启动同卡 MPS 基础设施

MPS 不是动态调度器，只是让两个独立 CUDA 进程在同一 GPU 上稳定共执行。

```bash
export SPECSTREAM_GPU_UUID="$SINGLE_GPU_UUID"
source <(SPECSTREAM_GPU_UUID="$SINGLE_GPU_UUID" bash scripts/specstream/mps/start_mps.sh | grep '^export ')
```

如果你的 shell 不方便 `source <(...)`，直接：

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-${USER}
export CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-${USER}
```

检查：

```bash
echo get_server_list | nvidia-cuda-mps-control
```

实验全部结束后：

```bash
bash scripts/specstream/mps/stop_mps.sh
```

---

# 19. 创新点二：先做最小可行性 TPC 扫描

不要立即做完整 profile。

第一轮只测：

```text
input = 16K（profile key 以 CSV 实测 bucket 为准，通常为 ctx32k）
verification batch ≈ 8
q = 4
TPC = 2,4,6,8,12（只保留 <= TOTAL_TPCS 的值）
每点 3 次
```

目标：先确认是否存在：

```text
Target slowdown <= 5%
且 Draft step 能正常前进
```

若所有点都不满足，先停止创新点二大规模实验。

这里的 5% 指 **matched verification-forward slowdown**：

```text
slowdown = median(target_forward_ms | overlap, same shape/bs/q)
           / median(target_forward_ms | no-overlap, same shape/bs/q) - 1
```

每个 TPC 的 no-overlap 与 overlap run 使用相同请求 manifest、batch shape、q 和 seed，并交替/随机顺序执行。校准数据只用于选择 safe TPC 与构建 resource profile；D3/D4/D5 的正式 slowdown 必须在未参与选择的新 run 上重新估计，避免“用同一数据选最优点并报告最优点”的选择偏倚。

---

# 20. TPC calibration：固定 q=4 命令模板

q=4 必须：

```text
num_steps=3
num_draft_tokens=4
```

## 20.1 Drafter calibration 命令

```bash
export CAL_DRAFT_CMD="python -m sglang.launch_server \
  --model-path '$DRAFT_MODEL' --port $DRAFT_PORT \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role draft \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --spectre-draft-priority --spectre-max-draft-priority-steps 8 \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library '$SMCTRL_LIB' \
  --specstream-smctrl-mask-scope global \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port $ZMQ_PORT"
```

## 20.2 Target calibration 公共命令

```bash
export CAL_TARGET_COMMON="python -m sglang.launch_server \
  --model-path '$TARGET_MODEL' --port $TARGET_PORT \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode parallel --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port $ZMQ_PORT \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule"
```

---

# 21. 每个 TPC 要跑两类 profile

对于 `TPC=4` 示例。

## 21.1 Baseline：Target forward 不允许 overlap

```bash
export TPC=4
export REP=1
export BASE_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_base.csv"

export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC \
  --specstream-profile-path '$BASE_PROFILE'"
export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_CMD"
export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_base"
export SPECSTREAM_BENCH_CMD="CASE_TAG=cal_q4_tpc${TPC}_base \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=4 SEED=$REP OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

## 21.2 Overlap：允许 fixed-TPC Draft 在 Target forward 期间执行

```bash
export OVER_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_overlap.csv"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-profile-path '$OVER_PROFILE'"
export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_overlap"
export SPECSTREAM_BENCH_CMD="CASE_TAG=cal_q4_tpc${TPC}_overlap \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=4 SEED=$REP OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

---

# 22. 从 calibration CSV 提取一个 interference sample

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

如果出现：

```text
no positive ... samples for shape=...
```

说明该 run 的真实 verification batch 没形成 `batch_size=8`。先检查：

```bash
python - <<PY
import csv
p="$OVER_PROFILE"
rows=list(csv.DictReader(open(p)))
from collections import Counter
print("batch sizes:", Counter(r.get('batch_size') for r in rows))
print("q:", Counter(r.get('q') for r in rows))
print("contexts:", Counter(r.get('context_tokens') for r in rows))
PY
```

必要时增加并发/请求数，不能伪造一个不存在的 batch shape。

对所有 TPC × 3 repetitions 重复以上流程。

这里使用 `ctx32k` 不是笔误：代码的 bucket 上界为 `... 16k, 32k ...`，输入正好 16,384 tokens 后一进入 decode/verification，`context_tokens>16,384`，绝大多数 round 属于 `32k` bucket。必须从 CSV 的真实 `context_tokens`/`target_shape` 统计决定 profile key，不能用 workload 名称“16K”猜 bucket。若确实同时出现 16k 与 32k，两者分别建 entry。

---

# 23. 构建创新点二 resource profile

当 `interference_samples.jsonl` 中每一个 `(shape,bs,ctx,tpc)` 至少有 3 条独立重复后：

```bash
export RESOURCE_PROFILE="$RESULT_ROOT/resource_profiles/specstream_q4_input16k_ctx32k_bs8.json"

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

筛选 safe TPC：

```bash
python - <<'PY'
import json, os
p=os.environ['RESOURCE_PROFILE']
x=json.load(open(p))
for e in x['entries']:
    safe=e['target_slowdown'] <= 0.05
    print(e['target_shape'], 'TPC=',e['draft_tpcs'],
          'draft_step_ms=',round(e['draft_step_ms'],4),
          'target_slowdown=',round(e['target_slowdown'],4),
          'SAFE' if safe else 'UNSAFE')
PY
```

这一表直接用于论文的 Target slowdown–Draft latency Pareto 图。

---

# 24. 论文正式 I2 profile 如何扩展

不要一次构建所有笛卡尔积。

推荐顺序：

### Stage A：可行性

```text
16K input / observed ctx32k / bs≈8 / q=4 / TPC sweep
```

### Stage B：论文主运行区域

扩展到：

```text
16K input / observed ctx32k / bs≈8,16 / q=4
30K input / observed ctx32k / bs≈8,16 / q=4
```

### Stage C：完整 D5 dynamic q

先看 K5/K6 的 `q_dist`，只优先补齐经常出现的 q，例如：

```text
q=2,4,6
```

再扩展 q=8。

未校准 shape 在 current code 中应该 fail closed，不允许随意做 nearest-neighbor 插值。因此最终主文只对**有实测 resource profile 覆盖的 workload**作强结论。

---

# 25. 创新点二 D-AR 与 D0

`D-AR` 使用 ACC-AR 启动参数，运行与 D0 相同的 context×concurrency 网格。它不是创新点二的直接竞争 baseline，但用于回答“推测执行是否比普通 Target-only 服务真正更快”，并防止所有 speculative 方案都退化却只在彼此之间比较。

## 25.1 D0：原生 SGLang 单卡 speculative

I2 的 isolated causal comparison 固定 `q=4`。D0 启动可复用 ACC-SGL，但必须把 `--speculative-num-steps 4/--speculative-num-draft-tokens 5` 改为 `3/4`。D1–D4 也全部固定 q=4；否则 q=5 的双卡/无控制 baseline 与 q=4 的 online policy 比较会混入 speculation horizon 差异。

D0 同时保留两种报告口径：`D0-opt` 使用当前 SGLang 对 STANDALONE 支持的最强生产配置（例如 CUDA Graph 若该版本/模型确实支持）；`D0-matched` 关闭 CUDA Graph/overlap/radix，与 SpecStream 做机制受控比较。主文至少展示 D0-opt，不能只用人为关闭优化的弱 baseline；D0-matched 放在 ablation/Extended Data 解释差异来自系统机制还是通用 runtime optimization。任何“支持/不支持”先以 smoke test 证明并记录日志。

为 D1/D2 定义不含 smctrl 的 q=4 SPECTRE 命令：

```bash
export I2_Q4_TARGET_CMD="${TARGET_COMMON/--speculative-num-steps 4/--speculative-num-steps 3}"
export I2_Q4_TARGET_CMD="${I2_Q4_TARGET_CMD/--speculative-num-draft-tokens 5/--speculative-num-draft-tokens 4}"
export I2_Q4_DRAFT_CMD="${DRAFT_CMD_COMMON/--speculative-num-steps 4/--speculative-num-steps 3}"
export I2_Q4_DRAFT_CMD="${I2_Q4_DRAFT_CMD/--speculative-num-draft-tokens 5/--speculative-num-draft-tokens 4}"
```

性能测试：

```bash
for INPUT_LEN in 4096 16384 30000; do
  for C in 1 4 8 16 32; do
    CASE_TAG="D0_${INPUT_LEN}_c${C}" \
    DATASET_NAME=random DATASET_PATH="$SHAREGPT_JSON" \
    INPUT_LEN="$INPUT_LEN" OUTPUT_LEN=128 \
    NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY="$C" \
    RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 \
    OUTPUT_DIR="$RESULT_ROOT/bench" \
    bash scripts/specstream/run_benchmark_case.sh
  done
done
```

---

# 26. D1：SPECTRE 双卡并行

使用 `I2_Q4_TARGET_CMD` + `I2_Q4_DRAFT_CMD` 的 dedicated Target/Draft 配置。K0（q=5）仍用于 I1，不直接拿来替代该 I2 baseline。

物理 GPU 数记录为：

```text
2
```

最终报告：

```text
raw output throughput
P99 TTFT
P99 TPOT
P99 E2E
throughput/GPU = output throughput / 2
```

---

# 27. D2：同卡 SPECTRE parallel，无资源控制

先启动 MPS，然后把 Target 和 Drafter 都绑定到：

```text
$SINGLE_GPU_UUID
```

Target KV 与 K0 相同保持 GPU-resident。为取得逐轮 `target_forward_ms`，D2 可加入 `--specstream-profile-only --specstream-profile-path <unique.csv>`，但**不加入** `--specstream-smctrl-enabled`、dynamic q 或任何 grant 参数；因此它仍是 uncontrolled same-GPU baseline。主文需明确 profile-only 这里只是 instrumentation，并确认 K0 与“profile-only、无控制”的 dedicated-GPU smoke 在吞吐/延迟上无实质偏差。

用 existing wrapper：

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export PROFILE_PATH="$RESULT_ROOT/profiles/D2_16k_c8_rep1.csv"
export SPECSTREAM_TARGET_CMD="$I2_Q4_TARGET_CMD --specstream-profile-only --specstream-profile-path '$PROFILE_PATH'"
export SPECSTREAM_DRAFT_CMD="$I2_Q4_DRAFT_CMD"
export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D2_16k_c8"
export SPECSTREAM_BENCH_CMD="CASE_TAG=D2_16k_c8 \
DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 \
REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 \
WARMUP_REQUESTS=4 SEED=1 OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

D2 是非常重要的负面对照：用于证明创新点二不是“两个进程放同一 GPU 就行”。

---

# 28. D3：最佳固定 TPC

从 calibration 中选择：

```text
Target slowdown <=5%
且 Draft step latency 最低
```

的固定 TPC，例如：

```bash
export BEST_TPC=4
```

D3 Target 使用 calibration fixed-TPC overlap：

```bash
export D3_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $BEST_TPC \
  --specstream-smctrl-calibration-allow-overlap"
```

Drafter 使用 `CAL_DRAFT_CMD`。

注意：`calibration-allow-overlap` 在这里仅作为**固定资源 baseline**，不是最终 SpecStream online policy。

---

# 29. D4：创新点二 online Target-priority TPC grant

Target KV 保持原生 GPU-resident，因此必须使用：

```text
--specstream-profile-only
```

而不是 `--specstream-enabled`。

Target：

```bash
export D4_TARGET_CMD="python -m sglang.launch_server \
  --model-path '$TARGET_MODEL' --port $TARGET_PORT \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode parallel --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-grant-token-quantum 1 \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --specstream-coexec-resource-profile-path '$RESOURCE_PROFILE' \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port $ZMQ_PORT \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule"
```

Drafter：

```bash
export D4_DRAFT_CMD="$CAL_DRAFT_CMD"
```

重点 profile 字段：

```text
grant_state
grant_epoch
grant_wait_ms
draft_step_ms
draft_tpc_low/high
target_forward_ms
fallback
```

在线系统至少应出现 `TARGET_EXCLUSIVE`。有实测 safe entry 且 predicted slack 足够时应出现 `SLACK_FILL`；`DRAFT_CATCHUP` 只在 Target 等待 Draft 的压力场景出现，不要求每个正常 workload 三种状态齐全：

```text
TARGET_EXCLUSIVE
SLACK_FILL
DRAFT_CATCHUP
```

且 ACK 到达前不应连续发多个 token grant。若长期只有 `TARGET_EXCLUSIVE`，结果只能说明 fail-closed 正常，不能声称 co-execution 带来性能收益；先检查 profile key 覆盖、slack 和 guard。

---

# 30. D5：创新点一 + 创新点二完整系统

D5 在 D4 基础上把 `profile-only` 替换成真正的 tiered KV：

```bash
export D5_TARGET_CMD="python -m sglang.launch_server \
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
  --specstream-smctrl-enabled \
  --specstream-grant-token-quantum 1 \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --specstream-coexec-resource-profile-path '$RESOURCE_PROFILE' \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port $ZMQ_PORT \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule"
```

Drafter 使用带 libsmctrl/global mask 的命令。

> 若当前 resource profile 只覆盖 q=4，D5 dynamic q 在其它 q/shape 上可能 fail closed 到 Target-exclusive。这不是运行错误。正式 full-system 结果前，应按 K5/K6 实际 `q_dist` 补齐主要 q/shape 的校准数据。

D3/D4/D5 与 K 系列一样，在每个独立 run 启动前追加唯一 `--specstream-profile-path "$PROFILE_PATH"`；不得使用固定的 `D3.csv`/`D4.csv`/`D5.csv` 跨 workload 追加。D3 的 `BEST_TPC` 和 D4/D5 的 `RESOURCE_PROFILE` 在 confirmatory 阶段锁定，不得针对每个正式结果点事后重新挑选。

---

# 31. 创新点二的主性能矩阵

先 screening：

```text
16K/C8
16K/C16
30K/C8
30K/C16
```

比较：

```text
D-AR D0 D1 D2 D3 D4 D5
```

通过后，再扩展：

```text
Context = 4K,16K,30K
Concurrency = 1,4,8,16,32
```

其中最重要的三组因果比较：

### D2 vs D3 vs D4

证明：

```text
uncontrolled same-GPU
 -> static safe TPC
 -> online target-priority grant
```

### D1 vs D4

证明：

```text
2 GPUs absolute performance
vs
1 GPU resource efficiency
```

### D4 vs D5

证明：

```text
GPU-resident KV single-GPU coexec
vs
streaming-KV single-GPU coexec
```

即创新点一是否扩大创新点二的 context/concurrency 可运行区域。

I2 结果同时报告两种不可混用的 slowdown：

1. **verification-forward slowdown**：相对 matched no-overlap calibration 的 `target_forward_ms` 中位数变化，用于验证 5% controller budget；只比较相同 `(target_shape, bs, q, context bucket)`。
2. **service-level change**：在相同 absolute offered load 下，TTFT/TPOT/E2E/throughput 相对 D1 或 D0 的变化，用于评价用户可见性能。

`tp_target_slowdown` 是 TP straggler 监控字段，单 GPU I2 中可能为零，不能直接充当 observed verification-forward slowdown。正式 Source Data 需从独立 D2–D5 profile 的 `target_forward_ms` 与 matched baseline 重新计算，并给出 95% CI。

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
吞吐主点：每个 run >= 1000 completed requests
P99 主点：每个 run >= 2000 completed requests（1000 仅约 10 个 P99 尾部样本，只可作 pilot）
```

每一个 rep：

1. 停止旧 Target/Draft；
2. 重新启动；
3. health check；
4. warmup；
5. 开启 GPU monitor；
6. 跑预注册数量的请求，并验证 completed 数达到门槛；
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

统计规则：

- **run 是系统重复单位**，单个 request 是 run 内观测，不能把 10,000 个请求伪装成 10,000 个独立硬件重复。
- 每个 endpoint 报告 5 个 run 的中位数与 IQR；差值/比值用以 run 为 cluster、request 为 cluster 内样本的 hierarchical bootstrap（10,000 次）给出 95% CI。
- P99 同时给出每个 run 的 P99 和分层 bootstrap CI；不把所有 run 简单拼接后只报一个无不确定性的 P99。
- 正式 run 使用未参与参数选择的 seed；发生基础设施故障时按预先定义的 failure rule 重跑，并保留失败日志。性能较差但成功完成的 run 不得删除。
- 对多个 context/concurrency 的探索性结果控制叙事，不逐格宣称显著；主文只对预注册 primary workloads/endpoints 作 confirmatory claim，其余标为 exploratory/Extended Data。

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

该脚本只做快速表格化，不负责论文统计。它当前不计算 hierarchical bootstrap、goodput、完成率 CI、运行顺序或 physical-GPU 归一化；正式分析必须读取 `bench_serving --output-details` 保存的逐请求数组，并与 `run_manifest.tsv` 按唯一 run ID 合并。`throughput/GPU` 只在记录实际 physical GPU 数后计算；不能从 `CUDA_VISIBLE_DEVICES` 的逻辑编号猜测。

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

| Method | GSM8K Accuracy | Δ vs AR [95% CI] | LongBench-v2-30K Accuracy | Δ vs AR [95% CI] | Token parity Gate |
|---|---:|---:|---:|---:|---:|
| Target-only AR | | reference | | reference | reference |
| SGLang STANDALONE | | | | | 100% required |
| SPECTRE parallel | | | | | 100% required |
| SpecStream-I1 | | | | | 100% required |

旁边配一张 attention consistency heatmap。表注给出 `n`、错误计分规则、paired bootstrap=10,000、非劣效 margin=−1.0 pp；LongBench 标明它是冻结的 8K–30K-token 子集。

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
Goodput at frozen TTFT/TPOT SLO
P99 TTFT
P99 TPOT
P99 E2E
Peak GPU memory
H2D MiB / accepted token
Mean accept length
```

所有主 endpoint 给出 95% CI，并在表注写 `5 independent server restarts` 与每个 run 的 completed-request 数。

---

## 创新点二主性能表

列：

```text
Physical GPUs
Output throughput
Output throughput / GPU
Goodput / GPU
P99 TTFT
P99 TPOT
P99 E2E
Peak GPU memory
Observed verification-forward slowdown [95% CI]
```

重点比较：

```text
D-AR Target-only
D0 SGLang STANDALONE
D1 two-GPU SPECTRE
D2 uncontrolled same-GPU
D3 static TPC
D4 online I2
D5 full SpecStream
```

---

# 38. 推荐论文图的最小集合

为了避免实验结果变成“指标大拼盘”，每张图只回答一个问题。

统一绘图规范：图先写一句可证伪 conclusion，再决定 panel；主比较显示独立 run 的点和 95% CI，不只画柱高；颜色采用色觉友好且全篇方法映射固定，不能在不同图中给 K6/D5 换颜色；轴标包含单位，P99 使用线性/对数轴时明确标注；PDF/SVG 为矢量主文件，热力图另存高分辨率 raster；每个 panel 对应一份带 run ID、`n`、统计量和原始值的 Source Data。图注写清样本数、独立重复数、误差线定义、检验/CI 方法和 SLO。

### Fig. Accuracy

**主张：SpecStream 不牺牲模型质量。**

- a：GSM8K + LongBench-v2-30K accuracy difference vs AR（95% CI 与 −1 pp margin）；
- b：attention consistency heatmap。

不再增加 token parity 等主图。

### Fig. KV streaming mechanism

**主张：多 Query 共用 History，并保持 bounded GPU staging。**

- q vs H2D MiB/accepted token；
- context vs GPU KV/staging memory；
- Nsight timeline。

### Fig. I1 end-to-end

**主张：各机制最终扩大长上下文 serving region。**

- K0–K6 throughput/goodput/P99（主文只选能支撑结论的最小指标组合）；
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
[3] 检查 GSM8K / 冻结 LongBench-v2-30K manifest / ShareGPT 本地数据
    ↓
[4] 正确性：ACC-AR / ACC-SGL / ACC-SP / ACC-SS
    ├─ ACC-AR oracle + 256-sample token parity Gate
    ├─ GSM8K Accuracy + paired non-inferiority
    ├─ LongBench-v2-30K Accuracy + paired non-inferiority
    └─ Attention Heatmap
    ↓
[5] 验证 K0-opt/D0-opt；I1 screening：K0–K6 × {16K/C1,16K/C8,30K/C8}
    ↓
[6] I1 mechanism：fixed-q / prefetch / Cohort / Nsight
    ↓
[7] I1 full context×concurrency + P99 rate sweep
    ↓
[8] build + validate libsmctrl
    ↓
[9] I2 q=4 / 16K / bs≈8 TPC feasibility scan
    ↓
[10] Build measured resource profile
    ↓
[11] D2 → D3 → D4 验证同卡控制链
    ↓
[12] D1 vs D4 资源效率
    ↓
[13] D4 vs D5 验证 I1+I2 协同
    ↓
[14] 只对最终主点做 5 个独立 confirmatory runs（吞吐 ≥1000、P99 ≥2000 completed requests/run）
    ↓
[15] summarize_benchmarks + summarize_specstream_profile
    ↓
[16] 固化 source_data，再开始论文绘图
```

---

# 40. 最重要的 Stop 条件

出现下面任一情况，不要继续堆实验数据。

### 正确性

```text
任一 speculative 方法相对 Target-only AR 出现 token first-divergence
或准确率差值 95% CI 下界 <= -1.0 pp
或 attention shadow global max_abs 超过预注册警戒线/出现系统性层间大误差
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
    ├── preregistered_endpoints.yaml
    ├── run_order.tsv
    ├── run_manifest.tsv
    ├── all_benchmarks.tsv
    ├── all_profiles.tsv
    ├── figure_panel_source_data/
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
greedy token parity 通过，任务准确率相对 AR 非劣
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
