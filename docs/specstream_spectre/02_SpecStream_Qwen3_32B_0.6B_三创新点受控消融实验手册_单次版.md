# 02 · Qwen3-32B / Qwen3-0.6B 三创新点受控消融（最终双 TP2 对照版）

> 更新：2026-09-09；服务器 `/root/lifei/SpecStream`；2 × A800 80 GiB。  
> **创新点三现在两卡即可完整运行主消融：同一 SpecStream 实现，Draft TP1/TP2 × serial/auto。**  
> 创新点一/二统一 Target TP2 + Draft TP2，使用同一对 GPU；ordinary 定义保留，K4/K5 不做 Target/Draft 并行。  
> 仅做创新点三：先完成 [01 手册](01_SpecStream_Qwen3_32B_0.6B_公开数据集端到端实验手册_单次版.md) 第 2–5 节，然后直接跳到本手册第 7 节，不必重跑 I1/I2。

## 0. 配置与结论边界

工程 P0/P1/P2 与论文创新点一/二/三不是同一编号。本版各组共享当前修复、split-KV、后台泵、History=8192、buffers=2、catchup quantum=1；只切换表内的变量。不要同时更改 TPC、缓存、prefetch 深度和 q，再把总收益归给 Draft 并行。

| 试验 | Target / Draft TP | Target cap / Draft cap | 并发 |
|---|---|---|---|
| I1/I2 | 2 / 2，ordinary | 65536 / 516224 | I1=1；I2=1/4/8/16 |
| I3 最终部署对照 | 2 / 2，serial 与 auto | 131072 / 196608 | LongBench=4 |
| I3 固定 q 机制对照 | 2 / 1 或 2，serial 与 auto | 131072 / 196608 | 默认 4；可扩展 1/4 |

两套 cap 分别服务于 I2 并发 16 的完整 Draft KV 与公共/I3 配置。**不能跨 I2 和 I3 的不同 cap 直接算创新点三收益。** 第 7 节入口显式恢复 01 的固定配置，防止继承本手册 I1/I2 环境。Draft TP2 的 token cap 是全局序列槽数，各 rank 存本 rank 的 KV 分片，不能理解为两份独立完整上下文。

`auto` 会在不适合重叠时退回多 token ordinary；`serial` 只禁用 Target/Draft 的计算并行，不关闭 Target 内 H2D/attention 的流水线。成功跑完、成功 SLACK_FILL、整体加速分别判定。若 auto 全部 ordinary，应报告机制未激活，不能把它标成“已实现并行加速”。

## 1. 消融定义

创新点一验证 Sealed History 是提交边界：只有不会被 speculative rollback 触及的 committed 前缀才可 D2H；CPU History 不需要版本或撤销协议；GPU History 只是可淘汰的只读副本，不参与提交和回滚。

创新点二使用严格累积的 K1–K5：

| ID | 唯一累计变化 |
|---|---|
| `K1` | CPU Sealed History + Full-Restore；单缓冲；严格 serialized H2D；固定 q=4 |
| `K2` | K1 改为 grouped bounded streaming；仍严格 serialized H2D |
| `K3` | K2 + 独立 copy stream、双缓冲和跨层预取，允许 copy/compute overlap |
| `K4` | K3 + 基于各 q 实测 Draft RTT 的 dynamic-q ordinary controller；候选 `{2,4,6,8}` |
| `K5` | K4 + Chunk-Cohort；最大 8 请求、最大等待 200 us |

创新点三改用第 7 节 S1/P1/S2/P2 及 D2S/D2P。旧 A/B/C 不再是本版主消融；专用第三卡 Draft 的硬件效率研究可另列补充，不能混入同硬件主表。

## 2. I1/I2 环境与数据 Gate

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
export TARGET_GPUS=0,1 TARGET_TP_SIZE=2 TARGET_GPU=0
export COLOCATED_GPU=0 COLOCATED_TP_RANK=0
export DRAFT_GPU=0
export TARGET_PORT=30000 DRAFT_PORT=30001 ZMQ_PORT=5557
export SERVER_CONTEXT_LEN=40960 FINAL_DRAFT_TPCS=34

export QWEN3_DATA_ROOT=$REPO/specstream_prepared/qwen3_offline
export LONGBENCH_QWEN3=$QWEN3_DATA_ROOT/longbench_v2_qwen3_8b_8k32k_sharegpt.json

export SPECSTREAM_TARGET_MEM_FRACTION=0.55
export SPECSTREAM_DRAFT_MEM_FRACTION=0.80
export SPECSTREAM_TARGET_MAX_TOTAL_TOKENS=65536
export SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=516224
export SPECSTREAM_TARGET_MIN_KV_TOKENS=65536
export SPECSTREAM_DRAFT_MIN_KV_TOKENS=516224
export SPECSTREAM_PREFILL_MAX_REQUESTS=1
export SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192
export SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0
export SPECSTREAM_REQUIRE_SLACK_FILL=0
export SPECSTREAM_SPLIT_KV=auto SPECSTREAM_BACKGROUND_GRANT_PUMP=1
export SPECSTREAM_NUM_BUFFERS=2 SPECSTREAM_GRANT_TOKEN_QUANTUM=1
export SPECSTREAM_DRAFT_TP_SIZE=2 SPECSTREAM_OVERLAP_MODE=serial SPECSTREAM_FIXED_Q=0
export PYTHONUNBUFFERED=1 TQDM_MININTERVAL=0.5
unset SPECSTREAM_DRY_RUN CUDA_VISIBLE_DEVICES REQUIRE_SEPARATE_DRAFT_GPU
```

数据无需重处理，理由和指纹 Gate 与 01 手册第 3 节完全相同：当前 32B/0.6B tokenizer 映射、特殊 token、non-thinking template 与冻结 manifest 一致。消融只使用 random-ids 和创新点一的冻结 LongBench-v2；换模型本身不要求重建文本。若 01 的 `QWEN3_32B_DATA_REUSE_GATE` 失败，必须先按 01 第 3.2 节重生成，不能继续。

Target cap 保持 65536，Draft cap 统一设为 516224。Draft TP2 将每层 KV head 和权重分摊至两张卡；516224 是全局序列 token 槽数，不是每张卡都保存完整模型的 KV，也不是两卡各容纳不同的请求。I2 入口会强制 `SPECSTREAM_DRAFT_TP_SIZE=$TARGET_TP_SIZE`，即使旧 shell 残留 TP1 也不会误启动单卡 Draft。

16 条 32000-token 请求的容量下限为 `16 × (32000 + 256 + 8) = 516224`。保留实际容量 Gate：两卡均参与计算不意味着物理显存必然够用；若启动后 Draft 实际 cap 小于 516224，仍须停止并统一重新标定，不能降低 MIN 值绕过。Target Full-Restore、运行临时空间和批次调度也会限制实际 GPU batch；客户端 c16 不保证每轮 GPU batch=16。

当前主手册明确使用 History=8192，便于与此前缓存预算比较；服务器用户设置的脚本默认 History=0 保留，不擅自覆盖。若要复现刚才的零缓存压力矩阵，将本手册两处显式 `SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192` 都改为 0，全组一致，并报告为单独配置。切换 Draft TP 后必须从新目录完整跑 K1–K5，不能混入旧 TP1 cell。

## 3. 前置 Gate 与 TP2 命令 Gate

```bash
export PREFLIGHT_ROOT=$REPO/results/${MODEL_TAG}_ablation_preflight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PREFLIGHT_ROOT"
(
set -euo pipefail
for f in \
  "$TARGET_MODEL/config.json" "$DRAFT_MODEL/config.json" "$LONGBENCH_QWEN3" \
  scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  scripts/specstream/paper_eval/qwen3/run_public_once.sh \
  scripts/specstream/paper_eval/qwen3/run_i2_k1_k5_once.sh \
  scripts/specstream/paper_eval/qwen3/analyze_sealed_history.py \
  scripts/specstream/paper_eval/qwen3/validate_online_attention_heatmap.py \
  scripts/specstream/paper_eval/qwen3/analyze_grant_events.py; do
  test -s "$f" || { echo "ERROR: missing $f" >&2; false; }
done

bash -n scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh
bash -n scripts/specstream/paper_eval/qwen3/run_public_once.sh
bash -n scripts/specstream/paper_eval/qwen3/run_i2_k1_k5_once.sh

bash scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  2>&1 | tee "$PREFLIGHT_ROOT/console.log"
grep -q 'QWEN3_PUBLIC_PREFLIGHT=PASS' "$PREFLIGHT_ROOT/console.log"
)
# 仅在上一块 PASS 后继续。
source "$PREFLIGHT_ROOT/runtime_env.sh"

```

不启动模型的命令 Gate（两个角色均 TP2，K1–K5 仍 ordinary）：

```bash
export DRY_ROOT=$PREFLIGHT_ROOT/ablation_dryrun
(
set -euo pipefail
for method in I1_GPU_ONLY I1_SEALED_HISTORY K1 K2 K3 K4 K5; do
  METHOD=$method DATASET_TAG=dry DATASET_NAME=random-ids \
  INPUT_LEN=32000 NUM_PROMPTS=1 OUTPUT_LEN=256 MAX_CONCURRENCY=8 \
  WARMUP_REQUESTS=0 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$DRY_ROOT" \
  SPECSTREAM_DRY_RUN=1 \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh
done

for config in "$DRY_ROOT"/logs/*/config.env; do
  grep -q '^TARGET_TP_SIZE=2$' "$config"
  grep -q '^COLOCATED_TP_RANK=0$' "$config"
  grep -q '^DRAFT_TP_SIZE=2$' "$config"
  grep -q '^TARGET_MAX_TOTAL_TOKENS=65536$' "$config"
  grep -q '^DRAFT_MAX_TOTAL_TOKENS=516224$' "$config"
done

for method in K1 K2; do
  f="$DRY_ROOT/logs/${method}_dry_c8/config.env"
  grep -q '^SERIALIZE_H2D=1$' "$f"
  grep -q '^H2D_EXECUTION=serialized$' "$f"
done
for method in K3 K4 K5; do
  f="$DRY_ROOT/logs/${method}_dry_c8/config.env"
  grep -q '^SERIALIZE_H2D=0$' "$f"
  grep -q '^H2D_EXECUTION=async_copy_stream$' "$f"
done
echo QWEN3_32B_ABLATION_DRY_GATE=PASS
)
```

## 4. 一次性 TP2 + Draft smoke

smoke 使用正式 K3 路径和正式 exact caps，以 12K 输入真实触发 Sealed History、GPU History 与 CPU miss。它只验证可运行性：

```bash
export SMOKE_ROOT=$REPO/results/${MODEL_TAG}_ablation_smoke_$(date +%Y%m%d_%H%M%S)
(
set -euo pipefail
METHOD=K3 DATASET_TAG=tp2_12k DATASET_NAME=random-ids \
INPUT_LEN=12288 NUM_PROMPTS=2 OUTPUT_LEN=16 MAX_CONCURRENCY=1 \
WARMUP_REQUESTS=1 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$SMOKE_ROOT" \
CASE_TIMEOUT_S=1800 \
bash scripts/specstream/paper_eval/qwen3/run_public_once.sh

case_root=$SMOKE_ROOT/logs/K3_tp2_12k_c1
test -s "$case_root/case_complete.marker"
grep -q 'target_max_total_num_tokens=65536' "$case_root/kv_capacity_gate.txt"
grep -q 'draft_max_total_num_tokens=516224' "$case_root/kv_capacity_gate.txt"
if grep -RniE 'CUDA out of memory|Traceback|Scheduler hit an exception|RecvTimeout|DraftFallback' \
  "$case_root"; then
  echo 'ERROR: inspect the matched runtime failures before continuing' >&2
  false
fi
test -s "$SMOKE_ROOT/profiles/K3_tp2_12k_c1.csv"
echo QWEN3_32B_TP2_DRAFT_SMOKE=PASS
)
```

smoke 不通过时不要启动任何正式矩阵。重点看 `target.log`、`draft.log`、`kv_capacity_gate.txt`、timeout/fallback 和 `process_placement.csv`。

## 5. 创新点一：生命周期与长上下文精确一致性

静态生命周期：

```bash
export I1_ROOT=$REPO/results/${MODEL_TAG}_i1_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I1_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
(
set -euo pipefail
"$SPECSTREAM_PYTHON" -m pytest -q \
  python/sglang/test/spectre_specstream/test_state_invariants.py \
  python/sglang/test/spectre_specstream/test_async_seal_lifecycle.py \
  python/sglang/test/spectre_specstream/test_control_profile.py \
  | tee "$I1_ROOT/summary/lifecycle_gate.txt"
test ${PIPESTATUS[0]} = 0
)
```

完整 LongBench-v2、并发 1、三路径逐 token 比较：

```bash
(
set -euo pipefail
i1_failed=0
for method in AR I1_GPU_ONLY I1_SEALED_HISTORY; do
  METHOD=$method CLIENT_MODE=accuracy ACCURACY_DATASET=longbench_v2 \
  ACCURACY_RETURN_TOKEN_IDS=1 DATASET_NAME=sharegpt DATASET_TAG=i1_sealed_history \
  DATASET_PATH="$LONGBENCH_QWEN3" NUM_PROMPTS=0 OUTPUT_LEN=256 \
  MAX_CONCURRENCY=1 WARMUP_REQUESTS=0 REQUEST_RATE=inf SEED=1 \
  RESULT_ROOT="$I1_ROOT" CASE_TIMEOUT_S=43200 \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh || { i1_failed=1; break; }
done
echo I1_FORMAL_CASES_FAILED=$i1_failed
test "$i1_failed" = 0

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/compare_exact_generations.py \
  --reference "$I1_ROOT/bench/AR_i1_sealed_history_c1.jsonl" \
  --candidate "$I1_ROOT/bench/I1_GPU_ONLY_i1_sealed_history_c1.jsonl" \
  --candidate "$I1_ROOT/bench/I1_SEALED_HISTORY_i1_sealed_history_c1.jsonl" \
  --output "$I1_ROOT/summary/exact_generation_gate.json"

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/analyze_sealed_history.py \
  --profile "$I1_ROOT/profiles/I1_SEALED_HISTORY_i1_sealed_history_c1.csv" \
  --chunk-tokens 2048 --output "$I1_ROOT/summary/sealed_history_gate.json"
)
```

论文有效门槛：三路径 `exact_match_rate=1.0`；真实 `sealed_rows>0`、D2H/H2D>0、rejection 和 rollback>0；`rollback_crossed_history=false`；history chunk 对齐且单调；无 timeout/fallback/OOM/invariant error。若 0.6B/32B 在该集合没有 rejection，不能用“未报错”替代证明，需固定更大的 q 或选择能产生 rejection 的冻结样本后整组三方法重跑。

## 6. 创新点二：CUDA 数学 Gate 与双 TP2 K1–K5 40-cell

先运行 full-KV / online-softmax / 实际 fused CUDA kernel 一致性：

```bash
export I2_ROOT=$REPO/results/${MODEL_TAG}_i2_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I2_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
(
set -euo pipefail
CUDA_VISIBLE_DEVICES="$(cut -d, -f1 <<<"$TARGET_UUIDS")" \
"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/validate_online_attention_heatmap.py \
  --seq-len 16384 --chunk-tokens 2048 --chunks-per-transfer 4 \
  --output-dir "$I2_ROOT/summary/attention_chunk8192"
CUDA_VISIBLE_DEVICES="$(cut -d, -f1 <<<"$TARGET_UUIDS")" \
"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/validate_online_attention_heatmap.py \
  --seq-len 16384 --chunk-tokens 2048 --chunks-per-transfer 1 \
  --output-dir "$I2_ROOT/summary/attention_chunk2048"
)
```

实际 runner 应满足 `fused_launches>0`、heatmap row sum/argmax Gate、fused output/LSE 阈值；失败时不可进入吞吐矩阵。

可先用下面的无 GPU dry-run 核对 c16 全矩阵计划和两角色启动命令。必须先有当前机器 preflight 的 UUID 环境；不运行模型、不重复硬件 validator，也不生成正式完成 marker：

```bash
I2_ROOT="$REPO/results/${MODEL_TAG}_i2_tp2_dry_$(date +%Y%m%d_%H%M%S)" \
I2_DRY_RUN=1 I2_INPUT_LENGTHS="16384 32000" I2_CONCURRENCIES="1 4 8 16" \
SPECSTREAM_TARGET_MAX_TOTAL_TOKENS=65536 SPECSTREAM_TARGET_MIN_KV_TOKENS=65536 \
SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=516224 SPECSTREAM_DRAFT_MIN_KV_TOKENS=516224 \
bash scripts/specstream/paper_eval/qwen3/run_i2_k1_k5_once.sh
```

期望 `I2_DRY_RUN=PASS 40`。它只证明配置、计划和命令正确；第 4 节的实际 smoke 才验证模型加载、容量和 TP 通信。正式入口如下，显式 `I2_DRY_RUN=0` 防止继承 dry-run 设置：

```bash
export I2_ROOT=$REPO/results/${MODEL_TAG}_i2_k1_k5_once_$(date +%Y%m%d_%H%M%S)
unset PREFLIGHT_ROOT
MODEL_TAG=qwen3_0p6b_32b \
TARGET_MODEL="$TARGET_MODEL" DRAFT_MODEL="$DRAFT_MODEL" \
TARGET_GPUS=0,1 TARGET_TP_SIZE=2 TARGET_GPU=0 \
COLOCATED_GPU=0 COLOCATED_TP_RANK=0 DRAFT_GPU=0 \
SPECSTREAM_TARGET_MEM_FRACTION=0.55 \
SPECSTREAM_DRAFT_MEM_FRACTION=0.80 \
SPECSTREAM_TARGET_MAX_TOTAL_TOKENS=65536 \
SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=516224 \
SPECSTREAM_TARGET_MIN_KV_TOKENS=65536 \
SPECSTREAM_DRAFT_MIN_KV_TOKENS=516224 \
SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192 \
SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0 \
I2_INPUT_LENGTHS="16384 32000" I2_CONCURRENCIES="1 4 8 16" \
I2_METHODS="K1 K2 K3 K4 K5" I2_DRY_RUN=0 \
bash scripts/specstream/paper_eval/qwen3/run_i2_k1_k5_once.sh
```

上述显式配置会覆盖脚本的容量和缓存默认值，顺序执行 `input ∈ {16384,32000}`、`concurrency ∈ {1,4,8,16}`、`method ∈ {K1..K5}`，共 40 个 cell；每个 cell 64 请求、输出 256、warmup 4、请求率 inf。任何 cell 失败即停，不续跑到旧目录。`GPU_HISTORY_CACHE_TOKENS=0` 只能作为 cold-cache 压力补充实验；它与下述 GPU History hit Gate 冲突，不能混入本节采用 8192-token 统一缓存的正式矩阵。

矩阵计划保存在 `summary/matrix_plan.json`，包含每个精确 case tag 和 `expected_cells`。当前默认与上述正式配置均为 40 cells；改为一档输入/四档并发时可以是 20 cells，脚本按计划核对而不硬编码 40。当前 Draft cap=516224 与 c16×32000 的计划下限一致；仍须通过启动后的实际容量 Gate。只测 1/4/8 可显式删去 16，但保持整组统一的 cap，得到独立的 30-cell 矩阵。

完成 Gate：

```bash
(
set -euo pipefail
test -s "$I2_ROOT/i2_matrix_complete.marker"
I2_ROOT="$I2_ROOT" "$SPECSTREAM_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["I2_ROOT"])
plan = json.loads((root / "summary/matrix_plan.json").read_text())
expected = {cell["case_tag"] for cell in plan["cells"]}
markers = {f.parent.name for f in (root / "logs").glob("*/case_complete.marker")}
bench = {f.stem for f in (root / "bench").glob("K*.jsonl")}
assert expected == markers == bench
assert len(expected) == plan["expected_cells"]
print("I2_EXACT_CASE_SET=PASS", len(expected))
PY
if grep -RniE 'CUDA out of memory|Traceback|Scheduler hit an exception|RecvTimeout|DraftFallback' \
  "$I2_ROOT/logs"; then
  echo 'ERROR: inspect the matched runtime failures before continuing' >&2
  false
fi

"$SPECSTREAM_PYTHON" scripts/specstream/summarize_benchmarks.py \
  "$I2_ROOT"/bench/*.jsonl | tee "$I2_ROOT/summary/benchmark.tsv"
"$SPECSTREAM_PYTHON" scripts/specstream/summarize_specstream_profile.py \
  "$I2_ROOT"/profiles/*.csv | tee "$I2_ROOT/summary/profile.tsv"
)
```

本节固定 8192-token 缓存的启用卸载正式 cell 必须同时出现 `gpu_history_hit_tokens>0` 与 `cpu_history_miss_tokens>0`。`0` 缓存的补充矩阵只要求 CPU miss，不能要求 GPU hit；`-1` 自动缓存可出现全命中，结果属于独立部署比较，不能用于宣称有 CPU streaming 重叠。K1/K2 必须 `SERIALIZE_H2D=1`；K3–K5 必须 `h2d_event_ops == h2d_wait_event_ops == h2d_ops`。K4/K5 检查每个 q=2/4/6/8 的 Draft RTT 样本覆盖与候选 ordinary cost；样本不足的负载不能证明 controller 已充分校准，不应伪造样本或要求各 cell 都选遍所有 q。主比较为 K2/K1、K3/K2、K4/K3、K5/K4 和 K5/K1；不要求每个 cell 人为单调。

TP2 的 TP0 规范 profile 位于 `profiles/<case>.csv`；未经合并的 TP0/TP1 原始分片保存在 `logs/<case>/profile_shards/`。汇总只读取前者，不能把 rank 分片再加一次。

## 7. 创新点三：Draft 分布与同卡并行

### 7.1 两组问题与控制变量

所有组都用 `METHOD=SPECSTREAM_1GPU`，Target TP2、同两张卡、TPC=34、当前 CUDA graph/attention/通信路径、相同 caps 与缓存。不要用旧 A 代替 serial，旧 A 的执行配置不能作为只关闭重叠的单因素对照。

**A. 最终部署对照：回答“当前双 TP2 auto 是否比双 TP2 串行快”。**

| ID | Draft TP | mode | q | 负载 |
|---|---:|---|---|---|
| D2S | 2 | serial | 动态 2/4/6/8 | 01 同一冻结 LongBench 全集，c4、out256、warmup4 |
| D2P | 2 | auto | 动态 2/4/6/8 | 完全相同；最终公开测试版本 |

共 2 个正式 cell。主指标 `T(D2P)/T(D2S)-1`，其中 T 为 output tok/s。动态 q 与准入会共同影响实际 q 分布、accept length 和吞吐，因此这组测完整策略的净收益，不能单独证明纯重叠收益。小于 0 就是该负载下观察到的负收益，不能因为设计为并行就排除这种结果。

**B. 固定 q 机制对照：分离 Draft TP 分布与允许重叠。**

| ID | Draft TP | mode | 候选 q | 与谁比较 |
|---|---:|---|---:|---|
| S1 | 1，仅 GPU0 | serial | 8 | 旧分布的串行控制 |
| P1 | 1，仅 GPU0 | auto | 8 | P1/S1：TP1 下允许重叠的净效果 |
| S2 | 2，两张卡 | serial | 8 | S2/S1：串行条件下 Draft TP2 分布的净效果 |
| P2 | 2，两张卡 | auto | 8 | P2/S2：TP2 下允许重叠的净效果；P2/P1：auto 下 TP 分布效果 |

负载为 random-ids，长度 16384 和 32000，seed=1、固定长度比例 1、64 请求、out256、warmup4、rate=inf，默认并发 4：**2 长度 × 1 并发 × 4 组 = 8 个正式 cell**。需要并发扩展曲线时，从一开始用 `--concurrencies 1 4` 得到 16 个 cell，避免把默认 c4 重跑两遍。random-ids 用于控制形状和传输需求，不作为任务准确率证据。

`SPECSTREAM_FIXED_Q=8` 限制候选 q，并未删除安全 q=1 回退。检查实际 q/模式直方图；若两组有大量不同回退或校准状态，其比值仍只是策略效果，不能视作理想固定 q 的纯 overlap 加速。接受长度即使在固定 q 下也可能受数值、输出轨迹和批次组成影响，应一并报告。

四组都占两张卡，但 Draft TP1 的 KV/权重与计算集中在 GPU0，TP2 则分摊到两卡；这正是被研究的分布效果，不能声称两组每卡显存字节数相等。196608 Draft cap 适用于本节 c4；不要直接扩到 c8×32K，若扩展须统一重新标定所有组。

### 7.2 前置环境与无模型 dry-run

只测 I3：执行 01 第 2–5 节。若已经完成本手册 I1/I2，同一机器的有效 preflight 环境可复用；本节编排入口会恢复 01 的 0.62/0.80、131072/196608、双缓冲等固定设置。此入口不读取旧 I3 A/B/C runner，不会额外使用第三张卡。

```bash
export I3_DRY_ROOT=$REPO/results/${MODEL_TAG}_i3_tp2_dry_$(date +%Y%m%d_%H%M%S)
"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/run_tp2_final_matrix.py \
  --kind i3-fixed --root "$I3_DRY_ROOT" --dry-run
```

期望 `TP2_MATRIX_DRY_RUN=PASS cells=8`；`S1/P1` 的 `DRAFT_TP_SIZE=1`，`S2/P2` 为 2；serial 组的 `FIXED_Q_MODE=ordinary`；auto 组为 parallel 配置，但是否实际执行必须看后续 profile。q 候选均为 8，Target 均 TP2。

### 7.3 先运行双 TP2 动态 q 部署对照

```bash
export I3_DEPLOY_ROOT=$REPO/results/${MODEL_TAG}_i3_tp2_deploy_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I3_DEPLOY_ROOT"
(
  set -euo pipefail
  "$SPECSTREAM_PYTHON" -u scripts/specstream/paper_eval/qwen3/run_tp2_final_matrix.py \
    --kind i3-deploy --root "$I3_DEPLOY_ROOT" \
    2>&1 | tee "$I3_DEPLOY_ROOT/console.log"
)
```

### 7.4 再运行固定 q 的四组机制对照

```bash
export I3_FIXED_ROOT=$REPO/results/${MODEL_TAG}_i3_tp2_fixedq8_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I3_FIXED_ROOT"
(
  set -euo pipefail
  "$SPECSTREAM_PYTHON" -u scripts/specstream/paper_eval/qwen3/run_tp2_final_matrix.py \
    --kind i3-fixed --concurrencies 4 --root "$I3_FIXED_ROOT" \
    2>&1 | tee "$I3_FIXED_ROOT/console.log"
)
```

如需并发 1/4 的完整对照，**将上面命令的 `--concurrencies 4` 替换成 `--concurrencies 1 4`**，新根目录只运行这一套。默认 I3 共 2+8=10 个正式 cell；扩展后共 2+16=18 个。每 cell 前不会重复 smoke。终端同时显示 `[当前/总数] case` 和客户端数据集处理进度。

### 7.5 完成验收与查看结果

```bash
(
  set -euo pipefail
  for root in "$I3_DEPLOY_ROOT" "$I3_FIXED_ROOT"; do
    test -s "$root/matrix_complete.marker"
    grep -q 'TP2_MATRIX_COMPLETE=PASS' "$root/console.log"
    cat "$root/matrix_complete.marker"
    cat "$root/summary/results.json"
    "$SPECSTREAM_PYTHON" scripts/specstream/summarize_benchmarks.py \
      "$root"/bench/*.jsonl | tee "$root/summary/benchmark.tsv"
  done
)
```

编排入口逐 cell 检查全部请求成功、完成 marker、fatal/fallback、配置与 Grant Gate，并核对计划/实际结果精确集合。`matrix_complete.marker` 表示这些运行完整性检查通过，**不表示性能提升或机制激活**。失败会保留部分结果并停止，不生成矩阵完成标记。使用新根重跑，不补造 marker。

| 证据 | 判据与报告 |
|---|---|
| 有效执行 | completed=计划样本数、errors 为空、无 fatal/timeout/DraftFallback、Grant Gate PASS |
| 并行租约激活 | auto 组 `slack_fill_success>0`；缺失/为 0 均不能称为有效重叠实验 |
| 联合 TP2 执行 | 保留两个 rank 的启动/分片日志与联合 ACK；仅看到两个进程不构成执行重叠证明 |
| 未激活 | `overlap_evidence=NOT_ACTIVATED`；仍保留性能观察，结论写“该负载未激活” |
| 时间与资源效果 | 结合 `target_forward_ms`、`round_ms`、H2D/host wait、coexec 原因；不要把相互重叠的阶段时间直接相加 |
| q 与接受长度 | `q_histogram`、`mode_histogram`、`planned_mode_histogram`、accept length 一起报告；区分计划并行与实际模式 |
| 串行控制 | successful SLACK_FILL=0；若非 0，控制组无效，先查配置/实现 |
| 吞吐与准确性 | output tok/s、TTFT/TPOT/P99 和任务评分分开；性能成功不能替代正确性 |

所有比值在相同长度、并发、数据 SHA、q 设定内计算。可报告交互比 `(P2/S2)/(P1/S1)`；它描述本配置中 TP 分布对 auto 相对效果的影响，不等于通用 TP 扩展效率。不要将随机 token 的输出与 LongBench 输出放在同一吞吐平均值中。

若 auto 比 serial 慢，依次查看是否激活、q=1/回退比例、Target forward 与 Draft 等待是否变长、成功租约覆盖率是否过低。单个 SLACK_FILL 成功不足以证明足以抵消 TP collective 和资源竞争的开销。只有加入 GPU 时间轴分析后才能量化真正隐藏的 H2D 时间；本手册默认不强制额外 profiler，以免改变正式负载。

### 7.6 可选进一步消融与报告限制

先完成上述 10 个默认 cell，再根据失败证据决定是否增加测试。后台泵 0/1、TPC 宽度、buffers=2/4、catchup quantum=1/4 应各自独立成组；本版入口固定这些项，不接受用遗留环境变量悄悄覆盖。`pump=0` 仍保留前台/idle 推进，因此它也不等价于彻底关闭重叠。

旧 B（第三卡独占 Draft）仅用于另一个硬件成本问题，如 tok/s/GPU，可单独设计，不能用三卡结果替代本节双卡串行控制。主 I3 不再要求租用第三张卡。

## 8. 全文最终检查

- I1：Sealed History 提交/回滚不变量与逐 token 一致性；若不一致，不能直接归为浮点误差。定位首个分歧与 logits/采样配置后再解释，不能放宽 exact gate 并声称通过。
- I2：保持 K1–K5 原消融定义、TP2 Draft、统一 caps；实际完成集合匹配计划。跨层 H2D 预取和 Target/Draft 并行是两种不同重叠。
- I3：同一 SpecStream 实现的四组 fixed-q 对照 + 两组动态 q 部署对照。报告实际 Draft TP、q 分布、成功 SLACK_FILL 与未激活组；不要求结果人为单调。
- 01 主性能只用最终双 TP2 auto；运行过 02 后，重新执行 01 第 2 节再跑手工 smoke/准确性，避免继承 I1/I2 容量或串行开关。
- 单次实验只报告观察值；需要论文误差条时另开重复实验并平衡组次序，不能凭一次正差就声称稳定改善。

本次交付只修改手册和实验编排，未替操作者运行 GPU 实验；脚本 dry-run 通过也不等于模型、性能或准确性测试通过。
