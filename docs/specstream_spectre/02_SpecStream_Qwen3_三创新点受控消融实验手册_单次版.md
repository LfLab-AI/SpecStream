# SpecStream Qwen3-32B / Qwen3-0.6B 三创新点受控消融实验手册（单次正式版）

> Target：Qwen3-32B，TP=2；Draft：Qwen3-0.6B  
> 当前 2 × A800 可完成创新点一、创新点二以及创新点三的 A/C；完整 A/B/C 需要第 3 张 A800 作为独立 Draft GPU  
> K1–K5：Draft 与 Target TP rank 0 同卡、ordinary；K4/K5 也禁止 parallel  
> 统一 I1/I2/A/C 显存契约：Target `0.55 / 65536 tokens`，Draft `0.80 / 270336 tokens`  
> GPU History：全局 8192 tokens，allocation guard=0  
> 正式 cell 单次运行；smoke 只做一次，不写入论文表格

## 0. 2026-09-07 P0 / P1 / P2 版本说明

本手册适用于当前 Qwen3-32B / Qwen3-0.6B、Target TP=2 服务器。这里的 P0/P1/P2 是本轮工程修改优先级，与论文中的创新点一/二/三不是同一编号。

| 优先级 | 已交付的代码行为 | 验证重点 |
|---|---|---|
| P0 | TP 基线按 batch、q、上下文、CPU miss/GPU hit、attention 实现和 TP 分桶；只用已排除 Draft 重叠的预热样本；干扰时保留 ordinary 多 token 验证；有界恢复 | 不再因 batch 变大直接长期锁 q=1；检查 `tp_baseline_ready`、`tp_shape_key`、fallback 原因 |
| P0 | 从 Draft 接收前到提交后的真实 round wall time；native q=1 也记录；CPU enqueue 与 GPU event 分开；经验成本及受控 parallel 探测 | `round_timing_source`、`target_forward_ms`、`target_enqueue_ms`；不把各阶段重叠时间相加 |
| P1 | History split-KV、稳定归并、批量 GPU History/Tail/finalize；连续 CPU slab；不可变 metadata 缓存 | CUDA 数值、H2D 源字节/实际字节、DMA 数和 host 等待 |
| P1 | 首批 H2D 在 GPU History 计算前排队；当前/下一层任务复用；共享有界 ring 的槽位保护和取消回收 | 真正执行多层、多请求、请求重排和 2/4 buffers 的测试 |
| P2 | Target Python forward 期间后台推进 ACK/grant；末轮/idle/reset 仍收割迟到 ACK | `grant_pump_iterations` 仅证明泵运行；成功 SLACK_FILL ACK 才证明执行了租约 |
| P2 | 可选 1–8 token catchup 租约；SLACK_FILL 仍最多 1 token；可选 GPU History 自动预算；矩阵计划与完成集合严格核对 | 多 token ACK 不超预算；自动预算留 prefill、Tail/Frontier 和分配余量 |

本手册比较配置固定为 GPU History=8192、buffers=2、catchup quantum=1；split-KV 与后台泵启用。自动预算和更大 catchup 是单独的可选实验，不能与固定缓存对照混为同一配置。增加 buffers 只增加有界预取深度，不承诺 PCIe 带宽线性增加。

交付状态：此前候选版本在 A800 上有 `270 passed, 2 skipped` 的回归记录，包含实际 CUDA 数值与多层预取；这不等于最终版本已完成端到端吞吐验收。后续 ACK 收尾、运行器和文档修订由操作者按本手册重新验证。原版并发 8 曾完成全部客户端请求，但末轮缺失一个 ACK，因此未通过机制完整性 Gate；不得补造完成 marker 或当作已通过的正式对照。

### 0.1 开关及作用范围

| 环境变量 | 默认/本手册固定值 | 作用 |
|---|---|---|
| `SPECSTREAM_SPLIT_KV` | `auto` | `off` 或 `1` 关闭 split；也可指定 1–32；只改变 History 分片，不关闭全部批量 attention 优化 |
| `SPECSTREAM_BACKGROUND_GRANT_PUMP` | `1` | `0` 关闭 forward 期间后台泵；idle ACK 收尾仍保留 |
| `SPECSTREAM_GPU_HISTORY_CACHE_TOKENS` | 本手册 `8192` | `0` 为无 GPU History 副本；`-1` 为压力感知自动预算；单位是全局 KV token 槽，不是每请求预算 |
| `SPECSTREAM_NUM_BUFFERS` | `2` | 公共运行器仅对 `C/SPECSTREAM_1GPU` 生效，可试 `4`；K1–K5 保持消融定义中的固定 buffers |
| `SPECSTREAM_GRANT_TOKEN_QUANTUM` | `1` | 公共运行器仅对 `C/SPECSTREAM_1GPU` 生效，范围 1–8；实测步耗时/剩余期限决定实际 catchup 长度，SLACK_FILL 不随之变长 |
| `SPECSTREAM_REQUIRE_SLACK_FILL` | smoke `0` | `1` 要求至少一个成功的一 token SLACK_FILL ACK；用于重叠机制验收 |

Qwen3-32B BF16 TP2 每个全局 KV token 在每个 rank 约占 128 KiB。自动缓存复用已有 Target KV pool 的 CPU-backed 副本，不额外扩大 pool；不能把 `-1` 理解为无限缓存。缓存收缩只逐出已有 CPU 副本，不回收尚未完成 D2H 的 Tail/Frontier。

### 0.2 换卡或重启后的环境

每次租用新 GPU 都重新执行本手册的 preflight，使用本次生成的 `runtime_env.sh`。不要 source 历史 results 下的旧环境文件：GPU 索引相同不代表 UUID 相同。历史 `SPECSTREAM_SMCTRL_VALIDATED=1` 也不能代替当前硬件/驱动的预检。

测试前明确使用正式代码目录 `PYTHONPATH=/root/lifei/SpecStream/python`，不要继续使用 `results/p012_20260907/candidate_work/python`。不在交互式 SSH shell 全局设置 `set -euo pipefail`；每个命令块如出现 ERROR/非零状态，停止后续步骤并保留日志。

### 0.3 LongBench 后台线程设备绑定修复（2026-09-07）

本次公开矩阵 `qwen3_0p6b_32b_public_e2e_once_20260907_144615` 的 LongBench-v2 正式用例（并发 4、131 请求、输出 256）在 14:51:37 报错：`Expected a torch.device with a specified index or an integer, but got:cuda`，随后为 `SpecStream background grant pump failed`。直接原因是后台泵将无编号的 `torch.device("cuda")` 传给 `torch.cuda.set_device()`，导致 Target 退出；本次故障并非数据集格式错误。

修复后在 Target 调度器主线程解析设备编号，再由后台线程绑定该编号。显式 `cuda:0` / `cuda:1` 保持原编号，无编号 `cuda` 使用父线程当前设备；不硬编码物理 GPU 0，不关闭后台泵或多流水线重叠。此前回归未覆盖这一实际 scheduler 启动路径，本次补充了上述三种设备用例。

按操作者要求，本次仅交付修复、回归用例和文档，未执行回归测试或重启模型。操作者可先在 `spectre` 环境、仓库根目录运行：

```bash
PYTHONPATH=/root/lifei/SpecStream/python python -m pytest -q \
  python/sglang/test/spectre_specstream/test_background_grant_pump.py
```

然后重新启动 Target/Draft，保持 `SPECSTREAM_BACKGROUND_GRANT_PUMP=1`，使用当前 preflight 环境和新的 `RESULT_ROOT` 重跑失败用例；如需正式矩阵结果，按本手册矩阵入口重新运行。旧失败目录保留，不补写完成 marker。32-token smoke 成功不能替代 256-token 正式负载验收；必须确认请求全部成功、无上述异常，并由运行器正常生成 `case_complete.marker`。修复此退出异常不代表吞吐提升已通过验收。

### 0.4 LongBench 中途 TP 通信死锁修复（2026-09-07）

运行 `qwen3_0p6b_32b_public_e2e_once_20260907_150221` 的 LongBench 正式用例在异常前进度约为 `38/131`。15:32:51–52 的 watchdog 堆栈显示：TP0 在 `_decide_speculative_num_draft_tokens` 的 `all_gather_object` 等待，TP1 则在同一函数的决策 `broadcast_pyobj` 等待，最终 300 秒 watchdog 终止服务。客户端的连接拒绝是服务退出后的连带错误。

根因是两个 rank 分别用本地采样状态判断是否同步 TP profile；异步完成的本地状态不能保证一致，使两端执行了不同的集体通信。修复后仅 TP0 计算采样开关，先广播给所有 rank，然后统一执行可选采样 all-gather，最后广播动态 q 决策。每轮新增一次小型控制广播，保留原有采样间隔、动态 q、TP 干扰监测、后台泵及跨层预取。不要通过增大 watchdog 超时掩盖该死锁。

新增回归直接调用实际 scheduler 方法，使用两个 CPU/Gloo 进程，先模拟一致的预热状态，再模拟两端相反的采样判断，验证采样轮次、决策和通信顺序一致。测试设置 15 秒集体通信超时，不加载模型。操作者在 `spectre` 环境、仓库根目录执行：

```bash
PYTHONPATH=/root/lifei/SpecStream/python python -m pytest -q \
  python/sglang/test/spectre_specstream/test_tp_profile_collective_order.py \
  python/sglang/test/spectre_specstream/test_background_grant_pump.py
```

本次仅完成代码语法解析与服务器文件哈希核对，未执行上述回归或模型测试。请重新启动服务，在新的 `RESULT_ROOT` 下重跑同一完整正式用例，按正常 Gate 验收所有请求和完成 marker。此轮 15:15:05 还记录过一次 `DraftFallback`（q=8、4/4 缺失），这与最终 TP 通信死锁是不同观察；本补丁不宣称已消除该回退。即使重测能跑完，若仍有 DraftFallback，也应保留日志并判定机制验收失败，不能只看客户端成功数或吞吐率。

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

创新点三比较：

| ID | 位置与执行 | 物理 GPU |
|---|---|---:|
| `A` | Target TP2；Draft 位于 rank 0 GPU；ordinary 串行 | 2 |
| `B` | Target TP2 用 GPU 0,1；Draft 独占 GPU 2；parallel | 3 |
| `C` | Target TP2；Draft 位于 rank 0 GPU；PCIe-Slack + MPS + SMCTRL | 2 |

因此完整创新点三不能在当前两卡实例上伪造 `B`。两卡上完成 A/C 只能作为局部结果；论文中的 `C/B` 与 tok/s/GPU 对比必须换三卡实例后运行完整 18-cell 矩阵。

## 2. 统一环境与数据 Gate

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
export SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=270336
export SPECSTREAM_TARGET_MIN_KV_TOKENS=65536
export SPECSTREAM_DRAFT_MIN_KV_TOKENS=270336
export SPECSTREAM_PREFILL_MAX_REQUESTS=1
export SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192
export SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0
export SPECSTREAM_REQUIRE_SLACK_FILL=0
export SPECSTREAM_SPLIT_KV=auto SPECSTREAM_BACKGROUND_GRANT_PUMP=1
export SPECSTREAM_NUM_BUFFERS=2 SPECSTREAM_GRANT_TOKEN_QUANTUM=1
```

数据无需重处理，理由和指纹 Gate 与 01 手册第 3 节完全相同：当前 32B/0.6B tokenizer 映射、特殊 token、non-thinking template 与冻结 manifest 一致。消融只使用 random-ids 和创新点一的冻结 LongBench-v2；换模型本身不要求重建文本。若 01 的 `QWEN3_32B_DATA_REUSE_GATE` 失败，必须先按 01 第 3.2 节重生成，不能继续。

65,536 Target tokens 对 Qwen3-32B TP2 约占每个 rank 8 GiB KV；270,336 Draft tokens 约占 28.9 GiB KV。它们是所有 K1–K5 和 A/C 的固定 cap。Target 历史会迁移到 CPU，因此 Target cap 不等于八条 32K 请求的总长度；Draft 保存完整上下文，所以它必须覆盖 `8 × (32000+256)=258048` 并保留推测余量。两个 fraction 是各进程容量上限，不是可直接相加的 GPU 百分比。

## 3. 前置 Gate 与 TP2 命令 Gate

```bash
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

export PREFLIGHT_ROOT=$REPO/results/${MODEL_TAG}_ablation_preflight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PREFLIGHT_ROOT"
bash scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  2>&1 | tee "$PREFLIGHT_ROOT/console.log"
grep -q 'QWEN3_PUBLIC_PREFLIGHT=PASS' "$PREFLIGHT_ROOT/console.log"
source "$PREFLIGHT_ROOT/runtime_env.sh"

"$SPECSTREAM_PYTHON" -m pytest -q python/sglang/test/spectre_specstream \
  | tee "$PREFLIGHT_ROOT/env/specstream_pytest.txt"
test ${PIPESTATUS[0]} = 0
```

不启动模型的命令 Gate：

```bash
export DRY_ROOT=$PREFLIGHT_ROOT/ablation_dryrun
for method in I1_GPU_ONLY I1_SEALED_HISTORY K1 K2 K3 K4 K5 A C; do
  METHOD=$method DATASET_TAG=dry DATASET_NAME=random-ids \
  INPUT_LEN=32000 NUM_PROMPTS=1 OUTPUT_LEN=256 MAX_CONCURRENCY=8 \
  WARMUP_REQUESTS=0 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$DRY_ROOT" \
  SPECSTREAM_DRY_RUN=1 \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh || break
done

for config in "$DRY_ROOT"/logs/*/config.env; do
  grep -q '^TARGET_TP_SIZE=2$' "$config"
  grep -q '^COLOCATED_TP_RANK=0$' "$config"
  grep -q '^TARGET_MAX_TOTAL_TOKENS=65536$' "$config"
  grep -q '^DRAFT_MAX_TOTAL_TOKENS=270336$' "$config"
done

for method in K1 K2; do
  f="$DRY_ROOT/logs/${method}_dry_c8/config.env"
  grep -q '^SERIALIZE_H2D=1$' "$f" && grep -q '^H2D_EXECUTION=serialized$' "$f"
done
for method in K3 K4 K5; do
  f="$DRY_ROOT/logs/${method}_dry_c8/config.env"
  grep -q '^SERIALIZE_H2D=0$' "$f" && grep -q '^H2D_EXECUTION=async_copy_stream$' "$f"
done
echo QWEN3_32B_ABLATION_DRY_GATE=PASS
```

## 4. 一次性 TP2 + Draft smoke

smoke 使用正式 K3 路径和正式 exact caps，以 12K 输入真实触发 Sealed History、GPU History 与 CPU miss。它只验证可运行性：

```bash
export SMOKE_ROOT=$REPO/results/${MODEL_TAG}_ablation_smoke_$(date +%Y%m%d_%H%M%S)
METHOD=K3 DATASET_TAG=tp2_12k DATASET_NAME=random-ids \
INPUT_LEN=12288 NUM_PROMPTS=2 OUTPUT_LEN=16 MAX_CONCURRENCY=1 \
WARMUP_REQUESTS=1 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$SMOKE_ROOT" \
CASE_TIMEOUT_S=1800 \
bash scripts/specstream/paper_eval/qwen3/run_public_once.sh

case_root=$SMOKE_ROOT/logs/K3_tp2_12k_c1
test -s "$case_root/case_complete.marker"
grep -q 'target_max_total_num_tokens=65536' "$case_root/kv_capacity_gate.txt"
grep -q 'draft_max_total_num_tokens=270336' "$case_root/kv_capacity_gate.txt"
if grep -RniE 'CUDA out of memory|Traceback|Scheduler hit an exception|RecvTimeout|DraftFallback' \
  "$case_root"; then
  echo 'ERROR: inspect the matched runtime failures before continuing' >&2
  false
fi
test -s "$SMOKE_ROOT/profiles/K3_tp2_12k_c1.csv"
echo QWEN3_32B_TP2_DRAFT_SMOKE=PASS
```

smoke 不通过时不要启动任何正式矩阵。重点看 `target.log`、`draft.log`、`kv_capacity_gate.txt`、timeout/fallback 和 `process_placement.csv`。

## 5. 创新点一：生命周期与长上下文精确一致性

静态生命周期：

```bash
export I1_ROOT=$REPO/results/${MODEL_TAG}_i1_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I1_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
"$SPECSTREAM_PYTHON" -m pytest -q \
  python/sglang/test/spectre_specstream/test_state_invariants.py \
  python/sglang/test/spectre_specstream/test_async_seal_lifecycle.py \
  python/sglang/test/spectre_specstream/test_control_profile.py \
  | tee "$I1_ROOT/summary/lifecycle_gate.txt"
test ${PIPESTATUS[0]} = 0
```

完整 LongBench-v2、并发 1、三路径逐 token 比较：

```bash
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

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/compare_exact_generations.py \
  --reference "$I1_ROOT/bench/AR_i1_sealed_history_c1.jsonl" \
  --candidate "$I1_ROOT/bench/I1_GPU_ONLY_i1_sealed_history_c1.jsonl" \
  --candidate "$I1_ROOT/bench/I1_SEALED_HISTORY_i1_sealed_history_c1.jsonl" \
  --output "$I1_ROOT/summary/exact_generation_gate.json"

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/analyze_sealed_history.py \
  --profile "$I1_ROOT/profiles/I1_SEALED_HISTORY_i1_sealed_history_c1.csv" \
  --chunk-tokens 2048 --output "$I1_ROOT/summary/sealed_history_gate.json"
```

论文有效门槛：三路径 `exact_match_rate=1.0`；真实 `sealed_rows>0`、D2H/H2D>0、rejection 和 rollback>0；`rollback_crossed_history=false`；history chunk 对齐且单调；无 timeout/fallback/OOM/invariant error。若 0.6B/32B 在该集合没有 rejection，不能用“未报错”替代证明，需固定更大的 q 或选择能产生 rejection 的冻结样本后整组三方法重跑。

## 6. 创新点二：CUDA 数学 Gate 与 K1–K5 30-cell

先运行 full-KV / online-softmax / 实际 fused CUDA kernel 一致性：

```bash
export I2_ROOT=$REPO/results/${MODEL_TAG}_i2_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I2_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
CUDA_VISIBLE_DEVICES="$(cut -d, -f1 <<<"$TARGET_UUIDS")" \
"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/validate_online_attention_heatmap.py \
  --seq-len 16384 --chunk-tokens 2048 --chunks-per-transfer 4 \
  --output-dir "$I2_ROOT/summary/attention_chunk8192"
CUDA_VISIBLE_DEVICES="$(cut -d, -f1 <<<"$TARGET_UUIDS")" \
"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/validate_online_attention_heatmap.py \
  --seq-len 16384 --chunk-tokens 2048 --chunks-per-transfer 1 \
  --output-dir "$I2_ROOT/summary/attention_chunk2048"
```

实际 runner 应满足 `fused_launches>0`、heatmap row sum/argmax Gate、fused output/LSE 阈值；失败时不可进入吞吐矩阵。

K1–K5 的正式入口已允许外部覆盖 32B 模型、TP 和容量，不再被脚本写回 8B 默认值：

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
SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=270336 \
SPECSTREAM_TARGET_MIN_KV_TOKENS=65536 \
SPECSTREAM_DRAFT_MIN_KV_TOKENS=270336 \
SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192 \
SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0 \
I2_INPUT_LENGTHS="16384 32000" I2_CONCURRENCIES="1 4 8" \
I2_METHODS="K1 K2 K3 K4 K5" \
bash scripts/specstream/paper_eval/qwen3/run_i2_k1_k5_once.sh
```

上述显式配置与脚本默认值一致，顺序执行 `input ∈ {16384,32000}`、`concurrency ∈ {1,4,8}`、`method ∈ {K1..K5}`，共 30 个 cell；每个 cell 64 请求、输出 256、warmup 4、请求率 inf。任何 cell 失败即停，不续跑到旧目录。`GPU_HISTORY_CACHE_TOKENS=0` 只能作为 cold-cache 压力补充实验；它与下述 GPU History hit Gate 冲突，不能混入本节采用 8192-token 统一缓存的正式矩阵。

矩阵计划保存在 `summary/matrix_plan.json`，包含每个精确 case tag 和 `expected_cells`。当前默认与上述正式配置均为 30 cells；改为一档输入/四档并发时可以是 20 cells，脚本按计划核对而不硬编码 30。当前 Draft cap=270336 无法容纳 16 条 32K 完整上下文，不能只把并发改为 16；脚本会在启动前拒绝不足的容量。

完成 Gate：

```bash
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
```

本节固定 8192-token 缓存的启用卸载正式 cell 必须同时出现 `gpu_history_hit_tokens>0` 与 `cpu_history_miss_tokens>0`。`0` 缓存的补充矩阵只要求 CPU miss，不能要求 GPU hit；`-1` 自动缓存可出现全命中，结果属于独立部署比较，不能用于宣称有 CPU streaming 重叠。K1/K2 必须 `SERIALIZE_H2D=1`；K3–K5 必须 `h2d_event_ops == h2d_wait_event_ops == h2d_ops`。K4/K5 必须为每个 q=2/4/6/8 至少记录 4 个 Draft RTT 样本，并写出全部候选 ordinary cost。主比较为 K2/K1、K3/K2、K4/K3、K5/K4 和 K5/K1；不要求每个 cell 人为单调。

TP2 的 TP0 规范 profile 位于 `profiles/<case>.csv`；未经合并的 TP0/TP1 原始分片保存在 `logs/<case>/profile_shards/`。汇总只读取前者，不能把 rank 分片再加一次。

## 7. 创新点三 A/B/C

### 7.1 当前两卡实例：只允许 A/C smoke 或补充结果

```bash
export I3_AC_ROOT=$REPO/results/${MODEL_TAG}_i3_ac_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I3_AC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
export SPECSTREAM_REQUIRE_SLACK_FILL=1

ac_failed=0
for input in 16384 32000; do
  for conc in 1 4 8; do
    for method in A C; do
      METHOD=$method DATASET_TAG=i3_${input} DATASET_NAME=random-ids \
      INPUT_LEN=$input NUM_PROMPTS=64 OUTPUT_LEN=256 MAX_CONCURRENCY=$conc \
      WARMUP_REQUESTS=4 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$I3_AC_ROOT" \
      CASE_TIMEOUT_S=21600 \
      bash scripts/specstream/paper_eval/qwen3/run_public_once.sh || { ac_failed=1; break 3; }
    done
  done
done
echo I3_AC_FAILED=$ac_failed
```

这 12 个 cell 只能用于 `C/A`；不得标成完整创新点三，也不得把 A 当作独立单 GPU Target。

### 7.2 三卡实例：完整 18-cell 正式矩阵

换到至少 3 × A800 80 GiB 后，先把 GPU 2 设为独立 Draft 并重新 preflight：

```bash
export TARGET_GPUS=0,1 TARGET_TP_SIZE=2 TARGET_GPU=0
export COLOCATED_GPU=0 COLOCATED_TP_RANK=0
export DRAFT_GPU=2 REQUIRE_SEPARATE_DRAFT_GPU=1
export PREFLIGHT_ROOT=$REPO/results/${MODEL_TAG}_i3_preflight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PREFLIGHT_ROOT"
bash scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  2>&1 | tee "$PREFLIGHT_ROOT/console.log"
grep -q 'QWEN3_PUBLIC_PREFLIGHT=PASS' "$PREFLIGHT_ROOT/console.log"
source "$PREFLIGHT_ROOT/runtime_env.sh"

export I3_ROOT=$REPO/results/${MODEL_TAG}_i3_abc_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$I3_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
export SPECSTREAM_REQUIRE_SLACK_FILL=1

i3_failed=0
for input in 16384 32000; do
  for conc in 1 4 8; do
    for method in A B C; do
      METHOD=$method DATASET_TAG=i3_${input} DATASET_NAME=random-ids \
      INPUT_LEN=$input NUM_PROMPTS=64 OUTPUT_LEN=256 MAX_CONCURRENCY=$conc \
      WARMUP_REQUESTS=4 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$I3_ROOT" \
      CASE_TIMEOUT_S=21600 \
      bash scripts/specstream/paper_eval/qwen3/run_public_once.sh || { i3_failed=1; break 3; }
    done
  done
done
echo I3_FORMAL_MATRIX_FAILED=$i3_failed
```

A/C 的 Draft 使用 `COLOCATED_UUID`，B 使用独立 `DRAFT_UUID`。C 的每个 cell 必须生成 `grant_event_gate.json`，至少一个成功 one-token `SLACK_FILL`，issued grant 与 ACK 一一对应，实际 TPC width=34；expired/deferred ACK 消耗 0 token。报告 C/A、C/B、绝对吞吐、tok/s/GPU、Target slowdown 和 grant utilization。

## 8. 最终有效性清单

- 所有正式目录均记录 `MODEL_TAG=qwen3_0p6b_32b`、Target TP2 的两个 UUID 和 colocated rank 0。
- 同一矩阵所有方法 exact cap 相同；capacity Gate 不通过时整矩阵停止。
- I1 有精确输出、真实 seal/rejection/rollback 且回滚从不越过 History。
- I2 marker、benchmark、profile 与 `matrix_plan.json` 的精确 case 集合一致；本节默认正式配置为 30 个；K1/K2 串行，K3–K5 真异步，K4/K5 controller 遥测完整。
- I3 论文正式表必须有 A/B/C 18 个 cell；当前两卡 A/C 不能冒充完整表。
- 任一 timeout、fallback、missing Draft、OOM、invariant error 或不完整 telemetry 都使对应 cell 无效。
- 单次实验只报告观察值；不报告标准差、置信区间或显著性。

## 9. P0/P1/P2 改造的独立消融与回退

先执行 01 第 9.1 节的一个 CUDA 正确性入口，再按 01 第 9.2–9.4 节使用相同冻结 workload 比较固定缓存、预取深度、catchup 和自动缓存。这里的 P0/P1/P2 修改适用于当前版本所有相关方法；K1–K5 仍只比较第 1 节定义的增量。split-KV 和批量 Tail 是基础 kernel 实现，不应只为 K5 开启再把全部收益归给 Cohort。

单独消融 split-KV 时，整组 K1–K5 使用 `SPECSTREAM_SPLIT_KV=off`，结果写入新根目录，并与整组 `auto` 比较。这个开关不会关闭其他批量 GPU attention 路径，不是完整旧版本回滚。K4/K5 保持 ordinary；`SPECSTREAM_GRANT_TOKEN_QUANTUM` 与后台泵的机制比较应放在 A/C 或完整 SpecStream，不能用 K4/K5 证明 Draft overlap。

验证创新点三时先做默认 quantum=1 的 A/C 固定预算比较；然后单独以 quantum=4 比较 C，最后才试自动 GPU History 和四缓冲组合。每次改变配置都建立新结果根。成功 `SLACK_FILL`、成功多 token `DRAFT_CATCHUP` 与更高 output throughput 是三个不同结论，分别由 grant Gate、issued/ACK 预算计数和 benchmark JSONL 支撑。

ACK 审计新增 `PREFILL_COMPLETE`、`PAUSED`、`FINISHED` 等终态；审计器会检查部分消耗和未使用预算归还。SLACK_FILL 仍最多一 token，catchup 可 1–8 token，重复/缺失 ACK 仍判失败。末轮 DRAFT 先于 ACK 到达时，Target idle loop 会补收真实 ACK；不允许在离线汇总中凭空补 ACK。

回退顺序：先使用固定预算 `8192 / buffers=2 / quantum=1`；若仅怀疑 split-KV，可设 `SPECSTREAM_SPLIT_KV=off`；若仅怀疑 forward 内控制泵，可设 `SPECSTREAM_BACKGROUND_GRANT_PUMP=0`。开关均在启动服务前设置。完整代码回滚使用交付记录中的服务器备份，先核对文件白名单并保留后续用户修改；不要执行仓库级 reset。
