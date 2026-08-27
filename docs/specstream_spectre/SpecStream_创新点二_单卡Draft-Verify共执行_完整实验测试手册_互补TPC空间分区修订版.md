# SpecStream 创新点二：单卡 Draft–Verify 安全共执行完整实验测试手册

> **独立版 / 2026-08-26 D0 + 互补 TPC 空间分区修订**
>
> 本文档仅覆盖 SpecStream 创新点二：**measurement-backed Target-priority single-GPU Draft–Verify co-execution**。创新点一的 CPU History KV streaming 仅在最终 D5 协同实验中被提及，不在本文展开。
>
> 本版已经吸收当前真实调试中暴露的问题：
>
> - 新增 **D0：原生 SGLang 单卡串行 Draft→Target Verify（STANDALONE）** 基线，用于在任何 TPC 控制之前建立单卡传统推测解码参考；
> - 同卡 Target/Drafter 显式限制 KV pool；
> - D1–D4 统一固定 `q=4`；
> - MPS 只负责两个 CUDA 进程同卡共存，不是最终调度器；
> - `libsmctrl` TPC mask 才是计算资源隔离机制；
> - **Drafter 与 Target 现在都安装 TPC mask：`SLACK_FILL` 重叠窗口内 Draft 使用 `[0,k)`，Target 使用互补 `[k,N)`；**
> - Target forward 完成后立即恢复 Target=`[0,N)`，避免无 Draft 工作时永久损失 Target 算力；
> - Target 和 Drafter 都必须通过 `/health`；
> - 修复 `first-grant-before-first-forward` 后才允许做 TPC calibration；
> - fixed-TPC calibration 时 Target 和 Drafter 两端都显式传入相同的 `--specstream-smctrl-calibration-tpcs`；
> - resource profile 必须按真实 `(batch_size, q, context_bucket)` 构建；
> - 新增 `target_tpc_low/target_tpc_high` profile 字段，必须用运行记录证明 Target/Draft 真正互斥；
> - D4/D5 未命中 profile 时必须 fail closed；
> - D5 dynamic-q 只有在高频 q/bs/context shape 已实测覆盖后才能作为正式结果。
>
> **适用仓库**
>
> ```text
> ~/lifei/SpecStream
> GitHub: LfLab-AI/SpecStream
> ```
>
> **当前实验平台示例**
>
> ```text
> GPU: NVIDIA A800 80GB × 2
> Target: Qwen2.5-7B-Instruct
> Drafter: Qwen2.5-0.5B-Instruct
> TP = 1
> ```

> **本版新增代码语义**
>
> ```text
> SLACK_FILL overlap:
> Draft  = [0,k)
> Target = [k,N)
>
> Target forward 完成后:
> Target = [0,N)
>
> TARGET_EXCLUSIVE:
> Draft 不执行
> Target = [0,N)
>
> DRAFT_CATCHUP:
> 仅在 Target CUDA forward 完成后允许 Draft 推进。
> ```
>
> 对应新增/修改代码：`tpc_partition.py`、`server_args.py`、`config.py`、`verifier.py`、`spectre_worker.py`、`profiler.py` 与互补分区单测。

---

# 0. 创新点二需要证明什么

创新点二不是证明“两个进程能放到一张 GPU 上”，也不是证明“把 Drafter 限到几个 TPC 就能跑”。本版要证明的是：

> 在消除 dedicated Drafter GPU 的情况下，通过 **离线实测 resource profile + Target/Draft 互补 TPC 空间分区 + Target-priority one-token grant/ACK**，让 Draft 在 Target 的安全重叠窗口中推进，并把 Target slowdown 控制在预算内，从而提高单位 GPU 的服务效率。

本版特别修正了旧方案的核心缺陷：旧实现只限制 Drafter 到 `[0,k)`，Target 仍可使用 `[0,N)`，本质上只是 **Draft throttling**；新实现要求在真正的 `SLACK_FILL` 重叠窗口中满足：

```text
Draft  = [0,k)
Target = [k,N)
Draft ∩ Target = ∅
```

并在 Target forward 完成后恢复：

```text
Target = [0,N)
```

最终实验需要回答六个问题：

1. **传统单卡串行 Draft→Verify（D0）到底有多快？**
2. 把 Draft/Verify 改成同卡 naive 并行后（D2），是否真的优于 D0？
3. 旧式 Draft-only throttling 为什么可能无效？真正的互补空间分区是否能进一步降低 Target 干扰？
4. 固定 `k` 的互补 TPC 分区是否存在安全 Pareto 点？
5. 在线 Target-priority grant 是否能根据 measured profile 动态选择重叠机会，并维持 `Target slowdown <= 5%`？
6. 与两卡 SPECTRE 相比，单卡 raw throughput 损失多少，但 throughput/GPU、P99 和资源效率改善多少？

---

# 1. 最终实验对照定义

| ID | GPU 数 | KV 路径 | Draft–Verify 调度 | TPC 语义 | 目的 |
|---|---:|---|---|---|---|
| D0 | 1 | GPU-resident | **原生 SGLang STANDALONE：Draft→Target Verify 串行执行** | 无 TPC | 单卡传统推测解码基线 |
| D1 | 2 | GPU-resident | SPECTRE parallel，Target/Draft 分卡 | 无同卡争用 | 绝对吞吐强基线 |
| D2 | 1 | GPU-resident | SPECTRE parallel，同卡，无控制 | Target=`[0,N)`，Draft=`[0,N)` | naive 干扰负面对照 |
| D3-T（可选消融） | 1 | GPU-resident | 同卡 + 旧式 Draft-only 固定 TPC | Draft=`[0,k)`，Target=`[0,N)` | 证明“仅限 Draft”是否足够 |
| **D3** | 1 | GPU-resident | **同卡 + 固定 k 的互补 TPC 分区** | **Draft=`[0,k)`，Target=`[k,N)`** | 静态空间隔离基线 |
| **D4** | 1 | GPU-resident | **online Target-priority one-token grant + 互补 TPC** | `SLACK_FILL` 时互补；其余恢复 full Target | **隔离创新点二** |
| D5 | 1 | bounded KV streaming | online grant + complementary TPC + dynamic q + Cohort | I1+I2 | 完整系统 |

创新点二正文核心比较：

```text
单卡因果主线：
D0（串行）
 → D2（同卡 naive 并行）
 → D3（固定 k 的真正互补空间分区）
 → D4（在线互补分区 + one-token grant）

可选机制消融：D3-T（旧式 Draft-only throttling）
双卡强参考：D1（Target/Draft 分卡并行）
```

其中 `D3-T` 不是必须主表项，但如果你已经有旧代码结果，强烈建议保留为消融。它能直接回答：**性能不变究竟是因为 TPC 本身无效，还是因为旧实现没有真正隔离 Target/Draft。**

---

# 2. 实验统一原则

## 2.1 D0–D4 固定 q=4

统一：

```text
speculative_num_steps = 3
speculative_num_draft_tokens = 4
q = 4
```

D0–D4 都统一使用相同的 `num_steps=3`、`num_draft_tokens=4`。不要再使用不同 q 配置进行横向比较。

## 2.2 同卡显式限制 KV pool

Target 与 Drafter 都显式设置：

```bash
--max-total-tokens 200000
```

`200000` 只是当前 `16K/C8` pilot 起始值。更长上下文或更高并发应重新计算容量。

粗略检查：

```text
(input_len + output_len + reserve) × concurrency × 1.25
```

例如 30K/C8 不要继续直接使用 200K，可从约 320K 起重新验证。

## 2.3 MPS 不是最终调度器

MPS 只负责：

```text
两个独立 CUDA 进程共享同一张物理 GPU
```

真正的 I2 调度：

```text
measured resource profile
+ libsmctrl global TPC masks
+ Target-priority one-token grant/ACK
+ complementary Target/Draft partition
```

不要把 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 当成最终创新点二调度方法。

## 2.4 无 grant 绝不允许受控 Draft forward

核心不变量：

```text
NO_GRANT
   ↓
NO_DRAFT_GPU_FORWARD
```

正确在线路径：

```text
Target 发现安全 overlap/slack
        ↓
grant(epoch, Draft=[0,k), token_budget=1)
        ↓
Drafter 安装 [0,k) TPC mask
        ↓
Target 同时安装互补 [k,N) mask
        ↓
执行恰好 1 个 Draft token + Target forward
        ↓
Target forward 完成后恢复 [0,N)
        ↓
Draft ACK
        ↓
Target 才允许下一 grant
```

## 2.5 互补分区只覆盖真实重叠窗口

不要把 Target 在整个服务生命周期永久固定到 `[k,N)`。正确行为是：

```text
无 overlap：
Target = [0,N)

SLACK_FILL overlap：
Draft  = [0,k)
Target = [k,N)

Target forward 完成：
Target = [0,N)
```

原因是如果 Draft 当前没有可执行工作，永久保留 `[0,k)` 会白白减少 Target 算力。

## 2.6 global mask 是“每个独立进程内”的 process-wide mask

本方案的 Target 与 Drafter 是两个独立 SPECTRE CUDA 进程：

```text
Target process  → global mask [k,N)
Draft process   → global mask [0,k)
```

因此可以形成同一物理 GPU 上的互补 TPC 子集。不要把这套逻辑直接用于 D0 的单进程 STANDALONE。

---

# 3. 开始实验前的代码前置条件

本版**不再引入第二套 grant 状态机**。在线控制必须复用当前仓库已有的：

```text
TargetGrantRuntime
        ↓
Target 发 GRANT / PAUSE
        ↓
DraftGrantTable / Drafter priority path
        ↓
安装 Draft TPC mask
        ↓
恰好一个 token
        ↓
GRANT_ACK
```

互补 TPC 修订新增/修改如下：

```text
新增：
python/sglang/srt/speculative/spectre/specstream/tpc_partition.py
python/sglang/test/spectre_specstream/test_complementary_tpc_partition.py

修改：
python/sglang/srt/server_args.py
python/sglang/srt/speculative/spectre/specstream/config.py
python/sglang/srt/speculative/spectre/specstream/verifier.py
python/sglang/srt/speculative/spectre/verifier/spectre_worker.py
python/sglang/srt/speculative/spectre/specstream/profiler.py
csrc/specstream_smctrl/README.md
```

代码必须满足：

```text
smctrl disabled
→ 保持原 SPECTRE 行为

smctrl enabled + complementary disabled
→ 可用于旧式 Draft-only throttling 消融

smctrl enabled + complementary enabled
→ 仅允许 --specstream-smctrl-mask-scope global
→ SLACK_FILL: Draft=[0,k), Target=[k,N)
→ Target forward 完成后 Target 恢复 [0,N)

fixed-TPC calibration
→ 首个 controlled Draft forward 前必须已有 bootstrap grant + Draft TPC mask
→ overlap run 必须同时存在 Target complement mask

online D4
→ 不写死 calibration_tpcs
→ 从同一个 TargetGrantRuntime outstanding decision 推导 Target complement
→ 不允许建立第二套独立 TPC 决策状态机
```

另外，`spectre_worker.py` 的 Target CUDA completion event 必须覆盖 Verify，并对 Extend/Prefill 路径保持同样的“Target GPU work 真正结束后才允许 catchup”的安全边界。

---

# 4. 环境初始化

```bash
cd ~/lifei/SpecStream
conda activate spectre

export REPO=$PWD
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export SPECSTREAM_PYTHON="$(command -v python)"
```

检查：

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import sglang; print(sglang.__file__)"
```

应指向：

```text
/root/miniconda3/envs/spectre/bin/python
/root/lifei/SpecStream/python/...
```

launcher 内必须使用：

```bash
bash -c
```

不要使用：

```bash
bash -lc
```

AutoDL login shell 可能把 Python 重置到 base 环境。

---

# 5. 结果目录与版本记录

```bash
export RESULT_ROOT=$REPO/results/innovation2

mkdir -p   "$RESULT_ROOT/logs"   "$RESULT_ROOT/bench"   "$RESULT_ROOT/profiles"   "$RESULT_ROOT/resource_profiles"   "$RESULT_ROOT/source_data"   "$RESULT_ROOT/smctrl"
```

记录版本：

```bash
git rev-parse HEAD | tee "$RESULT_ROOT/source_data/git_commit.txt"
git status --short | tee "$RESULT_ROOT/source_data/git_status.txt"

python - <<'PY' | tee "$RESULT_ROOT/source_data/python_env.txt"
import sys, torch, sglang
print("python =", sys.executable)
print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("sglang =", sglang.__file__)
PY

nvidia-smi -L | tee "$RESULT_ROOT/source_data/nvidia_smi_L.txt"
nvidia-smi | tee "$RESULT_ROOT/source_data/nvidia_smi.txt"
```

---

# 6. 模型、端口、GPU

根据实际路径修改：

```bash
export TARGET_MODEL=/root/autodl-tmp/model/Qwen2.5-7B-Instruct
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct

export TARGET_PORT=30000
export DRAFT_PORT=30001
export ZMQ_PORT=29000
```

双卡：

```bash
export TARGET_GPU=1
export DRAFT_GPU=0
```

查看 UUID：

```bash
nvidia-smi -L
```

设置同卡物理 GPU：

```bash
export SINGLE_GPU_UUID="GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

必须替换为当前机器实际 UUID。

---

# 7. 数据与统一 workload

```bash
export SHAREGPT_JSON=$REPO/specstream_prepared/sharegpt_v3_merged.json
test -f "$SHAREGPT_JSON" && echo "ShareGPT OK"
```

第一阶段统一：

```bash
export INPUT_LEN=16384
export OUTPUT_LEN=128
export MAX_CONCURRENCY=8
export NUM_PROMPTS=200
export REQUEST_RATE=inf
export RANGE_RATIO=1
export WARMUP_REQUESTS=4
export SEED=1
```

最小 Gate smoke：

```text
INPUT_LEN=16384
OUTPUT_LEN=32
NUM_PROMPTS=4
MAX_CONCURRENCY=1
WARMUP_REQUESTS=1
```

---

# 8. 统一 q=4 与显存池

```bash
export I2_NUM_STEPS=3
export I2_NUM_DRAFT_TOKENS=4

export I2_TARGET_MAX_TOTAL_TOKENS=200000
export I2_DRAFT_MAX_TOTAL_TOKENS=200000

export I2_CONTEXT_LENGTH=32768
export I2_MAX_PREFILL_TOKENS=16384
```

---

# 9. Target 公共启动命令

```bash
export I2_TARGET_COMMON="python -m sglang.launch_server   --model-path '$TARGET_MODEL'   --port $TARGET_PORT   --context-length $I2_CONTEXT_LENGTH   --max-prefill-tokens $I2_MAX_PREFILL_TOKENS   --max-total-tokens $I2_TARGET_MAX_TOTAL_TOKENS   --skip-server-warmup   --speculative-algorithm SPECTRE   --spectre-role target   --speculative-num-steps 3   --speculative-eagle-topk 1   --speculative-num-draft-tokens 4   --page-size 1   --attention-backend fa3   --spectre-fixed-q-mode parallel   --spectre-require-draft   --spectre-draft-timeout-action fallback   --spectre-recv-timeout-ms 5000   --spectre-initial-recv-timeout-ms 15000   --spectre-failure-threshold 3   --spectre-cooldown-rounds 32   --spectre-retry-min-count 1   --spectre-retry-fail-ratio 0   --spectre-reject-interval 1   --spectre-zmq-addr 127.0.0.1   --spectre-zmq-port $ZMQ_PORT   --disable-radix-cache   --disable-cuda-graph   --disable-overlap-schedule"
```

---

# 10. Drafter 公共启动命令

`--spectre-draft-priority` 必须保留。

```bash
export I2_DRAFT_COMMON="python -m sglang.launch_server   --model-path '$DRAFT_MODEL'   --port $DRAFT_PORT   --context-length $I2_CONTEXT_LENGTH   --max-total-tokens $I2_DRAFT_MAX_TOTAL_TOKENS   --skip-server-warmup   --speculative-algorithm SPECTRE   --spectre-role draft   --speculative-num-steps 3   --speculative-eagle-topk 1   --speculative-num-draft-tokens 4   --spectre-draft-priority   --spectre-max-draft-priority-steps 8   --disable-overlap-schedule   --spectre-zmq-addr 127.0.0.1   --spectre-zmq-port $ZMQ_PORT"
```

---

# 11. launcher 健康检查

使用：

```text
scripts/specstream/run_dedicated_draft_target_baseline.sh
```

设置：

```bash
export SPECSTREAM_TARGET_READY_CMD="curl -fsS http://127.0.0.1:${TARGET_PORT}/health"
export SPECSTREAM_DRAFT_READY_CMD="curl -fsS http://127.0.0.1:${DRAFT_PORT}/health"
export SPECSTREAM_READY_TIMEOUT_S=300
```

benchmark 前必须同时满足：

```text
Target PID alive
Target /health OK
Draft PID alive
Draft /health OK
```

---

# 12. D0：原生 SGLang 单卡串行 Draft→Target Verify 基线

D0 必须在任何 TPC 控制实验之前完成。

它回答的是一个比 “TPC 是否有效” 更基础的问题：

> **传统的单卡串行推测解码，与把 Draft/Verify 放到同一张 GPU 上并行执行相比，到底差多少？**

如果没有 D0，就无法区分：

```text
D2/D3/D4 的收益
到底来自“Draft–Verify 并行化”
还是来自“TPC / grant 控制”本身。
```

## 12.1 D0 的严格定义

D0 使用 **原生 SGLang STANDALONE speculative decoding**：Target 与 Drafter 权重位于同一个 SGLang server 进程、同一张物理 GPU 上，执行语义是传统推测解码：

```text
一轮开始
   ↓
Draft 连续产生 q 个候选 token
   ↓
Draft 阶段结束
   ↓
Target 对候选 token 进行验证
   ↓
接受 / 拒绝
   ↓
下一轮
```

也就是：

```text
Draft  █████
Target      ███████
Draft              █████
Target                  ███████
```

而不是 D2/D3/D4 的目标：

```text
Draft  █████     █████
Target   ███████   ███████
        ↑ 存在共执行/重叠 ↑
```

**D0 不使用：**

```text
SPECTRE remote Drafter
ZMQ
MPS
libsmctrl
TPC mask
TargetGrantRuntime
grant/ACK
SpecStream co-execution controller
```

注意不要把：

```bash
--spectre-fixed-q-mode ordinary
```

当作本节 D0。`ordinary` 是 SPECTRE 自己的执行模式；本节需要的是 **原生 SGLang STANDALONE 单进程传统推测解码**，这样才能作为最干净的“无 Draft–Verify 并行”参考。

## 12.2 D0 公平性参数

D0 与后续 D1–D4 保持：

```text
Target model      = Qwen2.5-7B-Instruct
Draft model       = Qwen2.5-0.5B-Instruct
context length    = 32768
input length      = 16384
output length     = 128
q                 = 4
num steps         = 3
request count     = 200
concurrency       = 8
seed              = 1
radix cache       = disabled
attention backend = fa3
```

D0 是单进程单卡，因此 `--max-total-tokens 200000` 是该 server 的总 token pool；它与 D2/D3/D4 的两个独立进程各自 200K pool 并非字节级等价。正式论文中应额外报告实际峰值 GPU memory，并把“KV pool 参数一致”和“实际显存占用一致”区分开。

## 12.3 D0 运行前必须关闭 MPS 环境影响

D0 不需要 MPS。若之前启动过 MPS，先清理：

```bash
printf 'quit\n' | nvidia-cuda-mps-control 2>/dev/null || true

unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
```

检查：

```bash
pgrep -af nvidia-cuda-mps || true
```

D0 运行过程中不要重新启动 MPS。

## 12.4 定义 D0 启动命令

```bash
export D0_PORT=$TARGET_PORT
export D0_RESULT_ROOT="$RESULT_ROOT/logs/D0_serial_16k_c8"
export D0_BENCH_LOG="$D0_RESULT_ROOT/benchmark.log"

mkdir -p "$D0_RESULT_ROOT" "$RESULT_ROOT/bench"
```

定义原生 STANDALONE server：

```bash
export D0_CMD="python -m sglang.launch_server \
  --model-path '$TARGET_MODEL' \
  --port $D0_PORT \
  --context-length $I2_CONTEXT_LENGTH \
  --max-prefill-tokens $I2_MAX_PREFILL_TOKENS \
  --max-total-tokens $I2_TARGET_MAX_TOTAL_TOKENS \
  --skip-server-warmup \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path '$DRAFT_MODEL' \
  --speculative-num-steps $I2_NUM_STEPS \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens $I2_NUM_DRAFT_TOKENS \
  --page-size 1 \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-overlap-schedule"
```

先打印确认：

```bash
printf '%s\n' "$D0_CMD"
```

必须确认里面**没有**：

```text
SPECTRE
--spectre-role
--spectre-fixed-q-mode
--specstream-smctrl-enabled
--specstream-enabled
--specstream-profile-only
```

## 12.5 启动 D0 server

```bash
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
  bash -c "$D0_CMD" \
  > "$D0_RESULT_ROOT/server.log" 2>&1 &

export D0_PID=$!
echo "D0 PID=$D0_PID"
```

等待 health：

```bash
D0_READY=0
for i in $(seq 1 300); do
  if ! kill -0 "$D0_PID" 2>/dev/null; then
    echo "D0 server exited before readiness"
    tail -200 "$D0_RESULT_ROOT/server.log"
    exit 1
  fi

  if curl -fsS "http://127.0.0.1:${D0_PORT}/health" >/dev/null 2>&1; then
    D0_READY=1
    echo "D0 server is ready"
    break
  fi

  sleep 1
done

if [[ "$D0_READY" != "1" ]]; then
  echo "D0 readiness timeout"
  tail -200 "$D0_RESULT_ROOT/server.log"
  kill "$D0_PID" 2>/dev/null || true
  exit 1
fi
```

## 12.6 运行与 D1–D4 完全一致的 workload

```bash
export D0_BENCH_CMD="BASE_URL='http://127.0.0.1:$D0_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=D0_serial_16k_c8 \
DATASET_NAME=random \
DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 \
OUTPUT_LEN=128 \
NUM_PROMPTS=200 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=8 \
RANGE_RATIO=1 \
WARMUP_REQUESTS=4 \
SEED=1 \
CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

set -o pipefail
PYTHONUNBUFFERED=1 bash -c "$D0_BENCH_CMD" \
  2>&1 | tee "$D0_BENCH_LOG"
```

运行结束后：

```bash
kill "$D0_PID" 2>/dev/null || true
wait "$D0_PID" 2>/dev/null || true
unset D0_PID
```

## 12.7 D0 必须记录的指标

至少保存：

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
peak_gpu_memory
```

同时记录：

```bash
nvidia-smi \
  | tee "$D0_RESULT_ROOT/nvidia_smi_after.txt"
```

## 12.8 D0 的判定条件

D0 必须满足：

```text
server 正常启动
200 requests 正常完成
error_count = 0
无 OOM
无 SPECTRE timeout（因为根本不应该存在 remote Draft）
accept length 合理且非全 1
```

检查日志：

```bash
grep -Eini \
'out of memory|oom|timeout|traceback|error|SPECTRE|specstream-smctrl' \
"$D0_RESULT_ROOT/server.log" \
"$D0_RESULT_ROOT/benchmark.log" \
| tail -200
```

如果 D0 自己都不能稳定完成，则不要开始 D1–D4 横向比较。

## 12.9 D0 与后续实验分别回答什么

```text
D0 → D1
传统单卡串行 vs 两卡并行 SPECTRE：
“理想增加一张 Drafter GPU 能换来多少性能？”

D0 → D2
传统单卡串行 vs 单卡 naive 共执行：
“仅仅把 Draft/Verify 并行化是否值得？”

D2 → D3
naive 共执行 vs static TPC：
“固定 TPC 限制是否真的缓解同卡干扰？”

D3 → D4
固定 TPC vs online grant：
“动态 Target-priority 控制是否比静态限制更好？”

D0 → D4
传统单卡串行 vs 最终 I2：
“创新点二最终有没有超过传统单卡 speculative decoding？”
```

因此，**D0 不是辅助结果，而是创新点二因果链的起点。**

---

# 13. Gate A：单元测试

先跑新版互补分区单测：

```bash
PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_complementary_tpc_partition.py
```

如果本地工作树中仍保留 first-grant 专项测试，再执行：

```bash
if [[ -f python/sglang/test/spectre_specstream/test_first_grant_gate.py ]]; then
  PYTHONPATH=python pytest -q \
    python/sglang/test/spectre_specstream/test_first_grant_gate.py
else
  echo "test_first_grant_gate.py not present; first-grant will be verified by Gate D runtime smoke"
fi
```

然后跑全部 SpecStream 测试：

```bash
PYTHONPATH=python pytest -q python/sglang/test/spectre_specstream
```

有失败：

```text
STOP
```

---

# 14. D1：双卡 SPECTRE q=4

D1 使用两张独立 GPU，不需要 MPS，也不使用 libsmctrl/TPC。为避免单卡实验残留影响 D1，运行前先清理 MPS 环境：

```bash
printf 'quit\n' | nvidia-cuda-mps-control 2>/dev/null || true
unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
```

然后再启动 D1。

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$TARGET_GPU"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$DRAFT_GPU"

export SPECSTREAM_TARGET_CMD="$I2_TARGET_COMMON"
export SPECSTREAM_DRAFT_CMD="$I2_DRAFT_COMMON"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D1_16k_c8"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' TARGET_MODEL='$TARGET_MODEL' CASE_TAG=D1_16k_c8 DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 CONTEXT_LEN=32768 OUTPUT_DIR='$RESULT_ROOT/bench' bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

D1 是两卡绝对吞吐强基线。

---


# 15. Gate B：libsmctrl 与互补 mask 验证（D1 完成后，为 D2–D4 单卡阶段准备）

构建：

```bash
cd "$REPO/csrc/specstream_smctrl"
make config
make build
```

设置：

```bash
export SMCTRL_LIB=$REPO/csrc/specstream_smctrl/build/libsmctrl.so
ls -lh "$SMCTRL_LIB"
sha256sum "$SMCTRL_LIB" | tee "$RESULT_ROOT/smctrl/libsmctrl.sha256"
```

先读取实际 TPC 数：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
SGLANG_SPECSTREAM_SMCTRL_LIBRARY="$SMCTRL_LIB" \
python - <<'PY_INNER_15'
from sglang.srt.speculative.spectre.specstream.sm_controller import SMController
c = SMController(mask_scope="global")
print("total_tpcs =", c.total_tpcs)
PY_INNER_15
```

根据真实输出设置，例如 A800 可能是：

```bash
export TOTAL_TPCS=54
```

**不要机械复制 54。**

旧版只验证 Drafter 前缀 `[0,k)` 已经不够。互补分区必须同时验证：

```text
Draft prefix = [0,k)
Target suffix = [k,N)
```

先以 `k=4` 验证：

```bash
cd "$REPO/csrc/specstream_smctrl"

CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
make validate-global TPC_LOW=0 TPC_HIGH=4

CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
make validate-global TPC_LOW=4 TPC_HIGH="$TOTAL_TPCS"
```

还建议验证 Target full-range 恢复：

```bash
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
make validate-global TPC_LOW=0 TPC_HIGH="$TOTAL_TPCS"
```

正式扫描 `k=2,4,6,8,12` 前，对每个候选都验证前缀与互补后缀：

```bash
for TPC in 2 4 6 8 12; do
  (( TPC < TOTAL_TPCS )) || continue
  echo "=== validate Draft [0,$TPC) ==="
  CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
    make validate-global TPC_LOW=0 TPC_HIGH="$TPC"

  echo "=== validate Target [$TPC,$TOTAL_TPCS) ==="
  CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" \
    make validate-global TPC_LOW="$TPC" TPC_HIGH="$TOTAL_TPCS"
done
```

任意必需范围失败：

```text
STOP
```

不能只验证 Draft mask 成功，就宣称空间分区可用。

---

# 16. Gate C：MPS（从 D2 开始启用）

```bash
export CUDA_MPS_PIPE_DIRECTORY="/tmp/specstream-mps-${USER}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/specstream-mps-log-${USER}"

mkdir -p   "$CUDA_MPS_PIPE_DIRECTORY"   "$CUDA_MPS_LOG_DIRECTORY"
```

启动：

```bash
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" nvidia-cuda-mps-control -d
```

检查：

```bash
echo get_server_list | nvidia-cuda-mps-control
pgrep -af nvidia-cuda-mps
```

CUDA client 尚未连接时 `get_server_list` 可以为空。

全部实验结束后：

```bash
printf 'quit\n' | nvidia-cuda-mps-control
```

---

# 17. D2：同卡无 TPC 控制

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$I2_TARGET_COMMON"
export SPECSTREAM_DRAFT_CMD="$I2_DRAFT_COMMON"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D2_16k_c8"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' TARGET_MODEL='$TARGET_MODEL' CASE_TAG=D2_16k_c8 DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 CONTEXT_LEN=32768 OUTPUT_DIR='$RESULT_ROOT/bench' bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

D2 的“无控制”只表示：

```text
无 TPC 计算隔离
```

并不表示允许两个 SGLang 使用无限制显存池。

---

# 18. D0 / D1 / D2 如何解释

首先必须看：

```text
D0 vs D2
```

其中：

```text
D0 = 单卡传统串行 Draft→Verify
D2 = 单卡无 TPC 控制的 Draft–Verify 共执行
```

这组比较直接回答“并行化本身是否值得”。只有 D2 明显优于 D0，或者 D2 虽受干扰但 D3/D4 能恢复并超过 D0，创新点二的端到端价值链才完整。

然后再用 D1 作为两卡并行的绝对吞吐参考。

双卡近似：

```text
max(Tdraft, Tverify)
```

同卡无控制可能接近：

```text
Tdraft + Tverify + interference
```

因此 D2 raw throughput 只有 D1 的 50%~70% 不一定异常。

必须同时报告：

```text
D1 throughput/GPU = D1 raw throughput / 2
D2 throughput/GPU = D2 raw throughput / 1
```

真正异常信号：

```text
raw throughput 下降数倍
P99 出现约 5000ms/10000ms 阶梯
remote_draft_timeout 大量出现
missing_drafts > 0
Draft 退出
OOM
```

---

# 19. Gate D：first-grant + complementary-partition smoke

先定义 Target/Drafter calibration 公共命令。

**与旧文档最大的区别：Target 现在也必须加载 `libsmctrl`，使用 global mask，并显式打开 complementary partition。**

```bash
export CAL_TARGET_COMMON="$I2_TARGET_COMMON \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library '$SMCTRL_LIB' \
  --specstream-smctrl-mask-scope global \
  --specstream-smctrl-complementary-partition \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200"

export CAL_DRAFT_COMMON="$I2_DRAFT_COMMON \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library '$SMCTRL_LIB' \
  --specstream-smctrl-mask-scope global"
```

TPC=4：

```bash
export TPC=4
export GATE_PROFILE="$RESULT_ROOT/profiles/gate_q4_tpc4_complementary.csv"

export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-profile-path '$GATE_PROFILE'"

export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/gate_q4_tpc4_complementary"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=gate_q4_tpc4_complementary \
DATASET_NAME=random \
DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 \
OUTPUT_LEN=32 \
NUM_PROMPTS=4 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=1 \
RANGE_RATIO=1 \
WARMUP_REQUESTS=1 \
SEED=1 \
CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

fixed-TPC bootstrap 读取 Drafter 自己的 server args，因此 Drafter 必须收到相同的：

```bash
--specstream-smctrl-calibration-tpcs $TPC
```

Target 则必须同时收到：

```bash
--specstream-smctrl-calibration-tpcs $TPC
--specstream-smctrl-calibration-allow-overlap
--specstream-smctrl-complementary-partition
```

这样 Gate D 验证的才是**真正的互补空间分区启动链路**，而不是只验证 Drafter mask。

---

# 20. Gate D 成功条件

Drafter 日志应看到与下列语义一致的信息：

```text
SpecStream Draft TPC control initialized
calibration bootstrap grant active before first Draft forward
Draft TPC range [0,4)
```

Target 侧应成功初始化 complementary controller，并在真实 overlap round 中使用：

```text
Target TPC range [4,TOTAL_TPCS)
```

如果 A800 当前实际 `TOTAL_TPCS=54`，则必须能在 profile 中找到真实重叠 round：

```text
draft_tpc_low   = 0
draft_tpc_high  = 4
target_tpc_low  = 4
target_tpc_high = 54
```

并且非 overlap / Target-exclusive round 应能看到 Target full range：

```text
target_tpc_low  = 0
target_tpc_high = 54
```

必须满足：

```text
ungranted Draft forward = 0
no TPC mask is active = 0
Draft exited before readiness = 0
Target smctrl init failure = 0
Draft/Target mask overlap = 0
Target range 未恢复 = 0
timeout storm = 0
OOM = 0
benchmark 正常完成
```

检查日志：

```bash
grep -Eini \
'bootstrap grant|complementary|TPC|grant|mask|ungranted|timeout|missing|oom|traceback|error' \
"$SPECSTREAM_RESULT_ROOT/draft.log" \
"$SPECSTREAM_RESULT_ROOT/target.log" \
| tail -300
```

检查 CSV：

```bash
python - <<'PY_INNER_20'
import csv, os
p=os.environ['GATE_PROFILE']
with open(p, newline='') as f:
    rows=list(csv.DictReader(f))
print('rows =', len(rows))
for r in rows[:20]:
    print(
        'grant=', r.get('grant_state'),
        'draft=', (r.get('draft_tpc_low'), r.get('draft_tpc_high')),
        'target=', (r.get('target_tpc_low'), r.get('target_tpc_high')),
    )
PY_INNER_20
```

如果仍有：

```text
SpecStream refused an ungranted Draft forward
```

或者 overlap round 中仍是：

```text
Draft  [0,4)
Target [0,54)
```

立即停止，不进入 calibration。

---

# 21. TPC 可行性扫描：从“Draft 限速”升级到“真正互补分区”

Gate D 通过后，第一阶段只扫：

```text
context = 16K
concurrency = 8
q = 4
TPC k = 2,4,6,8,12
rep = 1,2,3
```

只保留：

```text
0 < k < TOTAL_TPCS
```

对每个 `k`，主 calibration 至少测两类：

```text
A. wait-only / no-overlap baseline
   Draft 可以使用固定 [0,k)，但只在 Target forward 完成后推进
   Target 始终 [0,N)

B. complementary overlap
   SLACK_FILL 时：
   Draft  [0,k)
   Target [k,N)
```

如果你要完整解释旧实验“有 TPC/无 TPC基本不变”，建议再保留一个**可选机制消融**：

```text
C. legacy Draft-only overlap（D3-T）
   Draft  [0,k)
   Target [0,N)
```

这样能得到最有解释力的三阶段因果链：

```text
D2 naive:              Draft [0,N), Target [0,N)
D3-T throttle-only:    Draft [0,k), Target [0,N)
D3 complementary:      Draft [0,k), Target [k,N)
```

### 21.1 可选 D3-T：旧式 Draft-only throttling 精确复现

如果需要保留旧结果作为机制消融，可定义一个**不带 complementary flag** 的 Target：

```bash
export LEGACY_TARGET_COMMON="$I2_TARGET_COMMON \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200"
```

Drafter 仍使用真实 global Draft mask：

```bash
export LEGACY_DRAFT_COMMON="$I2_DRAFT_COMMON \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library '$SMCTRL_LIB' \
  --specstream-smctrl-mask-scope global"
```

以 `k=4` 为例：

```bash
export SPECSTREAM_TARGET_CMD="$LEGACY_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs 4 \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-profile-path '$RESULT_ROOT/profiles/D3_legacy_throttle_16k_c8.csv'"

export SPECSTREAM_DRAFT_CMD="$LEGACY_DRAFT_COMMON \
  --specstream-smctrl-calibration-tpcs 4"
```

这组消融的正确语义必须是：

```text
Draft  = [0,4)
Target = [0,N)
```

它只能命名为 `legacy throttle` / `D3-T`，不能再称为 spatial partition。

安全点必须同时满足：

```text
Target slowdown <= 5%
Draft step 有有效进展
error_count = 0
timeout ≈ 0
真实 overlap round 满足 Draft ∩ Target = ∅
```

如果互补分区仍几乎无收益，不要继续把问题归因于“没有真正做空间隔离”；此时应转而验证 HBM/L2/内存控制器竞争或 Drafter 本身太轻。

---

# 22. Calibration baseline：wait-only / Target full TPC

以 `TPC=4, REP=1` 为例：

```bash
export TPC=4
export REP=1
export BASE_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_base.csv"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC \
  --specstream-profile-path '$BASE_PROFILE'"

export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_base"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=cal_q4_tpc${TPC}_rep${REP}_base \
DATASET_NAME=random \
DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 \
OUTPUT_LEN=128 \
NUM_PROMPTS=200 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=8 \
RANGE_RATIO=1 \
WARMUP_REQUESTS=4 \
SEED=$REP \
CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

这里故意**不加**：

```bash
--specstream-smctrl-calibration-allow-overlap
```

因此该 run 用来测量同样 workload、同样 `k` 配置下的 Target baseline。即使 Target 启用了 complementary-capable 代码，也不应该存在重叠 SLACK_FILL；Target 应保持：

```text
Target = [0,TOTAL_TPCS)
```

这个 baseline 才能作为后续 Target slowdown 的分母。

---

# 23. Calibration overlap：固定 k 的真正互补空间分区

```bash
export OVER_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_complementary.csv"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-profile-path '$OVER_PROFILE'"

export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_COMMON \
  --specstream-smctrl-calibration-tpcs $TPC"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_complementary"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=cal_q4_tpc${TPC}_rep${REP}_complementary \
DATASET_NAME=random \
DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 \
OUTPUT_LEN=128 \
NUM_PROMPTS=200 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=8 \
RANGE_RATIO=1 \
WARMUP_REQUESTS=4 \
SEED=$REP \
CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

每个 overlap run 必须：

```text
ungranted Draft forward = 0
missing_drafts ≈ 0
timeout ≈ 0
Draft 结束仍健康
TPC mask/grant 记录非空
```

而且关键 profile 证据必须成立。对 `TOTAL_TPCS=54, TPC=4`：

```text
SLACK_FILL overlap round:
Draft  = [0,4)
Target = [4,54)
```

也就是：

```text
draft_tpc_low=0
draft_tpc_high=4
target_tpc_low=4
target_tpc_high=54
```

如果 `target_tpc_low=0` 且 `target_tpc_high=54` 出现在本应重叠的 SLACK_FILL round，就说明没有真正进入 complementary partition。

---

# 24. 统计真实 runtime shape 与真实 partition 证据

不要预设：

```text
verify_bs8_q4_ctx16k
```

因为：

```text
MAX_CONCURRENCY=8 ≠ 每轮 batch_size=8
```

而且输入 16384 后 decode/verification 很可能进入下一 context bucket。

执行：

```bash
export OVER_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc4_rep1_complementary.csv"

python - <<'PY_INNER_24'
import csv, os
from collections import Counter

p=os.environ['OVER_PROFILE']
with open(p, newline='') as f:
    rows=list(csv.DictReader(f))

print('rows =', len(rows))

for key in (
    'batch_size','q','context_tokens','grant_state','fallback_reason',
    'draft_tpc_low','draft_tpc_high','target_tpc_low','target_tpc_high'
):
    if rows and key in rows[0]:
        vals=[r.get(key) for r in rows if r.get(key) not in (None,'')]
        print(key, Counter(vals).most_common(20))

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

if rows and 'context_tokens' in rows[0]:
    vals=[bucket(r['context_tokens']) for r in rows if r.get('context_tokens')]
    print('context_bucket =', Counter(vals).most_common())

bad=[]
checked=0
for i,r in enumerate(rows):
    try:
        dl=int(float(r.get('draft_tpc_low','-1')))
        dh=int(float(r.get('draft_tpc_high','-1')))
        tl=int(float(r.get('target_tpc_low','-1')))
        th=int(float(r.get('target_tpc_high','-1')))
    except Exception:
        continue
    if dl >= 0 and dh > dl and tl >= 0 and th > tl:
        checked += 1
        if max(dl,tl) < min(dh,th):
            bad.append((i,dl,dh,tl,th,r.get('grant_state')))
print('partition rows checked =', checked)
print('overlap violations =', len(bad))
print('first violations =', bad[:10])
PY_INNER_24
```

正式结果必须能说明：

```text
真实 bs/q/context shape 是什么
哪些 round 进入了 SLACK_FILL
这些 round 中 Draft/Target 是否真的互斥
Target 是否在 forward 完成后恢复 full TPC
```

不要只从启动命令推断“应该分区了”。

---

# 25. Resource profile 覆盖原则

统计：

```text
(batch_size, q, context_bucket)
```

按真实 round 频率排序。例如：

```text
(bs8,q4,ctx32k) 61%
(bs7,q4,ctx32k) 12%
(bs6,q4,ctx32k)  9%
(bs4,q4,ctx32k)  8%
其它              10%
```

优先校准高频 shape，目标：

```text
累计 round coverage >= 95%
```

每一个用于在线 `SLACK_FILL` 的 profile entry 除了要有：

```text
Target baseline latency
Target overlap latency
Draft step latency
safe k
Target slowdown
```

还必须已经在真实运行中验证：

```text
Draft  = [0,k)
Target = [k,N)
```

未覆盖 shape：

```text
fail closed
→ TARGET_EXCLUSIVE
```

禁止未实测 shape 的最近邻 TPC 插值，也禁止把旧式 Draft-only throttling 的 slowdown 数据直接拿来构建新版 complementary resource profile。

---

# 26. 构建 measured resource profile

如果仓库脚本支持：

```bash
python scripts/specstream/extract_resource_profile.py \
  --input "$RESULT_ROOT/profiles" \
  --output "$RESULT_ROOT/resource_profiles/i2_q4_complementary.json"
```

实际参数以：

```bash
python scripts/specstream/extract_resource_profile.py --help
```

为准。

然后：

```bash
export RESOURCE_PROFILE="$RESULT_ROOT/resource_profiles/i2_q4_complementary.json"
test -f "$RESOURCE_PROFILE" && echo "resource profile OK"
```

在生成 profile 前先检查输入 CSV 包含新版字段：

```bash
python - <<'PY_INNER_26'
import csv, glob
files=glob.glob('results/innovation2/profiles/*complementary*.csv')
if not files:
    raise SystemExit('no complementary calibration CSV found')
with open(files[0], newline='') as f:
    fields=csv.DictReader(f).fieldnames or []
required={'draft_tpc_low','draft_tpc_high','target_tpc_low','target_tpc_high'}
print('file =', files[0])
print('missing =', sorted(required-set(fields)))
PY_INNER_26
```

新版 resource profile 只能由**真正互补分区的 calibration 数据**构建。

---

# 27. D3：最佳固定 k 的互补 TPC 空间分区

从 complementary calibration 中选择：

```text
Target slowdown <=5%
```

范围内 Draft step latency 最低或 Pareto 最优的 `k`。

例如：

```bash
export BEST_TPC=4
```

Target：

```bash
export D3_TARGET_CMD="$CAL_TARGET_COMMON \
  --specstream-smctrl-calibration-tpcs $BEST_TPC \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-profile-path '$RESULT_ROOT/profiles/D3_complementary_16k_c8.csv'"
```

Drafter：

```bash
export D3_DRAFT_CMD="$CAL_DRAFT_COMMON \
  --specstream-smctrl-calibration-tpcs $BEST_TPC"
```

运行：

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$D3_TARGET_CMD"
export SPECSTREAM_DRAFT_CMD="$D3_DRAFT_CMD"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D3_complementary_16k_c8"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=D3_complementary_16k_c8 \
DATASET_NAME=random \
DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 \
OUTPUT_LEN=128 \
NUM_PROMPTS=200 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=8 \
RANGE_RATIO=1 \
WARMUP_REQUESTS=4 \
SEED=1 \
CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

D3 不再表示“Drafter 固定 4 TPC、Target 不受限”。D3 的正式定义是：

```text
SLACK_FILL:
Draft  = [0,BEST_TPC)
Target = [BEST_TPC,TOTAL_TPCS)

Target forward 完成：
Target = [0,TOTAL_TPCS)
```

对 A800 `N=54, BEST_TPC=4`，必须验证 profile：

```text
draft=[0,4)
target=[4,54)
```

如果你保留旧式 Draft-only 结果，应命名为 `D3-T` 或 `legacy_throttle`，不能与新版 D3 混淆。

---

# 28. D4 前的在线 grant + complementary partition Gate

D4 与 fixed-TPC calibration 不同。

D4 前必须确认当前在线链路形成：

```text
Target create grant
↓
epoch monotonic
↓
SLACK_FILL decision 给出 Draft [0,k)
↓
Drafter acquire + apply [0,k)
↓
Target 从同一个 outstanding decision 推导 [k,N)
↓
Target/Draft 真正互补执行
↓
Draft exactly one token
↓
Target forward 完成后恢复 [0,N)
↓
ACK
↓
next epoch
```

至少要记录：

```text
grant_state
grant_epoch
grant_wait_ms
draft_tpc_low
draft_tpc_high
target_tpc_low
target_tpc_high
draft_step_ms
ACK
fallback
```

三种状态的空间语义必须区分：

```text
TARGET_EXCLUSIVE:
Target=[0,N), Draft 不执行

SLACK_FILL:
Draft=[0,k), Target=[k,N)

DRAFT_CATCHUP:
只有 Target CUDA forward 已完成后才能发放；此时不要求 Target/Draft 同时占用互补区，因为 Target 已离开关键 forward。
```

如果 online grant 能发但 Target 端没有真实 complementary mask：

```text
STOP D4/D5
```

D3 可以继续作为 fixed complementary baseline，但不能把缺少 Target mask 的结果称为完整 online I2。

---

# 29. D4：online Target-priority one-token grant + complementary TPC

D4 Target 保持 GPU-resident KV，因此使用：

```text
--specstream-profile-only
```

Target 现在也必须显式加载 libsmctrl/global mask，并开启 complementary partition：

```bash
export D4_TARGET_CMD="$I2_TARGET_COMMON \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library '$SMCTRL_LIB' \
  --specstream-smctrl-mask-scope global \
  --specstream-smctrl-complementary-partition \
  --specstream-grant-token-quantum 1 \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --specstream-coexec-resource-profile-path '$RESOURCE_PROFILE' \
  --specstream-profile-path '$RESULT_ROOT/profiles/D4_complementary_16k_c8.csv'"
```

Drafter：

```bash
export D4_DRAFT_CMD="$I2_DRAFT_COMMON \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library '$SMCTRL_LIB' \
  --specstream-smctrl-mask-scope global"
```

D4 是在线 grant，因此 **不要**给 Target 或 Drafter 写死：

```bash
--specstream-smctrl-calibration-tpcs
```

否则会污染在线策略。

运行：

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$D4_TARGET_CMD"
export SPECSTREAM_DRAFT_CMD="$D4_DRAFT_CMD"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D4_complementary_16k_c8"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' \
TARGET_MODEL='$TARGET_MODEL' \
CASE_TAG=D4_complementary_16k_c8 \
DATASET_NAME=random \
DATASET_PATH='$SHAREGPT_JSON' \
INPUT_LEN=16384 \
OUTPUT_LEN=128 \
NUM_PROMPTS=200 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=8 \
RANGE_RATIO=1 \
WARMUP_REQUESTS=4 \
SEED=1 \
CONTEXT_LEN=32768 \
OUTPUT_DIR='$RESULT_ROOT/bench' \
bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

D4 的正式含义是：**在线控制器决定什么时候允许 overlap 和允许多大的 Draft `[0,k)`，Target 在同一个 overlap 窗口自动使用互补 `[k,N)`。**

---

# 30. D4 结果必须检查什么

```bash
python scripts/specstream/summarize_specstream_profile.py \
  "$RESULT_ROOT/profiles/D4_complementary_16k_c8.csv"
```

重点：

```text
grant_state
grant_epoch
grant_wait_ms
draft_step_ms
draft_tpc_low
draft_tpc_high
target_tpc_low
target_tpc_high
target_forward_ms
fallback
fallback_reason
missing_drafts
```

正常 D4 应出现：

```text
TARGET_EXCLUSIVE
SLACK_FILL
DRAFT_CATCHUP
```

且满足：

```text
grant_epoch 单调
每 grant 恰好一个 Draft token
ACK 前无下一 token grant
draft_tpc_high > draft_tpc_low
draft_step_ms > 0
```

对每个真正 `SLACK_FILL` overlap round，必须验证：

```text
Draft=[dl,dh)
Target=[tl,th)
max(dl,tl) >= min(dh,th)
```

在当前 prefix/suffix 设计中进一步要求：

```text
dl = 0
tl = dh
th = TOTAL_TPCS
```

即：

```text
Draft  [0,k)
Target [k,N)
```

Target forward 结束后后续非 overlap round 应恢复：

```text
Target [0,N)
```

如果：

```text
TARGET_EXCLUSIVE ≈ 100%
```

先检查 resource-profile coverage，而不是直接判断 online controller 无效。

如果大量 `SLACK_FILL` 但 `target_tpc_low=0`，说明 online grant 虽存在，但真正空间隔离没有生效，该 run 不能作为 D4 正式结果。

---

# 31. D5：完整 I1+I2

只有 D4 在线链路完全正确后才进入 D5。

D5 startup/default：

```text
q=4
```

动态候选：

```text
1,2,4,6,8
```

D5 Target 除创新点一的 KV streaming 参数外，必须继续保留创新点二的：

```bash
--specstream-smctrl-enabled
--specstream-smctrl-library "$SMCTRL_LIB"
--specstream-smctrl-mask-scope global
--specstream-smctrl-complementary-partition
--specstream-grant-token-quantum 1
--specstream-coexec-resource-profile-path "$RESOURCE_PROFILE"
```

Drafter 保留：

```bash
--specstream-smctrl-enabled
--specstream-smctrl-library "$SMCTRL_LIB"
--specstream-smctrl-mask-scope global
```

正式 D5 前必须按 pilot 的真实 q 分布补齐高频：

```text
q
batch_size
context_bucket
```

目标：

```text
resource-profile coverage >=95%
```

而且这些 profile 必须来自新版 complementary calibration，不能混入旧 Draft-only throttling 数据。

否则 D5 只能作为 debug 结果。

---

# 32. workload 扩展

先只完成：

```text
16K / C8 / q4
```

然后扩展：

```text
Context:
4K
8K
16K
30K

Concurrency:
1
4
8
16
32
```

D0–D4 必须保持同一：

```text
input length
output length
q
seed
request count
token-pool policy
```

---

# 33. P99 rate sweep

建议 Poisson arrival rate：

```text
0.5
1
2
4
8 req/s
```

报告：

```text
P99 TTFT
P99 TPOT
P99 E2E
Goodput
timeout/error rate
```

不要只用 `REQUEST_RATE=inf` 描述尾延迟。

---

# 34. screening 与正式确认性实验

开发阶段：

```text
NUM_PROMPTS ≈ 200
rep = 1~3
```

最终论文主点：

```text
>=1000 requests / run
5 independent runs
```

4-request smoke、32-request debug、200-request screening 都不作为最终主结果。

---

# 35. 统一指标

基础 serving 指标：

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

创新点二额外：

```text
physical GPUs
output throughput / GPU
Target slowdown
Target forward latency
Draft step latency
grant state distribution
grant epoch
grant wait
draft_tpc_low
draft_tpc_high
target_tpc_low
target_tpc_high
partition violation count
missing drafts
fallback rate
```

其中下面四个字段是新版空间隔离最关键的证据：

```text
draft_tpc_low
draft_tpc_high
target_tpc_low
target_tpc_high
```

论文中不能只写“我们设置了 TPC=4”，而应证明真实 overlap round 中：

```text
Draft=[0,4)
Target=[4,N)
```

---

# 36. throughput/GPU

```text
D0:
throughput_per_gpu = raw_throughput / 1

D1:
throughput_per_gpu = raw_throughput / 2

D2/D3/D4:
throughput_per_gpu = raw_throughput / 1
```

必须同时报告 raw throughput 与 throughput/GPU。

不能用 throughput/GPU 隐藏 D1 的绝对吞吐优势，也不能只用 raw throughput 忽略单卡节省了一张 GPU。

---

# 37. Target slowdown 与安全 Pareto 点

定义：

```text
Target slowdown =
T_target_complementary_overlap / T_target_wait_only_baseline - 1
```

这里 baseline 与 overlap 必须保持：

```text
相同模型
相同 q
相同 workload
相同 REP/seed
相同 k
相同 KV pool
```

唯一核心差异是：

```text
baseline:
Target=[0,N), Draft 不与 Target forward 重叠

overlap:
Draft=[0,k), Target=[k,N)
```

正式安全预算：

```text
<=5%
```

如果所有 `k` 都 >5%：

```text
Innovation-2 No-Go for this shape/hardware
```

不要临时放宽到 10% 后继续宣称原目标成立。

如果所有 `k` 的 slowdown 都接近 0，且 D2/D3/D4 端到端也基本相同，则要把结论解释为：

> 该 workload 当前不是显著 SM/TPC contention region；下一步应测 Drafter 压力、HBM/L2、模型大小和并发，而不是继续无限扫 k。

---

# 38. 每次 run 后的故障扫描

```bash
grep -Eini 'out of memory|oom|timeout|missing|fallback|ungranted|mask|grant|traceback|error' "$SPECSTREAM_RESULT_ROOT/target.log" "$SPECSTREAM_RESULT_ROOT/draft.log" "$SPECSTREAM_RESULT_ROOT/benchmark.log" | tail -200
```

特别关注 5000ms/10000ms 阶梯，因为：

```bash
--spectre-recv-timeout-ms 5000
```

若 P99 呈 5 秒整数倍，优先查 Draft timeout/missing Draft，而不是解释为正常同卡竞争。

---

# 39. GPU/MPS 运行时检查

同卡运行期间：

```bash
watch -n 1 nvidia-smi
```

另一个终端：

```bash
echo get_server_list | nvidia-cuda-mps-control
pgrep -af 'launch_server|nvidia-cuda-mps'
```

必须确认：

```text
Target + Drafter 都存活
两者都绑定同一物理 GPU
显存没有接近耗尽
MPS client 正常
```

---

# 40. 避免旧 shell 变量污染

每个新阶段建议先清理：

```bash
unset CAL_TARGET_COMMON
unset CAL_DRAFT_COMMON
unset CAL_DRAFT_CMD

unset SPECSTREAM_TARGET_CMD
unset SPECSTREAM_DRAFT_CMD
unset SPECSTREAM_TARGET_VISIBLE_DEVICES
unset SPECSTREAM_DRAFT_VISIBLE_DEVICES
unset SPECSTREAM_RESULT_ROOT
unset SPECSTREAM_BENCH_CMD
```

然后重新按本文档定义。

运行前打印：

```bash
printf '%s
' "$SPECSTREAM_TARGET_CMD"
printf '%s
' "$SPECSTREAM_DRAFT_CMD"
```

---

# 41. 禁止出现的实验错误

不要：

```text
D1 q5 / D4 q4
```

不要：

```text
同卡两个 SGLang 都使用默认超大 KV pool
```

不要：

```text
Draft 只 sleep 5 秒，不检查 /health
```

不要：

```text
只校准 bs8/q4/ctx16k 就认为覆盖 16K/C8
```

不要：

```text
D5 q=1/2/4/6/8，但 resource profile 只有 q4
```

不要：

```text
no grant 仍放行 Draft forward
```

不要：

```text
为了跑通删除 tp_worker fail-closed guard
```

不要：

```text
把 MPS active-thread percentage 当成正式调度器
```

新版还特别禁止：

```text
只设置 Draft=[0,k)，却让 Target=[0,N)，然后把结果叫 spatial partitioning
```

不要：

```text
Target 启用了 --specstream-smctrl-complementary-partition，
却没有 --specstream-smctrl-mask-scope global
```

不要：

```text
只验证 validate-global [0,k)，不验证 Target 的 [k,N)
```

不要：

```text
把旧 Draft-only calibration 数据生成的 resource profile 用到新版 D4
```

不要：

```text
在单进程 D0 STANDALONE 上套用这套 process-global Target/Draft 互补逻辑
```

---

# 42. 主表推荐

| Method | GPUs | Execution | TPC Semantics | Raw Output Throughput | Throughput/GPU | P99 TTFT | P99 TPOT | P99 E2E | Target Slowdown |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| **D0 SGLang STANDALONE** | 1 | **Serial Draft→Verify** | none | | | | | | N/A |
| D1 2-GPU SPECTRE | 2 | Parallel / separate GPUs | no same-GPU contention | | | | | | |
| D2 Same-GPU uncontrolled | 1 | Parallel / naive | Draft=`[0,N)`, Target=`[0,N)` | | | | | | |
| D3-T Legacy throttle（可选） | 1 | Parallel / Draft-only | Draft=`[0,k)`, Target=`[0,N)` | | | | | | |
| **D3 Complementary fixed-k** | 1 | Parallel / fixed-k spatial partition | **Draft=`[0,k)`, Target=`[k,N)`** | | | | | | |
| **D4 Online Target-priority** | 1 | Online one-token grant | **SLACK_FILL complementary** | | | | | | |

D0 是单卡因果基线；D1 是双卡绝对吞吐参考；D3-T 是可选消融；D3/D4 才是新版空间分区主结果。D5 放完整系统表，不与 I2 isolation 主表混在一起。

---

# 43. 主图推荐

## 图 A：互补 TPC Pareto

横轴：

```text
Target slowdown (%)
```

纵轴：

```text
Draft step latency / Draft token throughput
```

点：

```text
k=2,4,6,8,12
```

每个点明确写：

```text
Draft=[0,k)
Target=[k,N)
```

画 5% slowdown 阈值。

## 图 B：D0–D4 throughput

横轴：

```text
D0 Serial
D1 2-GPU Parallel
D2 Same-GPU Naive
D3-T Draft-only（optional）
D3 Complementary Fixed-k
D4 Online Complementary
```

同时展示：

```text
raw output throughput
output throughput / GPU
```

## 图 C：TPC partition timeline

时间线上同时画：

```text
Target TPC range
Draft TPC range
TARGET_EXCLUSIVE
SLACK_FILL
DRAFT_CATCHUP
grant epoch
Draft token
ACK
```

理想时间片：

```text
Target full:       [0,N)
SLACK_FILL:        Draft [0,k) | Target [k,N)
Target done:       Target 恢复 [0,N)
DRAFT_CATCHUP:     Target forward 已完成后 Draft 推进
```

## 图 D：partition correctness / mode distribution

可画：

```text
SLACK_FILL rounds with disjoint masks = 100%
partition violation = 0
```

以及：

```text
TARGET_EXCLUSIVE %
SLACK_FILL %
DRAFT_CATCHUP %
```

## 图 E：P99

在：

```text
0.5/1/2/4/8 req/s
```

比较：

```text
D0/D1/D2/D3/D4
P99 TTFT/TPOT/E2E
```

---

# 44. 严格执行顺序

```text
[0] 固定 git commit / Python / GPU UUID / workload
        ↓
[1] 基础 import / pytest
        ↓
[2] D0：原生 SGLang STANDALONE，单卡串行 Draft→Verify
        ↓
[3] first-grant + complementary partition 单元测试
        ↓
[4] D1：双卡 SPECTRE parallel q4（此时不启用 MPS）
        ↓
[5] libsmctrl build
        ↓
[6] validate-global Draft [0,k)
        ↓
[7] validate-global Target [k,N)
        ↓
[8] 启动单卡 MPS
        ↓
[9] D2：同卡 q4，无 TPC 控制
        ↓
[10] Gate D：k=4 / C1 / 4 requests，验证 first grant + 双边互补 mask
        ↓
[11] Gate 成功后 complementary TPC feasibility scan
        ↓
[12] 可选 D3-T：旧 Draft-only throttling 消融
        ↓
[13] 统计真实 batch/q/context shape + partition fields
        ↓
[14] 构建 >=95% coverage complementary measured profile
        ↓
[15] D3：最佳 fixed-k complementary partition
        ↓
[16] 验证 online grant/ACK + Target complement
        ↓
[17] D4：online Target-priority complementary partition
        ↓
[18] D4 profile coverage >=95%
        ↓
[19] 扩展 context/concurrency/rate sweep
        ↓
[20] 5×>=1000-request confirmatory runs
        ↓
[21] D5 full I1+I2
```

不要再沿用“先只测 Drafter TPC，再默认 Target 自然受保护”的旧顺序。

---

# 45. Go / No-Go

## Gate 1：代码与环境

D0/D1 前必须：

```text
当前工作树 import 正确
基础 pytest 通过
GPU 可见性正常
```

进入 D2–D4 单卡控制阶段前额外要求：

```text
libsmctrl validate-global Draft prefix 通过
libsmctrl validate-global Target complement suffix 通过
MPS 状态正常
```

失败：STOP。

## Gate 2：first-grant + complementary mask

必须：

```text
k=4 smoke 无 ungranted Draft forward
首个受控 Draft forward 前已有真实 Draft mask
SLACK_FILL overlap round 同时存在 Target=[4,N) mask
partition violation = 0
Target forward 后恢复 [0,N)
```

失败：STOP D3/D4。

## Gate 3：TPC 可行性

至少一个 `k`：

```text
Target slowdown <=5%
Draft step 有进展
error_count=0
Draft/Target overlap mask 互斥
```

失败：当前硬件/shape 下 I2 No-Go。

## Gate 4：online control

必须：

```text
grant epoch 单调
one grant = one Draft token
ACK 前无下一 token
未校准 shape fail closed
SLACK_FILL 用互补 mask
Target slowdown 超预算后 latch exclusive
```

失败：D4 不进入论文。

## Gate 5：端到端价值

D4 相比 D0/D2/D3 至少形成一种稳定价值：

```text
更高 throughput/GPU
更低 P99
更低 Target interference
更稳定高负载 operating region
```

同时公开 D1 两卡 raw throughput。

如果 D2≈D3-T≈D3≈D4，且所有 mask/partition 证据都正确，则结论应转向“当前 workload 不是 SM/TPC contention-bound”，而不是继续修改代码制造差异。

---

# 46. 最小 complementary first-grant 复现清单

只验证当前代码修复时：

```text
1. pytest test_first_grant_gate（如仓库保留该测试）
2. pytest test_complementary_tpc_partition
3. validate-global Draft [0,4)
4. validate-global Target [4,N)
5. MPS ready
6. q=4
7. k=4
8. input=16K
9. output=32
10. requests=4
11. concurrency=1
12. Target/Draft /health 均成功
13. Draft log 无 ungranted forward
14. profile 有 draft_tpc_low/high
15. profile 有 target_tpc_low/high
16. SLACK_FILL 中 Draft=[0,4), Target=[4,N)
17. partition violation=0
18. benchmark 正常结束
```

全部满足后才开始 200-request calibration。

---

# 47. 结果目录建议

```text
results/innovation2/
├── logs/
│   ├── D0_serial_16k_c8/
│   ├── D1_16k_c8/
│   ├── D2_16k_c8/
│   ├── gate_q4_tpc4_complementary/
│   ├── cal_q4_tpc4_rep1_base/
│   ├── cal_q4_tpc4_rep1_complementary/
│   ├── D3_legacy_throttle_16k_c8/        # optional
│   ├── D3_complementary_16k_c8/
│   └── D4_complementary_16k_c8/
├── bench/
├── profiles/
├── resource_profiles/
├── smctrl/
└── source_data/
```

每个最终图表数据点必须能反查：

```text
benchmark JSONL
SpecStream CSV
Target log
Draft log
git commit
launch args
GPU UUID
TOTAL_TPCS
Draft/Target TPC ranges
seed
```

建议额外保存：

```text
validate-global [0,k) 输出
validate-global [k,N) 输出
```

这样论文中的空间分区结果具有可复现实验证据链。

---

# 48. 最终结果解释模板

首先比较：

```text
D0 vs D2
```

如果：

```text
D2 > D0
```

说明：

> 在当前 workload 下，同卡 Draft–Verify 并行本身相较传统串行 speculative decoding 有正收益；后续 TPC 控制是在这个并行收益基础上处理同卡争用。

如果：

```text
D2 ≈ D0
```

说明：

> 当前配置下并行收益与争用大致抵消，必须看 D3 的真正互补空间隔离能否打破这个平衡。

如果：

```text
D2 < D0
```

说明：

> naive 同卡并行产生的资源争用超过并行收益；D3/D4 的目标首先是恢复并超过 D0。

如果保留 D3-T：

```text
D2 ≈ D3-T
但 D3 > D3-T
```

说明：

> 旧版只限制 Drafter 并不足以真正隔离 SM/TPC；Target/Draft 互补空间分区才产生了有效的计算资源隔离。

如果：

```text
D3 > D2
且 Target slowdown <=5%
且 partition violation = 0
```

说明：

> 固定 `k` 的互补空间分区可以缓解同卡 SM/TPC 干扰，并存在安全 Pareto 区域。

如果：

```text
D4 >= D3
或动态负载下 D4 的 P99/Goodput 更优
且 grant/epoch/ACK/partition 全链路正确
```

说明：

> measurement-backed Target-priority online controller 能只在安全时间窗启用互补空间分区，相比固定 `k` 更适应动态 workload。

如果：

```text
D2 ≈ D3-T ≈ D3 ≈ D4
```

同时 profile 已证明：

```text
SLACK_FILL 中 Draft=[0,k), Target=[k,N)
```

则不要再说“代码可能没真正分区”。此时更合理的解释是：

> 当前 `0.5B→7B / A800 80GB / 当前并发与上下文` 很可能不是 SM/TPC contention-bound；需要进一步测 HBM/L2、提高 Drafter 压力、并发或更换模型组合。

如果：

```text
D4 raw throughput < D1
但 D4 throughput/GPU > D1 throughput/GPU
```

仍是合理结果：

> 单卡方案以部分绝对吞吐为代价，消除 dedicated Drafter GPU 资源税，提高单位 GPU 服务效率。

如果所有安全 `k` 都无法满足 5% Target slowdown：

> 如实报告适用边界，不放宽预算包装结果。

---

# 49. 一句话执行原则

> **先用 D0 建立传统单卡串行基线，再用 D2 判断 naive 并行是否值得；随后必须用 profile 证明 `SLACK_FILL` 中 Draft=`[0,k)`、Target=`[k,N)` 的真实互补空间分区，再讨论 fixed-k D3、online D4 与最终性能。先证明“真的隔离了”，再解释“为什么快或为什么不快”。**

---

