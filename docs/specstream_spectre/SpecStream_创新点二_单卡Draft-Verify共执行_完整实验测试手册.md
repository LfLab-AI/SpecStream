# SpecStream 创新点二：单卡 Draft–Verify 安全共执行完整实验测试手册

> **独立版 / 2026-08-26 修订**
>
> 本文档仅覆盖 SpecStream 创新点二：**measurement-backed Target-priority single-GPU Draft–Verify co-execution**。创新点一的 CPU History KV streaming 仅在最终 D5 协同实验中被提及，不在本文展开。
>
> 本版已经吸收当前真实调试中暴露的问题：
>
> - 同卡 Target/Drafter 显式限制 KV pool；
> - D1–D4 统一固定 `q=4`；
> - MPS 只负责两个 CUDA 进程同卡共存，不是最终调度器；
> - `libsmctrl` TPC mask 才是计算资源隔离机制；
> - Target 和 Drafter 都必须通过 `/health`；
> - 修复 `first-grant-before-first-forward` 后才允许做 TPC calibration；
> - fixed-TPC calibration 时 Target 和 Drafter 两端都显式传入相同的 `--specstream-smctrl-calibration-tpcs`；
> - resource profile 必须按真实 `(batch_size, q, context_bucket)` 构建；
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

---

# 0. 创新点二需要证明什么

创新点二不是证明“两个进程能放到一张 GPU 上”，而是证明：

> 在消除 dedicated Drafter GPU 的情况下，通过 **离线实测 resource profile + libsmctrl TPC mask + Target-priority one-token grant/ACK**，让 Draft 利用 Target 的安全空隙推进，同时把 Target slowdown 控制在预算内，从而提高单位 GPU 的服务效率。

最终需要回答四个问题：

1. 直接同卡是否会产生严重干扰？
2. 固定 TPC 隔离是否存在安全 Pareto 点？
3. 在线 Target-priority grant 是否能把 Target slowdown 保持在 5% 预算内？
4. 与两卡 SPECTRE 相比，单卡 raw throughput 损失多少，但 throughput/GPU 与资源效率改善多少？

---

# 1. 最终实验对照定义

| ID | GPU 数 | KV 路径 | Draft–Verify 调度 | 目的 |
|---|---:|---|---|---|
| D0 | 1 | GPU-resident | 原生 SGLang STANDALONE speculative | 单卡原生参考 |
| D1 | 2 | GPU-resident | SPECTRE parallel，Target/Draft 分卡 | 绝对吞吐强基线 |
| D2 | 1 | GPU-resident | SPECTRE parallel，同卡，无 TPC 控制 | 干扰负面对照 |
| D3 | 1 | GPU-resident | same-GPU + 最佳安全固定 TPC | 静态隔离基线 |
| D4 | 1 | GPU-resident | online Target-priority one-token TPC grant | **隔离创新点二** |
| D5 | 1 | bounded KV streaming | online TPC grant + dynamic q + Cohort | I1+I2 完整系统 |

创新点二正文核心比较：

```text
D1 → D2 → D3 → D4
```

D5 只用于最终完整系统协同实验。

---

# 2. 实验统一原则

## 2.1 D1–D4 固定 q=4

统一：

```text
speculative_num_steps = 3
speculative_num_draft_tokens = 4
q = 4
```

不要再使用 D1/D2 q=5、D3/D4 q=4 的不公平配置。

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
+ libsmctrl TPC mask
+ Target-priority one-token grant/ACK
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
Target 发现安全 slack
        ↓
grant(epoch, TPC range, token_budget=1)
        ↓
Drafter 安装 TPC mask
        ↓
执行恰好 1 个 Draft token
        ↓
ACK
        ↓
Target 才允许下一个 grant
```

---

# 3. 开始实验前的代码前置条件

当前 first-grant 修复至少涉及：

```text
修改：
python/sglang/srt/speculative/spectre/drafter/spectre_draft_scheduler_mixin.py
python/sglang/srt/managers/tp_worker.py

新增：
python/sglang/srt/speculative/spectre/specstream/draft_grant_runtime.py

建议新增：
python/sglang/test/spectre_specstream/test_first_grant_gate.py
```

代码必须满足：

```text
smctrl disabled
→ 保持原 SPECTRE priority 行为

smctrl enabled
→ 无 grant 不允许 run_batch()
→ 有 grant 最多推进 1 个受控 Draft token

tp_worker
→ 只有 controlled remote Draft forward 才要求 active mask

fixed-TPC calibration
→ 首个 controlled Draft forward 前必须已有 bootstrap grant + TPC mask
```

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

# 12. Gate A：单元测试

```bash
PYTHONPATH=python pytest -q   python/sglang/test/spectre_specstream/test_first_grant_gate.py
```

然后：

```bash
PYTHONPATH=python pytest -q   python/sglang/test/spectre_specstream
```

有失败：

```text
STOP
```

---

# 13. Gate B：libsmctrl

```bash
cd "$REPO/csrc/specstream_smctrl"
make config
make build
```

设置：

```bash
export SMCTRL_LIB=$REPO/csrc/specstream_smctrl/build/libsmctrl.so

ls -lh "$SMCTRL_LIB"

sha256sum "$SMCTRL_LIB"   | tee "$RESULT_ROOT/smctrl/libsmctrl.sha256"
```

global mask：

```bash
CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" make validate-global TPC_LOW=0 TPC_HIGH=4
```

失败：

```text
STOP
```

读取实际 TPC：

```bash
cd "$REPO"

CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID" SGLANG_SPECSTREAM_SMCTRL_LIBRARY="$SMCTRL_LIB" python - <<'PY'
from sglang.srt.speculative.spectre.specstream.sm_controller import SMController
c = SMController(mask_scope="global")
print("total_tpcs =", c.total_tpcs)
PY
```

根据实际输出设置：

```bash
export TOTAL_TPCS=54
```

不要机械复制 54。

---

# 14. Gate C：MPS

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
printf 'quit
' | nvidia-cuda-mps-control
```

---

# 15. D1：双卡 SPECTRE q=4

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

# 16. D2：同卡无 TPC 控制

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

# 17. D1 vs D2 如何解释

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

# 18. Gate D：first-grant-before-first-forward smoke

先定义：

```bash
export CAL_TARGET_COMMON="$I2_TARGET_COMMON   --specstream-profile-only   --specstream-smctrl-enabled   --specstream-coexec-target-slowdown-budget 0.05   --specstream-coexec-guard-us 200"

export CAL_DRAFT_COMMON="$I2_DRAFT_COMMON   --specstream-smctrl-enabled   --specstream-smctrl-library '$SMCTRL_LIB'   --specstream-smctrl-mask-scope global"
```

TPC=4：

```bash
export TPC=4
export GATE_PROFILE="$RESULT_ROOT/profiles/gate_q4_tpc4.csv"

export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON   --specstream-smctrl-calibration-tpcs $TPC   --specstream-profile-path '$GATE_PROFILE'"

export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_COMMON   --specstream-smctrl-calibration-tpcs $TPC"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/gate_q4_tpc4"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' TARGET_MODEL='$TARGET_MODEL' CASE_TAG=gate_q4_tpc4 DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' INPUT_LEN=16384 OUTPUT_LEN=32 NUM_PROMPTS=4 REQUEST_RATE=inf MAX_CONCURRENCY=1 RANGE_RATIO=1 WARMUP_REQUESTS=1 SEED=1 CONTEXT_LEN=32768 OUTPUT_DIR='$RESULT_ROOT/bench' bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

**注意：fixed-TPC bootstrap 读取 Drafter 自己的 server args，因此 Drafter 也必须收到相同的 `--specstream-smctrl-calibration-tpcs $TPC`。**

---

# 19. Gate D 成功条件

应看到与下列语义一致的日志：

```text
SpecStream Draft TPC control initialized
calibration bootstrap grant active before first Draft forward
TPC range [0,4)
```

必须满足：

```text
ungranted Draft forward = 0
no TPC mask is active = 0
Draft exited before readiness = 0
timeout storm = 0
OOM = 0
benchmark 正常完成
```

检查：

```bash
grep -Eini 'bootstrap grant|TPC|grant|mask|ungranted|timeout|missing|oom|traceback|error' "$SPECSTREAM_RESULT_ROOT/draft.log" "$SPECSTREAM_RESULT_ROOT/target.log" | tail -200
```

如果仍有：

```text
SpecStream refused an ungranted Draft forward
```

立即停止，不进入 calibration。

---

# 20. TPC 可行性扫描

Gate D 通过后，第一阶段只扫：

```text
context = 16K
concurrency = 8
q = 4
TPC = 2,4,6,8,12
rep = 1,2,3
```

只保留：

```text
TPC <= TOTAL_TPCS
```

每个 TPC 都测：

```text
baseline/wait-only
overlap
```

安全点：

```text
Target slowdown <= 5%
Draft step 有有效进展
error_count = 0
timeout ≈ 0
```

---

# 21. Calibration baseline

```bash
export TPC=4
export REP=1

export BASE_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_base.csv"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON   --specstream-smctrl-calibration-tpcs $TPC   --specstream-profile-path '$BASE_PROFILE'"

export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_COMMON   --specstream-smctrl-calibration-tpcs $TPC"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_base"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' TARGET_MODEL='$TARGET_MODEL' CASE_TAG=cal_q4_tpc${TPC}_rep${REP}_base DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=$REP CONTEXT_LEN=32768 OUTPUT_DIR='$RESULT_ROOT/bench' bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

---

# 22. Calibration overlap

```bash
export OVER_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc${TPC}_rep${REP}_overlap.csv"

export SPECSTREAM_TARGET_CMD="$CAL_TARGET_COMMON   --specstream-smctrl-calibration-tpcs $TPC   --specstream-smctrl-calibration-allow-overlap   --specstream-profile-path '$OVER_PROFILE'"

export SPECSTREAM_DRAFT_CMD="$CAL_DRAFT_COMMON   --specstream-smctrl-calibration-tpcs $TPC"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/cal_q4_tpc${TPC}_rep${REP}_overlap"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' TARGET_MODEL='$TARGET_MODEL' CASE_TAG=cal_q4_tpc${TPC}_rep${REP}_overlap DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=$REP CONTEXT_LEN=32768 OUTPUT_DIR='$RESULT_ROOT/bench' bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

每个 calibration run 必须：

```text
ungranted Draft forward = 0
missing_drafts ≈ 0
timeout ≈ 0
Draft 结束仍健康
TPC mask/grant 记录非空
```

---

# 23. 统计真实 runtime shape

不要再预设：

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
export OVER_PROFILE="$RESULT_ROOT/profiles/cal_q4_tpc4_rep1_overlap.csv"

python - <<'PY'
import csv, os
from collections import Counter

p=os.environ["OVER_PROFILE"]
with open(p, newline="") as f:
    rows=list(csv.DictReader(f))

print("rows =", len(rows))

for key in ("batch_size","q","context_tokens","grant_state","fallback_reason"):
    if rows and key in rows[0]:
        vals=[r.get(key) for r in rows if r.get(key) not in (None,"")]
        print(key, Counter(vals).most_common(20))

def bucket(v):
    try:
        n=int(float(v))
    except Exception:
        return "invalid"
    if n <= 4096: return "4k"
    if n <= 8192: return "8k"
    if n <= 16384: return "16k"
    if n <= 32768: return "32k"
    if n <= 65536: return "64k"
    return ">64k"

if rows and "context_tokens" in rows[0]:
    vals=[bucket(r["context_tokens"]) for r in rows if r.get("context_tokens")]
    print("context_bucket =", Counter(vals).most_common())
PY
```

---

# 24. Resource profile 覆盖原则

统计：

```text
(batch_size, q, context_bucket)
```

按真实 round 频率排序。

例如：

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

未覆盖 shape：

```text
fail closed
→ TARGET_EXCLUSIVE
```

禁止未实测 shape 的最近邻 TPC 插值。

---

# 25. 构建 measured resource profile

如果仓库脚本支持：

```bash
python scripts/specstream/extract_resource_profile.py   --input "$RESULT_ROOT/profiles"   --output "$RESULT_ROOT/resource_profiles/i2_q4.json"
```

实际参数以：

```bash
python scripts/specstream/extract_resource_profile.py --help
```

为准。

然后：

```bash
export RESOURCE_PROFILE="$RESULT_ROOT/resource_profiles/i2_q4.json"
test -f "$RESOURCE_PROFILE" && echo "resource profile OK"
```

---

# 26. D3：最佳固定 TPC

从 calibration 选择：

```text
Target slowdown <=5%
```

范围内 Draft step latency 最低或 Pareto 最优的点。

例如：

```bash
export BEST_TPC=4
```

Target：

```bash
export D3_TARGET_CMD="$CAL_TARGET_COMMON   --specstream-smctrl-calibration-tpcs $BEST_TPC   --specstream-smctrl-calibration-allow-overlap   --specstream-profile-path '$RESULT_ROOT/profiles/D3_16k_c8.csv'"
```

Drafter：

```bash
export D3_DRAFT_CMD="$CAL_DRAFT_COMMON   --specstream-smctrl-calibration-tpcs $BEST_TPC"
```

运行：

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$D3_TARGET_CMD"
export SPECSTREAM_DRAFT_CMD="$D3_DRAFT_CMD"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D3_16k_c8"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' TARGET_MODEL='$TARGET_MODEL' CASE_TAG=D3_16k_c8 DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 CONTEXT_LEN=32768 OUTPUT_DIR='$RESULT_ROOT/bench' bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

---

# 27. D4 前的在线 grant Gate

D4 与 fixed-TPC calibration 不同。

D4 前必须确认当前 `coexec_runtime.py` 已经形成完整：

```text
Target create grant
↓
epoch monotonic
↓
Drafter acquire
↓
apply real TPC mask
↓
exactly one Draft token
↓
ACK
↓
next epoch
```

至少要能记录：

```text
grant_state
grant_epoch
grant_wait_ms
draft_tpc_low
draft_tpc_high
draft_step_ms
ACK
fallback
```

如果当前 `coexec_runtime.py` 仍没有可供 Drafter 获取 first online grant 的真实接口：

```text
STOP D4/D5
```

D3 可以继续，但不能把 D3 当成 online I2。

---

# 28. D4：online Target-priority TPC grant

D4 Target 保持 GPU-resident KV，因此使用：

```text
--specstream-profile-only
```

定义：

```bash
export D4_TARGET_CMD="$I2_TARGET_COMMON   --specstream-profile-only   --specstream-smctrl-enabled   --specstream-grant-token-quantum 1   --specstream-coexec-target-slowdown-budget 0.05   --specstream-coexec-guard-us 200   --specstream-coexec-resource-profile-path '$RESOURCE_PROFILE'   --specstream-profile-path '$RESULT_ROOT/profiles/D4_16k_c8.csv'"
```

Drafter：

```bash
export D4_DRAFT_CMD="$I2_DRAFT_COMMON   --specstream-smctrl-enabled   --specstream-smctrl-library '$SMCTRL_LIB'   --specstream-smctrl-mask-scope global"
```

D4 是在线 grant，因此 **不要**给 Drafter 写死：

```bash
--specstream-smctrl-calibration-tpcs
```

否则会把在线策略污染成 persistent fixed mask。

运行：

```bash
export SPECSTREAM_TARGET_VISIBLE_DEVICES="$SINGLE_GPU_UUID"
export SPECSTREAM_DRAFT_VISIBLE_DEVICES="$SINGLE_GPU_UUID"

export SPECSTREAM_TARGET_CMD="$D4_TARGET_CMD"
export SPECSTREAM_DRAFT_CMD="$D4_DRAFT_CMD"

export SPECSTREAM_RESULT_ROOT="$RESULT_ROOT/logs/D4_16k_c8"

export SPECSTREAM_BENCH_CMD="BASE_URL='http://127.0.0.1:$TARGET_PORT' TARGET_MODEL='$TARGET_MODEL' CASE_TAG=D4_16k_c8 DATASET_NAME=random DATASET_PATH='$SHAREGPT_JSON' INPUT_LEN=16384 OUTPUT_LEN=128 NUM_PROMPTS=200 REQUEST_RATE=inf MAX_CONCURRENCY=8 RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 CONTEXT_LEN=32768 OUTPUT_DIR='$RESULT_ROOT/bench' bash scripts/specstream/run_benchmark_case.sh"

bash scripts/specstream/run_dedicated_draft_target_baseline.sh
```

---

# 29. D4 结果必须检查什么

```bash
python scripts/specstream/summarize_specstream_profile.py   "$RESULT_ROOT/profiles/D4_16k_c8.csv"
```

重点：

```text
grant_state
grant_epoch
grant_wait_ms
draft_step_ms
draft_tpc_low
draft_tpc_high
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
每 grant 恰好一个 token
ACK 前无下一 token grant
draft_tpc_high > draft_tpc_low
draft_step_ms > 0
```

如果：

```text
TARGET_EXCLUSIVE ≈ 100%
```

先检查 resource-profile coverage。

---

# 30. D5：完整 I1+I2

只有 D4 在线链路完全正确后才进入 D5。

D5 startup/default：

```text
q=4
```

动态候选：

```text
1,2,4,6,8
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

否则 D5 只能作为 debug 结果。

---

# 31. workload 扩展

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

D1–D4 必须保持同一：

```text
input length
output length
q
seed
request count
token-pool policy
```

---

# 32. P99 rate sweep

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

# 33. screening 与正式确认性实验

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

# 34. 统一指标

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
Draft step latency
grant state distribution
grant wait
TPC range
missing drafts
fallback rate
```

---

# 35. throughput/GPU

```text
D1:
throughput_per_gpu = raw_throughput / 2

D2/D3/D4:
throughput_per_gpu = raw_throughput
```

必须同时报告 raw throughput 与 throughput/GPU。

不能用 throughput/GPU 隐藏 D1 的绝对吞吐优势，也不能只用 raw throughput 忽略单卡节省了一张 GPU。

---

# 36. Target slowdown

定义：

```text
Target slowdown =
T_target_overlap / T_target_baseline - 1
```

正式安全预算：

```text
<=5%
```

如果所有 TPC 点都 >5%：

```text
Innovation-2 No-Go for this shape/hardware
```

不要临时放宽到 10% 后继续宣称原目标成立。

---

# 37. 每次 run 后的故障扫描

```bash
grep -Eini 'out of memory|oom|timeout|missing|fallback|ungranted|mask|grant|traceback|error' "$SPECSTREAM_RESULT_ROOT/target.log" "$SPECSTREAM_RESULT_ROOT/draft.log" "$SPECSTREAM_RESULT_ROOT/benchmark.log" | tail -200
```

特别关注 5000ms/10000ms 阶梯，因为：

```bash
--spectre-recv-timeout-ms 5000
```

若 P99 呈 5 秒整数倍，优先查 Draft timeout/missing Draft，而不是解释为正常同卡竞争。

---

# 38. GPU/MPS 运行时检查

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

# 39. 避免旧 shell 变量污染

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

# 40. 禁止出现的实验错误

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

---

# 41. 主表推荐

| Method | GPUs | Raw Output Throughput | Throughput/GPU | P99 TTFT | P99 TPOT | P99 E2E | Target Slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 2-GPU SPECTRE | 2 | | | | | | |
| D2 Same-GPU uncontrolled | 1 | | | | | | |
| D3 Static TPC | 1 | | | | | | |
| D4 Online Target-priority | 1 | | | | | | |

D0 可作为辅助参考。D5 放完整系统表，不与 I2 isolation 主表混在一起。

---

# 42. 主图推荐

## 图 A：TPC Pareto

横轴：

```text
Target slowdown (%)
```

纵轴：

```text
Draft step latency
```

点：

```text
TPC=2,4,6,8,12
```

画 5% slowdown 阈值。

## 图 B：D1–D4 throughput

同时展示：

```text
raw output throughput
output throughput / GPU
```

## 图 C：online grant timeline

展示：

```text
TARGET_EXCLUSIVE
SLACK_FILL
DRAFT_CATCHUP
grant epoch
TPC range
Draft token
ACK
```

## 图 D：P99

在：

```text
0.5/1/2/4/8 req/s
```

比较：

```text
D1/D2/D3/D4
P99 TTFT/TPOT/E2E
```

---

# 43. 严格执行顺序

```text
[0] 固定 git commit / 环境
        ↓
[1] editable install + SpecStream 全部单测
        ↓
[2] first-grant gate 单测
        ↓
[3] libsmctrl build + validate-global
        ↓
[4] 启动 MPS
        ↓
[5] D1 双卡 q4
        ↓
[6] D2 同卡 q4，无 smctrl
        ↓
[7] Gate D：TPC=4 / C1 / 4 requests
        ↓
[8] Gate 成功后 TPC feasibility scan
        ↓
[9] 统计真实 batch/q/context shape
        ↓
[10] 构建 >=95% coverage measured profile
        ↓
[11] D3 最佳 fixed TPC
        ↓
[12] 验证 coexec_runtime online grant/ACK
        ↓
[13] D4 online Target-priority
        ↓
[14] D4 profile coverage >=95%
        ↓
[15] 扩展 context/concurrency/rate sweep
        ↓
[16] 5×>=1000-request confirmatory runs
        ↓
[17] D5 full I1+I2
```

---

# 44. Go / No-Go

## Gate 1：代码与环境

必须：

```text
当前工作树 import 正确
pytest 通过
libsmctrl validate-global 通过
```

失败：STOP。

## Gate 2：first-grant

必须：

```text
TPC=4 smoke 无 ungranted Draft forward
首个受控 forward 前已有真实 TPC mask
```

失败：STOP D3/D4。

## Gate 3：TPC 可行性

至少一个 TPC：

```text
Target slowdown <=5%
Draft step 有进展
error_count=0
```

失败：当前硬件/shape 下 I2 No-Go。

## Gate 4：online control

必须：

```text
grant epoch 单调
one grant = one Draft token
ACK 前无下一 token
未校准 shape fail closed
Target slowdown 超预算后 latch exclusive
```

失败：D4 不进入论文。

## Gate 5：端到端价值

D4 相比 D2/D3 至少形成一种稳定价值：

```text
更高 throughput/GPU
更低 P99
更低 Target interference
更稳定高负载 operating region
```

同时公开 D1 两卡 raw throughput。

---

# 45. 最小 first-grant 复现清单

只验证当前代码修复时：

```text
1. pytest test_first_grant_gate
2. validate-global TPC [0,4)
3. MPS ready
4. q=4
5. TPC=4
6. input=16K
7. output=32
8. requests=4
9. concurrency=1
10. Target/Draft /health 均成功
11. Draft log 无 ungranted forward
12. benchmark 正常结束
```

全部满足后才开始 200-request calibration。

---

# 46. 结果目录建议

```text
results/innovation2/
├── logs/
│   ├── D1_16k_c8/
│   ├── D2_16k_c8/
│   ├── gate_q4_tpc4/
│   ├── cal_q4_tpc4_rep1_base/
│   ├── cal_q4_tpc4_rep1_overlap/
│   ├── D3_16k_c8/
│   └── D4_16k_c8/
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
seed
```

---

# 47. 最终结果解释模板

如果：

```text
D2 << D1 raw throughput
```

说明：

> 同卡直接共执行会产生显著资源竞争。

如果：

```text
D3 > D2
且 Target slowdown <=5%
```

说明：

> 固定 TPC 隔离可以缓解干扰，存在安全空间资源划分区域。

如果：

```text
D4 >= D3
或动态负载下 D4 的 P99/Goodput 更优
且 grant/epoch/ACK 全链路正确
```

说明：

> measurement-backed Target-priority online grant 能安全利用 Target slack，而不是依赖固定静态分区。

如果：

```text
D4 raw throughput < D1
但 D4 throughput/GPU > D1 throughput/GPU
```

仍是合理结果：

> 单卡方案以部分绝对吞吐为代价，消除 dedicated Drafter GPU 资源税，提高单位 GPU 服务效率。

如果所有安全 TPC 都无法满足 5% Target slowdown：

> 如实报告适用边界，不放宽预算包装结果。

---

# 48. 一句话执行原则

> **先证明 first-grant 正确，再做 fixed-TPC calibration；先按真实 runtime shape 建 profile，再跑 online grant；先证明机制真的执行，再讨论性能。**
