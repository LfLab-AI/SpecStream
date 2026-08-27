# SpecStream 创新点二：最简端到端实验测试手册

> 本版只做两个实验：
>
> 1. **正确性测试**：证明创新点二与原生 STANDALONE Spec V2 输出完全一致；
> 2. **性能测试**：证明创新点二相对原生 STANDALONE Spec V2 是否真正加速。
>
> 其余 h 扫描、C8、Nsight、GSM8K、LongBench、TPC 和故障注入均不作为首次验收内容。只有这两个测试出现问题时，才做额外诊断。

---

# 0. 您刚才的命令为什么失败

这次失败与创新点二正确性或性能无关，server 在加载模型前就没有识别到 GPU：

```text
RuntimeError: No accelerator ... is available.
```

直接原因有两个：

1. `CUDA_VISIBLE_DEVICES="$SINGLE_GPU_UUID"` 指向了当前容器不可用的 UUID、空值或示例占位符；
2. 当前 shell 中的 `OMP_NUM_THREADS` 不是合法整数，所以同时出现：

```text
libgomp: Invalid value for environment variable OMP_NUM_THREADS
```

后面的 `P0_NATIVE.jsonl` 不存在、`start_gpu_monitor: command not found` 都是 server 启动失败后的连锁错误，不是新的代码问题。

本版脚本统一使用容器内逻辑 GPU 编号：

```bash
GPU_ID=0
```

并强制：

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

脚本会先运行 `torch.cuda.is_available()`；GPU 不可见时立即退出，不会继续产生一串无意义错误。

---

# 1. 一次性准备

进入仓库和环境：

```bash
cd ~/lifei/SpecStream
conda activate spectre
```

设置实际模型和数据路径：

```bash
export TARGET_MODEL=/root/autodl-tmp/model/Qwen2.5-7B-Instruct
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct
export SHAREGPT_JSON=$PWD/specstream_prepared/sharegpt_v3_merged.json

export GPU_ID=0
export ATTENTION_BACKEND=fa3
```

先确认容器中的 GPU：

```bash
nvidia-smi -L

CUDA_VISIBLE_DEVICES="$GPU_ID" \
OMP_NUM_THREADS=1 \
python - <<'PY'
import torch
print("cuda_available =", torch.cuda.is_available())
print("device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_0 =", torch.cuda.get_device_name(0))
PY
```

必须得到：

```text
cuda_available = True
device_count >= 1
```

如果仍为 `False`，先解决 AutoDL 容器的 GPU 挂载或修改 `GPU_ID`，不要继续运行测试。

赋予脚本执行权限：

```bash
chmod +x \
  scripts/specstream/run_i2_correctness.sh \
  scripts/specstream/run_i2_performance.sh
```

---

# 2. 实验一：正确性测试

## 2.1 测什么

只比较三个系统：

| ID | 配置 | 作用 |
|---|---|---|
| P0-Native | 原生 STANDALONE Spec V2 | 正确性基准 |
| P2-H4 | 固定 `h=4` 的进程内 Draft-ahead | 创新点二核心路径 |
| P3-Auto | 自适应 Draft-ahead | 创新点二完整路径 |

脚本自动完成：

1. 检查 GPU；
2. 依次启动 P0、P2-H4 和 P3；
3. 对相同的 24 条固定 prompt 做贪心生成；
4. 每条生成 64 token；
5. 提取并逐 token 比较输出 token IDs；
6. 检查 P2/P3 profile，确认 Draft-ahead 确实被执行，而不是全部回退串行。

## 2.2 直接运行

```bash
bash scripts/specstream/run_i2_correctness.sh
```

若显存不足，可只调整容量，不改变算法参数：

```bash
MAX_TOTAL_TOKENS=32768 \
bash scripts/specstream/run_i2_correctness.sh
```

当前创新点二的 q=4 参数已经固定在脚本中：

```text
--speculative-num-steps 4
--speculative-num-draft-tokens 5
--speculative-eagle-topk 1
--page-size 1
```

不要改回旧文档的 `num_steps=3 / draft_tokens=4`。

## 2.3 成功判据

终端最后必须显示：

```text
P2_H4 mismatches = 0
P3_AUTO mismatches = 0
P2_H4 profile_rows > 0
P3_AUTO profile_rows > 0
ahead_generated > 0
CORRECTNESS TEST: PASS
```

只要出现一个 token mismatch，就判失败；不能用“文本看起来相同”或“准确率接近”替代。

结果默认保存在：

```text
results/innovation2_inproc_minimal/correctness_<timestamp>/
```

这一个测试足以回答：

> 当前创新点二是否在真正执行 Draft-ahead 时保持原生 Spec V2 的输出语义。

---

# 3. 实验二：性能测试

## 3.1 测什么

只比较四个系统：

| ID | 配置 | 回答的问题 |
|---|---|---|
| P0-Native | 原生 STANDALONE Spec V2 | 原始单卡性能 |
| P1-Serial | 加载创新点二框架，但强制串行 | 框架本身是否有额外开销 |
| P2-H4 | 固定 `h=4` Draft-ahead | 固定并行是否加速 |
| P3-Auto | 自适应 Draft-ahead | 完整创新点二是否加速 |

只跑一个最能隔离创新点二的 workload：

```text
数据：ShareGPT 文本构造的受控 random workload
input length：16K
output length：128
concurrency：1
prompts：200
repetitions：3
request rate：inf
```

选择 C1 是因为当前创新点二是 same-request Draft-ahead。首次验收不需要 C4/C8；高并发只在 C1 已证明有效后再测试。

## 3.2 直接运行

```bash
bash scripts/specstream/run_i2_performance.sh
```

快速 smoke 可以减少请求和重复数：

```bash
NUM_PROMPTS=32 \
OUTPUT_LEN=64 \
REPETITIONS=1 \
bash scripts/specstream/run_i2_performance.sh
```

得到正常结果后，再运行正式版本：

```bash
NUM_PROMPTS=1000 \
OUTPUT_LEN=128 \
REPETITIONS=5 \
bash scripts/specstream/run_i2_performance.sh
```

这仍然是同一个性能测试，只是把样本量从 smoke 提高到论文统计所需规模。

## 3.3 输出结果

脚本会自动输出：

```text
performance_summary.tsv
```

表中只有必要指标：

```text
method
runs
output_tok_s
speedup_vs_P0
p99_ttft_ms
p99_tpot_ms
p99_e2e_ms
```

结果默认保存在：

```text
results/innovation2_inproc_minimal/performance_<timestamp>/
```

## 3.4 判定方法

先看 P1：

```text
P1-Serial ≈ P0-Native
```

如果 P1 已明显慢于 P0，说明创新点二框架本身有额外开销，暂时不要解释 P2/P3。

再看 P2/P3：

```text
speedup_vs_P0 > 1.0
且 P99 没有明显恶化
```

建议首次验收使用：

```text
P1 throughput 不低于 P0 的 97%
P2 或 P3 throughput 高于 P0
P2/P3 P99 E2E 不超过 P0 的 105%
error_count = 0
```

只有同时满足正确性 PASS 和性能收益，才能说明当前创新点二有效。

---

# 4. 最终只保留这一张结果表

| Method | Correct | Output tok/s | Speedup vs P0 | P99 TTFT | P99 TPOT | P99 E2E |
|---|---:|---:|---:|---:|---:|---:|
| P0-Native | reference |  | 1.00× |  |  |  |
| P1-Serial | — |  |  |  |  |  |
| P2-H4 | 100% token parity |  |  |  |  |  |
| P3-Auto | 100% token parity |  |  |  |  |  |

论文中创新点二的实验结论只需要围绕这张表写：

1. P2/P3 是否与 P0 逐 token 一致；
2. P1 是否说明框架开销可忽略；
3. P2/P3 是否提高 output throughput；
4. 加速是否以严重 P99 退化为代价。

---

# 5. 只有失败时才做的诊断

这些不是正式实验，不要预先全部执行。

## 正确性失败

只检查首个 mismatch prompt，并查看：

```text
P2/P3 server.log
对应 profile JSONL
ahead_tokens_generated/reused/discarded
repair_tokens
```

## P1 明显慢于 P0

核对两个 `server_info.json` 中：

```text
SGLANG_ENABLE_SPEC_V2
disable_cuda_graph
disable_overlap_schedule
max_total_tokens
attention_backend
```

## P2/P3 没有加速

先看：

```text
ahead_reuse_ratio
actual_overlap_ms
repair_tokens
```

然后才考虑把性能测试改为 `h=1/2` 或增加 C8。不要在首次验收前运行完整 h/context/concurrency 网格。

---

# 6. 一句话执行顺序

```bash
# 1. GPU 必须可见
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# 2. 一个正确性测试
bash scripts/specstream/run_i2_correctness.sh

# 3. 一个性能测试
bash scripts/specstream/run_i2_performance.sh
```

> **正确性 PASS 后只跑一次固定 16K/C1 性能对照；先回答“对不对”和“快不快”，其余测试只有在解释失败原因时再增加。**
