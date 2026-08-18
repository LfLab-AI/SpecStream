# SpecStream B3 多 Query Tiled 内核优化、结果分析与服务器验证

## 1. 本轮结果的结论

附件中的 12 组结果按出现顺序解释为 B0、B1、B2、B3，每组依次为 4K、16K、30K。原命令的 `tag` 和 `output-file` 全部仍写成了 `B0_*`，因此以下归类依赖运行顺序；后续测试必须改为唯一 baseline 名称，避免覆盖或误归类。

| Context 参数 | B0 output tok/s | B1 | B2 | B3 | B3/B2 | B3/B1 | B3/B0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 48.95 | 41.03 | 40.75 | 38.69 | 0.949x | 0.943x | 0.790x |
| 16384 | 27.82 | 8.07 | 7.83 | 13.36 | 1.706x | 1.655x | 0.480x |
| 30000 | 10.33 | 2.96 | 3.19 | 5.48 | 1.718x | 1.851x | 0.531x |

B3 已经明显优于 B1/B2 的长上下文路径：16K、30K 的 output throughput 分别比 B2 高约 70.6% 和 71.8%，Mean TPOT 分别由 114.14/257.72 ms 降到 60.55/116.98 ms。但是 B3 仍只有 B0 的 48.0%/53.1%。这不是“online softmax 没生效”，而是旧 B3 内核和实验条件共同造成的：

1. 旧内核每个 `(query, query_head)` 一个 Triton program，用逐元素乘加计算 QK；K/V 没有在一个 program 内跨多 query 复用，也没有使用 Tensor Core `tl.dot`。
2. 每个 2048-token chunk 都有一次 Python 循环、一次 ready/free event 和一次 attention launch。
3. GPU Tail/Frontier 原先由多个 Torch `arange/index_select/einsum` 操作完成，产生额外临时张量与 launch。
4. 12 组结果的 `Accept length` 全部为 1.00。结合后续 Target 日志中持续出现的 `[Target] Recv timeout`，不能再断言这些轮次真实验证了 5 个有效 draft query；更准确的解释是远程草稿未按轮次送达，Target 静默走了 q=1 normal-decode fallback。必须先按 Step 1 的 `ordinary + require-draft` 校准流程修复草稿链路，再评价 B3 的 multi-query 摊薄收益。
5. 命令中的 `--random-range-ratio 0` 不是固定长度。SGLang 实现从 `[full_len * ratio, full_len]` 采样；固定长度必须用 `1`。旧日志实际平均输入只有约 2.17K、7.77K、15.20K，而不是 4K、16K、30K。

以 Qwen2.5-7B 的 28 层、4 个 KV heads、head_dim=128、BF16 为例，每个 token 的 Target KV 为 `2 × 4 × 128 × 2 = 2048 bytes/layer`。若 30K History 全部 sealed，一轮验证约需读取 `30000 × 2048 × 28 ≈ 1.72 GB`。即使有效 PCIe 带宽为 24 GB/s，纯 H2D 下界也约为 72 ms/round；当 accept length 约为 1 时，它不可能仅靠 kernel 优化稳定击败 GPU-resident B0。P0 任务是同时提高 draft acceptance。

## 2. 新 B3 数据路径

本轮代码把 B3 改为：

```text
CPU pinned sealed chunks
  -> 独立 CUDA copy stream
  -> 2 个预分配 bounded staging slots
  -> 每 4 个逻辑 chunk 组成一个 8192-token transfer group
  -> 每组只使用 1 对 ready/free event 和 1 次 tiled attention launch
  -> FP32 online (m, l, acc)
  -> 间接读取 GPU paged Tail/Frontier 的同一 tiled kernel
  -> finalize acc/l
```

关键变化如下：

- `BLOCK_M` 同时覆盖多个 verification query/GQA rows；一个 K/V tile 被这些 rows 共同使用。
- `BLOCK_N=64`，QK 和 PV 都用 `tl.dot`，遵循 FlashAttention 的 tiled online-softmax 结构。
- `m/l/acc` 始终为 FP32；`m` 保持自然对数 score 空间，History 与 Tail 可直接合并。
- Tail 内核通过 SGLang page-table slots 间接读取 KV cache，不再先执行两次 `index_select`。
- `submit_many()` 不在 CPU 上重新拼接数据；它在 copy stream 中把若干 pinned slabs 依次放进同一连续 staging view，然后只记录一个 ready event。
- ring pipeline 会先提交前 `num_buffers` 组，计算组 i 时复制组 i+1；复用 slot 前由 free event 保证计算已经消费完毕。
- staging 在 Target 初始化时一次性预分配，避免第一轮 verification 中扩容和 current-stream synchronize。

默认 `chunk_tokens=2048`、`chunks_per_transfer=4`、`num_buffers=2` 时，每 rank 的 staging 上界为：

```text
2 buffers × 8192 tokens × 2048 bytes/token = 32 MiB
```

它不随 History 长度增长。注意：逻辑 chunk 分组减少的是 attention launch、event 和 Python 调度次数；由于 CPU chunks 是独立 pinned allocations，底层仍会产生每源 chunk 一次 `cudaMemcpyAsync`。若要进一步合并成单一 DMA，需要把 CPUHistoryStore 改为连续 super-slab，这应单独做消融，因为会改变 seal 粒度和 CPU 内存管理。

## 3. 采用这些方法的依据

- Triton 官方 fused-attention 教程使用 `BLOCK_M × BLOCK_N` 的 Q/K tile、`tl.dot(q,k)`、online max/normalizer 和 `tl.dot(p,v)`；新内核按同一执行结构实现，但把 online state 暴露出来用于跨 CPU chunk 合并：<https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- FlashAttention 将性能问题表述为 HBM/SRAM I/O-aware tiling，而不是单纯减少 FLOPs：<https://arxiv.org/abs/2205.14135>
- NVIDIA CUDA Best Practices 明确要求 pinned host memory、不同的 non-default streams，并推荐 staged concurrent copy/execute：<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation>
- PyTorch 官方实验同样指出，只有 pinned CPU tensor、独立 stream 和 non-blocking copy 同时满足时，H2D 才能和 GPU kernel 重叠：<https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
- FlexGen/FlexLLMGen 使用 block schedule 提升复用并重叠 I/O 与计算；这里借鉴的是调度原则，不采用其量化或近似策略：<https://github.com/FMInference/FlexLLMGen>

## 4. 服务器更新方式

不要把整个新 `srt` 目录覆盖到旧 wheel 的 `site-packages`。在固定的 SGLang commit 工作树中同步这些源文件，然后使用 editable install：

```bash
cd ~/lifei/specdecode/baseline/sglang
python -m pip install -e ./python --no-deps

python - <<'PY'
import inspect
from sglang.srt.speculative.spectre.specstream import triton_stream_attn
print(inspect.getfile(triton_stream_attn))
PY
```

输出必须指向当前工作树的 `python/sglang/...`，而不是另一份残留 wheel。

## 5. 上线前正确性测试

先运行全部纯模块测试：

```bash
cd ~/lifei/specdecode/baseline/sglang
pytest -q python/sglang/test/spectre_specstream
```

GPU/Triton 专项测试会覆盖 q=1、q=5、BF16/FP16、513 个非整块 History tokens，以及 History + 间接 paged Tail 的 causal 合并：

```bash
pytest -q -s \
  python/sglang/test/spectre_specstream/test_triton_tiled_attention.py
```

然后启动一轮带 shadow 的 B3，仅做正确性，不记录性能：

```bash
--specstream-enabled \
--no-specstream-reference-attention \
--specstream-shadow-attention \
--specstream-chunk-tokens 2048 \
--specstream-chunks-per-transfer 4 \
--specstream-num-buffers 2 \
--specstream-profile-path results/profiles/B3_tiled_shadow.csv
```

必须满足：无 NaN/Inf、无 token mismatch、shadow error 在模型容差内。Shadow 会重复运行 reference attention，不得用于性能数字。

## 6. B3 性能启动参数

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port 30000 \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --specstream-enabled --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 \
  --specstream-chunks-per-transfer 4 \
  --specstream-num-buffers 2 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-profile-path results/profiles/B3_tiled.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

## 7. 修正后的固定长度对照

每次启动对应 Target 后，把 `BASELINE` 设置为真实名称。不能再把四轮都写入 `B0_*`：

```bash
export BASE_URL=http://127.0.0.1:30000
export DATASET_NAME=random
export DATASET_PATH="$SHAREGPT_JSON"

for BASELINE in B3; do
  for INPUT_LEN in 4096 16384 30000; do
    CASE_TAG="${BASELINE}_fixed_${INPUT_LEN}_c1" \
    INPUT_LEN="$INPUT_LEN" OUTPUT_LEN=128 \
    NUM_PROMPTS=30 REQUEST_RATE=1 MAX_CONCURRENCY=1 \
    RANGE_RATIO=1 WARMUP_REQUESTS=4 SEED=1 \
    bash scripts/specstream/run_benchmark_case.sh
  done
done
```

为了让所有请求确实生成 128 tokens，可在专门的 controlled microbenchmark 中使用 `ignore_eos=true`；若保持真实 EOS 行为，则必须核对四个 baseline 的实际 generated tokens 完全一致。

正式数据每点至少独立重启 Target 并重复 3 次；论文数字建议 5 次。报告 native B0 和关闭 graph/overlap、与 SpecStream 调度约束一致的 B0-fair 两个基线。

## 8. `chunks_per_transfer` 的硬件调优

默认 4 是针对 2K logical chunks 的稳健起点，不是对所有 GPU/PCIe 的理论全局最优。保持其他参数不变，分别重启 B3 Target 测量：

```bash
for GROUP in 1 2 4 8; do
  # 每轮将 Target 参数改为：
  # --specstream-chunks-per-transfer "$GROUP"
  # profile/result 名称必须含 g${GROUP}
  true
done
```

选择规则：

- 以 fixed 16K/30K 的 median TPOT、output tok/s、P99 和 staging bytes 联合判断；
- 若 Nsight Systems 中 copy 与 tiled kernel 几乎没有重叠，优先减小 group；
- 若 timeline 中大量极短 kernel/event/CPU gap，优先增大 group；
- group 增大后 staging 线性增长，但 History 总 H2D bytes 不变；
- 只有 `h2d_ops` 与 attention launch 数约按 group 倍数下降时，才说明新路径实际生效。

检查硬件 copy engine：

```bash
python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(p)
print("device=", torch.cuda.get_device_name(0))
PY
```

最终用 Nsight Systems 验证 copy stream 和 main compute stream 的时间线，而不能只依赖 Python `perf_counter`。现有 CSV 的 per-chunk `h2d_ms/stream_attn_ms` 是 host enqueue 区间，异步 CUDA 下不能作为精确 kernel 时间；端到端吞吐和 Nsight timeline 才是性能结论依据。

## 9. 下一轮 GO/NO-GO 条件

1. `test_triton_tiled_attention.py` 全部通过，shadow 无 token mismatch。
2. fixed-length 命令实际打印的 input token 总数约为 `num_prompts × input_len`。
3. B3 profile 中 `h2d_bytes > 0`，`h2d_ops` 相比 group=1 约下降 4 倍，staging bytes 不随 History 线性增加。
4. Nsight 显示 H2D 与前一组 tiled attention 有实质重叠。
5. B3 在 16K/30K 稳定优于 B2/B1；4K 的回退路径回归不超过预设阈值。
6. `avg_spec_accept_length` 不再长期等于 1.00。若仍为 1，先修复 Drafter/Target 模型匹配、tokenizer、采样配置或 stale response，再讨论 B3 是否能超过 B0。

只有服务器 GPU 上完成这些测试后，才能宣称“该硬件配置下的最优参数”。本地代码审查可以消除确定性的结构开销，但不能诚实地替代真实 PCIe、GPU、NUMA 和并发负载测量。
