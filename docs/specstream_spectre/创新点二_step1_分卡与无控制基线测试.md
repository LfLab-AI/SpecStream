# 创新点二 step1：分卡强基线与同卡无控制对照测试

## 1. 本步测试目的

本步建立两个参照组：

- `I2-S1-B0`：Target 在 GPU 0，Draft 在 GPU 1。它是没有同卡资源竞争的强 baseline。
- `I2-S1-B1`：Target 和 Draft 都在 GPU 0，不启用 MPS，也不开启卡内控制。它用于测量直接共卡造成的干扰。

本文件中的命令均为直接执行方式，不需要设置 `SPECSTREAM_TARGET_CMD`、`SPECSTREAM_DRAFT_CMD` 或 `SPECSTREAM_BENCH_CMD`。

本文件以及后续创新点二、三的实验都不启用创新点一。Target KV 始终完整保存在 GPU。命令中的 `--specstream-profile-only` 只负责记录控制指标，不会 seal KV、不会把 KV 搬到 CPU，也不会从 CPU 流回 GPU。

## 2. 检查 GPU 和目录

直接执行：

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

nvidia-smi -L
nvidia-smi
git rev-parse HEAD

mkdir -p logs/innovation2_step1 results/innovation2_step1 profiles/innovation2_step1
```

目的：确认当前会话能看到 GPU，并记录代码版本。后续命令使用 GPU 编号 `0`、`1`；如果 `nvidia-smi -L` 中编号不同，请按实际编号修改。

## 3. 运行代码测试

```bash
PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_draft_delivery_policy.py \
  python/sglang/test/spectre_specstream/test_draft_load_tracker.py \
  python/sglang/test/spectre_specstream/test_dynamic_q.py \
  python/sglang/test/spectre_specstream/test_single_gpu_coexec_policy.py \
  python/sglang/test/spectre_specstream/test_spectre_fixed_q_e2e.py
```

目的：先确认 Draft 交付、timeout 统计、固定 q 和单卡策略基础逻辑没有回归。测试失败时不要继续跑 GPU 性能实验。

## 4. I2-S1-B0：Target GPU 0、Draft GPU 1

### 4.1 终端 A：启动 Target

把 `/你的/Target模型目录` 替换为真实路径，然后直接执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /common_data/model/Qwen2.5-7B-Instruct --port 30000 \
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
  --specstream-profile-only \
  --specstream-profile-path profiles/innovation2_step1/B0_dedicated.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation2_step1/B0_target.log
```

目的：让 Target 独占 GPU 0，建立没有 Draft 干扰的 Target 路径。看到 Uvicorn ready 后再启动 Draft。

### 4.2 终端 B：启动 Draft

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path /common_data/model/Qwen2.5-0.5B-Instruct --port 30001 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role draft \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --mem-fraction-static 0.45 \
  --max-total-tokens 196608 \
  --max-running-requests 8 \
  --spectre-max-batch-size 8 \
  --chunked-prefill-size 2048 \
  --spectre-draft-priority --spectre-max-draft-priority-steps 8 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation2_step1/B0_draft.log
```

目的：让 Draft 独占 GPU 1，与 Target 分卡并行。这就是创新点二最重要的分卡 baseline。这里也使用小显存 Draft 配置，保证 B0/B1 的 Draft 容量相同，比较只反映 GPU 放置差异。

### 4.3 终端 C：健康检查和 16K smoke

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30001/health

python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /common_data/model/Qwen2.5-7B-Instruct \
  --tokenizer /common_data/model/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --dataset-path /common_data/dataset/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 8 \
  --random-input-len 16384 --random-output-len 64 \
  --random-range-ratio 1 \
  --request-rate 1 --max-concurrency 1 \
  --warmup-requests 1 --seed 1 --flush-cache --output-details \
  --tag I2-S1-B0-smoke \
  --output-file results/innovation2_step1/B0_smoke.jsonl
```

目的：确认 Target、Draft、ZMQ 和 SpecStream streaming 链路都能正常运行。请求必须全部成功，Target 不应退出，Accept length 不应长期固定为 1。

### 4.4 终端 C：16K 正式并发测试

先运行并发 8：

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /common_data/model/Qwen2.5-7B-Instruct \
  --tokenizer /common_data/model/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --dataset-path /common_data/dataset/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 32 \
  --random-input-len 16384 --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf --max-concurrency 8 \
  --warmup-requests 8 --seed 1 --flush-cache --output-details \
  --tag I2-S1-B0-16k-c8 \
  --output-file results/innovation2_step1/B0_16k_c8.jsonl
```

目的：建立分卡 baseline 的主要吞吐和延迟数据。之后把 `--max-concurrency 8` 分别改为 `1、4、16、32`，每个并发点完整重启 Target 和 Draft，并修改输出文件名。

## 5. I2-S1-B1：Target 与 Draft 同卡、无控制

先停止 B0 的 Target 和 Draft。然后使用与 B0 完全相同的参数，只改变 GPU 放置。

### 5.1 终端 A：Target 仍使用 GPU 0

重新执行第 4.1 节 Target 命令，只把 profile 改为：

```bash
--specstream-profile-path profiles/innovation2_step1/B1_uncontrolled.csv
```

目的：Target 参数保持不变，保证 B0/B1 只比较 GPU 放置。

### 5.2 终端 B：Draft 改到 GPU 0

直接执行第 4.2 节 Draft 命令，但把第一行改为：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
```

日志文件改为：

```bash
2>&1 | tee logs/innovation2_step1/B1_draft.log
```

目的：故意制造同卡、无 MPS、无新增控制的资源竞争。

### 5.3 终端 C：运行完全相同的 benchmark

执行第 4.3 和 4.4 节的 benchmark，只把标签和输出文件改为 `I2-S1-B1`：

```bash
--tag I2-S1-B1-16k-c8 \
--output-file results/innovation2_step1/B1_16k_c8.jsonl
```

目的：量化直接共卡造成的吞吐下降、P99 增加、Draft timeout 和 fallback。

## 6. Draft 小显存参数说明

0.5B 模型的权重通常不是 10GB。您看到 Draft 进程占用约 10128MiB，主要原因是 SGLang 自动把大量剩余显存预分配给 Draft KV Cache 池。本文档已经在所有 Draft 命令中加入：

```bash
--mem-fraction-static 0.45 \
--max-total-tokens 196608 \
--max-running-requests 8 \
--spectre-max-batch-size 8 \
--chunked-prefill-size 2048 \
--disable-radix-cache --disable-cuda-graph
```

各参数的简单含义：

- `--max-total-tokens 196608`：直接限制 Draft KV Cache 池最多容纳的 token 数，这是控制预分配显存最直接的参数。
- `--mem-fraction-static 0.45`：限制 Draft 的模型权重加 KV 池不要继续吞掉几乎全部剩余显存。
- `--max-running-requests 8` 和 `--spectre-max-batch-size 8`：限制 Draft 同时处理的请求数；更多请求会排队，而不是一次性占满显存。
- `--chunked-prefill-size 2048`：降低 Draft 长 prompt 预填充时的临时激活显存峰值。
- `--disable-cuda-graph`：避免 Draft 为 CUDA Graph 额外保留显存。
- `--disable-radix-cache`：避免已完成请求继续占用额外前缀缓存。

对 A100 40GB + 0.5B Draft，建议先使用上面的“平衡配置”。如果仍然 OOM，改成更保守配置：

```bash
--mem-fraction-static 0.35 \
--max-total-tokens 131072 \
--max-running-requests 4 \
--spectre-max-batch-size 4 \
--chunked-prefill-size 1024 \
--disable-radix-cache --disable-cuda-graph
```

如果只想先跑通 16K、并发 1 的 smoke，可临时使用：

```bash
--mem-fraction-static 0.30 \
--max-total-tokens 65536 \
--max-running-requests 2 \
--spectre-max-batch-size 2 \
--chunked-prefill-size 1024 \
--disable-radix-cache --disable-cuda-graph
```

不要一开始把 `--max-total-tokens` 设成 8192，因为单条 16K prompt 本身就可能超过该容量。

启动 Draft 后先检查：

```bash
nvidia-smi

grep -Ei 'mem_fraction_static|max_total|memory pool|kv cache|out of memory|oom' \
  logs/innovation2_step1/B1_draft.log
```

目的：确认 Draft 显存已经从约 10GB 降下来，并确认日志中的实际 token pool 容量。建议 Target+Draft 启动后至少保留 4GB 左右空闲显存，供 warmup、attention 和临时张量使用；不要在 39532MiB/40960MiB 的状态下继续 benchmark。

如果 Draft 已经降到合理范围但 Target 仍在请求期间 OOM，再减少 Target KV 池，而不是继续压缩 0.5B 权重。做公平实验时，Target 的显存参数必须在 B0 和 B1 中同时修改。

## 7. 汇总命令

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation2_step1/*.csv' \
  > results/innovation2_step1/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation2_step1/*.jsonl' \
  > results/innovation2_step1/benchmark_summary.tsv
```

目的：生成 B0/B1 对比表。必须记录吞吐、P99 TPOT、Accept length、Draft RTT、timeout、fallback 和物理 GPU 数。

## 8. 本步通过条件

1. B0 和 B1 请求均正确，无死锁。
2. B0 能稳定获得有效 Draft，Accept length 不长期为 1。
3. B1 即使竞争严重，也能在 Draft timeout 时回退，而不是杀死 Target。
4. 获得 B0/B1 的吞吐、P99、显存和 timeout 对比。

本步不要求 B1 优于 B0。B1 的作用是测出“不加控制直接共卡”的实际代价。

## 9. 本次报错的直接解释

错误命令使用了：

```bash
CUDA_VISIBLE_DEVICES='GPU-填入GPU0的完整UUID'
```

这段文字只是占位符，并不是真实 GPU UUID，所以 CUDA 隐藏了所有有效 GPU，最终出现：

```text
RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) is available.
```

现在本文档统一使用 `CUDA_VISIBLE_DEVICES=0/1`。执行前只需要通过 `nvidia-smi -L` 确认 GPU 编号。

重新启动 Target 前，先直接验证 GPU 0：

```bash
nvidia-smi -L

CUDA_VISIBLE_DEVICES=0 python -c \
  "import torch; print('cuda_available=', torch.cuda.is_available()); print('gpu_count=', torch.cuda.device_count()); print('gpu0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

目的：预期输出 `cuda_available=True`、`gpu_count=1` 和真实 GPU 名称。如果这里仍是 `False/0/NONE`，先检查 CUDA/PyTorch/容器 GPU 权限，不要继续启动 SGLang。
