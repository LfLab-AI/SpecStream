# 创新点一与创新点三 step1：分层缓存下的多卡独立草稿模型基线

## 1. 本步测试目的

建立 P0/H0 强 baseline：

```text
GPU 0 + GPU 1：Target，TP=2
GPU 2：Draft，独立运行
物理 GPU 总数：3
```

本步用于取得没有共卡干扰时的 TP 吞吐、P99、Draft RTT 和每个 Target rank 的时间，为后续两卡共置方案提供性能上限。

本步启用创新点一的分层 KV，但不启用创新点二、三的控制器。Target 使用 TP=2，Draft 独占 GPU2，用来测量“分层 KV + 无共卡干扰”的多卡性能上限。

## 2. 环境检查

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

nvidia-smi -L
nvidia-smi topo -m
git rev-parse HEAD

mkdir -p logs/innovation13_step1 results/innovation13_step1 profiles/innovation13_step1
```

目的：确认当前会话至少能看到 3 张 GPU，并记录 GPU0/GPU1 的互联拓扑。

## 3. 代码测试

```bash
PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_spectre_tp2.py \
  python/sglang/test/spectre_specstream/test_tp_straggler_monitor.py \
  python/sglang/test/spectre_specstream/test_multi_gpu_tp_policy.py \
  python/sglang/test/spectre_specstream/test_dynamic_q.py \
  python/sglang/test/spectre_specstream/test_control_profile.py
```

目的：确认 TP rank 使用一致决策、rank 时间统计正确，并确认 TP 保护关闭时不会影响 baseline。

## 4. 终端 A：启动 TP=2 Target

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /你的/Target模型目录 --port 30000 \
  --skip-server-warmup --tp-size 2 \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-require-draft --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-failure-threshold 3 --spectre-cooldown-rounds 32 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --specstream-enabled --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 --specstream-cpu-memory-gb 128 \
  --specstream-layer-prefetch --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path profiles/innovation13_step1/P0_tp2.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation13_step1/P0_target.log
```

目的：让分层 KV Target 使用 GPU0/GPU1 做 TP=2。该命令不包含卡内共执行或 TP 慢节点控制，长输入时应观察到 CPU History 和 H2D 数据。

## 5. 终端 B：启动独立 Draft GPU 2

Target ready 后执行：

```bash
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
  --model-path /你的/Draft模型目录 --port 30001 \
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
  2>&1 | tee logs/innovation13_step1/P0_draft.log
```

目的：让 Draft 独占 GPU2，避免对两个 Target ranks 造成资源竞争。

## 6. 终端 C：TP smoke

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30001/health

python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /你的/Target模型目录 \
  --tokenizer /你的/Target模型目录 \
  --dataset-name random \
  --dataset-path /common_data/dataset/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 8 \
  --random-input-len 16384 --random-output-len 64 \
  --random-range-ratio 1 \
  --request-rate 1 --max-concurrency 1 \
  --warmup-requests 1 --seed 1 --flush-cache --output-details \
  --tag I13-S1-P0-smoke \
  --output-file results/innovation13_step1/P0_smoke.jsonl
```

目的：确认 TP=2、Draft 通信、动态 q 和 streaming 同时工作。应生成带 rank 后缀的 profile 文件。

## 7. 16K 正式主实验

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /你的/Target模型目录 \
  --tokenizer /你的/Target模型目录 \
  --dataset-name random \
  --dataset-path /common_data/dataset/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 200 \
  --random-input-len 16384 --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf --max-concurrency 8 \
  --warmup-requests 8 --seed 1 --flush-cache --output-details \
  --tag I13-S1-P0-16k-c8 \
  --output-file results/innovation13_step1/P0_16k_c8.jsonl
```

目的：建立 P0 的主要吞吐、P99、q 分布和 Draft RTT。之后测试并发 `1、4、16、32`；每个点完整重启 Target/Draft，并修改输出文件名。

## 8. 30K 压力实验

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /你的/Target模型目录 \
  --tokenizer /你的/Target模型目录 \
  --dataset-name random \
  --dataset-path /common_data/dataset/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 80 \
  --random-input-len 30000 --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf --max-concurrency 4 \
  --warmup-requests 4 --seed 1 --flush-cache --output-details \
  --tag I13-S1-P0-30k-c4 \
  --output-file results/innovation13_step1/P0_30k_c4.jsonl
```

目的：记录长 History、TP=2 时的显存和 streaming 开销，供后续共卡组比较。

## 9. 汇总命令

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation13_step1/*.csv' \
  > results/innovation13_step1/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation13_step1/*.jsonl' \
  > results/innovation13_step1/benchmark_summary.tsv
```

目的：得到 P0 正式 baseline。必须保留所有 rank 的 profile，不能只分析 rank 0。

## 10. 本步通过条件

1. TP=2 所有 ranks 结果一致，无死锁和 collective 错误。
2. 至少 C=1/4/8 能获得有效推测，Accept length 不长期为 1。
3. 得到没有共卡干扰时的自然 rank 时间差。
4. 记录总吞吐、P99、Draft RTT、显存、物理 GPU 数=3 和 `goodput/GPU`。
