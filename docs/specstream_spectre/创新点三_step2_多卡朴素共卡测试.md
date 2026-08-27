# 创新点三 step2：多卡单进程朴素共卡与慢节点复现测试

## 1. 本步测试目的

本步测试 P1/H1 朴素共卡布局：

```text
GPU 0：Target rank 0 + Draft
GPU 1：Target rank 1
物理 GPU 总数：2
TP straggler 控制：关闭
```

目标是测出 Draft 共卡后，rank 0 是否比 rank 1 慢，并使整个 TP 组在 collective 前等待。

本步仍不启用创新点一。Target KV 全部保存在各自 TP rank 的 GPU 上；`--specstream-profile-only` 仅记录数据，不卸载 KV。

## 2. 环境和代码测试

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

nvidia-smi -L
mkdir -p logs/innovation3_step2 results/innovation3_step2 profiles/innovation3_step2

PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_spectre_tp2.py \
  python/sglang/test/spectre_specstream/test_tp_straggler_monitor.py \
  python/sglang/test/spectre_specstream/test_multi_gpu_tp_policy.py \
  python/sglang/test/spectre_specstream/test_control_profile.py
```

目的：确认 rank profile 和 skew 统计逻辑正确。P1 和后续 O1 必须使用与 Step 1 相同的 GPU0/GPU1。

## 3. 启动两卡 MPS daemon

```bash
mkdir -p /tmp/specstream-mps-$USER /tmp/specstream-mps-log-$USER

CUDA_VISIBLE_DEVICES=0,1 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control -d
```

目的：让两个 Target ranks 都连接同一个 MPS daemon。Target 两个 workers 使用相同的 80% 上限，只有 GPU0 额外运行 Draft。

## 4. 终端 A：启动无 TP 保护的 Target

```bash
CUDA_VISIBLE_DEVICES=0,1 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 \
CUDA_MPS_CLIENT_PRIORITY=0 \
python -m sglang.launch_server \
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
  --specstream-profile-only \
  --specstream-profile-path profiles/innovation3_step2/P1_naive_80_20.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation3_step2/P1_target_80_20.log
```

目的：运行 TP=2，但故意不开启 `--specstream-tp-straggler-control` 和 `--specstream-coexec-enabled`，作为朴素共卡对照。

## 5. 终端 B：Draft 与 rank 0 共用 GPU 0

Target ready 后执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
CUDA_MPS_CLIENT_PRIORITY=1 \
python -m sglang.launch_server \
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
  2>&1 | tee logs/innovation3_step2/P1_draft_80_20.log
```

目的：让 Draft 只与物理 GPU0 上的 Target rank 0 竞争资源，从而复现不对称 TP 慢节点。

## 6. 终端 C：smoke

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
  --tag I3-S2-P1-80-20-smoke \
  --output-file results/innovation3_step2/P1_80_20_smoke.jsonl
```

目的：确认 TP 共卡布局可以完成请求，并生成两个 ranks 的 profile。

## 7. 16K 并发主实验

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
  --tag I3-S2-P1-80-20-16k-c8 \
  --output-file results/innovation3_step2/P1_80_20_16k_c8.jsonl
```

目的：测量 80/20 配额下的 rank skew、Target slowdown、吞吐和 P99。之后测试并发 `1、4、16、32`，每个点重启两端。

## 8. 扫描 90/10 和 70/30

每一组完整重启 Target 和 Draft，只修改以下内容：

| 组别 | Target 百分比 | Draft 百分比 | profile/tag 名称 |
|---|---:|---:|---|
| 90/10 | 90 | 10 | `P1_naive_90_10` |
| 80/20 | 80 | 20 | `P1_naive_80_20` |
| 70/30 | 70 | 30 | `P1_naive_70_30` |

目的：得到 `Draft share -> tp_rank_skew_ms -> Target slowdown/P99` 曲线。Draft 份额越大，如果 rank 0 越明显落后，就说明第三创新点的慢节点问题真实存在。

## 9. GPU 监控

额外终端执行：

```bash
nvidia-smi dmon -s pucvmet -d 1 \
  -o DT > results/innovation3_step2/nvidia_smi_dmon.log
```

目的：对照 GPU0/GPU1 的利用率和显存，确认不对称负载确实来自 GPU0 上的 Draft。

## 10. 汇总命令

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation3_step2/*.csv' \
  > results/innovation3_step2/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation3_step2/*.jsonl' \
  > results/innovation3_step2/benchmark_summary.tsv
```

目的：记录每个 rank 的 Target forward、`tp_rank_skew_ms`、`tp_target_slowdown`、Draft timeout、总吞吐、P99 和物理 GPU 数=2。

## 11. 本步完成条件

1. 所有请求正确，无 TP 死锁。
2. 至少获得 90/10、80/20、70/30 三组 rank skew。
3. 能判断共卡 rank 是否随 Draft share 或并发增加而落后。
4. 与 P0 同时报告总吞吐和 `goodput/GPU`：P0 除以 3 张卡，P1 除以 2 张卡。

本步是问题复现实验，不要求 P1 优于 P0。

## 12. 停止 MPS

```bash
echo quit | \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control
```
