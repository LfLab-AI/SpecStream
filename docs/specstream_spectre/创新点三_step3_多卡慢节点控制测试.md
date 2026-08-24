# 创新点三 step3：多卡慢节点感知控制与分层并行测试

## 1. 本步测试目的

在与 Step 2 完全相同的共卡布局和 80/20 配额下，开启单卡资源感知和 TP 慢节点保护，验证：

1. rank 0 开始落后时能否限 q、切换 `SERIALIZE` 或 `FALLBACK`；
2. 相比朴素 P1，rank skew、Target slowdown 和 P99 是否下降；
3. 相比独立 Draft P0，少用一张 GPU 后的 `goodput/GPU` 是否有竞争力；
4. 在不使用任何 CPU KV 卸载的前提下，控制器是否仍能降低 TP 慢节点影响。

本步只启用创新点三的调度控制。Target KV 仍是原生 SPECTRE 的全 GPU KV；`--specstream-profile-only` 只启动控制和记录，不启用创新点一。

## 2. 代码测试

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

mkdir -p logs/innovation3_step3 results/innovation3_step3 profiles/innovation3_step3

PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_spectre_tp2.py \
  python/sglang/test/spectre_specstream/test_tp_straggler_monitor.py \
  python/sglang/test/spectre_specstream/test_multi_gpu_tp_policy.py \
  python/sglang/test/spectre_specstream/test_single_gpu_coexec_policy.py \
  python/sglang/test/spectre_specstream/test_dynamic_q.py \
  python/sglang/test/spectre_specstream/test_control_profile.py
```

目的：确认单卡策略和多卡策略相互独立、TP timing 汇总正确、严重 rank skew 能触发安全回退。

## 3. 启动 MPS

```bash
mkdir -p /tmp/specstream-mps-$USER /tmp/specstream-mps-log-$USER

CUDA_VISIBLE_DEVICES=0,1 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control -d
```

目的：保持与 Step 2 相同的两卡 MPS 环境。O1 与 P1 必须使用相同 GPU 和配额。

## 4. 终端 A：启动 O1 TP 保护 Target

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
  --specstream-dynamic-q --specstream-q-candidates 1,2,4,6,8 \
  --specstream-q-switch-threshold 0.08 \
  --specstream-coexec-enabled --specstream-coexec-require-mps \
  --specstream-coexec-draft-pressure-ratio 0.80 \
  --specstream-coexec-timeout-rate-threshold 0.10 \
  --specstream-coexec-pending-high-watermark 16 \
  --specstream-coexec-compute-ratio-threshold 0.90 \
  --specstream-tp-straggler-control \
  --specstream-colocated-tp-rank 0 \
  --specstream-tp-straggler-budget-ms 1.0 \
  --specstream-target-slowdown-budget 0.10 \
  --specstream-tp-monitor-interval 8 \
  --specstream-profile-path profiles/innovation3_step3/O1_tp_control.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation3_step3/O1_target.log
```

目的：开启创新点三完整策略。`--specstream-colocated-tp-rank 0` 必须与 Draft 实际共卡的 rank 一致。

## 5. 终端 B：Draft 与 rank 0 共卡

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
  2>&1 | tee logs/innovation3_step3/O1_draft.log
```

目的：保持与 P1 完全相同的 Draft 放置和 20% 配额，只比较 TP 保护是否有效。

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
  --tag I3-S3-O1-smoke \
  --output-file results/innovation3_step3/O1_smoke.jsonl
```

目的：确认两个 ranks 使用一致 q/mode，并且 profile 包含 `tp_rank_skew_ms`、`tp_target_slowdown` 和 coexec mode。

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
  --tag I3-S3-O1-16k-c8 \
  --output-file results/innovation3_step3/O1_16k_c8.jsonl
```

目的：与 Step 2 P1 80/20、并发 8 直接比较。之后测试并发 `1、4、16、32`；每个点重启 Target/Draft。

## 8. TP skew 预算消融

分别完整重启并测试以下参数：

```bash
--specstream-tp-straggler-budget-ms 0.5
--specstream-tp-straggler-budget-ms 1.0
--specstream-tp-straggler-budget-ms 2.0
```

目的：比较更保守和更积极的慢节点阈值。每个预算都运行同一个 16K、并发 8 benchmark，并修改 profile/tag/输出文件名；不能只报告最有利的一个点。

## 9. 故障注入

先找到 Draft PID：

```bash
pgrep -af 'spectre-role draft'
```

把 `12345` 替换成实际 PID：

```bash
kill -STOP 12345
sleep 5
kill -CONT 12345

curl -fsS http://127.0.0.1:30000/health
```

目的：确认 Draft 暂停时 TP Target 不会退出；应该出现限 q、SERIALIZE 或 FALLBACK。

还可以把 Target/Draft 配额改为 70/30 并完整重启，故意增加 rank 0 压力。目的：确认压力严重时控制器优先保护整个 TP critical path。

## 10. 汇总命令

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation3_step3/*.csv' \
  > results/innovation3_step3/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation3_step3/*.jsonl' \
  > results/innovation3_step3/benchmark_summary.tsv
```

目的：把原生分卡 P0、原生朴素共卡 P1、原生 GPU KV 加创新点三控制 O1 放在同一结果表中，比较总吞吐、P99、`goodput/GPU`、rank skew、Target slowdown 和 SERIALIZE/FALLBACK 次数。

## 11. 本步通过条件

1. O1 相比 P1 的 `tp_rank_skew_ms` 和 P99 明显下降。
2. 所有 TP ranks 使用一致 q/mode，无死锁和 collective 错误。
3. 共卡压力变大时能进入 SERIALIZE/FALLBACK；恢复后可以重新使用 COEXEC。
4. O1 与 P0 同时报告总吞吐和 `goodput/GPU`：P0 使用 3 张物理卡，O1 使用 2 张。
5. P0、P1、O1 的 `cpu_bytes`、`h2d_bytes` 和 `h2d_ops` 都必须为 0；否则说明错误地混入了创新点一。

## 12. 停止 MPS

```bash
echo quit | \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control
```
