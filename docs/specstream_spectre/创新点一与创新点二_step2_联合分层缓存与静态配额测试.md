# 创新点一与创新点二 step2：分层缓存下的单卡静态配额扫描

## 1. 本步测试目的

本步验证：同一张 GPU 上是否存在一个稳定的 Target/Draft 资源配额，使 Target slowdown 较小，同时 Draft 能按时返回候选。

依次测试：

```text
Target/Draft = 90/10、80/20、70/30、60/40
```

本步不开启 `--specstream-coexec-enabled`。这样测到的是纯静态 MPS baseline，不会与 Step 3 动态策略混在一起。

本步启用创新点一的 CPU/GPU 分层 KV，但仍不开启创新点二的动态控制。这样测到的是“分层 KV + 静态 MPS”的联合 baseline，可与不使用创新点一的静态扫描直接对比。

## 2. 检查环境和代码

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

nvidia-smi -L
mkdir -p logs/innovation12_step2 results/innovation12_step2 profiles/innovation12_step2

PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_mps_env.py \
  python/sglang/test/spectre_specstream/test_control_profile.py \
  python/sglang/test/spectre_specstream/test_single_gpu_coexec_policy.py
```

目的：确认 MPS 配额可以被运行时读取，并能写入 profile。

## 3. 启动 MPS

直接执行：

```bash
mkdir -p /tmp/specstream-mps-$USER /tmp/specstream-mps-log-$USER

CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control -d

ps -ef | grep -E 'nvidia-cuda-mps-control|nvidia-cuda-mps-server' | grep -v grep
```

目的：为物理 GPU 0 启动 MPS。后续 Target 和 Draft 命令必须使用相同的 `CUDA_MPS_PIPE_DIRECTORY` 和 `CUDA_MPS_LOG_DIRECTORY`。

## 4. 先测试 80/20 配额

### 4.1 终端 A：Target 使用 80%

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 \
CUDA_MPS_CLIENT_PRIORITY=0 \
python -m sglang.launch_server \
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
  --specstream-enabled --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 --specstream-cpu-memory-gb 128 \
  --specstream-layer-prefetch --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path profiles/innovation12_step2/static_80_20.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation12_step2/static_80_20_target.log
```

目的：限制 Target 最多使用约 80% 的 active threads，为 Draft 留出资源上限。看到 Target ready 后再执行下一节。

### 4.2 终端 B：Draft 使用 20%

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20 \
CUDA_MPS_CLIENT_PRIORITY=1 \
python -m sglang.launch_server \
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
  2>&1 | tee logs/innovation12_step2/static_80_20_draft.log
```

目的：把 Draft 的执行资源限制为 20%，避免无控制地抢占 Target。

### 4.3 终端 C：健康检查和 smoke

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
  --tag I12-S2-static-80-20-smoke \
  --output-file results/innovation12_step2/static_80_20_smoke.jsonl
```

目的：确认 80/20 下两端能够启动、Draft 能返回、Target 不会持续 fallback。

## 5. 运行 80/20 正式测试

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /common_data/model/Qwen2.5-7B-Instruct \
  --tokenizer /common_data/model/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --dataset-path /home/lifei/lifei/specdecode/baseline/sglang/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 200 \
  --random-input-len 16384 --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf --max-concurrency 8 \
  --warmup-requests 8 --seed 1 --flush-cache --output-details \
  --tag I12-S2-static-80-20-16k-c8 \
  --output-file results/innovation12_step2/static_80_20_16k_c8.jsonl
```

目的：取得静态 80/20 在 16K、并发 8 下的吞吐、P99、Accept length 和 timeout。之后把并发改为 `1、4、16、32`，每个点重启 Target 和 Draft。

## 6. 扫描其他配额

每一组都先停止 Target 和 Draft，再按第 4 节直接重新启动。只修改下面四项：

| 组别 | Target 命令 | Draft 命令 | profile/输出名称 |
|---|---|---|---|
| 90/10 | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=90` | `...=10` | `static_90_10` |
| 80/20 | `...=80` | `...=20` | `static_80_20` |
| 70/30 | `...=70` | `...=30` | `static_70_30` |
| 60/40 | `...=60` | `...=40` | `static_60_40` |

目的：得到资源配额与 Target slowdown、Draft latency、吞吐和 P99 的完整曲线。不能只跑单个 80/20 就宣布静态配额有效。

## 7. 30K 长上下文测试

对每个配额至少运行并发 4：

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
  --tag I12-S2-static-80-20-30k-c4 \
  --output-file results/innovation12_step2/static_80_20_30k_c4.jsonl
```

目的：检查长 History 和 streaming 阶段是否为同卡 Draft 提供更有价值的执行窗口。运行其他配额时同步修改标签和输出文件名。

## 8. GPU 监控命令

在额外终端运行：

```bash
nvidia-smi dmon -s pucvmet -d 1 \
  -o DT > results/innovation12_step2/nvidia_smi_dmon.log
```

目的：记录 GPU 利用率、显存和 PCIe 活动，帮助判断性能下降来自 SM/HBM 竞争还是 Draft 超时。

## 9. 汇总命令

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation12_step2/*.csv' \
  > results/innovation12_step2/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation12_step2/*.jsonl' \
  > results/innovation12_step2/benchmark_summary.tsv
```

目的：选择最佳静态配额。选择时同时考虑吞吐、P99、Target slowdown、Draft timeout 和 `goodput/GPU`，不能只看速度最高的一项。

## 10. Go/No-Go 条件

Go：至少一个配额满足以下要求：

1. Target slowdown 可接受，建议先用 10% 作为参考线。
2. Draft timeout 较低，Accept length 不长期为 1。
3. 相比 Step 1 同卡无控制组，吞吐或 P99 明显改善。
4. 与两卡分卡 baseline 相比，完整报告总吞吐差距，并且单位 GPU 有效吞吐具有竞争力。
5. 长输入 profile 中 `cpu_history_bytes`、`h2d_bytes`、`h2d_ops` 均大于 0；否则不能算作创新点一与二的联合实验。

No-Go：90/10 到 60/40 全部明显负优化，且没有单位 GPU 效率收益。此时应先换更小 Draft 或限制适用工作负载。

## 11. 停止 MPS

全部实验结束后执行：

```bash
echo quit | \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control
```

目的：关闭本次实验的 MPS daemon，避免影响后续非 MPS 实验。
