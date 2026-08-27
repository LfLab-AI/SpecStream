# 创新点二 step1：分卡强基线与同卡无控制基线（详细执行版）

## 1. 这一步要回答什么

step1 不启用创新点二的 execution grant，也不限制 Drafter TPC。它建立后续实验必须对照的两个基线：

- `I2-S1-B0`：Target 在 GPU0，Drafter 在 GPU1。两张卡互不争抢，是性能强基线；
- `I2-S1-B1`：Target 和 Drafter都在 GPU0，MPS 只允许两个进程共卡，但没有 TPC mask 和动态控制。这是直接共卡的负面对照。

本步同时验证 libsmctrl 是否具备进入 step2 的资格，但 B0/B1 的服务命令都不能带 `--specstream-smctrl-enabled`。比较目的不是要求单卡总吞吐超过两卡，而是给 step2/step3 提供总吞吐、尾延迟和 goodput/GPU 的参照。

## 2. 终端和执行规则

使用四个终端：终端 A 跑 Target，终端 B 跑 Drafter，终端 C 跑健康检查和 benchmark，终端 D 跑监控、日志检查和汇总。

每个正式点至少独立运行 3 次。每一轮都停止并重启两个服务，输出文件名使用 `r1`、`r2`、`r3`，禁止覆盖旧结果。

## 3. 四个终端共同执行的环境准备

```bash
cd ~/lifei/SpecStream
conda activate spectre

export TARGET_MODEL=/root/autodl-tmp/model/Qwen2.5-7B-Instruct
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct
export DATASET=$PWD/specstream_prepared/sharegpt_v3_merged.json
export SMCTRL_LIB=$PWD/csrc/specstream_smctrl/build/libsmctrl.so
export MPS_PIPE=/tmp/specstream-mps-$USER
export MPS_LOG=/tmp/specstream-mps-log-$USER

unset MASK_OFF
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

mkdir -p logs/innovation2_step1
mkdir -p results/innovation2_step1
mkdir -p profiles/innovation2_step1

test -d "$TARGET_MODEL"
test -d "$DRAFT_MODEL"
test -s "$DATASET"
```

记录环境，后续结果表必须附带：

```bash
nvidia-smi -L | tee results/innovation2_step1/gpu_list.txt
nvidia-smi | tee results/innovation2_step1/nvidia_smi.txt
nvcc --version | tee results/innovation2_step1/nvcc_version.txt
python --version | tee results/innovation2_step1/python_version.txt
git rev-parse HEAD | tee results/innovation2_step1/git_commit.txt

PYTHONPATH=python python - <<'PY' | tee results/innovation2_step1/runtime_versions.txt
import torch
import sglang
print("torch =", torch.__version__)
print("torch CUDA build =", torch.version.cuda)
print("sglang =", getattr(sglang, "__version__", "unknown"))
print("cuda available =", torch.cuda.is_available())
PY
```

目的：区分 Driver API 13.0、nvcc 12.8、PyTorch CUDA build 三个不同概念，并保证重复实验可追溯。

## 4. 代码正确性门禁

```bash
cd ~/lifei/SpecStream
PYTHONPATH=python pytest -q python/sglang/test/spectre_specstream
```

目的：在跑长时间 GPU 实验前，检查 SPECTRE、grant 状态机、SM controller、resource profile 等 CPU 可测逻辑。

通过条件：没有 failed。因缺少真实 GPU/TP 环境而明确标记的 skipped 可以接受，但必须记录数量。

## 5. libsmctrl 的预验证门禁

B0/B1 不会使用 mask，但必须提前证明同一台机器可以进入 step2。

```bash
cd ~/lifei/SpecStream/csrc/specstream_smctrl
make config
make build
test -s build/libsmctrl.so

unset MASK_OFF
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=0 TPC_HIGH=2
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=0 TPC_HIGH=4
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=4 TPC_HIGH=8
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=50 TPC_HIGH=54
cd ~/lifei/SpecStream
```

目的：A800/Driver 580 使用进程级 QMD/TMD callback 后端，不依赖 CUDA stream 私有结构偏移。

通过条件：每条都包含 `using process-global QMD/TMD mask backend` 和 `test passed`，且无任何 `SM ... shouldn't be used`。不要运行已经证实失败的 `MASK_OFF=24 make validate`。

如计划在 GPU1 上运行后续 mask 实验，再把四条命令中的 `CUDA_VISIBLE_DEVICES=0` 改为 `1` 重复一次。

## 6. I2-S1-B0：Target GPU0、Drafter GPU1

### 6.1 本组目的

这组使用两张物理 GPU，不启动 MPS，不启用任何 SM/TPC 控制。它表示没有共卡资源争抢时，当前模型和负载的最好 SPECTRE 基线。

### 6.2 终端 A：启动 Target

```bash
cd ~/lifei/SpecStream
conda activate spectre

CUDA_VISIBLE_DEVICES=0 \
python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role target \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --spectre-fixed-q-mode parallel \
  --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --specstream-profile-only \
  --specstream-profile-path profiles/innovation2_step1/B0_16k_c8_r1.csv \
  --page-size 1 \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation2_step1/B0_16k_c8_r1_target.log
```

等待 `The server is fired up and ready to roll!`。此命令不能出现 `--specstream-smctrl-enabled`。

### 6.3 终端 B：启动 Drafter

```bash
cd ~/lifei/SpecStream
conda activate spectre

CUDA_VISIBLE_DEVICES=1 \
python -m sglang.launch_server \
  --model-path "$DRAFT_MODEL" \
  --port 30001 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role draft \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --spectre-draft-priority \
  --spectre-max-draft-priority-steps 8 \
  --mem-fraction-static 0.45 \
  --max-total-tokens 196608 \
  --max-running-requests 8 \
  --spectre-max-batch-size 8 \
  --chunked-prefill-size 2048 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation2_step1/B0_16k_c8_r1_draft.log
```

目的：Drafter 独占 GPU1，得到不存在同卡干扰的参考 Draft RTT 和 accept length。

### 6.4 终端 C：健康检查和小请求

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30001/health

python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$TARGET_MODEL" \
  --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$DATASET" \
  --num-prompts 1 \
  --random-input-len 1024 \
  --random-output-len 16 \
  --random-range-ratio 1 \
  --request-rate inf \
  --max-concurrency 1 \
  --warmup-requests 0 \
  --seed 1 \
  --output-details \
  --tag I2-S1-B0-smoke \
  --output-file results/innovation2_step1/B0_smoke.jsonl
```

目的：先用低成本请求确认 HTTP 与 ZMQ 链路，避免直接用 16K 请求等待很久才发现服务错误。

### 6.5 终端 C：正式 B0 测试

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$TARGET_MODEL" \
  --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$DATASET" \
  --num-prompts 80 \
  --random-input-len 15360 \
  --random-output-len 128 \
  --random-range-ratio 1 \
  --request-rate inf \
  --max-concurrency 8 \
  --warmup-requests 8 \
  --seed 1 \
  --flush-cache \
  --output-details \
  --tag I2-S1-B0-16k-c8-r1 \
  --output-file results/innovation2_step1/B0_16k_c8_r1.jsonl
```

### 6.6 终端 D：监控和错误检查

benchmark 期间运行：

```bash
nvidia-smi dmon -s pucm -d 1 -c 120 \
  > results/innovation2_step1/B0_16k_c8_r1_dmon.txt
```

结束后运行：

```bash
grep -nE \
  'Traceback|Scheduler hit an exception|DraftFallback|Failed to send' \
  logs/innovation2_step1/B0_16k_c8_r1_*.log || true
```

通过条件：80 个请求全部完成、`error_count=0`、两个服务不崩溃。完成 r1 后停止两端，把所有文件名中的 r1 改成 r2/r3，完整重复。

## 7. I2-S1-B1：同卡、无 execution control

### 7.1 本组目的

Target 与 Drafter 都使用 GPU0。MPS 开启基础共享，但不设置百分比、不启用 libsmctrl、不发送 execution grant。它测量最直接的共卡争抢，是后续动态 gating 必须超越的负面对照。

### 7.2 启动 MPS

确保 B0 两端都已经停止，然后：

```bash
echo quit | \
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
  nvidia-cuda-mps-control 2>/dev/null || true

mkdir -p "$MPS_PIPE" "$MPS_LOG"

CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
nvidia-cuda-mps-control -d

echo get_server_list | \
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
  nvidia-cuda-mps-control
```

目的：允许两个 CUDA 进程共卡。`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 必须保持 unset，否则 B1 就不再是无控制基线。

### 7.3 终端 A：B1 Target 完整命令

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" \
  --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role target \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --spectre-fixed-q-mode parallel \
  --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --specstream-profile-only \
  --specstream-profile-path profiles/innovation2_step1/B1_16k_c8_r1.csv \
  --page-size 1 \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation2_step1/B1_16k_c8_r1_target.log
```

### 7.4 终端 B：B1 Drafter 完整命令

```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
python -m sglang.launch_server \
  --model-path "$DRAFT_MODEL" \
  --port 30001 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE \
  --spectre-role draft \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --spectre-draft-priority \
  --spectre-max-draft-priority-steps 8 \
  --mem-fraction-static 0.45 \
  --max-total-tokens 196608 \
  --max-running-requests 8 \
  --spectre-max-batch-size 8 \
  --chunked-prefill-size 2048 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation2_step1/B1_16k_c8_r1_draft.log
```

检查：A、B 两条命令都没有 `--specstream-smctrl-enabled`、`--specstream-smctrl-calibration-*`、`--specstream-coexec-resource-profile-path`。

### 7.5 终端 C：B1 benchmark

先运行与 B0 相同的两个 `/health` 和 1 请求 smoke，再运行：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$TARGET_MODEL" \
  --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$DATASET" \
  --num-prompts 80 \
  --random-input-len 15360 \
  --random-output-len 128 \
  --random-range-ratio 1 \
  --request-rate inf \
  --max-concurrency 8 \
  --warmup-requests 8 \
  --seed 1 \
  --flush-cache \
  --output-details \
  --tag I2-S1-B1-16k-c8-r1 \
  --output-file results/innovation2_step1/B1_16k_c8_r1.jsonl
```

完成后按 B0 相同方式检查日志、保存 dmon，并做 r2、r3。

## 8. 扩展负载矩阵

先确保 16K bucket/c8 的 B0、B1 各 3 次完全通过，再扩展。这里用 15360 输入并限制输出长度，是为了避免请求在生成过程中跨过 16384 边界而混入 32K bucket：

```text
context length: 16K, 30K
max concurrency: 1, 4, 8, 16, 32
output length: 128 或预先固定的 256
repetitions: r1, r2, r3
```

每个 B0/B1 配对必须保持模型、seed、请求数、输入/输出长度、并发完全一致。改变并发时，如果同时改变 `--max-running-requests` 或 `--spectre-max-batch-size`，必须对两个组做同样修改并写进结果说明。

## 9. 汇总命令和每项指标的目的

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation2_step1/*.csv' \
  > results/innovation2_step1/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation2_step1/*.jsonl' \
  > results/innovation2_step1/benchmark_summary.tsv

column -t -s $'\t' results/innovation2_step1/benchmark_summary.tsv | less -S
```

必须解释这些指标：

- output throughput：系统总生成能力；
- P99 TTFT/TPOT/E2E：共卡是否造成尾延迟恶化；
- accept length：草稿是否仍有价值；
- timeout/fallback：Drafter 是否因争抢而来不及返回；
- goodput/GPU：满足服务目标的吞吐除以物理 GPU 数；B0 除以 2，B1 除以 1；
- GPU power/SM/memory：确认两组的实际放置与资源利用。

## 10. step1 通过条件

1. B0、B1 每个正式点都完成 3 次，benchmark `error_count=0`；
2. B0 确认使用两张物理卡，B1 确认只使用一张物理卡；
3. B1 未启用 MPS 百分比、TPC mask 或 grant；
4. B0/B1 除 GPU 放置与基础 MPS 外参数完全一致；
5. 没有死锁或 Target 崩溃；允许 B1 出现可解释的 timeout/fallback，但必须统计；
6. 同时报告总吞吐和 goodput/GPU，不能用一个指标替代另一个。

## 11. 常见问题

### 两个 `/health` 都正常但 benchmark 失败

先看 Target 日志中的第一个异常。benchmark 的 `ClientPayloadError` 通常只是 Target 崩溃后流式响应被截断的结果。

### B1 显存不足

先确认没有残留服务，用 `pgrep -af 'sglang.launch_server'` 和 `nvidia-smi` 检查。不要只为 B1 任意降低模型或输入长度；若必须调整内存参数，B0 必须用相同设置重跑。

### B1 比 B0 更快

先核对 B0/B1 的请求数、warmup、并发、模型、seed、输出长度和是否误开控制。三次重复都稳定后才解释结果。

## 12. 完成后的清理

先在 A、B 中按 `Ctrl+C` 停止服务，再停止 MPS：

```bash
echo quit | \
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
  nvidia-cuda-mps-control
```

不要删除日志、JSONL、CSV 和环境记录；step2/step3 的结论必须引用这些基线。
