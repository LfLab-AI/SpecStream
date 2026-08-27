# 创新点一与创新点二 step3：分层缓存、动态共执行与高并发反压

## 1. 本步测试目的

本步在 Step 2 找到的最佳静态 MPS 配额上开启卡内资源感知策略，验证：

1. 控制器能否在 `COEXEC`、`THROTTLE`、`SERIALIZE`、`FALLBACK` 之间安全切换；
2. 动态 q 与 Draft 压力联合控制是否优于最佳静态配额；
3. 高并发或 Draft 暂停时，Target 是否能继续服务而不崩溃。

下面假设 Step 2 选出的最佳配额为 Target 80%、Draft 20%。如果您的最佳配额不同，请同时修改 Target 和 Draft 命令中的百分比。

本步同时启用创新点一和创新点二：较老 Target KV 卸载到 CPU，同时运行动态 q、卡内共执行和反压控制。它必须与“不使用创新点一”的创新点二 step3 使用相同模型、MPS 配额和 benchmark。

## 2. 运行代码测试

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

mkdir -p logs/innovation12_step3 results/innovation12_step3 profiles/innovation12_step3

PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_dynamic_q.py \
  python/sglang/test/spectre_specstream/test_draft_load_tracker.py \
  python/sglang/test/spectre_specstream/test_single_gpu_coexec_policy.py \
  python/sglang/test/spectre_specstream/test_circuit_breaker_fallback.py \
  python/sglang/test/spectre_specstream/test_control_profile.py
```

目的：确认 q 上限、Draft pressure、模式切换、circuit breaker 和 profile 字段正确。

## 3. 启动 MPS

```bash
mkdir -p /tmp/specstream-mps-$USER /tmp/specstream-mps-log-$USER

CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control -d
```

目的：为 Target/Draft 同卡实验提供静态资源上限。Step 2 baseline 和本步优化组必须使用同一配额。

## 4. 终端 A：启动动态控制 Target

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
  --specstream-dynamic-q --specstream-q-candidates 1,2,4,6,8 \
  --specstream-q-switch-threshold 0.08 \
  --specstream-coexec-enabled --specstream-coexec-require-mps \
  --specstream-coexec-draft-pressure-ratio 0.80 \
  --specstream-coexec-timeout-rate-threshold 0.10 \
  --specstream-coexec-pending-high-watermark 16 \
  --specstream-coexec-compute-ratio-threshold 0.90 \
  --specstream-profile-path profiles/innovation12_step3/O1_dynamic.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation12_step3/O1_target.log
```

目的：开启创新点二完整动态策略。MPS 的 80% 是静态上限；运行时动态改变的是 q 和执行模式，不是每个 kernel 重新分配 MPS 百分比。

## 5. 终端 B：启动同卡 Draft

看到 Target ready 后执行：

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
  2>&1 | tee logs/innovation12_step3/O1_draft.log
```

目的：Draft 与 Target 共用 GPU 0，但受 20% MPS 上限约束。

## 6. 终端 C：smoke

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30001/health

python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /common_data/model/Qwen2.5-7B-Instruct \
  --tokenizer /common_data/model/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --dataset-path /home/lifei/lifei/specdecode/baseline/sglang/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 8 \
  --random-input-len 16384 --random-output-len 64 \
  --random-range-ratio 1 \
  --request-rate 1 --max-concurrency 1 \
  --warmup-requests 1 --seed 1 --flush-cache --output-details \
  --tag I12-S3-O1-smoke \
  --output-file results/innovation12_step3/O1_smoke.jsonl
```

目的：确认动态控制器、MPS 环境和 Draft 通信正常。profile 中应出现 q、mode 和 coexec_mode 字段。

## 7. 16K 并发主实验

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /common_data/model/Qwen2.5-7B-Instruct \
  --tokenizer /common_data/model/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --dataset-path /home/lifei/lifei/specdecode/baseline/sglang/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 32 \
  --random-input-len 16384 --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf --max-concurrency 8 \
  --warmup-requests 8 --seed 1 --flush-cache --output-details \
  --tag I12-S3-O1-16k-c8 \
  --output-file results/innovation12_step3/O1_16k_c8.jsonl
```

目的：比较 O1 与 Step 2 最佳静态 baseline。之后分别测试并发 `1、4、16、32`；每个点重启 Target/Draft，并修改 profile、tag 和输出文件名。

## 8. Poisson 到达率测试

先测试 request-rate=4：

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /common_data/model/Qwen2.5-7B-Instruct \
  --tokenizer /common_data/model/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --dataset-path /home/lifei/lifei/specdecode/baseline/sglang/specstream_prepared/sharegpt_v3_merged.json \
  --num-prompts 300 \
  --random-input-len 16384 --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate 4 --max-concurrency 32 \
  --warmup-requests 8 --seed 1 --flush-cache --output-details \
  --tag I12-S3-O1-poisson-r4 \
  --output-file results/innovation12_step3/O1_poisson_r4.jsonl
```

目的：验证真实到达率变化下，低压力时能使用较积极 q，高压力时能转为保守模式。之后把 request-rate 改为 `1、2、8、16`。

## 9. Draft 暂停故障测试

在额外终端找到 Draft PID：

```bash
pgrep -af 'spectre-role draft'
```

把下面的数字 `12345` 替换为实际 Draft PID：

```bash
kill -STOP 12345
sleep 5
kill -CONT 12345

curl -fsS http://127.0.0.1:30000/health
```

目的：人为制造 Draft timeout。预期行为是 q 下降，并出现 `THROTTLE/SERIALIZE/FALLBACK`；Target 必须仍然存活。故障测试不能混入正式性能数据。

## 10. 汇总命令

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation12_step3/*.csv' \
  > results/innovation12_step3/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation12_step3/*.jsonl' \
  > results/innovation12_step3/benchmark_summary.tsv
```

目的：检查 `q_dist`、`coexec_dist`、Draft RTT、timeout、fallback、吞吐、P99 和 `goodput/GPU`。

## 11. 本步通过条件

1. O1 相比最佳静态 baseline，在至少一种混合负载上提高 `goodput/GPU` 或降低 P99。
2. C=16/32 和 Draft 暂停时不崩溃、不死锁。
3. Draft 过载时能先限 q/限并行，严重时回退；恢复后可以重新使用推测解码。
4. O1 与静态 baseline 使用完全相同的 MPS 配额。
5. 同时报告与两卡分卡 baseline 的总吞吐差距和单位 GPU 效率。
6. `cpu_history_bytes` 和 `h2d_ops` 必须大于 0，同时 controller 的 q/模式分布不能全为空。

### 可选联合完整组：增加分块分组

停止当前 Target 后，在第 4 节 Target 命令中额外增加：

```bash
--specstream-cohort-enabled \
--specstream-max-cohort-size 8 \
--specstream-max-cohort-delay-us 200
```

同时把 profile、tag 和结果文件名中的 `O1` 改成 `O2_cohort`。目的：检查创新点一的分块分组能否进一步降低每个接受 token 对应的 H2D 次数；该组不能代替不启用 Cohort 的 O1。

## 12. 停止 MPS

```bash
echo quit | \
CUDA_MPS_PIPE_DIRECTORY=/tmp/specstream-mps-$USER \
CUDA_MPS_LOG_DIRECTORY=/tmp/specstream-mps-log-$USER \
nvidia-cuda-mps-control
```

目的：避免 MPS 环境影响后续实验。
