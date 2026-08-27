# 创新点二 step2：单卡静态 TPC 校准与干扰曲线测试（Driver 580 / A800 详细版）

## 1. 这一步到底要证明什么

step2 不是正式的在线动态策略，而是离线校准。它需要回答四个问题：

1. Target 与 Drafter 放在同一张 A800 上时，Drafter 只使用指定 TPC 范围是否真的生效；
2. 不允许重叠时，Target 自身一次 forward 的基准延迟是多少；
3. 允许 Drafter 用 2、4、6、8、12 个 TPC 与 Target 重叠后，Target 变慢多少、Drafter 单步需要多久；
4. 哪些实测点满足 Target slowdown 预算，可以写入 step3 使用的 `resource_profile.json`。

这里的控制量是 libsmctrl 的 TPC mask，不是 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`。MPS 只负责让两个独立 CUDA 进程共享 GPU。

## 2. 本次 `Invalid SpectreAction: pause` 的原因和修复要求

报错不是网络慢，也不是 15 秒超时太短。Python 控制面已经会发送 `grant`、`pause`、`grant_ack`，但旧的 C++ ZMQ 扩展只认识 `draft`、`finish`、`abort`、`reject`，所以发送 `pause` 时直接失败。之后 Target 永远等不到 Drafter 响应，才继续出现：

```text
Failed to send: Invalid SpectreAction: pause
DraftFallback ... after 15000 ms
ClientPayloadError: Response payload is not completed
```

代码已把 C++ 协议升级为 schema 2，并传输全部 grant 字段。复制新代码到服务器后，必须强制重编译扩展并同时重启 Target、Drafter。只改 timeout 无效；旧进程也不会自动加载新 `.so`。

## 3. 终端规划

建议始终使用四个终端，避免把两个服务混在一起：

- 终端 A：Target 服务；
- 终端 B：Drafter 服务；
- 终端 C：健康检查、smoke、benchmark；
- 终端 D：日志检查、`nvidia-smi`、结果提取。

每次更换 `TPC`、`RUN`、输入长度或并发数，都在 A、B 中按 `Ctrl+C` 停掉两个服务，再重新启动。不要只重启其中一个。

## 4. 一次性环境准备

以下命令在四个终端都执行。路径按当前服务器日志填写；若模型路径不同，只修改最上面的三个变量。

```bash
cd ~/lifei/SpecStream
conda activate spectre

export TARGET_MODEL=/root/autodl-tmp/model/Qwen2.5-7B-Instruct
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct
export DATASET=$PWD/specstream_prepared/sharegpt_v3_merged.json
export SMCTRL_LIB=$PWD/csrc/specstream_smctrl/build/libsmctrl.so
export SMCTRL_SCOPE=global
export MPS_PIPE=/tmp/specstream-mps-$USER
export MPS_LOG=/tmp/specstream-mps-log-$USER

unset MASK_OFF
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

mkdir -p logs/innovation2_step2
mkdir -p results/innovation2_step2
mkdir -p profiles/innovation2_step2
mkdir -p profiles/innovation2_step2/samples

test -s "$DATASET"
test -d "$TARGET_MODEL"
test -d "$DRAFT_MODEL"
```

目的：固定所有实验使用的模型、数据集、MPS 目录和输出目录，并确保 Driver 580 下不会误用已经证实无效的 `MASK_OFF=24` stream 私有偏移。

通过条件：四个 `test`/目录检查不报错，`echo ${MASK_OFF-unset}` 输出 `unset`。

## 5. 必做的协议扩展重编译

先确保旧 Target、Drafter 已经停止。检查是否仍有残留服务：

```bash
pgrep -af 'sglang.launch_server|spectre-role'
```

若有输出，回到对应服务终端按 `Ctrl+C`，直到上述命令不再显示本实验的服务。然后执行：

```bash
cd ~/lifei/SpecStream/python/sglang/srt/speculative/spectre/cpp_zmq
python setup.py build_ext --inplace --force
cd ~/lifei/SpecStream
```

验证实际加载的是新扩展：

```bash
PYTHONPATH=python python - <<'PY'
from sglang.srt.speculative.spectre.cpp_zmq import protocol_schema_version
from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction

version = int(protocol_schema_version())
actions = {item.value for item in SpectreAction}
print("C++ protocol schema =", version)
print("Python actions =", sorted(actions))
assert version == 2, version
assert {"grant", "pause", "grant_ack"} <= actions
PY
```

目的：确认运行时不是旧 `.so`。仅看到源码中有 `pause` 不算通过，必须由 Python 实际加载扩展并打印 schema 2。

通过条件：输出 `C++ protocol schema = 2`，且无异常退出。

如果仍是旧版本，执行下面的定点清理后重编译。这里仅删除 cpp_zmq 自己的构建产物，不会删除模型或实验结果：

```bash
cd ~/lifei/SpecStream/python/sglang/srt/speculative/spectre/cpp_zmq
rm -rf build
rm -f spectre_zmq*.so
python setup.py build_ext --inplace --force
cd ~/lifei/SpecStream
```

随后重新运行 schema 检查。不要在旧服务进程上继续试。

## 6. CPU 正确性测试

```bash
cd ~/lifei/SpecStream
PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_cpp_protocol_schema.py \
  python/sglang/test/spectre_specstream/test_sm_controller.py \
  python/sglang/test/spectre_specstream/test_gpu_grant.py \
  python/sglang/test/spectre_specstream/test_gpu_grant_controller.py \
  python/sglang/test/spectre_specstream/test_coexec_runtime.py \
  python/sglang/test/spectre_specstream/test_resource_profile.py
```

目的：在占用 GPU 前检查三类错误：Python/C++ 动作不一致、一个 grant 错误执行多个 token、ACK/epoch 状态机错误。

通过条件：全部 passed；任何 failed 都必须先修复，不能通过加大 timeout 绕过。

## 7. libsmctrl 在 A800/Driver 580 上的硬门禁

### 7.1 编译

```bash
cd ~/lifei/SpecStream/csrc/specstream_smctrl
make config
make build
test -s build/libsmctrl.so
```

### 7.2 在实际用于实验的 GPU0 上验证 global 后端

```bash
unset MASK_OFF
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=0 TPC_HIGH=2
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=0 TPC_HIGH=4
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=4 TPC_HIGH=8
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=50 TPC_HIGH=54
cd ~/lifei/SpecStream
```

目的：验证进程级 QMD/TMD callback 后端，而不是猜 CUDA 13 stream 内部偏移。

每条命令必须同时满足：

- 出现 `using process-global QMD/TMD mask backend`；
- 出现 `test passed`；
- 不出现任何 `SM ... shouldn't be used`；
- 不发生崩溃。

任一条件不满足即为 No-Go，不要运行 step2 性能测试，也不要恢复 `MASK_OFF=24`。

### 7.3 用与 Drafter 相同的 Python wrapper 做启动前检查

validator 通过后，再验证服务实际使用的 ctypes wrapper 能加载同一个库：

```bash
cd ~/lifei/SpecStream
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=python python - <<'PY'
import os
from sglang.srt.speculative.spectre.specstream.sm_controller import SMController

controller = SMController(
    os.environ["SMCTRL_LIB"],
    device_index=0,
    mask_scope="global",
)
print("Python SMController scope =", controller.mask_scope)
print("Python SMController total_tpcs =", controller.total_tpcs)
assert controller.mask_scope == "global"
assert controller.total_tpcs == 54
PY
```

通过条件：打印 `scope = global` 和 `total_tpcs = 54`。这一步只加载并查询库，不会启动 benchmark。

## 8. 启动干净的 MPS

先检查是否有旧 daemon：

```bash
echo get_server_list | \
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
  nvidia-cuda-mps-control 2>/dev/null || true
```

若是上一次本实验留下的 daemon，可停止并重新建立目录：

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

目的：只提供同卡双进程并发环境。整个 step2 都不要设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`。

## 9. S2-2A：先跑 wait-only 基准

这一组的目的，是测 Target forward 的无重叠基线。Target 计算期间不允许 Drafter 重叠；只有 Target 已明确进入等草稿状态时，才给 Drafter 一个 token grant。

特别注意：本组命令中不能出现 `--specstream-smctrl-calibration-allow-overlap`。用户之前把该参数加在 `grant_wait_only.csv` 命令里，会导致文件名写“wait-only”但实际执行 overlap，实验语义错误。

正式资源表需要覆盖 step3 的动态候选 `q=2,4,6,8`。`q=1` 是不使用远端草稿的安全回退，不需要 TPC entry。下面先用 `q=4、r1` 演示：

```bash
export Q=4
export RUN=r1
```

### 9.1 终端 A：Target

```bash
cd ~/lifei/SpecStream
conda activate spectre

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
  --speculative-num-draft-tokens "$Q" \
  --spectre-fixed-q-mode parallel \
  --spectre-require-draft \
  --spectre-draft-timeout-action error \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-smctrl-calibration-tpcs 4 \
  --specstream-smctrl-library "$SMCTRL_LIB" \
  --specstream-profile-path "profiles/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}.csv" \
  --page-size 1 \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 29000 \
  2>&1 | tee "logs/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}_target.log"
```

等待出现 `The server is fired up and ready to roll!` 后再启动 Drafter。

### 9.2 终端 B：Drafter

```bash
cd ~/lifei/SpecStream
conda activate spectre

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
  --speculative-num-draft-tokens "$Q" \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library "$SMCTRL_LIB" \
  --specstream-smctrl-mask-scope "$SMCTRL_SCOPE" \
  --spectre-draft-priority \
  --spectre-max-draft-priority-steps 1 \
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
  2>&1 | tee "logs/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}_draft.log"
```

目的：Drafter 是独立进程，global mask 会限制其全部 CUDA kernel；Target 进程保持 full-device，不加载 Drafter 的 global mask。

Drafter 启动后、运行任何请求之前，日志必须同时出现：

```text
SpecStream Draft TPC control initialized: scope=global total_tpcs=54
SpecStream remote Drafter libsmctrl ready: scope=global total_tpcs=54
```

可在终端 D 检查：

```bash
grep -nE \
  'SpecStream Draft TPC control initialized|SpecStream remote Drafter libsmctrl ready' \
  "logs/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}_draft.log"
```

必须正好看到上述两类初始化信息。若没有，禁止开始 benchmark；这说明服务器仍在运行旧的 worker 初始化代码，或 Drafter 命令缺少 `--spectre-role draft` / `--specstream-smctrl-enabled`。

### 9.3 终端 C：健康检查与最小 smoke

先确认两个 HTTP 端口分别可用：

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30001/health
```

先跑 1 个短请求，目的是验证协议链路，不是测性能：

```bash
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
  --tag I2-S2-protocol-smoke \
  --output-file results/innovation2_step2/protocol_smoke.jsonl
```

### 9.4 终端 D：smoke 后立即查错

```bash
grep -nE \
  'Invalid SpectreAction|Failed to send|DraftFallback|Scheduler hit an exception|ClientPayloadError' \
  "logs/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}_target.log" \
  "logs/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}_draft.log" || true
```

通过条件：grep 无输出，Target 和 Drafter 都仍在运行。若再次出现 `Invalid SpectreAction`，说明服务器仍加载旧 `.so`；停止两端，回到第 5 节重编译，不能继续 benchmark。

### 9.5 终端 C：正式 wait-only 负载

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
  --tag "I2-S2-wait-q${Q}-bs8-ctx16k-${RUN}" \
  --output-file "results/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}.jsonl"
```

完成后在 A、B 中按 `Ctrl+C`。对 `Q=2,4,6,8` 分别运行 `r1,r2,r3`；每次修改 `Q/RUN` 后完整重启两端。三次必须使用相同参数和 seed。

## 10. S2-2B：固定 TPC overlap 校准

本组允许 Target forward 期间发出固定 TPC 的 `SLACK_FILL` grant，用来测干扰曲线。除多一个 overlap 参数和输出文件名外，其他参数必须与 wait-only 相同。

### 10.1 选择当前扫描点

下面先从 TPC=4、r1 开始：

```bash
export TPC=4
export RUN=r1
export Q=4
export SHAPE=verify_bs8_q4_ctx16k
export DRAFT_BS=8
export DRAFT_CTX_BUCKET=16k
```

### 10.2 终端 A：overlap Target

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
  --speculative-num-draft-tokens "$Q" \
  --spectre-fixed-q-mode parallel \
  --spectre-require-draft \
  --spectre-draft-timeout-action error \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --specstream-profile-only \
  --specstream-smctrl-enabled \
  --specstream-smctrl-calibration-tpcs "$TPC" \
  --specstream-smctrl-calibration-allow-overlap \
  --specstream-smctrl-library "$SMCTRL_LIB" \
  --specstream-profile-path "profiles/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}.csv" \
  --page-size 1 \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 29000 \
  2>&1 | tee "logs/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}_target.log"
```

### 10.3 终端 B：overlap Drafter

Drafter 启动命令与第 9.2 节完全相同，只修改日志名：

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
  --speculative-num-draft-tokens "$Q" \
  --specstream-smctrl-enabled \
  --specstream-smctrl-library "$SMCTRL_LIB" \
  --specstream-smctrl-mask-scope "$SMCTRL_SCOPE" \
  --spectre-draft-priority \
  --spectre-max-draft-priority-steps 1 \
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
  2>&1 | tee "logs/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}_draft.log"
```

### 10.4 终端 C：正式 overlap 负载

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
  --tag "I2-S2-overlap-q${Q}-tpc${TPC}-bs8-ctx16k-${RUN}" \
  --output-file "results/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}.jsonl"
```

### 10.5 每一轮结束后的强制检查

```bash
test -s "profiles/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}.csv"

grep -nE \
  'Invalid SpectreAction|Failed to send|DraftFallback|Scheduler hit an exception|Traceback' \
  "logs/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}_target.log" \
  "logs/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}_draft.log" || true

head -n 2 "profiles/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}.csv"
```

通过条件：无协议/调度异常；CSV 非空；表头至少能看到 `batch_size`、`target_forward_ms`、`draft_step_ms`、`grant_state`、`grant_epoch`、`draft_tpc_low`、`draft_tpc_high`。正式命令使用 15360 输入和 128 输出，是为了让整轮保持在 16K bucket 内，避免跨到 32K 后把不同 shape 混合统计。

## 11. 完整扫描矩阵怎么执行

对每个 q 和 TPC 组合做 3 次独立运行：

```text
Q = 2, 4, 6, 8
TPC = 2, 4, 6, 8, 12
RUN = r1, r2, r3
shape = verify_bs8_q${Q}_ctx16k
draft_bs = 8
draft_ctx_bucket = 16k
```

每个点的操作顺序固定为：

1. 在 A、B 中停止上一个点；
2. 修改 `export Q=...`、`export TPC=...` 和 `export RUN=...`；
3. 先启动 A，看到 ready；
4. 再启动 B，看到 ready；
5. C 检查两个 `/health`；
6. C 运行 benchmark；
7. D 执行错误 grep 和 CSV 检查；
8. 保存日志，再进入下一个点。

不要在服务运行中修改 `Q/TPC`；环境变量不会改变已启动进程。不要把多个重复写入同一个 CSV，否则无法证明独立重复。

## 12. 从 CSV 生成实测样本

wait-only 与 overlap 必须是完全相同的 q、shape、并发和输入长度。下面以 q=4、TPC=4、r1 为例：

```bash
python scripts/specstream/smctrl/extract_interference_sample.py \
  --baseline-profile profiles/innovation2_step2/wait_q4_bs8_ctx16k_r1.csv \
  --overlap-profile profiles/innovation2_step2/overlap_q4_tpc4_bs8_ctx16k_r1.csv \
  --target-shape verify_bs8_q4_ctx16k \
  --draft-bs 8 \
  --draft-ctx-bucket 16k \
  --draft-tpcs 4 \
  --output profiles/innovation2_step2/interference_samples.jsonl
```

目的：从 baseline 取 `target_baseline_ms`，从 overlap 取 `target_latency_ms` 和 `draft_step_ms`，形成一个可审计样本。提取脚本会按 q、`batch_size` 和 context bucket 严格过滤，warmup、batch ramp-up 和其他 shape 不会被错误标成 bs8/16K。

确认示例能成功后，先移走或删除仅用于试验的旧样本文件，再用下面的确定性循环一次提取全部 60 个已经完成的测量点：

```bash
mv profiles/innovation2_step2/interference_samples.jsonl \
  profiles/innovation2_step2/interference_samples.before_full_scan.jsonl \
  2>/dev/null || true

for Q in 2 4 6 8; do
  for TPC in 2 4 6 8 12; do
    for RUN in r1 r2 r3; do
      python scripts/specstream/smctrl/extract_interference_sample.py \
        --baseline-profile "profiles/innovation2_step2/wait_q${Q}_bs8_ctx16k_${RUN}.csv" \
        --overlap-profile "profiles/innovation2_step2/overlap_q${Q}_tpc${TPC}_bs8_ctx16k_${RUN}.csv" \
        --target-shape "verify_bs8_q${Q}_ctx16k" \
        --draft-bs 8 \
        --draft-ctx-bucket 16k \
        --draft-tpcs "$TPC" \
        --output profiles/innovation2_step2/interference_samples.jsonl
    done
  done
done
```

这只聚合已经存在的 CSV，不会启动 GPU 任务。对 q=2/4/6/8、TPC=2/4/6/8/12 的 r1/r2/r3 分别提取，共应得到 60 行：

```bash
wc -l profiles/innovation2_step2/interference_samples.jsonl
```

通过条件：输出 `60 .../interference_samples.jsonl`。如果某个 CSV 报 `has no positive draft_step_ms samples`，该轮没有产生有效 grant，不能伪造数据，必须查看 Target/Drafter 日志并重跑。

## 13. 构建 step3 使用的 resource profile

```bash
python scripts/specstream/smctrl/build_resource_profile.py \
  profiles/innovation2_step2/interference_samples.jsonl \
  --output profiles/innovation2_step2/resource_profile.json \
  --gpu 'NVIDIA A800 80GB PCIe' \
  --draft-model "$DRAFT_MODEL" \
  --target-model "$TARGET_MODEL" \
  --total-tpcs 54 \
  --min-repetitions 3
```

验证文件可被在线控制器读取：

```bash
PYTHONPATH=python python - <<'PY'
from sglang.srt.speculative.spectre.specstream.resource_profile import ResourceProfile

path = "profiles/innovation2_step2/resource_profile.json"
profile = ResourceProfile.load(path)
print("gpu =", profile.gpu)
print("total_tpcs =", profile.total_tpcs)
print("entries =", len(profile.entries))
for entry in profile.entries:
    print(entry)
assert profile.total_tpcs == 54
assert len(profile.entries) >= 20
PY
```

目的：生成 step3 唯一允许使用的、由真实重复测量构成的控制表。脚本按三次重复的中位数计算 slowdown，不插值、不猜未测试 shape。

## 14. 汇总结果

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation2_step2/*.csv' \
  > results/innovation2_step2/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation2_step2/*.jsonl' \
  > results/innovation2_step2/benchmark_summary.tsv

sed -n '1,20p' results/innovation2_step2/profile_summary.tsv
sed -n '1,20p' results/innovation2_step2/benchmark_summary.tsv
```

需要报告：每个 TPC 的 Target slowdown、Drafter 单步延迟、吞吐、P99、错误数、三次重复离散程度。不能只选最快的一次。

## 15. step2 最终通过条件

只有同时满足以下条件才进入 step3：

1. C++ ZMQ 运行时 schema 为 2；日志中不再出现 `Invalid SpectreAction: pause/grant/grant_ack`；
2. A800/Driver 580 的四个 `validate-global` 范围都严格通过；
3. wait-only 与每个 overlap 点都完成 3 次，benchmark `error_count=0`；
4. CSV 有真实 `grant_epoch`、`draft_step_ms`、Target forward 数据；
5. `resource_profile.json` 可加载，每个 entry 至少 3 次重复；
6. step3 的每个远端草稿候选 q=2/4/6/8 都至少存在一个 `target_slowdown <= 0.05` 的安全 TPC 点；q=1 是无远端草稿回退。缺失的 q 必须 fail closed，不能用另一 q 的 entry 冒充；
7. 若没有任何安全点，创新点二应报告 No-Go，而不是放宽数据或改用 MPS 百分比。

## 16. 常见故障对照

### 16.1 `Invalid SpectreAction: pause`

原因：旧 C++ 扩展仍被加载。处理：停止两个服务，执行第 5 节强制重编译，确认 schema=2，再启动。

### 16.2 `required a remote draft ... no valid response`

先向前查找是否已经出现 `Failed to send`、Drafter 崩溃或 schema 错误。只有协议正确、Drafter 正常且只是长上下文首次编译较慢时，才考虑临时增大 initial timeout；增大 timeout 不能修复协议错误。

如果 debug 日志显示 Target 已经收到 Drafter 的 `UNPACK`，Drafter 也已发出 `spec_cnt=0` 响应，但下一轮 `spec_cnt=1` 在 overlap 模式下仍整批超时，检查 Drafter 是否只打印了 `[Draft][Resume] ... (pending)` 而没有后续 `GRANT_ACK`。这不是 ZMQ 断链，而是零历史 slack 被错误转成 1 微秒 `SLACK_FILL` deadline：grant 在到达 Drafter 前已过期，而旧实现又静默删除过期 grant，使 Target 的 outstanding epoch 永远无法转入 `DRAFT_CATCHUP`。更新后的实现对显式 calibration overlap 保留可撤销的单 token grant，并对其他过期 grant 发送 `grant_tokens=0, grant_state=EXPIRED` ACK，不再死锁。该修复是 Python 代码，同步后必须停止并重启 Target/Drafter；如果同时更新了 C++ 协议文件，仍按第 5 节重编译。

debug 参数必须放在管道前，不能写在 `tee` 后面：

```bash
python -m sglang.launch_server \
  ... \
  --log-level debug \
  2>&1 | tee logs/innovation2_step2/overlap_debug_target.log
```

若写成 `... | tee file.log --log-level debug`，`--log-level` 会被 `tee` 解析，并报 `tee: unrecognized option '--log-level'`。

如果 Drafter 日志先出现 `SpecStream Draft TPC mask requested but libsmctrl is not initialized`，说明旧代码错误地用内部 EAGLE/MTP 的 `is_draft_worker` 标志判断远端 SPECTRE Drafter，导致启动时跳过初始化。更新 `tp_worker.py` 后，必须完整重启 Drafter，并先确认第 9.2 节的两条 ready 日志。

### 16.3 benchmark 的 `ClientPayloadError`

这是 Target 子进程崩溃后 HTTP 流被截断的下游症状。先看 Target 日志中的第一个异常，不要把它当作 aiohttp 网络问题。

### 16.4 `SM ... shouldn't be used`

说明 mask 没有生效。Driver 580 下只能判定该后端失败，禁止继续性能实验，也禁止尝试 `MASK_OFF=24`。

## 17. 完成后的清理

先在 A、B 中按 `Ctrl+C` 停止两个服务，再停止本实验 MPS：

```bash
echo quit | \
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
  nvidia-cuda-mps-control
```

保留 `logs/innovation2_step2`、`results/innovation2_step2`、`profiles/innovation2_step2`，它们是 step3 和最终论文结果的证据。
