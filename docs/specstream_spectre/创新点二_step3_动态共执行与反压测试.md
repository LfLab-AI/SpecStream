# 创新点二 step3：Target-priority 动态共执行与反压（详细执行版）

## 1. 这一步要证明什么

step3 使用 step2 的真实干扰曲线进行在线控制，不再使用固定 TPC 校准参数。它要证明：

1. Target 忙时，Drafter 默认暂停，不抢占未授权的 GPU 资源；
2. 只有实测 profile 证明安全且预测 slack 足够时，Target 才发一个 token 的 `SLACK_FILL` grant；
3. Target 已等待草稿时，只按一个 token 的 `DRAFT_CATCHUP` grant 推进；
4. 每个 grant 必须收到同 epoch 的 `GRANT_ACK` 后才能发下一个；
5. 未校准 shape、过期 grant、Drafter 故障或实测 slowdown 越界时 fail closed，回到 `TARGET_EXCLUSIVE` 或 q=1 fallback。

`desired_q` 与执行许可是两件事。例如 q=8 只表示希望得到 8 个 draft token，不表示 Drafter 可以一次连续跑 8 步；它仍由最多 8 个逐次 ACK 的 one-token grant 完成。

## 2. 开始前必须已有的文件

```bash
cd ~/lifei/SpecStream
conda activate spectre

export TARGET_MODEL=/root/autodl-tmp/model/Qwen2.5-7B-Instruct
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct
export DATASET=$PWD/specstream_prepared/sharegpt_v3_merged.json
export RESOURCE_PROFILE=$PWD/profiles/innovation2_step2/resource_profile.json
export SMCTRL_LIB=$PWD/csrc/specstream_smctrl/build/libsmctrl.so
export SMCTRL_SCOPE=global
export MPS_PIPE=/tmp/specstream-mps-$USER
export MPS_LOG=/tmp/specstream-mps-log-$USER

unset MASK_OFF
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

mkdir -p logs/innovation2_step3
mkdir -p results/innovation2_step3
mkdir -p profiles/innovation2_step3

test -d "$TARGET_MODEL"
test -d "$DRAFT_MODEL"
test -s "$DATASET"
test -s "$RESOURCE_PROFILE"
test -s "$SMCTRL_LIB"
```

任一 `test` 失败都不能启动在线实验。尤其不能手写一个假的 `resource_profile.json` 代替 step2 测量。

## 3. 协议与代码门禁

### 3.1 为什么还要检查 C++ 扩展

step3 会使用 `grant`、`pause` 和 `grant_ack`。旧扩展只认识四个原始动作，会再次产生 `Invalid SpectreAction: pause`。服务进程启动后不会热更新 `.so`，因此复制新代码后要先停掉旧服务并强制重编译。

### 3.2 停止旧服务并重编译

```bash
pgrep -af 'sglang.launch_server|spectre-role'
```

若有本实验的 Target/Drafter，回到对应终端按 `Ctrl+C`。确认停止后执行：

```bash
cd ~/lifei/SpecStream/python/sglang/srt/speculative/spectre/cpp_zmq
python setup.py build_ext --inplace --force
cd ~/lifei/SpecStream

PYTHONPATH=python python - <<'PY'
from sglang.srt.speculative.spectre.cpp_zmq import protocol_schema_version
from sglang.srt.speculative.spectre.spectre_protocol import SpectreAction

version = int(protocol_schema_version())
actions = {item.value for item in SpectreAction}
print("C++ protocol schema =", version)
print("Python actions =", sorted(actions))
assert version == 2
assert {"grant", "pause", "grant_ack"} <= actions
PY
```

通过条件：实际加载并打印 schema 2。若失败，按 step2 第 5 节删除 cpp_zmq 局部 build/`.so` 后重编译，并再次停止、重启两个服务。

### 3.3 CPU 测试

```bash
PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_cpp_protocol_schema.py \
  python/sglang/test/spectre_specstream/test_gpu_grant.py \
  python/sglang/test/spectre_specstream/test_gpu_grant_controller.py \
  python/sglang/test/spectre_specstream/test_coexec_runtime.py \
  python/sglang/test/spectre_specstream/test_resource_profile.py \
  python/sglang/test/spectre_specstream/test_dynamic_q.py
```

目的：检查 schema、one-token quantum、ACK/epoch、slowdown latch、profile fail-closed 和动态 q。

通过条件：全部 passed。

## 4. resource profile 内容门禁

先打印并验证所有 entry：

```bash
PYTHONPATH=python python - <<'PY'
from collections import defaultdict
from sglang.srt.speculative.spectre.specstream.resource_profile import ResourceProfile

path = "profiles/innovation2_step2/resource_profile.json"
profile = ResourceProfile.load(path)
print("gpu =", profile.gpu)
print("target_model =", profile.target_model)
print("draft_model =", profile.draft_model)
print("total_tpcs =", profile.total_tpcs)

safe = defaultdict(list)
for entry in profile.entries:
    print(entry)
    if entry.target_slowdown <= 0.05:
        safe[entry.target_shape].append(entry.draft_tpcs)

required = {f"verify_bs8_q{q}_ctx16k" for q in (2, 4, 6, 8)}
print("safe entries =", dict(safe))
missing = sorted(shape for shape in required if not safe.get(shape))
assert profile.total_tpcs == 54
assert not missing, f"missing <=5% safe entries: {missing}"
PY
```

目的：step3 的动态候选为 q=1/2/4/6/8。q=1 是 Target-only 回退；q=2/4/6/8 必须分别有同 shape 的安全 entry。只校准 `q=5` 无法支持这些动态候选，系统会正确地 fail closed，但也不会产生在线共执行收益。

## 5. 再次验证 Driver 580 global mask

```bash
cd ~/lifei/SpecStream/csrc/specstream_smctrl
unset MASK_OFF
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=0 TPC_HIGH=2
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=0 TPC_HIGH=4
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=4 TPC_HIGH=8
CUDA_VISIBLE_DEVICES=0 make validate-global TPC_LOW=50 TPC_HIGH=54
cd ~/lifei/SpecStream
```

目的：确认当前重启/驱动环境仍能严格限制 TPC。四项都必须出现 global backend 和 `test passed`，且无范围外 SM。

## 6. 启动 MPS

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

目的：提供双进程共卡。不要设置 MPS active-thread percentage；在线 TPC 范围来自 resource profile 和 grant。

## 7. 在线 smoke：先验证协议和状态机

使用四个终端。第一次只跑一个短请求，确认不再出现用户遇到的协议异常。

### 7.1 终端 A：Target 完整命令

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
  --speculative-num-draft-tokens 5 \
  --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 \
  --spectre-initial-recv-timeout-ms 15000 \
  --spectre-failure-threshold 3 \
  --spectre-cooldown-rounds 32 \
  --specstream-profile-only \
  --specstream-dynamic-q \
  --specstream-q-candidates 1,2,4,6,8 \
  --specstream-q-switch-threshold 0.08 \
  --specstream-smctrl-enabled \
  --specstream-grant-token-quantum 1 \
  --specstream-coexec-target-slowdown-budget 0.05 \
  --specstream-coexec-guard-us 200 \
  --specstream-coexec-resource-profile-path "$RESOURCE_PROFILE" \
  --specstream-profile-path profiles/innovation2_step3/online_16k_c8_r1.csv \
  --page-size 1 \
  --attention-backend fa3 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 29000 \
  2>&1 | tee logs/innovation2_step3/online_16k_c8_r1_target.log
```

参数目的：

- `--spectre-draft-timeout-action fallback`：正式在线服务不能因一次缺失草稿杀死 Target；
- `--specstream-dynamic-q`：根据运行历史选 q，但不直接授权 CUDA；
- `--specstream-grant-token-quantum 1`：每次只允许一个 token step；
- slowdown budget 0.05：只接受 step2 中 Target 变慢不超过 5% 的 entry；
- guard 200us：从预测 slack 中扣除安全余量；
- resource profile：禁止在线猜 TPC。

本命令不能加入 `--specstream-smctrl-calibration-tpcs` 或 `--specstream-smctrl-calibration-allow-overlap`，否则测到的是 step2 校准模式。

### 7.2 终端 B：Drafter 完整命令

Target 出现 ready 后执行：

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
  --speculative-num-draft-tokens 5 \
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
  2>&1 | tee logs/innovation2_step3/online_16k_c8_r1_draft.log
```

目的：Drafter 独立进程使用 global mask；所有 Drafter CUDA launch 受当前 grant TPC 范围限制。Target 进程不启用 global mask，继续使用整张卡。

运行请求前必须检查 Drafter 日志中存在：

```text
SpecStream Draft TPC control initialized: scope=global total_tpcs=54
SpecStream remote Drafter libsmctrl ready: scope=global total_tpcs=54
```

缺少任一行都不能开始 benchmark。否则第一次 grant 才会暴露初始化错误，Target 随后只会表现为 15 秒 remote-draft timeout。

### 7.3 终端 C：健康检查和 1 请求 smoke

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
  --tag I2-S3-protocol-smoke \
  --output-file results/innovation2_step3/protocol_smoke.jsonl
```

### 7.4 终端 D：立即检查原始故障是否消失

```bash
grep -nE \
  'Invalid SpectreAction|Failed to send|Scheduler hit an exception|Traceback' \
  logs/innovation2_step3/online_16k_c8_r1_*.log || true

pgrep -af 'sglang.launch_server|spectre-role'
```

通过条件：grep 无输出，两个服务进程仍存在。若出现 `ClientPayloadError`，先看 Target 日志第一个异常；它通常是服务崩溃的后果，不是 aiohttp 根因。

## 8. 16K bucket/c8 正式在线实验

smoke 通过后，在终端 C 执行。这里使用 15360 输入加 256 输出，使上下文在整轮生成中仍不超过 16384，避免从 16K profile 运行到 32K 未校准 shape：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$TARGET_MODEL" \
  --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$DATASET" \
  --num-prompts 200 \
  --random-input-len 15360 \
  --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf \
  --max-concurrency 8 \
  --warmup-requests 8 \
  --seed 1 \
  --flush-cache \
  --output-details \
  --tag I2-S3-online-16k-c8-r1 \
  --output-file results/innovation2_step3/online_16k_c8_r1.jsonl
```

benchmark 期间终端 D 记录 GPU：

```bash
nvidia-smi dmon -s pucm -d 1 -c 300 \
  > results/innovation2_step3/online_16k_c8_r1_dmon.txt
```

结束后检查：

```bash
test -s profiles/innovation2_step3/online_16k_c8_r1.csv

grep -nE \
  'Invalid SpectreAction|Failed to send|Scheduler hit an exception|Traceback' \
  logs/innovation2_step3/online_16k_c8_r1_*.log || true

head -n 2 profiles/innovation2_step3/online_16k_c8_r1.csv
```

然后在 A、B 按 `Ctrl+C`，把命令和输出中的 r1 改为 r2、r3，完整重启重复。

## 9. CSV 中每个字段要证明什么

CSV 至少检查这些字段：

- `target_phase`：grant 发出时 Target 所处阶段；
- `grant_state`：应出现 `TARGET_EXCLUSIVE`、`SLACK_FILL` 或 `DRAFT_CATCHUP`；
- `grant_epoch`：同一 request 单调递增；
- `grant_wait_ms`：发出 grant 到收到 ACK 的等待；
- `draft_step_ms`：一个已授权 token step 的实际耗时；
- `draft_tpc_low/high`：必须对应 resource profile 中的范围；
- `target_forward_ms`：在线 Target slowdown 复核依据；
- q/mode/fallback：动态 q 与安全回退是否工作。

协议/状态机通过条件：

1. 每个 `GRANT` 的 `grant_tokens=1`；
2. 仅 `PREFILL_DEFERRED` ACK 可以是 0，而且 ACK 前不得发生 Draft CUDA forward；
3. 同一 request 的 epoch 单调递增，过期 ACK 不推进状态；
4. 一个 ACK 到达前没有下一 grant；
5. 未匹配 profile 时不发 `SLACK_FILL`；
6. Drafter 没有 grant 时保持 paused；
7. Target 始终保持 full-device，不被 Drafter global mask 污染。

## 10. 正式负载矩阵

建议顺序从小到大，先发现问题再增加成本：

```text
context: 16K, 30K
max concurrency: 1, 4, 8, 16, 32
q candidates: 1,2,4,6,8（由在线控制器选择）
repetitions: r1, r2, r3
```

每个点都要：

1. 确认 step2 profile 有该 `verify_bs{batch}_q{q}_ctx{bucket}` 的 entry，或明确预期 fail closed；
2. 修改 Target/Drafter 的 `--max-running-requests`、`--spectre-max-batch-size` 时两边保持一致；
3. 修改三个输出文件名，避免覆盖；
4. 完整重启两个服务；
5. health、benchmark、错误 grep、CSV 检查全部执行；
6. 与 step1 B0、B1 的完全相同 shape 结果配对比较。

如果 profile 只校准了 bs8/16K，那么其他 batch/context 是安全性测试，不是收益测试。要宣称其他 shape 有动态收益，必须回到 step2 为它们补做 3 次校准；不得使用最近邻插值。

## 11. 故障与反压测试

故障测试与正式性能轮次分开，使用单独日志/CSV。

### 11.1 未校准 shape：验证 fail closed

如果 profile 没有 bs1/2K entry，可运行：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$TARGET_MODEL" \
  --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$DATASET" \
  --num-prompts 8 \
  --random-input-len 2048 \
  --random-output-len 64 \
  --random-range-ratio 1 \
  --request-rate 1 \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --seed 9 \
  --output-details \
  --tag I2-S3-unknown-shape \
  --output-file results/innovation2_step3/unknown_shape.jsonl
```

预期：无匹配 entry 时保持 `TARGET_EXCLUSIVE` 或 q=1 fallback；不能猜测 TPC，不能崩溃。

### 11.2 暂停 Drafter：验证 timeout/fallback

先在终端 C 启动一个持续 benchmark，再在终端 D 执行：

```bash
DRAFT_PID=$(pgrep -f 'spectre-role draft' | head -n 1)
test -n "$DRAFT_PID"
kill -STOP "$DRAFT_PID"
sleep 5
kill -CONT "$DRAFT_PID"
curl -fsS http://127.0.0.1:30000/health
```

预期：Target 记录 timeout/fallback，但不会因正式命令使用 `fallback` 而退出；Drafter 恢复后，新 request 能建立新的 grant/epoch 序列。该轮只验证容错，不计入性能中位数。

### 11.3 错误 profile：验证 Target 启动失败

停止正常服务后，把 Target 命令中的 profile 临时改为不存在路径：

```bash
export BAD_PROFILE=$PWD/profiles/innovation2_step2/not_found.json
test ! -e "$BAD_PROFILE"
```

用 `--specstream-coexec-resource-profile-path "$BAD_PROFILE"` 启动 Target。预期启动阶段明确报错，不得静默改用 MPS 百分比或随机 TPC。

### 11.4 错误 library：验证 Drafter 启动失败

把 Drafter 命令的 library 临时改为：

```bash
export BAD_SMCTRL=$PWD/csrc/specstream_smctrl/build/not_found.so
test ! -e "$BAD_SMCTRL"
```

预期 Drafter 启动失败并指出 library/mask 初始化问题，不得无 mask 继续 forward。

## 12. 结果汇总和对照

```bash
python scripts/specstream/summarize_specstream_profile.py \
  'profiles/innovation2_step3/*.csv' \
  > results/innovation2_step3/profile_summary.tsv

python scripts/specstream/summarize_benchmarks.py \
  'results/innovation2_step3/*.jsonl' \
  > results/innovation2_step3/benchmark_summary.tsv

sed -n '1,30p' results/innovation2_step3/profile_summary.tsv
sed -n '1,30p' results/innovation2_step3/benchmark_summary.tsv
```

每个 shape 要并排报告：

- step1 B0：分卡两 GPU 强基线；
- step1 B1：同卡无控制负面对照；
- step2：最佳安全固定 TPC 点；
- step3：在线动态 gating；
- output throughput、P99 TTFT/TPOT/E2E、accept length；
- Target slowdown、Draft timeout/fallback、grant 状态分布；
- 物理 GPU 数与 goodput/GPU。

## 13. step3 最终通过条件

1. schema=2，日志不再出现 `Invalid SpectreAction`；
2. 所有正常轮次请求结果正确、`error_count=0`，无死锁和进程崩溃；
3. one-token grant、ACK-before-next-grant、epoch 单调规则成立；
4. 在线实测 Target slowdown 不超过 5% 预算；一旦越界，后续决策回到 `TARGET_EXCLUSIVE`；
5. 未校准 shape 和错误/过期 grant 都 fail closed；
6. Drafter STOP/CONT 时 Target 可 fallback 并继续服务；
7. 相比同卡无控制 B1，在至少一个已校准 workload 上稳定提高 goodput/GPU 或降低 P99；
8. 同时诚实报告与两卡 B0 的总吞吐差距和物理 GPU 数。

若 profile 没有安全 entry 或在线没有稳定收益，应报告 No-Go/负结果，不得靠调 MPS 百分比或只挑单次最快结果包装收益。

## 14. 完成后的清理

先在 A、B 中按 `Ctrl+C` 停止 Target 和 Drafter，再执行：

```bash
echo quit | \
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
  nvidia-cuda-mps-control
```

保留 step1/step2/step3 的全部日志、JSONL、CSV、resource profile 和环境版本记录。
