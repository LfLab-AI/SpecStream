# 创新点一 step2：输入输出感知动态步长与串并行联合控制测试

## 1. 前置条件和测试目标

只有 Step 1 的 B3 fixed-q fused streaming 已通过，才能开始本步骤。Step 2 不改变 CPU History、staging、online softmax 和 Tail merge 数据路径，只在每个 verification batch/round 统一选择：

```text
(mode, q) = (ordinary 或 parallel, 本轮 verification horizon)
```

控制器同时考虑 History I/O、Target attention、network wait、acceptance/rollback、batch 状态以及 Remote Drafter 的实测 deadline pressure/timeout rate。当前实现是 batch-level q，不是 per-request q；TP=2 时由 rank 0 决策后广播。任一请求缺少草稿时，由于 Target verification shape 必须 batch-uniform，该轮整个 batch 安全降级到 q=1。

主数据集路径：

```bash
export SHAREGPT_V3_ROOT=/common_data/dataset/ShareGPT_V3
export PREPARED_ROOT=/common_data/dataset/specstream_prepared
export SHAREGPT_JSON=$PREPARED_ROOT/sharegpt_v3_merged.json
export BASE_URL=http://127.0.0.1:30000
mkdir -p logs/specstream results/bench results/profiles
```

如果服务器只同步了 `srt`，所有原始 benchmark 命令仍可执行；使用 `summarize_benchmarks.py` 和 `summarize_specstream_profile.py` 前需额外同步 `scripts/specstream/`。

主控制实验使用：

```text
dataset-name=random
dataset-path=/common_data/dataset/specstream_prepared/sharegpt_v3_merged.json
```

原因是它可以构造严格相同的 4K/16K/30K 输入长度，使 q 的比较不受 prompt 长度分布干扰。

本步骤只使用三个 ShareGPT V3 split 合并得到的 `$SHAREGPT_JSON`。必须先按 Step 1 第 4 节完成三个 split 的 schema 检查、完整合并和合并文件复核。

## 2. 单元测试

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_draft_delivery_policy.py \
  python/sglang/test/spectre_specstream/test_dynamic_q.py \
  python/sglang/test/spectre_specstream/test_circuit_breaker_fallback.py \
  python/sglang/test/spectre_specstream/test_state_invariants.py \
  python/sglang/test/spectre_specstream/test_spectre_fixed_q_e2e.py \
  python/sglang/test/spectre_specstream/test_spectre_tp2.py
```

检查候选 q、acceptance EWMA、ordinary/parallel 成本、8% hysteresis、安全回退和 TP 广播。

## 3. 固定 q baseline 的正确配置

在 `topk=1` 下必须保持：

```text
speculative_num_draft_tokens = speculative_num_steps + 1 = q
```

因此：

| 固定 q | `--speculative-num-steps` | `--speculative-num-draft-tokens` |
|---:|---:|---:|
| 2 | 1 | 2 |
| 4 | 3 | 4 |
| 5 | 4 | 5 |
| 8 | 7 | 8 |

当前固定 CLI 的 q=1 会要求 `num_steps=0`，而 pinned SPECTRE 的部分初始化代码把 0 当作 false/fallback 值，因此不要把 `num_steps=0` 作为可靠 q=1 baseline。q=1 由动态控制器选择时仍可在 profile 中观察；纯 ordinary/AR 应作为单独 baseline。

固定 q 实验必须同时重启 Drafter 和 Target，并在两端使用相同的 steps/tokens。

## 4. 固定 q=2/4/8 启动模板

下面以 q=4 为例。执行顺序必须是终端 B Target 先启动、终端 A Drafter 后启动；Target 使用 `--skip-server-warmup`，待两端 ready 后再发送显式 smoke。

终端 A：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$DRAFT_MODEL" --port 30001 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role draft \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --spectre-draft-priority --spectre-max-draft-priority-steps 8 \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

终端 B：

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode ordinary --spectre-require-draft \
  --spectre-draft-timeout-action error \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --specstream-enabled --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-profile-path results/profiles/D_fixed_q4.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

q=2 时把两端改为 steps=1、draft-tokens=2，profile 改为 `D_fixed_q2.csv`；q=8 时改为 steps=7、draft-tokens=8，profile 改为 `D_fixed_q8.csv`。`error` 只用于 C=1 的固定-q 链路校准；确认实际 q>1 后，正式 serving/高并发实验必须改为 `--spectre-draft-timeout-action fallback`，再按需把 ordinary 改为 parallel 做吞吐对照。

固定 q 的正式性能轮不能继续使用上面的 ordinary 校准参数。保持两端 q 配置一致，并把 Target 的以下参数替换为：

```bash
--spectre-fixed-q-mode parallel --spectre-require-draft \
--spectre-draft-timeout-action fallback \
--spectre-failure-threshold 3 --spectre-cooldown-rounds 32 \
--specstream-layer-prefetch \
--specstream-gpu-reserve-mb 1024
```

结果文件使用 `D_fixed_q${Q}_parallel.csv`。同一个 q 的 ordinary 与 parallel 必须分别重启 Target 和 Drafter、使用相同数据与随机种子。parallel 用于测量 `Draft(next round) || Target verify(current round)` 的收益，但只有 Accept length 与 ordinary 接近且没有持续 DraftFallback 时才是有效性能点；否则应使用 ordinary/dynamic 结果，不能把 timeout 退化归因给 SpecStream kernel。

## 5. 动态 `(mode,q)` 启动

仍按 Target → Drafter 顺序启动。

终端 A 可继续使用用户当前的 Drafter 配置：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$DRAFT_MODEL" --port 30001 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role draft \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --spectre-draft-priority --spectre-max-draft-priority-steps 8 \
  --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

终端 B：

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --specstream-enabled --no-specstream-reference-attention \
  --specstream-chunk-tokens 2048 --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 4 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-dynamic-q \
  --specstream-q-candidates 1,2,4,6,8 \
  --specstream-q-switch-threshold 0.08 \
  --specstream-layer-prefetch \
  --specstream-gpu-reserve-mb 1024 \
  --specstream-profile-path results/profiles/D_dynamic.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

当前 CLI 开启的是联合 `(mode,q)` 控制器，并没有“动态 q 但强制 parallel”的独立开关。若要做 q-only ablation，需要另行增加控制器选项，不能仅靠测试命令伪造。

动态控制器允许主动选择 q=1，所以动态实验的 Accept length 偶尔下降到 1 是正常的；但若全程严格为 `1.00`，必须先停止性能归因。检查 Target 日志是否有持续 timeout，并用 profile 汇总确认 `q_dist`。默认 `fallback` 不会杀死 Target：当前轮执行 batch-uniform q=1，控制器随后进入短 AR backoff；deadline P95 接近 timeout 时，候选上限会收缩到 q≤2。只有 C=1 校准并显式设置 `--spectre-draft-timeout-action error` 才会 fail-fast。

## 6. 每次启动后的检查

```bash
curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/v1/models" | python -m json.tool
```

然后先运行 8 条 4K smoke：

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name random-ids --tokenize-prompt \
  --num-prompts 8 --random-input-len 4096 --random-output-len 64 \
  --random-range-ratio 1 --request-rate 1 --max-concurrency 1 \
  --warmup-requests 1 --seed 1 --flush-cache --output-details \
  --tag D_dynamic_smoke \
  --output-file results/bench/D_dynamic_smoke.jsonl
```

## 7. 控制器实验矩阵

### 7.1 固定 q=2/4/8 的 16K 单请求

当前运行哪个 q，就把 `TAG` 设置成对应值：

```bash
export TAG=D_fixed_q4

python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$SHAREGPT_JSON" \
  --num-prompts 50 \
  --random-input-len 16384 --random-output-len 128 \
  --random-range-ratio 1 \
  --request-rate 1 --max-concurrency 1 \
  --warmup-requests 2 --seed 1 --flush-cache --output-details \
  --tag "${TAG}_16k_c1" \
  --output-file "results/bench/${TAG}_16k_c1.jsonl"
```

### 7.2 动态 q 的 4K/16K/30K 扫描

```bash
for INPUT_LEN in 4096 16384 30000; do
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 50 \
    --random-input-len "$INPUT_LEN" --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate 1 --max-concurrency 1 \
    --warmup-requests 2 --seed 1 --flush-cache --output-details \
    --tag "D_dynamic_${INPUT_LEN}_c1" \
    --output-file "results/bench/D_dynamic_${INPUT_LEN}_c1.jsonl"
done
```

4K 可能没有 sealed History，主要观察 cold-start 和回退；16K/30K 才用于 I/O-aware 决策。

### 7.3 动态 q 的并发扫描

```bash
for CONCURRENCY in 16; do
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 32 \
    --random-input-len 16384 --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate inf --max-concurrency "$CONCURRENCY" \
    --warmup-requests 4 --seed 1 --flush-cache --output-details \
    --tag "D_dynamic_16k_c${CONCURRENCY}" \
    --output-file "results/bench/D_dynamic_16k_c${CONCURRENCY}.jsonl"
done
```

### 7.4 Poisson 到达率扫描

```bash
for RATE in 0.5 1 2 4 8; do
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 50 \
    --random-input-len 16384 --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate "$RATE" --max-concurrency 32 \
    --warmup-requests 4 --seed 1 --flush-cache --output-details \
    --tag "D_dynamic_16k_r${RATE}_c32" \
    --output-file "results/bench/D_dynamic_16k_r${RATE}_c32.jsonl"
done
```

### 7.5 ShareGPT V3 原始长度分布

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name sharegpt --dataset-path "$SHAREGPT_JSON" \
  --num-prompts 300 --sharegpt-output-len 256 \
  --sharegpt-context-len 32768 \
  --request-rate 4 --max-concurrency 16 \
  --warmup-requests 4 --seed 1 --flush-cache --output-details \
  --tag D_dynamic_sharegpt_v3_r4_c16 \
  --output-file results/bench/D_dynamic_sharegpt_v3_r4_c16.jsonl
```

该实验观察控制器在自然长度分布上的 q/mode 行为；固定 16K/30K 仍是 I/O-aware 因果对照的主数据。

## 8. 必须保留的 baseline

最低对照集合：

| 名称 | 说明 |
|---|---|
| D-AR | 非 speculative 的 SGLang Target，作为 q=1/ordinary 参考 |
| D-F2 | fixed q=2 fused SpecStream |
| D-F4 | fixed q=4 fused SpecStream |
| D-F8 | fixed q=8 fused SpecStream |
| D-DYN | 当前联合 `(mode,q)` 控制器 |

所有 baseline 必须使用相同模型、GPU、chunk、tail、数据、seed、输出长度、并发和到达率。

## 9. 分析动态决策

当前实现的权威字段是 CSV 中的 `q` 和 `mode`：

```bash
python scripts/specstream/summarize_specstream_profile.py \
  results/profiles/D_fixed_q2.csv \
  results/profiles/D_fixed_q4.csv \
  results/profiles/D_fixed_q8.csv \
  results/profiles/D_dynamic.csv
```

也可以直接查看分布：

```bash
python - <<'PY'
import csv
from collections import Counter
p = "results/profiles/D_dynamic.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8")))
print("q=", Counter(r["q"] for r in rows))
print("mode=", Counter(r["mode"] for r in rows))
print("history_max=", max(int(r["history_len"] or 0) for r in rows))
print("h2d_GiB=", sum(int(r["h2d_bytes"] or 0) for r in rows) / 2**30)
print("fallback=", Counter(r["fallback_reason"] for r in rows if r["fallback"] == "True"))
print("missing_drafts=", sum(int(r.get("missing_draft_count") or 0) for r in rows))
PY
```

注意：当前 `controller_selected_q`、`controller_selected_mode` 和 controller cost 列尚未在所有路径写入，可能保持 0/空。分析当前运行时决策时使用 `q`、`mode`，不要用预留列得出“控制器未工作”的错误结论。

## 10. 动态控制正确性判据

- 同一个 verification batch 的请求使用相同 q/mode。
- q 始终属于 `1,2,4,6,8`。
- 达到 History 阈值后 profile 中 `h2d_bytes` 和 `stream_attn_ms` 非零。
- 稳态相邻 round 不应无收益地频繁切换；8% hysteresis 应减少抖动。
- reject、高 overhead 或无 draft 场景优先安全回退，不被成本模型覆盖。
- stale `spec_cnt` response 不改变当前 round。
- Drafter 断连时 Target 不能无限等待。
- timeout 时日志出现 `[Target][DraftFallback]`，HTTP 30000 保持可用，当前 batch 使用 q=1；不得出现 Scheduler 因 timeout 退出。
- timeout 后动态控制器出现 `draft_timeout_backoff`；接近 deadline 的成功响应应出现 `draft_pressure_limited` 并限制最大 q。
- TP=2 时两个 rank 的 q、mode、请求顺序和 round 一致。

## 11. 性能和控制效果判据

汇总 benchmark：

```bash
python scripts/specstream/summarize_benchmarks.py results/bench/D_*.jsonl
```

至少报告：

- output throughput；
- mean/P99 TTFT、TPOT、E2E；
- q/mode 分布；
- accepted tokens 和 rollback ratio；
- History、H2D GiB、H2D ops、effective H2D GB/s；
- stream attention、Target forward 和 network wait；
- 不同 ctx、并发和 rate 下的最佳 fixed q。

动态控制器的有效结论不是“总选择更大 q”，而是：

1. 不同 History/I/O/acceptance 区间选择发生变化；
2. D-DYN 在主要适用区间接近或超过最佳 fixed q；
3. D-DYN 的 P99 没有因平均吞吐优化而显著恶化；
4. 相对离线枚举得到的最佳 `(mode,q)`，regret 可接受。

离线 oracle 需要分别运行固定 q=2/4/8 和可获得的 mode baseline，再按场景选最优值；不能只用控制器自己的预测成本充当 oracle。

## 12. 故障注入

至少测试：

- 关闭 Drafter；
- 延迟或丢弃 draft response；
- stale `spec_cnt`；
- 强制 REJECT；
- batch 超过 `spectre_max_batch_size`；
- 低 acceptance 模型对；
- 长 History + 高并发。

每种故障记录 fallback、恢复时间、失败请求数和是否有无限等待。

高并发 timeout 回归至少运行 C=16，并在 benchmark 结束后再次检查服务存活：

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name random --num-prompts 64 \
  --random-input-len 16384 --random-output-len 128 \
  --random-range-ratio 1 --request-rate inf --max-concurrency 16 \
  --warmup-requests 4 --seed 1 --flush-cache --output-details \
  --tag D_dynamic_16k_c16_timeout_safe \
  --output-file results/bench/D_dynamic_16k_c16_timeout_safe.jsonl

curl -fsS "$BASE_URL/health"
```

允许出现：

```text
[Target][DraftFallback] ... using a batch-uniform q=1 round
```

但不允许出现 `Scheduler hit an exception`、`SIGQUIT`、`ConnectionRefusedError`。若 timeout 很多但服务仍存活，也只能判定“容错通过”，不能判定性能通过；需要结合 q/fallback 分布和 Drafter GPU 利用率继续调参。

## 13. Step 2 通过条件

1. 单元测试和 fixed q=2/4/8 全通过。
2. 动态 q 在 16K/30K profile 中真实选择合法 q/mode。
3. 安全 gate、hysteresis、断连和 stale response 行为正确。
4. 动态方案在主要负载下接近/优于最佳 fixed q，并同时报告 P99。
5. 所有决策、benchmark 和 profile 文件可由相同 seed 与请求轨迹复现。
