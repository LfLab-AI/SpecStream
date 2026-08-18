# Step 3：Continuous Serving Chunk-Cohort 完整测试流程

## 1. 前置条件和当前实现边界

开始本步骤前必须满足：

- Step 1 的 B3 fused streaming 正确性和稳定性通过；
- Step 2 的 dynamic `(mode,q)` 能稳定运行；
- 16K/30K profile 已确认 sealed History、H2D 和 stream attention 非零。

Chunk-Cohort 在同一个 Target verification batch 中，把具有兼容 History chunk 的多个请求合并成一次 packed H2D 和一次 cohort attention。兼容键包含：

```text
layer_id, q bucket, dtype, head_dim, local_kv_heads, chunk_tokens
```

每个请求仍维护独立的 online-softmax `(m,l,o)` 状态，padding token 不得进入 softmax。

当前实现的重要边界：cohort 对当前 verification batch 中已经到达的工作进行分组，并不会主动 sleep 等待未来请求。`max_cohort_delay_us` 当前主要参与 deadline/过期语义，不应预期仅调整该参数就出现明显的主动等待曲线。论文若要声称跨调度周期等待/聚合，需要另行实现真正的 pending queue。

## 2. 环境和数据集

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

export TARGET_MODEL=/你的/Target模型目录
export DRAFT_MODEL=/你的/Draft模型目录
export BASE_URL=http://127.0.0.1:30000

export SHAREGPT_V3_ROOT=/common_data/dataset/ShareGPT_V3
export PREPARED_ROOT=/common_data/dataset/specstream_prepared
export SHAREGPT_JSON=$PREPARED_ROOT/sharegpt_v3_merged.json

mkdir -p logs/specstream results/bench results/profiles
```

如果服务器只同步了 `srt`，正文中的 benchmark 命令仍可直接运行；两个结果汇总脚本位于 `scripts/specstream/`，需要一并复制到服务器。

主要工作负载：

| 工作负载 | 文件/模式 | 作用 |
|---|---|---|
| 同质固定长度 | 合并后的 ShareGPT V3 + `random` | 最容易形成相同 q/history/chunk 的 cohort |
| 原始分布 | 合并后的 ShareGPT V3 + `sharegpt` | 测 serving 延迟，但 cohort 兼容率可能较低 |
| shared-prefix 形状 | `generated-shared-prefix` | 无需文件，构造同步长 prompt；不用于 prefix-cache 收益结论 |

本步骤的数据集证据统一来自 `/common_data/dataset/ShareGPT_V3`。开始 Step 3 前必须已经按 Step 1 第 4 节检查并合并三个 split；C0/C1 必须使用同一个合并文件。

## 3. 单元和内核测试

```bash
PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_draft_delivery_policy.py \
  python/sglang/test/spectre_specstream/test_cohort_scheduler.py \
  python/sglang/test/spectre_specstream/test_online_softmax.py \
  python/sglang/test/spectre_specstream/test_streaming_attention.py \
  python/sglang/test/spectre_specstream/test_staging_reuse.py \
  python/sglang/test/spectre_specstream/test_spectre_fixed_q_e2e.py \
  python/sglang/test/spectre_specstream/test_spectre_tp2.py
```

必须覆盖：

- cohort size 1/2/4/8；
- 不同 History 长度和最后一块 valid length；
- q 1/4/8；
- BF16/FP16；
- GQA ratio 1/2/4；
- 不兼容 key 被拆组；
- deadline 已过的 item 单独下发；
- padding 不进入 online softmax；
- cohort 与逐请求 reference 在误差容限内一致。

## 4. 准备 Drafter 命令

先保存命令，等第 5 节 Target 启动并显示 Uvicorn ready 后再执行。每轮 C0/C1 都按 Target → Drafter 顺序启动。终端 A：

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

## 5. C0/C1 Target 对照

每次只运行一个 Target。C0 和 C1 除 cohort 开关及 profile/output 文件名外必须完全相同。

### 5.1 C0：dynamic SpecStream，不启用 cohort

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
  --specstream-profile-path results/profiles/C0_no_cohort.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

### 5.2 C1：dynamic SpecStream + cohort

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
  --specstream-cohort-enabled \
  --specstream-max-cohort-size 8 \
  --specstream-max-cohort-delay-us 200 \
  --specstream-profile-path results/profiles/C1_cohort.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

功能调试可把 C1 改为 `--specstream-reference-attention`；性能实验必须使用 `--no-specstream-reference-attention`，且关闭 shadow。C0/C1 只有在 profile 中实际 q>1 且没有持续草稿超时后才有比较意义；Accept length 恒为 1 时，cohort 没有获得有效的多-query verification 工作，不能据此判断 cohort 无收益。

## 6. 启动后检查

```bash
curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/v1/models" | python -m json.tool
```

先运行低并发 smoke：

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$SHAREGPT_JSON" \
  --num-prompts 8 \
  --random-input-len 16384 --random-output-len 64 \
  --random-range-ratio 1 \
  --request-rate 1 --max-concurrency 1 \
  --warmup-requests 1 --seed 1 --flush-cache --output-details \
  --tag C1_smoke_16k_c1 \
  --output-file results/bench/C1_smoke_16k_c1.jsonl
```

并发 1 时 `cohort_size=1` 是预期行为，只验证单请求回退正确性。

## 7. 同质长 History cohort 主实验

### 7.1 饱和并发扫描

当前运行 C0 时使用 `PREFIX=C0`，运行 C1 时改为 `PREFIX=C1`：

```bash
export PREFIX=C1

for CONCURRENCY in 2 4 8 16 32; do
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 128 \
    --random-input-len 16384 --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate inf --max-concurrency "$CONCURRENCY" \
    --warmup-requests 4 --seed 1 --flush-cache --output-details \
    --tag "${PREFIX}_random_16k_c${CONCURRENCY}" \
    --output-file "results/bench/${PREFIX}_random_16k_c${CONCURRENCY}.jsonl"
done
```

### 7.2 30K 压力点

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name random \
  --dataset-path "$SHAREGPT_JSON" \
  --num-prompts 64 \
  --random-input-len 30000 --random-output-len 128 \
  --random-range-ratio 1 \
  --request-rate inf --max-concurrency 8 \
  --warmup-requests 2 --seed 1 --flush-cache --output-details \
  --tag C1_random_30k_c8 \
  --output-file results/bench/C1_random_30k_c8.jsonl
```

如果真实上下文上限不允许 30K，按 benchmark JSONL 的 `server_info.max_req_input_len` 下调。

## 8. Poisson serving 负载

```bash
for RATE in 0.5 1 2 4 8; do
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 300 \
    --random-input-len 16384 --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate "$RATE" --max-concurrency 32 \
    --warmup-requests 4 --seed 1 --flush-cache --output-details \
    --tag "C1_random_16k_r${RATE}_c32" \
    --output-file "results/bench/C1_random_16k_r${RATE}_c32.jsonl"
done
```

相同命令在 C0 上运行一遍。低负载下 cohort size 接近 1 很正常；中高负载才有形成 cohort 的机会。

## 9. Shared-prefix 形状工作负载

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 8 --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 16000 \
  --gsp-question-len 128 --gsp-output-len 128 \
  --gsp-range-ratio 0 \
  --request-rate 8 --max-concurrency 32 \
  --warmup-requests 4 --seed 1 --output-details \
  --tag C1_gsp_16k_r8_c32 \
  --output-file results/bench/C1_gsp_16k_r8_c32.jsonl
```

SpecStream v1 会关闭 radix cache。此工作负载用于制造相近长度和同步到达，不把收益解释成 prefix-cache 命中。

## 10. ShareGPT V3 原始分布异质负载

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name sharegpt \
  --dataset-path "$SHAREGPT_JSON" \
  --num-prompts 300 --sharegpt-output-len 256 \
  --sharegpt-context-len 32768 \
  --request-rate 8 --max-concurrency 32 \
  --warmup-requests 4 --seed 1 --flush-cache --output-details \
  --tag C1_sharegpt_r8_c32 \
  --output-file results/bench/C1_sharegpt_r8_c32.jsonl
```

ShareGPT V3 的自然 prompt 长度会降低 cohort 兼容率，这正是服务实验需要报告的现象，不能只保留最容易合并的固定 16K 工作负载。相同命令必须在 C0 上重跑，并把输出标签改为 C0。

## 11. Cohort 参数消融

### 11.1 `max_cohort_size`

分别重启 C1 Target 并设置：

```text
--specstream-max-cohort-size 1
--specstream-max-cohort-size 2
--specstream-max-cohort-size 4
--specstream-max-cohort-size 8
--specstream-max-cohort-size 16
```

`size=1` 是保留 cohort 调度代码但禁止跨请求合并的消融。

### 11.2 chunk size

分别测试：

```text
--specstream-chunk-tokens 512
--specstream-chunk-tokens 1024
--specstream-chunk-tokens 2048
--specstream-chunk-tokens 4096
```

每次改变 chunk 后，C0/C1 都必须使用同一值；同时记录 CPU History block 数、H2D ops、有效带宽、stream attention 和 staging 占用。

### 11.3 delay 参数

可做健全性扫描：

```text
--specstream-max-cohort-delay-us 0
--specstream-max-cohort-delay-us 50
--specstream-max-cohort-delay-us 200
--specstream-max-cohort-delay-us 500
```

但当前实现不会为了未来请求主动等待，因此不能把该扫描包装成完整的 throughput/P99 wait-time trade-off。若数值几乎不变属于当前实现的合理结果。

## 12. Profile 分析

```bash
python scripts/specstream/summarize_specstream_profile.py \
  results/profiles/C0_no_cohort.csv \
  results/profiles/C1_cohort.csv
```

当前 CSV 每行对应一个 verification round，多个 rid 用 `|` 连接；`cohort_size` 是该 round 观察到的最大 cohort size。

C1 在并发长上下文下应观察到：

- `max_history >= 8192`；
- `stream_rows > 0`；
- `max_cohort > 1`；
- 中高负载 `mean_cohort > 1`；
- 相同工作负载下 H2D ops 或 launch/attention 开销相对 C0 降低；
- 所有请求正确完成。

当前 `staging_wait_ms`、copy-compute overlap、kernel launches/token 等列尚未在全部路径写入，可能为 0。要形成论文级 overlap/launch 结论，需要增加 CUDA event/NVTX/NSight 指标，不能把预留的 0 当作实测值。

## 13. Benchmark 分析

```bash
python scripts/specstream/summarize_benchmarks.py \
  results/bench/C0_*.jsonl \
  results/bench/C1_*.jsonl
```

报告：

- completed 和失败请求；
- request/output throughput；
- mean/P99 TTFT、TPOT、E2E；
- 实际并发；
- q/mode 分布；
- mean/max cohort；
- H2D GiB、ops、effective GB/s；
- CPU History 和 staging bytes；
- GPU 峰值显存。

SGLang `bench_serving` 不直接输出 SLO goodput。若论文报告 goodput，必须先定义 SLO，例如 `TTFT <= 2s 且 TPOT <= 50ms`，再基于 `--output-details` 的逐请求结果计算达标请求率，不能直接把 output throughput 改名为 goodput。

## 14. 公平性与重复实验

- C0/C1 使用完全相同的请求顺序、seed、模型、q controller、chunk、tail、GPU 和预热。
- 每个正式点至少运行 3 次；论文结果建议 5 次并报告置信区间。
- C1 不能单独获得更大的 batch、CPU memory 或 GPU memory 预算。
- profile 和 benchmark 文件不能在不同配置之间复用或追加混合。
- ShareGPT V3 原始分布与固定 16K 必须同时报告，避免只展示最有利的同质工作负载。
- 同时报告吞吐和 P99，不能只展示平均吞吐。

## 15. 正确性与故障注入

至少测试：

- 插入不同 dtype/head geometry 的不兼容项，应拆组；
- 不同最后一块 valid length，不允许 padding 进入 softmax；
- 一个超长请求不能阻塞已经可执行的其他请求；
- pinned allocation 失败的降级行为；
- CPU History budget 达到上限；
- Drafter 中断或 stale response；
- TP=2 一个 rank 延迟时，两个 rank 的 cohort plan、请求顺序和 staging slot 仍一致。

正确性失败先检查 valid length、rid-state 映射、q bucket、layer/geometry key；性能失败再检查 host packing、pinned memory、H2D 大小、kernel occupancy 和 batch 内兼容率。

## 16. Step 3 通过条件

1. 单元/内核测试全部通过。
2. C1 在并发 16K/30K 工作负载中实际出现 `cohort_size > 1`。
3. C1 与 C0 在算法误差容限内一致，无 padding 污染和状态串扰。
4. 中高负载下 DMA/attention 调用开销下降，吞吐改善或在相同吞吐下降低延迟。
5. P99 恶化不超过预先设定阈值；建议以 5% 为初始判据。
6. HBM 和 CPU History 始终受配置预算约束。
7. 对固定长度、ShareGPT V3 原始分布和 shared-prefix 形状工作负载均给出结果。
8. 对当前“batch 内 cohort、无主动跨周期等待”的实现边界做明确披露。
