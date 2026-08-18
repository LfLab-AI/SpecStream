# SpecStream-SPECTRE 服务启动后端到端数据集测试手册

> B3 新版 multi-query tiled kernel、旧结果复盘、固定长度参数纠正和硬件调优流程见 [`B3_MULTI_QUERY_TILED_OPTIMIZATION_CN.md`](B3_MULTI_QUERY_TILED_OPTIMIZATION_CN.md)。

本文从 Drafter 和 Target 已经启动的状态开始，说明请求应发到哪里、如何先做功能检查、如何使用 `random-ids`、ShareGPT V3、受控长文本和 shared-prefix 工作负载测试，以及如何对 B0-B3 做公平对比。

> 最重要的规则：用户请求只发到 Target 的 `30000` 端口。`30001` 是 Drafter 服务端口，不是本测试的生成入口。Drafter 与 Target 通过 `29000` 端口上的 SPECTRE ZMQ 通道协作。

## 1. 三个终端分别做什么

- 终端 A：运行 Drafter，保持不退出。
- 终端 B：运行一个 Target。B0、B1、B2、B3 不能同时运行；切换 baseline 时先停止旧 Target。
- 终端 C：运行 `curl` 和 `python -m sglang.bench_serving`，所有测试请求发往 `http://127.0.0.1:30000`。

在终端 C 重新设置环境变量。Shell 环境变量不会自动从终端 A/B 传到终端 C：

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

export TARGET_MODEL=/你的/Target模型目录
export BASE_URL=http://127.0.0.1:30000
export SHAREGPT_V3_ROOT=/common_data/dataset/ShareGPT_V3
export PREPARED_ROOT=/common_data/dataset/specstream_prepared
export SHAREGPT_JSON=$PREPARED_ROOT/sharegpt_v3_merged.json
mkdir -p results/bench results/profiles
```

若 `/common_data/dataset` 不可写，将 `PREPARED_ROOT` 改为 `~/lifei/specdecode/datasets/specstream_prepared`。

Target 与 Drafter 必须使用兼容的 tokenizer/vocabulary。测试客户端的 `--model` 和 `--tokenizer` 均指向 Target 模型。

## 2. 先确认服务真的可用

### 2.1 健康检查

在终端 C 执行：

```bash
curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/v1/models" | python -m json.tool
```

判定标准：两个命令均正常退出；第二个命令能显示 Target 模型。`/health` 返回体可能为空，只要 HTTP 状态为 200 即可。

如果失败，先不要运行数据集。检查：Target 是否仍在运行、端口是否为 `30000`、是否错误地访问了 `30001`，以及 Target 日志中是否已经出现 `The server is fired up and ready to roll!`。

### 2.2 单请求功能检查

```bash
curl -sS "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "请用一句话解释推测性解码。",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 32,
      "ignore_eos": false
    },
    "stream": false
  }' | python -m json.tool
```

应返回生成文本，Target 和 Drafter 均不应退出。不要用这一步测性能；它只证明 HTTP、Target、ZMQ 和 Drafter 的最短路径已经连通。

## 3. 第一个可直接运行的数据集测试：不下载任何数据

`random-ids` 会直接生成指定长度的 token ID，适合验证长度、并发、长上下文和基本稳定性，不需要 Hugging Face 网络。

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" \
  --tokenizer "$TARGET_MODEL" \
  --dataset-name random-ids \
  --tokenize-prompt \
  --num-prompts 8 \
  --random-input-len 4096 \
  --random-output-len 64 \
  --random-range-ratio 1 \
  --request-rate 1 \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --seed 1 \
  --flush-cache \
  --output-details \
  --tag B0_smoke_4k \
  --output-file results/bench/B0_smoke_4k.jsonl
```

成功条件：

- 最终显示 `Successful requests: 8` 或等价的全部完成统计；
- `Failed requests` 为 0；
- Target 和 Drafter 均未出现 traceback、CUDA assert、NaN/Inf；
- 产生 `results/bench/B0_smoke_4k.jsonl`。

`--random-range-ratio 1` 才表示每条请求具有固定的输入/输出长度。SGLang 的 `compute_random_lens()` 从 `[full_len * ratio, full_len]` 采样，因此 `0` 会产生从 1 到目标长度的宽分布。本次旧日志中 4K/16K/30K 的平均实际输入约为 2.17K/7.77K/15.20K，正是因为误用了 `0`。

注意：随机 token ID 在极少数模型上可能诱发 NaN。如果只有 `random-ids` 出现 NaN，而自然文本工作负载正常，先改用第 4 节的 `random` 文本模式复核，不要立即把它判定为 SpecStream 错误。

## 4. 主性能测试：ShareGPT 文本构造的受控长上下文

论文主性能数据建议使用 `random`，而不是 `random-ids`。该模式从 ShareGPT 文本取样，并重复或截断到指定 token 长度，因此既能控制长度，又比任意 token ID 更接近自然文本。

### 4.1 准备用户本地的 ShareGPT V3

本版只使用 `/common_data/dataset/ShareGPT_V3` 下三个 `split*.json`；`dataset_infos.json` 和 `README.md` 不作为 benchmark 输入。先检查并合并三个 split：

```bash
python scripts/specstream/prepare_datasets.py inspect \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split1.json"
python scripts/specstream/prepare_datasets.py inspect \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split2.json"
python scripts/specstream/prepare_datasets.py inspect \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split3.json"

mkdir -p "$PREPARED_ROOT"
python scripts/specstream/prepare_datasets.py merge-sharegpt \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split1.json" \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split2.json" \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split3.json" \
  --output "$SHAREGPT_JSON"

python scripts/specstream/prepare_datasets.py inspect "$SHAREGPT_JSON"
```

合并脚本只保留 SGLang 实际使用的前两轮 prompt/completion，并要求这两轮具有非空 `value` 或 `content`。ShareGPT V3 的后续多轮中若含无文本元数据，不会阻塞合并；少于两轮或前两轮无有效文本的记录会被跳过，并在日志中报告每个 split 的 `kept`/`skipped` 计数。需要严格审计时可额外添加 `--strict`。

### 4.2 推荐的长度扫描

当前日志显示 Qwen2.5 模型的最大请求输入约为 32762 token，因此不要直接测试 64K。输入、输出和 speculative frontier 的总长度必须小于模型上下文上限。32K 模型建议使用 30K 而不是顶到 32768。

依次执行 4K、16K、30K：

```bash
for INPUT_LEN in 4096 16384 30000; do
  python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" \
    --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 30 \
    --random-input-len "$INPUT_LEN" \
    --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate 1 \
    --max-concurrency 1 \
    --warmup-requests 2 \
    --seed 1 \
    --flush-cache \
    --output-details \
    --tag "B0_random_${INPUT_LEN}_c1" \
    --output-file "results/bench/B0_random_${INPUT_LEN}_c1.jsonl"
done
```

对于 `--specstream-min-history-tokens 8192`，4K 主要验证短上下文回退路径；16K 和 30K 才是验证 CPU sealed History、H2D 和 bounded streaming 的关键点。

## 5. 直接使用原始 ShareGPT 对话分布

`sharegpt` 模式使用每个样本前两轮对话的真实 prompt/completion 长度。它适合服务分布测试，但很多样本较短，因此不能替代受控的 16K/30K 测试。

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" \
  --tokenizer "$TARGET_MODEL" \
  --dataset-name sharegpt \
  --dataset-path "$SHAREGPT_JSON" \
  --num-prompts 200 \
  --sharegpt-output-len 256 \
  --sharegpt-context-len 32768 \
  --request-rate 4 \
  --max-concurrency 16 \
  --warmup-requests 4 \
  --seed 1 \
  --flush-cache \
  --output-details \
  --tag B0_sharegpt_r4_c16 \
  --output-file results/bench/B0_sharegpt_r4_c16.jsonl
```

默认 benchmark 会设置 `temperature=0` 并忽略 EOS，以保证每个请求生成完整的目标长度，适合吞吐对照。如果要评价真实完成行为，在命令中加入 `--disable-ignore-eos`；同一对照组必须统一是否加入该参数。

## 6. Continuous serving 和 Chunk-Cohort 压力测试

### 6.1 开环到达率扫描

有限的 `--request-rate` 使用 Poisson 到达。先从低负载逐步提高，直到 P99 急剧增长或失败率上升：

```bash
for RATE in 0.5 1 2 4 8; do
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 200 \
    --random-input-len 16384 --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate "$RATE" --max-concurrency 32 \
    --warmup-requests 4 --seed 1 --flush-cache \
    --output-details \
    --tag "B0_16k_r${RATE}_c32" \
    --output-file "results/bench/B0_16k_r${RATE}_c32.jsonl"
done
```

`--request-rate inf` 表示所有请求在时间零发出，适合测饱和吞吐，不代表在线流量。必须同时报告有限速率的 P99。

### 6.2 shared-prefix 形状工作负载

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 8 \
  --gsp-prompts-per-group 8 \
  --gsp-system-prompt-len 16000 \
  --gsp-question-len 128 \
  --gsp-output-len 128 \
  --gsp-range-ratio 0 \
  --request-rate 8 \
  --max-concurrency 32 \
  --warmup-requests 4 \
  --seed 1 \
  --output-details \
  --tag B7_gsp_16k_r8_c32 \
  --output-file results/bench/B7_gsp_16k_r8_c32.jsonl
```

SpecStream v1 会关闭 radix cache，因此这里使用 shared-prefix 数据主要是构造可形成 cohort 的相近历史长度与并发到达，不把结果解释成 prefix-cache 收益。

## 7. B0、B1、B2、B3 的公平比较流程

每个 baseline 都执行完全相同的 benchmark 命令，只改变 Target 启动参数和输出标签。最稳妥的做法是每个 baseline 都重启 Drafter 和 Target，等待健康检查通过，预热，再正式测试。至少必须重启 Target，且任何时刻只能有一个 Target 占用 `30000/29000`。

| Baseline | Target 数据路径 | 必须参数 |
|---|---|---|
| B0 | 原始 GPU-resident SPECTRE | 不带任何 `--specstream-*` 参数 |
| B1 | CPU History，每轮 Full-Restore | `--specstream-enabled --specstream-full-restore-baseline` |
| B2 | bounded reference streaming | `--specstream-enabled --specstream-reference-attention` |
| B3 | multi-query tiled fused streaming + grouped double buffer | `--specstream-enabled --no-specstream-reference-attention --specstream-num-buffers 2 --specstream-chunks-per-transfer 4` |

B1-B3 的公共参数建议为：

```bash
--specstream-chunk-tokens 2048 \
--specstream-active-tail-tokens 512 \
--specstream-min-history-tokens 8192 \
--specstream-cpu-memory-gb 128 \
--disable-radix-cache \
--disable-cuda-graph \
--disable-overlap-schedule
```

每个 Target 使用独立 profile 文件，例如：

```bash
--specstream-profile-path results/profiles/B3_16k_c1.csv
```

推荐最小对照矩阵：

| 阶段 | 数据 | 输入/输出 | prompts | 到达率/并发 | 目的 |
|---|---|---:|---:|---:|---|
| 功能 smoke | random-ids | 4K/64 | 8 | 1/1 | HTTP、ZMQ、SPECTRE 不崩溃 |
| 短上下文 | random text | 4K/128 | 30 | 1/1 | 验证未达到 seal 阈值的回退路径 |
| 长上下文单请求 | random text | 16K、30K/128 | 各 30 | 1/1 | B1-B3 数据路径、H2D、TPOT |
| 并发吞吐 | random text | 16K/128 | 200 | inf/1、4、8、16 | 吞吐和显存容量 |
| 在线负载 | random text | 16K/128 | 200 | 0.5-8/32 | goodput 和 P99 |
| 真实分布 | ShareGPT | 自然/256 | 200 | 4/16 | serving 分布 |

固定以下条件：commit、模型、GPU、CPU NUMA、端口、chunk、tail、seed、prompt 顺序、输出长度、request rate、max concurrency 和预热次数。每个正式点至少独立重复 3 次；论文结果建议 5 次并报告均值与置信区间。

## 8. 如何确认“真的走了分块流式 KV + online softmax”

只看到请求成功不足以证明流式路径被执行。B2/B3 的 `results/profiles/*.csv` 中至少要看到：

- `history_len >= specstream_min_history_tokens`；
- `num_chunks > 0`；
- `h2d_bytes > 0`、`h2d_ops > 0`；
- `stream_attn_ms > 0`；
- B3 使用两个 bounded staging buffer，显存不会随 sealed History 线性增长；
- `fallback`/`fallback_reason` 没有把所有请求都旁路回普通路径。

开启 fused-vs-reference 影子诊断时，给 B3 Target 加：

```bash
--specstream-shadow-attention \
--specstream-profile-path results/profiles/B3_shadow_16k.csv
```

检查 `*.shadow.csv` 的 `shadow_max_abs`、relative error、logit margin 和 token mismatch。这里的 exact 是完整历史 attention 语义，不等同于 bitwise identical。

## 9. benchmark JSONL 结果怎么看

重点字段：

- `completed`：完成请求数，应等于计划请求数；
- `request_throughput`：请求/秒；
- `output_throughput`：输出 token/秒；
- `total_throughput`：输入加输出 token/秒；
- `mean_ttft_ms`、`p99_ttft_ms`：首 token 延迟；
- `mean_tpot_ms`、`p99_tpot_ms`：每输出 token 时间；
- `mean_e2e_latency_ms`、`p99_e2e_latency_ms`：端到端延迟；
- `accept_length`：SPECTRE 接受长度；某些版本可能为 `null`，此时从服务日志/SpecStream CSV 统计；
- `server_info`：服务实际启动参数，可用于核对 baseline 是否跑错。

汇总多个 JSONL：

```bash
python scripts/specstream/summarize_benchmarks.py results/bench/*.jsonl
```

该脚本只汇总，不改写原结果。

## 10. 使用仓库自带的一键 benchmark 脚本

单个受控测试：

```bash
export TARGET_MODEL=/你的/Target模型目录
export BASE_URL=http://127.0.0.1:30000
export DATASET_PATH=/common_data/dataset/specstream_prepared/sharegpt_v3_merged.json

CASE_TAG=B0_16k_c1 \
DATASET_NAME=random \
INPUT_LEN=16384 OUTPUT_LEN=128 \
NUM_PROMPTS=30 REQUEST_RATE=1 MAX_CONCURRENCY=1 \
bash scripts/specstream/run_benchmark_case.sh
```

长度扫描：

```bash
BASELINE=B0 \
DATASET_NAME=random \
DATASET_PATH=/common_data/dataset/specstream_prepared/sharegpt_v3_merged.json \
CONTEXT_LENS="4096 16384 30000" \
OUTPUT_LEN=128 NUM_PROMPTS=30 \
REQUEST_RATE=1 MAX_CONCURRENCY=1 \
bash scripts/specstream/run_context_sweep.sh
```

切换到 B1/B2/B3 后只改 `BASELINE`，其余环境变量保持不变。

## 11. 常见错误定位

- benchmark 连接 `30001`：改成 `30000`；生成请求必须发给 Target。
- `$TARGET_MODEL` 为空：在 benchmark 所在终端重新 `export TARGET_MODEL=...`。
- `Connection refused`：Target 未完成启动或已经因 Drafter/ZMQ/CUDA 错误退出。
- 生成一直跑到约 32K 后越界：检查 Drafter 的 SPECTRE 请求是否在达到 `draft_tokens_target` 后暂停；确认包含 health-request finish 修复和 `_check_and_pause_draft_req` 修复。
- 4K 没有 `h2d_bytes`：这通常是预期行为，因为 4K 未达到 8192 的 seal 阈值；改跑 16K。
- `random-ids` NaN：改用 `--dataset-name random` 加 ShareGPT 文本复核。
- 30K 被拒绝：从 benchmark JSONL 的 `server_info.max_req_input_len` 查真实上限，并给输出和 speculative frontier 留余量。
- 所有 B0-B3 数字完全一样：核对 JSONL 的 `server_info.specstream_enabled`、`specstream_full_restore_baseline` 和 `specstream_reference_attention`，以及 Target 是否真的重启。
- B2/B3 没有 profile：确认 SpecStream 参数只加在 Target，且 `--specstream-profile-path` 对 Target 进程可写。
- 对照不公平：B0 开启 CUDA Graph/overlap，而 B1-B3 被自动禁用时，要同时报告“原生最佳 B0”和“同调度/同 Graph 约束 B0”两个版本，避免把框架约束误算成算法收益或损失。

## 12. 一轮测试的完整检查清单

1. 记录 git commit、模型、GPU 和启动命令。
2. 启动 Drafter，等待 ready。
3. 启动一个 Target，等待 ready。
4. 运行 `/health`、`/v1/models` 和单请求检查。
5. 运行 4K smoke。
6. 运行 16K/30K 单并发受控测试。
7. 运行并发和 request-rate 扫描。
8. 检查 `completed`、失败数、TTFT/TPOT/E2E 和吞吐。
9. B1-B3 同时检查 profile CSV 是否实际进入预期数据路径。
10. 保存 Drafter 日志、Target 日志、benchmark JSONL 和 SpecStream CSV，文件名必须包含 baseline、ctx、q、rate、concurrency、seed 和重复编号。
11. 停止 Target；切换 baseline 后重新执行相同流程。
