# Step 1：Target 分层 KV、分块流式传输与 online softmax 完整测试流程

## 1. 测试目标

本步骤验证以下数据路径是否真实运行：

```text
CPU Sealed History
  -> 分块 H2D
  -> 有界 GPU staging
  -> 对同一 verification round 的 q 个 query 执行 online-softmax history attention

GPU Active Tail + Speculative Frontier
  -> 原 GPU KV 路径

History 状态与 Tail 状态
  -> online-softmax merge
  -> 完整 full-history attention 输出
```

“exact”指不丢弃 History token、不做 sparse selection、不量化 KV，attention 的覆盖范围和 causal 语义与完整历史一致；不承诺不同浮点归约顺序下逐位相同。

本版本要求：

- SPECTRE Target；
- `page_size=1`；
- MHA/GQA，K/V head dimension 相同；
- `speculative_eagle_topk=1`；
- 不支持 MLA、sliding-window layer 和 context parallel；
- 状态始终满足 `0 <= history_len <= committed_len <= logical_len`。

## 2. 本机目录、模型和数据集

服务器仓库：

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre
```

在每个新终端中设置变量。变量不会自动跨终端继承：

```bash
export TARGET_MODEL=/你的/Target模型目录
export DRAFT_MODEL=/你的/Draft模型目录
export BASE_URL=http://127.0.0.1:30000

export SHAREGPT_V3_ROOT=/common_data/dataset/ShareGPT_V3
export PREPARED_ROOT=/common_data/dataset/specstream_prepared
export SHAREGPT_JSON=$PREPARED_ROOT/sharegpt_v3_merged.json

mkdir -p logs/specstream results/bench results/profiles
```

如果当前用户不能写 `/common_data/dataset`，把 `PREPARED_ROOT` 改为 `~/lifei/specdecode/datasets/specstream_prepared`；后续命令不需要其他修改。

如果服务器此前只替换了 `python/sglang/srt`，正文中的原始 `python -m sglang.bench_serving` 命令仍可直接执行；但要使用文末两个汇总命令，还需要把本次提供的 `scripts/specstream/` 目录同步到服务器仓库。

本步骤的数据使用规则：

| 文件 | benchmark 模式 | 用途 |
|---|---|---|
| `/common_data/dataset/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split*.json` | 先合并，再用 `random` | 构造精确 4K/16K/30K 长度，作为主性能工作负载 |
| 同上 | `sharegpt` | 保留原始 ShareGPT prompt 长度，测试真实服务分布 |
| 不使用文件 | `random-ids` | 最短功能 smoke，不作为主论文性能数据 |

本版测试文档只使用 `/common_data/dataset/ShareGPT_V3` 下的三个 split。`dataset_infos.json` 是数据集元信息，`README.md` 是说明文件，二者都不能作为 `--dataset-path`。Step 1-3 的正式实验统一使用三个 split 合并得到的文件。

## 3. 替换代码后的静态和单元测试

先确认运行的是预期提交和当前工作树：

```bash
git rev-parse HEAD
git status --short
python -c "import sglang; print(sglang.__file__)"
```

确认 Python 实际导入当前仓库，而不是另一个已安装版本。然后执行：

```bash
cd python/sglang/srt/speculative/spectre/cpp_zmq
python setup.py build_ext --inplace --force
cd ~/lifei/specdecode/baseline/sglang
python -m pip install -e ./python --no-deps

python - <<'PY'
from sglang.srt.speculative.spectre import cpp_zmq
from sglang.srt.speculative.spectre.cpp_zmq import spectre_zmq
print(cpp_zmq.__file__)
print(spectre_zmq.__file__)
PY

python -m sglang.launch_server --help | grep -E \
  'spectre-(recv-timeout|initial-recv-timeout|fixed-q-mode|require-draft)'

python -m compileall -q \
  python/sglang/srt/speculative/spectre/specstream \
  python/sglang/srt/speculative/spectre/verifier

PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_draft_delivery_policy.py \
  python/sglang/test/spectre_specstream/test_state_invariants.py \
  python/sglang/test/spectre_specstream/test_cpu_history_store.py \
  python/sglang/test/spectre_specstream/test_online_softmax.py \
  python/sglang/test/spectre_specstream/test_streaming_attention.py \
  python/sglang/test/spectre_specstream/test_staging_reuse.py \
  python/sglang/test/spectre_specstream/test_spectre_fixed_q_e2e.py \
  python/sglang/test/spectre_specstream/test_first_divergence_diagnostics.py
```

如果曾经直接覆盖整个 `srt`，必须重编译 `cpp_zmq`；仅替换 Python 文件不能证明通信二进制与当前协议一致。两个导入路径必须都指向当前工作树，帮助输出必须包含四个新参数；否则服务器仍在导入旧代码。只有上述测试通过后再启动 GPU 服务。

## 4. 准备与验证兼容数据集

### 4.1 新数据集文件角色

```text
/common_data/dataset/ShareGPT_V3/
├── dataset_infos.json
├── README.md
├── ShareGPT_V3_unfiltered_cleaned_split1.json
├── ShareGPT_V3_unfiltered_cleaned_split2.json
└── ShareGPT_V3_unfiltered_cleaned_split3.json
```

三个 `split*.json` 是 benchmark 输入；`dataset_infos.json` 和 `README.md` 仅用于元信息/说明。先把本次提供的 `scripts/specstream/prepare_datasets.py` 同步到服务器，再检查三个 split 的真实 schema：

```bash
python scripts/specstream/prepare_datasets.py inspect \
  /common_data/dataset/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split1.json

python scripts/specstream/prepare_datasets.py inspect \
  /common_data/dataset/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split2.json

python scripts/specstream/prepare_datasets.py inspect \
  /common_data/dataset/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split3.json
```

预期三个文件均显示 `JSON array`、`record_type=turn-list`，且 `first_item_keys` 包含 `value` 或 `content`。任一 split 失败时先停止，不要生成部分合并数据。

### 4.2 合并 ShareGPT V3

```bash
mkdir -p "$PREPARED_ROOT"

python scripts/specstream/prepare_datasets.py merge-sharegpt \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split1.json" \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split2.json" \
  "$SHAREGPT_V3_ROOT/ShareGPT_V3_unfiltered_cleaned_split3.json" \
  --output "$SHAREGPT_JSON"
```

输出文件是标准 ShareGPT JSON 数组。每条可用记录只保留 SGLang benchmark 实际会读取的前两轮，并统一为 `{"from": ..., "value": ...}`；后续轮次即使缺少文本也不会影响合并。少于两轮、前两轮不是对象或前两轮没有非空文本的记录会被跳过并计数，不会中止整个数据集。该文件可同时用于 `dataset-name=random` 和 `dataset-name=sharegpt`。

验证：

```bash
python scripts/specstream/prepare_datasets.py inspect "$SHAREGPT_JSON"
```

合并日志必须记录每个 split 的 `kept`/`skipped` 以及最终 `source_records`/`skipped`。正式实验保存该日志；如果要审计数据质量并在首条坏记录处停止，可在合并命令末尾加入 `--strict`。

后续所有 Step 1-3 命令统一使用 `$SHAREGPT_JSON`，不在不同 baseline 间切换 split 或数据版本。若只想快速验证，也可在 schema 检查通过后暂时把 `SHAREGPT_JSON` 指向 `split1.json`；正式对照必须使用同一个合并文件。

## 5. 准备 Drafter 命令（先不要执行）

本版必须采用 **Target 先启动、Drafter 后启动** 的顺序。原因是 Target 是三个 IPC endpoint 的 bind 端；先让 Target 建立 `/tmp/127_0_0_1_29000*`，再让 Drafter 连接，可以避免 Drafter 先连接旧/尚不存在的 IPC endpoint。先保存下面命令，完成第 6 节 Target 启动后再执行。终端 A 使用 GPU 0：

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

保持 Drafter 运行。客户端生成请求不能发送到 `30001`。`draft-priority` 让当前远程草稿尽快生成完 q 个 token，避免长上下文测试中草稿响应被普通调度延迟。

本步骤先使用 `ordinary` 固定-q 校准模式：Target 每轮等待本轮草稿再验证。它牺牲 Draft/Verify 并行，但最适合确认 q>1 数据路径真实生效。校准通过后，再把 Target 的 `--spectre-fixed-q-mode ordinary` 改为 `parallel` 测最终并行性能。

## 6. 依次启动 B0-B3 Target，然后启动 Drafter

终端 B 使用 GPU 1。每次只启动一个 Target；Target 日志出现 Uvicorn ready 后，执行第 5 节 Drafter 命令。切换 baseline 前按 `Ctrl+C` 停止旧 Target 和 Drafter，确认旧进程退出。正式结果每个 baseline 都按“Target → Drafter”顺序重启二者。

### 6.1 B0：原始 GPU-resident SPECTRE

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode ordinary --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  --disable-overlap-schedule
```

### 6.2 B1：CPU History + 每轮 Full-Restore

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode ordinary --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --specstream-enabled --specstream-full-restore-baseline \
  --specstream-chunk-tokens 2048 --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 1 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-profile-path results/profiles/B1.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

### 6.3 B2：bounded reference streaming

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode ordinary --spectre-require-draft \
  --spectre-draft-timeout-action fallback \
  --spectre-recv-timeout-ms 5000 --spectre-initial-recv-timeout-ms 15000 \
  --spectre-retry-min-count 1 --spectre-retry-fail-ratio 0 \
  --spectre-reject-interval 1 \
  --specstream-enabled --specstream-reference-attention \
  --specstream-chunk-tokens 2048 --specstream-num-buffers 2 \
  --specstream-chunks-per-transfer 1 \
  --specstream-active-tail-tokens 512 \
  --specstream-min-history-tokens 8192 \
  --specstream-cpu-memory-gb 128 \
  --specstream-profile-path results/profiles/B2.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

### 6.4 B3：multi-query tiled fused streaming + grouped double buffer

先不开 shadow 测性能：

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --port 30000 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role target \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --attention-backend fa3 \
  --spectre-fixed-q-mode ordinary --spectre-require-draft \
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
  --specstream-profile-path results/profiles/B3.csv \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000
```

正确性诊断另起一轮 B3，额外加入：

```bash
--specstream-shadow-attention \
--specstream-profile-path results/profiles/B3_shadow.csv
```

Shadow 会重复执行 reference attention，不用于吞吐结果。

## 7. Target 启动后的健康检查

终端 C：

```bash
export BASE_URL=http://127.0.0.1:30000

curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/v1/models" | python -m json.tool
```

单请求：

```bash
curl -sS "$BASE_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "text":"请用一句话解释推测性解码。",
    "sampling_params":{"temperature":0,"max_new_tokens":32,"ignore_eos":false},
    "stream":false
  }' | python -m json.tool
```

只有 Target 和 Drafter 都没有 traceback，才进入批量测试。正式 serving 使用 `--spectre-draft-timeout-action fallback`：q>1 草稿超时会明确记录 missing rid，并将当前 batch 降级到 q=1，但不会杀死 Target。仅在 C=1 链路校准时可临时改为 `error`；高并发性能测试不得使用 `error`。

首次显式 smoke 时，两端日志必须形成以下闭环：Target 出现 `[Target][DraftLink] registered=...; sending ...`，Drafter 出现 `received initial request` 和 `sent initial response`。缺哪一段就只排查对应链路，不要继续运行 benchmark。

### 7.1 accept length 一直为 1 的判定

SGLang 的 `Accept length` 包含 Target 自己产生的 1 个 token。因此 `Accept length: 1.00` 表示平均接受的草稿 token 数为 0，并不表示“成功接受了 1 个草稿 token”。若 Target 日志同时出现 `[Target] Recv timeout`、`No draft available` 或 circuit breaker，则根因是草稿没有按轮次送达，B1/B2/B3 内核尚未真正获得多-query 输入。

校准通过必须同时满足：

- Target 日志没有持续的 `Recv timeout`/`No draft available`；
- benchmark 的 `Accept length` 不再恒定为精确的 `1.00`；
- SpecStream profile 的实际 q 分布中存在 q>1；
- B1/B2/B3 使用同一 Drafter、Target、q、数据和采样参数。

若 C=1 严格校准（`--spectre-draft-timeout-action error`）不报错、profile 也确认实际 q>1，但 Accept length 仍接近 1，则才检查模型匹配问题：Draft/Target 必须使用同一 tokenizer、同一模型家族和兼容 chat template；0.5B→7B 的草稿质量也可能较低。传输问题解决不等于保证很高的接受率。

## 8. 数据集测试命令

以下命令中的 `B0` 必须随当前 Target 改为 `B1`、`B2` 或 `B3`。所有 baseline 的其他参数必须相同。

### 8.1 无下载 4K smoke

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name random-ids --tokenize-prompt \
  --num-prompts 8 \
  --random-input-len 4096 --random-output-len 64 \
  --random-range-ratio 1 \
  --request-rate 1 --max-concurrency 1 \
  --warmup-requests 1 --seed 1 --flush-cache --output-details \
  --tag B0_random_ids_4k_c1 \
  --output-file results/bench/B0_random_ids_4k_c1.jsonl
```

4K 低于 `min_history_tokens=8192`，主要验证普通/回退路径，不证明 streaming 已执行。

### 8.2 ShareGPT 文本固定 4K、16K、30K

```bash
for INPUT_LEN in 4096 16384 30000; do
  python -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
    --dataset-name random \
    --dataset-path "$SHAREGPT_JSON" \
    --num-prompts 30 \
    --random-input-len "$INPUT_LEN" --random-output-len 128 \
    --random-range-ratio 1 \
    --request-rate 1 --max-concurrency 1 \
    --warmup-requests 2 --seed 1 --flush-cache --output-details \
    --tag "B0_random_${INPUT_LEN}_c1" \
    --output-file "results/bench/B0_random_${INPUT_LEN}_c1.jsonl"
done
```

使用 30K 而不是 32768，是为了给输出 token 和 speculative frontier 留空间。如果 `/v1/models` 或 benchmark JSONL 中的 `max_req_input_len` 更小，应进一步降低输入长度。

### 8.3 固定长度并发扫描

```bash
for CONCURRENCY in 1 4 8 16; do
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
    --tag "B0_random_16k_c${CONCURRENCY}" \
    --output-file "results/bench/B0_random_16k_c${CONCURRENCY}.jsonl"
done
```

`request-rate=inf` 用于饱和吞吐；P99 serving 结论必须在有限 request rate 下另测。

### 8.4 原始 ShareGPT 分布

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url "$BASE_URL" \
  --model "$TARGET_MODEL" --tokenizer "$TARGET_MODEL" \
  --dataset-name sharegpt \
  --dataset-path "$SHAREGPT_JSON" \
  --num-prompts 200 --sharegpt-output-len 256 \
  --sharegpt-context-len 32768 \
  --request-rate 4 --max-concurrency 16 \
  --warmup-requests 4 --seed 1 --flush-cache --output-details \
  --tag B0_sharegpt_r4_c16 \
  --output-file results/bench/B0_sharegpt_r4_c16.jsonl
```

原始 ShareGPT 很多 prompt 较短，只用于服务分布，不替代固定 16K/30K。

## 9. B0-B3 的正确执行顺序

每个 baseline 都按以下顺序执行：

1. 启动 Drafter 和当前 Target。
2. `/health`、`/v1/models`、单请求检查。
3. 4K random-ids smoke。
4. 4K/16K/30K 单并发固定长度。
5. 16K 并发 1/4/8/16。
6. 使用合并后的 ShareGPT V3 运行原始长度分布 workload。
7. 保存 Target、Drafter 日志、benchmark JSONL 和 SpecStream CSV。
8. 停止 Target，切换下一个 baseline；正式实验同时重启 Drafter。

每个正式点至少独立运行 3 次；论文结果建议 5 次。输出文件名加入 `_rep1`、`_rep2` 等，不能把多个重复运行追加到同一结果文件后当作一次结果。

## 10. 如何证明 streaming 和 online softmax 已执行

B0 没有 SpecStream CSV。B1-B3 查看：

```bash
python scripts/specstream/summarize_specstream_profile.py \
  results/profiles/B1.csv \
  results/profiles/B2.csv \
  results/profiles/B3.csv
```

B2/B3 的长上下文必须观察到：

- `max_history >= 8192`；
- `stream_rows > 0`；
- `h2d_gib > 0`、`h2d_ops > 0`；
- `stream_attn_s > 0`；
- q 分布与固定 `q=5` 一致，安全回退 round 除外；
- 无 NaN/Inf、CUDA assert 和非法 page-table slot。

当前 CSV 中 `q` 和 `mode` 是实际执行值。`chunk_tokens`、`num_chunks`、`controller_selected_q/mode`、`gpu_kv_bytes`、overlap 和 staging-wait 列在当前实现中属于预留字段，可能保持 0/空；不要把这些预留字段为 0 误判为 streaming 未执行。

B3 shadow 轮还要检查同目录的 `B3_shadow.shadow.csv`，记录 first divergence、layer、`shadow_max_abs`、logit margin 和 token mismatch。性能轮必须关闭 shadow。

## 11. benchmark JSONL 的通过标准

汇总：

```bash
python scripts/specstream/summarize_benchmarks.py results/bench/*.jsonl
```

每个点检查：

- `completed` 等于 `num_prompts`；
- 失败请求为 0；
- `server_info` 中模型、SPECTRE、SpecStream 参数与 baseline 一致；
- 报告 `output_throughput`、TTFT、TPOT、E2E 的 mean/P99；
- B1-B3 同时报告 H2D、CPU History 和 stream attention；
- B3 对 B2 的结论同时包含吞吐/TPOT与内核开销；
- B3 对 B1 的结论同时包含时间性能与显存可扩展性。

## 12. 显存测试

在另一个终端采样：

```bash
nvidia-smi --query-compute-apps=timestamp,pid,used_memory \
  --format=csv -lms 200 > results/profiles/B0_gpu_memory.csv
```

为 B1-B3 分别改文件名。B2/B3 的 staging 理论上随 `chunk_tokens * num_buffers` 有界；进程总显存还包含模型权重、GPU tail、workspace 和框架缓存，所以不能只凭一次 `nvidia-smi` 数字直接等同于 staging 大小。

## 13. 常见故障

- 请求误发到 `30001`：改为 Target 的 `30000`。
- `$TARGET_MODEL` 为空：在 benchmark 终端重新 export。
- 4K 没有 H2D：正常；改跑 16K。
- 旧 `sharegpt.json` 顶层是 dict：不要使用；统一合并 `/common_data/dataset/ShareGPT_V3` 的三个 split。
- `dataset_infos.json`/`README.md` 传给 benchmark：这是元信息/说明文件，改用 `$SHAREGPT_JSON`。
- 三个 split schema 不一致：停止合并并记录 `inspect` 输出，不能只合并通过的部分后用于正式对照。
- 约 32K 后 Drafter index out of bounds：确认 health request finish 修复和 `_check_and_pause_draft_req` 修复均已部署；用 `CUDA_LAUNCH_BLOCKING=1` 定位首次越界。
- Target 已记录 `sending initial request`、Drafter 已记录 `received initial request`，但没有匹配的 `sent initial response`，随后 curl 显示 `Empty reply from server`：检查 `spectre_worker.py` 是否包含 `finish_normal_decode_bookkeeping`。旧路径会在 Drafter 为同一 q-token round 每生成一个 token 就错误执行 `req.spec_cnt += 1`；例如 Target 等待 `spec_cnt=0`，q=5 的 Drafter 却回传 `spec_cnt=4`，Target 会将它判为非当前轮响应并在 15 秒后 fail-fast。部署最新版 `spectre_worker.py` 与 `draft_delivery.py` 后，Drafter 必须在 q 个 autoregressive step 内保持 Target 分配的 `spec_cnt` 不变。
- 30K 请求被拒绝：检查 `server_info.max_req_input_len`，继续给输出和 q 留空间。
- B2/B3 CSV 没有 H2D：确认输入超过 seal 阈值、Target 加了 `--specstream-enabled`，且没有所有 round 都回退。
- B0 与 B1-B3 公平性：SpecStream 会关闭 CUDA Graph/radix cache。论文中同时报告“原生最佳 B0”和“同 Graph/cache 约束 B0”，不能隐藏框架配置差异。

## 14. Step 1 通过条件

Step 1 只有同时满足以下条件才通过：

1. 全部单元测试通过。
2. B0-B3 的 4K、16K 请求全部完成且服务不崩溃。
3. B2/B3 的 16K/30K profile 确认 CPU History、分块 H2D 和 stream attention 已执行。
4. B3 shadow 在预设 BF16/FP16 容差内；如 token 分叉，完成 first-divergence 记录。
5. HBM 随 History 增长不再呈 Full-Restore 的完整 history scratch 行为。
6. 合并文件校验结果、CSV、日志和启动命令均归档，可复现实验。
