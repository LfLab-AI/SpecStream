# SpecStream Qwen3-32B / Qwen3-0.6B 公开数据集端到端实验手册（单次正式版）

> 仓库：`/root/lifei/SpecStream`  
> Target：`/root/autodl-tmp/model/models/Qwen--Qwen3-32B/snapshots/master`，TP=2  
> Draft：`/root/autodl-tmp/model/Qwen3-0.6B`  
> 当前机器：2 × A800 80 GiB；外置 Draft 与 Target TP rank 0 同卡  
> 模型标签：`qwen3_0p6b_32b`；结果不得写入旧 8B 目录  
> 解码：Qwen3 non-thinking、greedy、`temperature=0`、`top_p=1`

## 0. 2026-09-07 P0 / P1 / P2 版本说明

本手册适用于当前 Qwen3-32B / Qwen3-0.6B、Target TP=2 服务器。这里的 P0/P1/P2 是本轮工程修改优先级，与论文中的创新点一/二/三不是同一编号。

| 优先级 | 已交付的代码行为 | 验证重点 |
|---|---|---|
| P0 | TP 基线按 batch、q、上下文、CPU miss/GPU hit、attention 实现和 TP 分桶；只用已排除 Draft 重叠的预热样本；干扰时保留 ordinary 多 token 验证；有界恢复 | 不再因 batch 变大直接长期锁 q=1；检查 `tp_baseline_ready`、`tp_shape_key`、fallback 原因 |
| P0 | 从 Draft 接收前到提交后的真实 round wall time；native q=1 也记录；CPU enqueue 与 GPU event 分开；经验成本及受控 parallel 探测 | `round_timing_source`、`target_forward_ms`、`target_enqueue_ms`；不把各阶段重叠时间相加 |
| P1 | History split-KV、稳定归并、批量 GPU History/Tail/finalize；连续 CPU slab；不可变 metadata 缓存 | CUDA 数值、H2D 源字节/实际字节、DMA 数和 host 等待 |
| P1 | 首批 H2D 在 GPU History 计算前排队；当前/下一层任务复用；共享有界 ring 的槽位保护和取消回收 | 真正执行多层、多请求、请求重排和 2/4 buffers 的测试 |
| P2 | Target Python forward 期间后台推进 ACK/grant；末轮/idle/reset 仍收割迟到 ACK | `grant_pump_iterations` 仅证明泵运行；成功 SLACK_FILL ACK 才证明执行了租约 |
| P2 | 可选 1–8 token catchup 租约；SLACK_FILL 仍最多 1 token；可选 GPU History 自动预算；矩阵计划与完成集合严格核对 | 多 token ACK 不超预算；自动预算留 prefill、Tail/Frontier 和分配余量 |

本手册比较配置固定为 GPU History=8192、buffers=2、catchup quantum=1；split-KV 与后台泵启用。自动预算和更大 catchup 是单独的可选实验，不能与固定缓存对照混为同一配置。增加 buffers 只增加有界预取深度，不承诺 PCIe 带宽线性增加。

交付状态：此前候选版本在 A800 上有 `270 passed, 2 skipped` 的回归记录，包含实际 CUDA 数值与多层预取；这不等于最终版本已完成端到端吞吐验收。后续 ACK 收尾、运行器和文档修订由操作者按本手册重新验证。原版并发 8 曾完成全部客户端请求，但末轮缺失一个 ACK，因此未通过机制完整性 Gate；不得补造完成 marker 或当作已通过的正式对照。

### 0.1 开关及作用范围

| 环境变量 | 默认/本手册固定值 | 作用 |
|---|---|---|
| `SPECSTREAM_SPLIT_KV` | `auto` | `off` 或 `1` 关闭 split；也可指定 1–32；只改变 History 分片，不关闭全部批量 attention 优化 |
| `SPECSTREAM_BACKGROUND_GRANT_PUMP` | `1` | `0` 关闭 forward 期间后台泵；idle ACK 收尾仍保留 |
| `SPECSTREAM_GPU_HISTORY_CACHE_TOKENS` | 本手册 `8192` | `0` 为无 GPU History 副本；`-1` 为压力感知自动预算；单位是全局 KV token 槽，不是每请求预算 |
| `SPECSTREAM_NUM_BUFFERS` | `2` | 公共运行器仅对 `C/SPECSTREAM_1GPU` 生效，可试 `4`；K1–K5 保持消融定义中的固定 buffers |
| `SPECSTREAM_GRANT_TOKEN_QUANTUM` | `1` | 公共运行器仅对 `C/SPECSTREAM_1GPU` 生效，范围 1–8；实测步耗时/剩余期限决定实际 catchup 长度，SLACK_FILL 不随之变长 |
| `SPECSTREAM_REQUIRE_SLACK_FILL` | smoke `0` | `1` 要求至少一个成功的一 token SLACK_FILL ACK；用于重叠机制验收 |

Qwen3-32B BF16 TP2 每个全局 KV token 在每个 rank 约占 128 KiB。自动缓存复用已有 Target KV pool 的 CPU-backed 副本，不额外扩大 pool；不能把 `-1` 理解为无限缓存。缓存收缩只逐出已有 CPU 副本，不回收尚未完成 D2H 的 Tail/Frontier。

### 0.2 换卡或重启后的环境

每次租用新 GPU 都重新执行本手册的 preflight，使用本次生成的 `runtime_env.sh`。不要 source 历史 results 下的旧环境文件：GPU 索引相同不代表 UUID 相同。历史 `SPECSTREAM_SMCTRL_VALIDATED=1` 也不能代替当前硬件/驱动的预检。

测试前明确使用正式代码目录 `PYTHONPATH=/root/lifei/SpecStream/python`，不要继续使用 `results/p012_20260907/candidate_work/python`。不在交互式 SSH shell 全局设置 `set -euo pipefail`；每个命令块如出现 ERROR/非零状态，停止后续步骤并保留日志。

### 0.3 LongBench 后台线程设备绑定修复（2026-09-07）

本次公开矩阵 `qwen3_0p6b_32b_public_e2e_once_20260907_144615` 的 LongBench-v2 正式用例（并发 4、131 请求、输出 256）在 14:51:37 报错：`Expected a torch.device with a specified index or an integer, but got:cuda`，随后为 `SpecStream background grant pump failed`。直接原因是后台泵将无编号的 `torch.device("cuda")` 传给 `torch.cuda.set_device()`，导致 Target 退出；本次故障并非数据集格式错误。

修复后在 Target 调度器主线程解析设备编号，再由后台线程绑定该编号。显式 `cuda:0` / `cuda:1` 保持原编号，无编号 `cuda` 使用父线程当前设备；不硬编码物理 GPU 0，不关闭后台泵或多流水线重叠。此前回归未覆盖这一实际 scheduler 启动路径，本次补充了上述三种设备用例。

按操作者要求，本次仅交付修复、回归用例和文档，未执行回归测试或重启模型。操作者可先在 `spectre` 环境、仓库根目录运行：

```bash
PYTHONPATH=/root/lifei/SpecStream/python python -m pytest -q \
  python/sglang/test/spectre_specstream/test_background_grant_pump.py
```

然后重新启动 Target/Draft，保持 `SPECSTREAM_BACKGROUND_GRANT_PUMP=1`，使用当前 preflight 环境和新的 `RESULT_ROOT` 重跑失败用例；如需正式矩阵结果，按本手册矩阵入口重新运行。旧失败目录保留，不补写完成 marker。32-token smoke 成功不能替代 256-token 正式负载验收；必须确认请求全部成功、无上述异常，并由运行器正常生成 `case_complete.marker`。修复此退出异常不代表吞吐提升已通过验收。

### 0.4 LongBench 中途 TP 通信死锁修复（2026-09-07）

运行 `qwen3_0p6b_32b_public_e2e_once_20260907_150221` 的 LongBench 正式用例在异常前进度约为 `38/131`。15:32:51–52 的 watchdog 堆栈显示：TP0 在 `_decide_speculative_num_draft_tokens` 的 `all_gather_object` 等待，TP1 则在同一函数的决策 `broadcast_pyobj` 等待，最终 300 秒 watchdog 终止服务。客户端的连接拒绝是服务退出后的连带错误。

根因是两个 rank 分别用本地采样状态判断是否同步 TP profile；异步完成的本地状态不能保证一致，使两端执行了不同的集体通信。修复后仅 TP0 计算采样开关，先广播给所有 rank，然后统一执行可选采样 all-gather，最后广播动态 q 决策。每轮新增一次小型控制广播，保留原有采样间隔、动态 q、TP 干扰监测、后台泵及跨层预取。不要通过增大 watchdog 超时掩盖该死锁。

新增回归直接调用实际 scheduler 方法，使用两个 CPU/Gloo 进程，先模拟一致的预热状态，再模拟两端相反的采样判断，验证采样轮次、决策和通信顺序一致。测试设置 15 秒集体通信超时，不加载模型。操作者在 `spectre` 环境、仓库根目录执行：

```bash
PYTHONPATH=/root/lifei/SpecStream/python python -m pytest -q \
  python/sglang/test/spectre_specstream/test_tp_profile_collective_order.py \
  python/sglang/test/spectre_specstream/test_background_grant_pump.py
```

本次仅完成代码语法解析与服务器文件哈希核对，未执行上述回归或模型测试。请重新启动服务，在新的 `RESULT_ROOT` 下重跑同一完整正式用例，按正常 Gate 验收所有请求和完成 marker。此轮 15:15:05 还记录过一次 `DraftFallback`（q=8、4/4 缺失），这与最终 TP 通信死锁是不同观察；本补丁不宣称已消除该回退。即使重测能跑完，若仍有 DraftFallback，也应保留日志并判定机制验收失败，不能只看客户端成功数或吞吐率。

## 1. 实验口径与硬件边界

正式性能矩阵仍比较四种方法：

| 运行器 ID | 论文含义 | Target | Draft | 独占 GPU 数 |
|---|---|---|---|---:|
| `AR` | 原生自回归 | TP2 | 无 | 2 |
| `SGLANG_SD` | 原生 in-process standalone SD | TP2 | Target 进程内 | 2 |
| `SGLANG_SD_KV_OFFLOAD` | 固定 q=4、ordinary、独立 chunk streaming | TP2 | 外置，位于 rank 0 GPU | 2 |
| `SPECSTREAM_1GPU` | 完整 SpecStream；此名称是兼容旧脚本的 ID | TP2 | 外置，位于 rank 0 GPU | 2 |

`SPECSTREAM_1GPU` 在本模型组合中不表示“整个 Target 只用一张卡”，只表示“Draft 不增加独占 GPU”；论文表格应写成 `SpecStream, Target TP2 + colocated Draft`。四种方法都占用相同的两张物理 GPU。

性能 workload 为 GSM8K 全集（并发 8）、LongBench-v2 全部合格样本（并发 4）和 MRCR 16K–32K 全部合格样本（并发 4），输出均为 256，`request-rate=inf`。准确性实验使用同一冻结样本：GSM8K 数值 EM、LongBench-v2 官方选项提取、MRCR token F1/marker recall。

## 2. 一次性环境

在新的交互式终端中执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/miniconda3/envs/spectre
cd /root/lifei/SpecStream

export REPO=$PWD
export SPECSTREAM_PYTHON=/root/miniconda3/envs/spectre/bin/python
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$(dirname "$SPECSTREAM_PYTHON"):$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=1; fi

export MODEL_TAG=qwen3_0p6b_32b
export TARGET_MODEL=/root/autodl-tmp/model/models/Qwen--Qwen3-32B/snapshots/master
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen3-0.6B

# UUID 顺序就是 Target TP rank 顺序；rank 0 与 Draft 同卡。
export TARGET_GPUS=0,1
export TARGET_TP_SIZE=2
export TARGET_GPU=0
export COLOCATED_GPU=0
export COLOCATED_TP_RANK=0
export DRAFT_GPU=0

export TARGET_PORT=30000 DRAFT_PORT=30001 ZMQ_PORT=5557
export SERVER_CONTEXT_LEN=40960 FINAL_DRAFT_TPCS=34

export QWEN3_DATA_ROOT=$REPO/specstream_prepared/qwen3_offline
export GSM8K_TEST_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/test-00000-of-00001.parquet
export GSM8K_TRAIN_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/train-00000-of-00001.parquet
export GSM8K_TEST=$QWEN3_DATA_ROOT/gsm8k_main_test.jsonl
export GSM8K_FEWSHOT=$QWEN3_DATA_ROOT/gsm8k_main_train_fewshot5.jsonl
export GSM8K_QWEN3=$QWEN3_DATA_ROOT/gsm8k_qwen3_nothink_sharegpt.json
export LONGBENCH_QWEN3=$QWEN3_DATA_ROOT/longbench_v2_qwen3_8b_8k32k_sharegpt.json
export MRCR_QWEN3=$QWEN3_DATA_ROOT/mrcr_qwen3_16k32k_sharegpt.json

# 32B/0.6B 公共矩阵固定容量；所有四种方法不得在 cell 间改动。
export SPECSTREAM_TARGET_MEM_FRACTION=0.62
export SPECSTREAM_DRAFT_MEM_FRACTION=0.80
export SPECSTREAM_TARGET_MAX_TOTAL_TOKENS=131072
export SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS=196608
export SPECSTREAM_TARGET_MIN_KV_TOKENS=131072
export SPECSTREAM_DRAFT_MIN_KV_TOKENS=196608
export SPECSTREAM_PREFILL_MAX_REQUESTS=1
export SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192
export SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0
export SPECSTREAM_REQUIRE_SLACK_FILL=0
export SPECSTREAM_SPLIT_KV=auto SPECSTREAM_BACKGROUND_GRANT_PUMP=1
export SPECSTREAM_NUM_BUFFERS=2 SPECSTREAM_GRANT_TOKEN_QUANTUM=1
```

不要全局设置 `CUDA_VISIBLE_DEVICES` 或 `CUDA_MPS_*`；运行器按进程设置并在退出时清理。

## 3. 数据集是否需要重处理

### 3.1 本次结论

此前预检支持复用已有数据；本次仍以随后指纹 Gate 为准。Qwen3-32B、Qwen3-8B 和 Qwen3-0.6B 使用相同 token-to-id 映射、特殊 token 与 non-thinking chat template；当前服务器已经实测 32B/0.6B 的 `VOCAB_EQUAL=True`、special token 一致、template 一致。因此原来由 Qwen3-8B Target tokenizer 生成的冻结 manifest 可以逐 token 复用。文件名中的 `qwen3_8b` 只是历史名称，不代表其中保存了 8B 模型状态。

正式运行前仍须执行下面的指纹 Gate；它失败时才重生成：

```bash
TARGET_MODEL="$TARGET_MODEL" DRAFT_MODEL="$DRAFT_MODEL" QWEN3_DATA_ROOT="$QWEN3_DATA_ROOT" \
"$SPECSTREAM_PYTHON" - <<'PY'
import hashlib, json, os
from transformers import AutoTokenizer

def load(path):
    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )

target = load(os.environ["TARGET_MODEL"])
draft = load(os.environ["DRAFT_MODEL"])
assert target.get_vocab() == draft.get_vocab()
for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
    assert getattr(target, key) == getattr(draft, key), key
probe_t = target.apply_chat_template(
    [{"role": "user", "content": "fingerprint"}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False,
)
probe_d = draft.apply_chat_template(
    [{"role": "user", "content": "fingerprint"}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False,
)
assert probe_t == probe_d
vocab = json.dumps(sorted(target.get_vocab().items()), ensure_ascii=False).encode()
actual = hashlib.sha256(vocab).hexdigest()
manifest = json.load(open(
    os.path.join(os.environ["QWEN3_DATA_ROOT"], "tokenizer_manifest.json"),
    encoding="utf-8",
))
assert actual == manifest["vocab_sha256"], (actual, manifest["vocab_sha256"])
assert manifest["context_length"] == 40960
assert manifest["reserved_output_length"] == 1024
assert manifest["enable_thinking"] is False
print("QWEN3_32B_DATA_REUSE_GATE=PASS", actual)
PY

(cd "$QWEN3_DATA_ROOT" && sha256sum -c dataset_sha256.txt)
```

### 3.2 仅当指纹 Gate 失败时重生成

不能覆盖冻结目录。先写入新目录，完成检查后再令 `QWEN3_DATA_ROOT` 指向它：

```bash
export GSM8K_SOURCE=/root/autodl-tmp/dataset/gsm8k/main/test-00000-of-00001.parquet
export LONGBENCH_SOURCE=/root/autodl-tmp/dataset/LongBench-v2
export MRCR_SOURCE=/root/autodl-tmp/dataset/mrcr
export NEW_DATA_ROOT=$REPO/specstream_prepared/qwen3_32b_0p6b_$(date +%Y%m%d_%H%M%S)
mkdir -p "$NEW_DATA_ROOT"

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/prepare_qwen3_offline.py \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --gsm8k-source "$GSM8K_SOURCE" \
  --longbench-source "$LONGBENCH_SOURCE" \
  --mrcr-source "$MRCR_SOURCE" \
  --output-root "$NEW_DATA_ROOT" \
  --context-length 40960 --reserved-output-length 1024 --minimum-rows 64 \
  2>&1 | tee "$NEW_DATA_ROOT/prepare_console.log"

"$SPECSTREAM_PYTHON" scripts/specstream/paper_eval/qwen3/prepare_gsm8k_native_eval.py \
  --test-source "$GSM8K_TEST_SOURCE" --train-source "$GSM8K_TRAIN_SOURCE" \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --output-root "$NEW_DATA_ROOT" --num-shots 5 --expected-test-rows 1319

grep -q 'QWEN3_OFFLINE_DATA_GATE=PASS' "$NEW_DATA_ROOT/prepare_console.log"
(cd "$NEW_DATA_ROOT" && sha256sum -c dataset_sha256.txt)
(cd "$NEW_DATA_ROOT" && sha256sum -c gsm8k_native_eval_sha256.txt)
export QWEN3_DATA_ROOT=$NEW_DATA_ROOT
```

脚本当前仍输出 `longbench_v2_qwen3_8b_8k32k_*.json*` 这一兼容文件名；以 `tokenizer_manifest.json` 的实际 SHA256 为准，不要手工改名后混用旧 checksum。

## 4. 代码、模型、拓扑和容量预检

```bash
for f in \
  "$TARGET_MODEL/config.json" "$DRAFT_MODEL/config.json" \
  "$GSM8K_TEST" "$GSM8K_FEWSHOT" "$GSM8K_QWEN3" \
  "$LONGBENCH_QWEN3" "$MRCR_QWEN3" \
  scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  scripts/specstream/paper_eval/qwen3/run_public_once.sh \
  scripts/specstream/paper_eval/qwen3/run_public_matrix_once.sh; do
  test -s "$f" || { echo "ERROR: missing $f" >&2; false; }
done

bash -n scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh
bash -n scripts/specstream/paper_eval/qwen3/run_public_once.sh
bash -n scripts/specstream/paper_eval/qwen3/run_public_matrix_once.sh

export PREFLIGHT_ROOT=$REPO/results/${MODEL_TAG}_preflight_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PREFLIGHT_ROOT"
bash scripts/specstream/paper_eval/qwen3/preflight_public_qwen3.sh \
  2>&1 | tee "$PREFLIGHT_ROOT/console.log"
grep -q 'QWEN3_PUBLIC_PREFLIGHT=PASS' "$PREFLIGHT_ROOT/console.log"
source "$PREFLIGHT_ROOT/runtime_env.sh"

test "$TARGET_TP_SIZE" = 2
test "$COLOCATED_TP_RANK" = 0
test "$(tr ',' '\n' <<<"$TARGET_UUIDS" | wc -l)" = 2
test "$(cut -d, -f1 <<<"$TARGET_UUIDS")" = "$COLOCATED_UUID"
```

先做不启动模型的四方法命令 Gate：

```bash
export DRY_ROOT=$PREFLIGHT_ROOT/public_dryrun
for method in AR SGLANG_SD SGLANG_SD_KV_OFFLOAD SPECSTREAM_1GPU; do
  METHOD=$method DATASET_TAG=dry DATASET_NAME=random-ids \
  INPUT_LEN=1024 NUM_PROMPTS=1 OUTPUT_LEN=8 MAX_CONCURRENCY=1 \
  WARMUP_REQUESTS=0 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$DRY_ROOT" \
  SPECSTREAM_DRY_RUN=1 \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh
done

for f in "$DRY_ROOT"/logs/*/config.env; do
  grep -E '^(MODEL_TAG|TARGET_TP_SIZE|COLOCATED_TP_RANK|TARGET_VISIBLE|DRAFT_VISIBLE|TARGET_MAX_TOTAL_TOKENS|DRAFT_MAX_TOTAL_TOKENS)=' "$f"
done
```

每个 Target 命令必须有 `--tp-size 2`；外置 Draft 方法的 `TARGET_VISIBLE` 是两个 UUID，`DRAFT_VISIBLE` 等于 rank 0 UUID。
TP2 原始 profile 会先按 `*.tp0.csv`、`*.tp1.csv` 生成；运行器以拥有 SPECTRE/ZMQ 控制路径的 TP0 为规范 profile，并把原始 rank 分片保存在对应 `logs/<case>/profile_shards/`，避免汇总脚本重复计算一个 cell。

## 5. 最小 smoke（正式矩阵前必须通过）

此 smoke 使用正式 TP2/同卡外置 Draft 路径，但缩短 workload；它验证模型能加载、SPECTRE 收发正常、无 timeout/fallback，并不作为性能结果：

```bash
export SMOKE_ROOT=$REPO/results/${MODEL_TAG}_public_smoke_$(date +%Y%m%d_%H%M%S)
METHOD=K3 DATASET_TAG=tp2_pair_smoke DATASET_NAME=random-ids \
INPUT_LEN=12288 NUM_PROMPTS=2 OUTPUT_LEN=16 MAX_CONCURRENCY=1 \
WARMUP_REQUESTS=1 REQUEST_RATE=inf SEED=1 RESULT_ROOT="$SMOKE_ROOT" \
CASE_TIMEOUT_S=1800 \
bash scripts/specstream/paper_eval/qwen3/run_public_once.sh

test -s "$SMOKE_ROOT/logs/K3_tp2_pair_smoke_c1/case_complete.marker"
if grep -RniE 'CUDA out of memory|Traceback|Scheduler hit an exception|RecvTimeout|DraftFallback' \
  "$SMOKE_ROOT/logs"; then
  echo 'ERROR: inspect the matched runtime failures before continuing' >&2
  false
fi
grep -E 'max_total_num_tokens=' "$SMOKE_ROOT/logs/K3_tp2_pair_smoke_c1"/{target,draft}.log
```

## 6. 准确性正式实验

每个 pair runner 会依次运行原生 SD 和 SpecStream、评分并检查完整样本数与请求错误。使用新的结果根：

```bash
export ACC_ROOT=$REPO/results/${MODEL_TAG}_public_accuracy_once_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ACC_ROOT"/{bench,logs,profiles,gpu_monitor,summary,env}
cp "$QWEN3_DATA_ROOT/dataset_sha256.txt" "$ACC_ROOT/env/"
git rev-parse HEAD > "$ACC_ROOT/env/git_commit.txt"
git status --short > "$ACC_ROOT/env/git_status.txt"
```

GSM8K：

```bash
GSM8K_TEST="$GSM8K_TEST" GSM8K_FEWSHOT="$GSM8K_FEWSHOT" ACC_ROOT="$ACC_ROOT" \
bash scripts/specstream/paper_eval/qwen3/run_gsm8k_accuracy_pair.sh \
  2>&1 | tee "$ACC_ROOT/gsm8k_console.log"
```

LongBench-v2：

```bash
DATASET_TAG=accuracy_longbench_v2 DATASET_PATH="$LONGBENCH_QWEN3" \
OUTPUT_LEN=256 MAX_CONCURRENCY=4 CASE_TIMEOUT_S=43200 ACC_ROOT="$ACC_ROOT" \
bash scripts/specstream/paper_eval/qwen3/run_longbench_accuracy_pair.sh \
  2>&1 | tee "$ACC_ROOT/longbench_console.log"
```

MRCR：

```bash
ACCURACY_DATASET=mrcr DATASET_TAG=accuracy_mrcr_all \
DATASET_PATH="$MRCR_QWEN3" OUTPUT_LEN=1024 MAX_CONCURRENCY=4 \
CASE_TIMEOUT_S=43200 ACC_ROOT="$ACC_ROOT" \
bash scripts/specstream/paper_eval/qwen3/run_manifest_accuracy_pair.sh \
  2>&1 | tee "$ACC_ROOT/mrcr_console.log"
```

必须分别出现 `GSM8K_ACCURACY_PAIR_GATE=PASS`、`LONGBENCH_ACCURACY_PAIR_GATE=PASS` 和 `MANIFEST_ACCURACY_PAIR_GATE=PASS`；两方法 `failures=0`。只有 HTTP `error=False` 不能代替任务准确率。

## 7. 12-cell 性能正式矩阵

```bash
# SGLANG_SD 也必须使用上文统一标定的 Target 比例。旧脚本在这里硬编码
# 0.58，只能得到约 123015 tokens，会被 131072 exact-cap gate 拒绝。
grep -A8 '^  SGLANG_SD)' scripts/specstream/paper_eval/qwen3/run_public_once.sh \
  | grep -Fq -- '--mem-fraction-static "$SPECSTREAM_TARGET_MEM_FRACTION"'

export RESULT_ROOT="$REPO/results/${MODEL_TAG}_public_e2e_once_$(date +%Y%m%d_%H%M%S)"
export CASE_TIMEOUT_S=43200 ALLOW_LONG_CASE=1
mkdir -p "$RESULT_ROOT"
bash scripts/specstream/paper_eval/qwen3/run_public_matrix_once.sh \
  2>&1 | tee "$RESULT_ROOT/console.log"
```

运行器会在每个正式 cell 前执行 8-request smoke 并预测时长。正式完成标准：

```bash
grep -q 'QWEN3_PUBLIC_MATRIX_ONCE=PASS' "$RESULT_ROOT/console.log"
test "$(find "$RESULT_ROOT/logs" -name case_complete.marker | wc -l)" = 24
test "$(find "$RESULT_ROOT/bench" -name '*.jsonl' | wc -l)" = 24
if grep -RniE 'CUDA out of memory|Traceback|Scheduler hit an exception|RecvTimeout|DraftFallback' \
  "$RESULT_ROOT/logs"; then
  echo 'ERROR: inspect the matched runtime failures before continuing' >&2
  false
fi
cat "$RESULT_ROOT/summary/benchmark_summary.tsv"
cat "$RESULT_ROOT/summary/specstream_profile_summary.tsv"
```

24 包括 12 个 smoke 和 12 个正式 case；论文汇总只使用不带 `smoke_` 的 12 个 JSONL。每个正式 cell 必须检查 `config.env` 中 `MODEL_TAG=qwen3_0p6b_32b`、`TARGET_TP_SIZE=2`、正确的两个 Target UUID，以及 capacity Gate 中 Target/Draft 的实际 token cap。若 131072/196608 exact-cap 启动失败，停止整套矩阵；不能只给某个方法改容量。先保留日志，再在所有方法上统一重新标定。

## 8. 论文报告清单

- 报告 output tok/s、request throughput、TTFT、TPOT、P95/P99、accept length 与 acceptance rate。
- 同时报告 Target TP=2、物理 GPU 数=2，以及 Draft 与 rank 0 同卡；不要把兼容 ID `SPECSTREAM_1GPU` 写成单卡 32B Target。
- SpecStream profile 必须有 CPU History、H2D、GPU History hit/miss、动态 q、cohort 和 grant 证据；无 timeout/fallback 才能使用。
- 单次实验只报告观察值；不写标准差或显著性。需要误差条时另开三次重复矩阵。

## 9. 本次 P0/P1/P2 的最小验收与性能对照

先完成第 2–4 节的模型、数据指纹与新硬件 preflight。以下代码只写入手册，供操作者执行；本次交付不自动启动模型或测试。

### 9.1 一个正确性入口

```bash
export P012_CHECK_ROOT="$REPO/results/${MODEL_TAG}_p012_correctness_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$P012_CHECK_ROOT"
CUDA_VISIBLE_DEVICES="$COLOCATED_UUID" "$SPECSTREAM_PYTHON" - <<'PY' \
  2>&1 | tee "$P012_CHECK_ROOT/correctness.log"
import torch, pytest
assert torch.cuda.is_available(), "CUDA is required; CPU skips are not a GPU pass"
print("CUDA_DEVICE=", torch.cuda.get_device_name())
raise SystemExit(pytest.main(["-q", "python/sglang/test/spectre_specstream"]))
PY
test ${PIPESTATUS[0]} = 0
```

重点包含 `test_split_kv_attention.py`、`test_verifier_gpu_integration.py`、`test_staging_reuse.py`、`test_async_seal_lifecycle.py`、`test_background_grant_pump.py`、`test_catchup_leases.py`、`test_analyze_grant_events.py`。CUDA 用例必须实际执行；仅 CPU 测试通过或全部 skip 不算此入口通过。数值回归检验 attention/output/LSE 误差与生命周期；完整模型逐 token 一致性另见 02 第 5 节，不能将 kernel 阈值测试宣称为全模型 bitwise 一致。

### 9.2 一个性能入口：同一工作负载扫描并发 4/8

保持本手册的 Target/Draft exact caps、模型、冻结数据、seed、输出长度和 GPU History 固定值。先使用固定配置，不同时开启自动缓存。

```bash
export P012_PERF_ROOT="$REPO/results/${MODEL_TAG}_p012_fixed_$(date +%Y%m%d_%H%M%S)"
export SPECSTREAM_GPU_HISTORY_CACHE_TOKENS=8192
export SPECSTREAM_GPU_HISTORY_MIN_FREE_TOKENS=0
export SPECSTREAM_NUM_BUFFERS=2 SPECSTREAM_GRANT_TOKEN_QUANTUM=1
export SPECSTREAM_SPLIT_KV=auto SPECSTREAM_BACKGROUND_GRANT_PUMP=1
export SPECSTREAM_REQUIRE_SLACK_FILL=0
p012_failed=0
for conc in 4 8; do
  METHOD=SPECSTREAM_1GPU DATASET_TAG=p012_longbench DATASET_NAME=sharegpt \
  DATASET_PATH="$LONGBENCH_QWEN3" NUM_PROMPTS=8 OUTPUT_LEN=64 \
  MAX_CONCURRENCY="$conc" WARMUP_REQUESTS=1 REQUEST_RATE=inf SEED=1 \
  RESULT_ROOT="$P012_PERF_ROOT" CASE_TIMEOUT_S=1800 \
  bash scripts/specstream/paper_eval/qwen3/run_public_once.sh \
    || { p012_failed=1; break; }
done
test "$p012_failed" = 0
```

每个 case 需要 `case_complete.marker`、8 个成功请求、512 个生成 token、空 errors，且 `grant_event_gate.json` 为 PASS。计数使用服务端生成 token 数，不使用 retokenized token 数。保存 `target_command.txt`、`draft_command.txt`、`config.env`、两个 rank 的 profile 与 GPU placement。

```bash
P012_PERF_ROOT="$P012_PERF_ROOT" "$SPECSTREAM_PYTHON" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["P012_PERF_ROOT"])
for conc in (4, 8):
    case = f"SPECSTREAM_1GPU_p012_longbench_c{conc}"
    assert (root / "logs" / case / "case_complete.marker").is_file(), case
    rows = [json.loads(s) for s in (root / "bench" / f"{case}.jsonl").read_text().splitlines() if s]
    assert len(rows) == 1, "a case must contain exactly one benchmark result"
    row = rows[0]
    assert row["completed"] == 8 and row["total_output_tokens"] == 512
    assert not any(row.get("errors", []))
    gate = json.loads((root / "logs" / case / "grant_event_gate.json").read_text())
    assert gate["status"] == "PASS", gate
    print(conc, "output_tok_s=", row["output_throughput"],
          "mean_tpot_ms=", row["mean_tpot_ms"], "p99_e2e_ms=", row["p99_e2e_latency_ms"])
PY
```

旧版本与新版本必须分开保存，只有相同配置、相同数据且完整性检查通过的两组才计算改进百分比。`SPECSTREAM_SPLIT_KV=off` 只消融 split-KV，不能把它称为完整旧代码基线。并发 8 相比并发 4 的提高幅度是结果，不是预设门槛；History miss 随并发增长时，PCIe 可成为吞吐上限。短 smoke 中样本不足以完成所有 q/shape 的预热时，按相同配置增加请求数/输出长度，再讨论稳态动态 q 和重叠。

### 9.3 P2 可选实验：一次只改变一个因素

以 9.2 为基准，每次换新的 `P012_PERF_ROOT`，重新执行并发循环与检查，分别比较：

| 实验 | GPU History | buffers | catchup quantum | 解释 |
|---|---:|---:|---:|---|
| 固定基准 | 8192 | 2 | 1 | P0/P1 与后台泵的固定预算配置 |
| 预取深度 | 8192 | 4 | 1 | 额外 staging 空间换取预取深度 |
| catchup | 8192 | 2 | 4 | 量子上限允许多 token，实际发放由测量与期限限制 |
| 自动缓存 | -1 | 2 | 1 | 明确报告有效 GPU hit/miss 与内存预算变化 |
| 组合部署 | -1 | 4 | 4 | 在各独立实验通过后再评估；不是严格单因素消融 |

自动缓存属于部署配置比较，不能与第 7 节固定预算论文矩阵混合汇总。catchup=4 也不保证每次 ACK=4：prefill、提前结束、到期、剩余候选较少时允许更短前缀；检查 `catchup_multi_token_grants` 和实际 issued/ACK 的 token 计数，不能只看命令行写了 4。

### 9.4 机制证据与判读

- P0：查看 q/mode/fallback 原因、`tp_baseline_ready`、`tp_baseline_samples`、`tp_shape_key`。冷启动或未完成 shape 预热可暂用 ordinary，不能因有 ordinary 就认定出错；不应因 batch 改变而长期无条件 q=1。
- P1：比较同 shape 的 `target_forward_ms` 与 `target_enqueue_ms`；`h2d_source_bytes`、`h2d_bytes`、`h2d_dma_ops`、`host_slot_wait_ms`、`host_pack_ms` 解释传输和主机成本。`h2d_padding_bytes=0` 表示没有传输 padding，不代表 staging 张量不存在 padding。
- 跨层预取：以真实 GPU 多层用例、prefetch active 日志、完整 H2D wait/event coverage 和 exposed-copy 变化为证；缓冲数量或 copy-stream 数量本身不是重叠证明。
- 后台泵：`grant_pump_iterations>0` 表示 forward 内发生轮询；`grant_pump_wall_ms` 是线程存续 wall time，含等待，不能计入额外 CPU 计算或当成 Draft/GPU 重叠毫秒数。
- SLACK_FILL：单独把 `SPECSTREAM_REQUIRE_SLACK_FILL=1` 用于长上下文重叠机制实验；需要至少一次成功的一 token ACK、TPC width=34、每 epoch 一 issue/一终态 ACK。若物理窗口过短或基线不足，允许保持 Target-exclusive，但该轮不能宣称创新点三已生效。
- `draft_overlap_timing_source=unmeasured` 时，重叠毫秒数仍未被直接测得，不能将默认 0 当作已测量值。实际端到端 round 成本用于经验决策；copy wait event 与 ACK 记录分别用于传输和执行资格的证据。

发现 timeout、fallback、OOM、数值/生命周期错误、缺失/重复 ACK 或无 marker 时保留原日志，停止正式汇总。不得通过延长超时、取消 Gate 或手写 marker 把失败改成成功。
