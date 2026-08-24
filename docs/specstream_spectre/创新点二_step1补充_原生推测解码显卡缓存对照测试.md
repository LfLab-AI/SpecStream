# 创新点二、三独立测试：原生推测解码全显卡缓存对照与隔离检查

## 1. 这一步真正要比较什么

创新点二和创新点三的实验都必须把 Target KV 完整保存在 GPU，不能启用创新点一的 CPU 卸载。

正确的三组关系是：

| 组别 | Target KV | 创新点二控制 | 创新点三控制 | 用途 |
|---|---|---|---|---|
| Native-B0 | 原生 SPECTRE，全在 GPU | 关闭 | 关闭 | 纯原生基线 |
| I2-O1 | 原生 SPECTRE，全在 GPU | 开启 | 关闭 | 单独测创新点二 |
| I3-O1 | 原生 SPECTRE，全在 GPU | 可作为底层共卡控制 | 开启 | 单独测创新点三 |

这里不再比较“GPU KV 与 CPU 卸载 KV”。创新点一的测试仍放在原来的 `STEP1/STEP2/STEP3` 文档中，不混入本组结果。

## 2. 两个开关的区别

- `--specstream-enabled`：启用创新点一，会把一部分 Target KV 搬到 CPU。本组实验禁止使用。
- `--specstream-profile-only`：Target 仍走原生 SPECTRE attention，KV 全在 GPU；它只启动统计和控制代码。本组需要用它来运行创新点二、三。

以下参数也属于创新点一，本组实验禁止出现：

```text
--specstream-chunk-tokens
--specstream-num-buffers
--specstream-chunks-per-transfer
--specstream-active-tail-tokens
--specstream-min-history-tokens
--specstream-cpu-memory-gb
--specstream-layer-prefetch
--specstream-gpu-reserve-mb
--specstream-cohort-enabled
--specstream-full-restore-baseline
```

## 3. 先运行代码测试

```bash
cd ~/lifei/specdecode/baseline/sglang
conda activate spectre

PYTHONPATH=python pytest -q \
  python/sglang/test/spectre_specstream/test_control_profile.py \
  python/sglang/test/spectre_specstream/test_dynamic_q.py \
  python/sglang/test/spectre_specstream/test_single_gpu_coexec_policy.py \
  python/sglang/test/spectre_specstream/test_multi_gpu_tp_policy.py
```

目的：确认 `--specstream-profile-only` 可以单独启动创新点二、三控制，同时拒绝 Cohort、Full-Restore 等卸载专用功能。

## 4. Native-B0：纯原生 SPECTRE 基线

Target 命令中不要写任何 `--specstream-*` 参数：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /你的/Target模型目录 --port 30000 \
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
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/native_B0_target.log
```

Draft 在独立 GPU1 上启动：

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path /你的/Draft模型目录 --port 30001 \
  --skip-server-warmup \
  --speculative-algorithm SPECTRE --spectre-role draft \
  --speculative-num-steps 4 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --mem-fraction-static 0.45 --max-total-tokens 196608 \
  --max-running-requests 8 --spectre-max-batch-size 8 \
  --chunked-prefill-size 2048 \
  --disable-radix-cache --disable-cuda-graph \
  --spectre-zmq-addr 127.0.0.1 --spectre-zmq-port 29000 \
  2>&1 | tee logs/native_B0_draft.log
```

目的：得到 Draft/Target 分卡、无创新控制、无 KV 卸载时的原生 SPECTRE 吞吐和延迟上限。

## 5. I2-O1：原生 GPU KV 加创新点二

使用创新点二 Step 3 的同卡 MPS 布局。Target 命令的关键部分必须是：

```bash
--specstream-profile-only \
--specstream-dynamic-q \
--specstream-q-candidates 1,2,4,6,8 \
--specstream-coexec-enabled \
--specstream-coexec-require-mps \
--specstream-profile-path profiles/innovation2_native_gpu_kv.csv
```

目的：只开启动态 q、共执行和反压。由于没有 `--specstream-enabled`，Target KV 不会离开 GPU。

完整 Target、Draft、MPS 和 benchmark 命令见：

```text
创新点二_step2_单卡静态配额扫描测试.md
创新点二_step3_动态共执行与反压测试.md
```

## 6. I3-O1：原生 GPU KV 加创新点三

使用创新点三 Step 3 的 TP=2 共卡布局。Target 命令的关键部分必须是：

```bash
--specstream-profile-only \
--specstream-dynamic-q \
--specstream-coexec-enabled \
--specstream-tp-straggler-control \
--specstream-colocated-tp-rank 0 \
--specstream-profile-path profiles/innovation3_native_gpu_kv.csv
```

目的：只开启共卡资源控制和 TP 慢节点保护。不能添加 Cohort，因为 Cohort 依赖创新点一的 CPU History/H2D 路径。

完整 Target、Draft、MPS 和 benchmark 命令见：

```text
创新点三_step1_多卡独立草稿模型基线测试.md
创新点三_step2_多卡朴素共卡测试.md
创新点三_step3_多卡慢节点控制测试.md
```

## 7. 三组使用同一条 benchmark 命令

每次完整重启 Target 和 Draft 后直接运行：

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --model /你的/Target模型目录 --tokenizer /你的/Target模型目录 \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 16384 --random-output-len 256 \
  --random-range-ratio 1 \
  --request-rate inf --max-concurrency 8 \
  --warmup-requests 8 --seed 1 --flush-cache --output-details \
  --tag native-gpu-kv-comparison \
  --output-file results/native_gpu_kv_comparison.jsonl
```

目的：三组保持模型、输入长度、输出长度、请求数、并发数和随机种子一致。每组应修改 `tag` 和输出文件名，正式测试重复 3 次并取中位数。

## 8. 检查是否错误地启用了卸载

先检查日志：

```bash
grep -E 'CPU History|H2D|tiered KV|B3 tiled|Full-Restore|cohort' \
  logs/innovation2_*/*.log logs/innovation3_*/*.log
```

目的：正常情况下不应看到 CPU History seal、H2D streaming 或 Cohort 被启用。看到 `control-only mode` 是正常的。

再检查 profile：

```bash
python - <<'PY'
import csv
from pathlib import Path

paths = list(Path('profiles').glob('innovation2*/**/*.csv'))
paths += list(Path('profiles').glob('innovation3*/**/*.csv'))
failed = False
for path in paths:
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    cpu = max((int(float(r.get('cpu_history_bytes', 0) or 0)) for r in rows), default=0)
    h2d_bytes = sum(int(float(r.get('h2d_bytes', 0) or 0)) for r in rows)
    h2d_ops = sum(int(float(r.get('h2d_ops', 0) or 0)) for r in rows)
    print(path, 'cpu_history_bytes=', cpu, 'h2d_bytes=', h2d_bytes, 'h2d_ops=', h2d_ops)
    failed |= cpu != 0 or h2d_bytes != 0 or h2d_ops != 0
raise SystemExit(1 if failed else 0)
PY
```

目的：退出码必须为 0，而且每个文件的 `cpu_history_bytes`、`h2d_bytes`、`h2d_ops` 都必须为 0。这是“没有混入创新点一”的直接证据。

## 9. 最终对比表

| 组别 | KV 位置 | 物理 GPU 数 | 控制器 | 吞吐 | P99 | goodput/GPU | CPU KV bytes | H2D ops |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Native-B0 | 全部 GPU | 2（I2）或 3（I3） | 无 |  |  |  | 0 | 0 |
| I2 静态 MPS | 全部 GPU | 1 | 仅静态配额 |  |  |  | 0 | 0 |
| I2-O1 | 全部 GPU | 1 | 动态 q/共执行/反压 |  |  |  | 0 | 0 |
| I3-P1 | 全部 GPU | 2 | 无 TP 保护 |  |  |  | 0 | 0 |
| I3-O1 | 全部 GPU | 2 | TP 慢节点保护 |  |  |  | 0 | 0 |

创新点二的主要比较是 `I2-O1 vs I2 静态 MPS`，并用 `Native-B0` 给出分卡性能上限。创新点三的主要比较是 `I3-O1 vs I3-P1`，并用 `Native-B0` 给出独立 Draft 的性能上限。
