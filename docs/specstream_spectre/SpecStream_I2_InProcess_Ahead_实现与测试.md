# SpecStream 创新点二：进程内 Optimistic Draft-Ahead

## 1. 实现边界

本实现以 `STANDALONE` Spec V2 为唯一主路径，在一个 SGLang 进程内让
Target Verify 与同一请求的下一轮 Draft 候选生成并发执行。Phase A 明确不
启用 MPS、ZMQ、GRANT/ACK、TPC mask 或 dynamic q；原有双进程 SPECTRE
代码保持不变，可继续作为双卡基线和同卡双进程负基线。

当前固定约束如下：

- 单 GPU：`--tp-size 1 --dp-size 1`；
- 贪心解码；非贪心请求自动回退到原生串行路径；
- `--page-size 1`、`--speculative-eagle-topk 1`；
- q=4：`--speculative-num-steps 4 --speculative-num-draft-tokens 5`；
- ahead depth `h ∈ {1, 2, 4}`；
- 有 grammar 的请求自动回退到原生串行路径；
- 当前按 batch 整体提升 ahead 结果：batch 内任一请求发生分叉时，整个
  batch 使用原生 Draft extend 修复。C1 是首要正确性和性能 Gate。

Target 和 Draft 使用独立物理 KV tensor，但 STANDALONE V2 共享位置表和
分配器。因此分叉时不能释放共享槽位。实现使用最长公共前缀确定 fork
point，将错误 Draft 后缀逻辑失效，再由原生 Draft extend 在共享位置上覆写
修复；Target KV 和 Target token 结果从不回滚。

## 2. 启动参数

下列公共参数中的模型路径需要替换为实验使用的 Target/Draft 模型：

```bash
COMMON_ARGS="\
  --model-path ${TARGET_MODEL} \
  --speculative-algorithm STANDALONE \
  --speculative-draft-model-path ${DRAFT_MODEL} \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --page-size 1 --tp-size 1 --dp-size 1"
```

P0 原生 STANDALONE：

```bash
python -m sglang.launch_server ${COMMON_ARGS}
```

P1 进程内框架串行语义控制组：

```bash
python -m sglang.launch_server ${COMMON_ARGS} \
  --specstream-inproc-enabled \
  --specstream-inproc-mode serial
```

P2 无 TPC optimistic ahead，分别测试 h=1/2/4：

```bash
python -m sglang.launch_server ${COMMON_ARGS} \
  --specstream-inproc-enabled \
  --specstream-inproc-mode ahead-free \
  --specstream-inproc-ahead-depth 4 \
  --specstream-inproc-profile-path results/i2_ahead_h4.jsonl
```

自适应模式在预热后根据 Verify 时间窗、每个 ahead token 时间、reuse EMA
和 repair cost 在 `SERIAL` 与 `AHEAD_FREE` 之间选择，并从 1/2/4 中选择
不超过配置上限的 h：

```bash
python -m sglang.launch_server ${COMMON_ARGS} \
  --specstream-inproc-enabled \
  --specstream-inproc-mode auto \
  --specstream-inproc-ahead-depth 4 \
  --specstream-inproc-min-reuse-ratio 0.25
```

`SGLANG_ENABLE_SPEC_V2` 会在启用该功能时自动打开，无需额外导出环境变量。

## 3. 正确性 Gate

必须在 temperature=0 下覆盖以下四种 reconciliation：

1. q 个候选和 bonus anchor 全部匹配；
2. 第一个候选 reject；
3. 中间候选 reject；
4. q 个候选全部接受，但 Target bonus 与 ahead anchor 不同。

四类场景均要求最终 Target token 序列与 P0 原生 STANDALONE 逐 token 完全
一致。纯状态机单元测试：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/spec/test_specstream_inproc.py
```

GPU 集成测试应对相同 prompts 分别运行 P0、P1、P2，并保存原始 response
JSON 后逐 request 比较 token IDs；不能只比较最终文本或任务准确率。

## 4. Profile 字段

JSONL 每轮记录：

- `ahead_tokens_generated/reused/discarded`；
- `local_rollback_tokens`、`repair_tokens`、`repair_ms`；
- `verify_ms`、`ahead_ms`、`actual_overlap_ms`、`overlap_ratio`；
- `target_slowdown`（Phase A 尚无串行在线参考时为 `null`）。

正常路径只使用 CUDA Event；没有 `torch.cuda.synchronize()`、stream-wide
`synchronize()` 或逐 token CPU 同步。reconciliation 在每轮边界仅等待
Target `verify_done` 和本轮 `ahead_done` 两个事件。

## 5. Go/No-Go

先在 C1/C4/C8 上比较 P0、P1、P2。只有无 TPC 的 P2 至少在一个 workload
超过 P0，并且 P1 与 P0 接近，才进入 Draft-only stream TPC 阶段。否则应先
分析 `ahead_reuse_ratio`、`overlap_ratio`、repair cost 和 Target 干扰，不应
继续叠加 TPC patch。
