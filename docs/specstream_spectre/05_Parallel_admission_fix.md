# 双 TP2 并行准入修复与收益验证

服务器：`/root/lifei/SpecStream`；实测时间：2026-09-07。

## 已修改

1. Target 已采集的 CPU 批次状态用于并行准入。当前批次存在未就绪 Draft 时，在发送请求前选择 ordinary，多 token q 保留；不再进入 parallel 后才被就绪门限降为 q=1。已有就绪 Draft 时仍保留并行候选及原有窗口/Target 保护条件。
2. 失败执行按“原计划 q + 工作负载形状”反馈，实际 q 的成本样本保持原义。无 SLACK_FILL 发放或实际 q 与计划不符时退避；前两次为 8/16 轮，第三次起为 128 轮，后续最多 512 轮，形状变化可重新评估。
3. 控制器保存最多 128 个失败条目。复用已有 round、Draft 步耗时和窗口记录识别窗口过短，不增加 CUDA 同步、GPU kernel、TP collective 或逐 token I/O；CSV 每轮增加计划模式和 SLACK_FILL 发放 token 数两列。

涉及三个运行时文件：`specstream/controller.py`、`specstream/verifier.py`、`specstream/profiler.py`，另新增 `test_parallel_admission.py`。源码均在 `python/sglang/srt/speculative/spectre/` 下。此次没有调整 KV 容量或 q 候选来制造收益。

## 验证结果

定向回归 **40 passed**，覆盖准入保留 q、失败归属、退避与恢复、形状隔离、原动态 q 行为和双 rank 控制广播顺序。

GPU 对照使用同一 LongBench 文件与 seed=1，8 请求、并发 4、每请求输出 256、预热 4；Target/Draft 均 TP2，动态 q=2/4/6/8，TPC=34，GPU History=8192，buffers=2，catchup quantum=1，KV 上限分别 131072/196608。其余模型和环境与最近正式运行一致。

| 配置 | output tokens/s | 相对旧 auto | Accept length | q=1 轮数 |
|---|---:|---:|---:|---:|
| 修复前 auto | 11.416 | +0.00% | 4.240 | 9 |
| 修复后 auto | 13.278 | +16.31% | 4.574 | 0 |
| 修复后 serial | 14.711 | +28.86% | 4.516 | 0 |

三组均完成 8/8 请求和 2048 输出 token，无客户端错误、无 DraftFallback，Grant Gate PASS。修复后 auto 与 serial 的生成文本一致数为 **5/8**。

本次消除了无效并行探测，单次 auto 吞吐提高约 16.3%。但 auto 仍比此次 serial 慢约 9.7%，且动态 q 对照中存在文本差异，因此不能把全部时间差都严格归因于准入修改。修复后的 auto 在此负载使用 ordinary，成功 SLACK_FILL 仍为 0；这不是双卡实际重叠加速的证据。新策略会在流水线未就绪时保守地持续串行；真正重叠的启动与收益仍需有就绪草稿和足够窗口作为前提。

这是单次小规模实测，不保证完整 131 请求或其他负载有相同比例提升。此前正式测试的 53 条文本分歧未逐条复核，本次输出对照不能替代其完整正确性验收。

## 分歧请求的定向重放

取本次第 3 条请求（索引 2），两次输出在第 6 个重分词后的 token 分歧，分别为 D（35）与 C（34）。固定共同前缀共 19044 个 input token，q=8、并发 1、输出 4 个 token 后，SpecStream auto/serial 均返回 `[320, 35, 8, 151645]`，两组检查均正常完成。

同一前缀的 Target-only 参考返回 `[320, 34, 8, 151645]`；其分歧位置 C 与 D 的 logprob **完全相同，均为 -0.7610415816307068**。追加共同 token 后单步 prefill 参考也观察到相同平局。这说明该位置对数值和并列最大值敏感，不能直接将文本差异解释为新增的并行状态错误；也不足以排除其他请求存在独立问题。此次未修改模型采样规则或降低精度来强行对齐输出。

证据：`prefix_comparison.json`、`prefix_reference.json`、`prefix_origin.json`，以及 `prefix2_auto/`、`prefix2_serial/`、`prefix_ar/`。初版临时诊断客户端存在路径替换语法错误，未发出模型请求，已修正并在新目录重试；其失败日志独立保留，不算验证通过。

## 使用与证据

继续使用此前完整测试命令即可，无新增必需开关：

```bash
export SPECSTREAM_DRAFT_TP_SIZE=2
export SPECSTREAM_FIXED_Q=0
export SPECSTREAM_OVERLAP_MODE=auto  # 串行对照设置 serial
```

每次使用新结果目录。公共运行器的实时进度条及日志保存方式保持一致。

结果根目录：`/root/lifei/SpecStream/results/tp2_admission_fix_20260907_211919`，其中 `old_auto/`、`new_auto/`、`new_serial/` 为三组原始结果；`comparison.json` 为汇总，`focused_tests.log` 为定向回归，`case.sh` 为实际参数。测试进程已全部退出。

代码备份：`/root/lifei/SpecStream_before_parallel_admission_20260907_212624.tar.gz`。本地服务器代码副本在 `artifacts/specstream_p012_20260907/work`，未覆盖本地其他运行时代码。
