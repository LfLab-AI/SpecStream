# Draft TP2 修改与服务器小规模实测（2026-09-07）

已部署到 `/root/lifei/SpecStream`。Draft 可在 Target 的两张卡上按 TP2 运行；本次验证了功能，但没有证明稳定的吞吐提升，保留 Draft TP1 默认值。

## 修改内容

- 增加 Draft TP 开关，共用 Target GPU 列表；Target、Draft 各自保留独立 TP 进程组。
- 各 Target rank 发布本地 KV 传输窗口，只有所有 rank 同一轮且窗口仍有效时才允许联合重叠租约；后台线程不调用 TP collective。
- Draft 各 rank 统一控制时钟、请求顺序和启动准入；任一 rank 不满足期限则共同延后，所有 rank GPU 工作完成后才报告 ACK。
- 修复 Target 接收竞态：GPU 恰在检查后完成时，旧逻辑可能睡满接收超时而没有发出 catchup；现在有租约控制器时最多等待 1 ms 后重新推进。

## 实测结果

2 × A800 PCIe，Target Qwen3-32B TP2、Draft Qwen3-0.6B。三组使用同一 LongBench 性能输入、4 请求、并发 4、每请求输出 64、greedy、固定 q=4；Target/Draft KV token 上限分别 131072/196608，GPU History=8192，buffers=2，catchup quantum=1，重叠 TPC=16、catchup TPC=54。TPC=16 是此次固定对照参数，未证明最优。

| 配置 | output tokens/s | 相对 TP1 串行 | Grant / ACK |
|---|---:|---:|---:|
| Draft TP1 / 串行 | 7.631 | +0.00% | 347 / 347 |
| Draft TP2 / 串行 | 7.752 | +1.58% | 351 / 351 |
| Draft TP2 / 自动重叠 | 7.590 | -0.54% | 351 / 351 |

三组各完成 4/4 请求、256 输出 token，生成文本逐项一致；均无 DraftFallback，Grant Gate 全部 PASS，完成 marker 齐全，任务退出码 0。测试进程已退出。

初次部署的三个相关测试文件共 13 passed；接收竞态修复后，相关文件 3 passed（含新增竞态回归），这些不是全量回归结果。首次四组运行曾发现一次回退；以上仅采用修复后的三组复测，首次结果保留作诊断。

**性能边界：** TP2 串行约 +1.6%，TP2 自动重叠约 -0.5%，单次小样本不足以认定稳定收益。本次自动模式只有 1 个 parallel 探测轮，成功 SLACK_FILL 为 0，因此尚未验收实际双卡气泡重叠。约 565 ms 的 H2D event 时间不等于可独占的空闲窗口；记录到的最大剩余窗口仅 1183 us。TP2 缩短了观测到的 Draft 接收等待中位数（约 61 → 43–44 ms），但 Target forward 中位数仍约 575 ms，因此整体收益有限；这些阶段时间不能直接相加。

## 使用方式

在 01 文档现有环境准备完成后，运行公共运行器的 `C/SPECSTREAM_1GPU` 用例前设置：

```bash
export SPECSTREAM_DRAFT_TP_SIZE=2
export SPECSTREAM_OVERLAP_MODE=auto  # serial 为串行对照
export SPECSTREAM_FIXED_Q=4         # 0 恢复动态 q
```

公共运行器为 TP2 自动设置独立窗口目录。默认值仍为 Draft TP1、auto、动态 q。各组必须使用不同 CASE_TAG / RESULT_ROOT，避免旧完成标记或结果混用。不要将 `SPECSTREAM_REQUIRE_SLACK_FILL=0` 下通过通用 Gate 解释为重叠成功；机制验收需要成功 SLACK_FILL 证据。本次没有运行完整 131 请求 LongBench 或重新比较原生 SGLang。

上述三个开关仅适用于公共运行器的 C/SPECSTREAM_1GPU；02 文档 K1–K5 消融定义没有随之改变，需要 TP2 对照时使用公共运行器的独立实验。

## 证据与备份

复测目录：`/root/lifei/SpecStream/results/draft_tp2_smoke_20260907_180500`，其中 `comparison.json` 为汇总，`bench/` 为原始结果，`logs/` 含 Grant Gate，`run.sh` 记录实际参数。

首次诊断：`/root/lifei/SpecStream/results/draft_tp2_smoke_20260907_175202`。

初次代码备份：`/root/lifei/SpecStream_before_draft_tp2_20260907_175051.tar.gz`。

接收修复前备份：`/root/lifei/SpecStream_before_tp2_receive_fix_20260907_180500.tar.gz`。

当前交付清单：`/root/lifei/SpecStream/results/p012_20260907/tp2_final_delivery.json`。本地服务器代码副本位于 `artifacts/specstream_p012_20260907/work`，没有覆盖本地其他未提交的运行时代码。
