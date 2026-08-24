# SpecStream 第二、三点代码实现说明

本次实现没有删除第一点，但把第二点、第三点与第一点解耦。使用 `--specstream-profile-only` 时，Target 走原生 SPECTRE attention，KV 全部驻留 GPU；只有显式使用 `--specstream-enabled` 时才进入第一点的 tiered-KV、CPU History 和 H2D streaming 路径。

## 第二点：卡内 Draft-Verify 协同执行

新增运行时控制链：

- `draft_load_tracker.py`：维护 Drafter RTT EMA/P95、deadline pressure、pending batch、timeout/missing/REJECT 比例。
- `controller.py`：在原 `(ordinary|parallel, q)` 决策上增加 `COEXEC / THROTTLE / SERIALIZE / FALLBACK`。timeout 仍具有最高优先级；压力升高时先限制 q，TP/负载严重超预算时执行 batch-uniform q=1 fallback。
- `mps_env.py`：只读取并校验进程启动时的 MPS 配置，不伪装运行中可修改 active-thread percentage。
- `profiler.py`：写出 coexec mode/reason、Drafter RTT/timeout/pending、MPS 配置和真实 GPU Target forward 时间。

Target 启动示例新增：

```bash
--specstream-profile-only \
--specstream-coexec-enabled \
--specstream-coexec-draft-pressure-ratio 0.80 \
--specstream-coexec-timeout-rate-threshold 0.10 \
--specstream-coexec-pending-high-watermark 16 \
--specstream-coexec-compute-ratio-threshold 0.90
```

若正式同卡实验必须通过 MPS 隔离，可再加 `--specstream-coexec-require-mps`。无 MPS 的 C1 feasibility baseline 不应加该参数。

`scripts/specstream/mps/` 提供 MPS 启停、90/10-60/40 静态扫描以及 H1 TP colocate 启动器。脚本通过环境变量接收完整 Target、Draft 和 benchmark 命令，不固定模型路径或端口。

## 第三点：TP-straggler 控制与层次化多 GPU

新增 `tp_straggler_monitor.py`，周期性 all-gather 各 Target rank 的 GPU forward 时间。rank0 使用统一快照决定 q/mode，随后广播给全部 TP ranks。控制优先级为：

1. 协议/语义安全与 Drafter timeout fallback；
2. 严重 TP slowdown/rank skew：q=1 `FALLBACK`；
3. 中度 TP skew：q 上限收缩并进入 `SERIALIZE`；
4. Drafter deadline pressure：`THROTTLE`；
5. 预算内：`COEXEC`。

TP=2 H1 Target 启动参数：

```bash
--specstream-profile-only \
--specstream-coexec-enabled \
--specstream-tp-straggler-control \
--specstream-colocated-tp-rank 0 \
--specstream-tp-straggler-budget-ms 1.0 \
--specstream-target-slowdown-budget 0.10 \
--specstream-tp-monitor-interval 8
```

当前 `rank_skew` 基于完整 Target forward 的 GPU event；它包含 collective 等待，是在线安全门控信号，不替代 Nsight/NCCL 的逐 collective 分解。创新点二、三的独立实验不使用 cohort，因为 cohort 属于第一点的 CPU History/H2D 路径。

## 验证

CPU 控制层测试：

```bash
PYTHONPATH=python pytest -q python/sglang/test/spectre_specstream
```

GPU/TP 正式验证仍需按附件 Gate 执行：MPS 静态份额扫描、TP=2 H0/H1 对照、Drafter timeout/REJECT 故障注入，以及 P99、goodput/GPU 和每 rank forward/collective wait 的联合报告。profile 中的 CPU History bytes 和 H2D ops 必须保持为 0。
