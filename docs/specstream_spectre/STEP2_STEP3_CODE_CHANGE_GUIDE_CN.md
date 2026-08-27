# SpecStream 第二点与第三点代码修改说明（易懂版）

版本日期：2026-08-22

## 1. 先回答最重要的问题

上一版虽然已经分别提供了单卡开关和多卡开关，但两套限制规则都写在 `controller.py` 的同一个判断链中，因此“运行开关分开了，核心代码还没有完全分开”。这不符合您希望第二点、第三点独立实现和独立测试的要求。

本次已经改成：

1. 第二点的单卡规则只放在 `single_gpu_coexec_policy.py`。
2. 第三点的多卡规则只放在 `multi_gpu_tp_policy.py`。
3. `controller.py` 只负责把规则给出的限制应用到 q 和运行方式，不再自己编写单卡或多卡判断。
4. 两个功能使用不同开关。只打开第二点时，不读取多卡 TP 数据；只打开第三点时，不会自动启用单卡 MPS 规则。
5. 第二点和第三点可以在 `--specstream-profile-only` 下运行：Target 继续使用原生 SPECTRE 的全 GPU KV，不再依赖第一点的 CPU 卸载。

## 2. 两点优化现在怎样区分

### 第二点：单卡内推测性解码

含义：Target 和 Draft 是两个进程，但放在同一张物理 GPU 上。推荐用 CUDA MPS 给它们分配不同的计算份额，例如 Target 90%、Draft 10%。

主要开关：

```bash
--specstream-profile-only \
--specstream-coexec-enabled \
--specstream-dynamic-q
```

正式 MPS 实验建议再加：

```bash
--specstream-coexec-require-mps
```

这一路不会启动 TP rank 监控，也不会因为多卡 rank 时间差而调整 q。

### 第三点：多卡环境下推测性解码

含义：Target 使用多张卡做 TP，例如 Target 使用 GPU 0、1；Draft 与某一个 Target rank 共用物理卡，例如也放在 GPU 0。这里最重要的问题是：GPU 0 如果被 Draft 抢占太多，GPU 1 会等待 GPU 0，最终拖慢整个 Target。

主要开关：

```bash
--specstream-profile-only \
--specstream-tp-straggler-control \
--specstream-colocated-tp-rank 0
```

第三点通常也会打开第二点的同卡保护：

```bash
--specstream-coexec-enabled
```

但这不是代码上的强制绑定。两套规则仍是两个独立模块，只是在第三点完整实验中按优先级组合使用。

## 3. 代码结构

### 3.1 第二点独立模块

文件：`python/sglang/srt/speculative/spectre/specstream/single_gpu_coexec_policy.py`

它只看两类信息：

1. Draft 是否越来越忙，例如返回时间接近超时、等待请求变多、出现超时或拒绝。
2. 当前 Target 是否处于计算很重的阶段。

如果压力变大，它会先缩短 q。简单理解：一次少让 Draft 猜一些 token，避免 Draft 长时间占用同一张卡。如果压力继续恶化，外层安全机制会暂时回到普通解码。

这个文件没有导入 TP rank 监控，不知道有多少张 Target 卡，也不处理多卡通信。

### 3.2 第三点独立模块

文件：`python/sglang/srt/speculative/spectre/specstream/multi_gpu_tp_policy.py`

它只看 Target 各个 TP rank 的耗时：

1. 轻微差异：保持当前运行。
2. 中等差异：把 q 限制到较小值，并暂时让 Draft 与 Target 错开，减少共卡 rank 被拖慢。
3. 严重差异：本轮回到 q=1 的普通解码，优先保证 Target 不被某一个慢 rank 卡住。

这个文件不读取 MPS 百分比，也不读取 Draft 的队列或超时数据。

### 3.3 公共协调层

文件：`python/sglang/srt/speculative/spectre/specstream/controller.py`

公共层保留三件事：

1. 根据已有成本数据选择 q。
2. 执行 Draft 真正超时后的短期回退。
3. 接收第二点或第三点模块给出的上限。

优先级为：语义安全回退 > Draft 真超时 > 严重多卡不平衡 > 中等多卡不平衡 > 单卡压力限制 > 正常成本选择。

### 3.4 运行时接线

文件：`python/sglang/srt/speculative/spectre/specstream/verifier.py`

接线规则如下：

1. `--specstream-coexec-enabled` 为真时，才创建 `SingleGPUCoexecPolicy`。
2. `--specstream-tp-straggler-control` 为真时，才创建 `MultiGPUTPPolicy` 和 TP 时间监控器。
3. 两个开关都关闭时，两套新增策略都不运行。

`--specstream-profile-only` 这个名字容易误解。现在它表示“只启用统计/控制运行时”：不安装创新点一的流式 attention backend，`_maybe_seal()` 会直接返回，因此 KV 不会从 GPU 搬到 CPU。只有 `--specstream-enabled` 才会真正执行创新点一。

## 4. 其他配套修改

### Draft 压力记录

`draft_load_tracker.py` 记录 Draft 返回时间、接近超时的程度、等待请求数、超时比例和拒绝比例。它记录的是 Target 实际观察到的服务情况，不假装知道 Draft 内部 CUDA 队列。

### MPS 环境检查

`mps_env.py` 读取进程启动时继承到的 MPS 配置。正式单卡实验可以要求必须存在 MPS；如果没有配置，会直接报错，避免把“没有隔离的同卡运行”误当成正式结果。

### 多卡时间监控

`tp_straggler_monitor.py` 保存各 Target rank 的前向时间，并计算共卡 rank 比其他 rank 慢多少。rank 0 做统一决策后广播，保证所有 Target rank 使用相同 q 和相同模式。

### 日志与汇总

`profiler.py` 增加了 Draft 返回时间、超时率、MPS 信息、Target 前向时间、多卡时间差、回退原因等字段。`scripts/specstream/summarize_specstream_profile.py` 可把每轮 CSV 汇总成一行，方便比较不同实验。

### 实验脚本

1. `scripts/specstream/run_dedicated_draft_target_baseline.sh`：Draft 和 Target 使用不同卡的 baseline。第二点可用 2 张卡；第三点 TP=2 baseline 需要 3 张卡。
2. `scripts/specstream/mps/run_static_percent_scan.sh`：同一物理卡下扫描 90/10、80/20、70/30、60/40。
3. `scripts/specstream/mps/run_h1_colocated_tp.sh`：Target 多卡 TP，Draft 与指定 rank 共卡，并执行一次 benchmark。
4. `scripts/specstream/mps/start_mps.sh`、`stop_mps.sh`：启动和停止 MPS。

## 5. 运行模式的简单解释

- q：Draft 一次计划向前猜多少步。q 越大，可能一次确认更多 token，但占用时间和失败代价也可能更大。
- parallel：尽量让已有的拷贝、计算和验证重叠。
- ordinary：更保守地执行，减少重叠带来的资源争用。
- COEXEC：允许 Draft 与 Target 同时工作。
- THROTTLE：仍可同时工作，但限制 q，减轻 Draft 压力。
- SERIALIZE：暂时错开 Draft 与 Target，优先保护 Target。
- FALLBACK：本轮回到 q=1 普通解码。

## 6. 本次新增的独立测试

1. `test_single_gpu_coexec_policy.py`：只测试单卡策略，不构造 TP 数据。
2. `test_multi_gpu_tp_policy.py`：只测试多卡策略，不构造 Draft 压力数据。
3. `test_dynamic_q.py`：增加“不开第二点就忽略 Draft 压力”“不开第三点就忽略 TP 时间差”的测试，防止两个功能再次被无意绑定。

本地 CPU 控制层回归结果：49 个测试通过，10 个需要真实 GPU/TP 环境的测试按条件跳过。

## 7. 当前边界

本次代码已经完成控制逻辑、监控、日志和实验脚本，但当前 Windows 工作区没有 NVIDIA MPS/多卡 Linux 运行环境，所以不能在这里代替您完成正式 GPU 性能结论。正式结论必须按照配套测试文档，在同一台 Linux 多 GPU 机器上执行 baseline 和优化组。
