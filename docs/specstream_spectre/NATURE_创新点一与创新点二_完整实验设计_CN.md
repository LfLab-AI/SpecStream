# SpecStream 创新点一与创新点二：Nature 风格完整实验设计

> **One-sentence argument (English).** In long-context speculative serving, SpecStream preserves the target model's authoritative full-history verification semantics while reducing the GPU-resident KV cost and the dedicated-drafter GPU cost through verification-native bounded KV streaming and target-priority single-GPU Draft–Verify co-execution.
>
> **一句话论断（中文）.** SpecStream 在保持 Target 完整历史权威验证语义的前提下，通过验证原生的有界 KV 流式执行降低 GPU KV 驻留成本，并通过 Target-priority 单卡 Draft–Verify 共执行降低专用 Drafter GPU 成本。

## 0. 本方案的范围与结论先行

本方案只覆盖当前论文的创新点一和创新点二，不把多 GPU TP/PP 的创新点三混入主实验。实验按照“语义正确性 → 机制证据 → 端到端收益 → 消融与边界 → 组合收益”的证据阶梯组织。

论文需要回答五个问题：

1. SpecStream 是否保持 Target 的完整历史验证语义，而不是仅保持平均任务分数？
2. 多 Query 是否真正复用同一请求的 History 传输，且 GPU staging 是否与 History 长度解耦？
3. 分块、双缓冲、跨层预取、动态 q 和 Chunk-Cohort 分别贡献了什么？
4. 单卡共执行是否在减少一张专用 Drafter GPU 后仍获得更高的资源效率，并控制 Target slowdown 与 P99？
5. 创新点一释放的显存和阶段空隙是否扩大创新点二的单卡可运行区域？

核心评价不应只用 raw throughput。主指标为 **SLO goodput/GPU**，并同时报告 output throughput、P99 TTFT、P99 TPOT/ITL、P99 E2E、Target slowdown、物理 GPU 数、峰值 GPU 显存、H2D/有效 token 和错误率。

## 1. 代码到论文主张的对应关系

| 论文组件 | 当前实现位置 | 可直接测量的证据 | 论文中允许的表述 |
|---|---|---|---|
| CPU sealed History 与异步 seal | `cpu_history_store.py`、`verifier.py::_maybe_seal/_poll_pending_seals` | `cpu_history_bytes`、`gpu_kv_bytes`、seal 生命周期测试 | 已确认 History 异步封存到 CPU，完成事件后才释放 GPU slot |
| Full-Restore 对照 | `verifier.py::_restore_history`、`--specstream-full-restore-baseline` | H2D bytes、round latency、峰值显存 | 每轮物化完整 CPU History 的对照；当前代码不应称为“已证明异步隐藏的全量加载” |
| 有界分块流式验证 | `staging_runtime.py`、`verifier.py::_stream_history_*` | `staging_bytes`、`h2d_bytes/ops`、`stream_attn_ms/ops` | 固定 staging 下覆盖完整 History |
| fused tiled/grouped transfer | `triton_stream_attn.py`、`submit_many`、`chunks_per_transfer` | H2D ops、attention launches、round latency | 相邻 chunk 共用 ready event 和 fused attention launch |
| 双缓冲与跨层预取 | `StagingWindowPool`、`layer_prefetch` | overlap、staging wait、Nsight 时间线 | Copy/compute 重叠与下一层 History 预取 |
| 动态 q | `IOAwareController`、`cost_model.py` | q 分布、accepted tokens、time/useful-token | 基于 I/O、验收率和在线 profile 的 q 选择 |
| ordinary/parallel 模式 | `--spectre-fixed-q-mode`、`IOAwareController` | mode 分布、fallback reason | 固定模式可直接比较；当前成本式下正常状态几乎总偏向 parallel，不能预设会主动频繁切换 |
| Chunk-Cohort | `cohort_scheduler.py`、`_stream_history_cohorts` | cohort size、H2D ops、attention ops、P99 | 对同一已调度 batch 内的兼容请求做 packed/fused 分组；不是跨请求 KV 内容去重 |
| 单卡静态校准 | `--specstream-smctrl-calibration-tpcs`、`libsmctrl` | Draft step、Target slowdown、TPC mask validation | 用固定 TPC 建立干扰曲线和离线安全表，不是最终在线策略 |
| Target-priority 动态共执行 | `GpuGrantController`、`TargetGrantRuntime`、grant/ACK 协议 | grant state/epoch、TPC 范围、slowdown latch、fallback | 基于实测 profile 的 one-token grant、Target-exclusive fail-closed 与在线反压 |
| 原生 GPU KV 的隔离实验 | `--specstream-profile-only` | 与创新点二相同的控制指标，但无 CPU History/H2D | 隔离创新点二，不混入创新点一 |

### 1.1 必须提前锁定的术语

| 规范术语 | 定义 | 不建议替代写法 |
|---|---|---|
| GPU-resident SPECTRE | Target KV 全驻 GPU 的 SPECTRE parallel baseline | 原生 SpecStream |
| Full-Restore | sealed History 每轮整体物化到 GPU 的 CPU-offload 对照 | 异步全量加载优化（当前未被代码独立证明） |
| bounded fused streaming | 有界 staging、grouped transfer、fused tiled attention | KV offload |
| dynamic horizon | 联合选择 q 和 ordinary/parallel mode 的控制层 | dynamic model |
| Chunk-Cohort | 同一 verify batch 内兼容请求的 chunk packing/fused execution | 相同 KV 共享或 KV 去重 |
| target-priority TPC grant | 由 Target 发出的单 token、实测 profile 约束的 Draft TPC 授权 | MPS 动态调度 |
| exact | 完整历史覆盖和 causal 语义 exact | bitwise identical |
| Goodput/GPU | 满足预设 SLO 的完成请求或有效输出 token，除以时间和物理 GPU 数 | 仅 tokens/s/GPU |

## 2. 研究问题、假设与确认性终点

| RQ | 预注册假设 | 主要终点 | 支撑证据 |
|---|---|---|---|
| RQ1 正确性 | SpecStream 不改变 Target 权威语义 | greedy token exact-match = 100%；任务分数非劣 | shadow attention、first divergence、GSM8K、LongBench v2 |
| RQ2 I/O 复用 | q 增大时物理 H2D/round 不按 q 线性增加 | H2D bytes/round 对 q 的斜率接近 0；H2D/useful-token 下降 | q=1/2/4/6/8 机制扫描 |
| RQ3 有界显存 | History 增长不导致 staging 线性增长 | staging 峰值在 4K–30K 近似恒定 | context sweep、GPU memory 分解 |
| RQ4 流水收益 | grouped transfer、双缓冲和预取降低暴露 I/O | exposed-copy、staging-wait、round time 下降 | B2/B3 及逐项消融、Nsight |
| RQ5 动态控制 | dynamic q 在混合负载下接近离线 oracle | time/useful-token 或 SLO goodput 与 oracle 差距 | fixed-q 与 dynamic-q 配对比较 |
| RQ6 Cohort | cohort 降低多请求 launch/event 开销且不放大 P99 | H2D ops、attn ops、round time 下降，P99 不恶化 | 同质/异质 workload，cohort-size 消融 |
| RQ7 单卡共执行 | 存在 Target slowdown 可控的 TPC Pareto 区 | slowdown ≤5% 且 Draft step 能覆盖验证窗口 | q×TPC 校准图 |
| RQ8 在线 gating | 在线策略优于同卡无控制和最佳固定安全 TPC | Goodput/GPU 提升或 P99 降低 | 独立 5 次运行、故障/负载跃迁 |
| RQ9 联合收益 | KV streaming 扩大单卡共执行的可运行区域 | 同卡 OOM 边界、Goodput/GPU、P99 | profile-only I2 与 I1+I2 配对 |

所有主要假设在看主结果前固定。若某一假设未通过，按预设 No-Go 收敛，不以事后挑选 workload 替代。

## 3. 实验环境与公平性锁定

### 3.1 主平台

- GPU：A800 80GB PCIe，记录 GPU UUID、总 TPC 数、功率上限、温度、驱动、CUDA driver/runtime、NVCC 与 PyTorch CUDA build。
- 主模型对：Qwen2.5-7B-Instruct Target + Qwen2.5-0.5B-Instruct Drafter。
- 泛化模型对：在资源允许时加入 Qwen2.5-14B-Instruct + 0.5B Drafter，至少重复准确性和 16K/c8 主性能点。
- dtype、attention backend、page size、tokenizer、chat template、max context、radix cache、CUDA graph 和 overlap scheduler 在一个对照块内完全一致。
- Target 与 Drafter checkpoint、代码 commit、工作树 diff、C++ ZMQ schema、libsmctrl commit/hash 均进入 run manifest。

### 3.2 公平性规则

1. 同一对照只改变目标机制对应的 flag；模型、输入 token IDs、输出长度、EOS 策略、seed 和请求到达序列保持一致。
2. Target 先启动、Drafter 后启动；每个独立重复完整重启两端，不能在同一服务进程上连续覆盖多个配置。
3. 性能模式统一关闭或统一开启 radix cache、CUDA graph 和 overlap schedule。机制实验建议先全部关闭；最终系统结果再补一组生产优化开启的外部有效性实验。
4. run 顺序采用随机区组或 Latin-square，避免所有 baseline 总在低温/空闲时先跑。
5. 每次正式 run 前预热 32 个请求或达到 60 s 稳态；预热数据不进入统计。
6. calibration、screening 和 confirmatory runs 分目录保存，禁止把调参数据作为确认性结果。

### 3.3 重复次数与请求数

| 阶段 | 独立重复 | 每个 run 请求数 | 用途 |
|---|---:|---:|---|
| smoke/门禁 | 1 | 8–16 | 检查服务、协议和无崩溃 |
| TPC 离线校准 | 3 | 80–200 | 建 resource profile；取重复中位数 |
| 大网格 screening | 3 | 200 | 找主效应、OOM 边界、候选最优点 |
| 论文确认性主点 | 5 | ≥1000 | 稳定估计 throughput 与 P99 |
| accuracy | 完整测试集 | 全量 | paired task-level comparison |

只用 32 或 80 个请求无法稳定支持 P99 结论。现有 `results/innovation2_step1/B0_16k_c8.jsonl` 仅可视为 pilot，不进入论文确认性统计。

## 4. 正确性与任务准确率实验

### 4.1 数据集

主文至少使用两套、性质互补的数据集：

1. **GSM8K test**：短到中等上下文推理，指标为最终数值答案 exact match。
2. **LongBench v2**：长上下文理解，使用仓库现有 LongBench v2 evaluator 和固定截断/格式化策略，指标为官方 macro score。

可选补充：HumanEval pass@1，用于检查代码生成边界。ShareGPT 只作为服务工作负载，不作为“准确率”数据集。

### 4.2 正确性配置

所有正确性主实验使用 temperature=0、top-p=1、相同 max_new_tokens、相同 EOS 策略和相同 prompt token IDs。Target-only autoregressive decoding 是语义 oracle。

| ID | 配置 | 目的 |
|---|---|---|
| A0 | Target-only autoregressive SGLang | 权威语义 oracle |
| A1 | SPECTRE ordinary、GPU-resident KV | 匹配模型对的串行 speculative baseline |
| A2 | SPECTRE parallel、GPU-resident KV | 并行调度是否改变输出 |
| A3 | SpecStream Full-Restore | CPU tiering + 完整物化是否改变输出 |
| A4 | SpecStream bounded reference streaming | Torch FP32 online-softmax 参考路径 |
| A5 | SpecStream fused streaming | Triton/fused 路径 |
| A6 | 完整创新点一：A5 + dynamic q + cohort | 最终 I1 系统 |
| A7 | 完整 I1+I2 单卡共执行 | 资源控制是否影响语义 |

若要加入“原生 SGLang EAGLE”作为外部参考，必须单独列为 A-ext，不与 A1 做因果消融，因为其 drafter 架构、接受率与资源路径可能不同。

### 4.3 三层正确性证据

**层 1：attention 数值检查。** 在 A4/A5 启用 `--specstream-shadow-attention`，报告每层/每轮 `shadow_max_abs` 分布、logit margin 和 token mismatch。重点展示低 logit-margin 样本，不只报告平均误差。

**层 2：端到端 token parity。** 对每个请求保存 output token IDs，并相对 A0 计算：

- request-level exact token match；
- token-level match；
- first-divergence position；
- divergence 后任务答案是否仍正确。

确认性门槛：temperature=0 时 A1–A7 相对 A0 的 request-level exact match 必须为 100%。任何 divergence 都进入 first-divergence 诊断，不允许仅用任务分数相同掩盖。

**层 3：任务非劣效。** 对 GSM8K 和 LongBench v2 使用 paired bootstrap（10,000 次，按样本重采样）计算分数差的 95% CI。预设非劣界限为绝对 1.0 percentage point；只有 CI 下界大于 -1.0 pp 才称为非劣。二元正确/错误结果同时给出 McNemar 检验，但不能用 `P>0.05` 代替非劣证明。

## 5. 创新点一：KV 数据路径与控制层性能实验

### 5.1 主对照序列

| ID | 配置 | 关键 flags | 该组唯一回答的问题 |
|---|---|---|---|
| K0 | GPU-resident SPECTRE parallel | 不加 `--specstream-*` | 原始并行推测强基线 |
| K1 | CPU Full-Restore | `--specstream-enabled --specstream-full-restore-baseline` | offload 后每轮完整恢复的代价 |
| K2 | bounded reference streaming | `--specstream-enabled --specstream-reference-attention` | 有界 streaming 的语义/容量收益 |
| K3 | fused streaming，无跨层预取 | `--no-specstream-reference-attention --no-specstream-layer-prefetch` | fused/grouped 路径本身 |
| K4 | K3 + 双缓冲/跨层预取 | `--specstream-num-buffers 2 --specstream-chunks-per-transfer 4 --specstream-layer-prefetch` | 异步重叠增益 |
| K5 | K4 + dynamic horizon | `--specstream-dynamic-q --specstream-q-candidates 1,2,4,6,8` | q 自适应增益 |
| K6 | K5 + Chunk-Cohort | `--specstream-cohort-enabled --specstream-max-cohort-size 8` | 完整创新点一 |

K1 在当前代码中是“non-blocking copy API + 立即消费”的 Full-Restore。除非用 Nsight 证明传输被其他计算隐藏，否则正文只写 Full-Restore，不写“异步重叠优化全量加载”。

### 5.2 工作负载矩阵

**受控长上下文。** 使用同一 ShareGPT 文本池构造固定 token 长度：

- input: 4K、8K、16K、30K；
- output: 128；
- concurrency: 1、4、8、16、32；
- request rate: 0.5、1、2、4、8、inf；
- q fixed ablation: 1、2、4、6、8。

4K 主要验证未达到 `min_history_tokens=8192` 时的回退，不应与真正 streaming 点混算。16K 是主点，30K 是容量与 P99 stress point。

**真实服务分布。** ShareGPT V3 自然 prompt 长度、output=256、Poisson arrival，至少选择低负载、拐点负载和过载三档。

**Cohort 友好/不友好负载。**

- 同质：固定 16K、相同 q、c=8/16/32；
- shared-prefix-shaped：仅用于形状同步和 cohort 压力，不宣称 prefix-cache 收益；
- 异质：ShareGPT 自然长度，验证 cohort 兼容率低时是否自动退化。

### 5.3 机制实验

#### E1：多 Query History 复用

固定 16K/c1，q=1/2/4/6/8，比较 K1、K2、K4。每组报告：

- `h2d_bytes/round` 与 q 的关系；
- `h2d_ops/round`；
- `h2d MiB/accepted token`；
- accepted tokens/round；
- round time/useful token。

主张成立要求：K4 的物理 H2D bytes/round 不随 q 近似线性增长，且 H2D/useful-token 随 q 显著下降。使用线性回归只用于描述斜率和 CI，不用小样本 P 值包装。

#### E2：显存有界性

在 K0/K1/K4/K6 下扫描 4K–30K 和 c=1–32，报告：

- 总 GPU peak memory；
- `gpu_kv_bytes`、`staging_bytes`、模型权重/框架其他显存；
- CPU History bytes；
- OOM/成功边界。

主图使用 context × concurrency operating-region heatmap。`staging_bytes` 应近似常量，但进程总显存仍含权重、Tail、workspace 和 allocator reserve，不能把总显存直接等同 staging。

#### E3：重叠与实现消融

对 K2/K3/K4 做：

- buffers = 1/2/3；
- chunk tokens = 512/1024/2048/4096；
- chunks per transfer = 1/2/4/8；
- layer prefetch on/off。

主要指标：`exposed_copy_ms`、`copy_compute_overlap`、`staging_wait_ms`、H2D ops、stream-attn ops、round time。对 16K/c1、16K/c8、30K/c8 三个代表点使用 Nsight Systems 验证 CUDA memcpy、copy stream、attention kernel 和 layer overlap 时间线。

#### E4：dynamic q 与 ordinary/parallel

比较 fixed q=2/4/6/8、dynamic q 和离线 oracle。离线 oracle 在同一 workload 上从 fixed 配置中选取最佳，但不参与在线控制。

报告 q 分布、mode 分布、hysteresis hold、accepted progress、timeout/fallback 和相对 oracle gap。当前成本模型中 `max(draft,verify)` 总不高于 `draft+verify`，正常状态下 parallel 理论上支配 ordinary；因此若实验没有 mode 切换，应如实把贡献限定为 dynamic q + safety fallback。若论文要主张“负载驱动的串并行主动切换”，需先修改成本/策略使并行干扰、排队或等待能够让 ordinary 在某些状态下更优，再新增单元测试和确认性实验。

#### E5：Chunk-Cohort

在 K5/K6 下扫描 max cohort size=1/2/4/8。报告 cohort-size 分布、H2D ops、attention ops、round time、throughput 和 P99。

当前 cohort 把不同请求的 CPU chunks 打包后仍传输各自 bytes，因此主预期是减少 event/launch 和提高 fused execution 效率，而不是减少总 H2D bytes。`max_cohort_delay_us` 当前没有形成跨调度 batch 的真实 admission wait，不应把 delay sweep 作为主消融；若以后实现 admission waiting，再研究 0/50/100/200/500 μs 的吞吐–P99 权衡。

## 6. 创新点二：单卡 Draft–Verify 并行实验

### 6.1 主对照序列

| ID | GPU 数 | KV 路径 | Draft–Verify 调度 | 角色 |
|---|---:|---|---|---|
| D0 | 1 | GPU-resident | Target-only AR | 计算下界/语义参考 |
| D1 | 1 | GPU-resident | SPECTRE ordinary、同卡、无并行重叠 | 公平的单卡串行 speculative baseline |
| D2 | 2 | GPU-resident | SPECTRE parallel、Target/Draft 分卡 | 绝对吞吐强基线 |
| D3 | 1 | GPU-resident | SPECTRE parallel、同卡无控制 | 干扰负面对照 |
| D4 | 1 | GPU-resident | 最佳安全固定 TPC | 静态资源隔离对照 |
| D5 | 1 | GPU-resident | online target-priority TPC grant | 隔离创新点二，Target 用 `--specstream-profile-only` |
| D6 | 1 | CPU History bounded streaming | online target-priority TPC grant + dynamic q + cohort | 完整创新点一+二 |

MPS 只作为两个 CUDA 进程同卡并发的基础设施。90/10、80/20、70/30、60/40 的 MPS 百分比扫描可放补充材料作为 legacy baseline；当前最终实现的核心是经过 `validate-global` 的 TPC mask、离线实测 resource profile 和 one-token grant。

### 6.2 静态 TPC 校准

主校准形状：16K、batch=8，q=2/4/6/8；TPC=2/4/6/8/12；每点 3 次独立运行。另为论文主确认性 workload 补齐 bs=1/4/16 和 ctx=30K 的 profile，不允许对未测 shape 最近邻插值。

每个点同时测 wait-only Target baseline 与 overlap，计算：

\[
\mathrm{Target\ slowdown}=T_{\mathrm{target,overlap}}/T_{\mathrm{target,alone}}-1
\]

安全点门槛：Target slowdown ≤5%、error_count=0、grant 和 Draft step 数据非空。没有安全点即创新点二 No-Go，不能放宽为 10% 后继续宣称原目标成立。

### 6.3 在线 Target-priority gating

D5/D6 使用 step2 的实测 profile，检查：

- `TARGET_EXCLUSIVE`、`SLACK_FILL`、`DRAFT_CATCHUP` 状态占比；
- 每个 grant 恰好 1 token；ACK 到达前没有下一 grant；epoch 单调；
- grant TPC 范围来自匹配 profile；
- 在线 Target slowdown 超预算后 latch 到 exclusive；
- 未校准 shape、过期 grant 和错误 profile fail closed；
- Drafter STOP/CONT、timeout 和 overload 时 Target 继续服务并 fallback。

### 6.4 端到端性能与资源效率

在 D1–D6 上运行相同 16K/30K、c=1/4/8/16/32 和 Poisson rate sweep。主要比较：

1. D2 vs D5：两卡绝对吞吐与单卡资源效率的权衡；
2. D3 vs D4 vs D5：无控制、最佳固定安全点和在线 gating；
3. D5 vs D6：KV streaming 是否扩大单卡可运行区域；
4. D1 vs D5/D6：单卡内 parallel overlap 是否相对单卡 serial speculative 有真实收益。

不能只报告 tokens/s/GPU。D2 使用两张物理 GPU，应同时给 raw throughput、GPU 数、throughput/GPU、SLO goodput/GPU、P99 和能耗/有效 token（若 NVML 数据完整）。

## 7. 指标定义与统计分析

### 7.1 服务指标

- request throughput：completed requests / wall time；
- output throughput：生成 output tokens / wall time；
- TTFT：请求到首 token；
- TPOT：`(E2E - TTFT)/(output_tokens-1)`；
- ITL：相邻流式 token 间隔，报告 median、P95、P99；
- SLO goodput：同时满足预注册 TTFT 和 TPOT 阈值的请求数 / wall time；
- Goodput/GPU：SLO goodput / 物理 GPU 数。

SLO 阈值在 pilot 后、正式主结果前冻结。建议主文固定一组服务目标，补充材料提供 TTFT × TPOT SLO 网格，避免只挑最有利阈值。

### 7.2 SpecStream 机制指标

- H2D bytes/round；
- H2D ops/round；
- H2D MiB/accepted token；
- copy floor、exposed copy、copy–compute overlap；
- staging wait、stream attention time/ops；
- accepted tokens/round、accept length、rollback ratio；
- cohort size 与 cohort hit ratio；
- GPU KV、staging、CPU History 和总 GPU peak memory。

### 7.3 共执行指标

- Target slowdown；
- Draft step latency 与 Draft RTT P95；
- grant wait、TPC range、grant state distribution；
- timeout、missing draft、fallback 和 reject rate；
- Target/Draft GPU utilization、HBM bandwidth proxy、power；
- 物理 GPU 数和资源成本。

### 7.4 统计方法

1. 性能主结果以 5 个独立 run 为统计单位，报告中位数、IQR 和 cluster bootstrap 95% CI；请求不能被当作完全独立的硬件重复。
2. 配对配置使用同一请求顺序和 seed，报告 paired difference/ratio 及 CI。
3. 多配置大网格不做大量逐格显著性检验；主文只对预注册主要终点做确认性比较，其余报告效应量和 CI。
4. P99 每个确认性 run 至少 1000 个成功请求；error/timeout 请求单独计数，不能从延迟分布中静默删除。
5. OOM、crash、timeout、fallback 均为结果，不用减少并发后替换原点。

## 8. Nature 风格主图与表格设计

所有图以一个明确结论为中心，方法颜色跨图固定：Target-only 为灰，GPU-resident SPECTRE 为深紫，Full-Restore 为浅蓝，SpecStream I1 为玫瑰，I2 为青绿，I1+I2 为深蓝。红色只表示超预算、错误或下降。最终作图前再锁定 Python/R 后端；主格式为可编辑 SVG/PDF，PNG 300 dpi 仅用于预览。

### Figure 1 | SpecStream system and evidence chain

- **核心结论：** 两项创新分别降低 KV residency tax 和 dedicated-drafter tax，并在组合后扩大单卡可运行区域。
- **archetype：** schematic-led composite。
- **hero panel a：** History/Tail/Frontier + bounded streaming + Target-priority grants 的统一时间线。
- **support b：** baseline/ablation 路径图；**c：** claim–evidence map。

### Figure 2 | SpecStream preserves target-authoritative decoding

- **核心结论：** A1–A7 在 greedy decoding 下保持 token parity，并在两数据集上任务分数非劣。
- **hero a：** paired task-score difference forest plot，显示 95% CI 和 -1 pp 非劣界限。
- **b：** request token exact-match；**c：** shadow max error vs logit margin；**d：** first-divergence 计数。

### Figure 3 | Verification-native streaming amortizes History transfer

- **hero a：** q vs H2D MiB/accepted token。
- **b：** q vs H2D bytes/round；**c：** context vs staging/GPU KV memory；**d：** Nsight copy/compute 时间线。
- 图注写明 n、误差定义、硬件、chunk/buffer 参数和 Source Data。

### Figure 4 | Each component expands the long-context serving region

- **hero a：** context × concurrency operating-region heatmap（K0–K6 small multiples）。
- **b：** K0–K6 SLO goodput；**c：** P99 TPOT；**d：** dynamic q 与 cohort size 分布。
- 不把 unrelated metrics 塞进一个柱状图；每个 panel 回答一个问题。

### Figure 5 | Target-priority TPC grants define a safe co-execution Pareto frontier

- **hero a：** Target slowdown vs Draft step latency，点按 TPC/q 编码，标出 5% slowdown 线和 Pareto region。
- **b：** q×TPC heatmap；**c：** online grant state 时间线；**d：** failure injection 的服务存活/回退。

### Figure 6 | SpecStream removes the dedicated-drafter GPU tax

- **hero a：** raw throughput vs physical GPU count，点大小表示 P99 或 goodput。
- **b：** Goodput/GPU；**c：** D1–D6 P99；**d：** D5 vs D6 可运行区域。
- 明确展示两卡 D2 的绝对吞吐优势与单卡 D5/D6 的资源效率优势，不隐藏 trade-off。

### 主表

- **Table 1：** 模型、硬件、软件和 workload；
- **Table 2：** A0–A7 correctness 与 paired non-inferiority；
- **Table 3：** K0–K6 16K/c8 和 30K/c8 主性能；
- **Table 4：** D1–D6 的 GPU 数、raw throughput、Goodput/GPU、P99、slowdown、peak memory；
- **Extended Data：** 完整 context/rate/concurrency 网格、ablation、负结果、resource profile 与 failure injection。

表格采用 booktabs，无竖线；列头标明单位和指标方向；同列精度一致；最好值只做克制强调。

## 9. 执行顺序与 Go/No-Go

### Gate 0：代码与协议

- 全部 `python/sglang/test/spectre_specstream` 测试通过；
- C++ ZMQ schema=2；grant/pause/grant_ack 可往返；
- `validate-global` 在实际 A800 和实际 TPC 范围通过；
- 正确导入当前工作树，不是旧安装包。

### Gate 1：正确性

- temperature=0 token parity 100%；
- shadow mismatch 可解释且无 token mismatch；
- GSM8K、LongBench v2 非劣 CI 下界 > -1 pp。

失败则停止性能主张，先做 first-divergence 修复。

### Gate 2：创新点一机制

- staging 有界；
- H2D/round 不按 q 线性增长；
- 至少一个 16K/30K workload 的 H2D/useful-token、可运行区域或 Goodput 有实质改善。

### Gate 3：创新点二可行性

- 至少一个 q/TPC 点 Target slowdown ≤5%，且 Draft 有有效进展；
- 未校准 shape fail closed；
- online run 不崩溃、无死锁、error_count=0。

若无安全点，创新点二收敛为负结果或适用边界，不进入动态策略包装。

### Gate 4：完整系统

- D6 相比 D3 稳定提高 Goodput/GPU 或降低 P99；
- D6 相比 D5 扩大可运行 context/concurrency；
- 与两卡 D2 的 raw throughput 差距和节省的 GPU 数同时公开。

## 10. 结果目录与 run manifest

每个 run 使用不可变 ID：

```text
{stage}_{variant}_{dataset}_ctx{K}_out{N}_rate{R}_c{C}_q{Q}_rep{n}
```

保存：

```text
results/nature_i1_i2/
  manifest.csv
  accuracy/
  benchmark/
  profiles/
  nsys/
  nvml/
  logs/
  summaries/
  source_data/
```

`manifest.csv` 至少包含 commit、dirty diff hash、模型 hash、dataset hash、GPU UUID、driver/CUDA/PyTorch、全部 launch args、seed、开始/结束时间、退出码、成功/错误请求数和输出文件 SHA256。任何图中的点都必须能反查到原始 JSONL/CSV 和日志。

## 11. 当前代码审计后必须正面处理的四个问题

1. **Full-Restore 不是独立的“异步重叠全量加载”实现。** 当前路径整体创建 host tensor、发起 non-blocking H2D 后立即用于 attention。实验可把它作为 Full-Restore baseline，但不能预先命名为 overlap-optimized full load。
2. **dynamic ordinary/parallel 在正常成本模型下可能退化。** parallel 成本用 `max(draft, verify)`，ordinary 用二者相加，因此若无额外干扰/排队惩罚，ordinary 很难被主动选择。主张应先限为 dynamic q + safety fallback，或先完善策略。
3. **Chunk-Cohort 不做跨请求 KV 内容去重，也没有真正跨 batch 等待。** 它主要减少兼容请求的 launch/event 开销。主指标应是 ops、round time 和 P99，而非总 H2D bytes 下降；`max_cohort_delay_us` 暂不作为主效应变量。
4. **创新点二最终实现不是 MPS 百分比控制。** `SingleGPUCoexecPolicy` 已明确退役；论文主线应使用实测 resource profile + Target-priority one-token TPC grant。MPS 百分比只保留为基础设施/legacy baseline。

解决以上命名和机制边界后，整套实验可以形成清晰的 Nature 风格因果链：先证明 exactness，再证明 I/O 与显存机制，再证明单卡资源 Pareto，最后证明组合系统在相同 GPU 预算下扩大长上下文 serving 区域。
