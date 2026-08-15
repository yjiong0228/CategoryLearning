# 推理

本目录只管理已装配 Bayesian 状态模型的隐状态推理：决定“运行一条状态轨迹”还是“用粒子对
多条轨迹积分”，并返回统一结果契约。`BayesianStateEngine`、perception、transition、memory 与
beta 都属于 `model/`；固定 likelihood evaluator 来自 `hypothesis_space/observation_model/`。

## 文件

| 文件 | 职责 |
|---|---|
| `dispatcher.py` | 解析 `engine_config.inference` 并选择 backend |
| `results.py` | 所有 backend 共享的 `InferenceResult` 契约与兼容属性 |
| `backends/trajectory.py` | 单条随机认知轨迹 |
| `backends/particle_filter.py` | bootstrap particle filter、ESS、重采样和粒子边际 |
| `posterior_predictive.py` | 观察前缀条件下的粒子后验预测与自主 suffix rollout |
| `__init__.py` | 重导出稳定的公共推理接口 |

依赖方向固定为 `inference -> model`。模型层不导入本包，因此无需局部导入来掩盖循环依赖。

## 推理后端分派

配置接口保持不变：

```yaml
inference:
  backend: particle_filter   # 或 trajectory
  particle_count: 512
  resample_threshold_fraction: 0.5
```

评价层需要定位 choice-transmission 瓶颈时，可在复制出的诊断配置中临时设置
`choice_transmission_audit: true`。PF 会从相同 pre-choice 粒子状态旁路计算 MAP hypothesis、
exploration-adaptive sharpening、exploration-gated choice uncertainty 和粒子预测分位数；正式
particle weights 仍只由原 fitted readout 更新。审计还记录每次重采样的父粒子索引，并从最终
posterior-weighted 粒子向前回溯完整的 prediction/strategy 祖先路径。该开关默认关闭，不应加入
正式拟合配置。

当 persistent execution 已启用时，审计同时返回执行规则在 strategy confidence 变换前后的
choice probability。两者共享完全相同的 pre-choice 粒子状态和权重，可用于估计纯读出层的即时
配对贡献；由于替代读出没有反过来更新后续粒子权重，这仍是条件分解，不是完整反事实模型拟合。

`resolve_inference_backend()` 负责规范化和验证配置，`run_inference_backend()` 负责执行。优化器
只消费 backend 输出并计算 metrics/loss，不再包含 particle-filter 实现。

两种 backend 的区别是：

- `trajectory`：条件于一组已实现的 perception/transition 随机变量，只保留一条 latent path。
- `particle_filter`：维护多份完整 `StateModel`，用 observed choice 更新权重，对 latent paths 求边际。

两者调用同一套 `StateModel + BayesianStateEngine + modules`，不是两套认知模型。

两者也返回同一个结果契约。公共字段分为 `observation_probabilities`,
`state_probabilities`, `latent_summaries`, `diagnostics`, `artifacts` 和 `metadata`；旧的
`TrajectoryInferenceResult`/`ParticleFilterResult` 名称保留为兼容构造器。

## 推理后端使用的模型状态

| 字段 | 含义 |
|---|---|
| `observation` | 当前 `(stimulus, choice, feedback)` |
| `prior` | 当前 trial choice 前的 hypothesis distribution |
| `likelihood` | 当前 observation 对各 hypothesis 的 likelihood |
| `posterior` | 当前 trial feedback 后的 hypothesis distribution |
| `hypotheses_mask` | 当前 active hypotheses 的 0/1 mask |
| `beta` | hypothesis-specific inverse temperature |
| `partition` | hypothesis geometry 与 category probability provider |
| `modules` | 已实例化 module mapping |
| `agenda` | trial 内 module 的调用顺序 |

## 生命周期

`model/assembly.py` 创建 engine、实例化配置中的 modules，然后逐个调用
`engine.register_module()`，最后用 `engine.validate_agenda()` 检查调度表。

```python
engine.register_module(name, module)
engine.validate_agenda()
```

正式路径的每个 trial 调用：

```python
prepared = model.begin_trial(stimulus)
prediction = model.predict_choice(...)
posterior, prior_snapshot, log = model.complete_trial(choice, feedback)
```

其中 `begin_trial()` 只运行 perception/transition；`complete_trial()` 在真实或模型生成的
choice/feedback 已经出现后，先固定计算 likelihood，再运行 agenda 中的 memory/beta。
Likelihood 由 `BayesianStateEngine.compute_likelihood()` 调用，不是可省略的 module。模块阶段由
`ModulePhase` 声明，不通过配置名称推断。`fit_step_by_step()` 与自主
`generate_step_by_step()` 共享这套生命周期。

不存在绕过上述生命周期的 `engine.infer_single()` 快捷路径。module 通过共享 engine 字段通信，
不应互相维护重复的全局状态。

## 粒子快照与重采样

`state_dict()` 保存 engine 核心字段及每个 module 的状态 payload；`load_state_dict()` 用于
恢复 particle ancestor。`clear_module_logs()` 清除复制后不应继承的轨迹日志。

快照接口是认知状态协议，不是长期磁盘序列化格式。不要假定其 payload 跨代码版本稳定。

dynamic-continuous persistent execution 开启时，快照还保存 executed hypothesis、dwell/switch
计数和独立 execution RNG。重采样复制这些认知状态，再为子粒子的未来 transition/execution
随机流重新设种；因此 overt strategy persistence 属于粒子状态，不是 filter 外部的绘图平滑。

粒子滤波公共入口为：

```python
from src.Bayesian_state.inference.backends.particle_filter import (
    run_state_model_particle_filter,
)
```

当前正式入口支持 condition 1、expectation 类 readout 和 uniform output lapse。条件 posterior
predictive 由 `posterior_predictive.py` 组合粒子状态与自主生成过程，不属于 optimizer。

机制审计可直接调用 PF 公共函数，用
`condition_on_observed_choice=false` 得到不做外层 choice importance weighting 的均匀轨迹
混合，并用 `resample_threshold_fraction=0` 明确关闭重采样。两项都是推断分解用的分析对照；
正式 dispatcher 配置仍要求 choice-conditioned filtering 和 `(0, 1]` 内的重采样阈值，默认行为
没有改变。

PF transition 诊断区分两种时间语义：已有 `transition_rate`、`search_range`、
`swap_probability` 等字段是观察当前 choice 后的 filtered 边缘量；用于解释 trial `t` 所采取策略的
`predictive_*` 字段则在当前 choice 更新粒子权重之前计算。连续策略进一步把每个粒子的理论
swap 概率分解为 `predictive_strategy_exploit`、`predictive_strategy_local_explore` 和
`predictive_strategy_global_explore`，随后用 pre-choice 权重求边缘；三者逐 trial 加和为 1。

persistent execution 还返回 pre-choice `executed_probability`、post-choice
`filtered_executed_probability`、`predictive_execution_switch_probability`、
`predictive_execution_switch_event_probability`、`predictive_execution_dwell_trials`、
`predictive_executed_beta` 和 `filtered_executed_beta`。
choice-transmission audit 的 ancestral paths 可直接检查 executed rule 的驻留和切换。

PF 是否“粒子足够”不能只看单次运行是否完成。`scripts/run_model_0815_p0_pf_convergence.py`
用独立 filter seeds 比较相邻 particle counts，同时检查 probability-averaged choice NLL、逐 trial
choice-probability RMSE、executed-rule posterior 的 Jensen--Shannon divergence、repeat split-half
稳定性和 post-choice ESS fraction。只有预先声明的门槛同时通过，较小 particle count 才可作为
正式预算；smoke 模式只验证工作流，不构成收敛证据。

## 扩展边界

- 新认知机制应实现为 `BaseModule`，放在 `model/modules/`。
- 新的 filter/smoother 应放在 `backends/`，通过快照协议操作 engine。
- backend 只返回推理状态、预测概率和诊断；loss、模型选择与结果写盘属于 `optimization/`。
- 不要在 engine 中硬编码某个模型名、被试数据路径、评价指标或绘图逻辑。
