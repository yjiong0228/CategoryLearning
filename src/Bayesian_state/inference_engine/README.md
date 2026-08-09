# inference_engine

本目录统一管理 Bayesian 状态模型的执行与隐状态推理。它包含单份认知状态的 `BaseEngine`，
也包含决定“运行一条状态轨迹”还是“用粒子对多条轨迹积分”的 backend 层。具体 transition、
memory 或 likelihood 仍由 `problems/modules/` 注入。

## 文件

| 文件 | 职责 |
|---|---|
| `bayesian_engine.py` | `BaseSet`、概率容器、`BaseEngine`、trial agenda 和快照协议 |
| `dispatcher.py` | 解析 `engine_config.inference` 并选择 backend |
| `results.py` | 所有 backend 共享的 `InferenceResult` 契约与兼容属性 |
| `backends/trajectory.py` | 单条随机认知轨迹 |
| `backends/particle_filter.py` | bootstrap particle filter、ESS、重采样和粒子边际 |
| `posterior_predictive.py` | 观察前缀条件下的粒子后验预测与自主 suffix rollout |
| `__init__.py` | 重导出稳定的公共推理接口 |

`problems/base_problem.py` 也会重导出这些类，用于兼容旧 import；真正定义位于本目录。

## Backend dispatch

配置接口保持不变：

```yaml
inference:
  backend: particle_filter   # 或 trajectory
  particle_count: 512
  resample_threshold_fraction: 0.5
```

`resolve_inference_backend()` 负责规范化和验证配置，`run_inference_backend()` 负责执行。优化器
只消费 backend 输出并计算 metrics/loss，不再包含 particle-filter 实现。

两种 backend 的区别是：

- `trajectory`：条件于一组已实现的 perception/transition 随机变量，只保留一条 latent path。
- `particle_filter`：维护多份完整 `StateModel`，用 observed choice 更新权重，对 latent paths 求边际。

两者调用同一套 `StateModel + BaseEngine + modules`，不是两套认知模型。

两者也返回同一个结果契约。公共字段分为 `observation_probabilities`,
`state_probabilities`, `latent_summaries`, `diagnostics`, `artifacts` 和 `metadata`；旧的
`TrajectoryInferenceResult`/`ParticleFilterResult` 名称保留为兼容构造器。

## BaseEngine 的核心状态

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

`StateModel` 创建 engine 后调用：

```python
engine.build_modules(engine_config["modules"])
```

正式路径的每个 trial 调用：

```python
prepared = model.begin_trial(stimulus)
prediction = model.predict_choice(...)
posterior, prior_snapshot, log = model.complete_trial(choice, feedback)
```

其中 `begin_trial()` 只运行 perception/transition；`complete_trial()` 在真实或模型生成的
choice/feedback 已经出现后运行 likelihood/memory/beta。`fit_step_by_step()` 与自主
`generate_step_by_step()` 共享这套生命周期。

`engine.infer_single(observation)` 仍作为完整 observation 的兼容入口，它会：

1. 将上一 posterior 复制为新 prior；
2. 设置 observation；
3. 按 `agenda` 调用 module `process()`；
4. 返回 posterior、transition 后的 prior snapshot 和轻量 step log。

module 通过共享 engine 字段通信，不应互相维护重复的全局状态。

## 粒子快照与重采样

`state_dict()` 保存 engine 核心字段及每个 module 的状态 payload；`load_state_dict()` 用于
恢复 particle ancestor。`clear_module_logs()` 清除复制后不应继承的轨迹日志。

快照接口是认知状态协议，不是长期磁盘序列化格式。不要假定其 payload 跨代码版本稳定。

粒子滤波公共入口为：

```python
from src.Bayesian_state.inference_engine.backends.particle_filter import (
    run_state_model_particle_filter,
)
```

当前正式入口支持 condition 1、expectation 类 readout 和 uniform output lapse。条件 posterior
predictive 由 `posterior_predictive.py` 组合粒子状态与自主生成过程，不属于 optimizer。

## 扩展边界

- 新认知机制应实现为 `BaseModule`，放在 `problems/modules/`。
- 新的 filter/smoother 应放在 `backends/`，通过快照协议操作 engine。
- backend 只返回推理状态、预测概率和诊断；loss、模型选择与结果写盘属于 `optimization/`。
- 不要在 engine 中硬编码某个模型名、被试数据路径、评价指标或绘图逻辑。
