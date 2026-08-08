# optimization

本目录是 `Bayesian_state` 的统一执行与模型选择层。它负责把 engine config、被试 trial 数据和
hyperparameters 转换成可比较的 `SingleRunResult`/`SimulationResult`，但不定义具体认知机制。
凡是会根据数据表现改变下一轮参数、候选权重或冻结配置的操作属于本目录；冻结模型的 PPC、
留出验证和外部通道解释属于 `model_evaluation/`。

## 文件说明

| 文件 | 职责 |
|---|---|
| `optimizer_common.py` | trial arrays、模型概率构造、metrics 调用和单次 StateModel 评价适配 |
| `optimizer_simulation.py` | 独立重复运行、representative run 选择和跨重复聚合 |
| `mechanism_candidates.py` | condition-1 单机制候选空间与 engine-config 参数注入 |
| `optimization_config.py` | YAML、subjects、prediction/loss、相对路径和 stream reference 解析 |
| `hyper_objectives.py` | 有容差和 anchor guard 的有序多目标比较 |
| `hyper_utils.py` | 候选展开、结果 schema、provenance、compact/full artifact 构建 |
| `hyper_grid_optimizer.py` | 显式 joint grid 搜索 |
| `hyper_cd_optimizer.py` | coarse/fine coordinate descent、多 restart 与 trace |
| `hyper_cli.py` | grid/CD 统一 CLI |
| `hyper_evaluation.py` | 已完成 hyper search 的收敛、plateau 和 selection diagnostics |

## 标准数据对象

`TrialArrays`：

```python
TrialArrays(
    stimulus=...,
    choices=...,
    feedback=...,
    categories=...,     # optional diagnostic truth
    target_probs=...,   # optional probabilistic target
)
```

`SingleRunResult` 保存一条 trajectory 或一次 particle marginal run 的：

- `metrics_by_mode`
- `mean_error`
- seeds 与 params
- optional state/trial/transition logs

`SimulationResult` 聚合多个 `SingleRunResult`，并保存 sample errors、representative run 和
`compute_simulation_statistics()` 的结构化结果。

## prediction mode

- `posterior_t_minus_1`：以上一 trial feedback 后 posterior 预测当前 trial。
- `prior_t`：使用当前 trial transition 后、choice 前 prior。
- `both`：同时计算两者；必须指定用于模型选择的 mode。

动态 transition 和粒子滤波的因果预测应使用 `prior_t`。

## loss

公共 loss 包括：

- accuracy curve MAE/MSE/BerHu
- accuracy/family Brier 与 NLL
- choice Brier/NLL
- wrong-choice 与 conditional-wrong-choice NLL
- target-probability Brier

loss strategy 与全部 loss 数值定义位于 `metrics/losses.py`，并只读取标准 metrics mapping。
Brier、NLL、CRPS、曲线和行为统计等其他纯数值定义也位于 `metrics/`；本目录只负责构造模型
预测、调用共享指标，以及规定如何把这些数值组成 selection objective、容差和 anchor guard。
增加新后端时，应把后端输出转换成公共 prediction/metrics contract，而不是在 optimizer 中另写
一套评分。

## 随机种子层级

种子由 `utils/seeding.py` 通过稳定 hash 分层派生；`optimizer_common.py` 保留同名重导出以兼容
已有调用：

```text
hyper_base_seed
  → hyper_candidate_seed
      → simulation_point_seed
          → trajectory/filter seed
              → module-specific seed
```

同一配置、被试、参数点和 repeat index 应产生相同 seed；不要依赖 Python 进程内置的随机 hash。

## Hyper-CD 与 Grid

Hyperparameter key 必须以以下前缀开头：

```text
engine.modules.memory_mod.kwargs.gamma
engine.choice_readout.kwargs
simulation.window_size
```

mapping-valued coordinate 会整体替换目标 mapping，适合把一个策略 profile 或 controller 当成
不可拆分候选。不要同时声明父路径和其子路径；`validate_no_nested_hyperparam_paths()` 会拒绝
这种歧义。

Hyper-CD 的 `objective_order` 按顺序比较目标；只有前一目标落入容差集合时，后一目标才用于
区分。coarse/fine stage 可分别覆盖 simulation repeats、particle count 或日志预算。

## 与推理后端的边界

model structure 中设置：

```yaml
inference:
  backend: particle_filter
  particle_count: 512
  resample_threshold_fraction: 0.5
```

`evaluate_state_model_run()` 调用 `inference_engine.dispatcher`，再将 trajectory 或 particle
输出转换成公共 metrics/loss。粒子实现位于
`inference_engine/backends/particle_filter.py`；optimization 不维护另一份推理算法。

choice/output-noise 的实现也不再位于 optimizer；统一从
`problems/modules/readout.py` 调用。RT/oral readout 已有状态到测量分布的接口，但尚未加入
当前 choice-only backend 的观测输入和 loss。

当前限制：condition 1、expectation 类 readout、uniform base lapse；RT emission 尚未接入。

## 入口与输出

```bash
python -m src.Bayesian_state.optimization.hyper_cli --backend cd --config <yaml>
python -m src.Bayesian_state.optimization.hyper_cli --backend grid --config <yaml>
```

顶层 workflow 与结果序列化由：

- `src.Bayesian_state.run_hyper_then_simulation`
- `src.Bayesian_state.run_simulation`
- `src.Bayesian_state.run_hyper_evaluation`

负责。optimizer 本身应保持可由测试和 notebook 直接调用。
