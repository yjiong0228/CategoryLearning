# 优化

本目录是 `Bayesian_state` 的参数搜索与模型选择层。它调用 `simulation/`，把候选
hyperparameters 对应的 `SingleRunResult`/`SimulationResult` 组成可比较的 objective，但不拥有
固定参数运行时、结果契约、指标公式或具体认知机制。
凡是会根据数据表现改变下一轮参数、候选权重或冻结配置的操作属于本目录；冻结模型的 PPC、
留出验证和外部通道解释属于 `evaluation/`。

## 文件说明

| 文件 | 职责 |
|---|---|
| `candidates.py` | condition-1 单机制候选空间与 engine-config 参数注入 |
| `objectives.py` | 有容差和 anchor guard 的有序多目标比较 |
| `artifacts.py` | 候选展开、结果 schema、provenance、compact/full artifact 构建 |
| `cli.py` | grid/CD 统一 CLI |
| `search/common.py` | Grid/CD 共用运行时、配置解析、候选注入和 JSONL I/O |
| `search/grid.py` | 显式 joint grid 搜索 |
| `search/coordinate_descent.py` | coarse/fine coordinate descent、多 restart 与 trace |
| `diagnostics/search.py` | 只读取既有搜索产物的收敛、plateau 和 selection diagnostics |
| `diagnostics/predictive.py` | 需要重新运行候选模型的 accuracy sampling 与 volatility diagnostics |

## 标准数据对象

这些对象的唯一实现分别位于 `simulation/data.py` 与 `simulation/results.py`。`TrialArrays`：

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
- 可选的 state/trial/transition 日志

`SimulationResult` 聚合多个 `SingleRunResult`，并保存 sample errors、representative run 和
`compute_simulation_statistics()` 的结构化结果。

## 预测模式

- `posterior_t_minus_1`：以上一 trial feedback 后 posterior 预测当前 trial。
- `prior_t`：使用当前 trial transition 后、choice 前 prior。
- `both`：同时计算两者；必须指定用于模型选择的 mode。

动态 transition 和粒子滤波的因果预测应使用 `prior_t`。

## 损失函数

公共 loss 包括：

- 准确率曲线 MAE/MSE/BerHu
- accuracy/family Brier 与 NLL
- 选择 Brier/NLL
- wrong-choice 与 conditional-wrong-choice NLL
- 目标概率 Brier

loss strategy 与全部 loss 数值定义位于 `metrics/losses.py`，并只读取标准 metrics mapping。
Brier、NLL、CRPS、曲线和行为统计等其他纯数值定义也位于 `metrics/`；本目录只负责构造模型
预测、调用共享指标，以及规定如何把这些数值组成 selection objective、容差和 anchor guard。
增加新后端时，应把后端输出转换成公共 prediction/metrics contract，而不是在 optimizer 中另写
一套评分。

## 随机种子层级

种子由 `utils/seeding.py` 通过稳定 hash 分层派生：

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

Grid/CD 都继承 `HyperSearchBase`。外部 workflow 运行单被试时调用
`optimizer.run_subject(subject_id, stage=...)`，批量运行调用 `optimizer.run(subjects, stage=...)`；
`_run_subject_pipeline()` 是后端内部实现，不是 orchestration API。

若基础 simulation config 声明 `evaluation_protocol.mode: sequential_holdout`，Grid 和 Hyper-CD
统一以 `optimization` 角色解析评分掩码。主 loss、边际预测、accuracy shape、history kernel、
switch behavior 与 distribution objectives 都从带该掩码的公共 metrics mapping 计算；完整序列
只用于保持在线状态递推，不允许留出后缀进入候选比较。每个候选的结构化结果保存 `scoring`
上下文，便于审计实际切分。

## 与推理后端的边界

model structure 中设置：

```yaml
inference:
  backend: particle_filter
  particle_count: 512
  resample_threshold_fraction: 0.5
```

`evaluate_state_model_run()` 调用 `inference.dispatcher`，再将 trajectory 或 particle
输出转换成公共 metrics/loss。粒子实现位于
`inference/backends/particle_filter.py`；optimization 不维护另一份推理算法。

choice/output-noise 的实现也不再位于 optimizer；统一从
`model/readout.py` 调用。RT/oral readout 已有状态到测量分布的接口，但尚未加入
当前 choice-only backend 的观测输入和 loss。

当前限制：condition 1、expectation 类 readout、uniform base lapse；RT emission 尚未接入。

## 入口与输出

```bash
python -m src.Bayesian_state.optimization.cli --backend cd --config <yaml>
python -m src.Bayesian_state.optimization.cli --backend grid --config <yaml>
```

顶层 workflow 与结果序列化由：

- `src.Bayesian_state.run_hyper_then_simulation`
- `src.Bayesian_state.run_simulation`
- `src.Bayesian_state.run_hyper_evaluation`

负责。optimizer 本身应保持可由测试和 notebook 直接调用。

`diagnostics/search.py` 不启动模型，适合快速检查搜索产物；
`diagnostics/predictive.py` 会重新采样或运行模型，应显式控制 repeats、subjects 和 `n_jobs`。
