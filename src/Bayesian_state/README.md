# Bayesian_state 模型框架

`Bayesian_state` 是本项目的试次级 Bayesian 状态模型包。它把“模型结构”“逐试次推理”
“潜在路径积分”“超参数搜索”“重复仿真”和“结果评价”分成相互独立的层，而不是把一个模型
写成一份从数据读取到画图的独立脚本。

当前正式建模路径以 `StateModel + BayesianStateEngine + modules` 为核心。`model_0806` 的动态连续
hypothesis-transition 模型也已经接入这条路径；`reference_models/` 中的同名实现保留作
恢复实验和数值参考，不是正式拟合入口。

## 1. 总体结构

核心包结构固定为：

```text
Bayesian_state/
├── model/                StateModel、BayesianStateEngine 与可插拔认知机制
├── hypothesis_space/     假设目录、几何、partition 与 likelihood
├── inference/            推理后端、后端分派与结果契约
├── simulation/           单次/重复运行与自主行为生成
├── optimization/         参数搜索与模型选择
├── evaluation/           已完成结果的统计与诊断
├── metrics/              各执行层共享的纯数值指标
└── reference_models/     冻结的论文数值实现与复现基准
```

此外，`utils/` 保存跨层基础工具，FFT 聚类归入 `evaluation/`，根目录的
`run_*.py` 只负责 orchestration；它们不改变上述八个核心职责边界。

```text
configs/*.yaml
    │
    ▼
run_* entrypoints
    │
    ├── metrics/             共享 proper scores、曲线、行为和跨重复统计
    ├── simulation/          观察数据执行、独立重复聚合与模型自主行为生成
    ├── optimization/        候选搜索、objective、Hyper-CD/Grid
    │
    ▼
inference/dispatcher
    │
    ├── trajectory backend
    └── particle-filter backend
    │
    ▼
model/StateModel → model/BayesianStateEngine
    │
    ▼
hypothesis_space/observation_model     fixed observation-likelihood evaluation
    │
    ▼
model/modules/                perception / transition / memory / beta
    │
    ▼
results/subjects/*.json + optional compressed run streams
    │
    ├── evaluation/         指标、PPC、状态轨迹、口述规则对齐
    └── evaluation/fft_clustering.py  run-level 轨迹的事后聚类
```

依赖方向原则是：执行与分析层可以调用模型层，模型层不应反向调用具体 CLI、结果目录或绘图
脚本。共享路径、数据解析和小型统计函数放在 `utils/`。

## 2. 目录职责

| 目录 | 职责 | 是否属于正式运行主路径 |
|---|---|---|
| `inference/` | backend dispatch、单轨迹/粒子推理与公共结果契约 | 是 |
| `hypothesis_space/` | 连续/离散假设目录、geometry、partition 与固定 observation-likelihood evaluator | 是 |
| `model/` | `StateModel`、状态 engine、trial scheduler 与认知 modules；不包含 hypothesis 实现 | 是 |
| `model/modules/` | 感知、transition、memory、beta 等可插拔认知机制 | 是 |
| `metrics/` | optimization、simulation 与 evaluation 共享的纯数值指标 | 是，底层支持 |
| `simulation/` | trial/run 结果、单次/重复固定参数运行、统计 schema 和自主行为生成 | 是 |
| `optimization/` | 候选参数、objective、机制候选与 Hyper-CD/Grid | 是 |
| `evaluation/` | 已完成仿真的统计、作图和 oral/model alignment | 是，属于后处理 |
| `reference_models/` | 论文阶段的独立数值实现、恢复和 oracle | 否；参考与复现用途 |
| `utils/` | 路径、数据集、subject override、stream、公共统计 | 是，底层支持 |

根目录的可执行文件只负责 orchestration：

| 文件 | 作用 |
|---|---|
| `run_simulation.py` | 固定参数、逐被试重复 simulation 并序列化；也提供公开 `run_simulation()` API |
| `run_hyper_then_simulation.py` | Hyper 搜索、生成 subjectwise simulation YAML、再通过公开 API 运行 simulation |
| `run_hyper_evaluation.py` | 已完成 Hyper-CD 输出的收敛和选择诊断 |
| `run_model_evaluation.py` | 已完成 simulation 输出的统一后处理 |
| `__init__.py` | package marker；公共对象应从职责明确的子包显式导入 |

更详细的说明见各目录 README：

- [`inference/README.md`](inference/README.md)
- [`metrics/README.md`](metrics/README.md)
- [`hypothesis_space/README.md`](hypothesis_space/README.md)
- [`model/README.md`](model/README.md)
- [`model/modules/README.md`](model/modules/README.md)
- [`optimization/README.md`](optimization/README.md)
- [`simulation/README.md`](simulation/README.md)
- [`evaluation/README.md`](evaluation/README.md)
- [`reference_models/README.md`](reference_models/README.md)
- [`utils/README.md`](utils/README.md)

根目录的 `PMH modules.svg` 是较早期的 module-loop 示意图，可用于理解黑板式调度，但它
早于当前的 inference backend 分层；目录边界和正式入口以本 README 与 YAML
配置为准。

## 3. 建议阅读顺序

第一次阅读代码时建议按以下顺序：

1. `model/config.py`：`ModelConfig` 与 `ModelContext` 如何分开结构和运行上下文。
2. `model/state_model.py` 与 `model/assembly.py`：模型生命周期与 YAML 装配。
3. `model/engine.py`：一个 trial 怎样按阶段和模块职责调度。
4. `model/modules/README.md`：各认知模块的输入、状态与公式。
   `model/readout.py` 集中 choice、output-noise、RT 和 oral-report 读出。
5. `inference/dispatcher.py` 与 `inference/backends/`：怎样选择和运行推理后端。
6. `metrics/README.md`：共享指标的数据契约与依赖边界。
7. `simulation/execution.py`：单次 StateModel 执行怎样转换为公共指标和损失。
8. `simulation/autonomous.py`：模型怎样自主采样 choice 并接收任务 feedback。
9. `simulation/runner.py`：独立随机重复怎样选择、聚合和生成统计 schema。
10. `optimization/search/coordinate_descent.py` 或 `optimization/search/grid.py`：超参数搜索。
11. `run_simulation.py`、`run_hyper_then_simulation.py`：顶层执行和序列化。

若只关心 0806，再阅读：

1. `model/modules/hypothesis_transition/README.md`
2. `model/modules/hypothesis_transition/contracts.py`
3. `model/modules/hypothesis_transition/dynamic_adaptive_control.py`
4. `inference/backends/particle_filter.py`
5. `configs/model_struct/pmh_model_cond1_0806.yaml`
6. `docs/model_0806_workflow.md`

## 4. 一个 trial 的数据流

标准 observation 为：

```python
(stimulus, choice, feedback)
```

典型 `agenda`：

```text
perception_mod
  → hypo_transitions_mod
  → memory_mod
  → beta_mod
```

各步职责：

1. `perception_mod` 产生内部感知刺激并写回 `engine.observation`。
2. `hypo_transitions_mod` 决定当前 active hypotheses，并把上一 posterior 映射为当前 prior。
3. choice/feedback 出现后，engine 固定调用 `observation_likelihood.process()`。
4. `memory_mod` 融合短期/长期证据，写入 `engine.posterior`。
5. `beta_mod` 更新下一 trial 使用的 hypothesis-specific inverse temperature。

模块的配置实例名不参与运行时语义判断。每个模块用 `ModulePhase` 声明执行时点，用
`ModuleRole` 声明唯一职责；需要模块间协作时，engine 按 role 查找。完成 choice/feedback 后，
`complete_trial()` 再统一广播 `record_outcome()`，保证当前 outcome 不进入同一 trial 的
pre-choice transition。

Likelihood 不在 `agenda` 中：它是每次 Bayesian learning 必须执行的观测模型，
其纯计算实现位于 `hypothesis_space/observation_model/likelihood.py`。

在下一个 trial 开始时，engine 先执行 `prior <- posterior`，transition module 可再对其进行
集合迁移。`prior_t` 因而表示看到 trial `t` 的 choice/feedback 之前的预测状态。

`StateModel` 将这一过程显式拆为共享的三段生命周期：

```text
begin_trial(stimulus) -> predict_choice() -> complete_trial(choice, feedback)
```

观察数据路径 `fit_step_by_step()` 在最后一步注入被试 choice/feedback；自主路径
`generate_step_by_step()` 先从预测分布采样 choice，再由任务环境产生 feedback。两条路径使用
同一 perception、transition、likelihood、memory 和 beta 更新，不维护平行的生成模型公式。

## 5. 两种推理后端

### 5.1 单轨迹后端

未配置 `inference`，或设置：

```yaml
inference:
  backend: trajectory
```

时，`inference/backends/trajectory.py` 调用 `StateModel.fit_step_by_step()`，运行一条
随机认知轨迹。重复仿真由
`simulation/runner.py` 中的 `StateModelSimulationRunner` 生成独立 trajectory seeds。

### 5.2 粒子后端

```yaml
inference:
  backend: particle_filter
  particle_count: 512
  resample_threshold_fraction: 0.5
```

`inference/dispatcher.py` 读取这段配置，标准 `evaluate_state_model_run()` 随后获得粒子
边际输出，对不可见的 perception/transition
路径求边际预测。每个粒子仍然是正常的 `StateModel`，重采样依靠 engine/module 的
`state_dict()` 与 `load_state_dict()`，没有另一套认知模型状态。

粒子 transition 日志显式区分 pre-choice predictive 策略和 post-choice filtered 诊断。
`predictive_strategy_exploit/local_explore/global_explore` 在当前 choice 进入权重更新之前求边缘，
用于解释该 trial 的学习策略；filtered controls 保留用于事后状态诊断。

两个 backend 均返回 `inference/results.py` 定义的 `InferenceResult`。优化与评估层
优先读取其中的公共 probability/state/latent/diagnostic mappings；旧 backend-specific 属性
继续作为兼容入口。

当前通用粒子入口支持 condition 1、`expectation`/`sharpened_expectation` readout，以及
uniform `base_lapse`。`choice_readout.kwargs.strategy_confidence_gain > 0` 还可在 hypothesis
已汇总为 category probability 后、lapse 之前加入策略条件化执行确信度。它只使用 pre-choice
controller state：令
`signal_t = max(mastery_evidence_t - failure_pressure_t, 0)^2`，再以
`precision_t = 1 + gain * signal_t` 对当前 category probability 做幂变换。该操作放大当前偏好，
不读取正确答案，也不改变 hypothesis learning；默认 `gain=0` 时严格退化为旧行为。历史依赖
lapse、RT emission 和其他 condition 尚未进入这一入口。

`continuous_controller.execution.enabled: true` 可让每个 trajectory/particle 维护一个
`executed_hypothesis`：active set 仍表示内部候选池，但 choice 只执行该 rule。执行 rule
占用一个受保护 slot，内部搜索使用其余 slots；只有部分已实现的搜索事件按
`execution.switch_scale` 转为 overt switch。该状态进入 engine/module snapshot 并随 PF
重采样传播，当前 choice 只在 pre-choice prediction 之后更新粒子权重。未配置时保持原来的
active-hypothesis 边际读出。

若再设置 `beta_mod.kwargs.update_scope: executed_hypothesis`，每次反馈只改变 overt rule
自己的 beta；配合非零 `correct_additive` 和 `decrease_rate`，成功会逐步强化当前规则的
判别锐度，失败会降低它的确信度，未执行候选不会被同步强化。

`continuous_controller.execution.misconception_capture.enabled: true` 可进一步维护严格
history-only 的规则—选择相容度：failure pressure 高时优先搜索并执行更能解释近期选择的
alternative rule，随后以最短 dwell 抑制立即反悔。它用于表达“自洽错误规则的短期固着”，
与动态 beta 的“对当前执行规则有多确信”是两个不同状态；两者均随 particle snapshot 和 PF
重采样传播。

更保守的实验性变体是 `continuous_controller.execution.rule_commitment`。它从完整的 19-rule
空间选择唯一最能解释历史选择的候选，并可用 `min_prior_mastery` 要求被试在进入固着前已经
达到过掌握水平；`min_hold_choice_compatibility` 则在后续选择不再支持该规则时允许及时释放。
所有门控只消费已完成 trial，状态与 `peak_mastery_evidence` 一并进入 particle snapshot。
`choice_readout.kwargs.rule_commitment_confidence_gain` 只在 commitment active 时放大当前规则的
category preference。该机制默认关闭，且不得与 `misconception_capture` 同时开启；当前 selected-eight
结构探针只支持把它作为待比较模型变体，尚不支持设为通用默认机制。

## 6. 配置分层

配置位于仓库根目录 `configs/`，分成三层：

```text
hyper_cd_cfg or hyper_grid_cfg
        │ base_sim_config_path
        ▼
simulation_cfg
        │ engine_config_path / engine_config
        ▼
model_struct
```

- `model_struct/*.yaml`：partition、module class/kwargs、agenda、readout、output noise、inference backend。
- `simulation_cfg/*.yaml`：subjects、dataset、repeats、prediction mode、loss、评价切分、输出目录。
- `hyper_*_cfg/*.yaml`：搜索空间、目标排序、coarse/fine 预算。

相对路径永远相对于“声明该路径的 YAML 所在目录”解析。subject-specific 设置通过
`subject_overrides` 合并；超参数路径必须以 `engine.` 或 `simulation.` 开头。

需要按时间顺序冻结参数并评价时，在 simulation config 中声明：

```yaml
evaluation_protocol:
  mode: sequential_holdout
  train_fraction: 0.50       # 或 train_trials，二选一
  optimization_partition: train
  simulation_partition: evaluation
```

Grid/CD 仍执行完整观察序列，但只用前缀 trial 计算候选 objective；冻结参数后的 simulation
仍执行同一完整序列，只用后缀 trial 报告 loss 和重复统计。后缀中的每个 one-step-ahead 预测可以
因果地使用之前已经观察到的 trial，但任何后缀结果都不会参与参数选择。未配置该字段的旧 YAML
继续对全部 trial 评分。实际切分点、角色和评分 trial 数写入结果的
`selection.selection_meta.score_context`。

## 7. 正式入口

### 超参数搜索

```bash
python -m src.Bayesian_state.optimization.cli \
  --backend cd \
  --config configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml
```

将 `cd` 替换为 `grid` 可运行显式网格搜索。

### 搜索后生成逐被试配置并仿真

```bash
python -m src.Bayesian_state.run_hyper_then_simulation \
  --backend hyper_cd \
  --hyper-config configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml
```

Model 0809 的 selected-eight 完整序列试跑使用一份独立配置，不改写历史 0806 输出：

```bash
python -m src.Bayesian_state.run_hyper_then_simulation \
  --backend hyper_cd \
  --hyper-config configs/hyper_cd_cfg/model0809_cond1_dynamic_continuous_selected8.yaml \
  --subjects 103 104 105 108 111 120 124 132 \
  --stage coarse \
  --skip-simulation \
  --generated-sim-config configs/simulation_cfg/generated_from_hyper/model0809_selected8_best.yaml \
  --sim-output-dir results/model_dynamic_continuous/0809_v1/simulation
```

该配置用全部 trial 的 `choice_nll` 搜索参数，并把 `capacity` 当作被试级坐标
`[3, 5, 7]`。选中的容量会写入生成 YAML 的 `subject_overrides`，在同一被试的所有 trial
保持固定。检查 coarse 结果后，将上面命令中的 `--stage coarse` 改成
`--stage fine --resume-from-coarse` 运行 fine 搜索；最后对生成配置运行：

```bash
python -m src.Bayesian_state.run_simulation \
  --config configs/simulation_cfg/generated_from_hyper/model0809_selected8_best.yaml
```

### 固定参数仿真

```bash
python -m src.Bayesian_state.run_simulation \
  --config configs/simulation_cfg/pmh_cond1_simulation_0806.yaml
```

### 超参数搜索诊断

```bash
python -m src.Bayesian_state.run_hyper_evaluation \
  --input-dir results/state-based-hyper-cd/pmh/cond1_0806_selected8
```

### 仿真结果评价

```bash
python -m src.Bayesian_state.run_model_evaluation \
  --input-dir results/state-based-simulation/pmh/cond1_0806
```

所有命令建议从仓库根目录执行。

## 8. 输出约定

Hyper 搜索通常写入：

```text
<hyper-output>/
  best_hyperparams.json
  subject_<id>/
    best_hyperparams.json
    all_combinations.jsonl
    stage_summary.json
    restart_summary.json          # Hyper-CD
    coordinate_trace.jsonl        # Hyper-CD
```

固定仿真通常写入：

```text
<simulation-output>/
  subjects/subject_<id>.json
  cache/subject_<id>_raw_runs.gz  # keep_logs=True 时可能存在
```

subject JSON 保存轻量 summary、representative run 和 stream reference；大规模逐 run 状态通过
`utils.streaming.StreamList` 压缩存储，避免 JSON 无限膨胀。需要 trajectory-rank、posterior-rank
或 FFT clustering 时，最终仿真必须保留相应日志。

## 9. 0806 在当前架构中的位置

0806 的 choice 主路径已经主框架化：

- dynamic-continuous hypothesis transition 是普通 `hypo_transitions_mod`；
- surprise、uncertainty 及联合 controller 都通过 module kwargs 表达；
- 粒子滤波由 `inference.backend` 选择；
- choice Brier、NLL、accuracy curves、Hyper-CD 和结果序列化复用公共实现。

正式 `StateModel` 已支持自主 choice/feedback trajectory；`simulation/autonomous.py`
提供类别学习任务入口。尚保留在 `reference_models/model_0806.py` 的 RT emission 和旧 rolling
实验属于参考工作流。迁移这些额外观测模型时，应增加独立 module/adapter，而不是继续扩展一套
平行的完整模型。

## 10. 扩展原则

新增模型机制时：

1. 如果改变 trial 内认知状态，放入 `model/modules/`。
2. 如果改变 hypothesis inventory、geometry 或 observation partition，放入 `hypothesis_space/`。
3. 如果改变潜在状态积分方法，放入 `inference/backends/`；不要复制认知更新方程。
4. 如果新增评价指标，先在 `metrics/` 定义纯数值计算，再由 `simulation/runner.py` 负责聚合。
5. 如果只是复现论文某阶段的冻结算法，可放在 `reference_models/`，但必须明确不是正式入口。
6. 为所有有状态 module 实现快照、恢复、日志清理；随机 module 还应实现 future reseeding。
7. 配置 class path、README、回归测试必须与代码一起更新。
