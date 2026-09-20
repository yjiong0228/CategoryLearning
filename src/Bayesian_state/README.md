# Bayesian_state 模型框架

> 本目录是期刊与博士论文共同维护的模型实现。期刊的配置、入口和验证位于
> [`CategoryLearning_codes/Bayesian_model`](../../CategoryLearning_codes/Bayesian_model/README.md)，
> 该目录依赖本包；本包不依赖期刊目录。修改机制只改这里，实验差异由配置和数据适配处理。
> 0826 支持 condition 2 的四分类二值反馈，以及 condition 3 的三值反馈和未知反应配对学习。
> Condition 3 当前开放 PMH 全模型；已有实现测试及少量被试的完整序列拟合试点，全体拟合与恢复验收尚未完成。
> 新入口和限制见 [condition 2 说明](docs/model_architecture/model_0826_condition2.md)。
> Task2 的 `category`、`choice`、`presskey` 不能混用：类别的科结构固定，按键配对随被试变化。
> 字段约定、实现入口与 condition 3 验证范围见
> [condition 3：类别、按键与配对学习](docs/model_architecture/model_0826_condition3_design.md)。
> 统一恢复入口为 `python -m src.Bayesian_state.run_recovery`；阶段调度位于 `workflows/recovery/run.py`。
> Model 0826 观察数据 PMH 拟合默认入口为 `python -m src.Bayesian_state.run_model_0826_fit`。
> 默认 v2 接入分散/联合搜索、一层分级独立审查、边界标记和断点续跑；旧 v1 配置保留。
> v2 九人完整拟合及定向诊断已结束；九份兼容结果可按原标记复用，原整库评分通过4/9，严格整体停止0/9。
> 当前保持共同范围和有界预算。按2026-09-21最新指示，下一批仅运行[每条件一人的完整拟合](optimization/THREE_SUBJECT_FIT_20260921.md)：S104/S215/S328；全体暂不启动。
> 软件接入不代表全体拟合、范围校准或恢复已完成，未解决结果保留明确状态。
> 用法见 [自适应拟合说明](optimization/ADAPTIVE_FIT.md)，通俗流程见 [Model 0826 Plus](docs/model_architecture/model_0826_plus.pdf)。

六人试点的作用是校准和验收拟合流程：每个 condition 各两人，按试次长度接近中位数和第75百分位选取，
不按拟合成绩挑选。六人已经各做过一次完整拟合；后续只针对发现的问题追加检查，例如固定参数后增加随机重复，
不等于反复重拟合全部六人。它们用于检查搜索覆盖、停止规则、评分波动、参数触边和实际成本，不能替代群体结论。
规则既然已根据这些人的结果调整，他们就属于开发/校准样本；随后S122/S222/S315已按冻结规则完成完整核查。
流程应明确何时停止、何时追加搜索或评分、何时保留未解决，而不是要求每个人都追加计算到通过。
三人核查的冻结规则、选人标准和上限见[原协议](optimization/FROZEN_VALIDATION.md)。后续有限外侧范围检查也已结束；
原始拟合状态与追加诊断并列保留，可复用不等于已通过。新增三人沿用既定规则，仍不为达到全通过而追加到通过。

`Bayesian_state` 是本项目的试次级 Bayesian 状态模型包。它把“模型结构”“逐试次推理”
“潜在路径积分”“超参数搜索”“重复仿真”和“结果评价”分成相互独立的层，而不是把一个模型
写成一份从数据读取到画图的独立脚本。

当前正式建模路径以 `StateModel + BayesianStateEngine + modules` 为核心。`model_0806` 的动态连续
hypothesis-transition 模型也已经接入这条路径。独立 `reference_models/` 已按维护范围删除；
当前 0826 核心与恢复流程不依赖它，历史实现需要从 Git 历史恢复。

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
└── metrics/              各执行层共享的纯数值指标
```

此外，`utils/` 保存跨层基础工具，FFT 聚类归入 `evaluation/`，根目录的
`run_*.py` 只负责 orchestration；它们不改变上述七个核心职责边界。

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
| `utils/` | 路径、数据集、subject override、stream、公共统计 | 是，底层支持 |

根目录的可执行文件只负责 orchestration：

| 文件 | 作用 |
|---|---|
| `run_simulation.py` | 固定参数、逐被试重复 simulation 并序列化；也提供公开 `run_simulation()` API |
| `run_model_0826_fit.py` | Model 0826 exp123 PMH 默认自适应拟合，输出选择概率与停止/边界诊断；不自动交付状态轨迹 |
| `run_hyper_then_simulation.py` | Hyper 搜索、生成 subjectwise simulation YAML、再通过公开 API 运行 simulation |
| `run_hyper_evaluation.py` | 已完成 Hyper-CD 输出的收敛和选择诊断 |
| `run_model_evaluation.py` | 已完成 simulation 输出的统一后处理 |
| `run_autonomous_trajectory_evaluation.py` | 冻结参数下生成完整自主行为轨迹，并输出形态分布、medoid、轨迹聚类与持续掌握起点 |
| `run_internal_cognitive_trajectory_evaluation.py` | 条件于被试完整观察历史，汇总多种子 PF 完整祖先路径、形态 archetype 与 genealogy 充分性诊断 |
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
5. `configs/exp123/model_struct/pmh_model_cond1_0806.yaml`
6. `src/Bayesian_state/docs/history/model_0806_workflow.md`

## 4. 一个 trial 的数据流

### Task2 的类别与按键编码

`data/exp123/processed/Task2_processed.csv` 中，`category` 是刺激的真实类别，
`choice` 是已经用同一类别编号表示的被试所选答案，`presskey` 是实际反应按键的编码。
`choice` 不是正确答案，也不能直接视为 `presskey`。每位被试有固定的一一对应
`choice ↔ presskey` 映射；该映射在被试之间不同。提取映射应使用同一次反应的
`choice` 与 `presskey`，不能把错误试次的 `category` 与 `presskey` 当作映射样本。

对于四分类任务，真实上层为科、下层为种：`category={1,2}` 属于一科，`{3,4}` 属于另一科。
Condition 3 被试预先知道“两科、每科两种”，但不知道哪两个反应按键属于同一科。
例如 301 的 `choice 1,2,3,4` 分别对应 `presskey 4,2,3,1`，两科的真实按键集合为
`{2,4}` 和 `{1,3}`。这些真实集合可用于任务反馈和事后评价，不能直接初始化被试的配对信念。

`simulation/data.py` 保留 `choice`、`presskey` 和双向映射；先用完整被试记录核查映射，
再按既有规则截取试次。映射是实验编码元数据，不进入学习先验。
统一使用 `choice` 编号本身只是重编码；若模型据编号直接认定 `{1,2}|{3,4}` 是已知配对，
就把实验者知识提供给了被试模型。Condition 3 使用 `choice` 作为无语义的反应 ID，
从均匀的三种配对开始学习；结果保存配对顺序、实际按键和概率列坐标。
反馈核已检查整体重编码等价性，旧四分类固定标签规则库的完整标签对称性仍需单独审计。

Condition 3 显式配置为
[`pmh_model_cond3_0826.yaml`](../../configs/exp123/model_struct/pmh_model_cond3_0826.yaml)。
它同时启用联合记忆、`hierarchical_pairing` 反馈、`hierarchical_feedback` 精度更新和
`full_success` 搜索解释；缺少其中一项会报错，不能用 condition 2 配置直接读取半分反馈。
配对权重是每条粒子轨迹的学习状态，不是额外的被试参数。现有 PMH 选择 NLL 与参数搜索接口
保留；condition 3 的消融及恢复流程尚未定义，不应套用 condition 1 的恢复配置。

单被试短序列检查入口（32 试次、8 粒子、1 个进程、固定参数）：

```bash
python -m src.Bayesian_state.run_simulation --config configs/exp123/simulation_cfg/model0826_cond3_smoke.yaml
```

重跑前设置新的 `output_dir`，已有输出受保护。PF 结果含选择预测、评分掩码、按键映射、
反馈前后配对边际、配对熵与最大配对概率；这些边际分别使用进入本试次和观察选择后的粒子权重。
自主生成按“种正确 1、同科另一种 0.5、异科 0”评分，科正确率、种正确率、平均得分分别解释。

### 试次调度

标准 observation 为：

```python
(stimulus, choice, feedback)
```

典型 `agenda`：

```text
perception_mod
  → hypo_transitions_mod
  → [optional mapping_mod after fixed likelihood]
  → memory_mod
  → beta_mod
```

各步职责：

1. `perception_mod` 产生内部感知刺激并写回 `engine.observation`。
2. `hypo_transitions_mod` 决定当前 active hypotheses，并把上一 posterior 映射为当前 prior。
3. choice/feedback 出现后，engine 固定调用 `observation_likelihood.process()`。
4. 可选 `mapping_mod` 对二分类 fixed/reversed label orientation 做解析边缘化并更新 belief。
5. `memory_mod` 融合短期/长期证据，写入 `engine.posterior`。
6. `beta_mod` 更新下一 trial 使用的 hypothesis-specific inverse temperature。

模块的配置实例名不参与运行时语义判断。每个模块用 `ModulePhase` 声明执行时点，用
`ModuleRole` 声明唯一职责；需要模块间协作时，engine 按 role 查找。完成 choice/feedback 后，
`complete_trial()` 再统一广播 `record_outcome()`，保证当前 outcome 不进入同一 trial 的
pre-choice transition。

Likelihood 不在 `agenda` 中：它是每次 Bayesian learning 必须执行的观测模型，
其纯计算实现位于 `hypothesis_space/observation_model/likelihood.py`。

在下一个 trial 开始时，engine 先执行 `prior <- posterior`，transition module 可再对其进行
集合迁移。`prior_t` 因而表示看到 trial `t` 的 choice/feedback 之前的预测状态。
Model 0815 H5 将这一步显式拆为 active-set selection 与 `similarity_transport` prior assignment：
实际替换比例控制信念重构强度，当前 local/global kernel 把旧 posterior 按规则相似度投影到新
workspace。H5 把固定标签的功能一致率解释为 functional similarity，并将共同
`tau_local=0.10` 同时用于 newcomer proposal 与 belief transport；该尺度不作为被试级坐标，
避免与 `global_search` 重复控制搜索宽度。历史 `pairwise_mass_transfer` 仍可配置，但不再是 H5 默认。
Model 0826 的一次性反事实可把 prior assignment 切为 `mass_preserving_similarity_transport`：
survivors 保留绝对 posterior mass，newcomers 总共只接收被删除质量，其他模块和参数保持不变。

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

Model 0815 的 M1 sensitivity 配置可选加入 parameter-free binary orientation state。PF 会同时
返回 predictive/filtered `executed_orientation_joint`（geometry × 两种方向）；这仍是 online
filtering 输出，不是事后 smoothing。M0 不配置该 module，因而与原 fixed-label 路径保持严格兼容。

当前通用粒子入口支持 condition 1（二分类）、condition 2（四分类二值反馈）和
condition 3（四分类层级部分反馈），以及 `expectation`/`sharpened_expectation` readout，以及
uniform `base_lapse`。`choice_readout.kwargs.strategy_confidence_gain > 0` 还可在 hypothesis
已汇总为 category probability 后、lapse 之前加入策略条件化执行确信度。它只使用 pre-choice
controller state：令
`signal_t = max(mastery_evidence_t - failure_pressure_t, 0)^2`，再以
`precision_t = 1 + gain * signal_t` 对当前 category probability 做幂变换。该操作放大当前偏好，
不读取正确答案，也不改变 hypothesis learning；默认 `gain=0` 时严格退化为旧行为。历史依赖
lapse 和 RT emission 尚未进入这一入口。`choice_transmission_audit` 仍仅支持 condition 1；
condition 3 已有测试及短序列验证，正式拟合、完整恢复和消融尚未完成。

`continuous_controller.execution.enabled: true` 可让每个 trajectory/particle 维护一个
`executed_hypothesis`：active set 仍表示内部候选池，但 choice 只执行该 rule。执行 rule
占用一个受保护 slot，内部搜索使用其余 slots；只有部分已实现的搜索事件按
`execution.switch_scale` 转为 overt switch。该状态进入 engine/module snapshot 并随 PF
重采样传播，当前 choice 只在 pre-choice prediction 之后更新粒子权重。未配置时保持原来的
active-hypothesis 边际读出。

若再设置 `beta_mod.kwargs.update_scope: executed_hypothesis`，每次反馈只改变 overt rule
自己的 beta；配合非零 `increase_rate` 和 `decrease_rate`，成功会逐步强化当前规则的
判别锐度，失败会降低它的确信度，未执行候选不会被同步强化。
`increase_rate` 表示一次完全支持性反馈所获得的剩余 beta headroom 比例；旧
`correct_additive` 继续作为严格等价的兼容入口。Model 0815 H5 使用统一动态 beta、
`update_scope: active_hypotheses`、固定 `beta_min=0.1`/`beta_max=25`，并以
`increase_rate=0.04` 替代旧的 `correct_additive=1`。

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

需要可中断且抗 PF 噪声的搜索时，Hyper-CD 2.0 通过
`search_schema_version: 2` 显式启用。它保留离散/联合参数块，按完整坐标多轮扫描，并可把
coarse shortlist 投影为 fine 起点。最终 shortlist 必须用独立 seed family、候选间共同随机数
和 `mean_probability` 聚合复评分；搜索控制与复评分都属于统计估计过程，不作为被试认知机制。
恢复实验若要求全部试次，应在基础配置及 stage override 中都保持 `max_trials: null`。

## 7. 正式入口

### 超参数搜索

```bash
python -m src.Bayesian_state.optimization.cli \
  --backend cd \
  --config configs/exp123/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml
```

将 `cd` 替换为 `grid` 可运行显式网格搜索。

Schema-v2 Hyper-CD 若因中断需要继续，使用同一命令并追加 `--resume`。系统会核对配置、基础
simulation 配置、被试顺序和 stage 的 fingerprint；不一致时拒绝复用旧 PF 评分。同一输出目录
已有产物而未显式续跑时也会报错，不会静默覆盖。

### 搜索后生成逐被试配置并仿真

```bash
python -m src.Bayesian_state.run_hyper_then_simulation \
  --backend hyper_cd \
  --hyper-config configs/exp123/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml
```

Model 0809 的 selected-eight 完整序列试跑使用一份独立配置，不改写历史 0806 输出：

```bash
python -m src.Bayesian_state.run_hyper_then_simulation \
  --backend hyper_cd \
  --hyper-config configs/exp123/hyper_cd_cfg/model0809_cond1_dynamic_continuous_selected8.yaml \
  --subjects 103 104 105 108 111 120 124 132 \
  --stage coarse \
  --skip-simulation \
  --generated-sim-config configs/exp123/simulation_cfg/generated_from_hyper/model0809_selected8_best.yaml \
  --sim-output-dir results/model_dynamic_continuous/0809_v1/simulation
```

该配置用全部 trial 的 `choice_nll` 搜索参数，并把 `capacity` 当作被试级坐标
`[3, 5, 7]`。选中的容量会写入生成 YAML 的 `subject_overrides`，在同一被试的所有 trial
保持固定。检查 coarse 结果后，将上面命令中的 `--stage coarse` 改成
`--stage fine --resume-from-coarse` 运行 fine 搜索；最后对生成配置运行：

```bash
python -m src.Bayesian_state.run_simulation \
  --config configs/exp123/simulation_cfg/generated_from_hyper/model0809_selected8_best.yaml
```

### 固定参数仿真

```bash
python -m src.Bayesian_state.run_simulation \
  --config configs/exp123/simulation_cfg/pmh_cond1_simulation_0806.yaml
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

### Model 0826 模块与参数恢复

冻结的 Model0826 恢复入口为：

```bash
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py --phase smoke
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py --phase generate --resume
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py --phase calibrate --resume
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py --phase module-fit --resume
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py --phase parameter-fit --resume
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py --phase summarize --resume
```

也可在全新输出目录用 `--phase all` 顺序执行。默认输出只写入
`results/model_0826/recovery_v1/`；已有目录必须显式 `--resume`，并且源 recovery YAML、
Model0826 engine、参数空间和基础 simulation 配置的联合 fingerprint 必须一致。

优化版使用独立配置和输出目录，不覆盖 v1：

```bash
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py \
  --config configs/exp123/specific_models/model_0826_recovery_v2.yaml \
  --phase smoke
```

v2 保留全部试次、模型结构和高预算 final-rescore，只把候选发现拆成 R16×B4 coarse 与
R32×B4 fine；进入恢复前必须验证这两个阶段分别保留 R128×B16 赢家于 top-4/top-2。边界距离
默认由 Numba 执行与历史 Dykstra 更新顺序相同的机器码循环，缺少 Numba 时自动回退 Python。
如果需要在有限墙钟时间内优先得到一个完整 subject-template 检查点，可用：

```bash
python src/Bayesian_state/workflows/runs/run_model_0826_recovery.py \
  --config configs/exp123/specific_models/model_0826_recovery_v2.yaml \
  --output-dir results/model_0826/recovery_v2_subject_first \
  --phase priority-all --priority-subject 101
```

该入口连续执行 smoke、PF 校准、全部数据生成，然后按 101→111→118 分别完成模块与参数恢复；
每个数据集后原子更新增量 score 表，每个被试完成后写出明确标记为 partial 的 subject-level
报告，最后才合并全体预注册结果。中断后用完全相同命令追加 `--resume`。最终 suffix 和
near-best 的多个 PF seeds 也按 `search.cd.parallel_budget` 并行，但仍先平均逐试次概率再计算 NLL。

`smoke`、正式生成和 PF 校准都使用 101/111/118 的全部 320/320/256 个试次。模块恢复只用
70% 时间前缀进行 Hyper-CD 2.0 参数选择，冻结参数后执行完整历史但只在 30% 后缀计算
held-out total choice NLL。参数恢复对 final-rescore shortlist 全部候选与生成真值使用另一组
共同 PF seeds 成对评分；先逐试次平均 PF 概率，再计算 NLL。PF 加权和多 seed 平均是不可见状态
积分的数值估计，不被解释为被试的额外认知步骤；不同自主生成轨迹始终是独立观测，不能彼此
平均成一条行为轨迹。

## 8. 输出约定

普通 simulation 在计算前拒绝已有的选中被试产物，并原子发布新 JSON/流文件；
输出保护与中断处理见 [simulation 输出保护](simulation/README.md#输出保护)。

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
    search_checkpoint.json        # Hyper-CD 2.0，原子断点状态
    final_rescore.jsonl           # Hyper-CD 2.0，独立高预算 shortlist 复评分
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

当 `repeat_aggregation: mean_probability` 时，subject JSON 的 `simulation.mean_error` 来自逐试次
平均 repeat 概率后计算的 proper score；`sample_errors` 只是 PF seed 噪声诊断。结果同时保存
`model_provenance`，明确 resolved 配置哈希、容量、初始化策略和数值积分预算。

Model 0815 P0 的未拟合结构、precision 标定和 PF 收敛入口分别为：

```text
configs/exp123/model_struct/pmh_model_cond1_0815_p0.yaml
configs/exp123/hyper_cd_cfg/model0815_p0_cond1_beta_screening.yaml
configs/exp123/hyper_cd_cfg/model0815_p0_cond1_beta_recalibration.yaml
configs/exp123/specific_models/model_0815_p0_pf_convergence.yaml
configs/exp123/model_struct/pmh_model_cond1_0815_p1_m1_orientation.yaml
configs/exp123/specific_models/model_0815_p1_mapping_sensitivity.yaml
configs/exp123/specific_models/model_0815_p1_state_identifiability.yaml
configs/exp123/model_struct/pmh_model_cond1_0815_h4_nested_feedback_accumulator.yaml
configs/exp123/model_struct/pmh_model_cond1_0815_h5_similarity_transport.yaml
configs/exp123/specific_models/model_0815_h4_nested_subject_screen.yaml
```

其中 `beta_screening` 只用 8 人各 64 个早期 trial 和低粒子预算排除明显不合适的尺度，不能作为
最终参数或论文结果。P0 不覆盖 0813 输出。先完成正式 beta recalibration 并冻结生成的 simulation config，再把该 config
填入 PF convergence 配置；收敛门槛通过后才能启动新的全样本拟合/评价。
该 calibration 配置使用 `hyperparam_selection_mode: shared`：selected-eight 仅共同校准一套
boundary-distance precision 参数，每位被试先得到自己的 choice NLL，再在被试间等权平均；
它不是为每位被试增加五个新的自由参数。

P1 mapping 入口 `src/Bayesian_state/workflows/runs/run_model_0815_p1_mapping_sensitivity.py` 严格配对 M0/M1 的 PF seeds，
先对 seed 重复的预测概率求平均再计分，并统一输出双向 fixed-parameter recovery、geometry/orientation state
recovery、early predictive NLL、执行规则轨迹敏感性以及 mapping effect/PF seed noise 比值。该小样本
pilot 只用于决定 mapping omission 是否会污染核心 strategy inference，不能单独用来选择最终模型；

当 pilot 的 M1 geometry recovery 较差时，
`src/Bayesian_state/workflows/runs/run_model_0815_p1_state_identifiability.py` 在固定的 M1 synthetic paths 上比较
R=64/128/256，并以四个共同 PF seeds 同时检查 choice probability、geometry、
geometry×orientation joint、后验期望 switch、ancestral support 与 true-path choice-likelihood
replay。最高粒子数另跑 O1 oracle：只把生成器完整的 pre-choice orientation belief vector
提供给 PF，不固定 geometry。这个 oracle 是 identifiability upper control，不是候选模型；
terminal ancestry 也只是 genealogy diagnostic，不等同于 FFBSi/PGAS smoothing。相邻粒子数和
seed-noise 门槛预先写在配置中，未通过时只在同一批冻结轨迹上升级粒子数。
升级轮可用 `--reference-cache-dir` 只读复用上一轮中匹配的低粒子数 cache；runner 会先核对
base seed、trial 数、被试和 model path，并把新粒子数结果写入新的 output directory，不覆盖
上一轮结果。
其他参数重新优化后的 recovery 必须等 P0 calibration 与 PF budget 冻结后再运行。

Model 0815 H4 把 constant、一步 feedback-reactive 与 accumulated-failure search 写成一个严格
嵌套的 bounded-workspace controller。`accumulator_logit_gain: 0` 直接走 H3 reactive 更新路径；
`global_search_failure_gain: 0` 则保留固定的 local/global mixture；两个 gain 都为 0 且 correct/error
event probability 相等时得到 constant 边界。正的 global gain 使用同一个 failure state：
`g_t = g_0 + (1-g_0)c_gF_t`，不会增加第二套 range accumulator、阈值或 rise/recovery controller。
当前模板在两个 gain 的精确零边界提供独立接口，尚未进行最终被试级参数估计。
历史入口 `src/Bayesian_state/workflows/runs/run_model_0815_h4_nested_subject_screen.py` 只用前32 trials和独立 training PF seeds选择
reactive 基线、共同 decay 与被试 gain；后32 trials及另一组 PF seeds 只用于最终预测比较。训练搜索
使用较低粒子预算，锁定参数后的效应估计恢复到32粒子×4 seeds。该切分结果现仅作为历史技术
产物保留，不再用于 accumulator 或动态 global search 的当前架构去留决定。
H5 不改动上述 nested controller，只把 prior assignment 换成 parameter-free similarity transport；
H4 配置继续保留为既有审计结果的生成来源。当前 H5 模板还把 persistent execution 接成默认关闭的
被试级二元结构坐标：关闭时严格保持原 H5 workspace-mixture readout，开启时每个 PF particle
维护并执行一条受保护规则；`switch_scale` 继续固定，避免把二元结构选择扩展成新的连续补偿参数。

## 9. 0806 在当前架构中的位置

0806 的 choice 主路径已经主框架化：

- dynamic-continuous hypothesis transition 是普通 `hypo_transitions_mod`；
- surprise、uncertainty 及联合 controller 都通过 module kwargs 表达；
- 粒子滤波由 `inference.backend` 选择；
- choice Brier、NLL、accuracy curves、Hyper-CD 和结果序列化复用公共实现。

正式 `StateModel` 已支持自主 choice/feedback trajectory；`simulation/autonomous.py`
提供类别学习任务入口。旧参考实现中的 RT emission 和 rolling 实验已退出当前维护范围。
未来若需要这些额外观测模型，应增加明确的 module/adapter。

## 10. 扩展原则

新增模型机制时：

1. 如果改变 trial 内认知状态，放入 `model/modules/`。
2. 如果改变 hypothesis inventory、geometry 或 observation partition，放入 `hypothesis_space/`。
3. 如果改变潜在状态积分方法，放入 `inference/backends/`；不要复制认知更新方程。
4. 如果新增评价指标，先在 `metrics/` 定义纯数值计算，再由 `simulation/runner.py` 负责聚合。
5. 已删除历史工作流的复现使用对应 Git 版本，不再扩展独立 reference_models 实现。
6. 为所有有状态 module 实现快照、恢复、日志清理；随机 module 还应实现 future reseeding。
7. 配置 class path、README、回归测试必须与代码一起更新。

## 工作流与模型文档

专题脚本已从根 scripts/ 迁入 [workflows/](workflows/README.md)，
目前保留 recovery、runs、analysis、benchmarks 中的必要工具。模型设计与历史说明位于
[docs/](docs/README.md)。生成的报告随 results 中对应模型保存。

恢复职责和当前工作流保留范围见 [恢复说明](workflows/recovery/README.md)及
[工作流索引](workflows/README.md)。此前 0813/0815 等历史脚本的命令示例按原版本理解，
是否仍提供入口以当前工作流索引为准。

### 2026-09-17 输入校验与断点溯源修复

反应编码在整数转换之前检查：必须有限、为整数且属于任务类别范围；不自动填补漏答或
截断小数。Loader 在 `stop_at` / `max_trials` 截取之前检查完整被试的编码、条件一致性
及试次键。含 `iTrial` 的表默认按现存的 `iSession, iRun, iBlock, iTrial` 列检查严格递增
和唯一性；自定义任务请在引擎中明确设置，例如：

```yaml
data:
  trial_order_columns: [iSession, iBlock, iTrial]
```

显式声明的列必须存在。无 `iTrial` 且未声明试次键的旧数组／表仍按输入行序运行，无法
验证其真实时序；正式新任务应声明完整键。检查不排序数据，也不在 session 边界重置认知状态。

Hyper-CD search schema 2 与 recovery manifest 增加独立的 `fingerprint_schema_version: 2`。
恢复前检查解析后的阶段／被试配置、引用文件内容、行为与感知数据、规则资源、实际连续规则
相似矩阵、共享核心全部 Python 源码和数值库版本。相似矩阵首次缺缓存时会按现有算法生成，
再建立指纹；不会把后续生成的缓存误作输入变更。输入文件须在运行期间保持不变。
当前任务与规则几何在候选间须固定；schema-2 搜索拒绝把数据路径、partition 或
`likelihood.distance_mode` 当搜索坐标。自定义类的直接源码及基类被记录，但额外外部资源
应通过配置中的文件路径显式声明，不能依赖隐藏的运行环境输入。

旧 checkpoint 缺少完整内容指纹，不能在新版中直接续算。保留旧目录；可在原代码及原环境下
复现，或使用新输出目录重算。没有自动给旧分数补盖新指纹的迁移操作。共享核心任一源码变更
都可能保守地拒绝续跑，即使只是与该任务无关的修改；固定代码快照后再启动正式长任务。

模型说明现在是独立 TeX 文档。编译到一个不存在的新文件：

```bash
bash src/Bayesian_state/docs/model_architecture/compile_model_0826.sh /tmp/model_0826_new_proof.pdf
```

脚本拒绝覆盖已有 PDF，构建日志留在独立临时目录。修复记录与六步研究流程说明见
[系统修复与后续步骤](docs/maintenance/model_0826_system_repairs_20260917.md)。


### Model 0826 等价加速（2026-09-17）

零 beta 的均匀预测快捷路径和 boundary 距离的有界精确缓存已默认启用。缓存以实际感知刺激为 key，
同次 PF 粒子共享，不改变随机数流、反馈核或全规则归一化。默认最多 4096 条／4 MiB 数组内容；
后续优化合并零 beta 的相同二值反馈似然列，减少工作区成员查询，并复用唯一祖先快照；后代
仍独立复制认知状态。确定性学习门控省去独立 RNG 构造，其他随机流保持原样。类别概率缓存
与轻量诊断仍只在性能原型中评估，正式拟合的输出字段和评分路径保持不变。
设置与关闭方法见 [假设空间 README](hypothesis_space/README.md#model-0826-的等价计算加速2026-09-17)，
验证入口见 [性能检查](workflows/benchmarks/README.md)。这属于等价工程优化；由于源码已更新，
旧代码创建的断点仍受内容指纹约束，正式新运行应使用新的输出目录。

### Model 0826 正式并行预算（2026-09-17）

正式 recovery 的搜索和固定仿真配置使用 **128 个进程的预算，不主动预留 CPU**；
新生成的 0826 搜索配置和轨迹评价 CLI 也默认使用 128。每个 worker 的数值计算线程为 1，
嵌套调用串行执行，避免进程数与 BLAS/OpenMP 线程数相乘。实际进程数取预算、可用 CPU
和当前就绪任务数的最小值。显式指定的较小预算和 smoke/test 配置仍有效。

这只改变计算资源分配，不改变粒子数、重复次数、随机种子、模型机制或候选选择顺序。
当前按被试串行推进的流程与 CD 坐标依赖仍然保留，因此小任务批次未必能持续占满 128 核。
已有运行进程和结果目录内的冻结配置不会自动更新；新正式运行使用新输出目录。
具体入口和验证见 [128 核并行策略](docs/maintenance/model_0826_parallel_policy_20260917.md)。
