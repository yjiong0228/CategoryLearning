# Bayesian_state 模型架构

本包包含 `StateModel` 装配层，以及 perception、memory、beta 和 hypothesis transition
等可插拔模块。假设空间和 partition geometry 位于 `../hypothesis_space/`；状态容器与模块调度器
位于本目录的 `engine.py`，潜在路径推理后端位于 `../inference/`。

第一次阅读这部分代码时，建议按以下顺序：

1. `state_model.py`：`StateModel` 如何构建，以及 trial 如何进入 engine。
2. `config.py`：模型结构与单次运行上下文的显式契约。
3. `engine.py`：共享模型状态、模块阶段、职责和 agenda 调度。
4. `../inference/README.md`：单轨迹和粒子推理后端。
5. `../hypothesis_space/README.md`：完整的假设空间包。
6. `modules/README.md`：可插拔 engine module 的详细行为。
7. `modules/*.py`：各认知模块的实现细节。

## 核心对象

### `model_1024.py` 中的 `model_1024_runtime`

该上下文为exp4二分类入口组合既有1024研究机制：规则信念、替换槽位的信念转移、结果依赖保持、
错误驱动的新规则初始信念份额、ADD10精度更新和STICKY持续执行。ADD10沿用既有反馈梯度，
更新步长为学习率乘梯度再乘10；STICKY在信念权重基础上增加重复执行上一条规则的倾向。
具体算法继续位于各模块中。配置沿用0923 full-v3的装配与校验，并显式声明1024坐标与运行身份。
新规则的精度继承由 `compact_belief` 按替换前全部活跃规则的更新后信念加权计算，
与新规则初始信念份额分别处理。Exp4迁移设计及候选调整见
[迁移文档](../docs/model_architecture/model_1024_exp4.tex)。

在上下文内完成模型创建、观测推断和自主生成；退出时恢复原方法。适配器修改当前进程中的
类方法，因此同一进程串行执行且上下文仅进入一次。exp4小规模入口限制为一个worker。
持续执行属于研究候选；内部状态恢复及个体参数解释继续遵循[研究指南](../docs/RESEARCH_GUIDE.md)。

### `state_model.py` 中的 `StateModel`

`StateModel` 是高层模型包装器。它接收经过 `ModelConfig` 校验的 `engine_config` 和显式
`ModelContext`，创建连续或离散 partition，
构造假设集合，初始化 `BayesianStateEngine`，并驱动 engine 逐 trial 处理数据。

主要职责：

- 从 `ModelContext` 读取 condition、subject 和数据路径；
- 创建 partition 和假设空间；
- 通过 `engine` 向各模块提供被试和数据集上下文；
- 根据 `engine_config["modules"]` 构建模块；
- 提供共享的 `begin_trial() -> predict_choice() -> complete_trial()` 生命周期；
- 用 `fit_step_by_step(data)` 处理观察行为；
- 用 `generate_step_by_step(stimulus, feedback_provider)` 生成自主行为；
- 保存 posterior、prior 和 trial 日志。

一个完整的观察 trial 表示为：

```python
(stimulus, choice, feedback)
```

部分评价工具还会额外使用真实类别标签，例如诊断预测准确率。观察路径与自主生成路径都遵循
以下因果 trial 顺序：

```text
begin_trial(stimulus)
    perception + hypothesis transition
predict_choice()
    latent state -> observable choice distribution
complete_trial(choice, feedback)
    likelihood + memory/posterior + beta update
```

`fit_step_by_step()` 在第三步注入记录的被试 choice/feedback。
`generate_step_by_step()` 先采样 choice，再从任务 callback 获取 feedback。因此正确类别属于
任务环境，不能出现在模型的 pre-choice 状态中。

### `engine.py` 中的 `BayesianStateEngine`

`BayesianStateEngine` 是共享状态容器和调度器，拥有当前 observation、prior、posterior、likelihood、
active hypothesis mask、partition、固定的 `observation_likelihood` 计算器以及各 module 实例。
`state_model.py` 和 `assembly.py` 直接使用本包的 engine；`inference/` 只向下依赖已装配模型。

Engine 按 `agenda` 处理每个 trial，例如：

```text
perception_mod -> hypo_transitions_mod -> [fixed likelihood] -> memory_mod -> beta_mod
```

方括号中的 likelihood 阶段由 `BayesianStateEngine.compute_likelihood()` 执行，不属于 agenda item。
各 module 通过 engine 通信，不直接互相调用。

每个 module 同时声明两项互补契约：`ModulePhase` 决定何时执行，`ModuleRole` 决定它承担
perception、hypothesis transition、memory 或 beta 中哪一种唯一职责。运行时代码通过 role 获取
模块，不依赖 `perception_mod`、`memory_mod` 等配置实例名；这些名称只用于配置和结果定位。

choice 前只执行 `process()`。完成 choice/feedback 后，`complete_trial()` 先统一调用所有模块的
`record_outcome()`，再执行 post-choice phase。需要因果历史的模块覆盖该钩子，其余模块使用
`BaseModule` 的空实现，因此 outcome 不会在 pre-choice 过程里被提前记录。

### `config.py` 中的配置与上下文

- `ModelConfig` 校验 agenda 与 modules 一一对应，并保存与调用方隔离的配置副本；
- `ModelContext` 保存 condition、subject_id、processed data 目录和数据集路径；
- `partition`、`hypotheses_set` 和 `module_overrides` 是 `StateModel` 的显式构造参数，不再从任意
  `**kwargs` 中猜测。

### 假设空间与 partition

假设空间、geometry 实现、observation partition 及其共享 likelihood 基类均位于顶层包
`src/Bayesian_state/hypothesis_space/`：

```text
../hypothesis_space/
├── spaces/                     continuous/discrete 假设定义
├── geometry/                   prototype、boundary 和 rule 计算
├── observation_model/          partition 和固定 observation likelihood
├── analysis/                   口述证据的假设空间审计
└── resources/similarity/       带版本的只读矩阵
```

`StateModel` 从该包导入 partition。`model/` 下不保存假设空间、geometry 实现、partition façade、
分析脚本或相似度缓存。规范依赖图和扩展规则见 `../hypothesis_space/README.md`。

## 模块

`modules/` 文件夹包含可插拔推理步骤。常用 PMH 组合为：

- `perception.py`：将原始 stimulus 映射为内部感知 stimulus；
- `hypothesis_transition/`：完整的 H 模块包，包括共享两步契约、被试内固定策略、
  离散策略状态动力学、连续控制和版本化候选；
- `memory.py`：整合 likelihood 与 prior/posterior memory；
- `beta.py`：更新每个 hypothesis 的 inverse temperature。

`modules/readout/` 集中实现输出层，内部区分 choice 和 oral，并保留反应时接口。
`StateModel` 在作答和口述阶段调用 readout；engine 的 agenda 继续管理认知状态更新。
调用统一使用 `modules/readout/` 下的对应分支。详见 [readout 说明](modules/readout/README.md)。

公式和各模块细节见 `modules/README.md`。

强制执行的 likelihood 计算器位于
`../hypothesis_space/observation_model/likelihood.py`，通过顶层 `engine.likelihood`
mapping 配置，而不是通过 `engine.modules` 配置。

## 配置流

大多数运行从本包之外的配置文件开始：

```text
configs/exp123/model_struct/*.yaml
configs/exp123/hyper_grid_cfg/*.yaml
configs/exp123/hyper_cd_cfg/*.yaml
configs/exp123/simulation_cfg/*.yaml
```

典型流程：

1. Hyper selector 加载 `hyper_grid` 或 `hyper_cd` 配置。
2. Hyper 配置指向基础 simulation 配置。
3. Simulation 配置指向 model-structure 配置。
4. `StateModel` 接收解析后的 `engine_config`。
5. `model/assembly.py` 实例化 `engine_config["modules"]` 中列出的 module class。
6. 逐 trial 推理遵循 `engine_config["agenda"]`。

## 阅读说明

以下实现细节是有意设计的：

- `partition.hypothesis_space.hypotheses` 是唯一的规则清单。
- 连续区域存放在 `hypothesis.categories`；prototype 由
  `partition.prototype_geometry` 推导，而不是作为第二份规则表保存。
- 基于 strategy 的 transition 使用 `partition.similarity_matrix`；它优先读取随代码发布的资源，
  新矩阵写入
  `results/cache/hypothesis_space/`。
- `modules/hypothesis_transition/selection.py` 中有一个局部 `cached_dist`；该缓存用于
  transition policy 内的中心间距离，不用于缓存 likelihood 距离。
- `../hypothesis_space/resources/similarity/README.md` 说明随代码发布的矩阵。
- `configs/shared/candidates/hypothesis_transition/README.md` 说明 Hyper-CD/Grid 配置加载的
  版本化 strategy 资源。

## 几何验证

测试覆盖二分类的 29 条规则、四分类的 116 条规则、自动 component 质心、非连通类别 prototype、
归一化概率和 hard-label 一致性审计。改变 prototype 构造会改变模型行为，需要重新估计 beta
并重新运行 choice-model 比较；它不是纯格式重构。

## 假设空间审计

在仓库根目录运行 Task2 口述证据和 partition library 审计：

```bash
python -m src.Bayesian_state.hypothesis_space.analysis
```

该命令校验当前 Task2 processed data 与
`results/oral_analysis/Task2_oral_trial_diagnostics.csv` 的 trial key 是否完全一致，然后将
可复现表格、PNG 图、provenance 和报告源文件写入 `results/hypothesis_analysis`。这是分析入口，
不会修改 partition 实现或模型行为。


## Oral readout：选择后的口述观测层

`modules/readout/oral.py` 提供 `OralReadout`、`OralReadoutParameters` 与 `OralPrediction`。
它与 decision readout 共用潜在规则，属于观测层，不放入学习模块的 agenda。
Model 0923 的 oral 只生成预测，用于与真实内容比较。表达设置事先固定，行为参数
由选择数据估计；真实口述不参与参数拟合或粒子权重更新。
在 engine 配置顶层加入 `oral_readout: {method: rule_content_v1, kwargs: ...}`；
参数字段见 `configs/exp123/model_struct/model_0923_cond{1,2,3}_B0.yaml`。
`ModelConfig` 校验方法与参数；未配置时行为接口兼容原有用法。

显式调用顺序为 `begin_trial → predict_choice → predict_oral(choice) → complete_trial`。
Oral 使用 `ChoicePrediction.rule_snapshot` 中实际决策权重、各规则类别概率及感知刺激，
不会独立重抽执行规则。`choice` 从 1 起始；给定该选择后计算口述分布，不更新学习状态。
`OralPrediction.sample(rng)` 应使用独立口述 RNG；`probability(code)` 对未知代码返回 0，
不将未解析文本自动归入 OTHER。`extra_probabilities` 是每种报告同时来自补充事件的概率质量；
`extra_given_report(code)` 才是给定报告后的补充来源概率，不是人工标注结论。

首版支持连续目录、固定标签、最近类别分量、独立省略、模糊躯干命名，以及至多一个未使用
部位的刺激描述。已产生口述的 trial 不能重新作 decision，也不能用另一选择完成反馈。
output lapse 或实际改变混合类别概率的置信变换，以及区域—类别映射模块，暂不能读取口述；
必须先定义明确的联合观测关系。行为运行仍可使用这些机制。

这里仅接通单模型路径，`generate_step_by_step` 也不自动生成报告。
批量口述预测须先按选择前粒子权重汇总 `P(choice, report)`，再作条件化。
等价地，可在反馈前按仅由选择更新的粒子权重平均条件报告分布；真实报告仅用于评价。
详细假设见 [0923 规范第 13 节](../docs/model_architecture/model_0923.md#oral-readout)，
可运行例子见 `../workflows/analysis/demo_model_0923_oral.py`。
