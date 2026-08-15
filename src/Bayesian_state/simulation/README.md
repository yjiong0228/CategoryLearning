# 仿真

本目录负责固定参数下的观察数据执行、独立重复运行与模型自主行为生成。
它不搜索或选择超参数；optimization 通过调用这里的稳定运行接口比较候选参数。

## 文件

| 文件 | 职责 |
|---|---|
| `data.py` | trial 数组、数据加载、被试切片与验证 |
| `results.py` | 单次和重复运行的稳定结果契约 |
| `execution.py` | 单次 trajectory/particle StateModel 执行 |
| `runner.py` | 独立重复、representative run 与聚合统计 |
| `config.py` | YAML/路径/loss/顺序评价协议解析及 packed profile 参数展开 |
| `parameters.py` | 固定超参数的提取、覆盖与可复现 candidate seed 解析 |
| `provenance.py` | 结构配置哈希、容量、初始化、precision/readout 与 repeat 聚合溯源 |
| `autonomous.py` | 在类别学习任务中运行模型自主选择并生成 feedback |

`autonomous.py` 提供 `run_autonomous_category_learning()`。物理
stimulus/category schedule 固定，但 category 只由任务环境在选择后用于产生 feedback，不会提前
进入模型。模型内部调用 `StateModel.generate_step_by_step()`，因此自主生成和观察数据拟合共享
`begin_trial() -> predict_choice() -> complete_trial()` 生命周期。这是参数恢复、模型恢复和自主
posterior-predictive validation 的生成基础，不在本文件中实现恢复候选搜索。

公共结果对象位于 `results.py`，trial 数据契约位于 `data.py`。本目录不定义 loss、metric 公式或搜索策略：loss 和
统计定义属于 `metrics/`，候选比较、容差和 anchor guard 属于 `optimization/`。

顶层 `run_simulation.py` 同时提供 CLI 和公开函数
`run_simulation(config_path, subjects=..., subject_range=...)`。组合式 workflow 直接调用该函数，
不再通过子进程重新进入 CLI；函数返回本次写出的 subject JSON 路径。
固定参数工具属于仿真领域接口，调用方应从 `simulation.parameters` 导入，而不是反向依赖 CLI
入口 `run_simulation.py`。

## 顺序训练/留出评价

`evaluation_protocol.mode: sequential_holdout` 只改变哪些 trial 进入指标和 loss，不截断
`StateModel` 的观察序列。`optimization_partition` 通常为 `train`，供 Grid/CD 用前缀选择参数；
`simulation_partition` 通常为 `evaluation`，供冻结参数后的 simulation 用后缀报告效果。
这保留了在线学习的因果状态更新，同时阻止留出段进入 hyperparameter selection。

切分可由 `train_trials` 或 `train_fraction` 指定，二者不可同时出现；
`min_train_trials`/`min_evaluation_trials` 用于拒绝过小分区。未声明协议时保持原有的全序列评分。
runner 将解析后的 `score_context`（切分点、角色、分区和评分 trial 数）写入
`selection_meta`，后处理不需要重新猜测切分。

particle-filter 的 `state_log` 同时保存 post-choice filtered transition 诊断和以
`predictive_*` 命名的 pre-choice 策略边缘量。后者用于回答“当前 trial 做选择之前偏向利用还是
探索”，不得与观察当前 choice 后重新加权得到的 filtered 状态混用。

内部诊断可向 `StateModelSimulationRunner.simulate_subject()` 传入与 repeats 等长的
`trajectory_seeds`，用于 common-random-number 反事实。默认调用仍按既有
`hyper_candidate_seed -> simulation_point_seed -> trajectory_seed` 链生成 seeds；只有显式传入
该参数时才覆盖。`compute_statistics=False` 可供不需要重复运行统计的诊断使用，默认保持 `True`。

## 独立重复的主分数

历史配置默认使用 `repeat_aggregation: mean_loss`：每个随机 repeat 先计算 loss，再平均标量。
需要把多次 PF 当作同一潜路径积分的独立数值近似时，应显式设置：

```yaml
simulation_repeats: 8
repeat_aggregation: mean_probability
```

此时 runner 先逐 trial、逐 category 平均所有 repeat 的预测概率，再从这个平均分布计算一次
choice NLL/Brier。`simulation.mean_error` 是这个 probability-mixture score；各 repeat 的原始 loss、
均值、标准差和 MCSE 仍保存在 `sample_errors` 与 `aggregation_diagnostics`，不能用 best repeat
替代主结果。Grid、Hyper-CD 和固定参数 simulation 共用同一定义。

固定仿真还把 `model_provenance` 写入 subject JSON。它包括完整 resolved engine config 的 SHA-256、
共同/被试级容量、初始 active set 是固定还是按 prior 无放回抽样、PF 粒子数、规则证据 precision、
动作 beta 配置及 repeat 聚合方式。声明性解释来自 engine YAML 的 `provenance` 字段；该字段不参与
认知计算。

Controller v2a 的三被试结构探针配置是
`configs/simulation_cfg/generated_from_hyper/model0809_controller_v2a_selected3_probe.yaml`。
它继承 0809 已选中的 memory/readout/noise/capacity 设置，只替换 continuous controller，并写入
新的 `results/model_dynamic_continuous/0810_controller_v2a_probe/`；它不是新的 Hyper-CD 拟合。
v2b 的受限先验重置探针是
`configs/simulation_cfg/generated_from_hyper/model0809_controller_v2b_selected3_probe.yaml`；除
`prior_reset.max_strength: 0.35` 与独立输出目录外，它与 v2a 完全相同，便于直接归因比较。
