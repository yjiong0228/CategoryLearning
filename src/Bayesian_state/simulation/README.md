# simulation

本目录负责固定参数下的观察数据执行、独立重复运行与模型自主行为生成。
它不搜索或选择超参数；optimization 通过调用这里的稳定运行接口比较候选参数。

## 文件

| 文件 | 职责 |
|---|---|
| `state_model_execution.py` | trial 数据、运行结果对象及单次 trajectory/particle StateModel 执行 |
| `repeated_simulation.py` | 独立重复、representative run、聚合统计和稳定结果 schema |
| `simulation_config.py` | YAML/路径/loss/顺序评价协议解析及 packed profile 参数展开 |
| `autonomous_model_execution.py` | 在类别学习任务中运行模型自主选择并生成 feedback |

`autonomous_model_execution.py` 提供 `run_autonomous_category_learning()`。物理
stimulus/category schedule 固定，但 category 只由任务环境在选择后用于产生 feedback，不会提前
进入模型。模型内部调用 `StateModel.generate_step_by_step()`，因此自主生成和观察数据拟合共享
`begin_trial() -> predict_choice() -> complete_trial()` 生命周期。这是参数恢复、模型恢复和自主
posterior-predictive validation 的生成基础，不在本文件中实现恢复候选搜索。

公共结果对象位于 `state_model_execution.py`。本目录不定义 loss、metric 公式或搜索策略：loss 和
统计定义属于 `metrics/`，候选比较、容差和 anchor guard 属于 `optimization/`。

顶层 `run_simulation.py` 同时提供 CLI 和公开函数
`run_simulation(config_path, subjects=..., subject_range=...)`。组合式 workflow 直接调用该函数，
不再通过子进程重新进入 CLI；函数返回本次写出的 subject JSON 路径。

## 顺序训练/留出评价

`evaluation_protocol.mode: sequential_holdout` 只改变哪些 trial 进入指标和 loss，不截断
`StateModel` 的观察序列。`optimization_partition` 通常为 `train`，供 Grid/CD 用前缀选择参数；
`simulation_partition` 通常为 `evaluation`，供冻结参数后的 simulation 用后缀报告效果。
这保留了在线学习的因果状态更新，同时阻止留出段进入 hyperparameter selection。

切分可由 `train_trials` 或 `train_fraction` 指定，二者不可同时出现；
`min_train_trials`/`min_evaluation_trials` 用于拒绝过小分区。未声明协议时保持原有的全序列评分。
runner 将解析后的 `score_context`（切分点、角色、分区和评分 trial 数）写入
`selection_meta`，后处理不需要重新猜测切分。
