# Readout：从内部状态到可观察反应

`readout/` 集中实现模型的输出层。`StateModel` 在作答阶段调用它，输入是当前认知状态，
输出是选择、口述或反应时的概率描述。学习更新由 engine 的认知模块负责。

| 文件 | 职责 |
|---|---|
| `choice.py` | choice readout：类别选择概率、输出噪声及本题规则来源快照 |
| `oral.py` | oral readout：读取所选类别的局部规则，生成省略、表达变体和刺激补充 |
| `report_space.py` | 固定口述词表与规则内容的表达核 |
| `observation.py` | 使用同一规则来源计算选择与口述的联合概率 |
| `oral_kernel.py` | 给定规则权重和报告映射的归一化观测核 |
| `reaction_time.py` | 反应时观测接口 |
| `shared.py` | 共享概率归一化和配置校验 |

典型时序为 `begin_trial → predict_choice → predict_oral → complete_trial`。
口述读取 choice 保存的内部刺激、规则来源和类别概率；采样使用调用方单独提供的口述 RNG。
选择和口述配置分别保留在 `choice_readout` 与 `oral_readout`，由 `StateModel` 调用。
engine 的 `agenda` 继续管理知觉、假设转换、记忆和精度的学习步骤。

新代码使用 `src.Bayesian_state.model.modules.readout.choice` 和 `.oral`。
原 `model/readout.py`、`model/oral_readout.py`、`model/oral_observation.py`、
`model/oral_report_space.py` 提供兼容导入，算法统一保存在本目录。
