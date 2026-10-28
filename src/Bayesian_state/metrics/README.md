# 指标

本目录是 `Bayesian_state` 的共享数值统计层。它只定义“怎样从预测与观测计算指标”，供
`optimization/`、`simulation/` 和 `evaluation/` 共同调用，不负责参数选择、模型执行、
文件读写、作图或科学结论判定。

## 依赖边界

```text
optimization ───────┐
simulation ─────────┼──> metrics
evaluation ─────────┘
```

`metrics` 不应导入上述三个工作流 package，也不应依赖某个 inference backend。上层先将结果
转换成 `TrialPrediction` 或 `RunPrediction`，再调用纯数值函数。

## 文件

| 文件 | 职责 |
|---|---|
| `prediction.py` | trial/run prediction 数据契约，以及 choice Brier/NLL、ECE、CRPS 和预测区间统计 |
| `losses.py` | 所有 `loss_metric` 的唯一实现，包括 accuracy-curve BerHu、Brier 与 NLL family |
| `trial.py` | trial 对齐、rolling/exponential accuracy、family/target-majority 和标准 metric bundle |
| `task.py` | 实验环境反馈，以及种正确率、科正确率、平均得分；真实类别只用于评分 |
| `probabilistic_task.py` | 二分类概率任务的选择质量、实际奖励和期望奖励差距 |
| `behavior.py` | 学习曲线、history kernel、switch、perseveration、win--stay/lose--shift |
| `trajectory.py` | 跨随机轨迹的边际 choice 预测，以及 loss、shape、history、switch 与分布汇总 |
| `selection.py` | accuracy shape、history、switch 和代表轨迹选择所用的复合分数 |
| `group.py` | 以被试等独立单位进行 paired delta、bootstrap 和 FDR 汇总 |
| `residuals.py` | 逐试次顺序残差与因果状态诊断所需的纯数值函数 |
| `oral_distribution_alignment.py` | 完整口述空间中的分布重合度、Jensen–Shannon 相似度，以及按实际试次窗口汇总的预测分布和真实口述频率 |
| `numeric.py` | metric 模块复用的小型数值 helper |

Condition 3 的 `condition3_task_metric_summaries()` 沿用调用方的 `valid_trial_mask`，
不改变首试次排除规则。分别返回 observed/predicted 的种、科、平均得分；相同试次下
`mean_reward = (species_accuracy + family_accuracy) / 2`。
没有真实类别时，可由三值反馈计算观察指标，预测任务指标则保持缺失，不能用所选反应概率代替。

exp4的 `probabilistic_task_metrics()` 使用1起点的类别选择、实际二值反馈及任务概率。
`optimal_choice_rate` 表示选择奖励概率最大类别的比例；两类恰好等概率时，两种选择都计为最优。
`mean_reward` 是正反馈比例。`expected_reward` 对每题所选类别的奖励概率取平均，
`optimal_expected_reward` 对每题可获得的最大奖励概率取平均；两者的差为 `expected_regret`。
这些分数描述传入的完整序列，首题计入分母。行为似然使用作答前概率，继续由 `choice_nll_loss()` 计算。

口述分布比较使用模型词表与真实完整编码的并集。真实口述超出模型词表时，对应的预测概率为0，
它仍进入真实口述频率的分母。窗口内的模型平均概率和真实频率使用相同的计分试次；默认窗口
覆盖最近64个实际试次，至少包含16条可计分口述。分布重合度将每种口述内容两侧较小的概率
相加，等于1减去总变差距离；Jensen–Shannon 相似度采用1减去以2为底的散度。两项指标均在
0到1之间，1表示两个分布一致。窗口指标衡量内容出现频率的关系，逐题口述概率衡量具体试次
内容的对应关系，两者分别保留。

配置名 `accuracy_curve_berhu` 的实现是 `losses.accuracy_curve_berhu()`；更短的
`accuracy_berhu()` 是同一实现的兼容入口。旧代码中的
`LOSS_METRIC_ACCURACY_BERHU` 和 `LOSS_METRIC_BERHU` 仍是指向该配置名的兼容别名。

## 设计约束

- 输入是数组、mapping 或本目录定义的数据契约，不读取 YAML/JSON/stream。
- 有效 trial 的类别概率必须有限、非负且归一化；choice index 使用零基索引。
- 函数可以返回指标值、有效样本数和诊断细节，但不决定模型是否“通过”。
- selection tolerance、anchor guard 属于 `optimization/`；验证门槛与结论等级属于
  `evaluation/`。
- 指标定义、trial mask、prediction timing 或聚合顺序的变化都属于科学行为变化，必须单独测试
  和记录。

重复运行结果 schema 由 `simulation/runner.py` 编排；新代码的数值定义应直接从本 package 导入。

包级 `__init__.py` 采用延迟导入：`import src.Bayesian_state.metrics` 不会立即加载各数值子模块；
首次访问公开指标名称时才导入其定义模块。现有包级名称导入保持兼容，新代码仍优先从职责明确的
子模块导入。
