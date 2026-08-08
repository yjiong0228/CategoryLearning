# metrics

本目录是 `Bayesian_state` 的共享数值统计层。它只定义“怎样从预测与观测计算指标”，供
`optimization/`、`simulation/` 和 `model_evaluation/` 共同调用，不负责参数选择、模型执行、
文件读写、作图或科学结论判定。

## 依赖边界

```text
optimization ───────┐
simulation ─────────┼──> metrics
model_evaluation ───┘
```

`metrics` 不应导入上述三个工作流 package，也不应依赖某个 inference backend。上层先将结果
转换成 `TrialPrediction` 或 `RunPrediction`，再调用纯数值函数。

## 文件

| 文件 | 职责 |
|---|---|
| `contracts.py` | trial/run prediction 与 scalar metric 的轻量、只读数据契约 |
| `losses.py` | 所有 `loss_metric` 的唯一实现，包括 accuracy-curve BerHu、Brier 与 NLL family |
| `trial_metrics.py` | trial 对齐、rolling/exponential accuracy、family/target-majority 和标准 metric bundle |
| `predictive.py` | choice Brier/NLL、ECE、经验 CRPS 和预测区间统计 |
| `learning_curves.py` | exponential smoothing、accuracy curve level/shape/volatility 指标 |
| `behavior.py` | history kernel、switch、perseveration、win--stay/lose--shift |
| `repeated_runs.py` | 跨 stochastic runs 的边际 choice 预测和 trajectory-distribution 指标 |
| `group_comparison.py` | 以被试等独立单位进行 paired delta、bootstrap 和 FDR 汇总 |
| `aggregation.py` | repeated-run 指标的纯数值聚合 |
| `numeric.py` | 仅供 metric 模块复用的小型数值 helper |

配置名 `accuracy_curve_berhu` 的实现是 `losses.accuracy_curve_berhu()`；更短的
`accuracy_berhu()` 是同一实现的兼容入口。旧代码中的
`LOSS_METRIC_ACCURACY_BERHU` 和 `LOSS_METRIC_BERHU` 仍是指向该配置名的兼容别名。

## 设计约束

- 输入是数组、mapping 或本目录定义的数据契约，不读取 YAML/JSON/stream。
- 有效 trial 的类别概率必须有限、非负且归一化；choice index 使用零基索引。
- 函数可以返回指标值、有效样本数和诊断细节，但不决定模型是否“通过”。
- selection tolerance、anchor guard 属于 `optimization/`；验证门槛与结论等级属于
  `model_evaluation/`。
- 指标定义、trial mask、prediction timing 或聚合顺序的变化都属于科学行为变化，必须单独测试
  和记录。

旧的 `utils/simulation_statistics.py` 暂时保留兼容路径；新代码应优先从本 package 导入。
