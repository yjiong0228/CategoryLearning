# 观测模型

本文件夹回答一个问题：**假设空间怎样接入 Bayesian 模型？**

- `base_partition.py` 负责距离模式校验和共享的 likelihood 流程。
- `continuous_partition.py` 分派 prototype 或 boundary 几何，并实现 Task2 的
  correct/related/incorrect feedback。
- `discrete_rule_partition.py` 分派精确的奇偶规则几何，并实现 Exp5 的二元 feedback。
- `likelihood.py` 是 `p(observation | hypothesis)` 的无状态计算器。
  `BayesianStateEngine` 在 outcome 出现后强制调用它，因此它不注册到 module agenda 中。
  `beta_source: action` 保留历史行为：规则证据使用动作策略当前的动态 beta；
  `beta_source: fixed` 则使用 `default_beta` 作为独立、固定的证据尺度，避免动作
  确信度同时改变规则排序。新模型若要区分“哪条规则更可信”和“按当前规则作答有多确定”，
  应显式选择后一种配置。
- `../similarity.py` 推导固定类别标签下的假设相似度，并管理随代码发布的资源与运行时生成缓存。

观测 partition 使用既有的 space 和 geometry，绝不自行枚举假设，也不复制 region 或
prototype 数组。


Model 0923 B0 v2 在上述共享 `ContinuousPartition` 上显式设置：

```yaml
kwargs:
  n_dims: 4
  n_cats: 4
  structural_extension: axis_pair_overlap_0923
```

该项仅支持 4 维、4 类和固定标签，调用 `spaces/structural_0923.py` 的 128 条目录。
省略该项保持原 116 条四分类 / 29 条二分类空间。规则先验、几何、反馈和搜索使用同一目录；
128 条的相似度按新版本独立生成，不能复用旧 116 条矩阵。详见 [0923 规范](../../docs/model_architecture/model_0923.md)。
