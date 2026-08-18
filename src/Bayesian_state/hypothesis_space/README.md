# 假设空间

文件夹按阅读代码时最自然的几个问题组织：

```text
hypothesis_space/
├── spaces/                         有哪些假设？
│   ├── continuous.py              ContinuousHypothesisSpace + Task2 families
│   ├── discrete.py                DiscreteHypothesisSpace + 奇偶规则
│   ├── regions.py                 Polytope 和 CategoryRegion
│   └── common.py                  共享的不可变元数据
│
├── geometry/                       如何对刺激进行分类？
│   ├── prototype.py               自动计算连通分量质心距离
│   ├── boundary.py                区域归属和边界距离
│   ├── discrete_rule.py           奇偶规则的精确计算
│   └── stimuli.py                 共享的输入校验
│
├── observation_model/              证据如何进入 Bayesian 模型？
│   ├── continuous_partition.py    geometry 分派 + Task2 feedback
│   ├── discrete_rule_partition.py rule 分派 + Exp5 feedback
│   ├── base_partition.py          共享 likelihood 流程
│   └── likelihood.py              固定的观测模型计算器
│
├── similarity.py                  相似度与缓存策略
├── analysis/                       离线口述证据审计
└── resources/similarity/           带版本的只读矩阵
```

依赖方向是单向的：

```text
spaces  →  geometry  →  observation_model
```

`ContinuousHypothesisSpace` 直接定义在 `spaces/continuous.py` 中，读者无需知道
内部 schema 文件名就能找到它。Prototype 和 boundary 实现使用同一个 space
对象，不会分别维护平行的规则列表。

## 规范接口

- `ContinuousHypothesisSpace.hypotheses` 和
  `DiscreteHypothesisSpace.hypotheses` 是仅有的规则清单。
- `ContinuousPartition` 公开 `hypothesis_space`、`prototype_geometry`、
  `boundary_geometry` 和 `similarity`。
- `DiscreteRulePartition` 公开 `hypothesis_space` 和 `rule_geometry`。
- `ObservationLikelihood` 将一个已完成 trial 转换为 inference 使用的
  likelihood 向量；它是固定的执行基础设施，不是可插拔认知模块。
- 连续类别区域使用 `CategoryRegion.components`。
- Prototype 通过以下方法从这些 component 推导：
  `prototype_geometry.get_category_prototypes(hypothesis, category)`。

代码有意不提供 `Partition` 这类过于宽泛的名称，也不提供 `.splits`、
`.regions`、`.rules` 和 `.prototypes` 这类重复视图。

## 相似度资源

连续假设的相似度定义为：在单位超立方体上均匀采样刺激，并计算 boundary
geometry 下固定类别标签的一致率。实现位于 `similarity.py`。

随模型发布的带版本矩阵位于 `resources/similarity/`。非标准矩阵写入
`results/cache/hypothesis_space/`，绝不写入 `src/`。运行时计算固定使用 seed 0，
并用带 seed 的文件名与历史未记录种子的缓存隔离；载入时检查 shape、有限性、
概率范围、对称性和单位对角线。

## 扩展假设空间

1. 在 `spaces/continuous.py` 或 `spaces/discrete.py` 中添加有类型的 family。
2. 用 `CategoryRegion` 表示连续类别；若该表示不足，则引入新的显式 geometry 契约。
3. 所有适用的 geometry 都应复用同一个 space。
4. 为标签、概率、prototype、相似度和任务特定 feedback 添加不变量检查。
5. 同步更新最近的 README 以及所有 YAML observation-model class path。

在仓库根目录运行口述证据审计：

```bash
python -m src.Bayesian_state.hypothesis_space.analysis
```

审计结果写入 `results/hypothesis_analysis`，模型执行路径不会导入该分析包。
