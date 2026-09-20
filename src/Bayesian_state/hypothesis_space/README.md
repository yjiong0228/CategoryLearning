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
- 连续模型必须在 `likelihood.distance_mode` 显式选择 `prototype` 或
  `boundary`。Hard assignment 直接使用相应 geometry，不经过 Beta；Beta 只软化
  category distance 得到概率。

Boundary geometry 支持两个具名 solver：

```yaml
partition:
  class: src.Bayesian_state.hypothesis_space.observation_model.continuous_partition.ContinuousPartition
  kwargs:
    n_dims: 4
    n_cats: 2
    boundary_distance_method: dykstra_iterative_projection
    boundary_dykstra_backend: auto
    boundary_distance_tolerance: 1.0e-9
    boundary_projection_iterations: 100
    label_permutation_policy: identity_only
likelihood:
  distance_mode: boundary
```

`dykstra_iterative_projection` 是兼容默认；`kkt_active_set_projection` 枚举 KKT
active constraints。二者都计算 stimulus 到单位立方体内 category region 的欧氏
距离。缓存只保存 region geometry 的 active sets 和投影算子，不缓存 stimulus
distance。`label_permutation_policy` 还可显式设为
`binary_identity_and_reverse`；它仅支持二分类，并在原规则之后追加标签反转规则。

当使用兼容默认 Dykstra solver 时，`boundary_dykstra_backend` 可为 `auto`、`numba`
或 `python`。`auto` 在 Numba 可用时编译历史循环（保持 100 次投影、更新顺序和
`fastmath=False`），否则回退到 Python；`python` 主要用于数值等价审计。单位立方体约束和
固定知觉统计会在进程内复用，但不缓存随 trial 改变的 stimulus distance。

代码有意不提供 `Partition` 这类过于宽泛的名称，也不提供 `.splits`、
`.regions`、`.rules` 和 `.prototypes` 这类重复视图。

## 三值反馈与历史兼容

连续 partition 的 `category_feedback` 用于二值反馈：正确时使用所选类别概率，
错误时使用其补概率。收到 `0.5` 时明确报错，避免将旧的 overlapping evidence
误当作三值反馈的互斥事件概率。`category`、`categorical`、`legacy` 等既有 alias
仍解析为 `category_feedback`，不会自动开启旧三值行为。

新的 Cond3 模型必须配置 `likelihood.feedback_likelihood_mode: hierarchical_pairing`
及 `HierarchicalPairingMemoryModule`。装配阶段会验证这一组合，即使当前数据前缀
只有 `0/1` 反馈也不会退化为二值模式。该 kernel 分别计算选对、同科错误、跨科错误，
并对三种未知反应配对进行联合学习。

仅在单独复现旧 partition / observation-likelihood 计算时，显式指定：

```yaml
likelihood:
  distance_mode: boundary
  feedback_likelihood_mode: legacy_category_feedback
```

这一兼容模式保留历史的 `q(choice)`、`neighbor_mass`、`1-q(choice)` 公式，
不修正其重叠事件；不要把它解释为正确的三值反馈概率模型。直接调用
`partition.calc_likelihood()` 时也必须传入同名参数。该配置会保存在 simulation
provenance 的 `resolved.encoding.feedback_likelihood_mode`，不能用它绕过当前
`StateModel(condition=3)` 的联合记忆检查；复现完整旧模型应使用历史版本。
离散任务不接受这一 continuous-only 兼容模式。

## 相似度资源

连续假设的 assignment-agreement 相似度定义为：在单位超立方体上均匀采样刺激，
并使用调用方显式指定的 `prototype` 或 `boundary` hard assignment 计算标签一致率。
调用 `partition.get_similarity_matrix(kind="assignment_agreement",
distance_mode=...)`；旧 `similarity_matrix` property 仅是 deprecated boundary adapter。

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
