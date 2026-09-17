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
距离。Region geometry 的 active sets 和投影算子沿用原有缓存；实际感知刺激的
boundary distance 另有实例级有界缓存（见下文）。`label_permutation_policy` 还可显式设为
`binary_identity_and_reverse`；它仅支持二分类，并在原规则之后追加标签反转规则。

当使用兼容默认 Dykstra solver 时，`boundary_dykstra_backend` 可为 `auto`、`numba`
或 `python`。`auto` 在 Numba 可用时编译历史循环（保持 100 次投影、更新顺序和
`fastmath=False`），否则回退到 Python；`python` 主要用于数值等价审计。单位立方体约束和
固定知觉统计会在进程内复用；刺激距离只按精确输入缓存，不对刺激进行量化或舍入。

代码有意不提供 `Partition` 这类过于宽泛的名称，也不提供 `.splits`、
`.regions`、`.rules` 和 `.prototypes` 这类重复视图。

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


## Model 0826 的等价计算加速（2026-09-17）

连续 partition 默认启用三个工程优化；模型公式、科学参数、随机数流和全规则似然归一化不变：

- 对 beta **精确等于 0** 且处于单位立方体内的有效刺激，直接返回均匀类别概率。不会把接近 0
  的正数视为 0；仍校验 mode、维度与规则索引。非有限、空批次及立方体外输入沿原计算路径处理。
- `BoundaryGeometry` 按规则 ID、实际感知刺激的 shape 和 float64 精确字节缓存距离；不缓存
  beta、预测概率或学习状态。不同 beta 仍执行原 softmax。Prototype 距离暂不缓存。
- 对上述有效刺激、二值 `category_feedback` 和精确为零的 beta，复用一次反馈似然计算填入
  相同的列；保留全部规则、列顺序和完整矩阵的归一化。部分反馈、其他反馈模型、自定义子类
  和非零 beta 走原求值路径；不缓存随认知更新而变化的似然。

```yaml
partition:
  class: src.Bayesian_state.hypothesis_space.observation_model.continuous_partition.ContinuousPartition
  kwargs:
    n_dims: 4
    n_cats: 2
    zero_beta_fast_path: true
    zero_beta_likelihood_batch: true
    boundary_distance_cache_max_entries: 4096
    boundary_distance_cache_max_bytes: 4194304
```

这四个字段可省略，以上为默认值。单独将 `zero_beta_likelihood_batch` 设为 `false` 可关闭
似然列复用。将 `zero_beta_fast_path` 设为 `false` 且任一缓存上限设为 0 可关闭上述三项加速，
便于复核。上限必须是非负整数。字节上限计算保留的刺激／距离数组内容；Python
字典与对象开销另受条目上限约束，不表示整个 PF 进程只占 4 MiB。

缓存是 LRU，属于单个 boundary geometry；同一次 PF 的粒子共享它，不跨独立 PF 运行共享。
超出单项字节预算的批量请求在复制缓存 key 前直接绕过缓存。规则空间对象、solver、容差、投影
迭代次数或后端改变会使旧条目失效。规则区域遵守原有不可变契约，禁止绕过只读保护原地改写。

缓存返回值由不可变 bytes 支撑，不能直接写入或重新设为可写；修改数组 shape/dtype 也不会破坏
其他调用。调用方如需编辑结果，使用 `.copy()`。没有缓存的结果仍可能可写，应统一按只读方式使用。
通过 `partition.boundary_geometry.distance_cache_info()` 查看命中、未命中、条目及内容字节数，
通过 `clear_distance_cache()` 释放条目。缓存不进入认知状态快照，pickle/deepcopy 后为空；对象
回收时随之释放。锁保护并发读写的缓存账目；几何配置应在计算期间保持固定。

复现、数值与内存验收入口见 [性能检查](../workflows/benchmarks/README.md)。此次验证只支持等价
加速结论，不替代参数恢复、状态恢复或 exp4/exp5 任务迁移。
