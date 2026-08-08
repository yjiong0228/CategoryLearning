# problems/cache

本目录保存 `Partition` 计算成本较高、但由模型结构唯一决定的 hypothesis similarity matrices。
transition modules 通过 `partition.similarity_matrix` 懒加载这些文件。

文件名编码了关键生成条件，例如：

- dimension/category 数量：`d4_c2`
- distance/prototype representation
- region label version
- Monte Carlo sample 数量
- label reversal 编码

## 使用原则

- 不要手工编辑 `.npy`。
- 新 cache 必须由 `Partition` 的生成逻辑产生，并验证 shape、有限值、对称性和对角线。
- 改变 hypothesis enumeration、prototype、region labels 或 similarity 定义时，应提升对应的
  cache/version 标识，不能静默复用旧矩阵。
- 删除 cache 后首次运行可能重新计算，成本可能很高。
- 配置通常不直接引用 cache 路径；应由 partition 根据结构解析。

这些文件是数值加速产物，不是被试数据，也不包含拟合后的参数。
