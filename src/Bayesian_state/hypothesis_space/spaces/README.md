# 假设集合

本文件夹只回答一个问题：**有哪些假设？** 其中不包含 likelihood、距离、拟合或缓存逻辑。

- `continuous.py` 直接定义 `ContinuousHypothesisSpec`、
  `ContinuousHypothesisSpace` 和 Task2 假设 family 构造器。
- `discrete.py` 直接定义 `DiscreteHypothesisSpec`、
  `DiscreteHypothesisSpace` 和奇偶规则构造器。
- `regions.py` 定义 `Polytope`、`CategoryRegion` 和 `Hyperplane`。
- `common.py` 保存两种假设空间共享的小型不可变元数据映射。

`ContinuousHypothesisSpace.hypotheses` 是 prototype geometry 和 boundary geometry
共享的唯一规则列表；两种 geometry 都不会另建一份假设空间。


## Model 0923 的重复特征扩展

`structural_0923.py::build_axis_pair_overlap_space()` 返回 0923 B0 v2 采用的 4 维、4 类空间。
原 `build_axis_pair_overlap_probe_space()` 保留为调用同一构造器的兼容入口。
它复用 `continuous.py` 的区域构造，在原 116 条之后添加 12 条阈值特征与比较特征重叠的规则，
保留固定比较方向和标签，不枚举标签排列。独立版本与 signature 避免和原目录混淆。
`ContinuousPartition` 通过显式 `structural_extension: axis_pair_overlap_0923` 接入它；
不带该项时仍使用原目录。新空间的相似度独立生成并按 signature 缓存，未进行真实拟合。
验证入口见 `workflows/analysis/probe_model_0923_structure.py`，结果解释见
[0923 规范](../../docs/model_architecture/model_0923.md)。
