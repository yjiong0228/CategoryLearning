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


## Model 0923 的独立结构候选

`structural_0923.py::build_axis_pair_overlap_probe_space()` 返回 4 维、4 类的有限候选空间。
它复用 `continuous.py` 的区域构造，在原 116 条之后添加 12 条阈值特征与比较特征重叠的规则，
保留固定比较方向和标签，不枚举标签排列。独立版本与 signature 避免和原目录混淆。
这不是默认 `ContinuousPartition` 空间，尚未用于正式拟合，也不提供新的相似度资源。
验证入口见 `workflows/analysis/probe_model_0923_structure.py`，结果解释见
[0923 规范](../../docs/model_architecture/model_0923.md)。
