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
