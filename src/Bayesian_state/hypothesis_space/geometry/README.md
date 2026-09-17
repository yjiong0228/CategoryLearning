# 假设几何

本文件夹定义如何评价一个假设，但绝不另行枚举独立的假设空间。

- `boundary.py` 计算刺激到连续类别区域的距离。
- `prototype.py` 为每个连通分量推导一个体积质心原型，并计算刺激到这些原型的距离。
- `discrete_rule.py` 在二值刺激上精确计算奇偶规则。
- `stimuli.py` 提供共享的刺激形状校验。
- `distance_cache.py` 为边界距离提供实例级、精确输入、有界 LRU 缓存；不保存认知状态。

`BoundaryGeometry` 和 `PrototypeGeometry` 使用同一个 `ContinuousHypothesisSpace`。
`DiscreteRuleGeometry` 使用 `DiscreteHypothesisSpace`，并不是连续空间的近似实现。

例如，`x0 <= x1` 的有效坐标质心为 `(1/3, 2/3)`，而
`x0 + x1 <= x2 + x3` 的四维体积质心为
`(23/60, 23/60, 37/60, 37/60)`。

缓存参数、只读返回与生命周期约定见 [假设空间 README](../README.md#model-0826-的等价计算加速2026-09-17)。
