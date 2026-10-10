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

## Model 0923 的 152 条目录

`build_relational_0923_space()` 在原128条后加入两组各12条规则：

- 围绕一个共同部位做两次长短比较，索引128–139。
- 判断两个部位是否相近，再判断第三个部位长短，索引140–151。

例如，后一组可以表达“脖子和腿差不多长，尾巴短”。相近区域直接复用二分类的
`pairwise_similarity_band`，默认容差为0.10；第三个部位的阈值为0.5。
类别顺序为短且相近、短且差距大、长且相近、长且差距大。
差距大的两个方向组成同一类别。`rule_programs.py` 将相近作为一个整体判断进行结构编辑。

完整模型full v2与B0 v3通过 `structural_extension: relational_152_0923` 使用此目录。
原128条的几何、编号与固定标签保持原值；新目录的签名记录版本与容差。
具体解释与验证范围见[规范第17节](../../docs/model_architecture/model_0923.md#17-四分类目录扩展到-152-条2026-10-11)。
