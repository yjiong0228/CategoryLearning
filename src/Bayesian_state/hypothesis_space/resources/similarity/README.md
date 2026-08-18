# 相似度资源

这些 NumPy 矩阵缓存均匀采样刺激在 boundary geometry 下的固定类别标签两两一致率。
文件名记录假设空间版本、维度、类别数、采样数和二分类带状容差。

这些文件只用于查表。缺失或非标准的矩阵会生成到
`results/cache/hypothesis_space`。运行时生成使用固定随机种子 0，文件名带
`seed0` 后缀，因此不会误用早期未记录采样种子的缓存。载入矩阵时会检查
shape、有限性、取值范围、对称性及单位对角线。

以 `similarity_matrix_shared_hypothesis_space_v1` 开头的文件是当前 continuous partition
使用的资源。其余 19-rule 和 label-expanded 文件仍被保留，是因为冻结的论文模型配置引用了
这些特定历史矩阵；它们不是运行时别名。
