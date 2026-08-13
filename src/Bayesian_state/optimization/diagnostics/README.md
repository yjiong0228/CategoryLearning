# 优化诊断

`search.py` 读取已有搜索产物，评价收敛性、plateau 和多目标选择。
`predictive.py` 可能重新运行选中的候选，以评价预测采样与波动性，因此调用方必须显式控制
subjects、repeats 和并行度。
