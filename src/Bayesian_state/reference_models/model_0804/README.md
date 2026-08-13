# 0804 参考模型

`core.py` 是冻结的数值实现。`forgetting.py`、`pgas.py` 和 `recovery.py` 是历史恢复实验与
诊断脚本使用的模型专用扩展。它们用于复现和对照，不属于当前 `StateModel` 执行路径。
