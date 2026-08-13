# 假设空间分析

本文件夹包含可复现的口述证据审计，并有意与模型执行路径分离。

- `oral_evidence.py` 解析口述报告，计算关系原语、出现率、假设空间、覆盖率和敏感性表格。
- `reporting.py` 写出 CSV 文件、PNG 图、简洁的 Markdown 报告和来源清单。
- `__main__.py` 是命令行编排层。

在仓库根目录运行：

```bash
python -m src.Bayesian_state.hypothesis_space.analysis
```

产物默认写入 `results/hypothesis_analysis`。
