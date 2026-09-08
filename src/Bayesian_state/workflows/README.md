# 当前模型工作流

本目录按当前 0826 工作需要精简。保留清单：

| 位置 | 用途与保留原因 |
| --- | --- |
| `recovery/design.py` | 读取、验证已注册的恢复实验设计 |
| `recovery/generation.py` | 配置真值模型，调用 simulation 自主生成并保存实验数据 |
| `recovery/run.py` | 恢复阶段调度、恢复运行及来源指纹 |
| `runs/run_model_0826_recovery.py` | 原工作流模块路径的兼容入口 |
| `runs/run_model_0826_belief_transport_counterfactual.py` | 当前 0826 信念迁移方法比较 |
| `runs/run_model_0818_boundary_recovery.py` | 0826 比较仍直接使用其数据读取、引擎设置和种子工具；保留的历史依赖 |
| `analysis/behavior_diagnostics.py` | 通用行为诊断 |
| `benchmarks/benchmark_boundary_geometry.py` | 几何计算性能检查 |

从仓库根目录运行：

```bash
python -m src.Bayesian_state.run_recovery --help
python -m src.Bayesian_state.workflows.runs.run_model_0826_belief_transport_counterfactual --help
python -m src.Bayesian_state.workflows.analysis.behavior_diagnostics --help
```

恢复拟合实现位于 `optimization/recovery.py`；冻结候选评分、预算评价和恢复统计/绘图位于
`evaluation/recovery.py`。工作流只组织这些步骤，不维护另一套 PF 或认知机制。

76 个已无当前依赖的历史 Python 工作流和 3 个历史 shell 批处理入口已删除。
其对应旧命令不再可用，历史结果、配置和文稿保留；需要复现已删除工具时检出原 Git 版本。
删除路径及哈希见 [精简记录](../docs/maintenance/recovery_cleanup_20260908.json)。

模型文稿与配套编译脚本已移到 [docs/model_architecture/](../docs/model_architecture/)。
