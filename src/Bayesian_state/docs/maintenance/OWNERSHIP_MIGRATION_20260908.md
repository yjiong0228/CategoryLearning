# 文档、脚本和报告归属调整 — 2026-09-08

本阶段取消根 docs/、reports/、scripts/、.github/，按所属模型归档内容。

| 原位置 | 当前位置 |
| --- | --- |
| docs/ 的模型说明、计划与审计 | src/Bayesian_state/docs/ 下对应目录 |
| docs/REPOSITORY_MANAGEMENT.md | 当前规则合并至根 README/AGENTS；原文件作为历史记录保留 |
| scripts/run_*、resume_* | src/Bayesian_state/workflows/runs/ |
| scripts/analyze_*、diagnose_*、compare_*、summarize_* 等 | src/Bayesian_state/workflows/analysis/ |
| scripts/build_*、plot_* | src/Bayesian_state/workflows/reports/ |
| scripts/benchmarks/ | src/Bayesian_state/workflows/benchmarks/ |
| reports/model_0826/ | results/model_0826/reports/ |
| reports/model_0813_pf_literature_audit_20260814/ | results/model_dynamic/0813_pf/reports/literature_audit_20260814/ |
| reports/internal_cognitive_smoke_qa/ | results/model_0826/reports/internal_cognitive_smoke_qa/ |
| reports/repository_maintenance/ | results/repository_maintenance/ |
| .github/prompts/CateLearnIntro.prompt.md | 删除；旧模型介绍由当前 AGENTS/README 取代 |

逐文件对应见 [ownership_20260908.json](ownership_20260908.json)。
之前 data_file_manifest.json 的新位置为
`results/repository_maintenance/layout_20260908/data_file_manifest.json`，文件内容及其 SHA256 未变。
前阶段清单仍保留当时路径，不回写历史记录。

## 验证

- 共移动 121 个文件，其中 16 个报告文件逐一核对 SHA256，移动前后完全一致。
- 所有工作流 Python 文件通过语法解析；shell 脚本通过 bash -n。
- 完整测试：`python -m pytest -q tests CategoryLearning_codes/Bayesian_model/tests CategoryLearning_codes/figures CategoryLearning_codes/tests`，470 passed。
- 新模块入口检查：0826 belief-transport counterfactual、0805 accuracy diagnosis、benchmark 的 --help。
- 深层脚本直接调用检查：0826 recovery 转发脚本的 --help。
- 已更新脚本间导入、测试导入、来源指纹使用的脚本路径、Python 根路径计算与 shell 的 cd 路径。
- Shell 中前一阶段残留的 configs/specific_models 已修正为 configs/exp123/specific_models。
- 更新当前文档导航；旧文档和报告快照按原内容保留。

没有运行正式拟合、恢复或 shell 批处理。未再次改写数据或用户已归档的模型结果。
旧模型工作流的历史结果默认路径可能需要根据用户的归档位置显式指定；本阶段没有推断并
批量重写所有历史运行路径。旧 scripts 导入/命令不再保留，使用清单中的新模块路径。

变更尚未提交，前面共享模型与目录整理的未提交工作保留。
