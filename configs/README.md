# 实验配置

- `shared/model_struct/`：通用基础模型（base/default/P/M/PM）。
- `shared/candidates/`：可复用的策略候选空间；是否适用于某实验仍由配置决定。
- `exp123/`：原 configs 中的 condition 1 模型版本、运行/恢复设计、超参数搜索及口述编码适配。
- `exp4/`：概率原型实验配置，以及[Model 1024观测数据接入](exp4/README.md)。
- `exp5/`：原 configs/exp5 配置。
- `meg/`：Task3b的Model 0923 full-v3拟合配置及单被试验证YAML，见[MEG配置说明](meg/README.md)。

`exp123/hyper_cd_cfg/model0923_selected12_full_v3.yaml` 为已有0826结果的固定12人拟合0923。
它引用 `exp123/simulation_cfg/model0923_selected12_full_v3.yaml`，按被试组选择三个共享full-v3引擎，
以完整有效试次估计13项行为参数；名单、数值预算和运行方法见[工作流说明](../src/Bayesian_state/workflows/README.md#model-0923已有0826结果的12人)。

期刊专用配置仍在 `CategoryLearning_codes/Bayesian_model/configs/`，调用同一共享模型。
数据与输出的 `../` 相对路径以所在 YAML 为基准；迁移增加了一级目录，已相应调整。
`dataset` 中的文件名通常相对于 processed_dir，不要再按 YAML 目录拼接。
原结果输出位置保持不变；新运行请覆盖 output_dir 为新的目录，避免覆盖历史产物。

旧路径与新路径的逐文件对应见 `src/Bayesian_state/docs/maintenance/layout_20260908.json`。
未保留旧顶层路径的软链接。历史结果内嵌配置不被改写；复现历史运行使用对应 Git 版本，
或将配置复制到新运行目录并按迁移清单显式更新路径。
