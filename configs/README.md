# 实验配置

- `shared/model_struct/`：通用基础模型（base/default/P/M/PM）。
- `shared/candidates/`：可复用的策略候选空间；是否适用于某实验仍由配置决定。
- `exp123/`：原 configs 中的 condition 1 模型版本、运行/恢复设计、超参数搜索及口述编码适配。
- `exp4/`：概率原型实验的原 configs/exp4 配置。
- `exp5/`：原 configs/exp5 配置。
- `meg/`：预留统一配置位置，目前没有已迁入的运行 YAML。

期刊专用配置仍在 `CategoryLearning_codes/Bayesian_model/configs/`，调用同一共享模型。
数据与输出的 `../` 相对路径以所在 YAML 为基准；迁移增加了一级目录，已相应调整。
`dataset` 中的文件名通常相对于 processed_dir，不要再按 YAML 目录拼接。
原结果输出位置保持不变；新运行请覆盖 output_dir 为新的目录，避免覆盖历史产物。

旧路径与新路径的逐文件对应见 `src/Bayesian_state/docs/maintenance/layout_20260908.json`。
未保留旧顶层路径的软链接。历史结果内嵌配置不被改写；复现历史运行使用对应 Git 版本，
或将配置复制到新运行目录并按迁移清单显式更新路径。
