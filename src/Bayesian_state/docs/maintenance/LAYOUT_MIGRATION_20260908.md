# 按实验统一目录 — 2026-09-08

## 当前结构

```text
data/
  exp123/{raw,processed,不用}/
  exp4/{raw,processed}/
  exp5/{raw,processed}/
  meg/{raw,processed}/
configs/
  shared/{model_struct,candidates}/
  exp123/{model_struct,simulation_cfg,hyper_cd_cfg,hyper_grid_cfg,specific_models}/
  exp123/oral_coding_adapters.yaml
  exp4/
  exp5/
  meg/                    # 说明与预留位置，无新增运行 YAML
scripts/benchmarks/
docs/
  README.md
  REPOSITORY_MANAGEMENT.md
  history/
  maintenance/
  superpowers/plans/
```

原 data/ 的全部内容归入 exp123，包括原有“ 不用 ”目录（实际目录名不含空格）；
目录名称不能作为删除依据，未删其中任何数据。
原 data_exp4、data_exp5、data_meg 分别移动到 data/exp4、data/exp5、data/meg。
期刊 simulation/recovery 的允许数据根已收紧为 data/exp123，仍拒绝其他实验输入。

通用 base/default/P/M/PM 模型及候选库归入 configs/shared；各版 condition 1 模型、运行与
恢复设计归入 configs/exp123。MODEL_STRUCT 懒加载查找同时覆盖 shared 和 exp123。
期刊专用配置仍在 CategoryLearning_codes/Bayesian_model/configs，引用共享实现。
MEG 只迁移已有数据和 notebook 路径，没有生成未经验证的模型配置。

## 更新与兼容边界

- 更新代码默认路径、预处理/探索 notebook 源单元、当前 README/AGENTS 和相关测试。
- 调整 YAML 中相对路径的层级；原结果输出位置不变。
- 旧技术文档归入 docs/history，保留其当时内容；实施计划仍在 docs/superpowers/plans。
- benchmarks 的唯一脚本移至 scripts/benchmarks，入口为
  `python -m scripts.benchmarks.benchmark_boundary_geometry --help`。
- 删除 tmp 下已确认的 81 个字体/Matplotlib 缓存文件；评价入口改用系统临时目录，
  不再因导入模块而重建仓库 tmp。
- 不保留旧顶层路径软链接。旧命令应改用新路径；历史结果内嵌的配置与 manifest 未改写。
  复现旧运行可检出历史版本；在当前版本继续使用历史配置时，应复制配置并显式更新路径。
- 代码/配置的内容哈希可能因路径字符串变化而不同，不能强行跨版本 resume。

逐文件配置/文档迁移及数据目录映射见 [layout_20260908.json](layout_20260908.json)。
全部数据逐文件清单保存在仓库本地 `reports/repository_maintenance/layout_20260908/data_file_manifest.json`；
该报告未被 Git 跟踪，其 SHA256 记录在上述小型清单中，备份时需一并归档。

## 核验结果

- 移动前、移动后逐文件 SHA256 校验：63,169 个数据文件、24,967,958,413 字节全部一致。
  同时核对文件集合，未发现缺失或额外数据文件。
- 160 个迁移后的原配置 YAML 与 HEAD 比较，除路径字符串外的配置值一致。
  期刊配置另外经过固定数值参考与 CLI 检查。
- Notebook 输出、执行计数及 cell metadata 保持不变，只更新源单元路径。
- `python -m pytest -q tests CategoryLearning_codes/Bayesian_model/tests CategoryLearning_codes/figures CategoryLearning_codes/tests`：470 passed。
  一处原测试写死 ../../results，现检查实际解析后的目录，仍要求原输出位置。
- 新增路径测试覆盖共享/exp123 模型查找及 exp123、exp4、exp5 配置的输入与输出解析。
- S101 的 32 试次、32 粒子、1 repeat 仿真：NLL 0.7377468323010447；
  除 provenance 外的 11 个顶层结果字段与此前 smoke 输出完全一致。
- Fig1 全流程读取 62,720 行、96 人；9 张 PNG 与 v10 逐字节一致。
- Fig2 从现有初步结果重新生成；完整图与框架图均与 v9 逐字节一致。
- 新 benchmark、期刊 recovery、共享 evaluation/autonomous CLI 的 --help 正常。
  缓存路径修改后另行验证两个 evaluation 入口不再创建仓库 tmp。
- 当前导航链接和 git diff --check 已检查。

临时验证产物保存在系统临时目录，未覆盖已有模型或配图结果。未运行正式拟合、恢复或
额外实验的科学模型扩展。数据位置统一不代表四分类/部分反馈/MEG 模型已经验证。

本次未提交 commit；工作区仍包含前面共享模型统一和管理文档的待提交改动。
