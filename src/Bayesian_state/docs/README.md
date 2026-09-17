# 项目文档

- [仓库管理](../../../README.md#仓库管理)：当前目录职责、清理和版本冻结规则。
- [目录迁移记录](maintenance/LAYOUT_MIGRATION_20260908.md)：数据/配置新位置、核验及兼容边界。
- `maintenance/`：清理、迁移的清单与验证记录。
- `history/`：早期模型工作流、设计说明与冻结记录；保留原始内容，内部旧路径按当时版本理解。
- `superpowers/plans/`：实施计划历史。计划不取代当前 README，也不表示其中的工作均已完成。

模型接口看 `src/Bayesian_state/README.md`，期刊流程看
`CategoryLearning_codes/Bayesian_model/README.md`，避免在多份文档中重复维护完整模型说明。

- [文档/脚本/报告归属迁移](maintenance/OWNERSHIP_MIGRATION_20260908.md)：当前分组和旧路径映射。

- [模型架构文稿](model_architecture/)：当前及历史模型定义，0826 来源为 model_0826.tex；配套编译脚本就近保存。

- [Condition 3：类别、按键与配对学习](model_architecture/model_0826_condition3_design.md)：Task2 编码约定、联合配对学习实现、运行入口和验证范围；正式拟合及恢复实验尚未开展。

- [Model 0826：从文献论证到科学问题](model_architecture/model_0826_scientific_questions_20260916.md)：基于人类认知文献地图，评估反馈利用、学习瓶颈与知识继承等八个问题，区分竞争解释、关键证据、模型边界和三条候选文章主线。

- [科学故事、Results 逻辑与摘要构想](model_architecture/model_0826_results_logic_bottlenecks_continuity.md)：保留此前路线与摘要记录；当前重点为第七部分 G3“验证个体信念解码 → 区分学习瓶颈 → 检验有限资源下的认知补偿”，包含模块设计依据、补偿关系及 Fig3/4 的证据逻辑。中英文摘要中的设想结果与现有证据分开标注。

- [摘要结论对应的分析方案](model_architecture/model_0826_bottlenecks_analysis_plan_20260917.md)：以用户确认的 G3 标题与英文摘要为准，逐项说明瓶颈识别、个体画像与分组、搜索补偿和信念迁移需要的分析、现有输出字段及优先执行顺序。

- [Model 0826 Plus：检验与论文呈现方案](model_architecture/model_0826_plus.tex)：中文内部讨论稿，区分数值/恢复/行为/口述/生成检验，说明最新 Fig2 a–g 与补充图分工，以及已完成和待检验内容。
