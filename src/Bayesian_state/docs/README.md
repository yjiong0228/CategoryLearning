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

- [科学故事、Results 逻辑与摘要构想](model_architecture/model_0826_results_logic_bottlenecks_continuity.md)：此前路线作为历史保留，当前以第八部分（2026-09-19）的用户确认英文摘要为准，连接信念解码、快速／缓慢／突变轨迹与补偿机制。版本确定不表示相应科学结论已完成验证。

- [Model 0826 Further：从信念动态理解学习轨迹](model_architecture/model_0826_further.pdf)
  （[TeX 源文件](model_architecture/model_0826_further.tex)）：按 2026-09-19 确认摘要重组全文，说明 Fig1–4 的证据连接、关键概念及每部分写法；旧 50 个方向压缩保留在参考附录。
  编译到新文件使用 `bash src/Bayesian_state/docs/model_architecture/compile_model_0826_further.sh NEW_OUTPUT.pdf`，脚本拒绝覆盖。
  用户要求更新的旧版保存在 `results/model_0826/further_story_20260919_v1/document_before/`。

- [摘要结论对应的分析方案（2026-09-17）](model_architecture/model_0826_bottlenecks_analysis_plan_20260917.md)：按当时 G3 摘要制定，保留瓶颈识别、个体画像、搜索补偿和信念迁移的详细分析约定；当前故事安排以 Further 的 2026-09-19 版为准。

- [Model 0826 Plus：怎样拟合，怎样检验](model_architecture/model_0826_plus.pdf)
  （[TeX 源文件](model_architecture/model_0826_plus.tex)）：2026-09-18 精简修订，先用通俗语言区分
  搜索覆盖、评分精度和状态精度，说明按被试追加计算与停止检查，再介绍恢复、行为、口述、
  自主生成和 Fig2 分工。同步修正 condition 3 已接入的状态；试点不等于正式流程已验证。
  编译到新文件使用 `bash src/Bayesian_state/docs/model_architecture/compile_model_0826_plus.sh NEW_OUTPUT.pdf`，
  脚本拒绝覆盖现有 PDF。此次用户要求修订的旧版已归档到
  `results/model_0826/search_stopping_pilot_20260918/document_before/`。
