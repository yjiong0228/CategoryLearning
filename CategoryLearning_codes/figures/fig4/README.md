# Fig4：学习机制图

## 当前期刊版（2026-09-21）

[Fig4](../outputs/fig4/journal_20260921_v3/Figure4.png) · [独立图注](../outputs/fig4/journal_20260921_v3/legend.md) ·
[整套Fig1–4/S2](../JOURNAL_SET_20260921.md)。四个阶跃改善个案的考虑/信念/执行变化为主体，连接反馈搜索和证据保持。

```bash
python -m CategoryLearning_codes.figures.fig4.build_journal_figure --source results/model_0826/fig34_twelve_subjects_20260921_v1/analysis --output CategoryLearning_codes/figures/outputs/fig4/journal_NEW
```

## 较早讨论稿：每条件4人（2026-09-21）

[12人Fig4](../outputs/fig4/abstract_story_20260921_v3/Figure4_learning_mechanisms.png) ·
[配套Fig3](../outputs/fig3/abstract_story_20260921_v1/Figure3_learning_dynamics.png) ·
[结果与图注](../../../results/model_0826/fig34_twelve_subjects_20260921_v1/README.md)

六个检验的科学问题与指标不变。a/b/d/e纳入全部12人；c/f按原行为规则纳入S102、S215、S307、S328四个阶跃案例，避免只补新点却遗漏新出现的突破路径。同任务案例在f用填心/空心进一步区分；c的空心继续表示近优参数，两个面板的图例分别说明。

```bash
python -m CategoryLearning_codes.figures.fig4.build_abstract_story --source results/model_0826/fig34_twelve_subjects_20260921_v1/analysis --output CategoryLearning_codes/figures/outputs/fig4/abstract_story_twelve_new
```

旧图和源表保留。新增拟合仍有原审计未解决项，图面不是已确认的人群分类或因果补偿结论。

## 当前讨论稿：检验摘要对学习动力学的解释（2026-09-20）

[Fig4：学习机制](../outputs/fig4/abstract_story_20260920_v2/Figure4_learning_mechanisms.png) ·
[Fig3：学习动力学](../outputs/fig3/abstract_story_20260920_v2/Figure3_learning_dynamics.png) ·
[结果与完整图注](../../../results/model_0826/abstract_story_20260920_v2/README.md) ·
[共同设计](../fig3/abstract_story_design.md)

三列对应摘要的三个解释环节：

- **a、d：早期集中与保持。** 较早达标者是否较早偏向简单规则，已有的目标支持是否在错误后保持？
- **b、e：搜索与重新积累。** 广泛搜索是否伴随渐进的行为改善，较弱的证据保持是否伴随更频繁的完整信念重分配？
- **c、f：突破来源。** 对全部两个明显偏向阶跃的案例，比较突破前考虑/支持水平，以及突破前后考虑、信念、适用时的执行、预测正确率的变化。近优参数直接显示在c中。

每列围绕Fig3中的一种现象提出具体检验，不再按反馈、参数或状态输出逐项罗列。九人结果不能确立人群类型；这里的重分配不等于人的信念重置，证据保持参数也不等于独立测得的记忆容量。因果补偿仍需进一步受控比较，不能由相关图替代。

```bash
python -m CategoryLearning_codes.figures.fig4.build_abstract_story --source results/model_0826/abstract_story_20260920_v2/analysis --output CategoryLearning_codes/figures/outputs/fig4/abstract_story_new
```

183 × 198 mm、450 dpi PNG。使用新输出目录；原图保留为下方参考诊断图。

## 九人过程诊断：2026-09-20

[Fig4 主图](../outputs/fig4/nine_subjects_20260920_v3/Figure4_process_diagnostics.png) ·
[Task2 配对状态核查](../outputs/fig4/nine_subjects_20260920_v3/task2_pairing_diagnostic.png) ·
[共同结果说明](../../../results/model_0826/fig34_nine_subjects_20260920_v1/README.md) ·
[Fig3/4 设计说明](../fig3/nine_subject_fig34_design.md)

`build_nine_subject_mechanisms.py` 读取 Fig3 同一份九人状态源表，不另行生成行为：

- a：按上一试次反馈比较下一试次选择前的搜索概率；不跨 session 连接。
- b：目标信念 Q≤.5 / Q>.5 时实际候选更替比例的被试内对照。
- c：Q>.75 的试次中，目标信念、预测正确率与实际正确率。右侧 n 是该人的试次数。
- d–f：全部三个持续执行模型的完整 Q、执行概率、行为和执行规则精度 beta。

这版提供原全文机制假设的过程诊断证据。仍需受控比较才能判断搜索补偿、记忆收益或相似性迁移作用；beta轨迹本身也不证明它导致行为突破。
保留没有明显反馈调节的被试，以及执行已经跟上信念的被试。原拟合问题、近优候选与状态精度检查见共同结果说明。

```bash
python -m CategoryLearning_codes.figures.fig4.build_nine_subject_mechanisms --source results/model_0826/fig34_nine_subjects_20260920_v1/analysis_v2 --output CategoryLearning_codes/figures/outputs/fig4/nine_subjects_new
```

183 × 205 mm、450 dpi PNG；输出目录必须不存在。逐试次源表、被试内分层样本量和输入哈希一起保存。

## 历史：六种文章主线的后续图预览

[打开 Fig3 / Fig4 配对图集](../outputs/fig3/story_previews_20260915_v2/index.html) · [来源与计算说明](../outputs/fig3/story_previews_20260915_v2/README.md) · [核查记录](../outputs/fig3/story_previews_20260915_v2/QA.md)

`build_story_previews.py` 提供六种后续方向：分类错误与泛化、改变与保持的价值、个体困难、任务分支差异、未来预测评价、自主生成。与每种方向的 Fig3 成对阅读；图中明确区分现有结果与后续检验设计。

从仓库根目录统一生成两组图：

```bash
MPLCONFIGDIR=/tmp/categorylearning_story_mpl python -m CategoryLearning_codes.figures.fig3.build_story_previews --version story_previews_YYYYMMDD_v1
```

版本目录必须同时在 `outputs/fig3/` 和 `outputs/fig4/` 下不存在。Fig4 单图保存到 `outputs/fig4/<version>/`；配对图、总览、HTML 图集、共享来源清单、汇总数据与代码快照保存到 `outputs/fig3/<version>/`。单图为 183 × 205 mm、300 dpi PNG。绘图共用 `fig3/story_preview_common.py`，不执行新拟合或模拟。

预测与迁移面板当前展示评价设计，没有留出评分。自主生成面板使用全部 500 条已保存的 S129 PMH 轨迹，包含 9 条未达到历史持续标准的轨迹；仅描述一名被试固定参数下的生成分布，不能代表群体或参数不确定性。
