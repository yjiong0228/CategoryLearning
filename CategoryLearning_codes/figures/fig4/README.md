# Fig4：学习机制图

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
