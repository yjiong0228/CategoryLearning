# Fig4：六种文章主线的后续图预览

[打开 Fig3 / Fig4 配对图集](../outputs/fig3/story_previews_20260915_v2/index.html) · [来源与计算说明](../outputs/fig3/story_previews_20260915_v2/README.md) · [核查记录](../outputs/fig3/story_previews_20260915_v2/QA.md)

`build_story_previews.py` 提供六种后续方向：分类错误与泛化、改变与保持的价值、个体困难、任务分支差异、未来预测评价、自主生成。与每种方向的 Fig3 成对阅读；图中明确区分现有结果与后续检验设计。

从仓库根目录统一生成两组图：

```bash
MPLCONFIGDIR=/tmp/categorylearning_story_mpl python -m CategoryLearning_codes.figures.fig3.build_story_previews --version story_previews_YYYYMMDD_v1
```

版本目录必须同时在 `outputs/fig3/` 和 `outputs/fig4/` 下不存在。Fig4 单图保存到 `outputs/fig4/<version>/`；配对图、总览、HTML 图集、共享来源清单、汇总数据与代码快照保存到 `outputs/fig3/<version>/`。单图为 183 × 205 mm、300 dpi PNG。绘图共用 `fig3/story_preview_common.py`，不执行新拟合或模拟。

预测与迁移面板当前展示评价设计，没有留出评分。自主生成面板使用全部 500 条已保存的 S129 PMH 轨迹，包含 9 条未达到历史持续标准的轨迹；仅描述一名被试固定参数下的生成分布，不能代表群体或参数不确定性。
