# Model 1024：训练分布约束选参的评价图

本组图检验固定选参规则对后段选择预测、自主分布及完整学习形态的影响。比较保留全部九名
开发被试及三种作答机制，来源为冻结后的共同种子评价。量化网格是两张图的共同结构。

当前审阅稿为[2026-10-25 v2](../outputs/fig3/model1024_generation_selection_20261025_v2/)，
完整表格见[本轮报告](../../../results/model_1024/generation_selection_report_20261025_v2/report.md)。

- `selection_outcomes.png`：四个面板分别显示新减原的后段NLL、后段Energy、完整序列Energy
  与后段正确率，误差线为两倍配对蒙特卡洛标准误。未更换参数的组保留在零处。
- `selection_focus_paths.png`：129、229、317各三种机制的真人块正确率、原估计和新估计的
  自主路径均值及新估计5%—95%逐点分位区间。均值和区间描述预测分布；陡变—保持的完整
  路径事件由报告中的逐路径指标检验。

图使用Python、7.2英寸宽、300 DPI，按仓库规定只输出PNG。颜色与线型在各面板一致，
横轴和区间说明直接标在图中。全部27组比较均进入总览，三名重点被试由上一轮研究事先确定。
图表追溯到`comparisons.csv`、`summary.csv`和`curves/`，不按评价成绩排除案例。
v1保留为审阅记录；v2减少重点个案图的横轴刻度，避免长序列标签拥挤，并注明同点估计的
均值曲线重合。数值、路径和面板范围相同。静态审查器的SVG/PDF/TIFF要求按仓库PNG限定
处理，300 DPI达到其最低分辨率；最终图另经目视核查。

```bash
MPLCONFIGDIR=/tmp/model1024_selection_figcache python -m CategoryLearning_codes.figures.fig3.model1024_generation_selection \
  --summary NEW_SUMMARY --output NEW_FIGURES
```
