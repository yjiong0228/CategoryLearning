# Model 1024：完整路径诊断图

当前审阅版本为[2026-10-25 v2](../outputs/fig3/model1024_trajectory_diagnosis_20261025_v2/)。
研究范围是九名exp123开发被试及129、229、317的参数敏感性检查。
[完整报告](../../../results/model_1024/trajectory_report_20261025_v2/report.md)给出定义、结果与边界。

五张PNG对应三项问题：

| 图 | 主要信息 |
| --- | --- |
| `nine_subject_path_distribution.png` | 九人真实后段正确率在全部原自主路径分布中的位置；每点一条路径，菱形为中位数，竖线为5%—95%分位数 |
| `subject_129_path_shapes.png`、`subject_229_path_shapes.png`、`subject_317_path_shapes.png` | 新批次完整代表路径、冻结类别的概率及真人到最近类别中心的距离位置 |
| `parameter_process_sensitivity.png` | 训练相近参数的正确规则在场、作答贡献与后段正确率，误差线为均值的1.96倍蒙特卡洛标准误 |

形态图以同类medoid作为代表，即到同类其他成员平均距离最小的一条完整路径。全部路径保存在
原模拟产物和分析表中。建类用旧模拟前半，每机制64条；类别概率用旧后半64条和新模拟128条
分别检查，误差线为95% Wilson蒙特卡洛区间。颜色和线型区分三种形态，面板标题注明作答机制。
其他图以固定颜色区分作答机制或过程量。横向抖动仅用于分开重叠散点，纵向值来自原始分析表。

Python绘图，7.2英寸宽、300 DPI，仅输出PNG，符合仓库格式要求。五张图使用全部要求的
被试、参数点与路径，图中数据均可追溯到`paths.csv`、`clusters.csv`及`curves/`。v1保留为
审阅记录；v2将形态图图例移至绘图区外，以完整显示317的低表现片段，数值及路径选择相同。

静态图审查器要求SVG/PDF/TIFF，与仓库PNG限定不同；本任务按用户常设要求输出PNG。
300 DPI达到检查器的最低分辨率。抖动随机数仅控制横向排布。最终图另经逐图目视审查。

```bash
MPLCONFIGDIR=/tmp/model1024_figcache python -m CategoryLearning_codes.figures.fig3.model1024_trajectory_diagnosis \
  --analysis results/model_1024/trajectory_diagnosis_20261025_v2 \
  --output NEW_FIGURE_DIRECTORY
```
