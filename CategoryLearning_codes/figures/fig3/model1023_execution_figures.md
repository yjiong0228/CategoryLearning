# Model 1023：执行机制检验图

绘图入口为 `model1023_execution.py`，使用Python和matplotlib。每张图包含九名开发被试，
每32题汇总一次完整正确率。129、229和317置于第一行，以便检查用户提示及训练筛查选出的案例。
全部6,208个有效试次进入图中，横轴保留各被试自己的有效试次编号。

黑线为实际行为；水平点线为均匀猜测基准；竖虚线为拟合前段与评价后段的分界，灰底标出后段。
模型比较图增加三个模型的自主生成均值，以及全部路径的10%–90%分位区间。
该区间描述生成路径之间的差异。每条自主路径从任务起点开始，反馈由其模拟选择产生。

从仓库根目录运行，并为每次输出指定新目录：

```bash
python -m CategoryLearning_codes.figures.fig3.model1023_execution --data PREPARED --output NEW_FIGURES
python -m CategoryLearning_codes.figures.fig3.model1023_execution --data PREPARED --study COMPLETED_FIT --output NEW_COMPARISON
python -m CategoryLearning_codes.figures.fig3.model1023_execution_states --capture COMPLETED_STATES --output NEW_STATES_FIGURES
python -m CategoryLearning_codes.figures.fig3.model1023_execution_recovery --summary COMPLETED_RECOVERY_SUMMARY --output NEW_RECOVERY_FIGURES
```

输出包括PNG、逐题数据、32题汇总数据及来源清单。画布为180×170毫米，分辨率600 dpi。
当前行为图已检查：坐标、拟合分界、标签和数据点均可辨认；绘图不筛选试次。
前期300 dpi草稿保留，当前审阅版本为600 dpi。

`model1023_execution_states.py`逐人绘制实际正确率及三个模型推断的正确规则在场概率、
规则信念和执行概率。单规则模型另列“正确规则信念至少为0.5，同时执行其他规则”的联合概率。
该事件包含两条规则各占0.5时的并列选择，MAP也可出现。图例直接标出阈值；
执行维持机制的归因结合模型比较与状态恢复结果判断。
曲线以冻结的前段参数为条件，使用当前及过去选择，在本题反馈前记录。
参数由整个训练段估计，训练段曲线属于拟合后的回顾性重建；评价段沿用冻结参数。
MIX的作答贡献等于其规则信念，图中显示在场与信念两条状态曲线。
每人的全部试次进入图和源数据表；灰底标记参数冻结后的评价段。

`model1023_execution_recovery.py`使用全部已登记的恢复案例绘制模型选择计数、执行维持参数
真值与估计值，以及生成参数和重新估计参数下的状态评分。后两项使用所有STICKY生成路径，
包含零维持边界。某条路径没有错误选择时，三类错误来源评分为空，并列明该路径。
两种状态重建均包含有限粒子误差。恢复仅覆盖指定生成点、任务序列及每点两次重复。
状态图与恢复图画布均为180×155毫米、600 dpi PNG。
九人的行为比较图和状态图已生成并检查坐标、标签、拟合分界及全部试次的显示。
状态图v2明确标出0.5阈值，源数据与v1一致。恢复图在正式恢复评分完成后生成并检查。

通用nature-figure检查器的字体、字号、尺寸、分辨率、数据筛选和语言后端检查通过。
其SVG/PDF与TIFF要求按仓库“仅PNG”的常设要求处理，产物只保存PNG。
这些图用于开发检验，行为曲线本身不提供规则进入、信念或执行状态的独立测量。
