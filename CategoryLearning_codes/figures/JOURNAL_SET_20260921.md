# Fig1–4 与 FigS2：期刊版整套审阅

本版恢复原 Fig2 的整体构图与完整 Fig2a，并把全套图重新组织成一条论证链：

**实验与学习差异 → 信念解码的外部验证 → 学习动力学 → 潜在过程与机制 → 模型补充检验。**

主图统一使用紧凑面板、固定的任务与状态配色、原生机制示意和独立图注。
不把大标题、分析清单或方法说明段落放进图内。PNG宽183 mm、450 dpi；全部保留旧版。

| 图 | 本图承担的问题 | 最新图与图注 |
|---|---|---|
| Fig1 | 实验测量了什么，学习者的行为有哪些差异？ | [图](outputs/fig1/journal_20260921_v2/Figure1_journal.png) · [图注](fig1/JOURNAL_FIGURE.md) |
| Fig2 | 从选择得到的信念，能否对应未用于拟合的当前口述？ | [图](outputs/fig2/journal_20260921_v2/Figure2_journal.png) · [图注](fig2/JOURNAL_FIGURE.md) |
| Fig3 | 早期改善、逐渐改善和突变中，信念与行为如何演化？ | [图](outputs/fig3/journal_20260921_v3/Figure3.png) · [图注](outputs/fig3/journal_20260921_v3/legend.md) |
| Fig4 | 相似的行为突变是否伴随不同的考虑、支持与执行变化？ | [图](outputs/fig4/journal_20260921_v3/Figure4.png) · [图注](outputs/fig4/journal_20260921_v3/legend.md) |
| FigS2 | 拟合参数与信念解码的数值分辨率如何？ | [模型补充图](outputs/figS2/journal_20260921_v4/FigureS2_model_resolution.png) · [完整轨迹](outputs/figS2/journal_20260921_v4/FigureS2_belief_trajectories.png) · [图注](outputs/figS2/journal_20260921_v4/README.md) |

## Fig1：以实验和行为现象开篇

恢复刺激和试次屏幕示意。96人的轨迹总览是主要数据面板，三个跨图个案展示行为与报告特征。
学习时间、前后行为变化和报告特征变化补充群体信息，不预先把人群硬分成认知类型。

![Fig1](outputs/fig1/journal_20260921_v2/Figure1_journal.png)

## Fig2：保留原结构，补齐三任务和群体证据

Fig2a 保留规则卡片、有限工作区、两种选择方式、反馈、记忆与搜索的完整图形。
b–d沿用三列任务对照，每列包括行为、全部规则信念、当前口述支持及规则兼容度。
e–g放选择校准、规则内容对应和报告变化对应。均匀/静态参照明确按实际含义命名，
原计划但尚未完成的消融不由其他指标冒充。

![Fig2](outputs/fig2/journal_20260921_v2/Figure2_journal.png)

## Fig3：从曲线形状进入潜在动力学

三个个体的行为和A/Q/E轨迹占据主体；下方连接全部12人的行为形状与持续信念区间。
展示信念何时建立、何时丢失，而不只报告第一次越过某个阈值。

![Fig3](outputs/fig3/journal_20260921_v3/Figure3.png)

## Fig4：分解突变，并连接模型机制

并列比较全部四个行为上更偏向阶跃改善的个体：同样表现改善，A/Q/E的起点及变化可以不同。
下方把反馈后的搜索、证据保持与信念变化连接起来。范围显示数值不确定性；
这些比较仍是模型内部过程的描述性证据，不宣称已证明某模块产生认知补偿的因果效应。

![Fig4](outputs/fig4/journal_20260921_v3/Figure4.png)

## FigS2：必要的模型补充结果

参数画像按资源、选择与搜索模块组织，显示近优参数点改变了哪些估计。
补充图同时呈现完整信念与选择预测的稳定性、口述验证的稳定性、报告变化阈值和突变窗口敏感性。
另给出全部12人的完整信念轨迹，包含不同参数点和随机重放的变化范围。

![FigS2](outputs/figS2/journal_20260921_v4/FigureS2_model_resolution.png)

## 证据状态与复现

Fig1使用96人的62,720试次。Fig2–4与FigS2使用最新12人、7,936试次的已保存拟合；
当前口述检验采用7,600条有效报告。本轮没有改动认知模型、重新拟合或生成假设性实测结果。
S122/S314的口述分歧、S215/S328的数值不稳定均予保留。
正式替代模型比较、模块消融和可用的参数恢复结果仍需另行完成。

新入口分别为各图目录下的 `build_journal_figure.py`，FigS2为 `build_journal_supplement.py`。
源表、实际版本、hash和QA随各输出目录保存；详细重现命令见相邻设计文档。
全套相关测试：60 passed。已检查两次绘制的Fig2与其他最终图的字形、对齐、端点、图例及色标。

版式与图注组织参考 [Nature 官方图形规格](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
及 [Nature Communications 的图与源数据要求](https://www.nature.com/ncomms/submit/how-to-submit)。
这里提供可审阅的期刊版草稿，未替换论文目录中的已确认图。
