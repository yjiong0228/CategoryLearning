# Fig2：保留原机制图和三任务对照的期刊版

## 合同

本图回答：选择拟合的模型是否在规则内容和变化上与未参与拟合的口述相对应？
结构沿用原版：a完整机制示意；b–d三个任务的个体行为/信念/口述；e–g群体检验。
这是 schematic-led composite，Fig2a 是主导图形，不用四个文字框替代。
使用Python，183 × 230 mm、450 dpi、PNG；保留旧图，代码绘制原生矢量artist后导出PNG。
主图没有报告式总标题、疑问句面板标题或大段脚注；详细定义放入图注。

## 各面板与数据

- a 原样调用 framework_wide.draw_framework 的机制图，保留规则卡片、有限工作区、两种readout、
  反馈、记忆衰减、精度变化与局部/全局搜索。图中的数值来自原有说明性算例，不是新实测结果。
- b–d S102 / S307 / S221，每任务一个示例，分别448/704/832试次。
  三个案例仅用于展示相同分析结构，群体面板仍包含全部12人，未删去口述对应弱的个体。
  每列：32试次行为/模型准确率、全部29或116规则的Q、当前报告对全部规则的相对支持、完整规则兼容度。
  不只展示目标规则，不用最新报告向后填充。全部规则保持原始索引顺序。
  口述热图是O/max(O)，模型热图是Q，分别标明不同量纲，不能将色值直接当作两种相同概率。
  最下方是32试次窗内有效当前报告的平均兼容度（至少16条），以及该窗的均匀规则参照。
- e 所有12人的选择校准，按固定概率区间分箱；黑线是合并结果、彩色小点是个体/分箱。
- f 同一被试的均匀规则、时间平均信念、动态信念三种口述兼容度，12条配对线，无试次级显著性检验。
- g 同类别报告保持/变化时的信念变化率，按间隔匹配，12条配对线。不是因果检验。

e–g均来自已核对的7,924个可评分试次、7,600条有效当前报告。
复用上版build_belief_validation的真实统计定义，改变呈现层次，未改变数值口径。
原Fig2g规划的消融没有完整、公平的三任务结果，因此此版g呈现时序对应；不会用参照条件冒充消融。
必要的参数和轨迹稳定性移入FigS2。

## Figure legend

**Fig. 2 | A resource-constrained model links choices to evolving rule representations.**
**a**, Finite hypothesis search, memory-dependent evidence accumulation, alternative choice readouts,
and feedback-dependent search and revision. Values illustrate one possible binary trial and are not
participant estimates. **b–d**, Individual examples from Tasks 1–3. Top: observed (black) and model
(blue) accuracy, trailing 32 trials. Middle: pre-choice belief over every catalogued rule (teal) and
support from the current verbal report (grey; relative report likelihood, normalized to its trialwise
maximum). Grey missing columns denote absent or invalid reports. Rule indices are unchanged;
the task-defined target is H0 for Task 1 and H42 for Tasks 2–3. Bottom: full-rule report compatibility
for inferred beliefs (teal) and uniform rules (dotted grey), averaged over current reports within
32-trial windows containing at least 16 valid reports. **e**, Choice calibration in five fixed probability
bins; small points denote participant/bin means and the black line pools trials. **f**, Paired report
compatibility under uniform rule weights, each participant's time-averaged beliefs, and dynamic beliefs.
**g**, Mean change in the report-distinguishable belief representation during stable and changing
same-category reports, matched by interval within participant. Colours in e–g indicate task; each line
in f–g is one participant (n=12, four per task). Parameter estimation used full-sequence choices;
reports were not fitted. Reports follow choice and precede feedback; comparison uses pre-choice beliefs.
Compatibility retains partial-report ambiguity and is not exact rule-identification accuracy.
Uniform and time-averaged weights are reference conditions, not fitted alternative cognitive models.
Computational sensitivity and parameter profiles are shown in Fig. S2. Source data accompany the figure.

## 视觉参考

参考 Nature 的[官方图形规格](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
及 Nature Human Behaviour 的[学习模型文章及图序](https://www.nature.com/articles/s41562-018-0297-4)：
把机制示意与个体/群体结果置于同一论证链中，统一图内字号和轴样式，细节进入独立图注。
只参考组织原则，不复制发表文章图形或数据。
