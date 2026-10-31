# Fig3：搜索与规则过程初稿

## Model 1031：S128同预算个案（2026-10-31）

[三图结果解释](../../../results/model_1031/subject128_pilot_20261031_v1/report_v3/interpretation.md) ·
[行为与表示](../outputs/fig3/model1031_subject128_20261031_v3/behavior_and_representation.png) ·
[部位对应](../outputs/fig3/model1031_subject128_20261031_v3/feature_use.png) ·
[候选差异](../outputs/fig3/model1031_subject128_20261031_v3/candidate_sensitivity.png)。
全部192题与每候选64条自主路径保留。S128没有数量概括，全部口述均有部位编码；
早期对应有所提高，中段局部概括的支持随参数变化。主展示参数只按选择NLL排序。
复用`model1031_subject118_pilot`入口，编号、题数、特征顺序和候选标签读取实际资料。
v3将图例移出曲线区域；前两版保留，数值完全一致。图为183毫米宽、300 dpi PNG。

## Model 1031：S118经验到规则的首个个案（2026-10-31）

[三张PNG和结果解释](../../../results/model_1031/subject118_pilot_20261031_v1/report_v1/interpretation.md) ·
[行为与表示变化](../outputs/fig3/model1031_subject118_20261031_v1/behavior_and_representation.png) ·
[特征使用](../outputs/fig3/model1031_subject118_20261031_v1/feature_use.png) ·
[候选与数值差异](../outputs/fig3/model1031_subject118_20261031_v1/candidate_sensitivity.png)。
入口为`python -m CategoryLearning_codes.figures.fig3.model1031_subject118_pilot --help`。
图保留256题、各最终参数全部64条自主路径及两次数值重复。状态为选择后、反馈前过滤分布，
参数使用全记录行为；50条未指明部位身份的口述保留原文并留空评分。早期部位范围对应提高，
经验阶段仍短；口述仍待人工复核，图中没有整句口述概率或独立验证结论。
输出为183毫米宽、300 dpi PNG，Python后端；详见图目录内的来源和审阅记录。

## Model 1024未学会阶段口述开发（2026-10-31）

[十张PNG及完整报告](../../../results/model_1024/oral_development_20261031_v1/report_v2/report.md) ·
[结果解释](../../../results/model_1024/oral_development_20261031_v1/report_v2/findings.md) ·
[图表定义](model1024_oral_development.md)。
六人全程图并列目录表达上限、状态支持、相对表达参照的留出收益和其他规则特有的内容。
S306另显示全部头—脖子方向比较，并区分长腿与短腿刺激；S205、S306展示少量口述辅助路径重建的结果。
原始转写和编码保留，当前属于待人工核对的开发比较。最新内部审阅图在
`../outputs/fig3/model1024_oral_development_20261031_v2/`，入口为
`python -m CategoryLearning_codes.figures.fig3.model1024_oral_development --help`。

## Model 1024口述对齐随试次的变化（2026-10-30）

[六人总览](../outputs/fig3/model1024_oral_timecourse_20261030_v1/overview.png) ·
[七张图与读图说明](../../../results/model_1024/oral_alignment_20261030_v2/timecourse_v1/report.md) ·
[绘图定义与复现](model1024_oral_timecourse.md)。
六人4288条记录、17组既有候选均保留。每人四行分别展示具体判断支持、双向部位覆盖、
支持与冲突等结果的构成，以及口述编码范围。曲线为最近32题内有效报告的平均，至少需要8条；
逐题值与编码状态同时显示。候选范围、暂不可评分时段及S129诊断参数身份均明确标注。
图使用选择后、反馈前的过滤状态，口述仍待人工复核。本轮只读取已有评分，输出300 dpi PNG。

## Model 1024状态推断使用范围（2026-10-26）

[概率校准](../outputs/fig3/model1024_state_validation_20261026_v2/calibration.png) ·
[完整历史及整段在场量](../outputs/fig3/model1024_state_validation_20261026_v2/history.png) ·
[阶段交付与使用范围](../../../results/model_1024/state_validation_report_20261026_v2/report.md)。
六个登记参数背景各生成8条新路径，全部保留；选择后、反馈前的过滤和回顾状态分别评价。
图中校准区间按完整路径分层自举，整段在场区间来自共同终点的连贯路径。
逐题明确标签继续保留未通过结论，有限概率范围和整段功能量见报告。
绘图入口为`python -m CategoryLearning_codes.figures.fig3.model1024_state_validation`，
输出183毫米宽、300 dpi的PNG研究审阅稿；真实口述待最终人工检查。

## Model 1022估参与学习过程检验（2026-10-23）

[51张图的总入口](../outputs/fig3/model1023_estimation_reliability_20261023_v1/index.html) ·
[研究结论与复现](../../../src/Bayesian_state/docs/model_architecture/model_1022_compact_refit.md)。
固定简化机制，完成五条模拟序列与六名被试的联合参数搜索、独立复核及冻结评价。
六人图保留每个参数点全部128条自主路径、后期选择预测、逐题口述支持、规则状态和搜索过程。
209、309的两类后期行为评分均改善；规则内容与过程参数的稳定程度分别报告。
口述保留两种既定读法、功能与结构评价、完整报告分母及覆盖限制。
全部图为183毫米宽、300 dpi的PNG审阅稿，来源、尺寸及显示已核对。
入口为`model1023_estimation_reliability`、`model1023_recovery_candidates`和`model1023_frozen_evaluate`，
均位于`CategoryLearning_codes.figures.fig3`。

## Model 1021的H模块简化研究（2026-10-22）

[三人主比较图与源表](../outputs/fig3/model1021_simplification_20261022_v2/index.html) ·
[205搜索与保持对照](../outputs/fig3/model1021_retention_probe_20261022_v2/search_and_retention.png) ·
[205初始支持与精度对照](../outputs/fig3/model1021_entry_diagnostics_20261022_v2/entry_support_and_precision.png) ·
[机制、结果及限制](../../../src/Bayesian_state/docs/model_architecture/model_1021_simplification.md)。
主比较在108、205、306重新估参，呈现全部128条自主路径、后期选择NLL、双读法口述相容性及训练候选。
205的两组定向对照固定原COMPACT参数。精度继承改善该参数点的自主预测，候选机制仍待重新估参和跨被试验证。
误差条为固定参数下的配对蒙特卡洛误差；口述未确定质量需要结合规则目录的表达范围理解。
绘图入口分别为`model1021_simplification`、`model1021_retention_probe`和`model1021_entry_diagnostics`，
均位于`CategoryLearning_codes.figures.fig3`。当前六张PNG为研究审阅稿，保留源表及视觉检查记录。

## Model 0923 重估后的联合证据（2026-10-19）

[RETAIN重估后的四面板图与源表](../outputs/fig3/refit_joint_precision_20261019_v2/index.html)。
上排显示前段、后段和全程的选择NLL及自主Energy差值；下排显示双读法的口述支持、冲突和未确定质量差值。
两方案均保留POOL与SEARCH，差值同时包括机制变化与行为重新估参；同一人的提议方案和重叠时间窗口分别标明。
图中误差为配对模拟标准误，所有完整报告均计入。行为、口述使用相同冻结参数的身份和逐题汇总在绘图前核对。
RETAIN的训练选择收益伴随后段选择与口述支持下降，口述下降主要转为未确定质量；具体机制解释见
[研究记录](../../../src/Bayesian_state/docs/model_architecture/model_0923_research_progress_20261018.md#当前结论与下一项决定)。

入口为`python -m CategoryLearning_codes.figures.fig3.model0923_refit_joint_evidence --comparison precision --output 新目录`。
`--comparison online`要求完整ONLINE重估比较及行为、口述独立核验均已完成；该图的自动等待队列已经启动。
当前已验证的是RETAIN完整结果的282项源表核对和PNG视觉检查，ONLINE数值及图形仍待现有计算完成。

## Model 0923 保留当前刺激的冻结规则对照（2026-10-19）

[四人双读法图与源表](../outputs/fig3/stimulus_controls_20261019_v1/index.html) ·
[方法与完整结果](../../../src/Bayesian_state/docs/model_architecture/model_0923_oral_constraints.md#保留当前刺激的冻结规则对照)。
空心圆表示固定前段末端规则贡献，实心菱形表示每题动态贡献；二者保留当前刺激与共同知觉采样。
BASE和ONLINE的绝对支持同时呈现。208的ONLINE较少出现后段支持下降；207、306的较高支持在前段末端已形成。
107保留整句表达范围限制，208的两种提议属于同一被试的敏感性比较。
入口为`python -m CategoryLearning_codes.figures.fig3.model0923_stimulus_controls --output 新目录`，输出PNG讨论稿。

## Model 0923 逐判断口述覆盖（2026-10-19）

[四人双读法比较图与源表](../outputs/fig3/predicate_alignment_20261019_v2/index.html) ·
[定义、九人补充评价与复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_oral_constraints.md#单项判断与整句联合支持的表达范围)。
每份报告先对全部判断等权平均，再对报告等权汇总；原整句指标保留。
上排展示后段支持、冲突和未确定质量，下排展示前后阶段ONLINE减BASE的支持差及配对模拟误差。
四人均为开发样本，208的POOL与SEARCH是同一人的敏感性分析；参数和状态沿用已有行为拟合。
入口为`python -m CategoryLearning_codes.figures.fig3.model0923_predicate_alignment --output 新目录`，输出PNG讨论稿。

## Model 0923 候选准入机制联合重拟合（2026-10-17）

[13张比较图与全部源表](../outputs/fig3/model0923_episodic_refit_20261017_v2/index.html) ·
[参数范围、结果及复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_episodic_refit.md)。
九人分别拟合BASE、NEW、POOL和DRAW，四方案使用共同起点及相同阶段预算，并交换各自发现的候选参数。
冻结参数后比较独立训练重复、后段选择预测、完整自主路径和正确规则过程。
NEW的平均自主Energy最低，BASE的平均后段选择NLL最低；结果明确显示两类表现的取舍。
绘图入口为 `python -m CategoryLearning_codes.figures.fig3.model0923_episodic_refit`，输出PNG讨论稿。

## Model 0923 历史反馈与候选准入（2026-10-17）

[比较图与全部源表](../outputs/fig3/model0923_episodic_admission_20261017_v1/index.html) ·
[机制、结果及复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_episodic_admission.md)。
九人固定ADD10参数，五方案各生成64条自主路径；训练似然另做独立滤波和更高预算复核。
比较候选历史证据、新旧规则共同评价与随机记忆权重，保留每人的完整路径分布。
绘图入口为 `python -m CategoryLearning_codes.figures.fig3.model0923_episodic_admission`，输出PNG讨论稿。

## Model 0923 精度学习与共同H参数重拟合（2026-10-16—17）

[九张比较图与全部源表](../outputs/fig3/model0923_precision_refit_20261016_v3/index.html) ·
[方案、结果及复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_precision_refit.md)。
九人分别比较四种精度更新方案，每人每方案搜索48个参数点；冻结后完成288次独立滤波与4608条自主路径。
另在109上固定其余参数，使用1024粒子、八个新种子逐值比较学习率。
图中呈现训练候选、后段预测、全部路径及参数边界。更宽的加法学习率范围值得继续检验，
207的条件预测与自主生成仍有明显分离。
绘图入口为 `python -m CategoryLearning_codes.figures.fig3.model0923_precision_refit`，输出PNG讨论稿。

## Model 0923 反馈与精度原型（2026-10-16）

[十张比较图与全部源表](../outputs/fig3/model0923_feedback_prototypes_20261016_v5/index.html) ·
[机制定义、结果及复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_feedback_prototypes.md)。
九人、三套已有参数、八方案的完整自主比较，加上固定正确规则的精度诊断，共8064条路径。
训练选择NLL先做低预算筛查，再对四人的H方案用256粒子、八个独立种子复核。
图中同时显示全过程曲线、全部路径的末段分布及数值预算导致的排名变化。
绘图入口为 `python -m CategoryLearning_codes.figures.fig3.model0923_feedback_prototypes`，输出PNG讨论稿。

## Model 0923 规则发现、保留与使用诊断（2026-10-16）

[五张诊断图与源表](../outputs/fig3/model0923_h_rule_diagnosis_20261016_v3/index.html) ·
[具体操作、结果与复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_h_rule_diagnosis.md)。
九人规则目录核对通过；207、208、308各保留三套已有参数，完成8064条自主诊断路径。
对照一次提供正确规则、持续保留、当题均匀先验、独用正确规则，以及固定候选记录后的可拒绝准入。
另在1152条独用正确规则路径上检查知觉输入与β对即时作答概率的作用。
所有方案和路径均纳入图表；任务答案仅用于明确标记的诊断干预。
绘图入口为 `python -m CategoryLearning_codes.figures.fig3.model0923_h_rule_diagnosis`，使用Python和PNG。

## Model 0923 证据保持机制重拟合（2026-10-16）

[五张比较图与源表](../outputs/fig3/model0923_h_retention_refit_20261016_v1/index.html) ·
[方案、结果与复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_h_retention_refit.md)。
九名被试分别拟合独立保持率、持续累积、共享记忆保持率三套方案；原六人与新增三人分别汇总。
相同预算重新估计全部共同参数，独立确认后冻结，再比较留出选择预测、起点自主生成和训练末尾自主续跑。
图中呈现逐人分差、完整路径分布、参数保持率及训练搜索误差，区分模拟波动和参数解释的限制。
绘图入口为 `python -m CategoryLearning_codes.figures.fig3.model0923_h_retention_refit`，使用Python、PNG及新输出目录。

## Model 0923 参数精度与207机制复核（2026-10-16）

[四张诊断图与源表](../outputs/fig3/model0923_parameter_precision_20261016_v2/index.html) ·
[结果解释与复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_parameter_precision.md)。
固定107、207、305、306的15组原最终候选，用独立种子和更高粒子预算检验排名。
完整重放207两组参数各1024条自主路径，追踪规则发现、保留及正确／错误回答的似然贡献，
并补充搜索与证据保持的固定参数对照。图中区分计算误差、路径分布与参数解释的限制。
全部使用原训练段，未重新拟合参数；局部对照的三个新参数点和原结果分别保存。

## Model 0923 个体参数选择解释（2026-10-16）

[六名被试图文报告](../outputs/fig3/model0923_parameter_selection_20261016_v3/index.html) ·
[目标建议与复现](../../../src/Bayesian_state/docs/model_architecture/model_0923_parameter_selection.md)。
覆盖107、108、205、207、305、306的全部23组共同最终候选；同图比较选择NLL、自主特征分数、
逐试次似然贡献和全部1024条自主路径的区块分布。误差条表示配对计算标准误差，
参数不确定性仍需单独估计。全部图使用原训练段，并复算既有评分；本次没有新增拟合。

## 当前期刊版（2026-09-21）

[Fig3](../outputs/fig3/journal_20260921_v3/Figure3.png) · [独立图注](../outputs/fig3/journal_20260921_v3/legend.md) ·
[整套Fig1–4/S2](../JOURNAL_SET_20260921.md)。三个行为选定个案的对齐轨迹为主体，连接全部12人的行为形状和持续信念区间。

```bash
python -m CategoryLearning_codes.figures.fig3.build_journal_figure --source results/model_0826/fig34_twelve_subjects_20260921_v1/analysis --output CategoryLearning_codes/figures/outputs/fig3/journal_NEW
```

## 较早讨论稿：每条件4人（2026-09-21）

[12人Fig3](../outputs/fig3/abstract_story_20260921_v1/Figure3_learning_dynamics.png) ·
[12人Fig4](../outputs/fig4/abstract_story_20260921_v3/Figure4_learning_mechanisms.png) ·
[新增数据与结果解读](../../../results/model_0826/fig34_twelve_subjects_20260921_v1/README.md)

新增S104/215/328，合计12人、7936试次；原摘要主线、行为形态定义、阈值和窗口不变。按相同规则重新选出S122、S206、S215，Fig4突破比较扩展到全部4个明确阶跃案例。原九人状态只读复用；只对新增三人导出固定参数状态，不重新拟合模型。

```bash
python -m CategoryLearning_codes.figures.fig3.nine_subject_states --config CategoryLearning_codes/figures/fig3/twelve_subject_config.json --reuse-states results/model_0826/fig34_nine_subjects_20260920_v1/states --output results/model_0826/fig34_twelve_new/states
python -m CategoryLearning_codes.figures.fig3.nine_subject_analysis --config CategoryLearning_codes/figures/fig3/twelve_subject_config.json --states results/model_0826/fig34_twelve_new/states --output results/model_0826/fig34_twelve_new/base_analysis
python -m CategoryLearning_codes.figures.fig3.abstract_story_analysis --source results/model_0826/fig34_twelve_new/base_analysis --states results/model_0826/fig34_twelve_new/states --output results/model_0826/fig34_twelve_new/analysis
python -m CategoryLearning_codes.figures.fig3.build_abstract_story --source results/model_0826/fig34_twelve_new/analysis --output CategoryLearning_codes/figures/outputs/fig3/abstract_story_twelve_new
python -m CategoryLearning_codes.figures.fig4.build_abstract_story --source results/model_0826/fig34_twelve_new/analysis --output CategoryLearning_codes/figures/outputs/fig4/abstract_story_twelve_new
```

既有`nine_subject_states`和`nine_subject_analysis`模块名及九人默认配置保留。新增可选`--config`指定队列；状态导出的`--reuse-states`核对来源候选、数值预算和种子后建立相对只读复用链接，只计算缺少的状态。不得删除被链接的旧结果。已完成的同配置导出再次运行只做核对，不新增PF或重写完成记录。分析与绘图仍要求新输出目录。

## 当前讨论稿：按确定摘要组织学习动力学（2026-09-20）

[Fig3：三种行为轨迹与内部过程](../outputs/fig3/abstract_story_20260920_v2/Figure3_learning_dynamics.png) ·
[Fig4：逐项检验摘要中的机制](../outputs/fig4/abstract_story_20260920_v2/Figure4_learning_mechanisms.png) ·
[读图与结果](../../../results/model_0826/abstract_story_20260920_v2/README.md) ·
[设计与指标](abstract_story_design.md)

本版以“为什么较早学会、缓慢改善、停滞后突然改善”为主线。Fig3a先用真实行为分开学习时间与变化方式，b–d再展示行为规则选出的S122、S206和S307，逐行对照行为、目标规则信念、搜索范围与完整信念重分配。Fig4检验这些过程是否符合摘要提出的简单规则、判断保持、搜索与重建，以及突破前已有信念等解释。没有按内部状态挑选符合预期的案例，也没有把九人硬分成三类。

复用下方九人的现有状态，不重新拟合认知模型或追加模拟。描述性的行为模型比较不调整刺激难度，因此不能作为正式认知分型。Fig4的相关关系也不等于补偿的因果证据；报告保留不支持摘要具体机制的结果。

```bash
python -m CategoryLearning_codes.figures.fig3.abstract_story_analysis --source results/model_0826/fig34_nine_subjects_20260920_v1/analysis_v2 --states results/model_0826/fig34_nine_subjects_20260920_v1/states --output results/model_0826/abstract_story_new/analysis
python -m CategoryLearning_codes.figures.fig3.build_abstract_story --source results/model_0826/abstract_story_new/analysis --output CategoryLearning_codes/figures/outputs/fig3/abstract_story_new
python -m CategoryLearning_codes.figures.fig4.build_abstract_story --source results/model_0826/abstract_story_new/analysis --output CategoryLearning_codes/figures/outputs/fig4/abstract_story_new
```

输出目录必须为新目录；只改构图时复用现有`abstract_story_20260920_v2/analysis`。PNG宽183 mm、450 dpi；源表、输入哈希和代码快照随图保存。下面九人全轨迹和近优参数图改作参考诊断图，原文件保留。

## 九人完整过程图：2026-09-20

[Fig3 主图](../outputs/fig3/nine_subjects_20260920_v2/Figure3_nine_subjects.png) ·
[近优参数核查](../outputs/fig3/nine_subjects_20260920_v2/belief_candidate_sensitivity.png) ·
[Fig4 主图](../outputs/fig4/nine_subjects_20260920_v3/Figure4_process_diagnostics.png) ·
[结果解读](../../../results/model_0826/fig34_nine_subjects_20260920_v1/README.md) ·
[设计说明](nine_subject_fig34_design.md)

使用两批已完成的原始拟合：S102/S118/S122、S307/S314/S315、S206/S221/S222，分别为 Task1/2/3。
固定原提名参数补做状态输出，每人8次PF；另对一份原近优候选做4次敏感性重放，每次128粒子。
调用共享核心，无重拟合或机制改动；6272个真实试次全部保留。

图3上部是九人的完整过程；下部对照较高考虑、较高信念与行为达标的操作性时点，以及全部98个64试次块的行为与信念。
主图时点比较已显示近优候选差异。六个混合读出模型不虚构唯一执行规则。配套图的阴影为PF种子标准差，不是人群或参数置信区间。
“A/Q连续16次>.5”与“过去64次正确率>.9”采用不同标准；时间差不等于直接测得的认知等待时间，首次达标也不等于此后不再回退。

从仓库根目录运行，输出目录均使用新名称：

```bash
python -m CategoryLearning_codes.figures.fig3.nine_subject_states --output results/model_0826/fig34_new/states
python -m CategoryLearning_codes.figures.fig3.nine_subject_analysis --states results/model_0826/fig34_new/states --output results/model_0826/fig34_new/analysis
python -m CategoryLearning_codes.figures.fig3.build_nine_subject_figures --source results/model_0826/fig34_new/analysis --output CategoryLearning_codes/figures/outputs/fig3/nine_subjects_new
python -m CategoryLearning_codes.figures.fig4.build_nine_subject_mechanisms --source results/model_0826/fig34_new/analysis --output CategoryLearning_codes/figures/outputs/fig4/nine_subjects_new
```

只调整图面时复用现有 `results/model_0826/fig34_nine_subjects_20260920_v1/analysis_v2`，无需重复状态重放。
`nine_subject_config.json` 保存被试、数值预算、窗口和阈值；状态重放不覆盖已有种子文件，分析与绘图拒绝已有输出目录。
复核和局限见结果目录 `QA.md`。模型比较优势、早期留出预测、参数恢复和因果补偿未在本次验证。

## 瓶颈个案分析预演：2026-09-17

[首轮结果与解读](../outputs/fig3/bottleneck_cases_20260917_v2/README.md) · [完整时间轴](../outputs/fig3/bottleneck_cases_20260917_v2/belief_bottleneck_timelines.png) · [阶段画像](../outputs/fig3/bottleneck_cases_20260917_v2/stage_learning_profiles.png) · [局部片段](../outputs/fig3/bottleneck_cases_20260917_v2/candidate_transition_closeups.png) · [核查记录](../outputs/fig3/bottleneck_cases_20260917_v2/QA.md)

对应[摘要分析方案](../../../src/Bayesian_state/docs/model_architecture/model_0826_bottlenecks_analysis_plan_20260917.md)的前两步：读取 S129/S229 已保存的全部 16 次 PF 输出，提取每试次的目标规则考虑概率 A、条件支持 C、总体信念质量 Q，以及适用时的执行概率 E，并与原始行为、当前口述和按类别保留的口述状态对齐。没有重新拟合、模拟或聚类。这里的两个跨任务案例不能支持人群分组或补偿机制结论。

```bash
MPLCONFIGDIR=/tmp/categorylearning_bottleneck_mpl python -m CategoryLearning_codes.figures.fig3.build_bottleneck_cases --output CategoryLearning_codes/figures/outputs/fig3/bottleneck_cases_YYYYMMDD_v1
```

输出目录必须是新目录。参数、来源和探索性筛选阈值在 `bottleneck_case_config.json`；数据提取在 `bottleneck_analysis.py`，三张图与本组案例的解释性报告在 `build_bottleneck_cases.py`。报告中的逐试次口述解读针对当前 S129/S229 来源；更换案例或来源后须重新审阅，不能直接沿用。PNG 均为宽 183 mm、450 dpi；不生成其它图像格式。

保存完整逐试次表、全规则表、PF 重复表、阶段摘要、候选区间及阈值敏感性表，连同输入哈希、版本、种子、原始评分 mask 和代码快照。C 按平均 Q / 平均 A 计算，阶段内使用 ΣQ / ΣA；不先平均各 PF 的比值。S129 没有唯一执行规则，E 记为不适用。当前低 A 不等于历史上从未考虑；阈值区间只供复核，不作为已识别的认知事件或人群标签。

v2 修正了首轮 v1 阶段画像的轴标签重叠；两版数值相同，v1 保留。该预演与下方历史故事图集、Fig3 v4 分开存放，尚未选定为正式 Fig3。

## 六种文章主线的 Fig3 / Fig4 候选预览

[打开六组配对图集](../outputs/fig3/story_previews_20260915_v2/index.html) · [全部配对总览](../outputs/fig3/story_previews_20260915_v2/all_six_pairs.png) · [来源与计算说明](../outputs/fig3/story_previews_20260915_v2/README.md) · [核查记录](../outputs/fig3/story_previews_20260915_v2/QA.md)

对应 `model_0826_further.tex` 在 2026-09-15 旧版中的六条主线：理解重组、改变时机、个体困难、任务差异、未来预测、自主生成。每条提供一张 Fig3 和一张 Fig4，共 12 张单图、6 张并排图及本地图集。用于比较候选叙事与构图；未完成分析采用明确标注的设计示意。2026-09-19 的 Further 已以用户确认摘要重组为学习轨迹主线，本图集继续作为历史参考，不自动成为新框架的结果图。

```bash
MPLCONFIGDIR=/tmp/categorylearning_story_mpl python -m CategoryLearning_codes.figures.fig3.build_story_previews --version story_previews_YYYYMMDD_v1
```

`--version` 必须是新目录名，且 `outputs/fig3/<version>` 与 `outputs/fig4/<version>` 均不存在。命令一起生成两组图；Fig4 实现在 `fig4/build_story_previews.py`，数据读取和样式共用 `story_preview_common.py`。单图 183 × 205 mm、300 dpi、PNG；并排图和总览为屏幕预览。

输入为 Fig1 v10、Fig2 complete v15、Fig3 v4 的已保存来源，以及 S129 PMH 的 500 条历史自主轨迹。命令只读取和汇总这些来源，不重新拟合或模拟。每个面板标明真实数据、已有拟合、历史模拟或设计示意；导出来源哈希、逐面板清单、汇总 CSV 和代码快照。后缀预测、迁移与公平机制对照仍是待开展分析。此候选图集不替代下方 Fig3 v4。

## 当前：v4，真实报告变化片段

```bash
python -m CategoryLearning_codes.figures.fig3.build_revision_fig3 --output CategoryLearning_codes/figures/outputs/fig3/fig3_v4
```

三个 panel：a 为两个真实片段，b 为跨事件对齐分析空位，c 为后续行为分析空位。去掉上一版的 replaced fraction、完整规则热图和跨被试原始 trial 热图。PNG 183 × 185 mm，450 dpi。

a 使用 S129 trials 132–144（完整记录256 trials中的13个），S229 trials 985–999（完整记录1088 trials中的15个）。窗口内所有反馈和搜索值均展示。口述轨道限定同一选择类别，分别有9个 choice 2 报告和6个 choice 3 报告；这样不会把不同类别的报告内容差异误认为同类别修订。轨道只画实际报告点，不作 forward-fill 或相邻报告连线。

S129：t136“脖子长。”至t138“尾巴长。”；S229：t988“腿长，脖子短。”至t992“腿长，尾巴短。”。同时保留窗口中其他内容，例如S129 t133“脖子短。”，不假装整个旧阶段完全稳定。浅色带是两个同类别报告之间的间隔，虚线仅表示第一次报告新内容的试次，不是精确认知切换时刻。中文引用来自processed文本，英文轨道标签为直译。

这些例子按文本可读性和同类别变化选择，没有按搜索峰值排序；不代表全体修订事件、不建立类别变化效应，也不预设搜索在报告变化前升高。不能把2个例子作为已完成的事件检验。模型使用16次在线PF边际量平均，参数仍为全序列拟合；没有参数不确定性、显著性或独立预测结论。后续需完整语义事件表、匹配对照、时间自相关处理、近优参数状态稳定性；b/c留空。

源数据保存完整逐trial模型量、完整规则质量，以及两段未经删选的trial子表和同choice口述表；manifest记录输入哈希、窗口、文本映射、选择依据与数值重复种子。

## 历史：v1/v2，全记录过程预览

仅使用已有的 S129（condition 1 / Task 1）和 S229（condition 2 / Task 3）PMH 输出；不拟合新模型、不生成替代数据。c、d 留空并标记 Pending。

```bash
python -m CategoryLearning_codes.figures.fig3.build_fig3 --output CategoryLearning_codes/figures/outputs/fig3/fig3_v1
```

输出目录必须不存在。PNG 为 183 × 225 mm、450 dpi。附逐试次 CSV、完整规则质量 CSV、被试摘要、输入 SHA256 与随机种子、绘图代码快照。

## 阅读方式

- a：左右分别为两名被试，真实 trial 轴独立。自上而下为保留/局部/全局搜索概率、工作空间替换比例、在线规则信念、口述证据。规则编号为各任务原始目录的零起始索引；29 与 116 条目录编号不能跨任务直接比较。模型与口述均保留完整目录、不筛选有利规则。
- b：三个连续概率热图，每行一人，统一真实 trial 尺度。S129 的记录在 256 结束，后面灰色留空；没有把记录末尾称为已经掌握。每张图白色=0、对应饱和色=1。三个热图不使用 argmax 分类。
- c：拟以人工核对的同类别口述修订为零点，对照匹配的未修订时点；当前没有可靠事件表，所以只有坐标意图和空位。
- d：拟检验修订前搜索与修订后行为变化；当前没有完成事件分析及未来结果验证，所以不画点、不画拟合线。

## 计算与证据边界

所有模型量均从保存的 16 个 PF run 的 pre-choice 边际量作等权平均，不用代表 seed 替代，也不使用全历史 genealogy 平滑结果。16 个 run 是数值重复，不是 16 名被试。图中没有置信区间或显著性检验；CSV 的 seed SD 只描述数值重复差异，不是参数不确定性。

每粒子的保留倾向为 1-p，局部搜索为 p(1-g)，全局搜索为 pg，再按 PF 权重边际化；p 为搜索事件概率，g 为条件搜索范围。保留不表示信念完全不变，搜索概率也不是已经发生的替换事件。第二轨道是 PF 边际化的已实现替换比例，两者分别呈现。

口述 sigma=0.05，采用原有 category-specific carry-forward 口述状态，但仅在 valid_oral_report 为真的 trial 显示；其余列灰色。有效报告处可能仍带有其他类别的历史报告信息。图中颜色变化不能直接算作经核验的口述策略修订，也不说明即时报告完整描述了全部规则。模型使用作答前在线信念，口述遵循已有报告时点，并非假定两个通道严格同时测量。

现有参数在完整选择序列上估计，因此图是样本内描述。S129 为混合读出，不虚构其唯一执行规则；两例均展示共同可用的规则信念。图不支持群体条件差异、心理类型划分、因果机制或留出预测结论。近优参数间的状态稳定性尚待验证。

## 初稿取舍

先展示真实已有量，把经过核验的修订事件和未来行为检验留空。完整目录热图保留了小概率背景，缩小阅读时细节有限；逐规则源 CSV 保留供后续改成有明确语义的规则家族展示。未使用人工或模拟数据补齐 panel。

## Model 0923 的拟合后诊断草稿

当前逐人标准报告使用 `python -m CategoryLearning_codes.figures.fig3.model0923_evaluation`。
该入口默认执行[逐被试评估v2](../../../src/Bayesian_state/evaluation/README.md#model-0923逐被试评估-v2)，
每人固定输出 `behavior_alignment.png`、`oral_alignment.png`、`behavior.png`、`representations.png`、`diagnostics.png`，另有两张总体图，共 `5S+2` 张PNG。
行为与口述分别有独立指标及源表；没有口述数据时，同名口述图页说明原因。v1三图契约保留。
诊断图呈现ESS、重采样、搜索与候选进入事件，以及每人的配对粒子预算检查；逐图源表保留有效试次顺序。
历史12人的初始后分析及以下26张图保留原报告与用途。

`model0923_postfit.py` 提供12人参数端点图、独立粒子种子的评分对照，以及逐人的行为与表征图。
行为面板比较真实序列上的条件预测和自主生成的学习曲线；表征面板区分候选可访问性、
规则信念、作答贡献、有效参与数量及精度。条件3另显示反馈后的配对信念。

图中的隐藏状态区间以固定入选参数为条件；自主行为阴影为模拟轨迹的逐点10%–90%范围。
所有有效试次及完整规则目录保留在配套源表中。当前图为分析报告草稿，使用183毫米宽、
300 dpi的PNG。运行方法与证据边界见[工作流说明](../../../src/Bayesian_state/workflows/README.md#model-092312人拟合后的第一阶段分析)。

2026-10-13完成的12人分析入口为[完整图文报告](../../../results/model_0923/exp123_selected12_postfit_20261013_v1/report.html)。
[图文件夹](../outputs/fig3/model0923_postfit_20261013_v1/)保留26张PNG：
`cohort_behavior.png`和`parameter_audit.png`是总体图；每个`subject_编号/`下各有
`behavior.png`和`representations.png`。同一结果目录的`findings.html`提供文字摘要。
