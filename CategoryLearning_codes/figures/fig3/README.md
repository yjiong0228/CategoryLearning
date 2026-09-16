# Fig3：搜索与规则过程初稿

## 六种文章主线的 Fig3 / Fig4 候选预览

[打开六组配对图集](../outputs/fig3/story_previews_20260915_v2/index.html) · [全部配对总览](../outputs/fig3/story_previews_20260915_v2/all_six_pairs.png) · [来源与计算说明](../outputs/fig3/story_previews_20260915_v2/README.md) · [核查记录](../outputs/fig3/story_previews_20260915_v2/QA.md)

对应 `model_0826_further.tex` 的六条文章主线：理解重组、改变时机、个体困难、任务差异、未来预测、自主生成。每条主线提供一张 Fig3 和一张 Fig4，共 12 张单图、6 张并排图，并提供本地图集。用于比较候选叙事与构图；正式分析尚未完成的部分采用明确标注的设计示意。

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
