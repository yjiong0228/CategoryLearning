# Fig2 drafts

[12 人口述分布总览](../outputs/fig2/generated_oral_distribution_20261011_v2/oral_distribution_overview.png) ·
[逐人分布图与读图说明](../outputs/fig2/generated_oral_distribution_20261011_v2/README.md)，
2026-10-11，新审阅草稿。

口述分布入口为 `build_generated_oral_distribution.py`，使用已保存的生成口述概率。
每行代表一种完整的口述内容，例如“脖子短 + 尾巴长”。每名被试的上方热图展示模型的
平均预测概率，下方热图展示真实口述的出现频率，两图共用行顺序和 0–1 色阶。
窗口包含当前试次在内的最近 64 个实际试次，至少需要 16 条可计分、完整编码的口述。
模型平均概率与真实频率使用同一组试次。逐试次图另外保留每题的模型概率与单次口述观测。

每张图的曲线给出完整口述空间中的分布重合度：逐种内容取两侧较小的概率，再相加。
数值在 0–1 之间，1 表示两侧分布一致。完整空间包含模型词表与真实口述编码的并集，
词表外口述的模型概率为 0，仍计入真实频率。窗口指标衡量口述内容的频率关系；
逐题口述概率衡量具体试次的内容对应，两项结果可结合阅读。

热图从完整空间中选出最多 24 种主要内容，星号标记模型词表外的内容。
另外四行展示空描述（EMPTY）、模型定义的其他描述（OTHER）、其余词表内容和其余词表外内容。
这些汇总行保留全部概率质量；重合度在汇总之前计算。每人完整编码、可读标签、显示行映射
和滚动分布随图保存。不同被试的特征顺序及显示内容可以不同，逐人图提供对应行标签。

```bash
MPLCONFIGDIR=/tmp/model0826_oral_mpl \
python -m CategoryLearning_codes.figures.fig2.build_generated_oral_distribution \
  --source results/model_0826/oral_readout_alignment_20261011_v1 \
  --output CategoryLearning_codes/figures/outputs/fig2/generated_oral_distribution_NEW
```

输出目录须不存在。保留 12 人总览、每人的窗口图及逐试次图，共 25 张 PNG，
以及完整空间源数组、指标表、图注、QA 与来源哈希。模型预测时点、表达参数和开发样本边界
见下方生成口述评价说明及输出目录中的说明。

[生成口述评价总览](../outputs/fig2/generated_oral_alignment_20261011_v3/oral_alignment_summary.png) ·
[12 人学习轨迹](../outputs/fig2/generated_oral_alignment_20261011_v3/oral_alignment_trajectories.png)，
2026-10-11，新审阅草稿。

生成口述的 12 人评价入口为 `build_generated_oral_alignment.py`。它读取
`results/model_0826/oral_readout_alignment_20261011_v1/` 的新评分表，展示完整报告
的 top-1／top-3 命中率、真实报告的平均预测概率、表达覆盖率及逐人学习轨迹。
选择模型沿用原拟合参数和保存的状态种子；口述使用 Model 0923 第 8 节的共用表达设置，
在作答后、反馈前生成。结构编码采用自动解析，这 12 人属于开发样本。

```bash
MPLCONFIGDIR=/tmp/model0826_oral_mpl \
python -m CategoryLearning_codes.figures.fig2.build_generated_oral_alignment \
  --source results/model_0826/oral_readout_alignment_20261011_v1 \
  --output CategoryLearning_codes/figures/outputs/fig2/generated_oral_alignment_NEW
```

输出目录须不存在，保留 PNG、源表、图注、QA 和来源哈希。
精确指标包含所有可完整编码的计分报告，词表外报告给零概率；覆盖图包括未解析及缺失报告。
轨迹采用 64 个实际试次的滚动窗口，至少需要 16 条可计分报告，缺失题留断点。
本次每人使用 128 粒子、一个固定种子，图中给出描述性指标。

最新期刊版：[Fig2](../outputs/fig2/journal_20260921_v2/Figure2_journal.png) ·
[设计与图注](JOURNAL_FIGURE.md) · [整套Fig1–4/S2](../JOURNAL_SET_20260921.md)。
保留原 Fig2a 原生机制示意及 a/b–d/e–g 结构，补齐三任务与全部12人的检验。
入口为 `build_journal_figure.py`；没有用参照指标代替未完成的模型消融。

2026-09-21 较早审阅入口：[12人口述验证版 Fig2](../outputs/fig2/belief_validation_20260921_v2/Figure2_belief_validation.png)。
使用保存的12人拟合与7,600条当前有效口述，包含内容、动态增益和时序对应；
[读图、边界与复现](VALIDATION_STORY.md)，入口为 `build_belief_validation.py`。
下面保留原框架子图的历史说明。

Current panel: ../outputs/fig2/fig2a_v3/fig2a_framework_draft.png.
Python/matplotlib, 120×90 mm, 450 dpi PNG, prepared as a subpanel draft. Further
reflow may be needed when the actual Fig2b–e dimensions are fixed.

```bash
python -m CategoryLearning_codes.figures.fig2.build_framework --output CategoryLearning_codes/figures/outputs/fig2/fig2a_v4
```

Refuses an existing output directory. Code snapshot and source/output hashes accompany
rendered drafts. No model execution, data changes or simulated findings are involved.

## Figure contract

The diagram explains how a limited set of rules produces choice and how feedback
changes future inference. It inherits the manuscript's causal timing and the reference
article's compact input/inference/recursion organization. It is not a literal copy of
either reference and does not transplant the reference article's hazard-rate mechanism.
Use gray/blue only; no task-specific results or posterior bars.

- Perception includes clipping to [0,1]; noise is subject-specific.
- The three cards illustrate M=3 only, not a universal fitted capacity or identified rules.
- Choice prediction combines perceived stimulus with workspace beliefs and precision.
  The two readout structures are belief mixture and persistent execution; the compact
  p_t(c) symbol deliberately avoids an incomplete list of conditioning variables.
- Choice plus task feedback produce rule evidence L_t, fading belief update and
  dynamic rule precision updates. Dashed returns denote effects on a subsequent trial.
- Search event E and local/global mixture g are distinct controls derived from failure
  history. Candidate selection and belief transport are condensed into the return loop;
  the full formulas remain in Methods. E and g use different lagged summaries, as in
  the manuscript; the schematic does not assert they use an identical timing input.
- PF is shown only as a numerical-inference note, not as a cognitive module.
- No oral/RT measurement enters this model-framework diagram as a fitting input.

## Draft legend

**a, Finite rule search and inference.** Subject-specific perceptual noise transforms
stimuli into internal representations, clipped to the feature range. A limited
workspace maintains active rules, their relative beliefs and dynamic rule precisions.
Choices follow either a belief-weighted mixture or a persistent execution rule.
Observed choices and task feedback update rule evidence, fading beliefs and precision.
Feedback history controls the probability of searching and the local/global scope of
candidate proposals. Dashed connections indicate next-trial effects. Three illustrated
rules are schematic and do not specify fitted capacity or a participant's recovered
state. Particle filtering integrates possible latent cognitive paths using observed
choice likelihoods; it is an inference procedure rather than an additional cognitive
mechanism. The current model specification is binary-category Model 0826.
