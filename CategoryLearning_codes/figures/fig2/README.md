# Fig2a framework draft

2026-09-21 完整图的新审阅入口：[12人口述验证版 Fig2](../outputs/fig2/belief_validation_20260921_v2/Figure2_belief_validation.png)。
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
