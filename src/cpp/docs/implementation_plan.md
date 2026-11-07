> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Model 1031 C++ Implementation Plan

**Goal:** Independent portable C++ implementation of the adopted ten-parameter
1031-unified-v2 with three-condition fit/simulate/evaluate and offline PNG output.

**Architecture:** Reuse checked I/O, geometry, data and task-parallel optimization
infrastructure from GenModelOfLatentState_cpp. Replace 0923 rule-only scientific
state and updates with an experience workspace. Do not import old compatibility,
ablation or research-supervisor frameworks.

**Source:** CategoryLearning main 249144de, including model_1031.tex, the unified_v2
YAMLs, ExperienceWorkspacePartition/Likelihood and experience_workspace modules.
User approved targeting the final ten-parameter version only.

## Migration log

- [x] Fetch GitHub and fast-forward CategoryLearning main to 249144de.
- [x] Preserve earlier emission workspace edits including untracked files in stash
  947be25f99ebd0f673182ad0baff53023de6d5f1; do not apply incompatible old edits.
- [x] Inspect scientific specification, adopted version and Python mechanisms.
- [x] Create independent build and copy immutable data/resources.
- [x] Test then implement workspace state, template emission, conditional labels,
  posterior/pairing updates and ADD10 Beta derivative using predictive state.
- [x] Test then implement first completed example, dropout-of-one-feature,
  endpoint-probe rule proposals, deduplication and feedback-dependent admission.
- [x] Implement causal sampled execution with uid persistence; PF choice weighting,
  resampling and autonomous self-feedback; preserve miss and first-score timing.
- [x] Adapt finite-support optimization to ten named parameters, single task layer,
  independent scoring stages, deterministic seeds and input-aware resume/cache.
- [x] Replace rule-index diagnostics with semantic-kind summaries and explicit
  path-local content records; produce behavior and workspace-state PNGs.
- [x] Local deterministic unit tests and small Python/C++ statistical comparison
  for all three conditions. No remote jobs and no full cohort fit in this task.
- [x] README: build on Windows/Linux, three condition YAMLs, full Cond1 invocation,
  input/output semantics, evidence limits and GitHub publication instructions.

## Scientific contract

Parameters: capacity, similarity_width, baseline_probability, error_gain,
failure_decay, global_search, retention, execution_inertia, beta_init, learning_rate.
Each run starts with an uncommitted uniform predictor. Completed experiences can
create templates, partial templates and 29/152 fixed-label catalogue rules.
Capacity may be one. Search uses accumulated failures, not 0923's EMA; no autonomous
exit, structural-edit mixture or functional-distance newcomer transfer remains.
Execution is sampled by belief and previous content uid, not weighted expectation
over all contents. All contents still learn from feedback.

For Cond3, retain q_h(g) and theta_hg(category), not independent marginal labels.
Content retention equals one on full success and fitted retention otherwise;
conditional pairing retention uses fitted retention. New example labels encode
allowed categories per pairing, inheriting the workspace pairing marginal.
Generalizations inherit parent conditional labels and pairing. Beta steps use
10*learning_rate and pre-feedback labels/pairing; template labels update last.

## Validation protocol

Preserve the original Python source as independent reference, with temporary
caches/output confined to this repository. Direct-input template emissions,
gradients, label/pairing posterior, admission and geometry must match reference
within justified floating point tolerances (typically 1e-9; geometry 1e-7/1e-6).
Test boundary cases, zero support, capacity one, duplicate proposals, no general
templates, ties, conditional labels, inactive execution and ancestry copying.
Use multiple independent seeds on short real S122/S206/S314 sequences, comparing
NLL/probabilities and semantic-state summaries with Monte Carlo uncertainty.
No identical cross-language RNG bitstream requirement; no claim that a single
statistical screen proves equivalence for every parameter combination.

Keep per-task seed assignment independent of worker/completion order. Validate
1/multiple-worker outputs and a small native fit -> evaluation -> PNG workflow.
Use dynamic task scheduling for mixed heavy PF/light autonomous evaluation tasks
to avoid the known contiguous-static-block imbalance in the previous C++ version.
