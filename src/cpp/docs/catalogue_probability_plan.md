> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Per-Rule Conditional Probability Recording

## Scope

The completed 96-subject batch stores four semantic groups, not separate
catalogue identities. Keep those groups intact and add stable catalogue-ID
diagnostics. Do not map sample/generalization templates to catalogue rules.
Do not modify original results, model learning, RNG use, or optimization output.

For H catalogue rules, add six [T,H] arrays to kept-state runs:

- `catalogue_prior`: predictive-particle-weighted pre-feedback content belief.
- `catalogue_filtered_prior`: current-choice-filtered pre-feedback belief.
- `catalogue_posterior`: current-choice-filtered post-feedback belief.
- `catalogue_predictive_execution_probability`: pre-choice execution mass.
- `catalogue_execution_probability`: execution mass filtered by current choice,
  matching the original Python rule-diagnostic interpretation.
- `catalogue_active_probability`: predictive probability the rule is present.

Rules absent from a particle contribute zero, skipped trials stay NaN. Arrays
do not renormalize away template mass. Predictive/filtered execution use the
sampled content, not a new RNG draw or future-data smoothing. Rule 0/4/6 correspond
to columns 0/4/6; record IDs and program metadata to make this explicit.

## Steps

- [x] Add failing C++ tests for stable per-rule IDs, mass conservation versus
  semantic group 3, missing trials, and identical predictions with logs on/off.
- [x] Record six arrays when keep_states is true. No cost to fitting except
  existing flags; no per-particle JSON logs or dense per-particle output cubes.
- [x] Add efficient C++ mean aggregation, rule IDs, axis/timing metadata, and
  evaluation export separate from four-group state arrays.
- [x] Preserve/validate these arrays in offline evaluation, with standalone
  tests for optional legacy bundles and malformed/new shapes/probabilities.
- [x] Provide explicit export of old fitted parameters to fresh evaluation
  configs; preserve strict old fit-code identity checks. No automatic refitting
  or cohort replay. Record source hashes for the exported parameter origin.
- [x] Build; run native and Python suites plus a small three-condition
  evaluation/plot smoke. Review and document old-result recovery limits.

## Verification

- Native: 36 cases, 267,199 assertions passed. Tests include independent Bayes
  conditioning, stable IDs, semantic-group mass checks, missed trials, and
  identical forecasts with diagnostic logging on/off.
- Python: 197 passed, 2 non-applicable parameterized cases skipped, 25 subtests
  passed. Tests cover all-stage CLI output, optional legacy bundles, shape and
  probability guards, export checksums and canonical input/CSV consistency.
- Exported and validated 96 original best-parameter configurations into
  `results/rule_probability_evaluation_configs_verified`; no optimization or
  full-cohort evaluation was started. Preserve its `source.json` origin record.
- Re-evaluated only 8 trials for S101/S201/S301, 2 workers, smoke budgets, and
  verified new [8,29]/[8,152] outputs and offline preservation/plotting.
- Fresh review found an export-prefix binding issue; the exporter now checks
  original NPZ SHA, input shapes/dtypes/values and original CSV ordering before
  creating files. Focused re-review found no remaining actionable issues.
- Review also tightened presence conservation, predictive mass/presence bounds,
  and the binary/equality invariants of fixed autonomous path 0. All three gaps
  were reproduced with failing tests, fixed, and confirmed by read-only review.
