> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Offline Paper Tables

Run from the repository root in an environment with NumPy and matplotlib:

```powershell
python -B evaluation/export_paper_tables.py --input EVALUATION_SUBJECT_DIR --output NEW_PAPER_DIR --window 32
```

The input directory contains `evaluation_manifest.json` and
`evaluation_inputs.npz`. The exporter verifies the input SHA256, manifest array
shapes and dtypes, stable rule IDs, trial alignment and probability invariants.
The output directory must be new and separate from the input tree. Existing
directories are rejected, including empty ones. Partial failed exports remain
in place; choose a new destination for a retry. All processing is offline and
uses only the saved bundle.

## Outputs

- `rule_trials.csv`: one row per trial and stable catalogue rule ID, with all
  six saved conditional catalogue measures, rule family/program and feature order.
- `rule_windows.csv`: means for the whole run and nonoverlapping windows of
  `--window` trials. Start indices are inclusive, stop indices exclusive; indices
  are zero-based. The final window may be shorter. Invalid trials keep their
  positions and are excluded from means. `n_valid` gives the denominator.
- `feature_trials.csv`: bodypart-named marginal probabilities, all 16 dimension-set
  probabilities, oral dimension-use flags and two oral compatibility scores.
- `rule_probabilities.png`: H0/H4/H6 curves for all six measures, plus H42 as a
  conventional reference for four-category tasks when present. Missing requested IDs are listed in
  `summary.json`. Curves retain original mass; catalogue rows can sum below one.
- `feature_usage.png`: four bodypart curves with valid oral reports overlaid.
- `summary.json`: definitions, availability, scored oral count, input/exporter
  hashes and software versions.

Stable H IDs refer to subject-coordinate rule programs. Their physical bodypart
meaning follows `protocol.feature_order`; compare bodyparts before aggregating
rules across subjects. Both CSVs retain this order and the original rule programs.
`trial_index` is the preserved zero-based row position; `trial_id` stores the
original scalar or three-part ID as JSON, alongside the session ID.

## Timing And Oral Scores

Predictive rule belief, presence and execution use pre-current-choice particle
weights. Filtered belief and executed-rule probability use weights conditioned on
the current observed choice. Posterior belief includes the current feedback.
The feature arrays use current-choice-filtered executed content before feedback
conditioning. These quantities follow the observed history, with no future-data
smoothing. Each measure's definition is recorded in `summary.json`.

Feature-set code `c` encodes dimensions by bit position: bit `d` names
`protocol.feature_order[d]`. The marginal for a bodypart sums the probability of
all sets containing that bodypart. The exporter checks these marginals against
`conditional_feature_probability`.

Trial-aligned `oral_reports` entries require a nonblank string in `text` to
establish report availability. Four binary `feature1_use` through `feature4_use`
values then define the reported dimension set. Numeric strings are accepted.
The exact-set score is the probability assigned to that set. Expected
discrete dimension-set Jaccard averages intersection size divided by union size
over the 16 model sets. Two empty sets receive similarity 1. This score measures
discrete dimension overlap, not geometric region IoU. The two scores describe
content compatibility; they do not estimate the probability of generating an
oral report. Oral comparisons remain provisional pending manual review.

Missing, null, empty or whitespace-only `text` marks a missing report regardless
of the feature flags. These entries have blank oral-use and score cells and are
excluded from oral overlays. Nonbinary or incomplete flags also leave oral-use
and score cells blank. Nonblank text with four zero flags represents an explicit
empty coded set and is scored on valid trials. Invalid trials have blank score
cells. Reports are matched by saved trial position; sparse report lists are
rejected. With no saved feature
extension, feature probabilities and scores remain blank and availability is
explicitly marked. A partial feature extension is rejected. Missing catalogue
diagnostics require a new evaluation bundle from an updated binary; semantic
means cannot reconstruct individual rule probabilities.

Validation uses synthetic bundles and no model execution:

```powershell
python -B -m pytest -q tests/test_paper_tables.py -p no:cacheprovider --basetemp .tmp/paper_tables_test_NEW
```
