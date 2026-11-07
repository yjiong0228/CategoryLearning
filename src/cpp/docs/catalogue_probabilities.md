> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Per-Catalogue Diagnostics

Offline evaluation preserves all supplied NPZ arrays in `arrays.npz`. The six
catalogue diagnostics use a separate `[T,H]` axis: `T` retained trial slots and
`H` stable Model 1031 catalogue IDs. Existing semantic `STATE_KEYS` retain four
columns. Templates keep their own representation identities and probability mass.
No new plot or within-catalogue normalization is applied.

## Identity and Validation

The optional extension is complete when it contains:

- NPZ `catalogue_rule_ids`: `int64[H]`, ordered `0..H-1`.
- Manifest `catalogue_metadata`: H entries with `index`, `label` (`H0`, `H1`, ...),
  a nonempty `family` string and a nonempty `program` object.
- Manifest `catalogue_axis`: `rule_count: H` and nonempty timing/conditioning
  strings under each of the six names below; `axis`, when supplied, equals 1.
- Manifest `catalogue_state_keys`: all six names exactly once.
- All six `[T,H]` arrays under both `conditional_` and `autonomous_diagnostic_`.
  `parameters.capacity` supplies the active-count bound.

Thus `H0`, `H4`, and `H6` are columns 0, 4, and 6, when those IDs exist. Columns
represent catalogue programs, independent of workspace slots and path-local UIDs.
Every valid row must be finite, with each entry in `[0,1]`. Missing rules on a
valid trial contribute zero; every `valid=false` row must be entirely NaN.
Checks include valid trials excluded by `score_mask`. Numerical comparisons use
absolute tolerance `1e-9`; validation never clips or rescales values.
Interpret program feature indices using the same manifest's
`protocol.feature_order`, not a hard-coded body-part ordering.

## Definitions

For trial t and particle i, let `w_i^-` be its predictive weight before the
current choice. Choice filtering gives
`w_i^c = w_i^- L_i(a_t) / sum_j w_j^- L_j(a_t)`, where `L_i(a_t)` is that
particle's likelihood of the observed response. This normalization makes the
choice-filtered particle weights sum to one.

Let `b_ih^-` and `b_ih^+` be rule h's content belief before and after current
feedback, respectively. Let `I_ih` indicate rule presence and `E_ih` indicate
that the particle actually executed rule h. An absent rule has zero belief and
zero indicators. The quantities below describe the current trial:

| Array Suffix | Formula | Conditioning and Timing |
| --- | --- | --- |
| `catalogue_prior` | `sum_i w_i^- b_ih^-` | Predictive belief, before current choice/feedback |
| `catalogue_filtered_prior` | `sum_i w_i^c b_ih^-` | Current-choice-filtered pre-feedback belief |
| `catalogue_posterior` | `sum_i w_i^c b_ih^+` | Current-choice-filtered post-feedback belief |
| `catalogue_predictive_execution_probability` | `sum_i w_i^- E_ih` | Predictive mass of realized execution |
| `catalogue_execution_probability` | `sum_i w_i^c E_ih` | Execution mass filtered by current choice, before feedback conditioning |
| `catalogue_active_probability` | `sum_i w_i^- I_ih` | Predictive rule-presence probability |

Execution uses the recorded executed content. Feedback updates `b^+`; the
filtered execution probability uses `w^c` and the execution indicator. These
are distinct state quantities. All conditioning is forward in time.

Summing over h gives at most one for the first five quantities. Templates and
uncommitted content carry the remaining mass, so rule mass can sum below one.
Active probability sums to expected catalogue count, bounded by capacity.
The sums of prior, filtered prior, posterior and predictive execution match
semantic column 3 of `prior`, `filtered_prior`, `posterior` and
`dominant_probability`, respectively. These semantic arrays are required for
the checks. Current-choice-filtered execution has no matching saved semantic array.
Presence sums also match semantic `active_probability[:,3]`. Predictive belief
and execution cannot exceed predictive presence for the same rule. Fixed
autonomous path 0 has equal predictive/filtered prior and execution, with binary
presence/execution indicators; these are not PF-averaged quantities.

`conditional_` arrays are equal means across PF repeats using observed history.
`autonomous_diagnostic_` arrays describe fixed path 0 and its generated choices
and task feedback; its single-path indicators have unit particle weight.
`summary.catalogue_probabilities` records status, definitions, source
conditioning, IDs, metadata and the saved axis description.

## Legacy Results

Bundles without any extension field remain loadable with status `not_available`
and null catalogue metadata. Missing probabilities are not filled with zeros.
Partial or malformed extensions raise an error before an output directory is created.
The old 96-subject batch stores semantic means only; these cannot recover
conditional PF per-rule marginals. `content_path` records autonomous path 0.
The metadata-only `tools/export_evaluation_configs.py` transfers old parameter
values into fresh evaluation configurations while retaining source provenance
and strict fit-identity checks. Generating new diagnostics requires a separately
authorized evaluation into a new directory; existing results remain read-only.

## Re-evaluation of the Finished Batch

The current batch contains 96 completed subjects and no per-rule PF arrays.
Validated parameter exports are available in
`results/rule_probability_evaluation_configs_verified/`. `source.json` retains
old parameter/code identities, hashes of the fit/evaluation files and original
NPZs, input order checks and the unchanged evaluation budgets and seeds.
Exporting these configs does not run the model. The strict `--fit-dir` code
identity check remains unchanged; explicit transfer uses standalone parameters.

From the repository root, this command **starts** the re-evaluation (not fitting):

```powershell
$out = 'results/cond123_rule_probability_eval'
if (Test-Path -LiteralPath $out) { throw 'Choose a new output directory.' }
foreach ($cfg in Get-ChildItem 'results/rule_probability_evaluation_configs_verified/subject_*.json' | Sort-Object Name) {
    & .\build\Release\genmodel_cpp.exe evaluate --config $cfg.FullName --jobs 12 --output $out
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $($cfg.Name)" }
}
```

Each file is written to `$out/subject_ID/evaluation_inputs.npz`. The original
particle/autonomous-repeat budgets are preserved, so this is a new evaluation
workload, not an offline recovery from existing semantic means. It has not been
launched during this logging fix.

Read the requested conditional execution probabilities after evaluation:

```python
import numpy as np

with np.load('results/cond123_rule_probability_eval/subject_101/evaluation_inputs.npz') as saved:
    h0_h4_h6 = saved['conditional_catalogue_execution_probability'][:, [0, 4, 6]]
```

Local verification: 36 native tests and 267,199 assertions passed; Python had
197 passes and two non-applicable parameterized skips. Only three 8-trial
smoke evaluations and their plots were run. The finished batch was not edited.
