> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Independent Python Reference Validation

## Scope

The reference is the original CategoryLearning Python implementation, expected
commit `249144de`. Explicit `--reference-root` access loads its
`model_1031_cond{1,2,3}_unified_v2.yaml` configurations and constructs the original
`StateModel`. Source hashes and runtime versions identify the actual files used;
the expected commit label is supplied provenance, not a Git verification.
The source tree and old numerical references remain read-only.

`generate_1031_reference.py` follows the setup in the original
`tests/bayesian_state/test_model_1031_unified.py`: it instantiates `StateModel`
with deterministic, zero-noise perception for direct-input cases and uses its
`ExperienceWorkspacePartition` and `ExperienceWorkspaceLikelihood`. Expected
emissions, feedback kernels, gradients and updated labels come directly from
those original methods. The tooling contains input construction, array handling
and statistical summaries. Production Python imports require explicit reference
access; `--help`, `self-test` and offline `fixture` comparison do not load them.

## Deterministic Fixture

Run from this repository with the existing sibling virtual environment:

```powershell
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/generate_1031_reference.py --reference-root ../CategoryLearning
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/generate_1031_reference.py --reference-root ../CategoryLearning --check
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/compare_1031.py self-test
```

Generation exclusively creates `tests/fixtures/reference_1031.json` and refuses
to overwrite an existing file. `--check` reevaluates the original Python and
compares the complete JSON value without writing. Development caches use
`.reference_cache/` here, including the original similarity cache's runtime
destination. Bytecode writing is disabled. No old fixture is regenerated.

The authorized proposal extension adds a separate `proposals` key and preserves
all existing fields, including the 25 original cases and their provenance.
`--extend-proposals` verifies those fields against the original Python before
appending the new section. Existing proposal references are protected from
replacement; `--check` reevaluates both sections without writing.

```powershell
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/generate_1031_reference.py --self-test
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/generate_1031_reference.py --reference-root ../CategoryLearning --extend-proposals
```

Seven proposal cases cover one- and two-feature generalizations in each condition
and a Cond3 pairing contrast. Each supplies `parent` (the original template node),
`beta`, ascending zero-based `rule_ids`, and ignored-feature endpoint `probes`.
Every case includes the entire catalogue: 29 rules for Cond1 and 152 for Cond2/3.
Cond3 retains conditional labels `[3,4]` with pairing `[0.15,0.60,0.25]`; its
contrast changes only the pairing distribution to `[0.70,0.20,0.10]`.

Proposal expected outputs are:

- `rule_proposal_weights[R]`: the original method's normalized proposal distribution.
- `parent_category_belief[C]`: the original method's parent label marginal, using
  the supplied pairing distribution in Cond3.
- `catalogue_category_probabilities[R,P,C]`: original boundary-mode catalogue
  predictions for every rule and probe, with response categories on the last axis.

`R` is the catalogue size and `P` is two or four probes. Probe rows enumerate
ignored dimensions in ascending feature order with Cartesian endpoints `0,1`.
Expected quantities call the original partition and catalogue methods directly.
The per-rule probe probabilities expose geometry differences alongside their
effect on proposal weights. Native exports include a separate `proposals` list
with matching IDs and expected-output keys; offline comparison checks both lists.

The 25 cases cover all three conditions, null and uncommitted slots, complete
and partial templates, first and last catalogue rules, zero/low/moderate Beta,
correct/error feedback, and Cond3 half feedback. Tiny-affinity cases use opposite
cube corners at width 0.25. Cond3 templates retain a nontrivial `theta[3,4]` and
pre-feedback pairing `[0.15, 0.60, 0.25]`. Each case supplies the full stimulus,
original node dictionary (or JSON null), condition, Beta, choice and feedback.
`similarity_width` is a fixture-level constant, 0.25.

Response coordinates are explicit:

- Fixture `choice` is zero-based, in `0..C-1`; rule IDs are also zero-based.
- Original Python observations and real-data `choices` use IDs `1..C`.
  The generator adds one only at the original observation API boundary.
- `category_probabilities[C]` follows response columns, Python IDs `1..C`.
- Cond3 `conditional_category_probabilities[3,C]` has pairing rows and response
  columns, using original pairing order `12|34`, `13|24`, `14|23` in one-based
  Python response IDs (zero-based groups `01|23`, `02|13`, `03|12`). Pairing probabilities
  marginalize these rows to form `category_probabilities`.
- `feedback_kernel` has length one for Cond1/2 and three for Cond3.
- `log_feedback_gradient` is the derivative of log feedback probability with
  respect to Beta. Cond3 uses the supplied pre-feedback pairing distribution.
- `updated_template_labels` is `[C]` or `[3,C]` for templates and null otherwise.
  Label updates use predictive Beta and pre-feedback labels, after all other
  requested outputs have been evaluated.

The native test runner can read the fixture directly. To compare exported native
values, emit `{"cases": [{"id": "cond1_null", "expected": {...}}, ...]}` with
exactly the fixture IDs and expected output keys, then run:

```powershell
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/compare_1031.py fixture --actual path/to/native_1031.json
```

Shapes, finite values and nulls must match. Every numeric value uses absolute
tolerance `1e-9` and zero relative tolerance. Comparing the fixture to itself is
only a comparator smoke check; it provides no native-runtime evidence.

## Deferred Statistical Screen

Parent confirmation that the ten-parameter native executable is ready is required
before running this screen. No statistical loop was run during fixture creation.
The planned budget is subjects 122/206/314 (conditions 1/2/3), the first 16 or 32
answered trials, 32 particles, and 16 to 32 independent seeds per language. Runs
are serial with `--jobs 1` and one numerical thread, within the eight-worker cap.

The original `SubjectTrialDataLoader` preserves input order, validates trial
keys, and skips missing responses. The actual original subject-specific perception
module remains enabled for this screen. `inputs.npz` records the exact retained
stimuli, one-based responses, feedback and score mask. The first retained trial
initializes learning and is excluded from scoring; all retained prefix trials
still enter the model. Native masks must match this prespecified policy exactly.

Supply a JSON object containing exactly these ten named parameters. Their original
engine destinations are:

| Name | Original Dotted Field |
| --- | --- |
| capacity | partition.kwargs.capacity |
| similarity_width | partition.kwargs.similarity_width |
| baseline_probability | modules.hypo_transitions_mod.kwargs.baseline_probability |
| error_gain | modules.hypo_transitions_mod.kwargs.error_gain |
| failure_decay | modules.hypo_transitions_mod.kwargs.failure_decay |
| global_search | modules.hypo_transitions_mod.kwargs.global_search |
| retention | modules.memory_mod.kwargs.gamma |
| execution_inertia | modules.hypo_transitions_mod.kwargs.execution_inertia |
| beta_init | modules.beta_mod.kwargs.beta_init |
| learning_rate | modules.beta_mod.kwargs.learning_rate |

The native configuration schema is owned by the parent. The provided
`tools/native_config_template.json` uses whole-value tokens `@SEED@`,
`@PARTICLES@`, `@PARAMETERS@`, `@SUBJECT@`, `@DATA_ROOT@` and `@RESOURCE_ROOT@`
in accepted native fields. The first four tokens and either `@DATA@` or
`@DATA_ROOT@` are required by the renderer. `@DATA@` retains its CSV-file meaning;
`@DATA_ROOT@` is the native data directory and defaults to this repository's
`data/`. The optional `@CONDITION@` replacement remains available for compatible
external templates. The provided native template infers condition from subject
data and uses an absolute resource root. Tokens are replaced with typed JSON
values, including the full parameter object. The parent must confirm model
version, readout and resampling policy alongside the input/perception audit.

```powershell
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/compare_1031.py run --reference-root ../CategoryLearning --native build/Release/genmodel_cpp.exe --native-ready --native-config-template tools/native_config_template.json --parameters tests/fixtures/statistical_parameters.json --data ../CategoryLearning/data/exp123/processed/Task2_processed.csv --trials 16 --seeds 16 --output validation/screen_NEW
```

Each native invocation is:

```text
EXE filter --config PATH --subjects SID --max-trials N --jobs 1 --output NEW
```

It must produce `NEW/subject_SID/filter.npz` with numeric
`choice_probabilities[T]`, `category_probabilities[T,C]`, and boolean
`score_mask[T]` in binary bool or uint8 form. Nonbinary values are rejected;
accepted masks are compared as booleans. Chosen probabilities must match category columns selected by the
original responses. State arrays may accompany these outputs; this screen scores
behavioral probabilities only. Latent-state equivalence, admission, ancestry,
pairing posteriors and whole-trajectory equivalence require additional native
tests. The deterministic fixture covers emissions and template-label learning.

Before any seed run, `protocol.json` records thresholds, parameters, independent
seed banks, source versions/hashes, the data digest, executable digest and config
template. Each subject receives its effective reference engine configuration,
retained trial identifiers, probability banks and compact `comparison.json`.
Process records retain the actual native commands and diagnostics. The output
directory must be new and inside this repository; existing results are refused.

NLL is the negative sum of log chosen probabilities. The report retains per-seed
NLL, mean NLL and mean chosen probability, as well as NLL of the seed-averaged
chosen probabilities. These two NLL summaries have different averaging orders.
Zero chosen probabilities stop scoring explicitly.

Monte Carlo standard error (MCSE) quantifies variation of the estimated mean over
independent random seeds. Each comparison uses the square root of the sum of the
two languages' seed-variance-over-seed-count terms. Per-seed trial averages retain
within-trajectory correlations. Log seed-mean probability uses a first-order
delta-method MCSE, with each seed's average centered probability divided by the
trial's seed-mean probability as its influence value.

Default acceptance limits are declared before running: absolute error at most
`0.05 + 4 * combined_MCSE` for each scored category/chosen probability, and
`0.10 + 4 * combined_MCSE` for mean NLL and mean log seed-mean probability.
CLI threshold overrides are recorded before computation. All tested quantities
must pass; failure returns a nonzero exit status. These are development-screen
tolerances, not a calibrated familywise statistical test. Sixteen seeds give a
limited estimate of Monte Carlo variability. A passing short-prefix screen
supports the tested settings; parameter recovery, scientific validity and wider
parameter/task equivalence remain separate questions.

## Frozen Native Recheck

`recheck-native` reuses the completed screen's Python banks and reruns exactly
32 native seeds for each of S122/S206/S314, with 32 trials, 32 particles and one
worker. It preserves the original seeds and thresholds. Before execution it
validates original source/runtime provenance, behavioral and perception hashes,
canonical parameter hashes, saved engine configurations, retained inputs and
agreement between cached banks and completed scores. No Python PF is rerun.

```powershell
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/compare_1031.py recheck-native --reference-root ../CategoryLearning --screen-root results/statistical_screen_1031 --native .tmp/validation_binary/genmodel_cpp.exe --native-ready --parameters tests/fixtures/statistical_parameters.json --data ../CategoryLearning/data/exp123/processed/Task2_processed.csv --output results/statistical_screen_1031_final
```

The output directory must be new. The first screen remains read-only. New
provenance records the frozen native hash, hashes of the reused Python banks and
source files, and the canonical parameter hash. Bank hashes are captured for this
reuse; the first screen predates bank-hash recording. Source files and input data
are checked again before accepting the final result. Both ordinary `run` and
`recheck-native` guard executable hashes before and after every native process,
including failed or timed-out processes, and before final acceptance.
`--validate-only` performs input checks without native execution or output writes.

## Perception Audit

Preflight reads the original subject context and actual perception module without
running a particle filter. It verifies SHA-256 equality of the behavioral CSV and
both perception CSVs across the original and new repositories. With `--screen-root`,
it also checks available native summaries against original perception mode,
means and scales with absolute tolerance `1e-12`. Uniform scale means half-range;
normal scale means standard deviation. Incomplete or unavailable native summaries
remain pending. No native process is launched and no result file is changed.

```powershell
../GenModelOfLatentState/.venv/Scripts/python.exe -B tools/compare_1031.py preflight --reference-root ../CategoryLearning --data ../CategoryLearning/data/exp123/processed/Task2_processed.csv --trials 32 --screen-root results/statistical_screen_1031
```

For Cond2, source and native subject-selection policies both assign subjects
201-224 to uniform perception and 225-232 to normal perception. S206 therefore
uses the 72-subject threshold summary; its four subject-specific half-ranges
enter in that subject's feature order. The active screen contains S206, so its
available native summaries validate the uniform branch. The normal branch has
synthetic audit-function coverage and source-policy inspection; an actual normal
subject's native output remains a separate runtime check.
