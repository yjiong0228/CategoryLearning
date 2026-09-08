# Shared-core consolidation — 2026-09-08

## Scope

One implementation is maintained in src/Bayesian_state. Journal module paths forward
to canonical leaf modules; journal configurations now use canonical class paths.
The recovery orchestrator moved from scripts into src/Bayesian_state/run_recovery.py.
Both legacy and journal entrypoints remain. Journal-specific CLI defaults and the
0826 parameter-space restriction are policy wrappers, not duplicated algorithms.

No cognitive equations, default hyperparameters, trial timing, subject filters,
oral sigma (0.05), data files, or historical result files were changed.
The duplicated similarity resource was removed; canonical resource SHA256 is checked
against the frozen manuscript configuration.

## Numerical evidence

Before consolidation, generated tests/fixtures/pre_shared_core.npz using the journal
implementation at commit 65519f4e (full commit and hashes are in the adjacent JSON).
The six cases contain 232 numeric arrays:

- S101, first 16 trials, 4 particles, filter seed 8326: both belief-transport methods,
  each with persistent execution off/on; nonzero accumulator/global-search gains.
- S101, first 24 trials, autonomous trajectory seed 8261: execution off/on.

All 232 arrays match the shared implementation exactly. This checks against saved
pre-refactor output rather than comparing two imports of the same implementation.

A separate journal CLI smoke run used S101, 32 trials, 32 particles, one repeat,
one job, and seed 20260901. It wrote a fresh temporary output directory; no old
outputs were overwritten. Mean choice NLL = 0.7377468323010447. The 11 non-provenance
top-level fields of subject_101.json match the previous outputs/smoke32_v1 file
exactly, including simulation, statistics, selection/seeds and representative_run.
Only model_provenance differs because configured Python class paths are canonical.

## Checks performed

202 tests passed across the following two targeted invocations:

```bash
python -m pytest -q CategoryLearning_codes/Bayesian_model/tests CategoryLearning_codes/figures CategoryLearning_codes/tests tests/bayesian_state/test_model_0826_versioning.py tests/bayesian_state/evaluation/test_oral_alignment.py tests/bayesian_state/test_model_0826_recovery.py
# 134 passed
python -m pytest -q tests/bayesian_state/test_model_0806_framework.py tests/bayesian_state/test_model_0815_h5_similarity_transport.py tests/bayesian_state/simulation/test_repeat_probability_aggregation.py tests/bayesian_state/evaluation/test_encoding_recompute.py
# 68 passed
```

All journal production modules imported successfully. CLI --help succeeded for
journal recovery, canonical recovery, the legacy recovery script, journal optimizer,
and journal hyper evaluation. git diff --check passed.

Journal scope tests reject cross-dataset file overrides and selected-subject overrides.
Shared path-resolution tests cover data, data_exp4, data_exp5 and data_meg; these are
path-contract tests, not evidence of scientific model support for all experiments.

## Limits

No full-cohort fits, parameter recovery, model comparison or long PF calibration was
run. The whole repository test suite was not run. Four-category/partial-feedback and
MEG task semantics require separate modeling work. Existing recovery fingerprints
are historical and must not be forced to resume under changed source provenance.
