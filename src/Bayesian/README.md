# Bayesian Module

`Bayesian` contains the earlier Bayesian model implementation centered on `StandardModel`.

## What This Module Contains

- `problems/model.py`: `BaseModel` / `StandardModel` and fitting logic
- `problems/partitions.py`: partition definitions and likelihood helpers
- `problems/modules/`: legacy module components (decision, memory, perception, transitions)
- `inference_engine/bayesian_engine.py`: Bayesian inference engine for the legacy flow
- `utils/optimizer.py`: legacy parameter optimization helper

## Current Status

- This module is important for baseline behavior and historical compatibility.
- The current official batch optimization entry points are in `src/Bayesian_state`, not here.

## When To Use This Module

- Reproducing earlier experiments built around `StandardModel`.
- Comparing state-based pipeline results against legacy Bayesian baselines.
- Inspecting historical model behavior before migration/refactor.

## MEG M6_MH Posterior Workflow

The reusable replacement for the fitting/export cells in
`notebooks/old/Bayesian_meg.ipynb` is:

```bash
python -m src.Bayesian.run_meg_posterior --subject 334
```

The command reads `data/meg/processed/Task3b_processed.csv`, performs a fresh
M6_MH fit, predicts the subject trajectory, exports the four diagnostic PNGs,
and writes the trial-level posterior CSV required by top-down analyses. Its
default scientific settings are the notebook settings: window size 16, grid
repeat 64, and 1024 Monte Carlo samples. The worker budget defaults to 120 and
can be changed with `--n-jobs` without changing those fitting settings. The
effective fit worker count is bounded by the requested budget, available CPUs,
and ready grid tasks; the one-subject prediction step uses one worker. Numeric
libraries use one inner thread per worker.

Outputs are created under:

```text
results/model_static/model_results_meg/Model_results_sub<SUBJECT>_<YYMMDD>/
```

The top-down input is named
`Task3b_Sub<SUBJECT>_M6_MH_model_posterior.csv`. The directory also contains
`M6_MH.joblib`, `M6_MH_prediction.joblib`, the raw-step cache, four PNGs, and a
`run_manifest.json` with input hash, code revision, environment, parameters,
relevant source-file hashes, dirty-worktree entries, completion status, and
posterior diagnostics. Existing subject/date directories are never overwritten.
A failed run remains in place with `status: failed` in the manifest so partial
research artifacts are not silently deleted.

If a failed directory already occupies the subject/date name, preserve it and
use a labeled retry rather than overwriting it:

```bash
python -m src.Bayesian.run_meg_posterior --subject 336 --run-suffix retry1
```

This writes `Model_results_sub336_<YYMMDD>_retry1/`. Processed inputs must store
`choice` as an integer column; missed responses should be excluded explicitly
without renumbering an existing `iTrial` column so downstream MEG alignment
retains the original trial identity.

## Important Note

Some cross-module coupling still exists between `Bayesian` and `Bayesian_state`.
During refactor, this module should be gradually decoupled instead of removed directly.
