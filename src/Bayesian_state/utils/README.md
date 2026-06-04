# Bayesian State Utilities

The current optimization utilities support the new hyper-selection plus fixed-simulation workflow.

## Main Modules

- `optimizer_simulation.py`
  Execution layer for fixed-parameter repeated simulations. It owns `StateModelSimulationRunner`, repeat-level seeding, parallel execution, aggregation over repeated runs, and optional raw-run logging.
- `optimizer_common.py`
  Model-evaluation layer. It owns one-run `StateModel` evaluation, prediction/loss metrics, deterministic seed helpers, shared result containers, and the base data-preparation class.
- `optimization_config.py`
  Config-resolution layer. It owns YAML loading, path resolution, subject selection, engine-config resolution, prediction/loss/window parsing, JSON serialization, and stream-reference helpers.
- `hyperparam_values.py`
  Loads and validates hyperparameter value specifications.
- `config_subjects.py`
  Applies subject-specific config overrides.
- `datasets.py`
  Resolves processed-data inputs.
- `stream.py`
  Stores large optional logs out of the compact JSON payload.

## Expected Flow

Hyper selectors in `hyper_grid/` and `hyper_cd/` score candidate hyperparameters by repeated simulation. `run_simulation.py` then repeats the final fixed model many times using the selected subjectwise hyperparameters.

`optimization_config.py` is intentionally separate from `optimizer_common.py`: config parsing is runner-facing plumbing, while `optimizer_common.py` is model-facing evaluation logic. Keeping them separate avoids making the common evaluator responsible for CLI/YAML concerns.
