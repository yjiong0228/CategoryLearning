# CategoryLearning GitCode

This repository contains multiple category learning model implementations and experiment pipelines.

## Project Layout

- `src/Bayesian_state/`: current state-based model pipeline (active path)
- `src/Bayesian/`: baseline / legacy Bayesian model implementation
- `src/Hybrid/`, `src/RNN_old/`, `src/RNN_new/`, `src/SUSTAIN/`: other model families
- `configs/`: optimization and model-structure YAML configs
- `data/`, `data_meg/`: datasets
- `results/`: optimization outputs and aggregated artifacts
- `docs/`: usage notes for selected scripts
- `notebooks/`: exploratory notebooks
- `Old_version/`: historical code

## Recommended Entry Points

The recommended runnable pipeline is currently under `src/Bayesian_state`.

1. Grid optimization:
   - `python -m src.Bayesian_state.run_grid_optimization --config configs/grid_opt_cfg/pmh_cond1.yaml`
2. AMR optimization:
   - `python -m src.Bayesian_state.run_amr_optimization --opt-config configs/amr_opt_cfg/pmh_cond1.yaml`
3. Evaluation / plotting:
   - `python -m src.Bayesian_state.eval_grid_results --input-dir <result_dir>`
   - `python -m src.Bayesian_state.eval_amr_results --input-dir <result_dir>`

## Bayesian vs Bayesian_state

- `Bayesian_state`:
  - Modular engine with ordered `agenda` + pluggable modules from YAML.
  - Current optimization/evaluation scripts are maintained here.
- `Bayesian`:
  - Older `StandardModel`-centric pipeline.
  - Still useful for baseline comparison and backward compatibility.

## Current Refactor Direction

This repository is being refactored incrementally. Current priorities:

1. Clarify module responsibilities and official entry points.
2. Reduce duplicated optimization logic between state-based optimizers.
3. Mark and isolate legacy flows before any removals.

## Notes

- There is no behavior change introduced by this README update.
- If you are unsure where to start, begin with `src/Bayesian_state/README.md`.
