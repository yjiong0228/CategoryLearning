# Bayesian State Optimization Workflow

This package now uses a two-step workflow:

1. Select all hyperparameters jointly, including `gamma` and `w0`.
2. Run repeated fixed-parameter simulations under the selected hyperparameters.
3. Run model evaluation on the simulation outputs to produce diagnostic plots.

`gamma` and `w0` are no longer treated as an inner-layer exception. They live in the same `hyperparam_space` as strategy, active-set, and beta parameters.

## Entrypoints

- `python -m src.Bayesian_state.utils.hyper_cli --backend grid --config <yaml>`
  Runs explicit joint grid hyperparameter selection.
- `python -m src.Bayesian_state.utils.hyper_cli --backend cd --config <yaml>`
  Runs coordinate-descent hyperparameter selection over the same style of search space.
- `python -m src.Bayesian_state.run_hyper_then_simulation --backend hyper_grid --hyper-config <yaml>`
  Runs hyper-grid selection, materializes a subjectwise simulation config, then runs fixed simulations.
- `python -m src.Bayesian_state.run_hyper_then_simulation --backend hyper_cd --hyper-config <yaml>`
  Runs hyper-CD selection, materializes a subjectwise simulation config, then runs fixed simulations.
- `python -m src.Bayesian_state.run_simulation --config <yaml>`
  Runs repeated simulations under fixed hyperparameters.
- `python -m src.Bayesian_state.run_model_evaluation --input-dir <simulation-result-dir>`
  Runs post-simulation model evaluation plots. The input directory should contain `subjects/subject_*.json`.

## Configs

- `configs/hyper_grid_cfg/`
  Joint grid hyperparameter selection configs, e.g. `pmh_cond1_hyper_grid.yaml`.
- `configs/hyper_cd_cfg/`
  Coordinate-descent hyperparameter selection configs, e.g. `pmh_cond1_hyper_cd.yaml`.
- `configs/simulation_cfg/`
  Fixed-parameter repeated simulation configs, e.g. `pmh_cond1_simulation.yaml`.

Hyper configs use:

- `base_sim_config_path`: base simulation YAML.
- `hyperparam_space`: parameters to select. Supported prefixes are `engine.` and `simulation.`.
- `stages.coarse.simulation_overrides`: simulation budget/settings for coarse selection.
- `stages.fine.simulation_overrides`: simulation budget/settings for fine selection.
- `refine_policy.top_k`: number of coarse candidates used to build fine candidates.
- `refine_policy.expand`: optional fine-stage expansion for selected coordinates such as `gamma` and `w0`.

Simulation configs use:

- `simulation_repeats`: number of repeated simulations per subject.
- `fixed_hyperparams`: optional explicit fixed hyperparameters. If omitted, direct simulation infers the baseline hyperparameters from the resolved model structure. Hyper-generated subject overrides write this block explicitly.
- `engine_config_path` or `engine_config`: model structure and module settings.

## Outputs

Hyper selection writes per-subject `best_hyperparams.json`, `stage_summary.json`, and `all_combinations.jsonl`. Hyper-grid also writes `accepted_hyperparams.jsonl` when posterior-predictive acceptance selection is enabled. Hyper-CD also writes `restart_summary.json` and `coordinate_trace.jsonl`.

The workflow runner writes a generated subjectwise simulation config and then writes simulation results under `results/state-based-simulation/...`.

Model evaluation writes plots and CSV summaries under `<simulation-result-dir>/model_evaluation/` by default. It produces the core metric/log plots from simulation JSONs and, when oral data are available, the oral/model alignment plots as well. Trajectory-rank plots require final simulations to run with `keep_logs: true` so `raw_runs_ref` streams are present.

Example:

```bash
python -m src.Bayesian_state.run_model_evaluation \
  --input-dir results/state-based-simulation/pmh/cond1 \
  --window-size 8
```
