# Hyper-Opt-CD

`hyper_opt_cd` is a two-layer hyperparameter optimizer based on coordinate descent.

## Goal

Keep the same objective as the existing hyper optimizer:

`argmin_hyperparams min_inner_params(mean_error)`

- Outer layer: coordinate descent over hyperparameters
- Inner layer: existing `grid` or `amr` optimizer (unchanged)

## Run

Use `cate_learn` environment:

```bash
conda activate cate_learn
python -m src.Bayesian_state.hyper_opt_cd.cli \
  --config configs/hyper_opt_cd_cfg/example.yaml \
  --stage all
```

Optional subject overrides:

```bash
python -m src.Bayesian_state.hyper_opt_cd.cli \
  --config configs/hyper_opt_cd_cfg/example.yaml \
  --subjects 125 126
```

To run CD hyper-optimization and then immediately materialize normal GRID
results, use the shared workflow runner:

```bash
python -m src.Bayesian_state.run_hyper_then_grid \
  --hyper-config configs/hyper_opt_cd_cfg/example.yaml \
  --hyper-backend cd \
  --execution-mode per-subject
```

With `--hyper-backend cd`, the workflow runner uses CD-specific defaults so it
does not overwrite standard hyper-opt GRID outputs:

- generated GRID config: `configs/grid_opt_cfg/pmh_cond1_subjectwise_hyper_cd_best.yaml`
- GRID results: `results/state-based-grid-result/pmh/cond1_subjectwise_hyper_cd_best`

## Key Config Fields

- `inner_optimizer`: `grid` or `amr`
- `inner_base_config_path`: base inner config path
- `hyperparam_selection_mode`: `per_subject` (default) or `group_mean`
- `stages.coarse` / `stages.fine`:
  - `inner_overrides`: inner budget for that stage
  - optional `hyperparam_space` override for that stage
- `hyperparam_space`: outer hyperparameter candidates (explicit only)
  - Keep inner-grid params (for example `gamma`, `w0`) inside
    `stages.<coarse|fine>.inner_overrides.param_grid`.
  - `inner.param_grid.*` is intentionally disabled in `hyper_opt_cd` hyperparam space.
- `refine_policy.top_k`: coarse top-k used for fine fallback
- `cd`:
  - `n_restarts`
  - `max_outer_iters`
  - `init_strategy`: `random` or `anchor`
  - `coordinate_order`: `shuffle_each_iter` or `fixed`
  - `patience`
  - `min_delta`

## Outputs

- `all_combinations.jsonl`: all evaluated hyperparameter combinations
- `coordinate_trace.jsonl`: one row per coordinate update attempt, including candidate count, cache hits, and whether the coordinate improved the current best
- `restart_summary.json`: initialization, best point, stopping reason, number of improvements, cache hits, and new evaluations per restart
- `stage_summary.json`: top-k combinations by stage
- `best_hyperparams.json`: final best hyperparameters

When `hyperparam_selection_mode=per_subject`, each subject writes into its own folder.
