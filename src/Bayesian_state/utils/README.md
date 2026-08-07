# Bayesian State Utilities

This directory contains small shared utilities only:

- `paths.py`: centralized project, config, data, log, and result paths.
- `datasets.py`: dataset-path resolution.
- `config_subjects.py`: subject-specific configuration overrides.
- `basic_stat.py` and `classical_tools.py`: small numerical helpers.
- `base.py`, `console_styles.py`, and `load_config.py`: legacy logging,
  console, and configuration helpers.
- `simulation_statistics.py`: repeated-simulation summary statistics shared by
  optimization and final simulation runs.
- `stream.py`: compressed storage for large optional run logs.

Domain implementations now live in dedicated packages:

- `src.Bayesian_state.optimization`: config parsing, simulation execution,
  hyperparameter search, and hyper-search evaluation.
- `src.Bayesian_state.model_evaluation`: post-simulation plots and oral/model
  alignment.
- `src.Bayesian_state.manuscript_models`: standalone manuscript model series.
- `src.Bayesian_state.active_set`: minimal active-set generation, filtering,
  mechanism variants, and posterior-predictive rollouts.
