"""Utilities for evaluating hyper-CD optimization trajectories.

The hyper-CD runner already writes enough information to inspect convergence:

- ``all_combinations.jsonl`` stores every evaluated hyperparameter point.
- ``coordinate_trace.jsonl`` stores every coordinate-descent step.
- ``restart_summary.json`` stores initial/final state and improvements per restart.

This module flattens those artifacts into CSV files and produces lightweight
diagnostic plots for restart-level convergence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.optimization.optimization_config import (
    DEFAULT_DATA_PATH,
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.optimization.hyper_utils import (
    subject_best_hyperparams,
    subject_hyper_candidate_seed,
)
from src.Bayesian_state.optimization.optimizer_common import (
    BaseStateOptimizer,
    derive_simulation_point_seed,
    derive_trajectory_seed,
    evaluate_state_model_run,
    stable_seed,
)
from src.Bayesian_state.optimization.optimizer_simulation import StateModelSimulationRunner
from src.Bayesian_state.utils.simulation_statistics import get_stat_value
from src.Bayesian_state.utils.paths import ROOT_DIR


MEMORY_KEY = "engine.modules.memory_mod.kwargs"
TRANSITION_KEY = "engine.modules.hypo_transitions_mod.kwargs"
PRIOR_RESET_BASE_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_base"
PRIOR_RESET_POST_ERROR_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_post_error"
PRIOR_RESET_LOW_ACCURACY_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_low_accuracy"
PRIOR_RESET_THRESHOLD_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_threshold"
PRIOR_RESET_WINDOW_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_window"
PRIOR_RESET_DECAY_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_decay"
PRIOR_RESET_MAX_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_max"
PRIOR_RESET_TARGET_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_target"
PRIOR_RESET_SOURCE_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_source"
PRIOR_RESET_VOLATILITY_GAIN_KEY = "engine.modules.hypo_transitions_mod.kwargs.prior_reset_volatility_gain"
LATENT_VOLATILITY_BASE_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_base"
LATENT_VOLATILITY_ERROR_GAIN_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_error_gain"
LATENT_VOLATILITY_LOW_ACCURACY_GAIN_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_low_accuracy_gain"
LATENT_VOLATILITY_THRESHOLD_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_threshold"
LATENT_VOLATILITY_WINDOW_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_window"
LATENT_VOLATILITY_DECAY_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_decay"
LATENT_VOLATILITY_MAX_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_max"
LATENT_VOLATILITY_FEEDBACK_MODE_KEY = "engine.modules.hypo_transitions_mod.kwargs.latent_volatility_feedback_mode"
BETA_INIT_KEY = "engine.modules.beta_mod.kwargs.beta_init"
DECREASE_RATE_KEY = "engine.modules.beta_mod.kwargs.decrease_rate"
PRIOR_BETA_SCALE_KEY = "engine.modules.beta_mod.kwargs.prior_beta_scale"
CORRECT_ADDITIVE_KEY = "engine.modules.beta_mod.kwargs.correct_additive"
BETA_UPDATE_MODE_KEY = "engine.modules.beta_mod.kwargs.beta_update_mode"
PROBABILISTIC_FEEDBACK_LAPSE_KEY = "engine.modules.beta_mod.kwargs.probabilistic_feedback_lapse"
DISTANCE_MODE_KEY = "engine.modules.likelihood_mod.kwargs.distance_mode"
OUTPUT_NOISE_BASE_LAPSE_KEY = "engine.output_noise.kwargs.base_lapse"
OUTPUT_NOISE_POST_ERROR_LAPSE_KEY = "engine.output_noise.kwargs.post_error_lapse"
OUTPUT_NOISE_LOW_ACCURACY_LAPSE_KEY = "engine.output_noise.kwargs.low_accuracy_lapse"
OUTPUT_NOISE_LOW_ACCURACY_THRESHOLD_KEY = "engine.output_noise.kwargs.low_accuracy_threshold"
OUTPUT_NOISE_LAPSE_DECAY_KEY = "engine.output_noise.kwargs.lapse_decay"
OUTPUT_NOISE_MAX_LAPSE_KEY = "engine.output_noise.kwargs.max_lapse"
OUTPUT_NOISE_RECENT_ACCURACY_WINDOW_KEY = "engine.output_noise.kwargs.recent_accuracy_window"
OUTPUT_NOISE_LAPSE_TARGET_KEY = "engine.output_noise.kwargs.lapse_target"
OUTPUT_NOISE_LATENT_VOLATILITY_LAPSE_KEY = "engine.output_noise.kwargs.latent_volatility_lapse"
OUTPUT_NOISE_LATENT_VOLATILITY_POWER_KEY = "engine.output_noise.kwargs.latent_volatility_power"

NUMERIC_PARAM_COLUMNS = (
    "gamma",
    "w0",
    "beta_init",
    "decrease_rate",
    "prior_beta_scale",
    "correct_additive",
    "probabilistic_feedback_lapse",
    "prior_reset_base",
    "prior_reset_post_error",
    "prior_reset_low_accuracy",
    "prior_reset_threshold",
    "prior_reset_window",
    "prior_reset_decay",
    "prior_reset_max",
    "prior_reset_volatility_gain",
    "latent_volatility_base",
    "latent_volatility_error_gain",
    "latent_volatility_low_accuracy_gain",
    "latent_volatility_threshold",
    "latent_volatility_window",
    "latent_volatility_decay",
    "latent_volatility_max",
    "output_base_lapse",
    "output_post_error_lapse",
    "output_low_accuracy_lapse",
    "output_low_accuracy_threshold",
    "output_lapse_decay",
    "output_max_lapse",
    "output_recent_accuracy_window",
    "output_latent_volatility_lapse",
    "output_latent_volatility_power",
)
CATEGORICAL_PARAM_COLUMNS = (
    "strategy_id",
    "beta_update_mode",
    "distance_mode",
    "prior_reset_target",
    "prior_reset_source",
    "latent_volatility_feedback_mode",
    "output_lapse_target",
)

DEFAULT_BASE_SIM_CONFIG = Path("configs/simulation_cfg/pmh_cond1_simulation.yaml")

ACCURACY_SHAPE_COLUMNS = (
    "accuracy_shape_score",
    "accuracy_shape_choice_error",
    "accuracy_shape_repeat_index",
    "accuracy_shape_acc_mae",
    "accuracy_shape_acc_rmse",
    "accuracy_shape_acc_corr",
    "accuracy_shape_vol_ratio",
    "accuracy_shape_range_ratio",
    "accuracy_shape_slope_agree",
    "accuracy_shape_run_choice_cutoff",
    "accuracy_shape_eligible_run_count",
    "accuracy_shape_all_run_count",
    "accuracy_shape_score_mean",
    "accuracy_shape_score_q10",
    "accuracy_shape_eligible_score_mean",
)

HISTORY_KERNEL_COLUMNS = (
    "history_kernel_score",
    "history_kernel_choice_error",
    "history_kernel_repeat_index",
    "history_kernel_mse",
    "history_kernel_corr",
    "history_kernel_corr_loss",
    "history_kernel_norm_ratio",
    "history_kernel_human_norm",
    "history_kernel_model_norm",
    "history_kernel_max_lag",
    "history_kernel_n_rows",
    "history_kernel_human",
    "history_kernel_model",
    "history_kernel_run_choice_cutoff",
    "history_kernel_eligible_run_count",
    "history_kernel_all_run_count",
    "history_kernel_score_mean",
    "history_kernel_score_q10",
    "history_kernel_eligible_score_mean",
)

SWITCH_BEHAVIOR_COLUMNS = (
    "switch_behavior_score",
    "switch_behavior_choice_error",
    "switch_behavior_repeat_index",
    "switch_behavior_switch_human",
    "switch_behavior_switch_model",
    "switch_behavior_switch_abs_diff",
    "switch_behavior_perseveration_human",
    "switch_behavior_perseveration_model",
    "switch_behavior_perseveration_abs_diff",
    "switch_behavior_win_stay_human",
    "switch_behavior_win_stay_model",
    "switch_behavior_win_stay_abs_diff",
    "switch_behavior_lose_shift_human",
    "switch_behavior_lose_shift_model",
    "switch_behavior_lose_shift_abs_diff",
    "switch_behavior_n_pairs",
    "switch_behavior_n_win_pairs",
    "switch_behavior_n_loss_pairs",
    "switch_behavior_run_choice_cutoff",
    "switch_behavior_eligible_run_count",
    "switch_behavior_all_run_count",
    "switch_behavior_score_mean",
    "switch_behavior_score_q10",
    "switch_behavior_eligible_score_mean",
)

DISTRIBUTION_BEHAVIOR_COLUMNS = (
    "distribution_score",
    "distribution_component_max_raw",
    "distribution_intersection_score",
    "distribution_intersection_violation_count",
    "distribution_intersection_accept",
    "distribution_intersection_acc_mae_violation",
    "distribution_intersection_vol_ratio_low_violation",
    "distribution_intersection_vol_ratio_high_violation",
    "distribution_intersection_history_corr_violation",
    "distribution_intersection_switch_rate_violation",
    "distribution_intersection_win_stay_violation",
    "distribution_intersection_lose_shift_violation",
    "distribution_intersection_perseveration_violation",
    "distribution_ppc_interval_score",
    "distribution_ppc_interval_violation_count",
    "distribution_ppc_interval_accept",
    "distribution_ppc_interval_alpha",
    "distribution_ppc_interval_stat_count",
    "distribution_ppc_interval_lower_quantile",
    "distribution_ppc_interval_upper_quantile",
    "distribution_ppc_interval_acc_mean_human",
    "distribution_ppc_interval_acc_mean_model_q05",
    "distribution_ppc_interval_acc_mean_model_q95",
    "distribution_ppc_interval_acc_mean_model_median",
    "distribution_ppc_interval_acc_mean_percentile",
    "distribution_ppc_interval_acc_mean_tail_score",
    "distribution_ppc_interval_acc_mean_violation",
    "distribution_ppc_interval_acc_mean_accept",
    "distribution_ppc_interval_acc_vol_human",
    "distribution_ppc_interval_acc_vol_model_q05",
    "distribution_ppc_interval_acc_vol_model_q95",
    "distribution_ppc_interval_acc_vol_model_median",
    "distribution_ppc_interval_acc_vol_percentile",
    "distribution_ppc_interval_acc_vol_tail_score",
    "distribution_ppc_interval_acc_vol_violation",
    "distribution_ppc_interval_acc_vol_accept",
    "distribution_ppc_interval_acc_range_human",
    "distribution_ppc_interval_acc_range_model_q05",
    "distribution_ppc_interval_acc_range_model_q95",
    "distribution_ppc_interval_acc_range_model_median",
    "distribution_ppc_interval_acc_range_percentile",
    "distribution_ppc_interval_acc_range_tail_score",
    "distribution_ppc_interval_acc_range_violation",
    "distribution_ppc_interval_acc_range_accept",
    "distribution_ppc_interval_history_kernel_norm_human",
    "distribution_ppc_interval_history_kernel_norm_model_q05",
    "distribution_ppc_interval_history_kernel_norm_model_q95",
    "distribution_ppc_interval_history_kernel_norm_model_median",
    "distribution_ppc_interval_history_kernel_norm_percentile",
    "distribution_ppc_interval_history_kernel_norm_tail_score",
    "distribution_ppc_interval_history_kernel_norm_violation",
    "distribution_ppc_interval_history_kernel_norm_accept",
    "distribution_ppc_interval_switch_rate_human",
    "distribution_ppc_interval_switch_rate_model_q05",
    "distribution_ppc_interval_switch_rate_model_q95",
    "distribution_ppc_interval_switch_rate_model_median",
    "distribution_ppc_interval_switch_rate_percentile",
    "distribution_ppc_interval_switch_rate_tail_score",
    "distribution_ppc_interval_switch_rate_violation",
    "distribution_ppc_interval_switch_rate_accept",
    "distribution_ppc_interval_win_stay_human",
    "distribution_ppc_interval_win_stay_model_q05",
    "distribution_ppc_interval_win_stay_model_q95",
    "distribution_ppc_interval_win_stay_model_median",
    "distribution_ppc_interval_win_stay_percentile",
    "distribution_ppc_interval_win_stay_tail_score",
    "distribution_ppc_interval_win_stay_violation",
    "distribution_ppc_interval_win_stay_accept",
    "distribution_ppc_interval_lose_shift_human",
    "distribution_ppc_interval_lose_shift_model_q05",
    "distribution_ppc_interval_lose_shift_model_q95",
    "distribution_ppc_interval_lose_shift_model_median",
    "distribution_ppc_interval_lose_shift_percentile",
    "distribution_ppc_interval_lose_shift_tail_score",
    "distribution_ppc_interval_lose_shift_violation",
    "distribution_ppc_interval_lose_shift_accept",
    "distribution_ppc_interval_perseveration_human",
    "distribution_ppc_interval_perseveration_model_q05",
    "distribution_ppc_interval_perseveration_model_q95",
    "distribution_ppc_interval_perseveration_model_median",
    "distribution_ppc_interval_perseveration_percentile",
    "distribution_ppc_interval_perseveration_tail_score",
    "distribution_ppc_interval_perseveration_violation",
    "distribution_ppc_interval_perseveration_accept",
    "distribution_run_count",
    "distribution_choice_error_mean",
    "distribution_choice_error_median",
    "distribution_choice_error_q10",
    "distribution_choice_error_std",
    "distribution_accuracy_shape_score",
    "distribution_acc_mae_mean",
    "distribution_acc_mae_median",
    "distribution_acc_mae_q90",
    "distribution_acc_rmse_mean",
    "distribution_vol_ratio_mean",
    "distribution_vol_ratio_median",
    "distribution_vol_ratio_q10",
    "distribution_vol_ratio_q90",
    "distribution_slope_agree_mean",
    "distribution_history_kernel_score",
    "distribution_history_kernel_mse",
    "distribution_history_kernel_corr",
    "distribution_history_kernel_corr_loss",
    "distribution_history_kernel_norm_ratio",
    "distribution_history_kernel_human_norm",
    "distribution_history_kernel_model_norm",
    "distribution_history_kernel_run_count",
    "distribution_switch_behavior_score",
    "distribution_switch_human",
    "distribution_switch_model",
    "distribution_switch_abs_diff",
    "distribution_perseveration_human",
    "distribution_perseveration_model",
    "distribution_perseveration_abs_diff",
    "distribution_win_stay_human",
    "distribution_win_stay_model",
    "distribution_win_stay_abs_diff",
    "distribution_lose_shift_human",
    "distribution_lose_shift_model",
    "distribution_lose_shift_abs_diff",
    "distribution_switch_run_count",
)

METRIC_COLUMN_PATHS = {
    "hyper_selection_error": "selection.primary.value",
    "hyper_mean_error": "simulation.mean_error",
    "hyper_best_error": "simulation.best_error",
    "hyper_best10_mean_error": "simulation.best10_mean_error",
    "hyper_q10_error": "simulation.q10_error",
    "hyper_std_error": "simulation.std_error",
    "hyper_simulation_repeats": "simulation.simulation_repeats",
    "accuracy_shape_score": "statistics.scores.accuracy_shape.value",
    "accuracy_shape_choice_error": "statistics.scores.accuracy_shape.choice_error",
    "accuracy_shape_repeat_index": "statistics.scores.accuracy_shape.repeat_index",
    "accuracy_shape_acc_mae": "statistics.diagnostics.accuracy_curve.selected.metrics.mae",
    "accuracy_shape_acc_rmse": "statistics.diagnostics.accuracy_curve.selected.metrics.rmse",
    "accuracy_shape_acc_corr": "statistics.diagnostics.accuracy_curve.selected.metrics.corr",
    "accuracy_shape_vol_ratio": "statistics.diagnostics.accuracy_curve.selected.metrics.volatility_ratio",
    "accuracy_shape_range_ratio": "statistics.diagnostics.accuracy_curve.selected.metrics.range_ratio",
    "accuracy_shape_slope_agree": "statistics.diagnostics.accuracy_curve.selected.metrics.slope_agree",
    "accuracy_shape_run_choice_cutoff": "statistics.diagnostics.accuracy_curve.run_gate.choice_error_cutoff",
    "accuracy_shape_eligible_run_count": "statistics.diagnostics.accuracy_curve.run_gate.eligible_count",
    "accuracy_shape_all_run_count": "statistics.diagnostics.accuracy_curve.run_gate.run_count",
    "accuracy_shape_score_mean": "statistics.diagnostics.accuracy_curve.score_summary.mean",
    "accuracy_shape_score_q10": "statistics.diagnostics.accuracy_curve.score_summary.q10",
    "accuracy_shape_eligible_score_mean": "statistics.diagnostics.accuracy_curve.score_summary.eligible_mean",
    "history_kernel_score": "statistics.scores.history_kernel.value",
    "history_kernel_choice_error": "statistics.scores.history_kernel.choice_error",
    "history_kernel_repeat_index": "statistics.scores.history_kernel.repeat_index",
    "history_kernel_mse": "statistics.diagnostics.history_kernel.selected.metrics.mse",
    "history_kernel_corr": "statistics.diagnostics.history_kernel.selected.metrics.corr",
    "history_kernel_corr_loss": "statistics.diagnostics.history_kernel.selected.metrics.corr_loss",
    "history_kernel_norm_ratio": "statistics.diagnostics.history_kernel.selected.metrics.norm_ratio",
    "history_kernel_human_norm": "statistics.diagnostics.history_kernel.selected.metrics.human_norm",
    "history_kernel_model_norm": "statistics.diagnostics.history_kernel.selected.metrics.model_norm",
    "history_kernel_max_lag": "statistics.diagnostics.history_kernel.selected.metrics.max_lag",
    "history_kernel_n_rows": "statistics.diagnostics.history_kernel.selected.metrics.n_rows",
    "history_kernel_human": "statistics.diagnostics.history_kernel.selected.metrics.human_kernel",
    "history_kernel_model": "statistics.diagnostics.history_kernel.selected.metrics.model_kernel",
    "history_kernel_run_choice_cutoff": "statistics.diagnostics.history_kernel.run_gate.choice_error_cutoff",
    "history_kernel_eligible_run_count": "statistics.diagnostics.history_kernel.run_gate.eligible_count",
    "history_kernel_all_run_count": "statistics.diagnostics.history_kernel.run_gate.run_count",
    "history_kernel_score_mean": "statistics.diagnostics.history_kernel.score_summary.mean",
    "history_kernel_score_q10": "statistics.diagnostics.history_kernel.score_summary.q10",
    "history_kernel_eligible_score_mean": "statistics.diagnostics.history_kernel.score_summary.eligible_mean",
    "switch_behavior_score": "statistics.scores.switch_behavior.value",
    "switch_behavior_choice_error": "statistics.scores.switch_behavior.choice_error",
    "switch_behavior_repeat_index": "statistics.scores.switch_behavior.repeat_index",
    "switch_behavior_switch_human": "statistics.diagnostics.switch_behavior.selected.metrics.switch.human",
    "switch_behavior_switch_model": "statistics.diagnostics.switch_behavior.selected.metrics.switch.model",
    "switch_behavior_switch_abs_diff": "statistics.diagnostics.switch_behavior.selected.metrics.switch.abs_diff",
    "switch_behavior_perseveration_human": "statistics.diagnostics.switch_behavior.selected.metrics.perseveration.human",
    "switch_behavior_perseveration_model": "statistics.diagnostics.switch_behavior.selected.metrics.perseveration.model",
    "switch_behavior_perseveration_abs_diff": "statistics.diagnostics.switch_behavior.selected.metrics.perseveration.abs_diff",
    "switch_behavior_win_stay_human": "statistics.diagnostics.switch_behavior.selected.metrics.win_stay.human",
    "switch_behavior_win_stay_model": "statistics.diagnostics.switch_behavior.selected.metrics.win_stay.model",
    "switch_behavior_win_stay_abs_diff": "statistics.diagnostics.switch_behavior.selected.metrics.win_stay.abs_diff",
    "switch_behavior_lose_shift_human": "statistics.diagnostics.switch_behavior.selected.metrics.lose_shift.human",
    "switch_behavior_lose_shift_model": "statistics.diagnostics.switch_behavior.selected.metrics.lose_shift.model",
    "switch_behavior_lose_shift_abs_diff": "statistics.diagnostics.switch_behavior.selected.metrics.lose_shift.abs_diff",
    "switch_behavior_n_pairs": "statistics.diagnostics.switch_behavior.selected.metrics.counts.pairs",
    "switch_behavior_n_win_pairs": "statistics.diagnostics.switch_behavior.selected.metrics.counts.win_pairs",
    "switch_behavior_n_loss_pairs": "statistics.diagnostics.switch_behavior.selected.metrics.counts.loss_pairs",
    "switch_behavior_run_choice_cutoff": "statistics.diagnostics.switch_behavior.run_gate.choice_error_cutoff",
    "switch_behavior_eligible_run_count": "statistics.diagnostics.switch_behavior.run_gate.eligible_count",
    "switch_behavior_all_run_count": "statistics.diagnostics.switch_behavior.run_gate.run_count",
    "switch_behavior_score_mean": "statistics.diagnostics.switch_behavior.score_summary.mean",
    "switch_behavior_score_q10": "statistics.diagnostics.switch_behavior.score_summary.q10",
    "switch_behavior_eligible_score_mean": "statistics.diagnostics.switch_behavior.score_summary.eligible_mean",
    "distribution_score": "statistics.scores.distribution.multiobjective.score",
    "distribution_component_max_raw": "statistics.scores.distribution.multiobjective.component_max_raw",
    "distribution_intersection_score": "statistics.scores.distribution.intersection.score",
    "distribution_intersection_violation_count": "statistics.scores.distribution.intersection.violation_count",
    "distribution_intersection_accept": "statistics.scores.distribution.intersection.accept",
    "distribution_run_count": "statistics.diagnostics.distribution.run_count",
    "distribution_choice_error_mean": "statistics.diagnostics.distribution.choice_error.mean",
    "distribution_choice_error_median": "statistics.diagnostics.distribution.choice_error.median",
    "distribution_choice_error_q10": "statistics.diagnostics.distribution.choice_error.q10",
    "distribution_choice_error_std": "statistics.diagnostics.distribution.choice_error.std",
    "distribution_accuracy_shape_score": "statistics.scores.distribution.multiobjective.components.accuracy_shape",
    "distribution_acc_mae_mean": "statistics.diagnostics.distribution.accuracy_curve.mae.mean",
    "distribution_acc_mae_median": "statistics.diagnostics.distribution.accuracy_curve.mae.median",
    "distribution_acc_mae_q90": "statistics.diagnostics.distribution.accuracy_curve.mae.q90",
    "distribution_acc_rmse_mean": "statistics.diagnostics.distribution.accuracy_curve.rmse.mean",
    "distribution_vol_ratio_mean": "statistics.diagnostics.distribution.accuracy_curve.volatility_ratio.mean",
    "distribution_vol_ratio_median": "statistics.diagnostics.distribution.accuracy_curve.volatility_ratio.median",
    "distribution_vol_ratio_q10": "statistics.diagnostics.distribution.accuracy_curve.volatility_ratio.q10",
    "distribution_vol_ratio_q90": "statistics.diagnostics.distribution.accuracy_curve.volatility_ratio.q90",
    "distribution_slope_agree_mean": "statistics.diagnostics.distribution.accuracy_curve.slope_agree.mean",
    "distribution_history_kernel_score": "statistics.scores.distribution.multiobjective.components.history_kernel",
    "distribution_history_kernel_mse": "statistics.diagnostics.distribution.history_kernel.mse",
    "distribution_history_kernel_corr": "statistics.diagnostics.distribution.history_kernel.corr",
    "distribution_history_kernel_corr_loss": "statistics.diagnostics.distribution.history_kernel.corr_loss",
    "distribution_history_kernel_norm_ratio": "statistics.diagnostics.distribution.history_kernel.norm_ratio",
    "distribution_history_kernel_human_norm": "statistics.diagnostics.distribution.history_kernel.human_norm",
    "distribution_history_kernel_model_norm": "statistics.diagnostics.distribution.history_kernel.model_norm",
    "distribution_history_kernel_run_count": "statistics.diagnostics.distribution.history_kernel.run_count",
    "distribution_switch_behavior_score": "statistics.scores.distribution.multiobjective.components.switch_behavior",
    "distribution_switch_human": "statistics.diagnostics.distribution.switch_behavior.switch.human",
    "distribution_switch_model": "statistics.diagnostics.distribution.switch_behavior.switch.model",
    "distribution_switch_abs_diff": "statistics.diagnostics.distribution.switch_behavior.switch.abs_diff",
    "distribution_perseveration_human": "statistics.diagnostics.distribution.switch_behavior.perseveration.human",
    "distribution_perseveration_model": "statistics.diagnostics.distribution.switch_behavior.perseveration.model",
    "distribution_perseveration_abs_diff": "statistics.diagnostics.distribution.switch_behavior.perseveration.abs_diff",
    "distribution_win_stay_human": "statistics.diagnostics.distribution.switch_behavior.win_stay.human",
    "distribution_win_stay_model": "statistics.diagnostics.distribution.switch_behavior.win_stay.model",
    "distribution_win_stay_abs_diff": "statistics.diagnostics.distribution.switch_behavior.win_stay.abs_diff",
    "distribution_lose_shift_human": "statistics.diagnostics.distribution.switch_behavior.lose_shift.human",
    "distribution_lose_shift_model": "statistics.diagnostics.distribution.switch_behavior.lose_shift.model",
    "distribution_lose_shift_abs_diff": "statistics.diagnostics.distribution.switch_behavior.lose_shift.abs_diff",
    "distribution_switch_run_count": "statistics.diagnostics.distribution.switch_behavior.run_count",
    "distribution_ppc_interval_score": "statistics.scores.distribution.ppc_interval.score",
    "distribution_ppc_interval_violation_count": "statistics.scores.distribution.ppc_interval.violation_count",
    "distribution_ppc_interval_accept": "statistics.scores.distribution.ppc_interval.accept",
    "distribution_ppc_interval_alpha": "statistics.scores.distribution.ppc_interval.alpha",
    "distribution_ppc_interval_stat_count": "statistics.scores.distribution.ppc_interval.stat_count",
    "distribution_ppc_interval_lower_quantile": "statistics.scores.distribution.ppc_interval.lower_quantile",
    "distribution_ppc_interval_upper_quantile": "statistics.scores.distribution.ppc_interval.upper_quantile",
}

for _name in (
    "acc_mae",
    "vol_ratio_low",
    "vol_ratio_high",
    "history_corr",
    "switch_rate",
    "win_stay",
    "lose_shift",
    "perseveration",
):
    METRIC_COLUMN_PATHS[f"distribution_intersection_{_name}_violation"] = (
        f"statistics.scores.distribution.intersection.violations.{_name}"
    )

for _label in (
    "acc_mean",
    "acc_vol",
    "acc_range",
    "history_kernel_norm",
    "switch_rate",
    "win_stay",
    "lose_shift",
    "perseveration",
):
    for _column_suffix, _path_suffix in (
        ("human", "human"),
        ("model_q05", "model_lower"),
        ("model_q95", "model_upper"),
        ("model_median", "model_median"),
        ("percentile", "percentile"),
        ("tail_score", "tail_score"),
        ("violation", "violation"),
        ("accept", "accept"),
    ):
        METRIC_COLUMN_PATHS[f"distribution_ppc_interval_{_label}_{_column_suffix}"] = (
            f"statistics.scores.distribution.ppc_interval.stats.{_label}.{_path_suffix}"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _metric_column(metrics: Mapping[str, Any], column: str, default: Any = None) -> Any:
    if column in metrics:
        return metrics[column]
    path = METRIC_COLUMN_PATHS.get(column)
    if path:
        return get_stat_value(metrics, path, default)
    return default


def _metrics_summary_for_subject(summary: Any, subject_id: int | None) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        return {}
    subjects = summary.get("subjects")
    if isinstance(subjects, Mapping):
        if subject_id is not None and str(subject_id) in subjects and isinstance(subjects[str(subject_id)], Mapping):
            return subjects[str(subject_id)]
        for value in subjects.values():
            if isinstance(value, Mapping):
                return value
        return {}
    return summary


def _strategy_signature(strategy_kwargs: Mapping[str, Any] | None) -> str:
    if not isinstance(strategy_kwargs, Mapping):
        return "missing"
    init_num = strategy_kwargs.get("init_num", "?")
    strategies = strategy_kwargs.get("strategies")
    if not isinstance(strategies, list):
        return f"init{init_num}:no_strategies"
    parts = []
    for strat in strategies:
        if not isinstance(strat, Mapping):
            continue
        parts.append(
            "|".join(
                str(strat.get(key, ""))
                for key in ("label", "amount", "method", "pool")
            )
        )
    return f"init{init_num}:" + "+".join(parts)


def load_strategy_lookup(
    candidates_json: Path | None,
    *,
    candidate_key: str = "cond1",
    value_key: str = "hypo_transitions_kwargs",
) -> dict[str, dict[str, str]]:
    """Map canonical transition kwargs to candidate metadata."""
    if candidates_json is None or not candidates_json.is_file():
        return {}
    payload = _load_json(candidates_json)
    candidates = payload.get(candidate_key) if isinstance(payload, Mapping) else None
    if not isinstance(candidates, list):
        return {}

    lookup: dict[str, dict[str, str]] = {}
    for idx, item in enumerate(candidates):
        if not isinstance(item, Mapping) or value_key not in item:
            continue
        kwargs = item[value_key]
        lookup[_canonical_json(kwargs)] = {
            "id": str(item.get("id", f"candidate_{idx}")),
            "description": str(item.get("description", "")),
        }
    return lookup


def identify_strategy(
    strategy_kwargs: Mapping[str, Any] | None,
    strategy_lookup: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[str, str]:
    if not isinstance(strategy_kwargs, Mapping):
        return "missing", "missing"
    key = _canonical_json(strategy_kwargs)
    if strategy_lookup and key in strategy_lookup:
        meta = strategy_lookup[key]
        return str(meta.get("id", "unknown")), str(meta.get("description", ""))
    return _strategy_signature(strategy_kwargs), ""


def flatten_hyperparams(
    hyperparams: Mapping[str, Any] | None,
    strategy_lookup: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Extract analysis-friendly fields from a hyperparameter dictionary."""
    hp = hyperparams if isinstance(hyperparams, Mapping) else {}
    memory = hp.get(MEMORY_KEY)
    if not isinstance(memory, Mapping):
        memory = {}
    transition = hp.get(TRANSITION_KEY)
    if not isinstance(transition, Mapping):
        transition = {}
    strategy_id, strategy_description = identify_strategy(transition, strategy_lookup)

    return {
        "gamma": _safe_float(memory.get("gamma", hp.get("gamma"))),
        "w0": _safe_float(memory.get("w0", hp.get("w0"))),
        "strategy_id": strategy_id,
        "strategy_description": strategy_description,
        "strategy_signature": _strategy_signature(transition),
        "init_num": transition.get("init_num"),
        "beta_init": _safe_float(hp.get(BETA_INIT_KEY, hp.get("beta_init"))),
        "decrease_rate": _safe_float(hp.get(DECREASE_RATE_KEY, hp.get("decrease_rate"))),
        "prior_beta_scale": _safe_float(
            hp.get(PRIOR_BETA_SCALE_KEY, hp.get("prior_beta_scale"))
        ),
        "correct_additive": _safe_float(
            hp.get(CORRECT_ADDITIVE_KEY, hp.get("correct_additive"))
        ),
        "beta_update_mode": hp.get(BETA_UPDATE_MODE_KEY, hp.get("beta_update_mode")),
        "probabilistic_feedback_lapse": _safe_float(
            hp.get(
                PROBABILISTIC_FEEDBACK_LAPSE_KEY,
                hp.get("probabilistic_feedback_lapse", hp.get("feedback_lapse")),
            )
        ),
        "prior_reset_base": _safe_float(
            transition.get(
                "prior_reset_base",
                hp.get(PRIOR_RESET_BASE_KEY, hp.get("prior_reset_base")),
            )
        ),
        "prior_reset_post_error": _safe_float(
            transition.get(
                "prior_reset_post_error",
                hp.get(PRIOR_RESET_POST_ERROR_KEY, hp.get("prior_reset_post_error")),
            )
        ),
        "prior_reset_low_accuracy": _safe_float(
            transition.get(
                "prior_reset_low_accuracy",
                hp.get(PRIOR_RESET_LOW_ACCURACY_KEY, hp.get("prior_reset_low_accuracy")),
            )
        ),
        "prior_reset_threshold": _safe_float(
            transition.get(
                "prior_reset_threshold",
                hp.get(PRIOR_RESET_THRESHOLD_KEY, hp.get("prior_reset_threshold")),
            )
        ),
        "prior_reset_window": _safe_float(
            transition.get(
                "prior_reset_window",
                hp.get(PRIOR_RESET_WINDOW_KEY, hp.get("prior_reset_window")),
            )
        ),
        "prior_reset_decay": _safe_float(
            transition.get(
                "prior_reset_decay",
                hp.get(PRIOR_RESET_DECAY_KEY, hp.get("prior_reset_decay")),
            )
        ),
        "prior_reset_max": _safe_float(
            transition.get(
                "prior_reset_max",
                hp.get(PRIOR_RESET_MAX_KEY, hp.get("prior_reset_max")),
            )
        ),
        "prior_reset_target": transition.get(
            "prior_reset_target",
            hp.get(
                PRIOR_RESET_TARGET_KEY,
                hp.get("prior_reset_target"),
            ),
        ),
        "prior_reset_source": transition.get(
            "prior_reset_source",
            hp.get(PRIOR_RESET_SOURCE_KEY, hp.get("prior_reset_source")),
        ),
        "prior_reset_volatility_gain": _safe_float(
            transition.get(
                "prior_reset_volatility_gain",
                hp.get(
                    PRIOR_RESET_VOLATILITY_GAIN_KEY,
                    hp.get("prior_reset_volatility_gain"),
                ),
            )
        ),
        "latent_volatility_base": _safe_float(
            transition.get(
                "latent_volatility_base",
                hp.get(LATENT_VOLATILITY_BASE_KEY, hp.get("latent_volatility_base")),
            )
        ),
        "latent_volatility_error_gain": _safe_float(
            transition.get(
                "latent_volatility_error_gain",
                hp.get(
                    LATENT_VOLATILITY_ERROR_GAIN_KEY,
                    hp.get("latent_volatility_error_gain"),
                ),
            )
        ),
        "latent_volatility_low_accuracy_gain": _safe_float(
            transition.get(
                "latent_volatility_low_accuracy_gain",
                hp.get(
                    LATENT_VOLATILITY_LOW_ACCURACY_GAIN_KEY,
                    hp.get("latent_volatility_low_accuracy_gain"),
                ),
            )
        ),
        "latent_volatility_threshold": _safe_float(
            transition.get(
                "latent_volatility_threshold",
                hp.get(
                    LATENT_VOLATILITY_THRESHOLD_KEY,
                    hp.get("latent_volatility_threshold"),
                ),
            )
        ),
        "latent_volatility_window": _safe_float(
            transition.get(
                "latent_volatility_window",
                hp.get(LATENT_VOLATILITY_WINDOW_KEY, hp.get("latent_volatility_window")),
            )
        ),
        "latent_volatility_decay": _safe_float(
            transition.get(
                "latent_volatility_decay",
                hp.get(LATENT_VOLATILITY_DECAY_KEY, hp.get("latent_volatility_decay")),
            )
        ),
        "latent_volatility_max": _safe_float(
            transition.get(
                "latent_volatility_max",
                hp.get(LATENT_VOLATILITY_MAX_KEY, hp.get("latent_volatility_max")),
            )
        ),
        "latent_volatility_feedback_mode": transition.get(
            "latent_volatility_feedback_mode",
            hp.get(
                LATENT_VOLATILITY_FEEDBACK_MODE_KEY,
                hp.get("latent_volatility_feedback_mode"),
            ),
        ),
        "output_base_lapse": _safe_float(
            hp.get(OUTPUT_NOISE_BASE_LAPSE_KEY, hp.get("output_base_lapse"))
        ),
        "output_post_error_lapse": _safe_float(
            hp.get(OUTPUT_NOISE_POST_ERROR_LAPSE_KEY, hp.get("output_post_error_lapse"))
        ),
        "output_low_accuracy_lapse": _safe_float(
            hp.get(OUTPUT_NOISE_LOW_ACCURACY_LAPSE_KEY, hp.get("output_low_accuracy_lapse"))
        ),
        "output_low_accuracy_threshold": _safe_float(
            hp.get(
                OUTPUT_NOISE_LOW_ACCURACY_THRESHOLD_KEY,
                hp.get("output_low_accuracy_threshold"),
            )
        ),
        "output_lapse_decay": _safe_float(
            hp.get(OUTPUT_NOISE_LAPSE_DECAY_KEY, hp.get("output_lapse_decay"))
        ),
        "output_max_lapse": _safe_float(
            hp.get(OUTPUT_NOISE_MAX_LAPSE_KEY, hp.get("output_max_lapse"))
        ),
        "output_recent_accuracy_window": _safe_float(
            hp.get(
                OUTPUT_NOISE_RECENT_ACCURACY_WINDOW_KEY,
                hp.get("output_recent_accuracy_window"),
            )
        ),
        "output_lapse_target": hp.get(
            OUTPUT_NOISE_LAPSE_TARGET_KEY,
            hp.get("output_lapse_target"),
        ),
        "output_latent_volatility_lapse": _safe_float(
            hp.get(
                OUTPUT_NOISE_LATENT_VOLATILITY_LAPSE_KEY,
                hp.get("output_latent_volatility_lapse"),
            )
        ),
        "output_latent_volatility_power": _safe_float(
            hp.get(
                OUTPUT_NOISE_LATENT_VOLATILITY_POWER_KEY,
                hp.get("output_latent_volatility_power"),
            )
        ),
        "distance_mode": hp.get(DISTANCE_MODE_KEY, hp.get("distance_mode")),
        "hyperparam_signature": _canonical_json(hp),
    }


def _combination_map(subject_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _iter_jsonl(subject_dir / "all_combinations.jsonl"):
        stage = str(row.get("stage", ""))
        idx = row.get("combination_index")
        if idx is None:
            continue
        out[(stage, int(idx))] = row
    return out


def _subject_id_from_dir(subject_dir: Path) -> int | None:
    name = subject_dir.name
    if name.startswith("subject_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def discover_subject_dirs(hyper_dir: Path, subjects: Sequence[int] | None = None) -> list[Path]:
    wanted = {int(sid) for sid in subjects} if subjects else None
    dirs = []
    for path in sorted(hyper_dir.glob("subject_*")):
        if not path.is_dir():
            continue
        sid = _subject_id_from_dir(path)
        if sid is None:
            continue
        if wanted is not None and sid not in wanted:
            continue
        dirs.append(path)
    return dirs


def load_restart_table(
    subject_dir: Path,
    *,
    stage: str = "coarse",
    strategy_lookup: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    subject_id = _subject_id_from_dir(subject_dir)
    summary_path = subject_dir / "restart_summary.json"
    if not summary_path.is_file():
        return pd.DataFrame()
    payload = _load_json(summary_path)
    restarts = payload.get(stage) if isinstance(payload, Mapping) else None
    if not isinstance(restarts, list):
        return pd.DataFrame()

    combo_map = _combination_map(subject_dir)
    rows: list[dict[str, Any]] = []
    for restart in restarts:
        if not isinstance(restart, Mapping):
            continue
        restart_id = int(restart.get("restart_id", len(rows)))
        initial_idx = restart.get("initial_combination_index")
        final_idx = restart.get("best_combination_index")
        initial_record = combo_map.get((stage, int(initial_idx))) if initial_idx is not None else None
        final_record = combo_map.get((stage, int(final_idx))) if final_idx is not None else None

        initial_hp = (
            initial_record.get("hyperparams")
            if isinstance(initial_record, Mapping)
            else restart.get("initial_hyperparams")
        )
        final_hp = (
            restart.get("best_params")
            or restart.get("best_hyperparams")
            or (final_record.get("hyperparams") if isinstance(final_record, Mapping) else None)
        )
        initial_flat = flatten_hyperparams(initial_hp, strategy_lookup)
        final_flat = flatten_hyperparams(final_hp, strategy_lookup)
        row: dict[str, Any] = {
            "subject_id": subject_id,
            "stage": stage,
            "restart_id": restart_id,
            "initial_combination_index": initial_idx,
            "final_combination_index": final_idx,
            "initial_error": _safe_float(
                restart.get(
                    "initial_error",
                    initial_record.get("aggregated_error") if isinstance(initial_record, Mapping) else None,
                )
            ),
            "final_error": _safe_float(restart.get("best_error")),
            "outer_iters_completed": restart.get("outer_iters_completed"),
            "stopped_by": restart.get("stopped_by"),
            "num_improvements": restart.get("num_improvements"),
            "num_new_evaluations": restart.get("num_new_evaluations"),
            "num_cache_hits": restart.get("num_cache_hits"),
        }
        for key, value in initial_flat.items():
            row[f"initial_{key}"] = value
        for key, value in final_flat.items():
            row[f"final_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_coordinate_trace_table(
    subject_dir: Path,
    *,
    stage: str = "coarse",
    strategy_lookup: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    subject_id = _subject_id_from_dir(subject_dir)
    trace_rows = [
        row for row in _iter_jsonl(subject_dir / "coordinate_trace.jsonl")
        if str(row.get("stage", "")) == stage
    ]
    if not trace_rows:
        return pd.DataFrame()

    combo_map = _combination_map(subject_dir)
    rows: list[dict[str, Any]] = []
    for row in trace_rows:
        restart_id = int(row.get("restart_id", -1))
        end_idx = row.get("end_best_combination_index")
        end_record = combo_map.get((stage, int(end_idx))) if end_idx is not None else None
        end_hp = end_record.get("hyperparams") if isinstance(end_record, Mapping) else None
        end_flat = flatten_hyperparams(end_hp, strategy_lookup)
        out = dict(row)
        out["subject_id"] = subject_id
        for key, value in end_flat.items():
            out[f"current_{key}"] = value
        out["restart_id"] = restart_id
        rows.append(out)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_cols = [c for c in ("restart_id", "iter_id", "coordinate_index") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df["step_in_restart"] = df.groupby("restart_id").cumcount() + 1
    return df


def build_best_error_trajectory(restart_df: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not restart_df.empty:
        for _, row in restart_df.iterrows():
            rows.append(
                {
                    "subject_id": row.get("subject_id"),
                    "stage": row.get("stage"),
                    "restart_id": row.get("restart_id"),
                    "step_in_restart": 0,
                    "coordinate": "init",
                    "best_error": row.get("initial_error"),
                    "improved": True,
                    "current_gamma": row.get("initial_gamma"),
                    "current_w0": row.get("initial_w0"),
                    "current_strategy_id": row.get("initial_strategy_id"),
                    "current_beta_init": row.get("initial_beta_init"),
                    "current_decrease_rate": row.get("initial_decrease_rate"),
                    "current_prior_beta_scale": row.get("initial_prior_beta_scale"),
                    "current_correct_additive": row.get("initial_correct_additive"),
                }
            )
    if not trace_df.empty:
        for _, row in trace_df.iterrows():
            out = {
                "subject_id": row.get("subject_id"),
                "stage": row.get("stage"),
                "restart_id": row.get("restart_id"),
                "step_in_restart": row.get("step_in_restart"),
                "coordinate": row.get("coordinate"),
                "best_error": row.get("end_best_error"),
                "improved": row.get("improved"),
            }
            for key in (
                "gamma",
                "w0",
                "strategy_id",
                "beta_init",
                "decrease_rate",
                "prior_beta_scale",
                "correct_additive",
            ):
                out[f"current_{key}"] = row.get(f"current_{key}")
            rows.append(out)
    return pd.DataFrame(rows)


def _pairwise_distances(rows: pd.DataFrame, *, prefix: str) -> list[float]:
    if len(rows) < 2:
        return []
    numeric_cols = [f"{prefix}_{name}" for name in NUMERIC_PARAM_COLUMNS]
    categorical_cols = [f"{prefix}_{name}" for name in CATEGORICAL_PARAM_COLUMNS]
    ranges: dict[str, float] = {}
    for col in numeric_cols:
        if col not in rows.columns:
            continue
        values = pd.to_numeric(rows[col], errors="coerce")
        finite = values[np.isfinite(values)]
        span = float(finite.max() - finite.min()) if len(finite) else 0.0
        ranges[col] = span if span > 0 else 1.0

    distances: list[float] = []
    for left_idx in range(len(rows)):
        left = rows.iloc[left_idx]
        for right_idx in range(left_idx + 1, len(rows)):
            right = rows.iloc[right_idx]
            components: list[float] = []
            for col in numeric_cols:
                if col not in rows.columns:
                    continue
                left_val = _safe_float(left.get(col))
                right_val = _safe_float(right.get(col))
                if np.isfinite(left_val) and np.isfinite(right_val):
                    components.append(abs(left_val - right_val) / ranges.get(col, 1.0))
            for col in categorical_cols:
                if col not in rows.columns:
                    continue
                left_val = left.get(col)
                right_val = right.get(col)
                if pd.notna(left_val) and pd.notna(right_val):
                    components.append(0.0 if str(left_val) == str(right_val) else 1.0)
            if components:
                distances.append(float(np.mean(components)))
    return distances


def summarize_restart_convergence(restart_df: pd.DataFrame) -> pd.DataFrame:
    if restart_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for subject_id, group in restart_df.groupby("subject_id", dropna=False):
        final_errors = pd.to_numeric(group["final_error"], errors="coerce")
        best_idx = final_errors.idxmin()
        best_error = float(final_errors.loc[best_idx])
        final_signatures = group.get("final_hyperparam_signature", pd.Series(dtype=object))
        exact_mode_count = int(final_signatures.value_counts(dropna=True).max()) if len(final_signatures) else 0
        strategy_counts = group.get("final_strategy_id", pd.Series(dtype=object)).value_counts(dropna=True)
        strategy_mode_count = int(strategy_counts.max()) if len(strategy_counts) else 0
        init_dist = _pairwise_distances(group.reset_index(drop=True), prefix="initial")
        final_dist = _pairwise_distances(group.reset_index(drop=True), prefix="final")
        rows.append(
            {
                "subject_id": subject_id,
                "n_restarts": int(len(group)),
                "best_restart_id": int(group.loc[best_idx, "restart_id"]),
                "best_error": best_error,
                "final_error_mean": float(np.nanmean(final_errors)),
                "final_error_std": float(np.nanstd(final_errors)),
                "final_error_range": float(np.nanmax(final_errors) - np.nanmin(final_errors)),
                "exact_final_mode_count": exact_mode_count,
                "exact_final_mode_fraction": exact_mode_count / max(1, len(group)),
                "strategy_mode_count": strategy_mode_count,
                "strategy_mode_fraction": strategy_mode_count / max(1, len(group)),
                "initial_pairwise_distance_mean": float(np.mean(init_dist)) if init_dist else np.nan,
                "final_pairwise_distance_mean": float(np.mean(final_dist)) if final_dist else np.nan,
                "distance_contraction": (
                    float(np.mean(init_dist) - np.mean(final_dist))
                    if init_dist and final_dist
                    else np.nan
                ),
                "best_gamma": group.loc[best_idx].get("final_gamma"),
                "best_w0": group.loc[best_idx].get("final_w0"),
                "best_strategy_id": group.loc[best_idx].get("final_strategy_id"),
                "best_beta_init": group.loc[best_idx].get("final_beta_init"),
                "best_decrease_rate": group.loc[best_idx].get("final_decrease_rate"),
                "best_prior_beta_scale": group.loc[best_idx].get("final_prior_beta_scale"),
                "best_correct_additive": group.loc[best_idx].get("final_correct_additive"),
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def _plot_error_trajectory(traj_df: pd.DataFrame, path: Path, subject_id: Any) -> None:
    if traj_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for restart_id, group in traj_df.groupby("restart_id"):
        group = group.sort_values("step_in_restart")
        ax.plot(
            group["step_in_restart"],
            group["best_error"],
            marker="o",
            linewidth=1.2,
            markersize=3,
            alpha=0.85,
            label=f"R{int(restart_id)}",
        )
    ax.set_title(f"Subject {subject_id} hyper-CD best error by restart")
    ax.set_xlabel("Coordinate step")
    ax.set_ylabel("Best error so far")
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_final_memory(restart_df: pd.DataFrame, path: Path, subject_id: Any) -> None:
    if restart_df.empty or "final_gamma" not in restart_df or "final_w0" not in restart_df:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    errors = pd.to_numeric(restart_df["final_error"], errors="coerce")
    scatter = ax.scatter(
        restart_df["final_gamma"],
        restart_df["final_w0"],
        c=errors,
        s=80,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.5,
    )
    for _, row in restart_df.iterrows():
        ax.text(row["final_gamma"], row["final_w0"], str(int(row["restart_id"])), fontsize=8)
    ax.set_title(f"Subject {subject_id} final memory params by restart")
    ax.set_xlabel("Final gamma")
    ax.set_ylabel("Final w0")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Final error")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_distance_summary(summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary_df))
    width = 0.38
    ax.bar(
        x - width / 2,
        summary_df["initial_pairwise_distance_mean"],
        width=width,
        label="Initial",
        color="#9aa7b1",
    )
    ax.bar(
        x + width / 2,
        summary_df["final_pairwise_distance_mean"],
        width=width,
        label="Final",
        color="#2f6f9f",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in summary_df["subject_id"]], rotation=45)
    ax.set_ylabel("Mean pairwise mixed distance")
    ax.set_xlabel("Subject")
    ax.set_title("Restart convergence: initial vs final hyperparameter spread")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_strategy_counts(restart_df: pd.DataFrame, path: Path) -> None:
    if restart_df.empty or "final_strategy_id" not in restart_df:
        return
    table = pd.crosstab(restart_df["subject_id"], restart_df["final_strategy_id"])
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(table)), 5.5))
    table.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_ylabel("Restart count")
    ax.set_xlabel("Subject")
    ax.set_title("Final transition strategy selected by restarts")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_hyper_cd_convergence(
    hyper_dir: Path,
    *,
    output_dir: Path | None = None,
    subjects: Sequence[int] | None = None,
    stage: str = "coarse",
    candidates_json: Path | None = None,
    candidate_key: str = "cond1",
) -> dict[str, Path]:
    """Create restart-convergence CSVs and plots for a hyper-CD result directory."""
    hyper_dir = Path(hyper_dir).resolve()
    if output_dir is None:
        output_dir = hyper_dir / "hyper_evaluation"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_lookup = load_strategy_lookup(candidates_json, candidate_key=candidate_key)
    subject_dirs = discover_subject_dirs(hyper_dir, subjects)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject_* directories found under {hyper_dir}")

    restart_tables = []
    trace_tables = []
    trajectory_tables = []
    for subject_dir in subject_dirs:
        subject_id = _subject_id_from_dir(subject_dir)
        restart_df = load_restart_table(
            subject_dir,
            stage=stage,
            strategy_lookup=strategy_lookup,
        )
        trace_df = load_coordinate_trace_table(
            subject_dir,
            stage=stage,
            strategy_lookup=strategy_lookup,
        )
        traj_df = build_best_error_trajectory(restart_df, trace_df)
        if not restart_df.empty:
            restart_tables.append(restart_df)
            _plot_final_memory(
                restart_df,
                output_dir / f"subject_{subject_id}_final_memory.png",
                subject_id,
            )
        if not trace_df.empty:
            trace_tables.append(trace_df)
        if not traj_df.empty:
            trajectory_tables.append(traj_df)
            _plot_error_trajectory(
                traj_df,
                output_dir / f"subject_{subject_id}_optimization_trace.png",
                subject_id,
            )

    restart_all = pd.concat(restart_tables, ignore_index=True) if restart_tables else pd.DataFrame()
    trace_all = pd.concat(trace_tables, ignore_index=True) if trace_tables else pd.DataFrame()
    trajectory_all = (
        pd.concat(trajectory_tables, ignore_index=True) if trajectory_tables else pd.DataFrame()
    )
    summary = summarize_restart_convergence(restart_all)

    paths = {
        "restart_summary_flat": output_dir / "restart_summary_flat.csv",
        "coordinate_trace_flat": output_dir / "coordinate_trace_flat.csv",
        "best_error_trajectory": output_dir / "best_error_trajectory.csv",
        "convergence_summary": output_dir / "convergence_summary.csv",
        "distance_summary_plot": output_dir / "convergence_distance_summary.png",
        "strategy_counts_plot": output_dir / "final_strategy_counts.png",
    }
    restart_all.to_csv(paths["restart_summary_flat"], index=False)
    trace_all.to_csv(paths["coordinate_trace_flat"], index=False)
    trajectory_all.to_csv(paths["best_error_trajectory"], index=False)
    summary.to_csv(paths["convergence_summary"], index=False)
    _plot_distance_summary(summary, paths["distance_summary_plot"])
    _plot_strategy_counts(restart_all, paths["strategy_counts_plot"])
    return paths


def load_all_combinations_table(
    subject_dir: Path,
    *,
    stage: str = "coarse",
    strategy_lookup: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """Flatten all evaluated hyper-CD combinations for one subject."""
    subject_id = _subject_id_from_dir(subject_dir)
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(subject_dir / "all_combinations.jsonl"):
        if str(row.get("stage", "")) != str(stage):
            continue
        hp = row.get("hyperparams")
        metrics = _metrics_summary_for_subject(row.get("metrics_summary"), subject_id)
        if not isinstance(hp, Mapping) or not isinstance(metrics, Mapping):
            continue
        flat = flatten_hyperparams(hp, strategy_lookup)
        shape_metrics = {
            key: _safe_float(_metric_column(metrics, key))
            for key in ACCURACY_SHAPE_COLUMNS
        }
        history_metrics = {
            key: (
                _metric_column(metrics, key)
                if key in {"history_kernel_human", "history_kernel_model"}
                else _safe_float(_metric_column(metrics, key))
            )
            for key in HISTORY_KERNEL_COLUMNS
        }
        switch_metrics = {
            key: _safe_float(_metric_column(metrics, key))
            for key in SWITCH_BEHAVIOR_COLUMNS
        }
        distribution_metrics = {
            key: _safe_float(_metric_column(metrics, key))
            for key in DISTRIBUTION_BEHAVIOR_COLUMNS
        }
        rows.append(
            {
                "subject_id": subject_id,
                "stage": row.get("stage"),
                "combination_index": int(row.get("combination_index", -1)),
                "restart_id": row.get("restart_id"),
                "iter_id": row.get("iter_id"),
                "coordinate": row.get("coordinate"),
                "hyper_candidate_seed": row.get("hyper_candidate_seed"),
                "hyperparams": deepcopy(dict(hp)),
                "hyperparam_signature": flat["hyperparam_signature"],
                "hyper_selection_error": _safe_float(
                    _metric_column(metrics, "hyper_selection_error", row.get("aggregated_error"))
                ),
                "hyper_mean_error": _safe_float(_metric_column(metrics, "hyper_mean_error")),
                "hyper_best_error": _safe_float(_metric_column(metrics, "hyper_best_error")),
                "hyper_best10_mean_error": _safe_float(_metric_column(metrics, "hyper_best10_mean_error")),
                "hyper_q10_error": _safe_float(_metric_column(metrics, "hyper_q10_error")),
                "hyper_std_error": _safe_float(_metric_column(metrics, "hyper_std_error")),
                "hyper_simulation_repeats": _metric_column(metrics, "hyper_simulation_repeats"),
                **flat,
                **shape_metrics,
                **history_metrics,
                **switch_metrics,
                **distribution_metrics,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["hyper_selection_error", "combination_index"]).reset_index(drop=True)


def _load_subject_best_signature(
    subject_dir: Path,
    strategy_lookup: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    path = subject_dir / "best_hyperparams.json"
    if not path.is_file():
        return None, {}
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        return None, {}
    selected = payload.get("selected")
    if isinstance(selected, Mapping):
        hp = selected.get("best_hyperparams") or selected.get("hyperparams")
    else:
        hp = payload.get("best_hyperparams") or payload.get("hyperparams")
    if not isinstance(hp, Mapping):
        return None, {}
    flat = flatten_hyperparams(hp, strategy_lookup)
    return str(flat["hyperparam_signature"]), flat


def _deduplicate_combinations(combo_df: pd.DataFrame) -> pd.DataFrame:
    if combo_df.empty or "hyperparam_signature" not in combo_df.columns:
        return combo_df.copy()
    sort_cols = [
        col
        for col in (
            "hyper_selection_error",
            "accuracy_shape_score",
            "hyper_best_error",
            "combination_index",
        )
        if col in combo_df.columns
    ]
    dedup = combo_df.copy()
    if sort_cols:
        dedup = dedup.sort_values(sort_cols, na_position="last")
    dedup = dedup.drop_duplicates("hyperparam_signature", keep="first")
    return dedup.reset_index(drop=True)


def select_near_optimal_plateau(
    combo_df: pd.DataFrame,
    *,
    primary_metric: str = "hyper_selection_error",
    abs_tol: float = 0.02,
    rel_tol: float = 0.08,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return unique candidates whose primary score is close to the subject best."""
    if combo_df.empty or primary_metric not in combo_df.columns:
        return pd.DataFrame(), {
            "best_primary_error": np.nan,
            "plateau_cutoff": np.nan,
            "primary_abs_tol": float(abs_tol),
            "primary_rel_tol": float(rel_tol),
        }

    unique = _deduplicate_combinations(combo_df)
    values = pd.to_numeric(unique[primary_metric], errors="coerce")
    finite = np.isfinite(values)
    if not finite.any():
        return pd.DataFrame(), {
            "best_primary_error": np.nan,
            "plateau_cutoff": np.nan,
            "primary_abs_tol": float(abs_tol),
            "primary_rel_tol": float(rel_tol),
        }

    best = float(np.nanmin(values))
    cutoff = max(best + float(abs_tol), best * (1.0 + float(rel_tol)))
    near = unique.loc[values <= cutoff].copy()
    near["plateau_primary_metric"] = str(primary_metric)
    near["plateau_best_primary_error"] = best
    near["plateau_cutoff"] = cutoff
    near["plateau_delta"] = pd.to_numeric(near[primary_metric], errors="coerce") - best
    near = near.sort_values(
        [primary_metric, "accuracy_shape_score", "combination_index"],
        na_position="last",
    ).reset_index(drop=True)
    near["plateau_rank"] = np.arange(1, len(near) + 1)
    return near, {
        "best_primary_error": best,
        "plateau_cutoff": cutoff,
        "primary_abs_tol": float(abs_tol),
        "primary_rel_tol": float(rel_tol),
    }


def _mode_count(values: pd.Series) -> tuple[Any, int, float]:
    clean = values.dropna()
    if clean.empty:
        return np.nan, 0, np.nan
    counts = clean.astype(str).value_counts()
    mode = counts.index[0]
    count = int(counts.iloc[0])
    return mode, count, count / max(1, int(len(clean)))


def summarize_near_optimal_plateau(
    unique_df: pd.DataFrame,
    near_df: pd.DataFrame,
    *,
    primary_metric: str = "hyper_selection_error",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if unique_df.empty:
        return pd.DataFrame()

    for subject_id, all_group in unique_df.groupby("subject_id", dropna=False):
        near_group = near_df[near_df["subject_id"] == subject_id] if not near_df.empty else pd.DataFrame()
        all_values = pd.to_numeric(all_group.get(primary_metric), errors="coerce")
        near_values = pd.to_numeric(near_group.get(primary_metric), errors="coerce") if not near_group.empty else pd.Series(dtype=float)
        best_value = float(np.nanmin(all_values)) if np.isfinite(all_values).any() else np.nan
        cutoff = float(near_group["plateau_cutoff"].iloc[0]) if not near_group.empty else np.nan
        strategy_mode, strategy_mode_count, strategy_mode_fraction = _mode_count(
            near_group.get("strategy_id", pd.Series(dtype=object))
        )
        selected_rows = (
            near_group[near_group.get("is_subject_selected", False).astype(bool)]
            if "is_subject_selected" in near_group
            else pd.DataFrame()
        )
        best_shape_row = None
        if "accuracy_shape_score" in near_group and not near_group.empty:
            shape_values = pd.to_numeric(near_group["accuracy_shape_score"], errors="coerce")
            if np.isfinite(shape_values).any():
                best_shape_row = near_group.loc[shape_values.idxmin()]
        best_history_row = None
        if "history_kernel_score" in near_group and not near_group.empty:
            history_values = pd.to_numeric(near_group["history_kernel_score"], errors="coerce")
            if np.isfinite(history_values).any():
                best_history_row = near_group.loc[history_values.idxmin()]
        best_primary_row = all_group.loc[all_values.idxmin()] if np.isfinite(all_values).any() else None

        row: dict[str, Any] = {
            "subject_id": subject_id,
            "primary_metric": primary_metric,
            "n_unique_evaluated": int(len(all_group)),
            "n_near_optimal": int(len(near_group)),
            "near_optimal_fraction": int(len(near_group)) / max(1, int(len(all_group))),
            "best_primary_error": best_value,
            "plateau_cutoff": cutoff,
            "near_primary_min": float(np.nanmin(near_values)) if np.isfinite(near_values).any() else np.nan,
            "near_primary_median": float(np.nanmedian(near_values)) if np.isfinite(near_values).any() else np.nan,
            "near_primary_max": float(np.nanmax(near_values)) if np.isfinite(near_values).any() else np.nan,
            "strategy_unique_count": int(near_group["strategy_id"].nunique(dropna=True)) if "strategy_id" in near_group else 0,
            "strategy_mode": strategy_mode,
            "strategy_mode_count": strategy_mode_count,
            "strategy_mode_fraction": strategy_mode_fraction,
            "memory_pair_unique_count": (
                int(near_group[["gamma", "w0"]].dropna().drop_duplicates().shape[0])
                if {"gamma", "w0"}.issubset(near_group.columns)
                else 0
            ),
            "selected_in_plateau": bool(not selected_rows.empty),
            "selected_combination_index": (
                int(selected_rows["combination_index"].iloc[0]) if not selected_rows.empty else np.nan
            ),
            "selected_primary_error": (
                float(selected_rows[primary_metric].iloc[0]) if not selected_rows.empty and primary_metric in selected_rows else np.nan
            ),
        }
        for col in NUMERIC_PARAM_COLUMNS:
            if col not in near_group.columns:
                continue
            values = pd.to_numeric(near_group[col], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{col}_unique_count"] = int(finite.nunique()) if len(finite) else 0
            row[f"{col}_min"] = float(finite.min()) if len(finite) else np.nan
            row[f"{col}_median"] = float(finite.median()) if len(finite) else np.nan
            row[f"{col}_max"] = float(finite.max()) if len(finite) else np.nan
            row[f"{col}_range"] = float(finite.max() - finite.min()) if len(finite) else np.nan

        if best_primary_row is not None:
            row["best_primary_combination_index"] = int(best_primary_row.get("combination_index", -1))
            row["best_primary_gamma"] = best_primary_row.get("gamma")
            row["best_primary_w0"] = best_primary_row.get("w0")
            row["best_primary_strategy_id"] = best_primary_row.get("strategy_id")
        if best_shape_row is not None:
            row["best_shape_combination_index"] = int(best_shape_row.get("combination_index", -1))
            row["best_shape_score"] = float(best_shape_row.get("accuracy_shape_score"))
            row["best_shape_gamma"] = best_shape_row.get("gamma")
            row["best_shape_w0"] = best_shape_row.get("w0")
            row["best_shape_strategy_id"] = best_shape_row.get("strategy_id")
            row["best_shape_primary_error"] = best_shape_row.get(primary_metric)
        if best_history_row is not None:
            row["best_history_combination_index"] = int(best_history_row.get("combination_index", -1))
            row["best_history_score"] = float(best_history_row.get("history_kernel_score"))
            row["best_history_kernel_mse"] = best_history_row.get("history_kernel_mse")
            row["best_history_kernel_corr"] = best_history_row.get("history_kernel_corr")
            row["best_history_gamma"] = best_history_row.get("gamma")
            row["best_history_w0"] = best_history_row.get("w0")
            row["best_history_strategy_id"] = best_history_row.get("strategy_id")
            row["best_history_primary_error"] = best_history_row.get(primary_metric)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def _build_plateau_strategy_counts(near_df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    if near_df.empty or "strategy_id" not in near_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (subject_id, strategy_id), group in near_df.groupby(["subject_id", "strategy_id"], dropna=False):
        values = pd.to_numeric(group[primary_metric], errors="coerce")
        shape = pd.to_numeric(group.get("accuracy_shape_score"), errors="coerce")
        rows.append(
            {
                "subject_id": subject_id,
                "strategy_id": strategy_id,
                "count": int(len(group)),
                "fraction_within_subject": int(len(group)) / max(1, int((near_df["subject_id"] == subject_id).sum())),
                "primary_min": float(np.nanmin(values)) if np.isfinite(values).any() else np.nan,
                "primary_median": float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan,
                "shape_min": float(np.nanmin(shape)) if np.isfinite(shape).any() else np.nan,
                "shape_median": float(np.nanmedian(shape)) if np.isfinite(shape).any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["subject_id", "count", "primary_min"], ascending=[True, False, True])


def _build_plateau_memory_grid(near_df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    if near_df.empty or not {"gamma", "w0"}.issubset(near_df.columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (subject_id, gamma, w0), group in near_df.groupby(["subject_id", "gamma", "w0"], dropna=False):
        values = pd.to_numeric(group[primary_metric], errors="coerce")
        shape = pd.to_numeric(group.get("accuracy_shape_score"), errors="coerce")
        strategy_mode, strategy_mode_count, strategy_mode_fraction = _mode_count(group.get("strategy_id", pd.Series(dtype=object)))
        rows.append(
            {
                "subject_id": subject_id,
                "gamma": gamma,
                "w0": w0,
                "count": int(len(group)),
                "primary_min": float(np.nanmin(values)) if np.isfinite(values).any() else np.nan,
                "primary_median": float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan,
                "shape_min": float(np.nanmin(shape)) if np.isfinite(shape).any() else np.nan,
                "shape_median": float(np.nanmedian(shape)) if np.isfinite(shape).any() else np.nan,
                "strategy_unique_count": int(group["strategy_id"].nunique(dropna=True)) if "strategy_id" in group else 0,
                "strategy_mode": strategy_mode,
                "strategy_mode_count": strategy_mode_count,
                "strategy_mode_fraction": strategy_mode_fraction,
            }
        )
    return pd.DataFrame(rows).sort_values(["subject_id", "primary_min", "gamma", "w0"])


def _plot_subject_plateau_memory(
    unique_df: pd.DataFrame,
    near_df: pd.DataFrame,
    path: Path,
    subject_id: Any,
    *,
    primary_metric: str,
) -> None:
    if unique_df.empty or not {"gamma", "w0"}.issubset(unique_df.columns):
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    all_gamma = pd.to_numeric(unique_df["gamma"], errors="coerce")
    all_w0 = pd.to_numeric(unique_df["w0"], errors="coerce")
    ax.scatter(all_gamma, all_w0, s=16, color="#c7cdd3", alpha=0.35, label="Evaluated unique")

    if not near_df.empty:
        near_gamma = pd.to_numeric(near_df["gamma"], errors="coerce")
        near_w0 = pd.to_numeric(near_df["w0"], errors="coerce")
        color_values = pd.to_numeric(near_df[primary_metric], errors="coerce")
        scatter = ax.scatter(
            near_gamma,
            near_w0,
            c=color_values,
            s=58,
            cmap="viridis_r",
            edgecolor="black",
            linewidth=0.35,
            label="Near-optimal",
        )
        fig.colorbar(scatter, ax=ax, label=primary_metric)
        if "plateau_rank" in near_df.columns:
            top = near_df.sort_values("plateau_rank").head(10)
            for _, row in top.iterrows():
                ax.text(row["gamma"], row["w0"], str(int(row["plateau_rank"])), fontsize=7)
        if "is_subject_selected" in near_df.columns and near_df["is_subject_selected"].astype(bool).any():
            selected = near_df[near_df["is_subject_selected"].astype(bool)]
            ax.scatter(
                selected["gamma"],
                selected["w0"],
                marker="X",
                s=130,
                color="#d62728",
                edgecolor="white",
                linewidth=0.8,
                label="Final selected",
            )

    ax.set_title(f"Subject {subject_id}: near-optimal memory plateau")
    ax.set_xlabel("gamma")
    ax.set_ylabel("w0")
    finite_w0 = all_w0[np.isfinite(all_w0) & (all_w0 > 0)]
    if len(finite_w0) and float(finite_w0.max() / max(float(finite_w0.min()), 1e-12)) > 20:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_plateau_strategy_counts(strategy_counts: pd.DataFrame, path: Path) -> None:
    if strategy_counts.empty:
        return
    table = strategy_counts.pivot_table(
        index="subject_id",
        columns="strategy_id",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(table.columns)), 5.0))
    table.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title("Near-optimal plateau: transition strategy composition")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Unique candidate count")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_plateau_spread(summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df.empty:
        return
    cols = [
        col for col in ("gamma_unique_count", "w0_unique_count", "strategy_unique_count", "memory_pair_unique_count")
        if col in summary_df.columns
    ]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(summary_df))
    width = min(0.18, 0.75 / max(1, len(cols)))
    offsets = (np.arange(len(cols)) - (len(cols) - 1) / 2.0) * width
    labels = {
        "gamma_unique_count": "gamma",
        "w0_unique_count": "w0",
        "strategy_unique_count": "strategy",
        "memory_pair_unique_count": "gamma-w0 pair",
    }
    for offset, col in zip(offsets, cols):
        ax.bar(x + offset, summary_df[col], width=width, label=labels.get(col, col))
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in summary_df["subject_id"]])
    ax.set_title("Near-optimal plateau spread")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Unique values within plateau")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_plateau_shape_tradeoff(near_df: pd.DataFrame, path: Path, primary_metric: str) -> None:
    if near_df.empty or "accuracy_shape_score" not in near_df.columns:
        return
    finite = near_df[
        np.isfinite(pd.to_numeric(near_df[primary_metric], errors="coerce"))
        & np.isfinite(pd.to_numeric(near_df["accuracy_shape_score"], errors="coerce"))
    ]
    if finite.empty:
        return
    subjects = sorted(finite["subject_id"].dropna().unique())
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for subject_id in subjects:
        sub = finite[finite["subject_id"] == subject_id]
        ax.scatter(
            sub[primary_metric],
            sub["accuracy_shape_score"],
            s=42,
            alpha=0.75,
            label=f"S{int(subject_id)}",
        )
    ax.set_title("Near-optimal candidates: primary objective vs accuracy shape")
    ax.set_xlabel(primary_metric)
    ax.set_ylabel("accuracy_shape_score")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_plateau_report(
    summary_df: pd.DataFrame,
    strategy_counts: pd.DataFrame,
    path: Path,
    *,
    primary_metric: str,
    abs_tol: float,
    rel_tol: float,
) -> None:
    lines = [
        "# Near-Optimal Plateau Analysis",
        "",
        f"- Primary metric: `{primary_metric}`",
        f"- Plateau rule: candidates with metric <= max(best + {abs_tol:g}, best * (1 + {rel_tol:g}))",
        "- Unit of analysis: unique hyperparameter signatures.",
        "",
    ]
    if summary_df.empty:
        lines.append("No near-optimal candidates were found.")
    else:
        lines.append("## Subject Summary")
        lines.append("")
        for _, row in summary_df.iterrows():
            subject_id = int(row["subject_id"])
            lines.append(
                f"- Subject {subject_id}: {int(row['n_near_optimal'])}/{int(row['n_unique_evaluated'])} "
                f"unique candidates in plateau "
                f"(best={row['best_primary_error']:.4f}, cutoff={row['plateau_cutoff']:.4f}); "
                f"gamma unique={int(row.get('gamma_unique_count', 0))}, "
                f"w0 unique={int(row.get('w0_unique_count', 0))}, "
                f"strategy unique={int(row.get('strategy_unique_count', 0))}, "
                f"mode strategy={row.get('strategy_mode')} "
                f"({row.get('strategy_mode_fraction', np.nan):.2f})."
            )
            if pd.notna(row.get("best_shape_score", np.nan)):
                lines.append(
                    f"  Best shape within plateau: c{int(row['best_shape_combination_index'])}, "
                    f"shape={row['best_shape_score']:.4f}, "
                    f"primary={row['best_shape_primary_error']:.4f}, "
                    f"gamma={row['best_shape_gamma']}, w0={row['best_shape_w0']}, "
                    f"strategy={row['best_shape_strategy_id']}."
                )
            if pd.notna(row.get("best_history_score", np.nan)):
                lines.append(
                    f"  Best history kernel within plateau: c{int(row['best_history_combination_index'])}, "
                    f"history={row['best_history_score']:.4f}, "
                    f"primary={row['best_history_primary_error']:.4f}, "
                    f"gamma={row['best_history_gamma']}, w0={row['best_history_w0']}, "
                    f"strategy={row['best_history_strategy_id']}."
                )
            if bool(row.get("selected_in_plateau", False)):
                lines.append(
                    f"  Final selected combination c{int(row['selected_combination_index'])} is inside the plateau."
                )
            else:
                lines.append("  Final selected combination was not matched inside the plateau.")
        if not strategy_counts.empty:
            lines.extend(["", "## Dominant Strategies", ""])
            for subject_id, group in strategy_counts.groupby("subject_id"):
                top = group.sort_values(["count", "primary_min"], ascending=[False, True]).head(3)
                desc = "; ".join(
                    f"{row.strategy_id}: n={int(row['count'])}, min={row['primary_min']:.4f}"
                    for _, row in top.iterrows()
                )
                lines.append(f"- Subject {int(subject_id)}: {desc}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_near_optimal_plateau(
    hyper_dir: Path,
    *,
    output_dir: Path | None = None,
    subjects: Sequence[int] | None = None,
    stage: str = "coarse",
    candidates_json: Path | None = None,
    candidate_key: str = "cond1",
    primary_metric: str = "hyper_selection_error",
    abs_tol: float = 0.02,
    rel_tol: float = 0.08,
) -> dict[str, Path]:
    """Analyze the spread of near-equivalent hyperparameter solutions."""
    hyper_dir = Path(hyper_dir).resolve()
    if output_dir is None:
        output_dir = hyper_dir / "hyper_evaluation" / "near_optimal_plateau"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_lookup = load_strategy_lookup(candidates_json, candidate_key=candidate_key)
    subject_dirs = discover_subject_dirs(hyper_dir, subjects)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject_* directories found under {hyper_dir}")

    all_unique_tables: list[pd.DataFrame] = []
    all_near_tables: list[pd.DataFrame] = []
    for subject_dir in subject_dirs:
        subject_id = _subject_id_from_dir(subject_dir)
        combo_df = load_all_combinations_table(
            subject_dir,
            stage=stage,
            strategy_lookup=strategy_lookup,
        )
        if combo_df.empty:
            continue
        unique_df = _deduplicate_combinations(combo_df)
        selected_signature, _ = _load_subject_best_signature(subject_dir, strategy_lookup)
        unique_df["is_subject_selected"] = (
            unique_df["hyperparam_signature"].astype(str) == selected_signature
            if selected_signature
            else False
        )
        near_df, _ = select_near_optimal_plateau(
            unique_df,
            primary_metric=primary_metric,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        if not near_df.empty:
            near_df["is_subject_selected"] = (
                near_df["hyperparam_signature"].astype(str) == selected_signature
                if selected_signature
                else False
            )
        all_unique_tables.append(unique_df)
        all_near_tables.append(near_df)
        _plot_subject_plateau_memory(
            unique_df,
            near_df,
            output_dir / f"subject_{subject_id}_plateau_memory.png",
            subject_id,
            primary_metric=primary_metric,
        )

    unique_all = pd.concat(all_unique_tables, ignore_index=True) if all_unique_tables else pd.DataFrame()
    near_all = pd.concat(all_near_tables, ignore_index=True) if all_near_tables else pd.DataFrame()
    summary_df = summarize_near_optimal_plateau(unique_all, near_all, primary_metric=primary_metric)
    strategy_counts = _build_plateau_strategy_counts(near_all, primary_metric)
    memory_grid = _build_plateau_memory_grid(near_all, primary_metric)

    paths = {
        "unique_combinations": output_dir / "unique_combinations.csv",
        "near_optimal_candidates": output_dir / "near_optimal_candidates.csv",
        "summary": output_dir / "near_optimal_summary.csv",
        "strategy_counts": output_dir / "near_optimal_strategy_counts.csv",
        "memory_grid": output_dir / "near_optimal_memory_grid.csv",
        "strategy_counts_plot": output_dir / "near_optimal_strategy_counts.png",
        "spread_plot": output_dir / "near_optimal_spread.png",
        "shape_tradeoff_plot": output_dir / "near_optimal_shape_tradeoff.png",
        "report": output_dir / "near_optimal_report.md",
        "manifest": output_dir / "manifest.json",
    }
    unique_all.to_csv(paths["unique_combinations"], index=False)
    near_all.to_csv(paths["near_optimal_candidates"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    strategy_counts.to_csv(paths["strategy_counts"], index=False)
    memory_grid.to_csv(paths["memory_grid"], index=False)
    _plot_plateau_strategy_counts(strategy_counts, paths["strategy_counts_plot"])
    _plot_plateau_spread(summary_df, paths["spread_plot"])
    _plot_plateau_shape_tradeoff(near_all, paths["shape_tradeoff_plot"], primary_metric)
    _write_plateau_report(
        summary_df,
        strategy_counts,
        paths["report"],
        primary_metric=primary_metric,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    manifest = {
        "hyper_dir": str(hyper_dir),
        "output_dir": str(output_dir),
        "stage": str(stage),
        "subjects": [int(_subject_id_from_dir(path)) for path in subject_dirs],
        "candidate_key": str(candidate_key),
        "primary_metric": str(primary_metric),
        "abs_tol": float(abs_tol),
        "rel_tol": float(rel_tol),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return paths


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _available_minimize_objectives(
    df: pd.DataFrame,
    objectives: Sequence[str],
) -> list[str]:
    out: list[str] = []
    for column in objectives:
        if column not in df.columns:
            continue
        values = _numeric_series(df, column)
        if np.isfinite(values).any():
            out.append(str(column))
    return out


def _minimize_percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(1.0, index=values.index, dtype=float)
    ranks = numeric.rank(method="average", na_option="bottom", ascending=True)
    denom = max(1.0, float(len(numeric) - 1))
    scaled = (ranks - 1.0) / denom
    return scaled.clip(lower=0.0, upper=1.0).fillna(1.0)


def _pearson_or_nan(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce")
    y = pd.to_numeric(right, errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _pareto_front_mask(df: pd.DataFrame, objectives: Sequence[str]) -> pd.Series:
    if df.empty or not objectives:
        return pd.Series(False, index=df.index, dtype=bool)
    values = df.loc[:, list(objectives)].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    valid = values.notna().all(axis=1)
    mask = pd.Series(False, index=df.index, dtype=bool)
    if not bool(valid.any()):
        return mask

    arr = values.loc[valid].to_numpy(dtype=float)
    valid_index = values.loc[valid].index.to_list()
    keep = np.ones(arr.shape[0], dtype=bool)
    for idx in range(arr.shape[0]):
        if not keep[idx]:
            continue
        dominated_by = np.all(arr <= arr[idx] + 1e-12, axis=1) & np.any(arr < arr[idx] - 1e-12, axis=1)
        dominated_by[idx] = False
        if bool(np.any(dominated_by)):
            keep[idx] = False
    mask.loc[[valid_index[i] for i, is_keep in enumerate(keep) if is_keep]] = True
    return mask


def _weight_profiles(n_objectives: int, levels: Sequence[float]) -> list[tuple[float, ...]]:
    if n_objectives <= 0:
        return []
    profiles: set[tuple[float, ...]] = set()
    level_values = [float(x) for x in levels]

    def build(prefix: tuple[float, ...]) -> None:
        if len(prefix) == n_objectives:
            total = float(sum(prefix))
            if total <= 0.0:
                return
            profiles.add(tuple(round(x / total, 10) for x in prefix))
            return
        for value in level_values:
            build(prefix + (value,))

    build(())
    return sorted(profiles)


def _mode_summary(values: pd.Series, *, denominator: int | None = None) -> tuple[str, int, float]:
    clean = values.dropna().astype(str)
    if clean.empty:
        return "", 0, float("nan")
    counts = clean.value_counts()
    mode = str(counts.index[0])
    count = int(counts.iloc[0])
    denom = int(denominator) if denominator is not None else int(len(clean))
    return mode, count, count / max(1, denom)


def _count_memory_pairs(df: pd.DataFrame) -> int:
    if df.empty or not {"gamma", "w0"}.issubset(df.columns):
        return 0
    return int(df[["gamma", "w0"]].dropna().drop_duplicates().shape[0])


def _weight_sensitivity_for_subject(
    gate_df: pd.DataFrame,
    *,
    objectives: Sequence[str],
    weight_levels: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    profiles = _weight_profiles(len(objectives), weight_levels)
    if gate_df.empty or not objectives or not profiles:
        return pd.DataFrame(), {
            "n_weight_profiles": 0,
            "n_unique_winners": 0,
            "top_winner_combination_index": np.nan,
            "top_winner_count": 0,
            "top_winner_fraction": np.nan,
            "n_winner_memory_pairs": 0,
            "top_winner_memory_pair": "",
            "top_winner_memory_pair_count": 0,
            "top_winner_memory_pair_fraction": np.nan,
            "n_winner_strategies": 0,
            "top_winner_strategy_id": "",
            "top_winner_strategy_count": 0,
            "top_winner_strategy_fraction": np.nan,
        }

    ranked = gate_df.copy()
    rank_cols = [f"{column}_rank01" for column in objectives]
    for column, rank_col in zip(objectives, rank_cols):
        if rank_col not in ranked.columns:
            ranked[rank_col] = _minimize_percentile_rank(ranked[column])

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        score = np.zeros(len(ranked), dtype=float)
        for weight, rank_col in zip(profile, rank_cols):
            score = score + float(weight) * pd.to_numeric(ranked[rank_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        winner_pos = int(np.nanargmin(score))
        winner = ranked.iloc[winner_pos]
        row = {
            "subject_id": winner.get("subject_id"),
            "weight_profile": "|".join(f"{column}:{weight:.3f}" for column, weight in zip(objectives, profile)),
            "weighted_rank_score": float(score[winner_pos]),
            "combination_index": int(winner.get("combination_index", -1)),
            "hyperparam_signature": winner.get("hyperparam_signature"),
            "gamma": winner.get("gamma"),
            "w0": winner.get("w0"),
            "strategy_id": winner.get("strategy_id"),
            "strategy_description": winner.get("strategy_description"),
        }
        for column in objectives:
            row[column] = winner.get(column)
        rows.append(row)

    detail = pd.DataFrame(rows)
    winner_mode, winner_count, winner_fraction = _mode_summary(
        detail["combination_index"],
        denominator=len(detail),
    )
    memory_key = detail["gamma"].astype(str) + "/" + detail["w0"].astype(str)
    memory_mode, memory_count, memory_fraction = _mode_summary(memory_key, denominator=len(detail))
    strategy_mode, strategy_count, strategy_fraction = _mode_summary(
        detail.get("strategy_id", pd.Series(dtype=object)),
        denominator=len(detail),
    )
    summary = {
        "n_weight_profiles": int(len(detail)),
        "n_unique_winners": int(detail["combination_index"].nunique(dropna=True)),
        "top_winner_combination_index": int(float(winner_mode)) if str(winner_mode) else np.nan,
        "top_winner_count": winner_count,
        "top_winner_fraction": winner_fraction,
        "n_winner_memory_pairs": int(memory_key.nunique(dropna=True)),
        "top_winner_memory_pair": memory_mode,
        "top_winner_memory_pair_count": memory_count,
        "top_winner_memory_pair_fraction": memory_fraction,
        "n_winner_strategies": int(detail["strategy_id"].nunique(dropna=True)) if "strategy_id" in detail else 0,
        "top_winner_strategy_id": strategy_mode,
        "top_winner_strategy_count": strategy_count,
        "top_winner_strategy_fraction": strategy_fraction,
    }
    return detail, summary


def _write_multiobjective_report(
    summary_df: pd.DataFrame,
    weight_summary_df: pd.DataFrame,
    path: Path,
    *,
    primary_metric: str,
    objectives: Sequence[str],
    primary_abs_tol: float,
    primary_rel_tol: float,
    behavior_top_quantile: float,
    acc_mae_max: float,
    vol_ratio_min: float,
    vol_ratio_max: float | None,
    strict_vol_ratio_min: float,
    history_corr_min: float | None,
    switch_abs_max: float | None,
    include_legacy_multiobjective: bool = False,
) -> None:
    vol_range = (
        f"{vol_ratio_min:g} <= vol_ratio <= {vol_ratio_max:g}"
        if vol_ratio_max is not None
        else f"vol_ratio >= {vol_ratio_min:g}"
    )
    lines = [
        "# Hyper-CD Selection Diagnostic",
        "",
        f"- Primary metric: `{primary_metric}`",
        f"- Primary gate: metric <= max(best + {primary_abs_tol:g}, best * (1 + {primary_rel_tol:g}))",
        f"- Distribution absolute set: distribution_acc_mae_mean <= {acc_mae_max:g}, {vol_range}",
        "- PPC interval set: all selected distribution_ppc_interval statistics inside their central interval "
        "(or distribution_ppc_interval_score <= 1 when only scores are available)",
    ]
    if include_legacy_multiobjective:
        lines.extend(
            [
                f"- Pareto objectives: {', '.join(f'`{x}`' for x in objectives)}",
                f"- Top-quantile behavior set: non-primary objectives <= subject q{behavior_top_quantile:g}",
                f"- Legacy single-run absolute set: acc_mae <= {acc_mae_max:g}, vol_ratio >= {vol_ratio_min:g}",
                f"- Strict volatility count: vol_ratio >= {strict_vol_ratio_min:g}",
            ]
        )
    if history_corr_min is not None:
        if include_legacy_multiobjective:
            lines.append(
                f"- History criterion: single-run history_kernel_corr >= {history_corr_min:g}; "
                f"distribution_history_kernel_corr >= {history_corr_min:g}"
            )
        else:
            lines.append(f"- History criterion: distribution_history_kernel_corr >= {history_corr_min:g}")
    if switch_abs_max is not None:
        if include_legacy_multiobjective:
            lines.append(
                f"- Switch criterion: single-run switch abs diff <= {switch_abs_max:g}; "
                f"distribution_switch_behavior_score <= {switch_abs_max:g}"
            )
        else:
            lines.append(f"- Switch criterion: distribution_switch_behavior_score <= {switch_abs_max:g}")
    lines.extend(["", "## Subject Summary", ""])

    if summary_df.empty:
        lines.append("No candidate rows were available.")
    else:
        for _, row in summary_df.iterrows():
            subject_id = int(row["subject_id"])
            subject_line = (
                f"- Subject {subject_id}: primary gate {int(row['n_primary_gate'])}/"
                f"{int(row['n_unique_evaluated'])} "
                f"({row['primary_gate_fraction']:.2f}); "
                f"distribution absolute={int(row.get('n_distribution_accept', 0))}; "
                f"PPC interval={int(row.get('n_ppc_interval_accept', 0))}"
            )
            if include_legacy_multiobjective:
                subject_line += (
                    f"; Pareto={int(row['n_pareto'])}; top-q={int(row['n_behavior_topq'])}; "
                    f"legacy absolute={int(row['n_absolute_accept'])}; strict-vol={int(row['n_strict_vol_accept'])}"
                )
            lines.append(subject_line + ".")
            lines.append(
                f"  Gate diversity: memory pairs={int(row['memory_pairs_primary_gate'])}, "
                f"strategies={int(row['strategy_unique_primary_gate'])}."
            )
            if include_legacy_multiobjective:
                lines.append(
                    f"  Legacy top-q diversity: memory pairs={int(row['memory_pairs_behavior_topq'])}, "
                    f"strategies={int(row['strategy_unique_behavior_topq'])}."
                )
            lines.append(
                f"  Distribution pass counts: acc={int(row.get('n_distribution_acc_mae_le_threshold', 0))}, "
                f"vol={int(row.get('n_distribution_vol_ratio_in_range', 0))}, "
                f"history={int(row.get('n_distribution_history_corr_ge_threshold', 0))}, "
                f"switch={int(row.get('n_distribution_switch_score_le_threshold', 0))}, "
                f"acc+vol={int(row.get('n_distribution_acc_and_vol', 0))}, "
                f"acc+vol+history={int(row.get('n_distribution_acc_vol_history', 0))}."
            )
            if "n_ppc_interval_accept" in row:
                lines.append(
                    f"  PPC interval: accept={int(row.get('n_ppc_interval_accept', 0))}; "
                    f"score<=1={int(row.get('n_ppc_interval_score_le_1', 0))}; "
                    f"min score={float(row.get('min_ppc_interval_score', np.nan)):.3f}; "
                    f"median score={float(row.get('median_ppc_interval_score', np.nan)):.3f}."
                )
            if int(row.get("n_distribution_accept", 0)) == 0:
                lines.append("  Distribution-level absolute criteria selected no candidates under the current thresholds.")

    if include_legacy_multiobjective and not weight_summary_df.empty:
        lines.extend(["", "## Weight Sensitivity", ""])
        for _, row in weight_summary_df.iterrows():
            lines.append(
                f"- Subject {int(row['subject_id'])}: {int(row['n_unique_winners'])} different winners "
                f"across {int(row['n_weight_profiles'])} weight profiles; top winner "
                f"c{int(row['top_winner_combination_index']) if np.isfinite(row['top_winner_combination_index']) else 'NA'} "
                f"wins {row['top_winner_fraction']:.2f} of profiles; "
                f"winner memory pairs={int(row['n_winner_memory_pairs'])}, "
                f"winner strategies={int(row['n_winner_strategies'])}."
            )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- If the primary gate remains broad, the original choice objective does not identify a point estimate.",
            "- For v9-style runs, use the PPC interval set as the primary behavior-distribution acceptance diagnostic.",
            "- For v8 and later, interpret distribution-level absolute counts as the main PPC-style acceptance diagnostic.",
            "- If distribution absolute criteria select no candidates, the current model class is likely missing a mechanism for that behavior dimension.",
        ]
    )
    if include_legacy_multiobjective:
        lines.insert(
            -1,
            "- If the Pareto/top-q sets shrink but weight sensitivity has many winners, the candidate set is constrained but not point-identifiable.",
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_multiobjective_selection(
    hyper_dir: Path,
    *,
    output_dir: Path | None = None,
    subjects: Sequence[int] | None = None,
    stage: str = "coarse",
    candidates_json: Path | None = None,
    candidate_key: str = "cond1",
    primary_metric: str = "hyper_selection_error",
    primary_abs_tol: float = 0.015,
    primary_rel_tol: float = 0.05,
    objectives: Sequence[str] = (
        "hyper_selection_error",
        "accuracy_shape_score",
        "history_kernel_score",
        "switch_behavior_score",
    ),
    behavior_top_quantile: float = 0.20,
    acc_mae_max: float = 0.10,
    vol_ratio_min: float = 0.60,
    vol_ratio_max: float | None = 1.50,
    strict_vol_ratio_min: float = 0.80,
    history_corr_min: float | None = 0.80,
    switch_abs_max: float | None = 0.10,
    slope_agree_min: float | None = None,
    weight_levels: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
    include_legacy_multiobjective: bool = False,
) -> dict[str, Path]:
    """Evaluate candidate selection diagnostics inside the near-optimal choice plateau."""
    hyper_dir = Path(hyper_dir).resolve()
    if output_dir is None:
        output_dir = hyper_dir / "hyper_evaluation" / "selection_diagnostic"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_lookup = load_strategy_lookup(candidates_json, candidate_key=candidate_key)
    subject_dirs = discover_subject_dirs(hyper_dir, subjects)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject_* directories found under {hyper_dir}")

    candidate_tables: list[pd.DataFrame] = []
    pareto_tables: list[pd.DataFrame] = []
    topq_tables: list[pd.DataFrame] = []
    absolute_tables: list[pd.DataFrame] = []
    strict_tables: list[pd.DataFrame] = []
    distribution_absolute_tables: list[pd.DataFrame] = []
    ppc_interval_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    weight_detail_tables: list[pd.DataFrame] = []
    weight_summary_rows: list[dict[str, Any]] = []
    objective_union: list[str] = []

    for subject_dir in subject_dirs:
        subject_id = _subject_id_from_dir(subject_dir)
        combo_df = load_all_combinations_table(
            subject_dir,
            stage=stage,
            strategy_lookup=strategy_lookup,
        )
        if combo_df.empty:
            continue
        unique_df = _deduplicate_combinations(combo_df)
        selected_signature, _ = _load_subject_best_signature(subject_dir, strategy_lookup)
        unique_df["is_subject_selected"] = (
            unique_df["hyperparam_signature"].astype(str) == selected_signature
            if selected_signature
            else False
        )
        if primary_metric not in unique_df.columns:
            continue
        primary_values = _numeric_series(unique_df, primary_metric)
        if not np.isfinite(primary_values).any():
            continue

        best_primary = float(np.nanmin(primary_values))
        gate_cutoff = max(best_primary + float(primary_abs_tol), best_primary * (1.0 + float(primary_rel_tol)))
        gate = unique_df.loc[primary_values <= gate_cutoff].copy()
        gate["multiobjective_primary_metric"] = str(primary_metric)
        gate["multiobjective_best_primary"] = best_primary
        gate["multiobjective_gate_cutoff"] = gate_cutoff
        gate["multiobjective_primary_delta"] = _numeric_series(gate, primary_metric) - best_primary

        available_objectives = _available_minimize_objectives(gate, objectives)
        if primary_metric not in available_objectives and primary_metric in gate.columns:
            available_objectives = [primary_metric] + available_objectives
        available_objectives = list(dict.fromkeys(available_objectives))
        for objective in available_objectives:
            if objective not in objective_union:
                objective_union.append(objective)
            if include_legacy_multiobjective:
                gate[f"{objective}_rank01"] = _minimize_percentile_rank(gate[objective])

        secondary_objectives = [column for column in available_objectives if column != primary_metric]
        topq_mask = pd.Series(True, index=gate.index, dtype=bool)
        if include_legacy_multiobjective:
            for column in secondary_objectives:
                values = _numeric_series(gate, column)
                cutoff = float(np.nanquantile(values[np.isfinite(values)], behavior_top_quantile))
                gate[f"{column}_topq_cutoff"] = cutoff
                gate[f"{column}_topq_pass"] = values <= cutoff
                topq_mask &= gate[f"{column}_topq_pass"].astype(bool)
        if not include_legacy_multiobjective or not secondary_objectives:
            topq_mask = pd.Series(False, index=gate.index, dtype=bool)

        pareto_mask = (
            _pareto_front_mask(gate, available_objectives)
            if include_legacy_multiobjective
            else pd.Series(False, index=gate.index, dtype=bool)
        )
        gate["is_pareto_front"] = pareto_mask
        gate["is_behavior_topq"] = topq_mask

        acc_mae = _numeric_series(gate, "accuracy_shape_acc_mae")
        vol_ratio = _numeric_series(gate, "accuracy_shape_vol_ratio")
        switch_abs = _numeric_series(gate, "switch_behavior_switch_abs_diff")
        if include_legacy_multiobjective:
            absolute_mask = (acc_mae <= float(acc_mae_max)) & (vol_ratio >= float(vol_ratio_min))
            strict_mask = (acc_mae <= float(acc_mae_max)) & (vol_ratio >= float(strict_vol_ratio_min))
            if history_corr_min is not None and "history_kernel_corr" in gate.columns:
                history_corr = _numeric_series(gate, "history_kernel_corr")
                if np.isfinite(history_corr).any():
                    absolute_mask &= history_corr >= float(history_corr_min)
                    strict_mask &= history_corr >= float(history_corr_min)
            if switch_abs_max is not None and np.isfinite(switch_abs).any():
                absolute_mask &= switch_abs <= float(switch_abs_max)
                strict_mask &= switch_abs <= float(switch_abs_max)
            if slope_agree_min is not None and "accuracy_shape_slope_agree" in gate.columns:
                slope = _numeric_series(gate, "accuracy_shape_slope_agree")
                absolute_mask &= slope >= float(slope_agree_min)
                strict_mask &= slope >= float(slope_agree_min)
        else:
            absolute_mask = pd.Series(False, index=gate.index, dtype=bool)
            strict_mask = pd.Series(False, index=gate.index, dtype=bool)
        gate["is_absolute_accept"] = absolute_mask.fillna(False)
        gate["is_strict_vol_accept"] = strict_mask.fillna(False)

        dist_acc_mae = _numeric_series(gate, "distribution_acc_mae_mean")
        dist_vol_ratio = _numeric_series(gate, "distribution_vol_ratio_median")
        if not np.isfinite(dist_vol_ratio).any():
            dist_vol_ratio = _numeric_series(gate, "distribution_vol_ratio_mean")
        dist_history_corr = _numeric_series(gate, "distribution_history_kernel_corr")
        dist_switch_score = _numeric_series(gate, "distribution_switch_behavior_score")
        dist_acc_pass = dist_acc_mae <= float(acc_mae_max)
        dist_vol_pass = dist_vol_ratio >= float(vol_ratio_min)
        if vol_ratio_max is not None:
            dist_vol_pass &= dist_vol_ratio <= float(vol_ratio_max)
        dist_history_pass = pd.Series(True, index=gate.index, dtype=bool)
        if history_corr_min is not None and np.isfinite(dist_history_corr).any():
            dist_history_pass = dist_history_corr >= float(history_corr_min)
        dist_switch_pass = pd.Series(True, index=gate.index, dtype=bool)
        if switch_abs_max is not None and np.isfinite(dist_switch_score).any():
            dist_switch_pass = dist_switch_score <= float(switch_abs_max)
        distribution_accept_mask = dist_acc_pass & dist_vol_pass & dist_history_pass & dist_switch_pass
        distribution_strict_vol_mask = (
            dist_acc_pass
            & (dist_vol_ratio >= float(strict_vol_ratio_min))
            & dist_history_pass
            & dist_switch_pass
        )
        if vol_ratio_max is not None:
            distribution_strict_vol_mask &= dist_vol_ratio <= float(vol_ratio_max)
        gate["distribution_acc_mae_pass"] = dist_acc_pass.fillna(False)
        gate["distribution_vol_ratio_pass"] = dist_vol_pass.fillna(False)
        gate["distribution_history_corr_pass"] = dist_history_pass.fillna(False)
        gate["distribution_switch_score_pass"] = dist_switch_pass.fillna(False)
        gate["is_distribution_accept"] = distribution_accept_mask.fillna(False)
        gate["is_distribution_strict_vol_accept"] = distribution_strict_vol_mask.fillna(False)

        ppc_interval_score = _numeric_series(gate, "distribution_ppc_interval_score")
        ppc_interval_accept_raw = _numeric_series(gate, "distribution_ppc_interval_accept")
        if np.isfinite(ppc_interval_accept_raw).any():
            ppc_interval_accept_mask = ppc_interval_accept_raw >= 0.5
        else:
            ppc_interval_accept_mask = ppc_interval_score <= 1.0
        gate["is_ppc_interval_accept"] = ppc_interval_accept_mask.fillna(False)

        rank_cols = [f"{column}_rank01" for column in available_objectives if f"{column}_rank01" in gate.columns]
        if include_legacy_multiobjective and rank_cols:
            gate["equal_weight_rank_score"] = gate[rank_cols].mean(axis=1)
            gate = gate.sort_values(
                ["equal_weight_rank_score", primary_metric, "combination_index"],
                na_position="last",
            ).reset_index(drop=True)
        else:
            selection_sort_score = ppc_interval_score.copy()
            fallback_score = _numeric_series(gate, "distribution_intersection_score")
            selection_sort_score = selection_sort_score.where(np.isfinite(selection_sort_score), fallback_score)
            fallback_score = _numeric_series(gate, "distribution_score")
            selection_sort_score = selection_sort_score.where(np.isfinite(selection_sort_score), fallback_score)
            selection_sort_score = selection_sort_score.where(
                np.isfinite(selection_sort_score),
                _numeric_series(gate, primary_metric) - best_primary,
            )
            gate["selection_sort_score"] = selection_sort_score
            gate = gate.sort_values(
                ["is_ppc_interval_accept", "selection_sort_score", primary_metric, "combination_index"],
                ascending=[False, True, True, True],
                na_position="last",
            ).reset_index(drop=True)
        gate["selection_rank"] = np.arange(1, len(gate) + 1)
        gate["multiobjective_rank"] = gate["selection_rank"]

        pareto_df = gate[gate["is_pareto_front"].astype(bool)].copy()
        topq_df = gate[gate["is_behavior_topq"].astype(bool)].copy()
        absolute_df = gate[gate["is_absolute_accept"].astype(bool)].copy()
        strict_df = gate[gate["is_strict_vol_accept"].astype(bool)].copy()
        distribution_absolute_df = gate[gate["is_distribution_accept"].astype(bool)].copy()
        ppc_interval_df = gate[gate["is_ppc_interval_accept"].astype(bool)].copy()

        if include_legacy_multiobjective:
            weight_detail, weight_summary = _weight_sensitivity_for_subject(
                gate,
                objectives=available_objectives,
                weight_levels=weight_levels,
            )
            if not weight_detail.empty:
                weight_detail_tables.append(weight_detail)
            weight_summary["subject_id"] = subject_id
            weight_summary_rows.append(weight_summary)

        def count_condition(mask: pd.Series) -> int:
            return int(mask.fillna(False).sum())

        summary_rows.append(
            {
                "subject_id": subject_id,
                "available_objectives": "|".join(available_objectives),
                "n_unique_evaluated": int(len(unique_df)),
                "best_primary_error": best_primary,
                "primary_gate_cutoff": gate_cutoff,
                "n_primary_gate": int(len(gate)),
                "primary_gate_fraction": int(len(gate)) / max(1, int(len(unique_df))),
                "n_pareto": int(len(pareto_df)),
                "n_behavior_topq": int(len(topq_df)),
                "n_absolute_accept": int(len(absolute_df)),
                "n_strict_vol_accept": int(len(strict_df)),
                "n_distribution_accept": int(len(distribution_absolute_df)),
                "n_ppc_interval_accept": int(len(ppc_interval_df)),
                "n_distribution_strict_vol_accept": count_condition(gate["is_distribution_strict_vol_accept"].astype(bool)),
                "memory_pairs_primary_gate": _count_memory_pairs(gate),
                "memory_pairs_pareto": _count_memory_pairs(pareto_df),
                "memory_pairs_behavior_topq": _count_memory_pairs(topq_df),
                "memory_pairs_absolute_accept": _count_memory_pairs(absolute_df),
                "memory_pairs_distribution_accept": _count_memory_pairs(distribution_absolute_df),
                "memory_pairs_ppc_interval_accept": _count_memory_pairs(ppc_interval_df),
                "strategy_unique_primary_gate": int(gate["strategy_id"].nunique(dropna=True)) if "strategy_id" in gate else 0,
                "strategy_unique_pareto": int(pareto_df["strategy_id"].nunique(dropna=True)) if "strategy_id" in pareto_df else 0,
                "strategy_unique_behavior_topq": int(topq_df["strategy_id"].nunique(dropna=True)) if "strategy_id" in topq_df else 0,
                "strategy_unique_absolute_accept": int(absolute_df["strategy_id"].nunique(dropna=True)) if "strategy_id" in absolute_df else 0,
                "strategy_unique_distribution_accept": int(distribution_absolute_df["strategy_id"].nunique(dropna=True)) if "strategy_id" in distribution_absolute_df else 0,
                "strategy_unique_ppc_interval_accept": int(ppc_interval_df["strategy_id"].nunique(dropna=True)) if "strategy_id" in ppc_interval_df else 0,
                "n_acc_mae_le_threshold": count_condition(acc_mae <= float(acc_mae_max)),
                "n_vol_ratio_ge_threshold": count_condition(vol_ratio >= float(vol_ratio_min)),
                "n_vol_ratio_ge_strict": count_condition(vol_ratio >= float(strict_vol_ratio_min)),
                "n_acc_mae_and_vol": count_condition((acc_mae <= float(acc_mae_max)) & (vol_ratio >= float(vol_ratio_min))),
                "n_acc_mae_and_strict_vol": count_condition((acc_mae <= float(acc_mae_max)) & (vol_ratio >= float(strict_vol_ratio_min))),
                "n_switch_abs_le_threshold": (
                    count_condition(switch_abs <= float(switch_abs_max))
                    if switch_abs_max is not None
                    else 0
                ),
                "n_distribution_acc_mae_le_threshold": count_condition(dist_acc_pass),
                "n_distribution_vol_ratio_in_range": count_condition(dist_vol_pass),
                "n_distribution_vol_ratio_ge_strict": count_condition(
                    (dist_vol_ratio >= float(strict_vol_ratio_min))
                    & ((dist_vol_ratio <= float(vol_ratio_max)) if vol_ratio_max is not None else True)
                ),
                "n_distribution_history_corr_ge_threshold": (
                    count_condition(dist_history_pass)
                    if history_corr_min is not None and np.isfinite(dist_history_corr).any()
                    else 0
                ),
                "n_distribution_switch_score_le_threshold": (
                    count_condition(dist_switch_pass)
                    if switch_abs_max is not None and np.isfinite(dist_switch_score).any()
                    else 0
                ),
                "n_distribution_acc_and_vol": count_condition(dist_acc_pass & dist_vol_pass),
                "n_distribution_acc_vol_history": count_condition(dist_acc_pass & dist_vol_pass & dist_history_pass),
                "n_distribution_acc_vol_history_switch": count_condition(
                    dist_acc_pass & dist_vol_pass & dist_history_pass & dist_switch_pass
                ),
                "n_ppc_interval_score_le_1": count_condition(ppc_interval_score <= 1.0),
                "min_ppc_interval_score": float(np.nanmin(ppc_interval_score)) if np.isfinite(ppc_interval_score).any() else np.nan,
                "median_ppc_interval_score": float(np.nanmedian(ppc_interval_score)) if np.isfinite(ppc_interval_score).any() else np.nan,
                "min_acc_mae": float(np.nanmin(acc_mae)) if np.isfinite(acc_mae).any() else np.nan,
                "median_acc_mae": float(np.nanmedian(acc_mae)) if np.isfinite(acc_mae).any() else np.nan,
                "max_vol_ratio": float(np.nanmax(vol_ratio)) if np.isfinite(vol_ratio).any() else np.nan,
                "median_vol_ratio": float(np.nanmedian(vol_ratio)) if np.isfinite(vol_ratio).any() else np.nan,
                "min_switch_abs_diff": float(np.nanmin(switch_abs)) if np.isfinite(switch_abs).any() else np.nan,
                "median_switch_abs_diff": float(np.nanmedian(switch_abs)) if np.isfinite(switch_abs).any() else np.nan,
                "min_distribution_acc_mae_mean": float(np.nanmin(dist_acc_mae)) if np.isfinite(dist_acc_mae).any() else np.nan,
                "median_distribution_acc_mae_mean": float(np.nanmedian(dist_acc_mae)) if np.isfinite(dist_acc_mae).any() else np.nan,
                "max_distribution_vol_ratio": float(np.nanmax(dist_vol_ratio)) if np.isfinite(dist_vol_ratio).any() else np.nan,
                "median_distribution_vol_ratio": float(np.nanmedian(dist_vol_ratio)) if np.isfinite(dist_vol_ratio).any() else np.nan,
                "max_distribution_history_corr": float(np.nanmax(dist_history_corr)) if np.isfinite(dist_history_corr).any() else np.nan,
                "median_distribution_history_corr": float(np.nanmedian(dist_history_corr)) if np.isfinite(dist_history_corr).any() else np.nan,
                "min_distribution_switch_score": float(np.nanmin(dist_switch_score)) if np.isfinite(dist_switch_score).any() else np.nan,
                "median_distribution_switch_score": float(np.nanmedian(dist_switch_score)) if np.isfinite(dist_switch_score).any() else np.nan,
                "corr_primary_shape": (
                    _pearson_or_nan(gate[primary_metric], gate["accuracy_shape_score"])
                    if "accuracy_shape_score" in gate.columns
                    else np.nan
                ),
                "corr_primary_history": (
                    _pearson_or_nan(gate[primary_metric], gate["history_kernel_score"])
                    if "history_kernel_score" in gate.columns
                    else np.nan
                ),
                "corr_shape_history": (
                    _pearson_or_nan(gate["accuracy_shape_score"], gate["history_kernel_score"])
                    if {"accuracy_shape_score", "history_kernel_score"}.issubset(gate.columns)
                    else np.nan
                ),
                "corr_primary_switch": (
                    _pearson_or_nan(gate[primary_metric], gate["switch_behavior_score"])
                    if "switch_behavior_score" in gate.columns
                    else np.nan
                ),
                "corr_shape_switch": (
                    _pearson_or_nan(gate["accuracy_shape_score"], gate["switch_behavior_score"])
                    if {"accuracy_shape_score", "switch_behavior_score"}.issubset(gate.columns)
                    else np.nan
                ),
                "corr_history_switch": (
                    _pearson_or_nan(gate["history_kernel_score"], gate["switch_behavior_score"])
                    if {"history_kernel_score", "switch_behavior_score"}.issubset(gate.columns)
                    else np.nan
                ),
            }
        )

        candidate_tables.append(gate)
        pareto_tables.append(pareto_df)
        topq_tables.append(topq_df)
        absolute_tables.append(absolute_df)
        strict_tables.append(strict_df)
        distribution_absolute_tables.append(distribution_absolute_df)
        ppc_interval_tables.append(ppc_interval_df)

    candidates_df = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    pareto_all = pd.concat(pareto_tables, ignore_index=True) if pareto_tables else pd.DataFrame()
    topq_all = pd.concat(topq_tables, ignore_index=True) if topq_tables else pd.DataFrame()
    absolute_all = pd.concat(absolute_tables, ignore_index=True) if absolute_tables else pd.DataFrame()
    strict_all = pd.concat(strict_tables, ignore_index=True) if strict_tables else pd.DataFrame()
    distribution_absolute_all = pd.concat(distribution_absolute_tables, ignore_index=True) if distribution_absolute_tables else pd.DataFrame()
    ppc_interval_all = pd.concat(ppc_interval_tables, ignore_index=True) if ppc_interval_tables else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values("subject_id").reset_index(drop=True) if summary_rows else pd.DataFrame()
    weight_detail_df = pd.concat(weight_detail_tables, ignore_index=True) if weight_detail_tables else pd.DataFrame()
    weight_summary_df = pd.DataFrame(weight_summary_rows).sort_values("subject_id").reset_index(drop=True) if weight_summary_rows else pd.DataFrame()

    paths = {
        "selection_candidates": output_dir / "selection_candidates.csv",
        "distribution_accepted_candidates": output_dir / "distribution_accepted_candidates.csv",
        "ppc_interval_accepted_candidates": output_dir / "ppc_interval_accepted_candidates.csv",
        "summary": output_dir / "selection_summary.csv",
        "report": output_dir / "selection_report.md",
        "manifest": output_dir / "manifest.json",
    }
    if include_legacy_multiobjective:
        paths.update(
            {
                "multiobjective_candidates": output_dir / "multiobjective_candidates.csv",
                "pareto_front": output_dir / "pareto_front.csv",
                "behavior_topq_candidates": output_dir / "behavior_topq_candidates.csv",
                "accepted_candidates": output_dir / "accepted_candidates.csv",
                "strict_volatility_candidates": output_dir / "strict_volatility_candidates.csv",
                "weight_sensitivity": output_dir / "weight_sensitivity.csv",
                "weight_sensitivity_summary": output_dir / "weight_sensitivity_summary.csv",
                "multiobjective_summary": output_dir / "multiobjective_summary.csv",
                "multiobjective_report": output_dir / "multiobjective_report.md",
            }
        )
    candidates_df.to_csv(paths["selection_candidates"], index=False)
    distribution_absolute_all.to_csv(paths["distribution_accepted_candidates"], index=False)
    ppc_interval_all.to_csv(paths["ppc_interval_accepted_candidates"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    if include_legacy_multiobjective:
        candidates_df.to_csv(paths["multiobjective_candidates"], index=False)
        pareto_all.to_csv(paths["pareto_front"], index=False)
        topq_all.to_csv(paths["behavior_topq_candidates"], index=False)
        absolute_all.to_csv(paths["accepted_candidates"], index=False)
        strict_all.to_csv(paths["strict_volatility_candidates"], index=False)
        summary_df.to_csv(paths["multiobjective_summary"], index=False)
        weight_detail_df.to_csv(paths["weight_sensitivity"], index=False)
        weight_summary_df.to_csv(paths["weight_sensitivity_summary"], index=False)
    _write_multiobjective_report(
        summary_df,
        weight_summary_df,
        paths["report"],
        primary_metric=primary_metric,
        objectives=objective_union or list(objectives),
        primary_abs_tol=primary_abs_tol,
        primary_rel_tol=primary_rel_tol,
        behavior_top_quantile=behavior_top_quantile,
        acc_mae_max=acc_mae_max,
        vol_ratio_min=vol_ratio_min,
        vol_ratio_max=vol_ratio_max,
        strict_vol_ratio_min=strict_vol_ratio_min,
        history_corr_min=history_corr_min,
        switch_abs_max=switch_abs_max,
        include_legacy_multiobjective=include_legacy_multiobjective,
    )
    if include_legacy_multiobjective:
        _write_multiobjective_report(
            summary_df,
            weight_summary_df,
            paths["multiobjective_report"],
            primary_metric=primary_metric,
            objectives=objective_union or list(objectives),
            primary_abs_tol=primary_abs_tol,
            primary_rel_tol=primary_rel_tol,
            behavior_top_quantile=behavior_top_quantile,
            acc_mae_max=acc_mae_max,
            vol_ratio_min=vol_ratio_min,
            vol_ratio_max=vol_ratio_max,
            strict_vol_ratio_min=strict_vol_ratio_min,
            history_corr_min=history_corr_min,
            switch_abs_max=switch_abs_max,
            include_legacy_multiobjective=True,
        )
    manifest = {
        "hyper_dir": str(hyper_dir),
        "output_dir": str(output_dir),
        "stage": str(stage),
        "subjects": [int(_subject_id_from_dir(path)) for path in subject_dirs],
        "candidate_key": str(candidate_key),
        "primary_metric": str(primary_metric),
        "primary_abs_tol": float(primary_abs_tol),
        "primary_rel_tol": float(primary_rel_tol),
        "objectives": list(objectives),
        "available_objectives": objective_union,
        "behavior_top_quantile": float(behavior_top_quantile),
        "acc_mae_max": float(acc_mae_max),
        "vol_ratio_min": float(vol_ratio_min),
        "vol_ratio_max": None if vol_ratio_max is None else float(vol_ratio_max),
        "strict_vol_ratio_min": float(strict_vol_ratio_min),
        "history_corr_min": None if history_corr_min is None else float(history_corr_min),
        "switch_abs_max": None if switch_abs_max is None else float(switch_abs_max),
        "slope_agree_min": None if slope_agree_min is None else float(slope_agree_min),
        "weight_levels": [float(x) for x in weight_levels],
        "include_legacy_multiobjective": bool(include_legacy_multiobjective),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return paths


def select_accuracy_diagnostic_candidates(
    combo_df: pd.DataFrame,
    *,
    max_candidates: int = 12,
    top_choice_k: int = 4,
    top_best_k: int = 2,
    top_stochastic_k: int = 3,
    max_strategy_candidates: int = 8,
) -> pd.DataFrame:
    """Select a compact, diverse candidate set for accuracy-shape diagnostics."""
    if combo_df.empty:
        return pd.DataFrame()

    selected: dict[str, dict[str, Any]] = {}

    def add_rows(frame: pd.DataFrame, reason: str) -> None:
        for _, row in frame.iterrows():
            signature = str(row.get("hyperparam_signature"))
            payload = row.to_dict()
            if signature in selected:
                existing = str(selected[signature].get("selection_reason", ""))
                reasons = [x for x in existing.split("+") if x]
                if reason not in reasons:
                    reasons.append(reason)
                selected[signature]["selection_reason"] = "+".join(reasons)
                continue
            payload["selection_reason"] = reason
            selected[signature] = payload

    by_selection = combo_df.sort_values(
        ["hyper_selection_error", "hyper_best_error", "combination_index"],
        na_position="last",
    )
    add_rows(by_selection.head(max(0, int(top_choice_k))), "top_choice_brier")

    by_best = combo_df.sort_values(
        ["hyper_best_error", "hyper_selection_error", "combination_index"],
        na_position="last",
    )
    add_rows(by_best.head(max(0, int(top_best_k))), "top_single_run")

    finite_selection = pd.to_numeric(combo_df["hyper_selection_error"], errors="coerce")
    cutoff = float(np.nanquantile(finite_selection, 0.5)) if np.isfinite(finite_selection).any() else np.inf
    stochastic_pool = combo_df[finite_selection <= cutoff].copy()
    stochastic_pool = stochastic_pool.sort_values(
        ["hyper_std_error", "hyper_selection_error"],
        ascending=[False, True],
        na_position="last",
    )
    add_rows(stochastic_pool.head(max(0, int(top_stochastic_k))), "high_stochasticity")

    if "strategy_id" in combo_df.columns and max_strategy_candidates > 0:
        strategy_rows = []
        for _, group in combo_df.groupby("strategy_id", dropna=False):
            best = group.sort_values(
                ["hyper_selection_error", "hyper_best_error", "combination_index"],
                na_position="last",
            ).head(1)
            if not best.empty:
                strategy_rows.append(best)
        if strategy_rows:
            strategy_df = pd.concat(strategy_rows, ignore_index=True).sort_values(
                ["hyper_selection_error", "hyper_best_error", "combination_index"],
                na_position="last",
            )
            add_rows(strategy_df.head(max_strategy_candidates), "best_per_strategy")

    if len(selected) < int(max_candidates):
        add_rows(by_selection.head(int(max_candidates) * 2), "fill_top_choice_brier")

    out = pd.DataFrame(selected.values())
    if out.empty:
        return out
    out = out.sort_values(
        ["hyper_selection_error", "hyper_best_error", "combination_index"],
        na_position="last",
    ).head(int(max_candidates))
    out = out.reset_index(drop=True)
    out["diagnostic_candidate_id"] = [
        f"s{int(row.subject_id)}_c{int(row.combination_index)}" for row in out.itertuples()
    ]
    return out


def _set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
    curr = root
    for part in path.split(".")[:-1]:
        next_value = curr.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested path through non-mapping segment: {path}")
        curr = next_value
    curr[path.split(".")[-1]] = deepcopy(value)


def _apply_hyperparams_to_configs(
    point: Mapping[str, Any],
    sim_cfg: Mapping[str, Any],
    engine_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    next_sim = deepcopy(dict(sim_cfg))
    next_engine = deepcopy(dict(engine_cfg))
    for key, value in point.items():
        if key.startswith("engine."):
            _set_by_path(next_engine, key[len("engine."):], value)
        elif key.startswith("simulation."):
            _set_by_path(next_sim, key[len("simulation."):], value)
        else:
            raise ValueError(f"Hyperparameter key must start with engine. or simulation.: {key}")
    next_sim["fixed_hyperparams"] = deepcopy(dict(point))
    return next_sim, next_engine


def _curve_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    true = np.asarray(metrics.get("sliding_true_acc"), dtype=float)
    pred = np.asarray(metrics.get("sliding_pred_acc"), dtype=float)
    if true.shape != pred.shape or true.size == 0:
        return {
            "acc_mae": np.nan,
            "acc_rmse": np.nan,
            "acc_corr": np.nan,
            "true_vol": np.nan,
            "pred_vol": np.nan,
            "vol_ratio": np.nan,
            "true_range": np.nan,
            "pred_range": np.nan,
            "range_ratio": np.nan,
            "slope_agree": np.nan,
        }
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return {
            "acc_mae": np.nan,
            "acc_rmse": np.nan,
            "acc_corr": np.nan,
            "true_vol": np.nan,
            "pred_vol": np.nan,
            "vol_ratio": np.nan,
            "true_range": np.nan,
            "pred_range": np.nan,
            "range_ratio": np.nan,
            "slope_agree": np.nan,
        }
    true = true[mask]
    pred = pred[mask]
    diff = pred - true
    true_vol = float(np.mean(np.abs(np.diff(true)))) if true.size > 1 else np.nan
    pred_vol = float(np.mean(np.abs(np.diff(pred)))) if pred.size > 1 else np.nan
    true_range = float(np.nanmax(true) - np.nanmin(true))
    pred_range = float(np.nanmax(pred) - np.nanmin(pred))
    if np.nanstd(true) > 1e-12 and np.nanstd(pred) > 1e-12:
        acc_corr = float(np.corrcoef(true, pred)[0, 1])
    else:
        acc_corr = np.nan
    d_true = np.diff(true)
    d_pred = np.diff(pred)
    slope_mask = (np.abs(d_true) > 1e-12) & (np.abs(d_pred) > 1e-12)
    slope_agree = (
        float(np.mean(np.sign(d_true[slope_mask]) == np.sign(d_pred[slope_mask])))
        if slope_mask.any()
        else np.nan
    )
    return {
        "acc_mae": float(np.mean(np.abs(diff))),
        "acc_rmse": float(np.sqrt(np.mean(diff * diff))),
        "acc_corr": acc_corr,
        "true_vol": true_vol,
        "pred_vol": pred_vol,
        "vol_ratio": float(pred_vol / true_vol) if true_vol and np.isfinite(true_vol) else np.nan,
        "true_range": true_range,
        "pred_range": pred_range,
        "range_ratio": float(pred_range / true_range) if true_range > 0 else np.nan,
        "slope_agree": slope_agree,
    }


def _run_candidate_accuracy_diagnostic(
    *,
    subject_id: int,
    condition: int,
    arrays: Any,
    point: Mapping[str, Any],
    engine_config_template: Mapping[str, Any],
    processed_data_dir: Path,
    dataset_paths: Mapping[str, Path],
    window_size: int,
    prediction_mode: str,
    selection_prediction_mode: str,
    loss_metric: str,
    loss_delta: float | None,
    hyper_candidate_seed: int,
    simulation_repeats: int,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    simulation_point_seed = derive_simulation_point_seed(
        int(hyper_candidate_seed),
        int(subject_id),
        dict(point),
    )

    def one_run(repeat_index: int):
        trajectory_seed = derive_trajectory_seed(
            int(simulation_point_seed),
            "simulation",
            int(repeat_index),
        )
        return evaluate_state_model_run(
            int(subject_id),
            int(condition),
            arrays,
            dict(point),
            deepcopy(dict(engine_config_template)),
            Path(processed_data_dir),
            int(window_size),
            dataset_paths,
            keep_logs=False,
            include_step_log=False,
            prediction_mode=str(prediction_mode),
            selection_prediction_mode=str(selection_prediction_mode),
            loss_metric=str(loss_metric),
            loss_delta=loss_delta,
            simulation_point_seed=int(simulation_point_seed),
            trajectory_seed=trajectory_seed,
            seed_context={
                "hyper_candidate_seed": int(hyper_candidate_seed),
                "simulation_point_seed": int(simulation_point_seed),
                "trajectory_seed": trajectory_seed,
                "phase": "accuracy_diagnostic",
                "repeat_index": int(repeat_index),
            },
        )

    runs = Parallel(n_jobs=max(1, int(n_jobs)))(
        delayed(one_run)(repeat_index) for repeat_index in range(int(simulation_repeats))
    )

    rows: list[dict[str, Any]] = []
    best_acc_payload: dict[str, Any] = {}
    best_choice_payload: dict[str, Any] = {}
    best_acc = np.inf
    best_choice = np.inf
    for repeat_index, run in enumerate(runs):
        metrics = run.metrics_by_mode[str(selection_prediction_mode)]
        curve = _curve_metrics(metrics)
        choice_error = float(run.mean_error)
        row = {
            "repeat_index": int(repeat_index),
            "choice_error": choice_error,
            **curve,
        }
        rows.append(row)
        if np.isfinite(curve["acc_mae"]) and curve["acc_mae"] < best_acc:
            best_acc = curve["acc_mae"]
            best_acc_payload = {
                "repeat_index": int(repeat_index),
                "choice_error": choice_error,
                **curve,
                "sliding_true_acc": np.asarray(metrics["sliding_true_acc"], dtype=float).tolist(),
                "sliding_pred_acc": np.asarray(metrics["sliding_pred_acc"], dtype=float).tolist(),
            }
        if np.isfinite(choice_error) and choice_error < best_choice:
            best_choice = choice_error
            best_choice_payload = {
                "repeat_index": int(repeat_index),
                "choice_error": choice_error,
                **curve,
                "sliding_true_acc": np.asarray(metrics["sliding_true_acc"], dtype=float).tolist(),
                "sliding_pred_acc": np.asarray(metrics["sliding_pred_acc"], dtype=float).tolist(),
            }

    curves = {
        "simulation_point_seed": int(simulation_point_seed),
        "best_accuracy": best_acc_payload,
        "best_choice": best_choice_payload,
    }
    return rows, curves


def _candidate_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame()
    group_cols = ["subject_id", "diagnostic_candidate_id"]
    for (subject_id, candidate_id), group in run_df.groupby(group_cols, dropna=False):
        choice = pd.to_numeric(group["choice_error"], errors="coerce")
        acc = pd.to_numeric(group["acc_mae"], errors="coerce")
        vol = pd.to_numeric(group["vol_ratio"], errors="coerce")
        slope = pd.to_numeric(group["slope_agree"], errors="coerce")
        choice_idx = choice.idxmin()
        acc_idx = acc.idxmin()
        rows.append(
            {
                "subject_id": subject_id,
                "diagnostic_candidate_id": candidate_id,
                "simulation_repeats": int(len(group)),
                "choice_mean_error": float(np.nanmean(choice)),
                "choice_best_error": float(np.nanmin(choice)),
                "choice_q10_error": float(np.nanquantile(choice, 0.1)),
                "choice_best_acc_mae": float(group.loc[choice_idx, "acc_mae"]),
                "choice_best_vol_ratio": float(group.loc[choice_idx, "vol_ratio"]),
                "choice_best_repeat_index": int(group.loc[choice_idx, "repeat_index"]),
                "acc_best_mae": float(np.nanmin(acc)),
                "acc_best_choice_error": float(group.loc[acc_idx, "choice_error"]),
                "acc_best_vol_ratio": float(group.loc[acc_idx, "vol_ratio"]),
                "acc_best_slope_agree": float(group.loc[acc_idx, "slope_agree"]),
                "acc_best_repeat_index": int(group.loc[acc_idx, "repeat_index"]),
                "acc_mae_q10": float(np.nanquantile(acc, 0.1)),
                "acc_mae_median": float(np.nanmedian(acc)),
                "vol_ratio_median": float(np.nanmedian(vol)),
                "vol_ratio_max": float(np.nanmax(vol)),
                "slope_agree_median": float(np.nanmedian(slope)),
                "count_acc_mae_le_0p08": int(np.sum(acc <= 0.08)),
                "count_acc_mae_le_0p10": int(np.sum(acc <= 0.10)),
                "count_vol_ge_0p8": int(np.sum(vol >= 0.8)),
                "count_acc_good_vol_good": int(np.sum((acc <= 0.10) & (vol >= 0.8))),
            }
        )
    return pd.DataFrame(rows)


def _plot_accuracy_diagnostic_scatter(run_df: pd.DataFrame, path: Path) -> None:
    if run_df.empty:
        return
    subjects = sorted(run_df["subject_id"].dropna().unique())
    n = len(subjects)
    n_cols = min(3, max(1, n))
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 4.2 * n_rows), squeeze=False)
    for ax, subject_id in zip(axes.ravel(), subjects):
        sub = run_df[run_df["subject_id"] == subject_id]
        scatter = ax.scatter(
            sub["choice_error"],
            sub["acc_mae"],
            c=sub["vol_ratio"],
            s=12,
            alpha=0.45,
            cmap="viridis",
            vmin=0,
            vmax=max(1.2, float(np.nanquantile(run_df["vol_ratio"], 0.99))),
        )
        ax.axhline(0.10, color="#b33", linewidth=1, linestyle="--", alpha=0.8)
        ax.set_title(f"Subject {int(subject_id)}")
        ax.set_xlabel("Choice Brier")
        ax.set_ylabel("Accuracy curve MAE")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Pred/true volatility ratio", shrink=0.85)
    fig.suptitle("Run-level diagnostic: choice fit vs accuracy-curve fit", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate_summary(summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df.empty:
        return
    subjects = sorted(summary_df["subject_id"].dropna().unique())
    n = len(subjects)
    n_cols = min(3, max(1, n))
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.3 * n_rows), squeeze=False)
    for ax, subject_id in zip(axes.ravel(), subjects):
        sub = summary_df[summary_df["subject_id"] == subject_id]
        ax.scatter(
            sub["hyper_selection_error"],
            sub["acc_best_mae"],
            c=sub["acc_best_vol_ratio"],
            s=72,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.35,
        )
        for _, row in sub.iterrows():
            label = str(row["diagnostic_candidate_id"]).split("_c")[-1]
            ax.text(row["hyper_selection_error"], row["acc_best_mae"], label, fontsize=7)
        ax.axhline(0.10, color="#b33", linewidth=1, linestyle="--", alpha=0.8)
        ax.set_title(f"Subject {int(subject_id)}")
        ax.set_xlabel("Hyper-CD selection error")
        ax.set_ylabel("Best accuracy MAE after resampling")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Candidate-level diagnostic: hyper objective vs best sampled accuracy shape", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_top_accuracy_curves(
    curves: Sequence[Mapping[str, Any]],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    if summary_df.empty:
        return paths
    curve_by_id = {str(item.get("diagnostic_candidate_id")): item for item in curves}
    for subject_id, group in summary_df.groupby("subject_id"):
        top = group.sort_values(["acc_best_mae", "acc_best_choice_error"]).head(4)
        if top.empty:
            continue
        fig, axes = plt.subplots(len(top), 1, figsize=(8, 2.8 * len(top)), squeeze=False)
        for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
            curve_item = curve_by_id.get(str(row["diagnostic_candidate_id"]), {})
            best = curve_item.get("best_accuracy") if isinstance(curve_item, Mapping) else None
            if not isinstance(best, Mapping) or "sliding_true_acc" not in best:
                ax.text(0.5, 0.5, "No curve", ha="center", va="center", transform=ax.transAxes)
                continue
            true = np.asarray(best["sliding_true_acc"], dtype=float)
            pred = np.asarray(best["sliding_pred_acc"], dtype=float)
            x = np.arange(1, len(true) + 1)
            ax.plot(x, true, label="True", linewidth=2)
            ax.plot(x, pred, label="Predicted", linewidth=2)
            ax.set_ylim(0, 1)
            ax.set_title(
                f"{row['diagnostic_candidate_id']} | acc MAE={row['acc_best_mae']:.3f} | "
                f"choice={row['acc_best_choice_error']:.3f} | vol={row['acc_best_vol_ratio']:.2f}",
                fontsize=9,
            )
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
        fig.suptitle(f"Subject {int(subject_id)}: best sampled accuracy curves", y=1.01)
        fig.tight_layout()
        path = output_dir / f"subject_{int(subject_id)}_top_accuracy_curves.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def diagnose_hyper_accuracy_sampling(
    hyper_dir: Path,
    *,
    base_sim_config_path: Path = DEFAULT_BASE_SIM_CONFIG,
    output_dir: Path | None = None,
    subjects: Sequence[int] | None = None,
    stage: str = "coarse",
    candidates_json: Path | None = None,
    candidate_key: str = "cond1",
    simulation_repeats: int = 256,
    max_candidates_per_subject: int = 12,
    n_jobs: int | None = None,
) -> dict[str, Path]:
    """Resample selected hyper-CD candidates and test accuracy-curve expressiveness."""
    hyper_dir = Path(hyper_dir).resolve()
    base_sim_config_path = Path(base_sim_config_path)
    if not base_sim_config_path.is_absolute():
        base_sim_config_path = (Path.cwd() / base_sim_config_path).resolve()
    if output_dir is None:
        output_dir = hyper_dir / "accuracy_diagnostic"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_lookup = load_strategy_lookup(candidates_json, candidate_key=candidate_key)
    subject_dirs = discover_subject_dirs(hyper_dir, subjects)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject_* directories found under {hyper_dir}")

    base_cfg = load_yaml(base_sim_config_path)
    all_selected: list[pd.DataFrame] = []
    all_run_rows: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []

    for subject_dir in subject_dirs:
        subject_id = _subject_id_from_dir(subject_dir)
        if subject_id is None:
            continue
        combo_df = load_all_combinations_table(
            subject_dir,
            stage=stage,
            strategy_lookup=strategy_lookup,
        )
        selected_df = select_accuracy_diagnostic_candidates(
            combo_df,
            max_candidates=max_candidates_per_subject,
        )
        if selected_df.empty:
            continue
        all_selected.append(selected_df)

        subject_cfg = resolve_subject_config(base_cfg, int(subject_id))
        base_engine_cfg = resolve_engine_config(
            subject_cfg,
            base_sim_config_path.parent,
            subject_id=int(subject_id),
        )
        dataset_paths = resolve_dataset_paths(subject_cfg, base_sim_config_path.parent, DEFAULT_DATA_PATH)
        runner = StateModelSimulationRunner(
            engine_config=base_engine_cfg,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            n_jobs=int(n_jobs or subject_cfg.get("n_jobs", 1)),
        )
        runner.prepare_data(dataset_paths["learning_data"])
        subject_frame = runner._get_subject_frame(int(subject_id), float(subject_cfg.get("stop_at", 1.0)))
        condition = runner._get_condition_value(subject_frame)
        max_trials_raw = subject_cfg.get("max_trials")
        max_trials = int(max_trials_raw) if max_trials_raw is not None else None
        arrays = runner._extract_arrays(subject_frame, max_trials)
        prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
        loss_metric = resolve_loss_metric(subject_cfg)
        loss_delta = resolve_loss_delta(subject_cfg, loss_metric)
        window_size = resolve_window_size(subject_cfg, int(subject_id), [int(subject_id)])

        for _, cand in selected_df.iterrows():
            point = cand["hyperparams"]
            point_sim_cfg, point_engine_cfg = _apply_hyperparams_to_configs(
                point,
                subject_cfg,
                base_engine_cfg,
            )
            effective_prediction_mode = str(point_sim_cfg.get("prediction_mode", prediction_mode))
            effective_selection_mode = str(
                point_sim_cfg.get("selection_prediction_mode", selection_prediction_mode)
            )
            effective_loss_metric = str(point_sim_cfg.get("loss_metric", loss_metric))
            effective_loss_delta = resolve_loss_delta(point_sim_cfg, effective_loss_metric)
            effective_window_size = int(point_sim_cfg.get("window_size", window_size))
            hyper_candidate_seed = int(cand["hyper_candidate_seed"])
            repeat_rows, curves = _run_candidate_accuracy_diagnostic(
                subject_id=int(subject_id),
                condition=int(condition),
                arrays=arrays,
                point=point,
                engine_config_template=point_engine_cfg,
                processed_data_dir=dataset_paths["processed_dir"],
                dataset_paths=dataset_paths,
                window_size=effective_window_size,
                prediction_mode=effective_prediction_mode,
                selection_prediction_mode=effective_selection_mode,
                loss_metric=effective_loss_metric,
                loss_delta=effective_loss_delta,
                hyper_candidate_seed=hyper_candidate_seed,
                simulation_repeats=int(simulation_repeats),
                n_jobs=int(n_jobs or subject_cfg.get("n_jobs", 1)),
            )

            candidate_id = str(cand["diagnostic_candidate_id"])
            for row in repeat_rows:
                all_run_rows.append(
                    {
                        "subject_id": int(subject_id),
                        "diagnostic_candidate_id": candidate_id,
                        "combination_index": int(cand["combination_index"]),
                        "strategy_id": cand.get("strategy_id"),
                        "selection_reason": cand.get("selection_reason"),
                        "hyper_selection_error": cand.get("hyper_selection_error"),
                        "hyper_best_error": cand.get("hyper_best_error"),
                        "hyper_std_error": cand.get("hyper_std_error"),
                        **row,
                    }
                )
            curve_records.append(
                {
                    "subject_id": int(subject_id),
                    "diagnostic_candidate_id": candidate_id,
                    "combination_index": int(cand["combination_index"]),
                    "strategy_id": cand.get("strategy_id"),
                    "selection_reason": cand.get("selection_reason"),
                    "hyper_selection_error": cand.get("hyper_selection_error"),
                    "hyper_best_error": cand.get("hyper_best_error"),
                    "hyper_candidate_seed": hyper_candidate_seed,
                    **curves,
                }
            )

    selected_all = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    run_df = pd.DataFrame(all_run_rows)
    summary_df = _candidate_summary(run_df)
    if not summary_df.empty and not selected_all.empty:
        merge_cols = [
            "subject_id",
            "diagnostic_candidate_id",
            "combination_index",
            "selection_reason",
            "hyper_selection_error",
            "hyper_best_error",
            "hyper_best10_mean_error",
            "hyper_std_error",
            "gamma",
            "w0",
            "strategy_id",
            "init_num",
            "beta_init",
            "decrease_rate",
            "prior_beta_scale",
            "correct_additive",
        ]
        summary_df = summary_df.merge(
            selected_all[[c for c in merge_cols if c in selected_all.columns]],
            on=["subject_id", "diagnostic_candidate_id"],
            how="left",
        )
        summary_df = summary_df.sort_values(["subject_id", "acc_best_mae", "choice_best_error"])

    subject_summary_rows = []
    if not summary_df.empty:
        for subject_id, group in summary_df.groupby("subject_id"):
            best_acc = group.sort_values(["acc_best_mae", "acc_best_choice_error"]).iloc[0]
            best_choice = group.sort_values(["choice_best_error", "acc_best_mae"]).iloc[0]
            subject_summary_rows.append(
                {
                    "subject_id": int(subject_id),
                    "n_candidates": int(len(group)),
                    "simulation_repeats_per_candidate": int(simulation_repeats),
                    "best_acc_candidate": best_acc["diagnostic_candidate_id"],
                    "best_acc_mae": best_acc["acc_best_mae"],
                    "best_acc_choice_error": best_acc["acc_best_choice_error"],
                    "best_acc_vol_ratio": best_acc["acc_best_vol_ratio"],
                    "best_choice_candidate": best_choice["diagnostic_candidate_id"],
                    "best_choice_error": best_choice["choice_best_error"],
                    "best_choice_acc_mae": best_choice["choice_best_acc_mae"],
                    "best_choice_vol_ratio": best_choice["choice_best_vol_ratio"],
                    "any_acc_mae_le_0p08": bool((group["acc_best_mae"] <= 0.08).any()),
                    "any_acc_mae_le_0p10": bool((group["acc_best_mae"] <= 0.10).any()),
                    "any_acc_good_vol_good": bool((group["count_acc_good_vol_good"] > 0).any()),
                    "max_vol_ratio": float(np.nanmax(group["vol_ratio_max"])),
                }
            )
    subject_summary = pd.DataFrame(subject_summary_rows)

    paths = {
        "selected_candidates": output_dir / "selected_candidates.csv",
        "run_metrics": output_dir / "run_metrics.csv",
        "candidate_summary": output_dir / "candidate_summary.csv",
        "subject_summary": output_dir / "subject_summary.csv",
        "curve_records": output_dir / "best_run_curves.json",
        "manifest": output_dir / "manifest.json",
        "run_scatter_plot": output_dir / "run_choice_vs_accuracy_scatter.png",
        "candidate_summary_plot": output_dir / "candidate_summary.png",
    }
    selected_all.to_csv(paths["selected_candidates"], index=False)
    run_df.to_csv(paths["run_metrics"], index=False)
    summary_df.to_csv(paths["candidate_summary"], index=False)
    subject_summary.to_csv(paths["subject_summary"], index=False)
    with paths["curve_records"].open("w", encoding="utf-8") as f:
        json.dump(curve_records, f, ensure_ascii=False, indent=2)
    _plot_accuracy_diagnostic_scatter(run_df, paths["run_scatter_plot"])
    _plot_candidate_summary(summary_df, paths["candidate_summary_plot"])
    curve_paths = _plot_top_accuracy_curves(curve_records, summary_df, output_dir)
    manifest = {
        "hyper_dir": str(hyper_dir),
        "base_sim_config_path": str(base_sim_config_path),
        "output_dir": str(output_dir),
        "stage": str(stage),
        "subjects": [int(_subject_id_from_dir(path)) for path in subject_dirs],
        "simulation_repeats": int(simulation_repeats),
        "max_candidates_per_subject": int(max_candidates_per_subject),
        "n_jobs": n_jobs,
        "outputs": {key: str(value) for key, value in paths.items()},
        "curve_plots": [str(path) for path in curve_paths],
    }
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return paths


def _volatility_resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _volatility_load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a mapping: {path}")
    return payload


def _volatility_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _volatility_resolve_subjects(
    input_dir: Path,
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
) -> list[int]:
    if subjects:
        return [int(x) for x in subjects]
    if subject_range:
        start, end = [int(x) for x in subject_range]
        return list(range(start, end + 1))
    out: list[int] = []
    for path in sorted(input_dir.glob("subject_*")):
        if path.is_dir():
            try:
                out.append(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    if not out:
        raise ValueError(f"No subject_* directories found under {input_dir}")
    return out


def _volatility_infer_base_sim_config_path(input_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return _volatility_resolve_project_path(override)
    root_best = input_dir / "best_hyperparams.json"
    if root_best.is_file():
        payload = _volatility_load_json(root_best)
        hyper = payload.get("hyper") if isinstance(payload.get("hyper"), Mapping) else {}
        raw = hyper.get("base_sim_config_path") if isinstance(hyper, Mapping) else None
        if raw:
            return _volatility_resolve_project_path(Path(str(raw)))
    return _volatility_resolve_project_path(DEFAULT_BASE_SIM_CONFIG)


def _volatility_load_subject_best(
    input_dir: Path,
    subject_id: int,
) -> tuple[dict[str, Any], dict[str, Any], int | None]:
    path = input_dir / f"subject_{int(subject_id)}" / "best_hyperparams.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing subject best hyperparameters: {path}")
    payload = _volatility_load_json(path)
    params = subject_best_hyperparams(payload)
    if not isinstance(params, Mapping):
        raise ValueError(f"{path} is missing selected.best_hyperparams")
    seed = subject_hyper_candidate_seed(payload)
    return payload, deepcopy(dict(params)), None if seed is None else int(seed)


def _engine_local_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        key_str = str(key)
        if key_str.startswith("engine."):
            out[key_str[len("engine.") :]] = deepcopy(value)
        elif key_str.startswith("simulation."):
            continue
        else:
            out[key_str] = deepcopy(value)
    return out


def _sliding_curve(values: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < window_size + 1:
        return np.full(0, np.nan, dtype=float)
    out = []
    for start in range(1, values.size - window_size + 1):
        window = values[start : start + window_size]
        out.append(float(np.mean(window)) if np.all(np.isfinite(window)) else float("nan"))
    return np.asarray(out, dtype=float)


def _curve_volatility(curve: np.ndarray) -> float:
    curve = np.asarray(curve, dtype=float).reshape(-1)
    curve = curve[np.isfinite(curve)]
    if curve.size <= 1:
        return float("nan")
    return float(np.mean(np.abs(np.diff(curve))))


def _sample_binary_curves(
    pred_acc: np.ndarray,
    *,
    window_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(pred_acc, dtype=float).reshape(-1)
    probs = np.clip(probs, 0.0, 1.0)
    finite = np.isfinite(probs)
    if not finite.any():
        return np.full((0, 0), np.nan), np.full(0, np.nan)
    probs = np.where(finite, probs, np.nanmean(probs[finite]))

    curves: list[np.ndarray] = []
    vols: list[float] = []
    for _ in range(int(n_samples)):
        sampled = rng.binomial(1, probs).astype(float)
        curve = _sliding_curve(sampled, window_size)
        curves.append(curve)
        vols.append(_curve_volatility(curve))
    return np.vstack(curves) if curves else np.full((0, 0), np.nan), np.asarray(vols, dtype=float)


def _summarize_quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_q05": float("nan"),
            f"{prefix}_q10": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q90": float("nan"),
            f"{prefix}_q95": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_median": float(np.quantile(values, 0.50)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
    }


def _run_one_volatility_model_trajectory(
    *,
    subject_id: int,
    condition: int,
    arrays: Any,
    params: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    processed_data_dir: Path,
    dataset_paths: Mapping[str, Path],
    window_size: int,
    prediction_mode: str,
    selection_prediction_mode: str,
    loss_metric: str,
    loss_delta: float | None,
    simulation_point_seed: int,
    repeat_index: int,
) -> dict[str, Any]:
    trajectory_seed = derive_trajectory_seed(int(simulation_point_seed), "volatility_calibration", int(repeat_index))
    run = evaluate_state_model_run(
        int(subject_id),
        int(condition),
        arrays,
        dict(params),
        deepcopy(dict(engine_config)),
        processed_data_dir,
        int(window_size),
        dataset_paths,
        False,
        False,
        str(prediction_mode),
        str(selection_prediction_mode),
        str(loss_metric),
        loss_delta,
        simulation_point_seed=int(simulation_point_seed),
        trajectory_seed=int(trajectory_seed),
        seed_context={
            "phase": "volatility_calibration",
            "repeat_index": int(repeat_index),
            "simulation_point_seed": int(simulation_point_seed),
            "trajectory_seed": int(trajectory_seed),
        },
    )
    metrics = run.metrics_by_mode[str(selection_prediction_mode)]
    pred_acc = np.asarray(metrics.get("pred_acc"), dtype=float)
    pred_curve = np.asarray(metrics.get("sliding_pred_acc"), dtype=float)
    true_curve = np.asarray(metrics.get("sliding_true_acc"), dtype=float)
    return {
        "repeat_index": int(repeat_index),
        "trajectory_seed": int(trajectory_seed),
        "mean_error": float(run.mean_error),
        "pred_acc": pred_acc,
        "pred_curve": pred_curve,
        "true_curve": true_curve,
        "prob_vol": _curve_volatility(pred_curve),
    }


def _evaluate_volatility_subject(
    *,
    input_dir: Path,
    label: str,
    base_cfg: Mapping[str, Any],
    base_cfg_path: Path,
    subject_id: int,
    subjects: Sequence[int],
    model_repeats: int,
    binary_samples_per_run: int,
    n_jobs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subject_payload, raw_params, hyper_candidate_seed = _volatility_load_subject_best(input_dir, subject_id)
    params = _engine_local_params(raw_params)
    subject_cfg = resolve_subject_config(base_cfg, subject_id)
    base_cfg_dir = base_cfg_path.parent
    dataset_paths = resolve_dataset_paths(subject_cfg, base_cfg_dir, DEFAULT_DATA_PATH)
    engine_config = resolve_engine_config(subject_cfg, base_cfg_dir, subject_id=subject_id)
    window_size = resolve_window_size(subject_cfg, subject_id, subjects)
    prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
    loss_metric = resolve_loss_metric(subject_cfg)
    loss_delta = resolve_loss_delta(subject_cfg, loss_metric)
    stop_at = float(subject_cfg.get("stop_at", 1.0))
    max_trials_raw = subject_cfg.get("max_trials")
    max_trials = None if max_trials_raw is None else int(max_trials_raw)

    optimizer = BaseStateOptimizer(
        engine_config,
        processed_data_dir=dataset_paths["processed_dir"],
        n_jobs=max(1, int(n_jobs)),
        dataset_paths=dataset_paths,
    )
    optimizer.prepare_data(dataset_paths["learning_data"])
    subject_frame = optimizer._get_subject_frame(subject_id, stop_at)
    condition = optimizer._get_condition_value(subject_frame)
    arrays = optimizer._extract_arrays(subject_frame, max_trials)

    if hyper_candidate_seed is None:
        hyper_candidate_seed = stable_seed(
            {
                "seed_role": "volatility_calibration_fallback_hyper_seed",
                "input_dir": input_dir.as_posix(),
                "subject_id": int(subject_id),
                "params": raw_params,
                "seed": int(seed),
            }
        )
    simulation_point_seed = derive_simulation_point_seed(int(hyper_candidate_seed), int(subject_id), raw_params)

    runs = Parallel(n_jobs=max(1, int(n_jobs)))(
        delayed(_run_one_volatility_model_trajectory)(
            subject_id=int(subject_id),
            condition=int(condition),
            arrays=arrays,
            params=params,
            engine_config=engine_config,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            window_size=int(window_size),
            prediction_mode=str(prediction_mode),
            selection_prediction_mode=str(selection_prediction_mode),
            loss_metric=str(loss_metric),
            loss_delta=loss_delta,
            simulation_point_seed=int(simulation_point_seed),
            repeat_index=repeat_index,
        )
        for repeat_index in range(int(model_repeats))
    )

    true_curve = np.asarray(runs[0]["true_curve"], dtype=float) if runs else np.full(0, np.nan)
    human_vol = _curve_volatility(true_curve)
    sampled_vols: list[float] = []
    sampled_curves: list[np.ndarray] = []
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        rng = np.random.default_rng(
            stable_seed(
                {
                    "seed_role": "volatility_calibration_binary_sample",
                    "seed": int(seed),
                    "label": label,
                    "subject_id": int(subject_id),
                    "repeat_index": int(run["repeat_index"]),
                    "trajectory_seed": int(run["trajectory_seed"]),
                }
            )
        )
        curves, vols = _sample_binary_curves(
            np.asarray(run["pred_acc"], dtype=float),
            window_size=int(window_size),
            n_samples=int(binary_samples_per_run),
            rng=rng,
        )
        sampled_vols.extend([float(x) for x in vols if np.isfinite(x)])
        if curves.size:
            sampled_curves.extend([curve for curve in curves])
        prob_vol = float(run["prob_vol"])
        run_rows.append(
            {
                "label": label,
                "subject_id": int(subject_id),
                "repeat_index": int(run["repeat_index"]),
                "trajectory_seed": int(run["trajectory_seed"]),
                "mean_error": float(run["mean_error"]),
                "human_vol": human_vol,
                "prob_vol": prob_vol,
                "prob_vol_ratio": float(prob_vol / human_vol) if human_vol > 0 else float("nan"),
                "binary_sample_count": int(len(vols)),
                "sampled_binary_vol_mean": float(np.nanmean(vols)) if np.isfinite(vols).any() else float("nan"),
                "sampled_binary_vol_median": float(np.nanmedian(vols)) if np.isfinite(vols).any() else float("nan"),
            }
        )

    sampled_vol_arr = np.asarray(sampled_vols, dtype=float)
    prob_vol_arr = np.asarray([row["prob_vol"] for row in run_rows], dtype=float)
    sampled_ratio_arr = sampled_vol_arr / human_vol if human_vol > 0 else np.full(sampled_vol_arr.shape, np.nan)
    prob_ratio_arr = prob_vol_arr / human_vol if human_vol > 0 else np.full(prob_vol_arr.shape, np.nan)

    curve_stack = (
        np.vstack(sampled_curves)
        if sampled_curves and all(len(curve) == len(true_curve) for curve in sampled_curves)
        else np.full((0, 0), np.nan)
    )
    if curve_stack.size and true_curve.size:
        lower = np.nanquantile(curve_stack, 0.05, axis=0)
        upper = np.nanquantile(curve_stack, 0.95, axis=0)
        finite = np.isfinite(true_curve) & np.isfinite(lower) & np.isfinite(upper)
        curve_coverage = (
            float(np.mean((true_curve[finite] >= lower[finite]) & (true_curve[finite] <= upper[finite])))
            if finite.any()
            else float("nan")
        )
    else:
        curve_coverage = float("nan")

    summary = {
        "label": label,
        "subject_id": int(subject_id),
        "condition": int(condition),
        "model_repeats": int(model_repeats),
        "binary_samples_per_run": int(binary_samples_per_run),
        "total_binary_samples": int(sampled_vol_arr.size),
        "window_size": int(window_size),
        "prediction_mode": str(prediction_mode),
        "selection_prediction_mode": str(selection_prediction_mode),
        "loss_metric": str(loss_metric),
        "human_vol": human_vol,
        "human_vol_percentile": float(np.mean(sampled_vol_arr <= human_vol)) if sampled_vol_arr.size else float("nan"),
        "sampled_binary_vol_covered_90": bool(
            sampled_vol_arr.size
            and np.nanquantile(sampled_vol_arr, 0.05) <= human_vol <= np.nanquantile(sampled_vol_arr, 0.95)
        ),
        "sampled_curve_point_coverage_90": curve_coverage,
        "hyper_candidate_seed": int(hyper_candidate_seed),
        "simulation_point_seed": int(simulation_point_seed),
    }
    summary.update(_summarize_quantiles(prob_vol_arr, "prob_vol"))
    summary.update(_summarize_quantiles(prob_ratio_arr, "prob_vol_ratio"))
    summary.update(_summarize_quantiles(sampled_vol_arr, "sampled_binary_vol"))
    summary.update(_summarize_quantiles(sampled_ratio_arr, "sampled_binary_vol_ratio"))

    selected = subject_payload.get("selected") if isinstance(subject_payload.get("selected"), Mapping) else {}
    compact = selected.get("best_params") if isinstance(selected.get("best_params"), Mapping) else {}
    for key in (
        "gamma",
        "w0",
        "strategy_id",
        "prior_reset_target",
        "prior_reset_source",
        "prior_reset_volatility_gain",
        "latent_volatility_error_gain",
        "latent_volatility_low_accuracy_gain",
        "latent_volatility_decay",
        "latent_volatility_max",
        "output_base_lapse",
        "output_latent_volatility_lapse",
    ):
        if key in compact:
            summary[key] = compact[key]

    return run_rows, summary


def _write_volatility_report(output_dir: Path, label: str, summary_df: pd.DataFrame) -> Path:
    lines = [
        "# Volatility Calibration Diagnostic",
        "",
        f"- Label: `{label}`",
        "- `prob_vol_ratio` compares the expected model accuracy curve to the human 0/1 curve.",
        "- `sampled_binary_vol_ratio` compares Bernoulli-sampled model behavior to the same human 0/1 curve.",
        "- `human_vol_percentile` is the posterior predictive CDF value of the observed human volatility.",
        "- Coverage is good when the human volatility lies inside the sampled binary 5%-95% interval.",
        "",
        "## Subject Summary",
        "",
    ]
    for _, row in summary_df.sort_values("subject_id").iterrows():
        covered = "yes" if bool(row.get("sampled_binary_vol_covered_90")) else "no"
        lines.append(
            "- Subject {sid}: human percentile={pct:.3f}; sampled ratio median={med:.3f} "
            "[q05={q05:.3f}, q95={q95:.3f}]; prob ratio median={pmed:.3f}; covered90={covered}; "
            "curve point coverage={curve:.3f}.".format(
                sid=int(row["subject_id"]),
                pct=float(row.get("human_vol_percentile", np.nan)),
                med=float(row.get("sampled_binary_vol_ratio_median", np.nan)),
                q05=float(row.get("sampled_binary_vol_ratio_q05", np.nan)),
                q95=float(row.get("sampled_binary_vol_ratio_q95", np.nan)),
                pmed=float(row.get("prob_vol_ratio_median", np.nan)),
                covered=covered,
                curve=float(row.get("sampled_curve_point_coverage_90", np.nan)),
            )
        )
    if not summary_df.empty:
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- Covered subjects: {int(summary_df['sampled_binary_vol_covered_90'].sum())}/{len(summary_df)}",
                f"- Mean human percentile: {float(summary_df['human_vol_percentile'].mean()):.3f}",
                f"- Median sampled-binary volatility ratio: {float(summary_df['sampled_binary_vol_ratio_median'].median()):.3f}",
                f"- Median probability-curve volatility ratio: {float(summary_df['prob_vol_ratio_median'].median()):.3f}",
            ]
        )
    path = output_dir / "volatility_calibration_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate_volatility_calibration(
    *,
    input_dir: Path,
    output_dir: Path | None,
    base_sim_config: Path | None,
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
    model_repeats: int,
    binary_samples_per_run: int,
    n_jobs: int,
    seed: int,
) -> dict[str, str]:
    """Run binary posterior-predictive volatility calibration for selected hyper-CD fits."""
    input_dir = _volatility_resolve_project_path(input_dir)
    label = input_dir.name
    base_cfg_path = _volatility_infer_base_sim_config_path(input_dir, base_sim_config)
    base_cfg = load_yaml(base_cfg_path)
    subject_ids = _volatility_resolve_subjects(input_dir, subjects, subject_range)
    out_dir = (
        _volatility_resolve_project_path(output_dir)
        if output_dir
        else input_dir / "hyper_evaluation" / "volatility_calibration"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_run_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        run_rows, summary = _evaluate_volatility_subject(
            input_dir=input_dir,
            label=label,
            base_cfg=base_cfg,
            base_cfg_path=base_cfg_path,
            subject_id=int(subject_id),
            subjects=subject_ids,
            model_repeats=int(model_repeats),
            binary_samples_per_run=int(binary_samples_per_run),
            n_jobs=int(n_jobs),
            seed=int(seed),
        )
        all_run_rows.extend(run_rows)
        summaries.append(summary)

    run_df = pd.DataFrame(all_run_rows)
    summary_df = pd.DataFrame(summaries)
    run_path = out_dir / "volatility_calibration_runs.csv"
    summary_path = out_dir / "volatility_calibration_summary.csv"
    run_df.to_csv(run_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    report_path = _write_volatility_report(out_dir, label, summary_df)
    manifest_path = out_dir / "manifest.json"
    _volatility_write_json(
        manifest_path,
        {
            "input_dir": input_dir.as_posix(),
            "output_dir": out_dir.as_posix(),
            "base_sim_config": base_cfg_path.as_posix(),
            "subjects": [int(x) for x in subject_ids],
            "model_repeats": int(model_repeats),
            "binary_samples_per_run": int(binary_samples_per_run),
            "n_jobs": int(n_jobs),
            "seed": int(seed),
            "outputs": {
                "runs": run_path.as_posix(),
                "summary": summary_path.as_posix(),
                "report": report_path.as_posix(),
            },
        },
    )
    return {
        "runs": run_path.as_posix(),
        "summary": summary_path.as_posix(),
        "report": report_path.as_posix(),
        "manifest": manifest_path.as_posix(),
    }


def _parse_subjects(values: Sequence[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        if "-" in value:
            left, right = value.split("-", 1)
            out.extend(range(int(left), int(right) + 1))
        else:
            out.append(int(value))
    return sorted(set(out))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hyper_dir", type=Path, help="Hyper-CD output directory")
    parser.add_argument("--output-dir", type=Path, help="Directory for CSVs and plots")
    parser.add_argument("--subjects", nargs="+", help="Subject ids or ranges, e.g. 128 130-132")
    parser.add_argument("--stage", default="coarse", help="Stage to analyze")
    parser.add_argument(
        "--candidates-json",
        type=Path,
        default=Path(
            "src/Bayesian_state/problems/modules/hypo_transition/candidates/"
            "hypo_transition_strategy_candidates.json"
        ),
        help="Strategy candidate JSON used to recover candidate ids",
    )
    parser.add_argument("--candidate-key", default="cond1", help="Candidate key in JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = evaluate_hyper_cd_convergence(
        args.hyper_dir,
        output_dir=args.output_dir,
        subjects=_parse_subjects(args.subjects),
        stage=args.stage,
        candidates_json=args.candidates_json,
        candidate_key=args.candidate_key,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
