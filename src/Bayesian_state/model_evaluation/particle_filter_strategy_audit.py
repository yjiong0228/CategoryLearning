"""Counterfactual audit of particle-filter strategy contributions.

This module is invoked by the shared ``run_model_evaluation`` entry point.  It
reuses the standard simulation runner with common random seeds and compares the
fitted dynamic controller against two strategy-frozen counterfactuals:

``mean_matched_static``
    Constant exploration and local/global mixture matched to the fitted
    dynamic run's average pre-choice policy.

``controller_off``
    The fitted baseline ``m`` and ``g`` with both dynamic controllers disabled.

The audit is diagnostic.  It does not refit parameters or alter Hyper-CD
selection, and its conditional behavioral intervals are not autonomous
rollouts or fitted-parameter uncertainty intervals.
"""
from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.metrics import conditional_behavioral_accuracy_band_metrics
from src.Bayesian_state.run_simulation import (
    apply_fixed_hyperparams_to_engine_config,
    apply_fixed_hyperparams_to_subject_config,
    infer_fixed_hyperparams_from_engine_config,
)
from src.Bayesian_state.simulation.repeated_simulation import StateModelSimulationRunner
from src.Bayesian_state.simulation.simulation_config import (
    DEFAULT_DATA_PATH,
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_subjects,
    resolve_window_size,
)
from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths


LOGGER = logging.getLogger(__name__)

DYNAMIC = "dynamic"
MEAN_MATCHED_STATIC = "mean_matched_static"
CONTROLLER_OFF = "controller_off"
VARIANT_ORDER = (DYNAMIC, MEAN_MATCHED_STATIC, CONTROLLER_OFF)
VARIANT_LABELS = {
    DYNAMIC: "Fitted dynamic",
    MEAN_MATCHED_STATIC: "Mean-matched static",
    CONTROLLER_OFF: "Controller off",
}
VARIANT_COLORS = {
    DYNAMIC: "#0072B2",
    MEAN_MATCHED_STATIC: "#D55E00",
    CONTROLLER_OFF: "#7A7A7A",
}

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
}


def _transition_kwargs(engine_config: dict[str, Any]) -> dict[str, Any]:
    modules = engine_config.setdefault("modules", {})
    transition = modules.setdefault("hypo_transitions_mod", {})
    kwargs = transition.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError("hypo_transitions_mod.kwargs must be a mapping")
    return kwargs


def _controller_setting(
    kwargs: Mapping[str, Any],
    controller_name: str,
    setting_name: str,
    default: float,
) -> float:
    controller = kwargs.get(controller_name) or {}
    if isinstance(controller, Mapping) and setting_name in controller:
        return float(controller[setting_name])
    return float(kwargs.get(setting_name, default))


def _disable_strategy_controllers(
    engine_config: Mapping[str, Any],
    *,
    m: float,
    g: float,
) -> dict[str, Any]:
    resolved = deepcopy(dict(engine_config))
    transition = resolved.setdefault("modules", {}).setdefault(
        "hypo_transitions_mod",
        {},
    )
    transition["class"] = (
        "src.Bayesian_state.problems.modules.hypo_transition.static."
        "StaticWorkspaceHypothesisTransitionModule"
    )
    kwargs = _transition_kwargs(resolved)
    kwargs["m"] = float(m)
    kwargs["g"] = float(g)

    rate_controller = dict(kwargs.get("rate_controller") or {})
    rate_controller.update(
        {
            "m": float(m),
            "m_phi": 0.0,
            "m_beta_surprise": 0.0,
            "m_beta_uncertainty": 0.0,
        }
    )
    kwargs["rate_controller"] = rate_controller

    range_controller = dict(kwargs.get("range_controller") or {})
    range_controller.update(
        {
            "g": float(g),
            "g_phi": 0.0,
            "g_beta_surprise": 0.0,
            "g_beta_uncertainty": 0.0,
        }
    )
    kwargs["range_controller"] = range_controller

    # Top-level legacy aliases must also be neutralized because the runtime
    # falls back to them when a controller mapping omits a field.
    for field in (
        "m_phi",
        "m_beta_surprise",
        "m_beta_uncertainty",
        "g_phi",
        "g_beta_surprise",
        "g_beta_uncertainty",
    ):
        kwargs[field] = 0.0
    return resolved


def _mean_matched_controls(info: Mapping[str, Any]) -> tuple[float, float, float]:
    exploit = np.asarray(info["predictive_strategy_exploit"], dtype=float).reshape(-1)
    local = np.asarray(
        info["predictive_strategy_local_explore"], dtype=float
    ).reshape(-1)
    global_explore = np.asarray(
        info["predictive_strategy_global_explore"], dtype=float
    ).reshape(-1)
    n_trials = min(exploit.size, local.size, global_explore.size)
    if n_trials < 2:
        raise ValueError("Mean-matched strategy audit requires at least two trials")
    strategy = np.column_stack(
        [exploit[:n_trials], local[:n_trials], global_explore[:n_trials]]
    )
    if not np.all(np.isfinite(strategy)):
        raise ValueError("Mean-matched strategy controls contain non-finite values")

    # Trial 0 initializes the workspace without a replacement draw and should
    # not dilute the fitted later-trial exploration rate.
    later = strategy[1:]
    exploration = np.clip(later[:, 1] + later[:, 2], 0.0, 1.0)
    mean_exploration = float(np.mean(exploration))
    total_exploration = float(np.sum(exploration))
    matched_g = (
        float(np.sum(later[:, 2]) / total_exploration)
        if total_exploration > 1e-12
        else 0.0
    )
    capacity = int(info.get("capacity") or (info.get("best_params") or {}).get("capacity") or 0)
    if capacity <= 0:
        params = info.get("best_params") or info.get("fixed_hyperparams") or {}
        capacity = int(
            params.get("engine.modules.hypo_transitions_mod.kwargs.capacity", 0)
        )
    if capacity <= 0:
        nested = (info.get("best_params") or {}).get(
            "engine.modules.hypo_transitions_mod.kwargs"
        )
        if isinstance(nested, Mapping):
            capacity = int(nested.get("capacity", 0))
    if capacity <= 0:
        raise ValueError("Strategy audit requires a positive workspace capacity")
    matched_m = 1.0 - (1.0 - mean_exploration) ** (1.0 / float(capacity))
    return float(matched_m), float(np.clip(matched_g, 0.0, 1.0)), mean_exploration


def _causal_recent_accuracy(
    true_accuracy: np.ndarray,
    window_size: int,
) -> np.ndarray:
    values = np.asarray(true_accuracy, dtype=float).reshape(-1)
    output = np.full(values.size, np.nan, dtype=float)
    for trial_index in range(int(window_size), values.size):
        history = values[trial_index - int(window_size) : trial_index]
        if np.all(np.isfinite(history)):
            output[trial_index] = float(np.mean(history))
    return output


def _event_indices(
    true_accuracy: np.ndarray,
    causal_accuracy: np.ndarray,
    *,
    low_threshold: float,
) -> dict[str, list[int]]:
    correct = np.asarray(true_accuracy, dtype=float).reshape(-1)
    errors = np.isfinite(correct) & (correct < 0.5)
    error_streak: list[int] = []
    for trial_index in range(2, errors.size):
        if errors[trial_index - 2] and errors[trial_index - 1]:
            already_active = trial_index >= 3 and errors[trial_index - 3]
            if not already_active:
                error_streak.append(int(trial_index))

    low_entry: list[int] = []
    finite = np.isfinite(causal_accuracy)
    low = finite & (causal_accuracy <= float(low_threshold))
    for trial_index in np.flatnonzero(low):
        previous = int(trial_index) - 1
        if previous < 0 or not finite[previous] or not low[previous]:
            low_entry.append(int(trial_index))
    return {"error_streak": error_streak, "low_performance_entry": low_entry}


def _curve_volatility(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_pairs = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    if not np.any(finite_pairs):
        return float("nan")
    return float(np.mean(np.abs(np.diff(values)[finite_pairs])))


def _rolling_probability_runs(
    probability_runs: np.ndarray,
    *,
    window_size: int,
    start_index: int = 1,
) -> np.ndarray:
    probabilities = np.asarray(probability_runs, dtype=float)
    starts = np.arange(
        int(start_index),
        probabilities.shape[1] - int(window_size) + 1,
        dtype=int,
    )
    return np.column_stack(
        [
            np.nanmean(
                probabilities[:, start : start + int(window_size)],
                axis=1,
            )
            for start in starts
        ]
    )


def _subject_runtime(
    cfg: Mapping[str, Any],
    config_path: Path,
    subject_id: int,
    selected_subjects: Sequence[int],
) -> dict[str, Any]:
    subject_cfg = resolve_subject_config(cfg, int(subject_id))
    explicit_fixed = dict(subject_cfg.get("fixed_hyperparams") or {})
    subject_cfg = apply_fixed_hyperparams_to_subject_config(subject_cfg, explicit_fixed)
    engine_config = resolve_engine_config(
        subject_cfg,
        config_path.parent,
        subject_id=int(subject_id),
    )
    fixed_hyperparams = {
        **infer_fixed_hyperparams_from_engine_config(engine_config),
        **explicit_fixed,
    }
    engine_config = apply_fixed_hyperparams_to_engine_config(
        engine_config,
        fixed_hyperparams,
    )
    prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
    if selection_prediction_mode != "prior_t":
        raise ValueError("Particle strategy audit requires prior_t prediction semantics")
    dataset_paths = resolve_dataset_paths(
        subject_cfg,
        config_path.parent,
        DEFAULT_DATA_PATH,
    )
    return {
        "subject_cfg": subject_cfg,
        "engine_config": engine_config,
        "fixed_hyperparams": fixed_hyperparams,
        "prediction_mode": prediction_mode,
        "selection_prediction_mode": selection_prediction_mode,
        "loss_metric": resolve_loss_metric(subject_cfg),
        "loss_delta": resolve_loss_delta(
            subject_cfg,
            resolve_loss_metric(subject_cfg),
        ),
        "window_size": resolve_window_size(
            subject_cfg,
            int(subject_id),
            selected_subjects,
        ),
        "stop_at": float(subject_cfg.get("stop_at", 1.0)),
        "max_trials": (
            int(subject_cfg["max_trials"])
            if subject_cfg.get("max_trials") is not None
            else None
        ),
        "evaluation_protocol": subject_cfg.get("evaluation_protocol"),
        "dataset_paths": dataset_paths,
        "data_path": dataset_paths["learning_data"],
    }


def _run_variant(
    runtime: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    *,
    subject_id: int,
    seeds: Sequence[int],
    n_jobs: int,
) -> list[Mapping[str, Any]]:
    runner = StateModelSimulationRunner(
        engine_config=engine_config,
        processed_data_dir=runtime["dataset_paths"]["processed_dir"],
        dataset_paths=runtime["dataset_paths"],
        n_jobs=max(1, min(int(n_jobs), len(seeds))),
    )
    runner.prepare_data(runtime["data_path"])
    result = runner.simulate_subject(
        subject_id=int(subject_id),
        simulation_repeats=len(seeds),
        fixed_hyperparams=runtime["fixed_hyperparams"],
        window_size=int(runtime["window_size"]),
        stop_at=float(runtime["stop_at"]),
        max_trials=runtime["max_trials"],
        keep_logs=True,
        prediction_mode=str(runtime["prediction_mode"]),
        selection_prediction_mode=str(runtime["selection_prediction_mode"]),
        loss_metric=str(runtime["loss_metric"]),
        loss_delta=runtime["loss_delta"],
        hyper_candidate_seed=None,
        representative_run_selection="min_error",
        statistics_config={"enabled": False},
        evaluation_protocol=runtime["evaluation_protocol"],
        evaluation_role="simulation",
        trajectory_seeds=seeds,
        compute_statistics=False,
    )
    runs = list(result["best"].raw_runs or [])
    if len(runs) != len(seeds):
        raise RuntimeError(
            f"Strategy audit returned {len(runs)} runs for {len(seeds)} seeds"
        )
    return runs


def _summarize_variant(
    runs: Sequence[Mapping[str, Any]],
    *,
    subject_id: int,
    variant: str,
    window_size: int,
    n_draws: int,
    band_seed: int,
) -> dict[str, Any]:
    probability_runs: list[np.ndarray] = []
    exploit_runs: list[np.ndarray] = []
    local_runs: list[np.ndarray] = []
    global_runs: list[np.ndarray] = []
    true_accuracy = None
    observed_curve = None
    score_mask = None
    resolved_particle_count = None

    for run in runs:
        metrics = (run.get("metrics_by_mode") or {}).get("prior_t") or {}
        state_log = run.get("state_log") or {}
        probability_runs.append(np.asarray(metrics["pred_acc"], dtype=float))
        exploit_runs.append(
            np.asarray(state_log["predictive_strategy_exploit"], dtype=float)
        )
        local_runs.append(
            np.asarray(state_log["predictive_strategy_local_explore"], dtype=float)
        )
        global_runs.append(
            np.asarray(state_log["predictive_strategy_global_explore"], dtype=float)
        )
        if true_accuracy is None:
            true_accuracy = np.asarray(metrics["true_acc"], dtype=float)
            observed_curve = np.asarray(metrics["sliding_true_acc"], dtype=float)
            raw_mask = metrics.get("score_trial_mask")
            if raw_mask is None:
                raw_mask = metrics.get("valid_trial_mask")
            score_mask = (
                np.ones(true_accuracy.size, dtype=bool)
                if raw_mask is None
                else np.asarray(raw_mask, dtype=bool)
            )
            resolved_particle_count = int(metrics.get("particle_count", 0))

    probabilities = np.vstack(probability_runs)
    exploit = np.mean(np.vstack(exploit_runs), axis=0)
    local = np.mean(np.vstack(local_runs), axis=0)
    global_explore = np.mean(np.vstack(global_runs), axis=0)
    strategy_sum = exploit + local + global_explore
    if not np.allclose(strategy_sum, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{variant} strategy probabilities do not sum to one")
    if true_accuracy is None or observed_curve is None or score_mask is None:
        raise RuntimeError(f"{variant} did not produce observed accuracy metrics")

    band = conditional_behavioral_accuracy_band_metrics(
        probabilities,
        observed_curve,
        window_size=int(window_size),
        n_draws=int(n_draws),
        seed=int(band_seed),
        score_trial_mask=score_mask,
    )
    expected_trial = np.asarray(band["expected_trial_probability"], dtype=float)
    expected_curve_runs = _rolling_probability_runs(
        probabilities,
        window_size=int(window_size),
    )
    behavioral_variance = np.nanmean(probabilities * (1.0 - probabilities), axis=0)
    pf_numerical_variance = np.nanvar(probabilities, axis=0, ddof=0)
    causal_accuracy = _causal_recent_accuracy(true_accuracy, int(window_size))
    exploration = local + global_explore
    return {
        "subject_id": int(subject_id),
        "variant": str(variant),
        "window_size": int(window_size),
        "n_trials": int(expected_trial.size),
        "n_seeds": int(probabilities.shape[0]),
        "particle_count": int(resolved_particle_count or 0),
        "true_accuracy": true_accuracy,
        "causal_accuracy": causal_accuracy,
        "observed_curve": observed_curve,
        "expected_trial": expected_trial,
        "expected_curve": np.asarray(band["expected_curve"], dtype=float),
        "expected_curve_runs": expected_curve_runs,
        "q05": np.asarray(band["q05"], dtype=float),
        "q95": np.asarray(band["q95"], dtype=float),
        "exploit": exploit,
        "local": local,
        "global": global_explore,
        "exploration": exploration,
        "behavioral_variance": behavioral_variance,
        "pf_numerical_variance": pf_numerical_variance,
        "mean_behavioral_variance": float(np.nanmean(behavioral_variance)),
        "mean_pf_numerical_variance": float(np.nanmean(pf_numerical_variance)),
        "mean_pf_curve_sd": float(
            np.nanmean(np.nanstd(expected_curve_runs, axis=0, ddof=0))
        ),
        "mean_expected_accuracy": float(np.nanmean(expected_trial)),
        "expected_curve_mae": float(band["expected_curve_mae"]),
        "expected_curve_volatility": _curve_volatility(band["expected_curve"]),
        "mean_band_width_90": float(band["mean_width_90"]),
        "coverage_90": float(band["coverage_90"]),
        "mean_exploration": float(np.mean(exploration[1:])),
        "exploration_sd": float(np.std(exploration[1:])),
        "exploration_q90_range": float(
            np.quantile(exploration[1:], 0.95)
            - np.quantile(exploration[1:], 0.05)
        ),
    }


def _trial_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    n_trials = int(profile["n_trials"])
    rolling = {
        key: np.full(n_trials, np.nan, dtype=float)
        for key in ("observed_curve", "expected_curve", "q05", "q95")
    }
    window_size = int(profile["window_size"])
    curve_x = np.arange(window_size + 1, n_trials + 1, dtype=int)
    for key in rolling:
        values = np.asarray(profile[key], dtype=float)
        n = min(values.size, curve_x.size)
        if n:
            rolling[key][curve_x[:n] - 1] = values[:n]

    rows = []
    for index in range(n_trials):
        rows.append(
            {
                "subject_id": int(profile["subject_id"]),
                "variant": str(profile["variant"]),
                "trial": int(index + 1),
                "true_correct": float(profile["true_accuracy"][index]),
                "causal_recent_accuracy": float(profile["causal_accuracy"][index]),
                "expected_correct_probability": float(profile["expected_trial"][index]),
                "strategy_exploit": float(profile["exploit"][index]),
                "strategy_local_explore": float(profile["local"][index]),
                "strategy_global_explore": float(profile["global"][index]),
                "strategy_explore": float(profile["exploration"][index]),
                "behavioral_variance": float(profile["behavioral_variance"][index]),
                "pf_numerical_variance": float(
                    profile["pf_numerical_variance"][index]
                ),
                "rolling_observed_accuracy": float(rolling["observed_curve"][index]),
                "rolling_expected_accuracy": float(rolling["expected_curve"][index]),
                "rolling_behavioral_q05": float(rolling["q05"][index]),
                "rolling_behavioral_q95": float(rolling["q95"][index]),
            }
        )
    return rows


def _event_rows(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    low_threshold: float,
    relative_start: int,
    relative_end: int,
) -> list[dict[str, Any]]:
    dynamic = profiles[DYNAMIC]
    events = _event_indices(
        dynamic["true_accuracy"],
        dynamic["causal_accuracy"],
        low_threshold=low_threshold,
    )
    rows: list[dict[str, Any]] = []
    for event_type, indices in events.items():
        for event_number, event_index in enumerate(indices, start=1):
            for variant, profile in profiles.items():
                for relative_trial in range(int(relative_start), int(relative_end) + 1):
                    index = int(event_index) + int(relative_trial)
                    if not 0 <= index < int(profile["n_trials"]):
                        continue
                    rows.append(
                        {
                            "subject_id": int(profile["subject_id"]),
                            "event_type": event_type,
                            "event_number": int(event_number),
                            "event_trial": int(event_index + 1),
                            "variant": variant,
                            "relative_trial": int(relative_trial),
                            "strategy_explore": float(profile["exploration"][index]),
                            "expected_correct_probability": float(
                                profile["expected_trial"][index]
                            ),
                            "causal_recent_accuracy": float(
                                profile["causal_accuracy"][index]
                            ),
                        }
                    )
    return rows


def _summary_rows(
    profiles: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    dynamic = profiles[DYNAMIC]
    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        profile = profiles[variant]
        curve_n = min(
            len(dynamic["expected_curve"]),
            len(profile["expected_curve"]),
        )
        dynamic_curve = np.asarray(dynamic["expected_curve"][:curve_n], dtype=float)
        variant_curve = np.asarray(profile["expected_curve"][:curve_n], dtype=float)
        finite = np.isfinite(dynamic_curve) & np.isfinite(variant_curve)
        curve_mae = (
            float(np.mean(np.abs(dynamic_curve[finite] - variant_curve[finite])))
            if np.any(finite)
            else float("nan")
        )
        dynamic_curve_runs = np.asarray(dynamic["expected_curve_runs"], dtype=float)
        variant_curve_runs = np.asarray(profile["expected_curve_runs"], dtype=float)
        run_count = min(dynamic_curve_runs.shape[0], variant_curve_runs.shape[0])
        curve_count = min(dynamic_curve_runs.shape[1], variant_curve_runs.shape[1])
        paired_effects = np.nanmean(
            np.abs(
                dynamic_curve_runs[:run_count, :curve_count]
                - variant_curve_runs[:run_count, :curve_count]
            ),
            axis=1,
        )
        paired_effect_mean = float(np.nanmean(paired_effects))
        paired_effect_sd = float(np.nanstd(paired_effects, ddof=0))
        pf_curve_sd = float(dynamic["mean_pf_curve_sd"])
        rows.append(
            {
                "subject_id": int(profile["subject_id"]),
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "n_trials": int(profile["n_trials"]),
                "n_common_seeds": int(profile["n_seeds"]),
                "audit_particle_count": int(profile["particle_count"]),
                "mean_expected_accuracy": profile["mean_expected_accuracy"],
                "expected_curve_mae_to_subject": profile["expected_curve_mae"],
                "expected_curve_volatility": profile["expected_curve_volatility"],
                "mean_behavioral_band_width_90": profile["mean_band_width_90"],
                "behavioral_band_coverage_90": profile["coverage_90"],
                "mean_behavioral_variance": profile["mean_behavioral_variance"],
                "mean_pf_numerical_variance": profile[
                    "mean_pf_numerical_variance"
                ],
                "mean_pf_rolling_curve_sd": profile["mean_pf_curve_sd"],
                "mean_exploration": profile["mean_exploration"],
                "exploration_sd": profile["exploration_sd"],
                "exploration_q90_range": profile["exploration_q90_range"],
                "counterfactual_curve_mae_from_dynamic": curve_mae,
                "paired_counterfactual_curve_mae_mean": paired_effect_mean,
                "paired_counterfactual_curve_mae_sd": paired_effect_sd,
                "paired_effect_to_dynamic_pf_curve_sd_ratio": (
                    paired_effect_mean / pf_curve_sd
                    if np.isfinite(pf_curve_sd) and pf_curve_sd > 1e-12
                    else float("nan")
                ),
                "counterfactual_band_width_delta_from_dynamic": float(
                    profile["mean_band_width_90"] - dynamic["mean_band_width_90"]
                ),
                "counterfactual_volatility_delta_from_dynamic": float(
                    profile["expected_curve_volatility"]
                    - dynamic["expected_curve_volatility"]
                ),
            }
        )
    return rows


def _save_subject_curves(
    trial_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    save_path: Path,
) -> None:
    subjects = sorted(int(value) for value in trial_df["subject_id"].unique())
    n_cols = min(4, len(subjects))
    n_rows = int(np.ceil(len(subjects) / n_cols))
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4.4, n_rows * 3.0 + 0.8),
            squeeze=False,
            sharey=True,
        )
        legend_items: dict[str, Any] = {}
        for ax, subject_id in zip(axes.flat, subjects):
            subject = trial_df[trial_df["subject_id"] == subject_id]
            dynamic = subject[subject["variant"] == DYNAMIC]
            rolling = dynamic.dropna(subset=["rolling_expected_accuracy"])
            if not rolling.empty:
                band = ax.fill_between(
                    rolling["trial"].to_numpy(dtype=float),
                    rolling["rolling_behavioral_q05"].to_numpy(dtype=float),
                    rolling["rolling_behavioral_q95"].to_numpy(dtype=float),
                    color="#9DB9D8",
                    alpha=0.25,
                    linewidth=0,
                    label="Dynamic 90% behavioral PI",
                )
                legend_items.setdefault("Dynamic 90% behavioral PI", band)
                observed = ax.plot(
                    rolling["trial"],
                    rolling["rolling_observed_accuracy"],
                    color="#111111",
                    linewidth=2.0,
                    label="Subject accuracy",
                    zorder=6,
                )[0]
                legend_items.setdefault("Subject accuracy", observed)
            for variant in VARIANT_ORDER:
                variant_rows = subject[subject["variant"] == variant].dropna(
                    subset=["rolling_expected_accuracy"]
                )
                if variant_rows.empty:
                    continue
                line = ax.plot(
                    variant_rows["trial"],
                    variant_rows["rolling_expected_accuracy"],
                    color=VARIANT_COLORS[variant],
                    linewidth=2.0 if variant == DYNAMIC else 1.5,
                    linestyle="-" if variant != CONTROLLER_OFF else "--",
                    label=VARIANT_LABELS[variant],
                    zorder=5 if variant == DYNAMIC else 4,
                )[0]
                legend_items.setdefault(VARIANT_LABELS[variant], line)
            effect = summary_df[
                (summary_df["subject_id"] == subject_id)
                & (summary_df["variant"] == MEAN_MATCHED_STATIC)
            ]["paired_counterfactual_curve_mae_mean"]
            effect_text = float(effect.iloc[0]) if not effect.empty else float("nan")
            ax.text(
                0.02,
                0.03,
                f"Paired dynamic vs mean-static MAE: {effect_text:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.80,
                },
            )
            ax.set(
                title=f"Subject {subject_id}",
                xlabel="Trial",
                ylabel="Rolling accuracy",
                ylim=(0.0, 1.0),
            )
            ax.grid(axis="y", alpha=0.18)
        for ax in axes.flat[len(subjects) :]:
            ax.axis("off")
        fig.suptitle(
            "Strategy-freezing counterfactuals",
            fontsize=12,
            y=0.99,
        )
        fig.text(
            0.5,
            0.945,
            "Common PF seeds; only the exploration controller is changed",
            ha="center",
            va="top",
            fontsize=9,
        )
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=min(5, len(legend_items)),
            fontsize=8,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.86))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _save_summary_figure(summary_df: pd.DataFrame, save_path: Path) -> None:
    subjects = sorted(int(value) for value in summary_df["subject_id"].unique())
    x = np.arange(len(subjects), dtype=float)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))
        ax = axes[0, 0]
        for variant, offset in ((MEAN_MATCHED_STATIC, -0.08), (CONTROLLER_OFF, 0.08)):
            rows = summary_df[summary_df["variant"] == variant].set_index("subject_id")
            values = np.asarray(
                [rows.loc[sid, "paired_counterfactual_curve_mae_mean"] for sid in subjects],
                dtype=float,
            )
            errors = np.asarray(
                [rows.loc[sid, "paired_counterfactual_curve_mae_sd"] for sid in subjects],
                dtype=float,
            )
            ax.errorbar(
                x + offset,
                values,
                yerr=errors,
                color=VARIANT_COLORS[variant],
                fmt="o",
                markersize=5,
                capsize=2,
                label=VARIANT_LABELS[variant],
                zorder=3,
            )
        ax.set(
            title="a  Expected-accuracy change after freezing",
            ylabel="Paired common-seed MAE from dynamic curve",
            xticks=x,
            xticklabels=subjects,
        )
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.18)

        ax = axes[0, 1]
        width = 0.24
        for index, variant in enumerate(VARIANT_ORDER):
            rows = summary_df[summary_df["variant"] == variant].set_index("subject_id")
            values = np.asarray(
                [rows.loc[sid, "mean_behavioral_band_width_90"] for sid in subjects],
                dtype=float,
            )
            ax.bar(
                x + (index - 1) * width,
                values,
                width=width,
                color=VARIANT_COLORS[variant],
                alpha=0.78,
                label=VARIANT_LABELS[variant],
            )
        ax.set(
            title="b  Conditional behavioral interval width",
            ylabel="Mean 90% PI width",
            xticks=x,
            xticklabels=subjects,
        )
        ax.grid(axis="y", alpha=0.18)

        ax = axes[1, 0]
        dynamic = summary_df[summary_df["variant"] == DYNAMIC].set_index("subject_id")
        behavioral = np.asarray(
            [dynamic.loc[sid, "mean_behavioral_variance"] for sid in subjects],
            dtype=float,
        )
        numerical = np.asarray(
            [dynamic.loc[sid, "mean_pf_numerical_variance"] for sid in subjects],
            dtype=float,
        )
        ax.scatter(
            x - 0.08,
            behavioral,
            color="#4F81B8",
            s=34,
            label="Behavioral Bernoulli variance",
        )
        ax.scatter(
            x + 0.08,
            np.maximum(numerical, 1e-12),
            color="#8C6BB1",
            marker="D",
            s=28,
            label="Across-seed PF numerical variance",
        )
        ax.set_yscale("log")
        ax.set(
            title="c  Predictive variance sources",
            ylabel="Mean trialwise variance (log scale)",
            xticks=x,
            xticklabels=subjects,
        )
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.18)

        ax = axes[1, 1]
        matched = summary_df[summary_df["variant"] == MEAN_MATCHED_STATIC].set_index(
            "subject_id"
        )
        strategy_range = np.asarray(
            [dynamic.loc[sid, "exploration_q90_range"] for sid in subjects],
            dtype=float,
        )
        effects = np.asarray(
            [
                matched.loc[sid, "paired_counterfactual_curve_mae_mean"]
                for sid in subjects
            ],
            dtype=float,
        )
        ax.scatter(strategy_range, effects, color="#0072B2", s=40)
        for sid, x_value, y_value in zip(subjects, strategy_range, effects):
            ax.annotate(
                str(sid),
                (x_value, y_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set(
            title="d  Strategy dynamics versus counterfactual effect",
            xlabel="Exploration 95th–5th percentile range",
            ylabel="Dynamic vs mean-static curve MAE",
        )
        ax.grid(alpha=0.18)

        fig.suptitle(
            "Particle-filter strategy contribution audit",
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def _event_group_summary(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame()
    per_subject = (
        event_df.groupby(
            ["event_type", "variant", "relative_trial", "subject_id"],
            as_index=False,
        )[
            [
                "strategy_explore",
                "expected_correct_probability",
                "causal_recent_accuracy",
            ]
        ]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for keys, values in per_subject.groupby(
        ["event_type", "variant", "relative_trial"]
    ):
        event_type, variant, relative_trial = keys
        row: dict[str, Any] = {
            "event_type": event_type,
            "variant": variant,
            "relative_trial": int(relative_trial),
            "n_subjects": int(values["subject_id"].nunique()),
        }
        for column in (
            "strategy_explore",
            "expected_correct_probability",
            "causal_recent_accuracy",
        ):
            finite = values[column].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            row[f"{column}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            row[f"{column}_sem"] = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                if finite.size >= 2
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _save_event_figure(
    event_df: pd.DataFrame,
    event_summary: pd.DataFrame,
    save_path: Path,
) -> None:
    if event_df.empty or event_summary.empty:
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(7.2, 2.8))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No qualifying error-streak or low-performance entry events",
                ha="center",
                va="center",
                fontsize=10,
            )
            fig.suptitle("Event-aligned strategy response audit", fontsize=12)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
        return
    event_types = ("error_streak", "low_performance_entry")
    titles = {
        "error_streak": "After two consecutive errors",
        "low_performance_entry": "Entry into causal low-performance phase",
    }
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.8), sharex=True)
        legend_items: dict[str, Any] = {}
        for row_index, event_type in enumerate(event_types):
            event_rows = event_summary[event_summary["event_type"] == event_type]
            for column_index, (metric, ylabel) in enumerate(
                (
                    ("strategy_explore", "Exploration tendency"),
                    ("expected_correct_probability", "Predicted correct probability"),
                )
            ):
                ax = axes[row_index, column_index]
                for variant in VARIANT_ORDER:
                    values = event_rows[event_rows["variant"] == variant].sort_values(
                        "relative_trial"
                    )
                    if values.empty:
                        continue
                    x = values["relative_trial"].to_numpy(dtype=float)
                    mean = values[f"{metric}_mean"].to_numpy(dtype=float)
                    sem = values[f"{metric}_sem"].to_numpy(dtype=float)
                    line = ax.plot(
                        x,
                        mean,
                        color=VARIANT_COLORS[variant],
                        linewidth=2.0 if variant == DYNAMIC else 1.5,
                        linestyle="-" if variant != CONTROLLER_OFF else "--",
                        label=VARIANT_LABELS[variant],
                    )[0]
                    legend_items.setdefault(VARIANT_LABELS[variant], line)
                    if variant == DYNAMIC:
                        ax.fill_between(
                            x,
                            mean - sem,
                            mean + sem,
                            color=VARIANT_COLORS[variant],
                            alpha=0.16,
                            linewidth=0,
                        )
                if metric == "expected_correct_probability":
                    observed = event_rows[event_rows["variant"] == DYNAMIC].sort_values(
                        "relative_trial"
                    )
                    if not observed.empty:
                        observed_line = ax.plot(
                            observed["relative_trial"],
                            observed["causal_recent_accuracy_mean"],
                            color="#111111",
                            linewidth=1.7,
                            label="Causal subject accuracy",
                        )[0]
                        legend_items.setdefault("Causal subject accuracy", observed_line)
                ax.axvline(0.0, color="#555555", linestyle=":", linewidth=1.0)
                ax.set(
                    title=(
                        f"{titles[event_type]} — {ylabel.lower()}"
                        if column_index == 0
                        else ylabel
                    ),
                    xlabel="Trials relative to event",
                    ylabel=ylabel,
                    ylim=(0.0, 1.0),
                )
                ax.grid(axis="y", alpha=0.18)
        counts = (
            event_df.groupby("event_type")
            .agg(events=("event_number", "count"), subjects=("subject_id", "nunique"))
            if not event_df.empty
            else pd.DataFrame()
        )
        count_text = []
        for event_type in event_types:
            if event_type in counts.index:
                # Each event appears once per variant and relative trial; report
                # unique subject-event pairs instead of raw long-form rows.
                unique_events = event_df[event_df["event_type"] == event_type][
                    ["subject_id", "event_number"]
                ].drop_duplicates()
                count_text.append(
                    f"{event_type}: {len(unique_events)} events / "
                    f"{unique_events['subject_id'].nunique()} subjects"
                )
        fig.suptitle(
            "Event-aligned strategy response audit",
            fontsize=12,
            y=0.995,
        )
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=min(4, len(legend_items)),
            fontsize=8,
        )
        if count_text:
            fig.text(
                0.5,
                0.01,
                " | ".join(count_text),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444444",
            )
        fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.90))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def run_particle_filter_strategy_audit(
    results: Mapping[int, Mapping[str, Any]],
    *,
    simulation_config_path: str | Path,
    output_dir: str | Path,
    subjects: Sequence[int] | None = None,
    common_seeds: Sequence[int] = (20260821, 20260822, 20260823, 20260824),
    n_jobs: int = 1,
    particle_count: int = 32,
    n_behavioral_draws: int = 3000,
    behavioral_seed: int = 20260825,
    low_accuracy_threshold: float = 0.60,
    event_relative_start: int = -8,
    event_relative_end: int = 16,
) -> dict[str, Any]:
    """Run common-seed strategy-freezing counterfactuals and save the audit."""
    config_path = Path(simulation_config_path).resolve()
    cfg = load_yaml(config_path)
    configured_subjects = resolve_subjects(subjects, None, cfg)
    selected_subjects = [sid for sid in configured_subjects if sid in results]
    if not selected_subjects:
        raise ValueError("No strategy-audit subjects overlap the loaded results")
    seeds = tuple(int(seed) for seed in common_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Strategy-audit common seeds must be non-empty and unique")
    particle_count = int(particle_count)
    if particle_count <= 0:
        raise ValueError("Strategy-audit particle_count must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for subject_id in selected_subjects:
        info = results[int(subject_id)]
        if str(info.get("state_distribution_kind", "")) != "particle_marginal":
            raise ValueError(
                f"Subject {subject_id} is not a particle-marginal result"
            )
        LOGGER.info("Strategy audit subject %s", subject_id)
        runtime = _subject_runtime(cfg, config_path, subject_id, selected_subjects)
        base_engine = runtime["engine_config"]
        base_kwargs = _transition_kwargs(deepcopy(base_engine))
        baseline_m = _controller_setting(
            base_kwargs,
            "rate_controller",
            "m",
            0.15,
        )
        baseline_g = _controller_setting(
            base_kwargs,
            "range_controller",
            "g",
            0.35,
        )
        matched_m, matched_g, _ = _mean_matched_controls(info)
        engines = {
            DYNAMIC: base_engine,
            MEAN_MATCHED_STATIC: _disable_strategy_controllers(
                base_engine,
                m=matched_m,
                g=matched_g,
            ),
            CONTROLLER_OFF: _disable_strategy_controllers(
                base_engine,
                m=baseline_m,
                g=baseline_g,
            ),
        }
        for engine in engines.values():
            inference = engine.setdefault("inference", {})
            inference["particle_count"] = particle_count

        profiles: dict[str, Mapping[str, Any]] = {}
        for variant in VARIANT_ORDER:
            runs = _run_variant(
                runtime,
                engines[variant],
                subject_id=subject_id,
                seeds=seeds,
                n_jobs=n_jobs,
            )
            profiles[variant] = _summarize_variant(
                runs,
                subject_id=subject_id,
                variant=variant,
                window_size=int(runtime["window_size"]),
                n_draws=int(n_behavioral_draws),
                band_seed=int(
                    np.random.SeedSequence(
                        [int(behavioral_seed), int(subject_id), VARIANT_ORDER.index(variant)]
                    ).generate_state(1)[0]
                ),
            )
            trial_rows.extend(_trial_rows(profiles[variant]))
        summary_rows.extend(_summary_rows(profiles))
        event_rows.extend(
            _event_rows(
                profiles,
                low_threshold=low_accuracy_threshold,
                relative_start=event_relative_start,
                relative_end=event_relative_end,
            )
        )

    trial_df = pd.DataFrame(trial_rows)
    summary_df = pd.DataFrame(summary_rows)
    event_df = pd.DataFrame(
        event_rows,
        columns=[
            "subject_id",
            "event_type",
            "event_number",
            "event_trial",
            "variant",
            "relative_trial",
            "strategy_explore",
            "expected_correct_probability",
            "causal_recent_accuracy",
        ],
    )
    event_summary = _event_group_summary(event_df)
    if event_summary.empty:
        event_summary = pd.DataFrame(
            columns=[
                "event_type",
                "variant",
                "relative_trial",
                "n_subjects",
                "strategy_explore_mean",
                "strategy_explore_sem",
                "expected_correct_probability_mean",
                "expected_correct_probability_sem",
                "causal_recent_accuracy_mean",
                "causal_recent_accuracy_sem",
            ]
        )

    trial_csv = output_dir / "strategy_audit_trial_data.csv"
    summary_csv = output_dir / "strategy_audit_summary.csv"
    event_csv = output_dir / "strategy_audit_event_data.csv"
    event_summary_csv = output_dir / "strategy_audit_event_summary.csv"
    trial_df.to_csv(trial_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    event_df.to_csv(event_csv, index=False)
    event_summary.to_csv(event_summary_csv, index=False)

    curves_png = output_dir / "strategy_counterfactual_accuracy.png"
    summary_png = output_dir / "strategy_contribution_summary.png"
    event_png = output_dir / "strategy_event_alignment.png"
    _save_subject_curves(trial_df, summary_df, curves_png)
    _save_summary_figure(summary_df, summary_png)
    _save_event_figure(event_df, event_summary, event_png)

    return {
        "subjects": selected_subjects,
        "common_seeds": list(seeds),
        "variants": list(VARIANT_ORDER),
        "outputs": [
            trial_csv,
            summary_csv,
            event_csv,
            event_summary_csv,
            curves_png,
            summary_png,
            event_png,
        ],
        "summary": summary_df,
        "event_summary": event_summary,
    }


__all__ = [
    "CONTROLLER_OFF",
    "DYNAMIC",
    "MEAN_MATCHED_STATIC",
    "VARIANT_ORDER",
    "run_particle_filter_strategy_audit",
]
