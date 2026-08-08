"""Model evaluation facade.

The public surface follows the output layout used by
``run_model_evaluation.py``:

- basic:
  - accuracy_comparison and accuracy_family_comparison
  - accuracy_band
  - dynamic_strategy_profile and hypothesis_active_set_counts
  - choice_brier
  - posterior_probabilities and prior_probabilities
  - beta_dynamics
- oral_alignment:
  - oral_mass_distribution
  - distribution_based_alignment: oral reports -> hypothesis distribution
    (optionally projected into oral-equivalence groups)
  - oral_based_alignment: model belief -> oral center/region representation
  - target_based_alignment: target prior probability vs oral target mass
  - hit_based_alignment: target hit in model active/top-k set vs oral top-N/top-k set
  - coverage_based_alignment: model active-set coverage of oral top-N set
- trajectory:
  - trajectory_accuracy
  - trajectory_posterior

Oral mass and Oral/model alignment are implemented in
``oral_model_alignment.py`` and mixed in here so existing ``ModelEval`` call
sites keep working.
"""

from collections import defaultdict
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.Bayesian_state.metrics import (
    accuracy_curve_metrics,
    accuracy_metrics_from_info,
    behavior_ppc_group_metrics,
    choice_brier_curve_metrics_from_info,
    exponential_accuracy_metrics_from_info,
    family_correct,
    family_indices,
    history_kernel_metrics,
    predictive_accuracy_band_metrics,
    sliding_binary_metrics,
    switch_behavior_metrics,
    target_majority_accuracy_metrics_from_info,
    validate_exp_accuracy_alpha,
)
from src.Bayesian_state.model_evaluation.oral_model_alignment import OralModelAlignmentMixin
from src.Bayesian_state.problems.partitions import Partition
from src.Bayesian_state.utils.stream import StreamList


logger = logging.getLogger(__name__)


class ModelEval(OralModelAlignmentMixin):
    """Evaluation and plotting entry point for state-based model results."""

    DEFAULT_TRAJECTORY_RANKS = (
        1, 2, 3, 4,
        100, 101, 102, 103,
        1000, 1001, 1002, 1003,
        4000, 4001, 4002, 4003,
    )
    DEFAULT_TOP16_RANKS = tuple(range(1, 17))
    PROFILE_POLICY_ORDER = ("conservative", "stable", "aggressive", "stubborn")
    PROFILE_POLICY_COLORS = {
        "conservative": "#0072B2",
        "stable": "#009E73",
        "aggressive": "#D55E00",
        "stubborn": "#CC79A7",
    }

    # Shared plotting helpers -------------------------------------------------

    @staticmethod
    def _filter_results(results, subjects):
        if subjects is not None:
            return {iSub: results[iSub] for iSub in subjects if iSub in results}
        return results

    @staticmethod
    def _group_by_condition(results):
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info["condition"]].append((iSub, info))
        return grouped

    @staticmethod
    def _layout_by_condition(grouped, kwargs):
        max_group = max(len(lst) for lst in grouped.values())
        if "n_cols" in kwargs and kwargs.get("n_cols") is not None:
            n_cols = int(kwargs.get("n_cols"))
        else:
            n_cols = min(int(kwargs.get("max_subjects_per_row", 8)), max_group)
        n_cols = max(1, n_cols)
        rows_by_condition = {
            condition: int(np.ceil(len(subs) / n_cols))
            for condition, subs in grouped.items()
        }
        n_rows = max(1, sum(rows_by_condition.values()))
        return n_rows, n_cols, rows_by_condition

    def _plot_by_condition(self, results, subjects, save_path, title, plot_body, **kwargs):
        results = self._filter_results(results, subjects)
        grouped = self._group_by_condition(results)

        if not grouped:
            raise RuntimeError(f"No results to plot for: {title}")

        n_rows, n_cols, rows_by_condition = self._layout_by_condition(grouped, kwargs)
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(title, fontsize=kwargs.get("fontsize", 16), y=kwargs.get("y", 0.99))

        row_offset = 0
        for condition, subs in sorted(grouped.items()):
            for idx, (iSub, info) in enumerate(subs):
                local_row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(n_rows, n_cols, (row_offset + local_row) * n_cols + col + 1)
                plot_body(ax, condition, iSub, info)
            row_offset += rows_by_condition[condition]

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            logger.info("%s saved to %s", title, save_path)

    @staticmethod
    def _as_float_1d(values, field_name):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{field_name} is empty")
        return arr

    @staticmethod
    def _safe_float(value, default=np.nan):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if np.isfinite(out) else default

    @staticmethod
    def _extract_beta_log(info):
        beta_log = info.get("beta_log")
        if beta_log is not None:
            try:
                arr = np.asarray(beta_log, dtype=float)
            except (TypeError, ValueError):
                arr = np.asarray([], dtype=float)
            if arr.size:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                return arr

        rows = []
        for step in info.get("best_step_results") or info.get("step_results") or []:
            if not isinstance(step, dict):
                continue
            beta = step.get("beta")
            if beta is None:
                beta = step.get("beta_values")
            if beta is None:
                continue
            try:
                rows.append(np.asarray(beta, dtype=float).reshape(-1))
            except (TypeError, ValueError):
                continue
        if not rows:
            return np.empty((0, 0), dtype=float)
        try:
            return np.vstack(rows)
        except ValueError:
            return np.empty((0, 0), dtype=float)

    @staticmethod
    def _extract_subject_trials(info):
        trials = info.get("subject_trials") or info.get("trials")
        if trials is None:
            return None, None, None

        if isinstance(trials, dict):
            choices = trials.get("choice")
            categories = trials.get("category")
            feature_cols = [f"feature{i}" for i in range(1, 5)]
            if all(col in trials for col in feature_cols):
                stimulus = np.column_stack([np.asarray(trials[col], dtype=float) for col in feature_cols])
            else:
                stimulus = None
        else:
            df = pd.DataFrame(trials)
            choices = df["choice"].to_numpy() if "choice" in df else None
            categories = df["category"].to_numpy() if "category" in df else None
            feature_cols = [f"feature{i}" for i in range(1, 5)]
            stimulus = df[feature_cols].to_numpy(dtype=float) if all(col in df for col in feature_cols) else None

        if choices is None or categories is None:
            return None, None, None
        return np.asarray(choices, dtype=int), np.asarray(categories, dtype=int), stimulus

    def compute_target_majority_accuracy_metrics(self, info, window_size=None):
        """Compute Exp4 choice accuracy against the higher-probability target option."""
        return target_majority_accuracy_metrics_from_info(
            info, window_size=window_size
        )

    def compute_accuracy_metrics(self, info, window_size=None):
        return accuracy_metrics_from_info(info, window_size=window_size)

    _validate_exp_accuracy_alpha = staticmethod(validate_exp_accuracy_alpha)

    def compute_exponential_accuracy_metrics(self, info, exp_accuracy_alpha=None):
        return exponential_accuracy_metrics_from_info(
            info, exp_accuracy_alpha=exp_accuracy_alpha
        )

    @staticmethod
    def _resolve_beta_for_hypo(beta_vec, hypo, default_beta):
        if beta_vec is None:
            return float(default_beta)
        arr = np.asarray(beta_vec, dtype=float).reshape(-1)
        if hypo < arr.size and np.isfinite(arr[hypo]) and arr[hypo] > 0:
            return float(arr[hypo])
        return float(default_beta)

    def compute_family_accuracy_metrics(
        self,
        info,
        condition=None,
        window_size=None,
        prediction_mode=None,
        default_beta=10.0,
        distance_mode="prototype",
    ):
        """Compute relaxed family-level accuracy curves from saved trial logs."""
        choices, categories, stimulus = self._extract_subject_trials(info)
        if choices is None or categories is None:
            return {}

        condition = int(condition if condition is not None else info.get("condition", 1))
        n_cats = 2 if condition == 1 else 4
        partition = Partition(n_dims=4, n_cats=n_cats)
        steps = info.get("best_step_results") or info.get("step_results") or []
        posterior_log = info.get("posterior_log") or []
        prior_log = info.get("prior_log") or []

        mode = (
            prediction_mode
            or info.get("eval_prediction_mode")
            or (info.get("selection_meta") or {}).get("selection_prediction_mode")
            or info.get("selection_prediction_mode")
            or "posterior_t_minus_1"
        )

        n_trials = min(len(choices), len(categories))
        if mode == "prior_t":
            prior_len = len(prior_log) if prior_log else len(steps)
            n_trials = min(n_trials, prior_len)
        else:
            n_trials = min(n_trials, len(posterior_log))
        if n_trials <= 1:
            return {}

        win = window_size if window_size is not None else info.get("window_size")
        try:
            win = int(win)
        except (TypeError, ValueError):
            win = 16
        if win <= 0 or n_trials < win + 1:
            return {}

        choices = choices[:n_trials]
        categories = categories[:n_trials]
        true_family_acc = family_correct(categories, choices, n_cats)
        pred_family_acc = np.full(n_trials, np.nan, dtype=float)

        beta_arr = self._extract_beta_log(info)
        beta_vec = beta_arr[-1] if beta_arr.size else None

        for trial_idx in range(1, n_trials):
            if mode == "prior_t":
                if prior_log:
                    current_dist = np.asarray(prior_log[trial_idx], dtype=float).reshape(-1)
                elif trial_idx < len(steps) and isinstance(steps[trial_idx], dict):
                    current_dist = np.asarray(steps[trial_idx].get("prior"), dtype=float).reshape(-1)
                else:
                    continue
            else:
                current_dist = np.asarray(posterior_log[trial_idx - 1], dtype=float).reshape(-1)

            if trial_idx < len(steps) and isinstance(steps[trial_idx], dict):
                perceived = steps[trial_idx].get("perceived_stimulus")
            else:
                perceived = None
            if perceived is None and stimulus is not None and trial_idx < len(stimulus):
                perceived = stimulus[trial_idx]
            if perceived is None:
                continue

            trial_slice = (
                [np.asarray(perceived, dtype=float)],
                [int(choices[trial_idx])],
                [float(true_family_acc[trial_idx])],
                [int(categories[trial_idx])],
            )
            family_idx = family_indices(int(categories[trial_idx]), n_cats)
            weighted_family_prob = 0.0
            for hypo, weight in enumerate(current_dist):
                if weight <= 0:
                    continue
                beta_for_hypo = self._resolve_beta_for_hypo(beta_vec, hypo, default_beta)
                prob = partition.get_category_probabilities(
                    hypo=hypo,
                    data=trial_slice,
                    beta=beta_for_hypo,
                    distance_mode=distance_mode,
                )
                if prob.ndim == 1:
                    prob = prob.reshape(-1, 1)
                valid_family_idx = family_idx[family_idx < prob.shape[0]]
                if valid_family_idx.size:
                    weighted_family_prob += float(weight) * float(np.sum(prob[valid_family_idx, 0]))
            pred_family_acc[trial_idx] = weighted_family_prob

        sliding_true, sliding_pred, sliding_std = sliding_binary_metrics(
            true_family_acc,
            pred_family_acc,
            window_size=win,
        )

        return {
            "true_family_acc": true_family_acc,
            "pred_family_acc": pred_family_acc,
            "sliding_true_family_acc": sliding_true,
            "sliding_pred_family_acc": sliding_pred,
            "sliding_pred_family_acc_std": sliding_std,
        }

    # posterior/prior ---------------------------------------------------------

    @staticmethod
    def _hypothesis_color_map(max_k, palette_name="husl"):
        max_k = max(0, int(max_k))
        if max_k == 0:
            return {}
        colors = sns.color_palette(palette_name, n_colors=max_k)
        return {k: colors[k] for k in range(max_k)}

    def plot_posterior_probabilities(self, results, subjects=None, save_path=None, limit=True, **kwargs):
        def body(ax, condition, iSub, info):
            posterior_log = info.get("posterior_log") or []
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Posterior Probability",
            )

            if not posterior_log:
                ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
                return

            probability_threshold = float(kwargs.get("probability_threshold", 1e-12))
            if limit:
                max_k = 19 if condition == 1 else 116
            else:
                max_k = max((len(posterior) for posterior in posterior_log), default=0)

            data = []
            for step, posterior in enumerate(posterior_log):
                try:
                    posterior = np.asarray(posterior, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    continue
                for k in range(min(max_k, posterior.size)):
                    value = posterior[k]
                    if not np.isfinite(value) or value <= probability_threshold:
                        continue
                    data.append({"Step": step + 1, "k": k, "Posterior": float(value)})

            df = pd.DataFrame(data)
            if df.empty or "Step" not in df.columns:
                ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
                return

            color_map = self._hypothesis_color_map(max_k, kwargs.get("hypothesis_palette", "husl"))
            sns.scatterplot(
                data=df,
                x="Step",
                y="Posterior",
                hue="k",
                hue_order=list(range(max_k)),
                palette=color_map,
                alpha=0.5,
                legend=False,
                ax=ax,
            )

            target_hypo = info.get("target_hypothesis")
            if target_hypo is None:
                target_hypo = 0 if condition == 1 else 42
            target_hypo = int(target_hypo)
            target_df = df[df["k"] == target_hypo]
            if not target_df.empty:
                sns.scatterplot(data=target_df, x="Step", y="Posterior", color="red", s=50, ax=ax)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Posterior Probabilities for k by Subject",
            body,
            **kwargs,
        )

    @staticmethod
    def _extract_prior_probability_log(info):
        prior_log = info.get("prior_log") or []
        if prior_log:
            out = []
            for prior in prior_log:
                try:
                    out.append(np.asarray(prior, dtype=float).reshape(-1))
                except (TypeError, ValueError):
                    out.append(np.asarray([], dtype=float))
            return out

        steps = info.get("step_results") or info.get("best_step_results") or []
        out = []
        for step in steps:
            if not isinstance(step, dict) or step.get("prior") is None:
                continue
            try:
                out.append(np.asarray(step.get("prior"), dtype=float).reshape(-1))
            except (TypeError, ValueError):
                out.append(np.asarray([], dtype=float))
        return out

    def plot_prior_probabilities(self, results, subjects=None, save_path=None, limit=True, **kwargs):
        def body(ax, condition, iSub, info):
            prior_log = self._extract_prior_probability_log(info)
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Prior Probability",
            )

            if not prior_log:
                ax.text(0.5, 0.5, "No prior data", ha="center", va="center", transform=ax.transAxes)
                return

            probability_threshold = float(kwargs.get("probability_threshold", 1e-12))
            if limit:
                max_k = 19 if condition == 1 else 116
            else:
                max_k = max((prior.size for prior in prior_log), default=0)

            data = []
            for step, prior in enumerate(prior_log):
                if prior.size == 0:
                    continue
                for k in range(min(max_k, prior.size)):
                    value = prior[k]
                    if not np.isfinite(value) or value <= probability_threshold:
                        continue
                    data.append({"Step": step + 1, "k": k, "Prior": float(value)})

            df = pd.DataFrame(data)
            if df.empty or "Step" not in df.columns:
                ax.text(0.5, 0.5, "No prior data", ha="center", va="center", transform=ax.transAxes)
                return

            color_map = self._hypothesis_color_map(max_k, kwargs.get("hypothesis_palette", "husl"))
            sns.scatterplot(
                data=df,
                x="Step",
                y="Prior",
                hue="k",
                hue_order=list(range(max_k)),
                palette=color_map,
                alpha=0.5,
                legend=False,
                ax=ax,
            )

            target_hypo = info.get("target_hypothesis")
            if target_hypo is None:
                target_hypo = 0 if condition == 1 else 42
            target_hypo = int(target_hypo)
            target_df = df[df["k"] == target_hypo]
            if not target_df.empty:
                sns.scatterplot(data=target_df, x="Step", y="Prior", color="red", s=50, ax=ax)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Prior Probabilities for k by Subject",
            body,
            **kwargs,
        )

    # accuracy_model_alignment ------------------------------------------------

    def plot_accuracy_comparison(self, results, subjects=None, save_path=None, window_size=None, **kwargs):
        visible_results = self._filter_results(results, subjects)
        use_target_majority_plot = any(
            self._has_target_probability_data(info) for info in visible_results.values()
        )

        def body(ax, condition, iSub, info):
            use_target_majority = self._has_target_probability_data(info)
            if use_target_majority:
                computed = {}
                if window_size is not None:
                    computed = self.compute_target_majority_accuracy_metrics(info, window_size=window_size)
                true_acc = computed.get("sliding_target_majority_acc") if computed else info.get("sliding_target_majority_acc")
                pred_acc = (
                    computed.get("sliding_pred_target_majority_acc")
                    if computed
                    else info.get("sliding_pred_target_majority_acc")
                )
                pred_std = (
                    computed.get("sliding_pred_target_majority_acc_std")
                    if computed
                    else info.get("sliding_pred_target_majority_acc_std")
                )
                if true_acc is None or pred_acc is None or pred_std is None:
                    computed = self.compute_target_majority_accuracy_metrics(info, window_size=window_size)
                    true_acc = computed.get("sliding_target_majority_acc")
                    pred_acc = computed.get("sliding_pred_target_majority_acc")
                    pred_std = computed.get("sliding_pred_target_majority_acc_std")
                pred_label = "Model"
                true_label = "Participant"
                ylabel = "Higher-probability option"
                empty_text = "No target probability data"
            else:
                computed = self.compute_accuracy_metrics(info, window_size=window_size) if window_size is not None else {}
                true_acc = computed.get("sliding_true_acc") if computed else info.get("sliding_true_acc")
                pred_acc = computed.get("sliding_pred_acc") if computed else info.get("sliding_pred_acc")
                pred_std = computed.get("sliding_pred_acc_std") if computed else info.get("sliding_pred_acc_std")
                pred_label = "Predicted"
                true_label = "True"
                ylabel = "Accuracy"
                empty_text = "No accuracy data"

            if true_acc is None or pred_acc is None or pred_std is None:
                ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel=ylabel)
                return

            true_acc = np.asarray(true_acc, dtype=float)
            pred_acc = np.asarray(pred_acc, dtype=float)
            pred_std = np.asarray(pred_std, dtype=float)
            if not (len(true_acc) == len(pred_acc) == len(pred_std)) or len(pred_acc) == 0:
                ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel=ylabel)
                return

            win = window_size or info.get("window_size")
            try:
                win = int(win)
            except (TypeError, ValueError):
                win = 1

            trial = np.arange(win + 1, win + 1 + len(pred_acc))
            df = pd.DataFrame(
                {
                    "Trial": trial,
                    "Pred": pred_acc,
                    "True": true_acc,
                    "Low": pred_acc - pred_std,
                    "High": pred_acc + pred_std,
                }
            )
            sns.lineplot(data=df, x="Trial", y="Pred", label=pred_label, ax=ax)
            sns.lineplot(data=df, x="Trial", y="True", label=true_label, ax=ax)
            ax.fill_between(df["Trial"], df["Low"], df["High"], alpha=0.2)

            n_trials = info.get("n_trials")
            if n_trials:
                ax.set_xlim(1, n_trials)
            ax.set_ylim(0, 1)
            ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel=ylabel)
            ax.legend()

        title = (
            "Model vs Participant Higher-Probability Choice by Subject"
            if use_target_majority_plot
            else "Predicted vs True Accuracy by Subject"
        )
        self._plot_by_condition(
            results,
            subjects,
            save_path,
            title,
            body,
            **kwargs,
        )

    def plot_exponential_accuracy_comparison(
        self,
        results,
        subjects=None,
        save_path=None,
        window_size=None,
        exp_accuracy_alpha=None,
        **kwargs,
    ):
        override_alpha = self._validate_exp_accuracy_alpha(exp_accuracy_alpha)
        if override_alpha is None and window_size is not None:
            override_alpha = self._validate_exp_accuracy_alpha(2.0 / (float(window_size) + 1.0))
        visible_results = self._filter_results(results, subjects)
        use_target_majority_plot = any(
            self._has_target_probability_data(info) for info in visible_results.values()
        )

        def body(ax, condition, iSub, info):
            exp_override = self.compute_exponential_accuracy_metrics(info, exp_accuracy_alpha=override_alpha)
            exp_source = exp_override if override_alpha is not None else info
            if self._has_target_probability_data(info):
                true_acc = exp_source.get("exp_target_majority_acc")
                pred_acc = exp_source.get("exp_pred_target_majority_acc")
                true_label = "Participant"
                pred_label = "Model"
                ylabel = "Exp. smoothed higher-probability option"
                empty_text = "No exponential target accuracy data"
            else:
                true_acc = exp_source.get("exp_true_acc")
                pred_acc = exp_source.get("exp_pred_acc")
                true_label = "True"
                pred_label = "Predicted"
                ylabel = "Exp. smoothed accuracy"
                empty_text = "No exponential accuracy data"

            if true_acc is None or pred_acc is None:
                ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel=ylabel)
                return

            true_acc = np.asarray(true_acc, dtype=float)
            pred_acc = np.asarray(pred_acc, dtype=float)
            if true_acc.shape != pred_acc.shape or pred_acc.size == 0:
                ax.text(0.5, 0.5, empty_text, ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel=ylabel)
                return

            trial = np.arange(1, pred_acc.size + 1)
            df = pd.DataFrame(
                {
                    "Trial": trial,
                    "Pred": pred_acc,
                    "True": true_acc,
                }
            )
            sns.lineplot(data=df, x="Trial", y="Pred", label=pred_label, ax=ax)
            sns.lineplot(data=df, x="Trial", y="True", label=true_label, ax=ax)
            n_trials = info.get("n_trials")
            if n_trials:
                ax.set_xlim(1, n_trials)
            ax.set_ylim(0, 1)
            alpha = exp_source.get("exp_accuracy_alpha")
            try:
                alpha_val = float(alpha)
            except (TypeError, ValueError):
                alpha_val = float("nan")
            suffix = f", alpha={alpha_val:.3f}" if np.isfinite(alpha_val) else ""
            ax.set(title=f"Subject {iSub} (Condition {condition}{suffix})", xlabel="Trial", ylabel=ylabel)
            ax.legend()

        title = (
            "Exponential Model vs Participant Higher-Probability Choice by Subject"
            if use_target_majority_plot
            else "Exponential Predicted vs True Accuracy by Subject"
        )
        self._plot_by_condition(
            results,
            subjects,
            save_path,
            title,
            body,
            **kwargs,
        )

    def plot_accuracy_family_comparison(self, results, subjects=None, save_path=None, window_size=None, **kwargs):
        filtered_results = self._filter_results(results, subjects)
        family_results = {
            sid: info
            for sid, info in filtered_results.items()
            if int(info.get("condition", -1)) in (2, 3)
        }
        if not family_results:
            raise RuntimeError("No condition 2/3 results available for family accuracy comparison")

        def body(ax, condition, iSub, info):
            computed = {}
            if window_size is not None:
                computed = self.compute_family_accuracy_metrics(
                    info,
                    condition=condition,
                    window_size=window_size,
                    prediction_mode=kwargs.get("prediction_mode"),
                    default_beta=kwargs.get("default_beta", 10.0),
                    distance_mode=kwargs.get("distance_mode", "prototype"),
                )
            true_acc = computed.get("sliding_true_family_acc") if computed else info.get("sliding_true_family_acc")
            pred_acc = computed.get("sliding_pred_family_acc") if computed else info.get("sliding_pred_family_acc")
            pred_std = computed.get("sliding_pred_family_acc_std") if computed else info.get("sliding_pred_family_acc_std")
            if true_acc is None or pred_acc is None or pred_std is None:
                computed = self.compute_family_accuracy_metrics(
                    info,
                    condition=condition,
                    window_size=window_size,
                    prediction_mode=kwargs.get("prediction_mode"),
                    default_beta=kwargs.get("default_beta", 10.0),
                    distance_mode=kwargs.get("distance_mode", "prototype"),
                )
                true_acc = computed.get("sliding_true_family_acc")
                pred_acc = computed.get("sliding_pred_family_acc")
                pred_std = computed.get("sliding_pred_family_acc_std")

            if true_acc is None or pred_acc is None or pred_std is None:
                ax.text(0.5, 0.5, "No family accuracy data", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Family Accuracy")
                return

            true_acc = np.asarray(true_acc, dtype=float)
            pred_acc = np.asarray(pred_acc, dtype=float)
            pred_std = np.asarray(pred_std, dtype=float)
            if not (len(true_acc) == len(pred_acc) == len(pred_std)) or len(pred_acc) == 0:
                ax.text(0.5, 0.5, "No family accuracy data", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Family Accuracy")
                return

            win = window_size or info.get("window_size")
            try:
                win = int(win)
            except (TypeError, ValueError):
                win = 1

            trial = np.arange(win + 1, win + 1 + len(pred_acc))
            df = pd.DataFrame(
                {
                    "Trial": trial,
                    "Pred": pred_acc,
                    "True": true_acc,
                    "Low": pred_acc - pred_std,
                    "High": pred_acc + pred_std,
                }
            )
            sns.lineplot(data=df, x="Trial", y="Pred", label="Predicted", ax=ax)
            sns.lineplot(data=df, x="Trial", y="True", label="True", ax=ax)
            ax.fill_between(df["Trial"], df["Low"], df["High"], alpha=0.2)

            n_trials = info.get("n_trials")
            if n_trials:
                ax.set_xlim(1, n_trials)
            ax.set_ylim(0, 1)
            ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Family Accuracy")
            ax.legend()

        self._plot_by_condition(
            family_results,
            None,
            save_path,
            "Predicted vs True Family Accuracy by Subject",
            body,
            **kwargs,
        )

    def plot_target_majority_accuracy_comparison(self, results, subjects=None, save_path=None, window_size=None, **kwargs):
        def body(ax, condition, iSub, info):
            computed = self.compute_target_majority_accuracy_metrics(info, window_size=window_size) if window_size is not None else {}
            true_acc = computed.get("sliding_target_majority_acc") if computed else info.get("sliding_target_majority_acc")
            pred_acc = (
                computed.get("sliding_pred_target_majority_acc")
                if computed
                else info.get("sliding_pred_target_majority_acc")
            )
            pred_std = (
                computed.get("sliding_pred_target_majority_acc_std")
                if computed
                else info.get("sliding_pred_target_majority_acc_std")
            )
            if true_acc is None or pred_acc is None or pred_std is None:
                computed = self.compute_target_majority_accuracy_metrics(info, window_size=window_size)
                true_acc = computed.get("sliding_target_majority_acc")
                pred_acc = computed.get("sliding_pred_target_majority_acc")
                pred_std = computed.get("sliding_pred_target_majority_acc_std")

            if true_acc is None or pred_acc is None or pred_std is None:
                ax.text(0.5, 0.5, "No target probability data", ha="center", va="center", transform=ax.transAxes)
                ax.set(
                    title=f"Subject {iSub} (Condition {condition})",
                    xlabel="Trial",
                    ylabel="Target-majority accuracy",
                )
                return

            true_acc = np.asarray(true_acc, dtype=float)
            pred_acc = np.asarray(pred_acc, dtype=float)
            pred_std = np.asarray(pred_std, dtype=float)
            if not (len(true_acc) == len(pred_acc) == len(pred_std)) or len(pred_acc) == 0:
                ax.text(0.5, 0.5, "No target probability data", ha="center", va="center", transform=ax.transAxes)
                ax.set(
                    title=f"Subject {iSub} (Condition {condition})",
                    xlabel="Trial",
                    ylabel="Target-majority accuracy",
                )
                return

            win = window_size or info.get("window_size")
            try:
                win = int(win)
            except (TypeError, ValueError):
                win = 1

            trial = np.arange(win + 1, win + 1 + len(pred_acc))
            df = pd.DataFrame(
                {
                    "Trial": trial,
                    "Pred": pred_acc,
                    "Observed": true_acc,
                    "Low": pred_acc - pred_std,
                    "High": pred_acc + pred_std,
                }
            )
            sns.lineplot(data=df, x="Trial", y="Pred", label="Model", ax=ax)
            sns.lineplot(data=df, x="Trial", y="Observed", label="Participant", ax=ax)
            ax.fill_between(df["Trial"], df["Low"], df["High"], alpha=0.2)

            n_trials = info.get("n_trials")
            if n_trials:
                ax.set_xlim(1, n_trials)
            ax.set_ylim(0, 1)
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Higher-probability option",
            )
            ax.legend()

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Model vs Participant Higher-Probability Choice by Subject",
            body,
            **kwargs,
        )

    # choice_model_alignment --------------------------------------------------

    def compute_choice_brier_metrics(self, info, window_size=None):
        """Compute trial-level and sliding-window Brier loss for observed choices."""
        return choice_brier_curve_metrics_from_info(info, window_size=window_size)

    def plot_choice_brier(self, results, subjects=None, save_path=None, window_size=None, **kwargs):
        def body(ax, condition, iSub, info):
            metrics = self.compute_choice_brier_metrics(info, window_size=window_size)
            sliding = metrics.get("sliding_choice_brier")
            chance = metrics.get("choice_brier_chance")

            if sliding is None:
                ax.text(0.5, 0.5, "No choice Brier data", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Choice Brier")
                return

            sliding = np.asarray(sliding, dtype=float)
            finite = np.isfinite(sliding)
            if sliding.size == 0 or not np.any(finite):
                ax.text(0.5, 0.5, "No choice Brier data", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Choice Brier")
                return

            win = metrics.get("choice_brier_window_size")
            try:
                win = int(win)
            except (TypeError, ValueError):
                win = 1

            trial = np.arange(win + 1, win + 1 + len(sliding))
            df = pd.DataFrame({"Trial": trial, "Choice Brier": sliding})
            sns.lineplot(data=df, x="Trial", y="Choice Brier", label="Model", ax=ax)

            y_top_values = [float(np.nanmax(sliding[finite]))]
            if chance is not None and np.isfinite(chance):
                ax.axhline(float(chance), color="gray", linestyle="--", linewidth=1.0, label="Chance")
                y_top_values.append(float(chance))

            n_trials = info.get("n_trials")
            if n_trials:
                ax.set_xlim(1, n_trials)
            y_top = max(y_top_values)
            ax.set_ylim(0, min(2.05, max(0.05, y_top * 1.15)))
            ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Choice Brier")
            ax.legend()

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Choice Brier by Subject",
            body,
            **kwargs,
        )

    # beta_dynamics -----------------------------------------------------------

    def plot_beta_dynamics(self, results, subjects=None, save_path=None, max_hypotheses=20, **kwargs):
        def body(ax, condition, iSub, info):
            beta = self._extract_beta_log(info)
            ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Beta")
            if beta.size == 0:
                ax.text(0.5, 0.5, "No beta data", ha="center", va="center", transform=ax.transAxes)
                return

            if beta.ndim == 1:
                beta = beta.reshape(-1, 1)
            trials = np.arange(1, beta.shape[0] + 1)
            beta = np.asarray(beta, dtype=float)
            active_beta = np.where(beta > 0, beta, np.nan)
            finite_by_hypo = np.isfinite(active_beta)
            max_by_hypo = np.full(active_beta.shape[1], np.nan, dtype=float)
            has_hypo = np.any(finite_by_hypo, axis=0)
            if np.any(has_hypo):
                max_by_hypo[has_hypo] = np.max(
                    np.where(finite_by_hypo[:, has_hypo], active_beta[:, has_hypo], -np.inf),
                    axis=0,
                )
            valid_hypo = np.where(np.isfinite(max_by_hypo))[0]
            if valid_hypo.size == 0:
                ax.text(0.5, 0.5, "No active beta data", ha="center", va="center", transform=ax.transAxes)
                return

            order = valid_hypo[np.argsort(max_by_hypo[valid_hypo])[::-1]]
            if max_hypotheses is not None:
                order = order[: max(1, int(max_hypotheses))]
            for hypo in order:
                ax.plot(trials, beta[:, hypo], lw=0.8, alpha=0.35)

            finite_by_trial = np.isfinite(active_beta)
            counts_by_trial = np.sum(finite_by_trial, axis=1)
            mean_active = np.full(active_beta.shape[0], np.nan, dtype=float)
            valid_trials = counts_by_trial > 0
            if np.any(valid_trials):
                mean_active[valid_trials] = (
                    np.nansum(active_beta[valid_trials], axis=1) / counts_by_trial[valid_trials]
                )
            max_active = np.full(active_beta.shape[0], np.nan, dtype=float)
            if np.any(valid_trials):
                max_active[valid_trials] = np.max(
                    np.where(finite_by_trial[valid_trials], active_beta[valid_trials], -np.inf),
                    axis=1,
                )
            ax.plot(trials, mean_active, color="black", lw=2.0, label="Active mean")
            ax.plot(trials, max_active, color="red", lw=1.5, alpha=0.75, label="Active max")
            ax.set_xlim(1, beta.shape[0])
            finite_max = max_active[np.isfinite(max_active)]
            y_max = float(np.max(finite_max)) if finite_max.size else np.nan
            if np.isfinite(y_max) and y_max > 0:
                ax.set_ylim(0, y_max * 1.08)
            ax.legend()

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Model Beta Dynamics by Subject",
            body,
            **kwargs,
        )

    # oral_model_alignment ----------------------------------------------------
    # Implemented by OralModelAlignmentMixin:
    # Supporting utility: full oral mass display
    # - compute_oral_mass_probabilities
    # - plot_oral_mass_probabilities
    # Main family 1: distribution-based alignment
    # - compute_distribution_based_alignment
    # - plot_distribution_based_alignment_group
    # - plot_distribution_based_alignment_subjectwise
    # - save_distribution_based_alignment_outputs
    # Main family 2: oral-based alignment
    # - compute_oral_based_alignment
    # - plot_oral_based_alignment_group
    # - plot_oral_based_alignment_subjectwise
    # - save_oral_based_alignment_outputs
    # Main family 3: target-based alignment
    # - compute_target_based_alignment
    # - plot_target_based_alignment_group
    # - plot_target_based_alignment_subjectwise
    # - save_target_based_alignment_outputs
    # Main family 4: hit-based alignment
    # - compute_hit_based_alignment
    # - plot_hit_based_alignment_group
    # - plot_hit_based_alignment_subjectwise
    # - save_hit_based_alignment_outputs
    # Main family 5: coverage-based alignment
    # - compute_coverage_based_alignment
    # - plot_coverage_based_alignment_group
    # - plot_coverage_based_alignment_subjectwise
    # - save_coverage_based_alignment_outputs

    # trajectory_accuracy / trajectory_posterior -----------------------------

    @staticmethod
    def load_subject_payload(subject_json_path):
        path = Path(subject_json_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _subject_json_files(input_dir):
        input_dir = Path(input_dir)
        return sorted((input_dir / "subjects").glob("subject_*.json"))

    @staticmethod
    def _load_run_stream(payload, subject_json_path):
        subject_json_path = Path(subject_json_path)
        ref = payload.get("raw_runs_ref") or {}
        rel = ref.get("path")
        count = int(ref.get("count", 0) or 0)
        if not rel or count <= 0:
            raise ValueError(f"{subject_json_path} is missing a valid raw_runs_ref stream")

        stream_path = (subject_json_path.parent / rel).resolve()
        if not stream_path.exists():
            raise FileNotFoundError(f"Run stream file not found: {stream_path}")
        return StreamList(str(stream_path), count)

    def rank_runs_by_error(self, subject_json_path):
        payload = self.load_subject_payload(subject_json_path)
        ref = payload.get("raw_runs_ref") or {}
        stream_count = int(ref.get("count", 0) or 0)

        summary = payload.get("simulation") or payload.get("simulation_summary") or {}
        sample_errors = summary.get("sample_errors")
        if isinstance(sample_errors, list) and sample_errors:
            rows = [
                {
                    "stream_index": int(idx),
                    "run_index": int(idx),
                    "error": float(err),
                }
                for idx, err in enumerate(sample_errors)
                if stream_count <= 0 or idx < stream_count
            ]
            if rows:
                df = pd.DataFrame(rows).sort_values("error", ascending=True).reset_index(drop=True)
                df["rank"] = np.arange(1, len(df) + 1)
                return df[["rank", "stream_index", "run_index", "error"]]

        stream = self._load_run_stream(payload, subject_json_path)

        rows = []
        for stream_index, run_obj in enumerate(stream):
            if not isinstance(run_obj, dict):
                continue
            err = run_obj.get("mean_error")
            if err is None:
                metrics = run_obj.get("metrics") or {}
                err = metrics.get("mean_error")
            if err is None:
                continue
            run_index = run_obj.get("run_index", stream_index)
            rows.append(
                {
                    "stream_index": int(stream_index),
                    "run_index": int(run_index),
                    "error": float(err),
                }
            )

        if not rows:
            raise ValueError(f"No run objects with mean_error found in {subject_json_path}")

        df = pd.DataFrame(rows).sort_values("error", ascending=True).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df[["rank", "stream_index", "run_index", "error"]]

    @staticmethod
    def _extract_run_eval_curves(run_obj, eval_prediction_mode=None):
        metrics = run_obj.get("metrics")
        if not isinstance(metrics, dict):
            metrics_by_mode = run_obj.get("metrics_by_mode")
            mode = eval_prediction_mode or run_obj.get("selection_prediction_mode")
            if mode is None and isinstance(metrics_by_mode, dict) and len(metrics_by_mode) == 1:
                mode = next(iter(metrics_by_mode))
            if isinstance(metrics_by_mode, dict) and mode in metrics_by_mode:
                metrics = metrics_by_mode[mode]
            else:
                available = sorted(metrics_by_mode) if isinstance(metrics_by_mode, dict) else []
                raise KeyError(
                    "Run object is missing metrics and no usable metrics_by_mode entry was found. "
                    f"eval_prediction_mode={mode!r}, available={available}"
                )

        true_acc = ModelEval._as_float_1d(metrics.get("sliding_true_acc"), "sliding_true_acc")
        pred_acc = ModelEval._as_float_1d(metrics.get("sliding_pred_acc"), "sliding_pred_acc")
        pred_std = ModelEval._as_float_1d(metrics.get("sliding_pred_acc_std"), "sliding_pred_acc_std")
        if not (len(true_acc) == len(pred_acc) == len(pred_std)):
            raise ValueError(
                "Metric length mismatch in run object: "
                f"true={len(true_acc)}, pred={len(pred_acc)}, std={len(pred_std)}"
            )
        return true_acc, pred_acc, pred_std

    @staticmethod
    def _load_selected_runs(stream, stream_indices):
        wanted = {int(idx) for idx in stream_indices}
        found = {}
        for stream_index, run_obj in enumerate(stream):
            if stream_index in wanted:
                found[int(stream_index)] = run_obj
                if len(found) == len(wanted):
                    break
        return found

    @staticmethod
    def _extract_run_metrics(run_obj: Mapping[str, Any], eval_prediction_mode=None):
        metrics = run_obj.get("metrics")
        if isinstance(metrics, Mapping):
            return metrics
        metrics_by_mode = run_obj.get("metrics_by_mode")
        mode = eval_prediction_mode or run_obj.get("selection_prediction_mode")
        if mode is None and isinstance(metrics_by_mode, Mapping) and len(metrics_by_mode) == 1:
            mode = next(iter(metrics_by_mode))
        if isinstance(metrics_by_mode, Mapping) and mode in metrics_by_mode:
            metrics = metrics_by_mode[mode]
            if isinstance(metrics, Mapping):
                return metrics
        return {}

    @staticmethod
    def _accuracy_curve_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
        shared = accuracy_curve_metrics(metrics)
        return {
            "acc_mae": shared["acc_mae"],
            "acc_rmse": shared["acc_rmse"],
            "true_vol": shared["true_vol"],
            "model_vol": shared["pred_vol"],
            "vol_ratio": shared["vol_ratio"],
            "slope_agree": shared["slope_agree"],
        }

    @staticmethod
    def _history_kernel_summary(metrics: Mapping[str, Any], max_lag=8) -> dict[str, Any]:
        shared = history_kernel_metrics(
            metrics,
            max_lag=int(max_lag),
            ridge=1e-3,
            standardize=True,
        )
        return {
            "history_corr": shared["kernel_corr"],
            "history_mse": shared["kernel_mse"],
            "human_kernel": shared["human_kernel"],
            "model_kernel": shared["model_kernel"],
        }

    @staticmethod
    def _switch_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
        shared = switch_behavior_metrics(metrics, min_trials=1)
        return {
            key: shared[key]
            for key in (
                "switch_human",
                "switch_model",
                "switch_abs_diff",
                "win_stay_abs_diff",
                "lose_shift_abs_diff",
                "switch_score",
            )
        }

    def _run_behavior_row(self, payload, subject_json_path, stream_index, run_obj, eval_prediction_mode=None):
        metrics = self._extract_run_metrics(run_obj, eval_prediction_mode=eval_prediction_mode)
        if not metrics:
            return None
        curve = self._accuracy_curve_summary(metrics)
        history = self._history_kernel_summary(metrics)
        switch = self._switch_summary(metrics)
        return {
            "subject_id": int(payload.get("subject_id", -1)),
            "condition": int(payload.get("condition", -1)),
            "stream_index": int(stream_index),
            "run_index": int(run_obj.get("run_index", stream_index)),
            "choice_error": self._safe_float(run_obj.get("mean_error")),
            "subject_json": str(subject_json_path),
            **curve,
            "history_corr": history["history_corr"],
            "history_mse": history["history_mse"],
            **switch,
        }

    def collect_behavior_ppc_rows(
        self,
        input_dir,
        eval_prediction_mode=None,
        max_runs_per_subject=None,
        subjects=None,
    ):
        input_dir = Path(input_dir)
        subject_set = {int(subject) for subject in subjects} if subjects is not None else None
        rows = []
        for subject_json in self._subject_json_files(input_dir):
            payload = self.load_subject_payload(subject_json)
            sid = int(payload.get("subject_id", subject_json.stem.replace("subject_", "-1")))
            if subject_set is not None and sid not in subject_set:
                continue
            stream = self._load_run_stream(payload, subject_json)
            for stream_index, run_obj in enumerate(stream):
                if max_runs_per_subject is not None and stream_index >= int(max_runs_per_subject):
                    break
                if not isinstance(run_obj, Mapping):
                    continue
                row = self._run_behavior_row(
                    payload,
                    subject_json,
                    stream_index,
                    run_obj,
                    eval_prediction_mode=eval_prediction_mode,
                )
                if row is not None:
                    rows.append(row)
        return pd.DataFrame(rows)

    def _predictive_accuracy_band_data(
        self,
        subject_json_path,
        *,
        eval_prediction_mode=None,
        max_runs=None,
    ):
        subject_json_path = Path(subject_json_path)
        payload = self.load_subject_payload(subject_json_path)
        stream = self._load_run_stream(payload, subject_json_path)
        pred_curves = []
        true_curve = None
        best_curve = None
        best_error = np.inf
        best_run_index = None
        for stream_index, run_obj in enumerate(stream):
            if max_runs is not None and stream_index >= int(max_runs):
                break
            if not isinstance(run_obj, Mapping):
                continue
            try:
                true_acc, pred_acc, _ = self._extract_run_eval_curves(
                    run_obj,
                    eval_prediction_mode=eval_prediction_mode,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if true_curve is None:
                true_curve = np.asarray(true_acc, dtype=float)
            if len(pred_acc) == len(true_curve):
                pred_curve = np.asarray(pred_acc, dtype=float)
                pred_curves.append(pred_curve)
                run_error = self._safe_float(run_obj.get("mean_error"), default=np.inf)
                if run_error < best_error:
                    best_error = run_error
                    best_curve = pred_curve
                    best_run_index = int(run_obj.get("run_index", stream_index))
        if true_curve is None or not pred_curves:
            raise ValueError(f"No usable run accuracy curves found in {subject_json_path}")
        pred_stack = np.vstack(pred_curves)
        band_metrics = predictive_accuracy_band_metrics(pred_stack, true_curve)

        representative = payload.get("representative_run") or {}
        representative_metrics = representative.get("metrics_by_mode") or {}
        selection = payload.get("selection") or {}
        mode = eval_prediction_mode or selection.get("selection_prediction_mode")
        rep_pred = None
        if isinstance(representative_metrics, Mapping) and mode in representative_metrics:
            try:
                rep_pred = self._as_float_1d(representative_metrics[mode].get("sliding_pred_acc"), "rep_pred")
            except (TypeError, ValueError):
                rep_pred = None
        if best_curve is None and rep_pred is not None and len(rep_pred) == len(true_curve):
            best_curve = rep_pred
            best_run_index = selection.get("representative_run_index")
            best_error = self._safe_float((payload.get("simulation") or {}).get("best_error"))

        summary = payload.get("simulation") or payload.get("simulation_summary") or {}
        win = int(summary.get("window_size") or (selection.get("selection_meta") or {}).get("window_size") or 1)
        x = np.arange(win + 1, win + 1 + len(true_curve))
        sid = int(payload.get("subject_id", -1))
        condition = int(payload.get("condition", -1))

        return {
            "subject_id": sid,
            "condition": condition,
            "n_runs": int(len(pred_curves)),
            "x": x,
            "true_curve": true_curve,
            "representative_curve": rep_pred,
            "best_curve": best_curve,
            "best_run_index": best_run_index,
            "best_error": float(best_error) if np.isfinite(best_error) else np.nan,
            **band_metrics,
        }

    @staticmethod
    def _draw_predictive_accuracy_band(ax, band, *, show_legend=True, compact_title=False):
        x = band["x"]
        ax.fill_between(x, band["q00"], band["q100"], color="#dce6f2", alpha=0.45, label="Model 100% band")
        ax.fill_between(x, band["q05"], band["q95"], color="#9db9d8", alpha=0.45, label="Model 90% band")
        ax.fill_between(x, band["q25"], band["q75"], color="#4f81b8", alpha=0.45, label="Model 50% band")
        best_curve = band.get("best_curve")
        if best_curve is not None and len(best_curve) == len(x):
            ax.plot(x, best_curve, color="#E69F00", lw=2.0, alpha=0.95, label="Best run")
        ax.plot(x, band["true_curve"], color="#111111", lw=2.4, label="Subject")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Accuracy")
        if compact_title:
            ax.set_title(f"S{band['subject_id']} | n={band['n_runs']}")
        else:
            ax.set_title(
                f"Predictive Accuracy Band | Subject {band['subject_id']} | "
                f"n={band['n_runs']} runs"
            )
        ax.grid(alpha=0.25)
        if show_legend:
            ax.legend(loc="best")

    def plot_predictive_accuracy_band(
        self,
        subject_json_path,
        *,
        save_path=None,
        eval_prediction_mode=None,
        max_runs=None,
    ):
        band = self._predictive_accuracy_band_data(
            subject_json_path,
            eval_prediction_mode=eval_prediction_mode,
            max_runs=max_runs,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        self._draw_predictive_accuracy_band(ax, band, show_legend=True)
        fig.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return {
            "subject_id": band["subject_id"],
            "n_runs": band["n_runs"],
            "median_curve_mae": band["median_curve_mae"],
            "coverage_50": band["coverage_50"],
            "coverage_90": band["coverage_90"],
            "median_vol_ratio": band["median_vol_ratio"],
            "plot_path": str(save_path) if save_path else "",
        }

    def plot_predictive_accuracy_band_group(
        self,
        input_dir,
        save_path=None,
        *,
        eval_prediction_mode=None,
        max_runs_per_subject=None,
        subjects=None,
        n_cols=None,
        max_subjects_per_row=8,
    ):
        input_dir = Path(input_dir)
        subject_set = {int(subject) for subject in subjects} if subjects is not None else None
        bands = []
        for subject_json in self._subject_json_files(input_dir):
            sid = int(subject_json.stem.replace("subject_", ""))
            if subject_set is not None and sid not in subject_set:
                continue
            try:
                bands.append(
                    self._predictive_accuracy_band_data(
                        subject_json,
                        eval_prediction_mode=eval_prediction_mode,
                        max_runs=max_runs_per_subject,
                    )
                )
            except (ValueError, FileNotFoundError, KeyError) as exc:
                logger.warning("Skipping predictive band for %s: %s", subject_json, exc)

        if not bands:
            raise ValueError(f"No usable predictive accuracy bands found in {input_dir}")

        bands = sorted(bands, key=lambda row: (row["condition"], row["subject_id"]))
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for band in bands:
            grouped[int(band["condition"])].append(band)
        layout_kwargs = {"max_subjects_per_row": max_subjects_per_row}
        if n_cols is not None:
            layout_kwargs["n_cols"] = int(n_cols)
        n_rows, n_cols, rows_by_condition = self._layout_by_condition(grouped, layout_kwargs)

        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        row_offset = 0
        legend_drawn = False
        for condition, condition_bands in sorted(grouped.items()):
            for idx, band in enumerate(condition_bands):
                local_row = idx // n_cols
                col = idx % n_cols
                ax = fig.add_subplot(n_rows, n_cols, (row_offset + local_row) * n_cols + col + 1)
                show_legend = not legend_drawn
                self._draw_predictive_accuracy_band(
                    ax,
                    band,
                    show_legend=show_legend,
                    compact_title=False,
                )
                legend_drawn = legend_drawn or show_legend
            row_offset += rows_by_condition[condition]

        used_axes = sum(len(items) for items in grouped.values())
        for idx in range(used_axes, n_rows * n_cols):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            ax.axis("off")

        fig.suptitle("Predictive Accuracy Band by Subject", fontsize=16, y=0.99)
        fig.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)

        return pd.DataFrame(
            [
                {
                    "subject_id": band["subject_id"],
                    "condition": band["condition"],
                    "n_runs": band["n_runs"],
                    "median_curve_mae": band["median_curve_mae"],
                    "coverage_50": band["coverage_50"],
                    "coverage_90": band["coverage_90"],
                    "median_vol_ratio": band["median_vol_ratio"],
                    "plot_path": str(save_path) if save_path else "",
                }
                for band in bands
            ]
        )

    def save_predictive_accuracy_bands(
        self,
        input_dir,
        output_dir,
        *,
        eval_prediction_mode=None,
        max_runs_per_subject=None,
        subjects=None,
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subject_set = {int(subject) for subject in subjects} if subjects is not None else None
        rows = []
        for subject_json in self._subject_json_files(input_dir):
            sid = int(subject_json.stem.replace("subject_", ""))
            if subject_set is not None and sid not in subject_set:
                continue
            save_path = output_dir / f"subject_{sid}_predictive_accuracy_band.png"
            try:
                rows.append(
                    self.plot_predictive_accuracy_band(
                        subject_json,
                        save_path=save_path,
                        eval_prediction_mode=eval_prediction_mode,
                        max_runs=max_runs_per_subject,
                    )
                )
            except (ValueError, FileNotFoundError, KeyError) as exc:
                logger.warning("Skipping predictive band for %s: %s", subject_json, exc)
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(output_dir / "predictive_accuracy_band_summary.csv", index=False)
        return df

    def save_behavior_ppc_outputs(
        self,
        input_dir,
        output_dir,
        *,
        eval_prediction_mode=None,
        max_runs_per_subject=None,
        subjects=None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_df = self.collect_behavior_ppc_rows(
            input_dir,
            eval_prediction_mode=eval_prediction_mode,
            max_runs_per_subject=max_runs_per_subject,
            subjects=subjects,
        )
        if run_df.empty:
            raise ValueError("No raw-run behavior metrics available")
        run_df.to_csv(output_dir / "behavior_ppc_run_metrics.csv", index=False)

        summary_rows = []
        for sid, group in run_df.groupby("subject_id", sort=True):
            aggregated = behavior_ppc_group_metrics(group)
            summary_rows.append(
                {
                    "subject_id": int(sid),
                    "n_runs": int(len(group)),
                    **aggregated,
                    "acc_mae_pass": bool(aggregated["acc_mae_mean"] <= 0.10),
                    "vol_ratio_pass": bool(
                        0.60 <= aggregated["vol_ratio_median"] <= 1.50
                    ),
                    "history_corr_pass": bool(
                        aggregated["history_corr_mean"] >= 0.80
                    ),
                    "switch_score_pass": bool(
                        aggregated["switch_score_mean"] <= 0.10
                    ),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "behavior_ppc_subject_summary.csv", index=False)

        self._plot_ppc_distribution(
            run_df,
            y="acc_mae",
            threshold=0.10,
            output_path=output_dir / "accuracy_mae_ppc.png",
            ylabel="Accuracy Curve MAE",
            title="Accuracy MAE PPC",
            threshold_label="target <= 0.10",
        )
        self._plot_ppc_distribution(
            run_df,
            y="vol_ratio",
            threshold=(0.60, 1.50),
            output_path=output_dir / "volatility_ppc.png",
            ylabel="Model / Human Volatility Ratio",
            title="Volatility PPC",
            threshold_label="target range [0.60, 1.50]",
        )
        self._plot_ppc_distribution(
            run_df,
            y="history_corr",
            threshold=0.80,
            output_path=output_dir / "history_kernel_ppc.png",
            ylabel="History Kernel Correlation",
            title="Feedback-History Kernel PPC",
            threshold_label="target >= 0.80",
        )
        self._plot_ppc_distribution(
            run_df,
            y="switch_score",
            threshold=0.10,
            output_path=output_dir / "switch_ppc.png",
            ylabel="Switch Profile Score",
            title="Switch / Perseveration PPC",
            threshold_label="target <= 0.10",
        )
        return {
            "run_metrics": run_df,
            "subject_summary": summary_df,
        }

    @staticmethod
    def _plot_ppc_distribution(df, *, y, threshold, output_path, ylabel, title, threshold_label):
        plot_df = df[["subject_id", y]].copy()
        plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df[y])]
        if plot_df.empty:
            raise ValueError(f"No finite values for {y}")
        plot_df["subject_id"] = plot_df["subject_id"].astype(str)
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * plot_df["subject_id"].nunique()), 5))
        sns.violinplot(data=plot_df, x="subject_id", y=y, inner="quartile", cut=0, ax=ax, color="#d8e2dc")
        sns.pointplot(
            data=plot_df,
            x="subject_id",
            y=y,
            estimator=np.mean,
            errorbar=None,
            color="#1b4965",
            markers="D",
            linestyles="",
            ax=ax,
        )
        if isinstance(threshold, tuple):
            ax.axhspan(float(threshold[0]), float(threshold[1]), color="#81b29a", alpha=0.18, label=threshold_label)
        else:
            ax.axhline(float(threshold), color="#d1495b", lw=1.5, ls="--", label=threshold_label)
        ax.set_title(title)
        ax.set_xlabel("Subject")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _extract_run_state_log(run_obj, key):
        state_log = run_obj.get("state_log") or {}
        values = state_log.get(key)
        if values is None:
            return []
        return values

    def _plot_run_posterior_body(
        self,
        ax,
        run_obj,
        condition,
        rank,
        run_idx,
        err,
        limit=True,
        target_hypothesis=None,
    ):
        posterior_log = self._extract_run_state_log(run_obj, "posterior")
        ax.set(xlabel="Trial", ylabel="Posterior Probability")

        if not posterior_log:
            ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
            return

        if limit:
            max_k = 19 if int(condition) == 1 else 116
        else:
            max_k = max((len(posterior) for posterior in posterior_log), default=0)

        data = []
        for step, posterior in enumerate(posterior_log):
            try:
                posterior = np.asarray(posterior, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            for k in range(min(max_k, posterior.size)):
                value = posterior[k]
                if not np.isfinite(value) or value <= 1e-12:
                    continue
                data.append({"Step": step + 1, "k": k, "Posterior": float(value)})

        df = pd.DataFrame(data)
        if df.empty:
            ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
            return

        sns.scatterplot(
            data=df,
            x="Step",
            y="Posterior",
            hue="k",
            hue_order=list(range(max_k)),
            palette=self._hypothesis_color_map(max_k),
            alpha=0.5,
            legend=False,
            ax=ax,
            s=16,
        )

        target_hypo = target_hypothesis
        if target_hypo is None:
            target_hypo = 0 if int(condition) == 1 else 42
        target_hypo = int(target_hypo)
        target_df = df[df["k"] == target_hypo]
        if not target_df.empty:
            sns.scatterplot(data=target_df, x="Step", y="Posterior", color="red", s=28, legend=False, ax=ax)

        ax.set_title(f"Rank {int(rank)} | Run #{int(run_idx)} | Error={float(err):.6f}", fontsize=9)

    def plot_runs_by_rank(
        self,
        subject_json_path,
        ranks: int | Sequence[int],
        *,
        n_cols=4,
        save_path=None,
        eval_prediction_mode=None,
    ):
        subject_json_path = Path(subject_json_path)
        payload = self.load_subject_payload(subject_json_path)
        rank_df = self.rank_runs_by_error(subject_json_path)
        stream = self._load_run_stream(payload, subject_json_path)

        if isinstance(ranks, int):
            ranks = [ranks]
        ranks = [int(rank) for rank in ranks]
        if not ranks:
            raise ValueError("Please provide at least one rank")

        selected = rank_df[rank_df["rank"].isin(ranks)].copy()
        if selected.empty:
            raise ValueError(f"None of the requested ranks exist for {subject_json_path}")
        selected = selected.sort_values("rank").reset_index(drop=True)
        run_by_stream_index = self._load_selected_runs(stream, selected["stream_index"].tolist())

        n = len(selected)
        n_cols = max(1, int(n_cols))
        n_rows = int(math.ceil(n / n_cols))
        sid = int(payload.get("subject_id", -1))
        cond = int(payload.get("condition", -1))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.ravel()
        for i, row in selected.iterrows():
            ax = axes_flat[i]
            stream_idx = int(row["stream_index"])
            run_idx = int(row["run_index"])
            err = float(row["error"])
            run_obj = run_by_stream_index.get(stream_idx)
            if run_obj is None:
                ax.text(0.5, 0.5, "Run missing", ha="center", va="center", transform=ax.transAxes)
                continue

            true_acc, pred_acc, pred_std = self._extract_run_eval_curves(
                run_obj,
                eval_prediction_mode=eval_prediction_mode,
            )
            win = run_obj.get("window_size") or payload.get("window_size")
            try:
                win = int(win)
            except (TypeError, ValueError):
                win = 1
            x = np.arange(win + 1, win + 1 + len(pred_acc))

            ax.plot(x, pred_acc, label="Predicted", lw=2)
            ax.plot(x, true_acc, label="True", lw=2)
            ax.fill_between(x, pred_acc - pred_std, pred_acc + pred_std, alpha=0.2)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Trial")
            ax.set_ylabel("Accuracy")
            ax.set_title(
                f"Sub {sid} | Cond {cond} | Rank {int(row['rank'])} | "
                f"Run #{run_idx} | Error={err:.6f}"
            )
            ax.grid(alpha=0.25)
            ax.legend(loc="best")

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            f"Predicted vs True Accuracy by Run (Subject {sid}, Condition {cond})",
            fontsize=14,
            y=1.02,
        )
        fig.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Trajectory plot saved to %s", save_path)
            plt.close(fig)
        return selected

    def plot_trajectory_analysis(
        self,
        input_dir,
        output_dir,
        ranks=None,
        n_cols=4,
        eval_prediction_mode=None,
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ranks = self.DEFAULT_TRAJECTORY_RANKS if ranks is None else ranks

        summaries = []
        for subject_json in self._subject_json_files(input_dir):
            try:
                payload = self.load_subject_payload(subject_json)
                sid = int(payload.get("subject_id", subject_json.stem.replace("subject_", "-1")))
                out_path = output_dir / f"subject_{sid}_trajectory_accuracy.png"
                selected = self.plot_runs_by_rank(
                    subject_json,
                    ranks=ranks,
                    n_cols=n_cols,
                    save_path=out_path,
                    eval_prediction_mode=eval_prediction_mode,
                )
                selected = selected.copy()
                selected["subject_id"] = sid
                selected["plot_path"] = str(out_path)
                summaries.append(selected)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                logger.warning("Skipping trajectory plot for %s: %s", subject_json, exc)

        if not summaries:
            return pd.DataFrame()
        summary = pd.concat(summaries, ignore_index=True)
        summary_path = output_dir / "trajectory_rank_summary.csv"
        summary.to_csv(summary_path, index=False)
        return summary

    def plot_run_posteriors_by_rank(
        self,
        subject_json_path,
        ranks: int | Sequence[int] = DEFAULT_TOP16_RANKS,
        *,
        n_cols=4,
        save_path=None,
        limit=True,
        target_hypothesis=None,
    ):
        subject_json_path = Path(subject_json_path)
        payload = self.load_subject_payload(subject_json_path)
        rank_df = self.rank_runs_by_error(subject_json_path)
        stream = self._load_run_stream(payload, subject_json_path)

        if isinstance(ranks, int):
            ranks = [ranks]
        ranks = [int(rank) for rank in ranks]
        if not ranks:
            raise ValueError("Please provide at least one rank")

        selected = rank_df[rank_df["rank"].isin(ranks)].copy()
        if selected.empty:
            raise ValueError(f"None of the requested ranks exist for {subject_json_path}")
        selected = selected.sort_values("rank").reset_index(drop=True)
        run_by_stream_index = self._load_selected_runs(stream, selected["stream_index"].tolist())

        n = len(selected)
        n_cols = max(1, int(n_cols))
        n_rows = int(math.ceil(n / n_cols))
        sid = int(payload.get("subject_id", -1))
        cond = int(payload.get("condition", -1))
        target_hypothesis = payload.get("target_hypothesis", target_hypothesis)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.ravel()
        for i, row in selected.iterrows():
            ax = axes_flat[i]
            stream_idx = int(row["stream_index"])
            run_idx = int(row["run_index"])
            err = float(row["error"])
            run_obj = run_by_stream_index.get(stream_idx)
            if run_obj is None:
                ax.text(0.5, 0.5, "Run missing", ha="center", va="center", transform=ax.transAxes)
                continue
            self._plot_run_posterior_body(
                ax=ax,
                run_obj=run_obj,
                condition=cond,
                rank=int(row["rank"]),
                run_idx=run_idx,
                err=err,
                limit=limit,
                target_hypothesis=target_hypothesis,
            )

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            f"Posterior Probabilities for Top-16 Runs (Subject {sid}, Condition {cond})",
            fontsize=14,
            y=1.02,
        )
        fig.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Top-16 posterior trajectory plot saved to %s", save_path)
            plt.close(fig)
        return selected

    def plot_trajectory_posteriors(
        self,
        input_dir,
        output_dir,
        ranks=None,
        n_cols=4,
        limit=True,
        target_hypotheses_by_condition=None,
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ranks = self.DEFAULT_TOP16_RANKS if ranks is None else ranks

        summaries = []
        for subject_json in self._subject_json_files(input_dir):
            try:
                payload = self.load_subject_payload(subject_json)
                sid = int(payload.get("subject_id", subject_json.stem.replace("subject_", "-1")))
                condition = int(payload.get("condition", -1))
                target_hypothesis = None
                if isinstance(target_hypotheses_by_condition, dict):
                    target_hypothesis = target_hypotheses_by_condition.get(condition)
                out_path = output_dir / f"subject_{sid}_top16_posterior.png"
                selected = self.plot_run_posteriors_by_rank(
                    subject_json,
                    ranks=ranks,
                    n_cols=n_cols,
                    save_path=out_path,
                    limit=limit,
                    target_hypothesis=target_hypothesis,
                )
                selected = selected.copy()
                selected["subject_id"] = sid
                selected["plot_path"] = str(out_path)
                summaries.append(selected)
            except (ValueError, FileNotFoundError, KeyError) as exc:
                logger.warning("Skipping posterior trajectory plot for %s: %s", subject_json, exc)

        if not summaries:
            return pd.DataFrame()
        summary = pd.concat(summaries, ignore_index=True)
        summary_path = output_dir / "trajectory_posterior_rank_summary.csv"
        summary.to_csv(summary_path, index=False)
        return summary

    # dynamic strategy controller --------------------------------------------

    @classmethod
    def _policy_order(cls, policies: Sequence[str]) -> list[str]:
        unique = list(dict.fromkeys(str(policy) for policy in policies))
        canonical = [policy for policy in cls.PROFILE_POLICY_ORDER if policy in unique]
        return canonical + sorted(policy for policy in unique if policy not in canonical)

    @classmethod
    def _profile_activation_data(cls, info):
        logs = info.get("strategy_counts_log") or []
        probability_rows = []
        selected = []
        trials = []
        all_policies = []
        for trial_idx, step in enumerate(logs, start=1):
            if not isinstance(step, Mapping):
                continue
            probabilities = step.get("policy_probabilities")
            if not isinstance(probabilities, Mapping) or not probabilities:
                controller = step.get("state_controller") or {}
                probabilities = controller.get("policy_probabilities")
            if not isinstance(probabilities, Mapping) or not probabilities:
                probabilities = step.get("state_probabilities")
            if not isinstance(probabilities, Mapping) or not probabilities:
                continue

            clean = {}
            for key, value in probabilities.items():
                prob = cls._safe_float(value, default=np.nan)
                if np.isfinite(prob) and prob >= 0.0:
                    clean[str(key)] = float(prob)
            total = float(sum(clean.values()))
            if total <= 0.0:
                continue
            clean = {key: value / total for key, value in clean.items()}
            policy = step.get("selected_policy_method")
            if policy is None:
                policy = (step.get("profile_policy") or {}).get("policy_method")
            if policy is None:
                policy = step.get("selected_state")

            trials.append(int(trial_idx))
            probability_rows.append(clean)
            selected.append(str(policy) if policy is not None else "")
            all_policies.extend(clean)

        if not probability_rows:
            return None
        policies = cls._policy_order(all_policies)
        matrix = np.asarray(
            [[row.get(policy, 0.0) for policy in policies] for row in probability_rows],
            dtype=float,
        )
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0.0)
        return {
            "trial": np.asarray(trials, dtype=int),
            "policies": policies,
            "probabilities": matrix,
            "selected": selected,
        }

    def plot_dynamic_strategy_profile(
        self,
        results,
        window_size=None,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot controller probabilities, sampled policy, and representative accuracy."""

        def body(ax, condition, iSub, info):
            activation = self._profile_activation_data(info)
            if activation is None:
                ax.text(0.5, 0.5, "No dynamic strategy log", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})")
                return

            x = activation["trial"]
            policies = activation["policies"]
            probabilities = activation["probabilities"]
            colors = [
                self.PROFILE_POLICY_COLORS.get(policy, sns.color_palette("colorblind", len(policies))[idx])
                for idx, policy in enumerate(policies)
            ]
            ax.stackplot(
                x,
                probabilities.T,
                labels=[f"P({policy})" for policy in policies],
                colors=colors,
                alpha=0.24,
                linewidth=0,
                zorder=1,
            )

            cumulative = np.cumsum(probabilities, axis=1)
            selected_y = []
            selected_colors = []
            for row_idx, policy in enumerate(activation["selected"]):
                if policy not in policies:
                    selected_y.append(np.nan)
                    selected_colors.append("#444444")
                    continue
                policy_idx = policies.index(policy)
                lower = cumulative[row_idx, policy_idx - 1] if policy_idx > 0 else 0.0
                selected_y.append(lower + probabilities[row_idx, policy_idx] / 2.0)
                selected_colors.append(colors[policy_idx])
            ax.scatter(
                x,
                selected_y,
                c=selected_colors,
                marker="|",
                s=42,
                linewidths=1.2,
                alpha=0.95,
                zorder=3,
                label="Activated policy",
            )

            computed = self.compute_accuracy_metrics(info, window_size=window_size)
            true_acc = computed.get("sliding_true_acc")
            pred_acc = computed.get("sliding_pred_acc")
            win = int(computed.get("window_size", window_size or info.get("window_size") or 1))
            if true_acc is not None and pred_acc is not None:
                true_acc = np.asarray(true_acc, dtype=float)
                pred_acc = np.asarray(pred_acc, dtype=float)
                acc_x = np.arange(win + 1, win + 1 + min(len(true_acc), len(pred_acc)))
                ax.plot(acc_x, pred_acc[: len(acc_x)], color="#E69F00", lw=2.1, label="Best run", zorder=5)
                ax.plot(acc_x, true_acc[: len(acc_x)], color="#111111", lw=2.4, label="Subject", zorder=6)

            ax.set_xlim(1, max(int(x[-1]), int(info.get("n_trials") or x[-1])))
            ax.set_ylim(0, 1)
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Accuracy / activation probability",
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=8, ncol=4, loc="upper center")
            ax.grid(axis="x", alpha=0.18)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Dynamic Strategy Probabilities and Accuracy",
            body,
            **kwargs,
        )

    @staticmethod
    def _active_set_count_rows(info):
        rows = []
        for trial_idx, step in enumerate(info.get("strategy_counts_log") or [], start=1):
            if not isinstance(step, Mapping):
                continue
            policy_log = step.get("profile_policy")
            if isinstance(policy_log, Mapping):
                retained = ModelEval._safe_float(policy_log.get("retained_count"), default=0.0)
                newcomer = ModelEval._safe_float(policy_log.get("newcomer_count"), default=0.0)
            else:
                retained = 0.0
                newcomer = 0.0
                for strategy in step.get("strategies") or []:
                    if not isinstance(strategy, Mapping):
                        continue
                    count = ModelEval._safe_float(strategy.get("selected_count"), default=0.0)
                    if strategy.get("pool") == "active":
                        retained += count
                    elif strategy.get("pool") == "inactive":
                        newcomer += count
            total = ModelEval._safe_float(step.get("active_total"), default=retained + newcomer)
            rows.append(
                {
                    "trial": int(trial_idx),
                    "retained": float(retained),
                    "newcomer": float(newcomer),
                    "total": float(total),
                }
            )
        return pd.DataFrame(rows)

    def plot_hypothesis_active_set_counts(
        self,
        results,
        window_size=None,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot rolling retained, newcomer, and active-hypothesis counts."""

        def body(ax, condition, iSub, info):
            counts = self._active_set_count_rows(info)
            if counts.empty:
                ax.text(0.5, 0.5, "No active-set count log", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})")
                return
            win = max(1, int(window_size or info.get("window_size") or 16))
            smooth = counts.copy()
            value_columns = ["retained", "newcomer", "total"]
            smooth[value_columns] = counts[value_columns].rolling(window=win, min_periods=1).mean()

            for field, color in (
                ("retained", "#009E73"),
                ("newcomer", "#D55E00"),
                ("total", "#0072B2"),
            ):
                ax.step(
                    counts["trial"],
                    counts[field],
                    where="mid",
                    color=color,
                    lw=0.8,
                    alpha=0.12,
                    zorder=1,
                )

            ax.plot(smooth["trial"], smooth["total"], color="#0072B2", lw=2.2, linestyle="--", zorder=3, label="Active total")
            ax.plot(smooth["trial"], smooth["retained"], color="#009E73", lw=2.4, alpha=0.95, zorder=4, label="Retained")
            ax.plot(smooth["trial"], smooth["newcomer"], color="#D55E00", lw=2.2, alpha=0.95, zorder=4, label="Newcomers")
            ymax = max(1.0, float(np.nanmax(counts[["retained", "newcomer", "total"]].to_numpy())))
            ax.set_ylim(-0.15, ymax + 0.6)
            ax.set(
                title=f"Subject {iSub} (Condition {condition}) | rolling window={win}",
                xlabel="Trial",
                ylabel="Mean hypothesis count",
            )
            handles, labels = ax.get_legend_handles_labels()
            order = [labels.index(name) for name in ("Retained", "Newcomers", "Active total")]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="best")
            ax.grid(axis="y", alpha=0.22)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Hypothesis Active-Set Counts",
            body,
            **kwargs,
        )

    # Legacy static-strategy amount helpers ----------------------------------

    def plot_strategy_amount(self, results, window_size=16, subjects=None, save_path=None, **kwargs):
        def _first_numeric(value, default=0.0):
            if isinstance(value, (list, tuple)):
                if not value:
                    return float(default)
                return float(value[0])
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def body(ax, condition, iSub, info):
            steps = info.get("strategy_counts_log") or []
            exploitation = []
            exploration = []

            for step in steps:
                best_step_amount = step if isinstance(step, dict) else {}
                posterior_vals = [
                    _first_numeric(value)
                    for key, value in best_step_amount.items()
                    if "posterior" in key
                ]
                exploitation.append(sum(posterior_vals))
                exploration.append(_first_numeric(best_step_amount.get("random", 0.0)))

            rolling_exploitation = pd.Series(exploitation).rolling(window=window_size, min_periods=window_size).mean()
            rolling_exploration = pd.Series(exploration).rolling(window=window_size, min_periods=window_size).mean()

            x = np.arange(1, len(exploitation) + 1)
            ax.plot(x, rolling_exploitation, label="Exploitation", lw=2)
            ax.plot(x, rolling_exploration, label="Exploration", lw=2)
            ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Amount")
            ax.legend()

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Strategy Amount by Subject",
            body,
            **kwargs,
        )

    @staticmethod
    def _strategy_amount_rows(info, count_field="selected_count"):
        def _as_count(value, default=0.0):
            if isinstance(value, (list, tuple)):
                if not value:
                    return float(default)
                value = value[0]
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        logs = info.get("strategy_counts_log")
        if logs is None:
            steps = info.get("best_step_results") or info.get("step_results") or []
            logs = [
                step.get("best_step_amount")
                for step in steps
                if isinstance(step, dict) and isinstance(step.get("best_step_amount"), dict)
            ]

        rows = []
        for trial_idx, step in enumerate(logs or [], start=1):
            if not isinstance(step, dict):
                continue

            strategies = step.get("strategies")
            if isinstance(strategies, list):
                for strategy_idx, strategy in enumerate(strategies):
                    if not isinstance(strategy, dict):
                        continue
                    label = str(strategy.get("label", f"strategy_{strategy_idx}"))
                    amount = str(strategy.get("amount", ""))
                    method = str(strategy.get("method", ""))
                    pool = str(strategy.get("pool", ""))
                    value = _as_count(strategy.get(count_field, strategy.get("selected_count", 0)))
                    rows.append({
                        "trial": int(trial_idx),
                        "strategy_index": int(strategy_idx),
                        "label": label,
                        "amount": amount,
                        "method": method,
                        "pool": pool,
                        "strategy_key": f"{label} | {amount} | {method} | {pool}",
                        "count": value,
                    })
                continue

            # Backward-compatible path for older aggregate-only logs.
            for key, value in step.items():
                if key in {"active_total", "strategies"}:
                    continue
                count = _as_count(value, default=np.nan)
                if np.isnan(count):
                    continue
                method = str(key)
                rows.append({
                    "trial": int(trial_idx),
                    "strategy_index": -1,
                    "label": method,
                    "amount": "",
                    "method": method,
                    "pool": "",
                    "strategy_key": f"{method} |  | {method} | ",
                    "count": count,
                })

        return pd.DataFrame(rows)

    def plot_strategy_amount_details(
        self,
        results,
        window_size=1,
        subjects=None,
        save_path=None,
        count_field="selected_count",
        include_active_total=True,
        **kwargs,
    ):
        """Plot per-strategy transition counts from detailed strategy logs.

        Each line corresponds to one configured strategy identity, shown as
        ``label | amount | method | pool``. Use ``count_field="requested_count"``
        to inspect requested amounts instead of selected counts.
        """
        rolling_window = max(1, int(window_size))
        min_periods = int(kwargs.pop("min_periods", 1))
        high_contrast_colors = kwargs.pop("colors", None) or (
            "#E69F00",  # orange
            "#3969AC",  # deep blue
            "#009E73",  # bluish green
            "#7F3C8D",  # purple
            "#E73F74",  # pink red
        )

        def body(ax, condition, iSub, info):
            df = self._strategy_amount_rows(info, count_field=count_field)
            if df.empty:
                ax.text(0.5, 0.5, "No strategy count log", ha="center", va="center")
                ax.set(title=f"Subject {iSub} (Condition {condition})")
                return

            x_max = int(df["trial"].max())
            x_index = pd.Index(np.arange(1, x_max + 1), name="trial")
            keys = list(dict.fromkeys(df["strategy_key"].tolist()))

            for line_idx, key in enumerate(keys):
                sub = df[df["strategy_key"] == key]
                series = sub.groupby("trial")["count"].sum().reindex(x_index, fill_value=0.0)
                if rolling_window > 1:
                    series = series.rolling(window=rolling_window, min_periods=min_periods).mean()
                ax.plot(
                    x_index.to_numpy(),
                    series.to_numpy(),
                    label=key,
                    lw=kwargs.get("lw", 1.8),
                    alpha=kwargs.get("alpha", 0.7),
                    color=high_contrast_colors[line_idx % len(high_contrast_colors)],
                )

            if include_active_total:
                logs = info.get("strategy_counts_log")
                if logs is None:
                    steps = info.get("best_step_results") or info.get("step_results") or []
                    logs = [
                        step.get("best_step_amount")
                        for step in steps
                        if isinstance(step, dict) and isinstance(step.get("best_step_amount"), dict)
                    ]
                active_total = [
                    float(step.get("active_total", np.nan)) if isinstance(step, dict) else np.nan
                    for step in logs or []
                ]
                if active_total:
                    active_series = pd.Series(active_total, index=np.arange(1, len(active_total) + 1))
                    if rolling_window > 1:
                        active_series = active_series.rolling(window=rolling_window, min_periods=min_periods).mean()
                    ax.plot(
                        active_series.index.to_numpy(),
                        active_series.to_numpy(),
                        label="active_total",
                        color="black",
                        lw=kwargs.get("active_lw", 2.4),
                        linestyle="--",
                    )

            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel=count_field,
            )
            ax.legend(fontsize=kwargs.get("legend_fontsize", 8), loc=kwargs.get("legend_loc", "best"))

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Detailed Strategy Amount by Subject",
            body,
            **kwargs,
        )
