"""Model evaluation facade.

The public surface is organized around these evaluation layers:
- accuracy_model_alignment(full level/ family level)
- posterior_distribution
- oral_mass_distribution
- oral_model_alignment
  - distribution_based_alignment: oral reports -> hypothesis distribution
    (optionally projected into oral-equivalence groups)
  - oral_based_alignment: model belief -> oral center/region representation
  - target_based_alignment: target prior probability vs oral target mass
  - hit_based_alignment: target hit in model active/top-k set vs oral top-N/top-k set
  - coverage_based_alignment: model active-set coverage of oral top-N set
- cluster_amount_dynamics
- beta_dynamics
- error_grid
- trajectory_accuracy
- trajectory_posterior

Oral mass and Oral/model alignment is implemented in ``oral_model_alignment.py`` and mixed
in here so existing ``ModelEval`` call sites keep working.
"""

from collections import defaultdict
import json
import logging
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.Bayesian_state.utils.oral_model_alignment import OralModelAlignmentMixin
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

    @staticmethod
    def _family_correct(categories, choices, n_cats):
        if n_cats >= 4:
            category_family = np.where(np.isin(categories, [1, 2]), 0, 1)
            choice_family = np.where(np.isin(choices, [1, 2]), 0, 1)
            return (category_family == choice_family).astype(float)
        return (categories == choices).astype(float)

    @staticmethod
    def _family_indices(category, n_cats):
        category_idx = int(category) - 1
        if n_cats >= 4:
            if category_idx in (0, 1):
                return np.array([0, 1], dtype=int)
            return np.array([2, 3], dtype=int)
        return np.array([category_idx], dtype=int)

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
        true_family_acc = self._family_correct(categories, choices, n_cats)
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
            family_idx = self._family_indices(int(categories[trial_idx]), n_cats)
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

        sliding_true = []
        sliding_pred = []
        sliding_std = []
        for start in range(1, n_trials - win + 1):
            end = start + win
            true_window = true_family_acc[start:end]
            pred_window = pred_family_acc[start:end]
            sliding_true.append(float(np.mean(true_window)))
            sliding_pred.append(float(np.nanmean(pred_window)))
            valid = pred_window[~np.isnan(pred_window)]
            if valid.size:
                sliding_std.append(float(np.sqrt(np.sum(valid * (1 - valid))) / win))
            else:
                sliding_std.append(np.nan)

        return {
            "true_family_acc": true_family_acc,
            "pred_family_acc": pred_family_acc,
            "sliding_true_family_acc": np.asarray(sliding_true, dtype=float),
            "sliding_pred_family_acc": np.asarray(sliding_pred, dtype=float),
            "sliding_pred_family_acc_std": np.asarray(sliding_std, dtype=float),
        }

    # posterior ---------------------------------------------------------------

    def plot_posterior_probabilities(self, results, subjects=None, save_path=None, limit=True, **kwargs):
        def _get_post_max(hypo_details, k):
            if not isinstance(hypo_details, dict):
                return None

            entry = hypo_details.get(k)
            if entry is None:
                entry = hypo_details.get(str(k))
            if not isinstance(entry, dict):
                return None

            try:
                return float(entry.get("post_max"))
            except (TypeError, ValueError):
                return None

        def body(ax, condition, iSub, info):
            step_results = info.get("step_results") or info.get("best_step_results") or []
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Posterior Probability",
            )

            if not step_results:
                ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
                return

            if limit:
                max_k = 19 if condition == 1 else 116
            else:
                all_keys = []
                for step in step_results:
                    hypo_details = step.get("hypo_details", {})
                    if not isinstance(hypo_details, dict):
                        continue
                    for key in hypo_details.keys():
                        try:
                            all_keys.append(int(key))
                        except (TypeError, ValueError):
                            pass
                max_k = max(all_keys) + 1 if all_keys else 0

            data = []
            for step, step_info in enumerate(step_results):
                hypo_details = step_info.get("hypo_details", {})
                for k in range(max_k):
                    post_max = _get_post_max(hypo_details, k)
                    if post_max is None:
                        continue
                    data.append({"Step": step + 1, "k": k, "Posterior": post_max})

            df = pd.DataFrame(data)
            if df.empty or "Step" not in df.columns:
                ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
                return

            sns.scatterplot(
                data=df,
                x="Step",
                y="Posterior",
                hue="k",
                palette="tab10",
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

    # accuracy_model_alignment ------------------------------------------------

    def plot_accuracy_comparison(self, results, subjects=None, save_path=None, window_size=None, **kwargs):
        def body(ax, condition, iSub, info):
            true_acc = info["sliding_true_acc"]
            pred_acc = info["sliding_pred_acc"]
            pred_std = info["sliding_pred_acc_std"]
            win = info.get("window_size") or window_size
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
                    "Low": np.array(pred_acc) - pred_std,
                    "High": np.array(pred_acc) + pred_std,
                }
            )
            sns.lineplot(data=df, x="Trial", y="Pred", label="Predicted", ax=ax)
            sns.lineplot(data=df, x="Trial", y="True", label="True", ax=ax)
            ax.fill_between(df["Trial"], df["Low"], df["High"], alpha=0.2)

            n_trials = info.get("n_trials")
            if n_trials:
                ax.set_xlim(1, n_trials)
            ax.set_ylim(0, 1)
            ax.set(title=f"Subject {iSub} (Condition {condition})", xlabel="Trial", ylabel="Accuracy")
            ax.legend()

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Predicted vs True Accuracy by Subject",
            body,
            **kwargs,
        )

    def plot_accuracy_family_comparison(self, results, subjects=None, save_path=None, window_size=None, **kwargs):
        def body(ax, condition, iSub, info):
            true_acc = info.get("sliding_true_family_acc")
            pred_acc = info.get("sliding_pred_family_acc")
            pred_std = info.get("sliding_pred_family_acc_std")
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

            win = info.get("window_size") or window_size
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
            results,
            subjects,
            save_path,
            "Predicted vs True Family Accuracy by Subject",
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
            max_by_hypo = np.nanmax(active_beta, axis=0)
            valid_hypo = np.where(np.isfinite(max_by_hypo))[0]
            if valid_hypo.size == 0:
                ax.text(0.5, 0.5, "No active beta data", ha="center", va="center", transform=ax.transAxes)
                return

            order = valid_hypo[np.argsort(max_by_hypo[valid_hypo])[::-1]]
            if max_hypotheses is not None:
                order = order[: max(1, int(max_hypotheses))]
            for hypo in order:
                ax.plot(trials, beta[:, hypo], lw=0.8, alpha=0.35)

            mean_active = np.nanmean(active_beta, axis=1)
            max_active = np.nanmax(active_beta, axis=1)
            ax.plot(trials, mean_active, color="black", lw=2.0, label="Active mean")
            ax.plot(trials, max_active, color="red", lw=1.5, alpha=0.75, label="Active max")
            ax.set_xlim(1, beta.shape[0])
            y_max = np.nanmax(max_active)
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

        sample_errors = payload.get("sample_errors")
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
    def _get_post_max(hypo_details, k):
        if not isinstance(hypo_details, dict):
            return None
        entry = hypo_details.get(k)
        if entry is None:
            entry = hypo_details.get(str(k))
        if not isinstance(entry, dict):
            return None
        try:
            return float(entry.get("post_max"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_step_results_from_run(run_obj):
        step_results = run_obj.get("step_log") or run_obj.get("step_results") or []
        if isinstance(step_results, list) and step_results:
            return step_results

        posterior_log = run_obj.get("posterior_log") or []
        rebuilt = []
        for posterior in posterior_log:
            try:
                posterior = np.asarray(posterior, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            hypo_details = {
                int(k): {"post_max": float(p)}
                for k, p in enumerate(posterior)
                if float(p) > 1e-9
            }
            rebuilt.append({"hypo_details": hypo_details})
        return rebuilt

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
        step_results = self._extract_step_results_from_run(run_obj)
        ax.set(xlabel="Trial", ylabel="Posterior Probability")

        if not step_results:
            ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
            return

        if limit:
            max_k = 19 if int(condition) == 1 else 116
        else:
            all_keys = []
            for step in step_results:
                hypo_details = step.get("hypo_details", {}) if isinstance(step, dict) else {}
                for key in hypo_details.keys():
                    try:
                        all_keys.append(int(key))
                    except (TypeError, ValueError):
                        pass
            max_k = max(all_keys) + 1 if all_keys else 0

        data = []
        for step, step_info in enumerate(step_results):
            if not isinstance(step_info, dict):
                continue
            hypo_details = step_info.get("hypo_details", {})
            for k in range(max_k):
                post_max = self._get_post_max(hypo_details, k)
                if post_max is None:
                    continue
                data.append({"Step": step + 1, "k": k, "Posterior": post_max})

        df = pd.DataFrame(data)
        if df.empty:
            ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
            return

        sns.scatterplot(
            data=df,
            x="Step",
            y="Posterior",
            hue="k",
            palette="tab10",
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

    # cluster_amount ----------------------------------------------------------

    def plot_cluster_amount(self, results, window_size=16, subjects=None, save_path=None, **kwargs):
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
            steps = info.get("best_step_results", [])
            exploitation = []
            exploration = []

            for step in steps:
                best_step_amount = step.get("best_step_amount", {})
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

    # error_grid --------------------------------------------------------------

    def plot_error_grids(self, results, subjects=None, fname=None, save_path=None, **kwargs):
        labels = fname if isinstance(fname, (list, tuple)) and len(fname) >= 2 else ("gamma", "w0")

        def body(ax, condition, iSub, info):
            data = []
            for (gamma, w0), errs in info["grid_errors"].items():
                data.append({"gamma": gamma, "w0": w0, "Error": float(np.mean(errs))})

            df = pd.DataFrame(data)
            error_matrix = df.pivot(index="gamma", columns="w0", values="Error")
            sns.heatmap(error_matrix, cbar_kws={"label": "Error"}, ax=ax, cmap="viridis_r")
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel=labels[1],
                ylabel=labels[0],
            )
            ax.set_xticks(np.arange(len(error_matrix.columns)) + 0.5)
            ax.set_xticklabels([f"{v:.4f}" for v in error_matrix.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(error_matrix.index)) + 0.5)
            ax.set_yticklabels([f"{v:.2f}" for v in error_matrix.index], rotation=0)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Grid Search Error by Subject",
            body,
            **kwargs,
        )
