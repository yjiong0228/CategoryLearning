"""Model evaluation facade.

The public surface is organized around five evaluation layers:
- posterior
- accuracy_model_alignment
- oral_model_alignment
- cluster_amount
- error_grid

Oral/model alignment is implemented in ``oral_model_alignment.py`` and mixed
in here so existing ``ModelEval`` call sites keep working.
"""

from collections import defaultdict
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.Bayesian_state.utils.oral_model_alignment import OralModelAlignmentMixin


logger = logging.getLogger(__name__)


class ModelEval(OralModelAlignmentMixin):
    """Evaluation and plotting entry point for state-based model results."""

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

            target_hypo = 0 if condition == 1 else 42
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

    # oral_model_alignment ----------------------------------------------------
    # Implemented by OralModelAlignmentMixin:
    # - plot_k_oral_comparison
    # - compute_oral_model_alignment
    # - plot_oral_model_alignment
    # - compute_choice_conditioned_oral_alignment
    # - plot_choice_conditioned_oral_alignment

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
