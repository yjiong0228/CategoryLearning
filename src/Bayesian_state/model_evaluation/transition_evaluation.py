"""Evaluation helpers tied to hypothesis-transition log capabilities.

The mixin is intentionally separate from general model evaluation: these plots
are meaningful only when a transition implementation emits the corresponding
dynamic-discrete, dynamic-continuous, or active-set fields.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import seaborn as sns


class TransitionEvaluationMixin:
    """Diagnostics for model-specific hypothesis-transition mechanisms."""

    PROFILE_POLICY_ORDER = ("conservative", "stable", "aggressive", "stubborn")
    PROFILE_POLICY_COLORS = {
        "conservative": "#0072B2",
        "stable": "#009E73",
        "aggressive": "#D55E00",
        "stubborn": "#CC79A7",
    }

    @staticmethod
    def transition_capabilities(info: Mapping[str, Any]) -> set[str]:
        """Infer available transition diagnostics from persisted log fields."""
        capabilities: set[str] = set()
        if any(info.get(field) is not None for field in ("transition_rate", "search_range")):
            capabilities.add("dynamic_continuous")
        if info.get("marginal_active_probability") is not None:
            capabilities.add("particle_marginal")
        if any(info.get(field) is not None for field in ("pre_choice_ess", "post_choice_ess")):
            capabilities.add("particle_filter")
        for step in info.get("strategy_counts_log") or []:
            if not isinstance(step, Mapping):
                continue
            controller = step.get("state_controller")
            profile_policy = step.get("profile_policy")
            views = [
                value
                for value in (step, controller, profile_policy)
                if isinstance(value, Mapping)
            ]
            if any(
                isinstance(view.get(field), Mapping) and bool(view.get(field))
                for view in views
                for field in ("state_probabilities", "policy_probabilities")
            ):
                capabilities.add("dynamic_discrete")
            if any(
                field in view
                for view in views
                for field in (
                    "predictive_m",
                    "predictive_g",
                    "control_logit",
                    "g_control_logit",
                )
            ):
                capabilities.add("dynamic_continuous")
            if (
                "active_total" in step
                or isinstance(step.get("strategies"), list)
                or isinstance(profile_policy, Mapping)
            ):
                capabilities.add("active_set")
        return capabilities

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

    @classmethod
    def _active_set_count_rows(cls, info):
        rows = []
        for trial_idx, step in enumerate(info.get("strategy_counts_log") or [], start=1):
            if not isinstance(step, Mapping):
                continue
            policy_log = step.get("profile_policy")
            if isinstance(policy_log, Mapping):
                retained = cls._safe_float(policy_log.get("retained_count"), default=0.0)
                newcomer = cls._safe_float(policy_log.get("newcomer_count"), default=0.0)
            else:
                retained = 0.0
                newcomer = 0.0
                for strategy in step.get("strategies") or []:
                    if not isinstance(strategy, Mapping):
                        continue
                    count = cls._safe_float(strategy.get("selected_count"), default=0.0)
                    if strategy.get("pool") == "active":
                        retained += count
                    elif strategy.get("pool") == "inactive":
                        newcomer += count
            total = cls._safe_float(step.get("active_total"), default=retained + newcomer)
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

    # dynamic-continuous and particle-marginal diagnostics -----------------

    @staticmethod
    def _transition_series(info: Mapping[str, Any], field: str) -> np.ndarray:
        direct = info.get(field)
        if direct is not None:
            try:
                values = np.asarray(direct, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                values = np.asarray([], dtype=float)
            if values.size:
                return values
        rows = []
        for step in info.get("strategy_counts_log") or []:
            if not isinstance(step, Mapping):
                rows.append(np.nan)
                continue
            value = step.get(field)
            if value is None and field == "transition_rate":
                value = step.get("predictive_m")
            if value is None and field == "search_range":
                value = step.get("predictive_g")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = np.nan
            rows.append(numeric if np.isfinite(numeric) else np.nan)
        return np.asarray(rows, dtype=float)

    @staticmethod
    def _add_holdout_split(ax, info: Mapping[str, Any]) -> None:
        context = info.get("score_context") or {}
        if not isinstance(context, Mapping):
            return
        split_index = context.get("split_index")
        if split_index is None:
            return
        ax.axvline(
            float(split_index) + 0.5,
            color="#555555",
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label="Train/evaluation split",
        )

    def plot_dynamic_continuous_controls(
        self,
        results,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot filtered replacement/search controls and realized replacement."""

        def body(ax, condition, iSub, info):
            series = {
                "Replacement rate $m_t$": self._transition_series(info, "transition_rate"),
                "Search range $g_t$": self._transition_series(info, "search_range"),
                "Replacement fraction": self._transition_series(info, "replacement_fraction"),
            }
            colors = {
                "Replacement rate $m_t$": "#0072B2",
                "Search range $g_t$": "#009E73",
                "Replacement fraction": "#D55E00",
            }
            plotted = False
            for label, values in series.items():
                if values.size == 0 or not np.any(np.isfinite(values)):
                    continue
                ax.plot(
                    np.arange(1, values.size + 1),
                    values,
                    label=label,
                    color=colors[label],
                    linewidth=2.0 if "fraction" not in label.lower() else 1.5,
                    alpha=0.95 if "fraction" not in label.lower() else 0.65,
                )
                plotted = True
            self._add_holdout_split(ax, info)
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Probability / fraction",
                ylim=(-0.02, 1.02),
            )
            if plotted:
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(0.5, 0.5, "No continuous-control log", ha="center", va="center", transform=ax.transAxes)
            ax.grid(axis="y", alpha=0.2)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Dynamic-Continuous Transition Controls",
            body,
            **kwargs,
        )

    def plot_dynamic_continuous_signals(
        self,
        results,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot filtered feedback surprise and uncertainty on separate axes."""

        def body(ax, condition, iSub, info):
            surprise = self._transition_series(info, "feedback_surprise")
            uncertainty = self._transition_series(info, "feedback_uncertainty")
            surprise_ok = surprise.size and np.any(np.isfinite(surprise))
            uncertainty_ok = uncertainty.size and np.any(np.isfinite(uncertainty))
            lines = []
            labels = []
            if surprise_ok:
                line = ax.plot(
                    np.arange(1, surprise.size + 1),
                    surprise,
                    color="#D55E00",
                    linewidth=1.8,
                    label="Feedback surprise",
                )[0]
                lines.append(line)
                labels.append(line.get_label())
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Surprise",
            )
            if uncertainty_ok:
                right = ax.twinx()
                line = right.plot(
                    np.arange(1, uncertainty.size + 1),
                    uncertainty,
                    color="#0072B2",
                    linewidth=1.8,
                    alpha=0.9,
                    label="Feedback uncertainty",
                )[0]
                right.set_ylabel("Uncertainty")
                lines.append(line)
                labels.append(line.get_label())
            self._add_holdout_split(ax, info)
            if lines:
                ax.legend(lines, labels, fontsize=8, loc="best")
            else:
                ax.text(0.5, 0.5, "No control-signal log", ha="center", va="center", transform=ax.transAxes)
            ax.grid(axis="x", alpha=0.16)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Dynamic-Continuous Feedback Signals",
            body,
            **kwargs,
        )

    def plot_particle_filter_ess(
        self,
        results,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot normalized pre/post-choice ESS and resampling events."""

        def body(ax, condition, iSub, info):
            pre = self._transition_series(info, "pre_choice_ess")
            post = self._transition_series(info, "post_choice_ess")
            resampled = self._transition_series(info, "resampled")
            particle_count = self._safe_float(info.get("particle_count"), default=np.nan)
            normalize = bool(np.isfinite(particle_count) and particle_count > 0)
            plotted = False
            for values, label, color in (
                (pre, "Pre-choice ESS", "#0072B2"),
                (post, "Post-choice ESS", "#D55E00"),
            ):
                if values.size and np.any(np.isfinite(values)):
                    plotted_values = values / particle_count if normalize else values
                    ax.plot(
                        np.arange(1, values.size + 1),
                        plotted_values,
                        label=f"{label} / N" if normalize else label,
                        color=color,
                        linewidth=1.8,
                    )
                    plotted = True
            if resampled.size:
                trials = np.flatnonzero(np.asarray(resampled, dtype=bool)) + 1
                if trials.size:
                    ax.scatter(
                        trials,
                        np.full(trials.size, 0.03),
                        marker="|",
                        s=55,
                        color="#000000",
                        label="Resampled",
                        zorder=5,
                    )
                    plotted = True
            self._add_holdout_split(ax, info)
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Effective sample size / N" if normalize else "Effective sample size",
            )
            if normalize:
                ax.set_ylim(-0.02, 1.02)
            if plotted:
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(0.5, 0.5, "No particle ESS log", ha="center", va="center", transform=ax.transAxes)
            ax.grid(axis="y", alpha=0.2)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Particle-Filter Effective Sample Size",
            body,
            **kwargs,
        )

    def plot_marginal_active_probabilities(
        self,
        results,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot particle-marginal active-hypothesis probabilities by trial."""

        def body(ax, condition, iSub, info):
            raw = info.get("marginal_active_probability")
            try:
                values = np.asarray(raw, dtype=float)
            except (TypeError, ValueError):
                values = np.asarray([], dtype=float)
            if values.ndim != 2 or values.size == 0:
                ax.text(0.5, 0.5, "No marginal active-state data", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"Subject {iSub} (Condition {condition})")
                return
            image = ax.imshow(
                values.T,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            context = info.get("score_context") or {}
            if isinstance(context, Mapping) and context.get("split_index") is not None:
                ax.axvline(
                    float(context["split_index"]) - 0.5,
                    color="white",
                    linestyle=":",
                    linewidth=1.5,
                )
            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Hypothesis index",
            )
            ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="P(active)")

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Particle-Marginal Active-Hypothesis Probability",
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
