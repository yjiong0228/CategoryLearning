"""Particle-filter-specific evaluation inside the shared ModelEvaluator pipeline.

Particle-filter repeats estimate the same marginalized predictive probability.
Their across-seed spread is therefore a numerical stability diagnostic, not a
behavioral predictive interval.  This mixin keeps that distinction explicit:
the accuracy ribbon is generated from Bernoulli behavioral draws conditional
on the observed history, continuous strategy profiles use pre-choice particle
weights, and ESS/marginal-active plots remain numerical PF diagnostics.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.metrics import (
    conditional_behavioral_accuracy_band_metrics,
)


class ParticleFilterEvaluationMixin:
    """Behavioral predictive checks and diagnostics for particle-filter runs."""

    DEFAULT_BEHAVIORAL_BAND_DRAWS = 5000
    DEFAULT_BEHAVIORAL_BAND_SEED = 20260810

    @staticmethod
    def is_particle_filter_result(info: Mapping[str, Any]) -> bool:
        """Identify particle-filter results from persisted semantics, not names."""
        if str(info.get("state_distribution_kind", "")).lower() == "particle_marginal":
            return True
        if str(info.get("choice_readout_method", "")).lower() == "particle_marginal":
            return True
        return any(
            info.get(field) is not None
            for field in (
                "pre_choice_ess",
                "post_choice_ess",
                "marginal_active_probability",
            )
        )

    @classmethod
    def particle_filter_capabilities(cls, info: Mapping[str, Any]) -> set[str]:
        """Infer available PF diagnostics independently of transition features."""
        capabilities: set[str] = set()
        if any(info.get(field) is not None for field in ("pre_choice_ess", "post_choice_ess")):
            capabilities.add("particle_filter")
        if info.get("marginal_active_probability") is not None:
            capabilities.add("particle_marginal")
        if cls.is_particle_filter_result(info) and any(
            info.get(field) is not None
            for field in (
                "predictive_strategy_exploit",
                "predictive_swap_probability",
                "transition_rate",
            )
        ):
            capabilities.add("particle_continuous_strategy")
        return capabilities

    @staticmethod
    def _particle_series(info: Mapping[str, Any], field: str) -> np.ndarray:
        value = info.get(field)
        if value is None:
            return np.asarray([], dtype=float)
        try:
            return np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)

    @staticmethod
    def _add_particle_holdout_split(ax, info: Mapping[str, Any]) -> None:
        context = info.get("score_context") or {}
        if not isinstance(context, Mapping) or context.get("split_index") is None:
            return
        ax.axvline(
            float(context["split_index"]) + 0.5,
            color="#555555",
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label="Train/evaluation split",
        )

    @classmethod
    def _first_particle_series(
        cls,
        info: Mapping[str, Any],
        *fields: str,
    ) -> tuple[np.ndarray, str | None]:
        for field in fields:
            values = cls._particle_series(info, field)
            if values.size:
                return values, field
        return np.asarray([], dtype=float), None

    @staticmethod
    def _particle_capacity(info: Mapping[str, Any]) -> int:
        params = info.get("best_params") or info.get("fixed_hyperparams") or {}
        if not isinstance(params, Mapping):
            params = {}
        transition_kwargs = params.get("engine.modules.hypo_transitions_mod.kwargs")
        candidates = [
            params.get("capacity"),
            params.get("engine.modules.hypo_transitions_mod.kwargs.capacity"),
            transition_kwargs.get("capacity")
            if isinstance(transition_kwargs, Mapping)
            else None,
        ]
        for value in candidates:
            try:
                capacity = int(value)
            except (TypeError, ValueError):
                continue
            if capacity > 0:
                return capacity
        raise ValueError("Particle strategy profile requires a positive workspace capacity")

    @staticmethod
    def _causal_recent_accuracy(values: Any, window_size: int, n_trials: int) -> np.ndarray:
        try:
            observed = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return np.full(n_trials, np.nan, dtype=float)
        n = min(int(n_trials), observed.size)
        out = np.full(n_trials, np.nan, dtype=float)
        for trial_index in range(int(window_size), n):
            history = observed[trial_index - int(window_size) : trial_index]
            finite = history[np.isfinite(history)]
            if finite.size == int(window_size):
                out[trial_index] = float(np.mean(finite))
        return out

    @classmethod
    def _particle_continuous_strategy_data(
        cls,
        info: Mapping[str, Any],
        *,
        window_size: int | None = None,
        low_accuracy_threshold: float = 0.60,
        mastery_accuracy_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Build causal continuous exploit/local/global strategy tendencies."""
        exact_fields = (
            "predictive_strategy_exploit",
            "predictive_strategy_local_explore",
            "predictive_strategy_global_explore",
        )
        exact = [cls._particle_series(info, field) for field in exact_fields]
        has_exact = all(values.size for values in exact)
        capacity = cls._particle_capacity(info)

        if has_exact:
            n_trials = min(values.size for values in exact)
            strategy = np.column_stack([values[:n_trials] for values in exact])
            source_semantics = "pre_choice_particle_marginal"
            swap_probability, _ = cls._first_particle_series(
                info,
                "predictive_swap_probability",
            )
        else:
            transition_rate, rate_field = cls._first_particle_series(
                info,
                "predictive_transition_rate",
                "transition_rate",
            )
            search_range, range_field = cls._first_particle_series(
                info,
                "predictive_search_range",
                "search_range",
            )
            if not transition_rate.size or not search_range.size:
                raise ValueError("No usable continuous transition controls")
            n_trials = min(transition_rate.size, search_range.size)
            transition_rate = np.clip(transition_rate[:n_trials], 0.0, 1.0)
            search_range = np.clip(search_range[:n_trials], 0.0, 1.0)
            swap_probability, swap_field = cls._first_particle_series(
                info,
                "predictive_swap_probability",
                "swap_probability",
            )
            if swap_probability.size:
                n_trials = min(n_trials, swap_probability.size)
                transition_rate = transition_rate[:n_trials]
                search_range = search_range[:n_trials]
                swap_probability = np.clip(swap_probability[:n_trials], 0.0, 1.0)
                source_semantics = (
                    "pre_choice_particle_marginal"
                    if swap_field == "predictive_swap_probability"
                    else "post_choice_filtered_legacy"
                )
            else:
                swap_probability = 1.0 - np.power(
                    1.0 - transition_rate,
                    int(capacity),
                )
                # The initial workspace is constructed without a replacement draw.
                if swap_probability.size:
                    swap_probability[0] = 0.0
                source_semantics = "derived_from_filtered_controls_legacy"
            strategy = np.column_stack(
                [
                    1.0 - swap_probability,
                    swap_probability * (1.0 - search_range),
                    swap_probability * search_range,
                ]
            )
            if rate_field == "predictive_transition_rate" and range_field == "predictive_search_range":
                source_semantics = source_semantics.replace("filtered", "predictive")

        if strategy.shape[0] == 0 or not np.all(np.isfinite(strategy)):
            raise ValueError("Continuous strategy tendencies contain missing values")
        strategy = np.clip(strategy, 0.0, 1.0)
        row_sum = np.sum(strategy, axis=1, keepdims=True)
        if np.any(row_sum <= 0.0):
            raise ValueError("Continuous strategy tendencies have a zero-sum trial")
        strategy = strategy / row_sum
        n_trials = strategy.shape[0]
        if not swap_probability.size:
            swap_probability = 1.0 - strategy[:, 0]
        else:
            swap_probability = np.asarray(swap_probability[:n_trials], dtype=float)

        realized, realized_field = cls._first_particle_series(
            info,
            "predictive_swap_event_probability",
            "swap_event_probability",
            "predictive_replacement_fraction",
            "replacement_fraction",
        )
        realized = realized[:n_trials] if realized.size else np.full(n_trials, np.nan)
        distance, distance_field = cls._first_particle_series(
            info,
            "predictive_newcomer_distance",
            "newcomer_distance",
        )
        distance = distance[:n_trials] if distance.size else np.full(n_trials, np.nan)
        event_probability, _ = cls._first_particle_series(
            info,
            "predictive_swap_event_probability",
            "swap_event_probability",
        )
        if not event_probability.size:
            event_probability = swap_probability
        event_probability = np.asarray(event_probability[:n_trials], dtype=float)
        conditional_distance = np.divide(
            distance,
            event_probability,
            out=np.full(n_trials, np.nan, dtype=float),
            where=np.isfinite(event_probability) & (event_probability > 1e-12),
        )
        conditional_distance = np.clip(conditional_distance, 0.0, 1.0)
        controller_series = {}
        for field in (
            "predictive_failure_pressure",
            "predictive_mastery_evidence",
            "predictive_exploration_target",
            "predictive_global_target",
            "predictive_prior_reset_strength",
            "predictive_prior_reset_mass_shift",
            "predictive_misconception_capture_hold_probability",
            "predictive_misconception_capture_switch_event_probability",
            "predictive_misconception_capture_eligible_probability",
        ):
            values = cls._particle_series(info, field)
            controller_series[field] = (
                np.asarray(values[:n_trials], dtype=float)
                if values.size >= n_trials
                else np.full(n_trials, np.nan, dtype=float)
            )
        capture_diagnostics_available = bool(
            cls._particle_series(
                info, "predictive_misconception_capture_hold_probability"
            ).size
        )

        resolved_window = max(1, int(window_size or info.get("window_size") or 16))
        causal_accuracy = cls._causal_recent_accuracy(
            info.get("true_acc"),
            resolved_window,
            n_trials,
        )
        low_mask = np.isfinite(causal_accuracy) & (
            causal_accuracy <= float(low_accuracy_threshold)
        )
        mastery_mask = np.isfinite(causal_accuracy) & (
            causal_accuracy >= float(mastery_accuracy_threshold)
        )

        def phase_mean(values: np.ndarray, mask: np.ndarray) -> float:
            selected = np.asarray(values, dtype=float)[mask]
            selected = selected[np.isfinite(selected)]
            return float(np.mean(selected)) if selected.size else float("nan")

        finite_relation = np.isfinite(causal_accuracy) & np.isfinite(swap_probability)
        if (
            int(np.sum(finite_relation)) >= 2
            and float(np.std(causal_accuracy[finite_relation])) > 0.0
            and float(np.std(swap_probability[finite_relation])) > 0.0
        ):
            exploration_accuracy_correlation = float(
                np.corrcoef(
                    swap_probability[finite_relation],
                    causal_accuracy[finite_relation],
                )[0, 1]
            )
        else:
            exploration_accuracy_correlation = float("nan")

        return {
            "subject_id": int(info.get("subject_id", -1)),
            "condition": int(info.get("condition", -1)),
            "trial": np.arange(1, n_trials + 1, dtype=int),
            "capacity": int(capacity),
            "strategy": strategy,
            "swap_probability": swap_probability,
            "realized": realized,
            "realized_field": realized_field,
            "conditional_newcomer_distance": conditional_distance,
            "distance_field": distance_field,
            "source_semantics": source_semantics,
            "window_size": resolved_window,
            "causal_accuracy": causal_accuracy,
            "low_mask": low_mask,
            "mastery_mask": mastery_mask,
            "low_accuracy_threshold": float(low_accuracy_threshold),
            "mastery_accuracy_threshold": float(mastery_accuracy_threshold),
            "low_explore": phase_mean(1.0 - strategy[:, 0], low_mask),
            "mastery_explore": phase_mean(1.0 - strategy[:, 0], mastery_mask),
            "low_local": phase_mean(strategy[:, 1], low_mask),
            "mastery_local": phase_mean(strategy[:, 1], mastery_mask),
            "low_global": phase_mean(strategy[:, 2], low_mask),
            "mastery_global": phase_mean(strategy[:, 2], mastery_mask),
            "low_distance": phase_mean(conditional_distance, low_mask),
            "mastery_distance": phase_mean(conditional_distance, mastery_mask),
            "controller_series": controller_series,
            "low_failure_pressure": phase_mean(
                controller_series["predictive_failure_pressure"], low_mask
            ),
            "mastery_failure_pressure": phase_mean(
                controller_series["predictive_failure_pressure"], mastery_mask
            ),
            "low_mastery_evidence": phase_mean(
                controller_series["predictive_mastery_evidence"], low_mask
            ),
            "mastery_mastery_evidence": phase_mean(
                controller_series["predictive_mastery_evidence"], mastery_mask
            ),
            "low_prior_reset_strength": phase_mean(
                controller_series["predictive_prior_reset_strength"], low_mask
            ),
            "mastery_prior_reset_strength": phase_mean(
                controller_series["predictive_prior_reset_strength"], mastery_mask
            ),
            "low_prior_reset_mass_shift": phase_mean(
                controller_series["predictive_prior_reset_mass_shift"], low_mask
            ),
            "mastery_prior_reset_mass_shift": phase_mean(
                controller_series["predictive_prior_reset_mass_shift"], mastery_mask
            ),
            "capture_diagnostics_available": capture_diagnostics_available,
            "capture_hold_probability": controller_series[
                "predictive_misconception_capture_hold_probability"
            ],
            "capture_switch_event_probability": controller_series[
                "predictive_misconception_capture_switch_event_probability"
            ],
            "capture_eligible_probability": controller_series[
                "predictive_misconception_capture_eligible_probability"
            ],
            "low_capture_hold": phase_mean(
                controller_series[
                    "predictive_misconception_capture_hold_probability"
                ],
                low_mask,
            ),
            "mastery_capture_hold": phase_mean(
                controller_series[
                    "predictive_misconception_capture_hold_probability"
                ],
                mastery_mask,
            ),
            "exploration_accuracy_correlation": exploration_accuracy_correlation,
        }

    def _particle_filter_accuracy_band_data(
        self,
        subject_json_path,
        *,
        eval_prediction_mode=None,
        max_runs=None,
        n_draws=DEFAULT_BEHAVIORAL_BAND_DRAWS,
        seed=DEFAULT_BEHAVIORAL_BAND_SEED,
    ):
        """Build an observed-history-conditional behavioral accuracy interval."""
        subject_json_path = Path(subject_json_path)
        payload = self.load_subject_payload(subject_json_path)
        stream = self._load_run_stream(payload, subject_json_path)
        probability_runs: list[np.ndarray] = []
        true_curve = None
        score_mask = None

        for stream_index, run_obj in enumerate(stream):
            if max_runs is not None and stream_index >= int(max_runs):
                break
            if not isinstance(run_obj, Mapping):
                continue
            try:
                metrics = self._extract_run_metrics(
                    run_obj,
                    eval_prediction_mode=eval_prediction_mode,
                )
                run_probabilities = self._as_float_1d(
                    metrics.get("pred_acc"),
                    "pred_acc",
                )
                run_true_curve = self._as_float_1d(
                    metrics.get("sliding_true_acc"),
                    "sliding_true_acc",
                )
            except (KeyError, TypeError, ValueError):
                continue

            raw_mask = metrics.get("score_trial_mask")
            if raw_mask is None:
                raw_mask = metrics.get("valid_trial_mask")
            run_score_mask = (
                np.ones(run_probabilities.size, dtype=bool)
                if raw_mask is None
                else np.asarray(raw_mask, dtype=bool).reshape(-1)
            )
            if run_score_mask.size != run_probabilities.size:
                continue

            if true_curve is None:
                true_curve = np.asarray(run_true_curve, dtype=float)
                score_mask = run_score_mask
            elif not np.allclose(
                np.asarray(run_true_curve, dtype=float),
                true_curve,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Observed rolling-accuracy curves differ across PF runs in {subject_json_path}"
                )
            elif not np.array_equal(run_score_mask, score_mask):
                raise ValueError(
                    f"Score masks differ across PF runs in {subject_json_path}"
                )

            if probability_runs and run_probabilities.size != probability_runs[0].size:
                continue
            probability_runs.append(np.asarray(run_probabilities, dtype=float))

        if true_curve is None or score_mask is None or not probability_runs:
            raise ValueError(
                f"No usable particle-filter probability runs found in {subject_json_path}"
            )

        selection = payload.get("selection") or {}
        summary = payload.get("simulation") or payload.get("simulation_summary") or {}
        window_size = int(
            summary.get("window_size")
            or (selection.get("selection_meta") or {}).get("window_size")
            or 1
        )
        subject_id = int(payload.get("subject_id", -1))
        subject_seed = int(
            np.random.SeedSequence([int(seed), subject_id]).generate_state(1)[0]
        )
        band_metrics = conditional_behavioral_accuracy_band_metrics(
            np.vstack(probability_runs),
            true_curve,
            window_size=window_size,
            n_draws=int(n_draws),
            seed=subject_seed,
            score_trial_mask=score_mask,
        )
        x = np.arange(
            window_size + 1,
            window_size + 1 + len(true_curve),
        )
        return {
            "subject_id": subject_id,
            "condition": int(payload.get("condition", -1)),
            "n_runs": int(len(probability_runs)),
            "window_size": int(window_size),
            "base_seed": int(seed),
            "x": x,
            "true_curve": true_curve,
            **band_metrics,
        }

    @staticmethod
    def _draw_particle_filter_accuracy_band(
        ax,
        band: Mapping[str, Any],
        *,
        show_legend: bool = True,
    ) -> None:
        x = np.asarray(band["x"], dtype=float)
        ax.fill_between(
            x,
            band["q05"],
            band["q95"],
            color="#9DB9D8",
            alpha=0.38,
            linewidth=0,
            label="90% behavioral PI",
        )
        ax.fill_between(
            x,
            band["q25"],
            band["q75"],
            color="#4F81B8",
            alpha=0.50,
            linewidth=0,
            label="50% behavioral PI",
        )
        ax.plot(
            x,
            band["expected_curve"],
            color="#E69F00",
            linewidth=2.0,
            label="PF expected accuracy",
            zorder=4,
        )
        ax.plot(
            x,
            band["true_curve"],
            color="#111111",
            linewidth=2.1,
            label="Subject",
            zorder=5,
        )
        ax.set(
            title=f"Subject {band['subject_id']} | PF runs={band['n_runs']}",
            xlabel="Trial",
            ylabel="Rolling accuracy",
            ylim=(0.0, 1.0),
        )
        ax.grid(axis="y", alpha=0.20)
        if show_legend:
            ax.legend(loc="best", fontsize=8, frameon=False)

    @plt.rc_context(
        {
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
    )
    def plot_particle_filter_accuracy_band_group(
        self,
        input_dir,
        save_path=None,
        *,
        summary_path=None,
        eval_prediction_mode=None,
        max_runs_per_subject=None,
        subjects=None,
        n_draws=DEFAULT_BEHAVIORAL_BAND_DRAWS,
        seed=DEFAULT_BEHAVIORAL_BAND_SEED,
        n_cols=None,
        max_subjects_per_row=8,
    ) -> pd.DataFrame:
        """Plot one non-duplicated PF behavioral interval figure for all subjects."""
        input_dir = Path(input_dir)
        subject_set = {int(subject) for subject in subjects} if subjects is not None else None
        bands = []
        for subject_json in self._subject_json_files(input_dir):
            subject_id = int(subject_json.stem.replace("subject_", ""))
            if subject_set is not None and subject_id not in subject_set:
                continue
            bands.append(
                self._particle_filter_accuracy_band_data(
                    subject_json,
                    eval_prediction_mode=eval_prediction_mode,
                    max_runs=max_runs_per_subject,
                    n_draws=n_draws,
                    seed=seed,
                )
            )
        if not bands:
            raise ValueError(f"No usable particle-filter accuracy data found in {input_dir}")

        bands = sorted(bands, key=lambda row: (row["condition"], row["subject_id"]))
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for band in bands:
            grouped[int(band["condition"])].append(band)
        layout_kwargs = {"max_subjects_per_row": max_subjects_per_row}
        if n_cols is not None:
            layout_kwargs["n_cols"] = int(n_cols)
        n_rows, n_cols, rows_by_condition = self._layout_by_condition(
            grouped,
            layout_kwargs,
        )

        fig = plt.figure(figsize=(n_cols * 4.2, n_rows * 3.1))
        row_offset = 0
        legend_drawn = False
        for condition, condition_bands in sorted(grouped.items()):
            for index, band in enumerate(condition_bands):
                local_row = index // n_cols
                col = index % n_cols
                ax = fig.add_subplot(
                    n_rows,
                    n_cols,
                    (row_offset + local_row) * n_cols + col + 1,
                )
                self._draw_particle_filter_accuracy_band(
                    ax,
                    band,
                    show_legend=not legend_drawn,
                )
                legend_drawn = True
            row_offset += rows_by_condition[condition]

        used_axes = sum(len(items) for items in grouped.values())
        for index in range(used_axes, n_rows * n_cols):
            ax = fig.add_subplot(n_rows, n_cols, index + 1)
            ax.axis("off")
        fig.suptitle(
            "Particle-filter conditional behavioral predictive accuracy\n"
            f"50% and 90% pointwise intervals | {int(n_draws):,} draws",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            png_path = save_path.with_suffix(".png")
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            save_path = png_path

        summary = pd.DataFrame(
            [
                {
                    "subject_id": band["subject_id"],
                    "condition": band["condition"],
                    "band_type": band["band_type"],
                    "n_pf_runs": band["n_runs"],
                    "n_behavioral_draws": band["n_draws"],
                    "base_seed": band["base_seed"],
                    "subject_seed": band["seed"],
                    "window_size": band["window_size"],
                    "expected_curve_mae": band["expected_curve_mae"],
                    "coverage_50": band["coverage_50"],
                    "coverage_90": band["coverage_90"],
                    "mean_width_50": band["mean_width_50"],
                    "mean_width_90": band["mean_width_90"],
                    "plot_path": str(save_path) if save_path else "",
                }
                for band in bands
            ]
        )
        if summary_path:
            summary_path = Path(summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
        return summary

    @staticmethod
    def _format_strategy_phase(value: float) -> str:
        return "--" if not np.isfinite(value) else f"{100.0 * float(value):.1f}%"

    @plt.rc_context(
        {
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
    )
    def plot_particle_filter_dynamic_strategy_profile(
        self,
        results,
        subjects=None,
        save_path=None,
        *,
        summary_path=None,
        window_size=None,
        low_accuracy_threshold=0.60,
        mastery_accuracy_threshold=0.85,
        n_cols=None,
        max_subjects_per_row=4,
    ) -> pd.DataFrame:
        """Plot continuous exploit/local/global tendencies against performance."""
        visible = self._filter_results(results, subjects)
        grouped_info = self._group_by_condition(visible)
        if not grouped_info:
            raise ValueError("No particle-filter strategy results to plot")

        grouped: dict[int, list[tuple[int, Mapping[str, Any], dict[str, Any]]]] = (
            defaultdict(list)
        )
        for condition, rows in sorted(grouped_info.items()):
            for subject_id, info in rows:
                profile = self._particle_continuous_strategy_data(
                    info,
                    window_size=window_size,
                    low_accuracy_threshold=low_accuracy_threshold,
                    mastery_accuracy_threshold=mastery_accuracy_threshold,
                )
                grouped[int(condition)].append((int(subject_id), info, profile))

        layout_kwargs = {"max_subjects_per_row": max_subjects_per_row}
        if n_cols is not None:
            layout_kwargs["n_cols"] = int(n_cols)
        n_rows, n_cols, rows_by_condition = self._layout_by_condition(
            grouped,
            layout_kwargs,
        )
        # Reserve a fixed-height header so the shared legend does not collide
        # with the subtitle when a small subject set produces only one row.
        fig = plt.figure(figsize=(n_cols * 4.8, n_rows * 3.5 + 0.9))
        colors = ("#9DB9D8", "#83CDBA", "#E8A06C")
        labels = ("Exploit / retain", "Local exploration", "Global exploration")
        legend_items: dict[str, Any] = {}
        summary_rows: list[dict[str, Any]] = []

        row_offset = 0
        for condition, rows in sorted(grouped.items()):
            for index, (subject_id, info, profile) in enumerate(rows):
                local_row = index // n_cols
                col = index % n_cols
                ax = fig.add_subplot(
                    n_rows,
                    n_cols,
                    (row_offset + local_row) * n_cols + col + 1,
                )
                x = np.asarray(profile["trial"], dtype=float)
                strategy = np.asarray(profile["strategy"], dtype=float)
                ax.stackplot(
                    x,
                    strategy.T,
                    labels=labels,
                    colors=colors,
                    alpha=0.56,
                    linewidth=0,
                    zorder=1,
                )

                computed = self.compute_accuracy_metrics(
                    info,
                    window_size=int(profile["window_size"]),
                )
                observed_curve = computed.get("sliding_true_acc")
                predicted_curve = computed.get("sliding_pred_acc")
                if observed_curve is not None and predicted_curve is not None:
                    observed_curve = np.asarray(observed_curve, dtype=float)
                    predicted_curve = np.asarray(predicted_curve, dtype=float)
                    curve_n = min(observed_curve.size, predicted_curve.size)
                    curve_x = np.arange(
                        int(profile["window_size"]) + 1,
                        int(profile["window_size"]) + 1 + curve_n,
                    )
                    ax.plot(
                        curve_x,
                        predicted_curve[:curve_n],
                        color="#E69F00",
                        linewidth=1.8,
                        label="PF expected accuracy",
                        zorder=5,
                    )
                    ax.plot(
                        curve_x,
                        observed_curve[:curve_n],
                        color="#111111",
                        linewidth=2.0,
                        label="Subject accuracy",
                        zorder=6,
                    )

                realized = np.asarray(profile["realized"], dtype=float)
                if realized.size and np.any(np.isfinite(realized)):
                    realized_label = (
                        "Realized PF swap probability"
                        if "swap_event_probability" in str(profile["realized_field"])
                        else "Realized replacement fraction"
                    )
                    ax.plot(
                        x[: realized.size],
                        realized,
                        color="#7A5195",
                        linewidth=1.0,
                        linestyle=":",
                        alpha=0.85,
                        label=realized_label,
                        zorder=4,
                    )
                capture_hold = np.asarray(
                    profile["capture_hold_probability"],
                    dtype=float,
                )
                if (
                    profile["capture_diagnostics_available"]
                    and np.any(np.isfinite(capture_hold))
                ):
                    ax.plot(
                        x[: capture_hold.size],
                        capture_hold,
                        color="#C51B7D",
                        linewidth=1.8,
                        linestyle="--",
                        alpha=0.95,
                        label="Wrong-rule lock-in",
                        zorder=5,
                    )

                low_explore = float(profile["low_explore"])
                mastery_explore = float(profile["mastery_explore"])
                low_distance = float(profile["low_distance"])
                mastery_distance = float(profile["mastery_distance"])
                low_capture = float(profile["low_capture_hold"])
                mastery_capture = float(profile["mastery_capture_hold"])
                finite_capture = capture_hold[np.isfinite(capture_hold)]
                mean_capture = (
                    float(np.mean(finite_capture))
                    if finite_capture.size
                    else float("nan")
                )
                annotation = (
                    "Explore  low/mastery: "
                    f"{self._format_strategy_phase(low_explore)} / "
                    f"{self._format_strategy_phase(mastery_explore)}\n"
                    "Search distance: "
                    f"{self._format_strategy_phase(low_distance)} / "
                    f"{self._format_strategy_phase(mastery_distance)}\n"
                    "Lock-in: "
                    f"{self._format_strategy_phase(low_capture)} / "
                    f"{self._format_strategy_phase(mastery_capture)}"
                )
                ax.text(
                    0.015,
                    0.025,
                    annotation,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                    zorder=8,
                )
                self._add_particle_holdout_split(ax, info)
                ax.set(
                    title=f"Subject {subject_id} (M={profile['capacity']})",
                    xlabel="Trial",
                    ylabel="Accuracy / strategy tendency",
                    xlim=(1, max(1, int(x[-1]))),
                    ylim=(0.0, 1.0),
                )
                ax.grid(axis="x", alpha=0.12)
                handles, axis_labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, axis_labels):
                    legend_items.setdefault(label, handle)

                summary_rows.append(
                    {
                        "subject_id": int(subject_id),
                        "condition": int(condition),
                        "n_trials": int(strategy.shape[0]),
                        "capacity": int(profile["capacity"]),
                        "source_semantics": profile["source_semantics"],
                        "window_size": int(profile["window_size"]),
                        "low_accuracy_threshold": float(low_accuracy_threshold),
                        "mastery_accuracy_threshold": float(mastery_accuracy_threshold),
                        "low_trial_count": int(np.sum(profile["low_mask"])),
                        "mastery_trial_count": int(np.sum(profile["mastery_mask"])),
                        "mean_exploit": float(np.mean(strategy[:, 0])),
                        "mean_local_explore": float(np.mean(strategy[:, 1])),
                        "mean_global_explore": float(np.mean(strategy[:, 2])),
                        "low_explore": low_explore,
                        "mastery_explore": mastery_explore,
                        "low_local_explore": float(profile["low_local"]),
                        "mastery_local_explore": float(profile["mastery_local"]),
                        "low_global_explore": float(profile["low_global"]),
                        "mastery_global_explore": float(profile["mastery_global"]),
                        "low_conditional_newcomer_distance": low_distance,
                        "mastery_conditional_newcomer_distance": mastery_distance,
                        "low_failure_pressure": float(
                            profile["low_failure_pressure"]
                        ),
                        "mastery_failure_pressure": float(
                            profile["mastery_failure_pressure"]
                        ),
                        "low_mastery_evidence": float(
                            profile["low_mastery_evidence"]
                        ),
                        "mastery_mastery_evidence": float(
                            profile["mastery_mastery_evidence"]
                        ),
                        "low_prior_reset_strength": float(
                            profile["low_prior_reset_strength"]
                        ),
                        "mastery_prior_reset_strength": float(
                            profile["mastery_prior_reset_strength"]
                        ),
                        "low_prior_reset_mass_shift": float(
                            profile["low_prior_reset_mass_shift"]
                        ),
                        "mastery_prior_reset_mass_shift": float(
                            profile["mastery_prior_reset_mass_shift"]
                        ),
                        "low_misconception_capture_hold": low_capture,
                        "mastery_misconception_capture_hold": mastery_capture,
                        "mean_misconception_capture_hold": mean_capture,
                        "exploration_accuracy_correlation": float(
                            profile["exploration_accuracy_correlation"]
                        ),
                        "realized_field": profile["realized_field"] or "",
                        "newcomer_distance_field": profile["distance_field"] or "",
                        "plot_path": str(save_path) if save_path else "",
                    }
                )
            row_offset += rows_by_condition[condition]

        used_axes = sum(len(items) for items in grouped.values())
        for index in range(used_axes, n_rows * n_cols):
            ax = fig.add_subplot(n_rows, n_cols, index + 1)
            ax.axis("off")
        fig.suptitle(
            "Continuous Learning Strategy Profile",
            fontsize=12,
            y=0.985,
        )
        fig.text(
            0.5,
            0.94,
            "Exploit, local exploration, and global exploration",
            ha="center",
            va="top",
            fontsize=10,
        )
        if legend_items:
            fig.legend(
                list(legend_items.values()),
                list(legend_items.keys()),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.90),
                ncol=min(6, len(legend_items)),
                fontsize=8,
            )
        semantics = sorted(
            {
                profile["source_semantics"]
                for rows in grouped.values()
                for _, _, profile in rows
            }
        )
        fig.text(
            0.5,
            0.012,
            (
                "Phase definitions use only the previous "
                f"{int(next(iter(grouped.values()))[0][2]['window_size'])} trials: "
                f"low <= {float(low_accuracy_threshold):.2f}, "
                f"mastery >= {float(mastery_accuracy_threshold):.2f}. "
                f"Strategy source: {', '.join(semantics)}."
            ),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#444444",
        )
        axes_top = 0.80 if n_rows == 1 else 0.85
        fig.tight_layout(rect=(0.0, 0.045, 1.0, axes_top))
        resolved_save_path = None
        if save_path:
            resolved_save_path = Path(save_path).with_suffix(".png")
            resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(resolved_save_path, dpi=600, bbox_inches="tight")
            plt.close(fig)

        summary = pd.DataFrame(summary_rows)
        if resolved_save_path is not None:
            summary["plot_path"] = str(resolved_save_path)
        if summary_path:
            summary_path = Path(summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
        return summary

    def plot_particle_filter_ess(
        self,
        results,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot normalized pre/post-choice ESS and resampling events."""

        def body(ax, condition, iSub, info):
            pre = self._particle_series(info, "pre_choice_ess")
            post = self._particle_series(info, "post_choice_ess")
            resampled = self._particle_series(info, "resampled")
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
            self._add_particle_holdout_split(ax, info)
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
                ax.text(
                    0.5,
                    0.5,
                    "No particle ESS log",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.grid(axis="y", alpha=0.2)

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Particle-Filter Effective Sample Size",
            body,
            **kwargs,
        )

    @classmethod
    def _particle_filter_active_set_count_data(
        cls,
        info: Mapping[str, Any],
        *,
        window_size: int | None = None,
    ) -> dict[str, Any]:
        """Build fixed-capacity PF workspace-composition counts.

        Particle-filter transition logs do not contain the trajectory-only
        ``profile_policy.retained_count`` and ``newcomer_count`` fields. The
        expected PF counts instead follow directly from the particle-marginal
        replacement fraction: every replaced slot removes one incumbent and
        admits one newcomer while workspace capacity remains fixed.
        """
        capacity = cls._particle_capacity(info)
        replacement_fraction, source_field = cls._first_particle_series(
            info,
            "predictive_replacement_fraction",
            "replacement_fraction",
        )
        if replacement_fraction.size:
            expected_newcomers = replacement_fraction * float(capacity)
        else:
            replacement_count, source_field = cls._first_particle_series(
                info,
                "replacement_count",
            )
            if not replacement_count.size:
                raise ValueError(
                    "No particle-filter replacement fraction or count is available"
                )
            expected_newcomers = replacement_count

        finite = np.isfinite(expected_newcomers)
        if not np.any(finite):
            raise ValueError("Particle-filter replacement counts contain no finite values")
        tolerance = 1e-8
        if np.any(expected_newcomers[finite] < -tolerance) or np.any(
            expected_newcomers[finite] > float(capacity) + tolerance
        ):
            raise ValueError(
                "Particle-filter expected replacement count falls outside workspace capacity"
            )
        expected_newcomers = np.where(
            finite,
            np.clip(expected_newcomers, 0.0, float(capacity)),
            np.nan,
        )
        expected_retained = float(capacity) - expected_newcomers
        active_total = np.full(expected_newcomers.size, float(capacity), dtype=float)
        if not np.allclose(
            expected_retained[finite] + expected_newcomers[finite],
            active_total[finite],
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError("PF retained and newcomer counts do not sum to capacity")

        win = max(1, int(window_size or info.get("window_size") or 16))
        raw = pd.DataFrame(
            {
                "trial": np.arange(1, expected_newcomers.size + 1, dtype=int),
                "retained": expected_retained,
                "newcomer": expected_newcomers,
                "total": active_total,
            }
        )
        smooth = raw.copy()
        columns = ["retained", "newcomer", "total"]
        smooth[columns] = raw[columns].rolling(window=win, min_periods=1).mean()
        source_semantics = (
            "pre_choice_particle_marginal"
            if source_field == "predictive_replacement_fraction"
            else "post_choice_filtered_legacy"
        )
        return {
            "capacity": int(capacity),
            "window_size": int(win),
            "source_field": source_field,
            "source_semantics": source_semantics,
            "raw": raw,
            "smooth": smooth,
        }

    def plot_particle_filter_active_set_counts(
        self,
        results,
        window_size=None,
        subjects=None,
        save_path=None,
        **kwargs,
    ):
        """Plot expected retained/newcomer counts for fixed-capacity PF states."""

        def body(ax, condition, iSub, info):
            counts = self._particle_filter_active_set_count_data(
                info,
                window_size=window_size,
            )
            raw = counts["raw"]
            smooth = counts["smooth"]
            capacity = int(counts["capacity"])

            for field, color in (
                ("retained", "#009E73"),
                ("newcomer", "#D55E00"),
            ):
                ax.step(
                    raw["trial"],
                    raw[field],
                    where="mid",
                    color=color,
                    linewidth=0.8,
                    alpha=0.13,
                    zorder=1,
                )

            ax.plot(
                smooth["trial"],
                smooth["retained"],
                color="#009E73",
                linewidth=2.4,
                label="Expected retained",
                zorder=4,
            )
            ax.plot(
                smooth["trial"],
                smooth["newcomer"],
                color="#D55E00",
                linewidth=2.2,
                label="Expected newcomers",
                zorder=4,
            )
            ax.plot(
                smooth["trial"],
                smooth["total"],
                color="#0072B2",
                linewidth=1.8,
                linestyle="--",
                label=f"Fixed capacity = {capacity}",
                zorder=3,
            )
            self._add_particle_holdout_split(ax, info)
            source_label = (
                "pre-choice particle marginal"
                if counts["source_semantics"] == "pre_choice_particle_marginal"
                else "filtered legacy fallback"
            )
            ax.set(
                title=(
                    f"Subject {iSub} (Condition {condition}) | "
                    f"rolling window={counts['window_size']}\n{source_label}"
                ),
                xlabel="Trial",
                ylabel="Expected hypothesis count",
                ylim=(-0.08, float(capacity) + 0.35),
            )
            ax.legend(loc="best", fontsize=8)
            ax.grid(axis="y", alpha=0.22)

        kwargs.setdefault("max_subjects_per_row", 4)
        self._plot_by_condition(
            results,
            subjects,
            save_path,
            (
                "Particle-Filter Active-Workspace Composition\n"
                "Expected retained + expected newcomers = fixed capacity"
            ),
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
                ax.text(
                    0.5,
                    0.5,
                    "No marginal active-state data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
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
            ax.figure.colorbar(
                image,
                ax=ax,
                fraction=0.035,
                pad=0.02,
                label="P(active)",
            )

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Particle-Marginal Active-Hypothesis Probability",
            body,
            **kwargs,
        )


__all__ = ["ParticleFilterEvaluationMixin"]
