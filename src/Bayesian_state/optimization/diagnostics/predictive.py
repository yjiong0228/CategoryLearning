"""Predictive diagnostics that rerun selected hyperparameter candidates."""
from __future__ import annotations

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

from src.Bayesian_state.optimization.diagnostics.search import (
    DEFAULT_BASE_SIM_CONFIG,
    _subject_id_from_dir,
    discover_subject_dirs,
    load_all_combinations_table,
    load_strategy_lookup,
)
from src.Bayesian_state.optimization.artifacts import (
    subject_best_hyperparams,
    subject_hyper_candidate_seed,
)
from src.Bayesian_state.simulation.runner import StateModelSimulationRunner
from src.Bayesian_state.simulation.config import (
    DEFAULT_DATA_PATH,
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
from src.Bayesian_state.simulation.execution import evaluate_state_model_run
from src.Bayesian_state.utils.subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.seeding import (
    derive_simulation_point_seed,
    derive_trajectory_seed,
    stable_seed,
)


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
            "increase_rate",
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

    optimizer = SubjectTrialDataLoader(
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


__all__ = [
    "diagnose_hyper_accuracy_sampling",
    "evaluate_volatility_calibration",
    "select_accuracy_diagnostic_candidates",
]
