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
from src.Bayesian_state.utils.optimization_config import (
    DEFAULT_DATA_PATH,
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.utils.optimizer_common import (
    derive_simulation_point_seed,
    derive_trajectory_seed,
    evaluate_state_model_run,
)
from src.Bayesian_state.utils.optimizer_simulation import StateModelSimulationRunner


MEMORY_KEY = "engine.modules.memory_mod.kwargs"
TRANSITION_KEY = "engine.modules.hypo_transitions_mod.kwargs"
BETA_INIT_KEY = "engine.modules.beta_mod.kwargs.beta_init"
DECREASE_RATE_KEY = "engine.modules.beta_mod.kwargs.decrease_rate"
PRIOR_BETA_SCALE_KEY = "engine.modules.beta_mod.kwargs.prior_beta_scale"
CORRECT_ADDITIVE_KEY = "engine.modules.beta_mod.kwargs.correct_additive"
DISTANCE_MODE_KEY = "engine.modules.likelihood_mod.kwargs.distance_mode"

NUMERIC_PARAM_COLUMNS = (
    "gamma",
    "w0",
    "beta_init",
    "decrease_rate",
    "prior_beta_scale",
    "correct_additive",
)
CATEGORICAL_PARAM_COLUMNS = ("strategy_id", "distance_mode")

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
        metrics = row.get("metrics_summary")
        if not isinstance(hp, Mapping) or not isinstance(metrics, Mapping):
            continue
        flat = flatten_hyperparams(hp, strategy_lookup)
        shape_metrics = {
            key: _safe_float(metrics.get(key))
            for key in ACCURACY_SHAPE_COLUMNS
        }
        history_metrics = {
            key: (
                metrics.get(key)
                if key in {"history_kernel_human", "history_kernel_model"}
                else _safe_float(metrics.get(key))
            )
            for key in HISTORY_KERNEL_COLUMNS
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
                "hyper_selection_error": _safe_float(metrics.get("selection_error", row.get("aggregated_error"))),
                "hyper_mean_error": _safe_float(metrics.get("mean_error")),
                "hyper_best_error": _safe_float(metrics.get("best_error")),
                "hyper_best10_mean_error": _safe_float(metrics.get("best10_mean_error")),
                "hyper_q10_error": _safe_float(metrics.get("q10_error")),
                "hyper_std_error": _safe_float(metrics.get("std_error")),
                "hyper_simulation_repeats": metrics.get("simulation_repeats"),
                **flat,
                **shape_metrics,
                **history_metrics,
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
        default=Path("src/Bayesian_state/problems/modules/hypo_transition_strategy_candidates.json"),
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
