"""Coherent autonomous learning-trajectory ensembles and shape summaries.

Each rollout samples its own choices, receives feedback for those choices, and
updates its own state. This is distinct from an observed-history-conditional
Bernoulli accuracy band and from particle-filter repeat error.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..metrics.task import category_learning_metrics
from ..simulation.autonomous import run_autonomous_category_learning
from ..simulation.config import (
    DEFAULT_DATA_PATH,
    load_yaml,
    resolve_engine_config,
    resolve_window_size,
)
from ..simulation.data import SubjectTrialDataLoader, TrialArrays
from ..simulation.parameters import (
    apply_fixed_hyperparams_to_engine_config,
    apply_fixed_hyperparams_to_subject_config,
    infer_fixed_hyperparams_from_engine_config,
)
from ..utils.datasets import resolve_dataset_paths
from ..utils.paths import ROOT_DIR
from ..utils.seeding import stable_seed
from ..utils.subjects import resolve_subject_config


@dataclass(frozen=True)
class AutonomousEvaluationSpec:
    """Frozen model and task inputs resolved from one simulation config."""

    config_path: Path
    label: str
    subject_id: int
    condition: int
    window_size: int
    engine_config: dict[str, Any]
    fixed_hyperparams: dict[str, Any]
    arrays: TrialArrays
    processed_data_dir: Path
    dataset_paths: dict[str, Path]
    config_sha256: str
    engine_config_sha256: str


@dataclass(frozen=True)
class AutonomousEnsemble:
    """Minimal arrays retained from coherent autonomous rollouts."""

    choices: np.ndarray
    feedback: np.ndarray
    expected_correct_probability: np.ndarray
    trajectory_seeds: np.ndarray


@dataclass(frozen=True)
class TrajectoryShapeSummary:
    """Whole-curve summaries used by the autonomous trajectory figure."""

    trial: np.ndarray
    rolling_accuracy: np.ndarray
    rolling_expected_accuracy: np.ndarray
    observed_rolling_accuracy: np.ndarray
    medoid_index: int
    distance_to_medoid: np.ndarray
    central_50_indices: np.ndarray
    central_90_indices: np.ndarray
    central_50_lower: np.ndarray
    central_50_upper: np.ndarray
    central_90_lower: np.ndarray
    central_90_upper: np.ndarray
    cluster_labels: np.ndarray
    cluster_medoid_indices: np.ndarray
    cluster_shares: np.ndarray
    cluster_silhouette: float
    cluster_count: int
    mastery_onset: np.ndarray
    observed_mastery_onset: float
    overall_accuracy: np.ndarray
    final_block_accuracy: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return resolved.name


def load_autonomous_evaluation_spec(
    config_path: str | Path,
    *,
    subject_id: int,
    label: str | None = None,
    window_size: int | None = None,
) -> AutonomousEvaluationSpec:
    """Resolve the same engine, subject overrides, and trials as simulation."""

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Simulation config not found: {cfg_path}")
    cfg = load_yaml(cfg_path)
    sid = int(subject_id)
    subject_cfg = resolve_subject_config(cfg, sid)
    explicit_fixed = dict(subject_cfg.get("fixed_hyperparams") or {})
    subject_cfg = apply_fixed_hyperparams_to_subject_config(
        subject_cfg,
        explicit_fixed,
    )
    engine_config = resolve_engine_config(
        subject_cfg,
        cfg_path.parent,
        subject_id=sid,
    )
    fixed_hyperparams = {
        **infer_fixed_hyperparams_from_engine_config(engine_config),
        **explicit_fixed,
    }
    engine_config = apply_fixed_hyperparams_to_engine_config(
        engine_config,
        fixed_hyperparams,
    )
    dataset_paths = resolve_dataset_paths(
        subject_cfg,
        cfg_path.parent,
        DEFAULT_DATA_PATH,
    )
    loader = SubjectTrialDataLoader(
        engine_config=engine_config,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=1,
    )
    loader.prepare_data(dataset_paths["learning_data"])
    stop_at = float(subject_cfg.get("stop_at", 1.0))
    raw_max_trials = subject_cfg.get("max_trials")
    max_trials = int(raw_max_trials) if raw_max_trials is not None else None
    subject_frame = loader._get_subject_frame(sid, stop_at)
    arrays = loader._extract_arrays(subject_frame, max_trials)
    condition = loader._get_condition_value(subject_frame)
    if arrays.categories is None:
        raise ValueError(
            "Autonomous category-learning evaluation requires hard task categories."
        )
    if arrays.stimulus.shape[0] != arrays.categories.shape[0]:
        raise ValueError("Stimulus and category arrays are not trial-aligned.")
    resolved_window = (
        int(window_size)
        if window_size is not None
        else resolve_window_size(subject_cfg, sid, [sid])
    )
    if not 0 < resolved_window < arrays.stimulus.shape[0]:
        raise ValueError(
            "window_size must be positive and shorter than the full trajectory."
        )
    provenance = engine_config.get("provenance") or {}
    default_label = provenance.get("model_id") or cfg_path.stem
    return AutonomousEvaluationSpec(
        config_path=cfg_path,
        label=str(label or default_label),
        subject_id=sid,
        condition=int(condition),
        window_size=resolved_window,
        engine_config=engine_config,
        fixed_hyperparams=fixed_hyperparams,
        arrays=arrays,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        config_sha256=_sha256_file(cfg_path),
        engine_config_sha256=_sha256_mapping(engine_config),
    )


def _simulate_autonomous_minimal(
    *,
    engine_config: Mapping[str, Any],
    subject_id: int,
    condition: int,
    stimulus: np.ndarray,
    categories: np.ndarray,
    trajectory_seed: int,
    processed_data_dir: Path,
    dataset_paths: Mapping[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = run_autonomous_category_learning(
        engine_config=engine_config,
        subject_id=subject_id,
        condition=condition,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=trajectory_seed,
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    trajectory = result.trajectory
    category_index = np.asarray(categories, dtype=int) - 1
    expected_correct = trajectory.observed_probabilities[
        np.arange(category_index.size), category_index
    ]
    return (
        np.asarray(trajectory.choices, dtype=np.int16),
        np.asarray(trajectory.feedback, dtype=np.float32),
        np.asarray(expected_correct, dtype=np.float32),
    )


def generate_autonomous_ensemble(
    spec: AutonomousEvaluationSpec,
    *,
    rollout_count: int,
    analysis_seed: int,
    n_jobs: int = 1,
) -> AutonomousEnsemble:
    """Generate coherent, full-task virtual-subject trajectories."""

    count = int(rollout_count)
    if count < 2:
        raise ValueError("rollout_count must be at least 2.")
    seeds = np.asarray(
        [
            stable_seed(
                {
                    "seed_role": "autonomous_trajectory_ensemble",
                    "analysis_seed": int(analysis_seed),
                    "subject_id": int(spec.subject_id),
                    "rollout_index": int(index),
                }
            )
            for index in range(count)
        ],
        dtype=np.uint32,
    )
    if np.unique(seeds).size != seeds.size:
        raise RuntimeError("Autonomous trajectory seed collision detected.")
    categories = np.asarray(spec.arrays.categories, dtype=int)
    results = Parallel(n_jobs=int(n_jobs), prefer="processes")(
        delayed(_simulate_autonomous_minimal)(
            engine_config=spec.engine_config,
            subject_id=spec.subject_id,
            condition=spec.condition,
            stimulus=spec.arrays.stimulus,
            categories=categories,
            trajectory_seed=int(seed),
            processed_data_dir=spec.processed_data_dir,
            dataset_paths=spec.dataset_paths,
        )
        for seed in seeds
    )
    choices = np.stack([item[0] for item in results], axis=0)
    feedback = np.stack([item[1] for item in results], axis=0)
    expected_correct = np.stack([item[2] for item in results], axis=0)
    expected_shape = (count, int(spec.arrays.stimulus.shape[0]))
    if choices.shape != expected_shape or feedback.shape != expected_shape:
        raise RuntimeError("Autonomous rollout arrays have unexpected shapes.")
    if expected_correct.shape != expected_shape:
        raise RuntimeError("Expected-correct probability array has unexpected shape.")
    if not np.all(np.isfinite(expected_correct)):
        raise RuntimeError("Autonomous expected-correct probabilities are non-finite.")
    if np.any((expected_correct < 0.0) | (expected_correct > 1.0)):
        raise RuntimeError("Autonomous expected-correct probabilities fall outside [0, 1].")
    allowed_feedback = (0.0, 0.5, 1.0) if int(spec.condition) == 3 else (0.0, 1.0)
    if not np.all(np.isin(feedback, allowed_feedback)):
        raise RuntimeError(f"Autonomous feedback must be one of {allowed_feedback}.")
    return AutonomousEnsemble(
        choices=choices,
        feedback=feedback,
        expected_correct_probability=expected_correct,
        trajectory_seeds=seeds,
    )


def rolling_binary_ensemble(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the pipeline's established start-at-index-one rolling alignment."""

    array = np.asarray(values, dtype=float)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("values must have shape (trajectories, trials).")
    if not np.all(np.isfinite(array)):
        raise ValueError("rolling values contain non-finite entries.")
    window = int(window_size)
    if not 0 < window < array.shape[1]:
        raise ValueError("window_size must be positive and shorter than the trajectory.")
    starts = np.arange(1, array.shape[1] - window + 1, dtype=int)
    cumulative = np.pad(
        np.cumsum(array, axis=1),
        ((0, 0), (1, 0)),
        mode="constant",
    )
    curves = (
        cumulative[:, starts + window] - cumulative[:, starts]
    ) / float(window)
    trial = starts + window
    return trial, curves[0] if squeeze else curves


def sustained_mastery_onsets(
    curves: Sequence[Sequence[float]] | np.ndarray,
    trial: Sequence[int] | np.ndarray,
    *,
    threshold: float,
    sustain_windows: int,
) -> np.ndarray:
    """First rolling-window endpoint starting a sustained mastery period."""

    values = np.asarray(curves, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    x = np.asarray(trial, dtype=int).reshape(-1)
    if values.ndim != 2 or values.shape[1] != x.size:
        raise ValueError("curves and trial coordinates are not aligned.")
    sustain = int(sustain_windows)
    if sustain <= 0 or sustain > x.size:
        raise ValueError("sustain_windows must lie within the rolling trajectory.")
    threshold_value = float(threshold)
    if not 0.0 <= threshold_value <= 1.0:
        raise ValueError("mastery threshold must lie in [0, 1].")
    onsets = np.full(values.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(values):
        passed = np.asarray(row >= threshold_value, dtype=np.int16)
        sustained = np.convolve(
            passed,
            np.ones(sustain, dtype=np.int16),
            mode="valid",
        ) == sustain
        indices = np.flatnonzero(sustained)
        if indices.size:
            onsets[row_index] = float(x[int(indices[0])])
    return onsets[0] if squeeze else onsets


def _whole_curve_central_regions(
    curves: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    distances = cdist(curves, curves, metric="cityblock") / float(curves.shape[1])
    medoid_index = int(np.argmin(np.mean(distances, axis=1)))
    distance_to_medoid = distances[:, medoid_index]
    order = np.lexsort((np.arange(curves.shape[0]), distance_to_medoid))
    central_50 = np.sort(order[: max(1, int(math.ceil(curves.shape[0] * 0.50)))])
    central_90 = np.sort(order[: max(1, int(math.ceil(curves.shape[0] * 0.90)))])
    return medoid_index, distance_to_medoid, central_50, central_90


def _cluster_trajectory_shapes(
    curves: np.ndarray,
    mastery_onset: np.ndarray,
    final_accuracy: np.ndarray,
    *,
    max_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    n_trajectories = int(curves.shape[0])
    if n_trajectories < 4 or np.max(np.std(curves, axis=0)) < 1e-10:
        labels = np.zeros(n_trajectories, dtype=int)
        return labels, np.asarray([0], dtype=int), np.asarray([1.0]), float("nan"), 1

    upper = max(2, min(int(max_clusters), n_trajectories - 1))
    min_cluster_size = max(2, int(math.ceil(0.05 * n_trajectories)))
    candidates: list[tuple[float, int, np.ndarray]] = []
    fallback: list[tuple[float, int, np.ndarray]] = []
    for cluster_count in range(2, upper + 1):
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=int(random_state),
            n_init=30,
        ).fit_predict(curves).astype(int)
        sizes = np.bincount(labels, minlength=cluster_count)
        if np.any(sizes == 0):
            continue
        score = float(silhouette_score(curves, labels, metric="euclidean"))
        item = (score, cluster_count, labels)
        fallback.append(item)
        if int(np.min(sizes)) >= min_cluster_size:
            candidates.append(item)
    usable = candidates or fallback
    if not usable:
        labels = np.zeros(n_trajectories, dtype=int)
        return labels, np.asarray([0], dtype=int), np.asarray([1.0]), float("nan"), 1
    score, cluster_count, raw_labels = max(
        usable,
        key=lambda item: (item[0], -item[1]),
    )

    ordering: list[tuple[tuple[float, float, float], int]] = []
    for raw_label in range(cluster_count):
        indices = np.flatnonzero(raw_labels == raw_label)
        finite_onset = mastery_onset[indices][np.isfinite(mastery_onset[indices])]
        median_onset = (
            float(np.median(finite_onset)) if finite_onset.size else float("inf")
        )
        median_final = float(np.median(final_accuracy[indices]))
        key = (
            0.0 if np.isfinite(median_onset) else 1.0,
            median_onset,
            -median_final,
        )
        ordering.append((key, raw_label))
    ordering.sort(key=lambda item: item[0])
    remap = {
        raw_label: new_label
        for new_label, (_, raw_label) in enumerate(ordering)
    }
    labels = np.asarray([remap[int(value)] for value in raw_labels], dtype=int)

    medoids: list[int] = []
    shares: list[float] = []
    for label in range(cluster_count):
        indices = np.flatnonzero(labels == label)
        distances = cdist(curves[indices], curves[indices], metric="cityblock")
        local_index = int(np.argmin(np.mean(distances, axis=1)))
        medoids.append(int(indices[local_index]))
        shares.append(float(indices.size / n_trajectories))
    return (
        labels,
        np.asarray(medoids, dtype=int),
        np.asarray(shares, dtype=float),
        score,
        int(cluster_count),
    )


def summarize_trajectory_shapes(
    ensemble: AutonomousEnsemble,
    *,
    observed_feedback: Sequence[float] | np.ndarray,
    window_size: int,
    mastery_threshold: float = 0.80,
    mastery_sustain_windows: int = 8,
    final_block_trials: int = 64,
    max_clusters: int = 4,
    cluster_seed: int = 20260831,
) -> TrajectoryShapeSummary:
    """Summarize species-accuracy curves; retain graded task rewards separately."""

    species_success = np.asarray(ensemble.feedback, dtype=float) == 1.0
    trial, rolling = rolling_binary_ensemble(
        species_success,
        window_size=window_size,
    )
    expected_trial, rolling_expected = rolling_binary_ensemble(
        ensemble.expected_correct_probability,
        window_size=window_size,
    )
    observed_trial, observed_rolling = rolling_binary_ensemble(
        np.asarray(observed_feedback, dtype=float) == 1.0,
        window_size=window_size,
    )
    if not np.array_equal(trial, expected_trial) or not np.array_equal(
        trial,
        observed_trial,
    ):
        raise RuntimeError("Rolling trajectory coordinates are inconsistent.")
    medoid, distance, central_50, central_90 = _whole_curve_central_regions(rolling)
    mastery_onset = sustained_mastery_onsets(
        rolling,
        trial,
        threshold=mastery_threshold,
        sustain_windows=mastery_sustain_windows,
    )
    observed_onset = float(
        sustained_mastery_onsets(
            observed_rolling,
            trial,
            threshold=mastery_threshold,
            sustain_windows=mastery_sustain_windows,
        )
    )
    final_count = min(max(1, int(final_block_trials)), ensemble.feedback.shape[1])
    overall_accuracy = np.mean(species_success, axis=1)
    final_accuracy = np.mean(species_success[:, -final_count:], axis=1)
    labels, cluster_medoids, shares, silhouette, cluster_count = (
        _cluster_trajectory_shapes(
            rolling,
            np.asarray(mastery_onset, dtype=float),
            final_accuracy,
            max_clusters=max_clusters,
            random_state=cluster_seed,
        )
    )
    return TrajectoryShapeSummary(
        trial=trial,
        rolling_accuracy=rolling,
        rolling_expected_accuracy=rolling_expected,
        observed_rolling_accuracy=observed_rolling,
        medoid_index=medoid,
        distance_to_medoid=distance,
        central_50_indices=central_50,
        central_90_indices=central_90,
        central_50_lower=np.min(rolling[central_50], axis=0),
        central_50_upper=np.max(rolling[central_50], axis=0),
        central_90_lower=np.min(rolling[central_90], axis=0),
        central_90_upper=np.max(rolling[central_90], axis=0),
        cluster_labels=labels,
        cluster_medoid_indices=cluster_medoids,
        cluster_shares=shares,
        cluster_silhouette=silhouette,
        cluster_count=cluster_count,
        mastery_onset=np.asarray(mastery_onset, dtype=float),
        observed_mastery_onset=observed_onset,
        overall_accuracy=overall_accuracy,
        final_block_accuracy=final_accuracy,
    )


def _configure_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def plot_autonomous_trajectory_ensemble(
    spec: AutonomousEvaluationSpec,
    ensemble: AutonomousEnsemble,
    summary: TrajectoryShapeSummary,
    *,
    save_path: Path,
    visible_trajectories: int = 48,
    plot_seed: int = 20260831,
    mastery_threshold: float = 0.80,
    mastery_sustain_windows: int = 8,
) -> np.ndarray:
    """Draw the hero ensemble, cluster archetypes, and mastery-onset CDF."""

    _configure_figure_style()
    species_chance = 0.5 if int(spec.condition) == 1 else 0.25
    colors = ["#4878A8", "#D9822B", "#6A9F58", "#9A6FB0"]
    x = summary.trial
    n_rollouts = summary.rolling_accuracy.shape[0]
    visible_count = min(max(1, int(visible_trajectories)), n_rollouts)
    rng = np.random.default_rng(int(plot_seed))
    visible_indices = np.sort(
        rng.choice(n_rollouts, size=visible_count, replace=False)
    )

    fig = plt.figure(figsize=(10.0, 7.2), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.55, 1.0),
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.12,
        hspace=0.42,
        wspace=0.30,
    )
    ax_ensemble = fig.add_subplot(grid[0, :])
    ax_clusters = fig.add_subplot(grid[1, 0])
    ax_mastery = fig.add_subplot(grid[1, 1])

    ax_ensemble.fill_between(
        x,
        summary.central_90_lower,
        summary.central_90_upper,
        color="#DCE8F2",
        alpha=0.75,
        linewidth=0,
        label="Central 90% of whole trajectories",
    )
    ax_ensemble.fill_between(
        x,
        summary.central_50_lower,
        summary.central_50_upper,
        color="#AFC9DF",
        alpha=0.80,
        linewidth=0,
        label="Central 50% of whole trajectories",
    )
    for index in visible_indices:
        ax_ensemble.plot(
            x,
            summary.rolling_accuracy[index],
            color="#477DAA",
            alpha=0.13,
            linewidth=0.65,
            zorder=2,
        )
    ax_ensemble.plot(
        x,
        summary.rolling_accuracy[summary.medoid_index],
        color="#D9822B",
        linewidth=2.0,
        label="Autonomous medoid",
        zorder=4,
    )
    ax_ensemble.plot(
        x,
        summary.observed_rolling_accuracy,
        color="#111111",
        linewidth=2.2,
        label="Observed subject",
        zorder=5,
    )
    ax_ensemble.axhline(
        species_chance,
        color="#888888",
        linewidth=0.8,
        linestyle=":",
        zorder=1,
    )
    ax_ensemble.set(
        xlim=(1, int(spec.arrays.stimulus.shape[0])),
        ylim=(0, 1.02),
        xlabel="Trial",
        ylabel=f"Rolling accuracy (window = {spec.window_size})",
    )
    ax_ensemble.set_title(
        "a  Coherent autonomous learning trajectories",
        loc="left",
        fontweight="bold",
    )
    ax_ensemble.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    handles, labels = ax_ensemble.get_legend_handles_labels()
    ax_ensemble.legend(
        handles[::-1],
        labels[::-1],
        loc="lower right",
        ncol=2,
        handlelength=2.8,
    )

    for cluster_label, medoid_index in enumerate(summary.cluster_medoid_indices):
        onset_values = summary.mastery_onset[
            summary.cluster_labels == cluster_label
        ]
        finite_onsets = onset_values[np.isfinite(onset_values)]
        onset_label = (
            f"median onset {int(round(float(np.median(finite_onsets))))}"
            if finite_onsets.size
            else "criterion not reached"
        )
        ax_clusters.plot(
            x,
            summary.rolling_accuracy[int(medoid_index)],
            color=colors[cluster_label % len(colors)],
            linewidth=1.8,
            label=(
                f"C{cluster_label + 1}: {summary.cluster_shares[cluster_label] * 100:.1f}%"
                f" | {onset_label}"
            ),
        )
    ax_clusters.plot(
        x,
        summary.observed_rolling_accuracy,
        color="#111111",
        linewidth=1.4,
        linestyle="--",
        label="Observed subject",
    )
    ax_clusters.axhline(species_chance, color="#888888", linewidth=0.7, linestyle=":")
    ax_clusters.set(
        xlim=(1, int(spec.arrays.stimulus.shape[0])),
        ylim=(0, 1.02),
        xlabel="Trial",
        ylabel="Rolling accuracy",
    )
    ax_clusters.set_title(
        f"b  Trajectory archetypes (k = {summary.cluster_count})",
        loc="left",
        fontweight="bold",
    )
    ax_clusters.grid(axis="y", color="#E1E1E1", linewidth=0.55, alpha=0.65)
    ax_clusters.legend(loc="lower right", fontsize=6.4, handlelength=2.4)

    finite_onsets = np.sort(
        summary.mastery_onset[np.isfinite(summary.mastery_onset)]
    )
    if finite_onsets.size:
        cumulative = np.arange(1, finite_onsets.size + 1, dtype=float) / float(
            n_rollouts
        )
        ax_mastery.step(
            np.r_[x[0], finite_onsets],
            np.r_[0.0, cumulative],
            where="post",
            color="#4878A8",
            linewidth=2.0,
            label="Autonomous trajectories",
        )
    if np.isfinite(summary.observed_mastery_onset):
        ax_mastery.axvline(
            summary.observed_mastery_onset,
            color="#111111",
            linewidth=1.6,
            linestyle="--",
            label=f"Observed onset: {int(round(summary.observed_mastery_onset))}",
        )
    not_reached = float(np.mean(~np.isfinite(summary.mastery_onset)))
    ax_mastery.text(
        0.98,
        0.94,
        f"Not reached: {not_reached * 100:.1f}%",
        transform=ax_mastery.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )
    ax_mastery.set(
        xlim=(1, int(spec.arrays.stimulus.shape[0])),
        ylim=(0, 1.02),
        xlabel="Trial of sustained-mastery onset",
        ylabel="Cumulative share of all rollouts",
    )
    ax_mastery.set_title(
        (
            "c  Sustained mastery onset "
            f"(accuracy ≥ {mastery_threshold:.2f} for "
            f"{mastery_sustain_windows} windows)"
        ),
        loc="left",
        fontweight="bold",
        fontsize=8.4,
    )
    ax_mastery.grid(color="#E1E1E1", linewidth=0.55, alpha=0.65)
    if finite_onsets.size or np.isfinite(summary.observed_mastery_onset):
        ax_mastery.legend(loc="lower right")

    fig.suptitle(
        (
            f"Subject {spec.subject_id} | {spec.label}\n"
            f"Autonomous learning-trajectory distribution "
            f"({n_rollouts} full rollouts)"
        ),
        x=0.075,
        y=0.975,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.035,
        (
            f"All {spec.arrays.stimulus.shape[0]} task trials were simulated. "
            f"Curves use the pipeline's {spec.window_size}-trial rolling alignment "
            f"(first point: trial {spec.window_size + 1}). Central regions rank "
            "complete curves by distance to the medoid; clusters are descriptive. "
            "Parameters are fixed, so the ensemble reflects latent/process and "
            "choice variation, not parameter uncertainty."
        ),
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#4D4D4D",
        wrap=True,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return visible_indices


def _trajectory_summary_frame(
    ensemble: AutonomousEnsemble,
    summary: TrajectoryShapeSummary,
    *,
    categories: np.ndarray | None = None,
    n_categories: int = 2,
) -> pd.DataFrame:
    medoid_for = {
        int(index): int(label) + 1
        for label, index in enumerate(summary.cluster_medoid_indices)
    }
    frame = pd.DataFrame(
        {
            "rollout_index": np.arange(ensemble.feedback.shape[0], dtype=int),
            "trajectory_seed": ensemble.trajectory_seeds.astype(np.uint64),
            "cluster": summary.cluster_labels.astype(int) + 1,
            "overall_accuracy": summary.overall_accuracy,
            "final_block_accuracy": summary.final_block_accuracy,
            "sustained_mastery_onset": summary.mastery_onset,
            "distance_to_overall_medoid": summary.distance_to_medoid,
            "is_overall_medoid": (
                np.arange(ensemble.feedback.shape[0]) == summary.medoid_index
            ),
            "cluster_medoid_for": [
                medoid_for.get(index, np.nan)
                for index in range(ensemble.feedback.shape[0])
            ],
        }
    )
    if categories is not None:
        task_metrics = pd.DataFrame(
            [
                category_learning_metrics(
                    choices=choices,
                    categories=categories,
                    feedback=feedback,
                    n_categories=n_categories,
                )
                for choices, feedback in zip(ensemble.choices, ensemble.feedback)
            ]
        )
        frame = pd.concat([frame, task_metrics], axis=1)
    return frame


def _plot_source_frame(
    summary: TrajectoryShapeSummary,
    visible_indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_curve(
        kind: str,
        curve_id: str,
        values: np.ndarray,
        cluster: int | None = None,
        share: float | None = None,
    ) -> None:
        rows.extend(
            {
                "curve_kind": kind,
                "curve_id": curve_id,
                "cluster": cluster,
                "cluster_share": share,
                "trial": int(trial),
                "rolling_accuracy": float(value),
            }
            for trial, value in zip(summary.trial, values)
        )

    add_curve("observed", "observed_subject", summary.observed_rolling_accuracy)
    add_curve(
        "overall_medoid",
        f"rollout_{summary.medoid_index}",
        summary.rolling_accuracy[summary.medoid_index],
    )
    for name, values in (
        ("central_50_lower", summary.central_50_lower),
        ("central_50_upper", summary.central_50_upper),
        ("central_90_lower", summary.central_90_lower),
        ("central_90_upper", summary.central_90_upper),
    ):
        add_curve("central_region", name, values)
    for label, index in enumerate(summary.cluster_medoid_indices):
        add_curve(
            "cluster_medoid",
            f"cluster_{label + 1}_rollout_{int(index)}",
            summary.rolling_accuracy[int(index)],
            label + 1,
            float(summary.cluster_shares[label]),
        )
    for index in visible_indices:
        add_curve(
            "displayed_rollout",
            f"rollout_{int(index)}",
            summary.rolling_accuracy[int(index)],
            int(summary.cluster_labels[int(index)]) + 1,
        )
    return pd.DataFrame(rows)


def save_autonomous_trajectory_evaluation(
    spec: AutonomousEvaluationSpec,
    ensemble: AutonomousEnsemble,
    summary: TrajectoryShapeSummary,
    *,
    output_dir: str | Path,
    analysis_seed: int,
    visible_trajectories: int,
    plot_seed: int,
    mastery_threshold: float,
    mastery_sustain_windows: int,
    final_block_trials: int,
    max_clusters: int,
) -> dict[str, Path]:
    """Save one PNG plus compact source arrays, tables, and provenance."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing autonomous evaluation directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    figure_path = output / f"subject_{spec.subject_id}_autonomous_trajectories.png"
    visible_indices = plot_autonomous_trajectory_ensemble(
        spec,
        ensemble,
        summary,
        save_path=figure_path,
        visible_trajectories=visible_trajectories,
        plot_seed=plot_seed,
        mastery_threshold=mastery_threshold,
        mastery_sustain_windows=mastery_sustain_windows,
    )
    array_path = output / "autonomous_trajectory_arrays.npz"
    np.savez_compressed(
        array_path,
        choices=ensemble.choices,
        feedback=ensemble.feedback,
        expected_correct_probability=ensemble.expected_correct_probability,
        trajectory_seeds=ensemble.trajectory_seeds,
        rolling_trial=summary.trial,
        rolling_accuracy=summary.rolling_accuracy.astype(np.float32),
        rolling_expected_accuracy=summary.rolling_expected_accuracy.astype(np.float32),
        observed_feedback=np.asarray(spec.arrays.feedback, dtype=np.float32),
        observed_rolling_accuracy=summary.observed_rolling_accuracy.astype(np.float32),
        cluster_labels=summary.cluster_labels,
        cluster_medoid_indices=summary.cluster_medoid_indices,
        central_50_indices=summary.central_50_indices,
        central_90_indices=summary.central_90_indices,
    )
    summary_path = output / "autonomous_trajectory_summary.csv"
    _trajectory_summary_frame(
        ensemble,
        summary,
        categories=spec.arrays.categories,
        n_categories=2 if spec.condition == 1 else 4,
    ).to_csv(summary_path, index=False)
    source_path = output / "autonomous_trajectory_plot_source.csv"
    _plot_source_frame(summary, visible_indices).to_csv(source_path, index=False)

    finite_onsets = summary.mastery_onset[np.isfinite(summary.mastery_onset)]
    manifest = {
        "analysis": "autonomous_learning_trajectory_distribution",
        "interpretation": (
            "Each rollout samples its own choices and learns from its own feedback. "
            "This is not an observed-history-conditional Bernoulli band and not "
            "particle-filter repeat error."
        ),
        "subject_id": int(spec.subject_id),
        "condition": int(spec.condition),
        "metric_definitions": {
            "species_accuracy": "mean(choice == category)",
            "family_accuracy": "mean(task family(choice) == task family(category))",
            "mean_reward": "mean(raw feedback)",
            "accuracy_curves": "species accuracy: feedback == 1",
        },
        "label": spec.label,
        "config_path": _project_relative(spec.config_path),
        "config_sha256": spec.config_sha256,
        "resolved_engine_config_sha256": spec.engine_config_sha256,
        "dataset": {
            key: _project_relative(path) for key, path in spec.dataset_paths.items()
        },
        "fixed_hyperparams": spec.fixed_hyperparams,
        "trial_count": int(spec.arrays.stimulus.shape[0]),
        "rollout_count": int(ensemble.feedback.shape[0]),
        "window_size": int(spec.window_size),
        "final_block_trials": int(final_block_trials),
        "rolling_alignment": {
            "start_index_zero_based": 1,
            "first_displayed_trial": int(spec.window_size + 1),
            "last_displayed_trial": int(spec.arrays.stimulus.shape[0]),
        },
        "analysis_seed": int(analysis_seed),
        "plot_seed": int(plot_seed),
        "visible_trajectory_count": int(visible_indices.size),
        "whole_curve_region": {
            "distance": "mean_absolute_distance_across_complete_rolling_curve",
            "central_50_count": int(summary.central_50_indices.size),
            "central_90_count": int(summary.central_90_indices.size),
            "overall_medoid_rollout_index": int(summary.medoid_index),
        },
        "clustering": {
            "method": "kmeans_on_complete_rolling_accuracy_curves",
            "candidate_cluster_counts": list(range(2, int(max_clusters) + 1)),
            "selected_cluster_count": int(summary.cluster_count),
            "silhouette": (
                None
                if not np.isfinite(summary.cluster_silhouette)
                else float(summary.cluster_silhouette)
            ),
            "cluster_shares": summary.cluster_shares.astype(float).tolist(),
            "cluster_medoid_rollout_indices": (
                summary.cluster_medoid_indices.astype(int).tolist()
            ),
        },
        "mastery": {
            "threshold": float(mastery_threshold),
            "sustain_rolling_windows": int(mastery_sustain_windows),
            "reached_count": int(finite_onsets.size),
            "not_reached_count": int(np.sum(~np.isfinite(summary.mastery_onset))),
            "median_onset": (
                None if not finite_onsets.size else float(np.median(finite_onsets))
            ),
            "observed_onset": (
                None
                if not np.isfinite(summary.observed_mastery_onset)
                else float(summary.observed_mastery_onset)
            ),
        },
        "uncertainty_scope": (
            "Fixed subject-level parameters; latent/process and sampled-choice "
            "variation only. Parameter uncertainty is not included."
        ),
        "outputs": {
            "figure": figure_path.name,
            "arrays": array_path.name,
            "trajectory_summary": summary_path.name,
            "plot_source": source_path.name,
        },
    }
    manifest_path = output / "analysis_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str)
    return {
        "figure": figure_path,
        "arrays": array_path,
        "trajectory_summary": summary_path,
        "plot_source": source_path,
        "manifest": manifest_path,
    }


def run_autonomous_trajectory_evaluation(
    *,
    config_path: str | Path,
    subject_id: int,
    output_dir: str | Path,
    label: str | None = None,
    rollout_count: int = 500,
    n_jobs: int = 1,
    analysis_seed: int = 20260831,
    window_size: int | None = None,
    visible_trajectories: int = 48,
    mastery_threshold: float = 0.80,
    mastery_sustain_windows: int = 8,
    final_block_trials: int = 64,
    max_clusters: int = 4,
) -> dict[str, Path]:
    """Complete fixed-parameter autonomous trajectory evaluation."""

    spec = load_autonomous_evaluation_spec(
        config_path,
        subject_id=subject_id,
        label=label,
        window_size=window_size,
    )
    ensemble = generate_autonomous_ensemble(
        spec,
        rollout_count=rollout_count,
        analysis_seed=analysis_seed,
        n_jobs=n_jobs,
    )
    summary = summarize_trajectory_shapes(
        ensemble,
        observed_feedback=spec.arrays.feedback,
        window_size=spec.window_size,
        mastery_threshold=mastery_threshold,
        mastery_sustain_windows=mastery_sustain_windows,
        final_block_trials=final_block_trials,
        max_clusters=max_clusters,
        cluster_seed=analysis_seed,
    )
    return save_autonomous_trajectory_evaluation(
        spec,
        ensemble,
        summary,
        output_dir=output_dir,
        analysis_seed=analysis_seed,
        visible_trajectories=visible_trajectories,
        plot_seed=analysis_seed,
        mastery_threshold=mastery_threshold,
        mastery_sustain_windows=mastery_sustain_windows,
        final_block_trials=final_block_trials,
        max_clusters=max_clusters,
    )


__all__ = [
    "AutonomousEnsemble",
    "AutonomousEvaluationSpec",
    "TrajectoryShapeSummary",
    "generate_autonomous_ensemble",
    "load_autonomous_evaluation_spec",
    "plot_autonomous_trajectory_ensemble",
    "rolling_binary_ensemble",
    "run_autonomous_trajectory_evaluation",
    "save_autonomous_trajectory_evaluation",
    "summarize_trajectory_shapes",
    "sustained_mastery_onsets",
]
