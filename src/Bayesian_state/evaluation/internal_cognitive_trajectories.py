"""Observed-history-conditioned internal cognitive trajectory analysis.

The analysis traces complete bootstrap-particle genealogies after conditioning
on one subject's observed choices. Particle importance weights remain a
numerical inference device: figures use equal-weight draws from the terminal
genealogical approximation and explicitly report ancestry collapse.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ..utils.parallel import parallel_job_count, single_threaded_processes
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from ..inference.dispatcher import run_inference_backend
from ..model import StateModel
from ..model.config import ModelContext
from ..model.readout import (
    resolve_choice_readout_config,
    resolve_output_noise_config,
)
from ..utils.paths import ROOT_DIR
from ..utils.seeding import stable_seed
from .autonomous_trajectories import (
    AutonomousEvaluationSpec,
    load_autonomous_evaluation_spec,
)


PATH_FIELDS = (
    "correct_probability",
    "observed_choice_probability",
    "strategy_exploit",
    "strategy_local_explore",
    "strategy_global_explore",
    "swap_event",
    "replacement_fraction",
    "transition_rate",
    "search_range",
    "failure_pressure",
    "mastery_evidence",
    "hypothesis_prior",
    "hypothesis_posterior",
    "active_hypothesis_mask",
    "executed_hypothesis",
    "execution_switch_event",
    "execution_dwell_trials",
    "executed_beta",
)

EXECUTION_PATH_FIELDS = (
    "executed_hypothesis", "execution_switch_event", "execution_dwell_trials",
    "executed_beta",
)

FAMILY_ORDER = (
    "univariate_threshold",
    "pairwise_order",
    "pairwise_sum_threshold",
    "paired_sum_order",
    "pairwise_similarity_band",
    "univariate_center_band",
)
FAMILY_LABELS = {
    "univariate_threshold": "Single-feature threshold",
    "pairwise_order": "Feature comparison",
    "pairwise_sum_threshold": "Two-feature sum",
    "paired_sum_order": "Paired-sum comparison",
    "pairwise_similarity_band": "Feature similarity",
    "univariate_center_band": "Feature center band",
}
FAMILY_COLORS = {
    "univariate_threshold": "#4C78A8",
    "pairwise_order": "#F2A65A",
    "pairwise_sum_threshold": "#59A14F",
    "paired_sum_order": "#9C755F",
    "pairwise_similarity_band": "#B279A2",
    "univariate_center_band": "#76B7B2",
}


@dataclass(frozen=True)
class CognitivePathEnsemble:
    """Combined weighted genealogies and online filtering summaries."""

    spec: AutonomousEvaluationSpec
    filter_seeds: np.ndarray
    particle_count: int
    weights: np.ndarray
    seed_index: np.ndarray
    terminal_particle: np.ndarray
    particle_indices: np.ndarray
    paths: Mapping[str, np.ndarray]
    marginal_choice_probability: np.ndarray
    online_hypothesis_prior: np.ndarray
    online_active_probability: np.ndarray
    online_executed_probability: np.ndarray | None
    online_swap_probability: np.ndarray
    pre_choice_ess: np.ndarray
    post_choice_ess: np.ndarray
    resampled: np.ndarray


@dataclass(frozen=True)
class CognitivePathSummary:
    """Equal-weight path draws, archetypes, and genealogy diagnostics."""

    sample_source_index: np.ndarray
    sampled_paths: Mapping[str, np.ndarray]
    representative_index: int
    cluster_labels: np.ndarray
    cluster_medoid_indices: np.ndarray
    cluster_shares: np.ndarray
    cluster_silhouette: float
    cluster_count: int
    ancestor_unique_count: np.ndarray
    ancestor_effective_count: np.ndarray
    per_seed_start_ancestor_count: np.ndarray
    genealogy_status: str
    genealogy_message: str


@dataclass(frozen=True)
class CompletePathSelection:
    """One fixed genealogy selected by full-sequence choice likelihood."""

    path_index: int
    sequence_nll: float
    per_path_sequence_nll: np.ndarray


@dataclass(frozen=True)
class FixedPathFitSummary:
    """Behavioral fit of one unchanged genealogy over the full experiment."""

    rolling_human_accuracy: np.ndarray
    rolling_model_correct_probability: np.ndarray
    empirical_accuracy: float
    model_expected_accuracy: float
    sequence_nll: float
    mean_trial_nll: float
    rolling_rmse: float
    rolling_correlation: float


def select_best_complete_path(
    observed_choice_probability: np.ndarray,
) -> CompletePathSelection:
    """Select one path once using its likelihood over the complete sequence."""

    probability = np.asarray(observed_choice_probability, dtype=float)
    if probability.ndim != 2 or 0 in probability.shape:
        raise ValueError(
            "observed_choice_probability must have shape (paths, trials)."
        )
    if (
        not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError(
            "observed_choice_probability must contain finite probabilities in [0, 1]."
        )
    clipped = np.clip(probability, np.finfo(float).tiny, 1.0)
    per_path_nll = -np.sum(np.log(clipped), axis=1)
    best = int(np.argmin(per_path_nll))
    return CompletePathSelection(
        path_index=best,
        sequence_nll=float(per_path_nll[best]),
        per_path_sequence_nll=per_path_nll,
    )


def _normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if (
        array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
        or float(np.sum(array)) <= 0.0
    ):
        raise ValueError("weights must be finite, non-negative, and non-empty.")
    return array / float(np.sum(array))


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return resolved.name


def generate_filter_seeds(
    *, analysis_seed: int, subject_id: int, seed_count: int
) -> np.ndarray:
    count = int(seed_count)
    if count < 2:
        raise ValueError("seed_count must be at least two.")
    seeds = np.asarray(
        [
            stable_seed(
                {
                    "seed_role": "internal_cognitive_genealogy",
                    "analysis_seed": int(analysis_seed),
                    "subject_id": int(subject_id),
                    "seed_index": int(index),
                }
            )
            for index in range(count)
        ],
        dtype=np.uint32,
    )
    if np.unique(seeds).size != seeds.size:
        raise RuntimeError("filter seed collision detected.")
    return seeds


def _run_genealogy_seed(
    spec: AutonomousEvaluationSpec,
    *,
    particle_count: int,
    filter_seed: int,
) -> dict[str, Any]:
    engine_config = deepcopy(spec.engine_config)
    inference = engine_config.setdefault("inference", {})
    inference["backend"] = "particle_filter"
    inference["particle_count"] = int(particle_count)
    inference["choice_transmission_audit"] = True

    readout = resolve_choice_readout_config(None, engine_config)
    noise = resolve_output_noise_config(None, engine_config)
    unsupported_lapse = sum(
        float(noise.get(key, 0.0))
        for key in (
            "post_error_lapse",
            "low_accuracy_lapse",
            "latent_volatility_lapse",
        )
    )
    if unsupported_lapse > 0.0:
        raise ValueError(
            "Internal path tracing currently supports only uniform base_lapse."
        )
    output_lapse = (
        float(noise.get("base_lapse", 0.0))
        if bool(noise.get("enabled", False))
        else 0.0
    )
    result = run_inference_backend(
        engine_config=engine_config,
        subject_id=int(spec.subject_id),
        condition=int(spec.condition),
        stimulus=spec.arrays.stimulus,
        choices=spec.arrays.choices,
        feedback=spec.arrays.feedback,
        inference_seed=int(filter_seed),
        choice_readout_power=float(readout["power"]),
        strategy_confidence_gain=float(
            readout.get("strategy_confidence_gain", 0.0)
        ),
        rule_commitment_confidence_gain=float(
            readout.get("rule_commitment_confidence_gain", 0.0)
        ),
        output_lapse=output_lapse,
        valid_trial_mask=np.ones(spec.arrays.choices.size, dtype=bool),
        processed_data_dir=spec.processed_data_dir,
        dataset_paths=spec.dataset_paths,
    )
    ancestral = result.artifacts.get("audit_ancestral_paths")
    if not isinstance(ancestral, Mapping):
        raise RuntimeError("particle filter did not return ancestral paths.")
    executed = result.state_probabilities.get("executed_probability")
    fields = tuple(
        field for field in PATH_FIELDS
        if executed is not None or field not in EXECUTION_PATH_FIELDS
    )
    missing = [field for field in fields if field not in ancestral]
    if missing:
        raise RuntimeError(f"ancestral path audit is missing fields: {missing}")
    diagnostics = result.diagnostics
    latent = result.latent_summaries
    return {
        "filter_seed": int(filter_seed),
        "weights": np.asarray(ancestral["weights"], dtype=float),
        "particle_indices": np.asarray(
            ancestral["particle_indices"], dtype=np.int32
        ),
        "paths": {
            field: np.asarray(ancestral[field]) for field in fields
        },
        "marginal_choice_probability": np.asarray(
            result.observation_probabilities["prior_t"], dtype=float
        ),
        "online_hypothesis_prior": np.asarray(
            result.state_probabilities["hypothesis_prior"], dtype=float
        ),
        "online_active_probability": np.asarray(
            result.state_probabilities["active_probability"], dtype=float
        ),
        "online_executed_probability": (
            None if executed is None else np.asarray(executed, dtype=float)
        ),
        "online_swap_probability": np.asarray(
            latent["predictive_swap_probability"], dtype=float
        ),
        "pre_choice_ess": np.asarray(
            diagnostics["pre_choice_ess"], dtype=float
        ),
        "post_choice_ess": np.asarray(
            diagnostics["post_choice_ess"], dtype=float
        ),
        "resampled": np.asarray(diagnostics["resampled"], dtype=bool),
    }


@single_threaded_processes()
def generate_cognitive_path_ensemble(
    spec: AutonomousEvaluationSpec,
    *,
    particle_count: int,
    filter_seeds: Sequence[int] | np.ndarray,
    n_jobs: int = 1,
) -> CognitivePathEnsemble:
    """Run independent PF genealogies and combine their terminal mixtures."""

    n_particles = int(particle_count)
    if n_particles < 2:
        raise ValueError("particle_count must be at least two.")
    seeds = np.asarray(filter_seeds, dtype=np.uint32).reshape(-1)
    if seeds.size < 2 or np.unique(seeds).size != seeds.size:
        raise ValueError("filter_seeds must contain at least two unique values.")
    runs = Parallel(n_jobs=parallel_job_count(n_jobs, len(seeds)))(
        delayed(_run_genealogy_seed)(
            spec,
            particle_count=n_particles,
            filter_seed=int(seed),
        )
        for seed in seeds
    )
    path_weights = []
    seed_index = []
    terminal_particle = []
    particle_indices = []
    fields = tuple(runs[0]["paths"])
    paths: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    for run_index, run in enumerate(runs):
        if set(run["paths"]) != set(fields):
            raise RuntimeError("PF seeds disagree on available cognitive path fields.")
        weights = _normalize(run["weights"])
        if weights.size != n_particles:
            raise RuntimeError("terminal particle count does not match request.")
        path_weights.append(weights / float(len(runs)))
        seed_index.append(np.full(n_particles, run_index, dtype=np.int16))
        terminal_particle.append(np.arange(n_particles, dtype=np.int32))
        particle_indices.append(run["particle_indices"])
        for field in fields:
            paths[field].append(run["paths"][field])
    combined_paths = {
        field: np.concatenate(values, axis=0) for field, values in paths.items()
    }
    n_trials = int(spec.arrays.choices.size)
    n_hypotheses = int(combined_paths["hypothesis_prior"].shape[2])
    expected_2d = (len(runs) * n_particles, n_trials)
    for field in fields:
        values = combined_paths[field]
        expected = (
            (*expected_2d, n_hypotheses)
            if field in {
                "hypothesis_prior",
                "hypothesis_posterior",
                "active_hypothesis_mask",
            }
            else expected_2d
        )
        if values.shape != expected:
            raise RuntimeError(
                f"combined path field {field!r} has shape {values.shape}; "
                f"expected {expected}."
            )
    return CognitivePathEnsemble(
        spec=spec,
        filter_seeds=seeds,
        particle_count=n_particles,
        weights=_normalize(np.concatenate(path_weights)),
        seed_index=np.concatenate(seed_index),
        terminal_particle=np.concatenate(terminal_particle),
        particle_indices=np.concatenate(particle_indices, axis=0),
        paths=combined_paths,
        marginal_choice_probability=np.mean(
            np.stack([run["marginal_choice_probability"] for run in runs]),
            axis=0,
        ),
        online_hypothesis_prior=np.mean(
            np.stack([run["online_hypothesis_prior"] for run in runs]),
            axis=0,
        ),
        online_active_probability=np.mean(
            np.stack([run["online_active_probability"] for run in runs]),
            axis=0,
        ),
        online_executed_probability=None if runs[0]["online_executed_probability"] is None else np.mean(
            np.stack([run["online_executed_probability"] for run in runs]),
            axis=0,
        ),
        online_swap_probability=np.mean(
            np.stack([run["online_swap_probability"] for run in runs]),
            axis=0,
        ),
        pre_choice_ess=np.stack([run["pre_choice_ess"] for run in runs]),
        post_choice_ess=np.stack([run["post_choice_ess"] for run in runs]),
        resampled=np.stack([run["resampled"] for run in runs]),
    )


def _systematic_equal_weight_draws(
    weights: np.ndarray, *, draw_count: int, random_state: int
) -> np.ndarray:
    probability = _normalize(weights)
    count = int(draw_count)
    if count < 2:
        raise ValueError("draw_count must be at least two.")
    rng = np.random.default_rng(int(random_state))
    positions = (np.arange(count, dtype=float) + rng.random()) / float(count)
    cumulative = np.cumsum(probability)
    cumulative[-1] = 1.0
    indices = np.searchsorted(cumulative, positions, side="right")
    rng.shuffle(indices)
    return indices.astype(int)


def _genealogy_diagnostics(
    ensemble: CognitivePathEnsemble,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    n_trials = int(ensemble.particle_indices.shape[1])
    n_runs = int(ensemble.filter_seeds.size)
    n_particles = int(ensemble.particle_count)
    unique_count = np.zeros(n_trials, dtype=int)
    effective_count = np.zeros(n_trials, dtype=float)
    for trial_index in range(n_trials):
        codes = (
            ensemble.seed_index.astype(np.int64) * n_particles
            + ensemble.particle_indices[:, trial_index].astype(np.int64)
        )
        mass = np.bincount(
            codes,
            weights=ensemble.weights,
            minlength=n_runs * n_particles,
        )
        mass = mass[mass > 0.0]
        unique_count[trial_index] = int(mass.size)
        effective_count[trial_index] = 1.0 / float(np.sum(np.square(mass)))
    per_seed_start = np.asarray(
        [
            np.unique(
                ensemble.particle_indices[ensemble.seed_index == run_index, 0]
            ).size
            for run_index in range(n_runs)
        ],
        dtype=int,
    )
    start_effective = float(effective_count[0])
    median_seed_start = float(np.median(per_seed_start))
    if start_effective >= 32.0 and median_seed_start >= 4.0:
        status = "adequate_genealogical_support"
        message = "Early ancestry retains substantial within- and between-seed diversity."
    elif start_effective >= 12.0:
        status = "limited_genealogical_support"
        message = (
            "Independent seeds retain several early histories, but within-seed "
            "ancestry is collapsed; interpret archetype shares cautiously."
        )
    else:
        status = "severe_genealogical_collapse"
        message = (
            "Too few effective early ancestors remain for calibrated path shares; "
            "the paths are illustrative only."
        )
    return unique_count, effective_count, per_seed_start, status, message


def _mixed_path_distance(paths: Mapping[str, np.ndarray]) -> np.ndarray:
    executed = np.asarray(paths["executed_hypothesis"], dtype=int)
    swap = np.asarray(paths["swap_event"], dtype=float)
    h0 = np.asarray(paths["hypothesis_prior"], dtype=float)[:, :, 0]
    correct = np.asarray(paths["correct_probability"], dtype=float)
    executed_distance = np.mean(
        executed[:, None, :] != executed[None, :, :], axis=2
    )
    swap_distance = np.mean(swap[:, None, :] != swap[None, :, :], axis=2)
    h0_distance = cdist(h0, h0, metric="cityblock") / float(h0.shape[1])
    correct_distance = cdist(correct, correct, metric="cityblock") / float(
        correct.shape[1]
    )
    return (
        0.45 * executed_distance
        + 0.20 * swap_distance
        + 0.20 * h0_distance
        + 0.15 * correct_distance
    )


def _agglomerative_labels(distance: np.ndarray, cluster_count: int) -> np.ndarray:
    kwargs = {
        "n_clusters": int(cluster_count),
        "linkage": "average",
    }
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return model.fit_predict(distance).astype(int)


def _cluster_sampled_paths(
    sampled_paths: Mapping[str, np.ndarray], *, max_clusters: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    distance = _mixed_path_distance(sampled_paths)
    representative = int(np.argmin(np.mean(distance, axis=1)))
    n_paths = int(distance.shape[0])
    min_size = max(2, int(math.ceil(0.05 * n_paths)))
    candidates = []
    fallback = []
    for cluster_count in range(2, min(int(max_clusters), n_paths - 1) + 1):
        labels = _agglomerative_labels(distance, cluster_count)
        sizes = np.bincount(labels, minlength=cluster_count)
        if np.any(sizes == 0):
            continue
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        item = (score, cluster_count, labels)
        fallback.append(item)
        if int(np.min(sizes)) >= min_size:
            candidates.append(item)
    usable = candidates or fallback
    if not usable:
        labels = np.zeros(n_paths, dtype=int)
        return representative, labels, np.asarray([representative]), np.asarray([1.0]), float("nan")
    score, cluster_count, labels = max(usable, key=lambda item: (item[0], -item[1]))
    medoids = []
    shares = []
    for cluster in range(cluster_count):
        members = np.flatnonzero(labels == cluster)
        within = distance[np.ix_(members, members)]
        medoids.append(int(members[int(np.argmin(np.mean(within, axis=1)))]))
        shares.append(float(members.size / n_paths))
    order = np.argsort(-np.asarray(shares), kind="stable")
    remap = np.empty(cluster_count, dtype=int)
    remap[order] = np.arange(cluster_count)
    return (
        representative,
        remap[labels],
        np.asarray(medoids, dtype=int)[order],
        np.asarray(shares, dtype=float)[order],
        score,
    )


def summarize_cognitive_paths(
    ensemble: CognitivePathEnsemble,
    *,
    draw_count: int,
    analysis_seed: int,
    max_clusters: int = 4,
) -> CognitivePathSummary:
    source = _systematic_equal_weight_draws(
        ensemble.weights,
        draw_count=int(draw_count),
        random_state=int(analysis_seed),
    )
    sampled = {field: values[source] for field, values in ensemble.paths.items()}
    representative, labels, medoids, shares, silhouette = _cluster_sampled_paths(
        sampled,
        max_clusters=int(max_clusters),
    )
    unique, effective, per_seed, status, message = _genealogy_diagnostics(ensemble)
    return CognitivePathSummary(
        sample_source_index=source,
        sampled_paths=sampled,
        representative_index=representative,
        cluster_labels=labels,
        cluster_medoid_indices=medoids,
        cluster_shares=shares,
        cluster_silhouette=silhouette,
        cluster_count=int(shares.size),
        ancestor_unique_count=unique,
        ancestor_effective_count=effective,
        per_seed_start_ancestor_count=per_seed,
        genealogy_status=status,
        genealogy_message=message,
    )


def _hypothesis_catalog(spec: AutonomousEvaluationSpec) -> list[dict[str, Any]]:
    model = StateModel(
        spec.engine_config,
        context=ModelContext(
            condition=int(spec.condition),
            subject_id=int(spec.subject_id),
            processed_data_dir=spec.processed_data_dir,
            dataset_paths=spec.dataset_paths,
        ),
    )
    catalog = []
    for hypothesis in model.partition_model.hypothesis_space:
        parameters = dict(hypothesis.parameters)
        detail = ""
        if "dimension" in parameters:
            detail = f"F{int(parameters['dimension']) + 1}"
        elif "dimensions" in parameters:
            dims = tuple(int(value) + 1 for value in parameters["dimensions"])
            detail = "F" + "/F".join(str(value) for value in dims)
        elif "dimension_pairs" in parameters:
            pairs = parameters["dimension_pairs"]
            detail = " vs ".join(
                "+".join(f"F{int(value) + 1}" for value in pair)
                for pair in pairs
            )
        label = f"H{int(hypothesis.index)}"
        if detail:
            label = f"{label} {detail}"
        catalog.append(
            {
                "hypothesis": int(hypothesis.index),
                "label": label,
                "family": str(hypothesis.family),
                "family_label": FAMILY_LABELS.get(
                    str(hypothesis.family), str(hypothesis.family)
                ),
                "detail": detail,
            }
        )
    return catalog


def _weighted_mean(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.tensordot(_normalize(weights), np.asarray(values, dtype=float), axes=(0, 0))


def _rolling_accuracy(values: np.ndarray, window_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    window = int(window_size)
    result = np.full(array.size, np.nan, dtype=float)
    if array.size >= window:
        result[window - 1 :] = np.convolve(
            array, np.ones(window) / float(window), mode="valid"
        )
    return result


def summarize_fixed_path_fit(
    *,
    observed_feedback: np.ndarray,
    selected_correct_probability: np.ndarray,
    selected_observed_choice_probability: np.ndarray,
    window_size: int,
    condition: int = 1,
) -> FixedPathFitSummary:
    """Compare one fixed complete path with full species-correct behavior."""

    feedback = np.asarray(observed_feedback, dtype=float).reshape(-1)
    correct_probability = np.asarray(
        selected_correct_probability, dtype=float
    ).reshape(-1)
    observed_probability = np.asarray(
        selected_observed_choice_probability, dtype=float
    ).reshape(-1)
    if not (
        feedback.size
        == correct_probability.size
        == observed_probability.size
    ):
        raise ValueError("Human and fixed-path arrays must have equal length.")
    allowed_feedback = (0.0, 0.5, 1.0) if int(condition) == 3 else (0.0, 1.0)
    if not np.all(np.isfinite(feedback)) or not np.all(np.isin(feedback, allowed_feedback)):
        description = "0, 0.5, or 1" if int(condition) == 3 else "binary values"
        raise ValueError(f"observed_feedback must contain finite {description}.")
    if (
        not np.all(np.isfinite(correct_probability))
        or not np.all(np.isfinite(observed_probability))
        or np.any(correct_probability < 0.0)
        or np.any(correct_probability > 1.0)
        or np.any(observed_probability < 0.0)
        or np.any(observed_probability > 1.0)
    ):
        raise ValueError(
            "Fixed-path arrays must contain finite probabilities in [0, 1]."
        )
    species_success = (feedback == 1.0).astype(float)
    rolling_human = _rolling_accuracy(species_success, int(window_size))
    rolling_model = _rolling_accuracy(correct_probability, int(window_size))
    valid = np.isfinite(rolling_human) & np.isfinite(rolling_model)
    difference = rolling_model[valid] - rolling_human[valid]
    rolling_rmse = float(np.sqrt(np.mean(np.square(difference))))
    if (
        np.std(rolling_human[valid]) > 0.0
        and np.std(rolling_model[valid]) > 0.0
    ):
        rolling_correlation = float(
            np.corrcoef(rolling_human[valid], rolling_model[valid])[0, 1]
        )
    else:
        rolling_correlation = float("nan")
    clipped = np.clip(observed_probability, np.finfo(float).tiny, 1.0)
    sequence_nll = float(-np.sum(np.log(clipped)))
    return FixedPathFitSummary(
        rolling_human_accuracy=rolling_human,
        rolling_model_correct_probability=rolling_model,
        empirical_accuracy=float(np.mean(species_success)),
        model_expected_accuracy=float(np.mean(correct_probability)),
        sequence_nll=sequence_nll,
        mean_trial_nll=sequence_nll / float(feedback.size),
        rolling_rmse=rolling_rmse,
        rolling_correlation=rolling_correlation,
    )


def _rolling_path_means(values: np.ndarray, window_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("rolling path values must be a 2-D array.")
    return np.vstack(
        [
            _rolling_accuracy(row, int(window_size))
            for row in array
        ]
    )


def _plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_fixed_path_model_human_figure(
    *,
    trial: np.ndarray,
    observed_feedback: np.ndarray,
    fit: FixedPathFitSummary,
    subject_id: int,
    window_size: int,
    output_path: Path,
    condition: int = 1,
) -> Path:
    """Render the direct human-versus-one-fixed-path comparison."""

    _plot_style()
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.plot(
        trial,
        fit.rolling_human_accuracy,
        color="#111111",
        lw=1.55,
        label="Human empirical accuracy",
        zorder=3,
    )
    axis.plot(
        trial,
        fit.rolling_model_correct_probability,
        color="#2C7FB8",
        lw=1.55,
        label="Best fixed complete path",
        zorder=2,
    )
    error_trial = trial[np.asarray(observed_feedback, dtype=float) < 1.0]
    axis.scatter(
        error_trial,
        np.full(error_trial.size, 0.025),
        marker="|",
        s=28,
        color="#C44E52",
        linewidths=0.8,
        label="Observed error",
        zorder=4,
    )
    species_chance = 0.5 if int(condition) == 1 else 0.25
    axis.axhline(species_chance, color="#999999", lw=0.75, ls=":")
    axis.set_xlim(float(trial[0]), float(trial[-1]))
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Trial")
    axis.set_ylabel(f"{int(window_size)}-trial accuracy / P(correct)")
    axis.set_title(
        f"Subject {int(subject_id)}: human behavior versus one fixed complete path",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    metric_text = (
        f"Full-sequence NLL = {fit.sequence_nll:.2f}   |   "
        f"mean NLL/trial = {fit.mean_trial_nll:.3f}   |   "
        f"rolling RMSE = {fit.rolling_rmse:.3f}   |   "
        f"rolling r = {fit.rolling_correlation:.3f}"
    )
    axis.text(
        0.01,
        0.08,
        metric_text,
        transform=axis.transAxes,
        fontsize=6.3,
        color="#444444",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    fig.text(
        0.01,
        0.005,
        "One complete genealogy was selected once using all observed choices; "
        "the same path is used at every trial. No trialwise particle averaging.",
        fontsize=6.2,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_best_complete_path_model_human_from_artifacts(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    window_size: int = 16,
) -> dict[str, Path]:
    """Select one saved genealogy globally and compare it with human behavior."""

    source_dir = Path(input_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(
            f"Refusing to reuse existing output; choose a new output directory: {output}"
        )
    trial_path = source_dir / "internal_cognitive_trial_summary.csv"
    arrays_path = source_dir / "internal_cognitive_path_samples.npz"
    source_manifest_path = source_dir / "analysis_manifest.json"
    for path in (trial_path, arrays_path, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required cognitive-path artifact not found: {path}")

    trial_frame = pd.read_csv(trial_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as arrays:
        observed_probability = np.asarray(
            arrays["path_observed_choice_probability"], dtype=float
        )
        correct_probability = np.asarray(
            arrays["path_correct_probability"], dtype=float
        )
        source_seed = np.asarray(arrays["sampled_source_seed_index"], dtype=int)
        sample_source_index = np.asarray(
            arrays["sample_source_index"], dtype=int
        )
        terminal_particle = np.asarray(
            arrays["sampled_source_terminal_particle"], dtype=int
        )
        clustering_unavailable = (
            source_manifest.get("archetypes", {}).get("status") == "not_applicable"
        )
        cluster_labels = (
            None if clustering_unavailable
            else np.asarray(arrays["cluster_labels"], dtype=int)
        )

    candidate_count = int(observed_probability.shape[0])
    for name, values in (
        ("sample_source_index", sample_source_index),
        ("sampled_source_seed_index", source_seed),
        ("sampled_source_terminal_particle", terminal_particle),
        ("cluster_labels", cluster_labels),
    ):
        if values is not None and values.shape != (candidate_count,):
            raise ValueError(
                f"{name} must align with the candidate path count "
                f"({candidate_count})."
            )
    raw_path_count = int(source_manifest["raw_terminal_path_count"])
    if raw_path_count <= 0 or np.any(sample_source_index < 0) or np.any(
        sample_source_index >= raw_path_count
    ):
        raise ValueError(
            "sample_source_index must lie within the raw terminal path range."
        )
    selection = select_best_complete_path(observed_probability)
    best = int(selection.path_index)
    observed_feedback = np.asarray(
        trial_frame["observed_feedback"], dtype=float
    )
    n_trials = int(observed_feedback.size)
    if int(source_manifest["trial_count"]) != n_trials:
        raise ValueError(
            "Source manifest trial_count must match the human trial summary."
        )
    if correct_probability.shape != observed_probability.shape:
        raise ValueError("Saved correct and observed-choice path arrays must align.")
    if observed_probability.shape[1] != n_trials:
        raise ValueError("Saved paths and human trial summary must align.")
    if not 0 < int(window_size) <= n_trials:
        raise ValueError("window_size must be in [1, trial_count].")

    fit = summarize_fixed_path_fit(
        observed_feedback=observed_feedback,
        selected_correct_probability=correct_probability[best],
        selected_observed_choice_probability=observed_probability[best],
        window_size=int(window_size),
        condition=int(source_manifest.get("condition", 1)),
    )
    trial_raw = pd.to_numeric(trial_frame["trial"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(trial_raw)) or not np.all(
        trial_raw == np.floor(trial_raw)
    ):
        raise ValueError("Human trial values must be finite integers.")
    trial = trial_raw.astype(int)
    expected_trial = np.arange(1, n_trials + 1, dtype=int)
    if not np.array_equal(trial, expected_trial):
        raise ValueError(
            "Human trial summary must use contiguous 1-based trial order."
        )
    subject_id = int(source_manifest["subject_id"])
    selected_sample_source = int(sample_source_index[best])
    selected_seed = int(source_seed[best])
    selected_terminal = int(terminal_particle[best])
    selected_cluster = None if cluster_labels is None else int(cluster_labels[best]) + 1

    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / f"subject_{subject_id}_best_complete_path_vs_human.png"
    _save_fixed_path_model_human_figure(
        trial=trial,
        observed_feedback=observed_feedback,
        fit=fit,
        subject_id=subject_id,
        window_size=int(window_size),
        output_path=figure_path,
        condition=int(source_manifest.get("condition", 1)),
    )

    trial_source = pd.DataFrame(
        {
            "trial": trial,
            "observed_accuracy": observed_feedback,
            "human_rolling_accuracy": fit.rolling_human_accuracy,
            "model_correct_probability": correct_probability[best],
            "model_rolling_correct_probability": (
                fit.rolling_model_correct_probability
            ),
            "model_observed_choice_probability": observed_probability[best],
            "selected_path_index": np.full(n_trials, best, dtype=int),
            "selected_sample_source_index": np.full(
                n_trials, selected_sample_source, dtype=int
            ),
            "selected_source_seed_index": np.full(
                n_trials, selected_seed, dtype=int
            ),
            "selected_terminal_particle": np.full(
                n_trials, selected_terminal, dtype=int
            ),
            "selected_cluster": [selected_cluster] * n_trials,
        }
    )
    trial_source_path = output / "best_complete_path_vs_human_trial_data.csv"
    trial_source.to_csv(trial_source_path, index=False)

    summary_frame = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "trial_count": n_trials,
                "window_size": int(window_size),
                "candidate_complete_path_count": int(
                    observed_probability.shape[0]
                ),
                "selected_path_index": best,
                "selected_sample_source_index": selected_sample_source,
                "selected_source_seed_index": selected_seed,
                "selected_terminal_particle": selected_terminal,
                "selected_cluster": selected_cluster,
                "sequence_nll": fit.sequence_nll,
                "mean_trial_nll": fit.mean_trial_nll,
                "rolling_rmse": fit.rolling_rmse,
                "rolling_correlation": fit.rolling_correlation,
                "empirical_accuracy": fit.empirical_accuracy,
                "model_expected_accuracy": fit.model_expected_accuracy,
                "in_sample_descriptive_fit": True,
            }
        ]
    )
    summary_path = output / "best_complete_path_vs_human_summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    manifest = {
        "analysis": "best_complete_path_model_vs_human",
        "subject_id": subject_id,
        "trial_count": n_trials,
        "window_size": int(window_size),
        "source_analysis": _project_relative(source_dir),
        "selection": {
            "criterion": "minimum full-sequence observed-choice NLL",
            "candidate_complete_path_count": int(
                observed_probability.shape[0]
            ),
            "selected_path_index": best,
            "selected_sample_source_index": selected_sample_source,
            "selected_source_seed_index": selected_seed,
            "selected_terminal_particle": selected_terminal,
            "selected_cluster": selected_cluster,
        },
        "trialwise_particle_reweighting": False,
        "path_switching_after_selection": False,
        "interpretation": (
            "One saved complete genealogy is selected once using all observed "
            "choices and then used unchanged for all trials. Metrics are "
            "descriptive in-sample fit, not held-out predictive performance."
        ),
        "outputs": {
            "figure": figure_path.name,
            "trial_source": trial_source_path.name,
            "summary": summary_path.name,
        },
    }
    manifest_path = output / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "figure": figure_path,
        "trial_source": trial_source_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def _save_overview_figure(
    ensemble: CognitivePathEnsemble,
    summary: CognitivePathSummary,
    catalog: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Path:
    _plot_style()
    trial = np.arange(1, ensemble.spec.arrays.choices.size + 1)
    correct = (np.asarray(ensemble.spec.arrays.feedback, dtype=float) == 1.0).astype(float)
    categories = np.asarray(ensemble.spec.arrays.categories, dtype=int) - 1
    sampled = summary.sampled_paths
    h0_prior = np.asarray(sampled["hypothesis_prior"], dtype=float)[:, :, 0]
    h0_q = np.quantile(h0_prior, (0.10, 0.50, 0.90), axis=0)
    correct_paths = np.asarray(sampled["correct_probability"], dtype=float)
    rolling_correct_paths = _rolling_path_means(
        correct_paths, ensemble.spec.window_size
    )
    correct_q = np.full((3, rolling_correct_paths.shape[1]), np.nan, dtype=float)
    rolling_start = int(ensemble.spec.window_size) - 1
    correct_q[:, rolling_start:] = np.quantile(
        rolling_correct_paths[:, rolling_start:], (0.10, 0.50, 0.90), axis=0
    )
    smoothed_prior = _weighted_mean(
        ensemble.weights, ensemble.paths["hypothesis_prior"]
    )
    executed = np.asarray(ensemble.paths["executed_hypothesis"], dtype=int)
    n_hypotheses = int(smoothed_prior.shape[1])
    executed_onehot = np.eye(n_hypotheses, dtype=float)[executed]
    smoothed_executed = _weighted_mean(ensemble.weights, executed_onehot)
    smoothed_search_event = _weighted_mean(
        ensemble.weights, ensemble.paths["swap_event"]
    )
    smoothed_replacement = _weighted_mean(
        ensemble.weights, ensemble.paths["replacement_fraction"]
    )
    smoothed_failure = _weighted_mean(
        ensemble.weights, ensemble.paths["failure_pressure"]
    )
    online_correct = ensemble.marginal_choice_probability[
        np.arange(trial.size), categories
    ]
    representative = int(summary.representative_index)
    online_correct_rolling = _rolling_accuracy(online_correct, ensemble.spec.window_size)
    representative_rolling = rolling_correct_paths[representative]

    fig = plt.figure(figsize=(10.0, 12.0), constrained_layout=False)
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=(0.65, 1.45, 1.15, 2.2, 1.9, 1.45),
        hspace=0.34,
    )
    ax_behavior = fig.add_subplot(grid[0])
    ax_search = fig.add_subplot(grid[1], sharex=ax_behavior)
    ax_h0 = fig.add_subplot(grid[2], sharex=ax_behavior)
    ax_prior = fig.add_subplot(grid[3], sharex=ax_behavior)
    ax_execute = fig.add_subplot(grid[4], sharex=ax_behavior)
    ax_choice = fig.add_subplot(grid[5], sharex=ax_behavior)

    observed_rolling = _rolling_accuracy(correct, ensemble.spec.window_size)
    ax_behavior.scatter(
        trial,
        correct,
        s=5,
        color=np.where(correct > 0.5, "#7A7A7A", "#C44E52"),
        alpha=0.55,
        linewidths=0,
    )
    ax_behavior.plot(trial, observed_rolling, color="#111111", lw=1.3)
    ax_behavior.set_ylabel("Observed\naccuracy")
    ax_behavior.set_ylim(-0.08, 1.08)
    ax_behavior.set_yticks((0, 0.5, 1))
    ax_behavior.set_title(
        f"Subject {ensemble.spec.subject_id}: observed-history-conditioned cognitive paths",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )

    ax_search.plot(
        trial,
        ensemble.online_swap_probability,
        color="#4C78A8",
        lw=1.2,
        label="Online search probability",
    )
    ax_search.plot(
        trial,
        smoothed_search_event,
        color="#E07A5F",
        lw=1.2,
        label="Smoothed realized search",
    )
    ax_search.plot(
        trial,
        smoothed_replacement,
        color="#59A14F",
        lw=1.0,
        label="Smoothed replacement fraction",
    )
    failure_axis = ax_search.twinx()
    failure_axis.plot(
        trial,
        smoothed_failure,
        color="#777777",
        lw=0.9,
        ls="--",
        label="Failure pressure",
    )
    failure_axis.set_ylim(-0.03, 1.03)
    failure_axis.set_ylabel("Failure pressure", color="#666666")
    failure_axis.spines["top"].set_visible(False)
    failure_axis.spines["right"].set_visible(True)
    ax_search.set_ylim(-0.03, 1.03)
    ax_search.set_ylabel("Probability / fraction")
    handles, labels = ax_search.get_legend_handles_labels()
    h2, l2 = failure_axis.get_legend_handles_labels()
    ax_search.legend(handles + h2, labels + l2, ncol=4, loc="upper right")
    ax_search.text(-0.055, 1.04, "a", transform=ax_search.transAxes, fontsize=10, fontweight="bold")

    ax_h0.fill_between(trial, h0_q[0], h0_q[2], color="#9ECAE1", alpha=0.35, lw=0)
    ax_h0.plot(trial, h0_q[1], color="#2C7FB8", lw=1.25, label="Smoothed path median")
    ax_h0.plot(trial, smoothed_prior[:, 0], color="#08519C", lw=1.0, label="Smoothed mean belief")
    ax_h0.plot(trial, ensemble.online_hypothesis_prior[:, 0], color="#E07A5F", lw=1.0, label="Online pre-choice belief")
    ax_h0.plot(trial, smoothed_executed[:, 0], color="#59A14F", lw=1.0, label="P(execute H0 | all data)")
    ax_h0.set_ylim(-0.03, 1.03)
    ax_h0.set_ylabel("H0 probability")
    ax_h0.legend(ncol=4, loc="upper left")
    ax_h0.text(-0.055, 1.05, "b", transform=ax_h0.transAxes, fontsize=10, fontweight="bold")

    prior_image = ax_prior.imshow(
        smoothed_prior.T,
        origin="lower",
        aspect="auto",
        extent=(0.5, trial.size + 0.5, -0.5, n_hypotheses - 0.5),
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        interpolation="nearest",
    )
    ax_prior.set_ylabel("Rule")
    ax_prior.set_title("Smoothed expected internal pre-choice belief", loc="left", fontsize=7)
    tick_values = sorted(set([0, *range(4, n_hypotheses, 4), n_hypotheses - 1]))
    ax_prior.set_yticks(tick_values)
    ax_prior.set_yticklabels([catalog[index]["label"] for index in tick_values])
    fig.colorbar(prior_image, ax=ax_prior, pad=0.01, fraction=0.02, label="Belief mass")

    execute_image = ax_execute.imshow(
        smoothed_executed.T,
        origin="lower",
        aspect="auto",
        extent=(0.5, trial.size + 0.5, -0.5, n_hypotheses - 0.5),
        vmin=0.0,
        vmax=1.0,
        cmap="YlGn",
        interpolation="nearest",
    )
    ax_execute.set_ylabel("Rule")
    ax_execute.set_title("Smoothed probability that each rule is overtly executed", loc="left", fontsize=7)
    ax_execute.set_yticks(tick_values)
    ax_execute.set_yticklabels([catalog[index]["label"] for index in tick_values])
    fig.colorbar(execute_image, ax=ax_execute, pad=0.01, fraction=0.02, label="Occupancy")

    ax_choice.fill_between(trial, correct_q[0], correct_q[2], color="#BFD7EA", alpha=0.45, lw=0, label="Smoothed paths 10–90%")
    ax_choice.plot(trial, correct_q[1], color="#2C7FB8", lw=1.2, label="Smoothed path median")
    ax_choice.plot(trial, online_correct_rolling, color="#E07A5F", lw=1.15, label="Online one-step readout")
    ax_choice.plot(trial, representative_rolling, color="#111111", lw=0.9, alpha=0.80, label="Representative complete path")
    error_trials = trial[correct < 0.5]
    ax_choice.scatter(error_trials, np.full(error_trials.size, 0.03), marker="|", s=30, color="#C44E52", label="Observed error")
    ax_choice.axhline(1.0 / ensemble.marginal_choice_probability.shape[1], color="#999999", lw=0.7, ls=":")
    ax_choice.set_ylim(-0.03, 1.03)
    ax_choice.set_ylabel("16-trial mean P(correct)")
    ax_choice.set_xlabel("Trial")
    ax_choice.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.31),
    )
    ax_choice.text(-0.055, 1.05, "c", transform=ax_choice.transAxes, fontsize=10, fontweight="bold")

    for axis in (ax_behavior, ax_search, ax_h0, ax_prior, ax_execute, ax_choice):
        axis.set_xlim(1, trial.size)
    for axis in (ax_behavior, ax_search, ax_h0, ax_prior, ax_execute):
        axis.tick_params(labelbottom=False)
    fig.text(
        0.01,
        0.005,
        "Weights are used only to draw complete genealogies. Shaded regions summarize equal-weight complete-path draws; "
        f"genealogy status: {summary.genealogy_status.replace('_', ' ')}.",
        fontsize=6.4,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_family_strip(
    axis: plt.Axes,
    executed: np.ndarray,
    catalog: Sequence[Mapping[str, Any]],
    swap_event: np.ndarray,
) -> None:
    family_lookup = {family: index for index, family in enumerate(FAMILY_ORDER)}
    family_index = np.asarray(
        [family_lookup[catalog[int(rule)]["family"]] for rule in executed],
        dtype=int,
    )
    colors = [FAMILY_COLORS[family] for family in FAMILY_ORDER]
    axis.imshow(
        family_index[None, :],
        aspect="auto",
        interpolation="nearest",
        extent=(0.5, executed.size + 0.5, 0.0, 1.0),
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=len(colors) - 0.5,
    )
    boundaries = np.r_[0, 1 + np.flatnonzero(executed[1:] != executed[:-1]), executed.size]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start >= 8:
            axis.text(
                (start + stop + 1) / 2.0,
                0.5,
                f"H{int(executed[start])}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white",
                fontweight="bold",
            )
    event_trial = 1 + np.flatnonzero(np.asarray(swap_event) >= 0.5)
    axis.scatter(event_trial, np.full(event_trial.size, 1.12), marker="v", s=12, color="#C44E52", clip_on=False)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def _save_archetype_figure(
    ensemble: CognitivePathEnsemble,
    summary: CognitivePathSummary,
    catalog: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Path:
    _plot_style()
    trial = np.arange(1, ensemble.spec.arrays.choices.size + 1)
    count = int(summary.cluster_count)
    fig = plt.figure(figsize=(10.0, max(4.5, 3.0 * count)))
    grid = fig.add_gridspec(
        count * 3,
        1,
        height_ratios=np.tile((0.36, 0.82, 0.82), count),
        hspace=0.30,
    )
    axes = []
    for cluster_index, medoid in enumerate(summary.cluster_medoid_indices):
        row = cluster_index * 3
        strip = fig.add_subplot(grid[row])
        belief = fig.add_subplot(grid[row + 1], sharex=strip)
        readout = fig.add_subplot(grid[row + 2], sharex=strip)
        executed = np.asarray(summary.sampled_paths["executed_hypothesis"][medoid], dtype=int)
        swap = np.asarray(summary.sampled_paths["swap_event"][medoid], dtype=float)
        _plot_family_strip(strip, executed, catalog, swap)
        strip.set_title(
            f"Archetype {cluster_index + 1}  ·  {100.0 * summary.cluster_shares[cluster_index]:.1f}% of equal-weight path draws",
            loc="left",
            fontsize=8,
            fontweight="bold",
        )
        h0_prior = np.asarray(summary.sampled_paths["hypothesis_prior"][medoid], dtype=float)[:, 0]
        h0_posterior = np.asarray(summary.sampled_paths["hypothesis_posterior"][medoid], dtype=float)[:, 0]
        belief.plot(trial, h0_prior, color="#2C7FB8", lw=1.0, label="H0 pre-choice belief")
        belief.plot(trial, h0_posterior, color="#9ECAE1", lw=0.9, label="H0 post-feedback belief")
        belief.set_ylim(-0.03, 1.03)
        belief.set_ylabel("H0 belief")
        if cluster_index == 0:
            belief.legend(ncol=2, loc="upper left")
        q = np.asarray(summary.sampled_paths["correct_probability"][medoid], dtype=float)
        readout.plot(
            trial,
            q,
            color="#7CA6BF",
            lw=0.55,
            alpha=0.35,
            label="Trial-level P(correct)",
        )
        readout.plot(
            trial,
            _rolling_accuracy(q, ensemble.spec.window_size),
            color="#1B4F72",
            lw=1.15,
            label=f"{ensemble.spec.window_size}-trial mean",
        )
        readout.scatter(
            trial[np.asarray(ensemble.spec.arrays.feedback) < 1.0],
            np.full(np.sum(np.asarray(ensemble.spec.arrays.feedback) < 1.0), 0.03),
            marker="|",
            s=25,
            color="#C44E52",
        )
        readout.axhline(1.0 / ensemble.marginal_choice_probability.shape[1], color="#999999", lw=0.7, ls=":")
        readout.set_ylim(-0.03, 1.03)
        readout.set_ylabel("Readout")
        axes.extend((strip, belief, readout))
    axes[-1].set_xlabel("Trial")
    for axis in axes:
        axis.set_xlim(1, trial.size)
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    legend_handles = [
        mpl.patches.Patch(color=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
        for family in FAMILY_ORDER
    ]
    fig.legend(legend_handles, [item.get_label() for item in legend_handles], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Complete internal cognitive-path archetypes", x=0.08, ha="left", fontsize=10, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_genealogy_figure(
    ensemble: CognitivePathEnsemble,
    summary: CognitivePathSummary,
    output_path: Path,
) -> Path:
    _plot_style()
    trial = np.arange(1, ensemble.spec.arrays.choices.size + 1)
    pre_fraction = ensemble.pre_choice_ess / float(ensemble.particle_count)
    post_fraction = ensemble.post_choice_ess / float(ensemble.particle_count)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.7), sharex=True)
    for values, color, label in (
        (pre_fraction, "#4C78A8", "Pre-choice ESS / N"),
        (post_fraction, "#E07A5F", "Post-choice ESS / N"),
    ):
        q = np.quantile(values, (0.10, 0.50, 0.90), axis=0)
        axes[0].fill_between(trial, q[0], q[2], color=color, alpha=0.18, lw=0)
        axes[0].plot(trial, q[1], color=color, lw=1.1, label=label)
    axes[0].axhline(0.5, color="#999999", ls=":", lw=0.8)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("ESS fraction")
    axes[0].legend(ncol=2)
    axes[0].set_title("Particle-filter genealogy diagnostics", loc="left", fontsize=9, fontweight="bold")

    axes[1].plot(trial, summary.ancestor_unique_count, color="#76B7B2", lw=1.0, label="Unique ancestors")
    axes[1].plot(trial, summary.ancestor_effective_count, color="#2C7FB8", lw=1.2, label="Effective ancestors")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Ancestor count (log)")
    axes[1].legend(ncol=2)
    axes[1].text(
        0.99,
        0.04,
        summary.genealogy_message,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#555555",
    )

    resampling_rate = np.mean(ensemble.resampled.astype(float), axis=0)
    axes[2].fill_between(trial, 0, resampling_rate, color="#B279A2", alpha=0.45, lw=0)
    axes[2].plot(trial, resampling_rate, color="#7A5195", lw=1.0)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("Seeds resampled")
    axes[2].set_xlabel("Trial")
    for axis in axes:
        axis.set_xlim(1, trial.size)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _source_frames(
    ensemble: CognitivePathEnsemble,
    summary: CognitivePathSummary,
    catalog: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_trials = int(ensemble.spec.arrays.choices.size)
    trial = np.arange(1, n_trials + 1)
    categories = np.asarray(ensemble.spec.arrays.categories, dtype=int) - 1
    sampled = summary.sampled_paths
    prior = np.asarray(sampled["hypothesis_prior"], dtype=float)
    posterior = np.asarray(sampled["hypothesis_posterior"], dtype=float)
    correct = np.asarray(sampled["correct_probability"], dtype=float)
    h0_q = np.quantile(prior[:, :, 0], (0.10, 0.50, 0.90), axis=0)
    correct_q = np.quantile(correct, (0.10, 0.50, 0.90), axis=0)
    smoothed_prior = _weighted_mean(ensemble.weights, ensemble.paths["hypothesis_prior"])
    smoothed_posterior = _weighted_mean(ensemble.weights, ensemble.paths["hypothesis_posterior"])
    smoothed_active = _weighted_mean(ensemble.weights, ensemble.paths["active_hypothesis_mask"])
    executed = np.asarray(ensemble.paths["executed_hypothesis"], dtype=int)
    n_hypotheses = int(prior.shape[2])
    smoothed_executed = _weighted_mean(
        ensemble.weights, np.eye(n_hypotheses, dtype=float)[executed]
    )
    trial_frame = pd.DataFrame(
        {
            "trial": trial,
            "observed_choice": ensemble.spec.arrays.choices,
            "observed_feedback": ensemble.spec.arrays.feedback,
            "online_correct_probability": ensemble.marginal_choice_probability[
                np.arange(n_trials), categories
            ],
            "smoothed_correct_q10": correct_q[0],
            "smoothed_correct_q50": correct_q[1],
            "smoothed_correct_q90": correct_q[2],
            "online_search_probability": ensemble.online_swap_probability,
            "smoothed_search_event_probability": _weighted_mean(
                ensemble.weights, ensemble.paths["swap_event"]
            ),
            "smoothed_replacement_fraction": _weighted_mean(
                ensemble.weights, ensemble.paths["replacement_fraction"]
            ),
            "online_h0_prior": ensemble.online_hypothesis_prior[:, 0],
            "smoothed_h0_prior_mean": smoothed_prior[:, 0],
            "smoothed_h0_prior_q10": h0_q[0],
            "smoothed_h0_prior_q50": h0_q[1],
            "smoothed_h0_prior_q90": h0_q[2],
            "smoothed_h0_posterior_mean": smoothed_posterior[:, 0],
            "smoothed_h0_active_probability": smoothed_active[:, 0],
            "smoothed_h0_executed_probability": smoothed_executed[:, 0],
            "ancestor_unique_count": summary.ancestor_unique_count,
            "ancestor_effective_count": summary.ancestor_effective_count,
            "resampling_seed_fraction": np.mean(ensemble.resampled, axis=0),
        }
    )
    belief_rows = []
    for hypothesis, item in enumerate(catalog):
        for trial_index in range(n_trials):
            belief_rows.append(
                {
                    "trial": trial_index + 1,
                    "hypothesis": hypothesis,
                    "label": item["label"],
                    "family": item["family"],
                    "online_prior": ensemble.online_hypothesis_prior[trial_index, hypothesis],
                    "smoothed_prior": smoothed_prior[trial_index, hypothesis],
                    "smoothed_posterior": smoothed_posterior[trial_index, hypothesis],
                    "smoothed_active_probability": smoothed_active[trial_index, hypothesis],
                    "smoothed_executed_probability": smoothed_executed[trial_index, hypothesis],
                }
            )
    belief_frame = pd.DataFrame(belief_rows)
    archetype_rows = []
    for cluster, medoid in enumerate(summary.cluster_medoid_indices):
        for trial_index in range(n_trials):
            archetype_rows.append(
                {
                    "archetype": cluster + 1,
                    "share": summary.cluster_shares[cluster],
                    "trial": trial_index + 1,
                    "executed_hypothesis": int(sampled["executed_hypothesis"][medoid, trial_index]),
                    "search_event": float(sampled["swap_event"][medoid, trial_index]),
                    "replacement_fraction": float(sampled["replacement_fraction"][medoid, trial_index]),
                    "h0_prior": float(prior[medoid, trial_index, 0]),
                    "h0_posterior": float(posterior[medoid, trial_index, 0]),
                    "correct_probability": float(correct[medoid, trial_index]),
                }
            )
    archetype_frame = pd.DataFrame(archetype_rows)
    seed_frame = pd.DataFrame(
        {
            "filter_seed": ensemble.filter_seeds.astype(np.uint64),
            "particle_count": ensemble.particle_count,
            "resampling_count": np.sum(ensemble.resampled, axis=1),
            "start_unique_ancestor_count": summary.per_seed_start_ancestor_count,
            "minimum_pre_choice_ess_fraction": np.min(
                ensemble.pre_choice_ess / float(ensemble.particle_count), axis=1
            ),
            "minimum_post_choice_ess_fraction": np.min(
                ensemble.post_choice_ess / float(ensemble.particle_count), axis=1
            ),
        }
    )
    return trial_frame, belief_frame, archetype_frame, seed_frame


def save_cognitive_trajectory_outputs(
    ensemble: CognitivePathEnsemble,
    summary: CognitivePathSummary,
    *,
    output_dir: str | Path,
    analysis_seed: int,
) -> dict[str, Path]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    catalog = _hypothesis_catalog(ensemble.spec)
    overview = output / f"subject_{ensemble.spec.subject_id}_internal_cognitive_overview.png"
    archetypes = output / f"subject_{ensemble.spec.subject_id}_internal_path_archetypes.png"
    genealogy = output / f"subject_{ensemble.spec.subject_id}_genealogy_diagnostics.png"
    _save_overview_figure(ensemble, summary, catalog, overview)
    _save_archetype_figure(ensemble, summary, catalog, archetypes)
    _save_genealogy_figure(ensemble, summary, genealogy)
    trial_df, belief_df, archetype_df, seed_df = _source_frames(
        ensemble, summary, catalog
    )
    trial_csv = output / "internal_cognitive_trial_summary.csv"
    belief_csv = output / "internal_cognitive_belief_source.csv"
    archetype_csv = output / "internal_cognitive_archetypes.csv"
    seed_csv = output / "genealogy_seed_diagnostics.csv"
    catalog_csv = output / "hypothesis_catalog.csv"
    trial_df.to_csv(trial_csv, index=False)
    belief_df.to_csv(belief_csv, index=False)
    archetype_df.to_csv(archetype_csv, index=False)
    seed_df.to_csv(seed_csv, index=False)
    pd.DataFrame(catalog).to_csv(catalog_csv, index=False)
    arrays_path = output / "internal_cognitive_path_samples.npz"
    np.savez_compressed(
        arrays_path,
        sample_source_index=summary.sample_source_index,
        sampled_source_seed_index=ensemble.seed_index[summary.sample_source_index],
        sampled_source_terminal_particle=ensemble.terminal_particle[
            summary.sample_source_index
        ],
        cluster_labels=summary.cluster_labels,
        cluster_medoid_indices=summary.cluster_medoid_indices,
        cluster_shares=summary.cluster_shares,
        raw_terminal_weights=ensemble.weights,
        raw_seed_index=ensemble.seed_index,
        raw_terminal_particle=ensemble.terminal_particle,
        raw_particle_indices=ensemble.particle_indices,
        pre_choice_ess=ensemble.pre_choice_ess,
        post_choice_ess=ensemble.post_choice_ess,
        resampled=ensemble.resampled,
        **{f"path_{key}": value for key, value in summary.sampled_paths.items()},
    )
    manifest_path = output / "analysis_manifest.json"
    manifest = {
        "analysis": "observed_history_conditioned_internal_cognitive_trajectories",
        "method": "bootstrap_particle_filter_terminal_genealogy_approximation",
        "not_method": ["FFBSi", "PGAS", "independent posterior path sampler"],
        "interpretation": (
            "Particle weights are numerical inference weights only. Equal-weight "
            "draws of complete terminal genealogies are displayed as possible "
            "internal histories; genealogy diagnostics calibrate their limits."
        ),
        "subject_id": int(ensemble.spec.subject_id),
        "condition": int(ensemble.spec.condition),
        "config_path": _project_relative(ensemble.spec.config_path),
        "config_sha256": ensemble.spec.config_sha256,
        "resolved_engine_config_sha256": ensemble.spec.engine_config_sha256,
        "analysis_engine_config_sha256": _sha256_mapping(ensemble.spec.engine_config),
        "trial_count": int(ensemble.spec.arrays.choices.size),
        "particle_count_per_seed": int(ensemble.particle_count),
        "filter_seed_count": int(ensemble.filter_seeds.size),
        "filter_seeds": [int(seed) for seed in ensemble.filter_seeds],
        "raw_terminal_path_count": int(ensemble.weights.size),
        "equal_weight_draw_count": int(summary.sample_source_index.size),
        "unique_drawn_terminal_paths": int(np.unique(summary.sample_source_index).size),
        "analysis_seed": int(analysis_seed),
        "timing": {
            "hypothesis_prior": "after pre-choice search/replacement and before current choice",
            "choice_probability": "before current observed choice is weighted",
            "hypothesis_posterior": "after current observed choice and feedback update",
        },
        "h0_definition": "H0 = hypothesis index 0, feature-1 threshold at 0.5",
        "genealogy": {
            "status": summary.genealogy_status,
            "message": summary.genealogy_message,
            "start_unique_ancestor_count": int(summary.ancestor_unique_count[0]),
            "start_effective_ancestor_count": float(summary.ancestor_effective_count[0]),
            "terminal_effective_path_count": float(
                1.0 / np.sum(np.square(ensemble.weights))
            ),
            "per_seed_start_ancestor_count": [
                int(value) for value in summary.per_seed_start_ancestor_count
            ],
        },
        "archetypes": {
            "distance": (
                "0.45 executed-rule Hamming + 0.20 search-event Hamming + "
                "0.20 H0-belief MAE + 0.15 correct-readout MAE"
            ),
            "cluster_method": "average-linkage agglomerative clustering",
            "candidate_cluster_counts": list(
                range(2, min(4, summary.sample_source_index.size - 1) + 1)
            ),
            "selected_cluster_count": int(summary.cluster_count),
            "silhouette": float(summary.cluster_silhouette),
            "shares": [float(value) for value in summary.cluster_shares],
        },
        "uncertainty_scope": (
            "Fixed subject-level parameters; latent path uncertainty conditional "
            f"on all {ensemble.spec.arrays.choices.size} observed choices and feedback. "
            "Parameter uncertainty is excluded."
        ),
        "outputs": {
            "overview_figure": overview.name,
            "archetype_figure": archetypes.name,
            "genealogy_figure": genealogy.name,
            "arrays": arrays_path.name,
            "trial_source": trial_csv.name,
            "belief_source": belief_csv.name,
            "archetype_source": archetype_csv.name,
            "seed_diagnostics": seed_csv.name,
            "hypothesis_catalog": catalog_csv.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "overview": overview,
        "archetypes": archetypes,
        "genealogy": genealogy,
        "arrays": arrays_path,
        "trial_source": trial_csv,
        "belief_source": belief_csv,
        "archetype_source": archetype_csv,
        "seed_diagnostics": seed_csv,
        "catalog": catalog_csv,
        "manifest": manifest_path,
    }


def save_workspace_trajectory_outputs(
    ensemble: CognitivePathEnsemble,
    *,
    output_dir: str | Path,
    draw_count: int,
    analysis_seed: int,
) -> dict[str, Path]:
    """Report mixture-readout beliefs without inventing an executed rule.

    The existing executed-rule distance and clustering are undefined here.
    Preserve complete sampled genealogies, but report their support directly.
    """
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    draws = _systematic_equal_weight_draws(
        ensemble.weights, draw_count=draw_count, random_state=analysis_seed,
    )
    unique, effective, per_seed, status, message = _genealogy_diagnostics(ensemble)
    posterior = _weighted_mean(ensemble.weights, ensemble.paths["hypothesis_posterior"])
    trial = np.arange(1, ensemble.spec.arrays.choices.size + 1)
    n_hypotheses = posterior.shape[1]
    outputs = {
        "overview": output / f"subject_{ensemble.spec.subject_id}_workspace_beliefs.png",
        "arrays": output / "internal_cognitive_path_samples.npz",
        "trial_source": output / "internal_cognitive_trial_summary.csv",
        "belief_source": output / "internal_cognitive_belief_source.csv",
        "seed_diagnostics": output / "genealogy_seed_diagnostics.csv",
        "catalog": output / "hypothesis_catalog.csv",
        "manifest": output / "analysis_manifest.json",
    }
    np.savez_compressed(
        outputs["arrays"], sample_source_index=draws,
        sampled_source_seed_index=ensemble.seed_index[draws],
        sampled_source_terminal_particle=ensemble.terminal_particle[draws],
        raw_terminal_weights=ensemble.weights,
        raw_seed_index=ensemble.seed_index,
        raw_particle_indices=ensemble.particle_indices,
        pre_choice_ess=ensemble.pre_choice_ess,
        post_choice_ess=ensemble.post_choice_ess,
        resampled=ensemble.resampled,
        **{f"path_{key}": values[draws] for key, values in ensemble.paths.items()},
    )
    pd.DataFrame({
        "trial": trial, "observed_feedback": ensemble.spec.arrays.feedback,
        "online_swap_probability": ensemble.online_swap_probability,
        "unique_ancestors": unique, "effective_ancestors": effective,
        "mean_pre_choice_ess": ensemble.pre_choice_ess.mean(axis=0),
        "mean_post_choice_ess": ensemble.post_choice_ess.mean(axis=0),
        "resampling_seed_fraction": ensemble.resampled.mean(axis=0),
    }).to_csv(outputs["trial_source"], index=False)
    pd.DataFrame({
        "trial": np.repeat(trial, n_hypotheses),
        "hypothesis": np.tile(np.arange(n_hypotheses), trial.size),
        "online_prior": ensemble.online_hypothesis_prior.ravel(),
        "online_active_probability": ensemble.online_active_probability.ravel(),
        "terminal_genealogy_posterior": posterior.ravel(),
    }).to_csv(outputs["belief_source"], index=False)
    pd.DataFrame({"filter_seed": ensemble.filter_seeds,
                  "start_unique_ancestor_count": per_seed}).to_csv(
        outputs["seed_diagnostics"], index=False,
    )
    pd.DataFrame(_hypothesis_catalog(ensemble.spec)).to_csv(outputs["catalog"], index=False)
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), constrained_layout=True)
    for ax, values, title in zip(axes[:3], (
        ensemble.online_hypothesis_prior, ensemble.online_active_probability, posterior,
    ), ("Online pre-choice belief", "Online active-rule probability",
        "Post-feedback belief: terminal genealogy approximation")):
        artist = ax.imshow(values.T, aspect="auto", origin="lower", vmin=0, vmax=1,
                           extent=(0.5, trial.size + 0.5, -0.5, n_hypotheses - 0.5),
                           cmap="viridis", interpolation="nearest")
        ax.set(title=title, ylabel="Hypothesis index")
        fig.colorbar(artist, ax=ax, label="Probability")
    axes[3].plot(trial, ensemble.online_swap_probability, color="#4C78A8")
    axes[3].set(ylabel="Search-event probability", ylim=(-0.02, 1.02))
    axes[4].plot(trial, unique, label="Unique ancestors", color="#9C755F")
    axes[4].plot(trial, effective, label="Effective ancestors", color="#4C78A8")
    axes[4].set(xlabel="Trial", ylabel="Ancestor count", title=status.replace("_", " "))
    axes[4].legend(frameon=False)
    fig.suptitle(f"S{ensemble.spec.subject_id}: mixture readout; no single executed rule")
    fig.savefig(outputs["overview"], dpi=300, facecolor="white")
    plt.close(fig)
    manifest = {
        "analysis": "observed_history_conditioned_workspace_trajectories",
        "method": "bootstrap_particle_filter_terminal_genealogy_approximation",
        "persistent_execution": False,
        "subject_id": int(ensemble.spec.subject_id), "condition": int(ensemble.spec.condition),
        "trial_count": int(trial.size), "particle_count_per_seed": ensemble.particle_count,
        "filter_seed_count": int(ensemble.filter_seeds.size),
        "filter_seeds": ensemble.filter_seeds.tolist(), "analysis_seed": analysis_seed,
        "equal_weight_draw_count": int(draws.size),
        "raw_terminal_path_count": int(ensemble.weights.size),
        "config_path": _project_relative(ensemble.spec.config_path),
        "config_sha256": ensemble.spec.config_sha256,
        "resolved_engine_config_sha256": ensemble.spec.engine_config_sha256,
        "genealogy": {"status": status, "message": message},
        "archetypes": {"status": "not_applicable", "reason": "Executed-rule path distance is undefined for mixture readout."},
        "execution_metrics": {"status": "not_applicable", "reason": "No single rule is executed; execution beta and dwell are undefined."},
        "uncertainty_scope": "Fixed parameters; terminal genealogy is a path approximation, not FFBSi or an independent posterior path sampler.",
        "timing": {"online_prior": "Before current choice", "terminal_genealogy_posterior": "Post-feedback state conditioned on the complete observed history via terminal genealogy weights"},
        "outputs": {key: path.name for key, path in outputs.items() if key != "manifest"},
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2))
    return outputs


def run_internal_cognitive_trajectory_evaluation(
    *,
    config_path: str | Path,
    subject_id: int,
    output_dir: str | Path,
    particle_count: int = 128,
    seed_count: int = 16,
    path_draw_count: int = 500,
    analysis_seed: int = 20260831,
    n_jobs: int = 1,
    label: str | None = None,
) -> dict[str, Path]:
    """Resolve inputs, run genealogies, summarize, render, and serialize."""

    spec = load_autonomous_evaluation_spec(
        config_path,
        subject_id=int(subject_id),
        label=label,
    )
    seeds = generate_filter_seeds(
        analysis_seed=int(analysis_seed),
        subject_id=int(subject_id),
        seed_count=int(seed_count),
    )
    ensemble = generate_cognitive_path_ensemble(
        spec,
        particle_count=int(particle_count),
        filter_seeds=seeds,
        n_jobs=int(n_jobs),
    )
    if ensemble.online_executed_probability is None:
        return save_workspace_trajectory_outputs(
            ensemble, output_dir=output_dir, draw_count=int(path_draw_count),
            analysis_seed=int(analysis_seed),
        )
    summary = summarize_cognitive_paths(
        ensemble,
        draw_count=int(path_draw_count),
        analysis_seed=int(analysis_seed),
    )
    return save_cognitive_trajectory_outputs(
        ensemble,
        summary,
        output_dir=output_dir,
        analysis_seed=int(analysis_seed),
    )


__all__ = [
    "CognitivePathEnsemble",
    "CognitivePathSummary",
    "generate_cognitive_path_ensemble",
    "generate_filter_seeds",
    "run_internal_cognitive_trajectory_evaluation",
    "save_cognitive_trajectory_outputs",
    "summarize_cognitive_paths",
]
