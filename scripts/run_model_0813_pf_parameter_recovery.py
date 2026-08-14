#!/usr/bin/env python3
"""Run joint parameter recovery for the 0813 v2f particle-filter model.

The experiment autonomously generates one latent trajectory per synthetic
dataset and fits every candidate with the ordinary bootstrap particle filter.
An L9 orthogonal array varies memory gamma, the exploration failure threshold,
and the overt execution-switch scale jointly while keeping all other v2f
settings fixed.  Fitted candidates use a common random-number seed within each
dataset so their likelihood differences are not driven by unrelated particle
draws.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.model.readout import (  # noqa: E402
    resolve_choice_readout_config,
    resolve_output_noise_config,
)
from src.Bayesian_state.simulation.autonomous import (  # noqa: E402
    run_autonomous_category_learning,
)
from src.Bayesian_state.simulation.config import (  # noqa: E402
    load_yaml,
    resolve_engine_config,
)
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0813_pf_parameter_recovery.yaml"
)
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")
ORDER_COLUMNS = ("iSession", "iBlock", "iTrial")

# Three-level L9 orthogonal array.  P05 is the all-centre baseline.  Every pair
# of factor levels occurs exactly once, which is the key joint-design invariant.
L9_LEVEL_INDICES = (
    (0, 0, 2),
    (0, 1, 0),
    (0, 2, 1),
    (1, 0, 0),
    (1, 1, 1),
    (1, 2, 2),
    (2, 0, 1),
    (2, 1, 2),
    (2, 2, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("generate", "fit", "stability", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two profiles, one subject, one replicate, 32 trials, and 4 particles.",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _softmax(log_values: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    return weights / float(np.sum(weights))


def _wilson_interval(
    successes: int, total: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    # z=1.959963984540054 is the two-sided 95% standard-normal quantile.  The
    # configured report currently supports 95%; rejecting another value avoids
    # silently mislabelling an interval.
    if not math.isclose(float(confidence_level), 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the recovery report currently supports 95% Wilson intervals")
    z = 1.959963984540054
    n = float(total)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def build_profile_grid(factors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    factor_names = list(factors)
    if len(factor_names) != 3:
        raise ValueError("the L9 recovery design requires exactly three factors")
    for name in factor_names:
        levels = list(factors[name].get("levels", ()))
        if len(levels) != 3:
            raise ValueError(f"factor {name!r} must define exactly three levels")
        if len({float(value) for value in levels}) != 3:
            raise ValueError(f"factor {name!r} levels must be distinct")
        if not str(factors[name].get("path", "")).startswith("engine."):
            raise ValueError(f"factor {name!r} path must start with 'engine.'")

    profiles: list[dict[str, Any]] = []
    for profile_index, level_indices in enumerate(L9_LEVEL_INDICES, start=1):
        values = {
            name: float(factors[name]["levels"][level_index])
            for name, level_index in zip(factor_names, level_indices)
        }
        hyperparams = {
            str(factors[name]["path"]): values[name] for name in factor_names
        }
        profiles.append(
            {
                "profile_id": f"P{profile_index:02d}",
                "level_indices": [int(value) for value in level_indices],
                "values": values,
                "hyperparams": hyperparams,
                "is_baseline": bool(profile_index == 5),
            }
        )
    return profiles


def validate_profile_balance(profiles: Sequence[Mapping[str, Any]]) -> None:
    indices = np.asarray([profile["level_indices"] for profile in profiles], dtype=int)
    if indices.shape != (9, 3):
        raise AssertionError(f"unexpected L9 index shape: {indices.shape}")
    for factor_index in range(3):
        counts = np.bincount(indices[:, factor_index], minlength=3)
        if not np.array_equal(counts, np.asarray([3, 3, 3])):
            raise AssertionError("each factor level must occur three times")
    for left, right in combinations(range(3), 2):
        pairs = {(int(row[left]), int(row[right])) for row in indices}
        if len(pairs) != 9:
            raise AssertionError("each pair of factor levels must occur exactly once")
    if not np.array_equal(indices[4], np.asarray([1, 1, 1])):
        raise AssertionError("P05 must be the all-centre baseline")


def _resolved_paths(config_path: Path, config: Mapping[str, Any]) -> tuple[Path, Path]:
    base_path = Path(str(config["base_simulation_config"]))
    if not base_path.is_absolute():
        base_path = (ROOT / base_path).resolve()
    output = Path(str(config["output_dir"]))
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    return base_path, output


def _subject_engine(
    base_config: Mapping[str, Any], base_path: Path, subject_id: int
) -> dict[str, Any]:
    subject_config = resolve_subject_config(base_config, int(subject_id))
    return resolve_engine_config(subject_config, base_path.parent)


def _readout_args(engine_config: Mapping[str, Any]) -> dict[str, float]:
    readout = resolve_choice_readout_config(None, engine_config)
    noise = resolve_output_noise_config(None, engine_config)
    unsupported_noise = (
        float(noise.get("post_error_lapse", 0.0))
        + float(noise.get("low_accuracy_lapse", 0.0))
        + float(noise.get("latent_volatility_lapse", 0.0))
    )
    if unsupported_noise != 0.0 or str(noise.get("lapse_target", "uniform")) != "uniform":
        raise ValueError(
            "the current recovery scorer requires the uniform base-lapse PF readout"
        )
    return {
        "choice_readout_power": float(readout["power"]),
        "strategy_confidence_gain": float(readout["strategy_confidence_gain"]),
        "rule_commitment_confidence_gain": float(
            readout["rule_commitment_confidence_gain"]
        ),
        "output_lapse": float(noise.get("base_lapse", 0.0)),
    }


def _dataset_id(profile_id: str, subject_id: int, replicate: int) -> str:
    return f"{profile_id}_subject_{int(subject_id)}_replicate_{int(replicate):02d}"


def _synthetic_path(output: Path, dataset_id: str) -> Path:
    return output / "synthetic" / f"{dataset_id}.npz"


def _primary_cache_path(output: Path, dataset_id: str) -> Path:
    return output / "cache" / "primary" / f"{dataset_id}.json"


def _stability_cache_path(
    output: Path, dataset_id: str, particle_count: int, filter_repeat: int
) -> Path:
    return (
        output
        / "cache"
        / "stability"
        / dataset_id
        / f"particles_{int(particle_count)}_seed_{int(filter_repeat):02d}.json"
    )


def _load_subject_frames(
    base_config: Mapping[str, Any], base_path: Path, subjects: Sequence[int]
) -> tuple[dict[int, pd.DataFrame], dict[str, Path]]:
    dataset_paths = resolve_dataset_paths(base_config, base_path.parent)
    data = pd.read_csv(dataset_paths["learning_data"])
    condition_one = data.loc[data["condition"].eq(1)].copy()
    frames: dict[int, pd.DataFrame] = {}
    for subject_id in subjects:
        frame = (
            condition_one.loc[condition_one["iSub"].eq(int(subject_id))]
            .sort_values(list(ORDER_COLUMNS))
            .reset_index(drop=True)
        )
        if frame.empty:
            raise ValueError(f"subject {subject_id} is absent from condition 1")
        frames[int(subject_id)] = frame
    return frames, dataset_paths


def generate_one_dataset(
    *,
    output: Path,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    subject_frame: pd.DataFrame,
    profile: Mapping[str, Any],
    replicate: int,
    trials_per_dataset: int,
    base_seed: int,
    force: bool,
) -> dict[str, Any]:
    subject_id = int(subject_frame["iSub"].iloc[0])
    dataset_id = _dataset_id(str(profile["profile_id"]), subject_id, replicate)
    path = _synthetic_path(output, dataset_id)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
        return metadata

    if len(subject_frame) < int(trials_per_dataset):
        raise ValueError(
            f"subject {subject_id} has {len(subject_frame)} trials, fewer than "
            f"the requested {trials_per_dataset}"
        )
    frame = subject_frame.iloc[: int(trials_per_dataset)]
    stimulus = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    categories = frame["category"].to_numpy(dtype=int)
    engine = apply_fixed_hyperparams_to_engine_config(
        _subject_engine(base_config, base_path, subject_id),
        profile["hyperparams"],
    )
    generation_seed = stable_seed(
        {
            "seed_role": "model0813_parameter_recovery_generation",
            "base_seed": int(base_seed),
            "dataset_id": dataset_id,
            "true_hyperparams": dict(profile["hyperparams"]),
        }
    )
    generated = run_autonomous_category_learning(
        engine_config=engine,
        subject_id=subject_id,
        condition=1,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=int(generation_seed),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
    ).trajectory
    metadata = {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "replicate": int(replicate),
        "trial_count": int(stimulus.shape[0]),
        "true_profile_id": str(profile["profile_id"]),
        "true_values": dict(profile["values"]),
        "generation_seed": int(generation_seed),
        "generated_accuracy": float(np.mean(generated.feedback)),
        "independent_unit": "one autonomous synthetic choice trajectory",
    }
    _atomic_npz(
        path,
        stimulus=stimulus.astype(np.float32),
        categories=categories.astype(np.int8),
        choices=generated.choices.astype(np.int8),
        feedback=generated.feedback.astype(np.float32),
        generated_choice_probabilities=generated.observed_probabilities.astype(
            np.float32
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def _score_probabilities(
    probabilities: np.ndarray, choices: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(probabilities, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1) - 1
    if values.shape != (observed.size, 2):
        raise ValueError("particle probabilities do not match the synthetic choices")
    selected = values[np.arange(observed.size), observed]
    if not np.all(np.isfinite(selected)):
        raise ValueError("particle-filter choice probabilities contain non-finite values")
    log_likelihood = float(np.log(np.clip(selected, 1e-12, 1.0)).sum())
    return log_likelihood, float(-log_likelihood / observed.size)


def _fit_candidate_bank(
    *,
    dataset_path: Path,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    profiles: Sequence[Mapping[str, Any]],
    particle_count: int,
    resample_threshold_fraction: float,
    filter_seed: int,
    analysis_role: str,
    filter_repeat: int,
) -> list[dict[str, Any]]:
    with np.load(dataset_path, allow_pickle=False) as payload:
        stimulus = payload["stimulus"].astype(float)
        choices = payload["choices"].astype(int)
        feedback = payload["feedback"].astype(float)
        metadata = json.loads(str(payload["metadata_json"].item()))
    subject_id = int(metadata["subject_id"])
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        engine = apply_fixed_hyperparams_to_engine_config(
            _subject_engine(base_config, base_path, subject_id),
            profile["hyperparams"],
        )
        readout_args = _readout_args(engine)
        result = run_state_model_particle_filter(
            engine_config=engine,
            subject_id=subject_id,
            stimulus=stimulus,
            choices=choices,
            feedback=feedback,
            particle_count=int(particle_count),
            filter_seed=int(filter_seed),
            resample_threshold_fraction=float(resample_threshold_fraction),
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            **readout_args,
        )
        log_likelihood, nll = _score_probabilities(
            np.asarray(result.marginal_probabilities, dtype=float), choices
        )
        row: dict[str, Any] = {
            "analysis_role": str(analysis_role),
            "dataset_id": str(metadata["dataset_id"]),
            "subject_id": subject_id,
            "replicate": int(metadata["replicate"]),
            "trial_count": int(metadata["trial_count"]),
            "true_profile_id": str(metadata["true_profile_id"]),
            "fit_profile_id": str(profile["profile_id"]),
            "generation_seed": int(metadata["generation_seed"]),
            "filter_seed": int(filter_seed),
            "filter_repeat": int(filter_repeat),
            "particle_count": int(particle_count),
            "log_likelihood": log_likelihood,
            "nll": nll,
            "generated_accuracy": float(metadata["generated_accuracy"]),
            "mean_pre_choice_ess": float(
                np.mean(np.asarray(result.pre_choice_ess, dtype=float))
            ),
            "mean_post_choice_ess": float(
                np.mean(np.asarray(result.post_choice_ess, dtype=float))
            ),
            "resampling_fraction": float(
                np.mean(np.asarray(result.resampled, dtype=float))
            ),
        }
        for factor_name, value in metadata["true_values"].items():
            row[f"true_{factor_name}"] = float(value)
        for factor_name, value in profile["values"].items():
            row[f"fit_{factor_name}"] = float(value)
        rows.append(row)
    return rows


def fit_one_primary_dataset(
    *,
    output: Path,
    dataset_metadata: Mapping[str, Any],
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    profiles: Sequence[Mapping[str, Any]],
    particle_count: int,
    resample_threshold_fraction: float,
    base_seed: int,
    force: bool,
) -> list[dict[str, Any]]:
    dataset_id = str(dataset_metadata["dataset_id"])
    cache_path = _primary_cache_path(output, dataset_id)
    if cache_path.exists() and not force:
        return list(json.loads(cache_path.read_text(encoding="utf-8")))
    filter_seed = stable_seed(
        {
            "seed_role": "model0813_parameter_recovery_paired_filter",
            "base_seed": int(base_seed),
            "dataset_id": dataset_id,
            "particle_count": int(particle_count),
        }
    )
    rows = _fit_candidate_bank(
        dataset_path=_synthetic_path(output, dataset_id),
        base_config=base_config,
        base_path=base_path,
        dataset_paths=dataset_paths,
        profiles=profiles,
        particle_count=int(particle_count),
        resample_threshold_fraction=float(resample_threshold_fraction),
        filter_seed=int(filter_seed),
        analysis_role="primary_recovery",
        filter_repeat=0,
    )
    _atomic_json(cache_path, rows)
    return rows


def fit_one_stability_setting(
    *,
    output: Path,
    dataset_metadata: Mapping[str, Any],
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    profiles: Sequence[Mapping[str, Any]],
    particle_count: int,
    resample_threshold_fraction: float,
    filter_repeat: int,
    base_seed: int,
    force: bool,
) -> list[dict[str, Any]]:
    dataset_id = str(dataset_metadata["dataset_id"])
    cache_path = _stability_cache_path(
        output, dataset_id, int(particle_count), int(filter_repeat)
    )
    if cache_path.exists() and not force:
        return list(json.loads(cache_path.read_text(encoding="utf-8")))
    filter_seed = stable_seed(
        {
            "seed_role": "model0813_parameter_recovery_stability_filter",
            "base_seed": int(base_seed),
            "dataset_id": dataset_id,
            "particle_count": int(particle_count),
            "filter_repeat": int(filter_repeat),
        }
    )
    rows = _fit_candidate_bank(
        dataset_path=_synthetic_path(output, dataset_id),
        base_config=base_config,
        base_path=base_path,
        dataset_paths=dataset_paths,
        profiles=profiles,
        particle_count=int(particle_count),
        resample_threshold_fraction=float(resample_threshold_fraction),
        filter_seed=int(filter_seed),
        analysis_role="numerical_stability",
        filter_repeat=int(filter_repeat),
    )
    _atomic_json(cache_path, rows)
    return rows


def summarize_primary(
    scores: pd.DataFrame,
    *,
    factors: Mapping[str, Mapping[str, Any]],
    near_best_delta_nll: float,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_profiles = sorted(scores["fit_profile_id"].astype(str).unique())
    recovered_rows: list[dict[str, Any]] = []
    for dataset_id, frame in scores.groupby("dataset_id", sort=True):
        frame = frame.sort_values("fit_profile_id").reset_index(drop=True)
        if frame["fit_profile_id"].astype(str).tolist() != expected_profiles:
            raise ValueError(f"dataset {dataset_id} has an incomplete candidate bank")
        log_values = frame["log_likelihood"].to_numpy(dtype=float)
        posterior = _softmax(log_values)
        order = np.argsort(-log_values)
        best_index = int(order[0])
        true_profile = str(frame["true_profile_id"].iloc[0])
        predicted_profile = str(frame["fit_profile_id"].iloc[best_index])
        true_mask = frame["fit_profile_id"].astype(str).eq(true_profile).to_numpy()
        if int(true_mask.sum()) != 1:
            raise ValueError(f"dataset {dataset_id} true profile is not unique")
        true_index = int(np.flatnonzero(true_mask)[0])
        best_nll = float(-log_values[best_index])
        second_nll = float(-log_values[int(order[1])])
        rank_lookup = np.empty(len(order), dtype=int)
        rank_lookup[order] = np.arange(1, len(order) + 1)
        row: dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "subject_id": int(frame["subject_id"].iloc[0]),
            "replicate": int(frame["replicate"].iloc[0]),
            "trial_count": int(frame["trial_count"].iloc[0]),
            "true_profile_id": true_profile,
            "predicted_profile_id": predicted_profile,
            "exact_profile_recovered": bool(predicted_profile == true_profile),
            "true_profile_rank": int(rank_lookup[true_index]),
            "true_profile_posterior": float(posterior[true_index]),
            "runner_up_delta_nll": float(second_nll - best_nll),
            "true_delta_nll": float(-log_values[true_index] - best_nll),
            "true_within_delta_nll": bool(
                -log_values[true_index] <= best_nll + float(near_best_delta_nll)
            ),
            "near_best_profile_count": int(
                np.sum(-log_values <= best_nll + float(near_best_delta_nll))
            ),
            "effective_profile_count": float(
                np.exp(-np.sum(posterior * np.log(np.clip(posterior, 1e-300, 1.0))))
            ),
            "generated_accuracy": float(frame["generated_accuracy"].iloc[0]),
            "mean_pre_choice_ess": float(frame["mean_pre_choice_ess"].mean()),
            "resampling_fraction": float(frame["resampling_fraction"].mean()),
        }
        for factor_name in factors:
            true_value = float(frame[f"true_{factor_name}"].iloc[0])
            predicted_value = float(frame.loc[best_index, f"fit_{factor_name}"])
            posterior_mean = float(
                np.sum(posterior * frame[f"fit_{factor_name}"].to_numpy(dtype=float))
            )
            row[f"true_{factor_name}"] = true_value
            row[f"predicted_{factor_name}"] = predicted_value
            row[f"posterior_mean_{factor_name}"] = posterior_mean
            row[f"exact_{factor_name}_recovered"] = bool(
                math.isclose(predicted_value, true_value, rel_tol=0.0, abs_tol=1e-12)
            )
            row[f"absolute_error_{factor_name}"] = abs(posterior_mean - true_value)
        recovered_rows.append(row)
    recovered = pd.DataFrame(recovered_rows)

    parameter_rows: list[dict[str, Any]] = []
    for factor_name, specification in factors.items():
        correct = recovered[f"exact_{factor_name}_recovered"].astype(bool)
        successes = int(correct.sum())
        low, high = _wilson_interval(successes, len(correct), confidence_level)
        true_values = recovered[f"true_{factor_name}"].to_numpy(dtype=float)
        posterior_means = recovered[
            f"posterior_mean_{factor_name}"
        ].to_numpy(dtype=float)
        correlation_value = (
            float("nan")
            if np.unique(true_values).size < 2
            or np.unique(posterior_means).size < 2
            else float(spearmanr(true_values, posterior_means).statistic)
        )
        parameter_rows.append(
            {
                "factor": factor_name,
                "label": str(specification.get("label", factor_name)),
                "dataset_n": int(len(correct)),
                "exact_recovery_count": successes,
                "exact_recovery_rate": float(correct.mean()),
                "wilson_95_low": float(low),
                "wilson_95_high": float(high),
                "chance_rate": 1.0 / 3.0,
                "mean_absolute_error": float(
                    recovered[f"absolute_error_{factor_name}"].mean()
                ),
                "spearman_true_posterior_mean": correlation_value,
            }
        )
    parameter_summary = pd.DataFrame(parameter_rows)

    confusion = (
        recovered.groupby(["true_profile_id", "predicted_profile_id"], sort=True)
        .size()
        .rename("dataset_n")
        .reindex(
            pd.MultiIndex.from_product(
                [expected_profiles, expected_profiles],
                names=["true_profile_id", "predicted_profile_id"],
            ),
            fill_value=0,
        )
        .reset_index()
    )
    confusion["row_proportion"] = confusion["dataset_n"] / confusion.groupby(
        "true_profile_id"
    )["dataset_n"].transform("sum")

    by_subject_rows: list[dict[str, Any]] = []
    for subject_id, frame in recovered.groupby("subject_id", sort=True):
        row = {
            "subject_id": int(subject_id),
            "dataset_n": int(len(frame)),
            "exact_profile_recovery_rate": float(
                frame["exact_profile_recovered"].mean()
            ),
        }
        for factor_name in factors:
            row[f"exact_{factor_name}_recovery_rate"] = float(
                frame[f"exact_{factor_name}_recovered"].mean()
            )
        by_subject_rows.append(row)
    by_subject = pd.DataFrame(by_subject_rows)

    successes = int(recovered["exact_profile_recovered"].sum())
    profile_low, profile_high = _wilson_interval(
        successes, len(recovered), confidence_level
    )
    within_successes = int(recovered["true_within_delta_nll"].sum())
    within_low, within_high = _wilson_interval(
        within_successes, len(recovered), confidence_level
    )
    summary = {
        "independent_unit": "one autonomous synthetic choice trajectory",
        "inference_scope": (
            "Monte Carlo uncertainty conditional on four fixed real stimulus/category "
            "schedules; not a population-subject confidence interval"
        ),
        "dataset_n": int(len(recovered)),
        "template_subject_n": int(recovered["subject_id"].nunique()),
        "profile_count": int(len(expected_profiles)),
        "profile_chance_rate": 1.0 / float(len(expected_profiles)),
        "exact_profile_recovery_count": successes,
        "exact_profile_recovery_rate": float(
            recovered["exact_profile_recovered"].mean()
        ),
        "exact_profile_recovery_wilson_95": [
            float(profile_low),
            float(profile_high),
        ],
        "true_profile_within_delta_nll_count": within_successes,
        "true_profile_within_delta_nll_rate": float(
            recovered["true_within_delta_nll"].mean()
        ),
        "true_profile_within_delta_nll_wilson_95": [
            float(within_low),
            float(within_high),
        ],
        "near_best_delta_nll": float(near_best_delta_nll),
        "median_runner_up_delta_nll": float(
            recovered["runner_up_delta_nll"].median()
        ),
        "median_true_profile_rank": float(recovered["true_profile_rank"].median()),
        "mean_true_profile_posterior": float(
            recovered["true_profile_posterior"].mean()
        ),
        "median_effective_profile_count": float(
            recovered["effective_profile_count"].median()
        ),
        "median_near_best_profile_count": float(
            recovered["near_best_profile_count"].median()
        ),
        "generated_accuracy_mean": float(recovered["generated_accuracy"].mean()),
        "generated_accuracy_range": [
            float(recovered["generated_accuracy"].min()),
            float(recovered["generated_accuracy"].max()),
        ],
        "parameter_results": parameter_summary.to_dict(orient="records"),
    }
    return recovered, parameter_summary, confusion, by_subject, summary


def summarize_stability(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    recovered_rows: list[dict[str, Any]] = []
    for keys, frame in scores.groupby(
        ["dataset_id", "particle_count", "filter_repeat"], sort=True
    ):
        dataset_id, particle_count, filter_repeat = keys
        frame = frame.sort_values("fit_profile_id")
        best = frame.iloc[int(np.argmax(frame["log_likelihood"].to_numpy(dtype=float)))]
        recovered_rows.append(
            {
                "dataset_id": str(dataset_id),
                "true_profile_id": str(frame["true_profile_id"].iloc[0]),
                "particle_count": int(particle_count),
                "filter_repeat": int(filter_repeat),
                "setting": f"{int(particle_count)}p · seed {int(filter_repeat) + 1}",
                "predicted_profile_id": str(best["fit_profile_id"]),
                "true_profile_recovered": bool(
                    str(best["fit_profile_id"])
                    == str(frame["true_profile_id"].iloc[0])
                ),
                "best_nll": float(-best["log_likelihood"]),
            }
        )
    recovered = pd.DataFrame(recovered_rows)

    correlation_rows: list[dict[str, Any]] = []
    for dataset_id, frame in scores.groupby("dataset_id", sort=True):
        settings = {
            (int(particle_count), int(filter_repeat)): group.sort_values(
                "fit_profile_id"
            )["nll"].to_numpy(dtype=float)
            for (particle_count, filter_repeat), group in frame.groupby(
                ["particle_count", "filter_repeat"], sort=True
            )
        }
        for left, right in combinations(sorted(settings), 2):
            statistic = spearmanr(settings[left], settings[right]).statistic
            correlation_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "left_setting": f"{left[0]}p_seed{left[1] + 1}",
                    "right_setting": f"{right[0]}p_seed{right[1] + 1}",
                    "candidate_nll_spearman": float(statistic),
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    modal_agreements = []
    for _, frame in recovered.groupby("dataset_id", sort=True):
        counts = frame["predicted_profile_id"].value_counts()
        modal_agreements.append(float(counts.iloc[0] / len(frame)))
    by_particle_count: dict[str, Any] = {}
    working = scores.copy()
    working["total_nll"] = -working["log_likelihood"].astype(float)
    for particle_count, particle_frame in working.groupby(
        "particle_count", sort=True
    ):
        seed_rank_correlations: list[float] = []
        for _, dataset_frame in particle_frame.groupby("dataset_id", sort=True):
            repeated = {
                int(filter_repeat): group.sort_values("fit_profile_id")[
                    "total_nll"
                ].to_numpy(dtype=float)
                for filter_repeat, group in dataset_frame.groupby(
                    "filter_repeat", sort=True
                )
            }
            for left, right in combinations(sorted(repeated), 2):
                seed_rank_correlations.append(
                    float(spearmanr(repeated[left], repeated[right]).statistic)
                )
        candidate_seed_sd = (
            particle_frame.groupby(["dataset_id", "fit_profile_id"])["total_nll"]
            .std(ddof=1)
            .dropna()
        )
        within_setting_range = particle_frame.groupby(
            ["dataset_id", "filter_repeat"]
        )["total_nll"].agg(lambda values: float(values.max() - values.min()))
        by_particle_count[str(int(particle_count))] = {
            "median_seed_candidate_rank_spearman": float(
                np.median(seed_rank_correlations)
            ),
            "minimum_seed_candidate_rank_spearman": float(
                np.min(seed_rank_correlations)
            ),
            "median_candidate_total_nll_sd_across_seeds": float(
                candidate_seed_sd.median()
            ),
            "median_within_setting_candidate_total_nll_range": float(
                within_setting_range.median()
            ),
        }
    summary = {
        "dataset_n": int(recovered["dataset_id"].nunique()),
        "numerical_setting_n": int(
            recovered[["particle_count", "filter_repeat"]].drop_duplicates().shape[0]
        ),
        "setting_run_n": int(len(recovered)),
        "true_profile_recovery_rate_across_settings": float(
            recovered["true_profile_recovered"].mean()
        ),
        "mean_within_dataset_modal_winner_agreement": float(
            np.mean(modal_agreements)
        ),
        "median_pairwise_candidate_nll_spearman": float(
            correlations["candidate_nll_spearman"].median()
        ),
        "minimum_pairwise_candidate_nll_spearman": float(
            correlations["candidate_nll_spearman"].min()
        ),
        "by_particle_count": by_particle_count,
        "interpretation": (
            "descriptive PF approximation audit on three predeclared datasets; "
            "not an additional recovery replicate"
        ),
    }
    return recovered, correlations, summary


def _write_figure(
    output: Path,
    recovered: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    confusion: pd.DataFrame,
    stability_recovered: pd.DataFrame,
    stability_summary: Mapping[str, Any],
    factors: Mapping[str, Mapping[str, Any]],
    filename: str,
) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    blue = "#3B6FB6"
    orange = "#D9822B"
    charcoal = "#252A34"
    grey = "#D7DCE2"
    light_blue = "#DCE8F6"

    fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(0.94, 1.18), height_ratios=(1.0, 1.05))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    profile_rate = float(recovered["exact_profile_recovered"].mean())
    profile_low, profile_high = _wilson_interval(
        int(recovered["exact_profile_recovered"].sum()), len(recovered)
    )
    labels = ["Joint\nprofile"] + parameter_summary["label"].astype(str).tolist()
    rates = [profile_rate] + parameter_summary["exact_recovery_rate"].tolist()
    lows = [profile_low] + parameter_summary["wilson_95_low"].tolist()
    highs = [profile_high] + parameter_summary["wilson_95_high"].tolist()
    chances = [1.0 / 9.0] + parameter_summary["chance_rate"].tolist()
    positions = np.arange(len(labels))
    ax_a.bar(positions, rates, color=[charcoal] + [blue] * (len(labels) - 1), width=0.68)
    ax_a.errorbar(
        positions,
        rates,
        yerr=[np.asarray(rates) - np.asarray(lows), np.asarray(highs) - np.asarray(rates)],
        fmt="none",
        ecolor=charcoal,
        elinewidth=1.0,
        capsize=2.5,
    )
    ax_a.scatter(positions, chances, color=orange, marker="D", s=24, zorder=3, label="Discrete chance")
    ax_a.set_xticks(positions, labels, rotation=24, ha="right")
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_ylabel("Exact recovery rate")
    ax_a.set_title("Exact recovery with 95% Wilson intervals", loc="left", fontsize=9)
    ax_a.legend(loc="upper right", fontsize=7)
    ax_a.grid(axis="y", color="#E6E9ED", linewidth=0.7)

    profile_ids = sorted(recovered["true_profile_id"].unique())
    matrix = (
        confusion.pivot(
            index="true_profile_id", columns="predicted_profile_id", values="row_proportion"
        )
        .reindex(index=profile_ids, columns=profile_ids)
        .to_numpy(dtype=float)
    )
    image = ax_b.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
    ax_b.set_xticks(np.arange(len(profile_ids)), profile_ids, rotation=45, ha="right")
    ax_b.set_yticks(np.arange(len(profile_ids)), profile_ids)
    ax_b.set_xlabel("Recovered joint profile")
    ax_b.set_ylabel("Generating joint profile")
    ax_b.set_title("Joint-profile confusion (row proportion)", loc="left", fontsize=9)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value >= 0.125:
                ax_b.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > 0.50 else charcoal,
                )
    colorbar = fig.colorbar(image, ax=ax_b, fraction=0.045, pad=0.02)
    colorbar.set_label("Row proportion", fontsize=7)
    colorbar.ax.tick_params(labelsize=7)

    factor_colors = [blue, orange, "#7A8F3A"]
    offsets = np.linspace(-0.22, 0.22, len(factors))
    rng = np.random.default_rng(20260813)
    for factor_index, (factor_name, specification) in enumerate(factors.items()):
        levels = np.asarray(specification["levels"], dtype=float)
        denominator = float(levels[-1] - levels[0])
        normalized = (
            recovered[f"posterior_mean_{factor_name}"].to_numpy(dtype=float) - levels[0]
        ) / denominator
        true_values = recovered[f"true_{factor_name}"].to_numpy(dtype=float)
        true_indices = np.asarray(
            [int(np.argmin(np.abs(levels - value))) for value in true_values], dtype=int
        )
        x = true_indices + offsets[factor_index]
        jitter = rng.uniform(-0.035, 0.035, size=x.size)
        ax_c.scatter(
            x + jitter,
            normalized,
            s=9,
            alpha=0.24,
            color=factor_colors[factor_index],
            linewidths=0,
        )
        medians = [float(np.median(normalized[true_indices == level])) for level in range(3)]
        ax_c.plot(
            np.arange(3) + offsets[factor_index],
            medians,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=factor_colors[factor_index],
            label=str(specification.get("label", factor_name)),
        )
    ax_c.plot([0, 2], [0, 1], linestyle="--", color="#777C85", linewidth=1.0, label="Ideal ordering")
    ax_c.set_xticks([0, 1, 2], ["Low", "Baseline", "High"])
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_ylabel("Posterior mean (range-normalized)")
    ax_c.set_xlabel("Generating level")
    ax_c.set_title("Posterior ordering and shrinkage", loc="left", fontsize=9)
    ax_c.grid(axis="y", color="#E6E9ED", linewidth=0.7)
    ax_c.legend(fontsize=6.5, loc="upper left", ncol=1)

    stability_profiles = sorted(stability_recovered["true_profile_id"].unique())
    setting_order = (
        stability_recovered[["particle_count", "filter_repeat", "setting"]]
        .drop_duplicates()
        .sort_values(["particle_count", "filter_repeat"])["setting"]
        .tolist()
    )
    stability_matrix = np.zeros((len(stability_profiles), len(setting_order)), dtype=float)
    annotation = np.full(stability_matrix.shape, "", dtype=object)
    for row, true_profile in enumerate(stability_profiles):
        for column, setting in enumerate(setting_order):
            match = stability_recovered.loc[
                stability_recovered["true_profile_id"].eq(true_profile)
                & stability_recovered["setting"].eq(setting)
            ]
            if len(match) != 1:
                raise ValueError("stability summary is missing a unique setting row")
            stability_matrix[row, column] = float(match["true_profile_recovered"].iloc[0])
            annotation[row, column] = str(match["predicted_profile_id"].iloc[0])
    from matplotlib.colors import ListedColormap

    ax_d.imshow(
        stability_matrix,
        vmin=0.0,
        vmax=1.0,
        cmap=ListedColormap([grey, light_blue]),
        aspect="auto",
    )
    ax_d.set_xticks(np.arange(len(setting_order)), setting_order, rotation=30, ha="right")
    ax_d.set_yticks(np.arange(len(stability_profiles)), stability_profiles)
    ax_d.set_xlabel("PF numerical setting")
    ax_d.set_ylabel("Generating profile")
    rho = float(stability_summary["median_pairwise_candidate_nll_spearman"])
    ax_d.set_title(
        f"Numerical stability (median candidate-rank ρ={rho:.2f})",
        loc="left",
        fontsize=9,
    )
    for row in range(annotation.shape[0]):
        for column in range(annotation.shape[1]):
            ax_d.text(
                column,
                row,
                annotation[row, column],
                ha="center",
                va="center",
                fontsize=8,
                color=charcoal,
                fontweight="bold" if stability_matrix[row, column] else "normal",
            )
    ax_d.text(
        0.0,
        -0.20,
        "Blue cells recover the generating profile; grey cells select another profile.",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#555A63",
    )

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
    fig.suptitle(
        "0813 particle-filter joint parameter recovery",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=charcoal,
    )
    destination = output / filename
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def _write_notebook(
    output: Path,
    filename: str,
    summary: Mapping[str, Any],
    stability_summary: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import nbformat as nbf
        from nbclient import NotebookClient
    except ImportError as exc:
        return {"status": "not_created", "reason": f"missing notebook dependency: {exc}"}

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"- Exact joint-profile recovery: **{summary['exact_profile_recovery_rate']:.3f}** "
            f"({summary['exact_profile_recovery_count']}/{summary['dataset_n']}; discrete chance 0.111).\n"
            f"- Median effective candidate count: **{summary['median_effective_profile_count']:.2f}** of 9.\n"
            f"- Numerical audit median candidate-rank Spearman ρ: "
            f"**{stability_summary['median_pairwise_candidate_nll_spearman']:.3f}**."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "This companion notebook reads the saved recovery tables. Each independent "
            "unit is one autonomously generated synthetic choice trajectory. The 72 "
            "trajectories are conditional on four fixed real stimulus/category schedules; "
            "particles and trials are not treated as independent replicates.\n\n"
            "### Key Assumptions\n\n"
            "- The fitted candidate set is the predeclared balanced L9 profile bank.\n"
            "- All non-target v2f parameters are fixed to the 0813 configuration.\n"
            "- Candidate likelihoods within a dataset use paired particle-filter seeds."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "ROOT = Path.cwd()\n"
            "recovered = pd.read_csv(ROOT / 'recovered_datasets.csv')\n"
            "parameters = pd.read_csv(ROOT / 'parameter_recovery_summary.csv')\n"
            "stability = pd.read_csv(ROOT / 'stability_recovered_settings.csv')\n"
            "summary = json.loads((ROOT / 'recovery_summary.json').read_text(encoding='utf-8'))"
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            "display(recovered.head(10))\n"
            "print(f\"datasets={len(recovered)}, profiles={recovered.true_profile_id.nunique()}, \"\n"
            "      f\"subjects={recovered.subject_id.nunique()}, trials={recovered.trial_count.unique().tolist()}\")"
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            "display(parameters[['label', 'dataset_n', 'exact_recovery_rate', "
            "'wilson_95_low', 'wilson_95_high', 'chance_rate', "
            "'spearman_true_posterior_mean']])\n"
            "display(Image(filename=str(ROOT / 'parameter_recovery_overview.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "Use exact profile recovery to judge joint identifiability, parameter-level "
            "recovery to locate the principal ambiguity, and the numerical audit to "
            "separate model non-identifiability from particle-filter Monte Carlo noise. "
            "This targeted audit does not establish recovery for beta, lapse, capacity, "
            "or every controller parameter."
        ),
    ]
    path = output / filename
    nbf.write(notebook, path)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3")
    executed = client.execute(cwd=str(output))
    nbf.write(executed, path)
    return {"status": "executed", "path": path.name, "cell_count": len(executed.cells)}


def _write_chart_map(output: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# Figure contract and chart map

Core conclusion: The selected 0813 hyperparameters show partial rather than complete joint identifiability; recovery evidence must be interpreted together with candidate ambiguity and PF numerical stability.

- Figure archetype: quantitative grid with the confusion matrix as the hero evidence.
- Target output: technical recovery report; Python/matplotlib; 7.2 × 7.0 inches; PNG at 300 dpi.
- Export policy: PNG only, following the repository-wide artifact rule; redundant SVG/PDF/TIFF copies are intentionally omitted.
- Independent unit: one autonomous synthetic choice trajectory (n={summary['dataset_n']}).
- Interval definition: two-sided 95% Wilson interval over synthetic trajectories, conditional on four fixed schedules.
- Image integrity: no observations or candidates are removed; all nine profiles and all generated datasets enter every applicable panel.

| Panel | Question | Form | Evidence role | Palette |
|---|---|---|---|---|
| a | Are joint profiles and their component parameters recovered above discrete chance? | Bar + Wilson interval + chance marker | Primary summary | charcoal/blue with orange chance marker |
| b | Which generating and fitted profiles are confused? | Row-normalized heatmap | Hero evidence for equivalence | sequential blue |
| c | Does the posterior preserve low-to-high parameter ordering even when exact recovery fails? | Jittered points + median lines | Graded identifiability | blue/orange/olive plus neutral ideal line |
| d | Do PF seed and particle count change the winning candidate? | Annotated binary heatmap | Numerical robustness | light blue/neutral grey |

Reviewer risk: the intervals quantify simulation uncertainty conditional on four fixed stimulus schedules and must not be presented as population-subject confidence intervals. The L9 bank is a targeted three-factor design, not the full v2f parameter space.

QA exceptions: the source validator's `dropna` warning refers only to undefined stability standard deviations, not exclusion of plotted recovery observations; the RNG is used only for deterministic point jitter; logarithms use explicit clipping. Vector/TIFF export warnings are accepted because this repository requires PNG-only plots unless another format is explicitly requested.
"""
    (output / "chart_map.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    base_path, configured_output = _resolved_paths(config_path, config)
    output = (args.output_dir.resolve() if args.output_dir else configured_output)
    base_config = load_yaml(base_path)
    design = deepcopy(dict(config["design"]))
    factors = deepcopy(dict(config["factors"]))
    profiles = build_profile_grid(factors)
    validate_profile_balance(profiles)

    subjects = [int(value) for value in design["template_subjects"]]
    repeats = int(design["datasets_per_profile_per_subject"])
    trials = int(design["trials_per_dataset"])
    particle_count = int(design["particle_count"])
    n_jobs = int(args.n_jobs if args.n_jobs is not None else design["n_jobs"])
    if args.smoke:
        profiles = profiles[:2]
        subjects = subjects[:1]
        repeats = 1
        trials = min(32, trials)
        particle_count = 4
        output = output / "smoke"
        n_jobs = min(n_jobs, 2)
    if repeats <= 0 or trials <= 0 or particle_count < 2 or n_jobs <= 0:
        raise ValueError("invalid recovery compute budget")

    frames, dataset_paths = _load_subject_frames(base_config, base_path, subjects)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "analysis_config_snapshot.json", config)
    _atomic_json(output / "candidate_profiles.json", profiles)
    base_seed = int(design["base_seed"])
    jobs = [
        (profile, subject_id, replicate)
        for profile in profiles
        for subject_id in subjects
        for replicate in range(repeats)
    ]
    manifest = {
        "analysis_id": str(config["analysis_id"]),
        "scope": str(config["scope"]),
        "status": "configured",
        "base_simulation_config": str(base_path.relative_to(ROOT)),
        "base_simulation_config_sha256": _sha256(base_path),
        "recovery_config": str(config_path.relative_to(ROOT)),
        "recovery_config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "bayesian_state_python_tree_sha256": _python_tree_sha256(
            ROOT / "src/Bayesian_state"
        ),
        "subjects": subjects,
        "profile_count": len(profiles),
        "datasets_per_profile_per_subject": repeats,
        "expected_dataset_n": len(jobs),
        "trials_per_dataset": trials,
        "particle_count": particle_count,
        "resample_threshold_fraction": float(
            design["resample_threshold_fraction"]
        ),
        "n_jobs": n_jobs,
        "paired_filter_seed_across_candidates": True,
        "synthetic_generator": "single autonomous StateModel trajectory",
        "independent_unit": "one autonomous synthetic choice trajectory",
        "fixed_schedule_note": (
            "schedules are crossed fixed blocks and do not increase the population-subject n"
        ),
        "smoke": bool(args.smoke),
    }
    _atomic_json(output / "analysis_manifest.json", manifest)

    if args.phase in {"generate", "all"}:
        generated = Parallel(
            n_jobs=min(n_jobs, len(jobs)), backend="loky", verbose=10
        )(
            delayed(generate_one_dataset)(
                output=output,
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                subject_frame=frames[subject_id],
                profile=profile,
                replicate=replicate,
                trials_per_dataset=trials,
                base_seed=base_seed,
                force=bool(args.force),
            )
            for profile, subject_id, replicate in jobs
        )
        _atomic_json(output / "synthetic_manifest.json", generated)
        print(f"[recovery] generated={len(generated)}", flush=True)
        if args.phase == "generate":
            return
    elif not (output / "synthetic_manifest.json").exists():
        raise FileNotFoundError("synthetic_manifest.json is required; run --phase generate")

    generated = list(
        json.loads((output / "synthetic_manifest.json").read_text(encoding="utf-8"))
    )
    if len(generated) != len(jobs):
        raise ValueError("synthetic dataset count does not match the configured design")

    if args.phase in {"fit", "all"}:
        primary_results = Parallel(
            n_jobs=min(n_jobs, len(generated)), backend="loky", verbose=10
        )(
            delayed(fit_one_primary_dataset)(
                output=output,
                dataset_metadata=metadata,
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                profiles=profiles,
                particle_count=particle_count,
                resample_threshold_fraction=float(
                    design["resample_threshold_fraction"]
                ),
                base_seed=base_seed,
                force=bool(args.force),
            )
            for metadata in generated
        )
        primary_scores = pd.DataFrame(
            [row for dataset_rows in primary_results for row in dataset_rows]
        )
        _atomic_csv(output / "fit_scores.csv", primary_scores)
        print(f"[recovery] primary_fit_rows={len(primary_scores)}", flush=True)
        if args.phase == "fit":
            return
    elif not (output / "fit_scores.csv").exists():
        raise FileNotFoundError("fit_scores.csv is required; run --phase fit")

    stability_config = dict(config.get("numerical_stability", {}))
    stability_enabled = bool(stability_config.get("enabled", False)) and not args.smoke
    if args.phase in {"stability", "all"} and stability_enabled:
        selected = [
            metadata
            for metadata in generated
            if str(metadata["true_profile_id"])
            in {str(value) for value in stability_config["profile_ids"]}
            and int(metadata["subject_id"]) == int(stability_config["subject_id"])
            and int(metadata["replicate"]) == int(stability_config["replicate"])
        ]
        expected_selected = len(stability_config["profile_ids"])
        if len(selected) != expected_selected:
            raise ValueError("stability dataset selection is incomplete")
        stability_jobs = [
            (metadata, int(count), repeat)
            for metadata in selected
            for count in stability_config["particle_counts"]
            for repeat in range(int(stability_config["filter_seed_repeats"]))
        ]
        stability_results = Parallel(
            n_jobs=min(n_jobs, len(stability_jobs)), backend="loky", verbose=10
        )(
            delayed(fit_one_stability_setting)(
                output=output,
                dataset_metadata=metadata,
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                profiles=profiles,
                particle_count=count,
                resample_threshold_fraction=float(
                    design["resample_threshold_fraction"]
                ),
                filter_repeat=repeat,
                base_seed=base_seed,
                force=bool(args.force),
            )
            for metadata, count, repeat in stability_jobs
        )
        stability_scores = pd.DataFrame(
            [row for setting_rows in stability_results for row in setting_rows]
        )
        _atomic_csv(output / "stability_scores.csv", stability_scores)
        print(f"[recovery] stability_fit_rows={len(stability_scores)}", flush=True)
        if args.phase == "stability":
            return
    elif args.phase == "stability":
        print("[recovery] numerical stability disabled for this run", flush=True)
        return

    if args.phase in {"summarize", "all"}:
        primary_scores = pd.read_csv(output / "fit_scores.csv")
        expected_primary_rows = len(generated) * len(profiles)
        if len(primary_scores) != expected_primary_rows:
            raise ValueError(
                f"primary score rows are incomplete: {len(primary_scores)} vs "
                f"{expected_primary_rows}"
            )
        confidence_level = float(config["report"]["confidence_level"])
        (
            recovered,
            parameter_summary,
            confusion,
            by_subject,
            summary,
        ) = summarize_primary(
            primary_scores,
            factors=factors,
            near_best_delta_nll=float(design["near_best_delta_nll"]),
            confidence_level=confidence_level,
        )
        _atomic_csv(output / "recovered_datasets.csv", recovered)
        _atomic_csv(output / "parameter_recovery_summary.csv", parameter_summary)
        _atomic_csv(output / "profile_confusion.csv", confusion)
        _atomic_csv(output / "recovery_by_subject.csv", by_subject)

        if stability_enabled:
            stability_scores = pd.read_csv(output / "stability_scores.csv")
            stability_recovered, stability_correlations, stability_summary = (
                summarize_stability(stability_scores)
            )
        else:
            # Smoke runs retain the same output schema but clearly mark the
            # formal stability audit as unavailable.
            stability_recovered = pd.DataFrame(
                [
                    {
                        "dataset_id": recovered.iloc[0]["dataset_id"],
                        "true_profile_id": recovered.iloc[0]["true_profile_id"],
                        "particle_count": particle_count,
                        "filter_repeat": 0,
                        "setting": f"{particle_count}p · smoke",
                        "predicted_profile_id": recovered.iloc[0][
                            "predicted_profile_id"
                        ],
                        "true_profile_recovered": bool(
                            recovered.iloc[0]["exact_profile_recovered"]
                        ),
                        "best_nll": float("nan"),
                    }
                ]
            )
            stability_correlations = pd.DataFrame(
                columns=(
                    "dataset_id",
                    "left_setting",
                    "right_setting",
                    "candidate_nll_spearman",
                )
            )
            stability_summary = {
                "dataset_n": 0,
                "numerical_setting_n": 0,
                "setting_run_n": 0,
                "true_profile_recovery_rate_across_settings": None,
                "mean_within_dataset_modal_winner_agreement": None,
                "median_pairwise_candidate_nll_spearman": None,
                "minimum_pairwise_candidate_nll_spearman": None,
                "interpretation": "not run in smoke mode",
            }
        _atomic_csv(
            output / "stability_recovered_settings.csv", stability_recovered
        )
        _atomic_csv(
            output / "stability_candidate_rank_correlations.csv",
            stability_correlations,
        )
        _atomic_json(output / "stability_summary.json", stability_summary)

        summary.update(
            {
                "analysis_id": str(config["analysis_id"]),
                "scope": str(config["scope"]),
                "particle_count": int(particle_count),
                "trials_per_dataset": int(trials),
                "profile_design": "balanced L9 orthogonal array",
                "targeted_factors": list(factors),
                "numerical_stability": stability_summary,
                "limitations": [
                    "Only three predeclared v2f parameters were varied.",
                    "The L9 bank is not the complete 3x3x3 Cartesian grid.",
                    "All other beta, lapse, readout, capacity, and controller parameters were fixed.",
                    "Simulation intervals are conditional on four fixed real schedules.",
                ],
            }
        )
        _atomic_json(output / "recovery_summary.json", summary)
        if stability_enabled:
            _write_figure(
                output,
                recovered,
                parameter_summary,
                confusion,
                stability_recovered,
                stability_summary,
                factors,
                str(config["report"]["output_png"]),
            )
        _write_chart_map(output, summary)
        notebook_status = (
            {"status": "not_created", "reason": "smoke mode"}
            if args.smoke
            else _write_notebook(
                output,
                str(config["report"]["output_notebook"]),
                summary,
                stability_summary,
            )
        )
        manifest.update(
            {
                "status": "complete",
                "observed_dataset_n": int(len(recovered)),
                "observed_primary_score_rows": int(len(primary_scores)),
                "notebook": notebook_status,
            }
        )
        _atomic_json(output / "analysis_manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
