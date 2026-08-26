#!/usr/bin/env python3
"""Run the Model 0818 zero-versus-positive boundary recovery smoke.

The runner uses real condition-1 stimulus/category schedules only as fixed task
templates.  Choices and feedback are generated autonomously, and no observed
choice enters generation, fitting, or scoring.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
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
from src.Bayesian_state.optimization.parameter_space import (  # noqa: E402
    load_parameter_space,
    reactive_error_probability,
    spike_and_positive_values,
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


DEFAULT_CONFIG = ROOT / "configs/specific_models/model_0818_boundary_recovery.yaml"
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")
ORDER_COLUMNS = ("iSession", "iBlock", "iTrial")
CONTROLLER_PATH = (
    "engine.modules.hypo_transitions_mod.kwargs."
    "nested_feedback_accumulator_controller"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("generate", "fit", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    tier = parser.add_mutually_exclusive_group()
    tier.add_argument("--smoke", action="store_true")
    tier.add_argument("--pilot", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_boundary_profiles(
    parameter_space: Mapping[str, Any],
    positive_probes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the all-zero and three one-positive-at-a-time profiles."""

    parameters = dict(parameter_space["subject_parameters"])
    event_correct = float(parameters["E_C"]["anchor"])
    probes: dict[str, float] = {}
    for parameter in ("delta_E", "c_A", "c_G"):
        probe = float(positive_probes[parameter])
        if probe not in spike_and_positive_values(parameter_space, parameter)[1:]:
            raise ValueError(
                f"positive_probes.{parameter} must be a declared positive candidate."
            )
        probes[parameter] = probe

    definitions = (
        ("B000_all_zero", 0.0, 0.0, 0.0),
        ("B100_delta_E", probes["delta_E"], 0.0, 0.0),
        ("B010_c_A", 0.0, probes["c_A"], 0.0),
        ("B001_c_G", 0.0, 0.0, probes["c_G"]),
    )
    profiles: list[dict[str, Any]] = []
    for profile_id, delta_e, c_a, c_g in definitions:
        event_error = reactive_error_probability(event_correct, delta_e)
        values = {
            "delta_E": float(delta_e),
            "c_A": float(c_a),
            "c_G": float(c_g),
        }
        profiles.append(
            {
                "profile_id": profile_id,
                "values": values,
                "hyperparams": {
                    f"{CONTROLLER_PATH}.event_after_correct": event_correct,
                    f"{CONTROLLER_PATH}.event_after_error": event_error,
                    f"{CONTROLLER_PATH}.initial_event_probability": event_correct,
                    f"{CONTROLLER_PATH}.accumulator_logit_gain": float(c_a),
                    f"{CONTROLLER_PATH}.global_search_failure_gain": float(c_g),
                },
                "is_all_zero_boundary": bool(
                    delta_e == 0.0 and c_a == 0.0 and c_g == 0.0
                ),
            }
        )
    return profiles


def summarize_boundary_recovery(
    scores: pd.DataFrame,
    profiles: Sequence[Mapping[str, Any]],
    *,
    near_best_delta_nll: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize candidate winners without treating smoke rates as evidence."""

    expected_profiles = sorted(str(profile["profile_id"]) for profile in profiles)
    profile_lookup = {
        str(profile["profile_id"]): dict(profile["values"]) for profile in profiles
    }
    recovered_rows: list[dict[str, Any]] = []
    for dataset_id, frame in scores.groupby("dataset_id", sort=True):
        candidate_ids = sorted(frame["fit_profile_id"].astype(str).tolist())
        if candidate_ids != expected_profiles:
            raise ValueError(f"dataset {dataset_id} has an incomplete candidate bank")
        if not np.all(np.isfinite(frame["total_nll"].to_numpy(dtype=float))):
            raise ValueError(f"dataset {dataset_id} has non-finite candidate scores")
        ranked = frame.sort_values(["total_nll", "fit_profile_id"]).reset_index(drop=True)
        winner = ranked.iloc[0]
        true_profile_id = str(frame["true_profile_id"].iloc[0])
        predicted_profile_id = str(winner["fit_profile_id"])
        true_row = frame.loc[frame["fit_profile_id"].astype(str).eq(true_profile_id)]
        if len(true_row) != 1:
            raise ValueError(f"dataset {dataset_id} does not have one true candidate")
        best_nll = float(winner["total_nll"])
        true_nll = float(true_row["total_nll"].iloc[0])
        row: dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "subject_id": int(frame["subject_id"].iloc[0]),
            "trial_count": int(frame["trial_count"].iloc[0]),
            "true_profile_id": true_profile_id,
            "predicted_profile_id": predicted_profile_id,
            "exact_profile_recovered": predicted_profile_id == true_profile_id,
            "true_delta_nll": true_nll - best_nll,
            "true_within_delta_nll": (
                true_nll <= best_nll + float(near_best_delta_nll)
            ),
            "generated_accuracy": float(frame["generated_accuracy"].iloc[0]),
        }
        for parameter in ("delta_E", "c_A", "c_G"):
            true_value = float(profile_lookup[true_profile_id][parameter])
            predicted_value = float(profile_lookup[predicted_profile_id][parameter])
            row[f"true_{parameter}"] = true_value
            row[f"predicted_{parameter}"] = predicted_value
            row[f"true_{parameter}_positive"] = true_value > 0.0
            row[f"predicted_{parameter}_positive"] = predicted_value > 0.0
            row[f"boundary_{parameter}_recovered"] = (
                (true_value > 0.0) == (predicted_value > 0.0)
            )
        recovered_rows.append(row)
    recovered = pd.DataFrame(recovered_rows)

    boundary_rows: list[dict[str, Any]] = []
    for parameter in ("delta_E", "c_A", "c_G"):
        correct = recovered[f"boundary_{parameter}_recovered"].astype(bool)
        true_positive = recovered[f"true_{parameter}_positive"].astype(bool)
        boundary_rows.append(
            {
                "parameter": parameter,
                "dataset_n": int(len(correct)),
                "true_zero_dataset_n": int((~true_positive).sum()),
                "true_positive_dataset_n": int(true_positive.sum()),
                "boundary_recovery_count": int(correct.sum()),
                "boundary_recovery_rate": float(correct.mean()),
                "formal_recovery_evidence": False,
            }
        )
    boundary_summary = pd.DataFrame(boundary_rows)
    summary = {
        "status": "smoke_complete",
        "interpretation": "pipeline_smoke_not_recovery_evidence",
        "independent_unit": "one autonomous synthetic choice trajectory",
        "dataset_n": int(len(recovered)),
        "profile_count": int(len(expected_profiles)),
        "exact_profile_recovery_count": int(
            recovered["exact_profile_recovered"].sum()
        ),
        "exact_profile_recovery_rate": float(
            recovered["exact_profile_recovered"].mean()
        ),
        "true_within_delta_nll_count": int(
            recovered["true_within_delta_nll"].sum()
        ),
        "true_within_delta_nll_rate": float(
            recovered["true_within_delta_nll"].mean()
        ),
        "near_best_delta_nll": float(near_best_delta_nll),
        "all_candidate_banks_complete": True,
        "all_scores_finite": True,
        "all_zero_positive_boundaries_exercised": bool(
            np.all(boundary_summary["true_zero_dataset_n"].gt(0))
            and np.all(boundary_summary["true_positive_dataset_n"].gt(0))
        ),
        "boundary_results": boundary_summary.to_dict(orient="records"),
        "warning": (
            "This staged design is not the formal recovery experiment; winner rates "
            "remain diagnostic until the PF budget and recovery design are frozen."
        ),
    }
    return recovered, boundary_summary, summary


def _repo_path(raw: Any) -> Path:
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
    if unsupported_noise != 0.0 or str(noise.get("lapse_target")) != "uniform":
        raise ValueError("Model 0818 recovery requires the frozen uniform no-lapse readout.")
    return {
        "choice_readout_power": float(readout["power"]),
        "strategy_confidence_gain": float(readout["strategy_confidence_gain"]),
        "rule_commitment_confidence_gain": float(
            readout["rule_commitment_confidence_gain"]
        ),
        "output_lapse": float(noise.get("base_lapse", 0.0)),
    }


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


def _dataset_id(profile_id: str, subject_id: int, replicate: int) -> str:
    return f"{profile_id}_subject_{subject_id}_replicate_{replicate:02d}"


def _synthetic_path(output: Path, dataset_id: str) -> Path:
    return output / "synthetic" / f"{dataset_id}.npz"


def _score_cache_path(
    output: Path, dataset_id: str, cache_namespace: str
) -> Path:
    return output / "cache" / str(cache_namespace) / f"{dataset_id}.json"


def resolve_filter_seeds(
    *,
    dataset_id: str,
    base_seed: int,
    particle_count: int,
    filter_seed_count: int,
    seed_family: str | None = None,
) -> list[int]:
    """Resolve candidate-paired seeds, optionally shared across PF budgets."""

    if filter_seed_count < 1:
        raise ValueError("filter_seed_count must be positive")
    seeds: list[int] = []
    for repeat in range(filter_seed_count):
        payload: dict[str, Any] = {
            "seed_role": "model0818_boundary_recovery_paired_filter",
            "base_seed": int(base_seed),
            "dataset_id": str(dataset_id),
            "filter_repeat": repeat,
        }
        if seed_family is None:
            payload["particle_count"] = int(particle_count)
        else:
            payload["seed_family"] = str(seed_family)
        seeds.append(stable_seed(payload))
    return seeds


def generate_dataset(
    *,
    output: Path,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    subject_frame: pd.DataFrame,
    profile: Mapping[str, Any],
    replicate: int,
    trials: int,
    base_seed: int,
    force: bool,
) -> dict[str, Any]:
    subject_id = int(subject_frame["iSub"].iloc[0])
    dataset_id = _dataset_id(str(profile["profile_id"]), subject_id, replicate)
    path = _synthetic_path(output, dataset_id)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as payload:
            return dict(json.loads(str(payload["metadata_json"].item())))
    if len(subject_frame) < trials:
        raise ValueError(f"subject {subject_id} has fewer than {trials} schedule trials")
    frame = subject_frame.iloc[:trials]
    stimulus = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    categories = frame["category"].to_numpy(dtype=int)
    engine = apply_fixed_hyperparams_to_engine_config(
        _subject_engine(base_config, base_path, subject_id),
        profile["hyperparams"],
    )
    generation_seed = stable_seed(
        {
            "seed_role": "model0818_boundary_recovery_generation",
            "base_seed": int(base_seed),
            "dataset_id": dataset_id,
            "true_values": dict(profile["values"]),
        }
    )
    trajectory = run_autonomous_category_learning(
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
        "trial_count": int(trials),
        "true_profile_id": str(profile["profile_id"]),
        "true_values": dict(profile["values"]),
        "generation_seed": int(generation_seed),
        "generated_accuracy": float(np.mean(trajectory.feedback)),
        "observed_choices_used": False,
        "schedule_source": "condition_1_stimulus_and_category_columns_only",
    }
    _atomic_npz(
        path,
        stimulus=stimulus.astype(np.float32),
        categories=categories.astype(np.int8),
        choices=trajectory.choices.astype(np.int8),
        feedback=trajectory.feedback.astype(np.float32),
        generated_choice_probabilities=trajectory.observed_probabilities.astype(
            np.float32
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def fit_dataset(
    *,
    output: Path,
    metadata: Mapping[str, Any],
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    profiles: Sequence[Mapping[str, Any]],
    particle_count: int,
    filter_seed_count: int,
    resample_threshold_fraction: float,
    base_seed: int,
    force: bool,
    cache_namespace: str = "primary",
    filter_seed_family: str | None = None,
    seed_ensemble: str = "primary",
) -> list[dict[str, Any]]:
    dataset_id = str(metadata["dataset_id"])
    cache_path = _score_cache_path(output, dataset_id, cache_namespace)
    if cache_path.exists() and not force:
        cached_rows = list(json.loads(cache_path.read_text(encoding="utf-8")))
        for row in cached_rows:
            cached_family = row.get("filter_seed_family")
            if (
                filter_seed_family is not None
                and cached_family is not None
                and cached_family != filter_seed_family
            ):
                raise ValueError("cached score seed family does not match config")
            row["seed_ensemble"] = str(seed_ensemble)
        return cached_rows
    with np.load(_synthetic_path(output, dataset_id), allow_pickle=False) as payload:
        stimulus = payload["stimulus"].astype(float)
        choices = payload["choices"].astype(int)
        feedback = payload["feedback"].astype(float)
    subject_id = int(metadata["subject_id"])
    filter_seeds = resolve_filter_seeds(
        dataset_id=dataset_id,
        base_seed=base_seed,
        particle_count=particle_count,
        filter_seed_count=filter_seed_count,
        seed_family=filter_seed_family,
    )
    rows: list[dict[str, Any]] = []
    observed_index = choices - 1
    for profile in profiles:
        engine = apply_fixed_hyperparams_to_engine_config(
            _subject_engine(base_config, base_path, subject_id),
            profile["hyperparams"],
        )
        readout_args = _readout_args(engine)
        probability_runs: list[np.ndarray] = []
        mean_pre_choice_ess: list[float] = []
        resampling_fraction: list[float] = []
        for filter_seed in filter_seeds:
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
            probability_runs.append(
                np.asarray(result.marginal_probabilities, dtype=float)
            )
            mean_pre_choice_ess.append(
                float(np.mean(np.asarray(result.pre_choice_ess, dtype=float)))
            )
            resampling_fraction.append(
                float(np.mean(np.asarray(result.resampled, dtype=float)))
            )
        mean_probability = np.mean(np.stack(probability_runs, axis=0), axis=0)
        selected = mean_probability[np.arange(choices.size), observed_index]
        if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
            raise ValueError("seed-averaged observed-choice probabilities are invalid")
        total_nll = float(-np.log(np.clip(selected, 1e-12, 1.0)).sum())
        rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "trial_count": int(choices.size),
                "true_profile_id": str(metadata["true_profile_id"]),
                "fit_profile_id": str(profile["profile_id"]),
                "total_nll": total_nll,
                "mean_trial_nll": total_nll / float(choices.size),
                "particle_count": int(particle_count),
                "filter_seed_count": int(filter_seed_count),
                "filter_seeds": [int(value) for value in filter_seeds],
                "probability_aggregation": "mean_probability_then_nll",
                "generated_accuracy": float(metadata["generated_accuracy"]),
                "mean_pre_choice_ess": float(np.mean(mean_pre_choice_ess)),
                "resampling_fraction": float(np.mean(resampling_fraction)),
                "cache_namespace": str(cache_namespace),
                "filter_seed_family": filter_seed_family,
                "seed_ensemble": str(seed_ensemble),
            }
        )
    _atomic_json(cache_path, rows)
    return rows


def summarize_numerical_stability(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize PF-budget sensitivity without adding recovery replicates."""

    required = {
        "dataset_id",
        "true_profile_id",
        "fit_profile_id",
        "particle_count",
        "filter_seed_count",
        "total_nll",
    }
    if not required.issubset(scores):
        raise ValueError("numerical-stability scores are missing required columns")
    scores = scores.copy()
    if "seed_ensemble" not in scores:
        scores["seed_ensemble"] = "A"
    winners: list[dict[str, Any]] = []
    for keys, frame in scores.groupby(
        ["dataset_id", "particle_count", "filter_seed_count", "seed_ensemble"],
        sort=True,
    ):
        dataset_id, particle_count, filter_seed_count, seed_ensemble = keys
        ranked = frame.sort_values(["total_nll", "fit_profile_id"])
        winner = ranked.iloc[0]
        winners.append(
            {
                "dataset_id": str(dataset_id),
                "true_profile_id": str(frame["true_profile_id"].iloc[0]),
                "particle_count": int(particle_count),
                "filter_seed_count": int(filter_seed_count),
                "seed_ensemble": str(seed_ensemble),
                "setting": (
                    f"R{int(particle_count)}_B{int(filter_seed_count)}_"
                    f"{seed_ensemble}"
                ),
                "predicted_profile_id": str(winner["fit_profile_id"]),
                "true_profile_recovered": bool(
                    str(winner["fit_profile_id"])
                    == str(frame["true_profile_id"].iloc[0])
                ),
                "best_total_nll": float(winner["total_nll"]),
            }
        )
    winner_frame = pd.DataFrame(winners)

    correlation_rows: list[dict[str, Any]] = []
    for dataset_id, frame in scores.groupby("dataset_id", sort=True):
        settings = {
            (
                f"R{int(particle_count)}_B{int(filter_seed_count)}_"
                f"{seed_ensemble}"
            ): group.sort_values("fit_profile_id")["total_nll"].to_numpy(dtype=float)
            for (particle_count, filter_seed_count, seed_ensemble), group in frame.groupby(
                ["particle_count", "filter_seed_count", "seed_ensemble"], sort=True
            )
        }
        setting_names = sorted(settings)
        for left_index, left in enumerate(setting_names):
            for right in setting_names[left_index + 1 :]:
                statistic = float(spearmanr(settings[left], settings[right]).statistic)
                correlation_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "left_setting": left,
                        "right_setting": right,
                        "candidate_nll_spearman": statistic,
                    }
                )
    correlations = pd.DataFrame(correlation_rows)
    modal_agreement: list[float] = []
    for _, frame in winner_frame.groupby("dataset_id", sort=True):
        modal_agreement.append(
            float(frame["predicted_profile_id"].value_counts().iloc[0] / len(frame))
        )
    summary = {
        "interpretation": "targeted_pf_budget_diagnostic_not_additional_recovery_n",
        "dataset_n": int(winner_frame["dataset_id"].nunique()),
        "setting_n": int(
            winner_frame[["particle_count", "filter_seed_count", "seed_ensemble"]]
            .drop_duplicates()
            .shape[0]
        ),
        "winner_setting_rows": int(len(winner_frame)),
        "mean_within_dataset_modal_winner_agreement": float(
            np.mean(modal_agreement)
        ),
        "median_candidate_nll_rank_spearman": float(
            correlations["candidate_nll_spearman"].median()
        ),
        "minimum_candidate_nll_rank_spearman": float(
            correlations["candidate_nll_spearman"].min()
        ),
        "true_profile_recovery_rate_across_settings": float(
            winner_frame["true_profile_recovered"].mean()
        ),
    }
    return winner_frame, correlations, summary


def summarize_high_budget_calibration(
    scores: pd.DataFrame,
    gates: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply predeclared convergence gates to R32/R64 and seed ensembles A/B."""

    required = {
        "dataset_id",
        "fit_profile_id",
        "particle_count",
        "filter_seed_count",
        "seed_ensemble",
        "total_nll",
    }
    if not required.issubset(scores):
        raise ValueError("high-budget calibration scores are missing required columns")
    particle_counts = [int(value) for value in gates["required_particle_counts"]]
    seed_count = int(gates["required_filter_seed_count"])
    ensembles = [str(value) for value in gates["required_seed_ensembles"]]
    if particle_counts != [32, 64] or ensembles != ["A", "B"]:
        raise ValueError("high-budget calibration requires R32/R64 and ensembles A/B")

    selected = scores[
        scores["particle_count"].astype(int).isin(particle_counts)
        & scores["filter_seed_count"].astype(int).eq(seed_count)
        & scores["seed_ensemble"].astype(str).isin(ensembles)
    ].copy()
    comparison_specs = (
        ("particle_scaling", "R32_A_vs_R64_A", (32, "A"), (64, "A")),
        ("particle_scaling", "R32_B_vs_R64_B", (32, "B"), (64, "B")),
        ("seed_replication_R32", "R32_A_vs_B", (32, "A"), (32, "B")),
        ("seed_replication_R64", "R64_A_vs_B", (64, "A"), (64, "B")),
    )
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset_scores in selected.groupby("dataset_id", sort=True):
        setting_scores: dict[tuple[int, str], pd.DataFrame] = {}
        for key, frame in dataset_scores.groupby(
            ["particle_count", "seed_ensemble"], sort=True
        ):
            particle_count, seed_ensemble = key
            ranked = frame.sort_values("fit_profile_id")
            if ranked["fit_profile_id"].duplicated().any():
                raise ValueError("duplicate candidates in high-budget setting")
            setting_scores[(int(particle_count), str(seed_ensemble))] = ranked
        expected = {(r, ensemble) for r in particle_counts for ensemble in ensembles}
        if set(setting_scores) != expected:
            raise ValueError(
                f"incomplete high-budget settings for dataset {dataset_id}"
            )
        for category, comparison, left_key, right_key in comparison_specs:
            left = setting_scores[left_key]
            right = setting_scores[right_key]
            left_profiles = left["fit_profile_id"].astype(str).tolist()
            right_profiles = right["fit_profile_id"].astype(str).tolist()
            if left_profiles != right_profiles:
                raise ValueError("candidate profiles differ across calibration settings")
            left_nll = left["total_nll"].to_numpy(dtype=float)
            right_nll = right["total_nll"].to_numpy(dtype=float)
            rank_rho = float(spearmanr(left_nll, right_nll).statistic)
            left_delta = left_nll - float(np.min(left_nll))
            right_delta = right_nll - float(np.min(right_nll))
            left_winner = left_profiles[int(np.argmin(left_nll))]
            right_winner = right_profiles[int(np.argmin(right_nll))]
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "comparison_category": category,
                    "comparison": comparison,
                    "left_setting": f"R{left_key[0]}_B{seed_count}_{left_key[1]}",
                    "right_setting": f"R{right_key[0]}_B{seed_count}_{right_key[1]}",
                    "candidate_nll_rank_spearman": rank_rho,
                    "same_winner": bool(left_winner == right_winner),
                    "left_winner": left_winner,
                    "right_winner": right_winner,
                    "maximum_absolute_delta_nll_difference": float(
                        np.max(np.abs(left_delta - right_delta))
                    ),
                }
            )
    comparison_frame = pd.DataFrame(rows)

    median_gate = float(gates["minimum_median_candidate_rank_spearman"])
    worst_gate = float(gates["minimum_worst_candidate_rank_spearman"])
    winner_gate = float(gates["minimum_winner_agreement"])
    category_summary: dict[str, dict[str, Any]] = {}
    for category, frame in comparison_frame.groupby("comparison_category", sort=True):
        finite_rho = frame["candidate_nll_rank_spearman"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        median_rho = float(finite_rho.median()) if len(finite_rho) else None
        worst_rho = float(finite_rho.min()) if len(finite_rho) else None
        winner_agreement = float(frame["same_winner"].mean())
        passed = bool(
            median_rho is not None
            and worst_rho is not None
            and median_rho >= median_gate
            and worst_rho >= worst_gate
            and winner_agreement >= winner_gate
        )
        category_summary[str(category)] = {
            "comparison_n": int(len(frame)),
            "median_candidate_nll_rank_spearman": median_rho,
            "minimum_candidate_nll_rank_spearman": worst_rho,
            "winner_agreement": winner_agreement,
            "passed": passed,
        }

    r32_pass = bool(category_summary["seed_replication_R32"]["passed"])
    r64_pass = bool(category_summary["seed_replication_R64"]["passed"])
    scaling_pass = bool(category_summary["particle_scaling"]["passed"])
    if r32_pass and r64_pass and scaling_pass:
        budget_status = "provisional_minimum_R32_B4"
        minimum_budget = {"particle_count": 32, "filter_seed_count": 4}
    elif r64_pass:
        budget_status = "provisional_minimum_R64_B4"
        minimum_budget = {"particle_count": 64, "filter_seed_count": 4}
    else:
        budget_status = "not_frozen"
        minimum_budget = None
    winner_gaps: list[float] = []
    for _, frame in selected.groupby(
        ["dataset_id", "particle_count", "filter_seed_count", "seed_ensemble"],
        sort=True,
    ):
        ordered_nll = np.sort(frame["total_nll"].to_numpy(dtype=float))
        if ordered_nll.size < 2:
            raise ValueError("at least two candidates are required for calibration")
        winner_gaps.append(float(ordered_nll[1] - ordered_nll[0]))
    comparison_changes = comparison_frame[
        "maximum_absolute_delta_nll_difference"
    ].to_numpy(dtype=float)
    summary = {
        "interpretation": "predeclared_targeted_pf_budget_calibration",
        "dataset_n": int(comparison_frame["dataset_id"].nunique()),
        "gates": {
            "minimum_median_candidate_rank_spearman": median_gate,
            "minimum_worst_candidate_rank_spearman": worst_gate,
            "minimum_winner_agreement": winner_gate,
        },
        "category_summary": category_summary,
        "budget_status": budget_status,
        "provisional_minimum_budget": minimum_budget,
        "formal_recovery_authorized": bool(minimum_budget is not None),
        "observed_data_fit_authorized": False,
        "score_separation_diagnostics": {
            "setting_dataset_n": len(winner_gaps),
            "median_winner_to_runner_up_delta_nll": float(
                np.median(winner_gaps)
            ),
            "maximum_winner_to_runner_up_delta_nll": float(np.max(winner_gaps)),
            "median_cross_setting_maximum_absolute_delta_nll_difference": float(
                np.median(comparison_changes)
            ),
            "maximum_cross_setting_absolute_delta_nll_difference": float(
                np.max(comparison_changes)
            ),
        },
    }
    if {"mean_pre_choice_ess", "resampling_fraction"}.issubset(selected):
        summary["particle_filter_diagnostics"] = {
            "mean_pre_choice_ess_fraction": float(
                np.mean(
                    selected["mean_pre_choice_ess"].to_numpy(dtype=float)
                    / selected["particle_count"].to_numpy(dtype=float)
                )
            ),
            "mean_resampling_fraction": float(
                selected["resampling_fraction"].mean()
            ),
        }
    return comparison_frame, summary


def validate_high_budget_seed_design(scores: pd.DataFrame) -> dict[str, Any]:
    """Verify paired R comparisons and independent A/B seed ensembles."""

    required = {
        "dataset_id",
        "particle_count",
        "filter_seed_count",
        "seed_ensemble",
        "filter_seeds",
    }
    if not required.issubset(scores):
        raise ValueError("high-budget seed audit is missing required columns")
    audited = scores[
        scores["particle_count"].astype(int).isin([32, 64])
        & scores["filter_seed_count"].astype(int).eq(4)
        & scores["seed_ensemble"].astype(str).isin(["A", "B"])
    ]
    dataset_rows: list[dict[str, Any]] = []
    for dataset_id, frame in audited.groupby("dataset_id", sort=True):
        seeds_by_setting: dict[tuple[int, str], tuple[int, ...]] = {}
        for key, setting in frame.groupby(
            ["particle_count", "seed_ensemble"], sort=True
        ):
            unique_seeds = {
                tuple(int(seed) for seed in value)
                for value in setting["filter_seeds"]
            }
            if len(unique_seeds) != 1:
                raise ValueError("candidates do not share filter seeds within setting")
            particle_count, ensemble = key
            seeds_by_setting[(int(particle_count), str(ensemble))] = (
                unique_seeds.pop()
            )
        expected = {(32, "A"), (64, "A"), (32, "B"), (64, "B")}
        if set(seeds_by_setting) != expected:
            raise ValueError("high-budget seed settings are incomplete")
        paired_a = seeds_by_setting[(32, "A")] == seeds_by_setting[(64, "A")]
        paired_b = seeds_by_setting[(32, "B")] == seeds_by_setting[(64, "B")]
        disjoint = set(seeds_by_setting[(32, "A")]).isdisjoint(
            seeds_by_setting[(32, "B")]
        )
        if not (paired_a and paired_b and disjoint):
            raise ValueError("high-budget paired/independent seed design is invalid")
        dataset_rows.append(
            {
                "dataset_id": str(dataset_id),
                "paired_R32_R64_within_A": paired_a,
                "paired_R32_R64_within_B": paired_b,
                "A_B_seed_sets_disjoint": disjoint,
            }
        )
    if not dataset_rows:
        raise ValueError("no datasets were available for high-budget seed audit")
    return {
        "dataset_n": len(dataset_rows),
        "candidates_share_seeds_within_setting": True,
        "paired_R32_R64_within_ensemble": True,
        "A_B_seed_sets_disjoint": True,
    }


def _write_report(output: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Model 0818 boundary recovery",
        "",
        f"- Status: `{summary['status']}`",
        f"- Synthetic datasets: {summary['dataset_n']}",
        f"- Candidate profiles: {summary['profile_count']}",
        (
            "- Exact profile recovery: "
            f"{summary['exact_profile_recovery_count']}/{summary['dataset_n']}"
        ),
        (
            "- Generating profile within ΔNLL threshold: "
            f"{summary['true_within_delta_nll_count']}/{summary['dataset_n']}"
        ),
        "",
        "This staged run verifies synthetic generation, seed-averaged PF scoring, "
        "and zero/positive recovery. A smoke or pilot tier is not formal recovery evidence.",
    ]
    stability = dict(summary.get("numerical_stability", {}))
    if int(stability.get("dataset_n", 0)) > 0:
        lines.extend(
            [
                "",
                "## PF numerical stability",
                "",
                (
                    "- Mean within-dataset modal winner agreement: "
                    f"{float(stability['mean_within_dataset_modal_winner_agreement']):.3f}"
                ),
                (
                    "- Median candidate-NLL rank Spearman: "
                    f"{float(stability['median_candidate_nll_rank_spearman']):.3f}"
                ),
                (
                    "- Minimum candidate-NLL rank Spearman: "
                    f"{float(stability['minimum_candidate_nll_rank_spearman']):.3f}"
                ),
            ]
        )
    calibration = dict(summary.get("pf_budget_calibration", {}))
    if int(calibration.get("dataset_n", 0)) > 0:
        category_summary = dict(calibration["category_summary"])
        lines.extend(
            [
                "",
                "## Predeclared high-budget calibration",
                "",
                f"- Budget decision: `{calibration['budget_status']}`",
                (
                    "- Formal recovery authorized: "
                    f"`{str(bool(calibration['formal_recovery_authorized'])).lower()}`"
                ),
                "- Observed-data fitting authorized: `false`",
                (
                    "- Particle scaling (R32→R64): median/min rank rho "
                    f"{category_summary['particle_scaling']['median_candidate_nll_rank_spearman']:.3f}/"
                    f"{category_summary['particle_scaling']['minimum_candidate_nll_rank_spearman']:.3f}; "
                    f"winner agreement {category_summary['particle_scaling']['winner_agreement']:.3f}"
                ),
                (
                    "- Independent seeds at R32/B4: median/min rank rho "
                    f"{category_summary['seed_replication_R32']['median_candidate_nll_rank_spearman']:.3f}/"
                    f"{category_summary['seed_replication_R32']['minimum_candidate_nll_rank_spearman']:.3f}; "
                    f"winner agreement {category_summary['seed_replication_R32']['winner_agreement']:.3f}"
                ),
                (
                    "- Independent seeds at R64/B4: median/min rank rho "
                    f"{category_summary['seed_replication_R64']['median_candidate_nll_rank_spearman']:.3f}/"
                    f"{category_summary['seed_replication_R64']['minimum_candidate_nll_rank_spearman']:.3f}; "
                    f"winner agreement {category_summary['seed_replication_R64']['winner_agreement']:.3f}"
                ),
                "",
                "This two-trajectory audit can freeze a provisional PF budget for the "
                "formal synthetic recovery. Observed choices remain locked until formal "
                "recovery succeeds.",
            ]
        )
    (output / "recovery_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


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
    parameter_space_path = _repo_path(config["parameter_space"])
    base_path = _repo_path(config["base_simulation_config"])
    configured_output = _repo_path(config["output_dir"])
    output = args.output_dir.resolve() if args.output_dir else configured_output
    if args.smoke:
        output = output / "smoke"
        analysis_tier = "smoke"
    elif args.pilot:
        output = output / "pilot"
        analysis_tier = "pilot"
    else:
        analysis_tier = "configured_design"
    parameter_space = load_parameter_space(parameter_space_path)
    base_config = load_yaml(base_path)
    profiles = build_boundary_profiles(parameter_space, config["positive_probes"])
    if len(profiles) != int(config["report"]["expected_profile_count"]):
        raise ValueError("boundary profile count does not match the recovery config")

    design = deepcopy(dict(config["design"]))
    if args.smoke:
        design.update(dict(config["smoke"]))
    elif args.pilot:
        design.update(dict(config["pilot"]))
    subjects = [int(value) for value in design["template_subjects"]]
    repeats = int(design["datasets_per_profile_per_subject"])
    trials = int(design["trials_per_dataset"])
    particle_count = int(design["particle_count"])
    filter_seed_count = int(design["filter_seed_count"])
    n_jobs = int(args.n_jobs if args.n_jobs is not None else design["n_jobs"])
    if repeats <= 0 or trials <= 0 or particle_count < 2:
        raise ValueError("invalid boundary-recovery compute budget")
    if filter_seed_count < 2 or n_jobs <= 0:
        raise ValueError("boundary recovery requires at least two filter seeds and one job")

    frames, dataset_paths = _load_subject_frames(base_config, base_path, subjects)
    output.mkdir(parents=True, exist_ok=True)
    engine_path_raw = Path(str(base_config["engine_config_path"]))
    engine_path = (
        engine_path_raw.resolve()
        if engine_path_raw.is_absolute()
        else (base_path.parent / engine_path_raw).resolve()
    )
    manuscript_path = ROOT / "manuscript/model_0818.tex"
    manifest = {
        "analysis_id": str(config["analysis_id"]),
        "scope": str(config["scope"]),
        "status": "configured",
        "smoke": bool(args.smoke),
        "pilot": bool(args.pilot),
        "analysis_tier": analysis_tier,
        "resolved_output_dir": str(output),
        "observed_choices_used": False,
        "schedule_columns_used": [*FEATURE_COLUMNS, "category"],
        "subjects": subjects,
        "trials_per_dataset": trials,
        "profile_count": len(profiles),
        "expected_dataset_n": len(profiles) * len(subjects) * repeats,
        "particle_count": particle_count,
        "filter_seed_count": filter_seed_count,
        "probability_aggregation": "mean_probability_then_nll",
        "paired_filter_seeds_across_candidates": True,
        "stability_filter_seeds_paired_across_particle_counts": True,
        "stability_independent_seed_ensembles": ["A", "B"],
        "parameter_space_sha256": _sha256(parameter_space_path),
        "base_simulation_config_sha256": _sha256(base_path),
        "engine_config_sha256": _sha256(engine_path),
        "manuscript_sha256": _sha256(manuscript_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_json(output / "analysis_manifest.json", manifest)
    _atomic_json(output / "analysis_config_snapshot.json", config)
    _atomic_json(output / "parameter_space_snapshot.json", parameter_space)
    _atomic_json(output / "candidate_profiles.json", profiles)

    jobs = [
        (profile, subject_id, replicate)
        for profile in profiles
        for subject_id in subjects
        for replicate in range(repeats)
    ]
    base_seed = int(design["base_seed"])
    if args.phase in {"generate", "all"}:
        generated = Parallel(n_jobs=min(n_jobs, len(jobs)), backend="loky")(
            delayed(generate_dataset)(
                output=output,
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                subject_frame=frames[subject_id],
                profile=profile,
                replicate=replicate,
                trials=trials,
                base_seed=base_seed,
                force=bool(args.force),
            )
            for profile, subject_id, replicate in jobs
        )
        _atomic_json(output / "synthetic_manifest.json", generated)
        print(f"[0818 boundary recovery] generated={len(generated)}", flush=True)
        if args.phase == "generate":
            return
    elif not (output / "synthetic_manifest.json").exists():
        raise FileNotFoundError("synthetic_manifest.json is required; run generation first")

    generated = list(
        json.loads((output / "synthetic_manifest.json").read_text(encoding="utf-8"))
    )
    if len(generated) != len(jobs):
        raise ValueError("synthetic dataset count does not match the configured design")
    if args.phase in {"fit", "all"}:
        fitted = Parallel(n_jobs=min(n_jobs, len(generated)), backend="loky")(
            delayed(fit_dataset)(
                output=output,
                metadata=metadata,
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                profiles=profiles,
                particle_count=particle_count,
                filter_seed_count=filter_seed_count,
                resample_threshold_fraction=float(
                    design["resample_threshold_fraction"]
                ),
                base_seed=base_seed,
                force=bool(args.force),
            )
            for metadata in generated
        )
        scores = pd.DataFrame([row for rows in fitted for row in rows])
        _atomic_csv(output / "fit_scores.csv", scores)
        print(f"[0818 boundary recovery] fit_rows={len(scores)}", flush=True)
        if args.phase == "fit":
            return
    elif not (output / "fit_scores.csv").exists():
        raise FileNotFoundError("fit_scores.csv is required; run fitting first")

    stability_summary: dict[str, Any] = {
        "interpretation": "not_run_for_this_tier",
        "dataset_n": 0,
        "setting_n": 0,
    }
    calibration_summary: dict[str, Any] = {
        "interpretation": "not_run_for_this_tier",
        "dataset_n": 0,
        "budget_status": "not_frozen",
        "formal_recovery_authorized": False,
        "observed_data_fit_authorized": False,
    }
    stability_config = dict(config.get("numerical_stability", {}))
    stability_enabled = bool(stability_config.get("enabled", False)) and args.pilot
    if args.phase == "all" and stability_enabled:
        selected_profile_ids = {
            str(value) for value in stability_config["profile_ids"]
        }
        selected = [
            metadata
            for metadata in generated
            if str(metadata["true_profile_id"]) in selected_profile_ids
            and int(metadata["subject_id"]) == int(stability_config["subject_id"])
            and int(metadata["replicate"]) == int(stability_config["replicate"])
        ]
        if len(selected) != len(selected_profile_ids):
            raise ValueError("numerical-stability dataset selection is incomplete")
        settings = [dict(value) for value in stability_config["settings"]]
        setting_keys = [
            (
                int(value["particle_count"]),
                int(value["filter_seed_count"]),
                str(value["seed_ensemble"]),
            )
            for value in settings
        ]
        if len(setting_keys) != len(set(setting_keys)):
            raise ValueError("numerical-stability R/B/ensemble settings must be unique")
        stability_jobs = [
            (metadata, setting)
            for metadata in selected
            for setting in settings
        ]
        stability_fitted = Parallel(
            n_jobs=min(n_jobs, len(stability_jobs)), backend="loky"
        )(
            delayed(fit_dataset)(
                output=output,
                metadata=metadata,
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                profiles=profiles,
                particle_count=int(setting["particle_count"]),
                filter_seed_count=int(setting["filter_seed_count"]),
                resample_threshold_fraction=float(
                    design["resample_threshold_fraction"]
                ),
                base_seed=base_seed,
                force=bool(args.force),
                cache_namespace=str(setting["cache_namespace"]),
                filter_seed_family=str(setting["seed_family"]),
                seed_ensemble=str(setting["seed_ensemble"]),
            )
            for metadata, setting in stability_jobs
        )
        stability_scores = pd.DataFrame(
            [row for rows in stability_fitted for row in rows]
        )
        expected_stability_rows = len(stability_jobs) * len(profiles)
        if len(stability_scores) != expected_stability_rows:
            raise ValueError("numerical-stability score table is incomplete")
        _atomic_csv(output / "stability_scores.csv", stability_scores)
        (
            stability_winners,
            stability_correlations,
            stability_summary,
        ) = summarize_numerical_stability(stability_scores)
        _atomic_csv(output / "stability_winners.csv", stability_winners)
        _atomic_csv(
            output / "stability_candidate_rank_correlations.csv",
            stability_correlations,
        )
        _atomic_json(output / "stability_summary.json", stability_summary)
        high_budget_scores = stability_scores[
            stability_scores["particle_count"].astype(int).isin([32, 64])
            & stability_scores["filter_seed_count"].astype(int).eq(4)
            & stability_scores["seed_ensemble"].astype(str).isin(["A", "B"])
        ]
        high_budget_comparisons, calibration_summary = (
            summarize_high_budget_calibration(
                high_budget_scores,
                dict(stability_config["calibration_gates"]),
            )
        )
        calibration_summary["seed_design_audit"] = (
            validate_high_budget_seed_design(high_budget_scores)
        )
        _atomic_csv(
            output / "high_budget_comparisons.csv", high_budget_comparisons
        )
        _atomic_json(output / "pf_budget_calibration.json", calibration_summary)
        print(
            f"[0818 boundary recovery] stability_rows={len(stability_scores)}",
            flush=True,
        )

    scores = pd.read_csv(output / "fit_scores.csv")
    expected_rows = len(generated) * len(profiles)
    if len(scores) != expected_rows:
        raise ValueError(f"incomplete fit score table: {len(scores)} vs {expected_rows}")
    recovered, boundary_summary, summary = summarize_boundary_recovery(
        scores,
        profiles,
        near_best_delta_nll=float(design["near_best_delta_nll"]),
    )
    summary.update(
        {
            "analysis_id": str(config["analysis_id"]),
            "smoke": bool(args.smoke),
            "pilot": bool(args.pilot),
            "analysis_tier": analysis_tier,
            "trials_per_dataset": trials,
            "particle_count": particle_count,
            "filter_seed_count": filter_seed_count,
            "numerical_stability": stability_summary,
            "pf_budget_calibration": calibration_summary,
        }
    )
    if args.smoke:
        summary["status"] = "smoke_complete"
        summary["interpretation"] = "pipeline_smoke_not_recovery_evidence"
    elif args.pilot:
        summary["status"] = "pilot_complete"
        summary["interpretation"] = "targeted_recovery_pilot_not_formal_evidence"
    else:
        summary["status"] = "configured_design_complete"
    _atomic_csv(output / "recovered_datasets.csv", recovered)
    _atomic_csv(output / "boundary_recovery_summary.csv", boundary_summary)
    _atomic_json(output / "recovery_summary.json", summary)
    _write_report(output, summary)
    manifest.update({"status": str(summary["status"])})
    _atomic_json(output / "analysis_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
