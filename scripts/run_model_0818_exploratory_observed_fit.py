#!/usr/bin/env python3
"""Fit frozen Model 0818 exploratorily to observed condition-1 choices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0818_boundary_recovery import (  # noqa: E402
    FEATURE_COLUMNS,
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _load_subject_frames,
    _readout_args,
    _sha256,
    _subject_engine,
    resolve_filter_seeds,
)
from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.optimization.observed_fit import (  # noqa: E402
    build_model_0818_hyper_config,
    extract_model_0818_parameters,
    summarize_observed_choice_fit,
)
from src.Bayesian_state.optimization.parameter_space import (  # noqa: E402
    load_parameter_space,
)
from src.Bayesian_state.optimization.search.coordinate_descent import (  # noqa: E402
    HyperCDOptimizer,
)
from src.Bayesian_state.run_hyper_then_simulation import (  # noqa: E402
    aggregate_per_subject_best,
    materialize_simulation_config_from_hyper_best,
    save_yaml,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0818_exploratory_observed_fit.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=(
            "validate",
            "search-one",
            "search",
            "rescore",
            "summarize",
            "evaluation-config",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--subjects", nargs="+", type=int)
    parser.add_argument("--parallel-budget", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _select_subject_trials(
    frame: pd.DataFrame,
    trial_scope: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply the declared per-subject trial scope without hidden truncation."""

    mode = str(trial_scope.get("mode", "")).strip().lower()
    if mode == "all_trials_per_subject":
        if trial_scope.get("max_trials") is not None:
            raise ValueError("all_trials_per_subject requires max_trials: null")
        return frame.copy()
    if mode == "first_n_trials_per_subject":
        max_trials = int(trial_scope["max_trials"])
        if max_trials <= 0:
            raise ValueError("first_n_trials_per_subject requires max_trials > 0")
        if len(frame) < max_trials:
            raise ValueError(
                f"subject has {len(frame)} trials, fewer than requested {max_trials}"
            )
        return frame.iloc[:max_trials].copy()
    raise ValueError(f"unsupported trial_scope.mode: {mode!r}")


def _trial_scope_id(trial_scope: Mapping[str, Any]) -> str:
    mode = str(trial_scope.get("mode", "")).strip().lower()
    if mode == "all_trials_per_subject":
        return "all_trials"
    if mode == "first_n_trials_per_subject":
        return f"first_{int(trial_scope['max_trials'])}_trials"
    raise ValueError(f"unsupported trial_scope.mode: {mode!r}")


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_hyperparams(best_path: Path) -> dict[str, Any]:
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("best_hyperparams"), Mapping
    ):
        raise ValueError(f"search result lacks selected hyperparameters: {best_path}")
    return dict(selected["best_hyperparams"])


def _search_best_path(output: Path, subject_id: int) -> Path:
    return output / "search" / f"subject_{int(subject_id)}" / "best_hyperparams.json"


def _rescore_subject_dir(output: Path, subject_id: int) -> Path:
    return output / "final_rescore" / f"subject_{int(subject_id)}"


def _rescore_cache_path(
    output: Path,
    cache_namespace: str,
    subject_id: int,
    filter_seed: int,
) -> Path:
    return (
        output
        / "final_rescore"
        / f"subject_{int(subject_id)}"
        / "cache"
        / str(cache_namespace)
        / f"seed_{int(filter_seed)}.npz"
    )


def _read_cache(
    path: Path,
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = dict(json.loads(str(payload["metadata_json"].item())))
        probabilities = payload["marginal_probabilities"].astype(float)
        ess = payload["pre_choice_ess"].astype(float)
        resampled = payload["resampled"].astype(bool)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"cache metadata mismatch for {key}: {path}")
    if probabilities.shape != (int(metadata["trial_count"]), 2):
        raise ValueError(f"invalid cached probability shape: {path}")
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-8
    ):
        raise ValueError(f"invalid cached probabilities: {path}")
    if ess.shape != (probabilities.shape[0],) or resampled.shape != (
        probabilities.shape[0],
    ):
        raise ValueError(f"invalid cached PF diagnostics: {path}")
    return metadata, probabilities, ess, resampled


def validate_inputs(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    parameter_space: Mapping[str, Any],
    base_config: Mapping[str, Any],
    base_path: Path,
    output: Path,
) -> tuple[dict[int, pd.DataFrame], dict[str, Path]]:
    authorization = dict(config["authorization"])
    if not bool(authorization.get("user_authorized_observed_fit")):
        raise ValueError("observed-data fitting requires explicit user authorization")
    if not bool(authorization.get("exploratory_only")):
        raise ValueError("this runner is restricted to exploratory observed fitting")
    if any(
        bool(authorization.get(key))
        for key in (
            "formal_parameter_inference_authorized",
            "formal_module_inference_authorized",
            "manuscript_result_authorized",
        )
    ):
        raise ValueError("exploratory observed fitting cannot authorize formal inference")
    subjects = [int(value) for value in config["subjects"]]
    if subjects != list(range(101, 133)):
        raise ValueError("observed-fit subjects must be the ordered condition-1 cohort")
    if list(parameter_space["scope"]["subjects"]) != subjects:
        raise ValueError("parameter-space and observed-fit subjects differ")
    trial_scope = dict(config["trial_scope"])
    if int(trial_scope["condition"]) != 1:
        raise ValueError("observed fitting is restricted to condition 1")
    _trial_scope_id(trial_scope)
    if base_config.get("max_trials") != trial_scope.get("max_trials"):
        raise ValueError(
            "analysis and base simulation configs disagree about max_trials: "
            f"{trial_scope.get('max_trials')!r} vs {base_config.get('max_trials')!r}"
        )

    manuscript = ROOT / "manuscript/model_0818.tex"
    expected_manuscript_hash = str(
        parameter_space["provenance"]["manuscript_sha256"]
    )
    if _sha256(manuscript) != expected_manuscript_hash:
        raise ValueError("frozen Model 0818 manuscript hash changed")
    frames, dataset_paths = _load_subject_frames(base_config, base_path, subjects)
    data_rows: list[dict[str, Any]] = []
    for subject_id in subjects:
        frame = frames[subject_id]
        selected = _select_subject_trials(frame, trial_scope)
        if selected.empty:
            raise ValueError(f"subject {subject_id} has no selected condition-1 trials")
        required = [*FEATURE_COLUMNS, "category", "choice", "feedback"]
        if selected[required].isna().any().any():
            raise ValueError(f"subject {subject_id} has missing selected-trial data")
        if not set(selected["choice"].astype(int)).issubset({1, 2}):
            raise ValueError(f"subject {subject_id} has invalid choices")
        data_rows.append(
            {
                "subject_id": subject_id,
                "available_condition1_trials": int(len(frame)),
                "fitted_trials": int(len(selected)),
                "ambiguous_trial_count": int(selected["ambiguous"].astype(bool).sum()),
                "observed_accuracy": float(selected["feedback"].mean()),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis_id": str(config["analysis_id"]),
        "scope": str(config["scope"]),
        "status": "inputs_validated",
        "observed_choices_used": True,
        "exploratory_only": True,
        "formal_parameter_inference_authorized": False,
        "formal_module_inference_authorized": False,
        "manuscript_result_authorized": False,
        "subjects": subjects,
        "subject_count": len(subjects),
        "trial_scope": dict(config["trial_scope"]),
        "model_scope": dict(config["model_scope"]),
        "optimization": dict(config["optimization"]),
        "analysis_config_path": str(config_path),
        "analysis_config_sha256": _sha256(config_path),
        "base_simulation_config_path": str(base_path),
        "base_simulation_config_sha256": _sha256(base_path),
        "parameter_space_path": str(_repo_path(config["parameter_space"])),
        "parameter_space_sha256": _sha256(_repo_path(config["parameter_space"])),
        "manuscript_sha256": expected_manuscript_hash,
        "learning_data_path": str(dataset_paths["learning_data"]),
        "learning_data_sha256": _sha256(dataset_paths["learning_data"]),
        "perception_summary_sha256": _sha256(dataset_paths["perception_summary"]),
        "perception_summary_72_sha256": _sha256(
            dataset_paths["perception_summary_72"]
        ),
    }
    _atomic_json(output / "analysis_manifest.json", manifest)
    _atomic_json(output / "analysis_config_snapshot.json", config)
    _atomic_csv(output / "observed_data_audit.csv", pd.DataFrame(data_rows))
    return frames, dataset_paths


def run_search_subject(
    *,
    subject_id: int,
    hyper_config: Mapping[str, Any],
    config_path: Path,
    output: Path,
    force: bool,
) -> dict[str, Any]:
    best_path = _search_best_path(output, subject_id)
    if best_path.exists() and not force:
        return {"subject_id": int(subject_id), "cache_hit": True, "best_path": str(best_path)}
    optimizer = HyperCDOptimizer(hyper_config, config_path)
    result = optimizer.run_subject(int(subject_id), stage="coarse")
    return {
        "subject_id": int(subject_id),
        "cache_hit": False,
        "best_path": str(result["best_hyperparams"]),
    }


def run_rescore_job(
    *,
    output: Path,
    cache_namespace: str,
    subject_id: int,
    seed_index: int,
    filter_seed: int,
    frame: pd.DataFrame,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    hyperparams: Mapping[str, Any],
    particle_count: int,
    resample_threshold_fraction: float,
    seed_family: str,
    learning_data_sha256: str,
    trial_scope: Mapping[str, Any],
    force: bool,
) -> dict[str, Any]:
    cache_path = _rescore_cache_path(
        output, cache_namespace, subject_id, filter_seed
    )
    selected = _select_subject_trials(frame, trial_scope)
    hyperparams_sha256 = _json_sha256(hyperparams)
    expected = {
        "format_version": 1,
        "subject_id": int(subject_id),
        "trial_count": int(len(selected)),
        "seed_index": int(seed_index),
        "filter_seed": int(filter_seed),
        "seed_family": str(seed_family),
        "particle_count": int(particle_count),
        "hyperparams_sha256": hyperparams_sha256,
        "learning_data_sha256": learning_data_sha256,
        "observed_choices_used": True,
        "exploratory_only": True,
    }
    started = time.perf_counter()
    if cache_path.exists() and not force:
        _, probabilities, ess, resampled = _read_cache(cache_path, expected)
        cache_hit = True
    else:
        engine = apply_fixed_hyperparams_to_engine_config(
            _subject_engine(base_config, base_path, subject_id), hyperparams
        )
        result = run_state_model_particle_filter(
            engine_config=engine,
            subject_id=int(subject_id),
            stimulus=selected[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            choices=selected["choice"].to_numpy(dtype=int),
            feedback=selected["feedback"].to_numpy(dtype=float),
            particle_count=int(particle_count),
            filter_seed=int(filter_seed),
            resample_threshold_fraction=float(resample_threshold_fraction),
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            **_readout_args(engine),
        )
        probabilities = np.asarray(result.marginal_probabilities, dtype=float)
        ess = np.asarray(result.pre_choice_ess, dtype=float)
        resampled = np.asarray(result.resampled, dtype=bool)
        metadata = {**expected, "cache_namespace": str(cache_namespace)}
        _atomic_npz(
            cache_path,
            marginal_probabilities=probabilities,
            pre_choice_ess=ess,
            resampled=resampled.astype(np.uint8),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        )
        cache_hit = False
    observed = selected["choice"].to_numpy(dtype=int)
    chosen = probabilities[np.arange(observed.size), observed - 1]
    return {
        **expected,
        "total_nll_single_seed": float(-np.log(np.clip(chosen, 1e-12, 1.0)).sum()),
        "mean_pre_choice_ess": float(np.mean(ess)),
        "resampling_fraction": float(np.mean(resampled)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "cache_hit": bool(cache_hit),
        "cache_relpath": str(cache_path.relative_to(output)),
    }


def rescore_subjects(
    *,
    subjects: Sequence[int],
    config: Mapping[str, Any],
    output: Path,
    frames: Mapping[int, pd.DataFrame],
    dataset_paths: Mapping[str, Path],
    base_config: Mapping[str, Any],
    base_path: Path,
    n_jobs: int,
    force: bool,
) -> None:
    final = dict(config["optimization"]["final_rescore"])
    particle_count = int(final["particle_count"])
    filter_seed_count = int(final["filter_seed_count"])
    seed_family = str(final["seed_family"])
    cache_namespace = str(final["cache_namespace"])
    learning_hash = _sha256(dataset_paths["learning_data"])
    trial_scope = dict(config["trial_scope"])
    scope_id = _trial_scope_id(trial_scope)
    jobs: list[dict[str, Any]] = []
    for subject_id in subjects:
        best_path = _search_best_path(output, subject_id)
        if not best_path.exists():
            raise FileNotFoundError(f"search result required before rescore: {best_path}")
        hyperparams = _selected_hyperparams(best_path)
        filter_seeds = resolve_filter_seeds(
            dataset_id=f"observed_subject_{int(subject_id)}_{scope_id}",
            base_seed=int(config["hyper_base_seed"]),
            particle_count=particle_count,
            filter_seed_count=filter_seed_count,
            seed_family=seed_family,
        )
        for seed_index, filter_seed in enumerate(filter_seeds):
            jobs.append(
                {
                    "subject_id": int(subject_id),
                    "seed_index": int(seed_index),
                    "filter_seed": int(filter_seed),
                    "hyperparams": hyperparams,
                }
            )
    rows = Parallel(
        n_jobs=min(int(n_jobs), len(jobs)),
        backend="loky",
        batch_size=1,
        pre_dispatch="n_jobs",
        verbose=10,
    )(
        delayed(run_rescore_job)(
            output=output,
            cache_namespace=cache_namespace,
            subject_id=job["subject_id"],
            seed_index=job["seed_index"],
            filter_seed=job["filter_seed"],
            frame=frames[job["subject_id"]],
            base_config=base_config,
            base_path=base_path,
            dataset_paths=dataset_paths,
            hyperparams=job["hyperparams"],
            particle_count=particle_count,
            resample_threshold_fraction=float(
                config["resample_threshold_fraction"]
            ),
            seed_family=seed_family,
            learning_data_sha256=learning_hash,
            trial_scope=trial_scope,
            force=force,
        )
        for job in jobs
    )
    frame = pd.DataFrame(rows)
    for subject_id, subject_rows in frame.groupby("subject_id", sort=True):
        subject_dir = _rescore_subject_dir(output, int(subject_id))
        _atomic_csv(
            subject_dir / "seed_run_index.csv",
            subject_rows.sort_values("seed_index").reset_index(drop=True),
        )


def summarize_subject(
    *,
    subject_id: int,
    config: Mapping[str, Any],
    output: Path,
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    final = dict(config["optimization"]["final_rescore"])
    subject_dir = _rescore_subject_dir(output, subject_id)
    index_path = subject_dir / "seed_run_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    run_index = pd.read_csv(index_path).sort_values("seed_index")
    expected_count = int(final["filter_seed_count"])
    if len(run_index) != expected_count or run_index["filter_seed"].duplicated().any():
        raise ValueError(f"subject {subject_id} has an incomplete rescore seed bank")
    selected = _select_subject_trials(frame, config["trial_scope"])
    probability_runs: list[np.ndarray] = []
    for row in run_index.itertuples():
        cache_path = output / str(row.cache_relpath)
        expected = {
            "format_version": 1,
            "subject_id": int(subject_id),
            "trial_count": int(len(selected)),
            "seed_index": int(row.seed_index),
            "filter_seed": int(row.filter_seed),
            "seed_family": str(final["seed_family"]),
            "particle_count": int(final["particle_count"]),
            "hyperparams_sha256": str(row.hyperparams_sha256),
            "learning_data_sha256": str(row.learning_data_sha256),
            "observed_choices_used": True,
            "exploratory_only": True,
        }
        _, probabilities, _, _ = _read_cache(cache_path, expected)
        probability_runs.append(probabilities)
    fit, mean_probability = summarize_observed_choice_fit(
        np.stack(probability_runs),
        selected["choice"].to_numpy(dtype=int),
        selected["category"].to_numpy(dtype=int),
        window_size=int(config["window_size"]),
    )
    best_path = _search_best_path(output, subject_id)
    hyperparams = _selected_hyperparams(best_path)
    named_parameters = extract_model_0818_parameters(hyperparams)
    best_payload = json.loads(best_path.read_text(encoding="utf-8"))
    fit.update(
        {
            "subject_id": int(subject_id),
            "observed_accuracy": float(selected["feedback"].mean()),
            "ambiguous_trial_count": int(selected["ambiguous"].astype(bool).sum()),
            "particle_count": int(final["particle_count"]),
            "search_mean_trial_nll": float(best_payload["selection"]["value"]),
            "final_minus_search_mean_trial_nll": float(
                fit["mean_trial_nll"] - float(best_payload["selection"]["value"])
            ),
            "all_probabilities_finite": True,
            "formal_inference_authorized": False,
        }
    )
    _atomic_json(subject_dir / "fit_summary.json", fit)
    _atomic_json(subject_dir / "selected_parameters.json", named_parameters)
    _atomic_npz(
        subject_dir / "mean_probabilities.npz",
        mean_probabilities=mean_probability,
        choices=selected["choice"].to_numpy(dtype=np.int8),
        categories=selected["category"].to_numpy(dtype=np.int8),
        feedback=selected["feedback"].to_numpy(dtype=np.float32),
    )
    return fit, {"subject_id": int(subject_id), **named_parameters}


def summarize_subjects(
    *,
    subjects: Sequence[int],
    config: Mapping[str, Any],
    output: Path,
    frames: Mapping[int, pd.DataFrame],
) -> None:
    fit_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for subject_id in subjects:
        fit, parameters = summarize_subject(
            subject_id=int(subject_id),
            config=config,
            output=output,
            frame=frames[int(subject_id)],
        )
        fit_rows.append(fit)
        parameter_rows.append(parameters)
    fit_frame = pd.DataFrame(fit_rows).sort_values("subject_id")
    parameter_frame = pd.DataFrame(parameter_rows).sort_values("subject_id")
    complete_cohort = [int(value) for value in subjects] == [
        int(value) for value in config["subjects"]
    ]
    if complete_cohort:
        metrics_path = output / "subject_fit_metrics.csv"
        parameters_path = output / "selected_parameters.csv"
        summary_path = output / "group_fit_summary.json"
        report_path = output / "exploratory_fit_report.md"
    else:
        suffix = "_".join(str(value) for value in subjects)
        metrics_path = output / f"smoke_subject_fit_metrics_{suffix}.csv"
        parameters_path = output / f"smoke_selected_parameters_{suffix}.csv"
        summary_path = output / f"smoke_group_fit_summary_{suffix}.json"
        report_path = output / f"smoke_exploratory_fit_report_{suffix}.md"
    _atomic_csv(metrics_path, fit_frame)
    _atomic_csv(parameters_path, parameter_frame)
    weighted_mean_nll = float(
        fit_frame["total_nll"].sum() / fit_frame["trial_count"].sum()
    )
    group_summary = {
        "analysis_id": str(config["analysis_id"]),
        "subject_count": int(len(fit_frame)),
        "trial_count": int(fit_frame["trial_count"].sum()),
        "trial_scope": dict(config["trial_scope"]),
        "weighted_mean_trial_nll": weighted_mean_nll,
        "median_subject_mean_trial_nll": float(fit_frame["mean_trial_nll"].median()),
        "mean_choice_brier": float(fit_frame["choice_brier"].mean()),
        "mean_choice_accuracy": float(fit_frame["choice_accuracy"].mean()),
        "mean_observed_accuracy": float(fit_frame["observed_accuracy"].mean()),
        "subject_wins_vs_random": int(
            fit_frame["nll_improvement_vs_random"].gt(0.0).sum()
        ),
        "subject_wins_vs_causal_bias": int(
            fit_frame["nll_improvement_vs_causal_bias"].gt(0.0).sum()
        ),
        "median_learning_curve_correlation": float(
            fit_frame["learning_curve_correlation"].median()
        ),
        "median_learning_curve_mae": float(
            fit_frame["learning_curve_mae"].median()
        ),
        "median_choice_calibration_ece10": float(
            fit_frame["choice_calibration_ece10"].median()
        ),
        "maximum_trial_probability_mcse": float(
            fit_frame["maximum_trial_probability_mcse"].max()
        ),
        "zero_parameter_counts": {
            name: int(parameter_frame[name].eq(0.0).sum())
            for name in ("delta_E", "c_A", "c_G")
        },
        "observed_choices_used": True,
        "exploratory_only": True,
        "formal_parameter_inference_authorized": False,
        "formal_module_inference_authorized": False,
        "manuscript_result_authorized": False,
    }
    _atomic_json(summary_path, group_summary)
    lines = [
        "# Model 0818 exploratory observed fit",
        "",
        f"- Subjects: {group_summary['subject_count']}",
        f"- Fitted trials: {group_summary['trial_count']} ({config['trial_scope']['mode']})",
        f"- Weighted mean trial NLL: {weighted_mean_nll:.6f}",
        f"- Subjects better than random choice: {group_summary['subject_wins_vs_random']}/{len(fit_frame)}",
        f"- Subjects better than causal choice-bias baseline: {group_summary['subject_wins_vs_causal_bias']}/{len(fit_frame)}",
        f"- Mean choice Brier: {group_summary['mean_choice_brier']:.6f}",
        f"- Mean choice accuracy: {group_summary['mean_choice_accuracy']:.6f}",
        "",
        "These are descriptive pre-recovery results. They do not authorize parameter, module, or manuscript claims.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_standard_evaluation_config(
    *,
    subjects: Sequence[int],
    config: Mapping[str, Any],
    config_path: Path,
    hyper_config: Mapping[str, Any],
    output: Path,
) -> Path:
    """Materialize fixed subject parameters for the shared evaluation runner."""

    evaluation = dict(config["evaluation"])
    search_dir = output / "search"
    optimizer = HyperCDOptimizer(hyper_config, config_path)
    aggregate = aggregate_per_subject_best(
        search_dir,
        optimizer,
        config_path,
        "hyper_cd",
        subjects,
        require_all=True,
    )
    generated_path = output / "model_evaluation_simulation_config.yaml"
    simulation_dir = _repo_path(evaluation["simulation_dir"])
    generated = materialize_simulation_config_from_hyper_best(
        hyper_best_path=Path(aggregate["best_hyperparams"]),
        generated_sim_config_path=generated_path,
        sim_output_dir=simulation_dir,
        keep_logs=bool(evaluation["keep_logs"]),
        subjects=subjects,
    )

    generated["simulation_repeats"] = int(evaluation["filter_seed_count"])
    generated["n_jobs"] = int(evaluation["n_jobs_per_subject"])
    generated["hyper_base_seed"] = int(evaluation["hyper_base_seed"])
    generated["keep_logs"] = bool(evaluation["keep_logs"])
    generated["prediction_mode"] = str(evaluation["prediction_mode"])
    generated["selection_prediction_mode"] = str(evaluation["prediction_mode"])
    generated["repeat_aggregation"] = "mean_probability"
    max_trials_raw = config["trial_scope"].get("max_trials")
    generated["max_trials"] = (
        None if max_trials_raw is None else int(max_trials_raw)
    )
    generated["evaluation_protocol"] = {"mode": "all"}
    generated.setdefault("engine_config", {}).setdefault("inference", {})[
        "particle_count"
    ] = int(evaluation["particle_count"])

    # Evaluation must use a seed panel independent of Hyper-CD selection.
    overrides = generated.get("subject_overrides") or {}
    for subject_override in overrides.values():
        if isinstance(subject_override, dict):
            subject_override.pop("hyper_candidate_seed", None)
    save_yaml(generated_path, generated)

    _atomic_json(
        output / "model_evaluation_plan.json",
        {
            "reference_dir": str(_repo_path(evaluation["reference_dir"])),
            "simulation_config": str(generated_path),
            "simulation_dir": str(simulation_dir),
            "output_dir": str(_repo_path(evaluation["output_dir"])),
            "subjects": [int(value) for value in subjects],
            "particle_count": int(evaluation["particle_count"]),
            "filter_seed_count": int(evaluation["filter_seed_count"]),
            "hyper_base_seed": int(evaluation["hyper_base_seed"]),
            "seed_relation_to_search": "independent",
            "keep_logs": bool(evaluation["keep_logs"]),
            "prediction_mode": str(evaluation["prediction_mode"]),
            "oral_mode": str(evaluation["oral_mode"]),
            "evaluation_entrypoint": "src.Bayesian_state.run_model_evaluation",
            "trial_scope": dict(config["trial_scope"]),
            "interpretation": (
                "in_sample_descriptive_evaluation_of_fixed_fitted_parameters"
            ),
        },
    )
    return generated_path


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
    parameter_path = _repo_path(config["parameter_space"])
    parameter_space = load_parameter_space(parameter_path)
    base_path = _repo_path(config["base_simulation_config"])
    base_config = load_yaml(base_path)
    output = _repo_path(config["output_dir"])
    subjects = (
        [int(value) for value in args.subjects]
        if args.subjects
        else [int(value) for value in config["subjects"]]
    )
    if not subjects or not set(subjects).issubset(set(config["subjects"])):
        raise ValueError("requested subjects are outside the configured cohort")
    parallel_budget = int(
        args.parallel_budget
        if args.parallel_budget is not None
        else config["execution"]["search_parallel_budget_single_subject"]
    )
    n_jobs = int(
        args.n_jobs
        if args.n_jobs is not None
        else config["execution"]["final_rescore_n_jobs"]
    )
    logical_cpus = os.cpu_count() or 1
    if not 1 <= parallel_budget <= logical_cpus or not 1 <= n_jobs <= logical_cpus:
        raise ValueError("parallel budgets must lie within available logical CPUs")

    frames, dataset_paths = validate_inputs(
        config=config,
        config_path=config_path,
        parameter_space=parameter_space,
        base_config=base_config,
        base_path=base_path,
        output=output,
    )
    hyper_config = build_model_0818_hyper_config(
        config,
        parameter_space,
        root=ROOT,
        parallel_budget=parallel_budget,
    )
    _atomic_json(output / "generated_hyper_config_snapshot.json", hyper_config)
    if args.phase == "validate":
        print(json.dumps({"status": "inputs_validated", "subjects": subjects}, indent=2))
        return

    if args.phase in {"search-one", "search", "all"}:
        if args.phase == "search-one" and len(subjects) != 1:
            raise ValueError("search-one requires exactly one subject")
        search_rows = []
        for subject_id in subjects:
            row = run_search_subject(
                subject_id=subject_id,
                hyper_config=hyper_config,
                config_path=config_path,
                output=output,
                force=bool(args.force),
            )
            search_rows.append(row)
            print(
                f"[0818 observed search] subject={subject_id} cache_hit={row['cache_hit']}",
                flush=True,
            )
        _atomic_csv(
            output / f"search_invocation_{'_'.join(str(value) for value in subjects)}.csv",
            pd.DataFrame(search_rows),
        )
        if args.phase in {"search-one", "search"}:
            return

    if args.phase in {"rescore", "all"}:
        rescore_subjects(
            subjects=subjects,
            config=config,
            output=output,
            frames=frames,
            dataset_paths=dataset_paths,
            base_config=base_config,
            base_path=base_path,
            n_jobs=n_jobs,
            force=bool(args.force),
        )
        if args.phase == "rescore":
            summarize_subjects(
                subjects=subjects,
                config=config,
                output=output,
                frames=frames,
            )
            return

    if args.phase in {"summarize", "all"}:
        summarize_subjects(
            subjects=subjects,
            config=config,
            output=output,
            frames=frames,
        )

    if args.phase in {"evaluation-config", "all"}:
        generated_path = prepare_standard_evaluation_config(
            subjects=subjects,
            config=config,
            config_path=config_path,
            hyper_config=hyper_config,
            output=output,
        )
        print(f"[0818 model evaluation] generated_config={generated_path}")


if __name__ == "__main__":
    main()
