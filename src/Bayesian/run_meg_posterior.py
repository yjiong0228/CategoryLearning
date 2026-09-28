"""Fit legacy MEG M6_MH and export its trial-level posterior table.

Run from the repository root::

    python -m src.Bayesian.run_meg_posterior --subject 334

The scientific settings mirror the active cells in
``notebooks/old/Bayesian_meg.ipynb``. Each invocation performs a fresh fit and
refuses to overwrite an existing subject/date result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from math import prod
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.Bayesian.problems.config import config_fgt
from src.Bayesian.problems.fit_config import module_configs_M6, window_size_configs
from src.Bayesian.problems.model import StandardModel
from src.Bayesian.utils.model_evaluation import ModelEval
from src.Bayesian.utils.optimizer import Optimizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "M6_MH"
TASK_NAME = "Task3b"
WINDOW_SIZE = 16
GRID_REPEAT = 64
MC_SAMPLES = 1024
DEFAULT_N_JOBS = 120
POSTERIOR_CROSSCHECK_ATOL = 1e-12
DEFAULT_PROCESSED_CSV = Path("data/meg/processed/Task3b_processed.csv")
DEFAULT_RESULTS_ROOT = Path("results/model_static/model_results_meg")

POSTERIOR_COLUMNS = [
    "trial",
    "true_choice",
    "predicted_choice",
    "hit",
    "nonhit",
    "rating",
    "origin_true_choice_probability",
    "origin_posterior",
    "origin_pred_probs",
    "origin_entropy",
    "origin_top1_hypothesis",
    "origin_top1_probability",
    "origin_top3_mass",
]

REQUIRED_DATA_COLUMNS = {
    "iSub",
    "condition",
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "category",
    "choice",
    "rating",
    "feedback",
}


@dataclass(frozen=True)
class MegPosteriorRunPaths:
    """All stable output paths for one subject/date run."""

    output_dir: Path
    fit_results: Path
    prediction_results: Path
    posterior_csv: Path
    posterior_plot: Path
    accuracy_plot: Path
    grid_plot: Path
    amount_plot: Path
    manifest: Path


def build_run_paths(
    project_root: Path,
    subject_id: int,
    run_date: date | None = None,
    run_suffix: str | None = None,
) -> MegPosteriorRunPaths:
    """Build the non-overwriting output contract for a subject run."""

    subject_id = int(subject_id)
    date_token = (run_date or date.today()).strftime("%y%m%d")
    suffix_token = ""
    if run_suffix is not None:
        normalized_suffix = str(run_suffix).strip()
        if not normalized_suffix or any(
            not (character.isalnum() or character in "-_")
            for character in normalized_suffix
        ):
            raise ValueError(
                "run suffix must contain only letters, digits, '-' or '_'"
            )
        suffix_token = f"_{normalized_suffix}"
    output_dir = (
        Path(project_root)
        / DEFAULT_RESULTS_ROOT
        / f"Model_results_sub{subject_id}_{date_token}{suffix_token}"
    )
    return MegPosteriorRunPaths(
        output_dir=output_dir,
        fit_results=output_dir / f"{MODEL_NAME}.joblib",
        prediction_results=output_dir / f"{MODEL_NAME}_prediction.joblib",
        posterior_csv=(
            output_dir
            / f"{TASK_NAME}_Sub{subject_id}_{MODEL_NAME}_model_posterior.csv"
        ),
        posterior_plot=output_dir / f"{MODEL_NAME}_post.png",
        accuracy_plot=output_dir / f"{MODEL_NAME}_acc.png",
        grid_plot=output_dir / f"{MODEL_NAME}_grid.png",
        amount_plot=output_dir / f"{MODEL_NAME}_amount.png",
        manifest=output_dir / "run_manifest.json",
    )


def prepare_output_directory(paths: MegPosteriorRunPaths) -> None:
    """Create a new result directory, never replacing an earlier run."""

    paths.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        paths.output_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing result directory: {paths.output_dir}"
        ) from exc


def effective_fit_jobs(
    requested_jobs: int,
    *,
    available_cpus: int,
    ready_tasks: int,
) -> int:
    """Bound fit workers without changing the frozen scientific budget."""

    return max(1, min(int(requested_jobs), int(available_cpus), int(ready_tasks)))


def _ready_fit_tasks(optimizer: Optimizer, subject_id: int) -> int:
    parameter_grid = optimizer.optimize_params_dict[int(subject_id)]
    return max(1, prod(len(values) for values in parameter_grid.values()))


def _lookup_by_int(mapping: Mapping[Any, Any], key: int, name: str) -> Any:
    if key in mapping:
        return mapping[key]
    for candidate_key, value in mapping.items():
        try:
            matches = int(candidate_key) == int(key)
        except (TypeError, ValueError):
            matches = False
        if matches:
            return value
    raise KeyError(f"{name} does not contain subject {key}")


def _normalize_posterior(posterior: Mapping[Any, Any] | None) -> dict[int, float] | None:
    if not isinstance(posterior, Mapping) or not posterior:
        return None
    keys = [int(key) for key in posterior]
    values = np.asarray([float(value) for value in posterior.values()], dtype=float)
    values = np.clip(values, 0.0, None)
    value_sum = float(values.sum())
    if not np.isfinite(value_sum) or value_sum <= 0.0:
        values = np.full(len(keys), 1.0 / len(keys), dtype=float)
    else:
        values = values / value_sum
    return dict(zip(keys, values.tolist()))


def _posterior_to_json(posterior: Mapping[Any, Any] | None) -> str:
    normalized = _normalize_posterior(posterior)
    if normalized is None:
        return ""
    return json.dumps(
        {str(key): value for key, value in normalized.items()},
        ensure_ascii=True,
        sort_keys=True,
    )


def _probabilities_to_json(probabilities: Sequence[float]) -> str:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    return json.dumps(values.tolist(), ensure_ascii=True)


def _posterior_entropy(
    posterior: Mapping[Any, Any] | None, eps: float = 1e-12
) -> float:
    normalized = _normalize_posterior(posterior)
    if normalized is None:
        return float("nan")
    probabilities = np.asarray(list(normalized.values()), dtype=float)
    probabilities = np.clip(probabilities, eps, 1.0)
    probabilities = probabilities / probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _top1_hypothesis(posterior: Mapping[Any, Any] | None) -> int | float:
    normalized = _normalize_posterior(posterior)
    if normalized is None:
        return float("nan")
    return int(max(normalized, key=normalized.get))


def _top1_probability(posterior: Mapping[Any, Any] | None) -> float:
    normalized = _normalize_posterior(posterior)
    if normalized is None:
        return float("nan")
    return float(max(normalized.values()))


def _topk_mass(posterior: Mapping[Any, Any] | None, k: int = 3) -> float:
    normalized = _normalize_posterior(posterior)
    if normalized is None:
        return float("nan")
    probabilities = np.sort(np.asarray(list(normalized.values()), dtype=float))[::-1]
    return float(probabilities[:k].sum())


def _validate_subject_data(data: pd.DataFrame, subject_id: int) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_DATA_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Processed MEG data is missing columns: {missing_columns}")
    subject_data = data.loc[data["iSub"] == int(subject_id)].copy().reset_index(drop=True)
    if subject_data.empty:
        raise ValueError(f"No behavioral rows found for subject {subject_id}")
    if len(subject_data) < 2:
        raise ValueError(f"Subject {subject_id} needs at least two trials")
    if subject_data["condition"].nunique(dropna=False) != 1:
        raise ValueError(f"Subject {subject_id} has inconsistent condition values")
    if not pd.api.types.is_integer_dtype(subject_data["choice"].dtype):
        raise ValueError(
            "Processed MEG choice column must use an integer dtype; "
            "remove missing responses before fitting and serialize choices as integers"
        )
    if int(subject_id) not in module_configs_M6:
        raise ValueError(f"Subject {subject_id} has no M6_MH module configuration")
    return subject_data


def build_posterior_table(
    subject_data: pd.DataFrame,
    prediction_result: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the top-down trial table, intentionally omitting trial 1."""

    n_trials = len(subject_data)
    predicted_choice = np.asarray(prediction_result["pred_choice"])
    predicted_probabilities = np.asarray(prediction_result["pred_probs"], dtype=float)
    original_posterior = prediction_result["original_posterior"]

    if predicted_choice.shape != (n_trials,):
        raise ValueError(
            "Prediction length mismatch: "
            f"behavior={n_trials}, pred_choice={predicted_choice.shape}"
        )
    if predicted_probabilities.ndim != 2 or predicted_probabilities.shape[0] != n_trials:
        raise ValueError(
            "Prediction probability shape mismatch: "
            f"behavior={n_trials}, pred_probs={predicted_probabilities.shape}"
        )
    if len(original_posterior) != n_trials:
        raise ValueError(
            "Posterior length mismatch: "
            f"behavior={n_trials}, posterior={len(original_posterior)}"
        )
    trial_index = np.arange(1, n_trials)
    trial_probabilities = predicted_probabilities[trial_index]
    if not np.all(np.isfinite(trial_probabilities)):
        raise ValueError("Prediction probabilities contain non-finite values")
    if np.any(trial_probabilities < 0.0):
        raise ValueError("Prediction probabilities contain negative values")
    if not np.allclose(trial_probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Prediction probabilities are not row-normalized")

    true_choice = subject_data["category"].to_numpy(dtype=int)[trial_index]
    predicted_choice = predicted_choice[trial_index].astype(int)
    n_categories = predicted_probabilities.shape[1]
    if np.any((true_choice < 1) | (true_choice > n_categories)):
        raise ValueError("True categories fall outside prediction probability columns")
    if np.any((predicted_choice < 1) | (predicted_choice > n_categories)):
        raise ValueError("Predicted categories fall outside prediction probability columns")

    true_choice_probability = trial_probabilities[
        np.arange(len(trial_index)), true_choice - 1
    ]
    trial_posterior = [original_posterior[int(index)] for index in trial_index]
    hit = (true_choice == predicted_choice).astype(int)
    if "iTrial" in subject_data:
        output_trials = pd.to_numeric(
            subject_data.loc[trial_index, "iTrial"], errors="raise"
        ).to_numpy(dtype=float)
        if (
            not np.all(np.isfinite(output_trials))
            or not np.all(output_trials == np.floor(output_trials))
            or np.any(output_trials < 1)
            or len(np.unique(output_trials)) != len(output_trials)
        ):
            raise ValueError("iTrial must contain unique positive integer identifiers")
        output_trials = output_trials.astype(int)
    else:
        output_trials = (trial_index + 1).astype(int)

    return pd.DataFrame(
        {
            "trial": output_trials,
            "true_choice": true_choice,
            "predicted_choice": predicted_choice,
            "hit": hit,
            "nonhit": 1 - hit,
            "rating": subject_data.loc[trial_index, "rating"].to_numpy(),
            "origin_true_choice_probability": true_choice_probability,
            "origin_posterior": [
                _posterior_to_json(posterior) for posterior in trial_posterior
            ],
            "origin_pred_probs": [
                _probabilities_to_json(probabilities)
                for probabilities in trial_probabilities
            ],
            "origin_entropy": [
                _posterior_entropy(posterior) for posterior in trial_posterior
            ],
            "origin_top1_hypothesis": [
                _top1_hypothesis(posterior) for posterior in trial_posterior
            ],
            "origin_top1_probability": [
                _top1_probability(posterior) for posterior in trial_posterior
            ],
            "origin_top3_mass": [
                _topk_mass(posterior, k=3) for posterior in trial_posterior
            ],
        },
        columns=POSTERIOR_COLUMNS,
    )


def export_m6_mh_model_posterior_csv(
    subject_id: int,
    result_dir: Path,
    processed_csv: Path,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    """Recompute M6_MH probabilities and write the top-down posterior CSV."""

    result_dir = Path(result_dir)
    processed_csv = Path(processed_csv)
    fit_results = joblib.load(result_dir / f"{MODEL_NAME}.joblib")
    subject_fit = _lookup_by_int(fit_results, subject_id, f"{MODEL_NAME}.joblib")
    step_results = subject_fit["best_step_results"]

    subject_data = _validate_subject_data(pd.read_csv(processed_csv), subject_id)
    condition = int(subject_data["condition"].iloc[0])
    if len(step_results) != len(subject_data):
        raise ValueError(
            f"Fit/data trial mismatch for subject {subject_id}: "
            f"fit={len(step_results)}, behavior={len(subject_data)}"
        )
    fit_condition = int(subject_fit["condition"])
    if fit_condition != condition:
        raise ValueError(
            f"Fit/data condition mismatch for subject {subject_id}: "
            f"fit={fit_condition}, behavior={condition}"
        )
    stimulus = subject_data[
        ["feature1", "feature2", "feature3", "feature4"]
    ].to_numpy()
    categories = subject_data["category"].to_numpy(dtype=int)
    feedback = subject_data["feedback"].to_numpy()

    model = StandardModel(
        config_fgt,
        module_config=_lookup_by_int(
            module_configs_M6, subject_id, "module_configs_M6"
        ),
        condition=condition,
    )
    prediction_result = model.predict_probs(
        (stimulus, categories, feedback, categories),
        step_results,
        use_cached_dist=False,
        window_size=WINDOW_SIZE,
    )
    posterior_table = build_posterior_table(subject_data, prediction_result)
    output_path = (
        result_dir
        / f"{TASK_NAME}_Sub{int(subject_id)}_{MODEL_NAME}_model_posterior.csv"
    )

    diagnostics: dict[str, Any] = {
        "subject_id": int(subject_id),
        "n_behavior_trials": len(subject_data),
        "n_rows": len(posterior_table),
        "n_columns": posterior_table.shape[1],
        "mean_hit": float(posterior_table["hit"].mean()),
    }

    prediction_path = result_dir / f"{MODEL_NAME}_prediction.joblib"
    if prediction_path.exists():
        saved_prediction = joblib.load(prediction_path)
        subject_prediction = _lookup_by_int(
            saved_prediction, subject_id, f"{MODEL_NAME}_prediction.joblib"
        )
        saved_probability = np.asarray(subject_prediction["pred_acc"], dtype=float)[1:]
        current_probability = posterior_table[
            "origin_true_choice_probability"
        ].to_numpy(dtype=float)
        if saved_probability.shape != current_probability.shape:
            raise ValueError(
                "Posterior/prediction crosscheck failed: "
                f"saved={saved_probability.shape}, regenerated={current_probability.shape}"
            )
        if not np.all(np.isfinite(saved_probability)):
            raise ValueError(
                "Posterior/prediction crosscheck failed: saved probabilities "
                "contain non-finite values after trial 1"
            )
        max_difference = float(
            np.max(np.abs(saved_probability - current_probability), initial=0.0)
        )
        diagnostics["pred_acc_crosscheck_max_abs_diff"] = max_difference
        diagnostics["pred_acc_crosscheck_atol"] = POSTERIOR_CROSSCHECK_ATOL
        if max_difference > POSTERIOR_CROSSCHECK_ATOL:
            raise ValueError(
                "Posterior/prediction crosscheck failed: "
                f"max_abs_diff={max_difference:.17g} exceeds "
                f"atol={POSTERIOR_CROSSCHECK_ATOL:.1e}"
            )

    posterior_table.to_csv(output_path, index=False)

    return output_path, posterior_table, diagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_hashes(project_root: Path) -> dict[str, str]:
    """Hash the legacy model code and shared hypothesis implementation."""

    project_root = Path(project_root)
    source_files: set[Path] = set()
    for source_root in (
        project_root / "src/Bayesian",
        project_root / "src/Bayesian_state/hypothesis_space",
    ):
        if source_root.is_dir():
            source_files.update(source_root.rglob("*.py"))
    for source_file in (
        project_root / "src/Bayesian_state/__init__.py",
        project_root / "requirements.txt",
    ):
        if source_file.is_file():
            source_files.add(source_file)
    return {
        str(path.relative_to(project_root)): _sha256(path)
        for path in sorted(source_files)
    }


def _git_provenance(project_root: Path) -> dict[str, Any]:
    command_prefix = [
        "git",
        "-c",
        f"safe.directory={Path(project_root).resolve()}",
    ]
    try:
        commit = subprocess.run(
            [*command_prefix, "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [*command_prefix, "status", "--short"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None, "dirty_entries": None}
    return {
        "commit": commit,
        "worktree_dirty": bool(status),
        "dirty_entries": status,
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _initial_manifest(
    subject_id: int,
    project_root: Path,
    processed_csv: Path,
    n_jobs: int,
    run_suffix: str | None,
) -> dict[str, Any]:
    try:
        input_path = str(processed_csv.relative_to(project_root))
    except ValueError:
        input_path = str(processed_csv)
    return {
        "schema_version": 1,
        "status": "running",
        "subject_id": int(subject_id),
        "run_suffix": run_suffix,
        "task": TASK_NAME,
        "model": MODEL_NAME,
        "started_at": datetime.now().astimezone().isoformat(),
        "completed_at": None,
        "fit": {
            "window_size": WINDOW_SIZE,
            "grid_repeat": GRID_REPEAT,
            "mc_samples": MC_SAMPLES,
            "requested_n_jobs": int(n_jobs),
            "effective_fit_jobs": None,
            "prediction_jobs": 1,
            "inner_max_num_threads": 1,
            "random_seed": None,
        },
        "input": {"processed_csv": input_path, "sha256": _sha256(processed_csv)},
        "code": {
            **_git_provenance(project_root),
            "source_files_sha256": _source_file_hashes(project_root),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }


def _verify_artifacts(paths: MegPosteriorRunPaths, subject_id: int) -> None:
    required_files = [
        paths.fit_results,
        paths.prediction_results,
        paths.posterior_csv,
        paths.posterior_plot,
        paths.accuracy_plot,
        paths.grid_plot,
        paths.amount_plot,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    cache_file = paths.output_dir / "cache" / MODEL_NAME / f"{subject_id}.gz"
    if not cache_file.is_file():
        missing.append(str(cache_file))
    if missing:
        raise RuntimeError(f"MEG posterior run is missing artifacts: {missing}")


def run_meg_posterior(
    subject_id: int,
    *,
    project_root: Path = PROJECT_ROOT,
    processed_csv: Path | None = None,
    run_date: date | None = None,
    n_jobs: int = DEFAULT_N_JOBS,
    run_suffix: str | None = None,
) -> MegPosteriorRunPaths:
    """Run a complete fresh M6_MH fit and posterior export for one subject."""

    subject_id = int(subject_id)
    project_root = Path(project_root).resolve()
    processed_csv = Path(processed_csv or DEFAULT_PROCESSED_CSV)
    if not processed_csv.is_absolute():
        processed_csv = project_root / processed_csv
    processed_csv = processed_csv.resolve()
    if not processed_csv.is_file():
        raise FileNotFoundError(f"Processed MEG data not found: {processed_csv}")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")

    _validate_subject_data(pd.read_csv(processed_csv), subject_id)
    paths = build_run_paths(
        project_root, subject_id, run_date, run_suffix=run_suffix
    )
    prepare_output_directory(paths)
    manifest = _initial_manifest(
        subject_id, project_root, processed_csv, n_jobs, run_suffix
    )
    _write_manifest(paths.manifest, manifest)

    try:
        optimizer = Optimizer(module_configs_M6, n_jobs=n_jobs)
        ready_tasks = _ready_fit_tasks(optimizer, subject_id)
        available_cpus = os.cpu_count() or 1
        fit_jobs = effective_fit_jobs(
            n_jobs,
            available_cpus=available_cpus,
            ready_tasks=ready_tasks,
        )
        optimizer.n_jobs = fit_jobs
        manifest["fit"]["available_cpus"] = available_cpus
        manifest["fit"]["ready_fit_tasks"] = ready_tasks
        manifest["fit"]["effective_fit_jobs"] = fit_jobs
        _write_manifest(paths.manifest, manifest)

        optimizer.prepare_data(processed_csv)
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = optimizer.optimize_params_with_subs_parallel(
                config_fgt,
                [subject_id],
                window_size_configs,
                GRID_REPEAT,
                MC_SAMPLES,
            )
        optimizer.save_results(results, MODEL_NAME, paths.output_dir)

        optimizer.set_results(results)
        optimizer.n_jobs = 1
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            prediction = optimizer.predict_with_subs_parallel(config_fgt, [subject_id])
        joblib.dump(prediction, paths.prediction_results)

        _, _, posterior_diagnostics = export_m6_mh_model_posterior_csv(
            subject_id,
            paths.output_dir,
            processed_csv,
        )

        evaluator = ModelEval()
        evaluator.plot_posterior_probabilities(results, save_path=paths.posterior_plot)
        evaluator.plot_accuracy_comparison(prediction, save_path=paths.accuracy_plot)
        evaluator.plot_error_grids(
            results, fname=["gamma", "w0"], save_path=paths.grid_plot
        )
        evaluator.plot_cluster_amount(
            results, window_size=WINDOW_SIZE, save_path=paths.amount_plot
        )

        _verify_artifacts(paths, subject_id)
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["posterior"] = posterior_diagnostics
        manifest["artifacts"] = {
            "output_dir": str(paths.output_dir.relative_to(project_root)),
            "posterior_csv": str(paths.posterior_csv.relative_to(project_root)),
        }
        _write_manifest(paths.manifest, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_manifest(paths.manifest, manifest)
        raise

    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freshly fit legacy MEG M6_MH for one subject and export the "
            "trial-level posterior CSV used by top-down analyses."
        )
    )
    parser.add_argument("--subject", type=int, required=True, help="MEG subject ID")
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=DEFAULT_PROCESSED_CSV,
        help="Processed MEG Task3b table, relative to the repository root by default",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=f"Parallel worker budget (default: {DEFAULT_N_JOBS})",
    )
    parser.add_argument(
        "--run-suffix",
        help=(
            "Optional safe suffix for a preserved retry directory, for example "
            "'retry1'"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_meg_posterior(
        args.subject,
        processed_csv=args.processed_csv,
        n_jobs=args.n_jobs,
        run_suffix=args.run_suffix,
    )
    print(paths.output_dir)
    print(paths.posterior_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
