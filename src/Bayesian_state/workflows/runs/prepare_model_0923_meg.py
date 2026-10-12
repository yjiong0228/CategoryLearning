"""Validate MEG inputs and run a bounded one-subject Model 0923 fitting check.

The workflow uses the public Hyper-CD and simulation APIs. Generated YAMLs
snapshot the effective inputs; the declared batch fitting config is left intact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import yaml

from src.Bayesian_state.model.modules.perception import _get_uniform_threshold_stats
from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.run_simulation import run_simulation
from src.Bayesian_state.simulation.config import load_yaml, resolve_engine_config, save_json
from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.provenance import DependencySnapshot, file_digest


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _write_yaml(path: Path, config: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(dict(config), stream, sort_keys=False, allow_unicode=True)


def validate_meg_inputs(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Check every declared subject, including feedback and uniform half-widths."""
    engine = resolve_engine_config(config, config_path.parent)
    paths = resolve_dataset_paths(config, config_path.parent)
    loader = SubjectTrialDataLoader(engine, dataset_paths=paths)
    loader.prepare_data(paths["learning_data"])
    half_widths = _get_uniform_threshold_stats(paths["processed_dir"], paths)
    rows = []
    for subject in config["subjects"]:
        frame = loader._get_subject_frame(int(subject), 1.0)
        arrays = loader._extract_arrays(frame, None)
        if loader._get_condition_value(frame) != 3:
            raise ValueError(f"Subject {subject}: expected hierarchical condition 3")
        if not np.all(np.isfinite(arrays.stimulus)) or not np.all(
            (arrays.stimulus >= 0) & (arrays.stimulus <= 1)
        ):
            raise ValueError(f"Subject {subject}: invalid normalized stimuli")
        if not np.all(np.isin(arrays.feedback, [0.0, 0.5, 1.0])):
            raise ValueError(f"Subject {subject}: invalid three-level feedback")
        if not np.all(np.isin(arrays.choices, [1, 2, 3, 4])):
            raise ValueError(f"Subject {subject}: invalid choice coordinates")
        width = np.asarray(half_widths[int(subject)])
        if width.shape != (4,) or not np.all(np.isfinite(width) & (width >= 0)):
            raise ValueError(f"Subject {subject}: invalid perception half-widths")
        selection = frame.attrs["trial_selection"]
        rows.append({
            "subject": int(subject),
            "source_trials": selection["source_trial_count"],
            "valid_trials": len(frame),
            "omissions": selection["source_missed_trial_count"],
            "uniform_half_widths": width.tolist(),
        })
    return {"subjects": rows, "valid_trials": sum(row["valid_trials"] for row in rows),
            "omissions": sum(row["omissions"] for row in rows)}


def _smoke_space(search: Mapping[str, Any], engine: Mapping[str, Any]) -> tuple[dict, dict]:
    """Use the engine example and its next support value for each coordinate."""
    space, anchor = {}, {}
    for key, spec in search["hyperparam_space"].items():
        value: Any = engine
        for component in key.removeprefix("engine.").split("."):
            value = value[component]
        values = list(spec["values"])
        if value not in values or len(values) < 2:
            raise ValueError(f"Smoke anchor {key} must lie inside a varying support")
        index = values.index(value)
        alternate = values[index + 1] if index + 1 < len(values) else values[index - 1]
        space[key] = {"values": [value, alternate]}
        anchor[key] = value
    if len(space) != 13:
        raise ValueError("Model 0923 full-v3 fitting requires 13 behavior coordinates")
    return space, anchor


def audit_simulation(payload: Mapping[str, Any], expected_ids: list[list[int]]) -> dict[str, Any]:
    """Independently recompute scored NLL and check probabilities and alignment."""
    metrics = payload["representative_run"]["metrics_by_mode"]["prior_t"]
    probabilities = np.asarray(metrics["pred_category_probs"], dtype=float)
    n_trials = len(expected_ids)
    if probabilities.shape != (n_trials, 4):
        raise ValueError("Prediction shape does not match valid MEG trials")
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    expected_mask = np.ones(n_trials, dtype=bool)
    expected_mask[0] = False  # The shared pipeline uses the first valid trial to initialize.
    np.testing.assert_array_equal(valid, expected_mask)
    selected = probabilities[valid]
    if not np.all(np.isfinite(selected)) or np.any(selected < 0):
        raise ValueError("Invalid predictive category probabilities")
    np.testing.assert_allclose(selected.sum(axis=1), 1.0, atol=1e-10)
    choices = np.asarray(metrics["observed_choice"], dtype=int)
    observed = selected[np.arange(valid.sum()), choices[valid] - 1]
    total_nll = float(-np.log(np.clip(observed, 1e-12, 1.0)).sum())
    np.testing.assert_allclose(payload["simulation"]["mean_error"], total_nll / valid.sum(),
                               rtol=1e-10, atol=1e-12)
    selection = payload["selection"]["selection_meta"]["trial_selection"]
    if selection["retained_trial_identifiers"] != expected_ids:
        raise ValueError("Original MEG trial identifiers changed during execution")
    state = payload["representative_run"]["state_log"]
    checked = []
    for key in ("marginal_prior", "choice_weights", "dominant_probability",
                "filtered_choice_weights", "filtered_dominant_probability",
                "pairing_prior", "pairing_posterior"):
        values = np.asarray(state[key], dtype=float)
        if values.shape[0] != n_trials or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Invalid probability state: {key}")
        np.testing.assert_allclose(values.sum(axis=-1), 1.0, atol=1e-10)
        checked.append(key)
    return {"modeled_trials": n_trials, "scored_trials": int(valid.sum()),
            "total_choice_nll": total_nll, "mean_choice_nll": total_nll / valid.sum(),
            "minimum_observed_choice_probability": float(observed.min()),
            "normalized_states": checked, "source_alignment": "passed"}


def run_preparation(config_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run short coarse/fine fitting, independent rescoring, and full-history scoring."""
    settings = load_yaml(config_path)
    search_path = _resolve(config_path.parent, settings["hyper_config_path"])
    search = load_yaml(search_path)
    base_path = _resolve(search_path.parent, search["base_sim_config_path"])
    base = load_yaml(base_path)
    engine = resolve_engine_config(base, base_path.parent)
    subject = int(settings["subject"])
    maximum = int(settings["max_trials"])
    if subject not in base["subjects"] or maximum < int(base["window_size"]) + 1:
        raise ValueError("Smoke subject or trial budget is invalid")
    output = output_dir or _resolve(config_path.parent, settings["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_at_start = {str(path): file_digest(path)
                       for path in (ROOT_DIR / "src/Bayesian_state").rglob("*.py")}
    preflight = validate_meg_inputs(base, base_path)
    save_json(preflight, output / "data_validation.json")
    print(f"Inputs passed: {len(preflight['subjects'])} subjects; {preflight['valid_trials']} valid trials", flush=True)

    # Absolute references occur only in generated snapshots inside this run directory.
    simulation = deepcopy(base)
    simulation.pop("engine_config_path")
    simulation["engine_config"] = engine
    engine["partition"]["kwargs"]["similarity_cache_dir"] = str(output / "cache")
    simulation["dataset"] = {key: str(value) for key, value in resolve_dataset_paths(base, base_path.parent).items()}
    simulation.update(subjects=[subject], n_jobs=1, simulation_repeats=1, keep_logs=False,
                      max_trials=maximum, output_dir=str(output / "unused_base"))
    simulation["engine_config"]["inference"]["particle_count"] = int(settings["particle_count"])
    sim_path = output / "smoke_simulation.yaml"
    _write_yaml(sim_path, simulation)
    space, anchor = _smoke_space(search, engine)
    smoke = deepcopy(search)
    smoke.update(base_sim_config_path=str(sim_path), subjects=[subject],
                 output_dir=str(output / "hyper_cd"), hyperparam_space=space,
                 hyper_base_seed=int(settings["hyper_base_seed"]))
    smoke["cd"].update(n_restarts=1, max_outer_iters=1, init_strategy="anchor", anchor=anchor,
                       coordinate_order="fixed", parallel_budget=1)
    smoke["refine_policy"]["top_k"] = 1
    for stage in smoke["stages"].values():
        stage.update(hyperparam_space=deepcopy(space), cd_parallel={"max_repeat_jobs": 1},
                     simulation_overrides={"simulation_repeats": 1, "n_jobs": 1,
                                           "engine_config": {"inference": {"particle_count": int(settings["particle_count"])}}})
    smoke["final_rescore"].update(shortlist_size=int(settings["final_shortlist_size"]),
                                    seed_family="model0923_meg_preparation_final_v1")
    smoke["final_rescore"]["simulation_overrides"] = {
        "simulation_repeats": int(settings["final_repeats"]), "n_jobs": 1,
        "repeat_aggregation": "mean_probability",
        "engine_config": {"inference": {"particle_count": int(settings["final_particle_count"])}}}
    smoke_path = output / "smoke_search.yaml"
    _write_yaml(smoke_path, smoke)
    snapshot = DependencySnapshot()
    snapshot.add_file(config_path)
    snapshot.add_file(search_path, follow_yaml=True)
    snapshot.simulation(simulation, output, [subject])
    provenance = snapshot.payload()
    provenance["source_digests_at_start"] = source_at_start
    save_json(provenance, output / "input_provenance.json")
    if any(file_digest(Path(path)) != digest for path, digest in source_at_start.items()):
        raise ValueError("Model source changed while preparing provenance; rerun in a new directory")

    print("Starting bounded 13-coordinate search", flush=True)
    fit_started = time.perf_counter()
    result = HyperCDOptimizer(smoke, smoke_path).run(subjects=[subject], stage="all")
    fit_seconds = time.perf_counter() - fit_started
    best_path = Path(result["per_subject_outputs"][str(subject)]["best_hyperparams"])
    winner = json.loads(best_path.read_text())["final_rescore_best"]
    parameters = winner["hyperparams"]
    if set(parameters) != set(space) or not np.isfinite(winner["aggregated_error"]):
        raise ValueError("Final rescoring did not produce a complete finite parameter estimate")

    loader = SubjectTrialDataLoader(engine, dataset_paths=resolve_dataset_paths(base, base_path.parent))
    loader.prepare_data(simulation["dataset"]["learning_data"])
    frame = loader._get_subject_frame(subject, 1.0)
    identifiers = frame.attrs["trial_selection"]["retained_trial_identifiers"]
    checks, timings, short_predictions = {}, {}, []
    for name, limit in (("short", maximum), ("short_repeat", maximum), ("full_history", None)):
        print(f"Checking {name} predictions", flush=True)
        cfg = deepcopy(simulation)
        cfg.update(output_dir=str(output / name), keep_logs=True, max_trials=limit,
                   fixed_hyperparams=parameters, hyper_base_seed=int(settings["hyper_base_seed"]))
        path = output / f"{name}.yaml"
        _write_yaml(path, cfg)
        check_started = time.perf_counter()
        saved = run_simulation(path)
        timings[name] = time.perf_counter() - check_started
        payload = json.loads(saved[0].read_text())
        expected = identifiers if limit is None else identifiers[:limit]
        checks[name] = audit_simulation(payload, expected)
        if limit is not None:
            short_predictions.append(np.asarray(payload["representative_run"]["metrics_by_mode"]["prior_t"]["pred_category_probs"]))
    np.testing.assert_array_equal(short_predictions[0], short_predictions[1])
    initial_digests = {**provenance["files"], **source_at_start}
    changed_inputs = [path for path, digest in initial_digests.items()
                      if file_digest(Path(path)) != digest]
    if changed_inputs:
        raise ValueError(f"Inputs changed during validation: {changed_inputs}")
    report = {"status": "passed", "scope": "input_and_small_fit_validation",
              "subject": subject, "short_valid_trials": maximum,
              "behavior_parameter_count": len(parameters), "checks": checks,
              "fixed_seed_reproducibility": "passed", "fit_seconds": fit_seconds,
              "simulation_seconds": timings, "total_seconds": time.perf_counter() - started,
              "short_fit_mean_choice_nll": winner["aggregated_error"],
              "selected_smoke_parameters": parameters,
              "batch_fit_executed": False, "pf_budget_calibrated": False,
              "fit_support_calibrated": False}
    save_json(report, output / "validation_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/meg/model0923_task3b_smoke.yaml"))
    parser.add_argument("--output-dir", type=Path, help="A new output directory")
    parser.add_argument("--freeze-source", action="store_true",
                        help="Run from a read-only snapshot of the model, configs and MEG inputs")
    args = parser.parse_args()
    config = _resolve(ROOT_DIR, args.config)
    output = None if args.output_dir is None else _resolve(ROOT_DIR, args.output_dir)
    if args.freeze_source:
        run_frozen_preparation(config, output)
    else:
        run_preparation(config, output)


def run_frozen_preparation(config_path: Path, output_dir: Path | None = None) -> None:
    """Isolate a run from concurrent repository edits and retain its exact sources."""
    if not config_path.is_relative_to(ROOT_DIR / "configs"):
        raise ValueError("Frozen preparation requires a repository configuration")
    output = output_dir or _resolve(config_path.parent, load_yaml(config_path)["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    tracked = list((ROOT_DIR / "src/Bayesian_state").rglob("*.py"))
    tracked += list((ROOT_DIR / "configs").rglob("*.yaml"))
    tracked += [ROOT_DIR / "data/meg/processed" / name
                for name in ("Task3b_processed.csv", "Task1b_errorsummary.csv")]
    before = {str(path.relative_to(ROOT_DIR)): file_digest(path) for path in tracked}
    source = output / "source"
    source.mkdir()
    (source / "src").mkdir()
    shutil.copy2(ROOT_DIR / "src/__init__.py", source / "src/__init__.py")
    shutil.copytree(ROOT_DIR / "src/Bayesian_state", source / "src/Bayesian_state",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    shutil.copytree(ROOT_DIR / "configs", source / "configs")
    (source / "data/meg/processed").mkdir(parents=True)
    for name in ("Task3b_processed.csv", "Task1b_errorsummary.csv"):
        shutil.copy2(ROOT_DIR / "data/meg/processed" / name, source / "data/meg/processed" / name)
    if any(file_digest(ROOT_DIR / path) != digest or file_digest(source / path) != digest
           for path, digest in before.items()):
        raise ValueError("Inputs changed during snapshot creation; rerun in a new directory")
    save_json({"scope": "read_only_source_and_meg_input_snapshot", "files": before},
              output / "snapshot_manifest.json")
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(output / "numba_cache"))
    subprocess.run([
        sys.executable, "-m", "src.Bayesian_state.workflows.runs.prepare_model_0923_meg",
        "--config", str(config_path.relative_to(ROOT_DIR)),
        "--output-dir", str(output / "run"),
    ], cwd=source, env=environment, check=True)
    changed = [path for path, digest in before.items() if file_digest(ROOT_DIR / path) != digest]
    save_json({"current_repository_matches_snapshot": not changed,
               "changed_paths": changed, "validated_run": "run/validation_report.json"},
              output / "live_source_comparison.json")


if __name__ == "__main__":
    main()
