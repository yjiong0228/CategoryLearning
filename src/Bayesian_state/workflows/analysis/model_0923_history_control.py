"""Run fixed-parameter condition B along observed choices and feedback.

B uses the public particle filter with condition_on_observed_choice=False and
resample_threshold_fraction=0. Cognitive feedback learning remains enabled.
A and C are read from a saved evaluation; no parameters are re-estimated.
Run from the repository root. See docs/model_architecture/model_0923_history_control.md.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

# Re-enter before numerical imports so archived fits run their archived core.
# run_path makes analysis functions serializable by value for joblib workers.
if __name__ == "__main__" and "--runtime-source" in sys.argv:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--runtime-source", type=Path, required=True)
    runtime, remaining = bootstrap.parse_known_args()
    frozen = runtime.runtime_source.resolve()
    if not (frozen / "src/Bayesian_state/inference/backends/particle_filter.py").is_file():
        raise ValueError(f"Missing archived model source: {frozen}")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    launcher = ("import runpy,sys; source,script=sys.argv[1:3]; "
                "sys.path.insert(0,source); sys.argv=[script]+sys.argv[3:]; "
                "runpy.run_path(script,run_name='__main__')")
    os.execv(sys.executable, [sys.executable, "-c", launcher, str(frozen),
                             str(Path(__file__).resolve()), *remaining])

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.evaluation.model_0923_fit_source import resolve_fit_subject, validate_fit_source
from src.Bayesian_state.evaluation.model_0923_postfit import score_probabilities
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.utils.parallel import parallel_job_count, single_threaded_processes
from src.Bayesian_state.utils.paths import ROOT_DIR


RULE_FIELDS = ("belief", "active", "choice_weights", "dominant")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class UniformHistoryCollector:
    """Verify B's uniform paths and retain pre-choice state measurements."""

    def __init__(self, categories: np.ndarray) -> None:
        self.categories = np.asarray(categories, int)
        self.rows: list[dict[str, np.ndarray]] = []
        self.pending: dict[str, np.ndarray] | None = None
        self.filtered_count = 0

    def __call__(self, trial: int, phase: str, snapshot: dict) -> None:
        weights = np.asarray(snapshot["weights"])
        np.testing.assert_allclose(weights, 1 / len(weights), atol=1e-14, rtol=0)
        if phase == "filtered":
            if self.pending is None or trial != self.filtered_count:
                raise ValueError("Filtered observer order is inconsistent")
            for key in (*RULE_FIELDS, "prediction"):
                np.testing.assert_array_equal(snapshot[key], self.pending[key])
            self.filtered_count += 1
            self.pending = None
            return
        if phase != "predictive" or trial != len(self.rows) or self.pending is not None:
            raise ValueError("Predictive observer order is inconsistent")
        prediction = np.asarray(snapshot["prediction"])
        if not np.isfinite(prediction).all() or np.any(prediction < 0):
            raise ValueError("Non-finite or negative path prediction")
        np.testing.assert_allclose(prediction.sum(axis=1), 1, atol=1e-12, rtol=0)
        count = snapshot["active"].sum(axis=1)
        participation = 1 / np.sum(snapshot["choice_weights"]**2, axis=1)
        if np.any(participation > count + 1e-10):
            raise ValueError("Path participation exceeds candidate count")
        row = {key: weights @ snapshot[key] for key in RULE_FIELDS}
        row.update(correct_paths=prediction[:, self.categories[trial]-1].copy(),
                   candidate_count=np.asarray(weights @ count),
                   effective_participation=np.asarray(weights @ participation),
                   choice_weighted_beta=np.asarray(weights @ np.sum(
                       snapshot["choice_weights"] * snapshot["beta"], axis=1)),
                   active_beta=weights @ (snapshot["active"] * snapshot["beta"]))
        for key in ("search", "global_search", "failure", "admission_event"):
            row[key] = np.asarray(weights @ snapshot[key])
        self.rows.append(row)
        self.pending = {key: snapshot[key] for key in (*RULE_FIELDS, "prediction")}

    def arrays(self) -> dict[str, np.ndarray]:
        if self.filtered_count != len(self.rows) or self.pending is not None:
            raise ValueError("Observer phases are incomplete")
        return {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}


def run_filter(inputs: dict, seed: int, particles: int, *, control: bool,
               max_trials: int | None = None, observer=None):
    """Keep the cognitive engine and recorded history fixed across controls."""
    arrays = inputs["arrays"]
    np.random.seed(seed)
    return run_state_model_particle_filter(
        engine_config=deepcopy(inputs["engine"]), subject_id=inputs["subject"],
        condition=inputs["condition"], stimulus=arrays.stimulus[:max_trials],
        choices=arrays.choices[:max_trials], feedback=arrays.feedback[:max_trials],
        particle_count=particles, particle_blocks=1, choice_readout_power=1.,
        filter_seed=seed, condition_on_observed_choice=not control,
        resample_threshold_fraction=0. if control else .5,
        processed_data_dir=inputs["paths"]["processed_dir"], dataset_paths=inputs["paths"],
        state_observer=observer)


def run_control_task(inputs: dict, seed_file: Path, output: Path, particles: int) -> str:
    start = time.monotonic()
    with np.load(seed_file, allow_pickle=False) as saved:
        seed = int(saved["seed"])
        if int(saved["particles"]) != particles:
            raise ValueError("A/B particle budgets must match")
        original_first = saved["probabilities"][0].copy()
    observer = UniformHistoryCollector(inputs["arrays"].categories)
    result = run_filter(inputs, seed, particles, control=True, observer=observer)
    p = np.asarray(result.observation_probabilities["prior_t"])
    np.testing.assert_allclose(p[0], original_first, atol=1e-12, rtol=0)
    for key in ("pre_choice_ess", "post_choice_ess"):
        np.testing.assert_allclose(result.diagnostics[key], particles, atol=1e-9, rtol=0)
    if np.any(result.diagnostics["resampled"]):
        raise ValueError("Condition B must never resample")
    states = observer.arrays()
    np.testing.assert_allclose(states["correct_paths"].mean(axis=1),
                               p[np.arange(len(p)), inputs["arrays"].categories-1], atol=1e-12, rtol=0)
    if inputs["engine"]["modules"]["hypo_transitions_mod"]["kwargs"]["execution_inertia"] == 0:
        np.testing.assert_allclose(states["belief"], states["choice_weights"], atol=1e-12, rtol=0)
    with output.open("xb") as stream:
        np.savez_compressed(stream, probabilities=p, seed=seed, particles=particles,
                            elapsed_seconds=time.monotonic()-start, **states)
    return str(output)


def summarize(inputs: dict, evaluation: Path, files: list[Path], output: Path,
              settings: dict) -> tuple[list[dict], dict]:
    """Compare expected correct probabilities on a common trial schedule."""
    sid = inputs["subject"]
    n = len(inputs["arrays"].choices)
    original = evaluation / f"subject_{sid}"
    runs = []
    for file in files:
        with np.load(file, allow_pickle=False) as saved:
            runs.append({key: saved[key] for key in saved.files})
    with np.load(original / "pooled_states.npz", allow_pickle=False) as saved:
        a_states = {key: saved[key][:n] for key in saved.files}
    with np.load(original / "autonomous_rollouts.npz", allow_pickle=False) as saved:
        c_expected = saved["expected_correct_probability"][:, :n].astype(float)
        c_feedback = saved["feedback"][:, :n]
    categories = inputs["arrays"].categories
    index = np.arange(n)
    b_prob = np.mean([run["probabilities"] for run in runs], axis=0)
    b_seed_correct = np.stack([run["probabilities"][index, categories-1] for run in runs])
    b_paths = np.concatenate([run["correct_paths"].T for run in runs], axis=0)
    a_correct = a_states["probabilities"][index, categories-1]
    frame = pd.read_csv(original / "trials.csv").iloc[:n].copy()
    for key, values in (("choice", inputs["arrays"].choices),
                        ("category", categories), ("feedback", inputs["arrays"].feedback)):
        np.testing.assert_array_equal(frame[key], values)
    b_states = {key: np.mean([run[key] for run in runs], axis=0)
                for key in (*RULE_FIELDS, "candidate_count", "effective_participation",
                            "choice_weighted_beta", "search", "global_search", "failure", "admission_event")}
    # Select a descriptive reference rule from A, without treating it as ground truth.
    late_start = max(1, n-int(settings["late_window"]))
    reference = int(a_states["predictive_choice_weights"][late_start:].mean(axis=0).argmax())
    source = frame[[key for key in ("iSub", "condition", "iSession", "iRun", "iBlock", "iTrial",
                                    "valid_trial", "scored", "choice", "category", "feedback") if key in frame]].copy()
    source["observed_correct"] = (inputs["arrays"].feedback == 1).astype(int)
    source["A_correct_probability"] = a_correct
    source["B_correct_probability"] = b_prob[index, categories-1]
    source["C_mean_correct_probability"] = c_expected.mean(axis=0)
    source["C_median_correct_probability"] = np.median(c_expected, axis=0)
    for field in RULE_FIELDS:
        source[f"A_reference_{field}"] = a_states["predictive_"+field][:, reference]
        source[f"B_reference_{field}"] = b_states[field][:, reference]
    for field in ("candidate_count", "effective_participation", "choice_weighted_beta"):
        source[f"A_{field}"] = a_states["predictive_"+field][:, 0]
        source[f"B_{field}"] = b_states[field]
    for field in ("search", "global_search", "failure", "admission_event"):
        source[f"A_{field}"] = a_states["predictive_"+field]
        source[f"B_{field}"] = b_states[field]
    source.to_csv(output / "trials.csv", index=False)
    with (output / "comparison.npz").open("xb") as stream:
        np.savez_compressed(stream, A_correct=a_correct, B_seed_correct=b_seed_correct,
                            B_path_correct=b_paths, C_path_correct=c_expected,
                            C_feedback=c_feedback, B_probabilities=b_prob, **b_states)
    periods = {"overall": slice(1, n), "early": slice(1, min(n, 1+int(settings["late_window"]))),
               "late": slice(late_start, n)}
    rows = []
    for name, span in periods.items():
        seed_values = b_seed_correct[:, span].mean(axis=1)
        c_values = c_expected[:, span].mean(axis=1)
        rows.append(dict(subject=sid, period=name, first_valid_trial=span.start+1,
                         last_valid_trial=span.stop, trials=span.stop-span.start,
                         observed_accuracy=float(source.observed_correct.iloc[span].mean()),
                         A_expected=float(a_correct[span].mean()),
                         B_expected=float(seed_values.mean()),
                         B_seed_mean_sd=float(seed_values.std(ddof=1)) if len(runs)>1 else None,
                         B_seed_mean_min=float(seed_values.min()), B_seed_mean_max=float(seed_values.max()),
                         C_expected_mean=float(c_values.mean()), C_expected_median=float(np.median(c_values)),
                         C_realized_median=float(np.median((c_feedback[:, span]==1).mean(axis=1))),
                         A_minus_B=float(a_correct[span].mean()-seed_values.mean()),
                         B_minus_C_mean=float(seed_values.mean()-c_values.mean())))
    a_scores = score_probabilities(a_states["probabilities"], inputs["arrays"].choices)
    b_scores = score_probabilities(b_prob, inputs["arrays"].choices)
    details = dict(subject=sid, condition=inputs["condition"], trials=n,
                   inertia=inputs["engine"]["modules"]["hypo_transitions_mod"]["kwargs"]["execution_inertia"],
                   reference_rule=reference, reference_rule_definition=pd.read_csv(original / "rule_catalog.csv").iloc[reference].to_dict(),
                   reference_selection="largest_mean_A_choice_weight_in_late_window",
                   particles=int(runs[0]["particles"]), B_repeats=len(runs), C_trajectories=len(c_expected),
                   fit_status=inputs["fit_status"], fit_issues=inputs["fit_issues"],
                   A_choice_nll=a_scores["mean_choice_nll"], B_choice_nll=b_scores["mean_choice_nll"],
                   B_scoring_interpretation="uniform_observed_history_control_score_not_fitted_model_likelihood",
                   B_late_reference={field:float(b_states[field][late_start:, reference].mean()) for field in RULE_FIELDS},
                   A_late_reference={field:float(a_states["predictive_"+field][late_start:, reference].mean()) for field in RULE_FIELDS})
    save_json(output / "summary.json", details)
    return rows, details


def run(config: Path, output: Path, *, smoke: bool = False, jobs: int | None = None) -> None:
    start = time.monotonic()
    settings = yaml.safe_load(config.read_text())
    if settings["protocol"] != "model0923_history_control_v1":
        raise ValueError("Unknown history-control protocol")
    for key in ("particle_count", "pf_repeats", "parallel_budget", "window_size", "late_window", "replay_trials"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"Invalid setting: {key}")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if smoke:
        settings.update(subjects=settings["subjects"][:1], pf_repeats=1,
                        parallel_budget=1, window_size=4, late_window=4)
    elif jobs is not None:
        settings["parallel_budget"] = jobs
    output.mkdir(parents=True)
    (output / "effective_config.yaml").write_text(yaml.safe_dump(dict(settings, smoke=smoke), sort_keys=False))
    tasks, sources, subjects, checks = [], {str(config): digest(config)}, [], []
    for item in settings["subjects"]:
        sid = int(item["subject"])
        evaluation = Path.cwd() / item["evaluation_dir"]
        manifest_file = evaluation / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        fit = Path(manifest["source_fit"])
        source_validation = validate_fit_source(fit)
        directory = output / f"subject_{sid}"
        directory.mkdir()
        cache = directory / "resources_cache"
        cache.mkdir()
        for file in sorted((evaluation / "resources_cache").glob("*.npz")):
            shutil.copy2(file, cache / file.name)
            sources[str(file)] = digest(file)
        inputs = resolve_fit_subject(fit, sid, cache, settings["replay_trials"] if smoke else None)
        save_json(directory / "fit_source_validation.json", source_validation)
        original = evaluation / f"subject_{sid}"
        files = sorted((original / "seeds").glob("pf_*.npz"))
        if len(files) < settings["pf_repeats"]:
            raise ValueError("Saved A evaluation has fewer independent seeds than requested")
        selected = files[:settings["pf_repeats"]]
        for file in (manifest_file, original / "trials.csv", original / "pooled_states.npz",
                     original / "autonomous_rollouts.npz", original / "fixed_simulation.yaml",
                     original / "rule_catalog.csv", *selected):
            sources[str(file)] = digest(file)
            if file != manifest_file:
                key = "report/" + file.relative_to(evaluation).as_posix()
                if sources[str(file)] != manifest["artifacts_sha256"][key]:
                    raise ValueError(f"Saved evaluation artifact hash differs: {file}")
        with np.load(selected[0], allow_pickle=False) as saved:
            seed, particles = int(saved["seed"]), int(saved["particles"])
            expected = saved["probabilities"][:settings["replay_trials"]]
        if particles != settings["particle_count"]:
            raise ValueError("Saved A particle count differs from requested B budget")
        with single_threaded_processes():
            replay = run_filter(inputs, seed, particles, control=False, max_trials=settings["replay_trials"])
        actual = np.asarray(replay.observation_probabilities["prior_t"])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        checks.append(dict(subject=sid, A_replay_trials=len(actual),
                           A_replay_max_difference=float(np.max(np.abs(actual-expected)))))
        seeds_dir = directory / "seeds"
        seeds_dir.mkdir()
        b_files = []
        for i, file in enumerate(selected):
            target = seeds_dir / f"b_{i:02d}.npz"
            tasks.append((inputs, file, target, settings["particle_count"]))
            b_files.append(target)
        subjects.append((inputs, evaluation, b_files, directory))
        print(f"Prepared S{sid}: {len(inputs['arrays'].choices)} trials, {len(selected)} seeds", flush=True)
    workers = parallel_job_count(settings["parallel_budget"], len(tasks))
    print(f"Running {len(tasks)} B tasks with {workers} workers; one numeric thread each", flush=True)
    with single_threaded_processes():
        results = Parallel(n_jobs=workers, batch_size=1, return_as="generator_unordered")(
            delayed(run_control_task)(*task) for task in tasks)
        for i, path in enumerate(results, 1):
            print(f"B completed {i}/{len(tasks)}: {Path(path).parent.parent.name}/{Path(path).name}", flush=True)
    rows, summaries = [], []
    for inputs, evaluation, files, directory in subjects:
        row, details = summarize(inputs, evaluation, files, directory, settings)
        rows.extend(row)
        summaries.append(details)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    script = Path(__file__)
    sources[str(script)] = digest(script)
    save_json(output / "manifest.json", dict(protocol=settings["protocol"], smoke=smoke,
        analysis_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runtime_source_root=str(ROOT_DIR),
        controls=dict(condition_on_observed_choice=False, resample_threshold_fraction=0., learning_update_probability=1.),
        settings=settings, actual_workers=workers, elapsed_seconds=time.monotonic()-start,
        source_sha256=sources, versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        subjects=summaries, seed_matching="B reuses A seeds and particle counts; paths diverge after A conditions/resamples",
        C_source="saved independent autonomous ensemble; no new C simulation",
        artifact_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    save_json(output / "validation.json", dict(status="passed", replay=checks, B_tasks=len(tasks),
        checks=["uniform_weights_at_every_phase", "predictive_filtered_states_identical", "no_resampling",
                "ESS_equals_particle_count", "same_initial_prediction_as_A", "normalized_finite_predictions",
                "path_mean_matches_pooled_prediction", "saved_history_identity", "lambda_zero_identity"],
        new_parameter_fits=0, new_autonomous_simulations=0))
    print(pd.DataFrame(rows).query("period == 'late'").to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shared/model0923_history_control.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--runtime-source", type=Path, help="Run the shared model core from an existing fit source snapshot")
    args = parser.parse_args()
    run(args.config, args.output_dir, smoke=args.smoke, jobs=args.jobs)


if __name__ == "__main__":
    main()
