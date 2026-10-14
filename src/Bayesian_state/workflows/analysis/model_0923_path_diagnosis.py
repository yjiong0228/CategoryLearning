"""Replay saved B/C paths and locate rule proposal, retention and support failures.

Run from the repository root with the original fit's --runtime-source. An
analysis-local StateModel subclass records lifecycle outputs; every cognitive
operation delegates to the frozen shared core. Existing probabilities and
autonomous choices must replay before a task is accepted. See
docs/model_architecture/model_0923_path_diagnosis.md for definitions and limits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

if __name__ == "__main__" and "--runtime-source" in sys.argv:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--runtime-source", type=Path, required=True)
    runtime, remaining = bootstrap.parse_known_args()
    frozen = runtime.runtime_source.resolve()
    if not (frozen / "src/Bayesian_state/model/state_model.py").is_file():
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

import src.Bayesian_state.model as model_package
import src.Bayesian_state.simulation.autonomous as autonomous
from src.Bayesian_state.evaluation.model_0923_fit_source import resolve_fit_subject, validate_fit_source
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
from src.Bayesian_state.utils.parallel import parallel_job_count, single_threaded_processes
from src.Bayesian_state.utils.paths import ROOT_DIR


SLOT_FLOATS = ("prior", "posterior", "forgotten", "contribution", "beta_pre", "beta_post",
               "rule_correct", "evidence", "pairing_pre_true", "pairing_post_true")
EVENT_CODES = {"initial": 0, "global": 1, "structural": 2, "functional": 3}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def true_pairing_index() -> int:
    """Identify the task's {1,2}|{3,4} grouping for retrospective scoring only."""
    from src.Bayesian_state.hypothesis_space.observation_model.pairing import PAIRING_LABELS
    return PAIRING_LABELS.index("12|34")


class PathRecorder:
    """Store sparse active-rule states, with stable path identity and event timing."""

    def __init__(self, categories: np.ndarray, paths: int, capacity: int,
                 category_count: int = 4, has_pairing: bool = True) -> None:
        self.categories = np.asarray(categories, dtype=int)
        self.has_pairing = has_pairing
        shape = (len(categories), paths, capacity)
        self.arrays = {key: np.full(shape, np.nan) for key in SLOT_FLOATS}
        self.arrays["rule"] = np.full(shape, -1, dtype=np.int16)
        for key in ("proposed", "capacity_dropped", "source"):
            self.arrays[key] = np.full(shape[:2], -1, dtype=np.int16)
        self.arrays["exits"] = np.full(shape, -1, dtype=np.int16)
        self.arrays["joint_dropped"] = np.full(shape, -1, dtype=np.int16)
        self.arrays["prediction"] = np.full((*shape[:2], category_count), np.nan)
        self.arrays["choice"] = np.zeros(shape[:2], dtype=np.int16)
        self.arrays["feedback"] = np.full(shape[:2], np.nan)
        self.counts = np.zeros(paths, dtype=int)
        self.next_path = 0

    def before(self, model, choice: int, feedback: float) -> tuple:
        path, prepared = model.diagnostic_path, model._pending_trial
        trial = prepared.trial_index
        if trial != self.counts[path]:
            raise ValueError("Nonsequential path callback")
        transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        memory = model.engine.get_module(ModuleRole.MEMORY)
        expected_mode = "factorized" if self.has_pairing else "exact_power"
        if memory.forgetting_mode != expected_mode:
            raise ValueError(f"This diagnostic requires full-v3 {expected_mode} forgetting")
        prediction, snapshot = read_choice_probabilities_from_model(
            model, prepared.perceived_stimulus, power=1., lapse=0., return_snapshot=True)
        ids = np.asarray(snapshot.hypothesis_indices, int)
        row = (trial, path, slice(0, len(ids)))
        event = transition.transition_log[-1]
        a = self.arrays
        a["rule"][row] = ids
        a["prior"][row] = prepared.prior[ids]
        a["contribution"][row] = snapshot.rule_weights
        a["beta_pre"][row] = prepared.beta[ids]
        a["rule_correct"][row] = snapshot.category_probabilities[:, self.categories[trial]-1]
        forgotten = prepared.prior[ids] ** memory.gamma
        forgotten /= forgotten.sum()
        a["forgotten"][row] = forgotten
        pairing = true_pairing_index() if self.has_pairing else None
        if self.has_pairing:
            a["pairing_pre_true"][row] = prepared.log["conditional_pairing_pre"][ids, pairing]
        a["prediction"][trial, path] = prediction
        a["choice"][trial, path], a["feedback"][trial, path] = choice, feedback
        for key in ("proposed", "capacity_dropped"):
            value = event["proposed_hypothesis" if key == "proposed" else key]
            a[key][trial, path] = -1 if value is None else value
        a["source"][trial, path] = EVENT_CODES.get(event["proposal_source"], -1)
        exits = event["autonomous_exits"]
        a["exits"][trial, path, :len(exits)] = exits
        joint = event.get('joint_dropped', [])
        a['joint_dropped'][trial,path,:len(joint)] = joint
        return row, ids, pairing, forgotten

    def after(self, model, token: tuple, result: tuple) -> None:
        row, ids, pairing, forgotten = token
        posterior, _, log = result
        memory = model.engine.get_module(ModuleRole.MEMORY)
        # Binary feedback uses the engine likelihood directly; condition 3
        # marginalizes feedback over the forgotten conditional pairing belief.
        evidence = (memory.feedback_evidence if self.has_pairing else model.engine.likelihood)[ids]
        expected = forgotten * evidence
        expected /= expected.sum()
        np.testing.assert_allclose(posterior[ids], expected, atol=2e-12, rtol=0)
        a = self.arrays
        a["posterior"][row] = posterior[ids]
        a["evidence"][row] = evidence
        a["beta_post"][row] = model.engine.beta[ids]
        if self.has_pairing:
            a["pairing_post_true"][row] = log["conditional_pairing_post"][ids, pairing]
        self.counts[model.diagnostic_path] += 1

    def validate(self) -> None:
        if self.next_path != len(self.counts) or np.any(self.counts != len(self.categories)):
            raise ValueError("Incomplete path recording")
        active = self.arrays["rule"] >= 0
        for key in SLOT_FLOATS:
            if key.startswith("pairing_") and not self.has_pairing:
                if not np.isnan(self.arrays[key]).all():
                    raise ValueError("Pairing is undefined for binary-feedback tasks")
                continue
            if not np.isfinite(self.arrays[key][active]).all():
                raise ValueError(f"Nonfinite active state: {key}")
        for key in ("prior", "posterior", "forgotten", "contribution"):
            np.testing.assert_allclose(np.nansum(self.arrays[key], axis=2), 1, atol=2e-12, rtol=0)
        np.testing.assert_allclose(self.arrays["prediction"].sum(axis=-1), 1, atol=2e-12, rtol=0)
        mixed = np.nansum(self.arrays["contribution"] * self.arrays["rule_correct"], axis=2)
        correct = np.take_along_axis(self.arrays["prediction"], self.categories[:, None, None]-1, axis=2)[..., 0]
        np.testing.assert_allclose(mixed, correct, atol=2e-12, rtol=0)


@contextmanager
def record_lifecycle(recorder: PathRecorder):
    """Install a process-local observation wrapper and restore both exports."""
    original_model, original_autonomous = model_package.StateModel, autonomous.StateModel

    class RecordedStateModel(StateModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.diagnostic_path = recorder.next_path
            recorder.next_path += 1

        def complete_trial(self, choice, feedback, **kwargs):
            token = recorder.before(self, choice, feedback)
            result = super().complete_trial(choice, feedback, **kwargs)
            recorder.after(self, token, result)
            return result

    model_package.StateModel = autonomous.StateModel = RecordedStateModel
    try:
        yield
    finally:
        model_package.StateModel, autonomous.StateModel = original_model, original_autonomous


def replay_task(inputs: dict, kind: str, source: Path, indices: list[int], output: Path) -> dict:
    """Replay saved numerical paths, rejecting any altered prediction/history."""
    start = time.monotonic()
    arrays = inputs["arrays"]
    n = len(arrays.choices)
    capacity = inputs["engine"]["modules"]["hypo_transitions_mod"]["kwargs"]["max_capacity"]
    with np.load(source, allow_pickle=False) as saved:
        if kind == "B":
            paths, seed = int(saved["particles"]), int(saved["seed"])
            expected = saved["probabilities"][:n].copy()
            expected_paths = saved["correct_paths"][:n].copy()
            seeds = [seed]
        else:
            seeds = saved["seeds"][indices].astype(int).tolist()
            paths = len(seeds)
            expected = saved["expected_correct_probability"][indices, :n].T.copy()
            expected_choices = saved["choices"][indices, :n].T.copy()
            expected_feedback = saved["feedback"][indices, :n].T.copy()
    recorder = PathRecorder(arrays.categories, paths, capacity,
                            category_count=2 if inputs["condition"] == 1 else 4,
                            has_pairing=inputs["condition"] == 3)
    with record_lifecycle(recorder):
        if kind == "B":
            np.random.seed(seed)
            result = run_state_model_particle_filter(
                engine_config=deepcopy(inputs["engine"]), subject_id=inputs["subject"],
                condition=inputs["condition"], stimulus=arrays.stimulus, choices=arrays.choices,
                feedback=arrays.feedback, particle_count=paths, particle_blocks=1, filter_seed=seed,
                choice_readout_power=1.,
                condition_on_observed_choice=False, resample_threshold_fraction=0.,
                processed_data_dir=inputs["paths"]["processed_dir"], dataset_paths=inputs["paths"])
            if np.any(result.diagnostics["resampled"]):
                raise ValueError("B path identity requires no resampling")
            actual = np.asarray(result.observation_probabilities["prior_t"])
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(recorder.arrays["prediction"].mean(axis=1), actual, atol=1e-12, rtol=0)
            correct = np.take_along_axis(recorder.arrays["prediction"], arrays.categories[:, None, None]-1, axis=2)[..., 0]
            np.testing.assert_allclose(correct, expected_paths, atol=1e-12, rtol=0)
            error = float(np.max(np.abs(actual-expected)))
        else:
            for path, seed in enumerate(seeds):
                result = autonomous.run_autonomous_category_learning(
                    engine_config=deepcopy(inputs["engine"]), subject_id=inputs["subject"],
                    condition=inputs["condition"], stimulus=arrays.stimulus, categories=arrays.categories,
                    trajectory_seed=seed, processed_data_dir=inputs["paths"]["processed_dir"],
                    dataset_paths=inputs["paths"])
                np.testing.assert_allclose(recorder.arrays["prediction"][:, path],
                                           result.trajectory.observed_probabilities, atol=1e-12, rtol=0)
            correct = np.take_along_axis(recorder.arrays["prediction"], arrays.categories[:, None, None]-1, axis=2)[..., 0]
            # The original autonomous export stores probabilities as float32.
            np.testing.assert_allclose(correct, expected, atol=6e-8, rtol=0)
            np.testing.assert_array_equal(recorder.arrays["choice"], expected_choices)
            np.testing.assert_array_equal(recorder.arrays["feedback"], expected_feedback)
            error = float(np.max(np.abs(correct-expected)))
    recorder.validate()
    with output.open("xb") as stream:
        np.savez_compressed(stream, **recorder.arrays, seeds=seeds, categories=arrays.categories)
    return dict(subject=inputs["subject"], condition=kind, paths=paths, trials=n,
                replay_max_difference=error, elapsed_seconds=time.monotonic()-start, file=str(output))


def geometric_catalog(inputs: dict, catalog: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Score physical stimuli after the fact; tied category maxima share credit."""
    model = StateModel(deepcopy(inputs["engine"]), context=ModelContext(
        condition=inputs["condition"], subject_id=inputs["subject"],
        processed_data_dir=inputs["paths"]["processed_dir"], dataset_paths=inputs["paths"]))
    x, categories = inputs["arrays"].stimulus, inputs["arrays"].categories
    accuracies = []
    for rule in catalog.rule:
        probabilities = model.partition_model.get_category_probabilities(
            hypo=int(rule), data=(x, np.ones(len(x), int), np.ones(len(x))),
            beta=beta, distance_mode=model.engine.distance_mode)
        maxima = np.isclose(probabilities, probabilities.max(axis=0), atol=1e-12, rtol=0)
        accuracies.append(float(np.mean(maxima[categories-1, np.arange(len(x))]/maxima.sum(axis=0))))
    return catalog.assign(physical_argmax_accuracy=accuracies)


def extract_episodes(a: dict[str, np.ndarray], subject: int, kind: str, batch: str) -> list[dict]:
    """One uninterrupted candidate residence; final residences are right-censored."""
    ids = a["rule"]
    n, paths, _ = ids.shape
    rows = []
    for path in range(paths):
        open_episodes = {}
        for trial in range(n+1):
            present = {} if trial == n else {int(h): k for k, h in enumerate(ids[trial, path]) if h >= 0}
            for rule in list(open_episodes):
                if rule in present:
                    continue
                entry = open_episodes.pop(rule)
                start = entry.pop("start_index")
                values = entry.pop("values")
                cause = "censored" if trial == n else (
                    "joint_selection" if 'joint_dropped' in a and rule in a['joint_dropped'][trial,path] else
                    "capacity" if a["capacity_dropped"][trial, path] == rule else
                    "autonomous_exit" if rule in a["exits"][trial, path] else "unknown")
                if cause == "unknown":
                    raise ValueError("Rule disappeared without a recorded exit")
                row = dict(subject=subject, condition=kind, batch=batch, path=path, rule=rule,
                           start_trial=start+1, last_trial=trial, duration=trial-start,
                           right_censored=trial == n, exit_cause=cause, **entry)
                for key in SLOT_FLOATS:
                    data = np.asarray(values[key])
                    row[key+"_entry"], row[key+"_last"] = float(data[0]), float(data[-1])
                    row[key+"_mean"], row[key+"_max"] = float(data.mean()), float(data.max())
                row["feedback_support_gain_mean"] = float(np.mean(np.asarray(values["posterior"])-values["forgotten"]))
                row["net_belief_gain_mean"] = float(np.mean(np.asarray(values["posterior"])-values["prior"]))
                rows.append(row)
            for rule, slot in present.items():
                if rule not in open_episodes:
                    source = "initial" if trial == 0 else next(
                        (name for name, code in EVENT_CODES.items() if code == a["source"][trial, path]), "unknown")
                    if trial > 0 and a["proposed"][trial, path] != rule:
                        raise ValueError("Candidate entry lacks a proposal event")
                    open_episodes[rule] = dict(start_index=trial, proposal_source=source,
                                              values={key: [] for key in SLOT_FLOATS})
                for key in SLOT_FLOATS:
                    open_episodes[rule]["values"][key].append(a[key][trial, path, slot])
    return rows


def summarize_subject(directory: Path, inputs: dict, reference: int, settings: dict) -> None:
    catalog = pd.read_csv(directory / "rule_catalog.csv")
    useful = catalog.loc[catalog.physical_argmax_accuracy >= settings["geometric_accuracy_threshold"], "rule"].tolist()
    targets = {"reference": [reference], "task_compatible": useful}
    episodes, trials, path_rows = [], [], []
    for file in sorted((directory / "traces").glob("*.npz")):
        kind = file.stem[0]
        with np.load(file, allow_pickle=False) as saved:
            a = {key: saved[key] for key in saved.files}
        episodes.extend(extract_episodes(a, inputs["subject"], kind, file.stem))
        n, paths, _ = a["rule"].shape
        for target, rules in targets.items():
            mask = np.isin(a["rule"], rules)
            active = mask.any(axis=2)
            mass = {key: np.where(mask, a[key], 0).sum(axis=2)
                    for key in ("prior", "posterior", "contribution")}
            for trial in range(n):
                trials.append(dict(subject=inputs["subject"], condition=kind, batch=file.stem,
                    target=target, trial=trial+1, paths=paths, active=float(active[trial].mean()),
                    **{key: float(value[trial].mean()) for key, value in mass.items()}))
            correct = np.take_along_axis(a["prediction"], a["categories"][:, None, None]-1, axis=2)[..., 0]
            late = slice(max(1, n-settings["late_window"]), n)
            for path in range(paths):
                path_rows.append(dict(subject=inputs["subject"], condition=kind, batch=file.stem, path=path,
                    target=target, ever_active=bool(active[:, path].any()),
                    ever_supported=bool((mass["posterior"][:, path] >= settings["support_threshold"]).any()),
                    ever_contributing=bool((mass["contribution"][:, path] >= settings["support_threshold"]).any()),
                    late_active=float(active[late, path].mean()), late_contribution=float(mass["contribution"][late, path].mean()),
                    late_correct=float(correct[late, path].mean())))
    ep, paths = pd.DataFrame(episodes), pd.DataFrame(path_rows)
    ep.to_csv(directory / "episodes.csv", index=False)
    paths.to_csv(directory / "paths.csv", index=False)
    pd.DataFrame(trials).to_csv(directory / "trials.csv", index=False)
    summaries = []
    n = len(inputs["arrays"].choices)
    for (kind, target), group in paths.groupby(["condition", "target"]):
        visits = ep[(ep.condition == kind) & ep.rule.isin(targets[target])]
        eligible = visits[visits.start_trial <= n-settings["retention_trials"]+1]
        row = dict(subject=inputs["subject"], condition=kind, target=target, rules=targets[target],
                   paths=len(group), ever_active=float(group.ever_active.mean()),
                   ever_supported=float(group.ever_supported.mean()), ever_contributing=float(group.ever_contributing.mean()),
                   late_active=float(group.late_active.mean()), late_contribution=float(group.late_contribution.mean()),
                   late_correct=float(group.late_correct.mean()), episodes=len(visits),
                   retention_eligible=len(eligible), right_censored=int(visits.right_censored.sum()))
        for name, value in {
            "retention_fraction": (eligible.duration >= settings["retention_trials"]).mean(),
            "exit_fraction": (visits.exit_cause == "autonomous_exit").mean(),
            "capacity_fraction": (visits.exit_cause == "capacity").mean(),
            "median_completed_duration": visits.loc[~visits.right_censored, "duration"].median(),
            "median_entry_prior": visits.prior_entry.median(),
            "median_entry_beta": visits.beta_pre_entry.median(),
            "median_last_posterior": visits.loc[~visits.right_censored, "posterior_last"].median(),
            "fraction_ever_support_per_episode": (visits.posterior_max >= settings["support_threshold"]).mean(),
        }.items():
            row[name] = None if pd.isna(value) else float(value)
        summaries.append(row)
    save_json(directory / "summary.json", dict(reference_rule=reference, useful_rules=useful, groups=summaries))


def run(config: Path, output: Path, *, smoke: bool = False, jobs: int | None = None) -> None:
    start = time.monotonic()
    settings = yaml.safe_load(config.read_text())
    if settings["protocol"] != "model0923_path_diagnosis_v1":
        raise ValueError("Unknown diagnostic protocol")
    for key in ("b_repeats", "autonomous_chunk_size", "parallel_budget", "late_window", "retention_trials", "smoke_trials"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if smoke:
        settings.update(subjects=settings["subjects"][:1], b_repeats=1, parallel_budget=1)
    if jobs is not None and not smoke:
        if jobs < 1:
            raise ValueError("jobs must be positive")
        settings["parallel_budget"] = jobs
    output.mkdir(parents=True)
    (output / "effective_config.yaml").write_text(yaml.safe_dump(dict(settings, smoke=smoke), sort_keys=False))
    sources, tasks, subjects = {str(config): digest(config)}, [], []
    for item in settings["subjects"]:
        sid, evaluation = int(item["subject"]), Path(item["evaluation_dir"])
        manifest_file = evaluation / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        fit = Path(manifest["source_fit"])
        source_validation = validate_fit_source(fit)
        directory = output / f"subject_{sid}"
        (directory / "traces").mkdir(parents=True)
        cache = directory / "resources_cache"
        cache.mkdir()
        for file in (evaluation / "resources_cache").glob("*.npz"):
            shutil.copy2(file, cache / file.name)
            sources[str(file)] = digest(file)
        inputs = resolve_fit_subject(fit, sid, cache, settings["smoke_trials"] if smoke else None)
        save_json(directory / "source_validation.json", source_validation)
        original = evaluation / f"subject_{sid}"
        for name in ("rule_catalog.csv", "pooled_states.npz", "autonomous_rollouts.npz", "trials.csv"):
            file = original / name
            sources[str(file)] = digest(file)
            if sources[str(file)] != manifest["artifacts_sha256"]["report/"+file.relative_to(evaluation).as_posix()]:
                raise ValueError(f"Evaluation source changed: {file}")
        sources[str(manifest_file)] = digest(manifest_file)
        frame = pd.read_csv(original / "trials.csv").iloc[:len(inputs["arrays"].choices)]
        for key, values in (("choice", inputs["arrays"].choices), ("category", inputs["arrays"].categories), ("feedback", inputs["arrays"].feedback)):
            np.testing.assert_array_equal(frame[key], values)
        frame.to_csv(directory / "observed_trials.csv", index=False)
        with np.load(original / "pooled_states.npz", allow_pickle=False) as saved:
            reference = int(saved["predictive_choice_weights"][-settings["late_window"]:].mean(axis=0).argmax())
        with single_threaded_processes():
            catalog = geometric_catalog(inputs, pd.read_csv(original / "rule_catalog.csv"), settings["geometry_beta"])
        catalog.to_csv(directory / "rule_catalog.csv", index=False)
        history = Path(settings["history_control_dir"])
        history_manifest = json.loads((history / "manifest.json").read_text())
        sources[str(history / "manifest.json")] = digest(history / "manifest.json")
        b_files = sorted((history / f"subject_{sid}/seeds").glob("b_*.npz"))
        if len(b_files) < settings["b_repeats"]:
            raise ValueError("Insufficient saved B seeds")
        for file in b_files[:settings["b_repeats"]]:
            sources[str(file)] = digest(file)
            if sources[str(file)] != history_manifest["artifact_sha256"][str(file.relative_to(history))]:
                raise ValueError(f"History source changed: {file}")
            tasks.append((inputs, "B", file, [], directory / "traces" / ("B_"+file.stem+".npz")))
        c_file = original / "autonomous_rollouts.npz"
        with np.load(c_file, allow_pickle=False) as saved:
            count = 1 if smoke else len(saved["seeds"])
        size = settings["autonomous_chunk_size"]
        for first in range(0, count, size):
            indices = list(range(first, min(count, first+size)))
            tasks.append((inputs, "C", c_file, indices, directory / "traces" / f"C_{first:03d}.npz"))
        subjects.append((directory, inputs, reference))
        print(f"Prepared S{sid}: {len(inputs['arrays'].choices)} trials; reference R{reference}", flush=True)
    workers = parallel_job_count(settings["parallel_budget"], len(tasks))
    print(f"Replaying {len(tasks)} tasks on {workers} workers", flush=True)
    checks = []
    with single_threaded_processes():
        for check in Parallel(n_jobs=workers, batch_size=1, return_as="generator_unordered")(
                delayed(replay_task)(*task) for task in tasks):
            checks.append(check)
            print(f"Completed {len(checks)}/{len(tasks)}: S{check['subject']} {check['condition']}, max replay difference={check['replay_max_difference']:.3g}", flush=True)
    for directory, inputs, reference in subjects:
        summarize_subject(directory, inputs, reference, settings)
    sources[str(Path(__file__))] = digest(Path(__file__))
    save_json(output / "validation.json", dict(status="passed", tasks=checks,
        checks=["saved_source_hashes", "original_B_path_probabilities", "original_C_choices_and_feedback",
                "original_C_probabilities_float32_tolerance", "normalized_states", "mixture_readout_identity",
                "feedback_posterior_reconstruction", "entry_and_exit_accounting"],
        refits=0, new_random_trajectories=0))
    save_json(output / "manifest.json", dict(protocol=settings["protocol"], smoke=smoke,
        runtime_source_root=str(ROOT_DIR), actual_workers=workers, source_sha256=sources,
        analysis_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        elapsed_seconds=time.monotonic()-start,
        artifact_sha256={str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/shared/model0923_path_diagnosis.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    run(args.config, args.output_dir, smoke=args.smoke, jobs=args.jobs)


if __name__ == "__main__":
    main()
