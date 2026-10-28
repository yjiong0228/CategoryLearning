"""Fit two bounded single-rule baselines and compare identical particle inference.

The public fixed-workspace, feedback, likelihood and particle-filter algorithms
are reused. See docs/model_architecture/model_0923_simple_baselines.md. All
outputs require a fresh directory; inputs and frozen source remain read-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if __name__ == "__main__" and any(a.split("=", 1)[0]=="--runtime-source" for a in sys.argv[1:]):
    boot = argparse.ArgumentParser(add_help=False)
    boot.add_argument("--runtime-source", type=Path, required=True)
    runtime, rest = boot.parse_known_args()
    frozen = runtime.runtime_source.resolve()
    if not (frozen / "src/Bayesian_state/inference/backends/particle_filter.py").is_file():
        raise ValueError("The runtime source must be an existing model snapshot")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    code = ("import sys,runpy; source,script=sys.argv[1:3]; sys.path.insert(0,source); "
            "sys.argv=[script]+sys.argv[3:]; runpy.run_path(script,run_name='__main__')")
    os.execv(sys.executable, [sys.executable,"-c",code,str(frozen),str(Path(__file__).resolve()),*rest])

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import qmc
import yaml

from src.Bayesian_state.evaluation.model_0923_fit_source import resolve_fit_subject, validate_fit_source
from src.Bayesian_state.evaluation.model_0923_postfit import score_probabilities
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.utils.parallel import single_threaded_processes, parallel_job_count
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.seeding import stable_seed


SIMPLE_CLASS = "src.Bayesian_state.model.modules.hypothesis_transition.simple_rule.SimpleRuleHypothesisTransitionModule"
EXTENSIONS = ("src/Bayesian_state/model/model_0923.py",
              "src/Bayesian_state/model/modules/hypothesis_transition/simple_rule.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def install_extension(project: str) -> None:
    """Load only the new configuration contract and subclass from working source.

    Archived numerical algorithms stay on their original package paths. Both
    explicit additions are hashed separately, including in every worker result.
    """
    root = Path(project)
    name = "src.Bayesian_state.model.model_0923"
    spec = importlib.util.spec_from_file_location(name, root / EXTENSIONS[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    importlib.import_module("src.Bayesian_state.model").model_0923 = module
    package = importlib.import_module("src.Bayesian_state.model.modules.hypothesis_transition")
    added = str((root / EXTENSIONS[1]).parent)
    if added not in package.__path__:
        package.__path__.append(added)


def make_engine(original: dict, variant: str, point: dict) -> dict:
    """Replace the search mechanism while retaining full-v3 sensory/feedback semantics."""
    if variant not in ("fixed_rule", "random_rule"):
        raise ValueError("Unknown simple-rule variant")
    engine = deepcopy(original)
    engine["provenance"].update(variant=variant, specification_version="0923-simple-v1",
        scientific_status="bounded_baseline_comparison", source_specification_version="0923-full-v3")
    engine["modules"]["hypo_transitions_mod"] = {
        "class": SIMPLE_CLASS, "kwargs": dict(capacity=1, m=point["switch_probability"], g=1.,
                                               prior_assignment={"method":"pairwise_mass_transfer"})}
    engine["modules"]["beta_mod"]["kwargs"].update(beta_init=point["beta_init"],
                                                   learning_rate=point["learning_rate"])
    engine["modules"]["memory_mod"]["kwargs"]["gamma"] = point["gamma"]
    engine["oral_readout"] = None
    return engine


def candidate_points(settings: dict, original: dict, variant: str, sid: int) -> list[dict]:
    """Predeclared discrete Sobol screen plus the original nuisance-parameter anchor."""
    beta = original["modules"]["beta_mod"]["kwargs"]
    anchor = dict(beta_init=beta["beta_init"], learning_rate=beta["learning_rate"],
                  gamma=original["modules"]["memory_mod"]["kwargs"]["gamma"],
                  switch_probability=0. if variant=="fixed_rule" else .02)
    points = [anchor]
    fields = ["beta_init", "gamma", "learning_rate"]
    if variant == "random_rule":
        fields.append("switch_probability")
    draw = qmc.Sobol(len(fields), scramble=True,
                     seed=stable_seed([settings["base_seed"],sid,variant,"candidates"]))
    for sample in draw.random_base2(9):
        point = {field: settings["support"][field][int(u*len(settings["support"][field]))]
                 for field, u in zip(fields, sample)}
        point.setdefault("switch_probability", 0.)
        if point not in points:
            points.append(point)
        if len(points) == settings["search"]["candidates_per_variant"]:
            return points
    raise ValueError("Candidate request exceeds the bounded support screen")


def filter_run(inputs: dict, engine: dict, seed: int, particles: int, *, uniform=False, n=None):
    arrays = inputs["arrays"]
    np.random.seed(seed)
    return run_state_model_particle_filter(engine_config=deepcopy(engine), subject_id=inputs["subject"],
        condition=inputs["condition"], stimulus=arrays.stimulus[:n], choices=arrays.choices[:n],
        feedback=arrays.feedback[:n], particle_count=particles, particle_blocks=1,
        choice_readout_power=1., filter_seed=seed, condition_on_observed_choice=not uniform,
        resample_threshold_fraction=0. if uniform else .5, processed_data_dir=inputs["paths"]["processed_dir"],
        dataset_paths=inputs["paths"])


def task_run(inputs: dict, variant: str, point: dict, seeds: list[int], particles: int,
             folder: Path, project: str, *, uniform=False, full=False) -> dict:
    install_extension(project)
    start = time.monotonic()
    folder.mkdir(parents=True)
    engine = inputs["engine"] if full else make_engine(inputs["engine"], variant, point)
    runs = []
    for i, seed in enumerate(seeds):
        result = filter_run(inputs, engine, seed, particles, uniform=uniform)
        p = np.asarray(result.observation_probabilities["prior_t"])
        active = np.asarray(result.state_probabilities["active_probability"])
        belief = np.asarray(result.state_probabilities["hypothesis_prior"])
        if not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("Invalid prediction")
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12, rtol=0)
        if not full:
            np.testing.assert_allclose(active.sum(axis=1), 1, atol=1e-12, rtol=0)
            np.testing.assert_allclose(active, belief, atol=1e-12, rtol=0)
            if variant == "fixed_rule":
                np.testing.assert_array_equal(active[:, active[0]==0], 0.)
                np.testing.assert_allclose(result.latent_summaries["replacement_count"], 0, atol=0, rtol=0)
        if uniform:
            np.testing.assert_allclose(result.diagnostics["pre_choice_ess"], particles, atol=1e-9, rtol=0)
            if np.any(result.diagnostics["resampled"]):
                raise ValueError("Uniform control must not resample")
        scores = score_probabilities(p, inputs["arrays"].choices)
        np.savez_compressed(folder/f"seed_{i:02d}.npz", probabilities=p, active=active, belief=belief,
            pre_ess=result.diagnostics["pre_choice_ess"], post_ess=result.diagnostics["post_choice_ess"],
            resampled=result.diagnostics["resampled"], seed=seed, particles=particles,
            core_path=str(Path(importlib.import_module(run_state_model_particle_filter.__module__).__file__).resolve()),
            runtime_source=str(ROOT_DIR), extension_sha256=np.asarray([digest(Path(project)/f) for f in EXTENSIONS]))
        runs.append(dict(seed=seed, **scores, resampling_count=int(np.sum(result.diagnostics["resampled"])),
                         initial_supported_rules=int(np.count_nonzero(active[0])),
                         final_supported_rules=int(np.count_nonzero(active[-1]))))
    probabilities = np.mean([np.load(folder/f"seed_{i:02d}.npz")["probabilities"] for i in range(len(seeds))],axis=0)
    scores = score_probabilities(probabilities, inputs["arrays"].choices)
    out = dict(variant=variant, point=point, particles=particles, seeds=seeds, uniform=uniform,
               **scores, repeats=runs, elapsed_seconds=time.monotonic()-start, directory=str(folder))
    write_json(folder/"scores.json", out)
    return out


def run_tasks(tasks: list[tuple], settings: dict, project: str, flags=None) -> list[dict]:
    workers = parallel_job_count(settings["parallel_budget"], len(tasks))
    print(f"Running {len(tasks)} independent candidates with {workers} workers", flush=True)
    with single_threaded_processes():
        rows = Parallel(n_jobs=workers, batch_size=1, return_as="generator_unordered")(
            delayed(task_run)(*task, project, **(flags or {})) for task in tasks)
        out = []
        for row in rows:
            out.append(row)
            if len(out)%8==0 or len(out)==len(tasks):
                print(f"Completed {len(out)}/{len(tasks)} candidates", flush=True)
    return out


def phase_seeds(settings: dict, sid: int, phase: str, count: int) -> list[int]:
    return [stable_seed([settings["base_seed"], "simple_baselines", sid, phase, i]) for i in range(count)]


def run(config: Path, output: Path, smoke=False) -> None:
    started = time.monotonic()
    project = str(Path.cwd().resolve())
    settings = yaml.safe_load(config.read_text())
    if output.exists():
        raise FileExistsError("Use a new analysis directory")
    output.mkdir(parents=True)
    if smoke:
        settings["subjects"] = settings["subjects"][:1]
        settings["search"].update(candidates_per_variant=2, particles=8, repeats=1, finalists=1,
                                  confirmation_particles=8, confirmation_repeats=1)
        settings["evaluation"].update(particles=8, repeats=1, audit_particles=16, audit_repeats=1)
        settings["parallel_budget"] = 1
    write_json(output/"protocol.json", dict(settings=settings, smoke=smoke, runtime_source=str(ROOT_DIR),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        extension_sha256={f:digest(Path(project)/f) for f in EXTENSIONS},
        driver_sha256=digest(Path(__file__)), config_sha256=digest(config)))
    subjects, checks, source_hashes = {}, [], {}
    for item in settings["subjects"]:
        sid = item["subject"]
        evaluation = Path(project)/item["evaluation_dir"]
        manifest = json.loads((evaluation/"manifest.json").read_text())
        fit = Path(manifest["source_fit"])
        validated = validate_fit_source(fit)
        directory = output/f"subject_{sid}"
        directory.mkdir()
        write_json(directory/"source_validation.json", validated)
        cache = directory/"resources_cache"
        cache.mkdir()
        for p in (evaluation/"resources_cache").glob("*.npz"):
            shutil.copy2(p,cache/p.name)
        inputs = resolve_fit_subject(fit,sid,cache,settings["smoke_trials"] if smoke else None)
        original = evaluation/f"subject_{sid}"
        for name in ("trials.csv","pooled_states.npz","rule_catalog.csv","fixed_simulation.yaml",
                     *[f"seeds/pf_{i:02d}.npz" for i in range(16)]):
            p=original/name
            h=digest(p)
            if h != manifest["artifacts_sha256"][f"report/subject_{sid}/{name}"]:
                raise ValueError(f"Saved evaluation hash differs: {p}")
            source_hashes[str(p)]=h
        with np.load(original/"seeds/pf_00.npz") as saved:
            seed, count = int(saved["seed"]),int(saved["particles"])
            expected=saved["probabilities"][:len(inputs["arrays"].choices)]
        install_extension(project)
        with single_threaded_processes():
            replay=filter_run(inputs,inputs["engine"],seed,count)
        actual=replay.observation_probabilities["prior_t"]
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
        checks.append(dict(subject=sid, A_replay_trials=len(actual), A_max_difference=float(np.max(abs(actual-expected)))))
        subjects[sid]=(inputs,original,directory)
        print(f"S{sid}: full-model replay matched {len(actual)} trials",flush=True)
    screen_tasks=[]
    for sid,(inputs,original,directory) in subjects.items():
        for variant in settings["variants"]:
            points=candidate_points(settings,inputs["engine"],variant,sid)
            write_json(directory/f"{variant}_candidates.json", points)
            seeds=phase_seeds(settings,sid,"screen",settings["search"]["repeats"])
            for i,point in enumerate(points):
                screen_tasks.append((inputs,variant,point,seeds,settings["search"]["particles"],
                                     directory/"screen"/variant/f"candidate_{i:03d}"))
    screen=run_tasks(screen_tasks,settings,project)
    confirmation_tasks=[]
    for sid,(inputs,original,directory) in subjects.items():
        for variant in settings["variants"]:
            selected=sorted([r for r in screen if Path(r["directory"]).parents[2]==directory and r["variant"]==variant],
                            key=lambda r:r["mean_choice_nll"])[:settings["search"]["finalists"]]
            if len(selected)!=settings["search"]["finalists"]:
                raise ValueError("Missing screen candidates")
            seeds=phase_seeds(settings,sid,"confirmation",settings["search"]["confirmation_repeats"])
            for row in selected:
                confirmation_tasks.append((inputs,variant,row["point"],seeds,settings["search"]["confirmation_particles"],
                    directory/"confirmation"/variant/Path(row["directory"]).name))
    confirmed=run_tasks(confirmation_tasks,settings,project)
    selected_rows=[]
    for sid,(inputs,original,directory) in subjects.items():
        seeds=[]
        for i in range(settings["evaluation"]["repeats"]):
            with np.load(original/f"seeds/pf_{i:02d}.npz") as saved:
                seeds.append(int(saved["seed"]))
        for variant in settings["variants"]:
            selected=min([r for r in confirmed if Path(r["directory"]).parents[2]==directory and r["variant"]==variant],
                         key=lambda r:r["mean_choice_nll"])
            write_json(directory/f"{variant}_selected.json", selected)
            (directory/f"{variant}_engine.yaml").write_text(yaml.safe_dump(make_engine(inputs["engine"],variant,selected["point"]),sort_keys=False))
            selected_rows.append((sid,variant,selected["point"],seeds))
    # Split final repeats into independent jobs while retaining the original A seeds.
    eval_tasks=[]
    for sid,variant,point,seeds in selected_rows:
        inputs,original,directory=subjects[sid]
        for i,seed in enumerate(seeds):
            eval_tasks.append((inputs,variant,point,[seed],settings["evaluation"]["particles"],
                               directory/"evaluation"/variant/f"repeat_{i:02d}"))
    final=run_tasks(eval_tasks,settings,project)
    uniform_tasks=[(*t[:-1],t[-1].parents[1].parent/"uniform"/t[-1].parent.name/t[-1].name) for t in eval_tasks]
    uniform=run_tasks(uniform_tasks,settings,project,dict(uniform=True))
    audit_tasks=[]
    for sid,variant,point,seeds in selected_rows:
        inputs,original,directory=subjects[sid]
        for i,seed in enumerate(phase_seeds(settings,sid,"audit",settings["evaluation"]["audit_repeats"])):
            audit_tasks.append((inputs,variant,point,[seed],settings["evaluation"]["audit_particles"],
                                directory/"audit"/variant/f"repeat_{i:02d}"))
    audit=run_tasks(audit_tasks,settings,project)
    write_json(output/"run_index.json",dict(screen=screen,confirmation=confirmed,evaluation=final,uniform=uniform,audit=audit))
    write_json(output/"validation.json",dict(status="passed",replays=checks,
        invariants=["probability normalization","single active rule","belief equals rule membership",
                    "fixed rule has no cognitive replacements","uniform control has no resampling"],
        original_source_sha256=source_hashes,smoke=smoke,elapsed_seconds=time.monotonic()-started,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)))
    print("Completed baseline fitting and evaluation",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/shared/model0923_simple_baselines.yaml"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--runtime-source",type=Path,help="Existing frozen fit source")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    run(args.config,args.output_dir,args.smoke)


if __name__=="__main__":
    main()
