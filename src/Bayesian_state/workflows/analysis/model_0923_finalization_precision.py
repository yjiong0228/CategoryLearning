"""Run a bounded precision comparison at one frozen H and memory point.

Use --config YAML --output NEW_DIRECTORY [--smoke]. The declared cells are
SEARCH_ADD10, SEARCH_LOG25 and POOL_LOG25, or a subset needed to fill a missing
comparison. All cells start from the same behavior-selected reference arm.
Only beta_init and learning_rate are selected from training choices. The POOL
cell changes candidate proposal while preserving the reference H coordinates.

ONLINE and SEPARATE use the original cohort worker, retained precision and MIX
choice readout. Grid construction, nomination, random banks, execution audits
and scores reuse existing functions. Smoke runs one eight-trial training filter
and one sixteen-trial autonomous path per cell, plus exact reproduction of an
original autonomous prefix. Full scientific comparison requires a separate run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

from . import model_0923_cohort_belief_refit as driver
from .model_0923_fixed_h_precision import (
    INIT, RATE, choose_finalists, precision_points, seed_bank,
)
from .model_0923_fixed_h_precision_cohort import resolved_family, validate_boundary
from .model_0923_h_validation import combine_evaluation_chunks, write
from ...evaluation.model_0923_cohort_belief_refit import audit_job
from ...evaluation.model_0923_estimation_scope import paired_difference, validate_generation
from ...evaluation.model_0923_h_validation import digest, jackknife_scores, jackknife_se, load, nll
from ...evaluation.model_0923_reentry_refit import window_estimate

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
CELLS = {
    "SEARCH_ADD10": ("SEARCH", "ADD10"),
    "SEARCH_LOG25": ("SEARCH", "LOG25"),
    "POOL_LOG25": ("POOL", "LOG25"),
}
COMPARISONS = (
    ("SEARCH_LOG25", "SEARCH_ADD10", "LOG25_minus_ADD10"),
    ("SEARCH_LOG25", "POOL_LOG25", "SEARCH_minus_POOL_LOG25"),
)


def validate_settings(settings: dict) -> None:
    """Keep cell identity, precision definitions and numerical budgets explicit."""
    cells = settings["cells"]
    if (not isinstance(cells, list) or not cells or len(set(cells)) != len(cells)
            or not set(cells) <= CELLS.keys()
            or settings["scope"] not in ("software_validation", "development_comparison")
            or settings["belief_mechanism"] not in driver.INTERVENTIONS
            or settings["reference_arm"] not in ("SEARCH", "POOL")
            or type(settings["subject"]) is not int
            or type(settings["parallel_budget"]) is not int
            or not 1 <= settings["parallel_budget"] <= 128
            or settings["block_width"] != 32):
        raise ValueError("Declare bounded cells, mechanism, participant and worker budget")
    if set(settings["families"]) != {"ADD10", "LOG25"}:
        raise ValueError("Declare the two existing precision families")
    for name, mode, scale in (("ADD10", "fast", 10.), ("LOG25", "log", 1.)):
        family = settings["families"][name]
        if (family["precision"], family["rate_scale"], family["upper"]) != (mode, scale, 25.):
            raise ValueError("Precision definition or common ceiling changed")
        if len(settings["initial_values"]) * len(family["rates"]) > 40:
            raise ValueError("Precision grid exceeds the declared bounded comparison")
    for stage in ("screen", "confirmation", "evaluation"):
        budget = settings[stage]
        if not 2 <= budget["repeats"] <= 8 or not 2 <= budget["particles"] <= 512:
            raise ValueError("Require bounded independent filters")
    evaluation = settings["evaluation"]
    if (not 4 <= evaluation["paths"] <= 128
            or not 1 <= evaluation["path_chunk"] <= 4
            or settings["confirmation"]["finalists"] not in (1, 2)):
        raise ValueError("Require bounded path banks and finalist counts")


def build_cells(settings: dict, reference: dict) -> dict:
    """Construct every proposal cell from the same frozen H and memory point."""
    validate_settings(settings)
    result = {}
    for name in settings["cells"]:
        arm, family_name = CELLS[name]
        family = resolved_family(reference["hyperparams"], settings["families"][family_name])
        points = precision_points(reference["hyperparams"], settings["initial_values"], family)
        anchor = next(p for p in points if [p["hyperparams"][INIT], p["hyperparams"][RATE]] == family["anchor"])
        if family_name == "ADD10" and anchor["hyperparams"] != reference["hyperparams"]:
            raise ValueError("ADD10 anchor must reproduce the frozen reference")
        result[name] = dict(arm=arm, family=family_name, points=points,
                            anchor=anchor["id"], reference=reference)
    return result


def check_template(job: dict, settings: dict, point: dict, phase: str) -> None:
    """Reject a different source mechanism, choice policy or fitted point."""
    mechanism = settings["belief_mechanism"]
    if (job["phase"] != phase or job["candidate"] != point or job["smoke"]
            or job.get("execution_policy", "MIX") != "MIX"
            or job.get("probe_mechanism") != mechanism
            or job.get("cohort_belief_mechanism") != mechanism
            or job.get("belief_intervention") != driver.INTERVENTIONS[mechanism]
            or job.get("online_belief") is not (mechanism == "ONLINE")
            or job.get("retain_precision") is not True):
        raise ValueError("Source template differs from the frozen MIX reference")


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError("Use a new finalization precision directory")
    settings = yaml.safe_load(config.read_text())
    validate_settings(settings)
    if settings["scope"] == "software_validation" and not smoke:
        raise ValueError("Software-validation configurations require --smoke")
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        value = digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError("Bound input changed: " + str(path))
        inputs[str(path)] = value
        return path

    source = ROOT / settings["reference_evaluation"]
    bind(config)
    protocol, done = [load(bind(source / name)) for name in ("protocol.json", "completion.json")]
    frozen_file = Path(protocol["fit"]) / "nominees.json"
    frozen = load(bind(frozen_file))
    if (done["status"] != "complete" or frozen["smoke"]
            or frozen["status"] != "frozen_before_test"
            or protocol["nomination_sha256"] != digest(frozen_file)
            or done["nomination_sha256"] != digest(frozen_file)):
        raise ValueError("Require a completed evaluation of frozen training nominees")
    nominees = {n["model"]: n["candidate"] for n in frozen["nominees"] if n["subject"] == settings["subject"]}
    if settings["reference_arm"] not in nominees:
        raise ValueError("Missing frozen reference arm")
    reference = nominees[settings["reference_arm"]]
    jobs = [load(bind(file)) for file in sorted((source / "jobs").glob("*.json"))]
    templates = {}
    for phase in ("evaluate", "autonomous"):
        matching = [j for j in jobs if j["model"] == settings["reference_arm"]
                    and j["phase"] == phase and j["candidate"] == reference]
        if not matching:
            raise ValueError("Missing reference role")
        job = matching[0]
        check_template(job, settings, reference, phase)
        for name, expected in job["source_sha256"].items():
            if digest(bind(ROOT / name)) != expected:
                raise ValueError("Source implementation changed: " + name)
        templates[phase] = job
    template = templates["evaluate"]
    if any(j["folder"] != template["folder"] or j["runtime"] != template["runtime"] for j in templates.values()):
        raise ValueError("Source roles use different task data or runtime")
    metadata = load(bind(Path(template["folder"]) / "prepared.json"))
    training_trials, total_trials = validate_boundary(settings, metadata)
    for name, expected in metadata["array_sha256"].items():
        if digest(bind(Path(template["folder"]) / name)) != expected:
            raise ValueError("Prepared task data changed")
    cells = build_cells(settings, reference)
    banks = {stage: seed_bank(settings, stage, settings[stage]["repeats"])
             for stage in ("screen", "confirmation", "evaluation")}
    banks["autonomous"] = seed_bank(settings, "autonomous", settings["evaluation"]["paths"])
    all_seeds = [s for bank in banks.values() for s in bank]
    if len(set(all_seeds)) != len(all_seeds) or set(all_seeds) & {s for j in jobs for s in j["seeds"]}:
        raise ValueError("Stage banks must be independent of each other and the reference evaluation")
    additions = {ENTRY, driver.ENTRY}
    for function in (precision_points, resolved_family, audit_job, window_estimate, paired_difference):
        module = sys.modules[function.__module__]
        additions.add(str(Path(module.__file__).resolve().relative_to(ROOT)))
    for name in additions:
        bind(ROOT / name)
    output.mkdir(parents=True)
    for name in ("jobs", "chunks", "banks"):
        (output / name).mkdir()
    shutil.copyfile(config, output / "config.yaml")
    for name in additions | set(template["source_sha256"]):
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    workers = min(1 if smoke else settings["parallel_budget"], len(os.sched_getaffinity(0)))
    write(output / "protocol.json", dict(settings=settings, smoke=smoke, cells=cells, seeds=banks,
          inputs_sha256=inputs.copy(), workers=workers, numeric_threads=1,
          changed_coordinates=[INIT, RATE], selection_data="training choices only", oral_used=False,
          scope="Fixed-H precision selection and proposal comparison; MIX readout",
          subject=settings["subject"], condition=metadata["condition"],
          training_trials=training_trials, total_trials=total_trials,
          versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "PyYAML")}))

    def make_job(cell, point, stage, seeds):
        phase = "autonomous" if stage == "autonomous" else "evaluate" if stage == "evaluation" else "training"
        declared = cells[cell]
        job = deepcopy(templates["autonomous" if phase == "autonomous" else "evaluate"])
        name = f"{stage}_{cell}_{point['id']}_{seeds[0]}"
        job.update(phase=phase, name=name, model=declared["arm"], candidate=point, seeds=seeds,
                   smoke=smoke, execution_policy="MIX", finalization_precision_cell=cell,
                   particles=2 if smoke else settings["evaluation" if stage == "autonomous" else stage]["particles"],
                   target=str((output / "chunks" / name).resolve()),
                   arm_settings={k: v for k, v in settings["families"][declared["family"]].items() if k != "anchor"})
        job["source_sha256"].update({name: digest(ROOT / name) for name in additions})
        return job

    def launch(job):
        file = output / "jobs" / (job["name"] + ".json")
        write(file, job)
        launcher = ("import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);"
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix="finalization_precision_") as cache:
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                       NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix(".log").open("x") as stream:
                subprocess.run([sys.executable, "-c", launcher, job["runtime"], str(ROOT / driver.ENTRY),
                                "--worker", str(file.resolve())], cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)

    def execute(group):
        print(f"{len(group)} jobs; {min(workers, len(group))} workers; numeric threads=1", flush=True)
        with ThreadPoolExecutor(max_workers=min(workers, len(group))) as pool:
            for index, future in enumerate(as_completed([pool.submit(launch, job) for job in group]), 1):
                future.result()
                if index % 8 == 0 or index == len(group):
                    print(f"{index}/{len(group)} complete", flush=True)
        for job in group:
            audit_job(output / "jobs" / (job["name"] + ".json"), settings["belief_mechanism"], bind, smoke=smoke)
            if job["phase"] == "autonomous":
                target = Path(job["target"])
                with np.load(bind(target.with_suffix(".npz"))) as arrays:
                    validate_generation(arrays, load(bind(target.with_suffix(".json"))))

    def training_stage(stage, points):
        group = [make_job(cell, point, stage, [seed]) for cell, values in points.items() for point in values
                 for seed in (banks[stage][:1] if smoke else banks[stage])]
        execute(group)
        rows = []
        for cell, values in points.items():
            for point in values:
                files = [Path(j["target"]) for j in group if j["finalization_precision_cell"] == cell
                         and j["candidate"]["id"] == point["id"]]
                arrays = []
                for file in files:
                    with np.load(file.with_suffix(".npz")) as saved:
                        arrays.append(saved["observed_probability"])
                values_array = np.concatenate(arrays)
                rows.append(dict(cell=cell, id=point["id"], training_nll=nll(values_array),
                                 mc_se=None if smoke else jackknife_se(jackknife_scores(values_array)),
                                 beta_init=point["hyperparams"][INIT], configured_rate=point["hyperparams"][RATE],
                                 score_files=[str(file.with_suffix(".json")) for file in files]))
        write(output / (stage + ".json"), dict(rows=rows))
        pd.DataFrame(rows).to_csv(output / (stage + ".csv"), index=False)
        return rows

    points = {cell: [p for p in c["points"] if not smoke or p["id"] == c["anchor"]] for cell, c in cells.items()}
    screened = training_stage("screen", points)
    if smoke:
        execute([make_job(cell, values[0], "autonomous", banks["autonomous"][:1]) for cell, values in points.items()])
        original = templates["autonomous"]
        regression = deepcopy(original)
        regression.update(name="original_autonomous_prefix", smoke=True, seeds=original["seeds"][:1],
                          target=str((output / "chunks" / "original_autonomous_prefix").resolve()))
        execute([regression])
        saved_path = bind(Path(original["target"]).with_suffix(".npz"))
        new_path = bind(Path(regression["target"]).with_suffix(".npz"))
        checks = 0
        with np.load(saved_path) as old, np.load(new_path) as new:
            for key in ("choices", "probabilities", "correct_probability", "present", "contribution", "beta"):
                np.testing.assert_array_equal(new[key], old[key][:1, :16])
                checks += new[key].size
        for path, expected in inputs.items():
            if digest(Path(path)) != expected:
                raise ValueError("Input changed during smoke: " + path)
        write(output / "completion.json", dict(status="complete", smoke=True,
              full_evaluation_validated=False, original_prefix_exact_values=checks,
              protocol_sha256=digest(output / "protocol.json"), inputs_sha256=inputs))
        return
    finalists = {}
    for cell, declared in cells.items():
        ids = choose_finalists([r for r in screened if r["cell"] == cell], declared["anchor"],
                               settings["confirmation"]["finalists"])
        finalists[cell] = [p for p in declared["points"] if p["id"] in ids]
    confirmed = training_stage("confirmation", finalists)
    nominees = {cell: next(p for p in points if p["id"] == min(
        [r for r in confirmed if r["cell"] == cell], key=lambda r: (r["training_nll"], r["id"]))["id"])
        for cell, points in finalists.items()}
    write(output / "nominees.json", dict(status="frozen_before_test", nominees=nominees,
          confirmation_sha256=digest(output / "confirmation.json"), test_choices_used=False, oral_used=False))
    nomination_sha = digest(output / "nominees.json")
    final_jobs = []
    evaluation = settings["evaluation"]
    for cell, point in nominees.items():
        final_jobs.extend(make_job(cell, point, "evaluation", [seed]) for seed in banks["evaluation"])
        final_jobs.extend(make_job(cell, point, "autonomous", banks["autonomous"][i:i + evaluation["path_chunk"]])
                          for i in range(0, evaluation["paths"], evaluation["path_chunk"]))
    execute(final_jobs)
    estimates, scores = {}, []
    for cell, point in nominees.items():
        combined = {}
        for phase in ("evaluate", "autonomous"):
            group = [j for j in final_jobs if j["finalization_precision_cell"] == cell and j["phase"] == phase]
            job = deepcopy(group[0])
            target = output / "banks" / f"{cell}_{phase}"
            job.update(target=str(target.resolve()), seeds=[seed for j in group for seed in j["seeds"]])
            combine_evaluation_chunks(job, group, phase == "autonomous")
            with np.load(bind(target.with_suffix(".npz"))) as arrays:
                combined[phase] = {key: arrays[key] for key in arrays.files}
            bind(target.with_suffix(".json"))
        for window, first, last in (("training", 0, training_trials), ("later", training_trials, total_trials),
                                    ("whole", 0, total_trials)):
            conditional, autonomous = combined["evaluate"], combined["autonomous"]
            estimate = window_estimate(conditional, autonomous, first, last, settings["block_width"])
            estimates[cell, window] = estimate
            scores.append(dict(cell=cell, window=window, candidate=point["id"], **estimate["metrics"],
                  choice_nll_mc_se=jackknife_se(estimate["jackknife"]["choice_nll"]),
                  autonomous_correct=float((autonomous["choices"][:, first:last] ==
                                            autonomous["categories"][None, first:last]).mean()),
                  human_correct=float((conditional["choices"][first:last] == conditional["categories"][first:last]).mean())))
    contrasts = [dict(comparison=label, window=window,
                     **paired_difference(estimates[left, window], estimates[right, window]))
                 for left, right, label in COMPARISONS if left in cells and right in cells
                 for window in ("training", "later", "whole")]
    pd.DataFrame(scores).to_csv(output / "scores.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "contrasts.csv", index=False)
    if digest(output / "nominees.json") != nomination_sha:
        raise ValueError("Frozen nominees changed")
    for path, expected in inputs.items():
        if digest(Path(path)) != expected:
            raise ValueError("Input changed: " + path)
    write(output / "completion.json", dict(status="complete", smoke=False,
          nomination_sha256=nomination_sha, protocol_sha256=digest(output / "protocol.json"),
          inputs_sha256=inputs, summary_sha256={name: digest(output / name) for name in ("scores.csv", "contrasts.csv")},
          oral_used=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)
