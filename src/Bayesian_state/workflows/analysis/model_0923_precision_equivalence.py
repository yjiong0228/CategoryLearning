"""Verify exact ADD10/LOG25 equivalence at identical zero-learning-rate points.

Run --config YAML --output NEW_DIRECTORY. Both complete behavior-prefix fits
must select identical parameters with zero precision learning. Their original
eight-seed confirmation arrays must agree exactly. One complete LOG25 path per
proposal arm is then regenerated with a previously evaluated seed and compared
with every saved path array. Completed ADD10 behavior and both oral readings
can be reused for these particular points; the LOG25 fitting search remains
separately audited. Reuse is recorded explicitly, without creating a fictitious
LOG25 evaluation or state-replay directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from ...evaluation import model_0923_finalization_comparison as comparison
from ...evaluation.model_0923_refit_budget import audit_exchange
from . import model_0923_cohort_belief_oral as oral_states
from . import model_0923_finalization_joint as joint
from . import model_0923_oral_parameter_comparison as oral

ROOT = comparison.ROOT
load, digest, write = comparison.load, comparison.digest, comparison.write
RATE = "engine.modules.beta_mod.kwargs.learning_rate"
BETA = "engine.modules.beta_mod.kwargs.beta_init"


def check_points(original: dict, revised: dict) -> dict:
    """Restrict reuse to the same complete nominated points with zero updates."""
    banks = []
    for frozen in (original, revised):
        if frozen.get("status") != "frozen_before_test" or frozen.get("smoke") is not False:
            raise ValueError("Require full frozen behavioral fits")
        bank = {(r["subject"], r["model"]): r["candidate"] for r in frozen["nominees"]}
        expected = {(s, a) for s in frozen["settings"]["subjects"] for a in ("POOL", "SEARCH")}
        if len(bank) != len(frozen["nominees"]) or set(bank) != expected:
            raise ValueError("Missing or duplicated fitted point")
        banks.append(bank)
    if set(banks[0]) != set(banks[1]):
        raise ValueError("Reuse requires the same fitted participants")
    for key, before in banks[0].items():
        after = banks[1][key]
        if (before["id"] != after["id"] or before["hyperparams"] != after["hyperparams"]
                or before["hyperparams"][RATE] != 0.
                or not 0 < before["hyperparams"][BETA] <= 25.):
            raise ValueError("Reuse requires identical zero-rate parameters within the shared precision range")
    return banks[1]


def run(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new precision-equivalence directory")
    settings = yaml.safe_load(config.read_text())
    original_summary, revised_fit = (ROOT / settings[k] for k in ("original_summary", "revised_fit"))
    manifest, old, tables, inputs = comparison.common.retained.load_summary(original_summary)
    inputs = dict(inputs)
    comparison.merge_bindings(inputs, comparison.common.bind_summary(original_summary, manifest, "ONLINE"))

    def bind(path):
        path = Path(path).resolve()
        comparison.merge_bindings(inputs, {str(path): digest(path)})
        return path

    bind(config)
    for file in (Path(__file__), Path(comparison.__file__), Path(joint.__file__),
                 ROOT / "src/Bayesian_state/model/modules/precision_interventions.py"):
        bind(file)
    original_fit = Path(manifest["fit"])
    new, done = [load(bind(revised_fit / name)) for name in ("nominees.json", "completion.json")]
    if (done["status"] != "complete" or done["smoke"] or done["test_choices_used"]
            or done["nomination_sha256"] != digest(revised_fit / "nominees.json")):
        raise ValueError("Revised fitting did not freeze before evaluation")
    protocols = [load(bind(p / "protocol.json")) for p in (original_fit, revised_fit)]
    spaces = [load(bind(p / "parameter_spaces.json")) for p in (original_fit, revised_fit)]
    if comparison.match_policy(old, new, *protocols, *spaces) != ("ADD10", "LOG25"):
        raise ValueError("Require the declared common-H precision comparison")
    points = check_points(old, new)
    if len(new["settings"]["subjects"]) != 1:
        raise ValueError("Verify one registered participant per reuse record")
    subject = new["settings"]["subjects"][0]
    for name, sha in new["source_sha256"].items():
        if digest(bind(ROOT / name)) != sha:
            raise ValueError("Revised fitting implementation changed")
    declaration = load(bind(revised_fit / "finalization_joint_protocol.json"))
    if (declaration["precision_family"] != "LOG25" or declaration["scope"] != "development_comparison"
            or declaration["oral_used_for_parameter_selection"] is not False):
        raise ValueError("Revised fit changed precision family or data use")
    for name, sha in declaration["input_sha256"].items():
        if digest(bind(name)) != sha:
            raise ValueError("Revised fitting input changed")
    audit = comparison.common.cohort.audit_jobs(revised_fit, "ONLINE", bind)
    for file in (revised_fit / "jobs").glob("*.json"):
        joint.check_job(load(file))

    confirmation_checks = 0
    for row in old["nominees"]:
        key = row["subject"], row["model"]
        left, right = [Path(point["score_file"]) for point in (row["candidate"], points[key])]
        for path in (left, right):
            record = load(bind(path))
            if digest(bind(path.with_suffix(".npz"))) != record["predictions_sha256"]:
                raise ValueError("Confirmation arrays changed")
        with np.load(left.with_suffix(".npz")) as a, np.load(right.with_suffix(".npz")) as b:
            if set(a.files) != set(b.files):
                raise ValueError("Confirmation array fields differ")
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
                confirmation_checks += a[name].size

    # The precision adapter returns the original gradient when learning_rate==0.
    # Both modes set the same upper bound. All remaining cognitive dependencies,
    # parameters, stimuli and seeds were checked above. Regenerate complete paths
    # as an execution check of this pointwise equivalence.
    evaluation = Path(manifest["evaluation"])
    references, jobs = {}, []
    for arm in ("POOL", "SEARCH"):
        files = sorted((evaluation / "jobs").glob(f"AUTO_*_{arm}_*.json"))
        if not files:
            raise ValueError("Missing original complete autonomous path")
        original_job = load(bind(files[0]))
        target = Path(original_job["target"])
        references[arm] = (bind(target.with_suffix(".npz")), original_job["seeds"][0])
        if digest(target.with_suffix(".npz")) != load(bind(target.with_suffix(".json")))["predictions_sha256"]:
            raise ValueError("Original autonomous arrays changed")
        job = deepcopy(original_job)
        job.update(name=f"zero_rate_{arm}", target=str((output / "paths" / arm).resolve()),
                   candidate=points[subject, arm], seeds=original_job["seeds"][:1],
                   source_sha256=new["source_sha256"], arm_settings=new["settings"]["arms"][arm])
        jobs.append(job)
    modules = joint.registration.cohort.helpers()
    restored, bindings = joint.prepare_settings(
        {k: new["settings"][k] for k in joint.CONFIG_KEYS}, "ONLINE", modules)
    if restored != new["settings"]:
        raise ValueError("Revised fit no longer matches its frozen policy")
    comparison.merge_bindings(inputs, bindings)
    joint.configure(modules, restored, bindings)
    output.mkdir(parents=True)
    write(output / "protocol.json", dict(fit=str(revised_fit.resolve()), subject=subject,
        status="verified_before_path_check", confirmation_array_values=confirmation_checks,
        path_checks=len(jobs), workers=1, inputs_sha256=inputs.copy(),
        rationale="At zero learning rate the precision adapter uses the same gradient and shared bound"))
    modules[-1].base.execute(jobs, output, 1)
    path_checks = 0
    for file in sorted((output / "jobs").glob("*.json")):
        job = load(bind(file))
        joint.check_job(job)
        comparison.common.cohort.audit_job(file, "ONLINE", bind)
        original_file, seed = references[job["model"]]
        with np.load(original_file) as old_arrays, np.load(Path(job["target"]).with_suffix(".npz")) as new_arrays:
            if set(old_arrays.files) != set(new_arrays.files):
                raise ValueError("Autonomous array fields differ")
            index, = np.flatnonzero(old_arrays["seeds"] == seed)
            for name in old_arrays.files:
                expected = old_arrays[name] if name == "categories" else old_arrays[name][[index]]
                np.testing.assert_array_equal(new_arrays[name], expected)
                path_checks += expected.size

    reused_oral = {}
    for reading in ("midpoint", "relative"):
        directory = ROOT / settings["oral_" + reading]
        oral_manifest = load(bind(directory / "manifest.json"))
        if Path(oral_manifest["cohort_summary"]).resolve() != original_summary.resolve():
            raise ValueError("Oral evidence uses another original behavior fit")
        for arm in ("POOL", "SEARCH"):
            _, identity, bound = oral.load_group(directory, subject, arm)
            comparison.merge_bindings(inputs, bound)
            comparison.merge_bindings(inputs, oral_states.validate_states(oral_manifest, subject, arm, "ONLINE"))
            if identity["candidate"]["hyperparams"] != points[subject, arm]["hyperparams"]:
                raise ValueError("Oral states use another parameter point")
        reused_oral[reading] = str(directory.resolve())
    fit_budgets = []
    for fit, label in ((original_fit, "ADD10"), (revised_fit, "LOG25")):
        _, bound = audit_exchange(fit, [subject])
        comparison.merge_bindings(inputs, bound)
        rows = [row for stage in ("screen", "refine", "exchange", "selection", "confirmation")
                for row in comparison.common.cohort.separate.training_chunk_budgets(fit / stage)]
        if not rows:
            raise ValueError("Missing actual fitting work")
        fit_budgets.append(pd.DataFrame(rows).assign(precision_family=label))
    pd.concat(fit_budgets, ignore_index=True).to_csv(output / "actual_fitting_budgets.csv", index=False)
    tables["scores"].assign(precision_family="LOG25", evaluation_source="reused_exact_zero_rate_ADD10").to_csv(
        output / "reused_behavior_scores.csv", index=False)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Equivalence input changed during verification")
    shutil.copy2(__file__, output / "verification_source.py")
    write(output / "completion.json", dict(status="complete", subject=subject,
        equivalence="Identical selected points with zero precision learning; ADD10 and LOG25 execute the same model",
        original_summary=str(original_summary.resolve()), revised_fit=str(revised_fit.resolve()),
        reused_oral=reused_oral, reused_states=oral_manifest["source"],
        confirmation_array_values=confirmation_checks, autonomous_array_values=path_checks,
        regenerated_complete_paths=2, audited_revised_fit_jobs=audit["jobs"],
        precision_contrast="All paired behavior and oral differences and simulation errors are exactly zero",
        scope="These nominated points and task only; nonzero-rate precision models can differ",
        new_full_evaluation_executed=False, new_full_state_replay_executed=False,
        search_reestimated=True, inputs_sha256=inputs,
        artifacts_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
