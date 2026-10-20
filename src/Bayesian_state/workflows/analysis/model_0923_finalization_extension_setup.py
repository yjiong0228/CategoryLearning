"""Prepare geometry diagnostics and neutral starting banks for the extension.

Run --config YAML --output NEW_DIRECTORY from the repository root. The three
participants come from the completed eligibility registration. Existing task
arrays and boundaries remain unchanged. Starting points depend only on the
common parameter support and declared seed. The existing geometry audit selects
a diagnostic target using training stimuli/categories, then describes its fit
to the later task schedule. No choice likelihood or oral alignment is scored.
Formal parameter fitting still requires a frozen recommended version.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml

from . import model_0923_cohort_belief_refit as cohort
from . import model_0923_h_rule_diagnosis as geometry
from . import model_0923_joint_parameter_recovery as recovery
from .model_0923_h_validation import digest, read, write, training_boundary
from ...utils.seeding import stable_seed

ROOT = cohort.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
KEYS = {"data", "eligibility", "policy_reference_fit", "geometry_beta", "parallel_budget"}


def validate_registration(eligibility: dict, prepared: dict) -> list[dict]:
    """Preserve the registered sample and its original chronological boundaries."""
    selected = eligibility.get("selected", [])
    if (eligibility.get("status") != "complete" or eligibility.get("model_scores_used") is not False
            or eligibility.get("behavioral_accuracy_used") is not False
            or eligibility.get("previous_development_influence") != "unknown"
            or eligibility.get("recommendation_freeze_required") is not True
            or prepared.get("status") != "complete" or prepared.get("smoke") is not False
            or [(r["subject"], r["condition"]) for r in selected] != [(110, 1), (209, 2), (309, 3)]
            or prepared["subjects"] != [r["subject"] for r in selected]):
        raise ValueError("Use the registered three-person extension and retain its unknown prior influence")
    for row in selected:
        if (row["training_trials"] != training_boundary(row["trials"])
                or row["training_trials"] + row["later_trials"] != row["trials"]):
            raise ValueError("Registered chronological boundary changed")
    return selected


def neutral_banks(settings: dict, space: dict, subjects: list[int]) -> list[dict]:
    """Generate one shared bank per participant before selecting the precision family."""
    rows = []
    for subject in subjects:
        seed = stable_seed(dict(base=settings["seed"], subject=subject,
                                role="finalization_extension_independent_starts"))
        bank = recovery.independent_starts(space, settings["initial_candidates"], seed)
        rows.append(dict(subject=subject, seed=seed, required_reference_id=bank[0]["id"], rows=bank))
    return rows


def setup(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new extension setup directory")
    own = yaml.safe_load(config.read_text())
    if (set(own) != KEYS or own["geometry_beta"] != 100.
            or type(own["parallel_budget"]) is not int or not 1 <= own["parallel_budget"] <= 3):
        raise ValueError("Declare the prepared sources, the existing geometry beta=100 and at most three workers")
    inputs = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f"Extension preparation input changed: {path}")
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError("Input changed during setup")
        inputs[str(path)] = sha
        return path

    bind(config)
    eligibility_path = ROOT / own["eligibility"] / "completion.json"
    preparation_path = ROOT / own["data"] / "preparation_manifest.json"
    eligibility, prepared = read(bind(eligibility_path)), read(bind(preparation_path))
    for name, sha in prepared["artifacts_sha256"].items():
        bind(preparation_path.parent / name, sha)
    for name, sha in eligibility["inputs_sha256"].items():
        bind(Path(name), sha)
    for name, sha in eligibility["artifacts_sha256"].items():
        bind(eligibility_path.parent / name, sha)
    selected = validate_registration(eligibility, prepared)
    policy = ROOT / own["policy_reference_fit"]
    protocol, frozen, done = [read(bind(policy / name)) for name in
                              ("protocol.json", "nominees.json", "completion.json")]
    if (done["status"] != "complete" or done["smoke"] or protocol["smoke"] or frozen["smoke"]
            or done["test_choices_used"] is not False or frozen["status"] != "frozen_before_test"
            or done["nomination_sha256"] != digest(policy / "nominees.json")
            or protocol["settings"] != frozen["settings"]
            or protocol["source_sha256"] != frozen["source_sha256"]):
        raise ValueError("Require the completed common training-only policy")
    settings = protocol["settings"]
    if (settings["belief_intervention"] != cohort.INTERVENTIONS["ONLINE"]
            or settings["structural_intervention"] != "retained_rule_precision"
            or settings["initial_candidates"] != 40 or list(settings["arms"]) != ["POOL", "SEARCH"]):
        raise ValueError("The extension must inherit the declared common parameter search policy")
    modules = cohort.helpers()
    _, online, _, recalled = modules
    spaces = {arm: online.fixed_space(recalled.base.model_space(settings, arm), settings["fixed_coordinates"])
              for arm in settings["arms"]}
    if spaces["POOL"] != spaces["SEARCH"] or spaces != read(bind(policy / "parameter_spaces.json")):
        raise ValueError("Common parameter supports changed")
    bind(ROOT / settings["parameter_support"])
    for name, sha in protocol["source_sha256"].items():
        bind(ROOT / name, sha)
    sources = [Path(__file__), Path(geometry.__file__), Path(recovery.__file__),
               Path(cohort.__file__), ROOT / geometry.BASE]
    hashes = {str(p.relative_to(ROOT)): digest(bind(p)) for p in sources}
    banks = neutral_banks(settings, spaces["SEARCH"], [r["subject"] for r in selected])
    runtime = ROOT / prepared["settings"]["source_fit"] / "source"
    jobs = []
    for row, bank in zip(selected, banks):
        folder = ROOT / own["data"] / f"subject_{row['subject']}"
        meta = read(bind(folder / "prepared.json"))
        expected = {k: row[k] for k in ("subject", "condition", "trials", "training_trials")}
        if (any(meta[k] != value for k, value in expected.items())
                or meta["test_trials"] != row["later_trials"]):
            raise ValueError("Prepared participant differs from its registration")
        if read(bind(folder / "engine.json"))["oral_readout"] is not None:
            raise ValueError("Behavioral setup must retain disabled oral input")
        for name, sha in meta["frozen_cognitive_source_sha256"].items():
            bind(runtime / name, sha)
        jobs.append(dict(phase="audit", name=f"extension_audit_{row['subject']}", subject=row["subject"],
            runtime=str(runtime.resolve()), folder=str(folder.resolve()), mechanism=deepcopy(settings["mechanism"]),
            candidate=deepcopy(bank["rows"][0]), geometry_beta=own["geometry_beta"],
            target=str((output / f"audit/subject_{row['subject']}").resolve()), source_sha256=hashes))
    output.mkdir(parents=True)
    write(output / "parameter_spaces.json", spaces)
    write(output / "initial_banks.json", dict(initialization="support_midpoint_capacity_rays_stratified",
        training_choices_used=False, later_choices_used=False, oral_contents_used=False,
        count_per_subject=40, banks=banks))
    write(output / "protocol.json", dict(status="prepared", settings=own, subjects=selected, jobs=jobs,
        inputs_sha256=inputs.copy(), source_sha256=hashes, model_fitting_started=False,
        recommendation_freeze_required=True, previous_development_influence="unknown",
        geometry_selection="Training physical stimuli and task categories only; later geometry is diagnostic",
        seed_policy="Original base seed plus participant and extension-start role"))
    geometry.execute(jobs, output, own["parallel_budget"])
    results = []
    for row in selected:
        directory = output / f"audit/subject_{row['subject']}"
        audit = read(directory / "audit.json")
        if (audit["subject"] != row["subject"] or audit["condition"] != row["condition"]
                or audit["training_trials"] != row["training_trials"] or audit["trials"] != row["trials"]
                or audit["catalog_sha256"] != digest(directory / "catalog.csv")):
            raise ValueError("Geometry audit changed task identity or catalogue")
        results.append(dict(**row, target_rule=audit["target"],
            training_perfect_rules=len(audit["training_perfect_rules"]),
            full_perfect_rules=len(audit["full_perfect_rules"]),
            target_later_geometry_accuracy=audit["target_test_accuracy"]))
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Extension input changed during geometry execution")
    pd.DataFrame(results).to_csv(output / "subjects.csv", index=False)
    shutil.copy2(__file__, output / "setup_source.py")
    write(output / "completion.json", dict(status="complete", subjects=[r["subject"] for r in selected],
        parameter_fitting_started=False, recommendation_freeze_required=True,
        previous_development_influence="unknown", geometry_jobs=len(jobs), neutral_points=len(banks)*40,
        variable_coordinates=13, fixed_coordinates=settings["fixed_coordinates"],
        inputs_sha256=inputs, source_sha256=hashes,
        artifacts_sha256={str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()},
        versions=dict(python=sys.version, **{k: importlib.metadata.version(k) for k in ("numpy", "pandas", "PyYAML")})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    setup(args.config, args.output)
