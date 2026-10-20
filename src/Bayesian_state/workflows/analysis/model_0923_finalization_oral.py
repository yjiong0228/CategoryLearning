"""Pair fixed oral readings for precision and proposal comparisons.

Use --original OLD_ALIGNMENT --original-arm SEARCH --revised NEW_ALIGNMENT
--revised-arm SEARCH --subject ID --output NEW_DIRECTORY for ADD10 to LOG25.
POOL to SEARCH uses the two arms of a common family. Existing joint-mass
scores and complete-filter errors are reused. Each input must reproduce its
saved scores and bind its exact state replay to behavior-selected parameters.
Run midpoint and relative readings separately; neither reading selects parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ...evaluation import model_0923_finalization_comparison as behavior
from . import model_0923_cohort_belief_oral as cohort_oral
from . import model_0923_finalization_states as states
from . import model_0923_online_evidence_refit_oral as refit
from . import model_0923_oral_feedback_strata as strata
from . import model_0923_oral_parameter_comparison as comparison

ROOT = behavior.ROOT
load, digest, write = behavior.load, behavior.digest, behavior.write


def match_identities(before: dict, after: dict, families: tuple[str, str]) -> None:
    """Keep task, readout, coding and H constant across behavior refits."""
    for key in ("scientific", "archived_pf_sha256", "task_sha256", "oral_manifest_sha256",
                "mechanism", "structural_intervention", "parameter_estimation_scope",
                "parameter_candidate_policy"):
        if before[key] != after[key]:
            raise ValueError("Oral comparison changes task, reports, readout, H or parameter selection")
    if (before["parameter_estimation_scope"] != "prefix"
            or before["mechanism"].get("online_belief") is not True
            or before["mechanism"].get("retain_learned_precision") is not True):
        raise ValueError("Require behavior-prefix ONLINE parameters and retained precision")
    definitions = []
    for identity, family in zip((before, after), families):
        arm = dict(identity["arm_settings"])
        if (arm["precision"], arm["rate_scale"], arm["upper"]) != states.FAMILIES[family]:
            raise ValueError("State precision differs from its behavioral fit")
        arm.pop("precision")
        arm.pop("rate_scale")
        definitions.append(arm)
    if definitions[0] != definitions[1]:
        raise ValueError("Oral comparison changed other arm settings")
    behavior.match_sources(before["model_sources"], after["model_sources"])


def normalized_group(scored: tuple, arm: str) -> tuple:
    """Copy the proposal label solely for shared paired-score identity checks."""
    trials, rows, estimates = scored
    if set(trials.arm.unique()) != {arm} or any(row["arm"] != arm for row in rows):
        raise ValueError("Scored group contains another proposal arm")
    return trials.assign(arm="paired"), [dict(row, arm="paired") for row in rows], estimates


def bind_states(manifest: dict, identity: dict, record: tuple, subject: int, arm: str) -> dict:
    generic, frozen, _ = record
    fit, summary = Path(manifest["cohort_fit"]), Path(manifest["cohort_summary"])
    if fit.resolve() != Path(generic["fit"]).resolve():
        raise ValueError("Oral states name a different fitted parameter bank")
    protocol, done = (load(fit / name) for name in ("protocol.json", "completion.json"))
    refit.validate_nomination(identity, manifest, frozen, protocol, done,
                              digest(fit / "nominees.json"), subject, arm)
    if (manifest["cohort_manifest_sha256"] != digest(summary / "cohort_manifest.json")
            or manifest["fixed_coordinates"] != frozen["settings"]["fixed_coordinates"]
            or any(identity["candidate"]["hyperparams"].get(k) != v
                   for k, v in frozen["settings"]["fixed_coordinates"].items())
            or identity["arm_settings"] != frozen["settings"]["arms"][arm]):
        raise ValueError("Oral state settings differ from the nominated behavior fit")
    source = Path(manifest["source"])
    state_protocol = load(source / "protocol.json")
    if (ROOT / state_protocol["settings"]["summary"]).resolve() != summary.resolve():
        raise ValueError("Oral state replay uses another summary")
    if "finalization_precision_family" in frozen["settings"]:
        states.check_precision_context(state_protocol)
        context, _ = states.estimation_context(state_protocol)
        if any(manifest.get(key) != value for key, value in context.items()):
            raise ValueError("Alignment lost its joint precision provenance")
    with states.precision_observer():
        files, metadata = states.source_jobs(state_protocol["settings"],
                                            ROOT / state_protocol["settings"]["source"], False)
    for key in ("cohort_fit", "cohort_summary", "cohort_manifest_sha256", "refit_nomination_sha256",
                "belief_mechanism", "belief_intervention", "selected_parameters"):
        if metadata[key] != manifest.get(key):
            raise ValueError("Oral states and executed evaluation select different parameters")
    inputs = dict(metadata["refit_replay_inputs"])
    behavior.merge_bindings(inputs, cohort_oral.validate_states(manifest, subject, arm, "ONLINE", files))
    return inputs


def run(original: Path, revised: Path, output: Path, subject: int,
        original_arm: str, revised_arm: str) -> None:
    if output.exists():
        raise FileExistsError("Use a new finalization oral-comparison directory")
    directories, arms = (original, revised), (original_arm, revised_arm)
    manifests = [load(p / "manifest.json") for p in directories]
    records, families, kind, inputs = behavior.load_pair(
        tuple(Path(m["cohort_summary"]) for m in manifests), arms, subject)
    groups, identities, feedback_banks = [], [], []
    for directory, arm, manifest, record in zip(directories, arms, manifests, records):
        scored, identity, bound = comparison.load_group(directory, subject, arm)
        behavior.merge_bindings(inputs, bound)
        behavior.merge_bindings(inputs, bind_states(manifest, identity, record, subject, arm))
        bank, bound = strata.load_bank(directory, subject, arm, scored)
        behavior.merge_bindings(inputs, bound)
        feedback_banks.append(dict(bank, trials=bank["trials"].assign(arm="paired")))
        groups.append(normalized_group(scored, arm))
        identities.append(identity)
    match_identities(*identities, families)
    scores, trials = comparison.paired_scores(*groups, seed_relation="paired")
    feedback, deleted = refit.feedback_comparison(*feedback_banks, subject, "paired")
    labels = dict(original_arm=original_arm, revised_arm=revised_arm,
                  original_family=families[0], revised_family=families[1])
    tables = {name: table.drop(columns="arm").assign(**labels) for name, table in
              (("scores", scores), ("trials", trials), ("feedback_scores", feedback))}
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Oral comparison input changed during scoring")
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    deleted = {key.replace("retained_", "original_", 1) if key.startswith("retained_")
               else key.replace("online_", "revised_", 1): value for key, value in deleted.items()}
    np.savez_compressed(output / "feedback_deleted_filters.npz", **deleted)
    heading = f"被试{subject}：{original_arm}/{families[0]} → {revised_arm}/{families[1]}"
    note = ("参数来自原训练段行为。口述评价保留同一编码、题目、时间边界和滤波种子。"
            "差值为后者减前者，包含机制与行为重估参数的共同作用；误差反映固定参数下的模拟波动。"
            "支持、冲突和未确定质量共同描述带噪声的口述证据。")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>' + heading + "</title><h1>" + heading +
        "</h1><p>" + note + "</p>" + tables["scores"].to_html(index=False), encoding="utf-8")
    sources = [Path(module.__file__) for module in
               (behavior, states, cohort_oral, refit, strata, comparison, comparison.alignment)]
    sources.append(Path(__file__))
    (output / "source").mkdir()
    for file in sources:
        shutil.copy2(file, output / "source" / file.name)
    artifacts = [name + ".csv" for name in tables] + ["feedback_deleted_filters.npz", "index.html"]
    write(output / "manifest.json", dict(status="complete", subject=subject, **labels,
        contrast_kind=kind, original=str(original.resolve()), revised=str(revised.resolve()),
        inputs_sha256=inputs, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        artifacts={name: digest(output / name) for name in artifacts},
        identities=identities, contrast="revised minus original",
        data_use="Behavior-selected parameters; fixed oral reading used for evaluation",
        readout="after_choice_before_current_feedback; identical reports and coverage",
        uncertainty="Paired complete-filter deletion errors at frozen behavior-selected parameters"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "revised", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("original-arm", "revised-arm"):
        parser.add_argument("--" + name, choices=("POOL", "SEARCH"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    args = parser.parse_args()
    run(args.original, args.revised, args.output, args.subject, args.original_arm, args.revised_arm)
