"""Compare precision or proposal candidates under the frozen common H policy.

Run --original-summary OLD --original-arm SEARCH --revised-summary NEW
--revised-arm SEARCH --subject ID --output NEW_DIRECTORY for ADD10 to LOG25.
Use POOL then SEARCH from one summary for a proposal comparison. Differences
include behavioral parameter re-estimation. Original scoring, complete-filter
and complete-path errors, trial boundaries and adaptive-work audits are reused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_0923_cohort_belief_comparison as common
from ..workflows.analysis import model_0923_finalization_states as states

ROOT = common.ROOT
load, digest, write = common.load, common.digest, common.write
JOINT = states.JOINT


def merge_bindings(target: dict, addition: dict) -> None:
    for name, sha in addition.items():
        if name in target and target[name] != sha:
            raise ValueError("Comparison binds different versions of the same input")
        target[name] = sha


def match_sources(old: dict, new: dict) -> None:
    """Permit the precision adapter while retaining every shared dependency."""
    banks = []
    for source in (old, new):
        bank = dict(source)
        if JOINT in bank and bank.pop(JOINT) != digest(ROOT / JOINT):
            raise ValueError("Joint precision adapter changed")
        banks.append(bank)
    if banks[0] != banks[1]:
        raise ValueError("Comparison changes a shared executing dependency")


def match_policy(old: dict, new: dict, old_protocol: dict, new_protocol: dict,
                 old_space: dict, new_space: dict) -> tuple[str, str]:
    """Allow only declared precision and scheduling differences in fit policy."""
    families, normalized, banks = [], [], []
    for frozen, protocol in ((old, old_protocol), (new, new_protocol)):
        settings = frozen["settings"]
        family = states.family_from_settings(settings)
        if (frozen.get("smoke") is not False or protocol.get("smoke") is not False
                or settings != protocol["settings"]
                or frozen["source_sha256"] != protocol["source_sha256"]
                or settings.get("belief_mechanism") != "ONLINE"
                or settings.get("belief_intervention") != common.cohort.driver.INTERVENTIONS["ONLINE"]
                or settings.get("scope") != "common_policy_development_belief_refit"
                or set(settings["arms"]) != {"POOL", "SEARCH"}
                or type(settings.get("parallel_budget")) is not int
                or not 1 <= settings["parallel_budget"] <= 128):
            raise ValueError("Require complete scientific common-ONLINE fitting policies")
        if ("finalization_precision_family" in settings
                and JOINT not in frozen["source_sha256"]):
            raise ValueError("Precision family lacks its executed joint adapter")
        normalized_settings = deepcopy(settings)
        for key in ("parallel_budget", "finalization_precision_family", "finalization_scope"):
            normalized_settings.pop(key, None)
        for arm in normalized_settings["arms"].values():
            arm.pop("precision")
            arm.pop("rate_scale")
        normalized.append(normalized_settings)
        families.append(family)
        entries = protocol["initial_banks"]
        bank = {(r["subject"], r["model"]): r["rows"] for r in entries}
        keys = {(s, a) for s in settings["subjects"] for a in settings["arms"]}
        if len(entries) != len(bank) or set(bank) != keys or any(not rows for rows in bank.values()):
            raise ValueError("Initial candidates are missing or duplicated")
        banks.append(bank)
    match_sources(old["source_sha256"], new["source_sha256"])
    if normalized[0] != normalized[1] or old_space != new_space or banks[0] != banks[1]:
        raise ValueError("Precision comparison changes supports, task, seeds, starts or search budgets")
    return tuple(families)


def contrast_kind(families: tuple[str, str], arms: tuple[str, str]) -> str:
    if any(arm not in ("POOL", "SEARCH") for arm in arms):
        raise ValueError("Unknown proposal arm")
    if arms[0] == arms[1] and families == ("ADD10", "LOG25"):
        return "precision_with_behavior_refit"
    if arms == ("POOL", "SEARCH") and families[0] == families[1]:
        return "proposal_with_behavior_refit"
    raise ValueError("Change exactly one declared mechanism: ADD10 to LOG25 or POOL to SEARCH")


def load_pair(directories: tuple[Path, Path], arms: tuple[str, str],
              subject: int) -> tuple[list, tuple[str, str], str, dict]:
    """Load complete summaries with executed H and precision provenance."""
    records, protocols, spaces, inputs = [], [], [], {}
    for directory in directories:
        manifest, frozen, tables, bound = common.retained.load_summary(directory)
        merge_bindings(inputs, bound)
        merge_bindings(inputs, common.bind_summary(directory, manifest, "ONLINE"))
        extra, _, precision_manifest = states.bind_precision_summary(directory, frozen["settings"])
        merge_bindings(inputs, extra)
        merge_bindings(inputs, {str(precision_manifest.resolve()): digest(precision_manifest)})
        fit = Path(manifest["fit"])
        for filename, bank in (("protocol.json", protocols), ("parameter_spaces.json", spaces)):
            path = fit / filename
            merge_bindings(inputs, {str(path.resolve()): digest(path)})
            bank.append(load(path))
        records.append((manifest, frozen, tables))
    families = match_policy(records[0][1], records[1][1], *protocols, *spaces)
    kind = contrast_kind(families, arms)
    if subject not in records[0][1]["settings"]["subjects"]:
        raise ValueError("Participant absent from frozen policy")
    return records, families, kind, inputs


def generic_columns(table: pd.DataFrame) -> pd.DataFrame:
    names = {}
    for name in table:
        for old, new in (("online", "original"), ("separate", "revised"),
                         ("reset", "original"), ("retain", "revised")):
            if name == old or name.startswith(old + "_"):
                names[name] = new + name[len(old):]
                break
    return table.rename(columns=names)


def compare(original: Path, revised: Path, output: Path, subject: int,
            original_arm: str, revised_arm: str) -> None:
    if output.exists():
        raise FileExistsError("Use a new finalization comparison directory")
    arms = (original_arm, revised_arm)
    records, families, kind, inputs = load_pair((original, revised), arms, subject)
    selected, saved, audits = [], [], []
    for (manifest, frozen, tables), arm in zip(records, arms):
        settings = frozen["settings"]
        keys = {(s, a) for s in settings["subjects"] for a in settings["arms"]}
        selected.append(common.shared.load_bank(manifest, frozen, keys, settings)[subject, arm])
        saved.append({name: frame.loc[frame.subject.eq(subject) & frame.model.eq(arm)].copy()
                      for name, frame in tables.items() if name in
                      ("scores", "curves", "parameters", "paths", "budgets")})
        audit, bound = common.audit_exchange(Path(manifest["fit"]), [subject])
        merge_bindings(inputs, bound)
        audits.append(audit.loc[audit.model.eq(arm)].copy())
    common.retained.match_task(*selected)
    conditional, _, row = selected[0]
    cut, length = row["training_trials"], len(conditional["choices"])
    if not 0 < cut < length:
        raise ValueError("Require nonempty original training and later windows")
    contrasts = []
    for phase, first, last in (("training", 0, cut), ("test", cut, length), ("whole", 0, length)):
        estimates = []
        for (c, a, _), tables in zip(selected, saved):
            estimate = common.retained.window_estimate(c, a, first, last, settings["features"]["block_width"])
            selected_score = tables["scores"].loc[tables["scores"].phase.eq(phase)]
            if len(selected_score) != 1:
                raise ValueError("Missing unique saved phase score")
            for metric, value in estimate["metrics"].items():
                np.testing.assert_allclose(value, selected_score.iloc[0][metric], rtol=0, atol=1e-12)
            estimates.append(estimate)
        contrasts.append(dict(subject=subject, phase=phase, trials=last-first,
                              **common.paired_difference(estimates[1], estimates[0])))
    feedback = np.array([common.category_learning_feedback(int(choice), int(category), condition=row["condition"])
                         for choice, category in zip(conditional["choices"], conditional["categories"])])
    feedback_scores, trials = common.shared.feedback_scores(selected[0][0], selected[1][0], feedback, cut)
    paths = common.shared.path_contrasts(saved[0]["paths"], saved[1]["paths"])
    # Shared budget mathematics expects a common cell label. Original proposal
    # labels remain explicit in every output; source tables are kept unchanged.
    budget_args = [x["budgets"].assign(model="paired") for x in saved]
    budget_args += [x.assign(model="paired") for x in audits]
    budgets = common.budget_contrasts(*budget_args).drop(columns="model")
    tables = {name: pd.concat([bank[name].assign(variant=role, precision_family=family)
                              for bank, role, family in zip(saved, ("original", "revised"), families)],
                             ignore_index=True)
              for name in ("scores", "curves", "parameters", "paths")}
    tables.update(contrasts=pd.DataFrame(contrasts), feedback_scores=generic_columns(feedback_scores),
                  trials=generic_columns(trials), path_contrasts=generic_columns(paths),
                  budgets=generic_columns(budgets),
                  exchange_candidates=pd.concat([a.assign(variant=v) for a, v in
                                                  zip(audits, ("original", "revised"))], ignore_index=True))
    labels = dict(subject=subject, original_arm=original_arm, revised_arm=revised_arm,
                  original_family=families[0], revised_family=families[1])
    for name in tables:
        tables[name] = tables[name].assign(**labels)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Comparison input changed during scoring")
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    sources = [Path(__file__), Path(states.__file__), Path(common.__file__),
               Path(common.retained.__file__), Path(common.shared.__file__)]
    (output / "source").mkdir()
    for file in sources:
        shutil.copy2(file, output / "source" / file.name)
    write(output / "manifest.json", dict(status="complete", **labels, contrast_kind=kind,
        original_summary=str(original.resolve()), revised_summary=str(revised.resolve()),
        inputs_sha256=inputs, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        tables_sha256={name + ".csv": digest(output / (name + ".csv")) for name in tables},
        actual_search_compute_equal=bool(budgets.actual_compute_equal.all()),
        contrast="revised minus original; mechanism and behavior-selected parameters jointly change",
        data_use="Original behavior prefix; later choices are development diagnostics; oral excluded",
        uncertainty="Paired complete-filter/path simulation errors at frozen parameters"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-summary", "revised-summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("original-arm", "revised-arm"):
        parser.add_argument("--" + name, choices=("POOL", "SEARCH"), required=True)
    parser.add_argument("--subject", type=int, required=True)
    args = parser.parse_args()
    compare(args.original_summary, args.revised_summary, args.output, args.subject,
            args.original_arm, args.revised_arm)
