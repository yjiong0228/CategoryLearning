"""Retain all high-particle alternatives for the registered finalization cells.

Run --source EVIDENCE --output NEW_DIRECTORY [--allow-partial]. Read the
completed evidence matrix and its actual fits, including separately fitted
zero-rate precision aliases. Score selection and confirmation separately using
their original training trials and whole-filter Monte Carlo errors. Relative
point likelihoods describe the saved adaptive candidate bank, not a parameter
posterior, confidence interval or proof of identifiability. No fitting occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_0923_episodic_refit as online
from . import model_0923_joint_parameter_recovery as recovery
from .model_0923_finalization_evidence import CANDIDATES
from .model_0923_h_validation import digest, load, nll
from ..workflows.analysis.model_0923_h_validation import write

ROOT = Path(__file__).resolve().parents[3]
H = "engine.modules.hypo_transitions_mod.kwargs."
M = "engine.modules.memory_mod.kwargs."
B = "engine.modules.beta_mod.kwargs."
COORDINATES = {
    H + "search_controller.baseline_probability",
    H + "search_controller.error_gain",
    H + "search_controller.failure_decay",
    H + "search_controller.global_search",
    H + "search_controller.global_search_failure_gain",
    H + "max_capacity", H + "exit_rate", H + "local_structure_weight",
    H + "transfer_scale", H + "execution_inertia", H + "intervention.decay",
    M + "gamma", B + "beta_init", B + "learning_rate",
}


def check_scope(manifest: dict, cells: pd.DataFrame, allow_partial: bool) -> None:
    """Require every candidate for each declared participant; label partial runs."""
    subjects = sorted(manifest["subjects"])
    complete = subjects == [108, 205, 306]
    expected = {(s, c) for s in subjects for c in CANDIDATES}
    if (manifest.get("status") != "complete" or not subjects
            or not set(subjects) <= {108, 205, 306}
            or len(set(subjects)) != len(subjects)
            or manifest.get("registered_subjects") != [108, 205, 306]
            or manifest.get("complete_primary_matrix") != complete
            or (not complete and not allow_partial)
            or cells.duplicated(["subject", "candidate"]).any()
            or set(zip(cells.subject, cells.candidate)) != expected):
        raise ValueError("Require the registered matrix or an explicit complete-participant subset")
    for row in cells.itertuples():
        if (row.precision_family, row.arm) != CANDIDATES[row.candidate]:
            raise ValueError("Candidate precision/proposal identity changed")


def rank_bank(banks: dict[str, dict]) -> list[dict]:
    """Compare candidate points within one original numerical stage only."""
    if not banks:
        raise ValueError("An empty stage has no parameter evidence")
    aligned = {}
    for identity, bank in banks.items():
        seeds = np.asarray(bank["seeds"])
        probability = np.asarray(bank["observed_probability"])
        if (seeds.ndim != 1 or probability.ndim != 2 or len(seeds) < 2
                or probability.shape[0] != len(seeds) or probability.shape[1] < 1
                or len(set(seeds.tolist())) != len(seeds)
                or not np.isfinite(probability).all()
                or np.any((probability <= 0) | (probability > 1))):
            raise ValueError("Need valid independent complete-filter probabilities")
        order = np.argsort(seeds)
        aligned[identity] = dict(seeds=seeds[order], observed_probability=probability[order])
    best = min(aligned, key=lambda key: (nll(aligned[key]["observed_probability"]), key))
    rows = []
    for identity, bank in sorted(aligned.items()):
        score = recovery.paired_scores(bank, aligned[best])
        delta = score["trials"] * score["estimate_minus_truth_nll"]
        rows.append(dict(point=identity, stage_best=best, training_nll=score["estimate_nll"],
            delta_total_nll=delta, delta_total_nll_mcse=score["trials"] * score["difference_mc_se"],
            relative_point_likelihood=float(np.exp(-delta)), trials=score["trials"], filters=score["filters"]))
    return rows


def point_channels(frozen: dict, subject: int, arm: str) -> dict[str, str]:
    """Use the existing ONLINE closure checks without equating openness with recovery."""
    checked = deepcopy(frozen)
    checked["settings"]["models"] = {a: checked["settings"]["mechanism"] for a in checked["settings"]["arms"]}
    closed = online.refit_inactive_channels(checked)
    return {r.parameter: r.reason for r in closed.itertuples() if r.subject == subject and r.model == arm}


def collect(source: Path, output: Path, allow_partial: bool = False) -> None:
    if output.exists():
        raise FileExistsError("Use a new parameter-support directory")
    bindings = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        path = path.resolve(); sha = digest(path)
        if ((expected is not None and sha != expected)
                or (str(path) in bindings and sha != bindings[str(path)])):
            raise ValueError(f"Changed parameter-evidence input: {path}")
        bindings[str(path)] = sha
        return path

    manifest = load(bind(source / "manifest.json"))
    for name, sha in manifest["artifacts_sha256"].items():
        bind(source / name, sha)
    for name, sha in manifest["inputs_sha256"].items():
        bind(Path(name), sha)
    for file in (Path(__file__), Path(online.__file__), Path(recovery.__file__)):
        bind(file)
    cells = pd.read_csv(source / "cells.csv")
    check_scope(manifest, cells, allow_partial)
    saved_parameters = pd.read_csv(source / "parameters.csv")
    points, coordinates, channels, groups = [], [], [], []
    for cell in cells.itertuples():
        fit = Path(cell.actual_fit)
        frozen = load(bind(fit / "nominees.json"))
        done = load(bind(fit / "completion.json"))
        protocol = load(bind(fit / "protocol.json"))
        settings = frozen["settings"]
        if (done["status"] != "complete" or done["smoke"] or frozen["smoke"]
                or done["test_choices_used"] or frozen["status"] != "frozen_before_test"
                or digest(fit / "nominees.json") != done["nomination_sha256"]
                or protocol["settings"] != settings or settings["belief_mechanism"] != "ONLINE"
                or settings["belief_intervention"] != "online_retention_evidence"):
            raise ValueError("Require the completed ONLINE behavior-prefix fit")
        for name, sha in frozen["source_sha256"].items():
            bind(ROOT / name, sha)
        selected = [r for r in frozen["nominees"] if (r["subject"], r["model"]) == (cell.subject, cell.arm)]
        if len(selected) != 1:
            raise ValueError("Missing unique fitted cell")
        selected = selected[0]; nominee = selected["candidate"]; reference = selected["reference_id"]
        values = nominee["hyperparams"]
        saved = saved_parameters[(saved_parameters.subject == cell.subject) & (saved_parameters.candidate == cell.candidate)]
        if (nominee["id"] != cell.candidate_id or set(values) != COORDINATES
                or saved.parameter.duplicated().any() or dict(zip(saved.parameter, saved.value)) != values):
            raise ValueError("The fourteen fitted coordinates differ from the evidence matrix")
        labels = dict(subject=cell.subject, candidate=cell.candidate, arm=cell.arm,
                      precision_family=cell.precision_family, nominee=nominee["id"])
        closed = point_channels(frozen, cell.subject, cell.arm)
        for parameter, value in values.items():
            fixed = parameter in settings["fixed_coordinates"]
            channels.append(dict(**labels, parameter=parameter, value=value, configured_fixed=fixed,
                exact_closure_detected=parameter in closed, closure_reason=closed.get(parameter, ""),
                interpretation_status="固定或延后解释" if fixed or parameter in closed else "待结合联合恢复与过程证据分级",
                scope="当前参数点的既有精确通道检查；开放通道的可识别性由恢复和过程证据另行判断"))
        for stage in ("selection", "confirmation"):
            completion = load(bind(fit / f"{stage}_completion.json"))
            found = [g for g in completion["groups"] if (g["subject"], g["model"]) == (cell.subject, cell.arm)]
            if completion["status"] != "complete" or len(found) != 1:
                raise ValueError("Incomplete high-particle stage")
            rows = found[0]["rows"]
            if (len({r["id"] for r in rows}) != len(rows)
                    or not {nominee["id"], reference} <= {r["id"] for r in rows}):
                raise ValueError("Nominee/reference absent or candidate duplicated in stage")
            banks = {}; records = {}
            for row in rows:
                file = bind(Path(row["score_file"]))
                record = load(file); budget = settings["budgets"][stage]
                if (record["candidate"]["hyperparams"] != row["hyperparams"]
                        or record["candidate"]["id"] != row["id"]
                        or (record["subject"], record["model"]) != (cell.subject, cell.arm)
                        or record["trials"] != cell.training_trials
                        or record["particles"] != budget["particles"]
                        or len(record["seeds"]) != budget["seeds"]
                        or set(row["hyperparams"]) != COORDINATES):
                    raise ValueError("Candidate identity, training boundary or numerical budget changed")
                with np.load(bind(file.with_suffix(".npz"), record["predictions_sha256"])) as arrays:
                    bank = {k: arrays[k].copy() for k in ("seeds", "observed_probability")}
                np.testing.assert_array_equal(bank["seeds"], record["seeds"])
                for expected in (row["mean_nll"], record["mean_nll"]):
                    np.testing.assert_allclose(nll(bank["observed_probability"]), expected, atol=1e-12, rtol=0)
                banks[row["id"]] = bank; records[row["id"]] = record
            ranked = rank_bank(banks)
            for row in ranked:
                identity = row["point"]
                points.append(dict(**labels, stage=stage, **row, particles=budget["particles"],
                    is_final_nominee=identity == nominee["id"], is_required_reference=identity == reference,
                    actual_fit=str(fit.resolve())))
                for parameter, value in records[identity]["candidate"]["hyperparams"].items():
                    coordinates.append(dict(**labels, stage=stage, point=identity, parameter=parameter,
                        value=value, nominee_value=values[parameter], differs_from_nominee=value != values[parameter]))
            ordered = sorted(ranked, key=lambda r: (r["training_nll"], r["point"]))
            nominated, = [r for r in ranked if r["point"] == nominee["id"]]
            if stage == "confirmation":
                np.testing.assert_allclose(nominated["delta_total_nll"], 0., rtol=0, atol=1e-10)
            groups.append(dict(**labels, stage=stage, points=len(rows), stage_best=ordered[0]["point"],
                second_point=ordered[1]["point"] if len(ordered) > 1 else "",
                second_relative_point_likelihood=ordered[1]["relative_point_likelihood"] if len(ordered) > 1 else np.nan))
    for file, sha in bindings.items():
        if digest(Path(file)) != sha:
            raise ValueError("Source changed during parameter-support collection")
    output.mkdir(parents=True)
    for name, rows in (("points", points), ("point_parameters", coordinates),
                       ("nominee_channels", channels), ("groups", groups)):
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    shutil.copy2(__file__, output / "collection_source.py")
    write(output / "manifest.json", dict(status="complete", source=str(source.resolve()),
        subjects=manifest["subjects"], complete_primary_matrix=manifest["complete_primary_matrix"],
        missing_subjects=manifest["missing_subjects"], cells=len(cells), coordinates_per_cell=14,
        points=len(points), inputs_sha256=bindings,
        artifacts_sha256={p.name: digest(p) for p in output.iterdir()},
        interpretation="Saved adaptive candidate point likelihoods; parameter posterior and global identifiability remain unestimated",
        stage_policy="Selection and confirmation retain their own seeds and integration budgets; all saved candidates retained",
        uncertainty="Paired whole-filter simulation error conditional on the training sequence; finalist selection uses these scores",
        recovery_evidence="Integrate original and extended joint/state recovery separately; no recovery or psychological grade inferred here"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    collect(args.source, args.output, args.allow_partial)
