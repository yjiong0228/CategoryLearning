"""Collect the three registered candidates under common behavioral/oral scores.

Run --config YAML --output NEW_DIRECTORY; --subjects ID ... collects a labelled
partial matrix while other participants are still running. This entry point
uses completed fits, evaluations and state replays. It performs no fitting or
simulation. A LOG25 cell can reuse ADD10 evidence only with the completed,
hash-bound zero-rate equivalence record. Actual fitting provenance stays separate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from . import model_0923_finalization_comparison as behavior
from .model_0923_h_validation import jackknife_se
from ..workflows.analysis import model_0923_finalization_oral as oral
from ..workflows.analysis import model_0923_precision_equivalence as equivalence

ROOT = behavior.ROOT
load, digest, write = behavior.load, behavior.digest, behavior.write
CANDIDATES = {"SEARCH_ADD10": ("ADD10", "SEARCH"),
              "SEARCH_LOG25": ("LOG25", "SEARCH"), "POOL_LOG25": ("LOG25", "POOL")}
CONTRASTS = {"precision": ("SEARCH_ADD10", "SEARCH_LOG25"),
             "proposal": ("POOL_LOG25", "SEARCH_LOG25")}
READINGS = ("midpoint", "relative")
PHASES = ("training", "test", "whole")


def phase_accuracy(conditional: dict, autonomous: dict, first: int, last: int) -> dict:
    """Use exact category correctness; partial feedback is not a correct answer."""
    if not 0 <= first < last <= len(conditional["choices"]):
        raise ValueError("Invalid chronological accuracy window")
    np.testing.assert_array_equal(conditional["categories"], autonomous["categories"])
    observed = (conditional["choices"][first:last] == conditional["categories"][first:last]).mean()
    paths = (autonomous["choices"][:, first:last] == autonomous["categories"][None, first:last]).mean(1)
    if len(paths) < 2 or not np.isfinite(paths).all():
        raise ValueError("Need replicated complete autonomous paths")
    return dict(observed_accuracy=float(observed), autonomous_accuracy=float(paths.mean()),
                autonomous_accuracy_mcse=float(paths.std(ddof=1) / np.sqrt(len(paths))),
                autonomous_accuracy_q05=float(np.quantile(paths, .05)),
                autonomous_accuracy_q95=float(np.quantile(paths, .95)),
                paths_reaching_observed_accuracy=float((paths >= observed).mean()))


def equal_subject_summary(contrasts: pd.DataFrame, expected_subjects: list[int]) -> pd.DataFrame:
    """Keep the average simulation error separate from between-person variation."""
    keys = ["contrast", "evidence", "reading", "window", "domain", "subset", "metric"]
    rows = []
    for key, group in contrasts.groupby(keys, dropna=False, sort=True):
        if (group.subject.duplicated().any()
                or set(group.subject) != set(expected_subjects)):
            raise ValueError("Every collected participant must contribute once to each contrast")
        delta, se = group.difference.to_numpy(float), group.mcse.to_numpy(float)
        valid = np.isfinite(delta) & np.isfinite(se)
        empty = (group.evidence.eq("oral") & group.get("reports", pd.Series(-1, index=group.index)).eq(0)
                 & np.isnan(delta) & np.isnan(se)).to_numpy()
        if not np.all(valid | empty) or np.any(se < 0):
            raise ValueError("Invalid paired effect or simulation error")
        # A zero-report stratum has no defined score. Keep the full participant
        # denominator and leave its aggregate undefined rather than replace it
        # with zero or silently average only the participants with coverage.
        complete = bool(valid.all())
        rows.append(dict(zip(keys, key), subjects=len(delta), contributing_subjects=int(valid.sum()),
            equal_subject_difference=float(delta.mean()) if complete else np.nan,
            equal_subject_mcse=float(np.linalg.norm(se) / len(se)) if complete else np.nan,
            between_subject_sd=float(delta.std(ddof=1)) if complete and len(delta) > 1 else np.nan,
            smallest_subject_difference=float(delta.min()) if complete else np.nan,
            largest_subject_difference=float(delta.max()) if complete else np.nan))
    return pd.DataFrame(rows)


def check_disjoint_subject_seeds(banks: dict[int, dict[str, set]]) -> None:
    """Independent complete simulation banks justify variance addition."""
    for kind in ("conditional", "autonomous"):
        seen = set()
        for subject in sorted(banks):
            seeds = banks[subject][kind]
            if not seeds or seen & seeds:
                raise ValueError("Equal-subject MCSE requires disjoint participant seed banks")
            seen.update(seeds)


def check_reuse(proof: dict, specification: dict, original: dict, revised: dict) -> None:
    """Require the explicit equivalence scope and exact nominated zero-rate points."""
    if (proof.get("status") != "complete"
            or proof.get("new_full_evaluation_executed") is not False
            or proof.get("new_full_state_replay_executed") is not False
            or proof.get("search_reestimated") is not True
            or proof.get("subject") not in revised["settings"]["subjects"]
            or Path(proof["original_summary"]).resolve() != (ROOT / specification["summary"]).resolve()
            or Path(proof["revised_fit"]).resolve() != (ROOT / specification["fit"]).resolve()
            or any(Path(proof["reused_oral"][r]).resolve() != (ROOT / specification["oral_" + r]).resolve()
                   for r in READINGS)):
        raise ValueError("Equivalence record does not cover the requested evidence")
    equivalence.check_points(original, revised)


class EvidenceReader:
    """Cache verified source bundles so exact aliases do not repeat state scoring."""

    def __init__(self) -> None:
        self.inputs: dict[str, str] = {}
        self.summaries: dict = {}
        self.alignments: dict = {}

    def bind(self, path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Evidence input changed: {path}")
        behavior.merge_bindings(self.inputs, {str(path): actual})
        return path

    def manifest(self, path: Path) -> dict:
        record = load(self.bind(path))
        if record["status"] != "complete":
            raise ValueError(f"Evidence is incomplete: {path}")
        for name, sha in record.get("inputs_sha256", {}).items():
            self.bind(Path(name), sha)
        for key in ("artifacts_sha256", "artifacts", "summary_sha256", "tables_sha256"):
            for name, sha in record.get(key, {}).items():
                self.bind(path.parent / name, sha)
        return record

    def summary(self, directory: Path) -> tuple:
        directory = directory.resolve()
        if directory not in self.summaries:
            manifest, frozen, tables, inputs = behavior.common.retained.load_summary(directory)
            behavior.merge_bindings(self.inputs, inputs)
            behavior.merge_bindings(self.inputs, behavior.common.bind_summary(directory, manifest, "ONLINE"))
            extra, _, precision_manifest = behavior.states.bind_precision_summary(directory, frozen["settings"])
            behavior.merge_bindings(self.inputs, extra)
            self.bind(precision_manifest)
            settings = frozen["settings"]
            keys = {(s, a) for s in settings["subjects"] for a in settings["arms"]}
            bank = behavior.common.shared.load_bank(manifest, frozen, keys, settings)
            self.summaries[directory] = (manifest, frozen, tables, bank)
        return self.summaries[directory]

    def alignment(self, directory: Path, subject: int, arm: str, record: tuple) -> tuple:
        key = (directory.resolve(), subject, arm)
        if key not in self.alignments:
            scored, identity, inputs = oral.comparison.load_group(directory, subject, arm)
            behavior.merge_bindings(self.inputs, inputs)
            manifest = load(self.bind(directory / "manifest.json"))
            behavior.merge_bindings(self.inputs, oral.bind_states(manifest, identity, record[:3], subject, arm))
            self.alignments[key] = scored, identity
        return self.alignments[key]

    def cell(self, subject: int, candidate: str, specification: dict) -> dict:
        family, arm = CANDIDATES[candidate]
        directory = ROOT / specification["summary"]
        record = self.summary(directory)
        manifest, source_frozen, tables, bank = record
        source_family = behavior.states.family_from_settings(source_frozen["settings"])
        actual_fit = ROOT / specification.get("fit", manifest["fit"])
        actual = load(self.bind(actual_fit / "nominees.json"))
        done = self.manifest(actual_fit / "completion.json")
        if (actual["status"] != "frozen_before_test" or actual["smoke"]
                or done["smoke"] or done["test_choices_used"]
                or done["nomination_sha256"] != digest(actual_fit / "nominees.json")
                or behavior.states.family_from_settings(actual["settings"]) != family):
            raise ValueError("Cell does not use the declared frozen behavioral fit")
        reuse = specification.get("equivalence")
        if reuse:
            if family != "LOG25" or source_family != "ADD10":
                raise ValueError("Only verified ADD10-to-LOG25 equivalence is supported")
            proof = self.manifest(ROOT / reuse / "completion.json")
            check_reuse(proof, specification, source_frozen, actual)
        elif actual_fit.resolve() != Path(manifest["fit"]).resolve() or family != source_family:
            raise ValueError("Summary and actual fit differ without a verified equivalence record")
        selected = [r for r in actual["nominees"] if (r["subject"], r["model"]) == (subject, arm)]
        if len(selected) != 1:
            raise ValueError("Missing unique nominated point")
        values = selected[0]["candidate"]["hyperparams"]
        parameters = tables["parameters"].query("subject == @subject and model == @arm")
        if (parameters.parameter.duplicated().any()
                or dict(zip(parameters.parameter, parameters.value)) != values):
            raise ValueError("Saved parameter coordinates differ from the fitted point")
        aligned = {r: self.alignment(ROOT / specification["oral_" + r], subject, arm, record) for r in READINGS}
        return dict(candidate=candidate, subject=subject, arm=arm, family=family,
                    source_family=source_family, source_summary=directory, actual_fit=actual_fit,
                    frozen=actual, selected=selected[0], record=record, bank=bank[subject, arm],
                    alignments=aligned, reuse=bool(reuse), equivalence=reuse or "")


def collect(config: Path, output: Path, subjects: list[int] | None = None) -> None:
    if output.exists():
        raise FileExistsError("Use a new evidence directory")
    settings = yaml.safe_load(config.read_text())
    registered = sorted(settings["subjects"])
    subjects = registered if subjects is None else sorted(subjects)
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(registered):
        raise ValueError("Collect a unique nonempty subset of the registered participants")
    reader = EvidenceReader()
    reader.bind(config)
    registration = load(reader.bind(ROOT / settings["registration"]))
    if settings["candidates"] != list(CANDIDATES) or registered != [108, 205, 306]:
        raise ValueError("The primary evidence matrix must keep the registered three-by-three scope")
    if (registration["status"] != "frozen_before_new_precision_fits"
            or registration["subjects"] != registered
            or registration["primary_candidates"] != list(CANDIDATES)
            or (registration["common_h"], registration["precision_storage"], registration["execution"])
            != ("ONLINE", "RETAIN", "MIX")):
        raise ValueError("Primary candidates must have a frozen registration")
    cells, scores, oral_scores, curves, parameters, paths, provenance = {}, [], [], [], [], [], []
    estimates, seed_banks, contrasts = {}, {}, []
    for subject in subjects:
        seed_banks[subject] = {k: set() for k in ("conditional", "autonomous")}
        for candidate, (family, arm) in CANDIDATES.items():
            cell = reader.cell(subject, candidate, settings["subjects"][subject][family])
            cells[subject, candidate] = cell
            c, a, row = cell["bank"]
            cut, length = row["training_trials"], len(c["choices"])
            seed_banks[subject]["conditional"].update(map(int, c["seeds"]))
            seed_banks[subject]["autonomous"].update(map(int, a["seeds"]))
            labels = dict(subject=subject, candidate=candidate, arm=arm, precision_family=family)
            provenance.append(dict(**labels, candidate_id=cell["selected"]["candidate"]["id"],
                actual_fit=str(cell["actual_fit"].resolve()), evaluation_summary=str(cell["source_summary"].resolve()),
                evaluation_source="verified_zero_rate_reuse" if cell["reuse"] else "executed",
                equivalence=cell["equivalence"], training_trials=cut, later_trials=length-cut,
                source_precision_family=cell["source_family"]))
            source_tables = cell["record"][2]
            saved = source_tables["scores"].query("subject == @subject and model == @arm")
            for phase, first, last in (("training", 0, cut), ("test", cut, length), ("whole", 0, length)):
                estimate = behavior.common.retained.window_estimate(c, a, first, last,
                    cell["frozen"]["settings"]["features"]["block_width"])
                old = saved.loc[saved.phase.eq(phase)]
                if len(old) != 1:
                    raise ValueError("Missing unique chronological score")
                for metric, value in estimate["metrics"].items():
                    np.testing.assert_allclose(value, old.iloc[0][metric], atol=1e-12, rtol=0)
                metric_errors = {m + "_mc_se": jackknife_se(v) for m, v in estimate["jackknife"].items()}
                for metric, value in metric_errors.items():
                    np.testing.assert_allclose(value, old.iloc[0][metric], atol=1e-12, rtol=0)
                estimates[subject, candidate, phase] = estimate
                scores.append(dict(**labels, phase=phase, trials=last-first, condition=row["condition"],
                    **estimate["metrics"], **metric_errors, **phase_accuracy(c, a, first, last)))
            for name, destination in (("curves", curves), ("parameters", parameters), ("paths", paths)):
                destination.append(source_tables[name].query("subject == @subject and model == @arm").assign(**labels))
            for reading, (scored, _) in cell["alignments"].items():
                oral_scores.append(pd.DataFrame(scored[1]).assign(candidate=candidate, reading=reading))
        for contrast, (before, after) in CONTRASTS.items():
            old, new = cells[subject, before], cells[subject, after]
            behavior.common.retained.match_task(old["bank"], new["bank"])
            fits = [cell["actual_fit"] for cell in (old, new)]
            protocols = [load(reader.bind(p / "protocol.json")) for p in fits]
            spaces = [load(reader.bind(p / "parameter_spaces.json")) for p in fits]
            families = behavior.match_policy(old["frozen"], new["frozen"], *protocols, *spaces)
            behavior.contrast_kind(families, (old["arm"], new["arm"]))
            for phase in PHASES:
                difference = behavior.common.paired_difference(estimates[subject, after, phase],
                                                               estimates[subject, before, phase])
                for metric in ("choice_nll", "energy", "crps"):
                    contrasts.append(dict(subject=subject, contrast=contrast, evidence="behavior",
                        reading="not_applicable", window=phase, domain="not_applicable", subset="all_trials",
                        metric=metric, difference=difference["delta_" + metric],
                        mcse=difference["delta_" + metric + "_mc_se"]))
            for reading in READINGS:
                groups, identities = zip(*(cell["alignments"][reading] for cell in (old, new)))
                oral.match_identities(*identities, (old["source_family"], new["source_family"]))
                paired, _ = oral.comparison.paired_scores(
                    *[oral.normalized_group(group, cell["arm"]) for group, cell in zip(groups, (old, new))])
                for row in paired.to_dict("records"):
                    for label in ("support", "conflict", "undetermined"):
                        contrasts.append(dict(subject=subject, contrast=contrast, evidence="oral", reading=reading,
                            window=row["window"], domain=row["domain"], subset=row["subset"], metric=label,
                            reports=row["reports"], window_trials=row["window_trials"],
                            difference=row[label + "_difference"], mcse=row[label + "_difference_mcse"]))
    check_disjoint_subject_seeds(seed_banks)
    tables = dict(cells=pd.DataFrame(provenance), behavior_scores=pd.DataFrame(scores),
        oral_scores=pd.concat(oral_scores, ignore_index=True), curves=pd.concat(curves, ignore_index=True),
        parameters=pd.concat(parameters, ignore_index=True), paths=pd.concat(paths, ignore_index=True),
        contrasts=pd.DataFrame(contrasts))
    tables["equal_subject_contrasts"] = equal_subject_summary(tables["contrasts"], subjects)
    # Every alias retains its actual fitting budget; evaluation reuse saves only
    # the second evaluation and state replay, not the completed LOG25 search.
    budget_rows = []
    for (subject, candidate), cell in cells.items():
        _, bound = behavior.common.audit_exchange(cell["actual_fit"], [subject])
        behavior.merge_bindings(reader.inputs, bound)
        for stage in ("screen", "refine", "exchange", "selection", "confirmation"):
            rows = behavior.common.cohort.separate.training_chunk_budgets(cell["actual_fit"] / stage)
            budget_rows.extend(dict(r, candidate=candidate) for r in rows if r["model"] == cell["arm"])
    tables["actual_fitting_budgets"] = pd.DataFrame(budget_rows)
    for file, sha in reader.inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError("Evidence changed during collection")
    output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    shutil.copy2(__file__, output / "collection_source.py")
    write(output / "manifest.json", dict(status="complete", subjects=subjects, registered_subjects=registered,
        complete_primary_matrix=subjects == registered, collected_cells=len(cells), registered_cells=9,
        missing_subjects=sorted(set(registered)-set(subjects)), inputs_sha256=reader.inputs,
        artifacts_sha256={p.name: digest(p) for p in output.iterdir()},
        uncertainty="Complete-filter/path simulation MCSE at fitted parameters; between-subject SD separately",
        aggregation="Participants receive equal weight; phase and oral coverage subsets stay separate",
        path_window="Raw paths.csv late_* fields refer to the final 128 trials of the complete sequence",
        accuracy="Exact choice == category, including in the partial-feedback condition",
        oral="Fixed noisy external evidence; support, conflict, undetermined and coverage retained",
        data_use="Behavior-prefix parameter selection; later choices are development diagnostics"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int)
    args = parser.parse_args()
    collect(args.config, args.output, args.subjects)
