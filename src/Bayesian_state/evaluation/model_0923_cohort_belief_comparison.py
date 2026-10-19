"""Compare completed ONLINE and SEPARATE cohort refits under one search policy.

Run --online-summary OLD --separate-summary NEW --output NEW_DIRECTORY after
both model_0923_cohort_belief_refit summaries complete. Both execution audits,
common initial banks and parameter supports are required. Existing score and
whole-replicate uncertainty functions retain every participant's original trial
boundary and paired seeds. Differences include mechanism and re-estimation.
Oral data do not enter this behavior comparison. Actual adaptive exchange work
is reported alongside equal search limits; no population inference is made.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_0923_cohort_belief_refit as cohort
from . import model_0923_reentry_refit as retained
from . import model_0923_separate_refit_comparison as shared
from .model_0923_estimation_scope import paired_difference
from .model_0923_h_validation import digest, load
from .model_0923_refit_budget import audit_exchange, budget_contrasts
from ..metrics.task import category_learning_feedback
from ..workflows.analysis.model_0923_h_validation import write

ROOT = cohort.ROOT
MECHANISMS = ("ONLINE", "SEPARATE")


def bind_summary(source: Path, manifest: dict, mechanism: str) -> dict[str, str]:
    """Bind the cohort audit to all actual workers and their executed updates."""
    extra_path, audit_path = source / "cohort_manifest.json", source / "execution_audit.json"
    extra, audits = load(extra_path), load(audit_path)
    if (mechanism not in MECHANISMS or extra.get("status") != "complete"
            or extra.get("belief_mechanism") != mechanism
            or extra.get("generic_manifest_sha256") != digest(source / "manifest.json")
            or extra.get("execution_audit_sha256") != digest(audit_path)
            or set(audits) != {"fit", "evaluation"}):
        raise ValueError("Cohort summary lacks the bound mechanism execution audit")
    inputs = dict(extra["inputs_sha256"])
    for path in (extra_path, audit_path):
        inputs[str(path.resolve())] = digest(path)
    for name, sha in extra["source_sha256"].items():
        inputs[str((ROOT / name).resolve())] = sha
    for path in (Path(cohort.__file__), Path(cohort.driver.__file__)):
        if extra["source_sha256"].get(str(path.relative_to(ROOT))) != digest(path):
            raise ValueError("Cohort audit used a different implementation")
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Cohort summary input changed")
    for role, audit in audits.items():
        run = Path(manifest[role]); files = sorted((run / "jobs").glob("*.json"))
        if (audit.get("status") != "complete" or not files
                or audit.get("jobs") != len(files) or len(audit.get("rows", [])) != len(files)):
            raise ValueError("Incomplete cohort execution audit")
        recorded = {r["target"]: r for r in audit["rows"]}
        if len(recorded) != len(files):
            raise ValueError("Repeated audited worker target")
        declared = set()
        for file in files:
            if str(file.resolve()) not in inputs:
                raise ValueError("Worker missing from bound cohort inputs")
            job = load(file); target = Path(job["target"])
            phase = job["phase"]
            allowed = {"training"} if role == "fit" else {"evaluate", "autonomous"}
            if job.get("smoke") is not False or phase not in allowed:
                raise ValueError("Worker changed scientific scope or data role")
            sidecar = ".online.json" if mechanism == "ONLINE" else ".belief.json"
            paths = [target.with_suffix(s) for s in (".json", sidecar, ".precision.json")]
            if any(str(p.resolve()) not in inputs for p in paths):
                raise ValueError("Worker execution absent from bound audit")
            result, belief, precision = (load(p) for p in paths)
            cohort.check_execution(job, result, belief, precision, mechanism)
            copies = len(job["seeds"]) * (1 if phase == "autonomous" else job["particles"])
            key = str(target.resolve())
            expected = dict(target=key, subject=result["subject"], model=result["model"],
                            phase=phase, state_updates=copies * result["trials"], runs=len(job["seeds"]))
            if key in declared or recorded.get(key) != expected:
                raise ValueError("Worker identity or actual update count differs from summary")
            declared.add(key)
        actual = {str(f.with_suffix("").resolve()) for f in run.glob("**/chunks/*.json")
                  if not f.name.endswith(cohort.separate.SIDECARS)}
        if actual != declared:
            raise ValueError("Unmatched cohort result chunks")
    return inputs


def match_policy(old: dict, new: dict, old_protocol: dict, new_protocol: dict,
                 old_space: dict, new_space: dict) -> None:
    """Require identical scientific settings, starts, supports and source code."""
    for frozen, protocol, mechanism in ((old, old_protocol, "ONLINE"), (new, new_protocol, "SEPARATE")):
        settings = frozen["settings"]
        if (settings.get("belief_mechanism") != mechanism
                or settings.get("belief_intervention") != cohort.driver.INTERVENTIONS[mechanism]
                or settings.get("scope") != "common_policy_development_belief_refit"
                or protocol.get("smoke") is not False or frozen.get("smoke") is not False
                or protocol["settings"] != settings
                or protocol["source_sha256"] != frozen["source_sha256"]
                or type(settings.get("parallel_budget")) is not int
                or not 1 <= settings["parallel_budget"] <= 128):
            raise ValueError("Require the declared scientific cohort mechanism and policy")
    # Worker concurrency changes scheduling; candidate and particle budgets stay equal.
    execution_keys = {"belief_mechanism", "belief_intervention", "parallel_budget"}
    settings = [{k: v for k, v in f["settings"].items() if k not in execution_keys} for f in (old, new)]
    if settings[0] != settings[1] or old_space != new_space or old["source_sha256"] != new["source_sha256"]:
        raise ValueError("Cohort search supports, budgets or executed sources differ")
    keys = {(s, a) for s in settings[0]["subjects"] for a in settings[0]["arms"]}
    banks = []
    for protocol in (old_protocol, new_protocol):
        entries = protocol["initial_banks"]
        bank = {(r["subject"], r["model"]): r["rows"] for r in entries}
        if len(bank) != len(entries) or set(bank) != keys or any(not rows for rows in bank.values()):
            raise ValueError("Initial bank is incomplete or duplicated")
        banks.append(bank)
    if banks[0] != banks[1]:
        raise ValueError("Cohort mechanisms must start from the same participant-specific candidates")


def score_banks(banks: list[dict], saved: list[dict], settings: dict) -> dict[str, pd.DataFrame]:
    """Apply the established paired scores to every declared participant and arm."""
    keys = {(s, a) for s in settings["subjects"] for a in settings["arms"]}
    if len(banks) != 2 or len(saved) != 2 or any(set(b) != keys for b in banks):
        raise ValueError("Need both complete cohort banks")
    contrasts, feedback_rows, trial_rows, path_rows = [], [], [], []
    for cell in sorted(keys):
        subject, arm = cell; labels = dict(subject=subject, model=arm)
        retained.match_task(banks[0][cell], banks[1][cell])
        conditional, _, row = banks[0][cell]
        cut, length = row["training_trials"], len(conditional["choices"])
        if not 0 < cut < length:
            raise ValueError("Need the original nonempty training and later windows")
        for phase, first, last in (("training", 0, cut), ("test", cut, length), ("whole", 0, length)):
            estimates = []
            for bank, tables in zip(banks, saved):
                c, a, _ = bank[cell]
                estimate = retained.window_estimate(c, a, first, last, settings["features"]["block_width"])
                table = tables["scores"]
                selected = table[(table.subject == subject) & (table.model == arm) & (table.phase == phase)]
                if len(selected) != 1:
                    raise ValueError("Missing or duplicated saved score")
                for metric, value in estimate["metrics"].items():
                    np.testing.assert_allclose(value, selected.iloc[0][metric], rtol=0, atol=1e-12)
                estimates.append(estimate)
            contrasts.append(dict(**labels, phase=phase, trials=last-first,
                                  **paired_difference(estimates[1], estimates[0])))
        feedback = np.array([category_learning_feedback(int(choice), int(category), condition=row["condition"])
                             for choice, category in zip(conditional["choices"], conditional["categories"])])
        f, t = shared.feedback_scores(banks[0][cell][0], banks[1][cell][0], feedback, cut)
        feedback_rows.append(f.assign(**labels)); trial_rows.append(t.assign(**labels))
        paths = [t["paths"].loc[(t["paths"].subject == subject) & (t["paths"].model == arm)] for t in saved]
        path_rows.append(shared.path_contrasts(*paths).assign(**labels))
    tables = {name: pd.concat([table[name].assign(variant=label) for table, label in zip(saved, MECHANISMS)])
              for name in ("scores", "curves", "parameters", "paths")}
    tables.update(contrasts=pd.DataFrame(contrasts), feedback_scores=pd.concat(feedback_rows),
                  trials=pd.concat(trial_rows), path_contrasts=pd.concat(path_rows))
    return tables


def compare(online_summary: Path, separate_summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new cohort comparison directory")
    records = [retained.load_summary(p) for p in (online_summary, separate_summary)]
    inputs = {}
    for directory, mechanism, (manifest, _, _, bound) in zip((online_summary, separate_summary), MECHANISMS, records):
        for group in (bound, bind_summary(directory, manifest, mechanism)):
            for path, sha in group.items():
                if path in inputs and inputs[path] != sha:
                    raise ValueError("Cohort summaries bind different versions of the same input")
                inputs[path] = sha

    def read(path):
        path = Path(path).resolve(); sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError("Comparison input changed")
        inputs[str(path)] = sha
        return load(path)

    fits = [Path(row[0]["fit"]) for row in records]
    protocols = [read(p / "protocol.json") for p in fits]
    spaces = [read(p / "parameter_spaces.json") for p in fits]
    match_policy(records[0][1], records[1][1], *protocols, *spaces)
    settings = records[0][1]["settings"]
    keys = {(s, a) for s in settings["subjects"] for a in settings["arms"]}
    banks = [shared.load_bank(manifest, frozen, keys, settings) for manifest, frozen, _, _ in records]
    saved = [row[2] for row in records]
    tables = score_banks(banks, saved, settings)
    audits = []
    for fit in fits:
        audit, bound = audit_exchange(fit, settings["subjects"])
        audits.append(audit); inputs.update(bound)
    budgets = budget_contrasts(saved[0]["budgets"], saved[1]["budgets"], *audits)
    budgets = budgets.rename(columns={k: ("online_" + k[6:] if k.startswith("reset_") else "separate_" + k[7:])
                                      for k in budgets if k.startswith(("reset_", "retain_"))})
    tables.update(budgets=budgets, exchange_candidates=pd.concat([a.assign(variant=m) for a, m in zip(audits, MECHANISMS)]))
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Cohort comparison input changed during scoring")
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    sources = [Path(__file__), Path(cohort.__file__), Path(cohort.driver.__file__),
               Path(retained.__file__), Path(shared.__file__), *[Path(__file__).with_name(name) for name in
               ("model_0923_refit_budget.py", "model_0923_estimation_scope.py", "model_0923_h_validation.py",
                "model_0923_h_generation_uncertainty.py", "model_0923_episodic_refit.py", "model_0923_online_refit.py")]]
    (output / "source").mkdir()
    for path in sources:
        shutil.copy2(path, output / "source" / path.name)
    write(output / "manifest.json", dict(status="complete", subjects=settings["subjects"], arms=settings["arms"],
        online_summary=str(online_summary.resolve()), separate_summary=str(separate_summary.resolve()),
        inputs_sha256=inputs, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        tables_sha256={name + ".csv": digest(output / (name + ".csv")) for name in tables},
        actual_search_compute_equal=bool(budgets.actual_compute_equal.all()),
        contrast="SEPARATE minus ONLINE; mechanism and behavioral re-estimation jointly change",
        data_use="Original behavior-prefix estimation; later choices are development diagnostics; oral excluded",
        uncertainty="Paired complete-filter/path simulation errors at frozen parameters; no population inference",
        policy="Identical supports, initial banks and stage limits; adaptive exchange work audited separately"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("online-summary", "separate-summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.online_summary, args.separate_summary, args.output)
