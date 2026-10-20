"""Register extension participants using task integrity and fixed oral coverage.

Run --config YAML --output NEW_DIRECTORY before inspecting new model scores.
The original raw/processed text join and both established report encoders are
reused. Each phase needs the declared number of complete reports under both
readings. Select the smallest eligible identifier per condition, retaining all
exclusions and coverage counts. Parsing coverage does not measure report truth.
This entry prepares the participant list; fitting follows recommendation freeze.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import pandas as pd
import yaml

from . import model_0923_oral_constraint_audit as audit
from . import model_0923_oral_reference_sensitivity as relative
from .model_0923_h_validation import training_boundary, write

ROOT = audit.ROOT
KEYS = ["iSub", "iSession", "iTrial"]


def select_participants(rows: pd.DataFrame, minimum_reports: int) -> pd.DataFrame:
    """Apply one phase-coverage threshold before selecting by identifier."""
    if minimum_reports < 1 or rows.subject.duplicated().any():
        raise ValueError("Require a positive coverage threshold and unique participants")
    result = rows.copy()
    result["eligible"] = (
        result.task_valid
        & result.training_common_complete.ge(minimum_reports)
        & result.later_common_complete.ge(minimum_reports)
    )
    result["selected"] = False
    eligible = result[result.eligible].sort_values(["condition", "subject"])
    chosen = eligible.groupby("condition", sort=True).head(1).index
    result.loc[chosen, "selected"] = True
    result["coverage_interpretation"] = "fixed_encoder_coverage; report_fidelity_unmeasured"
    return result.sort_values(["condition", "subject"]).reset_index(drop=True)


def run(config_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new extension eligibility directory")
    config = yaml.safe_load(config_path.read_text())
    if (config["protocol"] != "model0923_finalization_extension_eligibility_v1"
            or config["selection"] != "smallest_eligible_id_per_condition"
            or config["previous_development_influence"] != "unknown"
            or config["evaluation_gate"] != "recommended_version_frozen"):
        raise ValueError("Keep the registered selection rule and prior-use limitation")
    inputs: dict[str, str] = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        value = audit.digest(path)
        if expected is not None and value != expected:
            raise ValueError(f"Changed registered input: {path}")
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError(f"Input changed during registration: {path}")
        inputs[str(path)] = value
        return path

    def read(path: Path) -> dict:
        return json.loads(bind(path).read_text())

    bind(config_path)
    reservation = read(ROOT / config["reservation"] / "manifest.json")
    midpoint = read(ROOT / config["coding_reference"] / "manifest.json")
    sensitivity = read(ROOT / config["relative_reference"] / "manifest.json")
    declared = {(int(s), int(c)) for c, subjects in config["subjects_by_condition"].items()
                for s in subjects}
    reserved = {(r["subject"], r["condition"]) for r in reservation["selected"]}
    if (reservation["status"] != "complete" or declared != reserved or len(declared) != 6
            or midpoint["status"] != "complete" or sensitivity["status"] != "complete"
            or sensitivity["semantic_sensitivity"] != relative.READING):
        raise ValueError("Require the six registered participants and both completed coding references")
    # The relative-reading helper imports ROOT/read/digest from this module.
    # Report encoding uses the separately bound encoder and constraint sources.
    helper_only = "src/Bayesian_state/workflows/analysis/model_0923_decision_basis_alignment.py"
    helper_versions = {}
    for source in (midpoint, sensitivity):
        for name, value in source["source_sha256"].items():
            if name == helper_only:
                bind(ROOT / name)
                helper_versions[name] = dict(
                    reference_sha256=value, current_sha256=inputs[str((ROOT / name).resolve())],
                    role="imported path/hash/JSON helpers; fitted alignment is not executed")
            else:
                bind(ROOT / name, value)
    data_name = midpoint["config"]["processed"]
    bind(ROOT / data_name, midpoint["input_sha256"][data_name])
    for file in (Path(__file__), Path(audit.__file__), Path(relative.__file__)):
        bind(file)
    from . import model_0923_h_validation as split
    bind(Path(split.__file__))

    output.mkdir(parents=True)
    write(output / "protocol.json", dict(
        status="frozen_before_eligibility", created_utc=datetime.now(timezone.utc).isoformat(),
        config=config, inputs_sha256=inputs.copy(),
        helper_only_source_versions=helper_versions,
        selection_inputs="task identities, valid labels, raw/processed agreement and fixed report coverage",
        model_scores_used=False, behavioral_accuracy_used=False,
        limitation="Prior development influence unknown; report fidelity is not measured"))
    rows, encoded, changed = [], [], []
    for subject, condition in sorted(declared):
        settings = dict(midpoint["config"], subjects=[subject])
        row = dict(subject=subject, condition=condition, task_valid=False, issue="",
                   trials=0, training_trials=0, later_trials=0,
                   training_common_complete=0, later_common_complete=0)
        try:
            frame, hashes = audit.load_reports(settings)
            for name, value in hashes.items():
                bind(ROOT / name, value)
            if frame.condition.nunique() != 1 or int(frame.condition.iloc[0]) != condition:
                raise ValueError("Participant condition differs from the reserved identity")
            if len(frame) < config["minimum_valid_trials"]:
                raise ValueError("Insufficient valid task trials")
            names = [f"feature{i}_name" for i in range(1, 5)]
            if any(set(values) != set(settings["feature_order"]) for values in frame[names].itertuples(index=False, name=None)):
                raise ValueError("Task feature labels are incomplete")
            cut = training_boundary(len(frame), config["training_fraction"])
            absolute, _ = audit.encode_observations(
                frame, tolerance=settings["tolerance"],
                symbolic_half_reference=settings.get("symbolic_half_reference", False),
                half_reference=settings.get("half_reference"))
            alternative, _ = relative.change_reading(absolute)
            pd.testing.assert_frame_equal(absolute[KEYS], alternative[KEYS])
            common = absolute.scope.eq("complete") & alternative.scope.eq("complete")
            first = absolute.ordinal_trial.le(cut)
            row.update(task_valid=True, trials=len(frame), training_trials=cut,
                       later_trials=len(frame)-cut,
                       nonempty_reports=int(absolute.text.str.strip().ne("").sum()),
                       midpoint_complete=int(absolute.scope.eq("complete").sum()),
                       relative_complete=int(alternative.scope.eq("complete").sum()),
                       common_complete=int(common.sum()),
                       training_common_complete=int((common & first).sum()),
                       later_common_complete=int((common & ~first).sum()),
                       changed_readings=int(alternative.reference_sensitivity_changed.sum()))
            absolute["training"] = first
            absolute["common_complete"] = common
            alternative["training"] = first
            alternative["common_complete"] = common
            encoded.append(absolute)
            changed.append(alternative)
        except ValueError as error:
            row["issue"] = str(error)
        rows.append(row)
    table = select_participants(pd.DataFrame(rows), config["minimum_common_complete_per_phase"])
    table.loc[table.task_valid & ~table.eligible, "issue"] = "insufficient_common_complete_reports_in_a_phase"
    table.to_csv(output / "eligibility.csv", index=False)
    selected = table[table.selected][["subject", "condition", "trials", "training_trials", "later_trials"]]
    selected.to_csv(output / "selected.csv", index=False)
    for name, frames in (("midpoint_reports", encoded), ("relative_reports", changed)):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output / (name + ".csv"), index=False)
    for path, expected in inputs.items():
        if audit.digest(Path(path)) != expected:
            raise ValueError(f"Input changed during registration: {path}")
    write(output / "completion.json", dict(
        status="complete", selected=selected.to_dict("records"), all_conditions_covered=len(selected)==3,
        config=config, inputs_sha256=inputs, protocol_sha256=audit.digest(output / "protocol.json"),
        model_scores_used=False, behavioral_accuracy_used=False, parameter_fitting_started=False,
        previous_development_influence="unknown", recommendation_freeze_required=True,
        scope="Participant registration and encoding coverage; no rule-catalogue or fitted-model alignment scores",
        versions=dict(python=platform.python_version(), pandas=pd.__version__, pyyaml=yaml.__version__),
        artifacts_sha256={p.name: audit.digest(p) for p in output.iterdir() if p.is_file()}))
    print(table.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
