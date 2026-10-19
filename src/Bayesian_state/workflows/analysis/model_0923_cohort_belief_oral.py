"""Compare cohort ONLINE and SEPARATE refits against identical oral reports.

Use --original ONLINE_ALIGNMENT --revised SEPARATE_ALIGNMENT --subject ID
--arm POOL_OR_SEARCH --output NEW. Each alignment must come from the complete
cohort state replay and its behavior-prefix nomination. Common supports,
initial banks, search policy, reports, readout and filter seeds are checked.
The established SEPARATE comparison writes scores, feedback strata and paired
complete-filter Monte Carlo errors unchanged. cohort_manifest.json binds this
cohort adapter to that numerical report. Run each fixed oral reading separately.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil

from src.Bayesian_state.evaluation import model_0923_cohort_belief_comparison as behavior
from src.Bayesian_state.workflows.analysis import model_0923_cohort_belief_states as states
from src.Bayesian_state.workflows.analysis import model_0923_separate_belief_refit_oral as shared

ROOT = behavior.ROOT
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
load, digest = states.load, states.digest


def check_state(job: dict, record: dict, protocol: dict, manifest: dict, mechanism: str) -> None:
    """Bind local report evidence to the actual cohort mechanism and timing."""
    if mechanism not in behavior.MECHANISMS:
        raise ValueError("Unknown cohort belief mechanism")
    intervention = behavior.cohort.driver.INTERVENTIONS[mechanism]
    if (job.get("cohort_belief_mechanism") != mechanism or job.get("probe_mechanism") != mechanism
            or any(x.get("belief_mechanism") != mechanism for x in (job, protocol, manifest))
            or any(x.get("belief_intervention") != intervention for x in (job, protocol, manifest))
            or any(x.get("evaluation_use") != states.USE for x in (job, record, protocol, manifest))
            or job.get("parameter_estimation_scope") != "prefix" or record.get("parameter_estimation_scope") != "prefix"
            or record.get("arm") != job["model"] or record.get("proposal_arm") != job["model"]
            or record.get("exact_behavior_replay") is not True
            or record.get("timing") != "after_choice_before_current_feedback"
            or record.get("data_use") != "behavior_only; no_oral_input"
            or record.get("particle_transitions") != record["particles"] * record["trials"]):
        raise ValueError("Cohort oral state lost its mechanism, timing or behavior-only provenance")
    count = record["particles"] * (record["trials"] - 1)
    if mechanism == "ONLINE":
        if (job.get("reported_arm") != job["model"] or job.get("online_belief") is not True
                or record.get("online_belief") is not True
                or record.get("online_evidence_matches_retained_state") is not True
                or record.get("online_reconstructions") != count or record.get("separate_belief", False)):
            raise ValueError("Missing executed ONLINE evidence in cohort state")
    elif (job.get("reported_arm") != "SEPARATE_" + job["model"] or job.get("online_belief") is not False
            or record.get("online_belief") is not False or record.get("separate_belief") is not True
            or record.get("belief_intervention") != intervention
            or record.get("belief_gamma") != job["candidate"]["hyperparams"][behavior.cohort.separate.GAMMA]
            or record.get("belief_matches_saved_state") is not True or record.get("belief_updates") != count):
        raise ValueError("Missing executed SEPARATE evidence in cohort state")


def validate_states(manifest: dict, subject: int, arm: str, mechanism: str,
                    expected_files: list[Path] | None = None) -> dict[str, str]:
    source = Path(manifest["source"])
    protocol = load(source / "protocol.json")
    context, _ = states.estimation_context(protocol)
    if any(manifest.get(k) != v for k, v in context.items()):
        raise ValueError("Oral alignment changed the cohort parameter-selection context")
    inputs = {str((source / "protocol.json").resolve()): digest(source / "protocol.json")}
    originals = []
    for file in sorted((source / "states").glob("*.json")):
        record = load(file)
        if (record["subject"], record["arm"]) != (subject, arm):
            continue
        job_file = source / "jobs" / file.name
        job = load(job_file)
        check_state(job, record, protocol, manifest, mechanism)
        originals.append(str(Path(job["original_job"]).resolve()))
        for path in (file, job_file):
            inputs[str(path.resolve())] = digest(path)
    if len(originals) != protocol["settings"]["seeds_per_arm"] or len(set(originals)) != len(originals):
        raise ValueError("Missing or duplicated complete cohort filter seeds")
    if expected_files is not None:
        expected = set()
        for file in expected_files:
            job = load(file)
            metadata = load(Path(job["folder"]) / "prepared.json")
            if (metadata["subject"], job["model"]) == (subject, arm):
                expected.add(str(file.resolve()))
        if set(originals) != expected:
            raise ValueError("Oral states and frozen cohort evaluation use different filters")
    return inputs


def load_separate_group(directory: Path, subject: int, arm: str) -> tuple[tuple, dict, dict]:
    scored, identity, inputs = shared.comparison.load_group(directory, subject, arm)
    inputs.update(validate_states(load(directory / "manifest.json"), subject, arm, "SEPARATE"))
    identity["mechanism"]["separate_belief"] = True
    return scored, identity, inputs


def bind_refits(before: dict, after: dict, old_manifest: dict, new_manifest: dict,
                subject: int, arm: str) -> dict[str, str]:
    inputs, frozen_banks, protocols, spaces = {}, [], [], []

    def bind(path):
        path = Path(path).resolve(); sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError("Cohort comparison input changed")
        inputs[str(path)] = sha
        return path

    for identity, manifest, mechanism in zip((before, after), (old_manifest, new_manifest), behavior.MECHANISMS):
        fit = Path(manifest["cohort_fit"]); summary = Path(manifest["cohort_summary"])
        frozen, protocol, done = [load(bind(fit / name)) for name in ("nominees.json", "protocol.json", "completion.json")]
        shared.old_adapter.validate_nomination(identity, manifest, frozen, protocol, done,
                                               digest(fit / "nominees.json"), subject, arm)
        if (frozen["settings"]["belief_mechanism"] != mechanism
                or manifest.get("cohort_manifest_sha256") != digest(bind(summary / "cohort_manifest.json"))
                or manifest.get("fixed_coordinates") != frozen["settings"]["fixed_coordinates"]
                or any(identity["candidate"]["hyperparams"].get(k) != v
                       for k, v in frozen["settings"]["fixed_coordinates"].items())):
            raise ValueError("Cohort oral report changed its mechanism, summary or fixed coordinates")
        state_protocol = load(bind(Path(manifest["source"]) / "protocol.json"))
        if Path(state_protocol["settings"]["summary"]).resolve() != summary.resolve():
            raise ValueError("State replay refers to another cohort summary")
        evaluation = ROOT / state_protocol["settings"]["source"]
        files, metadata = states.source_jobs(state_protocol["settings"], evaluation, False)
        for key in ("cohort_fit", "cohort_summary", "cohort_manifest_sha256", "refit_nomination_sha256",
                    "belief_mechanism", "belief_intervention", "selected_parameters"):
            if metadata[key] != manifest.get(key):
                raise ValueError("Executed filters and oral states name different cohort nominees")
        for group in (metadata["refit_replay_inputs"], validate_states(manifest, subject, arm, mechanism, files)):
            for path, sha in group.items():
                if path in inputs and inputs[path] != sha:
                    raise ValueError("Cohort inputs disagree between fit and state replay")
                inputs[path] = sha
        frozen_banks.append(frozen); protocols.append(protocol)
        spaces.append(load(bind(fit / "parameter_spaces.json")))
    behavior.match_policy(*frozen_banks, *protocols, *spaces)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Cohort fit or state input changed")
    return inputs


def run(original: Path, revised: Path, output: Path, subject: int, arm: str) -> None:
    if output.exists():
        raise FileExistsError("Use a new cohort oral comparison directory")
    # A fresh adapter module keeps the existing 208 comparison and other callers
    # untouched while reusing its numerical scoring and report writer.
    spec = importlib.util.spec_from_file_location(
        "src.Bayesian_state.workflows.analysis._cohort_oral_comparison", Path(shared.__file__))
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    entry.bind_refits, entry.load_separate_group = bind_refits, load_separate_group
    entry.run(original, revised, output, subject, arm)
    sources = [Path(__file__), Path(states.__file__), Path(behavior.__file__), Path(shared.__file__)]
    for file in sources:
        target = output / "source" / file.name
        if target.exists():
            if digest(target) != digest(file):
                raise ValueError("Comparison archived a different cohort source")
        else:
            shutil.copy2(file, target)
    shared.states.helpers().helpers().helpers().write(output / "cohort_manifest.json", dict(
        status="complete", contrast_kind="cohort_belief_refit", subject=subject, arm=arm,
        numerical_manifest_sha256=digest(output / "manifest.json"),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        mechanisms=list(behavior.MECHANISMS), parameter_selection="Original behavior prefix; oral excluded",
        contrast="SEPARATE minus ONLINE; mechanism and behavior re-estimation jointly change",
        readout="Identical report identities, fixed reading, task boundary and complete filter seeds",
        score_definition="Existing paired joint-mass report scores and complete-filter deletion errors unchanged"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "revised", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--arm", choices=("POOL", "SEARCH"), required=True)
    args = parser.parse_args()
    run(args.original, args.revised, args.output, args.subject, args.arm)
