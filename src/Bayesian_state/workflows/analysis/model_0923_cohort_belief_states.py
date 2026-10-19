"""Replay complete cohort belief refits for external oral validation.

Use --phase replay --summary SUMMARY --output NEW [--workers N] [--smoke],
then --phase align --source REPLAY --oral-audit FIXED_CODING --output NEW.
Participants, proposal arms, parameters, particles and seeds come from the
completed cohort summary. Both mechanisms reuse their existing exact state
observers. Full replays retain all independent evaluation filters; smoke takes
the first participant, both arms and the first filter for at most 16 trials.
Oral alignment accepts complete scientific replays only. Parameters remain
those selected from the original behavior prefix; reports enter alignment.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
ONLINE = "src/Bayesian_state/workflows/analysis/model_0923_online_evidence_refit_states.py"
SEPARATE = "src/Bayesian_state/workflows/analysis/model_0923_separate_belief_refit_states.py"
USE = "development_cohort_belief_refit"


def helper(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Workers start with the archived cognitive package on sys.path. Load only the
# observer adapters from the current tree; preserve the archived cognitive core.
separate = helper("cohort_separate_refit_observer", SEPARATE)
_replay = separate.helpers().helpers().helpers()
load, digest = _replay.read, _replay.digest


def comparison_helpers():
    from src.Bayesian_state.evaluation import model_0923_cohort_belief_comparison
    return model_0923_cohort_belief_comparison


def check_filter(planned: dict, actual: dict, point: dict, settings: dict, mechanism: str,
                 subject: int, arm: str, seed: int, particles: int, folder: Path, metadata: dict,
                 record: dict, precision: dict, belief: dict, references: dict) -> None:
    """Require the selected behavior parameter and original evaluated filter."""
    comparison = comparison_helpers()
    expected = dict(planned, structural_intervention="retained_rule_precision", retain_precision=True,
        retention_reference_inputs=references, belief_intervention=comparison.cohort.driver.INTERVENTIONS[mechanism],
        online_belief=mechanism == "ONLINE", probe_mechanism=mechanism,
        cohort_belief_mechanism=mechanism, fixed_coordinates=settings["fixed_coordinates"])
    if (actual != expected or planned.get("phase") != "evaluate" or planned.get("smoke") is not False
            or planned["model"] != arm or planned["candidate"] != point or planned["seeds"] != [seed]
            or planned["particles"] != particles or Path(planned["folder"]).resolve() != folder.resolve()
            or planned["arm_settings"] != settings["arms"][arm] or planned["mechanism"] != settings["mechanism"]
            or metadata["subject"] != subject or record["subject"] != subject
            or record["condition"] != metadata["condition"] or record["trials"] != metadata["trials"]
            or record["training_trials"] != metadata["training_trials"]
            or record["test_trials"] != metadata["test_trials"]
            or any(record[k] != actual[k] for k in ("candidate", "model", "particles", "seeds"))):
        raise ValueError("Cohort replay changed its behavior nominee, task boundary or evaluated filter")
    comparison.cohort.check_execution(actual, record, belief, precision, mechanism)


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    from src.Bayesian_state.utils.seeding import stable_seed
    comparison = comparison_helpers()
    summary = ROOT / settings["summary"]
    manifest, frozen, _, bindings = comparison.retained.load_summary(summary)
    original = frozen["settings"]; mechanism = original["belief_mechanism"]
    bindings.update(comparison.bind_summary(summary, manifest, mechanism))
    if (source.resolve() != Path(manifest["evaluation"]).resolve()
            or settings["sample_role"] != "development"
            or type(settings["workers"]) is not int or not 1 <= settings["workers"] <= 128
            or settings["subjects"] != original["subjects"] or settings["arms"] != list(original["arms"])
            or settings["arms"] != ["POOL", "SEARCH"]
            or settings["seeds_per_arm"] != original["budgets"]["evaluate"]["seeds"]):
        raise ValueError("Preserve all declared cohort participants, proposal arms and filter seeds")
    fit = Path(manifest["fit"])
    declaration = load(fit / "cohort_belief_protocol.json")
    references = declaration["input_sha256"]
    evaluation = load(source / "protocol.json")
    plans = {j["name"]: j for j in evaluation["jobs"]}
    actual_files = {f.stem: f for f in (source / "jobs").glob("*.json")}
    nominees = {(r["subject"], r["model"]): r["candidate"] for r in frozen["nominees"]}
    keys = {(s, a) for s in original["subjects"] for a in original["arms"]}
    if (len(plans) != len(evaluation["jobs"]) or set(plans) != set(actual_files)
            or len(nominees) != len(frozen["nominees"]) or set(nominees) != keys):
        raise ValueError("Incomplete or duplicated cohort evaluation bank")
    files, selected = [], []
    subjects = settings["subjects"][:1] if smoke else settings["subjects"]
    for subject in subjects:
        folder = Path(frozen["data"]) / f"subject_{subject}"
        metadata = load(folder / "prepared.json")
        budget = original["budgets"]["evaluate"]
        particles = {int(k): v for k, v in original["particle_overrides"].items()}.get(subject, budget["particles"])
        seeds = [stable_seed(dict(protocol=original["protocol"], base=original["seed"],
                  stage="evaluate", subject=subject, repeat=i)) for i in range(budget["seeds"])]
        for arm in settings["arms"]:
            point = nominees[subject, arm]
            group = []
            for i, seed in enumerate(seeds):
                name = f'evaluate_S{subject}_{arm}_{point["id"]}_C{i:03d}'
                if name not in plans:
                    raise ValueError("Missing an original cohort evaluation filter")
                planned = plans[name]; file = actual_files[name]; actual = load(file)
                if planned["source_sha256"] != frozen["source_sha256"]:
                    raise ValueError("Evaluation used a different fitted implementation")
                target = Path(actual["target"])
                sidecar = ".online.json" if mechanism == "ONLINE" else ".belief.json"
                record, precision, belief = [load(target.with_suffix(s)) for s in (".json", ".precision.json", sidecar)]
                check_filter(planned, actual, point, original, mechanism, subject, arm, seed,
                             particles, folder, metadata, record, precision, belief, references)
                group.append(file)
            files.extend(group[:1] if smoke else group)
            selected.append(dict(subject=subject, arm=arm, parameter_id=point["id"]))
    replay = separate.helpers().helpers().helpers()
    sources = (ENTRY, ONLINE, SEPARATE, separate.STATES, separate.helpers().OBSERVER,
               separate.helpers().BELIEF, replay.REENTRY, replay.PROPOSAL,
               str(Path(comparison.__file__).relative_to(ROOT)))
    for relative in sources:
        bindings[str((ROOT / relative).resolve())] = digest(ROOT / relative)
    for path in (fit / "cohort_belief_protocol.json", source / "protocol.json", source / "completion.json"):
        bindings[str(path.resolve())] = digest(path)
    return files, dict(parameter_estimation_scope="prefix", evaluation_use=USE,
        parameter_candidate_policy="projected_common_search_policy_behavior_refit",
        structural_intervention="retained_rule_precision", belief_mechanism=mechanism,
        belief_intervention=comparison.cohort.driver.INTERVENTIONS[mechanism],
        selected_parameters=selected, cohort_fit=str(fit.resolve()), cohort_summary=str(summary.resolve()),
        cohort_manifest_sha256=digest(summary / "cohort_manifest.json"),
        refit_nomination_sha256=digest(fit / "nominees.json"), fixed_coordinates=original["fixed_coordinates"],
        refit_replay_inputs=bindings, scientific_scope=not smoke,
        report_arm_policy="POOL/SEARCH identify the original proposal; belief mechanism comes from the executed fit")


def worker(file: Path) -> None:
    """Dispatch to the original mechanism observer without changing state updates."""
    job = load(file); mechanism = job.get("cohort_belief_mechanism")
    if (mechanism not in ("ONLINE", "SEPARATE") or job.get("belief_mechanism") != mechanism
            or job.get("evaluation_use") != USE or job.get("parameter_estimation_scope") != "prefix"):
        raise ValueError("Cohort state worker lost its mechanism or parameter-selection scope")
    module = separate.helper("cohort_belief_state_worker", ONLINE if mechanism == "ONLINE" else SEPARATE)
    module.USE = USE
    module.worker(file)


def run(summary: Path, output: Path, workers: int = 128, smoke: bool = False) -> None:
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError("Use one to 128 workers, bounded by ready tasks and available CPUs")
    manifest = load(summary / "manifest.json")
    frozen = load(Path(manifest["fit"]) / "nominees.json")
    settings = frozen["settings"]
    config = dict(protocol="model0923_cohort_belief_states_v1", source_kind="cohort_belief_refit",
        source=manifest["evaluation"], summary=str(summary.resolve()), subjects=settings["subjects"],
        arms=list(settings["arms"]), workers=workers, seeds_per_arm=settings["budgets"]["evaluate"]["seeds"],
        sample_role="development")
    replay = separate.helpers().helpers().helpers(); original_prepare = replay.prepare
    replay.source_jobs, replay.ENTRY = source_jobs, ENTRY

    def prepare(own_config, own_output, own_smoke):
        protocol, jobs = original_prepare(own_config, own_output, own_smoke)
        for job in jobs:
            job["reported_arm"] = ("SEPARATE_" if job["belief_mechanism"] == "SEPARATE" else "") + job["model"]
        for relative in (ONLINE, SEPARATE, separate.STATES, separate.helpers().OBSERVER, separate.helpers().BELIEF):
            path = own_output / "source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            replay.shutil.copy2(ROOT / relative, path)
        return protocol, jobs

    replay.prepare = prepare
    with tempfile.TemporaryDirectory(prefix="cohort_belief_states_") as temporary:
        path = Path(temporary) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        replay.run(path, output, smoke)


def estimation_context(protocol: dict) -> tuple[dict, str]:
    comparison = comparison_helpers()
    mechanism = protocol.get("belief_mechanism")
    if (mechanism not in comparison.MECHANISMS or protocol.get("evaluation_use") != USE
            or protocol.get("parameter_estimation_scope") != "prefix"
            or protocol.get("parameter_candidate_policy") != "projected_common_search_policy_behavior_refit"
            or protocol.get("belief_intervention") != comparison.cohort.driver.INTERVENTIONS[mechanism]
            or protocol.get("structural_intervention") != "retained_rule_precision"
            or protocol.get("scientific_scope") is not True
            or any(not protocol.get(k) for k in ("cohort_fit", "cohort_summary", "cohort_manifest_sha256",
                                                 "refit_nomination_sha256", "fixed_coordinates", "selected_parameters"))):
        raise ValueError("Oral evaluation requires complete scientific cohort selection provenance")
    keys = ("parameter_estimation_scope", "evaluation_use", "parameter_candidate_policy", "belief_mechanism",
            "belief_intervention", "structural_intervention", "cohort_fit", "cohort_summary",
            "cohort_manifest_sha256", "refit_nomination_sha256", "fixed_coordinates", "selected_parameters")
    return dict({k: protocol[k] for k in keys}, window_definition="original_chronological_boundary"), (
        f"<p>本次参数由{mechanism}机制下各被试原前段选择估计，POOL与SEARCH使用共同参数范围、起点和搜索政策。"
        "状态重放保留完整评价的参数、粒子数及独立种子，并逐值复现保存的选择预测。"
        "口述检验本题选择后、反馈前的局部分类依据；后段作为开发阶段的时间外推诊断。"
        "各被试保留原试次顺序、训练边界及报告集合。</p>")


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from src.Bayesian_state.workflows.analysis import model_0923_decision_basis_alignment as alignment
    protocol, done = load(source / "protocol.json"), load(source / "completion.json")
    if (done["status"] != "complete" or done["smoke"] or protocol["smoke"]
            or done["jobs"] != done["exact_behavior_replays"]
            or protocol["source_sha256"].get(ENTRY) != digest(ROOT / ENTRY)):
        raise ValueError("Require complete exact scientific cohort state replays")
    estimation_context(protocol)
    previous = alignment.estimation_context
    alignment.estimation_context = estimation_context
    try:
        alignment.run(source, oral_audit, output, reference_model="POOL")
    finally:
        alignment.estimation_context = previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("replay", "align"), default="replay")
    for name in ("summary", "source", "oral-audit", "output", "worker"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == "replay" and args.summary and args.output:
        run(args.summary, args.output, args.workers, args.smoke)
    elif args.phase == "align" and args.source and args.oral_audit and args.output and not args.smoke:
        align(args.source, args.oral_audit, args.output)
    else:
        parser.error("Replay needs --summary; align needs --source and --oral-audit; supply a new --output")
