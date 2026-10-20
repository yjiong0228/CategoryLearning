"""Replay the frozen extension candidate and score its external oral support.

Use --phase replay --summary SUMMARY --output NEW [--workers N] [--smoke],
then --phase align --source STATES --oral-audit FIXED_CODING --output NEW.
The complete extension summary supplies the single recommended proposal arm,
its behavior-prefix estimates, original evaluation particles and all seeds.
The existing ONLINE observer checks exact prediction reproduction. Oral scoring
uses the unchanged support/conflict/undetermined definitions and fixed coding.

Smoke replays one original filter for at most sixteen trials and cannot enter
oral scoring. The worker also accepts an explicitly marked eight-trial software
evaluation reference for integration checks; the public replay command requires
a completed scientific extension summary. Prior developmental influence on
the extension participants remains unknown.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile

import yaml

# Workers keep the archived cognitive package on sys.path. Load current
# orchestration by file without replacing that package's model implementation.
ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = "src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_states.py"
FIT = "src/Bayesian_state/workflows/analysis/model_0923_finalization_extension_fit.py"
spec = importlib.util.spec_from_file_location("extension_state_cohort", ROOT / COHORT)
cohort = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cohort
spec.loader.exec_module(cohort)
extension = cohort.helper("extension_state_fit", FIT)
read, digest = cohort.load, cohort.digest
USE = "registered_extension_validation"
POLICY = "registered_neutral_starts_behavior_prefix"
EXTRA = ("extension_scope", "extension_candidate", "extension_setup", "extension_recommendation")


def check_filter(planned: dict, actual: dict, point: dict, settings: dict, mechanism: str,
                 subject: int, arm: str, seed: int, particles: int, folder: Path, metadata: dict,
                 record: dict, precision: dict, belief: dict, references: dict) -> None:
    """Preserve the recommendation and the original evaluated filter identity."""
    extension.check_job(actual)
    if (mechanism != "ONLINE" or settings.get("extension_scope") != "extension_validation"
            or any(actual.get(k) != settings[k] for k in EXTRA)):
        raise ValueError("State replay differs from the frozen extension recommendation")
    cohort.check_filter(planned, {k: v for k, v in actual.items() if k not in EXTRA},
                        point, settings, mechanism, subject, arm, seed, particles, folder,
                        metadata, record, precision, belief, references)


def load_summary(summary: Path) -> tuple[dict, dict, dict]:
    """Bind the completed single-arm summary and its existing execution audit."""
    from src.Bayesian_state.evaluation import model_0923_reentry_refit as shared
    manifest, frozen, _, bindings = shared.load_summary(summary)
    fit = Path(manifest["fit"])
    settings, inputs, _ = extension.restore(fit)
    extra_file, audit_file = summary / "extension_manifest.json", summary / "execution_audit.json"
    extra, audit = read(extra_file), read(audit_file)
    if (settings != frozen["settings"] or extra.get("status") != "complete"
            or extra.get("scope") != "extension_validation"
            or extra.get("candidate") != settings["extension_candidate"]
            or extra.get("subjects") != settings["subjects"]
            or extra.get("previous_development_influence") != "unknown"
            or extra.get("recommendation") != settings["extension_recommendation"]
            or extra.get("generic_manifest_sha256") != digest(summary / "manifest.json")
            or extra.get("execution_audit_sha256") != digest(audit_file)
            or digest(summary / "extension_analysis_source.py") != digest(ROOT / FIT)
            or set(audit) != {"fit", "evaluation"}):
        raise ValueError("Require the completed, audited extension summary")
    for added in (inputs, extra["input_sha256"],
                  {str((ROOT / k).resolve()): v for k, v in frozen["source_sha256"].items()}):
        for path, sha in added.items():
            if path in bindings and bindings[path] != sha:
                raise ValueError("Extension summary binds inconsistent inputs")
            bindings[path] = sha
    for path in (extra_file, audit_file, summary / "extension_analysis_source.py"):
        bindings[str(path.resolve())] = digest(path)
    for path, sha in bindings.items():
        if digest(Path(path)) != sha:
            raise ValueError("Extension summary source or numerical evidence changed")
    for role, checked in audit.items():
        files = sorted((Path(manifest[role]) / "jobs").glob("*.json"))
        if (checked.get("status") != "complete" or not files
                or checked.get("jobs") != len(files) or len(checked.get("rows", [])) != len(files)):
            raise ValueError("Incomplete extension execution audit")
        actual = []
        for file in files:
            job = read(file)
            extension.check_job(job)
            target = Path(job["target"])
            paths = [file, *(target.with_suffix(s) for s in
                      (".json", ".npz", ".precision.json", ".online.json", ".experiment.json"))]
            if (any(str(p.resolve()) not in bindings for p in paths)
                    or job["smoke"] or job["extension_candidate"] != settings["extension_candidate"]
                    or job["phase"] not in ({"training"} if role == "fit" else {"evaluate", "autonomous"})):
                raise ValueError("Executed extension job is absent from the bound audit")
            result = read(target.with_suffix(".json"))
            copies = len(job["seeds"]) * (1 if job["phase"] == "autonomous" else job["particles"])
            actual.append(dict(target=str(target.resolve()), subject=result["subject"], model=job["model"],
                phase=job["phase"], state_updates=copies * result["trials"], runs=len(job["seeds"])))
        if checked["rows"] != actual:
            raise ValueError("Extension execution audit does not describe the saved workers")
    return manifest, frozen, bindings


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    from src.Bayesian_state.utils.seeding import stable_seed
    summary = ROOT / settings["summary"]
    manifest, frozen, bindings = load_summary(summary)
    original = frozen["settings"]
    arm, = original["arms"]
    if (source.resolve() != Path(manifest["evaluation"]).resolve()
            or settings["subjects"] != original["subjects"] or settings["arms"] != [arm]
            or settings["seeds_per_arm"] != original["budgets"]["evaluate"]["seeds"]
            or settings["sample_role"] != "development"
            or type(settings["workers"]) is not int or not 1 <= settings["workers"] <= 128):
        raise ValueError("Preserve all registered extension participants, the single arm and filter seeds")
    fit = Path(manifest["fit"])
    references = read(fit / "extension_protocol.json")["input_sha256"]
    evaluation = read(source / "protocol.json")
    plans = {j["name"]: j for j in evaluation["jobs"]}
    actual_files = {f.stem: f for f in (source / "jobs").glob("*.json")}
    nominees = {(r["subject"], r["model"]): r["candidate"] for r in frozen["nominees"]}
    if (len(plans) != len(evaluation["jobs"]) or set(plans) != set(actual_files)
            or len(nominees) != len(frozen["nominees"])
            or set(nominees) != {(s, arm) for s in original["subjects"]}):
        raise ValueError("Missing or duplicated extension nominees or evaluated jobs")
    files, selected = [], []
    for subject in settings["subjects"][:1] if smoke else settings["subjects"]:
        folder = Path(frozen["data"]) / f"subject_{subject}"
        metadata = read(folder / "prepared.json")
        budget = original["budgets"]["evaluate"]
        particles = {int(k): v for k, v in original["particle_overrides"].items()}.get(subject, budget["particles"])
        seeds = [stable_seed(dict(protocol=original["protocol"], base=original["seed"],
                 stage="evaluate", subject=subject, repeat=i)) for i in range(budget["seeds"])]
        point, group = nominees[subject, arm], []
        for i, seed in enumerate(seeds):
            name = f'evaluate_S{subject}_{arm}_{point["id"]}_C{i:03d}'
            if name not in plans:
                raise ValueError("Missing an original extension evaluation filter")
            planned, file = plans[name], actual_files[name]
            actual = read(file)
            if planned["source_sha256"] != frozen["source_sha256"]:
                raise ValueError("Evaluation used a different fitted implementation")
            target = Path(actual["target"])
            record, precision, belief = [read(target.with_suffix(s)) for s in
                                         (".json", ".precision.json", ".online.json")]
            check_filter(planned, actual, point, original, "ONLINE", subject, arm, seed,
                         particles, folder, metadata, record, precision, belief, references)
            group.append(file)
        files.extend(group[:1] if smoke else group)
        selected.append(dict(subject=subject, arm=arm, parameter_id=point["id"]))
    replay = cohort._replay
    for relative in (ENTRY, COHORT, FIT, cohort.ONLINE, cohort.SEPARATE, cohort.separate.STATES,
                     cohort.separate.helpers().OBSERVER, cohort.separate.helpers().BELIEF,
                     replay.REENTRY, replay.PROPOSAL):
        bindings[str(ROOT / relative)] = digest(ROOT / relative)
    return files, dict(parameter_estimation_scope="prefix", evaluation_use=USE,
        parameter_candidate_policy=POLICY, structural_intervention="retained_rule_precision",
        belief_mechanism="ONLINE", belief_intervention="online_retention_evidence",
        selected_parameters=selected, extension_fit=str(fit.resolve()), extension_summary=str(summary.resolve()),
        extension_manifest_sha256=digest(summary / "extension_manifest.json"),
        refit_nomination_sha256=digest(fit / "nominees.json"), fixed_coordinates=original["fixed_coordinates"],
        extension_scope="extension_validation", extension_candidate=original["extension_candidate"],
        extension_recommendation=original["extension_recommendation"], previous_development_influence="unknown",
        refit_replay_inputs=bindings, scientific_scope=not smoke,
        report_arm_policy="Single frozen recommended candidate; parameters fitted from each original behavior prefix")


def worker(file: Path) -> None:
    job = read(file)
    extension.check_job(job)
    if (job.get("evaluation_use") != USE or job.get("parameter_estimation_scope") != "prefix"
            or job.get("belief_mechanism") != "ONLINE" or job.get("cohort_belief_mechanism") != "ONLINE"
            or job.get("phase") != "evaluate"):
        raise ValueError("Extension observer lost its selection scope or ONLINE mechanism")
    software = job["extension_scope"] == "software_validation"
    if software:
        if job.get("scientific_scope") is not False or not 1 <= job.get("trial_limit", 0) <= 8:
            raise ValueError("Software worker requires an explicitly short, non-scientific replay")
    elif job.get("scientific_scope") is not True and job.get("trial_limit") != 16:
        raise ValueError("Formal replay requires the original evaluation or its sixteen-trial smoke")
    module = cohort.separate.helper("extension_online_state_worker", cohort.ONLINE)
    module.USE = USE
    module.worker(file)


def run(summary: Path, output: Path, workers: int = 128, smoke: bool = False) -> None:
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError("Use one to 128 workers, bounded by available CPUs and ready jobs")
    manifest = read(summary / "manifest.json")
    frozen = read(Path(manifest["fit"]) / "nominees.json")
    settings = frozen["settings"]
    config = dict(protocol="model0923_finalization_extension_states_v1", source_kind="registered_extension",
        source=manifest["evaluation"], summary=str(summary.resolve()), subjects=settings["subjects"],
        arms=list(settings["arms"]), workers=workers, seeds_per_arm=settings["budgets"]["evaluate"]["seeds"],
        sample_role="development")
    replay = cohort.separate.helpers().helpers().helpers()
    original_prepare = replay.prepare
    replay.source_jobs, replay.ENTRY = source_jobs, ENTRY

    def prepare(own_config, own_output, own_smoke):
        protocol, jobs = original_prepare(own_config, own_output, own_smoke)
        for job in jobs:
            job["reported_arm"] = job["model"]
        for relative in (COHORT, FIT, cohort.ONLINE, cohort.separate.helpers().OBSERVER):
            path = own_output / "source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            replay.shutil.copy2(ROOT / relative, path)
        return protocol, jobs

    replay.prepare = prepare
    with tempfile.TemporaryDirectory(prefix="extension_states_") as temporary:
        path = Path(temporary) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        replay.run(path, output, smoke)


def estimation_context(protocol: dict) -> tuple[dict, str]:
    candidate = protocol.get("extension_candidate")
    if (candidate not in extension.CANDIDATES or protocol.get("extension_scope") != "extension_validation"
            or protocol.get("scientific_scope") is not True or protocol.get("parameter_estimation_scope") != "prefix"
            or protocol.get("evaluation_use") != USE or protocol.get("parameter_candidate_policy") != POLICY
            or protocol.get("belief_mechanism") != "ONLINE"
            or protocol.get("belief_intervention") != "online_retention_evidence"
            or protocol.get("structural_intervention") != "retained_rule_precision"
            or protocol.get("previous_development_influence") != "unknown"
            or protocol["settings"]["subjects"] != [110, 209, 309]
            or protocol["settings"]["arms"] != [extension.CANDIDATES[candidate][0]]
            or any(not protocol.get(k) for k in ("extension_fit", "extension_summary", "extension_manifest_sha256",
                "extension_recommendation", "refit_nomination_sha256", "fixed_coordinates", "selected_parameters"))):
        raise ValueError("Oral alignment requires the complete scientific extension provenance")
    if (digest(Path(protocol["extension_summary"]) / "extension_manifest.json") != protocol["extension_manifest_sha256"]
            or digest(Path(protocol["extension_fit"]) / "nominees.json") != protocol["refit_nomination_sha256"]):
        raise ValueError("Extension recommendation or behavior nomination changed after replay")
    for path, sha in protocol["refit_replay_inputs"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Extension replay source changed before oral scoring")
    keys = ("parameter_estimation_scope", "evaluation_use", "parameter_candidate_policy", "belief_mechanism",
            "belief_intervention", "structural_intervention", "extension_fit", "extension_summary",
            "extension_manifest_sha256", "extension_candidate", "extension_recommendation", "extension_scope",
            "refit_nomination_sha256", "fixed_coordinates", "selected_parameters", "previous_development_influence")
    return dict({k: protocol[k] for k in keys}, window_definition="original_chronological_boundary"), (
        f"<p>扩展检验采用预先冻结的{candidate}版本。各被试参数由原前段选择数据与已登记的共同起点估计，"
        "状态重放逐值复现完整行为评价的选择概率。口述检验本题选择后、反馈前的局部分类依据，"
        "按固定编码分别报告支持、冲突与未确定质量。口述提供带噪声的外部证据，结合行为与自主轨迹解释。"
        "110、209、309的既往开发影响未确定；后段保留原时间边界，属于扩展时间外推检验。</p>")


def align(source: Path, oral_audit: Path, output: Path) -> None:
    from src.Bayesian_state.workflows.analysis import model_0923_decision_basis_alignment as alignment
    protocol, done = read(source / "protocol.json"), read(source / "completion.json")
    if (done["status"] != "complete" or done["smoke"] or protocol["smoke"]
            or done["jobs"] != done["exact_behavior_replays"]
            or done["jobs"] != protocol["jobs"]
            or protocol["source_sha256"].get(ENTRY) != digest(ROOT / ENTRY)):
        raise ValueError("Require complete exact scientific extension state replays")
    estimation_context(protocol)
    previous = alignment.estimation_context
    alignment.estimation_context = estimation_context
    try:
        alignment.run(source, oral_audit, output, reference_model=extension.CANDIDATES[protocol["extension_candidate"]][0])
    finally:
        alignment.estimation_context = previous


def main() -> None:
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
        parser.error("Provide replay summary or full states and fixed oral coding, with a new output")


if __name__ == "__main__":
    main()
