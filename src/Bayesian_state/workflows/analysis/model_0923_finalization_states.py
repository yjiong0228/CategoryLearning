"""Replay joint precision fits and completed ADD10 references for oral checks.

Use --phase replay --summary SUMMARY --output NEW [--workers N] [--smoke],
then --phase align --source STATES --oral-audit FIXED_CODING --output NEW.
Each complete behavior summary supplies its original nominees, filter seeds,
particles and trial boundaries. The existing ONLINE/SEPARATE observer checks
exact behavior reproduction. Oral scoring retains the existing support,
conflict and undetermined definitions and the selected fixed coding.

Joint fits require their finalization manifest. Existing additive references
retain their original cohort/development provenance. A short replay uses one
filter per proposal arm for at most sixteen trials and cannot enter oral scoring.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = "src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_states.py"
JOINT = "src/Bayesian_state/workflows/analysis/model_0923_finalization_joint.py"
spec = importlib.util.spec_from_file_location("finalization_state_cohort", ROOT / COHORT)
cohort = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cohort
spec.loader.exec_module(cohort)
_check_filter = cohort.check_filter
_source_jobs = cohort.source_jobs
_estimation_context = cohort.estimation_context
FAMILIES = {"ADD10": ("fast", 10., 25.), "LOG25": ("log", 1., 25.)}
KINDS = {"joint_finalization", "completed_additive_reference"}
EXTRA = {"development_cohort_scope", "development_cohort_fit",
         "finalization_precision_family", "finalization_scope"}


def family_from_settings(settings: dict) -> str:
    """Resolve the executed mode, retaining the original rate interpretation."""
    family = settings.get("finalization_precision_family", "ADD10")
    if family not in FAMILIES or any(
        (arm["precision"], arm["rate_scale"], arm["upper"]) != FAMILIES[family]
        for arm in settings["arms"].values()
    ):
        raise ValueError("Declared precision family differs from the fitted update")
    if ("finalization_precision_family" in settings
            and settings.get("finalization_scope") != "development_comparison"):
        raise ValueError("Require a scientific joint precision fit")
    return family


def check_filter(planned: dict, actual: dict, point: dict, settings: dict, mechanism: str,
                 subject: int, arm: str, seed: int, particles: int, folder: Path, metadata: dict,
                 record: dict, precision: dict, belief: dict, references: dict) -> None:
    family = family_from_settings(settings)
    expected_extra = {}
    if "development_cohort_fit" in settings:
        expected_extra.update(development_cohort_scope="completed_behavior_fit_development_cohort",
                              development_cohort_fit=settings["development_cohort_fit"])
    if "finalization_precision_family" in settings:
        expected_extra.update(finalization_precision_family=family,
                              finalization_scope="development_comparison")
    if {k: actual[k] for k in EXTRA if k in actual} != expected_extra:
        raise ValueError("Executed filter lost its participant or precision provenance")
    original_actual = {k: v for k, v in actual.items() if k not in EXTRA}
    _check_filter(planned, original_actual, point, settings, mechanism, subject, arm, seed,
                  particles, folder, metadata, record, precision, belief, references)


def bind_precision_summary(summary: Path, settings: dict) -> tuple[dict, str, Path]:
    """Bind the added joint-fit evidence without changing the common audit."""
    family = family_from_settings(settings)
    if "finalization_precision_family" not in settings:
        if (summary / "development_manifest.json").exists():
            from src.Bayesian_state.evaluation import model_0923_development_belief_refit as development
            return development.bind_registration(summary), "completed_additive_reference", summary / "development_manifest.json"
        return {}, "completed_additive_reference", summary / "cohort_manifest.json"
    manifest_file = summary / "finalization_manifest.json"
    manifest = cohort.load(manifest_file)
    if (manifest.get("status") != "complete" or manifest.get("precision_family") != family
            or manifest.get("scope") != "development_comparison"
            or manifest.get("cohort_manifest_sha256") != cohort.digest(summary / "cohort_manifest.json")
            or manifest.get("source_sha256", {}).get(JOINT) != cohort.digest(ROOT / JOINT)):
        raise ValueError("Require the intact completed joint precision summary")
    bindings = dict(manifest["inputs_sha256"])
    for name, expected in bindings.items():
        if cohort.digest(Path(name)) != expected:
            raise ValueError("Joint precision summary input changed")
    bindings[str(manifest_file.resolve())] = cohort.digest(manifest_file)
    bindings[str(ROOT / JOINT)] = cohort.digest(ROOT / JOINT)
    return bindings, "joint_finalization", manifest_file


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    files, metadata = _source_jobs(settings, source, smoke)
    frozen = cohort.load(Path(metadata["cohort_fit"]) / "nominees.json")
    original = frozen["settings"]
    extra, kind, manifest_file = bind_precision_summary(ROOT / settings["summary"], original)
    bindings = metadata["refit_replay_inputs"]
    for name, expected in extra.items():
        if name in bindings and bindings[name] != expected:
            raise ValueError("State and precision summaries bind different inputs")
        bindings[name] = expected
    for relative in (COHORT, ENTRY):
        bindings[str(ROOT / relative)] = cohort.digest(ROOT / relative)
    metadata.update(finalization_precision_family=family_from_settings(original),
                    precision_source_kind=kind,
                    precision_summary_manifest=str(manifest_file.resolve()),
                    precision_summary_sha256=cohort.digest(manifest_file))
    return files, metadata


@contextmanager
def precision_observer():
    previous = {name: getattr(cohort, name) for name in
                ("ENTRY", "check_filter", "source_jobs", "estimation_context")}
    cohort.ENTRY, cohort.check_filter = ENTRY, check_filter
    cohort.source_jobs, cohort.estimation_context = source_jobs, estimation_context
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(cohort, name, value)


def check_precision_context(protocol: dict) -> None:
    family = protocol.get("finalization_precision_family")
    if (family not in FAMILIES or protocol.get("precision_source_kind") not in KINDS
            or not protocol.get("precision_summary_manifest") or not protocol.get("precision_summary_sha256")
            or (protocol["precision_source_kind"] == "completed_additive_reference" and family != "ADD10")):
        raise ValueError("Missing or inconsistent precision selection provenance")
    if cohort.digest(Path(protocol["precision_summary_manifest"])) != protocol["precision_summary_sha256"]:
        raise ValueError("Precision summary changed after state selection")


def worker(file: Path) -> None:
    job = cohort.load(file)
    check_precision_context(job)
    arm = job["arm_settings"]
    if (arm["precision"], arm["rate_scale"], arm["upper"]) != FAMILIES[job["finalization_precision_family"]]:
        raise ValueError("State worker would execute a different precision update")
    cohort.worker(file)


def run(summary: Path, output: Path, workers: int = 128, smoke: bool = False) -> None:
    with precision_observer():
        cohort.run(summary, output, workers, smoke)


def estimation_context(protocol: dict) -> tuple[dict, str]:
    check_precision_context(protocol)
    metadata, note = _estimation_context(protocol)
    metadata.update({key: protocol[key] for key in
                     ("finalization_precision_family", "precision_source_kind",
                      "precision_summary_manifest", "precision_summary_sha256")})
    return metadata, note + (
        f"<p>精度更新采用{protocol['finalization_precision_family']}，沿用该行为拟合的全部参数。"
        "口述提供带噪声的外部支持；支持、冲突和未确定质量结合行为及自主学习过程解释。</p>")


def align(source: Path, oral_audit: Path, output: Path) -> None:
    with precision_observer():
        cohort.align(source, oral_audit, output)


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
        parser.error("Provide replay summary or complete states and fixed oral coding, with a new output")


if __name__ == "__main__":
    main()
