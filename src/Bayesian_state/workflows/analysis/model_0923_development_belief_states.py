"""Replay registered development fits for external oral support checks.

Use --phase replay --summary SUMMARY --output NEW [--workers N] [--smoke],
then --phase align --source STATES --oral-audit FIXED_CODING --output NEW.
Compare the two mechanisms with --phase compare --original ONLINE_ALIGNMENT
--revised SEPARATE_ALIGNMENT --subject ID --arm POOL|SEARCH --output NEW.
The original cohort observer retains every fitted parameter, evaluation seed,
particle count and trial boundary. The registration adapter adds participant
provenance; state updates and oral scoring use the established implementations.
Full oral alignment requires complete, exact behavior replays. Reports provide
noisy external support and do not enter parameter estimation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = "src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_states.py"
REGISTRATION = "src/Bayesian_state/workflows/analysis/model_0923_development_belief_refit.py"

# Worker subprocesses import their archived cognitive package. Load these
# provenance/observer adapters by file while preserving that model package.
spec = importlib.util.spec_from_file_location("registered_development_state_cohort", ROOT / COHORT)
cohort = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cohort
spec.loader.exec_module(cohort)
registration = cohort.helper("registered_development_state_fit", REGISTRATION)
SCOPE = registration.SCOPE
_check_filter = cohort.check_filter
_source_jobs = cohort.source_jobs
_estimation_context = cohort.estimation_context


def summary_helpers():
    from src.Bayesian_state.evaluation import model_0923_development_belief_refit
    return model_0923_development_belief_refit


def check_filter(planned: dict, actual: dict, point: dict, settings: dict, mechanism: str,
                 subject: int, arm: str, seed: int, particles: int, folder: Path, metadata: dict,
                 record: dict, precision: dict, belief: dict, references: dict) -> None:
    """Validate the two added provenance fields before the original filter audit."""
    if (actual.get("development_cohort_scope") != SCOPE
            or not settings.get("development_cohort_fit")
            or actual.get("development_cohort_fit") != settings["development_cohort_fit"]):
        raise ValueError("State replay lost its registered development participants")
    original_actual = {key: value for key, value in actual.items()
                       if key not in ("development_cohort_scope", "development_cohort_fit")}
    _check_filter(planned, original_actual, point, settings, mechanism, subject, arm, seed,
                  particles, folder, metadata, record, precision, belief, references)


def source_jobs(settings: dict, source: Path, smoke: bool) -> tuple[list[Path], dict]:
    summary = ROOT / settings["summary"]
    extra = summary_helpers().bind_registration(summary)
    files, metadata = _source_jobs(settings, source, smoke)
    frozen = cohort.load(Path(metadata["cohort_fit"]) / "nominees.json")
    fit_source = frozen["settings"]["development_cohort_fit"]
    bindings = metadata["refit_replay_inputs"]
    for name, expected in extra.items():
        if name in bindings and bindings[name] != expected:
            raise ValueError("State and registered summary bind different inputs")
        bindings[name] = expected
    for relative in (COHORT, REGISTRATION, ENTRY):
        bindings[str(ROOT / relative)] = cohort.digest(ROOT / relative)
    metadata.update(development_cohort_scope=SCOPE, development_cohort_fit=fit_source,
                    development_manifest_sha256=cohort.digest(summary / "development_manifest.json"))
    return files, metadata


@contextmanager
def registered_observer():
    previous = {name: getattr(cohort, name) for name in
                ("ENTRY", "check_filter", "source_jobs", "estimation_context")}
    cohort.ENTRY = ENTRY
    cohort.check_filter = check_filter
    cohort.source_jobs = source_jobs
    cohort.estimation_context = estimation_context
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(cohort, name, value)


def worker(file: Path) -> None:
    job = cohort.load(file)
    if (job.get("development_cohort_scope") != SCOPE
            or not job.get("development_cohort_fit")
            or not job.get("development_manifest_sha256")):
        raise ValueError("Worker requires the registered development replay provenance")
    cohort.worker(file)


def run(summary: Path, output: Path, workers: int = 128, smoke: bool = False) -> None:
    with registered_observer():
        cohort.run(summary, output, workers, smoke)


def estimation_context(protocol: dict) -> tuple[dict, str]:
    if (protocol.get("development_cohort_scope") != SCOPE
            or not protocol.get("development_cohort_fit")
            or not protocol.get("development_manifest_sha256")):
        raise ValueError("Oral checks require the registered development selection provenance")
    metadata, note = _estimation_context(protocol)
    metadata.update({key: protocol[key] for key in
                     ("development_cohort_scope", "development_cohort_fit", "development_manifest_sha256")})
    return metadata, note + (
        "<p>口述提供带噪声的外部支持。遗漏、参照歧义及表达误差允许造成局部偏差；"
        "支持、冲突、未确定质量与完整编码覆盖率共同报告，机制判断同时结合行为和自主学习过程。</p>")


def align(source: Path, oral_audit: Path, output: Path) -> None:
    with registered_observer():
        cohort.align(source, oral_audit, output)


def compare(original: Path, revised: Path, output: Path, subject: int, arm: str) -> None:
    """Apply the existing paired oral scores with the registered state audit."""
    from src.Bayesian_state.workflows.analysis import model_0923_cohort_belief_oral as oral
    previous = oral.states
    with registered_observer():
        oral.states = cohort
        try:
            oral.run(original, revised, output, subject, arm)
        finally:
            oral.states = previous
    source = Path(__file__).resolve()
    shutil.copy2(source, output / "source" / source.name)
    cohort._replay.write(output / "development_oral_manifest.json", dict(
        status="complete", scope=SCOPE, subject=subject, arm=arm,
        cohort_manifest_sha256=cohort.digest(output / "cohort_manifest.json"),
        numerical_manifest_sha256=cohort.digest(output / "manifest.json"),
        source_sha256={ENTRY: cohort.digest(source)},
        interpretation="Paired support, conflict and undetermined mass under one fixed oral reading; noisy external evidence"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("replay", "align", "compare"), default="replay")
    for name in ("summary", "source", "oral-audit", "output", "worker", "original", "revised"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--subject", type=int)
    parser.add_argument("--arm", choices=("POOL", "SEARCH"))
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == "replay" and args.summary and args.output:
        run(args.summary, args.output, args.workers, args.smoke)
    elif args.phase == "align" and args.source and args.oral_audit and args.output and not args.smoke:
        align(args.source, args.oral_audit, args.output)
    elif (args.phase == "compare" and args.original and args.revised and args.output
          and args.subject and args.arm and not args.smoke):
        compare(args.original, args.revised, args.output, args.subject, args.arm)
    else:
        parser.error("Supply the inputs for replay, align or compare and a new --output")


if __name__ == "__main__":
    main()
