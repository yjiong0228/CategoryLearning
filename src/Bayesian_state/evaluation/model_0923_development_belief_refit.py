"""Summarize registered development participants with the common belief scores.

Use --phase summary --fit FIT --evaluation EVALUATION --output NEW, then
--phase compare --online-summary OLD --separate-summary NEW --output NEW.
The participant registration extends the original cohort's provenance checks.
Numerical scores, execution audits, trial boundaries and paired uncertainty
calculations use the existing cohort implementation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil

from . import model_0923_cohort_belief_comparison as comparison
from . import model_0923_cohort_belief_refit as cohort
from .model_0923_h_validation import digest, load
from ..workflows.analysis import model_0923_development_belief_refit as registration
from ..workflows.analysis.model_0923_h_validation import write

ROOT = registration.ROOT


def registered_inputs(fit: Path, evaluation: Path) -> dict[str, str]:
    """Check the added participant scope on every declared scientific worker."""
    frozen = load(fit / "nominees.json")
    settings = frozen["settings"]
    declaration = load(fit / "development_cohort_protocol.json")
    if (declaration.get("scope") != registration.SCOPE
            or declaration.get("subjects") != settings["subjects"]
            or declaration.get("development_cohort_fit") != settings["development_cohort_fit"]):
        raise ValueError("Participant registration differs from the fitted cohort")
    inputs = dict(declaration["input_sha256"])
    inputs[str((fit / "development_cohort_protocol.json").resolve())] = digest(
        fit / "development_cohort_protocol.json")
    registered_source = ROOT / registration.ENTRY
    expected_source = frozen["source_sha256"].get(registration.ENTRY)
    if expected_source is None or digest(registered_source) != expected_source:
        raise ValueError("Fitted participant registration source changed")
    inputs[str(registered_source)] = expected_source
    for source in (fit, evaluation):
        jobs = sorted((source / "jobs").glob("*.json"))
        if not jobs:
            raise ValueError("Both fit and evaluation require executed workers")
        for path in jobs:
            job = load(path)
            if (job.get("development_cohort_scope") != registration.SCOPE
                    or job.get("development_cohort_fit") != settings["development_cohort_fit"]
                    or job.get("source_sha256") != frozen["source_sha256"]):
                raise ValueError("Worker lost its registered development source")
            inputs[str(path.resolve())] = digest(path)
    verify_inputs(inputs)
    return inputs


def verify_inputs(inputs: dict[str, str]) -> None:
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Registered development input changed: {name}")


@contextmanager
def registered_policy():
    """Restore the common summary's driver after the local provenance adapter."""
    driver = cohort.driver
    previous = {name: getattr(driver, name) for name in
                ("CONFIG_KEYS", "prepare_settings", "configure")}
    driver.CONFIG_KEYS = registration.CONFIG_KEYS
    driver.prepare_settings = registration.prepare_settings
    driver.configure = registration.configure
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(driver, name, value)


def save_registration(output: Path, inputs: dict[str, str], *, role: str) -> None:
    verify_inputs(inputs)
    source = Path(__file__).resolve()
    shutil.copy2(source, output / "development_analysis_source.py")
    write(output / "development_manifest.json", dict(
        status="complete", role=role, scope=registration.SCOPE,
        inputs_sha256=inputs, source_sha256={str(source.relative_to(ROOT)): digest(source)},
        generic_manifest_sha256=digest(output / "manifest.json"),
        interpretation="Existing development participants; training-only parameters; oral excluded from fitting"))


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new registered development summary directory")
    inputs = registered_inputs(fit, evaluation)
    with registered_policy():
        cohort.summarize(fit, evaluation, output)
    inputs[str((output / "cohort_manifest.json").resolve())] = digest(output / "cohort_manifest.json")
    save_registration(output, inputs, role="summary")


def bind_registration(summary: Path) -> dict[str, str]:
    """Bind the extension as well as the original cohort execution audit."""
    file = summary / "development_manifest.json"
    extra = load(file)
    if (extra.get("status") != "complete" or extra.get("role") != "summary"
            or extra.get("scope") != registration.SCOPE
            or extra.get("generic_manifest_sha256") != digest(summary / "manifest.json")
            or extra.get("source_sha256") != {
                str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__))}):
        raise ValueError("Require the intact registered development summary")
    inputs = dict(extra["inputs_sha256"])
    inputs[str(file.resolve())] = digest(file)
    inputs[str(Path(__file__).resolve())] = digest(Path(__file__))
    verify_inputs(inputs)
    return inputs


def compare(online_summary: Path, separate_summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Use a new registered development comparison directory")
    inputs = {}
    for source in (online_summary, separate_summary):
        for name, expected in bind_registration(source).items():
            if name in inputs and inputs[name] != expected:
                raise ValueError("Development summaries bind different versions of a common input")
            inputs[name] = expected
    comparison.compare(online_summary, separate_summary, output)
    save_registration(output, inputs, role="comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("summary", "compare"), required=True)
    for name in ("fit", "evaluation", "online-summary", "separate-summary", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.phase == "summary" and args.fit and args.evaluation and args.output:
        summarize(args.fit, args.evaluation, args.output)
    elif args.phase == "compare" and args.online_summary and args.separate_summary and args.output:
        compare(args.online_summary, args.separate_summary, args.output)
    else:
        parser.error("Provide both inputs for the selected phase and a new output directory")


if __name__ == "__main__":
    main()
