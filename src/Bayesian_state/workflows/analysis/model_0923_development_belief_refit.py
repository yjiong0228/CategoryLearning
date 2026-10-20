"""Extend the common belief comparison to already fitted development subjects.

Run --phase search --config YAML --mechanism ONLINE|SEPARATE --output NEW
[--smoke], followed by --phase evaluate --fit FIT --output NEW. The YAML extends
the existing cohort configuration with development_cohort_fit, a completed
four-arm behavior fit that records the eligible participants. Both mechanisms
inherit the same supports, projected warm starts, stage limits and seed policy.

This adapter extends participant provenance. It reuses the existing cohort
search and cognitive workers. The original three-person development record and
its archived inputs remain intact. Oral reports are reserved for evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = "src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_refit.py"
spec = importlib.util.spec_from_file_location("registered_development_belief_cohort", ROOT / COHORT)
cohort = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cohort
spec.loader.exec_module(cohort)
CONFIG_KEYS = cohort.CONFIG_KEYS | {"development_cohort_fit"}
SCOPE = "completed_behavior_fit_development_cohort"
WARM_ARMS = {"BASE", "NEW", "POOL", "DRAW"}


def prepare_settings(config: dict, mechanism: str, modules: tuple) -> tuple[dict, dict]:
    """Register participants from an intact training-only four-arm fit."""
    if set(config) != CONFIG_KEYS:
        raise ValueError("Specify the common cohort settings and development_cohort_fit")
    validation = modules[-1].base.validation()
    source = (ROOT / config["development_cohort_fit"]).resolve()
    inputs = {}

    def read(path):
        path = Path(path).resolve()
        inputs[str(path)] = validation.digest(path)
        return validation.read(path)

    frozen, done = [read(source / name) for name in ("nominees.json", "completion.json")]
    if (frozen["status"] != "frozen_before_test" or frozen["smoke"]
            or done["status"] != "complete" or done["smoke"]
            or done["test_choices_used"] is not False
            or done["nomination_sha256"] != inputs[str(source / "nominees.json")]):
        raise ValueError("Require an intact completed training-only development fit")
    subject_arms = {}
    for point in frozen["nominees"]:
        subject_arms.setdefault(point["subject"], set()).add(point["model"])
    eligible = {subject for subject, arms in subject_arms.items() if arms == WARM_ARMS}
    if not set(config["subjects"]) <= eligible:
        raise ValueError("Each declared participant must have all four completed development arms")
    policy = read(ROOT / config["policy_reference_fit"] / "protocol.json")
    if (ROOT / policy["settings"]["warm_start_fit"]).resolve() != source:
        raise ValueError("Development registration and common-policy warm starts must share one fit")

    # The common validator still checks all model supports, every participant's
    # training data and boundaries, and both archived implementation bindings.
    # Only the declared eligible set expands; no scientific worker is replaced.
    original_expansion = cohort.expanded_settings

    def expanded(own, reference, previous_eligible, selected_mechanism):
        return original_expansion(own, reference, previous_eligible | eligible, selected_mechanism)

    cohort.expanded_settings = expanded
    try:
        settings, inherited = cohort.prepare_settings(
            {key: config[key] for key in cohort.CONFIG_KEYS}, mechanism, modules)
    finally:
        cohort.expanded_settings = original_expansion
    for name, expected in inherited.items():
        if name in inputs and inputs[name] != expected:
            raise ValueError(f"Conflicting input identity: {name}")
        inputs[name] = expected
    settings["development_cohort_fit"] = config["development_cohort_fit"]
    inputs[str((ROOT / COHORT).resolve())] = validation.digest(ROOT / COHORT)
    return settings, inputs


def configure(modules: tuple, settings: dict, inputs: dict) -> None:
    """Add scope records while delegating numerical work to the shared cohort."""
    cohort.ENTRY = ENTRY
    cohort.configure(modules, settings, inputs)
    recalled = modules[-1]
    prior_hashes = recalled.base.source_hashes
    prior_execute = recalled.base.execute
    prior_archive = recalled.base.archive

    def hashes(validation):
        result = prior_hashes(validation)
        result[COHORT] = validation.digest(ROOT / COHORT)
        return result

    def execute(jobs, output, budget):
        tagged = [dict(deepcopy(job), development_cohort_scope=SCOPE,
                       development_cohort_fit=settings["development_cohort_fit"]) for job in jobs]
        prior_execute(tagged, output, budget)

    def archive(output, config, source_hashes):
        prior_archive(output, config, source_hashes)
        recalled.base.validation().write(output / "development_cohort_protocol.json", dict(
            scope=SCOPE, subjects=settings["subjects"],
            development_cohort_fit=settings["development_cohort_fit"],
            input_sha256=inputs,
            eligibility="All four original behavior-fit arms; membership independent of oral alignment scores",
            selection="Original training choices only; unchanged supports, starts, stage limits and seeds",
            interpretation="Existing development participants; prior influence retained"))

    recalled.base.source_hashes = hashes
    recalled.base.execute = execute
    recalled.base.archive = archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "evaluate"))
    parser.add_argument("--mechanism", choices=tuple(cohort.INTERVENTIONS))
    for name in ("config", "fit", "output", "worker"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    modules = cohort.helpers()
    validation = modules[-1].base.validation()
    if args.worker:
        job = validation.read(args.worker)
        if job.get("development_cohort_scope") != SCOPE or not job.get("development_cohort_fit"):
            raise ValueError("Require the declared development cohort scope")
        cohort.worker(args.worker)
        return
    if args.phase == "search" and args.config and args.mechanism and args.output:
        settings, inputs = prepare_settings(yaml.safe_load(args.config.read_text()), args.mechanism, modules)
        inputs[str(args.config.resolve())] = validation.digest(args.config)
        if args.smoke:
            settings["parallel_budget"] = 1
        configure(modules, settings, inputs)
        with tempfile.TemporaryDirectory(prefix="development_belief_config_") as cache:
            expanded = Path(cache) / "config.yaml"
            expanded.write_text(yaml.safe_dump(settings, sort_keys=False))
            modules[-1].search(expanded, args.output, args.smoke)
    elif args.phase == "evaluate" and args.fit and args.output and not args.smoke:
        frozen = validation.read(args.fit / "nominees.json")["settings"]
        settings, inputs = prepare_settings(
            {key: frozen[key] for key in CONFIG_KEYS}, frozen["belief_mechanism"], modules)
        if settings != frozen:
            raise ValueError("Frozen settings differ from their declared common policy")
        inputs.update(validation.read(args.fit / "development_cohort_protocol.json")["input_sha256"])
        configure(modules, settings, inputs)
        modules[-1].base.evaluate(args.fit, args.output)
    else:
        parser.error("Use search with --config --mechanism, or evaluate with --fit, and a new --output")


if __name__ == "__main__":
    main()
