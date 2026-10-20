"""Joint H/precision fits using the completed common development policy.

Run --phase search --config YAML --mechanism ONLINE|SEPARATE --output NEW
[--smoke], then --phase evaluate --fit FIT --output NEW and --phase summary
--fit FIT --evaluation EVALUATION --output NEW.

ADD10 and LOG25 retain the reference parameter supports, starting points,
training boundaries, seed namespaces and search limits. The precision update
and its rate scale differ. Every active H, memory and precision coordinate
remains available to the original joint search. POOL and SEARCH share starts.
Oral reports enter subsequent evaluation. Software-validation configurations
require --smoke; scientific candidate selection is registered separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
REGISTRATION = "src/Bayesian_state/workflows/analysis/model_0923_development_belief_refit.py"
spec = importlib.util.spec_from_file_location("finalization_joint_registration", ROOT / REGISTRATION)
registration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = registration
spec.loader.exec_module(registration)
CONFIG_KEYS = registration.CONFIG_KEYS | {"finalization_precision_family", "finalization_scope"}
FAMILIES = {"ADD10": ("fast", 10.), "LOG25": ("log", 1.)}
SCOPES = {"software_validation", "development_comparison"}


def precision_settings(reference: dict, family: str, scope: str) -> dict:
    """Change the precision definition while preserving every search coordinate."""
    if family not in FAMILIES or scope not in SCOPES:
        raise ValueError("Declare ADD10 or LOG25 and the scientific/software scope")
    if list(reference["arms"]) != ["POOL", "SEARCH"]:
        raise ValueError("Require the common POOL and SEARCH policy")
    result = deepcopy(reference)
    for arm, settings in result["arms"].items():
        if (settings["precision"], settings["rate_scale"], settings["upper"]) != ("fast", 10., 25.):
            raise ValueError("The common reference must use ADD10 with ceiling 25")
        settings["precision"], settings["rate_scale"] = FAMILIES[family]
    result.update(finalization_precision_family=family, finalization_scope=scope)
    return result


def prepare_settings(config: dict, mechanism: str, modules: tuple) -> tuple[dict, dict]:
    """Validate the original policy before declaring the precision comparison."""
    if set(config) != CONFIG_KEYS:
        raise ValueError("Specify the registered development policy, precision family and scope")
    original, inputs = registration.prepare_settings(
        {key: config[key] for key in registration.CONFIG_KEYS}, mechanism, modules)
    settings = precision_settings(
        original, config["finalization_precision_family"], config["finalization_scope"])
    v = modules[-1].base.validation()
    inputs[str((ROOT / REGISTRATION).resolve())] = v.digest(ROOT / REGISTRATION)
    return settings, inputs


def check_job(job: dict) -> None:
    """Check the declared precision definition before entering the actual worker."""
    family = job.get("finalization_precision_family")
    if family not in FAMILIES or job.get("finalization_scope") not in SCOPES:
        raise ValueError("Missing joint precision experiment identity")
    if job["finalization_scope"] == "software_validation" and job.get("smoke") is not True:
        raise ValueError("Software-validation jobs require smoke execution")
    arm = job["arm_settings"]
    if (arm["precision"], arm["rate_scale"], arm["upper"]) != (*FAMILIES[family], 25.):
        raise ValueError("Executed precision differs from its declared family")
    if job.get("development_cohort_scope") != registration.SCOPE:
        raise ValueError("Missing registered development participant scope")


def worker(file: Path) -> None:
    modules = registration.cohort.helpers()
    check_job(modules[-1].base.validation().read(file))
    registration.cohort.worker(file)


def configure(modules: tuple, settings: dict, inputs: dict) -> None:
    """Reuse the common search and worker; append experiment provenance."""
    registration.ENTRY = ENTRY
    registration.configure(modules, settings, inputs)
    recalled = modules[-1]
    old_hashes, old_execute, old_archive = (
        recalled.base.source_hashes, recalled.base.execute, recalled.base.archive)

    def hashes(v):
        result = old_hashes(v)
        result[REGISTRATION] = v.digest(ROOT / REGISTRATION)
        return result

    def execute(jobs, output, budget):
        tagged = [dict(deepcopy(job),
                       finalization_precision_family=settings["finalization_precision_family"],
                       finalization_scope=settings["finalization_scope"]) for job in jobs]
        old_execute(tagged, output, budget)

    def archive(output, config, source_hashes):
        old_archive(output, config, source_hashes)
        recalled.base.validation().write(output / "finalization_joint_protocol.json", dict(
            precision_family=settings["finalization_precision_family"],
            scope=settings["finalization_scope"], input_sha256=inputs,
            parameter_selection="Original training choices only; all active coordinates jointly searched",
            comparison="Common supports, initial banks, seeds and stage limits; actual adaptive work recorded",
            rate_interpretation="ADD10 scales the additive rate by ten; LOG25 updates log precision",
            oral_used_for_parameter_selection=False))

    recalled.base.source_hashes = hashes
    recalled.base.execute = execute
    recalled.base.archive = archive


@contextmanager
def summary_policy():
    """Supply the new provenance adapter to the existing numerical summary."""
    from ...evaluation import model_0923_cohort_belief_refit as scores
    driver = scores.driver
    original = {name: getattr(driver, name) for name in ("CONFIG_KEYS", "prepare_settings", "configure")}
    driver.CONFIG_KEYS, driver.prepare_settings, driver.configure = CONFIG_KEYS, prepare_settings, configure
    try:
        yield scores
    finally:
        for name, value in original.items():
            setattr(driver, name, value)


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    from ...evaluation import model_0923_development_belief_refit as development
    from ...evaluation.model_0923_h_validation import digest, load
    from .model_0923_h_validation import write
    if output.exists():
        raise FileExistsError("Use a new finalization summary directory")
    inputs = development.registered_inputs(fit, evaluation)
    declaration = load(fit / "finalization_joint_protocol.json")
    frozen = load(fit / "nominees.json")
    if (declaration["scope"] != "development_comparison"
            or declaration["precision_family"] != frozen["settings"]["finalization_precision_family"]
            or declaration["oral_used_for_parameter_selection"] is not False):
        raise ValueError("Require the declared scientific joint precision fit")
    inputs[str((fit / "finalization_joint_protocol.json").resolve())] = digest(
        fit / "finalization_joint_protocol.json")
    for source in (fit, evaluation):
        for file in sorted((source / "jobs").glob("*.json")):
            job = load(file)
            check_job(job)
            if (job["finalization_precision_family"] != declaration["precision_family"]
                    or job["finalization_scope"] != declaration["scope"]):
                raise ValueError("Worker family or scope differs from the fitted experiment")
    with summary_policy() as scores:
        scores.summarize(fit, evaluation, output)
    development.verify_inputs(inputs)
    write(output / "finalization_manifest.json", dict(
        status="complete", precision_family=declaration["precision_family"],
        scope=declaration["scope"], inputs_sha256=inputs,
        source_sha256={ENTRY: digest(ROOT / ENTRY)},
        cohort_manifest_sha256=digest(output / "cohort_manifest.json"),
        interpretation="Joint behavior-only fit under the common finite policy; development evaluation"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "evaluate", "summary"))
    parser.add_argument("--mechanism", choices=tuple(registration.cohort.INTERVENTIONS))
    for name in ("config", "fit", "evaluation", "output", "worker"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.phase == "summary" and args.fit and args.evaluation and args.output and not args.smoke:
        summarize(args.fit, args.evaluation, args.output)
        return
    modules = registration.cohort.helpers()
    v = modules[-1].base.validation()
    if args.phase == "search" and args.config and args.mechanism and args.output:
        settings, inputs = prepare_settings(yaml.safe_load(args.config.read_text()), args.mechanism, modules)
        if settings["finalization_scope"] == "software_validation" and not args.smoke:
            raise ValueError("Software-validation configurations require --smoke")
        inputs[str(args.config.resolve())] = v.digest(args.config)
        if args.smoke:
            settings["parallel_budget"] = 1
        configure(modules, settings, inputs)
        with tempfile.TemporaryDirectory(prefix="finalization_joint_") as cache:
            expanded = Path(cache) / "config.yaml"
            expanded.write_text(yaml.safe_dump(settings, sort_keys=False))
            modules[-1].search(expanded, args.output, args.smoke)
    elif args.phase == "evaluate" and args.fit and args.output and not args.smoke:
        frozen = v.read(args.fit / "nominees.json")["settings"]
        settings, inputs = prepare_settings(
            {key: frozen[key] for key in CONFIG_KEYS}, frozen["belief_mechanism"], modules)
        if settings != frozen or settings["finalization_scope"] != "development_comparison":
            raise ValueError("Frozen scientific settings differ from the declared policy")
        inputs.update(v.read(args.fit / "finalization_joint_protocol.json")["input_sha256"])
        configure(modules, settings, inputs)
        modules[-1].base.evaluate(args.fit, args.output)
    else:
        parser.error("Use search with config/mechanism, evaluate with fit, or summary with fit/evaluation")


if __name__ == "__main__":
    main()
