"""Fit the frozen recommended candidate on the registered extension sample.

Use --phase search --config YAML --output NEW, then evaluate with --fit FIT,
and summary with --fit FIT --evaluation EVALUATION. Formal fits require the
recommendation and completed nine-cell evidence. --smoke requires a separate
software_validation config and executes the short original five-stage policy
on one participant. It cannot produce a scientific evaluation.

Only the recommended proposal/precision combination is fitted. The saved
support-only starts replace subject warm starts. Existing proposal generation,
particle scoring, shortlists, autonomous simulation and summary metrics are
reused. The exchange stage re-scores the leading candidates and required
neutral reference of this single arm at its original larger particle budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
COHORT = "src/Bayesian_state/workflows/analysis/model_0923_cohort_belief_refit.py"
IO = "src/Bayesian_state/workflows/analysis/model_0923_h_validation.py"
SCOPES = {"software_validation", "extension_validation"}
KEYS = {"setup", "recommendation", "subjects", "parallel_budget", "scope", "software_candidate"}
CANDIDATES = {"SEARCH_ADD10": ("SEARCH", "fast", 10.),
              "SEARCH_LOG25": ("SEARCH", "log", 1.), "POOL_LOG25": ("POOL", "log", 1.)}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Scientific workers import the archived cognitive package. Load this adapter's
# current orchestration by file, as in the existing common-policy entry points.
cohort = load_module("extension_common_cohort", ROOT / COHORT)
io = load_module("extension_fitting_io", ROOT / IO)
read, write, digest = io.read, io.write, io.digest


def validate_recommendation(recommendation: dict, evidence: dict) -> str:
    """A formal extension evaluates one decision made from the complete matrix."""
    candidate = recommendation.get("candidate")
    if (recommendation.get("status") != "frozen_recommendation" or candidate not in CANDIDATES
            or (recommendation.get("common_h"), recommendation.get("precision_storage"),
                recommendation.get("execution")) != ("ONLINE", "RETAIN", "MIX")
            or not isinstance(recommendation.get("rationale"), str) or not recommendation["rationale"].strip()
            or recommendation.get("extension_subjects") != [110, 209, 309]
            or recommendation.get("previous_extension_influence") != "unknown"
            or evidence.get("status") != "complete" or evidence.get("complete_primary_matrix") is not True
            or evidence.get("subjects") != [108, 205, 306]
            or evidence.get("collected_cells") != 9 or evidence.get("missing_subjects") != []):
        raise ValueError("Require a frozen recommendation based on all nine registered cells")
    return candidate


def selected_banks(setup: dict, settings: dict, smoke: bool) -> tuple[dict, dict]:
    """Use the immutable neutral banks, with six starts only in software checks."""
    from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
    arm, = settings["arms"]
    rows = {r["subject"]: r for r in setup["banks"]}
    if len(rows) != len(setup["banks"]) or setup["count_per_subject"] != 40:
        raise ValueError("Missing or duplicated registered neutral banks")
    banks, required = {}, {}
    for subject in settings["subjects"]:
        registered = rows[subject]
        points = registered["rows"]
        if len(points) != 40 or len({p["id"] for p in points}) != 40:
            raise ValueError("Require forty distinct saved starting points")
        for point in points:
            if candidate(point["hyperparams"], "identity_check")["id"] != point["id"]:
                raise ValueError("Saved starting point ID differs from its parameters")
        if registered["required_reference_id"] != points[0]["id"]:
            raise ValueError("Required neutral reference changed")
        banks[subject, arm] = deepcopy(points[:6] if smoke else points)
        required[subject, arm] = deepcopy(points[0])
    return banks, required


def prepare_settings(config: dict, smoke: bool) -> tuple[dict, dict, dict]:
    """Bind the saved setup and recommendation before opening a fit directory."""
    if (set(config) != KEYS or config["scope"] not in SCOPES
            or (config["scope"] == "software_validation") is not smoke
            or type(config["parallel_budget"]) is not int or not 1 <= config["parallel_budget"] <= 128):
        raise ValueError("Declare the extension scope; software validation requires --smoke")
    subjects = config["subjects"]
    if (not isinstance(subjects, list) or not subjects or len(set(subjects)) != len(subjects)
            or any(type(s) is not int or s not in (110, 209, 309) for s in subjects)
            or (smoke and (len(subjects) != 1 or config["parallel_budget"] != 1))
            or (not smoke and subjects != [110, 209, 309])):
        raise ValueError("Use the registered three participants, or one single-worker software case")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f"Frozen extension input changed: {path}")
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError("Extension input changed while restoring settings")
        inputs[str(path)] = sha
        return path

    setup = ROOT / config["setup"]
    done = read(bind(setup / "completion.json"))
    if (done["status"] != "complete" or done["subjects"] != [110, 209, 309]
            or done["parameter_fitting_started"] is not False or done["recommendation_freeze_required"] is not True
            or done["previous_development_influence"] != "unknown"):
        raise ValueError("Require the completed registered extension setup")
    for name, sha in done["inputs_sha256"].items():
        bind(name, sha)
    for name, sha in done["artifacts_sha256"].items():
        bind(setup / name, sha)
    prepared = read(bind(setup / "protocol.json"))
    neutral = read(bind(setup / "initial_banks.json"))
    if any(neutral[k] is not False for k in ("training_choices_used", "later_choices_used", "oral_contents_used")):
        raise ValueError("Extension starts must depend only on the registered support")
    if smoke:
        if config["recommendation"] is not None or config["software_candidate"] not in CANDIDATES:
            raise ValueError("Name the software candidate explicitly and keep recommendation unset")
        chosen = config["software_candidate"]
    else:
        if config["software_candidate"] is not None or not config["recommendation"]:
            raise ValueError("Formal extension fits require the frozen recommendation")
        recommendation = read(bind(ROOT / config["recommendation"]))
        evidence_path = bind(ROOT / recommendation["evidence_manifest"], recommendation["evidence_manifest_sha256"])
        evidence = read(evidence_path)
        chosen = validate_recommendation(recommendation, evidence)
        for name, sha in evidence["artifacts_sha256"].items():
            bind(evidence_path.parent / name, sha)
        for name, sha in evidence["inputs_sha256"].items():
            bind(name, sha)
    policy = ROOT / prepared["settings"]["policy_reference_fit"]
    protocol = read(bind(policy / "protocol.json"))
    reference = protocol["settings"]
    if (reference["belief_intervention"] != cohort.INTERVENTIONS["ONLINE"]
            or reference["structural_intervention"] != "retained_rule_precision"
            or reference["initial_candidates"] != 40):
        raise ValueError("The registered common numerical policy changed")
    arm, mode, scale = CANDIDATES[chosen]
    replaced = cohort.REFERENCE_KEYS | {"warm_start_fit", "initial_ray_coordinates"}
    settings = {k: deepcopy(v) for k, v in reference.items() if k not in replaced}
    arm_settings = deepcopy(reference["arms"][arm])
    arm_settings.update(precision=mode, rate_scale=scale)
    settings.update(subjects=subjects, parallel_budget=config["parallel_budget"],
        data=prepared["settings"]["data"], audit=config["setup"], arms={arm: arm_settings},
        scope="registered_extension_behavior_fit", belief_mechanism="ONLINE",
        extension_scope=config["scope"], extension_candidate=chosen, extension_setup=config["setup"],
        extension_recommendation=config["recommendation"], policy_reference_fit=prepared["settings"]["policy_reference_fit"],
        extension_initialization="saved_support_midpoint_capacity_rays_stratified",
        previous_development_influence="unknown")
    modules = cohort.helpers()
    _, online, _, recalled = modules
    actual_space = online.fixed_space(recalled.base.model_space(settings, arm), settings["fixed_coordinates"])
    registered_space = read(bind(setup / "parameter_spaces.json"))[arm]
    if actual_space != registered_space:
        raise ValueError("Recommended version changed the registered parameter support")
    banks, _ = selected_banks(neutral, settings, smoke)
    for rows in banks.values():
        for point in rows:
            if (set(point["hyperparams"]) != set(actual_space)
                    or any(point["hyperparams"][k] not in v for k, v in actual_space.items())):
                raise ValueError("Neutral point is outside the declared support")
    return settings, inputs, neutral


def check_job(job: dict) -> None:
    chosen, scope = job.get("extension_candidate"), job.get("extension_scope")
    if chosen not in CANDIDATES or scope not in SCOPES or (scope == "software_validation") is not job.get("smoke"):
        raise ValueError("Worker lost its recommendation or software-validation scope")
    arm, mode, scale = CANDIDATES[chosen]
    if (job["model"] != arm or
            (job["arm_settings"]["precision"], job["arm_settings"]["rate_scale"], job["arm_settings"]["upper"])
            != (mode, scale, 25.)):
        raise ValueError("Worker differs from the frozen proposal/precision candidate")
    if scope == "extension_validation" and not job.get("extension_recommendation"):
        raise ValueError("Formal worker lacks the recommendation binding")


def configure(settings: dict, inputs: dict, config: dict):
    """Reuse the original cognitive worker with explicit extension provenance."""
    modules = cohort.helpers()
    recalled = modules[-1]
    original_archive = recalled.base.archive
    cohort.ENTRY = ENTRY
    cohort.configure(modules, settings, inputs)
    old_hashes, old_execute = recalled.base.source_hashes, recalled.base.execute

    def hashes(v):
        return {**old_hashes(v), COHORT: v.digest(ROOT / COHORT), ENTRY: v.digest(ROOT / ENTRY)}

    def execute(jobs, output, budget):
        tagged = [dict(deepcopy(j), extension_scope=settings["extension_scope"],
            extension_candidate=settings["extension_candidate"], extension_setup=settings["extension_setup"],
            extension_recommendation=settings["extension_recommendation"]) for j in jobs]
        for job in tagged:
            check_job(job)
        old_execute(tagged, output, budget)

    def archive(output, expanded_config, source_hashes):
        original_archive(output, expanded_config, source_hashes)
        write(output / "extension_protocol.json", dict(config=config, input_sha256=inputs,
            scope=settings["extension_scope"], candidate=settings["extension_candidate"],
            initialization="Forty saved support-only starts per participant; six in software validation",
            selection="Original training behavior sequence likelihood; oral excluded",
            search="Single recommended arm; common per-arm stage limits, particles and seed policy",
            exchange="Own top candidates plus required neutral reference, re-scored at the exchange budget",
            previous_development_influence="unknown"))

    recalled.base.source_hashes, recalled.base.execute, recalled.base.archive = hashes, execute, archive
    return recalled


def worker(file: Path) -> None:
    check_job(read(file))
    cohort.worker(file)


def search(config_file: Path, output: Path, smoke: bool) -> None:
    from src.Bayesian_state.optimization.search.adaptive_proposals import propose_round, select_elites
    from src.Bayesian_state.utils.seeding import stable_seed
    if output.exists():
        raise FileExistsError("Use a new extension fitting directory")
    config = yaml.safe_load(config_file.read_text())
    settings, inputs, neutral = prepare_settings(config, smoke)
    inputs[str(config_file.resolve())] = digest(config_file)
    recalled = configure(settings, inputs, config)
    base, data = recalled.base, ROOT / settings["data"]
    banks, required = selected_banks(neutral, settings, smoke)
    seen = {key: {r["id"] for r in rows} for key, rows in banks.items()}
    output.mkdir(parents=True)
    hashes = base.source_hashes(base.validation())
    with tempfile.TemporaryDirectory(prefix="extension_fitting_") as temporary:
        expanded = Path(temporary) / "config.yaml"
        expanded.write_text(yaml.safe_dump(settings, sort_keys=False))
        base.archive(output, expanded, hashes)
    write(output / "parameter_spaces.json", {arm: base.model_space(settings, arm) for arm in settings["arms"]})
    write(output / "protocol.json", dict(settings=settings, smoke=smoke, source_sha256=hashes,
        data_manifest_sha256=digest(data / "preparation_manifest.json"),
        data_boundary="Original training choices only; later choices and oral reserved for evaluation",
        initial_banks=[dict(subject=k[0], model=k[1], rows=v) for k, v in banks.items()],
        python=sys.version, versions={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pandas", "PyYAML")}))
    screened = base.stage(settings, data, output, banks, "screen", smoke)
    for key, rows in screened.items():
        elites = select_elites(rows, 1 if smoke else settings["elites"], exclude_key=io.H_PATH + "max_capacity")
        proposals = propose_round(elites, base.model_space(settings, key[1]), seen[key],
            1 if smoke else settings["proposals_per_elite"],
            stable_seed(dict(base=settings["seed"], subject=key[0], role="joint_refine")), .5, .25)
        if len(proposals) != (1 if smoke else settings["elites"] * settings["proposals_per_elite"]):
            raise ValueError("Joint refinement did not produce its declared candidate count")
        banks[key] = proposals + elites
        seen[key].update(r["id"] for r in proposals)
    refined = base.stage(settings, data, output, banks, "refine", smoke)
    leading = {key: sorted({r["id"]: r for r in screened[key] + refined[key]}.values(),
                           key=lambda r: (r["mean_nll"], r["id"])) for key in screened}
    banks = recalled.refit.exchange_banks(leading, required, 2 if smoke else settings["exchange_per_arm"])
    for key, rows in banks.items():
        seen[key].update(r["id"] for r in rows)
    rows = base.stage(settings, data, output, banks, "exchange", smoke)
    for stage, count in (("selection", settings["selection_candidates"]),
                         ("confirmation", settings["confirmation_candidates"])):
        banks = {key: base.shortlist(values, required[key], 2 if smoke else min(count, len(values)))
                 for key, values in rows.items()}
        rows = base.stage(settings, data, output, banks, stage, smoke)
    for path, sha in {**inputs, **{str(ROOT / k): v for k, v in hashes.items()}}.items():
        if digest(Path(path)) != sha:
            raise ValueError("Extension inputs changed during fitting")
    nominees = [dict(subject=k[0], model=k[1], candidate=v[0], finalists=v,
        unique_candidates=len(seen[k]), reference_id=required[k]["id"]) for k, v in rows.items()]
    write(output / "nominees.json", dict(status="frozen_before_test", settings=settings, smoke=smoke,
        source_sha256=hashes, data=str(data.resolve()), nominees=nominees))
    write(output / "completion.json", dict(status="complete", smoke=smoke, test_choices_used=False,
        nomination_sha256=digest(output / "nominees.json"), extension_scope=settings["extension_scope"]))


def restore(fit: Path) -> tuple[dict, dict, dict]:
    frozen, done, declaration = [read(fit / n) for n in ("nominees.json", "completion.json", "extension_protocol.json")]
    if (done["status"] != "complete" or done["smoke"] or frozen["smoke"]
            or done["test_choices_used"] is not False or frozen["status"] != "frozen_before_test"
            or done["nomination_sha256"] != digest(fit / "nominees.json")):
        raise ValueError("Scientific evaluation requires a complete full-budget extension fit")
    config = declaration["config"]
    settings, inputs, neutral = prepare_settings(config, False)
    if settings != frozen["settings"] or declaration["scope"] != "extension_validation":
        raise ValueError("Extension settings differ from their frozen policy")
    for path, sha in declaration["input_sha256"].items():
        if digest(Path(path)) != sha or (path in inputs and inputs[path] != sha):
            raise ValueError("Extension fitting source changed")
        inputs[path] = sha
    protocol = read(fit / "protocol.json")
    banks, _ = selected_banks(neutral, settings, False)
    expected = {(r["subject"], r["model"]): r["rows"] for r in protocol["initial_banks"]}
    if (banks != expected or len(expected) != len(protocol["initial_banks"])
            or protocol["settings"] != settings or protocol["source_sha256"] != frozen["source_sha256"]):
        raise ValueError("Executed initial banks or fitting policy changed")
    return settings, inputs, config


def summarize(fit: Path, evaluation: Path, output: Path) -> None:
    from src.Bayesian_state.evaluation import model_0923_cohort_belief_refit as audit
    from src.Bayesian_state.evaluation import model_0923_episodic_refit as metrics
    if output.exists():
        raise FileExistsError("Use a new extension summary directory")
    settings, inputs, _ = restore(fit)
    arm, = settings["arms"]

    def bind(path):
        path = Path(path).resolve()
        sha = digest(path)
        if str(path) in inputs and inputs[str(path)] != sha:
            raise ValueError("Extension input changed during summary")
        inputs[str(path)] = sha
        return path

    for name in ("extension_protocol.json", "protocol.json", "nominees.json", "completion.json"):
        bind(fit / name)
    evaluation_protocol = read(bind(evaluation / "protocol.json"))
    completed = read(bind(evaluation / "completion.json"))
    if (completed["status"] != "complete"
            or evaluation_protocol["nomination_sha256"] != digest(fit / "nominees.json")
            or completed["nomination_sha256"] != digest(fit / "nominees.json")):
        raise ValueError("Evaluation did not use the frozen extension nominees")
    plans = {j["name"]: j for j in evaluation_protocol["jobs"]}
    files = sorted((evaluation / "jobs").glob("*.json"))
    if (len(plans) != len(evaluation_protocol["jobs"]) or {f.stem for f in files} != set(plans)
            or len(files) != completed["jobs"]):
        raise ValueError("Extension evaluation is incomplete")
    for file in files:
        actual = read(bind(file))
        if any(actual.get(k) != v for k, v in plans[file.stem].items()):
            raise ValueError("Evaluation changed its frozen execution plan")
    for directory in (fit, evaluation):
        for file in (directory / "jobs").glob("*.json"):
            job = read(bind(file))
            check_job(job)
            if (job["extension_scope"] != "extension_validation"
                    or job["extension_candidate"] != settings["extension_candidate"]):
                raise ValueError("Executed worker differs from the recommended candidate")
    audits = {role: audit.audit_jobs(path, "ONLINE", bind) for role, path in (("fit", fit), ("evaluation", evaluation))}
    if (any(row["phase"] != "training" for row in audits["fit"]["rows"])
            or {row["phase"] for row in audits["evaluation"]["rows"]} != {"evaluate", "autonomous"}):
        raise ValueError("Training and evaluation roles were mixed")
    metrics.summarize(fit, evaluation, output, reference_model=arm)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError("Extension input changed during scoring")
    write(output / "execution_audit.json", audits)
    shutil.copy2(__file__, output / "extension_analysis_source.py")
    write(output / "extension_manifest.json", dict(status="complete", candidate=settings["extension_candidate"],
        scope="extension_validation", subjects=settings["subjects"], previous_development_influence="unknown",
        recommendation=settings["extension_recommendation"], input_sha256=inputs,
        generic_manifest_sha256=digest(output / "manifest.json"),
        execution_audit_sha256=digest(output / "execution_audit.json"),
        interpretation="Frozen recommended version; original behavior-prefix estimates; later behavior and noisy oral checks"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "evaluate", "summary"))
    for name in ("config", "fit", "evaluation", "output", "worker"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase == "search" and args.config and args.output:
        search(args.config, args.output, args.smoke)
    elif args.phase == "evaluate" and args.fit and args.output and not args.smoke:
        settings, inputs, config = restore(args.fit)
        configure(settings, inputs, config).base.evaluate(args.fit, args.output)
    elif args.phase == "summary" and args.fit and args.evaluation and args.output and not args.smoke:
        summarize(args.fit, args.evaluation, args.output)
    else:
        parser.error("Use search with config, evaluate with fit, or summary with fit/evaluation and a new output")
