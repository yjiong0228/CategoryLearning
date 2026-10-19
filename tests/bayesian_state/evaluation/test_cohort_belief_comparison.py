"""Keep cohort mechanism comparisons tied to common policies and real workers."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.Bayesian_state.evaluation import model_0923_cohort_belief_comparison as comparison


def policy_fixture():
    shared = dict(subjects=[107, 207, 306], arms=["POOL", "SEARCH"],
                  scope="common_policy_development_belief_refit", parallel_budget=64,
                  budgets=dict(evaluate=dict(seeds=8, particles=512)), seed=42)
    frozen, protocols = [], []
    for mechanism in comparison.MECHANISMS:
        settings = dict(deepcopy(shared), belief_mechanism=mechanism,
                        belief_intervention=comparison.cohort.driver.INTERVENTIONS[mechanism])
        frozen.append(dict(smoke=False, settings=settings, source_sha256={"shared.py": "hash"}))
        protocols.append(dict(smoke=False, settings=deepcopy(settings), source_sha256={"shared.py": "hash"},
            initial_banks=[dict(subject=s, model=a, rows=[dict(id=str(s), hyperparams=dict(gamma=.6))])
                           for s in shared["subjects"] for a in shared["arms"]]))
    return frozen, protocols, [dict(gamma=[0, .6, 1]), dict(gamma=[0, .6, 1])]


def test_common_policy_allows_different_worker_concurrency():
    frozen, protocols, spaces = policy_fixture()
    frozen[1]["settings"]["parallel_budget"] = 32
    protocols[1]["settings"]["parallel_budget"] = 32
    comparison.match_policy(*frozen, *protocols, *spaces)


@pytest.mark.parametrize("fault", ["subject", "budget", "seed", "support", "source", "mechanism",
                                   "initial", "duplicate", "missing", "smoke", "concurrency"])
def test_rejects_scientifically_unmatched_cohorts(fault):
    frozen, protocols, spaces = policy_fixture()
    if fault == "subject": frozen[1]["settings"]["subjects"] = [107, 207]
    elif fault == "budget": frozen[1]["settings"]["budgets"]["evaluate"]["seeds"] = 4
    elif fault == "seed": frozen[1]["settings"]["seed"] = 43
    elif fault == "support": spaces[1]["gamma"] = [0, 1]
    elif fault == "source": frozen[1]["source_sha256"]["shared.py"] = "different"
    elif fault == "mechanism": frozen[1]["settings"]["belief_mechanism"] = "ONLINE"
    elif fault == "initial": protocols[1]["initial_banks"][0]["rows"][0]["hyperparams"]["gamma"] = 1
    elif fault == "duplicate": protocols[1]["initial_banks"].append(deepcopy(protocols[1]["initial_banks"][0]))
    elif fault == "missing": protocols[1]["initial_banks"].pop()
    elif fault == "smoke": frozen[1]["smoke"] = True
    elif fault == "concurrency": frozen[1]["settings"]["parallel_budget"] = 129
    if fault in {"subject", "budget", "seed", "mechanism", "concurrency"}:
        protocols[1]["settings"] = deepcopy(frozen[1]["settings"])
    if fault == "source": protocols[1]["source_sha256"] = deepcopy(frozen[1]["source_sha256"])
    with pytest.raises(ValueError): comparison.match_policy(*frozen, *protocols, *spaces)


def audit_fixture(tmp_path, mechanism):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    source = tmp_path / "summary"; source.mkdir()
    manifest, audits, inputs, jobs = {}, {}, {}, []
    prefix = comparison.cohort.separate.PREFIX
    transfer = comparison.cohort.separate.TRANSFER
    gamma = comparison.cohort.separate.GAMMA
    intervention = comparison.cohort.driver.INTERVENTIONS[mechanism]
    for role, phase in (("fit", "training"), ("evaluation", "evaluate")):
        run = tmp_path / role; file = run / "jobs" / "worker.json"
        target = run / "stage" / "chunks" / "worker"
        job = dict(phase=phase, smoke=False, target=str(target), seeds=[1, 2], particles=4,
            cohort_belief_mechanism=mechanism, probe_mechanism=mechanism,
            belief_intervention=intervention, online_belief=mechanism == "ONLINE",
            structural_intervention="retained_rule_precision", retain_precision=True,
            fixed_coordinates={transfer: .01},
            candidate=dict(hyperparams={transfer: .01, gamma: .6, prefix + "intervention.decay": .8}))
        result = dict(trials=8, subject=107, model="SEARCH")
        belief = dict(enabled=True, belief_intervention=intervention, fixed_coordinates={transfer: .01},
            exact_transition_timing=True, evidence_matches_retained_state=True, reconstructions=56,
            gamma=.6, delta=[.8], updates=56, completed_feedback_timing_exact=True, belief_matches_saved_state=True)
        precision = dict(structural_intervention="retained_rule_precision", retained_precision=True, transitions=64)
        sidecar = ".online.json" if mechanism == "ONLINE" else ".belief.json"
        for p, value in [(file, job), (target.with_suffix(".json"), result),
                         (target.with_suffix(sidecar), belief), (target.with_suffix(".precision.json"), precision)]:
            save(p, value); inputs[str(p.resolve())] = comparison.digest(p)
        manifest[role] = str(run); jobs.append(file)
        audits[role] = dict(status="complete", jobs=1, rows=[dict(target=str(target.resolve()),
            subject=107, model="SEARCH", phase=phase, state_updates=64, runs=2)])
    save(source / "manifest.json", manifest)
    save(source / "execution_audit.json", audits)
    extra = dict(status="complete", belief_mechanism=mechanism, inputs_sha256=inputs,
        source_sha256={str(Path(p).relative_to(comparison.ROOT)): comparison.digest(Path(p))
                       for p in (comparison.cohort.__file__, comparison.cohort.driver.__file__)},
        generic_manifest_sha256=comparison.digest(source / "manifest.json"),
        execution_audit_sha256=comparison.digest(source / "execution_audit.json"))
    save(source / "cohort_manifest.json", extra)
    return source, manifest, audits, extra, jobs, save


@pytest.mark.parametrize("mechanism", comparison.MECHANISMS)
def test_binds_both_cohort_execution_formats(tmp_path, mechanism):
    source, manifest, _, _, jobs, _ = audit_fixture(tmp_path, mechanism)
    inputs = comparison.bind_summary(source, manifest, mechanism)
    assert all(str(p.resolve()) in inputs for p in jobs)


@pytest.mark.parametrize("fault", ["summary", "audit", "job", "mechanism", "scope", "count",
                                   "updates", "row_identity", "missing_bound_job", "extra_result", "auditor"])
@pytest.mark.parametrize("mechanism", comparison.MECHANISMS)
def test_rejects_changed_or_incomplete_worker_audit(tmp_path, mechanism, fault):
    source, manifest, audits, extra, jobs, save = audit_fixture(tmp_path, mechanism)
    job = json.loads(jobs[0].read_text())
    if fault == "summary": save(source / "manifest.json", {})
    elif fault == "audit": save(source / "execution_audit.json", {})
    elif fault == "job": save(jobs[0], dict(job, particles=8))
    elif fault == "mechanism": extra["belief_mechanism"] = "RETAIN"
    elif fault == "scope":
        save(jobs[0], dict(job, smoke=True)); extra["inputs_sha256"][str(jobs[0])] = comparison.digest(jobs[0])
    elif fault == "missing_bound_job": del extra["inputs_sha256"][str(jobs[0])]
    elif fault == "extra_result": save(Path(job["target"]).with_name("extra.json"), {})
    elif fault == "auditor": extra["source_sha256"] = {}
    elif fault == "updates":
        p = Path(job["target"]).with_suffix(".precision.json")
        value = json.loads(p.read_text()); value["transitions"] = 65; save(p, value)
        extra["inputs_sha256"][str(p)] = comparison.digest(p)
    else:
        if fault == "count": audits["fit"]["jobs"] = 2
        elif fault == "row_identity": audits["fit"]["rows"][0]["subject"] = 207
        save(source / "execution_audit.json", audits)
        extra["execution_audit_sha256"] = comparison.digest(source / "execution_audit.json")
    save(source / "cohort_manifest.json", extra)
    with pytest.raises(ValueError): comparison.bind_summary(source, manifest, mechanism)
