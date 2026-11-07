"""Explicit, offline access to the unchanged Python 1031-unified-v2 reference."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARAMETER_PATHS = {
    "capacity": "partition.kwargs.capacity",
    "similarity_width": "partition.kwargs.similarity_width",
    "baseline_probability": "modules.hypo_transitions_mod.kwargs.baseline_probability",
    "error_gain": "modules.hypo_transitions_mod.kwargs.error_gain",
    "failure_decay": "modules.hypo_transitions_mod.kwargs.failure_decay",
    "global_search": "modules.hypo_transitions_mod.kwargs.global_search",
    "retention": "modules.memory_mod.kwargs.gamma",
    "execution_inertia": "modules.hypo_transitions_mod.kwargs.execution_inertia",
    "beta_init": "modules.beta_mod.kwargs.beta_init",
    "learning_rate": "modules.beta_mod.kwargs.learning_rate",
}


def activate_reference(reference_root: Path) -> Path:
    """Only this explicit opt-in enables production Python imports."""
    root = reference_root.resolve(strict=True)
    if root == ROOT or not (root / "tests/bayesian_state/test_model_1031_unified.py").is_file():
        raise ValueError("--reference-root must identify the original CategoryLearning repository")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    cache = ROOT / ".reference_cache"
    cache.mkdir(exist_ok=True)
    temporary = cache / "tmp"
    temporary.mkdir(exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(temporary)
    tempfile.tempdir = str(temporary)
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        os.environ[name] = str(cache / name.lower())
    os.chdir(ROOT)
    sys.path.insert(0, str(root))
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    ContinuousSimilarity.RUNTIME_CACHE_DIR = cache / "hypothesis_space"
    return root


def load_engine(root: Path, condition: int, parameters: dict | None = None) -> dict:
    import yaml
    path = root / f"configs/exp123/model_struct/model_1031_cond{condition}_unified_v2.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["provenance"]["specification_version"] != "1031-unified-v2":
        raise ValueError("Unexpected reference specification")
    if parameters is not None:
        if set(parameters) != set(PARAMETER_PATHS):
            raise ValueError("parameters must contain exactly the ten adopted names")
        for name, dotted in PARAMETER_PATHS.items():
            target = cfg
            *parents, leaf = dotted.split(".")
            for key in parents:
                target = target.setdefault(key, {})
            target[leaf] = parameters[name]
    return cfg


def provenance(root: Path) -> dict:
    import numpy as np
    import scipy
    import yaml
    files = list((root / "src/Bayesian_state").rglob("*.py"))
    files += list((root / "configs/exp123/model_struct").glob("model_1031_cond*_unified_v2.yaml"))
    files += [root / "tests/bayesian_state/test_model_1031_unified.py"]
    return dict(expected_source_commit="249144de", source_sha256={
        p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, pyyaml=yaml.__version__))


def cases() -> list[dict]:
    """Inputs only; expected values are evaluated by original methods below."""
    result = []
    theta3 = [[.65, .15, .12, .08], [.10, .25, .55, .10], [.05, .15, .20, .60]]
    for condition in (1, 2, 3):
        theta = [.8, .2] if condition == 1 else ([.1, .2, .3, .4] if condition == 2 else theta3)
        template = dict(kind="template", anchor=[.2, .3, .4, .5],
                        mask=[True] * 4, category_belief=theta, uid=1, parent_uid=0)
        variants = [
            ("null", None, 0., [.22, .31, .42, .53], 0, 1.),
            ("uncommitted", dict(kind="uncommitted", uid=1, parent_uid=0), 2.3,
             [.25, .35, .4, .5], 1, 0.),
            ("template_correct", template, 2.3, [.25, .35, .4, .5], 0, 1.),
            ("template_error", template, .4, [.22, .31, .42, .53], 1, 0.),
            ("partial", dict(template, mask=[True, False, True, True]), 5.,
             [.21, .95, .45, .55], 1, .5 if condition == 3 else 1.),
            ("tiny_affinity", dict(template, anchor=[0.] * 4), 2.3, [1.] * 4,
             0, .5 if condition == 3 else 0.),
            ("rule_first", dict(kind="rule", rule_id=0, uid=1, parent_uid=0), 2.3,
             [.17, .36, .68, .84], 0, 1.),
            ("rule_last", dict(kind="rule", rule_id=28 if condition == 1 else 151,
                               uid=1, parent_uid=0), .7, [.71, .28, .61, .19],
             1, .5 if condition == 3 else 0.),
        ]
        if condition == 3:
            variants.append(("template_half", template, 2.3, [.25, .35, .4, .5], 2, .5))
        for name, node, beta, x, choice, feedback in variants:
            item = dict(id=f"cond{condition}_{name}", condition=condition, stimulus=x,
                        node=deepcopy(node), beta=beta, choice=choice, feedback=feedback)
            if condition == 3:
                item["pairing"] = [.15, .60, .25]
            result.append(item)
    return result


def evaluate_case(case: dict, model) -> dict:
    import numpy as np
    partition = model.partition_model
    likelihood = model.engine.observation_likelihood
    partition.nodes = [deepcopy(case["node"])]
    beta = case["beta"]
    data = ([case["stimulus"]], [], [])
    # The external/native fixture uses zero-based response columns; Python uses 1..C.
    observation = (case["stimulus"], case["choice"] + 1, case["feedback"])
    kwargs = {}
    if case["condition"] == 3:
        q = np.asarray([case["pairing"]])
        partition.pairing_weights = q.copy()
        kwargs["pairing_weights"] = q
        kernel = likelihood.pairing_kernel(observation, [0], beta=[beta])[0]
    else:
        kernel = likelihood.binary_feedback_likelihood(observation, [0], beta=[beta])
    output = dict(category_probabilities=partition.get_category_probabilities(0, data, beta)[:, 0].tolist(),
                  feedback_kernel=kernel.tolist(),
                  log_feedback_gradient=float(likelihood.feedback_precision_gradient(
                      observation, [0], [beta], **kwargs)[0]))
    if case["condition"] == 3:
        output["conditional_category_probabilities"] = partition.conditional_category_probabilities(
            0, data, beta)[:, :, 0].tolist()
    # Labels are conditioned last, with exactly the predictive (not updated) Beta.
    partition.update_template_labels(observation, np.asarray([beta]))
    node = partition.nodes[0]
    output["updated_template_labels"] = node["category_belief"] if node and node["kind"] == "template" else None
    p = np.asarray(output["category_probabilities"])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1.):
        raise AssertionError("Invalid original emission")
    json.dumps(output, allow_nan=False)
    return output


def proposal_cases() -> list[dict]:
    result = []
    theta3 = [[.65, .15, .12, .08], [.10, .25, .55, .10], [.05, .15, .20, .60]]
    for condition in (1, 2, 3):
        theta = [.8, .2] if condition == 1 else ([.1, .2, .3, .4] if condition == 2 else theta3)
        for name, mask, anchor, beta in (
            ("drop_one", [True, False, True, True], [.23, .37, .61, .79], 2.3),
            ("drop_two", [False, True, False, True], [.71, .28, .61, .19], 5.),
        ):
            item = dict(id=f"cond{condition}_{name}", condition=condition, beta=beta,
                parent=dict(kind="template", uid=1, parent_uid=0, anchor=anchor,
                            mask=mask, category_belief=deepcopy(theta)))
            if condition == 3:
                item["pairing"] = [.15, .60, .25]
            result.append(item)
    contrast = deepcopy(next(case for case in result if case["id"] == "cond3_drop_one"))
    contrast.update(id="cond3_drop_one_pairing_contrast", pairing=[.70, .20, .10])
    result.append(contrast)
    return result


def evaluate_proposal(case: dict, model) -> dict:
    import numpy as np
    partition = model.partition_model
    partition.nodes = [deepcopy(case["parent"])]
    parent = partition.nodes[0]
    if case["condition"] == 3:
        partition.pairing_weights = np.asarray([case["pairing"]], float)
    rules = partition.compatible_rule_candidates(parent)
    count = 29 if case["condition"] == 1 else 152
    np.testing.assert_array_equal(rules, np.arange(count))
    ignored = np.flatnonzero(~np.asarray(parent["mask"], bool))
    probes = np.tile(parent["anchor"], (2 ** len(ignored), 1))
    probes[:, ignored] = list(product((0., 1.), repeat=len(ignored)))
    # All expected model quantities are calls to the original implementation.
    weights = partition.rule_proposal_weights(parent, rules, case["beta"])
    probabilities = np.asarray([partition.catalogue.get_category_probabilities(
        int(rule), (probes, [], []), case["beta"], distance_mode="boundary").T for rule in rules])
    if (not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1.) or not np.isfinite(probabilities).all()):
        raise AssertionError("Invalid original rule proposal distribution")
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1., atol=1e-12, rtol=0)
    return dict(case, rule_ids=rules.tolist(), probes=probes.tolist(), expected=dict(
        rule_proposal_weights=weights.tolist(),
        parent_category_belief=partition.category_belief(parent).tolist(),
        catalogue_category_probabilities=probabilities.tolist()))


def extend_payload(existing: dict, generated: dict) -> dict:
    """Keep every frozen field; authorize only addition of the proposals section."""
    if "proposals" in existing:
        raise FileExistsError("Proposals already frozen; use --check")
    base = {key: value for key, value in generated.items() if key != "proposals"}
    if existing != base:
        raise AssertionError("Frozen cases or provenance differ; refusing fixture extension")
    return dict(deepcopy(existing), proposals=deepcopy(generated["proposals"]))


def generate(root: Path) -> dict:
    from src.Bayesian_state.model import ModelContext, StateModel
    models = {}
    for condition in (1, 2, 3):
        cfg = load_engine(root, condition, dict(capacity=1, similarity_width=.25,
            baseline_probability=.15, error_gain=1., failure_decay=.8, global_search=.2,
            retention=.8, execution_inertia=1., beta_init=5., learning_rate=.1))
        cfg["modules"]["perception_mod"]["kwargs"] = dict(
            features=4, mean=[0.] * 4, std=[0.] * 4, module_seed=7)
        models[condition] = StateModel(cfg, context=ModelContext(condition=condition))
    return dict(schema_version=1, model="1031-unified-v2", similarity_width=.25,
        choice_index_base=0, category_axis="response choice, original Python IDs 1..C",
        conditional_axes=["pairing", "category"], absolute_tolerance=1e-9,
        provenance=provenance(root), cases=[dict(case, expected=evaluate_case(case, models[case["condition"]]))
                                          for case in cases()],
        proposals=[evaluate_proposal(case, models[case["condition"]]) for case in proposal_cases()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/reference_1031.json")
    parser.add_argument("--check", action="store_true", help="compare existing fixture without writing")
    parser.add_argument("--extend-proposals", action="store_true",
                        help="append proposals to the frozen fixture after exact existing-field validation")
    args = parser.parse_args()
    if args.self_test:
        existing = dict(cases=[dict(id=str(i)) for i in range(25)], provenance={"source": "fixed"})
        generated = dict(deepcopy(existing), proposals=[dict(id="new")])
        result = extend_payload(existing, generated)
        assert result["cases"] == existing["cases"] and "proposals" not in existing
        bad = deepcopy(generated)
        bad["cases"][0]["id"] = "changed"
        try:
            extend_payload(existing, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Changed frozen cases accepted")
        assert len(proposal_cases()) == 7
        assert not any(name.startswith("src.Bayesian_state") for name in sys.modules)
        print("self-test passed: append-only extension, frozen-case guard, seven proposal inputs, no production imports")
        return
    if args.reference_root is None:
        parser.error("--reference-root is required for original model access")
    output = args.output.resolve()
    if output != ROOT / "tests/fixtures/reference_1031.json":
        parser.error("Only the new tests/fixtures/reference_1031.json is writable")
    if args.check and args.extend_proposals:
        parser.error("--check and --extend-proposals are mutually exclusive")
    if args.extend_proposals and not output.is_file():
        parser.error("--extend-proposals requires the existing frozen fixture")
    if output.exists() and not (args.check or args.extend_proposals):
        raise FileExistsError("Fixture exists; use --check (never overwrite numeric references)")
    original_bytes = output.read_bytes() if args.extend_proposals else None
    root = activate_reference(args.reference_root)
    payload = generate(root)
    if args.check:
        if json.loads(output.read_text(encoding="utf-8")) != payload:
            raise AssertionError("Fixture differs from unchanged original Python")
    elif args.extend_proposals:
        payload = extend_payload(json.loads(original_bytes), payload)
        if output.read_bytes() != original_bytes:
            raise RuntimeError("Fixture changed during generation; refusing write")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                dir=ROOT / ".reference_cache/tmp", delete=False, suffix=".json") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary_path = Path(stream.name)
        temporary_path.replace(output)
    else:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(payload["cases"]), proposals=len(payload["proposals"]),
        frozen_cases_sha256=sha256(json.dumps(payload["cases"], sort_keys=True).encode()).hexdigest(),
        checked=args.check, output=str(output))))


if __name__ == "__main__":
    main()
