"""Small, read-only Model 0826 audit; experimental patches live in this process only.

Run from the repository root with ``python -m ...audit_model_0826_system
--output-dir <new-directory>``. This does not fit parameters or edit research data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.hypothesis_space.geometry.boundary import BoundaryGeometry
from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from src.Bayesian_state.hypothesis_space.observation_model.discrete_rule_partition import DiscreteRulePartition
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.model.modules.hypothesis_transition.prior_assignment import PriorAssignmentPolicyMixin
from src.Bayesian_state.optimization.search.cd_v2 import search_context_fingerprint
from src.Bayesian_state.simulation.data import SubjectTrialDataLoader


def read_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def public_arrays(result) -> dict[str, np.ndarray]:
    arrays = {}
    for field in ("observation_probabilities", "state_probabilities", "latent_summaries"):
        for key, value in getattr(result, field).items():
            if value is not None:
                arrays[f"{field}/{key}"] = np.asarray(value)
    for key in ("resampled", "pre_choice_ess", "post_choice_ess", "resampling_unique_ancestors"):
        arrays[key] = np.asarray(getattr(result, key))
    return arrays


def compare(left, right) -> int:
    a, b = public_arrays(left), public_arrays(right)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    return len(a)


def dataset_summary(path: str) -> dict:
    frame = pd.read_csv(path)
    keys = [c for c in ("iSub", "iSession", "iBlock", "iTrial") if c in frame]
    summary = {
        "path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "rows": len(frame), "subjects": frame.groupby("iSub").size().to_dict(),
        "duplicate_trial_keys": int(frame.duplicated(keys).sum()),
        "chronologically_sorted": frame.index.equals(frame.sort_values(keys, kind="stable").index),
        "feedback_counts": frame.feedback.value_counts().to_dict(),
        "sessions": frame.groupby(["iSub", "iSession"]).size().rename("n").reset_index().to_dict("records"),
    }
    if "probCat1" in frame:
        probs = frame[["probCat1", "probCat2"]].to_numpy()
        summary["probability_sum_max_error"] = float(np.max(np.abs(probs.sum(1) - 1)))
        summary["feedback_disagrees_argmax"] = int(np.sum(
            (frame.choice.to_numpy() == probs.argmax(1) + 1) != frame.feedback.to_numpy()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "pandas", "numba", "joblib", "PyYAML", "pytest")},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "datasets": [dataset_summary(p) for p in
                     ("data/exp4/processed/Task2_processed.csv", "data/exp5/processed/Rule1_processed.csv")],
    }
    base = read_yaml("configs/exp123/model_struct/pmh_model_cond1_0826.yaml")
    frame = pd.read_csv("data/exp123/processed/Task2_processed.csv")
    frame = frame.loc[frame.iSub.eq(129)].iloc[:32]
    kwargs = dict(engine_config=base, subject_id=129,
                  stimulus=frame[[f"feature{i}" for i in range(1, 5)]].to_numpy(),
                  choices=frame.choice.to_numpy(), feedback=frame.feedback.to_numpy(),
                  particle_count=16, choice_readout_power=1., filter_seed=8326)
    baseline = run_state_model_particle_filter(**kwargs)
    report["deterministic_repeat_array_count"] = compare(baseline, run_state_model_particle_filter(**kwargs))
    changed = dict(kwargs, choices=kwargs["choices"].copy(), feedback=kwargs["feedback"].copy())
    changed["choices"][16:] = 3 - changed["choices"][16:]
    changed["feedback"][16:] = 1 - changed["feedback"][16:]
    altered = run_state_model_particle_filter(**changed)
    np.testing.assert_array_equal(baseline.marginal_probabilities[:17], altered.marginal_probabilities[:17])
    report["future_outcome_causality_check"] = "first 17 pre-choice predictions unchanged when outcomes from trial 17 change"

    fractional = dict(kwargs, choices=kwargs["choices"].astype(float) + .25)
    try:
        fractional_result = run_state_model_particle_filter(**fractional)
        report["fractional_choice_probe"] = {"accepted": True, "equal_arrays": compare(baseline, fractional_result)}
    except ValueError as exc:
        report["fractional_choice_probe"] = {"accepted": False, "error": str(exc)}
    try:
        value = PriorAssignmentPolicyMixin()._post_to_prior_confidence(np.array([.3, .7]), {"confidence_source": "entropy"})
        report["entropy_probe"] = {"value": value}
    except Exception as exc:
        report["entropy_probe"] = {"error_type": type(exc).__name__, "message": str(exc)}
    with TemporaryDirectory(prefix="model0826-fingerprint-") as directory:
        source = Path(directory) / "engine.yaml"
        source.write_text("capacity: 3\n")
        sim = {"engine_config_path": str(source)}
        before = search_context_fingerprint({"schema_version": 2}, sim, [129], "all")
        source.write_text("capacity: 5\n")
        after = search_context_fingerprint({"schema_version": 2}, sim, [129], "all")
        report["resume_fingerprint_probe"] = {"unchanged_after_referenced_file_change": before == after}
    loader = SubjectTrialDataLoader(base)
    loader.learning_data = frame.iloc[::-1].copy()
    selected = loader._get_subject_frame(129, 1.)
    report["row_order_probe"] = {"reversed_rows_preserved": selected.iTrial.tolist() == frame.iloc[::-1].iTrial.tolist()}

    partition = DiscreteRulePartition()
    similarity = partition.get_similarity_matrix()
    values, counts = np.unique(similarity[~np.eye(partition.length, dtype=bool)], return_counts=True)
    exp5 = pd.read_csv("data/exp5/processed/Rule1_processed.csv")
    x = exp5[[f"feature{i}" for i in range(1, 6)]].to_numpy()
    errors = [int(np.sum(partition.rule_geometry.category_assignments(i, x) + 1 != exp5.category.to_numpy())) for i in range(partition.length)]
    report["exp5_rules"] = {"count": partition.length, "off_diagonal_similarity": dict(zip(map(str, values), map(int, counts))),
                            "best_rules": [partition.describe_rule(i) for i, e in enumerate(errors) if e == min(errors)],
                            "minimum_errors": min(errors)}
    raw = list(Path("data/exp5/raw/Rule1").glob("*_bhv.csv"))
    report["exp5_raw_inventory"] = {"files": len(raw), "subject_ids": sorted({int(p.name.split("_")[0]) for p in raw}),
                                    "rows": sum(len(pd.read_csv(p)) for p in raw)}

    # Same arithmetic result for beta=0; preserve all-H likelihood normalization.
    original_probability = ContinuousPartition.get_category_probabilities
    def zero_beta_probability(self, hypo, data, beta, distance_mode=None, **extra):
        self._resolve_distance_mode(distance_mode)
        if float(beta) == 0.:
            return np.full((self.n_cats, len(data[0])), 1. / self.n_cats)
        return original_probability(self, hypo, data, beta, distance_mode=distance_mode, **extra)

    original_distance = BoundaryGeometry.category_distances
    def cached_distance(self, hypo, stimuli):
        array = np.asarray(stimuli, dtype=float).reshape(-1, self.space.n_dims)
        key = (int(hypo), array.shape, array.tobytes())
        cache = getattr(self, "_audit_distance_cache", None)
        if cache is None:
            cache = self._audit_distance_cache = {}
        if key not in cache:
            cache[key] = original_distance(self, hypo, array)
        return cache[key]

    timings = {"baseline": [], "zero_beta": [], "zero_beta_and_cache": []}
    comparisons = []
    for _ in range(3):
        for mode in timings:
            with patch.object(ContinuousPartition, "get_category_probabilities", original_probability if mode == "baseline" else zero_beta_probability):
                with patch.object(BoundaryGeometry, "category_distances", cached_distance if mode.endswith("and_cache") else original_distance):
                    started = perf_counter()
                    output = run_state_model_particle_filter(**kwargs)
                    timings[mode].append(perf_counter() - started)
            comparisons.append({"mode": mode, "equal_arrays": compare(baseline, output)})
    report["benchmark"] = {"trials": 32, "particles": 16, "subject": 129, "seed": 8326,
                           "seconds": timings, "medians": {k: float(np.median(v)) for k, v in timings.items()},
                           "comparisons": comparisons}
    report["fast_path_cases"] = []
    for capacity, execution, seed in [(1, False, 21), (3, True, 22), (5, False, 23)]:
        config = deepcopy(base)
        config["modules"]["hypo_transitions_mod"]["kwargs"]["capacity"] = capacity
        config["modules"]["hypo_transitions_mod"]["kwargs"]["persistent_execution"]["enabled"] = execution
        case = dict(kwargs, engine_config=config, particle_count=8, filter_seed=seed)
        original = run_state_model_particle_filter(**case)
        with patch.object(ContinuousPartition, "get_category_probabilities", zero_beta_probability):
            with patch.object(BoundaryGeometry, "category_distances", cached_distance):
                accelerated = run_state_model_particle_filter(**case)
        report["fast_path_cases"].append({"capacity": capacity, "execution": execution, "seed": seed, "equal_arrays": compare(original, accelerated)})

    (args.output_dir / "pre_migration_evidence.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    # Adapt only in memory: these are integration probes, not validated new models.
    report["migration_smoke"] = []
    for experiment, subject, engine_path, data_path in [
        ("exp4", 401, "configs/exp4/model_struct/pmh_prob_proto.yaml", "data/exp4/processed/Task2_processed.csv"),
        ("exp5", 501, "configs/exp5/model_struct/pmh_rule1.yaml", "data/exp5/processed/Rule1_processed.csv"),
    ]:
        legacy = read_yaml(engine_path)
        config = deepcopy(base)
        config["partition"] = legacy["partition"]
        config["data"] = legacy["data"]
        if experiment == "exp4":
            # Boundary route is the minimal 0826 migration. Prototype route
            # needs a separately frozen similarity resource and scale calibration.
            config["likelihood"]["feedback_likelihood_mode"] = "bernoulli_choice"
            config["modules"]["perception_mod"]["kwargs"] = {"noise_mode": "uniform"}
        else:
            config["likelihood"]["distance_mode"] = "rule"
            # PF currently requires this named module, including discrete tasks.
            # The legacy clip maps -1 to 0; rule geometry maps nonpositive values
            # back to -1, so this smoke preserves parity-rule predictions only.
            config["modules"]["perception_mod"]["kwargs"] = {
                "features": 5, "mean": 0., "std": 0., "noise_mode": "normal",
            }
        data = pd.read_csv(data_path).query("iSub == @subject").iloc[:16]
        processed = Path(data_path).resolve().parent
        paths = {"learning_data": Path(data_path).resolve(), "feature_order_data": Path(data_path).resolve()}
        if experiment == "exp4":
            paths["perception_summary"] = processed / "Task1b_errorsummary.csv"
            paths["perception_summary_72"] = paths["perception_summary"]
        result = run_state_model_particle_filter(
            engine_config=config, subject_id=subject, condition=1,
            stimulus=data[legacy["data"]["feature_columns"]].to_numpy(),
            choices=data.choice.to_numpy(), feedback=data.feedback.to_numpy(),
            particle_count=4, choice_readout_power=1., filter_seed=20260917,
            processed_data_dir=processed, dataset_paths=paths,
        )
        probabilities = result.marginal_probabilities
        assert np.all(np.isfinite(probabilities))
        np.testing.assert_allclose(probabilities.sum(1), 1.)
        report["migration_smoke"].append({"experiment": experiment, "trials": len(data), "particles": 4,
                                          "finite_normalized": True, "mean_choice_nll": float(-np.log(probabilities[np.arange(len(data)), data.choice.to_numpy()-1]).mean())})
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path("src/Bayesian_state").rglob("*.py")}
    (args.output_dir / "evidence.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(args.output_dir / "baseline_arrays.npz", **public_arrays(baseline))
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
