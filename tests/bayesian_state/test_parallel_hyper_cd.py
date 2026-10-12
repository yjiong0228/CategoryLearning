"""Compare parallel restart scheduling to the original serial search decisions."""
from copy import deepcopy
import json
from pathlib import Path
import random
from types import MethodType
import time

import pytest
import yaml

from src.Bayesian_state.optimization.search.coordinate_descent import (
    CombinationResult, HyperCDOptimizer,
)


def optimizer_at(path: Path, parallel: bool) -> HyperCDOptimizer:
    path.mkdir()
    base = path / "base.yaml"
    base.write_text(yaml.safe_dump({"subjects": [334], "simulation_repeats": 4}))
    config = {
        "search_schema_version": 2, "base_sim_config_path": str(base),
        "output_dir": str(path / "fit"), "hyper_base_seed": 42,
        "common_random_numbers_within_candidate_comparisons": True,
        "objective_order": [{"path": "simulation.mean_error"}],
        "hyperparam_space": {"engine.x": {"values": [0, 1, 2]},
                             "engine.y": {"values": [0, 1, 2]}},
        "stages": {"coarse": {"cd_parallel": {"max_repeat_jobs": 4}}},
        "cd": {"resume_mode": "explicit", "n_restarts": 2,
               "max_outer_iters": 3, "parallel_budget": 4,
               "coordinate_order": "shuffle_per_restart", "min_delta": 0.0001,
               "parallel_restarts": parallel},
    }
    return HyperCDOptimizer(config, path / "search.yaml")


def install_surface(optimizer, surface):
    calls = []

    def evaluate(self, **arguments):
        rows = []
        for entry in arguments["missing_entries"]:
            point = entry["point"]
            calls.append(tuple(point[name] for name in sorted(point)))
            seed = self._hyper_candidate_seed(
                arguments["stage_name"], entry["combination_index"], point,
                arguments["restart_id"], arguments["iter_id"], arguments["coordinate"],
            )
            value = surface(point["engine.x"], point["engine.y"])
            rows.append(CombinationResult(
                stage=arguments["stage_name"], combination_index=entry["combination_index"],
                hyperparams=deepcopy(point), aggregated_error=value,
                objective_values={"simulation.mean_error": value},
                subject_metrics={334: {"hyper_candidate_seed": seed,
                                       "objectives": {"values": {"simulation.mean_error": value}}}},
                hyper_candidate_seed=seed, restart_id=arguments["restart_id"],
                iter_id=arguments["iter_id"], coordinate=arguments["coordinate"],
            ))
        return rows, {"flat_task_count": len(rows) * 4, "flat_jobs": 4,
                      "parallel_backend": "test", "planned_total_jobs": 4}
    optimizer._evaluate_missing_entries_uncached = MethodType(evaluate, optimizer)
    return calls


def run(optimizer, folder, existing=None):
    return optimizer._coordinate_descent(
        stage_name="coarse", stage_sim_cfg=optimizer.base_sim_config,
        subjects=[334], space={"engine.x": [0, 1, 2], "engine.y": [0, 1, 2]},
        all_combinations_path=folder / "all.jsonl", coordinate_trace_path=folder / "trace.jsonl",
        rng=random.Random(71), existing_combinations=existing,
    )


@pytest.mark.parametrize("surface", [
    lambda x, y: float((x - 1)**2 + (y - 2)**2),
    lambda x, y: 1.0,  # Exact ties exercise serial candidate ownership.
    lambda x, y: 1.0 - x * 0.00001 + y * 0.00001,  # Subthreshold moves.
])
def test_restart_scores_ids_seeds_ties_and_stopping_match_serial(tmp_path, surface):
    serial = optimizer_at(tmp_path / "serial", False)
    parallel = optimizer_at(tmp_path / "parallel", True)
    install_surface(serial, surface)
    calls = install_surface(parallel, surface)
    first = run(serial, tmp_path / "serial")
    second = run(parallel, tmp_path / "parallel")
    assert first == second
    assert len(calls) == len(set(calls))
    assert (tmp_path / "serial/all.jsonl").read_text() == (tmp_path / "parallel/all.jsonl").read_text()
    before = len(calls)
    replay = run(parallel, tmp_path / "parallel", existing=second[0])
    assert replay[0] == second[0]
    assert replay[2] == second[2]
    assert all(row["num_new_evaluations"] == 0 for row in replay[1])
    assert len(calls) == before


def test_parallel_restarts_requires_order_independent_scores(tmp_path):
    optimizer = optimizer_at(tmp_path / "initial", True)
    config = deepcopy(optimizer.config)
    config["common_random_numbers_within_candidate_comparisons"] = False
    with pytest.raises(ValueError, match="common random numbers"):
        HyperCDOptimizer(config, optimizer.config_path)
    config["common_random_numbers_within_candidate_comparisons"] = True
    config["cd"]["coordinate_order"] = "shuffle_each_iter"
    with pytest.raises(ValueError, match="shuffle_per_restart"):
        HyperCDOptimizer(config, optimizer.config_path)


def test_parallel_final_candidate_completion_order_and_resume_match_serial(tmp_path):
    serial = optimizer_at(tmp_path / "serial", False)
    parallel = optimizer_at(tmp_path / "parallel", True)
    shortlist = [CombinationResult(
        stage="fine", combination_index=i, hyperparams={"engine.x": i, "engine.y": 0},
        aggregated_error=float(i), objective_values={"simulation.mean_error": float(i)},
        subject_metrics={}, hyper_candidate_seed=1, restart_id=0, iter_id=1,
        coordinate="engine.x",
    ) for i in range(3)]
    for optimizer, enabled in [(serial, False), (parallel, True)]:
        optimizer.parallel_final_rescore = enabled
        optimizer._combination_counter = 100
        optimizer.config["final_rescore"] = {
            "enabled": True, "shortlist_size": 3, "seed_family": "independent",
            "simulation_overrides": {"simulation_repeats": 4,
                                     "repeat_aggregation": "mean_probability"},
        }

        def score(self, **arguments):
            x = arguments["point"]["engine.x"]
            time.sleep((2 - x) * 0.01)
            error = float(2 - x)
            return CombinationResult(
                stage="final_rescore", combination_index=arguments["combination_index"],
                hyperparams=arguments["point"], aggregated_error=error,
                objective_values={"simulation.mean_error": error}, subject_metrics={},
                hyper_candidate_seed=self._hyper_candidate_seed(
                    "final_rescore", arguments["combination_index"], arguments["point"],
                    -1, 0, "final_rescore", arguments["seed_family"],
                ), restart_id=-1, iter_id=0, coordinate="final_rescore",
            )
        optimizer._evaluate_point_with_index = MethodType(score, optimizer)
    first = serial._run_final_rescore(shortlist, subjects=[334],
                                      output_path=tmp_path / "serial/final.jsonl")
    second = parallel._run_final_rescore(shortlist, subjects=[334],
                                         output_path=tmp_path / "parallel/final.jsonl")
    assert first == second
    assert (tmp_path / "serial/final.jsonl").read_text() == (tmp_path / "parallel/final.jsonl").read_text()
    assert parallel._run_final_rescore(shortlist, subjects=[334],
                                       output_path=tmp_path / "parallel/final.jsonl", resume=True) == second
