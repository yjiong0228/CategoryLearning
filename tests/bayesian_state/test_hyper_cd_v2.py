from __future__ import annotations

from pathlib import Path
import json
import random

import pytest
import yaml

from src.Bayesian_state.optimization import cli as optimization_cli
from src.Bayesian_state.optimization.search.coordinate_descent import (
    CombinationResult,
    HyperCDOptimizer,
)
from src.Bayesian_state.optimization.search import cd_v2
from src.Bayesian_state.optimization.search.common import HyperSearchBase
from src.Bayesian_state.optimization.objectives import resolve_objective_order


def _minimal_hyper_config(tmp_path: Path) -> tuple[dict, Path]:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump({"subjects": [101]}), encoding="utf-8")
    config_path = tmp_path / "hyper.yaml"
    return (
        {
            "search_schema_version": 2,
            "base_sim_config_path": str(base_path),
            "output_dir": str(tmp_path / "output"),
            "hyper_base_seed": 20260901,
            "objective_order": [{"path": "simulation.mean_error"}],
            "hyperparam_space": {"engine.value": {"values": [0, 1]}},
            "stages": {"coarse": {"simulation_overrides": {}}},
            "cd": {
                "n_restarts": 1,
                "max_outer_iters": 2,
                "patience": 1,
                "min_delta": 0.0,
                "coordinate_order": "fixed",
                "parallel_budget": 1,
                "checkpoint_every_coordinate": True,
            },
        },
        config_path,
    )


def _run_surface_search(
    tmp_path: Path,
    *,
    surface: dict[tuple[int, ...], float],
    initial_point: dict[str, int],
    space: dict[str, list[int]],
    min_delta: float,
    initial_points_override: list[dict[str, int]] | None = None,
    existing_combinations: list[CombinationResult] | None = None,
    checkpoint_callback=None,
) -> tuple[CombinationResult, list[dict], Path]:
    optimizer = object.__new__(HyperCDOptimizer)
    optimizer.n_restarts = 1
    optimizer.max_outer_iters = 3
    optimizer.coordinate_order = "fixed"
    optimizer.patience = 1
    optimizer.min_delta = min_delta
    optimizer.init_strategy = "anchor"
    optimizer.anchor = {}
    optimizer.initial_points = [initial_point]
    optimizer.objective_order = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )
    optimizer.save_level = "compact"
    optimizer._combination_counter = (
        max(row.combination_index for row in existing_combinations) + 1
        if existing_combinations
        else 0
    )
    optimizer._coordinate_parallel_plan = lambda *args: (1, 1)

    coordinate_names = list(space)

    def evaluate_entries(**kwargs):
        rows = []
        for entry in kwargs["missing_entries"]:
            point = dict(entry["point"])
            error = surface[tuple(int(point[name]) for name in coordinate_names)]
            rows.append(
                CombinationResult(
                    stage=str(kwargs["stage_name"]),
                    combination_index=int(entry["combination_index"]),
                    hyperparams=point,
                    aggregated_error=error,
                    objective_values={"simulation.mean_error": error},
                    subject_metrics={},
                    hyper_candidate_seed=1,
                    restart_id=int(kwargs["restart_id"]),
                    iter_id=int(kwargs["iter_id"]),
                    coordinate=str(kwargs["coordinate"]),
                )
            )
        return rows, {
            "flat_task_count": len(rows),
            "flat_jobs": 1,
            "parallel_backend": "test",
            "planned_total_jobs": 1,
        }

    optimizer._evaluate_missing_entries_flat = evaluate_entries
    trace_path = tmp_path / "coordinate_trace.jsonl"
    search_kwargs = {}
    if initial_points_override is not None:
        search_kwargs["initial_points"] = initial_points_override
    if existing_combinations is not None:
        search_kwargs["existing_combinations"] = existing_combinations
    if checkpoint_callback is not None:
        search_kwargs["checkpoint_callback"] = checkpoint_callback
    _, restarts, best = optimizer._coordinate_descent(
        stage_name="coarse",
        stage_sim_cfg={},
        subjects=[101],
        space=space,
        all_combinations_path=tmp_path / "all_combinations.jsonl",
        coordinate_trace_path=trace_path,
        rng=random.Random(7),
        **search_kwargs,
    )
    return best, restarts, trace_path


def test_schema_v2_requires_explicit_resume_mode(tmp_path: Path) -> None:
    """Catch schema-v2 searches that could silently inherit legacy overwrite behavior."""

    config, config_path = _minimal_hyper_config(tmp_path)

    with pytest.raises(ValueError, match="cd.resume_mode"):
        HyperCDOptimizer(config, config_path)


def test_schema_v2_rejects_negative_min_delta(tmp_path: Path) -> None:
    """Catch a threshold whose sign would make every ordered improvement movable."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    config["cd"]["min_delta"] = -0.1

    with pytest.raises(ValueError, match="cd.min_delta"):
        HyperCDOptimizer(config, config_path)


def test_schema_v2_rejects_unknown_fine_initialization(tmp_path: Path) -> None:
    """Catch schema-v2 fine search silently falling back to legacy starts."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    config["refine_policy"] = {"fine_initialization": "random"}

    with pytest.raises(ValueError, match="fine_initialization"):
        HyperCDOptimizer(config, config_path)


def test_fine_projection_keeps_mapping_exact_and_uses_nearest_numeric_value() -> None:
    """Catch fine initialization that interpolates or decomposes a joint structure."""

    point = {"joint": {"M": 3, "chi": 1}, "x": 0.49}
    space = {
        "joint": [{"M": 2, "chi": 0}, {"M": 3, "chi": 1}],
        "x": [0.25, 0.50, 0.75],
    }

    assert cd_v2.project_point_to_space(point, space) == {
        "joint": {"M": 3, "chi": 1},
        "x": 0.50,
    }


def test_candidate_improvement_enforces_primary_min_delta() -> None:
    """Catch PF-noise-sized improvements being accepted as coordinate moves."""

    specs = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )
    current = {"simulation.mean_error": 1.0}

    assert not cd_v2.candidate_improves(
        current,
        {"simulation.mean_error": 0.99995},
        specs,
        min_delta=0.0001,
    )
    assert cd_v2.candidate_improves(
        current,
        {"simulation.mean_error": 0.9998},
        specs,
        min_delta=0.0001,
    )


def test_coordinate_descent_does_not_move_for_subthreshold_improvement(
    tmp_path: Path,
) -> None:
    """Catch the optimizer bypassing its parsed min_delta during a real scan."""

    best, _, trace_path = _run_surface_search(
        tmp_path,
        surface={(0,): 1.0, (1,): 0.99995},
        initial_point={"x": 0},
        space={"x": [0, 1]},
        min_delta=0.0001,
    )

    assert best.hyperparams == {"x": 0}
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert trace[0]["min_delta_reject_count"] == 1


def test_coordinate_descent_revisits_earlier_coordinate_on_second_sweep(
    tmp_path: Path,
) -> None:
    """Protect the multi-sweep search needed when later coordinates unlock earlier ones."""

    best, restarts, _ = _run_surface_search(
        tmp_path,
        surface={(0, 0): 3.0, (1, 0): 3.0, (0, 1): 2.0, (1, 1): 1.0},
        initial_point={"x": 0, "y": 0},
        space={"x": [0, 1], "y": [0, 1]},
        min_delta=0.0,
    )

    assert best.hyperparams == {"x": 1, "y": 1}
    assert restarts[0]["num_improvements"] == 2
    assert restarts[0]["outer_iters_completed"] >= 2


def test_fine_initial_points_are_ranked_projected_and_deduplicated() -> None:
    """Catch fine search restarting randomly or duplicating projected coarse points."""

    optimizer = object.__new__(HyperCDOptimizer)
    optimizer.config = {"refine_policy": {"top_k": 3}}
    optimizer.objective_order = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )

    def coarse(index, error, joint, x):
        return CombinationResult(
            stage="coarse",
            combination_index=index,
            hyperparams={"joint": joint, "x": x},
            aggregated_error=error,
            objective_values={"simulation.mean_error": error},
            subject_metrics={},
            hyper_candidate_seed=index,
            restart_id=0,
            iter_id=1,
            coordinate="x",
        )

    combinations = [
        coarse(1, 0.80, {"M": 3, "chi": 0}, 0.49),
        coarse(2, 0.90, {"M": 3, "chi": 1}, 0.74),
        coarse(3, 1.00, {"M": 3, "chi": 0}, 0.51),
    ]
    fine_space = {
        "joint": [{"M": 3, "chi": 0}, {"M": 3, "chi": 1}],
        "x": [0.50, 0.75],
    }

    assert optimizer._fine_initial_points(combinations, fine_space) == [
        {"joint": {"M": 3, "chi": 0}, "x": 0.50},
        {"joint": {"M": 3, "chi": 1}, "x": 0.75},
    ]


def test_stage_initial_points_override_global_restart_configuration(
    tmp_path: Path,
) -> None:
    """Catch fine restarts accidentally reusing the global coarse starts."""

    _, restarts, _ = _run_surface_search(
        tmp_path,
        surface={(0,): 2.0, (1,): 1.0},
        initial_point={"x": 0},
        space={"x": [0, 1]},
        min_delta=0.0,
        initial_points_override=[{"x": 1}, {"x": 0}],
    )

    assert len(restarts) == 2
    assert [row["initial_error"] for row in restarts] == [1.0, 2.0]


def test_all_stage_pipeline_passes_projected_coarse_shortlist_to_fine(
    tmp_path: Path,
) -> None:
    """Catch the pipeline computing fine starts but failing to use them."""

    class FineInvocationObserved(RuntimeError):
        pass

    optimizer = object.__new__(HyperCDOptimizer)
    optimizer.config = {
        "stages": {
            "coarse": {"hyperparam_space": {"x": {"values": [0.49, 0.74]}}},
            "fine": {"hyperparam_space": {"x": {"values": [0.50, 0.75]}}},
        },
        "refine_policy": {"top_k": 1},
    }
    optimizer.cd_v2 = cd_v2.CDV2Config(
        enabled=True,
        resume_mode="explicit",
        checkpoint_every_coordinate=True,
        fine_initialization="coarse_shortlist",
    )
    optimizer.hyper_base_seed = 20260901
    optimizer.base_sim_config = {}
    optimizer._prepare_stage_config = lambda stage_name: {}
    optimizer._combination_counter = 0
    optimizer.objective_order = resolve_objective_order(
        {"objective_order": [{"path": "simulation.mean_error"}]}
    )
    coarse = CombinationResult(
        stage="coarse",
        combination_index=0,
        hyperparams={"x": 0.49},
        aggregated_error=0.8,
        objective_values={"simulation.mean_error": 0.8},
        subject_metrics={},
        hyper_candidate_seed=1,
        restart_id=0,
        iter_id=1,
        coordinate="x",
    )

    def coordinate_descent(**kwargs):
        if kwargs["stage_name"] == "coarse":
            return [coarse], [], coarse
        assert kwargs["initial_points"] == [{"x": 0.50}]
        raise FineInvocationObserved

    optimizer._coordinate_descent = coordinate_descent

    with pytest.raises(FineInvocationObserved):
        optimizer._run_pipeline(
            subjects=[101],
            stage="all",
            output_dir=tmp_path,
        )


def test_checkpoint_is_written_atomically_and_round_trips(tmp_path: Path) -> None:
    """Catch partial checkpoint files being mistaken for resumable search state."""

    path = tmp_path / "search_checkpoint.json"
    payload = {
        "status": "active",
        "stage": "coarse",
        "restart_id": 1,
        "iter_id": 2,
        "current_point": {"x": 0.5},
        "subjects": [101, 111, 118],
    }

    cd_v2.atomic_write_checkpoint(path, payload)

    assert cd_v2.load_checkpoint(path) == payload
    assert list(tmp_path.glob(".search_checkpoint.json.*.tmp")) == []


def test_resume_fingerprint_covers_config_base_subject_order_and_stage() -> None:
    """Catch a resume operation reusing scores under a changed scientific context."""

    config = {"search_schema_version": 2, "cd": {"min_delta": 0.0}}
    base = {"max_trials": None, "prediction_mode": "prior_t"}
    expected = cd_v2.search_context_fingerprint(config, base, [101, 111], "all")

    assert expected == cd_v2.search_context_fingerprint(
        {"cd": {"min_delta": 0.0}, "search_schema_version": 2},
        {"prediction_mode": "prior_t", "max_trials": None},
        [101, 111],
        "all",
    )
    assert expected != cd_v2.search_context_fingerprint(
        config, base, [111, 101], "all"
    )
    assert expected != cd_v2.search_context_fingerprint(
        config, {**base, "max_trials": 64}, [101, 111], "all"
    )
    assert expected != cd_v2.search_context_fingerprint(
        config, base, [101, 111], "coarse"
    )


def test_resume_repairs_only_a_partial_final_jsonl_record(tmp_path: Path) -> None:
    """Catch crash-truncated tails blocking safe cache recovery."""

    path = tmp_path / "records.jsonl"
    path.write_text('{"combination_index": 0}\n{"combination_index":', encoding="utf-8")

    assert HyperSearchBase._load_jsonl_records(
        path, repair_trailing=True
    ) == [{"combination_index": 0}]
    assert path.read_text(encoding="utf-8") == '{"combination_index": 0}\n'


def test_resume_rejects_invalid_jsonl_before_the_final_line(tmp_path: Path) -> None:
    """Catch cache corruption being mislabeled as an interrupted final append."""

    path = tmp_path / "records.jsonl"
    path.write_text('{"combination_index":\n{"combination_index": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL"):
        HyperSearchBase._load_jsonl_records(path, repair_trailing=True)


def test_deterministic_replay_resume_matches_uninterrupted_search(
    tmp_path: Path,
) -> None:
    """Catch resume paths that repeat evaluation or diverge from the original trace."""

    surface = {(0, 0): 3.0, (1, 0): 3.0, (0, 1): 2.0, (1, 1): 1.0}
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full_best, full_restarts, full_trace = _run_surface_search(
        full_dir,
        surface=surface,
        initial_point={"x": 0, "y": 0},
        space={"x": [0, 1], "y": [0, 1]},
        min_delta=0.0,
    )

    interrupted_dir = tmp_path / "interrupted"
    interrupted_dir.mkdir()
    checkpoint_count = 0

    def interrupt_after_second_coordinate(payload):
        nonlocal checkpoint_count
        checkpoint_count += 1
        if checkpoint_count == 2:
            raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected interruption"):
        _run_surface_search(
            interrupted_dir,
            surface=surface,
            initial_point={"x": 0, "y": 0},
            space={"x": [0, 1], "y": [0, 1]},
            min_delta=0.0,
            checkpoint_callback=interrupt_after_second_coordinate,
        )

    combination_path = interrupted_dir / "all_combinations.jsonl"
    records = HyperSearchBase._load_jsonl_records(combination_path)
    loader = object.__new__(HyperCDOptimizer)
    cached = [loader._combination_from_record(record, combination_path) for record in records]
    (interrupted_dir / "coordinate_trace.jsonl").unlink()

    resumed_best, resumed_restarts, resumed_trace = _run_surface_search(
        interrupted_dir,
        surface=surface,
        initial_point={"x": 0, "y": 0},
        space={"x": [0, 1], "y": [0, 1]},
        min_delta=0.0,
        existing_combinations=cached,
    )

    def scientific_restarts(rows):
        return [
            {
                key: value
                for key, value in row.items()
                if key not in {"num_new_evaluations", "num_cache_hits"}
            }
            for row in rows
        ]

    def scientific_trace(path):
        runtime_fields = {
            "missing_value_count",
            "new_evaluations",
            "cache_hits",
            "flat_task_count",
            "flat_jobs",
            "parallel_backend",
        }
        return [
            {key: value for key, value in json.loads(line).items() if key not in runtime_fields}
            for line in path.read_text().splitlines()
        ]

    assert resumed_best.hyperparams == full_best.hyperparams
    assert scientific_restarts(resumed_restarts) == scientific_restarts(full_restarts)
    assert scientific_trace(resumed_trace) == scientific_trace(full_trace)
    assert combination_path.read_text() == (
        full_dir / "all_combinations.jsonl"
    ).read_text()


def test_schema_v2_existing_output_requires_explicit_resume(tmp_path: Path) -> None:
    """Catch a new schema-v2 invocation silently deleting prior search artifacts."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    optimizer = HyperCDOptimizer(config, config_path)
    subject_dir = optimizer.output_dir / "subject_101"
    subject_dir.mkdir()
    (subject_dir / "coordinate_trace.jsonl").write_text(
        '{"stage":"coarse"}\n', encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="--resume"):
        optimizer.run([101], stage="coarse", resume=False)


def test_schema_v2_resume_requires_checkpoint(tmp_path: Path) -> None:
    """Catch an alleged resume silently starting a fresh search."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    optimizer = HyperCDOptimizer(config, config_path)

    with pytest.raises(FileNotFoundError, match="search_checkpoint.json"):
        optimizer.run([101], stage="coarse", resume=True)


def test_schema_v2_resume_rejects_context_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    """Catch cached scores being reused under a different scientific context."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    optimizer = HyperCDOptimizer(config, config_path)
    subject_dir = optimizer.output_dir / "subject_101"
    subject_dir.mkdir()
    cd_v2.atomic_write_checkpoint(
        subject_dir / "search_checkpoint.json",
        {
            "schema_version": 2,
            "context_fingerprint": "not-the-current-context",
            "subjects": [101],
            "requested_stage": "coarse",
            "combination_record_count": 0,
        },
    )

    with pytest.raises(ValueError, match="fingerprint"):
        optimizer.run([101], stage="coarse", resume=True)


def test_schema_v2_resume_replays_stage_from_jsonl_cache(
    tmp_path: Path,
) -> None:
    """Catch resume deleting cached combinations instead of replaying from them."""

    config, config_path = _minimal_hyper_config(tmp_path)
    config["cd"]["resume_mode"] = "explicit"
    optimizer = HyperCDOptimizer(config, config_path)
    subject_dir = optimizer.output_dir / "subject_101"
    subject_dir.mkdir()
    record = {
        "schema_version": 1,
        "stage": "coarse",
        "combination_index": 0,
        "restart_id": 0,
        "iter_id": 0,
        "coordinate": "init",
        "hyperparams": {"engine.value": 0},
        "aggregated_error": 1.0,
        "objective_values": {"simulation.mean_error": 1.0},
        "hyper_candidate_seed": 1,
    }
    combinations_path = subject_dir / "all_combinations.jsonl"
    combinations_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    trace_path = subject_dir / "coordinate_trace.jsonl"
    trace_path.write_text('{"stale":true}\n', encoding="utf-8")
    fingerprint = cd_v2.search_context_fingerprint(
        optimizer.config,
        optimizer.base_sim_config,
        [101],
        "coarse",
    )
    cd_v2.atomic_write_checkpoint(
        subject_dir / "search_checkpoint.json",
        {
            "schema_version": 2,
            "context_fingerprint": fingerprint,
            "subjects": [101],
            "requested_stage": "coarse",
            "combination_record_count": 1,
        },
    )

    def capture_resume(**kwargs):
        assert not trace_path.exists()
        cached = kwargs["existing_combinations"]
        assert [row.hyperparams for row in cached] == [{"engine.value": 0}]
        kwargs["checkpoint_callback"](
            {
                "status": "active",
                "stage": "coarse",
                "combination_record_count": 1,
            }
        )
        raise RuntimeError("resume inputs captured")

    optimizer._coordinate_descent = capture_resume

    with pytest.raises(RuntimeError, match="resume inputs captured"):
        optimizer.run([101], stage="coarse", resume=True)

    checkpoint = cd_v2.load_checkpoint(subject_dir / "search_checkpoint.json")
    assert checkpoint["context_fingerprint"] == fingerprint
    assert checkpoint["subjects"] == [101]
    assert checkpoint["requested_stage"] == "coarse"


def test_hyper_optimization_cli_accepts_explicit_resume(monkeypatch) -> None:
    """Catch the public CLI omitting the opt-in needed by schema-v2 searches."""

    monkeypatch.setattr(
        "sys.argv",
        [
            "hyper-opt",
            "--backend",
            "cd",
            "--config",
            "hyper.yaml",
            "--resume",
        ],
    )

    args = optimization_cli.parse_args()

    assert args.resume is True
