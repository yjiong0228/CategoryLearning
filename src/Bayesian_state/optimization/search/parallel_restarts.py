"""Parallel score preparation with deterministic serial search publication.

Restart planners reuse the existing coordinate-descent implementation. Common
random numbers make each parameter point's score independent of planner order.
The owning optimizer subsequently publishes the original serial search order,
including candidate IDs, tie handling and checkpoints.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy, deepcopy
from functools import wraps
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any, Callable, Mapping

from joblib.externals.loky import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

from ...utils.parallel import parallel_job_count, single_threaded_processes
from ...inference.particle_parallel import initialize_particle_pool
from .cd_v2 import atomic_write_checkpoint, canonical_point_key


NUMERIC_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)}


def particle_block_count(config: Mapping[str, Any], base: Path) -> int:
    from ...simulation.config import resolve_engine_config
    engine = (resolve_engine_config(config, base) if config.get("engine_config_path")
              else config.get("engine_config") or {})
    return int((engine.get("inference") or {}).get("particle_blocks", 1))


def initialize_score_worker() -> None:
    threadpool_limits(limits=1)


def score_process_context(function: Callable[..., Any]) -> Callable[..., Any]:
    """Shared planners inherit the limits established by their owning thread.

    Scanning dynamic libraries from several planner threads can deadlock with
    pandas' optional-library imports. The owning thread enters the numeric
    context before launching planners and leaves it after they have joined.
    """
    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = (nullcontext() if getattr(self, "_shared_score_executor", None) is not None
                   else single_threaded_processes())
        with context:
            return function(self, *args, **kwargs)
    return wrapped


class PointScoreBank:
    """Deduplicate both completed and in-flight evaluations across planners."""

    def __init__(self, path: Path, optimizer: Any, existing: list[Any]) -> None:
        self.path = path
        self.optimizer = optimizer
        self.lock = Lock()
        self.futures: dict[str, Future] = {}
        self.execution: dict[str, Any] = {}
        if path.exists():
            for record in optimizer._load_jsonl_records(path, repair_trailing=True):
                row = optimizer._combination_from_record(record, path)
                self._seed(row)
        for row in existing:
            self._seed(row)

    def _seed(self, row: Any) -> None:
        key = canonical_point_key(row.hyperparams)
        if key in self.futures:
            prior = self.futures[key].result()
            if prior.objective_values != row.objective_values:
                raise ValueError("Parallel score cache disagrees with confirmed search scores")
        future: Future = Future()
        future.set_result(row)
        self.futures[key] = future

    def reserve(self, point: Mapping[str, Any]) -> tuple[Future, bool]:
        key = canonical_point_key(point)
        with self.lock:
            if key in self.futures:
                return self.futures[key], False
            future: Future = Future()
            self.futures[key] = future
            return future, True

    def publish(self, row: Any) -> None:
        with self.lock:
            # Flush complete point scores before exposing them to other planners.
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.optimizer._serialize_combination_record(row),
                                        ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            self.futures[canonical_point_key(row.hyperparams)].set_result(row)
            self._progress()

    def _progress(self) -> None:
        path = self.path.with_name(self.path.name.replace("parallel_scores_", "parallel_progress_")).with_suffix(".json")
        atomic_write_checkpoint(path, {
            **self.execution, "phase": "score_preparation",
            "completed_point_count": sum(future.done() and future.exception() is None
                                         for future in self.futures.values()),
            "reserved_point_count": len(self.futures),
        })

    def fail(self, entries: list[dict], error: BaseException) -> None:
        with self.lock:
            for entry in entries:
                future = self.futures[canonical_point_key(entry["point"])]
                if not future.done():
                    future.set_exception(error)


def evaluate_with_bank(optimizer: Any, arguments: dict[str, Any],
                       evaluate: Callable[..., Any]) -> tuple[list[Any], dict]:
    bank = optimizer._parallel_score_bank
    owned, futures = [], []
    for entry in arguments["missing_entries"]:
        future, is_new = bank.reserve(entry["point"])
        futures.append(future)
        if is_new:
            owned.append(entry)
    diagnostic = {
        "flat_task_count": 0, "flat_jobs": 0,
        "parallel_backend": "shared_restart_score_cache",
        "planned_total_jobs": arguments["value_jobs"] * arguments["repeat_jobs"],
    }
    if owned:
        try:
            rows, diagnostic = evaluate(**{**arguments, "missing_entries": owned})
            for row in rows:
                bank.publish(row)
        except BaseException as error:
            bank.fail(owned, error)
            raise
    result = []
    for entry, future in zip(arguments["missing_entries"], futures):
        row = deepcopy(future.result())
        row.combination_index = int(entry["combination_index"])
        row.restart_id = int(arguments["restart_id"])
        row.iter_id = int(arguments["iter_id"])
        row.coordinate = str(arguments["coordinate"])
        row.hyper_candidate_seed = optimizer._hyper_candidate_seed(
            row.stage, row.combination_index, row.hyperparams, row.restart_id,
            row.iter_id, row.coordinate,
        )
        for metrics in row.subject_metrics.values():
            metrics["hyper_candidate_seed"] = row.hyper_candidate_seed
        result.append(row)
    return result, diagnostic


@single_threaded_processes()
def prepare_restart_scores(optimizer: Any, arguments: dict[str, Any]) -> PointScoreBank:
    """Plan existing restarts concurrently using one bounded process pool."""
    rng = random.Random()
    supplied_rng = arguments.get("rng")
    rng.setstate((supplied_rng or optimizer._stage_rng(
        arguments["subjects"], arguments["stage_name"])).getstate())
    space = arguments["space"]
    supplied_points = arguments.get("initial_points")
    count = len(supplied_points) if supplied_points is not None else optimizer.n_restarts
    plans = []
    for restart in range(count):
        point = (deepcopy(supplied_points[restart]) if supplied_points is not None
                 else optimizer._init_point(space, rng, restart))
        coordinates = list(space)
        if optimizer.coordinate_order == "shuffle_per_restart":
            rng.shuffle(coordinates)
        plans.append((point, {name: space[name] for name in coordinates}))
    path = arguments["all_combinations_path"].parent / (
        f"parallel_scores_{arguments['stage_name']}.jsonl")
    bank = PointScoreBank(path, optimizer, list(arguments.get("existing_combinations") or []))
    repeats = int(arguments["stage_sim_cfg"].get("simulation_repeats", 1))
    blocks = particle_block_count(arguments["stage_sim_cfg"], optimizer.base_sim_config_path.parent)
    max_tasks = count * max(len(values) for values in space.values()) * repeats * blocks
    workers = parallel_job_count(optimizer.parallel_budget, max_tasks)
    bank.execution = {"stage": arguments["stage_name"], "process_budget": workers,
                      "restart_count": count, "particle_block_limit": blocks}
    bank._progress()
    with TemporaryDirectory(prefix="model_cd_restart_") as temporary:
        with ProcessPoolExecutor(max_workers=workers, env=NUMERIC_ENV,
                                 initializer=initialize_score_worker) as processes:
            initialize_particle_pool(processes, workers)
            def plan(index: int) -> None:
                point, ordered_space = plans[index]
                planner = copy(optimizer)
                planner.n_restarts = 1
                planner.coordinate_order = "fixed"
                planner._combination_counter = optimizer._combination_counter
                planner._parallel_score_bank = bank
                planner._shared_score_executor = processes
                planner._shared_score_workers = workers
                planner._coordinate_descent_serial(**{
                    **arguments, "space": ordered_space, "initial_points": [point],
                    "all_combinations_path": Path(temporary) / f"restart_{index}.jsonl",
                    "coordinate_trace_path": None, "checkpoint_callback": None,
                    "rng": random.Random(0),
                })
            with ThreadPoolExecutor(max_workers=count) as planners:
                list(planners.map(plan, range(count)))
    return bank


@single_threaded_processes()
def prepare_final_scores(optimizer: Any, shortlist: list[Any], cached: dict[str, Any],
                         stage_config: dict, subjects: Any, output_path: Path,
                         seed_family: str) -> dict[str, Any]:
    """Flatten shortlist candidates and repeats into the same process pool."""
    missing = [row for row in shortlist if canonical_point_key(row.hyperparams) not in cached]
    if not missing:
        return {}
    repeats = int(stage_config.get("simulation_repeats", 1))
    blocks = particle_block_count(stage_config, optimizer.base_sim_config_path.parent)
    workers = parallel_job_count(optimizer.parallel_budget, len(missing) * repeats * len(subjects) * blocks)
    reserved = [(row, optimizer._next_combination_index()) for row in missing]
    warm_path = output_path.with_name("parallel_scores_final_rescore.jsonl")
    warm = {}
    if warm_path.exists():
        for record in optimizer._load_jsonl_records(warm_path, repair_trailing=True):
            if record.get("seed_family") != seed_family:
                raise ValueError("Parallel final score seed family differs")
            row = optimizer._combination_from_record(record, warm_path)
            warm[canonical_point_key(row.hyperparams)] = row
    lock = Lock()
    with ProcessPoolExecutor(max_workers=workers, env=NUMERIC_ENV,
                             initializer=initialize_score_worker) as processes:
        initialize_particle_pool(processes, workers)
        def score(pair: tuple[Any, int]) -> tuple[str, Any]:
            search_row, index = pair
            key = canonical_point_key(search_row.hyperparams)
            row = warm.get(key)
            if row is not None:
                row = deepcopy(row)
                row.combination_index = index
                row.hyper_candidate_seed = optimizer._hyper_candidate_seed(
                    "final_rescore", index, row.hyperparams, -1, 0,
                    "final_rescore", seed_family,
                )
                for metrics in row.subject_metrics.values():
                    metrics["hyper_candidate_seed"] = row.hyper_candidate_seed
                return key, row
            planner = copy(optimizer)
            planner._shared_score_executor = processes
            planner._shared_score_workers = workers
            row = planner._evaluate_point_with_index(
                stage_name="final_rescore", point=deepcopy(search_row.hyperparams),
                stage_sim_cfg=stage_config, subjects=subjects, restart_id=-1,
                iter_id=0, coordinate="final_rescore", combination_index=index,
                seed_family=seed_family, force_common_random_numbers=True,
            )
            record = optimizer._serialize_combination_record(row)
            record["seed_family"] = seed_family
            with lock, warm_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            return key, row
        with ThreadPoolExecutor(max_workers=len(missing)) as planners:
            return dict(planners.map(score, reserved))
