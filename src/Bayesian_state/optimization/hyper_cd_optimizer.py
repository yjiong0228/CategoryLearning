"""Coordinate-descent hyperparameter optimizer backed by repeated simulations."""
from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed

from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.optimization.hyper_utils import (
    HYPER_RESULT_SCHEMA_VERSION,
    build_root_best_payload,
    build_hyper_provenance,
    build_subject_artifacts,
    build_subject_best_payload,
    combination_metrics_summary,
    compact_hyperparams,
    expand_profile_candidate_hyperparams,
    to_builtin,
    validate_no_nested_hyperparam_paths,
    values_from_json,
    values_product,
)
from src.Bayesian_state.optimization.optimization_config import (
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_simulation_repeats,
    resolve_window_size,
)
from src.Bayesian_state.optimization.optimizer_common import (
    derive_hyper_candidate_seed,
    derive_simulation_point_seed,
    derive_trajectory_seed,
    evaluate_state_model_run,
    stable_seed,
)
from src.Bayesian_state.optimization.hyper_objectives import (
    aggregate_objective_values,
    compare_objective_values,
    extract_subject_objective_values,
    first_objective_value,
    objective_order_payload,
    passes_anchor_guard,
    rank_by_objectives,
    resolve_objective_order,
    select_best_by_objectives,
    update_anchor_values,
)
from src.Bayesian_state.utils.simulation_statistics import resolve_simulation_stat_config
from src.Bayesian_state.optimization.optimizer_simulation import (
    StateModelSimulationRunner,
    aggregate_simulation_runs,
)


LOWER_TAIL_FRACTION = 0.10


@dataclass
class CombinationResult:
    stage: str
    combination_index: int
    hyperparams: Dict[str, Any]
    aggregated_error: float
    objective_values: Dict[str, float]
    subject_metrics: Dict[int, Dict[str, Any]]
    hyper_candidate_seed: int
    restart_id: int
    iter_id: int
    coordinate: str


def _lower_tail_error_metrics(sample_errors: Sequence[float], fallback_error: float) -> Dict[str, Any]:
    values = np.asarray(list(sample_errors or []), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.asarray([float(fallback_error)], dtype=float)
    ordered = np.sort(values)
    tail_count = max(1, int(np.ceil(ordered.size * LOWER_TAIL_FRACTION)))
    return {
        "best10_mean_error": float(np.mean(ordered[:tail_count])),
        "q10_error": float(np.quantile(ordered, LOWER_TAIL_FRACTION)),
        "lower_tail_fraction": float(LOWER_TAIL_FRACTION),
        "lower_tail_count": int(tail_count),
    }


def _evaluate_cd_flat_repeat_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    run = evaluate_state_model_run(
        int(task["subject_id"]),
        int(task["condition"]),
        task["arrays"],
        dict(task["params"]),
        task["engine_config_template"],
        task["processed_data_dir"],
        int(task["window_size"]),
        task["dataset_paths"],
        bool(task["keep_logs"]),
        bool(task["keep_logs"]),
        str(task["prediction_mode"]),
        str(task["selection_prediction_mode"]),
        str(task["loss_metric"]),
        task.get("loss_delta"),
        simulation_point_seed=int(task["simulation_point_seed"]),
        trajectory_seed=int(task["trajectory_seed"]),
        seed_context=task.get("seed_context"),
    )
    return {
        "position": int(task["position"]),
        "repeat_index": int(task["repeat_index"]),
        "run": run,
    }


class HyperCDOptimizer:
    """Choose model hyperparameters with coordinate descent."""

    def __init__(self, config: Mapping[str, Any], config_path: Path) -> None:
        self.config = dict(config)
        self.config_path = config_path
        self.config_dir = config_path.parent

        self.objective_order = resolve_objective_order(self.config)
        self.objective_order_config = objective_order_payload(self.objective_order)

        # `hyperparam_selection_mode` config key is optional now; keep internal
        # default as per-subject selection (only mode currently implemented).
        self.hyperparam_selection_mode = "per_subject"

        self.save_level = str(self.config.get("save_level", "compact")).strip().lower()
        if self.save_level not in {"compact", "full"}:
            raise ValueError("save_level must be 'compact' or 'full'")

        if "hyper_base_seed" not in self.config:
            raise ValueError("Hyper-CD config must include hyper_base_seed.")
        self.hyper_base_seed = int(self.config["hyper_base_seed"])

        base_sim = self.config.get("base_sim_config_path")
        if not base_sim:
            raise ValueError("base_sim_config_path is required")
        base_path = Path(base_sim)
        if not base_path.is_absolute():
            base_path = (self.config_dir / base_path).resolve()
        self.base_sim_config_path = base_path
        self.base_sim_config = load_yaml(self.base_sim_config_path)

        self.output_dir = self._resolve_path(
            self.config.get("output_dir", "../../results/state-based-hyper-cd/default")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cd_cfg = dict(self.config.get("cd") or {})
        self.n_restarts = int(cd_cfg.get("n_restarts", 5))
        self.max_outer_iters = int(cd_cfg.get("max_outer_iters", 8))
        self.coordinate_order = str(cd_cfg.get("coordinate_order", "shuffle_each_iter"))
        self.patience = int(cd_cfg.get("patience", 2))
        self.min_delta = float(cd_cfg.get("min_delta", 0.0))
        self.init_strategy = str(cd_cfg.get("init_strategy", "random"))
        self.anchor = dict(cd_cfg.get("anchor") or {})
        self.parallel_budget = self._positive_int(
            cd_cfg.get("parallel_budget", 1),
            "cd.parallel_budget",
        )
        self.statistics_config = self._resolve_statistics_config(
            self.config.get("statistics_config")
        )
        if self.coordinate_order not in {"shuffle_each_iter", "shuffle_per_restart", "fixed"}:
            raise ValueError("cd.coordinate_order must be 'shuffle_each_iter', 'shuffle_per_restart', or 'fixed'")
        if self.init_strategy not in {"random", "anchor"}:
            raise ValueError("cd.init_strategy must be 'random' or 'anchor'")

        self._combination_counter = 0

    def _resolve_statistics_config(self, raw: Any) -> Dict[str, Any]:
        return resolve_simulation_stat_config(raw, setting_name="statistics_config")

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
        if out != value and not (isinstance(value, str) and str(out) == value):
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if out <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        return out

    def _resolve_path(self, maybe_path: Any) -> Path:
        p = Path(maybe_path)
        if not p.is_absolute():
            p = (self.config_dir / p).resolve()
        return p

    def resolve_subjects(
        self,
        cli_subjects: Sequence[int] | None,
        cli_subject_range: Sequence[int] | None,
    ) -> List[int]:
        if cli_subjects:
            return [int(x) for x in cli_subjects]
        if cli_subject_range:
            start, end = [int(x) for x in cli_subject_range]
            return list(range(start, end + 1))
        if "subjects" in self.config and self.config["subjects"] is not None:
            return [int(x) for x in self.config["subjects"]]
        if "subject_range" in self.config and self.config["subject_range"] is not None:
            start, end = [int(x) for x in self.config["subject_range"]]
            return list(range(start, end + 1))

        base = self.base_sim_config
        if "subjects" in base and base["subjects"] is not None:
            return [int(x) for x in base["subjects"]]
        if "subject_range" in base and base["subject_range"] is not None:
            start, end = [int(x) for x in base["subject_range"]]
            return list(range(start, end + 1))
        raise ValueError("Unable to resolve subjects from CLI/hyper-CD config/base simulation config")

    def _linspace_values(self, spec: Mapping[str, Any]) -> List[float]:
        start = float(spec["start"])
        stop = float(spec["stop"])
        num = int(spec["num"])
        if num < 2:
            return [start]
        return [float(x) for x in np.linspace(start, stop, num=num, endpoint=True)]

    def _hyperparam_values(self, spec: Mapping[str, Any]) -> List[Any]:
        if "values" in spec:
            vals = list(spec["values"])
            if not vals:
                raise ValueError("hyperparameter values cannot be empty")
            return vals
        if "values_from_json" in spec:
            return values_from_json(spec, self.config_dir)
        if "values_product" in spec:
            return values_product(spec)
        if all(k in spec for k in ("start", "stop", "num")):
            return self._linspace_values(spec)
        raise ValueError(
            "Each hyperparameter spec must provide values, values_from_json, values_product, or (start, stop, num)"
        )

    def _param_specs_for_stage(self, stage_name: str) -> Dict[str, Dict[str, Any]]:
        stages = self.config.get("stages") or {}
        stage_cfg = stages.get(stage_name)
        if not isinstance(stage_cfg, Mapping):
            raise ValueError(f"Missing stage config: stages.{stage_name}")
        if "hyperparam_space" in stage_cfg:
            raw = stage_cfg["hyperparam_space"]
            if not isinstance(raw, Mapping):
                raise ValueError(f"stages.{stage_name}.hyperparam_space must be a mapping")
            validate_no_nested_hyperparam_paths(raw)
            return {k: dict(v) for k, v in raw.items()}

        raw = self.config.get("hyperparam_space")
        if not isinstance(raw, Mapping):
            raise ValueError("hyperparam_space must be a mapping")
        validate_no_nested_hyperparam_paths(raw)
        return {k: dict(v) for k, v in raw.items()}

    def _top_k_combinations_from_coarse(self, coarse_combinations: Sequence[CombinationResult]) -> List[Dict[str, Any]]:
        policy = self.config.get("refine_policy") or {}
        top_k = max(1, int(policy.get("top_k", 3)))
        ranked = rank_by_objectives(
            coarse_combinations,
            lambda combination: combination.objective_values,
            self.objective_order,
            tie_breaker=lambda combination: int(combination.combination_index),
        )
        selected: List[Dict[str, Any]] = []
        seen = set()
        for combination in ranked:
            key = json.dumps(_to_builtin(combination.hyperparams), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            selected.append(deepcopy(combination.hyperparams))
            if len(selected) >= top_k:
                break
        return selected

    def _space_from_combinations(
        self,
        combinations: Sequence[Dict[str, Any]],
        fallback_specs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {}
        expand_policy = (self.config.get("refine_policy") or {}).get("expand") or {}
        if not isinstance(expand_policy, Mapping):
            raise ValueError("refine_policy.expand must be a mapping when provided")
        for name in fallback_specs.keys():
            vals = [combination[name] for combination in combinations if name in combination]
            if not vals:
                vals = self._hyperparam_values(fallback_specs[name])
            if name in expand_policy:
                expanded_values = expand_policy[name]
                vals = (
                    self._hyperparam_values(expanded_values)
                    if isinstance(expanded_values, Mapping)
                    else list(expanded_values)
                )
                if not vals:
                    raise ValueError(f"refine_policy.expand.{name} cannot be empty")
            unique = []
            seen = set()
            for value in vals:
                key = json.dumps(_to_builtin(value), sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(value)
            out[name] = unique
        return out

    def _set_by_path(self, root: Dict[str, Any], path: str, value: Any) -> None:
        curr = root
        parts = path.split(".")
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = deepcopy(value)

    def _apply_single_hyperparam(
        self,
        key: str,
        value: Any,
        next_sim: Dict[str, Any],
        next_engine: Dict[str, Any],
    ) -> None:
        if key.startswith("engine."):
            self._set_by_path(next_engine, key[len("engine."):], value)
        elif key.startswith("simulation."):
            self._set_by_path(next_sim, key[len("simulation."):], value)
        else:
            raise ValueError(f"Hyperparameter key '{key}' must start with 'engine.' or 'simulation.'.")

    def _apply_hyperparams(
        self,
        point: Dict[str, Any],
        sim_cfg: Dict[str, Any],
        engine_cfg: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        next_sim = deepcopy(sim_cfg)
        next_engine = deepcopy(engine_cfg)
        for key, value in expand_profile_candidate_hyperparams(point).items():
            self._apply_single_hyperparam(key, value, next_sim, next_engine)
        next_sim["fixed_hyperparams"] = deepcopy(point)
        return next_sim, next_engine

    def _hyper_candidate_seed(
        self,
        stage_name: str,
        combination_index: int,
        point: Mapping[str, Any],
        restart_id: int,
        iter_id: int,
        coordinate: str,
    ) -> int:
        return derive_hyper_candidate_seed(
            hyper_base_seed=self.hyper_base_seed,
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=point,
            extra_context={
                "restart_id": int(restart_id),
                "iter_id": int(iter_id),
                "coordinate": str(coordinate),
            },
        )

    def _stage_rng(self, subjects: Sequence[int], stage_name: str) -> random.Random:
        seed = stable_seed(
            {
                "seed_role": "hyper_cd_stage_rng",
                "hyper_base_seed": self.hyper_base_seed,
                "stage": stage_name,
                "subjects": [int(sid) for sid in subjects],
            }
        )
        return random.Random(seed)

    def _prepare_stage_config(self, stage_name: str) -> Dict[str, Any]:
        stage_cfg = (self.config.get("stages") or {}).get(stage_name)
        if not isinstance(stage_cfg, Mapping):
            raise ValueError(f"Missing stages.{stage_name}")
        override = stage_cfg.get("simulation_overrides")
        if override is not None and not isinstance(override, Mapping):
            raise ValueError(f"stages.{stage_name}.simulation_overrides must be a mapping")
        merged = deepcopy(self.base_sim_config)
        if override is not None:
            merged = _deep_update(merged, dict(override))
        return merged

    def _resolve_sim_components(
        self,
        sim_cfg: Dict[str, Any],
        subject_id: int,
        subjects: Sequence[int],
    ):
        subject_cfg = resolve_subject_config(sim_cfg, subject_id)
        engine_cfg = resolve_engine_config(subject_cfg, self.base_sim_config_path.parent, subject_id=subject_id)
        prediction_mode, selection_prediction_mode = resolve_prediction_modes(subject_cfg)
        loss_metric = resolve_loss_metric(subject_cfg)
        loss_delta = resolve_loss_delta(subject_cfg, loss_metric)
        window_size = resolve_window_size(subject_cfg, subject_id, subjects)
        n_jobs = int(subject_cfg.get("n_jobs", 1))
        return subject_cfg, engine_cfg, prediction_mode, selection_prediction_mode, loss_metric, loss_delta, window_size, n_jobs

    def _build_runner(self, sim_cfg: Dict[str, Any], engine_cfg: Dict[str, Any]):
        dataset_paths = resolve_dataset_paths(sim_cfg, self.base_sim_config_path.parent)
        runner = StateModelSimulationRunner(
            engine_config=engine_cfg,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            n_jobs=int(sim_cfg.get("n_jobs", 1)),
        )
        runner.prepare_data(dataset_paths["learning_data"])
        return runner, dataset_paths

    def _simulate_runs_for_point(
        self,
        *,
        runner: StateModelSimulationRunner,
        dataset_paths: Mapping[str, Path | str],
        subject_id: int,
        point: Mapping[str, Any],
        simulation_repeats: int,
        window_size: int,
        stop_at: float,
        max_trials: Any,
        keep_logs: bool,
        prediction_mode: str,
        selection_prediction_mode: str,
        loss_metric: str,
        loss_delta: float | None,
        hyper_candidate_seed: int,
        n_jobs: int,
    ):
        subject_frame = runner._get_subject_frame(int(subject_id), float(stop_at))
        condition = runner._get_condition_value(subject_frame)
        max_trials_int = int(max_trials) if max_trials is not None else None
        arrays = runner._extract_arrays(subject_frame, max_trials_int)
        simulation_point_seed = derive_simulation_point_seed(
            int(hyper_candidate_seed),
            int(subject_id),
            dict(point),
        )

        tasks = []
        for repeat_index in range(int(simulation_repeats)):
            trajectory_seed = derive_trajectory_seed(
                int(simulation_point_seed),
                "simulation",
                int(repeat_index),
            )
            tasks.append(
                {
                    "repeat_index": int(repeat_index),
                    "trajectory_seed": trajectory_seed,
                }
            )

        runs = list(
            Parallel(n_jobs=max(1, int(n_jobs)))(
                delayed(evaluate_state_model_run)(
                    int(subject_id),
                    int(condition),
                    arrays,
                    dict(point),
                    runner._engine_config_template,
                    runner._processed_data_dir,
                    int(window_size),
                    dataset_paths,
                    bool(keep_logs),
                    bool(keep_logs),
                    str(prediction_mode),
                    str(selection_prediction_mode),
                    str(loss_metric),
                    loss_delta,
                    simulation_point_seed=int(simulation_point_seed),
                    trajectory_seed=task["trajectory_seed"],
                    seed_context={
                        "hyper_candidate_seed": int(hyper_candidate_seed),
                        "simulation_point_seed": int(simulation_point_seed),
                        "trajectory_seed": task["trajectory_seed"],
                        "phase": "hyper_cd",
                        "repeat_index": task["repeat_index"],
                    },
                )
                for task in tasks
            )
        )
        return [run for run in runs if run is not None], int(condition), int(simulation_point_seed)

    def _select_final_combination(
        self,
        combinations: Sequence[CombinationResult],
    ) -> tuple[CombinationResult, Dict[str, Any]]:
        selected, context = select_best_by_objectives(
            combinations,
            lambda result: result.objective_values,
            self.objective_order,
            tie_breaker=lambda result: (int(result.restart_id), int(result.combination_index)),
        )
        context.update(
            {
                "selected_combination_index": selected.combination_index,
                "selected_restart_id": selected.restart_id,
            }
        )
        return selected, context

    def _evaluate_point(
        self,
        stage_name: str,
        point: Dict[str, Any],
        stage_sim_cfg: Dict[str, Any],
        subjects: Sequence[int],
        restart_id: int,
        iter_id: int,
        coordinate: str,
    ) -> CombinationResult:
        combination_index = self._combination_counter
        self._combination_counter += 1
        return self._evaluate_point_with_index(
            stage_name=stage_name,
            point=point,
            stage_sim_cfg=stage_sim_cfg,
            subjects=subjects,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
            combination_index=combination_index,
        )

    def _evaluate_point_with_index(
        self,
        stage_name: str,
        point: Dict[str, Any],
        stage_sim_cfg: Dict[str, Any],
        subjects: Sequence[int],
        restart_id: int,
        iter_id: int,
        coordinate: str,
        combination_index: int,
    ) -> CombinationResult:
        hyper_candidate_seed = self._hyper_candidate_seed(
            stage_name,
            combination_index,
            point,
            restart_id,
            iter_id,
            coordinate,
        )

        subject_metrics: Dict[int, Dict[str, Any]] = {}
        subject_objectives: List[Dict[str, float]] = []
        for sid in subjects:
            subject_cfg, base_engine_cfg, pred_mode, sel_mode, loss_metric, loss_delta, window_size, n_jobs = self._resolve_sim_components(
                stage_sim_cfg,
                sid,
                subjects,
            )
            point_sim_cfg, point_engine_cfg = self._apply_hyperparams(point, subject_cfg, base_engine_cfg)
            runner, dataset_paths = self._build_runner(point_sim_cfg, point_engine_cfg)
            runner.n_jobs = n_jobs
            simulation_repeats = resolve_simulation_repeats(point_sim_cfg)
            effective_loss_metric = str(point_sim_cfg["loss_metric"])
            effective_loss_delta = resolve_loss_delta(point_sim_cfg, effective_loss_metric)

            runs, condition, simulation_point_seed = self._simulate_runs_for_point(
                runner=runner,
                dataset_paths=dataset_paths,
                subject_id=sid,
                simulation_repeats=simulation_repeats,
                point=point,
                window_size=int(point_sim_cfg.get("window_size", window_size)),
                stop_at=float(point_sim_cfg.get("stop_at", 1.0)),
                max_trials=point_sim_cfg.get("max_trials"),
                keep_logs=bool(point_sim_cfg.get("keep_logs", False)),
                prediction_mode=str(point_sim_cfg.get("prediction_mode", pred_mode)),
                selection_prediction_mode=str(point_sim_cfg.get("selection_prediction_mode", sel_mode)),
                loss_metric=effective_loss_metric,
                loss_delta=effective_loss_delta,
                hyper_candidate_seed=hyper_candidate_seed,
                n_jobs=n_jobs,
            )
            best = aggregate_simulation_runs(
                runs,
                params=point,
                subject_id=sid,
                condition=condition,
                window_size=int(point_sim_cfg.get("window_size", window_size)),
                selection_prediction_mode=str(point_sim_cfg.get("selection_prediction_mode", sel_mode)),
                simulation_repeats=simulation_repeats,
                simulation_point_seed=simulation_point_seed,
                keep_logs=bool(point_sim_cfg.get("keep_logs", False)),
                statistics_config=self.statistics_config,
            )

            mean_err = float(getattr(best, "mean_error"))
            best_err = float(getattr(best, "best_error", mean_err))
            sample_errors = list(getattr(best, "sample_errors", []) or [])
            tail_metrics = _lower_tail_error_metrics(sample_errors, mean_err)
            statistics_summary = dict(getattr(best, "statistics_summary", {}) or {})
            simulation_summary = {
                "mean_error": mean_err,
                "best_error": best_err,
                **tail_metrics,
                "std_error": float(getattr(best, "std_error", 0.0)),
                "sample_errors": sample_errors,
                "simulation_repeats": simulation_repeats,
            }
            subject_record = {
                "simulation": simulation_summary,
                "statistics": statistics_summary,
            }
            objective_values = extract_subject_objective_values(
                subject_record,
                self.objective_order,
            )
            subject_objectives.append(objective_values)
            subject_metrics[int(sid)] = {
                "simulation": simulation_summary,
                "statistics": statistics_summary,
                "objectives": {
                    "values": objective_values,
                },
                "fixed_hyperparams": deepcopy(point),
                "condition": int(condition),
                "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
                "hyper_candidate_seed": int(hyper_candidate_seed),
                "simulation_point_seed": int(simulation_point_seed),
            }

        aggregated_objectives = aggregate_objective_values(subject_objectives, self.objective_order)
        agg_error = first_objective_value(aggregated_objectives, self.objective_order)
        return CombinationResult(
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=deepcopy(point),
            aggregated_error=agg_error,
            objective_values=aggregated_objectives,
            subject_metrics=subject_metrics,
            hyper_candidate_seed=hyper_candidate_seed,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    def _evaluate_missing_entries_flat(
        self,
        *,
        stage_name: str,
        stage_sim_cfg: Dict[str, Any],
        subjects: Sequence[int],
        restart_id: int,
        iter_id: int,
        coordinate: str,
        missing_entries: Sequence[Dict[str, Any]],
        value_jobs: int,
        repeat_jobs: int,
    ) -> tuple[List[CombinationResult], Dict[str, Any]]:
        if not missing_entries:
            return [], {
                "flat_task_count": 0,
                "flat_jobs": 0,
                "parallel_backend": "flat_value_repeat_processes",
            }
        if len(subjects) != 1:
            raise ValueError("Flat CD coordinate evaluation expects exactly one subject.")

        sid = int(subjects[0])
        flat_tasks: List[Dict[str, Any]] = []
        candidate_meta: Dict[int, Dict[str, Any]] = {}

        for entry in missing_entries:
            position = int(entry["position"])
            point = dict(entry["point"])
            combination_index = int(entry["combination_index"])
            hyper_candidate_seed = self._hyper_candidate_seed(
                stage_name,
                combination_index,
                point,
                restart_id,
                iter_id,
                coordinate,
            )

            subject_cfg, base_engine_cfg, pred_mode, sel_mode, loss_metric, loss_delta, window_size, _ = self._resolve_sim_components(
                stage_sim_cfg,
                sid,
                subjects,
            )
            point_sim_cfg, point_engine_cfg = self._apply_hyperparams(point, subject_cfg, base_engine_cfg)
            runner, dataset_paths = self._build_runner(point_sim_cfg, point_engine_cfg)
            simulation_repeats = resolve_simulation_repeats(point_sim_cfg)
            effective_loss_metric = str(point_sim_cfg["loss_metric"])
            effective_loss_delta = resolve_loss_delta(point_sim_cfg, effective_loss_metric)
            effective_window_size = int(point_sim_cfg.get("window_size", window_size))
            keep_logs = bool(point_sim_cfg.get("keep_logs", False))

            subject_frame = runner._get_subject_frame(sid, float(point_sim_cfg.get("stop_at", 1.0)))
            condition = runner._get_condition_value(subject_frame)
            arrays = runner._extract_arrays(subject_frame, point_sim_cfg.get("max_trials"))
            simulation_point_seed = derive_simulation_point_seed(
                int(hyper_candidate_seed),
                sid,
                point,
            )

            candidate_meta[position] = {
                "point": point,
                "combination_index": combination_index,
                "hyper_candidate_seed": int(hyper_candidate_seed),
                "simulation_point_seed": int(simulation_point_seed),
                "subject_id": sid,
                "condition": int(condition),
                "window_size": effective_window_size,
                "selection_prediction_mode": str(point_sim_cfg.get("selection_prediction_mode", sel_mode)),
                "prediction_mode": str(point_sim_cfg.get("prediction_mode", pred_mode)),
                "loss_metric": effective_loss_metric,
                "loss_delta": effective_loss_delta,
                "simulation_repeats": simulation_repeats,
                "keep_logs": keep_logs,
                "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
                "engine_config_template": deepcopy(runner._engine_config_template),
                "processed_data_dir": runner._processed_data_dir,
                "arrays": arrays,
            }

            for repeat_index in range(simulation_repeats):
                trajectory_seed = derive_trajectory_seed(
                    int(simulation_point_seed),
                    "simulation",
                    repeat_index,
                )
                flat_tasks.append(
                    {
                        "position": position,
                        "repeat_index": int(repeat_index),
                        "subject_id": sid,
                        "condition": int(condition),
                        "arrays": arrays,
                        "params": point,
                        "engine_config_template": runner._engine_config_template,
                        "processed_data_dir": runner._processed_data_dir,
                        "window_size": effective_window_size,
                        "dataset_paths": dataset_paths,
                        "keep_logs": keep_logs,
                        "prediction_mode": str(point_sim_cfg.get("prediction_mode", pred_mode)),
                        "selection_prediction_mode": str(point_sim_cfg.get("selection_prediction_mode", sel_mode)),
                        "loss_metric": effective_loss_metric,
                        "loss_delta": effective_loss_delta,
                        "simulation_point_seed": int(simulation_point_seed),
                        "trajectory_seed": trajectory_seed,
                        "seed_context": {
                            "hyper_candidate_seed": int(hyper_candidate_seed),
                            "simulation_point_seed": int(simulation_point_seed),
                            "trajectory_seed": trajectory_seed,
                            "phase": "simulation",
                            "repeat_index": int(repeat_index),
                        },
                    }
                )

        flat_task_count = len(flat_tasks)
        flat_jobs = min(self.parallel_budget, flat_task_count)
        flat_results = list(
            Parallel(n_jobs=flat_jobs)(
                delayed(_evaluate_cd_flat_repeat_task)(task)
                for task in flat_tasks
            )
        )

        runs_by_position: Dict[int, Dict[int, Any]] = {
            position: {} for position in candidate_meta
        }
        for result in flat_results:
            runs_by_position[int(result["position"])][int(result["repeat_index"])] = result["run"]

        out: List[CombinationResult] = []
        for entry in missing_entries:
            position = int(entry["position"])
            meta = candidate_meta[position]
            simulation_repeats = int(meta["simulation_repeats"])
            runs_by_repeat = runs_by_position[position]
            runs = [runs_by_repeat[idx] for idx in range(simulation_repeats)]
            best = aggregate_simulation_runs(
                runs,
                params=meta["point"],
                subject_id=sid,
                condition=int(meta["condition"]),
                window_size=int(meta["window_size"]),
                selection_prediction_mode=str(meta["selection_prediction_mode"]),
                simulation_repeats=simulation_repeats,
                simulation_point_seed=int(meta["simulation_point_seed"]),
                keep_logs=bool(meta["keep_logs"]),
                statistics_config=self.statistics_config,
            )
            mean_error = float(best.mean_error)
            best_error = float(best.best_error if best.best_error is not None else mean_error)
            sample_errors = list(best.sample_errors or [])
            tail_metrics = _lower_tail_error_metrics(sample_errors, mean_error)
            statistics_summary = dict(best.statistics_summary or {})
            simulation_summary = {
                "mean_error": mean_error,
                "best_error": best_error,
                **tail_metrics,
                "std_error": float(best.std_error),
                "sample_errors": sample_errors,
                "simulation_repeats": simulation_repeats,
            }
            subject_record = {
                "simulation": simulation_summary,
                "statistics": statistics_summary,
            }
            objective_values = extract_subject_objective_values(
                subject_record,
                self.objective_order,
            )
            aggregated_objectives = aggregate_objective_values([objective_values], self.objective_order)
            selection_error = first_objective_value(aggregated_objectives, self.objective_order)
            subject_metrics = {
                sid: {
                    "simulation": simulation_summary,
                    "statistics": statistics_summary,
                    "objectives": {
                        "values": objective_values,
                    },
                    "fixed_hyperparams": deepcopy(meta["point"]),
                    "condition": int(meta["condition"]),
                    "dataset_paths": dict(meta["dataset_paths"]),
                    "hyper_candidate_seed": int(meta["hyper_candidate_seed"]),
                    "simulation_point_seed": int(meta["simulation_point_seed"]),
                }
            }
            out.append(
                CombinationResult(
                    stage=stage_name,
                    combination_index=int(meta["combination_index"]),
                    hyperparams=deepcopy(meta["point"]),
                    aggregated_error=selection_error,
                    objective_values=aggregated_objectives,
                    subject_metrics=subject_metrics,
                    hyper_candidate_seed=int(meta["hyper_candidate_seed"]),
                    restart_id=restart_id,
                    iter_id=iter_id,
                    coordinate=coordinate,
                )
            )

        return out, {
            "flat_task_count": flat_task_count,
            "flat_jobs": flat_jobs,
            "parallel_backend": "flat_value_repeat_processes",
            "planned_total_jobs": value_jobs * repeat_jobs,
        }

    def _next_combination_index(self) -> int:
        out = self._combination_counter
        self._combination_counter += 1
        return out

    def _stage_cd_parallel_config(self, stage_name: str) -> Dict[str, int]:
        stages = self.config.get("stages") or {}
        stage_cfg = stages.get(stage_name)
        if not isinstance(stage_cfg, Mapping):
            raise ValueError(f"Missing stages.{stage_name}")
        cd_parallel = stage_cfg.get("cd_parallel")
        if not isinstance(cd_parallel, Mapping):
            raise ValueError(f"stages.{stage_name}.cd_parallel.max_repeat_jobs is required for hyper-CD.")
        if "max_repeat_jobs" not in cd_parallel:
            raise ValueError(f"stages.{stage_name}.cd_parallel.max_repeat_jobs is required for hyper-CD.")
        return {
            "max_repeat_jobs": self._positive_int(
                cd_parallel["max_repeat_jobs"],
                f"stages.{stage_name}.cd_parallel.max_repeat_jobs",
            )
        }

    def _coordinate_parallel_plan(
        self,
        stage_name: str,
        stage_sim_cfg: Dict[str, Any],
        num_values: int,
    ) -> tuple[int, int]:
        if num_values <= 0:
            return 1, 1
        cd_parallel = self._stage_cd_parallel_config(stage_name)
        simulation_repeats = resolve_simulation_repeats(stage_sim_cfg)
        repeat_jobs = min(cd_parallel["max_repeat_jobs"], simulation_repeats, self.parallel_budget)
        value_jobs = min(int(num_values), max(1, self.parallel_budget // repeat_jobs))
        return max(1, int(value_jobs)), max(1, int(repeat_jobs))

    @staticmethod
    def _stage_sim_cfg_with_n_jobs(stage_sim_cfg: Dict[str, Any], repeat_jobs: int) -> Dict[str, Any]:
        out = deepcopy(stage_sim_cfg)
        out["n_jobs"] = int(repeat_jobs)
        return out

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_builtin(payload), ensure_ascii=False, allow_nan=False) + "\n")

    def _serialize_combination_record(self, result: CombinationResult) -> Dict[str, Any]:
        data = {
            "schema_version": HYPER_RESULT_SCHEMA_VERSION,
            "stage": result.stage,
            "combination_index": result.combination_index,
            "restart_id": result.restart_id,
            "iter_id": result.iter_id,
            "coordinate": result.coordinate,
            "hyperparams": result.hyperparams,
            "aggregated_error": result.aggregated_error,
            "objective_values": result.objective_values,
            "hyper_candidate_seed": result.hyper_candidate_seed,
        }
        metrics_summary = combination_metrics_summary(
            result.subject_metrics,
            aggregated_error=result.aggregated_error,
            objective_values=result.objective_values,
        )
        if metrics_summary:
            data["metrics_summary"] = metrics_summary
        if self.save_level == "full":
            data["subject_metrics"] = result.subject_metrics
        return data

    def _load_jsonl_records(self, path: Path) -> List[Dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot resume fine stage; missing coarse combinations file: {path}")
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(f"JSONL record must be a mapping at {path}:{line_no}")
                records.append(dict(payload))
        return records

    def _write_jsonl_records(self, path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(_to_builtin(record), ensure_ascii=False, allow_nan=False) + "\n")

    def _trim_jsonl_to_stage(self, path: Path, stage: str) -> None:
        if not path.is_file():
            return
        records = self._load_jsonl_records(path)
        kept = [record for record in records if record.get("stage") == stage]
        if len(kept) != len(records):
            self._write_jsonl_records(path, kept)

    def _combination_from_record(self, record: Mapping[str, Any], path: Path) -> CombinationResult:
        hyperparams = record.get("hyperparams")
        if not isinstance(hyperparams, Mapping):
            raise ValueError(f"Combination record is missing hyperparams in {path}")

        raw_subject_metrics = record.get("subject_metrics")
        subject_metrics: Dict[int, Dict[str, Any]] = {}
        if isinstance(raw_subject_metrics, Mapping):
            subject_metrics = {
                int(sid): dict(metrics)
                for sid, metrics in raw_subject_metrics.items()
                if isinstance(metrics, Mapping)
            }

        return CombinationResult(
            stage=str(record.get("stage", "")),
            combination_index=int(record["combination_index"]),
            hyperparams=deepcopy(dict(hyperparams)),
            aggregated_error=float(record["aggregated_error"]),
            objective_values=deepcopy(dict(record["objective_values"])),
            subject_metrics=subject_metrics,
            hyper_candidate_seed=int(record["hyper_candidate_seed"]),
            restart_id=int(record.get("restart_id", -1)),
            iter_id=int(record.get("iter_id", -1)),
            coordinate=str(record.get("coordinate", "loaded_coarse")),
        )

    def _load_coarse_for_fine_resume(self, path: Path) -> List[CombinationResult]:
        records = self._load_jsonl_records(path)
        coarse_records = [record for record in records if record.get("stage") == "coarse"]
        if not coarse_records:
            raise ValueError(f"Cannot resume fine stage; no coarse records found in {path}")
        if len(coarse_records) != len(records):
            self._write_jsonl_records(path, coarse_records)
        combinations = [self._combination_from_record(record, path) for record in coarse_records]
        max_existing_index = max(int(record["combination_index"]) for record in coarse_records)
        self._combination_counter = max(self._combination_counter, max_existing_index + 1)
        return combinations

    def _init_point(self, space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
        if self.init_strategy == "anchor":
            point = {}
            for name, vals in space.items():
                point[name] = self.anchor[name] if name in self.anchor else vals[0]
            return point
        return {name: rng.choice(list(vals)) for name, vals in space.items()}

    def _coordinate_descent(
        self,
        stage_name: str,
        stage_sim_cfg: Dict[str, Any],
        subjects: Sequence[int],
        space: Dict[str, List[Any]],
        all_combinations_path: Path,
        coordinate_trace_path: Path | None = None,
        rng: random.Random | None = None,
    ) -> tuple[List[CombinationResult], List[Dict[str, Any]], CombinationResult]:
        if rng is None:
            rng = self._stage_rng(subjects, stage_name)
        all_combinations: List[CombinationResult] = []
        restart_best: List[Dict[str, Any]] = []
        global_best: CombinationResult | None = None
        restart_local_bests: List[CombinationResult] = []
        cache: Dict[str, CombinationResult] = {}
        coords_base = list(space.keys())

        def eval_with_cache(
            point: Dict[str, Any],
            restart_id: int,
            iter_id: int,
            coordinate: str,
            repeat_jobs: int,
        ) -> tuple[CombinationResult, bool]:
            key = json.dumps(_to_builtin(point), sort_keys=True)
            if key in cache:
                return cache[key], False
            result = self._evaluate_point_with_index(
                stage_name=stage_name,
                point=point,
                stage_sim_cfg=self._stage_sim_cfg_with_n_jobs(stage_sim_cfg, repeat_jobs),
                subjects=subjects,
                restart_id=restart_id,
                iter_id=iter_id,
                coordinate=coordinate,
                combination_index=self._next_combination_index(),
            )
            cache[key] = result
            self._append_jsonl(all_combinations_path, self._serialize_combination_record(result))
            return result, True

        for restart_id in range(self.n_restarts):
            current = self._init_point(space, rng)
            restart_coords = list(coords_base)
            if self.coordinate_order == "shuffle_per_restart":
                rng.shuffle(restart_coords)
            _, init_repeat_jobs = self._coordinate_parallel_plan(stage_name, stage_sim_cfg, 1)
            current_result, current_is_new = eval_with_cache(
                current,
                restart_id,
                0,
                "init",
                init_repeat_jobs,
            )
            restart_new_evaluations = int(current_is_new)
            restart_cache_hits = int(not current_is_new)
            if current_is_new:
                all_combinations.append(current_result)
            best_local = current_result
            anchor_values = dict(best_local.objective_values)
            initial_result = current_result
            no_improve_rounds = 0
            outer_iters_completed = 0
            stopped_by = "max_outer_iters"
            improvements: List[Dict[str, Any]] = []

            for iter_id in range(1, self.max_outer_iters + 1):
                outer_iters_completed = iter_id
                if self.coordinate_order == "shuffle_each_iter":
                    coords = list(coords_base)
                    rng.shuffle(coords)
                elif self.coordinate_order == "shuffle_per_restart":
                    coords = list(restart_coords)
                else:
                    coords = list(coords_base)
                improved_this_round = False

                for coord_index, coord in enumerate(coords):
                    start_best = best_local
                    candidate_best = best_local
                    candidate_best_guard: Dict[str, Any] = {"checks": []}
                    base_point = deepcopy(current)
                    candidate_count = 0
                    coord_new_evaluations = 0
                    coord_cache_hits = 0
                    anchor_reject_count = 0
                    value_jobs, repeat_jobs = self._coordinate_parallel_plan(
                        stage_name,
                        stage_sim_cfg,
                        len(space[coord]),
                    )
                    candidate_entries: List[Dict[str, Any]] = []
                    missing_entries: List[Dict[str, Any]] = []
                    for value in space[coord]:
                        candidate_count += 1
                        candidate = deepcopy(base_point)
                        candidate[coord] = value
                        key = json.dumps(_to_builtin(candidate), sort_keys=True)
                        entry = {
                            "position": candidate_count - 1,
                            "key": key,
                            "point": candidate,
                        }
                        if key in cache:
                            entry["result"] = cache[key]
                            entry["is_new"] = False
                            coord_cache_hits += 1
                        else:
                            entry["is_new"] = True
                            entry["combination_index"] = self._next_combination_index()
                            missing_entries.append(entry)
                        candidate_entries.append(entry)

                    if missing_entries:
                        evaluated_missing, flat_diag = self._evaluate_missing_entries_flat(
                            stage_name=stage_name,
                            stage_sim_cfg=stage_sim_cfg,
                            subjects=subjects,
                            restart_id=restart_id,
                            iter_id=iter_id,
                            coordinate=coord,
                            missing_entries=missing_entries,
                            value_jobs=value_jobs,
                            repeat_jobs=repeat_jobs,
                        )
                        by_position = {
                            int(entry["position"]): result
                            for entry, result in zip(missing_entries, evaluated_missing)
                        }
                    else:
                        flat_diag = {
                            "flat_task_count": 0,
                            "flat_jobs": 0,
                            "parallel_backend": "flat_value_repeat_processes",
                            "planned_total_jobs": value_jobs * repeat_jobs,
                        }
                        by_position = {}

                    for entry in candidate_entries:
                        if entry["is_new"]:
                            candidate_result = by_position[int(entry["position"])]
                            cache[str(entry["key"])] = candidate_result
                            self._append_jsonl(all_combinations_path, self._serialize_combination_record(candidate_result))
                            all_combinations.append(candidate_result)
                            coord_new_evaluations += 1
                        else:
                            candidate_result = entry["result"]
                        passed_guard, guard_context = passes_anchor_guard(
                            candidate_result.objective_values,
                            anchor_values,
                            self.objective_order,
                        )
                        if not passed_guard:
                            anchor_reject_count += 1
                            continue
                        if (
                            compare_objective_values(
                                candidate_result.objective_values,
                                candidate_best.objective_values,
                                self.objective_order,
                            )
                            < 0
                        ):
                            candidate_best = candidate_result
                            candidate_best_guard = guard_context

                    restart_new_evaluations += coord_new_evaluations
                    restart_cache_hits += coord_cache_hits
                    improved_coord = False
                    if (
                        candidate_best.combination_index != best_local.combination_index
                        and compare_objective_values(
                            candidate_best.objective_values,
                            best_local.objective_values,
                            self.objective_order,
                        )
                        < 0
                    ):
                        current = deepcopy(candidate_best.hyperparams)
                        best_local = candidate_best
                        anchor_values = update_anchor_values(
                            anchor_values,
                            best_local.objective_values,
                            self.objective_order,
                        )
                        improved_this_round = True
                        improved_coord = True
                        improvements.append(
                            {
                                "iter_id": iter_id,
                                "coordinate": coord,
                                "from_combination_index": start_best.combination_index,
                                "to_combination_index": best_local.combination_index,
                                "from_error": start_best.aggregated_error,
                                "to_error": best_local.aggregated_error,
                                "from_objective_values": start_best.objective_values,
                                "to_objective_values": best_local.objective_values,
                                "anchor_values": anchor_values,
                                "anchor_guard": candidate_best_guard,
                                "selected_hyperparams": best_local.hyperparams,
                            }
                        )

                    if coordinate_trace_path is not None:
                        self._append_jsonl(
                            coordinate_trace_path,
                            {
                                "stage": stage_name,
                                "restart_id": restart_id,
                                "iter_id": iter_id,
                                "coordinate": coord,
                                "coordinate_index": coord_index,
                                "coordinate_order": coords,
                                "candidate_count": candidate_count,
                                "missing_value_count": len(missing_entries),
                                "new_evaluations": coord_new_evaluations,
                                "cache_hits": coord_cache_hits,
                                "anchor_reject_count": anchor_reject_count,
                                "value_jobs": value_jobs,
                                "repeat_jobs": repeat_jobs,
                                "planned_total_jobs": flat_diag["planned_total_jobs"],
                                "flat_task_count": flat_diag["flat_task_count"],
                                "flat_jobs": flat_diag["flat_jobs"],
                                "parallel_backend": flat_diag["parallel_backend"],
                                "start_best_combination_index": start_best.combination_index,
                                "start_best_error": start_best.aggregated_error,
                                "start_best_objective_values": start_best.objective_values,
                                "end_best_combination_index": best_local.combination_index,
                                "end_best_error": best_local.aggregated_error,
                                "end_best_objective_values": best_local.objective_values,
                                "anchor_values": anchor_values,
                                "improved": improved_coord,
                            },
                        )

                if improved_this_round:
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.patience:
                        stopped_by = "patience"
                        break

            restart_best.append(
                {
                    "restart_id": restart_id,
                    "initial_combination_index": initial_result.combination_index,
                    "initial_error": initial_result.aggregated_error,
                    "initial_objective_values": initial_result.objective_values,
                    "best_combination_index": best_local.combination_index,
                    "best_error": best_local.aggregated_error,
                    "best_objective_values": best_local.objective_values,
                    "anchor_values": anchor_values,
                    "best_hyperparams": best_local.hyperparams,
                    "best_params": compact_hyperparams(best_local.hyperparams),
                    "outer_iters_completed": outer_iters_completed,
                    "stopped_by": stopped_by,
                    "no_improve_rounds": no_improve_rounds,
                    "num_improvements": len(improvements),
                    "num_new_evaluations": restart_new_evaluations,
                    "num_cache_hits": restart_cache_hits,
                    "coordinate_order": restart_coords,
                    "improvements": improvements,
                }
            )
            restart_local_bests.append(best_local)

        if not restart_local_bests:
            raise RuntimeError("CD optimizer produced no combination")
        global_best, _ = select_best_by_objectives(
            restart_local_bests,
            lambda result: result.objective_values,
            self.objective_order,
            tie_breaker=lambda result: (int(result.restart_id), int(result.combination_index)),
        )
        return all_combinations, restart_best, global_best

    def _run_subject_pipeline(
        self,
        subject_id: int,
        stage: str,
        output_base: Path,
        resume_from_coarse: bool = False,
    ) -> Dict[str, Any]:
        subject_dir = output_base / f"subject_{int(subject_id)}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        return self._run_pipeline(
            subjects=[int(subject_id)],
            stage=stage,
            output_dir=subject_dir,
            resume_from_coarse=resume_from_coarse,
        )

    def _run_pipeline(
        self,
        subjects: Sequence[int],
        stage: str,
        output_dir: Path,
        resume_from_coarse: bool = False,
    ) -> Dict[str, Any]:
        if resume_from_coarse and stage != "fine":
            raise ValueError("resume_from_coarse requires stage='fine'")

        self._combination_counter = 0
        if stage == "all":
            configured_stages = self.config.get("stages") or {}
            stages_to_run = [name for name in ("coarse", "fine") if name in configured_stages]
            if not stages_to_run:
                raise ValueError("stage='all' requires at least one configured stage under stages.coarse or stages.fine")
        else:
            stages_to_run = [stage]
        all_combinations_path = output_dir / "all_combinations.jsonl"
        if resume_from_coarse:
            stage_combinations: Dict[str, List[CombinationResult]] = {
                "coarse": self._load_coarse_for_fine_resume(all_combinations_path)
            }
        elif all_combinations_path.exists():
            all_combinations_path.unlink()
            stage_combinations = {}
        else:
            stage_combinations = {}

        coordinate_trace_path = output_dir / "coordinate_trace.jsonl"
        if resume_from_coarse:
            self._trim_jsonl_to_stage(coordinate_trace_path, "coarse")
        elif coordinate_trace_path.exists():
            coordinate_trace_path.unlink()

        stage_restarts: Dict[str, Any] = {}
        for stage_name in stages_to_run:
            stage_sim_cfg = self._prepare_stage_config(stage_name)
            if stage_name == "fine":
                fine_stage_cfg = (self.config.get("stages") or {}).get("fine") or {}
                if "hyperparam_space" in fine_stage_cfg:
                    specs = self._param_specs_for_stage(stage_name)
                    space = {name: self._hyperparam_values(spec) for name, spec in specs.items()}
                else:
                    prior = stage_combinations.get("coarse")
                    if prior is None:
                        raise ValueError("fine stage without hyperparam_space requires coarse stage results")
                    coarse_top = self._top_k_combinations_from_coarse(prior)
                    coarse_specs = self._param_specs_for_stage("coarse")
                    space = self._space_from_combinations(coarse_top, coarse_specs)
            else:
                specs = self._param_specs_for_stage(stage_name)
                space = {name: self._hyperparam_values(spec) for name, spec in specs.items()}

            combinations, restarts, _ = self._coordinate_descent(
                stage_name=stage_name,
                stage_sim_cfg=stage_sim_cfg,
                subjects=subjects,
                space=space,
                all_combinations_path=all_combinations_path,
                coordinate_trace_path=coordinate_trace_path,
                rng=self._stage_rng(subjects, stage_name),
            )
            stage_combinations[stage_name] = combinations
            stage_restarts[stage_name] = restarts

        stage_summary = self._build_stage_summary(stage_combinations)
        final_stage = "fine" if "fine" in stage_combinations else "coarse"
        final_combinations = stage_combinations[final_stage]
        best_combination, final_selection_context = self._select_final_combination(final_combinations)

        stage_summary_path = output_dir / "stage_summary.json"
        with stage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2, allow_nan=False)

        restart_summary_path = output_dir / "restart_summary.json"
        with restart_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_restarts), f, ensure_ascii=False, indent=2, allow_nan=False)

        metrics = None
        if len(subjects) == 1:
            sid = int(subjects[0])
            metrics = best_combination.subject_metrics[sid]
        best_payload = build_subject_best_payload(
            subject_id=int(subjects[0]) if len(subjects) == 1 else -1,
            backend="hyper_cd",
            hyper_base_seed=self.hyper_base_seed,
            objective_order=self.objective_order_config,
            objective_values=best_combination.objective_values,
            best_stage=final_stage,
            best_combination_index=best_combination.combination_index,
            best_hyperparams=best_combination.hyperparams,
            aggregated_error=best_combination.aggregated_error,
            hyper_candidate_seed=best_combination.hyper_candidate_seed,
            metrics=metrics,
            search_context={
                "restart_id": best_combination.restart_id,
                "iter_id": best_combination.iter_id,
                "coordinate": best_combination.coordinate,
                "objectives": {"order": self.objective_order_config},
                "final_selection": final_selection_context,
            },
            provenance=build_hyper_provenance(
                config_path=self.config_path,
                output_dir=output_dir,
                base_sim_config_path=self.base_sim_config_path,
            ),
            artifacts=build_subject_artifacts(output_dir, include_cd=True),
            full_subject_metrics=(
                best_combination.subject_metrics
                if self.save_level == "full"
                else None
            ),
        )

        best_path = output_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(best_payload), f, ensure_ascii=False, indent=2, allow_nan=False)

        return {
            "output_dir": str(output_dir),
            "all_combinations": str(all_combinations_path),
            "stage_summary": str(stage_summary_path),
            "restart_summary": str(restart_summary_path),
            "coordinate_trace": str(coordinate_trace_path),
            "best_hyperparams": str(best_path),
            "best": best_payload,
        }

    def _build_stage_summary(self, stage_combinations: Mapping[str, Sequence[CombinationResult]]) -> Dict[str, Any]:
        top_k = int((self.config.get("refine_policy") or {}).get("top_k", 3))
        summary: Dict[str, Any] = {}
        for stage_name, combinations in stage_combinations.items():
            ranked = rank_by_objectives(
                combinations,
                lambda combination: combination.objective_values,
                self.objective_order,
                tie_breaker=lambda combination: (int(combination.restart_id), int(combination.combination_index)),
            )
            summary[stage_name] = {
                "num_combinations": len(combinations),
                "top_combinations": [
                    {
                        "combination_index": result.combination_index,
                        "aggregated_error": result.aggregated_error,
                        "objective_values": result.objective_values,
                        "hyperparams": result.hyperparams,
                        "best_params": compact_hyperparams(result.hyperparams),
                        "restart_id": result.restart_id,
                        "iter_id": result.iter_id,
                        "coordinate": result.coordinate,
                        "hyper_candidate_seed": result.hyper_candidate_seed,
                    }
                    for result in ranked[:max(1, top_k)]
                ],
            }
        return summary

    def run(self, subjects: Sequence[int], stage: str = "all", resume_from_coarse: bool = False) -> Dict[str, Any]:
        if stage not in {"coarse", "fine", "all"}:
            raise ValueError("stage must be one of: coarse, fine, all")
        if resume_from_coarse and stage != "fine":
            raise ValueError("resume_from_coarse requires stage='fine'")

        per_subject_best: Dict[str, Any] = {}
        per_subject_outputs: Dict[str, Any] = {}
        for sid in subjects:
            out = self._run_subject_pipeline(
                int(sid),
                stage,
                self.output_dir,
                resume_from_coarse=resume_from_coarse,
            )
            per_subject_outputs[str(int(sid))] = {
                "output_dir": out["output_dir"],
                "all_combinations": out["all_combinations"],
                "stage_summary": out["stage_summary"],
                "restart_summary": out["restart_summary"],
                "coordinate_trace": out["coordinate_trace"],
                "best_hyperparams": out["best_hyperparams"],
            }
            per_subject_best[str(int(sid))] = out["best"]

        best_payload = build_root_best_payload(
            backend="hyper_cd",
            config_path=self.config_path,
            output_dir=self.output_dir,
            base_sim_config_path=self.base_sim_config_path,
            hyper_base_seed=self.hyper_base_seed,
            objective_order=self.objective_order_config,
            save_level=self.save_level,
            subjects=subjects,
            per_subject_best=per_subject_best,
            per_subject_outputs=per_subject_outputs,
        )
        best_path = self.output_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(best_payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        return {
            "output_dir": str(self.output_dir),
            "per_subject_outputs": per_subject_outputs,
            "best_hyperparams": str(best_path),
            "best": best_payload,
        }


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _to_builtin(obj: Any) -> Any:
    return to_builtin(obj)


__all__ = ["HyperCDOptimizer", "CombinationResult"]
