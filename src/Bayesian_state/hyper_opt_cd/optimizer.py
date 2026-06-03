"""Coordinate-descent two-layer hyper optimizer for Bayesian_state."""
from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import yaml

from src.Bayesian_state.run_amr_optimization import (
    resolve_engine_config as resolve_engine_config_amr,
    resolve_loss_delta as resolve_loss_delta_amr,
    resolve_loss_metric as resolve_loss_metric_amr,
    resolve_param_grid as resolve_param_grid_amr,
    resolve_prediction_modes as resolve_prediction_modes_amr,
    resolve_window_size as resolve_window_size_amr,
)
from src.Bayesian_state.run_grid_optimization import (
    resolve_engine_config as resolve_engine_config_grid,
    resolve_loss_delta as resolve_loss_delta_grid,
    resolve_loss_metric as resolve_loss_metric_grid,
    resolve_param_grid as resolve_param_grid_grid,
    resolve_prediction_modes as resolve_prediction_modes_grid,
    resolve_window_size as resolve_window_size_grid,
)
from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.optimizer_amr import StateModelAMROptimizer
from src.Bayesian_state.utils.optimizer_common import derive_hyper_candidate_seed
from src.Bayesian_state.utils.optimizer_grid import StateModelGridOptimizer
from src.Bayesian_state.hyper_opt.value_sources import (
    validate_no_nested_hyperparam_paths,
    values_from_json,
)


@dataclass
class CombinationResult:
    stage: str
    combination_index: int
    hyperparams: Dict[str, Any]
    aggregated_error: float
    subject_metrics: Dict[int, Dict[str, Any]]
    hyper_candidate_seed: int
    restart_id: int
    iter_id: int
    coordinate: str


class HyperOptimizerCD:
    """Outer hyper optimizer using coordinate descent."""

    def __init__(self, config: Mapping[str, Any], config_path: Path) -> None:
        self.config = dict(config)
        self.config_path = config_path
        self.config_dir = config_path.parent

        self.inner_optimizer = str(self.config.get("inner_optimizer", "")).strip()
        if self.inner_optimizer not in {"grid", "amr"}:
            raise ValueError("inner_optimizer must be 'grid' or 'amr'")

        self.selection_metric = str(self.config.get("selection_metric", "min_inner_mean_error"))
        if self.selection_metric != "min_inner_mean_error":
            raise ValueError("Only selection_metric='min_inner_mean_error' is supported")

        self.hyperparam_selection_mode = str(
            self.config.get("hyperparam_selection_mode", "per_subject")
        ).strip().lower()
        if self.hyperparam_selection_mode not in {"per_subject", "group_mean"}:
            raise ValueError("hyperparam_selection_mode must be 'per_subject' or 'group_mean'")

        self.save_level = str(self.config.get("save_level", "compact")).strip().lower()
        if self.save_level not in {"compact", "full"}:
            raise ValueError("save_level must be 'compact' or 'full'")

        if "hyper_base_seed" not in self.config:
            raise ValueError("Hyper config must include hyper_base_seed. The old random_seed field is no longer supported.")
        self.hyper_base_seed = int(self.config["hyper_base_seed"])
        self.rng = random.Random(self.hyper_base_seed)

        inner_base = self.config.get("inner_base_config_path")
        if not inner_base:
            raise ValueError("inner_base_config_path is required")
        inner_path = Path(inner_base)
        if not inner_path.is_absolute():
            inner_path = (self.config_dir / inner_path).resolve()
        self.inner_base_config_path = inner_path
        self.inner_base_config = self._load_yaml(self.inner_base_config_path)

        self.output_dir = self._resolve_path(
            self.config.get("output_dir", "../../results/state-based-hyper-opt/default_cd")
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
        if self.coordinate_order not in {"shuffle_each_iter", "fixed"}:
            raise ValueError("cd.coordinate_order must be 'shuffle_each_iter' or 'fixed'")
        if self.init_strategy not in {"random", "anchor"}:
            raise ValueError("cd.init_strategy must be 'random' or 'anchor'")

        self._combination_counter = 0

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML must be a mapping: {path}")
        return data

    def _resolve_path(self, maybe_path: Any) -> Path:
        p = Path(maybe_path)
        if not p.is_absolute():
            p = (self.config_dir / p).resolve()
        return p

    def resolve_subjects(self, cli_subjects: Sequence[int] | None, cli_subject_range: Sequence[int] | None) -> List[int]:
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
        base = self.inner_base_config
        if "subjects" in base and base["subjects"] is not None:
            return [int(x) for x in base["subjects"]]
        if "subject_range" in base and base["subject_range"] is not None:
            start, end = [int(x) for x in base["subject_range"]]
            return list(range(start, end + 1))
        raise ValueError("Unable to resolve subjects from CLI/hyper config/inner config")

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
        if all(k in spec for k in ("start", "stop", "num")):
            return self._linspace_values(spec)
        raise ValueError("Each hyperparameter spec must provide values, values_from_json, or (start, stop, num)")

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
        ranked = sorted(coarse_combinations, key=lambda x: x.aggregated_error)
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

    def _space_from_combinations(self, combinations: Sequence[Dict[str, Any]], fallback_specs: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {}
        for name in fallback_specs.keys():
            vals = []
            for combination in combinations:
                if name in combination:
                    vals.append(combination[name])
            if not vals:
                vals = self._hyperparam_values(fallback_specs[name])
            unique = []
            seen = set()
            for v in vals:
                k = json.dumps(_to_builtin(v), sort_keys=True)
                if k in seen:
                    continue
                seen.add(k)
                unique.append(v)
            out[name] = unique
        return out

    def _set_by_path(self, root: Dict[str, Any], path: str, value: Any) -> None:
        curr = root
        parts = path.split(".")
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = deepcopy(value)

    def _apply_hyperparams(self, combination: Dict[str, Any], inner_cfg: Dict[str, Any], engine_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        next_inner = deepcopy(inner_cfg)
        next_engine = deepcopy(engine_cfg)
        for key, val in combination.items():
            if key.startswith("engine."):
                self._set_by_path(next_engine, key[len("engine."):], val)
            elif key.startswith("inner.param_grid."):
                raise ValueError(
                    "Do not tune inner.param_grid.* in hyperparam_space. "
                    "Put gamma/w0 and other inner-grid params under "
                    "stages.<coarse|fine>.inner_overrides.param_grid."
                )
            elif key.startswith("inner."):
                self._set_by_path(next_inner, key[len("inner."):], val)
            else:
                raise ValueError(
                    f"Hyperparameter key '{key}' must start with 'engine.' or 'inner.'"
                )
        return next_inner, next_engine

    def _hyper_candidate_seed(
        self,
        stage_name: str,
        combination_index: int,
        combination_params: Mapping[str, Any],
        restart_id: int,
        iter_id: int,
        coordinate: str,
    ) -> int:
        return derive_hyper_candidate_seed(
            hyper_base_seed=self.hyper_base_seed,
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=combination_params,
            extra_context={
                "restart_id": int(restart_id),
                "iter_id": int(iter_id),
                "coordinate": str(coordinate),
            },
        )

    def _prepare_stage_config(self, stage_name: str) -> Dict[str, Any]:
        stage_cfg = (self.config.get("stages") or {}).get(stage_name)
        if not isinstance(stage_cfg, Mapping):
            raise ValueError(f"Missing stages.{stage_name}")
        if "inner_overrides" in stage_cfg and not isinstance(stage_cfg["inner_overrides"], Mapping):
            raise ValueError(f"stages.{stage_name}.inner_overrides must be a mapping")
        merged = deepcopy(self.inner_base_config)
        if "inner_overrides" in stage_cfg:
            merged = _deep_update(merged, dict(stage_cfg["inner_overrides"]))
        return merged

    def _build_optimizer(self, inner_cfg: Dict[str, Any], engine_cfg: Dict[str, Any], cfg_path: Path):
        dataset_paths = resolve_dataset_paths(inner_cfg, cfg_path.parent)
        if self.inner_optimizer == "grid":
            optimizer = StateModelGridOptimizer(
                engine_config=engine_cfg,
                processed_data_dir=dataset_paths["processed_dir"],
                dataset_paths=dataset_paths,
                n_jobs=int(inner_cfg.get("n_jobs", 1)),
            )
        else:
            optimizer = StateModelAMROptimizer(
                engine_config=engine_cfg,
                processed_data_dir=str(dataset_paths["processed_dir"]),
                dataset_paths=dataset_paths,
                amr_kwargs=dict(inner_cfg.get("amr_kwargs") or {}),
                n_jobs=int(inner_cfg.get("n_jobs_inner", 1)),
            )
        optimizer.prepare_data(dataset_paths["learning_data"])
        return optimizer, dataset_paths

    def _resolve_inner_components(self, inner_cfg: Dict[str, Any], subject_id: int, subjects: Sequence[int], cfg_path: Path):
        subject_cfg = resolve_subject_config(inner_cfg, subject_id)
        if self.inner_optimizer == "grid":
            engine_cfg = resolve_engine_config_grid(subject_cfg, cfg_path.parent, subject_id=subject_id)
            prediction_mode, selection_prediction_mode = resolve_prediction_modes_grid(subject_cfg)
            loss_metric = resolve_loss_metric_grid(subject_cfg)
            loss_delta = resolve_loss_delta_grid(subject_cfg, loss_metric)
            window_size = resolve_window_size_grid(subject_cfg, subject_id, subjects)
            n_jobs = int(subject_cfg.get("n_jobs", 1))
        else:
            engine_cfg = resolve_engine_config_amr(subject_cfg, cfg_path.parent, subject_id=subject_id)
            prediction_mode, selection_prediction_mode = resolve_prediction_modes_amr(subject_cfg)
            loss_metric = resolve_loss_metric_amr(subject_cfg)
            loss_delta = resolve_loss_delta_amr(subject_cfg, loss_metric)
            window_size = resolve_window_size_amr(subject_cfg, subject_id, subjects)
            n_jobs = int(subject_cfg.get("n_jobs_inner", 1))
        return subject_cfg, engine_cfg, prediction_mode, selection_prediction_mode, loss_metric, loss_delta, window_size, n_jobs

    def _evaluate_point(
        self,
        stage_name: str,
        point: Dict[str, Any],
        stage_inner_cfg: Dict[str, Any],
        subjects: Sequence[int],
        restart_id: int,
        iter_id: int,
        coordinate: str,
    ) -> CombinationResult:
        combination_index = self._combination_counter
        self._combination_counter += 1
        hyper_candidate_seed = self._hyper_candidate_seed(
            stage_name,
            combination_index,
            point,
            restart_id,
            iter_id,
            coordinate,
        )

        subject_metrics: Dict[int, Dict[str, Any]] = {}
        errors: List[float] = []
        for sid in subjects:
            subject_cfg, base_engine_cfg, pred_mode, sel_mode, loss_metric, loss_delta, window_size, n_jobs = self._resolve_inner_components(
                stage_inner_cfg, sid, subjects, self.inner_base_config_path
            )
            combination_inner_cfg, combination_engine_cfg = self._apply_hyperparams(point, subject_cfg, base_engine_cfg)
            if self.inner_optimizer == "grid":
                param_grid = resolve_param_grid_grid(combination_inner_cfg)
            else:
                param_grid = resolve_param_grid_amr(combination_inner_cfg)

            optimizer, dataset_paths = self._build_optimizer(combination_inner_cfg, combination_engine_cfg, self.inner_base_config_path)
            optimizer.n_jobs = n_jobs
            effective_loss_metric = str(combination_inner_cfg["loss_metric"])
            if self.inner_optimizer == "grid":
                effective_loss_delta = resolve_loss_delta_grid(combination_inner_cfg, effective_loss_metric)
            else:
                effective_loss_delta = resolve_loss_delta_amr(combination_inner_cfg, effective_loss_metric)
            if "grid_repeats" not in combination_inner_cfg:
                raise ValueError("Inner config must include grid_repeats. The old n_repeats field is no longer supported.")
            repeat_count = int(combination_inner_cfg["grid_repeats"])
            common_kwargs = dict(
                subject_id=sid,
                param_grid=param_grid,
                refit_repeats=int(combination_inner_cfg.get("refit_repeats", 0)),
                window_size=int(combination_inner_cfg.get("window_size", window_size)),
                stop_at=float(combination_inner_cfg.get("stop_at", 1.0)),
                max_trials=combination_inner_cfg.get("max_trials"),
                keep_logs=bool(combination_inner_cfg.get("keep_logs", False)),
                prediction_mode=str(combination_inner_cfg.get("prediction_mode", pred_mode)),
                selection_prediction_mode=str(combination_inner_cfg.get("selection_prediction_mode", sel_mode)),
                loss_metric=effective_loss_metric,
                loss_delta=effective_loss_delta,
                random_seed=seed,
            )
            if self.inner_optimizer == "grid":
                result = optimizer.optimize_subject(
                    **common_kwargs,
                    grid_repeats=repeat_count,
                    hyper_candidate_seed=hyper_candidate_seed,
                )
            else:
                result = optimizer.optimize_subject(
                    **common_kwargs,
                    grid_repeats=repeat_count,
                )
            best = result["best"]
            mean_error = float(getattr(best, "mean_error"))
            errors.append(mean_error)
            subject_metrics[int(sid)] = {
                "mean_error": mean_error,
                "best_error": float(getattr(best, "best_error", mean_error)),
                "best_params": dict(getattr(best, "params", {})),
                "condition": int(result.get("condition", -1)),
                "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
                "hyper_candidate_seed": int(hyper_candidate_seed),
            }

        agg_error = float(np.mean(errors)) if errors else float("inf")
        return CombinationResult(
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=deepcopy(point),
            aggregated_error=agg_error,
            subject_metrics=subject_metrics,
            hyper_candidate_seed=hyper_candidate_seed,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_builtin(payload), ensure_ascii=False) + "\n")

    def _serialize_combination_record(self, tr: CombinationResult) -> Dict[str, Any]:
        data = {
            "stage": tr.stage,
            "combination_index": tr.combination_index,
            "restart_id": tr.restart_id,
            "iter_id": tr.iter_id,
            "coordinate": tr.coordinate,
            "hyperparams": tr.hyperparams,
            "aggregated_error": tr.aggregated_error,
            "hyper_candidate_seed": tr.hyper_candidate_seed,
        }
        if self.save_level == "full":
            data["subject_metrics"] = tr.subject_metrics
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
                f.write(json.dumps(_to_builtin(record), ensure_ascii=False) + "\n")

    def _trim_jsonl_to_stage(self, path: Path, stage: str) -> None:
        if not path.is_file():
            return
        records = self._load_jsonl_records(path)
        kept = [record for record in records if record.get("stage") == stage]
        if len(kept) != len(records):
            self._write_jsonl_records(path, kept)

    def _combination_from_record(self, record: Mapping[str, Any], path: Path) -> CombinationResult:
        stage = str(record.get("stage", "")).strip()
        if not stage:
            raise ValueError(f"Combination record is missing stage in {path}")

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
            stage=stage,
            combination_index=int(record["combination_index"]),
            hyperparams=deepcopy(dict(hyperparams)),
            aggregated_error=float(record["aggregated_error"]),
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

        # Drop stale or partial fine rows before appending the new fine stage.
        if len(coarse_records) != len(records):
            self._write_jsonl_records(path, coarse_records)

        combinations = [self._combination_from_record(record, path) for record in coarse_records]
        max_existing_index = max(int(record["combination_index"]) for record in coarse_records)
        self._combination_counter = max(self._combination_counter, max_existing_index + 1)
        return combinations

    def _init_point(self, space: Dict[str, List[Any]]) -> Dict[str, Any]:
        if self.init_strategy == "anchor":
            point = {}
            for k, vals in space.items():
                if k in self.anchor:
                    point[k] = self.anchor[k]
                else:
                    point[k] = vals[0]
            return point
        return {k: self.rng.choice(list(vals)) for k, vals in space.items()}

    def _coordinate_descent(
        self,
        stage_name: str,
        stage_inner_cfg: Dict[str, Any],
        subjects: Sequence[int],
        space: Dict[str, List[Any]],
        all_combinations_path: Path,
        coordinate_trace_path: Path | None = None,
    ) -> tuple[List[CombinationResult], List[Dict[str, Any]], CombinationResult]:
        all_combinations: List[CombinationResult] = []
        restart_best: List[Dict[str, Any]] = []
        global_best: CombinationResult | None = None
        cache: Dict[str, CombinationResult] = {}
        coords_base = list(space.keys())

        def eval_with_cache(
            point: Dict[str, Any],
            restart_id: int,
            iter_id: int,
            coordinate: str,
        ) -> tuple[CombinationResult, bool]:
            key = json.dumps(_to_builtin(point), sort_keys=True)
            if key in cache:
                cached = cache[key]
                return cached, False
            tr = self._evaluate_point(
                stage_name=stage_name,
                point=point,
                stage_inner_cfg=stage_inner_cfg,
                subjects=subjects,
                restart_id=restart_id,
                iter_id=iter_id,
                coordinate=coordinate,
            )
            cache[key] = tr
            self._append_jsonl(all_combinations_path, self._serialize_combination_record(tr))
            return tr, True

        for restart_id in range(self.n_restarts):
            current = self._init_point(space)
            current_tr, current_is_new = eval_with_cache(current, restart_id, 0, "init")
            restart_new_evaluations = int(current_is_new)
            restart_cache_hits = int(not current_is_new)
            if current_is_new:
                all_combinations.append(current_tr)
            best_local = current_tr
            initial_tr = current_tr
            no_improve_rounds = 0
            outer_iters_completed = 0
            stopped_by = "max_outer_iters"
            improvements: List[Dict[str, Any]] = []

            for iter_id in range(1, self.max_outer_iters + 1):
                outer_iters_completed = iter_id
                coords = list(coords_base)
                if self.coordinate_order == "shuffle_each_iter":
                    self.rng.shuffle(coords)
                improved_this_round = False

                for coord in coords:
                    start_best = best_local
                    candidate_best = best_local
                    base_point = deepcopy(current)
                    candidate_count = 0
                    coord_new_evaluations = 0
                    coord_cache_hits = 0
                    for val in space[coord]:
                        candidate_count += 1
                        cand = deepcopy(base_point)
                        cand[coord] = val
                        cand_tr, cand_is_new = eval_with_cache(cand, restart_id, iter_id, coord)
                        if cand_is_new:
                            all_combinations.append(cand_tr)
                            coord_new_evaluations += 1
                        else:
                            coord_cache_hits += 1
                        if cand_tr.aggregated_error + self.min_delta < candidate_best.aggregated_error:
                            candidate_best = cand_tr
                    restart_new_evaluations += coord_new_evaluations
                    restart_cache_hits += coord_cache_hits
                    improved_coord = False
                    if candidate_best.aggregated_error + self.min_delta < best_local.aggregated_error:
                        current = deepcopy(candidate_best.hyperparams)
                        best_local = candidate_best
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
                                "candidate_count": candidate_count,
                                "new_evaluations": coord_new_evaluations,
                                "cache_hits": coord_cache_hits,
                                "start_best_combination_index": start_best.combination_index,
                                "start_best_error": start_best.aggregated_error,
                                "end_best_combination_index": best_local.combination_index,
                                "end_best_error": best_local.aggregated_error,
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
                    "initial_combination_index": initial_tr.combination_index,
                    "initial_error": initial_tr.aggregated_error,
                    "best_combination_index": best_local.combination_index,
                    "best_error": best_local.aggregated_error,
                    "best_hyperparams": best_local.hyperparams,
                    "outer_iters_completed": outer_iters_completed,
                    "stopped_by": stopped_by,
                    "no_improve_rounds": no_improve_rounds,
                    "num_improvements": len(improvements),
                    "num_new_evaluations": restart_new_evaluations,
                    "num_cache_hits": restart_cache_hits,
                    "improvements": improvements,
                }
            )
            if global_best is None or best_local.aggregated_error < global_best.aggregated_error:
                global_best = best_local

        if global_best is None:
            raise RuntimeError("CD optimizer produced no combination")
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

        stages_to_run = ["coarse", "fine"] if stage == "all" else [stage]
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
            stage_inner_cfg = self._prepare_stage_config(stage_name)
            if stage_name == "fine":
                fine_stage_cfg = (self.config.get("stages") or {}).get("fine") or {}
                if "hyperparam_space" in fine_stage_cfg:
                    specs = self._param_specs_for_stage(stage_name)
                    space = {k: self._hyperparam_values(v) for k, v in specs.items()}
                else:
                    prior = stage_combinations.get("coarse")
                    if prior is None:
                        raise ValueError(
                            "fine stage without stages.fine.hyperparam_space requires coarse stage results in this run"
                        )
                    coarse_top = self._top_k_combinations_from_coarse(prior)
                    coarse_specs = self._param_specs_for_stage("coarse")
                    space = self._space_from_combinations(coarse_top, coarse_specs)
            else:
                specs = self._param_specs_for_stage(stage_name)
                space = {k: self._hyperparam_values(v) for k, v in specs.items()}

            combinations, restarts, _ = self._coordinate_descent(
                stage_name=stage_name,
                stage_inner_cfg=stage_inner_cfg,
                subjects=subjects,
                space=space,
                all_combinations_path=all_combinations_path,
                coordinate_trace_path=coordinate_trace_path,
            )
            stage_combinations[stage_name] = combinations
            stage_restarts[stage_name] = restarts

        stage_summary = {}
        top_k = int((self.config.get("refine_policy") or {}).get("top_k", 3))
        for stage_name, combinations in stage_combinations.items():
            ranked = sorted(combinations, key=lambda x: x.aggregated_error)
            stage_summary[stage_name] = {
                "num_combinations": len(combinations),
                "top_combinations": [
                    {
                        "combination_index": t.combination_index,
                        "aggregated_error": t.aggregated_error,
                        "hyperparams": t.hyperparams,
                        "restart_id": t.restart_id,
                        "iter_id": t.iter_id,
                        "coordinate": t.coordinate,
                        "hyper_candidate_seed": t.hyper_candidate_seed,
                    }
                    for t in ranked[:max(1, top_k)]
                ],
            }

        final_stage = "fine" if "fine" in stage_combinations else "coarse"
        final_combinations = stage_combinations[final_stage]
        best_combination = min(final_combinations, key=lambda x: x.aggregated_error)

        stage_summary_path = output_dir / "stage_summary.json"
        with stage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2)

        restart_summary_path = output_dir / "restart_summary.json"
        with restart_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_restarts), f, ensure_ascii=False, indent=2)

        best_payload: Dict[str, Any] = {
            "best_stage": final_stage,
            "best_combination_index": best_combination.combination_index,
            "best_hyperparams": best_combination.hyperparams,
            "aggregated_error": best_combination.aggregated_error,
            "hyper_candidate_seed": best_combination.hyper_candidate_seed,
            "hyper_base_seed": self.hyper_base_seed,
            "hyper_backend": "cd",
        }
        if len(subjects) == 1:
            sid = int(subjects[0])
            best_payload["mean_error"] = float(best_combination.subject_metrics[sid]["mean_error"])
            best_payload["best_error"] = float(best_combination.subject_metrics[sid]["best_error"])
        if self.save_level == "full":
            best_payload["subject_metrics"] = best_combination.subject_metrics

        best_path = output_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(best_payload), f, ensure_ascii=False, indent=2)

        return {
            "output_dir": str(output_dir),
            "all_combinations": str(all_combinations_path),
            "stage_summary": str(stage_summary_path),
            "restart_summary": str(restart_summary_path),
            "coordinate_trace": str(coordinate_trace_path),
            "best_hyperparams": str(best_path),
            "best": best_payload,
        }

    def run(self, subjects: Sequence[int], stage: str = "all", resume_from_coarse: bool = False) -> Dict[str, Any]:
        if stage not in {"coarse", "fine", "all"}:
            raise ValueError("stage must be one of: coarse, fine, all")
        if resume_from_coarse and stage != "fine":
            raise ValueError("resume_from_coarse requires stage='fine'")

        if self.hyperparam_selection_mode == "group_mean":
            return self._run_pipeline(
                subjects=subjects,
                stage=stage,
                output_dir=self.output_dir,
                resume_from_coarse=resume_from_coarse,
            )

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

        best_payload = {
            "selection_metric": self.selection_metric,
            "hyperparam_selection_mode": self.hyperparam_selection_mode,
            "save_level": self.save_level,
            "inner_base_config_path": str(self.inner_base_config_path),
            "hyper_config_path": str(self.config_path),
            "hyper_backend": "cd",
            "hyper_base_seed": self.hyper_base_seed,
            "per_subject_best": per_subject_best,
        }
        best_path = self.output_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(best_payload), f, ensure_ascii=False, indent=2)
        return {
            "output_dir": str(self.output_dir),
            "per_subject_outputs": per_subject_outputs,
            "best_hyperparams": str(best_path),
            "best": best_payload,
        }


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    return obj


__all__ = ["HyperOptimizerCD", "CombinationResult"]
