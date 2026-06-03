"""Two-layer hyperparameter optimization orchestrator for Bayesian_state."""
from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import yaml
from joblib import Parallel, delayed

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
from src.Bayesian_state.utils.optimizer_grid import StateModelGridOptimizer
from src.Bayesian_state.utils.paths import ROOT_DIR
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
    random_seed: int


class HyperOptimizer:
    """Outer optimizer that chooses hyperparameters via inner grid/amr optimizers."""

    def __init__(self, config: Mapping[str, Any], config_path: Path) -> None:
        self.config = dict(config)
        self.config_path = config_path
        self.config_dir = config_path.parent

        self.inner_optimizer = str(self.config.get("inner_optimizer", "")).strip()
        if self.inner_optimizer not in {"grid", "amr"}:
            raise ValueError("inner_optimizer must be 'grid' or 'amr'")

        self.selection_metric = str(self.config.get("selection_metric", "min_inner_mean_error"))
        if self.selection_metric != "min_inner_mean_error":
            raise ValueError("Only selection_metric='min_inner_mean_error' is supported in v1")
        self.hyperparam_selection_mode = str(
            self.config.get("hyperparam_selection_mode", "per_subject")
        ).strip().lower()
        if self.hyperparam_selection_mode not in {"per_subject", "group_mean"}:
            raise ValueError("hyperparam_selection_mode must be 'per_subject' or 'group_mean'")
        self.save_level = str(self.config.get("save_level", "compact")).strip().lower()
        if self.save_level not in {"compact", "full"}:
            raise ValueError("save_level must be 'compact' or 'full'")

        self.base_seed = int(self.config.get("random_seed", 1234))

        inner_base = self.config.get("inner_base_config_path")
        if not inner_base:
            raise ValueError("inner_base_config_path is required")
        inner_path = Path(inner_base)
        if not inner_path.is_absolute():
            inner_path = (self.config_dir / inner_path).resolve()
        self.inner_base_config_path = inner_path
        self.inner_base_config = self._load_yaml(self.inner_base_config_path)

        self.output_dir = self._resolve_path(self.config.get("output_dir", "../../results/state-based-hyper-opt/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _expand_combinations(self, param_specs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        names = list(param_specs.keys())
        value_lists = [self._hyperparam_values(param_specs[n]) for n in names]
        combos = []
        for combo in itertools.product(*value_lists):
            combos.append(dict(zip(names, combo)))
        return combos

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
            elif key.startswith("inner."):
                self._set_by_path(next_inner, key[len("inner."):], val)
            else:
                raise ValueError(f"Hyperparameter key '{key}' must start with 'engine.' or 'inner.'")

        return next_inner, next_engine

    def _combination_seed(self, stage_name: str, combination_index: int, combination_params: Mapping[str, Any]) -> int:
        payload = json.dumps({"stage": stage_name, "idx": combination_index, "params": combination_params, "base_seed": self.base_seed}, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

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

    def _evaluate_combination(self, stage_name: str, combination_index: int, combination_params: Dict[str, Any], stage_inner_cfg: Dict[str, Any], subjects: Sequence[int]) -> CombinationResult:
        seed = self._combination_seed(stage_name, combination_index, combination_params)
        np.random.seed(seed)

        subject_metrics: Dict[int, Dict[str, Any]] = {}
        errors: List[float] = []

        for sid in subjects:
            subject_cfg, base_engine_cfg, pred_mode, sel_mode, loss_metric, loss_delta, window_size, n_jobs = self._resolve_inner_components(
                stage_inner_cfg, sid, subjects, self.inner_base_config_path
            )
            combination_inner_cfg, combination_engine_cfg = self._apply_hyperparams(combination_params, subject_cfg, base_engine_cfg)
            if self.inner_optimizer == "grid":
                param_grid = resolve_param_grid_grid(combination_inner_cfg)
            else:
                param_grid = resolve_param_grid_amr(combination_inner_cfg)

            mod = combination_engine_cfg.get("modules", {}).get("hypo_transitions_mod", {}).get("kwargs", {})
            if isinstance(mod, dict) and "random_seed" not in mod:
                mod["random_seed"] = int(seed)

            optimizer, dataset_paths = self._build_optimizer(combination_inner_cfg, combination_engine_cfg, self.inner_base_config_path)
            optimizer.n_jobs = n_jobs

            effective_loss_metric = str(combination_inner_cfg["loss_metric"])
            if self.inner_optimizer == "grid":
                effective_loss_delta = resolve_loss_delta_grid(combination_inner_cfg, effective_loss_metric)
            else:
                effective_loss_delta = resolve_loss_delta_amr(combination_inner_cfg, effective_loss_metric)
            result = optimizer.optimize_subject(
                subject_id=sid,
                param_grid=param_grid,
                n_repeats=int(combination_inner_cfg.get("n_repeats", 1)),
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

            best = result["best"]
            mean_error = float(getattr(best, "mean_error"))
            errors.append(mean_error)
            subject_metrics[int(sid)] = {
                "mean_error": mean_error,
                "best_error": float(getattr(best, "best_error", mean_error)),
                "best_params": dict(getattr(best, "params", {})),
                "condition": int(result.get("condition", -1)),
                "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
            }

        agg_error = float(np.mean(errors)) if errors else float("inf")
        return CombinationResult(
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=deepcopy(combination_params),
            aggregated_error=agg_error,
            subject_metrics=subject_metrics,
            random_seed=seed,
        )

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_builtin(payload), ensure_ascii=False) + "\n")

    def _serialize_combination_record(self, combination: CombinationResult) -> Dict[str, Any]:
        data = {
            "stage": combination.stage,
            "combination_index": combination.combination_index,
            "hyperparams": combination.hyperparams,
            "aggregated_error": combination.aggregated_error,
            "random_seed": combination.random_seed,
        }
        if self.save_level == "full":
            data["subject_metrics"] = combination.subject_metrics
        return data

    def _evaluate_stage_combinations(
        self,
        stage_name: str,
        stage_inner_cfg: Dict[str, Any],
        stage_combinations: Sequence[Dict[str, Any]],
        subjects: Sequence[int],
    ) -> List[CombinationResult]:
        stage_cfg = (self.config.get("stages") or {}).get(stage_name) or {}
        n_jobs_combinations = max(1, int(stage_cfg.get("n_jobs_combinations", 1)))
        if n_jobs_combinations == 1:
            return [
                self._evaluate_combination(stage_name, idx, params, stage_inner_cfg, subjects)
                for idx, params in enumerate(stage_combinations)
            ]
        return list(
            Parallel(n_jobs=n_jobs_combinations)(
                delayed(self._evaluate_combination)(stage_name, idx, params, stage_inner_cfg, subjects)
                for idx, params in enumerate(stage_combinations)
            )
        )

    def _run_subject_pipeline(self, subject_id: int, stage: str, output_base: Path) -> Dict[str, Any]:
        stages_to_run = ["coarse", "fine"] if stage == "all" else [stage]
        subject_dir = output_base / f"subject_{int(subject_id)}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        all_combinations_path = subject_dir / "all_combinations.jsonl"
        if all_combinations_path.exists():
            all_combinations_path.unlink()

        all_stage_combinations: Dict[str, List[CombinationResult]] = {}
        for stage_name in stages_to_run:
            stage_inner_cfg = self._prepare_stage_config(stage_name)
            if stage_name == "fine":
                fine_stage_cfg = (self.config.get("stages") or {}).get("fine") or {}
                if "hyperparam_space" in fine_stage_cfg:
                    specs = self._param_specs_for_stage(stage_name)
                    stage_combinations = self._expand_combinations(specs)
                else:
                    prior = all_stage_combinations.get("coarse")
                    if prior is None:
                        raise ValueError(
                            "fine stage without stages.fine.hyperparam_space requires coarse stage results in this run"
                        )
                    stage_combinations = self._top_k_combinations_from_coarse(prior)
            else:
                specs = self._param_specs_for_stage(stage_name)
                stage_combinations = self._expand_combinations(specs)

            combination_results = self._evaluate_stage_combinations(
                stage_name,
                stage_inner_cfg,
                stage_combinations,
                [int(subject_id)],
            )
            for result in combination_results:
                self._append_jsonl(all_combinations_path, self._serialize_combination_record(result))
            all_stage_combinations[stage_name] = combination_results

        stage_summary: Dict[str, Any] = {}
        for stage_name, combinations in all_stage_combinations.items():
            ranked = sorted(combinations, key=lambda x: x.aggregated_error)
            top_k = int((self.config.get("refine_policy") or {}).get("top_k", 3))
            stage_summary[stage_name] = {
                "num_combinations": len(combinations),
                "top_combinations": [
                    {
                        "combination_index": t.combination_index,
                        "aggregated_error": t.aggregated_error,
                        "hyperparams": t.hyperparams,
                        "random_seed": t.random_seed,
                    }
                    for t in ranked[:max(1, top_k)]
                ],
            }

        final_stage = "fine" if "fine" in all_stage_combinations else "coarse"
        final_combinations = all_stage_combinations[final_stage]
        best_combination = min(final_combinations, key=lambda t: float(t.subject_metrics[int(subject_id)]["mean_error"]))

        stage_summary_path = subject_dir / "stage_summary.json"
        with stage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2)

        subject_best = {
            "best_stage": final_stage,
            "best_combination_index": best_combination.combination_index,
            "best_hyperparams": best_combination.hyperparams,
            "mean_error": float(best_combination.subject_metrics[int(subject_id)]["mean_error"]),
            "best_error": float(best_combination.subject_metrics[int(subject_id)]["best_error"]),
            "random_seed": best_combination.random_seed,
        }
        if self.save_level == "full":
            subject_best["subject_metrics"] = {str(int(subject_id)): best_combination.subject_metrics[int(subject_id)]}

        best_path = subject_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(subject_best), f, ensure_ascii=False, indent=2)

        return {
            "subject_id": int(subject_id),
            "output_dir": str(subject_dir),
            "all_combinations": str(all_combinations_path),
            "stage_summary": str(stage_summary_path),
            "best_hyperparams": str(best_path),
            "best": subject_best,
        }

    def run(self, subjects: Sequence[int], stage: str = "all") -> Dict[str, Any]:
        if stage not in {"coarse", "fine", "all"}:
            raise ValueError("stage must be one of: coarse, fine, all")

        best_payload: Dict[str, Any] = {
            "selection_metric": self.selection_metric,
            "hyperparam_selection_mode": self.hyperparam_selection_mode,
            "save_level": self.save_level,
            "inner_base_config_path": str(self.inner_base_config_path),
            "hyper_config_path": str(self.config_path),
        }

        if self.hyperparam_selection_mode == "group_mean":
            stages_to_run = ["coarse", "fine"] if stage == "all" else [stage]
            all_combinations_path = self.output_dir / "all_combinations.jsonl"
            if all_combinations_path.exists():
                all_combinations_path.unlink()

            all_stage_combinations: Dict[str, List[CombinationResult]] = {}
            for stage_name in stages_to_run:
                stage_inner_cfg = self._prepare_stage_config(stage_name)
                if stage_name == "fine":
                    fine_stage_cfg = (self.config.get("stages") or {}).get("fine") or {}
                    if "hyperparam_space" in fine_stage_cfg:
                        specs = self._param_specs_for_stage(stage_name)
                        stage_combinations = self._expand_combinations(specs)
                    else:
                        prior = all_stage_combinations.get("coarse")
                        if prior is None:
                            raise ValueError(
                                "fine stage without stages.fine.hyperparam_space requires coarse stage results in this run"
                            )
                        stage_combinations = self._top_k_combinations_from_coarse(prior)
                else:
                    specs = self._param_specs_for_stage(stage_name)
                    stage_combinations = self._expand_combinations(specs)

                combination_results = self._evaluate_stage_combinations(
                    stage_name,
                    stage_inner_cfg,
                    stage_combinations,
                    subjects,
                )
                for result in combination_results:
                    self._append_jsonl(all_combinations_path, self._serialize_combination_record(result))
                all_stage_combinations[stage_name] = combination_results

            stage_summary = {}
            for stage_name, combinations in all_stage_combinations.items():
                ranked = sorted(combinations, key=lambda x: x.aggregated_error)
                top_k = int((self.config.get("refine_policy") or {}).get("top_k", 3))
                stage_summary[stage_name] = {
                    "num_combinations": len(combinations),
                    "top_combinations": [
                        {
                            "combination_index": t.combination_index,
                            "aggregated_error": t.aggregated_error,
                            "hyperparams": t.hyperparams,
                            "random_seed": t.random_seed,
                        }
                        for t in ranked[:max(1, top_k)]
                    ],
                }

            final_stage = "fine" if "fine" in all_stage_combinations else "coarse"
            final_combinations = all_stage_combinations[final_stage]
            stage_summary_path = self.output_dir / "stage_summary.json"
            with stage_summary_path.open("w", encoding="utf-8") as f:
                json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2)

            best_payload["best_stage"] = final_stage
            best_combination = min(final_combinations, key=lambda x: x.aggregated_error)
            best_payload.update({
                "best_combination_index": best_combination.combination_index,
                "best_hyperparams": best_combination.hyperparams,
                "aggregated_error": best_combination.aggregated_error,
                "random_seed": best_combination.random_seed,
            })
            if self.save_level == "full":
                best_payload["subject_metrics"] = best_combination.subject_metrics
            best_path = self.output_dir / "best_hyperparams.json"
            with best_path.open("w", encoding="utf-8") as f:
                json.dump(_to_builtin(best_payload), f, ensure_ascii=False, indent=2)
            return {
                "output_dir": str(self.output_dir),
                "all_combinations": str(all_combinations_path),
                "stage_summary": str(stage_summary_path),
                "best_hyperparams": str(best_path),
                "best": best_payload,
            }
        else:
            per_subject_best: Dict[str, Any] = {}
            per_subject_outputs: Dict[str, Any] = {}
            for sid in subjects:
                out = self._run_subject_pipeline(int(sid), stage, self.output_dir)
                per_subject_outputs[str(int(sid))] = {
                    "output_dir": out["output_dir"],
                    "all_combinations": out["all_combinations"],
                    "stage_summary": out["stage_summary"],
                    "best_hyperparams": out["best_hyperparams"],
                }
                per_subject_best[str(int(sid))] = out["best"]
            best_payload["per_subject_best"] = per_subject_best
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


__all__ = ["HyperOptimizer", "CombinationResult"]
