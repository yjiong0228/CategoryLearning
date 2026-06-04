"""Joint grid hyperparameter optimizer backed by repeated simulations."""
from __future__ import annotations

import itertools
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed

from src.Bayesian_state.utils.optimization_config import (
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.run_simulation import resolve_simulation_repeats
from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.optimizer_common import derive_hyper_candidate_seed
from src.Bayesian_state.utils.optimizer_simulation import StateModelSimulationRunner
from src.Bayesian_state.utils.hyperparam_values import (
    validate_no_nested_hyperparam_paths,
    values_from_json,
)


SELECTION_METRICS = {"mean_simulation_error"}


@dataclass
class CombinationResult:
    stage: str
    combination_index: int
    hyperparams: Dict[str, Any]
    aggregated_error: float
    subject_metrics: Dict[int, Dict[str, Any]]
    hyper_candidate_seed: int


class HyperGridOptimizer:
    """Choose all model hyperparameters jointly via explicit grid candidates."""

    def __init__(self, config: Mapping[str, Any], config_path: Path) -> None:
        self.config = dict(config)
        self.config_path = config_path
        self.config_dir = config_path.parent

        self.selection_metric = str(self.config.get("selection_metric", "mean_simulation_error"))
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {sorted(SELECTION_METRICS)}")

        self.hyperparam_selection_mode = str(
            self.config.get("hyperparam_selection_mode", "per_subject")
        ).strip().lower()
        if self.hyperparam_selection_mode not in {"per_subject", "group_mean"}:
            raise ValueError("hyperparam_selection_mode must be 'per_subject' or 'group_mean'")

        self.save_level = str(self.config.get("save_level", "compact")).strip().lower()
        if self.save_level not in {"compact", "full"}:
            raise ValueError("save_level must be 'compact' or 'full'")

        if "hyper_base_seed" not in self.config:
            raise ValueError("Hyper-grid config must include hyper_base_seed.")
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
            self.config.get("output_dir", "../../results/state-based-hyper-grid/default")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        raise ValueError("Unable to resolve subjects from CLI/hyper-grid config/base simulation config")

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
        return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]

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

    def _refine_combinations_from_coarse(self, coarse_combinations: Sequence[CombinationResult]) -> List[Dict[str, Any]]:
        selected = self._top_k_combinations_from_coarse(coarse_combinations)
        policy = self.config.get("refine_policy") or {}
        expand = policy.get("expand") or {}
        if not isinstance(expand, Mapping) or not expand:
            return selected

        expand_names = list(expand.keys())
        expand_values = []
        for name in expand_names:
            values = expand[name]
            if isinstance(values, Mapping):
                values = self._hyperparam_values(values)
            else:
                values = list(values)
            if not values:
                raise ValueError(f"refine_policy.expand.{name} cannot be empty")
            expand_values.append(values)

        refined: List[Dict[str, Any]] = []
        seen = set()
        for base in selected:
            for combo in itertools.product(*expand_values):
                point = deepcopy(base)
                for name, value in zip(expand_names, combo):
                    point[name] = deepcopy(value)
                key = json.dumps(_to_builtin(point), sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                refined.append(point)
        return refined

    def _set_by_path(self, root: Dict[str, Any], path: str, value: Any) -> None:
        curr = root
        parts = path.split(".")
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = deepcopy(value)

    def _apply_hyperparams(
        self,
        combination: Dict[str, Any],
        sim_cfg: Dict[str, Any],
        engine_cfg: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        next_sim = deepcopy(sim_cfg)
        next_engine = deepcopy(engine_cfg)
        for key, val in combination.items():
            if key.startswith("engine."):
                self._set_by_path(next_engine, key[len("engine."):], val)
            elif key.startswith("simulation."):
                self._set_by_path(next_sim, key[len("simulation."):], val)
            else:
                raise ValueError(
                    f"Hyperparameter key '{key}' must start with 'engine.' or 'simulation.'."
                )
        next_sim["fixed_hyperparams"] = deepcopy(combination)
        return next_sim, next_engine

    def _hyper_candidate_seed(
        self,
        stage_name: str,
        combination_index: int,
        combination_params: Mapping[str, Any],
    ) -> int:
        return derive_hyper_candidate_seed(
            hyper_base_seed=self.hyper_base_seed,
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=combination_params,
        )

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

    def _evaluate_combination(
        self,
        stage_name: str,
        combination_index: int,
        combination_params: Dict[str, Any],
        stage_sim_cfg: Dict[str, Any],
        subjects: Sequence[int],
    ) -> CombinationResult:
        hyper_candidate_seed = self._hyper_candidate_seed(stage_name, combination_index, combination_params)
        subject_metrics: Dict[int, Dict[str, Any]] = {}
        errors: List[float] = []

        for sid in subjects:
            subject_cfg, base_engine_cfg, pred_mode, sel_mode, loss_metric, loss_delta, window_size, n_jobs = self._resolve_sim_components(
                stage_sim_cfg,
                sid,
                subjects,
            )
            combination_sim_cfg, combination_engine_cfg = self._apply_hyperparams(
                combination_params,
                subject_cfg,
                base_engine_cfg,
            )
            runner, dataset_paths = self._build_runner(combination_sim_cfg, combination_engine_cfg)
            runner.n_jobs = n_jobs
            simulation_repeats = resolve_simulation_repeats(combination_sim_cfg)
            effective_loss_metric = str(combination_sim_cfg["loss_metric"])
            effective_loss_delta = resolve_loss_delta(combination_sim_cfg, effective_loss_metric)

            result = runner.simulate_subject(
                subject_id=sid,
                simulation_repeats=simulation_repeats,
                fixed_hyperparams=combination_params,
                window_size=int(combination_sim_cfg.get("window_size", window_size)),
                stop_at=float(combination_sim_cfg.get("stop_at", 1.0)),
                max_trials=combination_sim_cfg.get("max_trials"),
                keep_logs=bool(combination_sim_cfg.get("keep_logs", False)),
                prediction_mode=str(combination_sim_cfg.get("prediction_mode", pred_mode)),
                selection_prediction_mode=str(combination_sim_cfg.get("selection_prediction_mode", sel_mode)),
                loss_metric=effective_loss_metric,
                loss_delta=effective_loss_delta,
                hyper_candidate_seed=hyper_candidate_seed,
            )

            best = result["best"]
            mean_error = float(getattr(best, "mean_error"))
            errors.append(mean_error)
            subject_metrics[int(sid)] = {
                "mean_error": mean_error,
                "best_error": float(getattr(best, "best_error", mean_error)),
                "std_error": float(getattr(best, "std_error", 0.0)),
                "sample_errors": list(getattr(best, "sample_errors", []) or []),
                "fixed_hyperparams": deepcopy(combination_params),
                "condition": int(result.get("condition", -1)),
                "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
                "hyper_candidate_seed": int(hyper_candidate_seed),
                "simulation_repeats": simulation_repeats,
            }

        agg_error = float(np.mean(errors)) if errors else float("inf")
        return CombinationResult(
            stage=stage_name,
            combination_index=combination_index,
            hyperparams=deepcopy(combination_params),
            aggregated_error=agg_error,
            subject_metrics=subject_metrics,
            hyper_candidate_seed=hyper_candidate_seed,
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
            "hyper_candidate_seed": combination.hyper_candidate_seed,
        }
        if self.save_level == "full":
            data["subject_metrics"] = combination.subject_metrics
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

    def _combination_from_record(self, record: Mapping[str, Any], path: Path) -> CombinationResult:
        hyperparams = record.get("hyperparams")
        if not isinstance(hyperparams, Mapping):
            raise ValueError(f"Combination record is missing hyperparams in {path}")
        return CombinationResult(
            stage=str(record.get("stage", "")),
            combination_index=int(record["combination_index"]),
            hyperparams=deepcopy(dict(hyperparams)),
            aggregated_error=float(record["aggregated_error"]),
            subject_metrics={},
            hyper_candidate_seed=int(record["hyper_candidate_seed"]),
        )

    def _load_coarse_for_fine_resume(self, path: Path) -> List[CombinationResult]:
        records = self._load_jsonl_records(path)
        coarse_records = [record for record in records if record.get("stage") == "coarse"]
        if not coarse_records:
            raise ValueError(f"Cannot resume fine stage; no coarse records found in {path}")
        if len(coarse_records) != len(records):
            self._write_jsonl_records(path, coarse_records)
        return [self._combination_from_record(record, path) for record in coarse_records]

    def _evaluate_stage_combinations(
        self,
        stage_name: str,
        stage_sim_cfg: Dict[str, Any],
        stage_combinations: Sequence[Dict[str, Any]],
        subjects: Sequence[int],
    ) -> List[CombinationResult]:
        stage_cfg = (self.config.get("stages") or {}).get(stage_name) or {}
        n_jobs_combinations = max(1, int(stage_cfg.get("n_jobs_combinations", 1)))
        if n_jobs_combinations == 1:
            return [
                self._evaluate_combination(stage_name, idx, params, stage_sim_cfg, subjects)
                for idx, params in enumerate(stage_combinations)
            ]
        return list(
            Parallel(n_jobs=n_jobs_combinations)(
                delayed(self._evaluate_combination)(stage_name, idx, params, stage_sim_cfg, subjects)
                for idx, params in enumerate(stage_combinations)
            )
        )

    def _run_subject_pipeline(
        self,
        subject_id: int,
        stage: str,
        output_base: Path,
        resume_from_coarse: bool = False,
    ) -> Dict[str, Any]:
        if resume_from_coarse and stage != "fine":
            raise ValueError("resume_from_coarse requires stage='fine'")

        stages_to_run = ["coarse", "fine"] if stage == "all" else [stage]
        subject_dir = output_base / f"subject_{int(subject_id)}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        all_combinations_path = subject_dir / "all_combinations.jsonl"
        if resume_from_coarse:
            all_stage_combinations: Dict[str, List[CombinationResult]] = {
                "coarse": self._load_coarse_for_fine_resume(all_combinations_path)
            }
        elif all_combinations_path.exists():
            all_combinations_path.unlink()
            all_stage_combinations = {}
        else:
            all_stage_combinations = {}

        for stage_name in stages_to_run:
            stage_sim_cfg = self._prepare_stage_config(stage_name)
            if stage_name == "fine":
                fine_stage_cfg = (self.config.get("stages") or {}).get("fine") or {}
                if "hyperparam_space" in fine_stage_cfg:
                    specs = self._param_specs_for_stage(stage_name)
                    stage_combinations = self._expand_combinations(specs)
                else:
                    prior = all_stage_combinations.get("coarse")
                    if prior is None:
                        raise ValueError("fine stage without hyperparam_space requires coarse stage results")
                    stage_combinations = self._refine_combinations_from_coarse(prior)
            else:
                specs = self._param_specs_for_stage(stage_name)
                stage_combinations = self._expand_combinations(specs)

            combination_results = self._evaluate_stage_combinations(
                stage_name,
                stage_sim_cfg,
                stage_combinations,
                [int(subject_id)],
            )
            for result in combination_results:
                self._append_jsonl(all_combinations_path, self._serialize_combination_record(result))
            all_stage_combinations[stage_name] = combination_results

        stage_summary = self._build_stage_summary(all_stage_combinations)
        final_stage = "fine" if "fine" in all_stage_combinations else "coarse"
        final_combinations = all_stage_combinations[final_stage]
        best_combination = min(final_combinations, key=lambda t: float(t.subject_metrics[int(subject_id)]["mean_error"]))

        stage_summary_path = subject_dir / "stage_summary.json"
        with stage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2)

        sid = int(subject_id)
        subject_best = {
            "best_stage": final_stage,
            "best_combination_index": best_combination.combination_index,
            "best_hyperparams": best_combination.hyperparams,
            "best_params": _compact_hyperparams(best_combination.hyperparams),
            "mean_error": float(best_combination.subject_metrics[sid]["mean_error"]),
            "best_error": float(best_combination.subject_metrics[sid]["best_error"]),
            "std_error": float(best_combination.subject_metrics[sid]["std_error"]),
            "hyper_candidate_seed": best_combination.hyper_candidate_seed,
            "simulation_repeats": int(best_combination.subject_metrics[sid]["simulation_repeats"]),
        }
        if self.save_level == "full":
            subject_best["subject_metrics"] = {str(sid): best_combination.subject_metrics[sid]}

        best_path = subject_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(subject_best), f, ensure_ascii=False, indent=2)

        return {
            "subject_id": sid,
            "output_dir": str(subject_dir),
            "all_combinations": str(all_combinations_path),
            "stage_summary": str(stage_summary_path),
            "best_hyperparams": str(best_path),
            "best": subject_best,
        }

    def _build_stage_summary(self, all_stage_combinations: Mapping[str, Sequence[CombinationResult]]) -> Dict[str, Any]:
        stage_summary: Dict[str, Any] = {}
        top_k = int((self.config.get("refine_policy") or {}).get("top_k", 3))
        for stage_name, combinations in all_stage_combinations.items():
            ranked = sorted(combinations, key=lambda x: x.aggregated_error)
            stage_summary[stage_name] = {
                "num_combinations": len(combinations),
                "top_combinations": [
                    {
                        "combination_index": t.combination_index,
                        "aggregated_error": t.aggregated_error,
                        "hyperparams": t.hyperparams,
                        "hyper_candidate_seed": t.hyper_candidate_seed,
                    }
                    for t in ranked[:max(1, top_k)]
                ],
            }
        return stage_summary

    def run(self, subjects: Sequence[int], stage: str = "all", resume_from_coarse: bool = False) -> Dict[str, Any]:
        if stage not in {"coarse", "fine", "all"}:
            raise ValueError("stage must be one of: coarse, fine, all")
        if resume_from_coarse and stage != "fine":
            raise ValueError("resume_from_coarse requires stage='fine'")

        best_payload: Dict[str, Any] = {
            "selection_metric": self.selection_metric,
            "hyperparam_selection_mode": self.hyperparam_selection_mode,
            "save_level": self.save_level,
            "base_sim_config_path": str(self.base_sim_config_path),
            "hyper_grid_config_path": str(self.config_path),
            "hyper_base_seed": self.hyper_base_seed,
            "hyper_backend": "hyper_grid",
        }

        if self.hyperparam_selection_mode == "group_mean":
            return self._run_group_pipeline(subjects, stage, resume_from_coarse, best_payload)

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

    def _run_group_pipeline(
        self,
        subjects: Sequence[int],
        stage: str,
        resume_from_coarse: bool,
        best_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        stages_to_run = ["coarse", "fine"] if stage == "all" else [stage]
        all_combinations_path = self.output_dir / "all_combinations.jsonl"
        if resume_from_coarse:
            all_stage_combinations: Dict[str, List[CombinationResult]] = {
                "coarse": self._load_coarse_for_fine_resume(all_combinations_path)
            }
        elif all_combinations_path.exists():
            all_combinations_path.unlink()
            all_stage_combinations = {}
        else:
            all_stage_combinations = {}

        for stage_name in stages_to_run:
            stage_sim_cfg = self._prepare_stage_config(stage_name)
            if stage_name == "fine":
                fine_stage_cfg = (self.config.get("stages") or {}).get("fine") or {}
                if "hyperparam_space" in fine_stage_cfg:
                    specs = self._param_specs_for_stage(stage_name)
                    stage_combinations = self._expand_combinations(specs)
                else:
                    prior = all_stage_combinations.get("coarse")
                    if prior is None:
                        raise ValueError("fine stage without hyperparam_space requires coarse stage results")
                    stage_combinations = self._refine_combinations_from_coarse(prior)
            else:
                specs = self._param_specs_for_stage(stage_name)
                stage_combinations = self._expand_combinations(specs)

            combination_results = self._evaluate_stage_combinations(
                stage_name,
                stage_sim_cfg,
                stage_combinations,
                subjects,
            )
            for result in combination_results:
                self._append_jsonl(all_combinations_path, self._serialize_combination_record(result))
            all_stage_combinations[stage_name] = combination_results

        stage_summary = self._build_stage_summary(all_stage_combinations)
        final_stage = "fine" if "fine" in all_stage_combinations else "coarse"
        final_combinations = all_stage_combinations[final_stage]
        best_combination = min(final_combinations, key=lambda x: x.aggregated_error)

        stage_summary_path = self.output_dir / "stage_summary.json"
        with stage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2)

        best_payload.update(
            {
                "best_stage": final_stage,
                "best_combination_index": best_combination.combination_index,
                "best_hyperparams": best_combination.hyperparams,
                "best_params": _compact_hyperparams(best_combination.hyperparams),
                "aggregated_error": best_combination.aggregated_error,
                "hyper_candidate_seed": best_combination.hyper_candidate_seed,
            }
        )
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


def _compact_hyperparams(hyperparams: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(hyperparams)
    shortcuts = {
        "engine.modules.memory_mod.kwargs.gamma": "gamma",
        "engine.modules.memory_mod.kwargs.w0": "w0",
        "engine.modules.beta_mod.kwargs.beta_init": "beta_init",
        "engine.modules.beta_mod.kwargs.decrease_rate": "decrease_rate",
        "engine.modules.beta_mod.kwargs.prior_beta_scale": "prior_beta_scale",
        "engine.modules.hypo_transitions_mod.kwargs.init_num": "init_num",
        "engine.modules.hypo_transitions_mod.kwargs.max_active_hypotheses": "max_active_hypotheses",
        "simulation.window_size": "window_size",
    }
    for source, target in shortcuts.items():
        if source in hyperparams:
            summary[target] = hyperparams[source]
    return summary


__all__ = ["HyperGridOptimizer", "CombinationResult"]
