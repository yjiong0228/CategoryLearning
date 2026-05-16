"""Coordinate-descent two-layer hyper optimizer for Bayesian_state."""
from __future__ import annotations

import hashlib
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
    resolve_param_grid as resolve_param_grid_amr,
    resolve_prediction_modes as resolve_prediction_modes_amr,
    resolve_window_size as resolve_window_size_amr,
)
from src.Bayesian_state.run_grid_optimization import (
    resolve_engine_config as resolve_engine_config_grid,
    resolve_param_grid as resolve_param_grid_grid,
    resolve_prediction_modes as resolve_prediction_modes_grid,
    resolve_window_size as resolve_window_size_grid,
)
from src.Bayesian_state.utils.config_subjects import resolve_subject_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.optimizer_amr import StateModelAMROptimizer
from src.Bayesian_state.utils.optimizer_grid import StateModelGridOptimizer


@dataclass
class TrialResult:
    stage: str
    trial_index: int
    hyperparams: Dict[str, Any]
    aggregated_error: float
    subject_metrics: Dict[int, Dict[str, Any]]
    random_seed: int
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

        self.base_seed = int(self.config.get("random_seed", 1234))
        self.rng = random.Random(self.base_seed)

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

        self._trial_counter = 0

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
        if all(k in spec for k in ("start", "stop", "num")):
            return self._linspace_values(spec)
        raise ValueError("Each hyperparameter spec must provide either values or (start, stop, num)")

    def _param_specs_for_stage(self, stage_name: str) -> Dict[str, Dict[str, Any]]:
        stages = self.config.get("stages") or {}
        stage_cfg = stages.get(stage_name)
        if not isinstance(stage_cfg, Mapping):
            raise ValueError(f"Missing stage config: stages.{stage_name}")
        if "hyperparam_space" in stage_cfg:
            raw = stage_cfg["hyperparam_space"]
            if not isinstance(raw, Mapping):
                raise ValueError(f"stages.{stage_name}.hyperparam_space must be a mapping")
            return {k: dict(v) for k, v in raw.items()}
        raw = self.config.get("hyperparam_space")
        if not isinstance(raw, Mapping):
            raise ValueError("hyperparam_space must be a mapping")
        return {k: dict(v) for k, v in raw.items()}

    def _top_k_trials_from_coarse(self, coarse_trials: Sequence[TrialResult]) -> List[Dict[str, Any]]:
        policy = self.config.get("refine_policy") or {}
        top_k = max(1, int(policy.get("top_k", 3)))
        ranked = sorted(coarse_trials, key=lambda x: x.aggregated_error)
        selected: List[Dict[str, Any]] = []
        seen = set()
        for trial in ranked:
            key = json.dumps(_to_builtin(trial.hyperparams), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            selected.append(deepcopy(trial.hyperparams))
            if len(selected) >= top_k:
                break
        return selected

    def _space_from_trials(self, trials: Sequence[Dict[str, Any]], fallback_specs: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {}
        for name in fallback_specs.keys():
            vals = []
            for t in trials:
                if name in t:
                    vals.append(t[name])
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
        curr[parts[-1]] = value

    def _apply_hyperparams(self, trial: Dict[str, Any], inner_cfg: Dict[str, Any], engine_cfg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        next_inner = deepcopy(inner_cfg)
        next_engine = deepcopy(engine_cfg)
        for key, val in trial.items():
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

    def _trial_seed(self, stage_name: str, trial_index: int, trial_params: Mapping[str, Any]) -> int:
        payload = json.dumps(
            {"stage": stage_name, "idx": trial_index, "params": trial_params, "base_seed": self.base_seed},
            sort_keys=True,
        )
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
            window_size = resolve_window_size_grid(subject_cfg, subject_id, subjects)
            n_jobs = int(subject_cfg.get("n_jobs", 1))
        else:
            engine_cfg = resolve_engine_config_amr(subject_cfg, cfg_path.parent, subject_id=subject_id)
            prediction_mode, selection_prediction_mode = resolve_prediction_modes_amr(subject_cfg)
            window_size = resolve_window_size_amr(subject_cfg, subject_id, subjects)
            n_jobs = int(subject_cfg.get("n_jobs_inner", 1))
        return subject_cfg, engine_cfg, prediction_mode, selection_prediction_mode, window_size, n_jobs

    def _evaluate_point(
        self,
        stage_name: str,
        point: Dict[str, Any],
        stage_inner_cfg: Dict[str, Any],
        subjects: Sequence[int],
        restart_id: int,
        iter_id: int,
        coordinate: str,
    ) -> TrialResult:
        trial_index = self._trial_counter
        self._trial_counter += 1
        seed = self._trial_seed(stage_name, trial_index, point)
        np.random.seed(seed)

        subject_metrics: Dict[int, Dict[str, Any]] = {}
        errors: List[float] = []
        for sid in subjects:
            subject_cfg, base_engine_cfg, pred_mode, sel_mode, window_size, n_jobs = self._resolve_inner_components(
                stage_inner_cfg, sid, subjects, self.inner_base_config_path
            )
            trial_inner_cfg, trial_engine_cfg = self._apply_hyperparams(point, subject_cfg, base_engine_cfg)
            if self.inner_optimizer == "grid":
                param_grid = resolve_param_grid_grid(trial_inner_cfg)
            else:
                param_grid = resolve_param_grid_amr(trial_inner_cfg)

            mod = trial_engine_cfg.get("modules", {}).get("hypo_transitions_mod", {}).get("kwargs", {})
            if isinstance(mod, dict) and "random_seed" not in mod:
                mod["random_seed"] = int(seed)

            optimizer, dataset_paths = self._build_optimizer(trial_inner_cfg, trial_engine_cfg, self.inner_base_config_path)
            optimizer.n_jobs = n_jobs
            result = optimizer.optimize_subject(
                subject_id=sid,
                param_grid=param_grid,
                n_repeats=int(trial_inner_cfg.get("n_repeats", 1)),
                refit_repeats=int(trial_inner_cfg.get("refit_repeats", 0)),
                window_size=int(trial_inner_cfg.get("window_size", window_size)),
                stop_at=float(trial_inner_cfg.get("stop_at", 1.0)),
                max_trials=trial_inner_cfg.get("max_trials"),
                keep_logs=bool(trial_inner_cfg.get("keep_logs", False)),
                prediction_mode=str(trial_inner_cfg.get("prediction_mode", pred_mode)),
                selection_prediction_mode=str(trial_inner_cfg.get("selection_prediction_mode", sel_mode)),
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
        return TrialResult(
            stage=stage_name,
            trial_index=trial_index,
            hyperparams=deepcopy(point),
            aggregated_error=agg_error,
            subject_metrics=subject_metrics,
            random_seed=seed,
            restart_id=restart_id,
            iter_id=iter_id,
            coordinate=coordinate,
        )

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_builtin(payload), ensure_ascii=False) + "\n")

    def _serialize_trial_record(self, tr: TrialResult) -> Dict[str, Any]:
        data = {
            "stage": tr.stage,
            "trial_index": tr.trial_index,
            "restart_id": tr.restart_id,
            "iter_id": tr.iter_id,
            "coordinate": tr.coordinate,
            "hyperparams": tr.hyperparams,
            "aggregated_error": tr.aggregated_error,
            "random_seed": tr.random_seed,
        }
        if self.save_level == "full":
            data["subject_metrics"] = tr.subject_metrics
        return data

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
        all_trials_path: Path,
    ) -> tuple[List[TrialResult], List[Dict[str, Any]], TrialResult]:
        all_trials: List[TrialResult] = []
        restart_best: List[Dict[str, Any]] = []
        global_best: TrialResult | None = None
        cache: Dict[str, TrialResult] = {}
        coords_base = list(space.keys())

        def eval_with_cache(
            point: Dict[str, Any],
            restart_id: int,
            iter_id: int,
            coordinate: str,
        ) -> tuple[TrialResult, bool]:
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
            self._append_jsonl(all_trials_path, self._serialize_trial_record(tr))
            return tr, True

        for restart_id in range(self.n_restarts):
            current = self._init_point(space)
            current_tr, current_is_new = eval_with_cache(current, restart_id, 0, "init")
            if current_is_new:
                all_trials.append(current_tr)
            best_local = current_tr
            no_improve_rounds = 0

            for iter_id in range(1, self.max_outer_iters + 1):
                coords = list(coords_base)
                if self.coordinate_order == "shuffle_each_iter":
                    self.rng.shuffle(coords)
                improved_this_round = False

                for coord in coords:
                    candidate_best = best_local
                    base_point = deepcopy(current)
                    for val in space[coord]:
                        cand = deepcopy(base_point)
                        cand[coord] = val
                        cand_tr, cand_is_new = eval_with_cache(cand, restart_id, iter_id, coord)
                        if cand_is_new:
                            all_trials.append(cand_tr)
                        if cand_tr.aggregated_error + self.min_delta < candidate_best.aggregated_error:
                            candidate_best = cand_tr
                    if candidate_best.aggregated_error + self.min_delta < best_local.aggregated_error:
                        current = deepcopy(candidate_best.hyperparams)
                        best_local = candidate_best
                        improved_this_round = True

                if improved_this_round:
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.patience:
                        break

            restart_best.append(
                {
                    "restart_id": restart_id,
                    "best_trial_index": best_local.trial_index,
                    "best_error": best_local.aggregated_error,
                    "best_hyperparams": best_local.hyperparams,
                }
            )
            if global_best is None or best_local.aggregated_error < global_best.aggregated_error:
                global_best = best_local

        if global_best is None:
            raise RuntimeError("CD optimizer produced no trial")
        return all_trials, restart_best, global_best

    def _run_pipeline(self, subjects: Sequence[int], stage: str, output_dir: Path) -> Dict[str, Any]:
        stages_to_run = ["coarse", "fine"] if stage == "all" else [stage]
        all_trials_path = output_dir / "all_trials.jsonl"
        if all_trials_path.exists():
            all_trials_path.unlink()

        stage_trials: Dict[str, List[TrialResult]] = {}
        stage_restarts: Dict[str, Any] = {}
        for stage_name in stages_to_run:
            stage_inner_cfg = self._prepare_stage_config(stage_name)
            if stage_name == "fine":
                fine_stage_cfg = (self.config.get("stages") or {}).get("fine") or {}
                if "hyperparam_space" in fine_stage_cfg:
                    specs = self._param_specs_for_stage(stage_name)
                    space = {k: self._hyperparam_values(v) for k, v in specs.items()}
                else:
                    prior = stage_trials.get("coarse")
                    if prior is None:
                        raise ValueError(
                            "fine stage without stages.fine.hyperparam_space requires coarse stage results in this run"
                        )
                    coarse_top = self._top_k_trials_from_coarse(prior)
                    coarse_specs = self._param_specs_for_stage("coarse")
                    space = self._space_from_trials(coarse_top, coarse_specs)
            else:
                specs = self._param_specs_for_stage(stage_name)
                space = {k: self._hyperparam_values(v) for k, v in specs.items()}

            trials, restarts, _ = self._coordinate_descent(
                stage_name=stage_name,
                stage_inner_cfg=stage_inner_cfg,
                subjects=subjects,
                space=space,
                all_trials_path=all_trials_path,
            )
            stage_trials[stage_name] = trials
            stage_restarts[stage_name] = restarts

        stage_summary = {}
        top_k = int((self.config.get("refine_policy") or {}).get("top_k", 3))
        for stage_name, trials in stage_trials.items():
            ranked = sorted(trials, key=lambda x: x.aggregated_error)
            stage_summary[stage_name] = {
                "num_trials": len(trials),
                "top_trials": [
                    {
                        "trial_index": t.trial_index,
                        "aggregated_error": t.aggregated_error,
                        "hyperparams": t.hyperparams,
                        "restart_id": t.restart_id,
                        "iter_id": t.iter_id,
                        "coordinate": t.coordinate,
                        "random_seed": t.random_seed,
                    }
                    for t in ranked[:max(1, top_k)]
                ],
            }

        final_stage = "fine" if "fine" in stage_trials else "coarse"
        final_trials = stage_trials[final_stage]
        best_trial = min(final_trials, key=lambda x: x.aggregated_error)

        stage_summary_path = output_dir / "stage_summary.json"
        with stage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2)

        restart_summary_path = output_dir / "restart_summary.json"
        with restart_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_restarts), f, ensure_ascii=False, indent=2)

        best_payload: Dict[str, Any] = {
            "best_stage": final_stage,
            "best_trial_index": best_trial.trial_index,
            "best_hyperparams": best_trial.hyperparams,
            "aggregated_error": best_trial.aggregated_error,
            "random_seed": best_trial.random_seed,
        }
        if self.save_level == "full":
            best_payload["subject_metrics"] = best_trial.subject_metrics

        best_path = output_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(best_payload), f, ensure_ascii=False, indent=2)

        return {
            "output_dir": str(output_dir),
            "all_trials": str(all_trials_path),
            "stage_summary": str(stage_summary_path),
            "restart_summary": str(restart_summary_path),
            "best_hyperparams": str(best_path),
            "best": best_payload,
        }

    def run(self, subjects: Sequence[int], stage: str = "all") -> Dict[str, Any]:
        if stage not in {"coarse", "fine", "all"}:
            raise ValueError("stage must be one of: coarse, fine, all")

        if self.hyperparam_selection_mode == "group_mean":
            return self._run_pipeline(subjects=subjects, stage=stage, output_dir=self.output_dir)

        per_subject_best: Dict[str, Any] = {}
        per_subject_outputs: Dict[str, Any] = {}
        for sid in subjects:
            subject_dir = self.output_dir / f"subject_{int(sid)}"
            subject_dir.mkdir(parents=True, exist_ok=True)
            out = self._run_pipeline(subjects=[int(sid)], stage=stage, output_dir=subject_dir)
            per_subject_outputs[str(int(sid))] = {
                "output_dir": out["output_dir"],
                "all_trials": out["all_trials"],
                "stage_summary": out["stage_summary"],
                "restart_summary": out["restart_summary"],
                "best_hyperparams": out["best_hyperparams"],
            }
            per_subject_best[str(int(sid))] = out["best"]

        best_payload = {
            "selection_metric": self.selection_metric,
            "hyperparam_selection_mode": self.hyperparam_selection_mode,
            "save_level": self.save_level,
            "inner_base_config_path": str(self.inner_base_config_path),
            "hyper_config_path": str(self.config_path),
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


__all__ = ["HyperOptimizerCD"]
