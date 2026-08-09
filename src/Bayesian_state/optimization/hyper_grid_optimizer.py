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

from src.Bayesian_state.simulation.simulation_config import (
    resolve_loss_delta,
    resolve_simulation_repeats,
)
from src.Bayesian_state.utils.seeding import derive_hyper_candidate_seed
from src.Bayesian_state.optimization.hyper_utils import (
    HYPER_RESULT_SCHEMA_VERSION,
    build_root_best_payload,
    build_hyper_provenance,
    build_subject_artifacts,
    build_subject_best_payload,
    combination_metrics_summary,
    compact_hyperparams,
    to_builtin as _to_builtin,
)
from src.Bayesian_state.optimization.hyper_search_common import HyperSearchRuntime
from src.Bayesian_state.simulation.repeated_simulation import (
    get_stat_value,
    resolve_selection_metric_path,
    resolve_simulation_stat_config,
)


LOWER_TAIL_FRACTION = 0.10
DEFAULT_TIE_BREAK_METRIC = "simulation.mean_error"


def _lower_tail_metrics(sample_errors: Sequence[Any], fallback_error: float) -> Dict[str, Any]:
    finite_values: List[float] = []
    for value in sample_errors or []:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            finite_values.append(numeric)
    values = np.asarray(finite_values, dtype=float)
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


def _metric_value(
    metric_path: str,
    subject_record: Mapping[str, Any],
) -> float:
    value = get_stat_value(
        subject_record,
        resolve_selection_metric_path(metric_path),
        float("inf"),
    )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return numeric if np.isfinite(numeric) else float("inf")


@dataclass
class CombinationResult:
    stage: str
    combination_index: int
    hyperparams: Dict[str, Any]
    aggregated_error: float
    subject_metrics: Dict[int, Dict[str, Any]]
    hyper_candidate_seed: int


class HyperGridOptimizer(HyperSearchRuntime):
    """Choose all model hyperparameters jointly via explicit grid candidates."""

    backend_label = "Hyper-grid"
    default_output_dir = "../../results/state-based-hyper-grid/default"

    def __init__(self, config: Mapping[str, Any], config_path: Path) -> None:
        super().__init__(config, config_path)

        self.tie_break_metric = resolve_selection_metric_path(
            self.config.get("tie_break_metric", DEFAULT_TIE_BREAK_METRIC)
        )

        requested_selection_mode = str(self.config.get("hyperparam_selection_mode", "per_subject"))
        if requested_selection_mode != "per_subject":
            raise ValueError("Only per_subject hyperparam_selection_mode is supported.")
        self.hyperparam_selection_mode = "per_subject"

        self.acceptance_selection = resolve_simulation_stat_config(
            self.config.get("acceptance_selection"),
            setting_name="acceptance_selection",
        )

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
            statistics_config = combination_sim_cfg.get("simulation_statistics")
            if statistics_config is None:
                statistics_config = self.config.get("simulation_statistics")
            if statistics_config is None:
                statistics_config = self.config.get("acceptance_selection")

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
                statistics_config=statistics_config,
                evaluation_protocol=combination_sim_cfg.get("evaluation_protocol"),
                evaluation_role="optimization",
            )

            best = result["best"]
            mean_err = float(getattr(best, "mean_error"))
            best_err = float(getattr(best, "best_error", mean_err))
            sample_errors = list(getattr(best, "sample_errors", []) or [])
            tail_metrics = _lower_tail_metrics(sample_errors, fallback_error=mean_err)
            statistics_summary = dict(getattr(best, "statistics_summary", {}) or {})
            simulation_summary = {
                "mean_error": mean_err,
                "best_error": best_err,
                "best10_mean_error": float(tail_metrics["best10_mean_error"]),
                "q10_error": float(tail_metrics["q10_error"]),
                "lower_tail_fraction": float(tail_metrics["lower_tail_fraction"]),
                "lower_tail_count": int(tail_metrics["lower_tail_count"]),
                "std_error": float(getattr(best, "std_error", 0.0)),
                "sample_errors": sample_errors,
                "simulation_repeats": simulation_repeats,
            }
            subject_record = {
                "simulation": simulation_summary,
                "statistics": statistics_summary,
            }
            value = _metric_value(
                self.tie_break_metric,
                subject_record,
            )
            errors.append(value)
            subject_metrics[int(sid)] = {
                "simulation": simulation_summary,
                "statistics": statistics_summary,
                "selection": {
                    "primary": {
                        "metric": self.tie_break_metric,
                        "value": float(value),
                    },
                    "tie_break": {
                        "metric": self.tie_break_metric,
                        "value": float(value),
                    },
                },
                "fixed_hyperparams": deepcopy(combination_params),
                "condition": int(result.get("condition", -1)),
                "dataset_paths": {k: str(v) for k, v in dataset_paths.items()},
                "hyper_candidate_seed": int(hyper_candidate_seed),
                "scoring": deepcopy(result.get("selection_meta", {}).get("score_context", {})),
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

    def _serialize_combination_record(self, combination: CombinationResult) -> Dict[str, Any]:
        data = {
            "schema_version": HYPER_RESULT_SCHEMA_VERSION,
            "stage": combination.stage,
            "combination_index": combination.combination_index,
            "hyperparams": combination.hyperparams,
            "aggregated_error": combination.aggregated_error,
            "hyper_candidate_seed": combination.hyper_candidate_seed,
        }
        metrics_summary = combination_metrics_summary(
            combination.subject_metrics,
            aggregated_error=combination.aggregated_error,
        )
        if metrics_summary:
            data["metrics_summary"] = metrics_summary
        if self.save_level == "full":
            data["subject_metrics"] = combination.subject_metrics
        return data

    def _serialize_accepted_record(
        self,
        combination: CombinationResult,
        *,
        mode: str,
    ) -> Dict[str, Any]:
        tie_break_value = self._combination_tie_break_value(combination)
        score = self._combination_acceptance_score(combination)
        data = {
            "schema_version": HYPER_RESULT_SCHEMA_VERSION,
            "result_type": "hyper_grid_accepted_candidate",
            "stage": combination.stage,
            "combination_index": combination.combination_index,
            "hyperparams": combination.hyperparams,
            "best_params": compact_hyperparams(combination.hyperparams),
            "aggregated_error": combination.aggregated_error,
            "hyper_candidate_seed": combination.hyper_candidate_seed,
            "selection": {
                "tie_break": {
                    "metric": self.tie_break_metric,
                    "value": tie_break_value,
                },
                "acceptance": {
                    "mode": mode,
                    "score": score,
                    "alpha": self.acceptance_selection.get("distribution_interval_alpha"),
                },
            },
        }
        metrics_summary = combination_metrics_summary(
            combination.subject_metrics,
            aggregated_error=combination.aggregated_error,
        )
        if metrics_summary:
            data["metrics_summary"] = metrics_summary
        if self.save_level == "full":
            data["subject_metrics"] = combination.subject_metrics
        return data

    def _write_accepted_records(
        self,
        path: Path,
        combinations: Sequence[CombinationResult],
        *,
        mode: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for combination in combinations:
                record = self._serialize_accepted_record(combination, mode=mode)
                f.write(json.dumps(_to_builtin(record), ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _finite_float(value: Any, default: float = float("inf")) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if np.isfinite(out) else default

    @staticmethod
    def _combination_metric_mean(result: CombinationResult, path: str) -> float:
        values: List[float] = []
        for metrics in (result.subject_metrics or {}).values():
            if not isinstance(metrics, Mapping):
                continue
            value = HyperGridOptimizer._finite_float(get_stat_value(metrics, path, float("inf")))
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else float("inf")

    def _combination_tie_break_value(self, result: CombinationResult) -> float:
        value = self._combination_metric_mean(result, "selection.tie_break.value")
        if not np.isfinite(value):
            value = self._combination_metric_mean(result, "selection.primary.value")
        return value if np.isfinite(value) else float(result.aggregated_error)

    def _combination_acceptance_score(self, result: CombinationResult) -> float:
        mode = self.acceptance_selection.get("mode")
        if mode == "distribution_ppc_interval":
            return self._combination_metric_mean(result, "statistics.scores.distribution.ppc_interval.score")
        if mode == "distribution_intersection":
            return self._combination_metric_mean(result, "statistics.scores.distribution.intersection.score")
        if mode == "accuracy_shape":
            return self._combination_metric_mean(result, "statistics.scores.accuracy_shape.value")
        if mode == "history_kernel":
            return self._combination_metric_mean(result, "statistics.scores.history_kernel.value")
        if mode == "switch_behavior":
            return self._combination_metric_mean(result, "statistics.scores.switch_behavior.value")
        if mode == "distribution_multiobjective":
            return self._combination_metric_mean(result, "statistics.scores.distribution.multiobjective.score")
        return float("inf")

    @staticmethod
    def _combination_all_subjects_accept(result: CombinationResult, path: str) -> bool:
        if not result.subject_metrics:
            return False
        for metrics in result.subject_metrics.values():
            if not isinstance(metrics, Mapping):
                return False
            if not bool(get_stat_value(metrics, path, False)):
                return False
        return True

    def _combination_accepts(self, result: CombinationResult) -> bool:
        mode = self.acceptance_selection.get("mode")
        if mode == "distribution_ppc_interval":
            return self._combination_all_subjects_accept(
                result,
                "statistics.scores.distribution.ppc_interval.accept",
            )
        if mode == "distribution_intersection":
            return self._combination_all_subjects_accept(
                result,
                "statistics.scores.distribution.intersection.accept",
            )
        return False

    def _select_final_combination(
        self,
        combinations: Sequence[CombinationResult],
    ) -> tuple[CombinationResult, Dict[str, Any], List[CombinationResult]]:
        if not combinations:
            raise RuntimeError("No combinations available for final selection")
        tie_break_best = min(
            combinations,
            key=lambda result: (
                self._combination_tie_break_value(result),
                int(result.combination_index),
            ),
        )
        if not self.acceptance_selection.get("enabled", False):
            return tie_break_best, {
                "enabled": False,
                "selected_by": "tie_break_metric",
                "tie_break_metric": self.tie_break_metric,
                "tie_break_best_combination_index": tie_break_best.combination_index,
                "final_metric": self.tie_break_metric,
                "final_value": self._combination_tie_break_value(tie_break_best),
            }, []

        mode = str(self.acceptance_selection["mode"])
        accepted = [
            result for result in combinations
            if self._combination_accepts(result)
        ]
        if accepted:
            selected = min(
                accepted,
                key=lambda result: (
                    self._combination_tie_break_value(result),
                    self._combination_acceptance_score(result),
                    int(result.combination_index),
                ),
            )
            selected_by = f"{mode}_accepted_set_tiebreak_metric"
            final_metric = self.tie_break_metric
            final_value = self._combination_tie_break_value(selected)
            selected_from_accepted = True
        else:
            with_acceptance_score = [
                result for result in combinations
                if self._combination_acceptance_score(result) < float("inf")
            ]
            pool = with_acceptance_score or list(combinations)
            selected = min(
                pool,
                key=lambda result: (
                    self._combination_acceptance_score(result),
                    self._combination_tie_break_value(result),
                    int(result.combination_index),
                ),
            )
            selected_by = f"fallback_min_{mode}_violation"
            acceptance_score = self._combination_acceptance_score(selected)
            final_metric = self._acceptance_metric_path(mode)
            final_value = acceptance_score if np.isfinite(acceptance_score) else self._combination_tie_break_value(selected)
            selected_from_accepted = False

        context = {
            "enabled": True,
            "mode": mode,
            "selected_by": selected_by,
            "tie_break_metric": self.tie_break_metric,
            "tie_break_best_combination_index": tie_break_best.combination_index,
            "tie_break_best_value": self._combination_tie_break_value(tie_break_best),
            "accepted_count": int(len(accepted)),
            "accepted_combination_indices": [int(result.combination_index) for result in accepted],
            "selected_from_accepted_set": selected_from_accepted,
            "selected_combination_index": int(selected.combination_index),
            "selected_tie_break_value": self._combination_tie_break_value(selected),
            "selected_acceptance_score": self._combination_acceptance_score(selected),
            "final_metric": final_metric,
            "final_value": final_value,
            "config": dict(self.acceptance_selection),
        }
        return selected, context, accepted

    @staticmethod
    def _acceptance_metric_path(mode: str) -> str:
        by_mode = {
            "distribution_ppc_interval": "statistics.scores.distribution.ppc_interval.score",
            "distribution_intersection": "statistics.scores.distribution.intersection.score",
            "accuracy_shape": "statistics.scores.accuracy_shape.value",
            "history_kernel": "statistics.scores.history_kernel.value",
            "switch_behavior": "statistics.scores.switch_behavior.value",
            "distribution_multiobjective": "statistics.scores.distribution.multiobjective.score",
        }
        return by_mode.get(mode, "selection.primary.value")

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

        if stage == "all":
            configured_stages = self.config.get("stages") or {}
            stages_to_run = [name for name in ("coarse", "fine") if name in configured_stages]
            if not stages_to_run:
                raise ValueError("Config must define at least one stage when stage='all'")
        else:
            stages_to_run = [stage]
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
        best_combination, final_selection_context, accepted_combinations = self._select_final_combination(
            final_combinations
        )

        stage_summary_path = subject_dir / "stage_summary.json"
        with stage_summary_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(stage_summary), f, ensure_ascii=False, indent=2, allow_nan=False)

        accepted_path = subject_dir / "accepted_hyperparams.jsonl"
        self._write_accepted_records(
            accepted_path,
            accepted_combinations,
            mode=str(final_selection_context.get("mode", "disabled")),
        )

        sid = int(subject_id)
        subject_best = build_subject_best_payload(
            subject_id=sid,
            backend="hyper_grid",
            hyper_base_seed=self.hyper_base_seed,
            selection_metric=self.tie_break_metric,
            best_stage=final_stage,
            best_combination_index=best_combination.combination_index,
            best_hyperparams=best_combination.hyperparams,
            aggregated_error=best_combination.aggregated_error,
            hyper_candidate_seed=best_combination.hyper_candidate_seed,
            metrics=best_combination.subject_metrics[sid],
            search_context={
                "final_selection": final_selection_context,
            },
            provenance=build_hyper_provenance(
                config_path=self.config_path,
                output_dir=subject_dir,
                base_sim_config_path=self.base_sim_config_path,
            ),
            artifacts=build_subject_artifacts(subject_dir, include_cd=False, include_accepted=True),
            full_subject_metrics=(
                {str(sid): best_combination.subject_metrics[sid]}
                if self.save_level == "full"
                else None
            ),
        )

        best_path = subject_dir / "best_hyperparams.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(subject_best), f, ensure_ascii=False, indent=2, allow_nan=False)

        return {
            "subject_id": sid,
            "output_dir": str(subject_dir),
            "all_combinations": str(all_combinations_path),
            "stage_summary": str(stage_summary_path),
            "accepted_hyperparams": str(accepted_path),
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
                        "best_params": compact_hyperparams(t.hyperparams),
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

        per_subject_best: Dict[str, Any] = {}
        per_subject_outputs: Dict[str, Any] = {}
        for sid in subjects:
            out = self.run_subject(
                int(sid),
                stage,
                resume_from_coarse=resume_from_coarse,
            )
            per_subject_outputs[str(int(sid))] = {
                "output_dir": out["output_dir"],
                "all_combinations": out["all_combinations"],
                "stage_summary": out["stage_summary"],
                "accepted_hyperparams": out["accepted_hyperparams"],
                "best_hyperparams": out["best_hyperparams"],
            }
            per_subject_best[str(int(sid))] = out["best"]

        best_payload = build_root_best_payload(
            backend="hyper_grid",
            config_path=self.config_path,
            output_dir=self.output_dir,
            base_sim_config_path=self.base_sim_config_path,
            hyper_base_seed=self.hyper_base_seed,
            selection_metric=self.tie_break_metric,
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


__all__ = ["HyperGridOptimizer", "CombinationResult"]
