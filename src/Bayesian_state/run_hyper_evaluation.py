"""Run the standard evaluation suite for hyper-CD search outputs.

Examples:
    python -m src.Bayesian_state.run_hyper_evaluation \
        --input-dir results/state-based-hyper-cd/pmh/cond1_v9

    # Include expensive resampling diagnostics as well.
    python -m src.Bayesian_state.run_hyper_evaluation \
        --input-dir results/state-based-hyper-cd/pmh/cond1_v9 \
        --all
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.Bayesian_state.optimization.hyper_evaluation import (
    DEFAULT_BASE_SIM_CONFIG,
    diagnose_hyper_accuracy_sampling,
    evaluate_hyper_cd_convergence,
    evaluate_multiobjective_selection,
    evaluate_near_optimal_plateau,
    evaluate_volatility_calibration,
)
from src.Bayesian_state.optimization.optimization_config import load_yaml
from src.Bayesian_state.utils.paths import RESULTS_DIR, ROOT_DIR


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_DIR = RESULTS_DIR / "state-based-hyper-cd" / "pmh" / "cond1_v9"
DEFAULT_CANDIDATES_JSON = (
    ROOT_DIR
    / "src"
    / "Bayesian_state"
    / "problems"
    / "modules"
    / "hypo_transition"
    / "candidates"
    / "hypo_transition_strategy_candidates.json"
)
TRANSITION_SPACE_KEY = "engine.modules.hypo_transitions_mod.kwargs"
DEFAULT_SELECTION_OBJECTIVES = (
    "hyper_selection_error",
    "distribution_ppc_interval_score",
    "distribution_intersection_score",
    "distribution_score",
    "accuracy_shape_score",
    "history_kernel_score",
    "switch_behavior_score",
)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def resolve_subjects(
    subjects: Sequence[int] | None,
    subject_range: Sequence[int] | None,
) -> list[int] | None:
    if subjects:
        return [int(x) for x in subjects]
    if subject_range:
        start, end = [int(x) for x in subject_range]
        return list(range(start, end + 1))
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be a mapping: {path}")
    return payload


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_maybe_project_path(raw: Any, *, base_dir: Path | None = None) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return resolve_project_path(path)


def infer_hyper_config_path(input_dir: Path, override: Path | None) -> Path | None:
    if override is not None:
        return resolve_project_path(override)
    best_path = input_dir / "best_hyperparams.json"
    if not best_path.is_file():
        return None
    payload = _load_json(best_path)
    raw = _as_mapping(payload.get("hyper")).get("config_path")
    return _resolve_maybe_project_path(raw)


def _stage_param_space(config: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    stages = _as_mapping(config.get("stages"))
    stage_cfg = _as_mapping(stages.get(str(stage)))
    stage_space = stage_cfg.get("hyperparam_space")
    if isinstance(stage_space, Mapping):
        return stage_space
    root_space = config.get("hyperparam_space")
    return root_space if isinstance(root_space, Mapping) else {}


def infer_candidate_source(
    *,
    hyper_config_path: Path | None,
    stage: str,
    candidates_json: Path | None,
    candidate_key: str | None,
) -> tuple[Path | None, str]:
    if candidates_json is not None:
        return resolve_project_path(candidates_json), str(candidate_key or "cond1")
    if hyper_config_path is not None and hyper_config_path.is_file():
        cfg = load_yaml(hyper_config_path)
        spec = _as_mapping(_stage_param_space(cfg, stage).get(TRANSITION_SPACE_KEY))
        source = _as_mapping(spec.get("values_from_json"))
        raw_path = source.get("path")
        raw_key = source.get("key")
        inferred_path = _resolve_maybe_project_path(raw_path, base_dir=hyper_config_path.parent)
        if inferred_path is not None and raw_key:
            return inferred_path, str(candidate_key or raw_key)
    return DEFAULT_CANDIDATES_JSON, str(candidate_key or "cond1")


def infer_base_sim_config_path(input_dir: Path, hyper_config_path: Path | None, override: Path | None) -> Path:
    if override is not None:
        return resolve_project_path(override)
    best_path = input_dir / "best_hyperparams.json"
    if best_path.is_file():
        payload = _load_json(best_path)
        raw = _as_mapping(payload.get("hyper")).get("base_sim_config_path")
        inferred = _resolve_maybe_project_path(raw)
        if inferred is not None:
            return inferred
    if hyper_config_path is not None and hyper_config_path.is_file():
        cfg = load_yaml(hyper_config_path)
        raw = cfg.get("base_sim_config_path")
        inferred = _resolve_maybe_project_path(raw, base_dir=hyper_config_path.parent)
        if inferred is not None:
            return inferred
    return resolve_project_path(DEFAULT_BASE_SIM_CONFIG)


def infer_secondary_config(hyper_config_path: Path | None) -> Mapping[str, Any]:
    if hyper_config_path is None or not hyper_config_path.is_file():
        return {}
    cfg = load_yaml(hyper_config_path)
    statistics_config = _as_mapping(cfg.get("statistics_config"))
    if statistics_config:
        return statistics_config
    return _as_mapping(cfg.get("acceptance_selection"))


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_csv_strings(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run convergence, plateau, and PPC/selection diagnostics for hyper-CD outputs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Hyper-CD output dir containing subject_*/ artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output dir; defaults to <input-dir>/hyper_evaluation",
    )
    parser.add_argument("--subjects", nargs="+", type=int, help="Subject IDs to evaluate")
    parser.add_argument(
        "--subject-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive subject range",
    )
    parser.add_argument("--stage", default="coarse", help="Hyper-CD stage to evaluate")
    parser.add_argument(
        "--hyper-config",
        type=Path,
        help="Hyper config YAML. Defaults to the path recorded in best_hyperparams.json",
    )
    parser.add_argument(
        "--base-sim-config",
        type=Path,
        help="Base simulation YAML for optional resampling diagnostics",
    )
    parser.add_argument(
        "--candidates-json",
        type=Path,
        help="Strategy candidate JSON. Defaults to values_from_json in the hyper config",
    )
    parser.add_argument(
        "--candidate-key",
        help="Candidate key in JSON. Defaults to values_from_json.key in the hyper config",
    )

    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--skip-plateau", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument(
        "--with-accuracy-diagnostic",
        action="store_true",
        help="Resimulate selected candidate points; expensive",
    )
    parser.add_argument(
        "--with-volatility-calibration",
        action="store_true",
        help="Run binary volatility PPC calibration for selected best points; expensive",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run core diagnostics plus expensive resampling diagnostics",
    )

    parser.add_argument(
        "--plateau-primary-metric",
        default="hyper_selection_error",
        help="Metric column used to define the near-optimal plateau",
    )
    parser.add_argument("--plateau-abs-tol", type=float, help="Defaults to hyper config statistics_config/acceptance_selection")
    parser.add_argument("--plateau-rel-tol", type=float, help="Defaults to hyper config statistics_config/acceptance_selection")
    parser.add_argument(
        "--selection-objectives",
        type=parse_csv_strings,
        default=list(DEFAULT_SELECTION_OBJECTIVES),
        help="Comma-separated minimization objectives for candidate diagnostics",
    )
    parser.add_argument("--selection-primary-metric", default="hyper_selection_error")
    parser.add_argument("--selection-primary-abs-tol", type=float, help="Defaults to hyper config statistics_config/acceptance_selection")
    parser.add_argument("--selection-primary-rel-tol", type=float, help="Defaults to hyper config statistics_config/acceptance_selection")

    parser.add_argument("--accuracy-repeats", type=int, default=256)
    parser.add_argument("--accuracy-max-candidates-per-subject", type=int, default=12)
    parser.add_argument("--accuracy-n-jobs", type=int)

    parser.add_argument("--volatility-model-repeats", type=int, default=128)
    parser.add_argument("--volatility-binary-samples-per-run", type=int, default=32)
    parser.add_argument("--volatility-n-jobs", type=int, default=8)
    parser.add_argument("--volatility-seed", type=int, default=20260622)
    return parser.parse_args()


def _add_paths(prefix: str, paths: Mapping[str, Any], out: dict[str, str]) -> None:
    for key, value in paths.items():
        out[f"{prefix}.{key}"] = str(value)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else input_dir / "hyper_evaluation"
    subjects = resolve_subjects(args.subjects, args.subject_range)
    hyper_config_path = infer_hyper_config_path(input_dir, args.hyper_config)
    secondary_cfg = infer_secondary_config(hyper_config_path)
    candidates_json, candidate_key = infer_candidate_source(
        hyper_config_path=hyper_config_path,
        stage=str(args.stage),
        candidates_json=args.candidates_json,
        candidate_key=args.candidate_key,
    )
    base_sim_config = infer_base_sim_config_path(input_dir, hyper_config_path, args.base_sim_config)

    primary_abs_tol = _float_or_default(secondary_cfg.get("primary_tolerance_abs"), 0.015)
    primary_rel_tol = _float_or_default(secondary_cfg.get("primary_tolerance_rel"), 0.05)
    plateau_abs_tol = float(args.plateau_abs_tol) if args.plateau_abs_tol is not None else primary_abs_tol
    plateau_rel_tol = float(args.plateau_rel_tol) if args.plateau_rel_tol is not None else primary_rel_tol
    selection_abs_tol = (
        float(args.selection_primary_abs_tol)
        if args.selection_primary_abs_tol is not None
        else primary_abs_tol
    )
    selection_rel_tol = (
        float(args.selection_primary_rel_tol)
        if args.selection_primary_rel_tol is not None
        else primary_rel_tol
    )

    LOGGER.info("Evaluating hyper-CD output: %s", input_dir)
    if hyper_config_path is not None:
        LOGGER.info("Using hyper config: %s", hyper_config_path)
    LOGGER.info("Using transition candidate lookup: %s [%s]", candidates_json, candidate_key)
    paths: dict[str, str] = {}

    if not args.skip_convergence:
        convergence_paths = evaluate_hyper_cd_convergence(
            input_dir,
            output_dir=output_dir,
            subjects=subjects,
            stage=str(args.stage),
            candidates_json=candidates_json,
            candidate_key=candidate_key,
        )
        _add_paths("convergence", convergence_paths, paths)

    if not args.skip_plateau:
        plateau_paths = evaluate_near_optimal_plateau(
            input_dir,
            output_dir=output_dir / "near_optimal_plateau",
            subjects=subjects,
            stage=str(args.stage),
            candidates_json=candidates_json,
            candidate_key=candidate_key,
            primary_metric=str(args.plateau_primary_metric),
            abs_tol=plateau_abs_tol,
            rel_tol=plateau_rel_tol,
        )
        _add_paths("near_optimal", plateau_paths, paths)

    if not args.skip_selection:
        selection_paths = evaluate_multiobjective_selection(
            input_dir,
            output_dir=output_dir / "selection_diagnostic",
            subjects=subjects,
            stage=str(args.stage),
            candidates_json=candidates_json,
            candidate_key=candidate_key,
            primary_metric=str(args.selection_primary_metric),
            primary_abs_tol=selection_abs_tol,
            primary_rel_tol=selection_rel_tol,
            objectives=list(args.selection_objectives),
            acc_mae_max=_float_or_default(secondary_cfg.get("distribution_accept_acc_mae_max"), 0.10),
            vol_ratio_min=_float_or_default(secondary_cfg.get("distribution_accept_vol_ratio_min"), 0.60),
            vol_ratio_max=_float_or_default(secondary_cfg.get("distribution_accept_vol_ratio_max"), 1.50),
            history_corr_min=_float_or_default(secondary_cfg.get("distribution_accept_history_corr_min"), 0.80),
            switch_abs_max=_float_or_default(secondary_cfg.get("distribution_accept_switch_score_max"), 0.10),
            include_legacy_multiobjective=False,
        )
        _add_paths("selection", selection_paths, paths)

    run_expensive = bool(args.all)
    if run_expensive or args.with_accuracy_diagnostic:
        accuracy_paths = diagnose_hyper_accuracy_sampling(
            input_dir,
            base_sim_config_path=base_sim_config,
            output_dir=output_dir / "accuracy_diagnostic",
            subjects=subjects,
            stage=str(args.stage),
            candidates_json=candidates_json,
            candidate_key=candidate_key,
            simulation_repeats=int(args.accuracy_repeats),
            max_candidates_per_subject=int(args.accuracy_max_candidates_per_subject),
            n_jobs=args.accuracy_n_jobs,
        )
        _add_paths("accuracy_diagnostic", accuracy_paths, paths)

    if run_expensive or args.with_volatility_calibration:
        volatility_paths = evaluate_volatility_calibration(
            input_dir=input_dir,
            output_dir=output_dir / "volatility_calibration",
            base_sim_config=base_sim_config,
            subjects=subjects,
            subject_range=None,
            model_repeats=int(args.volatility_model_repeats),
            binary_samples_per_run=int(args.volatility_binary_samples_per_run),
            n_jobs=int(args.volatility_n_jobs),
            seed=int(args.volatility_seed),
        )
        _add_paths("volatility_calibration", volatility_paths, paths)

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "hyper_config_path": str(hyper_config_path) if hyper_config_path else None,
                "base_sim_config_path": str(base_sim_config),
                "stage": str(args.stage),
                "subjects": subjects,
                "candidates_json": str(candidates_json) if candidates_json else None,
                "candidate_key": candidate_key,
                "outputs": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = str(manifest_path)
    LOGGER.info("Wrote hyper evaluation outputs to %s", output_dir)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
