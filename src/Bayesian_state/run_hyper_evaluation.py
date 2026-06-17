"""Run hyper-CD convergence evaluation for hyperparameter search outputs.

Usage:
    python -m src.Bayesian_state.run_hyper_evaluation \
        --input-dir results/state-based-hyper-cd/pmh/cond1_v3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from src.Bayesian_state.utils.hyper_evaluation import (
    evaluate_hyper_cd_convergence,
    evaluate_near_optimal_plateau,
)
from src.Bayesian_state.utils.paths import RESULTS_DIR, ROOT_DIR


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_DIR = RESULTS_DIR / "state-based-hyper-cd" / "pmh" / "cond1_v3"
DEFAULT_CANDIDATES_JSON = (
    ROOT_DIR
    / "src"
    / "Bayesian_state"
    / "problems"
    / "modules"
    / "hypo_transition_strategy_candidates.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run convergence diagnostics for hyper-CD output directories"
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
        "--candidates-json",
        type=Path,
        default=DEFAULT_CANDIDATES_JSON,
        help="Strategy candidate JSON used to recover readable candidate ids",
    )
    parser.add_argument("--candidate-key", default="cond1", help="Candidate key in JSON")
    parser.add_argument(
        "--skip-plateau",
        action="store_true",
        help="Only run restart-convergence diagnostics",
    )
    parser.add_argument(
        "--plateau-primary-metric",
        default="hyper_selection_error",
        help="Metric column used to define the near-optimal plateau",
    )
    parser.add_argument(
        "--plateau-abs-tol",
        type=float,
        default=0.02,
        help="Absolute tolerance above the best primary metric",
    )
    parser.add_argument(
        "--plateau-rel-tol",
        type=float,
        default=0.08,
        help="Relative tolerance above the best primary metric",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else input_dir / "hyper_evaluation"
    candidates_json = resolve_project_path(args.candidates_json)
    subjects = resolve_subjects(args.subjects, args.subject_range)

    LOGGER.info("Evaluating hyper-CD output: %s", input_dir)
    convergence_paths = evaluate_hyper_cd_convergence(
        input_dir,
        output_dir=output_dir,
        subjects=subjects,
        stage=str(args.stage),
        candidates_json=candidates_json,
        candidate_key=str(args.candidate_key),
    )
    paths = {f"convergence.{key}": value for key, value in convergence_paths.items()}
    if not args.skip_plateau:
        plateau_paths = evaluate_near_optimal_plateau(
            input_dir,
            output_dir=output_dir / "near_optimal_plateau",
            subjects=subjects,
            stage=str(args.stage),
            candidates_json=candidates_json,
            candidate_key=str(args.candidate_key),
            primary_metric=str(args.plateau_primary_metric),
            abs_tol=float(args.plateau_abs_tol),
            rel_tol=float(args.plateau_rel_tol),
        )
        paths.update({f"near_optimal.{key}": value for key, value in plateau_paths.items()})
    LOGGER.info("Wrote hyper evaluation outputs to %s", output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
