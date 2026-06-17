"""Run accuracy-shape diagnostics for hyper-CD candidate points.

Usage:
    python -m src.Bayesian_state.run_hyper_accuracy_diagnostic \
        --input-dir results/state-based-hyper-cd/pmh/cond1_v3 \
        --repeats 256
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from src.Bayesian_state.utils.hyper_evaluation import diagnose_hyper_accuracy_sampling
from src.Bayesian_state.utils.paths import RESULTS_DIR, ROOT_DIR


LOGGER = logging.getLogger(__name__)
DEFAULT_INPUT_DIR = RESULTS_DIR / "state-based-hyper-cd" / "pmh" / "cond1_v3"
DEFAULT_BASE_SIM_CONFIG = ROOT_DIR / "configs" / "simulation_cfg" / "pmh_cond1_simulation.yaml"
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
        description="Resample selected hyper-CD candidates and evaluate accuracy-curve shape"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Hyper-CD output dir containing subject_*/ artifacts",
    )
    parser.add_argument(
        "--base-sim-config",
        type=Path,
        default=DEFAULT_BASE_SIM_CONFIG,
        help="Base simulation YAML used by the hyper-CD run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output dir; defaults to <input-dir>/accuracy_diagnostic",
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
        "--repeats",
        type=int,
        default=256,
        help="Simulation repeats per selected candidate",
    )
    parser.add_argument(
        "--max-candidates-per-subject",
        type=int,
        default=12,
        help="Maximum hyper-CD candidates to resample per subject",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Parallel jobs for repeats; defaults to the base simulation config n_jobs",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    base_sim_config = resolve_project_path(args.base_sim_config)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else input_dir / "accuracy_diagnostic"
    candidates_json = resolve_project_path(args.candidates_json)
    subjects = resolve_subjects(args.subjects, args.subject_range)

    LOGGER.info("Running hyper accuracy diagnostic: %s", input_dir)
    paths = diagnose_hyper_accuracy_sampling(
        input_dir,
        base_sim_config_path=base_sim_config,
        output_dir=output_dir,
        subjects=subjects,
        stage=str(args.stage),
        candidates_json=candidates_json,
        candidate_key=str(args.candidate_key),
        simulation_repeats=int(args.repeats),
        max_candidates_per_subject=int(args.max_candidates_per_subject),
        n_jobs=args.n_jobs,
    )
    LOGGER.info("Wrote hyper accuracy diagnostic outputs to %s", output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
