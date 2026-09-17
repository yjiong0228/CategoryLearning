"""CLI for observed-history-conditioned internal cognitive trajectories."""

from __future__ import annotations

from src.Bayesian_state.utils.parallel import MODEL_0826_PARALLEL_BUDGET

import argparse
from pathlib import Path

from .evaluation.internal_cognitive_trajectories import (
    run_internal_cognitive_trajectory_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace complete particle genealogies conditioned on one subject's "
            "observed choices and render internal cognitive-trajectory figures."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--path-draws", type=int, default=500)
    parser.add_argument("--analysis-seed", type=int, default=20260831)
    parser.add_argument("--jobs", type=int, default=MODEL_0826_PARALLEL_BUDGET,
                        help="Process budget (default: 128); each worker uses one numeric thread")
    parser.add_argument("--label", type=str)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_internal_cognitive_trajectory_evaluation(
        config_path=args.config,
        subject_id=int(args.subject),
        output_dir=args.output_dir,
        particle_count=int(args.particles),
        seed_count=int(args.seed_count),
        path_draw_count=int(args.path_draws),
        analysis_seed=int(args.analysis_seed),
        n_jobs=int(args.jobs),
        label=args.label,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
