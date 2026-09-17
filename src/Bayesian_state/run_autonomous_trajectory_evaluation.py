"""Generate coherent autonomous learning-trajectory distribution figures.

Usage:
    python -m src.Bayesian_state.run_autonomous_trajectory_evaluation \
        --config path/to/frozen_simulation.yaml \
        --subject 101 \
        --output-dir path/to/new/autonomous_trajectories
"""

from __future__ import annotations

from src.Bayesian_state.utils.parallel import MODEL_0826_PARALLEL_BUDGET

import argparse
import os
import tempfile
from pathlib import Path

_PROJECT_TMP = Path(tempfile.gettempdir()) / "categorylearning-cache"
_MPL_CACHE = _PROJECT_TMP / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_TMP))

import matplotlib

matplotlib.use("Agg")

from .evaluation.autonomous_trajectories import (
    run_autonomous_trajectory_evaluation,
)
from .utils.paths import ROOT_DIR


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-parameter coherent autonomous learning trajectories "
            "and summarize whole-curve shape variation."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label")
    parser.add_argument("--rollouts", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=MODEL_0826_PARALLEL_BUDGET,
                        help="Process budget (default: 128); each worker uses one numeric thread")
    parser.add_argument("--analysis-seed", type=int, default=20260831)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--visible-trajectories", type=int, default=48)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument("--mastery-sustain-windows", type=int, default=8)
    parser.add_argument("--final-block-trials", type=int, default=64)
    parser.add_argument("--max-clusters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_autonomous_trajectory_evaluation(
        config_path=_project_path(args.config),
        subject_id=args.subject,
        output_dir=_project_path(args.output_dir),
        label=args.label,
        rollout_count=args.rollouts,
        n_jobs=args.n_jobs,
        analysis_seed=args.analysis_seed,
        window_size=args.window_size,
        visible_trajectories=args.visible_trajectories,
        mastery_threshold=args.mastery_threshold,
        mastery_sustain_windows=args.mastery_sustain_windows,
        final_block_trials=args.final_block_trials,
        max_clusters=args.max_clusters,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
