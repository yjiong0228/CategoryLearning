"""Unified CLI for hyperparameter optimization backends."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.Bayesian_state.optimization.hyper_cd_optimizer import HyperCDOptimizer
from src.Bayesian_state.optimization.hyper_grid_optimizer import HyperGridOptimizer
from src.Bayesian_state.optimization.hyper_utils import to_builtin
from src.Bayesian_state.simulation.simulation_config import load_yaml
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.base import configure_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter optimization")
    p.add_argument("--backend", choices=("grid", "cd"), required=True, help="Hyper optimizer backend")
    p.add_argument("--config", required=True, type=Path, help="Hyper YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Override subject list")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Override subject range")
    p.add_argument("--stage", choices=("coarse", "fine", "all"), default="all", help="Run coarse/fine/all stages")
    p.add_argument(
        "--resume-from-coarse",
        action="store_true",
        help="With --stage fine, load existing coarse all_combinations.jsonl and run only fine.",
    )
    return p.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()

    cfg = load_yaml(cfg_path)
    optimizer_cls = HyperGridOptimizer if args.backend == "grid" else HyperCDOptimizer
    optimizer = optimizer_cls(cfg, cfg_path)
    subjects = optimizer.resolve_subjects(args.subjects, args.subject_range)
    result = optimizer.run(
        subjects=subjects,
        stage=args.stage,
        resume_from_coarse=bool(args.resume_from_coarse),
    )

    print(f"Hyper-{args.backend} optimization done.")
    print(json.dumps(to_builtin(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
