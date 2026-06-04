"""CLI for simulation-based coordinate-descent hyperparameter optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.Bayesian_state.hyper_cd.optimizer import HyperCDOptimizer
from src.Bayesian_state.utils.optimization_config import load_yaml
from src.Bayesian_state.utils.paths import ROOT_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coordinate-descent hyperparameter optimization")
    p.add_argument("--config", required=True, type=Path, help="Hyper-CD YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Override subject list")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Override subject range")
    p.add_argument("--stage", choices=("coarse", "fine", "all"), default="all", help="Run coarse/fine/all stages")
    p.add_argument(
        "--resume-from-coarse",
        action="store_true",
        help="With --stage fine, load existing coarse all_combinations.jsonl and rerun only fine.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()

    cfg = load_yaml(cfg_path)
    optimizer = HyperCDOptimizer(cfg, cfg_path)
    subjects = optimizer.resolve_subjects(args.subjects, args.subject_range)
    result = optimizer.run(
        subjects=subjects,
        stage=args.stage,
        resume_from_coarse=bool(args.resume_from_coarse),
    )

    print("Hyper-CD optimization done.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
