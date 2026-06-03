"""CLI entrypoint for two-layer hyperparameter optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.Bayesian_state.hyper_opt.optimizer import HyperOptimizer
from src.Bayesian_state.utils.paths import ROOT_DIR


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Hyper config must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-layer hyperparameter optimization")
    p.add_argument("--config", required=True, type=Path, help="Hyper-optimizer YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Override subject list")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Override subject range")
    p.add_argument("--stage", choices=("coarse", "fine", "all"), default="all", help="Run only selected stage or all")
    p.add_argument(
        "--resume-from-coarse",
        action="store_true",
        help=(
            "With --stage fine, load existing coarse all_combinations.jsonl, "
            "trim stale fine rows, then run only fine."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()

    cfg = load_yaml(cfg_path)
    optimizer = HyperOptimizer(cfg, cfg_path)
    subjects = optimizer.resolve_subjects(args.subjects, args.subject_range)
    result = optimizer.run(
        subjects=subjects,
        stage=args.stage,
        resume_from_coarse=bool(args.resume_from_coarse),
    )

    print("Hyper optimization done.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
