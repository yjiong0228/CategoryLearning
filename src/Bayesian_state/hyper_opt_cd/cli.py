"""CLI for coordinate-descent two-layer hyper optimization."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.Bayesian_state.hyper_opt_cd.optimizer import HyperOptimizerCD
from src.Bayesian_state.utils.paths import ROOT_DIR


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coordinate-descent hyper optimization")
    p.add_argument("--config", required=True, type=Path, help="Hyper optimizer YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Override subject list")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Override subject range")
    p.add_argument("--stage", choices=["coarse", "fine", "all"], default="all", help="Run coarse/fine/all stages")
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
    args = _parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()

    cfg = _load_yaml(cfg_path)
    optimizer = HyperOptimizerCD(cfg, cfg_path)
    subjects = optimizer.resolve_subjects(args.subjects, args.subject_range)
    result = optimizer.run(
        subjects=subjects,
        stage=args.stage,
        resume_from_coarse=bool(args.resume_from_coarse),
    )
    print(f"Hyper CD done. Results saved to: {result.get('output_dir')}")


if __name__ == "__main__":
    main()
