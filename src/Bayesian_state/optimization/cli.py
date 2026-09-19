"""Unified CLI for hyperparameter optimization backends."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.optimization.search.grid import HyperGridOptimizer
from src.Bayesian_state.optimization.artifacts import to_builtin
from src.Bayesian_state.simulation.config import load_yaml
from src.Bayesian_state.utils.paths import ROOT_DIR
from src.Bayesian_state.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter optimization",
        allow_abbrev=False,
    )
    p.add_argument("--backend", choices=("grid", "cd", "adaptive"), help="Default: config backend, otherwise legacy cd")
    p.add_argument("--config", required=True, type=Path, help="Hyper YAML config")
    p.add_argument("--subjects", nargs="+", type=int, help="Override subject list")
    p.add_argument("--subject-range", nargs=2, type=int, metavar=("START", "END"), help="Override subject range")
    p.add_argument("--output-dir", type=Path, help="New adaptive fit output directory")
    p.add_argument("--conditions", nargs="+", type=int, choices=(1, 2, 3))
    p.add_argument("--smoke", action="store_true", help="Adaptive: one subject, 32 trials, one worker")
    p.add_argument("--dry-run", action="store_true", help="Adaptive: validate inputs without fitting")
    p.add_argument("--stage", choices=("coarse", "fine", "all"), default="all", help="Run coarse/fine/all stages")
    p.add_argument(
        "--resume-from-coarse",
        action="store_true",
        help="With --stage fine, load existing coarse all_combinations.jsonl and run only fine.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the same adaptive fit or schema-v2 Hyper-CD run from saved tasks."
        ),
    )
    return p.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (ROOT_DIR / cfg_path).resolve()

    cfg = load_yaml(cfg_path)
    backend = args.backend or ("adaptive" if cfg.get("backend") == "model0826_adaptive" else "cd")
    if backend == "adaptive":
        from .adaptive_fit import run_fit
        if args.stage != "all" or args.resume_from_coarse:
            raise ValueError("Adaptive fitting has no coarse/fine stage switch")
        if args.subjects and args.subject_range:
            raise ValueError("Use subjects or subject-range, not both")
        subjects = args.subjects
        if args.subject_range:
            subjects = list(range(args.subject_range[0], args.subject_range[1] + 1))
        result = run_fit(cfg_path, args.output_dir, subjects, args.conditions,
                         smoke=args.smoke, resume=args.resume, dry_run=args.dry_run)
        print(json.dumps(to_builtin(result), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if cfg.get("backend") == "model0826_adaptive":
        raise ValueError("Adaptive configuration cannot be passed to legacy grid/CD")
    if args.output_dir or args.conditions or args.smoke or args.dry_run:
        raise ValueError("output-dir/conditions/smoke/dry-run are adaptive backend options")
    args.backend = backend
    optimizer_cls = HyperGridOptimizer if args.backend == "grid" else HyperCDOptimizer
    optimizer = optimizer_cls(cfg, cfg_path)
    subjects = optimizer.resolve_subjects(args.subjects, args.subject_range)
    run_kwargs = {
        "subjects": subjects,
        "stage": args.stage,
        "resume_from_coarse": bool(args.resume_from_coarse),
    }
    if args.backend == "cd":
        run_kwargs["resume"] = bool(args.resume)
    elif args.resume:
        raise ValueError("--resume is supported only with --backend cd")
    result = optimizer.run(**run_kwargs)

    print(f"Hyper-{args.backend} optimization done.")
    print(json.dumps(to_builtin(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
