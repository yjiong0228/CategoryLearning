"""Fit one exp123 subject from independent starts and check integration accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from .exp123 import run_config
from .fit_diagnostics import summarize_study


def run_study(config: dict, seeds: Sequence[int], check_orders: Sequence[int] = (12, 16),
              nll_tolerance: float = 0.1, probability_tolerance: float = 0.001) -> dict:
    """Run sequential independent searches, parallelizing only GA candidates.

    This workflow fits the complete record of one subject. Higher-order replays
    evaluate the fitted initial parameters without further optimization.
    """
    if config.get("mode") != "fit" or len(config.get("subjects", [])) != 1:
        raise ValueError("study requires mode=fit and exactly one subject")
    if config.get("fit_first_n") is not None or config.get("max_trials") is not None:
        raise ValueError("study requires the complete subject record")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("provide at least two distinct nonnegative seeds")
    base_order = config.get("model", {}).get("quadrature_order", 8)
    if len(check_orders) < 2 or any(a >= b for a, b in zip([base_order, *check_orders], check_orders)):
        raise ValueError("provide at least two strictly increasing check orders above the fit order")
    if nll_tolerance <= 0 or probability_tolerance <= 0:
        raise ValueError("integration tolerances must be positive")
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    plan = {"status": "running", "config": config, "seeds": list(seeds),
            "check_orders": list(check_orders), "nll_tolerance": nll_tolerance,
            "probability_tolerance": probability_tolerance,
            "selection": "lowest complete-record NLL at the highest check order",
            "evidence_scope": "in-sample development fit; fixed fitted parameters",
            "completed_seeds": []}
    path = root / "study.json"

    def save() -> None:
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    save()
    try:
        for seed in seeds:
            print(f"Starting subject {config['subjects'][0]}, seed {seed}", flush=True)
            summary = run_config({**config, "seed": seed, "output_dir": str(root / f"seed_{seed}")})
            plan["completed_seeds"].append(seed)
            save()
            print(f"Finished seed {seed}: NLL={summary['runs'][0]['train_nll']:.6f}", flush=True)
        plan["diagnostics"] = summarize_study(root, plan)
        plan["status"] = "computationally_complete"
    except Exception as error:
        plan["status"] = "failed"
        plan["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    return plan


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--check-orders", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--nll-tolerance", type=float, default=0.1)
    parser.add_argument("--probability-tolerance", type=float, default=0.001)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    if args.outdir is not None:
        config["output_dir"] = str(args.outdir)
    result = run_study(config, args.seeds, args.check_orders,
                       args.nll_tolerance, args.probability_tolerance)
    print(json.dumps({"status": result["status"], "output_dir": config["output_dir"],
                      **result["diagnostics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
