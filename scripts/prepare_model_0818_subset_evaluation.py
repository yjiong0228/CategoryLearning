#!/usr/bin/env python3
"""Prepare a standard simulation config for completed Model 0818 subjects."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0818_exploratory_observed_fit import (  # noqa: E402
    prepare_standard_evaluation_config,
)
from src.Bayesian_state.optimization.observed_fit import (  # noqa: E402
    build_model_0818_hyper_config,
)
from src.Bayesian_state.optimization.parameter_space import (  # noqa: E402
    load_parameter_space,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402


DEFAULT_ANALYSIS = (
    ROOT / "configs/specific_models/model_0818_cond1_full_observed_fit.yaml"
)
DEFAULT_WORKSPACE = (
    ROOT / "results/model_0818/cond1/full_observed_fit_v1/evaluation_after_8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--subjects", nargs="+", type=int, required=True)
    parser.add_argument(
        "--additional-search-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Fallback search directory containing subject_<id>/best_hyperparams.json; "
            "may be repeated. The canonical analysis search directory is checked first."
        ),
    )
    parser.add_argument("--filter-seed-count", type=int, default=16)
    parser.add_argument("--hyper-base-seed", type=int, default=20260826)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.analysis_config.resolve()
    config = load_yaml(config_path)
    output = (ROOT / config["output_dir"]).resolve()
    workspace = args.workspace.resolve()
    subjects = sorted({int(value) for value in args.subjects})
    if not subjects or not set(subjects).issubset(set(config["subjects"])):
        raise ValueError("subset subjects are outside the configured cohort")
    if int(args.filter_seed_count) < 2:
        raise ValueError("filter-seed-count must be at least 2")

    subset_search = workspace / "search"
    search_dirs = [
        output / "search",
        *(path.resolve() for path in args.additional_search_dir),
    ]
    for subject_id in subjects:
        candidates = [
            directory / f"subject_{subject_id}" / "best_hyperparams.json"
            for directory in search_dirs
        ]
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                "no completed search result found in: "
                + ", ".join(str(path) for path in candidates)
            )
        target_dir = subset_search / f"subject_{subject_id}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)

    local = deepcopy(config)
    local["evaluation"] = deepcopy(config["evaluation"])
    local["evaluation"]["simulation_dir"] = str(workspace / "simulation")
    local["evaluation"]["output_dir"] = str(workspace / "model_evaluation")
    local["evaluation"]["filter_seed_count"] = int(args.filter_seed_count)
    local["evaluation"]["n_jobs_per_subject"] = int(args.filter_seed_count)
    local["evaluation"]["hyper_base_seed"] = int(args.hyper_base_seed)

    parameter_space = load_parameter_space((ROOT / config["parameter_space"]).resolve())
    hyper_config = build_model_0818_hyper_config(
        local,
        parameter_space,
        root=ROOT,
        parallel_budget=1,
    )
    generated = prepare_standard_evaluation_config(
        subjects=subjects,
        config=local,
        config_path=config_path,
        hyper_config=hyper_config,
        output=workspace,
    )
    print(generated)


if __name__ == "__main__":
    main()
