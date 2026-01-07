"""Aggregate per-subject AMR results and generate evaluation plots.

Usage example:

    conda activate cate_learn
    python -m src.Bayesian_state.eval_amr_results \
        --input-dir results/state-based-AMR-result/pmh/cond1 \
        --aggregate-output results/state-based-AMR-result/pmh/cond1_agg.json \
        --plot-accuracy results/state-based-AMR-result/pmh/cond1_accuracy.png

- Aggregates subject_*.json in the input directory (same format as run_amr_optimization outputs).
- Saves aggregated JSON.
- Runs ModelEval.plot_accuracy_comparison to produce accuracy figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Use non-interactive backend for batch plotting
matplotlib.use("Agg")

from src.Bayesian_state.aggregate_amr_results import aggregate
from src.Bayesian_state.utils.model_evaluation import ModelEval
from src.Bayesian_state.utils.oral_process import Oral_to_coordinate
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate AMR results and plot evaluation charts")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing subject_*.json")
    p.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help="Where to save aggregated JSON (default: <input-dir>/all_subjects.json)",
    )
    p.add_argument(
        "--plot-accuracy",
        type=Path,
        default=None,
        help="Where to save accuracy comparison plot (default: <input-dir>/accuracy.png)",
    )
    p.add_argument(
        "--plot-oral",
        type=Path,
        default=None,
        help="Optional: save oral vs model k comparison (requires best_step_results)",
    )
    p.add_argument(
        "--plot-cluster",
        type=Path,
        default=None,
        help="Optional: save cluster amount comparison (requires best_step_results)",
    )
    p.add_argument(
        "--oral-data",
        type=Path,
        default=Path("data/processed/Task2_processed.csv"),
        help="Path to Task2 processed CSV with oral fields (for oral vs model plot)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Defaults based on input_dir
    agg_out = args.aggregate_output or (input_dir / "all_subjects.json")
    plot_out = args.plot_accuracy or (input_dir / "accuracy.png")
    plot_oral = args.plot_oral or (input_dir / "oral_vs_model.png")
    plot_cluster = args.plot_cluster or (input_dir / "cluster_amount.png")
    oral_data_path = args.oral_data

    # Aggregate per-subject results
    aggregated = aggregate(input_dir)
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated {len(aggregated)} subjects -> {agg_out}")

    # Plot accuracy comparison
    me = ModelEval()
    me.plot_accuracy_comparison(aggregated, save_path=str(plot_out))
    print(f"Saved accuracy plot -> {plot_out}")

    # Optional plots when step logs exist
    has_steps = any(v.get("best_step_results") for v in aggregated.values())
    if has_steps:
        if args.plot_cluster is not None or plot_cluster:
            me.plot_cluster_amount(aggregated, save_path=str(plot_cluster))
            print(f"Saved cluster amount plot -> {plot_cluster}")
        else:
            print("Cluster plot targ0et not provided; skipping.")
        # Oral comparison: build oral hits from processed CSV if available
        if oral_data_path and oral_data_path.exists():
            oral_df = pd.read_csv(oral_data_path)
            oral_hits = Oral_to_coordinate().get_oral_hypo_hits(oral_df)
            me.plot_k_oral_comparison(aggregated, oral_hits, save_path=str(plot_oral))
            print(f"Saved oral vs model plot -> {plot_oral}")
        else:
            print(f"Oral data not found at {oral_data_path}; skipping oral vs model plot.")
    else:
        print("No best_step_results found; skipping oral/cluster plots.")


if __name__ == "__main__":
    main()
