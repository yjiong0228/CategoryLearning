#!/usr/bin/env python3
"""Directly compare the static cross-mechanism bank with frozen C1.

Both models are evaluated on the same reserved subjects and suffix trials.
The comparison answers whether deeper individual heterogeneity can substitute
for the dynamic C1 readout; it does not identify a unique cognitive mechanism.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_mechanism_screen import (  # noqa: E402
    _benjamini_hochberg,
    _bootstrap_delta,
    _paired_signflip_p,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-dir", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--base-seed", type=int, default=20261341)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_rows = pd.read_csv(args.global_dir / "mixture_subject_summary.csv")
    global_rows = global_rows.loc[
        global_rows["readout"].eq("static")
        & global_rows["model"].eq("global_candidate_bank")
    ].copy()
    screen_rows = pd.read_csv(args.screen_dir / "mixture_subject_summary.csv")
    c1_rows = screen_rows.loc[
        screen_rows["readout"].eq("c1")
        & screen_rows["family"].eq("F")
        & screen_rows["model"].eq("reference_candidate")
    ].copy()
    if global_rows["iSub"].duplicated().any() or c1_rows["iSub"].duplicated().any():
        raise ValueError("Each model must contribute exactly one row per subject.")
    merged = global_rows.merge(
        c1_rows,
        on="iSub",
        suffixes=("_global_static", "_c1"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No common subjects between the global bank and C1.")
    metrics = {
        "curve_crps": "negative",
        "summary_discrepancy": "negative",
        "combined_calibration_p": "positive",
        "curve_pointwise_interval_width_95": "negative",
    }
    rows: list[dict[str, Any]] = []
    for metric, direction in metrics.items():
        deltas = (
            merged[f"{metric}_global_static"].to_numpy(dtype=float)
            - merged[f"{metric}_c1"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "comparison": "static_global_minus_c1",
                "metric": metric,
                "better_direction": direction,
                **_bootstrap_delta(
                    deltas,
                    seed=stable_seed(
                        {
                            "seed_role": "static_global_vs_c1_bootstrap",
                            "base_seed": int(args.base_seed),
                            "metric": metric,
                        }
                    ),
                    repeats=int(args.bootstrap_repeats),
                ),
                "paired_signflip_p": _paired_signflip_p(
                    deltas,
                    seed=stable_seed(
                        {
                            "seed_role": "static_global_vs_c1_signflip",
                            "base_seed": int(args.base_seed),
                            "metric": metric,
                        }
                    ),
                    repeats=int(args.bootstrap_repeats),
                ),
                "global_better_subject_n": int(
                    np.sum(deltas < 0.0 if direction == "negative" else deltas > 0.0)
                ),
                "subject_n": int(deltas.size),
            }
        )
    q_values = _benjamini_hochberg([row["paired_signflip_p"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["paired_signflip_q"] = float(q_value)

    global_pass = merged["combined_pass_95_global_static"].astype(bool).to_numpy()
    c1_pass = merged["combined_pass_95_c1"].astype(bool).to_numpy()
    improved = int(np.sum(global_pass & ~c1_pass))
    worsened = int(np.sum(~global_pass & c1_pass))
    discordant = improved + worsened
    coverage = {
        "comparison": "static_global_minus_c1",
        "global_pass_n": int(global_pass.sum()),
        "c1_pass_n": int(c1_pass.sum()),
        "subject_n": int(global_pass.size),
        "global_only_pass_n": improved,
        "c1_only_pass_n": worsened,
        "exact_p": float(binomtest(improved, discordant, 0.5).pvalue) if discordant else 1.0,
    }
    subject_output = pd.DataFrame(
        {
            "subject_id": merged["iSub"].astype(int),
            "global_pass_95": global_pass,
            "c1_pass_95": c1_pass,
            **{
                f"delta_{metric}": (
                    merged[f"{metric}_global_static"].to_numpy(dtype=float)
                    - merged[f"{metric}_c1"].to_numpy(dtype=float)
                )
                for metric in metrics
            },
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_output.to_csv(args.output_dir / "subject_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "metric_comparison.csv", index=False)
    pd.DataFrame([coverage]).to_csv(args.output_dir / "coverage_comparison.csv", index=False)
    summary = {"metrics": rows, "coverage": coverage}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
