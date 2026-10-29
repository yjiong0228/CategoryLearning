"""Cli for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd
from .data import DEFAULT_EXP5_CSV, sort_subjects, subject_id_values
from .binary_workflow import run_one_subject, write_combined_outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=None,
        help="Combined processed CSV. Defaults to data/exp5/processed/Rule1_processed.csv.",
    )
    parser.add_argument("--outdir", default="results/Cohen_model/Exp5")
    parser.add_argument("--n-bits", type=int, default=5)
    parser.add_argument("--max-order", type=int, default=None)
    parser.add_argument("--lsb-first", action="store_true")
    parser.add_argument("--update-rule", choices=["si_ratio", "paper_eq3", "exact_gradient"], default="si_ratio")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--fit-first-n", type=int, default=None)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--anneal-steps", type=int, default=120)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--subjects", nargs="*", default=None, help="iSub values to fit. Defaults to all subjects in the CSV.")
    parser.add_argument("--isession", type=int, default=1, help="Value for the first column of formula CSV.")
    parser.add_argument("--formula-csv-name", default="formula_steps.csv", help="Formula CSV filename.")
    parser.add_argument("--formula-decimals", type=int, default=4, help="Decimals used in formula strings.")
    parser.add_argument("--formula-max-terms", type=int, default=31, help="Max number of feature terms shown in each vector; 31 shows all 5-bit features.")
    parser.add_argument("--include-after-update", action="store_true", help="Append post-feedback weight vector to each formula.")
    args = parser.parse_args()
    if args.fit_first_n is not None and not args.fit:
        raise ValueError("--fit-first-n requires --fit")

    csv_path = Path(args.csv) if args.csv else DEFAULT_EXP5_CSV
    outdir = Path(args.outdir)
    if outdir.exists():
        raise FileExistsError(f"Choose a new output directory: {outdir}")
    reserved = {"trial_predictions.csv", "final_feature_weights.csv", "summary.json",
                "alpha_pos_trajectory_wide.csv", "alpha_neg_trajectory_wide.csv",
                "alpha_diff_trajectory_wide.csv", "alpha_trajectory_long.csv"}
    if (Path(args.formula_csv_name).name != args.formula_csv_name or
            not args.formula_csv_name.endswith(".csv") or args.formula_csv_name in reserved):
        raise ValueError("formula-csv-name must be a distinct CSV basename")
    full_df = pd.read_csv(csv_path)
    subjects = sort_subjects(args.subjects) if args.subjects else subject_id_values(full_df)
    runs = []
    for subject_id in subjects:
        subject_df = full_df if subject_id == "all" else full_df[full_df["iSub"].astype(str) == subject_id].copy()
        if subject_df.empty:
            raise ValueError(f"No rows found for subject {subject_id}")
        runs.append(run_one_subject(subject_id, subject_df, csv_path, args))

    summary = write_combined_outputs(runs, outdir, csv_path, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
