"""Binary workflow for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any
import numpy as np
import argparse
import json
import math
import pandas as pd
from .features import feature_name
from .binary import FeatureMixtureStrict
from .fitting import FitConfig, fit_paper_style
from .data import DEFAULT_EXP5_CSV, prepare_dataframe, decode_binary_labels
from .trace import run_session_with_formula_trace
from .exports import make_wide_df, make_alpha_long_df, concat_run_df
from .plotting import plot_subject_learning_curves, plot_subject_feature_grid

Array = np.ndarray


def run_one_subject(subject_id: str, subject_df: pd.DataFrame, source_csv: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from .fitting import validate_prefix
    validate_prefix(args.fit_first_n, len(subject_df))
    max_order = args.max_order if args.max_order is not None else args.n_bits
    sort_cols = [c for c in ["iSession", "iBlock", "iTrial"] if c in subject_df.columns]
    if sort_cols:
        subject_df = subject_df.sort_values(sort_cols).reset_index(drop=True)
    feature_encoding = getattr(args, "feature_encoding", "pm1")
    label_encoding = getattr(args, "label_encoding", "category12")
    df, X, y, r = prepare_dataframe(subject_df, args.n_bits, msb_first=not args.lsb_first,
                                    feature_encoding=feature_encoding, label_encoding=label_encoding)

    if args.fit:
        cfg = FitConfig(
            population_size=args.population_size,
            generations=args.generations,
            anneal_steps=args.anneal_steps,
            seed=args.seed,
        )
        initial_model, fit_info = fit_paper_style(
            X, y, r, n_bits=args.n_bits, max_order=max_order,
            memory_window=1, fit_first_n=args.fit_first_n,
            config=cfg, verbose=not args.quiet, update_rule=getattr(args, "update_rule", "si_ratio")
        )
    else:
        initial_model = FeatureMixtureStrict(
            n_bits=args.n_bits, max_order=max_order,
            beta=args.beta, eta=args.eta, gamma=0.0,
            memory_window=1, seed=args.seed, update_rule=getattr(args, "update_rule", "si_ratio")
        )
        fit_info = {
            "mode": "no_fit_replay",
            "gamma0_fixed_to_zero": True,
            "alpha_update": initial_model.update_rule,
        }

    run_model = initial_model.copy()
    formula_max_terms = None if args.formula_max_terms <= 0 else args.formula_max_terms
    res, formula_strings = run_session_with_formula_trace(
        run_model,
        X,
        y,
        choices=r,
        update=True,
        decimals=args.formula_decimals,
        max_terms=formula_max_terms,
        include_after_update=args.include_after_update,
    )

    trial_values = df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1)
    formula_df = pd.DataFrame({
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), subject_id),
        "iSession": df["iSession"].to_numpy() if "iSession" in df.columns else np.full(len(df), args.isession),
        "iTrial": trial_values,
        "stiID": df["stiID"].to_numpy(),
        "formula": formula_strings,
    })

    feature_names = [feature_name(s) for s in run_model.subsets]
    alpha_pos_trials = res["alpha_pos_before_choice"]
    alpha_neg_trials = res["alpha_neg_before_choice"]
    alpha_diff_trials = res["alpha_pos_minus_alpha_neg_before_choice"]
    target_feature = getattr(args, "target_feature", None)
    if target_feature is None and source_csv.resolve() == DEFAULT_EXP5_CSV.resolve() and args.n_bits >= 4:
        target_feature = "x24"

    df["model_p_category1"] = res["p_pos"]
    df["model_p_correct"] = res["p_correct"]
    df["model_p_observed_choice"] = res["p_observed_choice"]
    df["model_pred_pm1"] = res["predicted_label"].astype(int)
    df["model_pred_category"] = decode_binary_labels(res["predicted_label"], label_encoding)
    df["model_pred_matches_choice"] = (res["predicted_label"] == r).astype(int)
    df["gamma_before_update"] = res["gamma_before_update"]

    trial_col = "iTrial" if "iTrial" in df.columns else "trial_index"
    if trial_col == "trial_index":
        df["trial_index"] = np.arange(1, len(df) + 1)

    df["rolling_accuracy_16"] = df["correct"].rolling(16, min_periods=16).mean()
    df["rolling_model_p_correct_16"] = df["model_p_correct"].rolling(16, min_periods=16).mean()

    feat_rows = []
    for i, s in enumerate(run_model.subsets):
        sep = float(run_model.alpha_pos[i] - run_model.alpha_neg[i])
        feat_rows.append({
            "iSub": subject_id,
            "feature": feature_name(s),
            "subset_zero_based": str(s),
            "subset_one_based": str(tuple(j + 1 for j in s)),
            "alpha_pos_final": float(run_model.alpha_pos[i]),
            "alpha_neg_final": float(run_model.alpha_neg[i]),
            "alpha_pos_minus_alpha_neg_final": sep,
            "abs_separation_final": abs(sep),
        })
    feat_df = pd.DataFrame(feat_rows).sort_values("abs_separation_final", ascending=False)

    task_feature = target_feature
    task_rank = None
    if task_feature in feat_df["feature"].to_numpy():
        task_rank = int(np.where(feat_df["feature"].to_numpy() == task_feature)[0][0] + 1)

    future = {}
    if args.fit_first_n is not None and args.fit_first_n < len(df):
        s = slice(args.fit_first_n, len(df))
        future = {
            "fit_first_n": int(args.fit_first_n),
            "future_choice_agreement": float(np.mean(res["predicted_label"][s] == r[s])),
            "future_mean_p_correct": float(np.mean(res["p_correct"][s])),
            "future_negative_log_likelihood": float(-np.sum(res["log_likelihood"][s])),
        }

    summary = {
        "iSub": subject_id,
        "source_csv": str(source_csv),
        "n_trials": int(len(df)),
        "n_unique_stimuli": int(df["stiID"].nunique()),
        "n_bits": args.n_bits,
        "max_order": max_order,
        "feature_encoding": feature_encoding,
        "label_encoding": label_encoding,
        "gamma0_fixed_to_zero": True,
        "memory_window": 1,
        "alpha_update": initial_model.update_rule,
        "observed_accuracy": float(df["correct"].mean()),
        "model_negative_log_likelihood": float(-np.sum(res["log_likelihood"])),
        "chance_negative_log_likelihood": float(len(df) * math.log(2)),
        "model_choice_agreement": float(df["model_pred_matches_choice"].mean()),
        "mean_model_p_correct": float(df["model_p_correct"].mean()),
        "mean_model_p_observed_choice": float(df["model_p_observed_choice"].mean()),
        "task_relevant_feature": task_feature,
        "task_relevant_feature_rank_by_abs_separation": task_rank,
        "rule_match_x2x4": float(df["rule_match_x2x4"].mean()) if "rule_match_x2x4" in df else None,
        "fit_info": fit_info,
        "initial_parameters": {"alpha_pos": initial_model.alpha_pos.tolist(),
                               "alpha_neg": initial_model.alpha_neg.tolist(),
                               "beta": initial_model.beta, "eta": initial_model.eta,
                               "gamma": initial_model.gamma},
        "future_prediction_metrics": future,
    }
    return {
        "subject_id": subject_id,
        "summary": summary,
        "trial_predictions": df,
        "formula_steps": formula_df,
        "final_feature_weights": feat_df,
        "alpha_pos_wide": make_wide_df(df, alpha_pos_trials, feature_names),
        "alpha_neg_wide": make_wide_df(df, alpha_neg_trials, feature_names),
        "alpha_diff_wide": make_wide_df(df, alpha_diff_trials, feature_names),
        "alpha_long": make_alpha_long_df(df, feature_names, alpha_pos_trials, alpha_neg_trials, alpha_diff_trials),
        "feature_names": feature_names,
        "trial_values": trial_values,
        "alpha_diff_trials": alpha_diff_trials,
        "target_feature": target_feature,
    }


def write_combined_outputs(runs: Sequence[Dict[str, Any]], outdir: Path, source_csv: Path, args: argparse.Namespace) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=False)
    paths = {
        "trial_predictions_csv": outdir / "trial_predictions.csv",
        "final_feature_weights_csv": outdir / "final_feature_weights.csv",
        "formula_steps_csv": outdir / args.formula_csv_name,
        "alpha_pos_trajectory_wide_csv": outdir / "alpha_pos_trajectory_wide.csv",
        "alpha_neg_trajectory_wide_csv": outdir / "alpha_neg_trajectory_wide.csv",
        "alpha_diff_trajectory_wide_csv": outdir / "alpha_diff_trajectory_wide.csv",
        "alpha_trajectory_long_csv": outdir / "alpha_trajectory_long.csv",
        "learning_curve_png": outdir / "learning_curve.png",
        "model_alpha_diff_over_trials_png": outdir / "model_alpha_diff_over_trials.png",
        "summary_json": outdir / "summary.json",
    }

    concat_run_df(runs, "trial_predictions").to_csv(paths["trial_predictions_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "final_feature_weights").to_csv(paths["final_feature_weights_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "formula_steps").to_csv(paths["formula_steps_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_pos_wide").to_csv(paths["alpha_pos_trajectory_wide_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_neg_wide").to_csv(paths["alpha_neg_trajectory_wide_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_diff_wide").to_csv(paths["alpha_diff_trajectory_wide_csv"], index=False, encoding="utf-8-sig")
    concat_run_df(runs, "alpha_long").to_csv(paths["alpha_trajectory_long_csv"], index=False, encoding="utf-8-sig")

    target_feature = runs[0]["target_feature"]
    plot_subject_learning_curves(runs, paths["learning_curve_png"])
    plot_subject_feature_grid(
        runs,
        "alpha_diff_trials",
        target_feature,
        "alpha_pos - alpha_neg",
        "Model feature weights over trials",
        paths["model_alpha_diff_over_trials_png"],
    )

    summary = {
        "source_csv": str(source_csv),
        "fit_subjects": [run["subject_id"] for run in runs],
        "n_subjects": len(runs),
        "n_bits": args.n_bits,
        "max_order": args.max_order if args.max_order is not None else args.n_bits,
        "optimizer_config": {
            "fit": args.fit,
            "population_size": args.population_size,
            "generations": args.generations,
            "anneal_steps": args.anneal_steps,
            "fit_first_n": args.fit_first_n,
            "seed": args.seed,
        },
        "runs": [run["summary"] for run in runs],
        "outputs": {key: str(path) for key, path in paths.items() if key != "summary_json"},
    }
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary
