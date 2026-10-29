"""Exports for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any
import numpy as np
import pandas as pd
from .data import id_columns

Array = np.ndarray


def save_alpha_trajectories(
    outdir: Path,
    df: pd.DataFrame,
    feature_names: Sequence[str],
    alpha_pos: Array,
    alpha_neg: Array,
    alpha_diff: Array,
) -> Dict[str, str]:
    id_cols = {
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), ""),
        "iTrial": df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1),
        "stiID": df["stiID"].to_numpy(),
    }
    wide_paths = {}
    for key, arr in [
        ("alpha_pos", alpha_pos),
        ("alpha_neg", alpha_neg),
        ("alpha_diff", alpha_diff),
    ]:
        wide = pd.DataFrame(arr, columns=feature_names)
        for col, values in reversed(list(id_cols.items())):
            wide.insert(0, col, values)
        path = outdir / f"{key}_trajectory_wide.csv"
        wide.to_csv(path, index=False, encoding="utf-8-sig")
        wide_paths[f"{key}_trajectory_wide_csv"] = str(path)

    rows = []
    for t in range(len(df)):
        for j, feature in enumerate(feature_names):
            rows.append({
                "iSub": id_cols["iSub"][t],
                "iTrial": id_cols["iTrial"][t],
                "stiID": id_cols["stiID"][t],
                "feature": feature,
                "order": len(feature) - 1,
                "alpha_pos": alpha_pos[t, j],
                "alpha_neg": alpha_neg[t, j],
                "alpha_pos_minus_alpha_neg": alpha_diff[t, j],
            })
    long_path = outdir / "alpha_trajectory_long.csv"
    pd.DataFrame(rows).to_csv(long_path, index=False, encoding="utf-8-sig")
    wide_paths["alpha_trajectory_long_csv"] = str(long_path)
    return wide_paths


def make_wide_df(df: pd.DataFrame, values: Array, feature_names: Sequence[str]) -> pd.DataFrame:
    wide = pd.DataFrame(values, columns=feature_names)
    for col, data in reversed(list(id_columns(df).items())):
        wide.insert(0, col, data)
    return wide


def make_alpha_long_df(df: pd.DataFrame, feature_names: Sequence[str], alpha_pos: Array, alpha_neg: Array, alpha_diff: Array) -> pd.DataFrame:
    ids = id_columns(df)
    rows = []
    for t in range(len(df)):
        for j, feature in enumerate(feature_names):
            rows.append({
                "iSub": ids["iSub"][t],
                "iSession": ids["iSession"][t],
                "iTrial": ids["iTrial"][t],
                "stiID": ids["stiID"][t],
                "feature": feature,
                "order": len(feature) - 1,
                "alpha_pos": alpha_pos[t, j],
                "alpha_neg": alpha_neg[t, j],
                "alpha_pos_minus_alpha_neg": alpha_diff[t, j],
            })
    return pd.DataFrame(rows)


def concat_run_df(runs: Sequence[Dict[str, Any]], key: str) -> pd.DataFrame:
    return pd.concat([run[key] for run in runs], ignore_index=True)
