"""Legacy oral for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import id_columns

Array = np.ndarray


def model_feature_to_csv_column(name: str) -> str:
    if not name.startswith("x"):
        raise ValueError(f"Unexpected feature name: {name}")
    return "*".join(f"X{idx}" for idx in name[1:])


def oral_feature_matrix(df: pd.DataFrame, feature_names: Sequence[str]) -> Array:
    cols = [model_feature_to_csv_column(name) for name in feature_names]
    values = []
    for col in cols:
        if col in df.columns:
            values.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        else:
            values.append(np.full(len(df), np.nan))
    return np.column_stack(values)


def save_oral_trajectories(
    outdir: Path,
    df: pd.DataFrame,
    feature_names: Sequence[str],
    oral_weights: Array,
) -> Dict[str, str]:
    id_data = {
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), ""),
        "iTrial": df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1),
        "stiID": df["stiID"].to_numpy(),
    }
    wide = pd.DataFrame(oral_weights, columns=feature_names)
    for col, values in reversed(list(id_data.items())):
        wide.insert(0, col, values)
    wide_path = outdir / "oral_feature_weights_wide.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    rows = []
    for t in range(len(df)):
        for j, feature in enumerate(feature_names):
            rows.append({
                "iSub": id_data["iSub"][t],
                "iTrial": id_data["iTrial"][t],
                "stiID": id_data["stiID"][t],
                "feature": feature,
                "order": len(feature) - 1,
                "oral_weight": oral_weights[t, j],
            })
    long_path = outdir / "oral_feature_weights_long.csv"
    pd.DataFrame(rows).to_csv(long_path, index=False, encoding="utf-8-sig")
    return {
        "oral_feature_weights_wide_csv": str(wide_path),
        "oral_feature_weights_long_csv": str(long_path),
    }


def row_cosine(a: Array, b: Array) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    aa, bb = a[mask], b[mask]
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom == 0:
        return np.nan
    return float(np.dot(aa, bb) / denom)


def row_pearson(a: Array, b: Array) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return np.nan
    aa, bb = a[mask], b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def save_similarity_outputs(
    outdir: Path,
    df: pd.DataFrame,
    trials: Array,
    model_weights: Array,
    oral_weights: Array,
    feature_names: Sequence[str],
    target_feature: str,
) -> Dict[str, str]:
    cosine = np.asarray([row_cosine(m, o) for m, o in zip(model_weights, oral_weights)])
    pearson = np.asarray([row_pearson(m, o) for m, o in zip(model_weights, oral_weights)])
    sim_df = pd.DataFrame({
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else "",
        "iTrial": trials,
        "stiID": df["stiID"].to_numpy(),
        "cosine_similarity": cosine,
        "pearson_correlation": pearson,
    })
    sim_path = outdir / "model_oral_similarity.csv"
    sim_df.to_csv(sim_path, index=False, encoding="utf-8-sig")

    sim_png = outdir / "model_oral_similarity.png"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(trials, cosine, color="black", linewidth=1.8, label="cosine")
    ax.plot(trials, pearson, color="#1f77b4", linewidth=1.2, linestyle="--", label="Pearson r")
    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Similarity")
    ax.set_title("Similarity between model and oral feature-weight distributions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(sim_png, dpi=200)
    plt.close(fig)

    target_png = outdir / f"target_{target_feature}_model_vs_oral.png"
    target_idx = feature_names.index(target_feature)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(trials, model_weights[:, target_idx], color="#d62728", linewidth=2.6, label=f"model {target_feature}")
    ax.plot(trials, oral_weights[:, target_idx], color="black", linewidth=1.5, linestyle="--", label=f"oral {target_feature}")
    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Weight")
    ax.set_title(f"Target feature {target_feature}: model vs oral weights")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(target_png, dpi=200)
    plt.close(fig)

    return {
        "model_oral_similarity_csv": str(sim_path),
        "model_oral_similarity_png": str(sim_png),
        "target_feature_model_vs_oral_png": str(target_png),
    }


def make_oral_long_df(df: pd.DataFrame, feature_names: Sequence[str], oral_weights: Array) -> pd.DataFrame:
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
                "oral_weight": oral_weights[t, j],
            })
    return pd.DataFrame(rows)


def make_similarity_df(df: pd.DataFrame, model_weights: Array, oral_weights: Array) -> pd.DataFrame:
    ids = id_columns(df)
    return pd.DataFrame({
        "iSub": ids["iSub"],
        "iSession": ids["iSession"],
        "iTrial": ids["iTrial"],
        "stiID": ids["stiID"],
        "cosine_similarity": [row_cosine(m, o) for m, o in zip(model_weights, oral_weights)],
        "pearson_correlation": [row_pearson(m, o) for m, o in zip(model_weights, oral_weights)],
    })


def plot_subject_similarity_grid(runs: Sequence[Dict[str, Any]], path: Path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 3.4 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        sim = run["similarity"]
        ax.plot(sim["iTrial"], sim["cosine_similarity"], color="black", linewidth=1.8, label="cosine")
        ax.plot(sim["iTrial"], sim["pearson_correlation"], color="#1f77b4", linewidth=1.2, linestyle="--", label="Pearson r")
        ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel("Similarity")
        ax.legend(frameon=False, loc="lower right")
    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_target_grid(runs: Sequence[Dict[str, Any]], target_feature: str, path: Path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 3.4 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        idx = run["feature_names"].index(target_feature)
        trials = run["trial_values"]
        ax.plot(trials, run["alpha_diff_trials"][:, idx], color="#d62728", linewidth=2.6, label=f"model {target_feature}")
        ax.plot(trials, run["oral_weights"][:, idx], color="black", linewidth=1.5, linestyle="--", label=f"oral {target_feature}")
        ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel("Weight")
        ax.legend(frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
