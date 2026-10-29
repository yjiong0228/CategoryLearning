"""Plotting for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


def plot_feature_trajectories(
    trials: Array,
    values: Array,
    feature_names: Sequence[str],
    target_feature: str,
    ylabel: str,
    title: str,
    path: Path,
):
    fig, ax = plt.subplots(figsize=(11, 6))
    target_idx = feature_names.index(target_feature) if target_feature in feature_names else None
    for j, name in enumerate(feature_names):
        if j == target_idx:
            continue
        ax.plot(trials, values[:, j], color="0.75", linewidth=0.7, alpha=0.75)
    if target_idx is not None:
        ax.plot(trials, values[:, target_idx], color="#d62728", linewidth=2.8, label=target_feature)
    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if target_idx is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_learning_curves(runs: Sequence[Dict[str, Any]], path: Path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 3.2 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        df = run["trial_predictions"]
        trial_col = "iTrial" if "iTrial" in df.columns else "trial_index"
        ax.plot(df[trial_col], df["rolling_accuracy_16"], label="Observed rolling accuracy")
        ax.plot(df[trial_col], df["rolling_model_p_correct_16"], label="Model rolling P(correct)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel("Accuracy / probability")
        ax.legend(frameon=False, loc="lower right")
    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_feature_grid(
    runs: Sequence[Dict[str, Any]],
    value_key: str,
    target_feature: str,
    ylabel: str,
    title: str,
    path: Path,
):
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 3.6 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        values = run[value_key]
        feature_names = run["feature_names"]
        trials = run["trial_values"]
        target_idx = feature_names.index(target_feature)
        for j, _name in enumerate(feature_names):
            if j != target_idx:
                ax.plot(trials, values[:, j], color="0.75", linewidth=0.65, alpha=0.7)
        ax.plot(trials, values[:, target_idx], color="#d62728", linewidth=2.8, label=target_feature)
        ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.6)
        ax.set_title(f"Subject {run['subject_id']}")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("Trial")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
