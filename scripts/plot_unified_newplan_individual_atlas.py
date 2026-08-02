#!/usr/bin/env python3
"""Reconstruct and visualize subject-level fit and inferred cognitive states.

This diagnostic atlas is intentionally different from a group model-comparison
figure.  It shows every subject, preserves the temporal holdout boundary, and
labels latent trajectories as model-inferred states rather than observations.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.getLogger("fontTools").setLevel(logging.WARNING)

from src.Bayesian_state.utils.unified_newplan import rule_predictions  # noqa: E402


BASE = ROOT / "results/zhuran/unified_newplan"
CORE = BASE / "core_sobol512_20260802"
DYNAMIC = BASE / "dynamic_readout_20260802"
JOINT = BASE / "joint_dynamic_nr2_20260802"
DEFAULT_OUTPUT = BASE / "individual_atlas_20260802"
SCORE_EPS = 1e-7

COLORS = {
    "rule": "#15599C",
    "rule_light": "#B9D3EA",
    "feature": "#2A8C82",
    "feature_light": "#BFE1DC",
    "feedback": "#282C34",
    "readout": "#8A5A9E",
    "holdout": "#FFF2D6",
    "negative": "#B64342",
    "neutral": "#777777",
    "neutral_light": "#D8D8D8",
    "ink": "#20242A",
    "grid": "#E5E7EB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--page-dpi", type=int, default=220)
    return parser.parse_args()


def configure_style() -> str:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.exists():
        candidates = list(Path("/usr/share/fonts").rglob("NotoSansCJK*.ttc"))
        if not candidates:
            raise RuntimeError("No CJK font found")
        font_path = candidates[0]
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Arial", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return family


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            frame.to_csv(compressed, index=False)
    temporary.replace(path)


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    )


def first_sustained(values: np.ndarray, threshold: float, window: int = 16) -> float:
    hit = np.asarray(values, dtype=float) >= float(threshold)
    if len(hit) < window:
        return float(1 if hit.all() else np.nan)
    counts = np.convolve(hit.astype(int), np.ones(window, dtype=int), mode="valid")
    indices = np.flatnonzero(counts == window)
    return float(indices[0] + 1) if len(indices) else float("nan")


def compressed_path(top_hypothesis: np.ndarray, limit: int = 12) -> str:
    top = np.asarray(top_hypothesis, dtype=int)
    switches = top[np.r_[True, top[1:] != top[:-1]]]
    labels = [f"h{value}" for value in switches]
    if len(labels) <= limit:
        return " → ".join(labels)
    keep = limit // 2
    return " → ".join(labels[:keep] + ["…"] + labels[-keep:])


def block_count(session: np.ndarray, block: np.ndarray) -> int:
    return int(len(set(zip(session.astype(int).tolist(), block.astype(int).tolist()))))


def load_existing_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(JOINT / "subject_metrics.csv")
    metrics = metrics[
        metrics["segment"].eq("holdout")
        & metrics["model"].isin(["R0KT_GLOBAL", "NR2T_JOINT_INDIVIDUAL"])
    ]
    pivot = metrics.pivot(
        index=["subject_id", "condition"],
        columns="model",
        values=["nll_per_trial", "accuracy", "brier"],
    )
    pivot.columns = ["__".join(column) for column in pivot.columns]
    return pivot.reset_index()


def reconstruct_subject(subject_id: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    core_path = CORE / "subject_predictions" / f"subject_{subject_id}.npz"
    dynamic_path = DYNAMIC / "subject_predictions" / f"subject_{subject_id}.npz"
    joint_path = JOINT / "subject_predictions" / f"subject_{subject_id}.npz"
    q_path = CORE / "q_cache" / f"subject_{subject_id}.npz"
    for path in (core_path, dynamic_path, joint_path, q_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(core_path, allow_pickle=False) as core:
        condition = int(core["condition"])
        choices = core["choice"].astype(int)
        feedback = core["feedback"].astype(float)
        holdout = core["holdout_mask"].astype(bool)
        session = core["iSession"].astype(int)
        block = core["iBlock"].astype(int)
        within_trial = core["iTrial"].astype(int)
        stored_belief_entropy = core["belief_entropy_R0"].astype(float)
    with np.load(q_path, allow_pickle=False) as archive:
        q = archive["q"].astype(float)
    with np.load(dynamic_path, allow_pickle=False) as dynamic:
        probability_rule = dynamic["p_R0KT_GLOBAL"].astype(float)
        kappa = dynamic["kappa_R0KT_GLOBAL"].astype(float)
        practice = dynamic["practice"].astype(float)
    with np.load(joint_path, allow_pickle=False) as joint:
        probability_feature = joint["p_NR2T_JOINT_INDIVIDUAL"].astype(float)

    n_trials = len(choices)
    arrays = (feedback, holdout, session, block, within_trial, kappa, practice)
    if any(len(array) != n_trials for array in arrays):
        raise ValueError(f"Length mismatch for subject {subject_id}")
    if probability_rule.shape != probability_feature.shape or probability_rule.shape[0] != n_trials:
        raise ValueError(f"Probability shape mismatch for subject {subject_id}")

    inferred = rule_predictions(
        q,
        choices,
        feedback,
        condition,
        return_beliefs=True,
    )
    if inferred.beliefs is None:
        raise RuntimeError("Rule trajectory reconstruction did not return beliefs")
    if not np.allclose(inferred.belief_entropy, stored_belief_entropy, atol=2e-6, rtol=2e-6):
        raise ValueError(f"Reconstructed belief entropy differs from frozen output for {subject_id}")

    beliefs = inferred.beliefs
    n_hypotheses = beliefs.shape[1]
    n_categories = probability_rule.shape[1]
    target = 0 if condition == 1 else 42
    if target >= n_hypotheses:
        raise ValueError(f"Target hypothesis is outside the rule set for {subject_id}")
    target_mass = beliefs[:, target]
    alternative = beliefs.copy()
    alternative[:, target] = -np.inf
    strongest_alternative = np.max(alternative, axis=1)
    top_hypothesis = np.argmax(beliefs, axis=1)
    rule_certainty = np.clip(1.0 - inferred.belief_entropy / math.log(n_hypotheses), 0.0, 1.0)

    rows = np.arange(n_trials)
    observed_probability_rule = probability_rule[rows, choices]
    observed_probability_feature = probability_feature[rows, choices]
    rule_choice_entropy = -np.sum(
        np.clip(probability_rule, 1e-12, 1.0) * np.log(np.clip(probability_rule, 1e-12, 1.0)),
        axis=1,
    )
    feature_choice_entropy = -np.sum(
        np.clip(probability_feature, 1e-12, 1.0)
        * np.log(np.clip(probability_feature, 1e-12, 1.0)),
        axis=1,
    )
    rule_choice_certainty = np.clip(1.0 - rule_choice_entropy / math.log(n_categories), 0.0, 1.0)
    feature_choice_certainty = np.clip(1.0 - feature_choice_entropy / math.log(n_categories), 0.0, 1.0)
    rule_argmax_hit = np.argmax(probability_rule, axis=1) == choices
    feature_argmax_hit = np.argmax(probability_feature, axis=1) == choices
    train_rows = np.flatnonzero(~holdout)
    holdout_rows = np.flatnonzero(holdout)
    if not len(train_rows) or not len(holdout_rows) or holdout_rows[0] != train_rows[-1] + 1:
        raise ValueError(f"Holdout is not a contiguous suffix for {subject_id}")

    frame = pd.DataFrame(
        {
            "subject_id": subject_id,
            "condition": condition,
            "trial": rows + 1,
            "session": session,
            "block": block,
            "within_experiment_trial": within_trial,
            "segment": np.where(holdout, "holdout", "train"),
            "choice": choices + 1,
            "feedback": feedback,
            "target_hypothesis": target,
            "target_rule_probability": target_mass,
            "strongest_alternative_probability": strongest_alternative,
            "top_hypothesis": top_hypothesis,
            "rule_state_certainty": rule_certainty,
            "rule_readout_kappa": kappa,
            "practice": practice,
            "rule_choice_certainty": rule_choice_certainty,
            "feature_choice_certainty": feature_choice_certainty,
            "observed_choice_probability_rule": observed_probability_rule,
            "observed_choice_probability_feature": observed_probability_feature,
            "rule_argmax_matches_choice": rule_argmax_hit,
            "feature_argmax_matches_choice": feature_argmax_hit,
        }
    )

    hold = holdout
    # Match the frozen model-comparison scorer exactly.  The original fitting
    # pipeline floors vanishing observed-choice probabilities at 1e-7 before
    # taking logs; retaining that convention matters for one extreme trial in
    # subject 116.  The unfloored probabilities remain in the trial export and
    # are used for all plotted trajectories.
    nll_rule = float(-np.log(np.clip(observed_probability_rule[hold], SCORE_EPS, 1.0)).mean())
    nll_feature = float(
        -np.log(np.clip(observed_probability_feature[hold], SCORE_EPS, 1.0)).mean()
    )
    final_window = min(32, n_trials)
    initial_window = min(32, n_trials)
    fit_winner = "动态规则" if nll_rule < nll_feature else "特征 RL"
    summary = {
        "subject_id": subject_id,
        "condition": condition,
        "n_trials": n_trials,
        "n_train": int((~holdout).sum()),
        "n_holdout": int(holdout.sum()),
        "n_sessions": int(len(np.unique(session))),
        "n_blocks": block_count(session, block),
        "chance_probability": 1.0 / n_categories,
        "target_hypothesis": target,
        "fit_winner_lower_holdout_nll": fit_winner,
        "holdout_nll_per_trial_rule": nll_rule,
        "holdout_nll_per_trial_feature": nll_feature,
        "holdout_delta_nll_rule_advantage": nll_feature - nll_rule,
        "holdout_accuracy_rule": float(rule_argmax_hit[hold].mean()),
        "holdout_accuracy_feature": float(feature_argmax_hit[hold].mean()),
        "holdout_mean_observed_probability_rule": float(observed_probability_rule[hold].mean()),
        "holdout_mean_observed_probability_feature": float(observed_probability_feature[hold].mean()),
        "holdout_feedback_mean": float(feedback[hold].mean()),
        "target_rule_t50_sustained16": first_sustained(target_mass, 0.50),
        "target_rule_t90_sustained16": first_sustained(target_mass, 0.90),
        "target_rule_t99_sustained16": first_sustained(target_mass, 0.99),
        "target_rule_probability_initial32": float(target_mass[:initial_window].mean()),
        "target_rule_probability_final32": float(target_mass[-final_window:].mean()),
        "rule_state_certainty_initial32": float(rule_certainty[:initial_window].mean()),
        "rule_state_certainty_final32": float(rule_certainty[-final_window:].mean()),
        "top_hypothesis_switches": int(np.sum(top_hypothesis[1:] != top_hypothesis[:-1])),
        "top_hypothesis_path": compressed_path(top_hypothesis),
        "final_top_hypothesis": int(top_hypothesis[-1]),
        "final_top_is_target": bool(top_hypothesis[-1] == target),
        "rule_kappa_first": float(kappa[0]),
        "rule_kappa_train_end": float(kappa[train_rows[-1]]),
        "rule_kappa_holdout_end": float(kappa[-1]),
        "rule_choice_certainty_train_mean": float(rule_choice_certainty[~holdout].mean()),
        "rule_choice_certainty_holdout_mean": float(rule_choice_certainty[hold].mean()),
        "feature_choice_certainty_train_mean": float(feature_choice_certainty[~holdout].mean()),
        "feature_choice_certainty_holdout_mean": float(feature_choice_certainty[hold].mean()),
        "holdout_start_trial": int(holdout_rows[0] + 1),
    }
    return frame, summary


def verify_against_formal(summary: pd.DataFrame) -> None:
    formal = load_existing_metrics()
    merged = summary.merge(formal, on=["subject_id", "condition"], validate="one_to_one")
    checks = {
        "holdout_nll_per_trial_rule": "nll_per_trial__R0KT_GLOBAL",
        "holdout_nll_per_trial_feature": "nll_per_trial__NR2T_JOINT_INDIVIDUAL",
        "holdout_accuracy_rule": "accuracy__R0KT_GLOBAL",
        "holdout_accuracy_feature": "accuracy__NR2T_JOINT_INDIVIDUAL",
    }
    for reconstructed, frozen in checks.items():
        if not np.allclose(merged[reconstructed], merged[frozen], atol=2e-6, rtol=2e-6):
            difference = np.max(np.abs(merged[reconstructed] - merged[frozen]))
            raise ValueError(f"{reconstructed} differs from formal metric by {difference}")


def draw_boundary(ax: plt.Axes, holdout_start: int, n_trials: int) -> None:
    ax.axvspan(holdout_start, n_trials, color=COLORS["holdout"], alpha=0.72, zorder=-5)
    ax.axvline(holdout_start, color="#C28B2C", linestyle=(0, (3, 3)), linewidth=1.0)
    ax.text(
        holdout_start + 0.008 * n_trials,
        0.96,
        "留出段",
        transform=ax.get_xaxis_transform(),
        color="#9A6A16",
        fontsize=7,
        va="top",
    )


def draw_session_boundaries(ax: plt.Axes, frame: pd.DataFrame) -> None:
    sessions = frame["session"].to_numpy()
    changes = np.flatnonzero(sessions[1:] != sessions[:-1]) + 2
    for trial in changes:
        ax.axvline(trial, color="#B7BBC2", linewidth=0.65, alpha=0.75, zorder=-2)


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.65, zorder=-6)
    ax.tick_params(length=3, width=0.7, color="#777777")
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")


def summary_card(
    fig: plt.Figure,
    x: float,
    width: float,
    title: str,
    value: str,
    subtitle: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, 0.77),
        width,
        0.105,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        transform=fig.transFigure,
        facecolor="#F8FAFC",
        edgecolor=color,
        linewidth=1.1,
    )
    fig.add_artist(patch)
    fig.text(x + 0.012, 0.852, title, fontsize=7.2, color="#5B6470", va="top")
    fig.text(x + 0.012, 0.817, value, fontsize=12.5, fontweight="bold", color=color, va="top")
    fig.text(x + 0.012, 0.786, subtitle, fontsize=6.7, color="#5B6470", va="top")


def plot_subject_page(frame: pd.DataFrame, row: pd.Series) -> plt.Figure:
    subject_id = int(row.subject_id)
    condition = int(row.condition)
    n_trials = int(row.n_trials)
    holdout_start = int(row.holdout_start_trial)
    chance = float(row.chance_probability)
    fit_advantage = float(row.holdout_delta_nll_rule_advantage)
    winner_color = COLORS["rule"] if fit_advantage > 0 else COLORS["feature"]

    fig = plt.figure(figsize=(11.5, 8.2))
    fig.text(0.06, 0.958, f"被试 {subject_id}｜条件 {condition}", fontsize=19, fontweight="bold", color=COLORS["ink"], va="top")
    fig.text(
        0.06,
        0.915,
        f"{n_trials}试次 · {int(row.n_sessions)}个session · {int(row.n_blocks)}个block · 最后{int(row.n_holdout)}试次为留出检验",
        fontsize=8.5,
        color="#5B6470",
        va="top",
    )
    fig.text(
        0.94,
        0.955,
        "以下状态均为模型推断，不是直接测量",
        ha="right",
        va="top",
        fontsize=7.5,
        color=COLORS["negative"],
    )

    winner = str(row.fit_winner_lower_holdout_nll)
    path_display = str(row.top_hypothesis_path)
    if len(path_display) > 34:
        path_display = f"{path_display[:16]} … {path_display[-13:]}"
    summary_card(
        fig,
        0.06,
        0.16,
        "留出拟合更优模型",
        winner,
        f"规则优势 ΔNLL={fit_advantage:+.3f}/试次",
        winner_color,
    )
    summary_card(
        fig,
        0.235,
        0.16,
        "规则模型留出准确率",
        f"{100 * row.holdout_accuracy_rule:.1f}%",
        f"特征RL {100 * row.holdout_accuracy_feature:.1f}%",
        COLORS["rule"],
    )
    t90 = row.target_rule_t90_sustained16
    t90_text = "未达到" if pd.isna(t90) else f"第 {int(t90)} 试次"
    summary_card(
        fig,
        0.410,
        0.16,
        "目标规则持续超过90%",
        t90_text,
        f"目标规则 h{int(row.target_hypothesis)}",
        COLORS["rule"],
    )
    summary_card(
        fig,
        0.585,
        0.16,
        "规则影响强度 κ",
        f"{row.rule_kappa_first:.2f} → {row.rule_kappa_holdout_end:.2f}",
        "斜率由全体被试共享",
        COLORS["readout"],
    )
    summary_card(
        fig,
        0.760,
        0.18,
        "最可能规则路径",
        f"切换 {int(row.top_hypothesis_switches)} 次",
        path_display,
        COLORS["neutral"],
    )

    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.965,
        bottom=0.085,
        top=0.715,
        height_ratios=[1.0, 0.93],
        hspace=0.42,
        wspace=0.27,
    )
    trial = frame["trial"].to_numpy()
    window = min(32, max(8, n_trials // 20))

    ax_fit = fig.add_subplot(gs[0, :])
    ax_fit.plot(
        trial,
        rolling(frame["observed_choice_probability_rule"].to_numpy(), window),
        color=COLORS["rule"],
        linewidth=2.0,
        label="动态规则：给实际选择的概率",
    )
    ax_fit.plot(
        trial,
        rolling(frame["observed_choice_probability_feature"].to_numpy(), window),
        color=COLORS["feature"],
        linewidth=1.7,
        label="特征RL：给实际选择的概率",
    )
    ax_fit.plot(
        trial,
        rolling(frame["feedback"].to_numpy(), window),
        color=COLORS["feedback"],
        linewidth=1.25,
        alpha=0.82,
        label="实际近期平均反馈",
    )
    ax_fit.axhline(chance, color="#9A9A9A", linestyle=(0, (2, 3)), linewidth=0.8)
    draw_boundary(ax_fit, holdout_start, n_trials)
    draw_session_boundaries(ax_fit, frame)
    ax_fit.set_xlim(1, n_trials)
    ax_fit.set_ylim(-0.02, 1.02)
    ax_fit.set_ylabel("滚动平均（0–1）")
    ax_fit.set_xlabel("试次")
    ax_fit.set_title(f"a  模型是否跟得上此人的实际选择（{window}试次滚动平均）", loc="left", fontweight="bold")
    ax_fit.legend(loc="lower right", ncol=3)
    style_axis(ax_fit, "y")

    ax_state = fig.add_subplot(gs[1, 0])
    zoom_end = min(n_trials, 160)
    early = frame[frame["trial"] <= zoom_end]
    ax_state.fill_between(
        early["trial"].to_numpy(),
        0,
        early["target_rule_probability"].to_numpy(),
        color=COLORS["rule_light"],
        alpha=0.45,
        linewidth=0,
    )
    ax_state.plot(
        early["trial"],
        early["target_rule_probability"],
        color=COLORS["rule"],
        linewidth=2.0,
        label=f"目标规则 h{int(row.target_hypothesis)}",
    )
    ax_state.plot(
        early["trial"],
        early["strongest_alternative_probability"],
        color=COLORS["neutral"],
        linewidth=1.35,
        label="最强备选规则",
    )
    if not pd.isna(t90) and t90 <= zoom_end:
        ax_state.axvline(t90, color=COLORS["rule"], linestyle=(0, (3, 3)), linewidth=1.0)
        ax_state.text(t90 + 2, 0.51, f"持续>90%：试次{int(t90)}", rotation=90, fontsize=6.5, color=COLORS["rule"], va="center")
    ax_state.axhline(0.9, color="#AAAAAA", linestyle=(0, (2, 3)), linewidth=0.7)
    ax_state.set_xlim(1, zoom_end)
    ax_state.set_ylim(-0.02, 1.02)
    ax_state.set_xlabel("试次（放大学习早期）")
    ax_state.set_ylabel("规则后验概率")
    ax_state.set_title("b  模型认为此人何时锁定目标规则", loc="left", fontweight="bold")
    ax_state.legend(loc="lower right")
    style_axis(ax_state, "y")

    ax_control = fig.add_subplot(gs[1, 1])
    rule_certainty = rolling(frame["rule_choice_certainty"].to_numpy(), window)
    feature_certainty = rolling(frame["feature_choice_certainty"].to_numpy(), window)
    ax_control.plot(trial, rule_certainty, color=COLORS["rule"], lw=1.9, label="规则选择确定度")
    ax_control.plot(trial, feature_certainty, color=COLORS["feature"], lw=1.6, label="特征RL选择确定度")
    ax_control.set_xlim(1, n_trials)
    ax_control.set_ylim(-0.02, 1.02)
    ax_control.set_xlabel("试次")
    ax_control.set_ylabel("模型选择确定度（0–1）")
    draw_boundary(ax_control, holdout_start, n_trials)
    draw_session_boundaries(ax_control, frame)
    style_axis(ax_control, "y")
    ax_kappa = ax_control.twinx()
    ax_kappa.plot(
        trial,
        frame["rule_readout_kappa"],
        color=COLORS["readout"],
        lw=1.15,
        linestyle=(0, (4, 2)),
        label="规则影响强度 κ",
    )
    kappa_max = max(0.1, float(frame["rule_readout_kappa"].max()) * 1.12)
    ax_kappa.set_ylim(0, kappa_max)
    ax_kappa.set_ylabel("规则影响强度 κ", color=COLORS["readout"])
    ax_kappa.tick_params(axis="y", colors=COLORS["readout"], labelsize=7)
    ax_kappa.spines["top"].set_visible(False)
    ax_kappa.spines["right"].set_color(COLORS["readout"])
    handles1, labels1 = ax_control.get_legend_handles_labels()
    handles2, labels2 = ax_kappa.get_legend_handles_labels()
    ax_control.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    ax_control.set_title("c  从内部规则到外显选择：控制强度如何变化", loc="left", fontweight="bold")

    fig.text(
        0.075,
        0.025,
        "解释边界：规则后验由刺激知觉分布和实际反馈递推；κ的练习斜率跨被试共享，只有起始水平按被试拟合；特征RL个体参数恢复较弱，不宜当作稳定人格特质。",
        fontsize=6.8,
        color="#606873",
    )
    return fig


def interpolate_subject(frame: pd.DataFrame, column: str, n_points: int = 100) -> np.ndarray:
    x = np.linspace(0.0, 1.0, len(frame))
    target = np.linspace(0.0, 1.0, n_points)
    return np.interp(target, x, frame[column].to_numpy(dtype=float))


def save_overview(
    states: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> list[Path]:
    ordered = summary.sort_values(["condition", "subject_id"]).reset_index(drop=True)
    n_subjects = len(ordered)
    early_trials = 160
    target_matrix = np.full((n_subjects, early_trials), np.nan, dtype=float)
    control_matrix = np.empty((n_subjects, 100), dtype=float)
    for index, row in ordered.iterrows():
        subject = states[states["subject_id"].eq(int(row.subject_id))]
        values = subject["target_rule_probability"].to_numpy(dtype=float)
        target_matrix[index, : min(early_trials, len(values))] = values[:early_trials]
        control_matrix[index] = interpolate_subject(subject, "rule_choice_certainty") - interpolate_subject(subject, "feature_choice_certainty")

    fig = plt.figure(figsize=(14.2, 18.0))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.05, 1.55, 1.05, 1.55],
        left=0.095,
        right=0.96,
        bottom=0.065,
        top=0.91,
        wspace=0.20,
    )
    fig.suptitle("96名被试的拟合与模型推断状态图谱", fontsize=20, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.946,
        "每一行是一名被试；相同被试在四个面板中严格对齐",
        ha="center",
        fontsize=10,
        color="#5B6470",
    )
    y = np.arange(n_subjects)
    subject_labels = [str(int(value)) for value in ordered["subject_id"]]

    ax_a = fig.add_subplot(gs[0, 0])
    delta = ordered["holdout_delta_nll_rule_advantage"].to_numpy(dtype=float)
    colors = np.where(delta >= 0, COLORS["rule"], COLORS["feature"])
    ax_a.barh(y, delta, color=colors, height=0.76, linewidth=0)
    ax_a.axvline(0, color="#444444", linewidth=0.9)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(subject_labels, fontsize=5.4)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("留出 ΔNLL/试次\n←特征RL更好｜规则更好→")
    ax_a.set_title("a  每个人谁拟合更好", loc="left", fontweight="bold")
    style_axis(ax_a, "x")

    cmap_target = LinearSegmentedColormap.from_list(
        "target_mass", ["#F2F4F7", COLORS["rule_light"], COLORS["rule"]]
    )
    cmap_target.set_bad("#ECECEC")
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    image_target = ax_b.imshow(
        target_matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
        cmap=cmap_target,
        origin="upper",
    )
    ax_b.tick_params(axis="y", labelleft=False, left=False)
    ax_b.set_xlabel("实际试次（前160）")
    ax_b.set_title("b  目标规则概率如何形成", loc="left", fontweight="bold")
    ax_b.set_xticks([0, 39, 79, 119, 159])
    ax_b.set_xticklabels([1, 40, 80, 120, 160])
    colorbar_b = fig.colorbar(image_target, ax=ax_b, fraction=0.025, pad=0.015)
    colorbar_b.set_label("目标规则概率", fontsize=7)
    colorbar_b.ax.tick_params(labelsize=6)

    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)
    start = np.log10(np.clip(ordered["rule_kappa_first"].to_numpy(dtype=float), 1e-4, None))
    end = np.log10(np.clip(ordered["rule_kappa_holdout_end"].to_numpy(dtype=float), 1e-4, None))
    for yi, x0, x1 in zip(y, start, end):
        ax_c.plot([x0, x1], [yi, yi], color=COLORS["rule_light"], linewidth=1.0)
    ax_c.scatter(start, y, s=8, color=COLORS["neutral"], label="开始", zorder=3)
    ax_c.scatter(end, y, s=9, color=COLORS["readout"], label="留出末", zorder=3)
    ax_c.tick_params(axis="y", labelleft=False, left=False)
    ticks = np.array([-2, -1, 0, 1], dtype=float)
    ax_c.set_xticks(ticks)
    ax_c.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax_c.set_xlabel("规则影响强度 κ")
    ax_c.set_title("c  规则能否控制选择", loc="left", fontweight="bold")
    ax_c.legend(loc="lower right", markerscale=1.4)
    style_axis(ax_c, "x")

    ax_d = fig.add_subplot(gs[0, 3], sharey=ax_a)
    maximum = max(0.15, float(np.nanmax(np.abs(control_matrix))))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    cmap_control = LinearSegmentedColormap.from_list(
        "control_difference", [COLORS["feature"], "#F7F7F7", COLORS["rule"]]
    )
    image_control = ax_d.imshow(
        control_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_control,
        norm=norm,
        origin="upper",
    )
    ax_d.tick_params(axis="y", labelleft=False, left=False)
    ax_d.set_xticks([0, 24, 49, 74, 99])
    ax_d.set_xticklabels(["开始", "25%", "50%", "75%", "结束"])
    ax_d.set_xlabel("归一化个人实验进程")
    ax_d.set_title("d  哪套系统的选择更确定", loc="left", fontweight="bold")
    colorbar_d = fig.colorbar(image_control, ax=ax_d, fraction=0.025, pad=0.015)
    colorbar_d.set_label("规则确定度 − 特征RL确定度", fontsize=7)
    colorbar_d.ax.tick_params(labelsize=6)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        for boundary in (31.5, 63.5):
            ax.axhline(boundary, color="#111111", linewidth=1.0)
    for center, label in ((15.5, "条件1"), (47.5, "条件2"), (79.5, "条件3")):
        ax_a.text(
            -0.37,
            center,
            label,
            transform=ax_a.get_yaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#4A5260",
            clip_on=False,
        )
    fig.text(
        0.095,
        0.025,
        "蓝色ΔNLL表示动态规则模型留出拟合更好，绿色表示动态特征RL更好。b中的灰色仅表示该被试试次数不足160；d为模型选择确定度差，不是直接观测的心理量。",
        fontsize=7.5,
        color="#5B6470",
    )

    paths = []
    svg_path = output / "fig_individual_overview.svg"
    pdf_path = output / "fig_individual_overview.pdf"
    png_path = output / "fig_individual_overview.png"
    tiff_path = output / "fig_individual_overview.tiff"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    paths.extend([svg_path, pdf_path, png_path, tiff_path])
    plt.close(fig)
    return paths


def write_index(output: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# 逐被试拟合与认知状态索引",
        "",
        "先打开 `fig_individual_overview.png` 看96名被试全貌；需要查看某个人时，点击下表中的被试编号。完整96页矢量图册为 `all_subjects_atlas.pdf`。",
        "",
        "ΔNLL为“特征RL留出NLL − 动态规则留出NLL”，正值表示规则模型拟合更好。单个被试没有重复实验可计算可靠的个体显著性区间，因此这里只报告方向和大小，不写‘显著’。",
        "",
    ]
    for condition, group in summary.groupby("condition", sort=True):
        lines.extend(
            [
                f"## 条件 {int(condition)}",
                "",
                "| 被试 | 留出更优模型 | 规则准确率 | 特征RL准确率 | 规则优势 ΔNLL | 目标规则持续>90% | κ 开始→末段 |",
                "|:--|:--|--:|--:|--:|--:|--:|",
            ]
        )
        for row in group.sort_values("subject_id").itertuples(index=False):
            t90 = "未达到" if pd.isna(row.target_rule_t90_sustained16) else f"试次{int(row.target_rule_t90_sustained16)}"
            lines.append(
                f"| [{int(row.subject_id)}](subjects/subject_{int(row.subject_id)}.png) | "
                f"{row.fit_winner_lower_holdout_nll} | {100 * row.holdout_accuracy_rule:.1f}% | "
                f"{100 * row.holdout_accuracy_feature:.1f}% | {row.holdout_delta_nll_rule_advantage:+.3f} | "
                f"{t90} | {row.rule_kappa_first:.2f}→{row.rule_kappa_holdout_end:.2f} |"
            )
        lines.append("")
    (output / "INDIVIDUAL_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_guide(output: Path, summary: pd.DataFrame) -> None:
    n_rule = int((summary["holdout_delta_nll_rule_advantage"] > 0).sum())
    n_feature = int((summary["holdout_delta_nll_rule_advantage"] < 0).sum())
    medians = summary.groupby("condition")["target_rule_t90_sustained16"].median().to_dict()
    slowest = summary.loc[summary["target_rule_t90_sustained16"].idxmax()]
    condition_counts = (
        summary.assign(rule_better=summary["holdout_delta_nll_rule_advantage"] > 0)
        .groupby("condition")["rule_better"]
        .agg(["sum", "count"])
    )
    lines = [
        "# 逐被试结果：先读这份说明",
        "",
        "## 最重要的发现",
        "",
        f"1. 动态规则模型在 {n_rule}/96 名被试上具有更低的留出NLL，动态特征RL在 {n_feature}/96 名被试上更低。按条件分别为 "
        + "、".join(
            f"条件{int(condition)} {int(row['sum'])}/{int(row['count'])}"
            for condition, row in condition_counts.iterrows()
        )
        + "。这说明总体规则优势并不等于每个人都由规则模型更好地解释。",
        f"2. 规则后验最终在96名被试中全部集中到已知任务规则。目标规则持续超过90%的中位试次分别为：条件1第{medians[1]:.1f}、条件2第{medians[2]:.1f}、条件3第{medians[3]:.1f}试次。最慢的是被试{int(slowest.subject_id)}，在第{int(slowest.target_rule_t90_sustained16)}试次达到。",
        "3. 因此个体差异主要不在‘最终相信哪条规则’，而在规则知识能否稳定控制选择，以及规则模型相对特征学习模型的概率校准。",
        "4. 当前R0KT模型给所有人使用同一个练习斜率，只单独拟合起始读出水平。每个人都呈现相似的相对增长形状，这是模型结构带来的结果，不能误写成96条独立发现。",
        "",
        "## 每张被试页怎么看",
        "",
        "- 顶部卡片：留出拟合胜者、两模型准确率、目标规则形成时间、规则影响强度和最可能规则路径。拟合胜者按留出NLL（实际选择的概率校准）判定；它与只看第一名是否猜中的准确率可能方向不同。",
        "- a：模型给此人实际选择分配的概率。线越高表示模型越能跟上此人的选择；黄色区域是完全未参与参数拟合的留出段。",
        "- b：模型推断的目标规则概率与最强备选规则概率，仅放大学习早期。",
        "- c：规则系统和特征RL的选择确定度，以及规则影响强度κ。",
        "",
        "## 解释限制",
        "",
        "这些曲线是依据刺激、反馈和选择推断的隐状态，不是直接测量到的脑状态。尤其是特征RL的个体学习率恢复较弱，不能把参数排名解释成人格或稳定认知能力。单名被试只有一个留出区块，个体模型胜负没有独立置信区间。",
        "",
        "## 文件",
        "",
        "- `fig_individual_overview.png/.svg/.pdf/.tiff`：96名被试对齐总览。",
        "- `all_subjects_atlas.pdf`：96页逐被试矢量图册。",
        "- `subjects/subject_*.png`：每名被试的单独诊断页。",
        "- `INDIVIDUAL_INDEX.md`：可点击的逐被试结果表。",
        "- `subject_summary.csv`：逐被试摘要指标。",
        "- `trial_states.csv.gz`：62,720行逐试次状态与拟合概率。",
        "",
    ]
    (output / "READ_ME_FIRST_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subject_output = output / "subjects"
    subject_output.mkdir(exist_ok=True)
    font_family = configure_style()

    available = sorted(
        int(path.stem.split("_")[-1])
        for path in (CORE / "subject_predictions").glob("subject_*.npz")
    )
    if args.subjects:
        requested = {int(value.strip()) for value in args.subjects.split(",") if value.strip()}
        missing = requested - set(available)
        if missing:
            raise ValueError(f"Unknown subject IDs: {sorted(missing)}")
        subject_ids = sorted(requested)
    else:
        subject_ids = available
    if not args.subjects and len(subject_ids) != 96:
        raise ValueError(f"Expected 96 subjects, found {len(subject_ids)}")

    state_frames = []
    summaries = []
    for subject_id in subject_ids:
        frame, summary = reconstruct_subject(subject_id)
        state_frames.append(frame)
        summaries.append(summary)
    states = pd.concat(state_frames, ignore_index=True)
    summary = pd.DataFrame(summaries).sort_values(["condition", "subject_id"]).reset_index(drop=True)
    verify_against_formal(summary)
    if not summary["final_top_is_target"].all():
        raise ValueError("At least one subject does not end with the known task rule on top")

    atomic_csv(output / "subject_summary.csv", summary)
    atomic_csv_gzip(output / "trial_states.csv.gz", states)
    overview_paths = save_overview(states, summary, output)

    atlas_path = output / "all_subjects_atlas.pdf"
    with PdfPages(atlas_path) as pdf:
        for row in summary.itertuples(index=False):
            subject_id = int(row.subject_id)
            subject_frame = states[states["subject_id"].eq(subject_id)]
            row_series = pd.Series(row._asdict())
            figure = plot_subject_page(subject_frame, row_series)
            pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
            figure.savefig(
                subject_output / f"subject_{subject_id}.png",
                dpi=max(180, int(args.page_dpi)),
                bbox_inches="tight",
                pad_inches=0.08,
            )
            plt.close(figure)

    write_index(output, summary)
    write_guide(output, summary)
    manifest = {
        "result_type": "unified_newplan_subject_fit_state_atlas",
        "status": "complete",
        "backend": "python_matplotlib",
        "font_family": font_family,
        "n_subjects": int(len(summary)),
        "n_trial_rows": int(len(states)),
        "n_subject_png": int(len(list(subject_output.glob("subject_*.png")))),
        "atlas_pdf": atlas_path.name,
        "overview_exports": [path.name for path in overview_paths],
        "state_definition": (
            "model-inferred pre-feedback rule posterior plus fitted rule/feature choice readouts; "
            "not a direct neural or cognitive measurement"
        ),
        "target_hypothesis": {"condition1": 0, "condition2": 42, "condition3": 42},
        "individual_inference_boundary": (
            "single temporal holdout block per subject; no individual uncertainty interval; "
            "R0KT practice slope shared across all subjects"
        ),
        "input_manifest_sha256": {
            "core": sha256_file(CORE / "manifest.json"),
            "dynamic": sha256_file(DYNAMIC / "manifest.json"),
            "joint_nr2": sha256_file(JOINT / "manifest.json"),
        },
        "runtime_seconds": float(time.time() - started),
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(
        f"[done] wrote {len(summary)} subject pages, one atlas PDF, and an aligned overview to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
