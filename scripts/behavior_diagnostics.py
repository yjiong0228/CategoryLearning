"""Behavior-only diagnostics for category-learning subjects.

This script intentionally does not import Bayesian_state model code.  It reads
trial-level behavior data and summarizes patterns that can motivate model
extensions: below-chance episodes, perseveration, win-stay/lose-stay behavior,
and simple stimulus-rule probes that explain observed choices.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SORT_COLUMNS = ("iSession", "iBlock", "iTrial")
REQUIRED_COLUMNS = ("iSub", "condition", "choice", "feedback")


@dataclass(frozen=True)
class RuleProbe:
    name: str
    predictions: np.ndarray


def _parse_subjects(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out: list[int] = []
    for value in values:
        if "-" in value:
            lo, hi = value.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(value))
    return sorted(set(out))


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive.")
    for idx in range(arr.shape[0]):
        start = max(0, idx - window + 1)
        segment = arr[start : idx + 1]
        finite = segment[np.isfinite(segment)]
        if finite.size:
            out[idx] = float(np.mean(finite))
    return out


def _find_episodes(
    smooth_acc: np.ndarray,
    *,
    threshold: float,
    min_duration: int,
) -> list[tuple[int, int]]:
    below = np.asarray(smooth_acc, dtype=float) < float(threshold)
    episodes: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(below):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_duration:
                episodes.append((start, idx - 1))
            start = None
    if start is not None and len(below) - start >= min_duration:
        episodes.append((start, len(below) - 1))
    return episodes


def _safe_rate(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _stay_rates(choice: np.ndarray, feedback: np.ndarray) -> dict[str, float]:
    choice = np.asarray(choice, dtype=float)
    feedback = np.asarray(feedback, dtype=float)
    if choice.size < 2:
        return {
            "stay_rate": float("nan"),
            "win_stay_rate": float("nan"),
            "lose_stay_rate": float("nan"),
        }
    valid_prev = np.isfinite(choice[:-1]) & np.isfinite(choice[1:]) & np.isfinite(feedback[:-1])
    stay = choice[1:] == choice[:-1]
    win = valid_prev & (feedback[:-1] >= 1.0)
    lose = valid_prev & (feedback[:-1] < 1.0)
    return {
        "stay_rate": _safe_rate(stay[valid_prev]),
        "win_stay_rate": _safe_rate(stay[win]),
        "lose_stay_rate": _safe_rate(stay[lose]),
    }


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    idx = 1
    while f"feature{idx}" in frame.columns:
        cols.append(f"feature{idx}")
        idx += 1
    return cols


def _build_rule_probes(frame: pd.DataFrame, n_cats: int) -> list[RuleProbe]:
    n = len(frame)
    choices = sorted(int(x) for x in frame["choice"].dropna().unique())
    if not choices:
        choices = list(range(1, n_cats + 1))
    probes = [RuleProbe(f"constant_{choice}", np.full(n, choice, dtype=int)) for choice in choices]

    if n_cats != 2:
        return probes

    for col in _feature_columns(frame):
        values = frame[col].to_numpy(dtype=float)
        for high_choice in (1, 2):
            low_choice = 1 if high_choice == 2 else 2
            pred = np.where(values >= 0.5, high_choice, low_choice).astype(int)
            probes.append(RuleProbe(f"{col}_high->{high_choice}", pred))
            pred_inv = np.where(values < 0.5, high_choice, low_choice).astype(int)
            probes.append(RuleProbe(f"{col}_low->{high_choice}", pred_inv))
    return probes


def _window_rule_fit(
    frame: pd.DataFrame,
    *,
    window: int,
    step: int,
    n_cats: int,
) -> pd.DataFrame:
    probes = _build_rule_probes(frame, n_cats)
    choices = frame["choice"].to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    for start in range(0, max(1, len(frame) - window + 1), step):
        end = min(len(frame), start + window)
        if end - start < max(4, min(window, 8)):
            continue
        best_name = ""
        best_acc = -1.0
        for probe in probes:
            score = float(np.mean(probe.predictions[start:end] == choices[start:end]))
            if score > best_acc:
                best_acc = score
                best_name = probe.name
        rows.append(
            {
                "window_start": start + 1,
                "window_end": end,
                "window_center": (start + end + 1) / 2.0,
                "best_rule": best_name,
                "best_rule_choice_acc": best_acc,
            }
        )
    return pd.DataFrame(rows)


def _best_rule_for_labels(
    frame: pd.DataFrame,
    labels: np.ndarray,
    *,
    n_cats: int,
) -> tuple[str, float]:
    probes = _build_rule_probes(frame, n_cats)
    labels = np.asarray(labels, dtype=int)
    best_name = ""
    best_acc = -1.0
    for probe in probes:
        score = float(np.mean(probe.predictions == labels))
        if score > best_acc:
            best_acc = score
            best_name = probe.name
    return best_name, best_acc


def _episode_rows(
    subject: int,
    frame: pd.DataFrame,
    smooth_acc: np.ndarray,
    episodes: Iterable[tuple[int, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    choices = frame["choice"].to_numpy(dtype=int)
    categories = (
        frame["category"].to_numpy(dtype=int)
        if "category" in frame.columns
        else np.full(len(frame), -1, dtype=int)
    )
    feedback = frame["feedback"].to_numpy(dtype=float)
    for ep_idx, (start, end) in enumerate(episodes, start=1):
        sl = slice(start, end + 1)
        wrong = feedback[sl] < 1.0
        episode_choices = choices[sl]
        dominant_choice_rate = float(pd.Series(episode_choices).value_counts(normalize=True).iloc[0])
        reverse_rate = float(np.mean(choices[sl] != categories[sl])) if np.all(categories[sl] > 0) else float("nan")
        stay = choices[start + 1 : end + 1] == choices[start:end] if end > start else np.asarray([], dtype=bool)
        rows.append(
            {
                "iSub": subject,
                "episode_index": ep_idx,
                "start_trial": start + 1,
                "end_trial": end + 1,
                "duration": end - start + 1,
                "min_smooth_acc": float(np.nanmin(smooth_acc[sl])),
                "mean_feedback": float(np.nanmean(feedback[sl])),
                "wrong_rate": float(np.mean(wrong)),
                "dominant_choice_rate": dominant_choice_rate,
                "reverse_category_rate": reverse_rate,
                "within_episode_stay_rate": _safe_rate(stay),
            }
        )
    return rows


def _switch_count(values: pd.Series) -> int:
    vals = [str(x) for x in values.dropna().tolist()]
    if len(vals) < 2:
        return 0
    return int(sum(1 for a, b in zip(vals[:-1], vals[1:]) if a != b))


def _classify_subject(row: pd.Series) -> str:
    if row["below_chance_total_duration"] <= 0:
        return "no_below_chance_episode"
    if row["episode_dominant_choice_rate_max"] >= 0.85:
        return "choice_bias_or_perseveration"
    if row["lose_stay_rate"] >= 0.60:
        return "post_error_perseveration"
    if (
        not bool(row.get("best_rule_matches_target", False))
        and row["best_rule_stability"] >= 0.60
        and row["best_rule_mean_choice_acc"] >= 0.65
    ):
        return "stable_wrong_or_dimension_rule"
    if bool(row.get("best_rule_matches_target", False)):
        return "transient_below_chance_target_rule"
    return "mixed_or_noisy_below_chance"


def analyze_subject(
    subject: int,
    frame: pd.DataFrame,
    *,
    window: int,
    episode_threshold: float,
    min_duration: int,
    rule_window: int,
    rule_step: int,
    out_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    sort_cols = [col for col in SORT_COLUMNS if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    condition = int(frame["condition"].iloc[0]) if "condition" in frame.columns else -1
    n_cats = int(max(frame["choice"].max(), frame.get("category", frame["choice"]).max()))
    feedback = frame["feedback"].to_numpy(dtype=float)
    smooth_acc = _rolling_mean(feedback, window)
    episodes = _find_episodes(
        smooth_acc,
        threshold=episode_threshold,
        min_duration=min_duration,
    )
    ep_rows = _episode_rows(subject, frame, smooth_acc, episodes)
    rule_df = _window_rule_fit(frame, window=rule_window, step=rule_step, n_cats=n_cats)
    if not rule_df.empty:
        top_rule = str(rule_df["best_rule"].mode().iloc[0])
        top_rule_rate = float(np.mean(rule_df["best_rule"] == top_rule))
        rule_acc = float(rule_df["best_rule_choice_acc"].mean())
        rule_switches = _switch_count(rule_df["best_rule"])
    else:
        top_rule = ""
        top_rule_rate = float("nan")
        rule_acc = float("nan")
        rule_switches = 0

    stay = _stay_rates(frame["choice"].to_numpy(dtype=int), feedback)
    choices = frame["choice"].to_numpy(dtype=int)
    categories = (
        frame["category"].to_numpy(dtype=int)
        if "category" in frame.columns
        else np.full(len(frame), -1, dtype=int)
    )
    if np.all(categories > 0):
        target_rule, target_rule_acc = _best_rule_for_labels(frame, categories, n_cats=n_cats)
    else:
        target_rule, target_rule_acc = "", float("nan")
    choice_rule, choice_rule_acc = _best_rule_for_labels(frame, choices, n_cats=n_cats)
    valid_category = np.all(categories > 0)
    summary: dict[str, object] = {
        "iSub": subject,
        "condition": condition,
        "n_trials": int(len(frame)),
        "mean_acc": float(np.nanmean(feedback)),
        "min_smooth_acc": float(np.nanmin(smooth_acc)),
        "below_chance_episode_count": int(len(ep_rows)),
        "below_chance_total_duration": int(sum(row["duration"] for row in ep_rows)),
        "below_chance_first_start": int(ep_rows[0]["start_trial"]) if ep_rows else "",
        "episode_dominant_choice_rate_max": (
            float(max(row["dominant_choice_rate"] for row in ep_rows)) if ep_rows else float("nan")
        ),
        "overall_reverse_category_rate": (
            float(np.mean(choices != categories)) if valid_category else float("nan")
        ),
        "best_rule_mode": top_rule,
        "best_rule_stability": top_rule_rate,
        "best_rule_mean_choice_acc": rule_acc,
        "best_rule_switch_count": rule_switches,
        "full_choice_rule": choice_rule,
        "full_choice_rule_acc": choice_rule_acc,
        "target_rule": target_rule,
        "target_rule_acc": target_rule_acc,
        "best_rule_matches_target": bool(top_rule == target_rule) if target_rule else False,
        **stay,
    }
    summary["recommended_extension"] = _classify_subject(pd.Series(summary))

    subject_dir = out_dir / "plots"
    subject_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    x = np.arange(1, len(frame) + 1)
    axes[0].plot(x, feedback, color="#9ca3af", linewidth=0.8, alpha=0.5, label="trial feedback")
    axes[0].plot(x, smooth_acc, color="#2563eb", linewidth=2.0, label=f"rolling acc w={window}")
    axes[0].axhline(episode_threshold, color="#dc2626", linestyle="--", linewidth=1.0, label="below threshold")
    for start, end in episodes:
        axes[0].axvspan(start + 1, end + 1, color="#fecaca", alpha=0.45)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Subject {subject} behavior diagnostics")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].step(x, frame["choice"].to_numpy(dtype=int), where="mid", color="#111827", linewidth=1.0, label="choice")
    if "category" in frame.columns:
        axes[1].step(x, frame["category"].to_numpy(dtype=int), where="mid", color="#16a34a", linewidth=1.0, alpha=0.7, label="category")
    axes[1].set_ylabel("Category / choice")
    axes[1].set_xlabel("Trial")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(subject_dir / f"subject_{subject}_behavior.png", dpi=150)
    plt.close(fig)

    return summary, pd.DataFrame(ep_rows), rule_df.assign(iSub=subject)


def write_markdown_report(
    out_path: Path,
    summary_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    *,
    window: int,
    threshold: float,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        cols = [str(col) for col in frame.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for col in frame.columns:
                value = row[col]
                text = "" if pd.isna(value) else str(value)
                values.append(text.replace("|", "\\|").replace("\n", " "))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    below = summary_df[summary_df["below_chance_episode_count"] > 0].copy()
    lines: list[str] = []
    lines.append("# Behavior Diagnostics Report")
    lines.append("")
    lines.append(f"- Rolling window: `{window}`")
    lines.append(f"- Below-chance threshold: `{threshold:.3f}`")
    lines.append(f"- Subjects analyzed: `{len(summary_df)}`")
    lines.append(f"- Subjects with below-chance episodes: `{len(below)}`")
    lines.append("")
    lines.append("## Subject Summary")
    cols = [
        "iSub",
        "n_trials",
        "mean_acc",
        "min_smooth_acc",
        "below_chance_episode_count",
        "below_chance_total_duration",
        "lose_stay_rate",
        "best_rule_mode",
        "best_rule_stability",
        "best_rule_mean_choice_acc",
        "target_rule",
        "best_rule_matches_target",
        "recommended_extension",
    ]
    display = summary_df[cols].copy()
    for col in ("mean_acc", "min_smooth_acc", "lose_stay_rate", "best_rule_stability", "best_rule_mean_choice_acc"):
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    lines.append(markdown_table(display))
    lines.append("")
    lines.append("## Below-Chance Episodes")
    if episode_df.empty:
        lines.append("No below-chance episodes found.")
    else:
        ep_cols = [
            "iSub",
            "episode_index",
            "start_trial",
            "end_trial",
            "duration",
            "min_smooth_acc",
            "dominant_choice_rate",
            "within_episode_stay_rate",
        ]
        ep_display = episode_df[ep_cols].copy()
        for col in ("min_smooth_acc", "dominant_choice_rate", "within_episode_stay_rate"):
            ep_display[col] = ep_display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        lines.append(markdown_table(ep_display))
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `choice_bias_or_perseveration`: below-chance periods are dominated by one repeated choice.")
    lines.append("- `post_error_perseveration`: high lose-stay rate suggests repeating choices after negative feedback.")
    lines.append("- `stable_wrong_or_dimension_rule`: a simple stimulus rule explains observed choices across many windows.")
    lines.append("- `mixed_or_noisy_below_chance`: below-chance behavior exists but is not captured by these simple probes.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/Task2_processed.csv", help="Trial-level CSV.")
    parser.add_argument("--condition", type=int, default=1)
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject ids or ranges like 101-132.")
    parser.add_argument("--window", type=int, default=16, help="Rolling accuracy window.")
    parser.add_argument("--below-margin", type=float, default=0.05, help="Chance minus margin threshold.")
    parser.add_argument("--min-duration", type=int, default=8)
    parser.add_argument("--rule-window", type=int, default=24)
    parser.add_argument("--rule-step", type=int, default=4)
    parser.add_argument("--output-dir", default="results/behavior_diagnostics/cond1")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")

    df = df[df["condition"] == int(args.condition)].copy()
    subjects = _parse_subjects(args.subjects)
    if subjects is not None:
        df = df[df["iSub"].isin(subjects)].copy()
    if df.empty:
        raise ValueError("No rows remain after condition/subject filtering.")

    n_cats = int(max(df["choice"].max(), df.get("category", df["choice"]).max()))
    chance = 1.0 / float(max(1, n_cats))
    threshold = chance - float(args.below_margin)
    if threshold < 0.0:
        threshold = 0.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    episode_frames: list[pd.DataFrame] = []
    rule_frames: list[pd.DataFrame] = []
    for subject, subject_df in df.groupby("iSub", sort=True):
        summary, episodes, rules = analyze_subject(
            int(subject),
            subject_df,
            window=int(args.window),
            episode_threshold=threshold,
            min_duration=int(args.min_duration),
            rule_window=int(args.rule_window),
            rule_step=int(args.rule_step),
            out_dir=out_dir,
        )
        summaries.append(summary)
        if not episodes.empty:
            episode_frames.append(episodes)
        if not rules.empty:
            rule_frames.append(rules)

    summary_df = pd.DataFrame(summaries).sort_values("iSub")
    episode_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    rule_df = pd.concat(rule_frames, ignore_index=True) if rule_frames else pd.DataFrame()

    summary_df.to_csv(out_dir / "subject_behavior_summary.csv", index=False)
    episode_df.to_csv(out_dir / "below_chance_episodes.csv", index=False)
    rule_df.to_csv(out_dir / "window_rule_fits.csv", index=False)
    write_markdown_report(
        out_dir / "behavior_diagnostics_report.md",
        summary_df,
        episode_df,
        window=int(args.window),
        threshold=threshold,
    )
    metadata = {
        "data": str(data_path),
        "condition": int(args.condition),
        "subjects": subjects,
        "window": int(args.window),
        "chance": chance,
        "below_threshold": threshold,
        "min_duration": int(args.min_duration),
        "rule_window": int(args.rule_window),
        "rule_step": int(args.rule_step),
        "n_subjects": int(len(summary_df)),
        "n_below_chance_subjects": int((summary_df["below_chance_episode_count"] > 0).sum()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
