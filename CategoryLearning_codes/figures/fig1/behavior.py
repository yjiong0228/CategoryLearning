"""Model-independent measurements; no source data are mutated."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["condition", "iSub", "iSession", "iBlock", "iTrial"]
FEATURE_PATTERNS = {"head": "头", "neck": "脖|颈", "tail": "尾", "leg": "腿|脚|肢"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep every observation and reconstruct chronological within-subject order."""
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate trial keys; cannot establish unique observations")
    if not frame.feedback.isin([0, .5, 1]).all():
        raise ValueError("Unexpected or missing feedback")
    out = frame.sort_values(KEYS, kind="stable").reset_index(drop=True).copy()
    out["source_row"] = frame.sort_values(KEYS, kind="stable").index.to_numpy() + 2
    out["trial"] = out.groupby(["condition", "iSub"]).cumcount() + 1
    out["correct"] = (out.feedback == 1).astype(int)
    return out


def adjacent_gain(values: np.ndarray, window: int) -> tuple[float, float]:
    """Maximum right-minus-left mean for adjacent complete windows; split after t."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 * window:
        return np.nan, np.nan
    if not np.isfinite(values).all():
        raise ValueError("Gain requires observed finite responses")
    sums = np.r_[0., np.cumsum(values)]
    splits = np.arange(window, len(values) - window + 1)
    gains = (sums[splits + window] - 2 * sums[splits] + sums[splits - window]) / window
    index = int(np.argmax(gains))
    return float(gains[index]), int(splits[index])


def explicit_features(text: object) -> frozenset[str]:
    """Literal anatomical mentions only; implicit references remain unclassified."""
    if not isinstance(text, str):
        return frozenset()
    return frozenset(name for name, pattern in FEATURE_PATTERNS.items() if re.search(pattern, text))


def task_categories(stimulus_categories: np.ndarray, condition: int) -> np.ndarray:
    """Raw stimulus tables retain four leaves even for the binary task.

    For every condition-1 record, raw behavior confirms leaves {1,2} -> 1 and
    {3,4} -> 2. Four-category conditions preserve the original leaf labels.
    """
    labels = np.asarray(stimulus_categories, dtype=int)
    return (labels + 1) // 2 if condition == 1 else labels.copy()


def validate_task_rule(frame: pd.DataFrame) -> None:
    """Verify normative feature roles against observations before oral evaluation."""
    binary = 1 + (frame.feature1 > .5).astype(int)
    four = np.where(frame.feature1 <= .5, 1 + (frame.feature2 > .5),
                    3 + (frame.feature3 > .5))
    expected = np.where(frame.condition == 1, binary, four)
    if not np.array_equal(expected, frame.category.to_numpy()):
        raise ValueError("Observed categories disagree with the assumed task rule")


def task_relative_reports(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe literal mentions relative to actual task features, not model rules."""
    out = frame.copy()
    anatomy = tuple(FEATURE_PATTERNS)
    mentions, required_sets, coverage, irrelevant = [], [], [], []
    for row in out.itertuples():
        feature_names = [getattr(row, f"feature{i}_name") for i in range(1, 5)]
        if set(feature_names) != set(anatomy):
            raise ValueError("Individual feature map must contain the four anatomical features")
        if row.condition == 1:
            required, unused = {feature_names[0]}, set(feature_names[1:])
        else:
            required = {feature_names[0], feature_names[1 if row.feature1 <= .5 else 2]}
            unused = {feature_names[3]}
        required_sets.append("|".join(sorted(required)))
        if row.report_recognized:
            named = set(row.feature_set.split("|"))
            mentions.append([float(part in named) for part in anatomy])
            coverage.append(len(named & required) / len(required))
            irrelevant.append(len(named & unused) / len(unused))
        else:
            mentions.append([np.nan]*4)
            coverage.append(np.nan)
            irrelevant.append(np.nan)
    for i, part in enumerate(anatomy):
        out[f"mention_{part}"] = [item[i] for item in mentions]
    out["required_features"] = required_sets
    out["path_coverage"] = coverage
    out["irrelevant_fraction"] = irrelevant
    return out


def report_features(frame: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Compare current report with preceding same-choice report in the same session.

    Missing/unrecognized reports replace the comparison state with an invalid state,
    rather than silently connecting valid reports across a gap of unknown content.
    """
    out = frame.copy()
    use_cols = [f"feature{i}_use" for i in range(1, 5)]
    if all(c in out for c in use_cols):
        # Use the persisted, participant-aligned coding as the single source.
        mentions = [frozenset(getattr(row, f"feature{i}_name") for i in range(1,5)
                              if pd.notna(getattr(row, f"feature{i}_use"))
                              and getattr(row, f"feature{i}_use") == 1)
                    for row in out.itertuples()]
    else:
        # Legacy helper compatibility; the figure entry point requires saved columns.
        mentions = [explicit_features(text) for text in out.text]
    out["report_present"] = out.text.fillna("").str.strip().ne("").to_numpy()
    out["report_recognized"] = [bool(item) for item in mentions]
    out["feature_count"] = [len(item) if item else np.nan for item in mentions]
    out["feature_set"] = ["|".join(sorted(item)) for item in mentions]
    changes, gaps = [], []
    previous = {}
    for row, current in zip(out.itertuples(), mentions):
        key = (row.iSession, row.choice)
        old, old_trial = previous.get(key, (frozenset(), -max_gap))
        gap = row.trial - old_trial
        valid = bool(current and old and 0 < gap <= max_gap)
        changes.append(float(current != old) if valid else np.nan)
        gaps.append(gap if valid else np.nan)
        previous[key] = (current, row.trial)
    out["report_set_change"] = changes
    out["report_comparison_gap"] = gaps
    return out


def reconcile_raw(frame: pd.DataFrame, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Independently join raw stimulus categories to trial keys for an audit."""
    pieces, hashes = [], {}
    for subject, group in frame.groupby("iSub", sort=True):
        bp = raw_dir / f"Task2_{subject}_bhv.csv"
        sp = raw_dir / f"Task2_{subject}_sti.csv"
        bhv, sti = pd.read_csv(bp), pd.read_csv(sp)
        hashes[str(bp)] = sha256(bp)
        hashes[str(sp)] = sha256(sp)
        joined = bhv.merge(sti[["iSession", "stiID", "category"]],
                           on=["iSession", "stiID"], suffixes=("_bhv", "_sti"),
                           validate="many_to_one")
        cols = ["condition", "iSession", "iBlock", "iTrial"]
        joined = joined.rename(columns={"category_bhv": "raw_behavior_category",
                                        "category_sti": "raw_stimulus_category",
                                        "feedback": "raw_feedback", "choice": "raw_choice"})
        merged = group.merge(joined[cols + ["raw_behavior_category", "raw_stimulus_category",
                                           "raw_feedback", "raw_choice"]],
                             on=cols, how="left", validate="one_to_one")
        if merged.raw_stimulus_category.isna().any():
            raise ValueError(f"Missing raw mapping for subject {subject}")
        if not ((merged.feedback == merged.raw_feedback) & (merged.choice == merged.raw_choice)).all():
            raise ValueError(f"Processed/raw response mismatch for subject {subject}")
        merged["raw_stimulus_leaf_category"] = merged.raw_stimulus_category
        merged["raw_stimulus_category"] = task_categories(merged.raw_stimulus_category.to_numpy(),
                                                          int(group.condition.iloc[0]))
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    out["label_feedback_mismatch"] = (out.choice == out.category) != (out.feedback == 1)
    out["stimulus_feedback_mismatch"] = (out.choice == out.raw_stimulus_category) != (out.feedback == 1)
    out["category_source_mismatch"] = out.category != out.raw_stimulus_category
    flagged = out[out.label_feedback_mismatch | out.stimulus_feedback_mismatch | out.category_source_mismatch]
    return out, flagged.copy(), hashes


def measure(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return trial source data, participant summaries and observed-block summaries."""
    trials, subjects, blocks = [], [], []
    for (condition, subject), group in frame.groupby(["condition", "iSub"], sort=True):
        group = task_relative_reports(report_features(group, config["oral_comparison_max_gap"]))
        w = config["rolling_window"]
        group["rolling_accuracy"] = group.correct.rolling(w, min_periods=w).mean()
        group["rolling_label_accuracy"] = (group.choice == group.category).rolling(w, min_periods=w).mean()
        # Mask ambiguity without compressing time: the window still covers w recorded trials.
        clear = group.ambiguous.eq(0)
        group["rolling_clear_n"] = clear.astype(int).rolling(w, min_periods=w).sum()
        group["rolling_clear_accuracy"] = (group.correct.where(clear).rolling(w, min_periods=1).sum()
                                               / group.rolling_clear_n.replace(0, np.nan))
        for gw in config["gain_windows"]:
            group[f"accuracy_w{gw}"] = group.correct.rolling(gw, min_periods=gw).mean()
        cw = config["criterion_window"]
        criterion = group.correct.rolling(cw, min_periods=cw).mean() > config["criterion_threshold"]
        crossing = group.loc[criterion, "trial"]
        summary = {"condition": condition, "iSub": subject, "n_trials": len(group),
                   "sessions": group.iSession.nunique(), "accuracy": group.correct.mean(),
                   "first64_accuracy": group.correct.iloc[:cw].mean(),
                   "last64_accuracy": group.correct.iloc[-cw:].mean(),
                   "first_crossing64": float(crossing.iloc[0]) if len(crossing) else np.nan,
                   "ambiguous_n": int(group.ambiguous.sum()),
                   "missing_text_n": int((~group.report_present).sum()),
                   "recognized_report_n": int(group.report_recognized.sum()),
                   "feature_count_mean": group.feature_count.mean(),
                   "report_change_rate": group.report_set_change.mean(),
                   "report_comparison_n": int(group.report_set_change.notna().sum()),
                   "label_feedback_mismatch_n": int(group.label_feedback_mismatch.sum())}
        for gw in config["gain_windows"]:
            gain, boundary = adjacent_gain(group.correct.to_numpy(), gw)
            summary[f"max_gain_w{gw}"] = gain
            summary[f"max_gain_split_w{gw}"] = boundary
        # First/last 64 trial reports; disjoint only when the recording has >=128 trials.
        for phase, part in [("first", group.iloc[:cw]), ("last", group.iloc[-cw:])]:
            summary[f"{phase}_feature_count"] = part.feature_count.mean() if len(group) >= 2*cw else np.nan
            summary[f"{phase}_report_n"] = int(part.report_recognized.sum()) if len(group) >= 2*cw else 0
            for metric in ["path_coverage", "irrelevant_fraction"]:
                summary[f"{phase}_{metric}"] = part[metric].mean() if len(group) >= 2*cw else np.nan
        for (session, block), part in group.groupby(["iSession", "iBlock"]):
            clear_part = part[part.ambiguous == 0]
            cat_acc = clear_part.groupby("raw_stimulus_category").correct.mean()
            blocks.append({"condition": condition, "iSub": subject, "iSession": session,
                           "iBlock": block, "trial_end": int(part.trial.max()), "n": len(part),
                           "accuracy": part.correct.mean(), "clear_n": len(clear_part),
                           "clear_accuracy": clear_part.correct.mean(),
                           "clear_balanced_accuracy": cat_acc.mean(), "clear_categories_n": len(cat_acc)})
        subjects.append(summary)
        trials.append(group)
    return pd.concat(trials, ignore_index=True), pd.DataFrame(subjects), pd.DataFrame(blocks)


def representatives(summary: pd.DataFrame) -> pd.DataFrame:
    """Illustrative records selected without model fits; not estimated learner classes."""
    selections = []
    for condition, rows in summary.groupby("condition"):
        ordered = rows.sort_values(["n_trials", "iSub"])
        short = ordered.iloc[:max(1, len(ordered)//3)]
        short_row = short.iloc[(len(short)-1)//2]
        long = ordered.iloc[len(ordered)//2:].sort_values(["max_gain_w32", "iSub"])
        for role, row in [("Shorter record", short_row),
                          ("Longer / larger gain", long.iloc[-1]),
                          ("Longer / smaller gain", long.iloc[0])]:
            selections.append({"condition": int(condition), "iSub": int(row.iSub),
                               "role": role, "n_trials": int(row.n_trials),
                               "max_gain_w32": row.max_gain_w32})
    return pd.DataFrame(selections)
