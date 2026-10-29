"""Data for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict, Any
import numpy as np
import pandas as pd
from .features import check_pm1

DEFAULT_EXP5_CSV = Path("data/exp5/processed/Rule1_processed.csv")

Array = np.ndarray


def sti_to_bits(sti_id: int, n_bits: int = 5, msb_first: bool = True) -> Array:
    if not np.isfinite(sti_id) or int(sti_id) != sti_id or not 1 <= sti_id <= 2 ** n_bits:
        raise ValueError("stiID must be an integer in 1..2**n_bits")
    val = int(sti_id) - 1
    bits = np.array([(val >> i) & 1 for i in range(n_bits)], dtype=int)
    if msb_first:
        bits = bits[::-1]
    return np.where(bits == 1, 1.0, -1.0)


def prepare_dataframe(df: pd.DataFrame, n_bits: int, msb_first: bool):
    df = df.copy()
    missing = {"category", "choice"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    feature_cols = [f"feature{j+1}" for j in range(n_bits)]
    if set(feature_cols).issubset(df.columns):
        X = df[feature_cols].to_numpy(dtype=float)
        check_pm1(X, "feature columns")
    else:
        if "stiID" not in df or any(c in df for c in feature_cols):
            raise ValueError("provide all binary feature columns or stiID")
        X = np.vstack([sti_to_bits(s, n_bits=n_bits, msb_first=msb_first) for s in df["stiID"]])

    for col in ("category", "choice"):
        if not df[col].isin([1, 2]).all():
            raise ValueError(f"{col} must contain only categories 1 and 2 in binary mode")
    if df.empty:
        raise ValueError("no trials supplied")
    if "stiID" not in df:
        df["stiID"] = np.arange(1, len(df) + 1)
    y = np.where(df["category"].to_numpy() == 1, 1.0, -1.0)
    r = np.where(df["choice"].to_numpy() == 1, 1.0, -1.0)
    for j in range(n_bits):
        df[f"x{j+1}"] = X[:, j].astype(int)
    df["category_pm1"] = y.astype(int)
    df["choice_pm1"] = r.astype(int)
    df["correct"] = (y == r).astype(int)
    if n_bits >= 4:
        df["x2_times_x4"] = (X[:, 1] * X[:, 3]).astype(int)
        df["rule_match_x2x4"] = (df["x2_times_x4"].to_numpy() == y).astype(int)
    return df, X, y, r


def load_csv(csv_path: Path, n_bits: int, msb_first: bool):
    return prepare_dataframe(pd.read_csv(csv_path), n_bits, msb_first)


def sort_subjects(subjects: Sequence[Any]) -> List[str]:
    return sorted({str(s) for s in subjects}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def subject_id_values(df: pd.DataFrame) -> List[str]:
    if "iSub" not in df.columns:
        return ["all"]
    return sort_subjects(df["iSub"].dropna().astype(str).unique())


def id_columns(df: pd.DataFrame) -> Dict[str, Array]:
    return {
        "iSub": df["iSub"].to_numpy() if "iSub" in df.columns else np.full(len(df), "all"),
        "iSession": df["iSession"].to_numpy() if "iSession" in df.columns else np.full(len(df), ""),
        "iTrial": df["iTrial"].to_numpy() if "iTrial" in df.columns else np.arange(1, len(df) + 1),
        "stiID": df["stiID"].to_numpy(),
    }


def prepare_exp123(df: pd.DataFrame, *, max_trials: int | None = None
                   ) -> tuple[pd.DataFrame, Array, Array, Array, Array, int]:
    """Validate one subject, preserve all trials, and order by session/trial.

    Session breaks carry learning state. The four feature columns are already
    normalized to [0, 1] and retain the subject-specific feature-name mapping.
    """
    from .features import positive_integer
    from .feedback import category_count, category_indices, task_feedback

    required = {"iSub", "condition", "iSession", "iTrial", "choice", "feedback", "category"}
    columns = [f"feature{i}" for i in range(1, 5)]
    missing = (required | set(columns)) - set(df.columns)
    if missing:
        raise ValueError(f"Missing exp123 columns: {sorted(missing)}")
    if df.empty or df.iSub.isna().any() or df.iSub.nunique() != 1 or df.condition.nunique() != 1:
        raise ValueError("prepare_exp123 requires one nonempty subject and condition")
    for col in ("condition", "iSession", "iTrial"):
        values = df[col].to_numpy(float)
        if not np.all(np.isfinite(values)) or np.any(values < 1) or np.any(values != np.floor(values)):
            raise ValueError(f"{col} must contain positive integers")
    if df.duplicated(["iSession", "iTrial"]).any():
        raise ValueError("duplicate session/trial IDs")
    frame = df.sort_values(["iSession", "iTrial"], kind="stable").copy().reset_index(drop=True)
    condition = int(frame.condition.iloc[0])
    k = category_count(condition)
    X = frame[columns].to_numpy(float)
    # Validate the input range without any data-dependent scaling.
    if not np.all(np.isfinite(X)) or np.any((X < 0) | (X > 1)):
        raise ValueError("feature1..4 must be finite values in [0, 1]")
    y = category_indices(frame.category, k, "category") + 1
    r = category_indices(frame.choice, k, "choice") + 1
    fb = frame.feedback.to_numpy(float)
    expected = np.asarray([task_feedback(a, b, condition) for a, b in zip(r, y)])
    if not np.array_equal(fb, expected):
        raise ValueError("feedback is inconsistent with choice/category and the task protocol")
    if max_trials is not None:
        positive_integer(max_trials, "max_trials")
        frame = frame.iloc[:max_trials].copy()
        X, y, r, fb = X[:max_trials], y[:max_trials], r[:max_trials], fb[:max_trials]
    return frame, X, y, r, fb, condition
