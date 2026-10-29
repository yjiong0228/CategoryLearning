"""Explicit stimulus encodings and feedback semantics for a single CSV history."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TrialData:
    frame: pd.DataFrame
    stimuli: np.ndarray
    targets: np.ndarray
    choices: np.ndarray
    dimensions: tuple[int, ...]
    category_values: list
    source: Path
    source_sha256: str


def encode_features(frame: pd.DataFrame, features: list[dict]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Encode nominal values and scale continuous values using declared bounds.

    Bounds and vocabularies come from the task specification, never the sample.
    Input columns outside ``features`` cannot enter the learner.
    """
    if not features or len({f["column"] for f in features}) != len(features):
        raise ValueError("features must list distinct stimulus columns")
    blocks, widths = [], []
    for feature in features:
        column, kind = feature["column"], feature["kind"]
        if column not in frame:
            raise ValueError(f"Missing feature column: {column}")
        values = frame[column]
        if kind == "continuous":
            lower, upper = map(float, feature["bounds"])
            if not np.isfinite([lower, upper]).all() or upper <= lower:
                raise ValueError("continuous bounds must be finite and increasing")
            block = (pd.to_numeric(values, errors="raise").to_numpy(float) - lower) / (upper - lower)
            if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
                raise ValueError(f"{column} contains missing values or values outside declared bounds")
            blocks.append(block[:, None])
            widths.append(1)
        elif kind == "nominal":
            vocabulary = feature["values"]
            if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
                raise ValueError("nominal values must contain at least two distinct entries")
            mapping = {value: i for i, value in enumerate(vocabulary)}
            index = values.map(mapping)
            if index.isna().any():
                raise ValueError(f"{column} contains missing or undeclared nominal values")
            blocks.append(np.eye(len(vocabulary))[index.to_numpy(int)])
            widths.append(len(vocabulary))
        else:
            raise ValueError("feature kind must be continuous or nominal")
    return np.column_stack(blocks), tuple(widths)


def load_csv_history(config: dict) -> TrialData:
    """Read one history without dropping or reordering trials.

    ``binary_correctness`` derives category feedback from the actual choice and
    binary correctness. ``category_label`` requires a declared column containing
    the category revealed to the learner. Four-category correctness/partial
    rewards require a separately specified learning extension.
    """
    path = Path(config["path"])
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    if source_sha256 != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Input CSV changed while it was being read")
    frame.insert(0, "sustain_source_row", np.arange(1, len(frame) + 1))
    for column, value in config.get("filters", {}).items():
        if column not in frame:
            raise ValueError(f"Unknown filter column: {column}")
        frame = frame.loc[frame[column] == value]
    if frame.empty:
        raise ValueError("No trials match the configured filters")
    # These identifiers must be constant. Sessions can continue only through
    # an explicit carry policy, while trial identities remain ordered.
    for column in config.get("group_columns", ["iSub", "condition"]):
        if column in frame and frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"Select a single {column}; independent histories cannot be concatenated")
    session_column = config.get("session_column", "iSession")
    if config.get("session_policy") not in ("single", "carry"):
        raise ValueError("session_policy must explicitly be single or carry")
    if (session_column in frame and config["session_policy"] == "single"
            and frame[session_column].nunique(dropna=False) != 1):
        raise ValueError("Multiple sessions require explicit selection or session_policy: carry")
    order_columns = config["order_columns"]
    if (not order_columns or any(c not in frame for c in order_columns)
            or frame[order_columns].isna().any().any()):
        raise ValueError("order_columns must identify each trial without missing values")
    order = pd.MultiIndex.from_frame(frame[order_columns])
    if not order.is_unique or not order.is_monotonic_increasing:
        raise ValueError("Trial identifiers must be unique and ordered; the loader does not sort or drop rows")
    # A smoke limit always preserves a prefix of the validated sequence.
    limit = config.get("max_trials")
    if limit is not None:
        if isinstance(limit, bool) or int(limit) != limit or limit < 1:
            raise ValueError("max_trials must be a positive integer")
        frame = frame.iloc[:int(limit)]
    frame = frame.reset_index(drop=True)
    categories = config["category_values"]
    if len(categories) < 2 or len(set(categories)) != len(categories):
        raise ValueError("category_values must specify the complete ordered response vocabulary")
    mapping = {value: i for i, value in enumerate(categories)}
    choice_column = config["choice_column"]
    if choice_column not in frame:
        raise ValueError(f"Missing choice column: {choice_column}")
    choices = frame[choice_column].map(mapping)
    if choices.isna().any():
        raise ValueError("Missing or undeclared choices; no trial filtering is performed")
    choices = choices.to_numpy(int)
    feedback = config["feedback"]
    mode = feedback["mode"]
    if feedback["column"] not in frame:
        raise ValueError("Missing feedback column")
    if mode == "binary_correctness":
        if len(categories) != 2:
            raise ValueError("Binary correctness identifies a category only in a two-category task")
        correct = frame[feedback["column"]].to_numpy()
        if not np.isin(correct, (0, 1)).all():
            raise ValueError("binary_correctness requires complete 0/1 feedback")
        targets = np.where(correct == 1, choices, 1 - choices)
    elif mode == "category_label":
        target_series = frame[feedback["column"]].map(mapping)
        if target_series.isna().any():
            raise ValueError("Category feedback contains missing or undeclared values")
        targets = target_series.to_numpy(int)
    else:
        raise ValueError("Supported feedback: binary_correctness or category_label")
    # Optional ground truth is used only to audit feedback, never as a fallback.
    audit_column = config.get("audit_category_column")
    if audit_column:
        if audit_column not in frame:
            raise ValueError("Missing audit category column")
        audited = frame[audit_column].map(mapping)
        if audited.isna().any() or not np.array_equal(targets, audited.to_numpy(int)):
            raise ValueError("Declared category differs from the category identified by feedback")
    forbidden = {choice_column, feedback["column"], audit_column, *order_columns,
                 *config.get("group_columns", ["iSub", "condition"])}
    if any(f["column"] in forbidden for f in config["features"]):
        raise ValueError("Stimulus features must be separate from choices, feedback and identifiers")
    stimuli, dimensions = encode_features(frame, config["features"])
    return TrialData(frame, stimuli, targets.astype(int), choices, dimensions, categories, path, source_sha256)
