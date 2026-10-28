"""Prepare exp4 observations and separate experimenter-side probabilities.

One subject supplies one continuous history across sessions. Only stimulus,
choice and recorded feedback enter the learner. Raw stimulus probabilities
remain available for the stochastic environment and task evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import SubjectTrialDataLoader, TrialArrays, prepare_trial_sequence
from ..model.config import ModelContext
from ..model.modules.perception import PerceptionModule
from ..utils.datasets import resolve_dataset_paths

FEATURES = [f"feature{i}" for i in range(1, 5)]
FEATURE_NAMES = [f"{name}_name" for name in FEATURES]
TRIAL_KEYS = ["iSession", "iTrial"]
PROBABILITIES = ["probCat1", "probCat2"]
COLORS = {"green", "yellow", "pink", "blue"}


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _probabilities(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
            or not np.allclose(values.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError(f"{label} must contain normalized finite binary probabilities")
    return values


def _align(frame: pd.DataFrame, reference: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
    """Align by explicit identity while retaining the reference's trial order."""
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {label} keys: {keys}")
    indexed = frame.set_index(keys)
    wanted = pd.MultiIndex.from_frame(reference[keys])
    if not wanted.isin(indexed.index).all():
        raise ValueError(f"Missing {label} trial identities")
    return indexed.loc[wanted].reset_index()


@dataclass
class Exp4SubjectData:
    """Validated continuous observations, identities and task-only metadata."""

    subject_id: int
    frame: pd.DataFrame
    arrays: TrialArrays
    feature_names: tuple[str, ...]
    perception_half_range: np.ndarray
    dataset_paths: dict[str, Path]
    source_hashes: dict[str, str]
    feedback_probability_disclosed: bool

    def model_context(self) -> ModelContext:
        """Use exp4's measured perception data in the shared model."""
        return ModelContext(condition=1, subject_id=self.subject_id,
                            processed_data_dir=self.dataset_paths["processed_dir"],
                            dataset_paths=self.dataset_paths)

    def observations(self) -> list:
        """Return one history; session boundaries introduce no model reset."""
        return prepare_trial_sequence(self.arrays.stimulus, self.arrays.choices, self.arrays.feedback)

    def metadata(self) -> dict[str, Any]:
        sessions = self.frame.iSession.to_numpy(dtype=int)
        starts = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
        ends = np.r_[starts[1:], len(sessions)]
        return dict(
            subject=self.subject_id, condition=1, trials=len(self.frame),
            sessions=[dict(session=int(sessions[start]), start_index=int(start),
                           stop_index=int(end)) for start, end in zip(starts, ends)],
            session_state_policy="carry_forward", trial_order_columns=TRIAL_KEYS,
            feature_names=list(self.feature_names),
            perception=dict(noise_mode="uniform", half_range=self.perception_half_range.tolist()),
            choice_to_presskey=self.arrays.choice_to_presskey,
            feedback_probability_disclosed=self.feedback_probability_disclosed,
            observation_fields=["stimulus", "choices", "feedback"],
            target_probability_source="raw_stimulus_prob_category1_prob_category2",
            processed_probability_max_difference=float(np.max(np.abs(
                self.frame[PROBABILITIES].to_numpy() - self.arrays.target_probs))),
            ambiguity_disagreement_trial_ids=self.frame.loc[
                self.frame.ambiguous.ne(self.frame.stimulus_ambiguous), TRIAL_KEYS
            ].to_numpy(dtype=int).tolist(),
            source_sha256=self.source_hashes,
        )


def load_exp4_subject(config_path: Path | str, subject_id: int) -> Exp4SubjectData:
    """Cross-check processed behavior against raw behavior and stimulus tables.

The returned target probabilities retain raw stimulus precision. Rounded
processed probabilities remain in ``frame``. The learner receives the actual
binary feedback, including negative outcomes after a higher-probability choice.
"""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("protocol") != "model1024_exp4_inputs_v1":
        raise ValueError("Expected model1024_exp4_inputs_v1 input protocol")
    if subject_id not in config["subjects"]:
        raise ValueError(f"Subject {subject_id} is not registered in the input config")
    disclosed = config["feedback_probability_disclosed"]
    if not isinstance(disclosed, bool):
        raise ValueError("feedback_probability_disclosed must be a boolean")
    expected = dict(subject_column="iSub", condition_column="condition",
                    feature_columns=FEATURES, trial_order_columns=TRIAL_KEYS,
                    target_type="probabilistic", probability_columns=PROBABILITIES)
    if config["data"] != expected:
        raise ValueError("exp4 data columns must retain the registered response and trial coordinates")
    paths = resolve_dataset_paths(config, config_path.parent)
    raw_dir = (config_path.parent / config["raw_task_dir"]).resolve()
    behavior_path = raw_dir / f"Task2_{subject_id}_bhv.csv"
    stimulus_path = raw_dir / f"Task2_{subject_id}_sti.csv"
    sources = dict(config=config_path, learning_data=paths["learning_data"],
                   perception_summary=paths["perception_summary_72"],
                   feature_order_data=paths["feature_order_data"],
                   raw_behavior=behavior_path, raw_stimulus=stimulus_path)
    hashes = {name: file_sha256(path) for name, path in sources.items()}

    loader = SubjectTrialDataLoader({"data": config["data"], "partition": {"kwargs": {"n_cats": 2}}},
                                    dataset_paths=paths)
    loader.prepare_data(paths["learning_data"])
    frame = loader._get_subject_frame(subject_id, stop_at=1.)
    frame.insert(0, "source_row", frame.index.to_numpy())
    frame = frame.reset_index(drop=True)
    if not frame.condition.eq(1).all():
        raise ValueError("exp4 requires condition 1 binary responses")
    features = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
        raise ValueError("exp4 features must be finite values in [0, 1]")
    if not frame.feedback.isin([0., 1.]).all():
        raise ValueError("exp4 observed feedback must be binary")
    _probabilities(frame[PROBABILITIES].to_numpy(), "Processed probabilities")
    orders = frame[FEATURE_NAMES].drop_duplicates()
    if len(orders) != 1 or set(orders.iloc[0]) != COLORS:
        raise ValueError("Feature names must give one fixed permutation of exp4 colors across sessions")
    feature_names = tuple(orders.iloc[0])

    raw = pd.read_csv(behavior_path)
    raw_keys = pd.MultiIndex.from_frame(raw[TRIAL_KEYS])
    source_frame = loader.learning_data.loc[loader.learning_data.iSub.eq(subject_id)]
    if (len(raw) != len(source_frame)
            or set(raw_keys) != set(pd.MultiIndex.from_frame(source_frame[TRIAL_KEYS]))):
        raise ValueError("Raw and processed behavior must contain the same trial identities")
    raw = _align(raw, frame, TRIAL_KEYS, "raw behavior")
    for name in ["condition", "iBlock", "choice", "feedback", "ambiguous", *PROBABILITIES]:
        if not np.array_equal(frame[name].to_numpy(), raw[name].to_numpy()):
            raise ValueError(f"Raw/processed mismatch in {name}")
    if "presskey" in frame and not np.array_equal(frame.presskey, raw.presskey):
        raise ValueError("Raw/processed mismatch in presskey")
    frame["presskey"] = raw.presskey.to_numpy()
    frame["stiID"] = raw.stiID.to_numpy()
    frame.attrs.pop("choice_to_presskey", None)  # Validate the recovered raw key mapping.

    stimuli = _align(pd.read_csv(stimulus_path), frame, ["iSession", "stiID"], "raw stimulus")
    if (not np.allclose(features, stimuli[FEATURES].to_numpy(), atol=1e-12, rtol=0)
            or not np.array_equal(frame[FEATURE_NAMES].to_numpy(), stimuli[FEATURE_NAMES].to_numpy())):
        raise ValueError("Raw/processed stimulus or feature-coordinate mismatch")
    if not frame.ambiguous.isin([0, 1]).all() or not stimuli.ambiguous.isin([0, 1]).all():
        raise ValueError("Ambiguity flags must be binary")
    frame["stimulus_ambiguous"] = stimuli.ambiguous.to_numpy()
    probabilities = _probabilities(stimuli[["prob_category1", "prob_category2"]].to_numpy(),
                                   "Raw stimulus probabilities")
    # Behavioral CSVs print probabilities to three decimal places.
    if np.max(np.abs(probabilities - frame[PROBABILITIES].to_numpy())) > .0005 + 1e-12:
        raise ValueError("Raw/processed probabilities disagree beyond three-decimal rounding")

    summary = pd.read_csv(paths["perception_summary_72"])
    measured = summary.loc[summary.iSub.eq(subject_id)]
    thresholds = measured.threshold_mean_mean.to_numpy(dtype=float)
    if (set(measured.feature_name) != COLORS or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0)):
        raise ValueError("Perception thresholds must be complete, finite and nonnegative")
    expected_half_range = measured.groupby("feature_name").threshold_mean_mean.mean().loc[list(feature_names)].to_numpy()
    half_range = np.asarray(PerceptionModule._load_uniform_half_range(subject_id, paths["processed_dir"], paths))
    if not np.allclose(half_range, expected_half_range, atol=1e-14, rtol=0):
        raise ValueError("Perception feature order differs from the learning stimuli")

    arrays = loader._extract_arrays(frame, max_trials=None)
    arrays.target_probs = probabilities.copy()
    if arrays.categories is not None:
        raise ValueError("exp4 inputs must retain probability targets")
    if hashes != {name: file_sha256(path) for name, path in sources.items()}:
        raise ValueError("Source files changed during exp4 input preparation")
    return Exp4SubjectData(subject_id, frame, arrays, feature_names, half_range.copy(), paths, hashes, disclosed)
