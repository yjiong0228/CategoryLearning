"""Validate all Exp5 Rule1 behavior and preserve session/protocol provenance.

Only stimuli, choices and feedback enter the learner. Subject 508's second
day is retained in the full table and excluded from this version's analysis
sequence. Same-day continuation subjects keep one continuous state history.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import SubjectTrialDataLoader, TrialArrays
from .exp4_data import file_sha256
from ..model.config import ModelContext

FEATURES = [f"feature{i}" for i in range(1, 6)]
TRIAL_KEYS = ["iSession", "iTrial"]
INTEGER_FIELDS = ["ruleID", "iSession", "iBlock", "iTrial", "stiID", "category", "choice", "presskey", "feedback"]


def _integers(frame: pd.DataFrame, fields: list[str], label: str) -> None:
    if not set(fields) <= set(frame):
        raise ValueError(f"Missing {label} columns: {sorted(set(fields) - set(frame))}")
    for field in fields:
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{label}.{field} must contain finite integers")
        frame[field] = values.astype(int)


@dataclass
class Exp5SubjectData:
    subject_id: int
    frame: pd.DataFrame
    arrays: TrialArrays
    source_hashes: dict[str, str]
    protocol: dict
    block_audit: list[dict]

    @property
    def analysis_frame(self) -> pd.DataFrame:
        return self.frame.loc[self.frame.analysis_included].reset_index(drop=True)

    def model_context(self) -> ModelContext:
        return ModelContext(condition=1, subject_id=self.subject_id)

    def metadata(self) -> dict:
        return dict(
            subject=self.subject_id, task="exp5_rule1", recorded_trials=len(self.frame),
            analysis_trials=len(self.arrays.choices),
            excluded_trial_ids=self.frame.loc[~self.frame.analysis_included, TRIAL_KEYS].values.tolist(),
            sessions=[dict(session=int(session), recorded_trials=len(group),
                           analysis_trials=int(group.analysis_included.sum()),
                           boundary=str(group.session_boundary.iloc[0]))
                      for session, group in self.frame.groupby("iSession", sort=False)],
            analysis_session_policy="carry_forward_within_selected_history",
            overnight_policy="first_day_only_second_day_retained_for_later_analysis",
            oral_extended=self.subject_id in self.protocol["oral_extended_subjects"],
            stopping_policy="condition_on_recorded_length_not_a_rule_mastery_label",
            stimulus_coordinates="ordered_left_to_right_black_minus1_white_plus1",
            observation_fields=["stimulus", "choices", "feedback"],
            choice_to_presskey=self.arrays.choice_to_presskey,
            protocol_evidence=self.protocol["protocol_evidence"],
            source_sha256=self.source_hashes, blocks=self.block_audit,
        )


def load_exp5_subject(config_path: Path | str, subject_id: int) -> Exp5SubjectData:
    """Join raw sessions to the complete stimulus design without trial filtering."""
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("protocol") != "model1024_exp5_rule1_inputs_v1":
        raise ValueError("Expected model1024_exp5_rule1_inputs_v1")
    subjects = config["subjects"]
    if len(subjects) != len(set(subjects)) or subject_id not in subjects:
        raise ValueError("Input subjects must be distinct and contain the requested subject")
    continued, overnight = set(config["continued_subjects"]), set(config["overnight_subjects"])
    if (continued & overnight or not (continued | overnight) <= set(subjects)
            or config["overnight_analysis_sessions"] != [1]
            or not set(config["oral_extended_subjects"]) <= continued):
        raise ValueError("Invalid registered continuation/overnight policy")
    if (config["block_size"] != 64 or config["stimulus_values"] != dict(black=-1, white=1)
            or config["correct_rule"] != dict(same_positions=[2, 4], same_category=1, different_category=2)):
        raise ValueError("Input config must retain Exp5 Rule1 coordinates and category mapping")
    root = (config_path.parent / config["raw_dir"]).resolve()
    stimulus_path = root / "Rule1_stimulus.csv"
    behavior_paths = sorted(root.glob(f"{subject_id}_*_bhv.csv"), key=lambda p: int(p.stem.split("_")[1]))
    expected_sessions = [1, 2] if subject_id in continued | overnight else [1]
    if [int(p.stem.split("_")[1]) for p in behavior_paths] != expected_sessions:
        raise ValueError(f"Subject {subject_id}: observed sessions differ from the registered protocol")
    sources = {"config": config_path, "stimulus": stimulus_path,
               **{p.name: p for p in behavior_paths}}
    hashes = {key: file_sha256(path) for key, path in sources.items()}
    stimuli = pd.read_csv(stimulus_path)
    _integers(stimuli, ["stiID", "category", *FEATURES], "stimulus")
    if (len(stimuli) != 32 or stimuli.stiID.duplicated().any()
            or stimuli.duplicated(FEATURES).any() or not stimuli[FEATURES].isin([-1, 1]).all().all()
            or not np.array_equal(stimuli.category, np.where(stimuli.feature2.eq(stimuli.feature4), 1, 2))):
        raise ValueError("Stimulus design must contain all 32 distinct patterns with Rule1 categories")
    batches = []
    audits = []
    for path, session in zip(behavior_paths, expected_sessions):
        batch = pd.read_csv(path)
        _integers(batch, INTEGER_FIELDS, path.name)
        if (batch.empty or not batch.ruleID.eq(1).all() or not batch.iSession.eq(session).all()
                or not np.array_equal(batch.iTrial, np.arange(1, len(batch) + 1))
                or not np.array_equal(batch.iBlock, (batch.iTrial - 1) // 64 + 1)):
            raise ValueError(f"Invalid rule/session/trial order in {path.name}")
        if not batch[["choice", "presskey", "category"]].isin([1, 2]).all().all():
            raise ValueError(f"Invalid binary response/category in {path.name}")
        if (not batch.feedback.isin([0, 1]).all()
                or not np.array_equal(batch.feedback, batch.choice.eq(batch.category).astype(int))):
            raise ValueError(f"Feedback differs from choice correctness in {path.name}")
        batch["source_file"] = path.name
        batch["source_row"] = np.arange(1, len(batch) + 1)  # One-based data rows; CSV line = row + 1.
        batch = batch.merge(stimuli.rename(columns={"category": "stimulus_category"}),
                            on="stiID", how="left", sort=False, validate="many_to_one")
        if batch[FEATURES].isna().any().any() or not batch.category.eq(batch.stimulus_category).all():
            raise ValueError(f"Missing stimulus or category mismatch in {path.name}")
        batch = batch.drop(columns="stimulus_category")
        batch["session_boundary"] = ("initial" if session == 1 else
                                      "overnight" if subject_id in overnight else "same_day_continuation")
        batch["analysis_included"] = subject_id not in overnight or session in config["overnight_analysis_sessions"]
        for block, group in batch.groupby("iBlock", sort=False):
            counts = group.stiID.value_counts()
            audits.append(dict(session=session, block=int(block), trials=len(group),
                               balanced_twice_each=bool(len(counts) == 32 and counts.eq(2).all()),
                               stimulus_counts={str(int(k)): int(v) for k, v in counts.sort_index().items()},
                               accuracy=float(group.feedback.mean())))
        batches.append(batch)
    frame = pd.concat(batches, ignore_index=True)
    frame.insert(0, "iSub", subject_id)
    frame.insert(1, "condition", 1)  # Shared binary-feedback interface; task remains exp5_rule1.
    frame["global_trial"] = np.arange(1, len(frame) + 1)
    if frame.duplicated(TRIAL_KEYS).any():
        raise ValueError("Duplicate session/trial identities")
    loader = SubjectTrialDataLoader(dict(data=dict(feature_columns=FEATURES, category_column="category",
                                                  target_type="hard"), partition=dict(kwargs=dict(n_cats=2))))
    loader._extract_arrays(frame, max_trials=None)  # Validate key mapping across every recorded session.
    selected = frame.loc[frame.analysis_included].reset_index(drop=True)
    arrays = loader._extract_arrays(selected, max_trials=None)
    if hashes != {key: file_sha256(path) for key, path in sources.items()}:
        raise ValueError("Exp5 inputs changed during preparation")
    return Exp5SubjectData(subject_id, frame, arrays, hashes, config, audits)
