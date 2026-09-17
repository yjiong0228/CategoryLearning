"""Descriptive bottleneck measures from saved Model 0826 inference outputs.

This module reads existing results only. It does not refit or implement a
cognitive mechanism. Marginal screens are not joint latent-state events.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    """Hash an input without materializing a large stream in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conditional_support(mass: np.ndarray, active: np.ndarray) -> np.ndarray:
    """E[belief | active] from aligned marginal mass and active probability.

    Across PF repeats aggregate numerator and denominator before this division.
    Zero active probability is undefined, rather than evidence of zero support.
    """
    mass, active = np.asarray(mass, dtype=float), np.asarray(active, dtype=float)
    if mass.shape != active.shape or not np.isfinite(mass).all() or not np.isfinite(active).all():
        raise ValueError("belief and active probability must be finite aligned arrays")
    if np.any(mass < -1e-9) or np.any(active < -1e-9) or np.any(active > 1 + 1e-9):
        raise ValueError("invalid marginal probability")
    if np.any(mass > active + 1e-8):
        raise ValueError("belief mass exceeds probability of being active")
    return np.divide(mass, active, out=np.full_like(mass, np.nan), where=active > 1e-12)


def contiguous_runs(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    """Return half-open index intervals; trial labels remain a separate column."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a >= minimum]


def validate_frame(raw: pd.DataFrame, subject: int, condition: int) -> None:
    """Reject alignment errors without silently sorting source trials."""
    if raw.empty or not raw.iSub.eq(subject).all() or not raw.condition.eq(condition).all():
        raise ValueError("subject/condition mismatch")
    keys = ["iSession", "iBlock", "iTrial"]
    index = pd.MultiIndex.from_frame(raw[keys])
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("source trial keys must be ordered and unique")
    n_categories = 2 if condition == 1 else 4
    for column in ["choice", "category"]:
        values = raw[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"non-integer {column}")
        if not ((values >= 1) & (values <= n_categories)).all():
            raise ValueError(f"out-of-range {column}")
    np.testing.assert_array_equal(raw.choice.eq(raw.category), raw.feedback.eq(1))


def _probability_matrix(value: Any, shape: tuple[int, int], normalized: bool) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid probability array, expected {shape}, got {array.shape}")
    if np.any(array < -1e-9) or np.any(array > 1 + 1e-9):
        raise ValueError("probability outside [0, 1]")
    if normalized:
        np.testing.assert_allclose(array.sum(axis=1), 1, atol=1e-8)
    return array


def read_case(spec: dict[str, Any], config: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
    """Aggregate all saved PF repeats and align the saved oral encoding."""
    subject = int(spec["subject"])
    model_dir = ROOT / spec["model_dir"]
    payload_path = model_dir / "simulation/subjects" / f"subject_{subject}.json"
    payload = json.loads(payload_path.read_text())
    condition = int(payload["condition"])
    if payload["subject_id"] != subject or condition not in (1, 2):
        raise ValueError("this descriptive reader supports fitted condition 1/2 cases")
    if spec["task"] != {1: 1, 2: 3}[condition]:
        raise ValueError("task labels must not be confused with condition numbers")
    raw = data.loc[data.iSub.eq(subject)].copy().reset_index(drop=True)
    validate_frame(raw, subject, condition)
    n = len(raw)
    oral_path = model_dir / spec["oral_file"]
    with np.load(oral_path, allow_pickle=False) as archive:
        rows = np.flatnonzero(archive["subjects"] == subject)
        if len(rows) != 1:
            raise ValueError("oral subject mapping is not unique")
        row = int(rows[0])
        if int(archive["n_trials"][row]) != n or int(archive["conditions"][row]) != condition:
            raise ValueError("oral trial count or condition mismatch")
        np.testing.assert_allclose(float(archive["oral_center_sigma"][row]), config["oral_sigma"])
        target = int(archive["target_hypos"][row])
        oral = np.asarray(archive["oral_mass"][row], dtype=float)
        instantaneous = np.asarray(archive["instantaneous_oral_mass"][row], dtype=float)
        report_valid = np.asarray(archive["valid_oral_report"][row], dtype=bool)
        state_valid = np.asarray(archive["valid_oral"][row], dtype=bool)
        report_category = np.asarray(archive["oral_state_update_category"][row], dtype=float)
        oral_meta = {key: np.asarray(archive[key])[row].tolist() for key in
                     ["oral_encoder_version", "oral_distribution_method", "oral_state_mode",
                      "hypothesis_space_signature"]}
    n_rules = oral.shape[1]
    for array, valid in [(oral, state_valid), (instantaneous, report_valid)]:
        if array.shape != (n, n_rules) or valid.shape != (n,):
            raise ValueError("oral shape mismatch")
        _probability_matrix(array[valid], (int(valid.sum()), n_rules), True)
    # The saved encoder exposes one-based response categories for report updates.
    np.testing.assert_array_equal(report_category[report_valid], raw.choice.to_numpy()[report_valid])
    params = payload["best_params"]
    capacity = int(params["capacity"])
    persistent = bool(params["engine.modules.hypo_transitions_mod.kwargs.persistent_execution.enabled"])
    stream = (payload_path.parent / payload["raw_runs_ref"]["path"]).resolve()
    sums: dict[str, np.ndarray] = {}
    sum_squares: dict[str, np.ndarray] = {}
    seed_frames, seeds = [], []
    score_mask = None
    valid_mask = None
    count = 0
    with gzip.open(stream, "rb") as handle:
        while True:
            try:
                run = pickle.load(handle)
            except EOFError:
                break
            if run["subject_id"] != subject or run["condition"] != condition or run["selection_prediction_mode"] != "prior_t":
                raise ValueError("saved run identity or timing mismatch")
            metrics, state = run["metrics_by_mode"]["prior_t"], run["state_log"]
            for key, expected in [("observed_choice", raw.choice), ("observed_feedback", raw.feedback)]:
                np.testing.assert_array_equal(metrics[key], expected.to_numpy())
            np.testing.assert_array_equal(metrics["true_category_index"], raw.category.to_numpy() - 1)
            current_valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
            current_mask = np.asarray(metrics["score_trial_mask"], dtype=bool)
            if current_valid.shape != (n,) or current_mask.shape != (n,):
                raise ValueError("saved evaluation mask shape mismatch")
            if score_mask is not None:
                np.testing.assert_array_equal(current_mask, score_mask)
                np.testing.assert_array_equal(current_valid, valid_mask)
            score_mask = current_mask
            valid_mask = current_valid
            prior = _probability_matrix(state["marginal_prior"], (n, n_rules), True)
            active = _probability_matrix(state["marginal_active_probability"], (n, n_rules), False)
            np.testing.assert_allclose(active.sum(axis=1), capacity, atol=1e-8)
            conditional_support(prior, active)
            values = {
                "prior": prior, "active": active,
                "prediction": _probability_matrix(metrics["pred_category_probs"], (n, 2 if condition == 1 else 4), True),
            }
            for key in ["predictive_swap_probability", "predictive_search_range", "predictive_replacement_fraction"]:
                values[key] = np.asarray(state[key], dtype=float)
                if values[key].shape != (n,) or not np.isfinite(values[key]).all():
                    raise ValueError(f"missing or invalid {key}")
            if persistent:
                executed = _probability_matrix(state["marginal_executed_probability"], (n, n_rules), True)
                if np.any(executed > active + 1e-8):
                    raise ValueError("executed rule must be active")
                values["executed"] = executed
            elif "marginal_executed_probability" in state:
                raise ValueError("mixture readout unexpectedly has an executed rule")
            seed = int(run["trajectory_seed"])
            seeds.append(seed)
            seed_frames.append(pd.DataFrame({
                "subject": subject, "repeat": count, "seed": seed, "trial": np.arange(1, n + 1),
                "target_active": active[:, target], "target_mass": prior[:, target],
                "target_execution": values["executed"][:, target] if persistent else np.full(n, np.nan),
            }))
            for key, array in values.items():
                if key not in sums:
                    sums[key], sum_squares[key] = np.zeros_like(array), np.zeros_like(array)
                sums[key] += array
                sum_squares[key] += array * array
            count += 1
    if count != int(payload["raw_runs_ref"]["count"]) or count < 2 or len(set(seeds)) != count:
        raise ValueError("PF repeat count or unique seed check failed")
    means = {key: value / count for key, value in sums.items()}
    sds = {key: np.sqrt(np.maximum((sum_squares[key] - value * value / count) / (count - 1), 0))
           for key, value in sums.items()}
    a, q = means["active"][:, target], means["prior"][:, target]
    c = conditional_support(q, a)
    # Denominators with very little active probability do not support a stable
    # displayed conditional ratio; retain the unmasked value in source tables.
    display_c = np.where(a >= config["conditional_display_min_active"], c, np.nan)
    prediction = means["prediction"]
    table = raw.copy()
    table.insert(0, "trial", np.arange(1, n + 1))
    table["subject"], table["task"] = subject, int(spec["task"])
    table["target_rule"], table["capacity"] = target, capacity
    table["target_active"], table["target_mass"] = a, q
    table["target_support_if_active"], table["target_support_display"] = c, display_c
    table["target_active_seed_sd"], table["target_mass_seed_sd"] = sds["active"][:, target], sds["prior"][:, target]
    table["target_execution"] = means["executed"][:, target] if persistent else np.nan
    table["target_execution_seed_sd"] = sds["executed"][:, target] if persistent else np.nan
    table["execution_applicable"] = persistent
    table["support_execution_gap"] = q - table["target_execution"]
    table["observed_accuracy"] = raw.choice.eq(raw.category).astype(float)
    table["model_correct_probability"] = prediction[np.arange(n), raw.category.to_numpy(dtype=int) - 1]
    table["model_observed_choice_probability"] = prediction[np.arange(n), raw.choice.to_numpy(dtype=int) - 1]
    table["score_trial_mask"] = score_mask
    table["saved_valid_trial_mask"] = valid_mask
    for key in ["predictive_swap_probability", "predictive_search_range", "predictive_replacement_fraction"]:
        table[key] = means[key]
    table["oral_report_valid"], table["oral_state_valid"] = report_valid, state_valid
    table["oral_target_state"] = np.where(state_valid, oral[:, target], np.nan)
    table["oral_target_current_report"] = np.where(report_valid, instantaneous[:, target], np.nan)
    table["oral_state_target_seed_independent"] = True
    table["oral_model_overlap"] = np.where(state_valid, np.minimum(means["prior"], oral).sum(axis=1), np.nan)
    prior_report: dict[int, tuple[int, str]] = {}
    previous_trial, previous_text, text_changed = [], [], []
    for i, row in table.iterrows():
        category = int(row.choice)
        old = prior_report.get(category)
        text = "" if pd.isna(row.text) else str(row.text).strip()
        previous_trial.append(np.nan if old is None else old[0])
        previous_text.append("" if old is None else old[1])
        text_changed.append(bool(report_valid[i] and old is not None and text != old[1]))
        if report_valid[i]:
            prior_report[category] = (i + 1, text)
    table["previous_same_choice_report_trial"] = previous_trial
    table["previous_same_choice_report_text"] = previous_text
    table["report_text_changed_same_choice"] = text_changed
    for key in ["observed_accuracy", "model_correct_probability"]:
        table[key + "_rolling"] = table[key].rolling(config["rolling_window"], min_periods=config["rolling_window"]).mean()
    source_paths = [payload_path, stream, oral_path, ROOT / config["data"]]
    metadata = {
        "subject": subject, "task": int(spec["task"]), "condition": condition,
        "n_trials": n, "n_rules": n_rules, "target_rule": target, "capacity": capacity,
        "persistent_execution": persistent, "pf_repeats": count, "seeds": seeds,
        "oral_metadata": oral_meta, "model_provenance": payload.get("model_provenance"),
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "score_trials": int(score_mask.sum()), "choice_nll": float(-np.log(np.clip(table.loc[score_mask, "model_observed_choice_probability"], 1e-12, 1)).mean()),
        "current_report_count": int(report_valid.sum()), "conditional_display_masked_trials": int(np.isnan(display_c).sum()),
        "fit_scope": "Saved full-sequence parameter estimate; no held-out evaluation",
    }
    return {"table": table, "prior": means["prior"], "active": means["active"],
            "execution": means.get("executed"), "oral": oral, "instantaneous_oral": instantaneous,
            "seed_table": pd.concat(seed_frames, ignore_index=True), "metadata": metadata}


def stage_profiles(case: dict[str, Any], count: int) -> pd.DataFrame:
    """Use all trials in contiguous equal-count stages; pool before division."""
    d, meta = case["table"], case["metadata"]
    rows = []
    for stage, indices in enumerate(np.array_split(np.arange(len(d)), count), 1):
        block = d.iloc[indices]
        active_total = float(block.target_active.sum())
        rows.append({
            "subject": meta["subject"], "task": meta["task"], "stage": stage,
            "first_trial": int(block.trial.iloc[0]), "last_trial": int(block.trial.iloc[-1]),
            "n_trials": len(block), "target_active_mean": float(block.target_active.mean()),
            "target_support_pooled": float(block.target_mass.sum() / active_total) if active_total > 1e-12 else np.nan,
            "active_probability_sum": active_total,
            "target_mass_mean": float(block.target_mass.mean()),
            "target_execution_mean": float(block.target_execution.mean()) if meta["persistent_execution"] else np.nan,
            "observed_accuracy": float(block.observed_accuracy.mean()),
            "model_accuracy": float(block.model_correct_probability.mean()),
            "oral_target_state_mean_at_new_reports": float(block.loc[block.oral_report_valid, "oral_target_state"].mean()),
            "oral_target_current_report_mean": float(block.oral_target_current_report.mean()),
            "oral_reports": int(block.oral_report_valid.sum()),
            "current_report_model_overlap_mean": float(block.loc[block.oral_report_valid, "oral_model_overlap"].mean()),
            "execution_applicable": meta["persistent_execution"],
        })
    return pd.DataFrame(rows)


def screen_episodes(case: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory marginal screens only; never assign a psychological subtype."""
    d, meta, setting = case["table"], case["metadata"], config["screening"]
    a, c = d.target_active.to_numpy(), d.target_support_if_active.to_numpy()
    e = d.target_execution.to_numpy()
    high_a, high_c = setting["high_active"], setting["high_support"]
    masks = {
        "low_current_availability": a <= setting["low_active"],
        "available_support_at_or_below_equal_candidate_weight": (a >= high_a) & (c <= 1 / meta["capacity"]),
    }
    if meta["persistent_execution"]:
        masks["supported_low_execution_marginal_screen"] = (a >= high_a) & (c >= high_c) & (e <= setting["low_execution"])
    rows = []
    for kind, mask in masks.items():
        for begin, stop in contiguous_runs(mask, setting["minimum_run"]):
            block = d.iloc[begin:stop]
            lookback = max(0, begin - setting["support_drop_lookback"])
            previous = np.flatnonzero((a[lookback:begin] >= high_a) & (c[lookback:begin] >= high_c)) + lookback
            last_high = int(previous[-1]) if len(previous) else None
            rows.append({
                "subject": meta["subject"], "screen": kind, "first_trial": begin + 1,
                "last_trial": stop, "n_trials": stop - begin,
                "mean_active": float(block.target_active.mean()),
                "pooled_support": float(block.target_mass.sum() / block.target_active.sum()) if block.target_active.sum() > 1e-12 else np.nan,
                "mean_execution": float(block.target_execution.mean()) if meta["persistent_execution"] else np.nan,
                "observed_accuracy": float(block.observed_accuracy.mean()),
                "current_reports": int(block.oral_report_valid.sum()),
                "last_prior_high_support_trial": np.nan if last_high is None else last_high + 1,
                "active_stayed_high_since_prior_support": bool(last_high is not None and np.all(a[last_high:stop] >= high_a)),
                "interpretation": "Marginal threshold screen; not a recovered latent event or human diagnosis",
            })
    columns = ["subject", "screen", "first_trial", "last_trial", "n_trials", "mean_active", "pooled_support", "mean_execution", "observed_accuracy", "current_reports", "last_prior_high_support_trial", "active_stayed_high_since_prior_support", "interpretation"]
    sensitivity = []
    for ah in config["sensitivity_high_active"]:
        for ch in config["sensitivity_high_support"]:
            eligible = (a >= ah) & (c >= ch)
            gap = eligible & (e <= setting["low_execution"])
            sensitivity.append({"subject": meta["subject"], "high_active": ah, "high_support": ch,
                                "high_support_trials": int(eligible.sum()),
                                "low_execution_trials": int(gap.sum()) if meta["persistent_execution"] else np.nan,
                                "low_execution_runs": len(contiguous_runs(gap, setting["minimum_run"])) if meta["persistent_execution"] else np.nan,
                                "execution_applicable": meta["persistent_execution"]})
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(sensitivity)


def save_case_sources(case: dict[str, Any], output: Path) -> None:
    """Write new derived tables, retaining the complete rule catalogue."""
    output.mkdir(exist_ok=False)
    d, meta = case["table"], case["metadata"]
    d.to_csv(output / "trial_states.csv", index=False)
    case["seed_table"].to_csv(output / "pf_seed_target_states.csv", index=False)
    n, j = case["prior"].shape
    pd.DataFrame({
        "subject": meta["subject"], "trial": np.repeat(np.arange(1, n + 1), j),
        "rule": np.tile(np.arange(j), n), "belief_mass": case["prior"].ravel(),
        "active_probability": case["active"].ravel(),
        "support_if_active": conditional_support(case["prior"], case["active"]).ravel(),
        "execution_probability": case["execution"].ravel() if case["execution"] is not None else np.full(n * j, np.nan),
        "oral_state_weight": case["oral"].ravel(), "oral_current_report_weight": case["instantaneous_oral"].ravel(),
    }).to_csv(output / "all_rule_states.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
