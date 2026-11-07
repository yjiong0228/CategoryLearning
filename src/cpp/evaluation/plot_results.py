"""Offline evaluation of C++ saved forecasts. No model runtime imports.

Consumes evaluation_inputs.npz + evaluation_manifest.json (gm1031-evaluation-inputs-v1).
Required arrays and manifest fields are documented by validate_inputs below.
Numerical metrics and bands retain the sibling evaluation-v1 definitions.
Outputs use the gm1031 namespace and retain C++ identity. Oral reports are unscored.

Usage: python -B evaluation/plot_results.py --input RUN_OR_SUBJECT [--window 32]
Default output is SUBJECT/evaluation. Every destination must be NEW, including
an explicitly supplied --output. Failed partial outputs are retained, never reused.
Only arrays and manifest are read; no resource/data/model lookup is performed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

SEMANTIC_KINDS = ["uncommitted", "example", "generalization", "catalogue_rule"]


def _positive_int(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def rolling_accuracy(
    values: Any,
    window: int = 32,
    *,
    valid: Any = None,
    session_ids: Any = None,
    min_periods: int = 1,
) -> np.ndarray:
    """Trailing means of correctness/probabilities, without compressing gaps.

    Nonfinite values are excluded. A masked current trial may display earlier
    valid trials within its window; a window with no valid trials stays NaN.
    Inputs may contain fractional expected correctness in [0,1].
    """
    window = _positive_int(window, "window")
    min_periods = _positive_int(min_periods, "min_periods")
    if min_periods > window:
        raise ValueError("min_periods cannot exceed window")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must have shape [T]")
    mask = (
        np.ones(values.shape, bool) if valid is None else np.asarray(valid, dtype=bool)
    )
    if mask.shape != values.shape:
        raise ValueError("valid must have shape [T]")
    mask = mask & np.isfinite(values)
    if np.any((values[mask] < 0) | (values[mask] > 1)):
        raise ValueError("valid correctness values must be in [0,1]")
    sessions = (
        np.zeros(len(values), int) if session_ids is None else np.asarray(session_ids)
    )
    if sessions.shape != values.shape:
        raise ValueError("session_ids must have shape [T]")
    sums = np.r_[0.0, np.cumsum(np.where(mask, values, 0.0))]
    counts = np.r_[0, np.cumsum(mask)]
    result = np.full(values.shape, np.nan)
    start = 0
    for t in range(len(values)):
        if t and sessions[t] != sessions[t - 1]:
            start = t
        left = max(start, t - window + 1)
        n = counts[t + 1] - counts[left]
        if n >= min_periods:
            result[t] = (sums[t + 1] - sums[left]) / n
    return result


def predictive_bands(
    samples: Any, window: int = 32, *, valid: Any = None, session_ids: Any = None
) -> dict[str, np.ndarray]:
    """Pointwise 100/90/50 percent empirical bands across rolling paths [R,T]."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("samples must have nonempty repeat dimension [R,T]")
    masks = np.ones(values.shape, bool) if valid is None else np.asarray(valid, bool)
    if masks.shape == values.shape[1:]:
        masks = np.broadcast_to(masks, values.shape)
    if masks.shape != values.shape:
        raise ValueError("valid must have shape [T] or [R,T]")
    rolling = np.stack(
        [
            rolling_accuracy(row, window, valid=mask, session_ids=session_ids)
            for row, mask in zip(values, masks)
        ]
    )
    names = ("q00", "q05", "q25", "q50", "q75", "q95", "q100")
    result = {key: np.full(values.shape[1], np.nan) for key in (*names, "mean")}
    for t in range(values.shape[1]):
        column = rolling[:, t]
        finite = column[np.isfinite(column)]
        if len(finite):
            quantiles = np.quantile(finite, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
            for name, value in zip(names, quantiles):
                result[name][t] = value
            result["mean"][t] = finite.mean()
    return result


def _response_mapping(data: Any) -> np.ndarray:
    mapping = np.asarray(data.category_to_response)
    n = len(data.choices)
    if mapping.ndim != 2 or mapping.shape[0] != n or mapping.shape[1] == 0:
        raise ValueError("category_to_response must have shape [T,C]")
    if mapping.dtype.kind not in "iu" or np.any(mapping < 0):
        raise ValueError(
            "category_to_response must contain zero-based integer responses"
        )
    return mapping


def _target_response(data: Any) -> tuple[np.ndarray, np.ndarray]:
    mapping = _response_mapping(data)
    categories = np.asarray(data.true_categories)
    if categories.shape != (len(mapping),):
        raise ValueError("true_categories must have shape [T]")
    finite = np.isfinite(categories)
    known = finite & (categories >= 0) & (categories < mapping.shape[1])
    known &= np.where(finite, categories, -1) == np.floor(
        np.where(finite, categories, -1)
    )
    target = np.full(len(mapping), -1, int)
    rows = np.flatnonzero(known)
    target[rows] = mapping[rows, categories[rows].astype(int)]
    return target, known


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask & np.isfinite(values)]
    return float(selected.mean()) if len(selected) else None


def behavior_accuracy(
    data: Any, *, choices: Any = None, feedback: Any = None, valid: Any = None
) -> dict[str, Any]:
    """Separate true-response accuracy from *received* positive-feedback rate.

    Feedback accuracy is mean received credit (including condition-3 half
    credit), NOT correctness when feedback is noisy/probabilistic. Full-credit
    rate is also returned with the same denominator. Missing or out-of-range
    feedback is excluded only from its own denominator.
    """
    choices = np.asarray(data.choices if choices is None else choices)
    feedback = np.asarray(data.feedback if feedback is None else feedback, float)
    n = len(data.choices)
    live = np.asarray(data.valid if valid is None else valid, bool)
    score = np.asarray(data.score_mask, bool)
    if any(array.shape != (n,) for array in (choices, feedback, live, score)):
        raise ValueError("choices, feedback and masks must have shape [T]")
    target, known = _target_response(data)
    choice_valid = live & score & known & np.isfinite(choices) & (choices >= 0)
    choice_valid &= choices == np.floor(choices)
    choice_correct = np.where(
        known & (choices >= 0) & np.isfinite(choices),
        (choices == target).astype(float),
        np.nan,
    )
    feedback_correct = np.where(
        np.isfinite(feedback) & (feedback >= 0) & (feedback <= 1), feedback, np.nan
    )
    feedback_valid = live & score & np.isfinite(feedback_correct)
    return dict(
        choice_correct=choice_correct,
        feedback_correct=feedback_correct,
        choice_mask=choice_valid,
        feedback_mask=feedback_valid,
        choice_accuracy=_mean(choice_correct, choice_valid),
        feedback_accuracy=_mean(feedback_correct, feedback_valid),
        full_feedback_rate=_mean((feedback_correct == 1).astype(float), feedback_valid),
        scored_trials=int((live & score).sum()),
        choice_trials=int(choice_valid.sum()),
        feedback_trials=int(feedback_valid.sum()),
        feedback_definition="mean received feedback credit (including partial credit); not necessarily true-response correctness",
    )


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("infinity" if value > 0 else "-infinity")
    if isinstance(value, Path):
        return str(value)
    return value




STATE_MATRICES = {"prior", "filtered_prior", "posterior", "beta", "active_probability",
                  "choice_weights", "dominant_probability"}
STATE_TRACES = {"failure_pressure", "search_probability", "global_probability",
                "active_count", "retained_count", "newcomer_count", "dropped_count",
                "newcomer_mass", "fallback"}
STATE_KEYS = STATE_MATRICES | STATE_TRACES | {"pairing", "proposal_source"}

# Stable catalogue identities are independent of the four semantic kinds.
CATALOGUE_KEYS = {
    "catalogue_prior", "catalogue_filtered_prior", "catalogue_posterior",
    "catalogue_predictive_execution_probability", "catalogue_execution_probability",
    "catalogue_active_probability",
}
CATALOGUE_PREFIXES = ("conditional_", "autonomous_diagnostic_")
CATALOGUE_DEFINITIONS = {
    "catalogue_prior": "Predictive particle-weighted pre-feedback belief mass for each stable rule ID.",
    "catalogue_filtered_prior": "Pre-feedback belief mass weighted by particles filtered on the current choice.",
    "catalogue_posterior": "Post-feedback belief mass weighted by particles filtered on the current choice.",
    "catalogue_predictive_execution_probability": "Predictive pre-current-choice mass of particles executing each rule.",
    "catalogue_execution_probability": "Executed-rule mass weighted by particles filtered on the current choice; distinct from post-feedback belief.",
    "catalogue_active_probability": "Predictive probability that each rule is present; sum is expected catalogue count, bounded by capacity.",
}


def validate_catalogue_probabilities(
    arrays: dict[str, np.ndarray], manifest: dict[str, Any], valid: np.ndarray,
) -> dict[str, Any]:
    """Validate the optional, complete catalogue extension without altering mass."""
    fields = {"catalogue_metadata", "catalogue_axis", "catalogue_state_keys"}
    names = {prefix + key for prefix in CATALOGUE_PREFIXES for key in CATALOGUE_KEYS}
    sources = {
        "conditional_": "Equal-repeat PF diagnostics conditioned on observed history; filtered quantities also use the current observed choice.",
        "autonomous_diagnostic_": "First fixed-seed autonomous path, conditioned on its own generated history and current generated choice; no observed-history replay.",
    }
    common = dict(definitions=CATALOGUE_DEFINITIONS.copy(), sources=sources,
                  normalization="Original mass retained; templates are separate, so rule mass can sum below one.")
    present = bool(fields & manifest.keys() or (names | {"catalogue_rule_ids"}) & arrays.keys())
    if not present:
        return dict(status="not_available", rule_count=None, metadata=None, axis=None, keys=[],
                    reason="No per-catalogue diagnostics saved; semantic means and autonomous path 0 cannot recover conditional PF rule marginals.",
                    **common)
    missing = (fields - manifest.keys()) | ((names | {"catalogue_rule_ids"}) - arrays.keys())
    if missing:
        raise ValueError(f"Incomplete catalogue diagnostics; missing: {sorted(missing)}")
    ids = arrays["catalogue_rule_ids"]
    if (ids.ndim != 1 or ids.dtype != np.dtype("int64") or not len(ids)
            or not np.array_equal(ids, np.arange(len(ids)))):
        raise ValueError("catalogue_rule_ids must be int64[H], contiguous 0..H-1")
    h = len(ids)
    metadata, axis = manifest["catalogue_metadata"], manifest["catalogue_axis"]
    if not isinstance(metadata, list) or len(metadata) != h:
        raise ValueError("catalogue_metadata must have H entries")
    for index, item in enumerate(metadata):
        if (not isinstance(item, dict) or type(item.get("index")) is not int
                or item["index"] != index or item.get("label") != f"H{index}"
                or not isinstance(item.get("family"), str) or not item["family"].strip()
                or not isinstance(item.get("program"), dict) or not item["program"]):
            raise ValueError(f"catalogue_metadata[{index}] must map index/label/family/program to H{index}")
    if (not isinstance(axis, dict) or type(axis.get("rule_count")) is not int
            or axis["rule_count"] != h):
        raise ValueError("catalogue_axis.rule_count must equal H")
    if "axis" in axis and (type(axis["axis"]) is not int or axis["axis"] != 1):
        raise ValueError("catalogue_axis.axis must be 1 for [T,H] arrays")
    for key in CATALOGUE_KEYS:
        if not isinstance(axis.get(key), str) or not axis[key].strip():
            raise ValueError(f"catalogue_axis.{key} must describe timing and conditioning")
    for key, value in axis.items():
        if "timing" in key and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"catalogue_axis.{key} must be a nonempty timing string")
    keys = manifest["catalogue_state_keys"]
    if (not isinstance(keys, list) or not all(isinstance(key, str) for key in keys)
            or len(keys) != len(CATALOGUE_KEYS) or set(keys) != CATALOGUE_KEYS):
        raise ValueError("catalogue_state_keys must list all six catalogue names exactly once")
    capacity = manifest["parameters"].get("capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("catalogue active probability requires parameters.capacity, an integer >= 1")
    semantic_matches = {
        "catalogue_prior": "prior", "catalogue_filtered_prior": "filtered_prior",
        "catalogue_posterior": "posterior",
        "catalogue_predictive_execution_probability": "dominant_probability",
        "catalogue_active_probability": "active_probability",
    }
    for prefix in CATALOGUE_PREFIXES:
        for key in sorted(CATALOGUE_KEYS):
            name = prefix + key
            value = arrays[name]
            if value.shape != (len(valid), h) or value.dtype.kind not in "fiu":
                raise ValueError(f"{name} must be numeric with shape [T,H]=({len(valid)},{h})")
            rows = value[valid]
            if not np.isfinite(rows).all():
                raise ValueError(f"{name} valid rows must contain finite probabilities")
            if (rows < -1e-9).any() or (rows > 1 + 1e-9).any():
                raise ValueError(f"{name} probabilities must be in [0,1]")
            if not np.isnan(value[~valid]).all():
                raise ValueError(f"{name} invalid trial rows must be NaN")
            bound = capacity if key == "catalogue_active_probability" else 1
            if (rows.sum(axis=1) > bound + 1e-9).any():
                raise ValueError(f"{name} row sum exceeds {bound}")
            if key in semantic_matches:
                semantic_name = prefix + semantic_matches[key]
                if semantic_name not in arrays:
                    raise ValueError(f"{name} requires matching semantic array {semantic_name}")
                semantic = arrays[semantic_name]
                if (semantic.shape != (len(valid), 4) or semantic.dtype.kind not in "fiu"
                        or not np.allclose(rows.sum(axis=1), semantic[valid, 3], atol=1e-9, rtol=0)):
                    raise ValueError(f"{name} sum disagrees with {semantic_name} catalogue group 3")
        presence = arrays[prefix + "catalogue_active_probability"][valid]
        for key in ("catalogue_prior", "catalogue_predictive_execution_probability"):
            if (arrays[prefix + key][valid] > presence + 1e-9).any():
                raise ValueError(f"{prefix + key} exceeds predictive catalogue presence")
        if prefix == "autonomous_diagnostic_":
            for left, right in (("catalogue_prior", "catalogue_filtered_prior"),
                                ("catalogue_predictive_execution_probability", "catalogue_execution_probability")):
                if not np.allclose(arrays[prefix+left][valid], arrays[prefix+right][valid], atol=1e-9, rtol=0):
                    raise ValueError(f"{prefix + right} disagrees with single-path catalogue identity")
            for key in ("catalogue_active_probability", "catalogue_execution_probability"):
                indicator = arrays[prefix + key][valid]
                if not np.all(np.isclose(indicator, 0, atol=1e-9, rtol=0) | np.isclose(indicator, 1, atol=1e-9, rtol=0)):
                    raise ValueError(f"{prefix + key} single-path catalogue indicators must be binary")
    return dict(status="available", rule_count=h, metadata=metadata, axis=axis,
                keys=sorted(CATALOGUE_KEYS), rule_ids=ids.tolist(), **common)


def _mask(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if value.shape != shape or value.dtype.kind not in "biu" or not np.isin(value, [0, 1]).all():
        raise ValueError(f"{name} must be a binary mask with shape {shape}")
    return value.astype(bool)


def _probabilities(value: np.ndarray, shape: tuple[int, ...], live: np.ndarray, name: str) -> None:
    if value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have shape {shape}")
    rows = value[live]
    if (not np.isfinite(rows).all() or (rows < 0).any() or
            not np.allclose(rows.sum(axis=-1), 1, atol=1e-9, rtol=0)):
        raise ValueError(f"{name} valid rows must be finite, nonnegative and normalized")


def task_feedback_definition(manifest: dict[str, Any]) -> str:
    """Retain an explicit definition, or the audited frozen C++ schema protocol."""
    if "task_feedback_definition" in manifest:
        definition = manifest["task_feedback_definition"]
        if not isinstance(definition, str) or not definition:
            raise ValueError("task_feedback_definition must retain the true task protocol")
        return definition
    if manifest.get("schema_version") != "gm1031-evaluation-inputs-v1" or manifest.get("condition") not in (1, 2, 3):
        raise ValueError("Explicit task_feedback_definition required for an unknown task protocol")
    definition = "Deterministic task feedback: inverse category_to_response maps chosen response to task category; exact true task category receives 1.0"
    if manifest["condition"] == 3:
        return definition + "; different category in the same task family {0,1} or {2,3} receives 0.5; other family receives 0.0. This is not learner pairing belief or a positive-feedback correctness label."
    return definition + "; all other categories receive 0.0. Received feedback credit and true-response accuracy are reported separately."


def validate_inputs(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> SimpleNamespace:
    """Validate gm1031-evaluation-inputs-v1 before any output is created.

    T=retained trial slots, C=response categories, R=PF repeats, S=autonomous
    repeats, H=4 ordered representation kinds. All IDs are zero based except the
    preserved trial/session labels. No object arrays/pickle are accepted.

    Required NPZ: observed_choices/observed_feedback/true_categories/valid/
    score_mask/conditional_score_mask/session_ids[T], category_to_response[T,C]
    (permutation), stimuli[T,4], trial_ids[T] or [T,K];
    conditional_category_probability_repeats[R,T,C] in RESPONSE coordinates,
    conditional_choice_probability_repeats[R,T] for the observed response;
    autonomous_choices/autonomous_feedback/autonomous_valid[S,T],
    autonomous_category_probabilities[S,T,C]; pf_seeds[R], simulation_seeds[S],
    band_seed[] (uint64 seeds are retained without truncation).

    Optional conditional_<state> is ALREADY aggregated over PF repeats upstream:
    pre-feedback beta averaged per active content, weighted by expected active
    count across repeats; pairing is filtered joint mass [T,4,3]. Optional
    autonomous_diagnostic_<state> is path index 0, never a selected best path.
    State matrices [T,H], traces [T], proposal_source[T,4]; NaN marks skipped
    trials, not fabricated beliefs. Every supplied state array is preserved.

    Manifest requires schema_version, scientific_model_contract, semantic_axis,
    state_metadata[4], subject_id, condition, parameters, config and C++ provenance.
    Optional task_feedback_definition specifies the task feedback protocol.
    Optional catalogue diagnostics use a separate dynamic [T,H] rule axis;
    see docs/catalogue_probabilities.md. Partial extensions are rejected.
    Raw oral reports are externally supplied, unscored metadata.
    """
    required = {"observed_choices", "observed_feedback", "true_categories", "valid", "score_mask",
                "conditional_score_mask", "session_ids", "category_to_response", "stimuli", "trial_ids",
                "conditional_category_probability_repeats", "conditional_choice_probability_repeats",
                "autonomous_choices", "autonomous_feedback", "autonomous_valid",
                "autonomous_category_probabilities", "pf_seeds", "simulation_seeds", "band_seed"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Missing evaluation arrays: {sorted(missing)}")
    for key in ("subject_id", "condition", "parameters", "config", "provenance"):
        if key not in manifest:
            raise ValueError(f"Missing manifest field: {key}")
    if manifest.get("schema_version") != "gm1031-evaluation-inputs-v1":
        raise ValueError("Expected gm1031-evaluation-inputs-v1 manifest")
    if manifest.get("scientific_model_contract") != "1031-unified-v2":
        raise ValueError("Expected final 1031-unified-v2 scientific contract")
    if manifest.get("semantic_axis", {}).get("kinds") != SEMANTIC_KINDS:
        raise ValueError("Expected four ordered semantic representation kinds")
    metadata = manifest.get("state_metadata", [])
    if [item.get("kind") for item in metadata] != SEMANTIC_KINDS:
        raise ValueError("State metadata must align with semantic representation kinds")
    if not isinstance(manifest["provenance"], dict) or not manifest["provenance"]:
        raise ValueError("C++ provenance must be a nonempty mapping")
    task_feedback_definition(manifest)
    if not isinstance(manifest["config"], dict) or not isinstance(manifest["parameters"], dict):
        raise ValueError("config and parameters must be mappings")
    choices = arrays["observed_choices"]
    if choices.ndim != 1 or not len(choices):
        raise ValueError("observed_choices must have nonempty shape [T]")
    n = len(choices)
    mapping = arrays["category_to_response"]
    if mapping.ndim != 2 or mapping.shape[0] != n or mapping.shape[1] == 0:
        raise ValueError("category_to_response must have shape [T,C]")
    c = mapping.shape[1]
    if manifest["condition"] not in (1, 2, 3) or c != (2 if manifest["condition"] == 1 else 4):
        raise ValueError("condition and response category count disagree")
    if mapping.dtype.kind not in "iu" or not np.all(np.sort(mapping, axis=1) == np.arange(c)):
        raise ValueError("category_to_response must be a per-trial response permutation")
    for key in ("observed_feedback", "true_categories", "session_ids"):
        if arrays[key].shape != (n,):
            raise ValueError(f"{key} must have shape [T]")
    for key in ("valid", "score_mask", "conditional_score_mask"):
        arrays[key] = _mask(arrays[key], (n,), key)
    valid = arrays["valid"]
    if choices.dtype.kind not in "iu" or np.any((choices[valid] < 0) | (choices[valid] >= c)):
        raise ValueError("valid observed_choices must be zero-based response IDs")
    stimuli, trials = arrays["stimuli"], arrays["trial_ids"]
    if stimuli.shape != (n, 4) or not np.isfinite(stimuli).all():
        raise ValueError("stimuli must be finite [T,4]")
    if trials.ndim not in (1, 2) or trials.shape[0] != n:
        raise ValueError("trial_ids must have shape [T] or [T,K]")
    probabilities = arrays["conditional_category_probability_repeats"]
    if probabilities.ndim != 3 or not probabilities.shape[0]:
        raise ValueError("conditional_category_probability_repeats must be nonempty [R,T,C]")
    r = probabilities.shape[0]
    _probabilities(probabilities, (r, n, c), np.broadcast_to(valid, (r, n)), "conditional forecasts")
    score = valid & arrays["score_mask"] & arrays["conditional_score_mask"]
    observed_p = arrays["conditional_choice_probability_repeats"]
    if observed_p.shape != (r, n):
        raise ValueError("conditional_choice_probability_repeats must have shape [R,T]")
    rows = np.flatnonzero(score)
    if not np.allclose(observed_p[:, rows], probabilities[:, rows, choices[rows]], atol=1e-9, rtol=0):
        raise ValueError("Observed choice probabilities disagree with response-coordinate forecasts")
    for key, expected in (("conditional_category_probabilities", probabilities.mean(axis=0)),
                          ("conditional_response_probabilities", probabilities.mean(axis=0)),
                          ("conditional_choice_probabilities", observed_p.mean(axis=0))):
        if key in arrays and (arrays[key].shape != expected.shape or
                not np.allclose(arrays[key], expected, atol=1e-9, rtol=0, equal_nan=True)):
            raise ValueError(f"Upstream aggregate {key} disagrees with equal-repeat forecasts")
    simulated = arrays["autonomous_choices"]
    if simulated.ndim != 2 or simulated.shape[1] != n or not simulated.shape[0]:
        raise ValueError("autonomous_choices must have nonempty shape [S,T]")
    s = simulated.shape[0]
    live = _mask(arrays["autonomous_valid"], (s, n), "autonomous_valid")
    arrays["autonomous_valid"] = live
    if simulated.dtype.kind not in "iu" or np.any((simulated[live] < 0) | (simulated[live] >= c)):
        raise ValueError("valid autonomous_choices must be zero-based responses")
    if not np.array_equal(live, np.broadcast_to(valid, live.shape)):
        raise ValueError("autonomous_valid must preserve the exact observed trial schedule")
    if arrays["autonomous_feedback"].shape != (s, n):
        raise ValueError("autonomous_feedback must have shape [S,T]")
    truth = arrays["true_categories"]
    if np.any(~np.isfinite(truth[valid]) | (truth[valid] != np.floor(truth[valid]))
              | (truth[valid] < 0) | (truth[valid] >= c)):
        raise ValueError("valid true_categories must contain task category IDs")
    # Feedback belongs to the fixed task, not to the learner's latent pairing.
    inverse_mapping = np.broadcast_to(np.argsort(mapping, axis=1), (s, n, c))
    chosen_category = np.take_along_axis(inverse_mapping,
        np.where(live, simulated, 0)[..., None], axis=2)[..., 0]
    task_credit = (chosen_category == truth[None, :]).astype(float)
    if manifest["condition"] == 3:
        same_family = chosen_category // 2 == truth[None, :] // 2
        task_credit = np.where((task_credit == 0) & same_family, .5, task_credit)
    if not np.array_equal(arrays["autonomous_feedback"][live], task_credit[live]):
        raise ValueError("autonomous_feedback disagrees with the fixed task protocol")
    _probabilities(arrays["autonomous_category_probabilities"], (s, n, c), live, "autonomous forecasts")
    for key, shape in (("pf_seeds", (r,)), ("simulation_seeds", (s,)), ("band_seed", ())):
        value = arrays[key]
        if value.shape != shape or value.dtype.kind not in "iu" or np.any(value < 0):
            raise ValueError(f"{key} must contain nonnegative integer seeds with shape {shape}")
        if key in manifest and not np.array_equal(value, np.asarray(manifest[key])):
            raise ValueError(f"{key} differs between manifest and NPZ")
    for key, count in (("pf_repeats", r), ("simulation_repeats", s)):
        if key in manifest["config"] and manifest["config"][key] != count:
            raise ValueError(f"config {key} differs from array repeat count")
    if "autonomous_diagnostic_seed" in arrays and not np.array_equal(
            arrays["autonomous_diagnostic_seed"], arrays["simulation_seeds"][0]):
        raise ValueError("autonomous diagnostic must be the first fixed-seed path")
    h = 4
    for prefix in ("conditional_", "autonomous_diagnostic_"):
        for key in sorted(STATE_KEYS):
            if prefix + key not in arrays:
                continue
            value = arrays[prefix + key]
            if value.dtype.kind not in "fiu" or np.isinf(value).any():
                raise ValueError(f"{prefix + key} must be numeric without infinity")
            if key in STATE_MATRICES or key == "pairing":
                if value.ndim != (3 if key == "pairing" else 2):
                    raise ValueError(f"{prefix + key} must have representation-kind dimension")
                h = value.shape[1] if h is None else h
                shape = (n, h, 3) if key == "pairing" else (n, h)
            else:
                shape = (n, 4) if key == "proposal_source" else (n,)
            if value.shape != shape:
                raise ValueError(f"{prefix + key} must have shape {shape}")
    catalogue = validate_catalogue_probabilities(arrays, manifest, valid)
    # Retain composite trial identities alongside externally supplied reports.
    oral_trials = np.empty(n, object)
    oral_trials[:] = [tuple(row.tolist()) for row in trials] if trials.ndim == 2 else trials.tolist()
    return SimpleNamespace(subject_id=manifest["subject_id"], condition=manifest["condition"],
                           choices=choices, feedback=arrays["observed_feedback"], valid=valid,
                           score_mask=arrays["score_mask"], true_categories=arrays["true_categories"],
                           category_to_response=mapping, trial_ids=oral_trials,
                           session_ids=arrays["session_ids"], oral_reports=manifest.get("oral_reports", []),
                           _protocol=manifest.get("protocol", {}), catalogue_probabilities=catalogue)


def load_bundle(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    subject = Path(path)
    if subject.is_file():
        subject = subject.parent
    manifest = json.loads((subject / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if "inputs_sha256" in manifest and hashlib.sha256((subject / "evaluation_inputs.npz").read_bytes()).hexdigest() != manifest["inputs_sha256"]:
        raise ValueError("evaluation_inputs.npz SHA-256 disagrees with manifest")
    with np.load(subject / "evaluation_inputs.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, spec in manifest.get("arrays", {}).items():
        if key not in arrays or list(arrays[key].shape) != spec["shape"] or str(arrays[key].dtype) != spec["dtype"]:
            raise ValueError(f"Manifest array shape/dtype disagrees for {key}")
    return arrays, manifest


def evaluate_bundle(input_path: str | Path, output: str | Path | None = None, *,
                    window: int | None = None, plots: bool = True) -> dict[str, Any]:
    """Score saved C++ outputs into a new gm1031-evaluation-v1 directory."""
    subject = Path(input_path).resolve()
    if subject.is_file():
        subject = subject.parent
    destination = Path(output).resolve() if output is not None else subject / "evaluation"
    if destination.exists():
        raise FileExistsError(destination)
    arrays, manifest = load_bundle(subject)
    data = validate_inputs(arrays, manifest)
    cfg = dict(manifest["config"])
    cfg["window"] = _positive_int(cfg.get("window", 32) if window is None else window, "window")
    cfg["band_draws"] = _positive_int(cfg.get("band_draws", 256), "band_draws")
    cfg["plots"] = plots
    probabilities = arrays["conditional_category_probability_repeats"]
    category_mean = probabilities.mean(axis=0)
    choice_mean = arrays["conditional_choice_probability_repeats"].mean(axis=0)
    score = data.valid & data.score_mask & arrays["conditional_score_mask"]
    n, c = category_mean.shape
    nll, brier = np.full(n, np.nan), np.full(n, np.nan)
    with np.errstate(divide="ignore"):
        nll[score] = -np.log(choice_mean[score])
    rows = np.flatnonzero(score)
    brier[rows] = ((category_mean[rows] - np.eye(c)[data.choices[rows]]) ** 2).sum(axis=1)
    target, known = _target_response(data)
    expected = np.full(n, np.nan)
    rows = np.flatnonzero(known & data.valid)
    expected[rows] = category_mean[rows, target[rows]]
    rng = np.random.default_rng(int(arrays["band_seed"]))
    draws = (rng.random((cfg["band_draws"], n)) < np.nan_to_num(expected, nan=0)).astype(float)
    conditional_bands = predictive_bands(draws, cfg["window"], valid=score & known, session_ids=data.session_ids)
    conditional_bands["mean"] = rolling_accuracy(expected, cfg["window"], valid=score & known,
                                                 session_ids=data.session_ids)
    observed = behavior_accuracy(data)
    simulations = [behavior_accuracy(data, choices=choice, feedback=feedback, valid=valid)
                   for choice, feedback, valid in zip(arrays["autonomous_choices"],
                       arrays["autonomous_feedback"], arrays["autonomous_valid"])]
    choice_samples = np.stack([m["choice_correct"] for m in simulations])
    feedback_samples = np.stack([m["feedback_correct"] for m in simulations])
    choice_masks = np.stack([m["choice_mask"] for m in simulations])
    feedback_masks = np.stack([m["feedback_mask"] for m in simulations])
    autonomous_bands = predictive_bands(choice_samples, cfg["window"], valid=choice_masks, session_ids=data.session_ids)
    feedback_bands = predictive_bands(feedback_samples, cfg["window"], valid=feedback_masks, session_ids=data.session_ids)
    states = {key: arrays[f"conditional_{key}"] for key in STATE_KEYS if f"conditional_{key}" in arrays}
    first_states = {key: arrays[f"autonomous_diagnostic_{key}"] for key in STATE_KEYS
                    if f"autonomous_diagnostic_{key}" in arrays}
    metadata = manifest["state_metadata"]
    oral = dict(status="not_implemented_1031", used_for_fit=False,
                raw_reports=manifest.get("oral_reports", []),
                reason="Raw external reports retained as unscored metadata; semantic groups describe representation kinds")
    arrays.update(conditional_category_probabilities=category_mean,
                  conditional_response_probabilities=category_mean,
                  conditional_choice_probabilities=choice_mean, conditional_choice_nll=nll,
                  conditional_response_brier=brier, conditional_expected_choice_correct=expected,
                  conditional_score_mask=score, observed_choice_correct=observed["choice_correct"],
                  observed_feedback_correct=observed["feedback_correct"],
                  autonomous_choice_correct=choice_samples, autonomous_feedback_correct=feedback_samples,
                  autonomous_diagnostic_seed=arrays["simulation_seeds"][0])
    for name, bands in (("conditional_choice", conditional_bands), ("autonomous_choice", autonomous_bands),
                        ("autonomous_feedback", feedback_bands)):
        arrays.update({f"{name}_{key}": value for key, value in bands.items()})
    provenance = dict(cpp=manifest["provenance"], input_manifest=manifest,
                      input_sha256={name: hashlib.sha256((subject / name).read_bytes()).hexdigest()
                                    for name in ("evaluation_inputs.npz", "evaluation_manifest.json")},
                      offline_evaluator=dict(python=platform.python_version(), numpy=np.__version__,
                          implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                 for name in ("plot_results.py", "figures.py")}),
                      band_seed=int(arrays["band_seed"]),
                      seed_scheme="C++ PF/autonomous seeds retained verbatim; NumPy default_rng only for offline outcome bands")
    summary = dict(schema_version="gm1031-evaluation-v1", subject_id=data.subject_id, condition=data.condition,
                   scientific_model_contract="1031-unified-v2", semantic_axis=manifest["semantic_axis"],
                   content_path=manifest.get("content_path"),
                   catalogue_probabilities=data.catalogue_probabilities,
                   trials=n, parameters=manifest["parameters"], config=cfg,
                   task_feedback_definition=task_feedback_definition(manifest),
                   feedback_definition_source="saved manifest" if "task_feedback_definition" in manifest
                       else "gm1031-evaluation-inputs-v1 task protocol and saved condition",
                   observed={k: v for k, v in observed.items() if not isinstance(v, np.ndarray)},
                   conditional=dict(pf_repeats=len(probabilities), particle_count=cfg.get("particle_count"),
                       scored_trials=int(score.sum()), mean_choice_nll=float(nll[score].mean()) if score.any() else None,
                       mean_response_brier=_mean(brier, score),
                       zero_choice_probability_trials=int((score & (choice_mean == 0)).sum()),
                       aggregation="equal mean of PF forecasts before scoring; never best repeat",
                       conditioning="current stimulus and observed history; pre-current-choice forecast",
                       band_semantics="independent conditional Bernoulli outcomes; pointwise, not joint paths or confidence intervals"),
                   internal_states=dict(status="available" if states else "not_available", keys=sorted(states),
                       semantics="PF semantic-kind marginals; active_probability is expected active count; beta is mean per active item, weighted across repeats by that count",
                       pairing_semantics="filtered joint representation-kind/pairing mass; sum over kinds without prior reweighting"),
                   autonomous=dict(repeats=len(simulations),
                       mean_choice_accuracy=_mean(choice_samples.ravel(), choice_masks.ravel()),
                       mean_feedback_accuracy=_mean(feedback_samples.ravel(), feedback_masks.ravel()),
                       state_diagnostic_seed=int(arrays["simulation_seeds"][0]),
                       state_retention="first fixed-seed path only; never best-of-N selection",
                       band_semantics="all independent autonomous paths; pointwise fixed-parameter rolling quantiles",
                       empirical_100_band="finite-repeat min/max, not full population support",
                       history="own generated choices and task feedback; not observed-history replay"),
                   oral=oral, artifacts=["summary.json", "arrays.npz"], provenance=provenance)
    if plots:
        # Import plotting before reserving output, so absent matplotlib leaves no artifacts.
        from matplotlib.figure import Figure  # noqa: F401
        if __package__:
            from .figures import plot_behavior, plot_states
        else:
            from figures import plot_behavior, plot_states
    destination.mkdir(parents=True, exist_ok=False)
    if plots:
        plot_behavior(data, autonomous_bands, destination / "behavior.png", window=cfg["window"],
                      conditional_bands=conditional_bands, feedback_bands=feedback_bands)
        summary["artifacts"].append("behavior.png")
        for name, values in (("conditional_states", states), ("autonomous_states", first_states)):
            if values:
                plot_states(values, destination / f"{name}.png", metadata=metadata,
                            title="Conditional PF state diagnostics (not forecasts)" if name == "conditional_states"
                            else "Autonomous path 1 (fixed seed; not selected for fit)")
                summary["artifacts"].append(f"{name}.png")
    summary = _json_safe(summary)
    arrays["provenance_json"] = np.asarray(json.dumps(summary["provenance"], ensure_ascii=True, allow_nan=False))
    with (destination / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (destination / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Subject NPZ/directory or parent run directory")
    parser.add_argument("--output", type=Path, help="Distinct NEW destination; parent input retains subject subdirectories")
    parser.add_argument("--window", type=int, help="Override rolling window without running the model")
    parser.add_argument("--no-plots", action="store_true", help="Numerical artifacts only; matplotlib not required")
    args = parser.parse_args(argv)
    source = args.input.resolve()
    if source.is_file():
        source = source.parent
    single = (source / "evaluation_inputs.npz").is_file()
    subjects = [source] if single else sorted(p.parent for p in source.glob("*/evaluation_inputs.npz"))
    if not subjects:
        parser.error("No evaluation_inputs.npz bundles found")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    destinations = [args.output if single else args.output / p.name for p in subjects] if args.output else [p / "evaluation" for p in subjects]
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(destination)
    for subject, destination in zip(subjects, destinations):
        summary = evaluate_bundle(subject, destination, window=args.window, plots=not args.no_plots)
        print(json.dumps(dict(subject_id=summary["subject_id"], output=str(destination))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
