"""Minimal, direct-likelihood models for ``manuscript/model_newplan.tex``.

This module is intentionally separate from the historical C1/B0/D0 simulation
code.  It implements the first model-selection gate in the new analysis plan:

* fixed Sobol integration of Task-1b perceptual uncertainty;
* the R0--R3 rule-belief models with pre-feedback choice predictions;
* four non-rule screening baselines (NR0--NR3);
* subject-wise temporal holdout scoring with frozen parameters.

The implementation is designed for screening/MAP work.  Hierarchical posterior
sampling, RT/oral measurement models, and autonomous posterior-predictive
generation are downstream steps and must not be conflated with these fits.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import norm, qmc

try:  # pragma: no cover - the test environment has numba, fallback is retained.
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate

from ..problems.partitions import Partition
from ..problems.modules.perception import (
    DEFAULT_NORMAL_SUBJECT_IDS,
    DEFAULT_UNIFORM_SUBJECT_IDS,
    _get_perception_noise_stats,
    _get_uniform_threshold_stats,
)


EPS = 1e-12
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")
ORDER_COLUMNS = ("iSession", "iBlock", "iTrial")
CORE_MODEL_NAMES = (
    "NR0",
    "NR1",
    "NR2",
    "NR3",
    "R0",
    "R0K",
    "R1",
    "R2",
    "R3",
)


@dataclass(frozen=True)
class PerceptionSpec:
    """Subject-specific Task-1b noise distribution in Task-2 feature order."""

    mode: str
    location: np.ndarray
    scale: np.ndarray


@dataclass
class PredictionResult:
    """Trialwise predictions and optional latent summaries."""

    probabilities: np.ndarray
    choice_entropy: np.ndarray
    belief_entropy: np.ndarray | None = None
    max_belief: np.ndarray | None = None
    beliefs: np.ndarray | None = None


def n_categories(condition: int) -> int:
    return 2 if int(condition) == 1 else 4


def stable_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    normalizer = logsumexp(values, axis=axis, keepdims=True)
    return np.exp(values - normalizer)


def probability_entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def feedback_compatible_categories(condition: int, choice: int, feedback: float) -> np.ndarray:
    r"""Return zero-based categories compatible with the observed feedback.

    This is the exact :math:`\mathcal Y_k(c,r)` mapping in the analysis plan.
    ``choice`` is zero based here, unlike the CSV representation.
    """

    condition = int(condition)
    choice = int(choice)
    cats = np.arange(n_categories(condition), dtype=np.int64)
    if choice < 0 or choice >= len(cats):
        raise ValueError(f"choice {choice} is invalid for condition {condition}")

    if condition in (1, 2):
        if np.isclose(float(feedback), 1.0):
            return np.array([choice], dtype=np.int64)
        if np.isclose(float(feedback), 0.0):
            return cats[cats != choice]
        raise ValueError(f"condition {condition} feedback must be 0 or 1, got {feedback}")

    if condition == 3:
        if np.isclose(float(feedback), 1.0):
            return np.array([choice], dtype=np.int64)
        family_start = 0 if choice < 2 else 2
        same_family = np.array([family_start, family_start + 1], dtype=np.int64)
        if np.isclose(float(feedback), 0.5):
            return same_family[same_family != choice]
        if np.isclose(float(feedback), 0.0):
            return cats[(cats // 2) != (choice // 2)]
        raise ValueError(f"condition 3 feedback must be 0, 0.5, or 1, got {feedback}")

    raise ValueError(f"unsupported condition: {condition}")


def feedback_target_matrix(condition: int, choices: np.ndarray, feedback: np.ndarray) -> np.ndarray:
    """Return a uniform categorical target over feedback-compatible categories."""

    choices = np.asarray(choices, dtype=np.int64)
    feedback = np.asarray(feedback, dtype=float)
    out = np.zeros((len(choices), n_categories(condition)), dtype=float)
    for trial, (choice, response) in enumerate(zip(choices, feedback)):
        compatible = feedback_compatible_categories(condition, int(choice), float(response))
        out[trial, compatible] = 1.0 / len(compatible)
    return out


def temporal_holdout_mask(subject_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Freeze the last complete block, or the last 25% for a one-block subject."""

    if subject_frame.empty:
        raise ValueError("cannot split an empty subject frame")
    frame = subject_frame.reset_index(drop=True)
    n_trials = len(frame)

    if all(column in frame.columns for column in ("iSession", "iBlock")):
        block_keys = list(
            dict.fromkeys(
                zip(frame["iSession"].astype(int), frame["iBlock"].astype(int))
            )
        )
    else:
        block_keys = []

    mask = np.zeros(n_trials, dtype=bool)
    if len(block_keys) >= 2:
        last_session, last_block = block_keys[-1]
        mask = (
            (frame["iSession"].to_numpy(dtype=int) == last_session)
            & (frame["iBlock"].to_numpy(dtype=int) == last_block)
        )
        method = "last_complete_block"
        boundary = {"iSession": int(last_session), "iBlock": int(last_block)}
    else:
        n_holdout = max(1, int(math.ceil(0.25 * n_trials)))
        mask[-n_holdout:] = True
        method = "last_25_percent"
        boundary = {"start_index": int(n_trials - n_holdout)}

    first_holdout = int(np.flatnonzero(mask)[0])
    if np.any(mask[:first_holdout]) or np.any(~mask[first_holdout:]):
        raise ValueError("temporal holdout must be one contiguous suffix")
    if first_holdout < 1:
        raise ValueError("temporal holdout left no parameter-training trials")

    metadata = {
        "method": method,
        "n_trials": int(n_trials),
        "n_train": int((~mask).sum()),
        "n_holdout": int(mask.sum()),
        "boundary": boundary,
    }
    return mask, metadata


def partition_prior(partition: Partition, mode: str = "uniform_rule") -> np.ndarray:
    """Return either a rule-uniform or split-family-uniform prior."""

    mode = str(mode).strip().lower()
    if mode == "uniform_rule":
        return np.full(partition.length, 1.0 / partition.length, dtype=float)
    if mode != "uniform_family":
        raise ValueError("prior mode must be 'uniform_rule' or 'uniform_family'")

    families = np.asarray([split.type for split in partition.splits], dtype=object)
    unique_families = sorted(set(families.tolist()))
    prior = np.zeros(partition.length, dtype=float)
    for family in unique_families:
        indices = np.flatnonzero(families == family)
        prior[indices] = 1.0 / (len(unique_families) * len(indices))
    return prior


def build_partition(condition: int) -> Partition:
    return Partition(
        n_dims=4,
        n_cats=n_categories(condition),
        include_label_reversals=int(condition) == 1,
    )


def encode_partition_regions(partition: Partition) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable-size region constraints into dense arrays for numba."""

    max_constraints = max(
        int(np.asarray(region["A"]).shape[0])
        for hypothesis in partition.regions
        for region in hypothesis
    )
    A = np.zeros(
        (partition.length, partition.n_cats, max_constraints, partition.n_dims),
        dtype=np.float64,
    )
    b = np.zeros((partition.length, partition.n_cats, max_constraints), dtype=np.float64)
    counts = np.zeros((partition.length, partition.n_cats), dtype=np.int64)
    for h, hypothesis in enumerate(partition.regions):
        for category, region in enumerate(hypothesis):
            region_A = np.asarray(region["A"], dtype=float)
            region_b = np.asarray(region["b"], dtype=float)
            n_rows = int(region_A.shape[0])
            A[h, category, :n_rows] = region_A
            b[h, category, :n_rows] = region_b
            counts[h, category] = n_rows
    return A, b, counts


@njit(cache=True, nogil=True)
def _integrated_region_counts(
    stimuli: np.ndarray,
    noise: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    constraint_counts: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Count region assignments for fixed perceptual draws.

    Region membership is tried first.  The minimum squared positive constraint
    violation is a deterministic fallback for floating-point gaps, matching the
    nearest-region intent of :class:`Partition` without expensive projections.
    """

    n_stimuli = stimuli.shape[0]
    n_draws = noise.shape[0]
    n_hypotheses = A.shape[0]
    n_cats = A.shape[1]
    n_dims = stimuli.shape[1]
    counts = np.zeros((n_stimuli, n_hypotheses, n_cats), dtype=np.int32)
    point = np.empty(n_dims, dtype=np.float64)

    for trial in range(n_stimuli):
        for draw in range(n_draws):
            for dim in range(n_dims):
                value = stimuli[trial, dim] + noise[draw, dim]
                if value < 0.0:
                    value = 0.0
                elif value > 1.0:
                    value = 1.0
                point[dim] = value

            for hypothesis in range(n_hypotheses):
                assigned = -1
                best_category = 0
                best_violation = 1e300
                for category in range(n_cats):
                    inside = True
                    violation = 0.0
                    n_constraints = constraint_counts[hypothesis, category]
                    for constraint in range(n_constraints):
                        value = -b[hypothesis, category, constraint]
                        for dim in range(n_dims):
                            value += A[hypothesis, category, constraint, dim] * point[dim]
                        if value > tolerance:
                            inside = False
                        if value > 0.0:
                            violation += value * value
                    if inside:
                        assigned = category
                        break
                    if violation < best_violation:
                        best_violation = violation
                        best_category = category
                if assigned < 0:
                    assigned = best_category
                counts[trial, hypothesis, assigned] += 1
    return counts


def integrated_rule_probabilities(
    stimuli: np.ndarray,
    noise: np.ndarray,
    region_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    probability_floor: float = 1e-7,
) -> np.ndarray:
    """Integrate category-region probabilities over fixed perceptual draws."""

    stimuli = np.asarray(stimuli, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if stimuli.ndim != 2 or stimuli.shape[1] != 4:
        raise ValueError("stimuli must have shape [trials, 4]")
    if noise.ndim != 2 or noise.shape[1] != 4 or len(noise) < 1:
        raise ValueError("noise must have shape [draws, 4]")
    A, b, constraint_counts = region_arrays
    counts = _integrated_region_counts(stimuli, noise, A, b, constraint_counts, 1e-10)
    probabilities = counts.astype(np.float64) / float(len(noise))
    if probability_floor > 0:
        probabilities = np.maximum(probabilities, float(probability_floor))
        probabilities /= probabilities.sum(axis=2, keepdims=True)
    return probabilities.astype(np.float32)


def subject_seed(base_seed: int, subject_id: int, role: str = "sobol_perception") -> int:
    digest = hashlib.blake2b(
        f"{int(base_seed)}:{int(subject_id)}:{role}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def sobol_noise(spec: PerceptionSpec, n_points: int, seed: int) -> np.ndarray:
    """Generate a nested, scrambled Sobol approximation to Task-1b noise."""

    n_points = int(n_points)
    exponent = int(round(math.log2(n_points)))
    if n_points < 2 or 2**exponent != n_points:
        raise ValueError("n_points must be a power of two")
    unit = qmc.Sobol(d=4, scramble=True, seed=int(seed)).random_base2(exponent)
    if spec.mode == "uniform":
        return (2.0 * unit - 1.0) * np.asarray(spec.scale, dtype=float)
    if spec.mode == "normal":
        safe = np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        return np.asarray(spec.location, dtype=float) + norm.ppf(safe) * np.asarray(
            spec.scale, dtype=float
        )
    raise ValueError(f"unsupported perception mode: {spec.mode}")


def load_perception_specs(
    processed_data_dir: Path | str,
    feature_order_path: Path | str | None = None,
) -> dict[int, PerceptionSpec]:
    """Load all subject noise distributions using the established mappings."""

    processed_data_dir = Path(processed_data_dir).resolve()
    dataset_paths: Mapping[str, Any] | None = None
    if feature_order_path is not None:
        dataset_paths = {
            "processed_dir": processed_data_dir,
            "feature_order_data": Path(feature_order_path).resolve(),
        }
    mean_map, std_map = _get_perception_noise_stats(processed_data_dir, dataset_paths)
    threshold_map = _get_uniform_threshold_stats(processed_data_dir, dataset_paths)
    normal_ids = {int(value) for value in DEFAULT_NORMAL_SUBJECT_IDS}
    uniform_ids = {int(value) for value in DEFAULT_UNIFORM_SUBJECT_IDS}
    all_ids = sorted(normal_ids | uniform_ids)
    specs: dict[int, PerceptionSpec] = {}
    for subject_id in all_ids:
        if subject_id in uniform_ids:
            if subject_id not in threshold_map:
                continue
            specs[subject_id] = PerceptionSpec(
                mode="uniform",
                location=np.zeros(4, dtype=float),
                scale=np.asarray(threshold_map[subject_id], dtype=float),
            )
        elif subject_id in normal_ids and subject_id in mean_map:
            specs[subject_id] = PerceptionSpec(
                mode="normal",
                location=np.asarray(mean_map[subject_id], dtype=float),
                scale=np.asarray(std_map[subject_id], dtype=float),
            )
    return specs


def unique_stimuli(stimuli: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact unique stimulus rows and an inverse map preserving all trials."""

    stimuli = np.ascontiguousarray(np.asarray(stimuli, dtype=np.float64))
    unique, inverse = np.unique(stimuli, axis=0, return_inverse=True)
    return unique, inverse


def rule_predictions(
    q_probabilities: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    condition: int,
    retention: float = 1.0,
    sensitivity: float = 1.0,
    prior: np.ndarray | None = None,
    reset_before: np.ndarray | None = None,
    return_beliefs: bool = False,
) -> PredictionResult:
    """Run the R-family state recursion in strict experimental order."""

    q_values = np.asarray(q_probabilities, dtype=float)
    choices = np.asarray(choices, dtype=np.int64)
    feedback = np.asarray(feedback, dtype=float)
    n_trials, n_hypotheses, n_cats = q_values.shape
    if n_cats != n_categories(condition):
        raise ValueError("q probability category count does not match condition")
    if len(choices) != n_trials or len(feedback) != n_trials:
        raise ValueError("q, choices, and feedback must have matching trial counts")
    if not (0.0 < float(retention) <= 1.0):
        raise ValueError("retention must lie in (0, 1]")
    if not (float(sensitivity) > 0.0):
        raise ValueError("sensitivity must be positive")

    if prior is None:
        prior_values = np.full(n_hypotheses, 1.0 / n_hypotheses, dtype=float)
    else:
        prior_values = np.asarray(prior, dtype=float)
        if prior_values.shape != (n_hypotheses,) or np.any(prior_values <= 0):
            raise ValueError("prior must be a positive vector over hypotheses")
        prior_values = prior_values / prior_values.sum()
    log_prior = np.log(prior_values)

    if reset_before is None:
        reset = np.zeros(n_trials, dtype=bool)
    else:
        reset = np.asarray(reset_before, dtype=bool)
        if reset.shape != (n_trials,):
            raise ValueError("reset_before must have one Boolean per trial")

    probabilities = np.empty((n_trials, n_cats), dtype=float)
    choice_entropy = np.empty(n_trials, dtype=float)
    belief_entropy = np.empty(n_trials, dtype=float)
    max_belief = np.empty(n_trials, dtype=float)
    # Keep optional belief trajectories in float64.  They are used for oral
    # compatible-set log scores, where float32 underflow can turn a very small
    # but finite mass into an artificial exact zero.
    beliefs = np.empty((n_trials, n_hypotheses), dtype=np.float64) if return_beliefs else None
    evidence_plus = np.zeros(n_hypotheses, dtype=float)

    for trial in range(n_trials):
        if trial == 0 or reset[trial]:
            evidence_minus = np.zeros(n_hypotheses, dtype=float)
        else:
            evidence_minus = float(retention) * evidence_plus

        belief = stable_softmax(log_prior + evidence_minus)
        core_probability = belief @ q_values[trial]
        core_probability = np.maximum(core_probability, EPS)
        core_probability /= core_probability.sum()
        if np.isclose(float(sensitivity), 1.0):
            choice_probability = core_probability
        else:
            choice_probability = stable_softmax(float(sensitivity) * np.log(core_probability))
        probabilities[trial] = choice_probability
        choice_entropy[trial] = probability_entropy(choice_probability)
        belief_entropy[trial] = probability_entropy(belief)
        max_belief[trial] = float(np.max(belief))
        if beliefs is not None:
            beliefs[trial] = belief

        compatible = feedback_compatible_categories(
            condition, int(choices[trial]), float(feedback[trial])
        )
        # Index the category axis after selecting the trial.  Using
        # ``q_values[trial, :, compatible]`` invokes NumPy advanced indexing
        # and moves the category index to the first axis, which can silently
        # collapse the hypothesis-specific likelihoods.
        likelihood = q_values[trial][:, compatible].sum(axis=1)
        evidence_plus = evidence_minus + np.log(np.clip(likelihood, EPS, 1.0))

    return PredictionResult(
        probabilities=probabilities,
        choice_entropy=choice_entropy,
        belief_entropy=belief_entropy,
        max_belief=max_belief,
        beliefs=beliefs,
    )


def score_probabilities(probabilities: np.ndarray, choices: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=float)
    choices = np.asarray(choices, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(mask)
    observed_probability = np.clip(probabilities[rows, choices[rows]], EPS, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=float)[choices[rows]]
    confidence = probabilities[rows].max(axis=1)
    predicted = probabilities[rows].argmax(axis=1)
    return {
        "n_trials": int(len(rows)),
        "nll": float(-np.log(observed_probability).sum()),
        "nll_per_trial": float(-np.log(observed_probability).mean()),
        "brier": float(np.mean(np.sum((probabilities[rows] - one_hot) ** 2, axis=1))),
        "accuracy": float(np.mean(predicted == choices[rows])),
        "mean_confidence": float(np.mean(confidence)),
        "mean_entropy": float(np.mean(probability_entropy(probabilities[rows]))),
    }


def _fit_rule_parameters(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    condition: int,
    train_mask: np.ndarray,
    prior: np.ndarray,
    fit_retention: bool,
    fit_sensitivity: bool,
) -> tuple[float, float, dict[str, Any]]:
    train_mask = np.asarray(train_mask, dtype=bool)

    def unpack(raw: np.ndarray) -> tuple[float, float]:
        index = 0
        retention = 1.0
        sensitivity = 1.0
        if fit_retention:
            retention = float(raw[index])
            index += 1
        if fit_sensitivity:
            sensitivity = float(np.exp(raw[index]))
        return retention, sensitivity

    def objective(raw: np.ndarray) -> float:
        retention, sensitivity = unpack(raw)
        result = rule_predictions(
            q_values,
            choices,
            feedback,
            condition,
            retention=retention,
            sensitivity=sensitivity,
            prior=prior,
        )
        return score_probabilities(result.probabilities, choices, train_mask)["nll"]

    if not fit_retention and not fit_sensitivity:
        return 1.0, 1.0, {"optimizer": "none", "success": True, "n_starts": 0}

    retention_starts = [0.25, 0.55, 0.8, 0.95, 1.0] if fit_retention else [None]
    sensitivity_starts = [0.5, 1.0, 2.0, 5.0] if fit_sensitivity else [None]
    starts = []
    for retention_start in retention_starts:
        for sensitivity_start in sensitivity_starts:
            values = []
            if fit_retention:
                values.append(float(retention_start))
            if fit_sensitivity:
                values.append(float(np.log(sensitivity_start)))
            starts.append(np.asarray(values, dtype=float))
    bounds = []
    if fit_retention:
        bounds.append((0.02, 1.0))
    if fit_sensitivity:
        bounds.append((math.log(0.01), math.log(20.0)))

    fits = []
    for start in starts:
        fit = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        fits.append(fit)
    best = min(fits, key=lambda value: float(value.fun))
    retention, sensitivity = unpack(np.asarray(best.x, dtype=float))
    converged_values = sorted(float(fit.fun) for fit in fits if bool(fit.success))
    same_region = sum(abs(float(fit.fun) - float(best.fun)) <= 1e-5 for fit in fits)
    diagnostics = {
        "optimizer": "multi_start_L-BFGS-B",
        "success": bool(best.success),
        "message": str(best.message),
        "n_starts": len(starts),
        "n_same_optimal_region": int(same_region),
        "best_train_nll": float(best.fun),
        "converged_train_nll": converged_values,
        "retention_at_boundary": bool(
            fit_retention and (retention <= 0.020001 or retention >= 0.999999)
        ),
        "sensitivity_at_boundary": bool(
            fit_sensitivity and (sensitivity <= 0.010001 or sensitivity >= 19.999)
        ),
    }
    return retention, sensitivity, diagnostics


def fit_rule_models(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    condition: int,
    train_mask: np.ndarray,
    partition: Partition,
) -> tuple[dict[str, PredictionResult], dict[str, dict[str, Any]]]:
    predictions: dict[str, PredictionResult] = {}
    parameters: dict[str, dict[str, Any]] = {}
    definitions = {
        "R0": ("uniform_rule", False, False),
        # Diagnostic single-mechanism ablation.  R2 is nested after R1 in the
        # planned sequence, but R0K is required to determine whether any R2
        # gain actually needs forgetting or is entirely stable readout noise.
        "R0K": ("uniform_rule", False, True),
        "R1": ("uniform_rule", True, False),
        "R2": ("uniform_rule", True, True),
        "R3": ("uniform_family", True, True),
    }
    for model, (prior_mode, fit_retention, fit_sensitivity) in definitions.items():
        prior = partition_prior(partition, prior_mode)
        retention, sensitivity, diagnostics = _fit_rule_parameters(
            q_values,
            choices,
            feedback,
            condition,
            train_mask,
            prior,
            fit_retention,
            fit_sensitivity,
        )
        predictions[model] = rule_predictions(
            q_values,
            choices,
            feedback,
            condition,
            retention=retention,
            sensitivity=sensitivity,
            prior=prior,
        )
        parameters[model] = {
            "retention": float(retention),
            "sensitivity": float(sensitivity),
            "prior_mode": prior_mode,
            "diagnostics": diagnostics,
        }
    return predictions, parameters


def nr0_predictions(choices: np.ndarray, train_mask: np.ndarray, n_cats: int) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.bincount(np.asarray(choices)[train_mask], minlength=n_cats).astype(float)
    probability = (counts + 0.5) / (counts.sum() + 0.5 * n_cats)
    predictions = np.repeat(probability[None, :], len(choices), axis=0)
    return predictions, {"dirichlet_half_counts": counts.tolist(), "probability": probability.tolist()}


def _feedback_code(condition: int, value: float) -> int:
    if condition == 3:
        if np.isclose(value, 0.0):
            return 0
        if np.isclose(value, 0.5):
            return 1
        return 2
    return 1 if np.isclose(value, 1.0) else 0


def nr1_predictions(
    choices: np.ndarray,
    feedback: np.ndarray,
    condition: int,
    train_mask: np.ndarray,
    base_probability: np.ndarray,
    shrinkage: float = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """First-order choice/feedback transition baseline with fixed shrinkage."""

    n_cats = n_categories(condition)
    n_feedback = 3 if condition == 3 else 2
    counts = np.zeros((n_cats, n_feedback, n_cats), dtype=float)
    for trial in range(1, len(choices)):
        if not train_mask[trial]:
            continue
        state = (int(choices[trial - 1]), _feedback_code(condition, float(feedback[trial - 1])))
        counts[state[0], state[1], int(choices[trial])] += 1.0
    transition = np.empty_like(counts)
    for previous in range(n_cats):
        for response in range(n_feedback):
            row = counts[previous, response]
            transition[previous, response] = (
                row + float(shrinkage) * base_probability
            ) / (row.sum() + float(shrinkage))
    predictions = np.repeat(base_probability[None, :], len(choices), axis=0)
    for trial in range(1, len(choices)):
        predictions[trial] = transition[
            int(choices[trial - 1]), _feedback_code(condition, float(feedback[trial - 1]))
        ]
    return predictions, {
        "shrinkage": float(shrinkage),
        "transition_counts": counts.tolist(),
    }


def nr2_feature_rl_predictions(
    stimuli: np.ndarray,
    choices: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    sensitivity: float,
) -> np.ndarray:
    """Prototype-free feature delta learner under partial feedback labels."""

    stimuli = np.asarray(stimuli, dtype=float)
    features = np.column_stack([np.ones(len(stimuli)), stimuli - 0.5])
    weights = np.zeros((targets.shape[1], features.shape[1]), dtype=float)
    predictions = np.empty((len(stimuli), targets.shape[1]), dtype=float)
    for trial, feature in enumerate(features):
        probability = stable_softmax(float(sensitivity) * (weights @ feature))
        predictions[trial] = probability
        gradient = np.outer(targets[trial] - probability, feature)
        weights += float(learning_rate) * gradient / max(1.0, float(feature @ feature))
    return predictions


@njit(cache=True, nogil=True)
def _nr2_dynamic_readout_predictions_impl(
    stimuli: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    intercept: float,
    practice_slope: float,
    practice: np.ndarray,
) -> np.ndarray:
    n_trials = stimuli.shape[0]
    n_cats = targets.shape[1]
    n_features = stimuli.shape[1] + 1
    weights = np.zeros((n_cats, n_features), dtype=np.float64)
    predictions = np.empty((n_trials, n_cats), dtype=np.float64)
    feature = np.empty(n_features, dtype=np.float64)
    logits = np.empty(n_cats, dtype=np.float64)
    for trial in range(n_trials):
        feature[0] = 1.0
        squared_norm = 1.0
        for dimension in range(stimuli.shape[1]):
            value = stimuli[trial, dimension] - 0.5
            feature[dimension + 1] = value
            squared_norm += value * value
        log_kappa = intercept + practice_slope * practice[trial]
        if log_kappa < math.log(0.01):
            log_kappa = math.log(0.01)
        elif log_kappa > math.log(20.0):
            log_kappa = math.log(20.0)
        kappa = math.exp(log_kappa)
        maximum = -np.inf
        for category in range(n_cats):
            value = 0.0
            for dimension in range(n_features):
                value += weights[category, dimension] * feature[dimension]
            logits[category] = kappa * value
            if logits[category] > maximum:
                maximum = logits[category]
        normalizer = 0.0
        for category in range(n_cats):
            logits[category] = math.exp(logits[category] - maximum)
            normalizer += logits[category]
        for category in range(n_cats):
            probability = logits[category] / normalizer
            predictions[trial, category] = probability
            error = targets[trial, category] - probability
            for dimension in range(n_features):
                weights[category, dimension] += (
                    learning_rate * error * feature[dimension] / squared_norm
                )
    return predictions


def nr2_dynamic_readout_predictions(
    stimuli: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    intercept: float,
    practice_slope: float,
    practice: np.ndarray,
) -> np.ndarray:
    """Feature delta learning with a practice-dependent choice sensitivity.

    Prediction precedes the current target-driven update.  The stable NR2
    model is nested at ``practice_slope=0``.
    """

    stimuli = np.asarray(stimuli, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    practice = np.asarray(practice, dtype=np.float64)
    if stimuli.ndim != 2 or targets.ndim != 2:
        raise ValueError("stimuli and targets must both be two-dimensional")
    if len(stimuli) != len(targets) or len(stimuli) != len(practice):
        raise ValueError("stimuli, targets, and practice must have equal length")
    if not (0.0 < float(learning_rate) < 1.0):
        raise ValueError("learning rate must lie strictly between zero and one")
    if not np.all(np.isfinite(stimuli)) or not np.all(np.isfinite(targets)):
        raise ValueError("stimuli and targets must be finite")
    probabilities = _nr2_dynamic_readout_predictions_impl(
        stimuli,
        targets,
        float(learning_rate),
        float(intercept),
        float(practice_slope),
        practice,
    )
    if not np.all(np.isfinite(probabilities)):
        raise FloatingPointError("dynamic NR2 produced non-finite probabilities")
    return probabilities


def fit_nr2(
    stimuli: np.ndarray,
    choices: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    def objective(raw: np.ndarray) -> float:
        alpha = float(expit(raw[0]))
        sensitivity = float(np.exp(raw[1]))
        probabilities = nr2_feature_rl_predictions(stimuli, choices, targets, alpha, sensitivity)
        return score_probabilities(probabilities, choices, train_mask)["nll"]

    starts = [
        np.array([math.log(a / (1.0 - a)), math.log(k)], dtype=float)
        for a in (0.02, 0.1, 0.35, 0.7)
        for k in (0.5, 1.0, 3.0)
    ]
    bounds = [
        (math.log(1e-4 / (1.0 - 1e-4)), math.log(0.999 / 0.001)),
        (math.log(0.01), math.log(20.0)),
    ]
    fits = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    best = min(fits, key=lambda fit: float(fit.fun))
    learning_rate = float(expit(best.x[0]))
    sensitivity = float(np.exp(best.x[1]))
    predictions = nr2_feature_rl_predictions(
        stimuli, choices, targets, learning_rate, sensitivity
    )
    return predictions, {
        "learning_rate": learning_rate,
        "sensitivity": sensitivity,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "n_starts": len(starts),
        "n_same_optimal_region": int(
            sum(abs(float(fit.fun) - float(best.fun)) <= 1e-5 for fit in fits)
        ),
        "learning_rate_at_boundary": bool(
            learning_rate <= 0.000101 or learning_rate >= 0.9989
        ),
        "sensitivity_at_boundary": bool(
            sensitivity <= 0.010001 or sensitivity >= 19.999
        ),
    }


def squared_distance_matrix(stimuli: np.ndarray) -> np.ndarray:
    stimuli = np.asarray(stimuli, dtype=np.float32)
    norms = np.sum(stimuli * stimuli, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * (stimuli @ stimuli.T)
    return np.maximum(distances, 0.0).astype(np.float32)


def nr3_exemplar_predictions(
    distances: np.ndarray,
    targets: np.ndarray,
    distance_sensitivity: float,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Generalized-context exemplar support from feedback-labelled past trials."""

    n_trials, n_cats = targets.shape
    probabilities = np.empty((n_trials, n_cats), dtype=float)
    base = np.full(n_cats, float(pseudocount), dtype=float)
    for trial in range(n_trials):
        if trial == 0:
            support = base.copy()
        else:
            kernel = np.exp(-float(distance_sensitivity) * distances[trial, :trial])
            support = base + kernel @ targets[:trial]
        probabilities[trial] = support / support.sum()
    return probabilities


def fit_nr3(
    distances: np.ndarray,
    choices: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache: dict[float, float] = {}

    def objective(log_sensitivity: float) -> float:
        key = float(log_sensitivity)
        if key not in cache:
            sensitivity = float(np.exp(key))
            probabilities = nr3_exemplar_predictions(distances, targets, sensitivity)
            cache[key] = score_probabilities(probabilities, choices, train_mask)["nll"]
        return cache[key]

    grid = np.log(np.array([0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]))
    grid_scores = np.array([objective(float(value)) for value in grid])
    best_index = int(np.argmin(grid_scores))
    lower = float(grid[max(0, best_index - 1)])
    upper = float(grid[min(len(grid) - 1, best_index + 1)])
    if lower == upper:
        best_log = float(grid[best_index])
        success = True
    else:
        fit = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
        best_log = float(fit.x)
        success = bool(fit.success)
    sensitivity = float(np.exp(best_log))
    predictions = nr3_exemplar_predictions(distances, targets, sensitivity)
    return predictions, {
        "distance_sensitivity": sensitivity,
        "pseudocount": 1.0,
        "optimizer_success": success,
        "grid_sensitivities": np.exp(grid).tolist(),
        "grid_train_nll": grid_scores.tolist(),
    }


def fit_nonrule_models(
    stimuli: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    condition: int,
    train_mask: np.ndarray,
) -> tuple[dict[str, PredictionResult], dict[str, dict[str, Any]]]:
    n_cats = n_categories(condition)
    predictions: dict[str, PredictionResult] = {}
    parameters: dict[str, dict[str, Any]] = {}

    nr0, parameters["NR0"] = nr0_predictions(choices, train_mask, n_cats)
    predictions["NR0"] = PredictionResult(nr0, probability_entropy(nr0))

    base_probability = nr0[0]
    nr1, parameters["NR1"] = nr1_predictions(
        choices, feedback, condition, train_mask, base_probability
    )
    predictions["NR1"] = PredictionResult(nr1, probability_entropy(nr1))

    targets = feedback_target_matrix(condition, choices, feedback)
    nr2, parameters["NR2"] = fit_nr2(stimuli, choices, targets, train_mask)
    predictions["NR2"] = PredictionResult(nr2, probability_entropy(nr2))

    distances = squared_distance_matrix(stimuli)
    nr3, parameters["NR3"] = fit_nr3(distances, choices, targets, train_mask)
    predictions["NR3"] = PredictionResult(nr3, probability_entropy(nr3))
    return predictions, parameters


def select_by_training_score(
    candidate_names: Sequence[str],
    predictions: Mapping[str, PredictionResult],
    choices: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[str, PredictionResult]:
    scores = {
        model: score_probabilities(predictions[model].probabilities, choices, train_mask)["nll"]
        for model in candidate_names
    }
    selected = min(candidate_names, key=lambda model: (scores[model], model))
    return selected, predictions[selected]


def fit_core_models(
    subject_frame: pd.DataFrame,
    q_values: np.ndarray,
    partition: Partition,
) -> tuple[dict[str, PredictionResult], dict[str, dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Fit all eight screening models and two train-selected family predictors."""

    frame = subject_frame.sort_values(list(ORDER_COLUMNS), kind="stable").reset_index(drop=True)
    holdout_mask, split_metadata = temporal_holdout_mask(frame)
    train_mask = ~holdout_mask
    condition = int(frame["condition"].iloc[0])
    stimuli = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    choices = frame["choice"].to_numpy(dtype=np.int64) - 1
    feedback = frame["feedback"].to_numpy(dtype=float)
    if len(q_values) != len(frame):
        raise ValueError("q cache and subject frame have different trial counts")

    nonrule_predictions, nonrule_parameters = fit_nonrule_models(
        stimuli, choices, feedback, condition, train_mask
    )
    rule_model_predictions, rule_parameters = fit_rule_models(
        q_values, choices, feedback, condition, train_mask, partition
    )
    predictions = {**nonrule_predictions, **rule_model_predictions}
    parameters = {**nonrule_parameters, **rule_parameters}

    nr_selected, nr_prediction = select_by_training_score(
        ("NR0", "NR1", "NR2", "NR3"), predictions, choices, train_mask
    )
    r_selected, r_prediction = select_by_training_score(
        ("R0", "R1", "R2", "R3"), predictions, choices, train_mask
    )
    predictions["NR_SELECT"] = nr_prediction
    predictions["R_SELECT"] = r_prediction
    parameters["NR_SELECT"] = {"selected_model": nr_selected, "selection_data": "training"}
    parameters["R_SELECT"] = {"selected_model": r_selected, "selection_data": "training"}
    return predictions, parameters, holdout_mask, split_metadata


def metric_rows(
    subject_id: int,
    condition: int,
    predictions: Mapping[str, PredictionResult],
    parameters: Mapping[str, Mapping[str, Any]],
    choices: np.ndarray,
    holdout_mask: np.ndarray,
    split_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, result in predictions.items():
        for segment, mask in (("train", ~holdout_mask), ("holdout", holdout_mask)):
            score = score_probabilities(result.probabilities, choices, mask)
            rows.append(
                {
                    "subject_id": int(subject_id),
                    "condition": int(condition),
                    "model": model,
                    "segment": segment,
                    **score,
                    "parameters_json": json.dumps(parameters[model], ensure_ascii=False, sort_keys=True),
                    "split_method": split_metadata["method"],
                    "split_metadata_json": json.dumps(split_metadata, ensure_ascii=False, sort_keys=True),
                }
            )
    return rows


def expected_feedback_from_category(condition: int, choices_one_based: np.ndarray, categories_one_based: np.ndarray) -> np.ndarray:
    choices = np.asarray(choices_one_based, dtype=int) - 1
    categories = np.asarray(categories_one_based, dtype=int) - 1
    if condition in (1, 2):
        return (choices == categories).astype(float)
    exact = choices == categories
    same_family = (choices // 2) == (categories // 2)
    return np.where(exact, 1.0, np.where(same_family, 0.5, 0.0))


def audit_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Run plan-critical data and rule-space integrity checks."""

    required = {
        "iSub",
        "condition",
        *ORDER_COLUMNS,
        *FEATURE_COLUMNS,
        "category",
        "choice",
        "feedback",
        "choRT",
        "text",
        "oral_center",
        "oral_A",
        "oral_b",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Task2 data are missing required columns: {missing}")

    payload: dict[str, Any] = {
        "n_rows": int(len(data)),
        "n_subjects": int(data["iSub"].nunique()),
        "conditions": {},
        "issues": [],
    }
    target_hypothesis = {1: 0, 2: 42, 3: 42}
    for condition in (1, 2, 3):
        frame = data[data["condition"] == condition].copy()
        partition = build_partition(condition)
        stimuli = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        category = frame["category"].to_numpy(dtype=int) - 1
        target_prediction = partition._get_category_assignments_region(
            target_hypothesis[condition], stimuli
        )
        expected_feedback = expected_feedback_from_category(
            condition,
            frame["choice"].to_numpy(dtype=int),
            frame["category"].to_numpy(dtype=int),
        )
        feedback_mismatch = ~np.isclose(expected_feedback, frame["feedback"].to_numpy(dtype=float))
        target_mismatch = target_prediction != category
        condition_payload = {
            "n_rows": int(len(frame)),
            "n_subjects": int(frame["iSub"].nunique()),
            "trial_count_min": int(frame.groupby("iSub").size().min()),
            "trial_count_max": int(frame.groupby("iSub").size().max()),
            "n_categories": int(n_categories(condition)),
            "n_hypotheses": int(partition.length),
            "target_hypothesis": int(target_hypothesis[condition]),
            "target_category_match_rate": float(np.mean(~target_mismatch)),
            "target_category_mismatch_rows": int(target_mismatch.sum()),
            "recorded_vs_category_feedback_mismatch_rows": int(feedback_mismatch.sum()),
            "feedback_values": sorted(float(value) for value in frame["feedback"].unique()),
            "missing_rt": int(frame["choRT"].isna().sum()),
            "missing_text": int(frame["text"].isna().sum()),
            "missing_oral_encoding": int(
                frame[["oral_center", "oral_A", "oral_b"]].isna().any(axis=1).sum()
            ),
        }
        if target_mismatch.any() or feedback_mismatch.any():
            affected = frame.loc[target_mismatch | feedback_mismatch, ["iSub", *ORDER_COLUMNS]]
            condition_payload["affected_subjects"] = sorted(
                int(value) for value in affected["iSub"].unique()
            )
            condition_payload["affected_sessions"] = [
                {"iSub": int(subject), "iSession": int(session), "n_rows": int(len(group))}
                for (subject, session), group in affected.groupby(["iSub", "iSession"])
            ]
            payload["issues"].append(
                {
                    "condition": condition,
                    "type": "category_or_feedback_inconsistency",
                    "affected_subjects": condition_payload["affected_subjects"],
                    "note": (
                        "Core one-step fits use the recorded feedback actually delivered to the participant; "
                        "category-derived autonomous feedback must not be used for affected rows until resolved."
                    ),
                }
            )
        payload["conditions"][str(condition)] = condition_payload
    return payload
