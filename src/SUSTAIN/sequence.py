"""Sequence scoring and generation on a fixed, externally labelled schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .model import Array, SUSTAIN


@dataclass
class SequenceResult:
    probabilities: Array
    log_probabilities: Array
    choices: np.ndarray | None
    negative_log_likelihood: float | None
    traces: list[dict]


def run_sequence(
    model: SUSTAIN,
    stimuli: ArrayLike,
    targets: ArrayLike,
    *,
    choices: ArrayLike | None = None,
    sample_choices: bool = False,
    seed: int = 0,
    trace: bool = False,
    learn_mask: ArrayLike | None = None,
) -> SequenceResult:
    """Continue a supervised history, scoring each choice before its feedback.

    Targets are explicit category feedback. In a deterministic binary task,
    choice plus correct/incorrect feedback uniquely determines the same target.
    Generation samples responses on a fixed schedule, with no adaptive stopping.
    Training boundaries and resets are controlled by the caller.
    """
    if model.n_categories is None:
        raise ValueError("run_sequence requires a supervised model")
    x = np.asarray(stimuli, dtype=float)
    y = np.asarray(targets)
    if x.ndim != 2 or x.shape[1] != sum(model.dimensions) or y.shape != (len(x),):
        raise ValueError("stimuli and targets must have matching trial counts and encoded width")
    if not len(x):
        raise ValueError("sequence must contain at least one trial")
    if sample_choices and choices is not None:
        raise ValueError("provide observed choices or request sampled choices")
    if choices is not None and model.ties != "first":
        raise ValueError("Choice likelihood requires ties: first; random tie histories are not marginalized")
    if not np.isin(y, np.arange(model.n_categories)).all():
        raise ValueError("targets must be zero-based category integers")
    observed = None if choices is None else np.asarray(choices)
    if observed is not None and (observed.shape != y.shape or not np.isin(observed, np.arange(model.n_categories)).all()):
        raise ValueError("choices must be zero-based categories with one value per trial")
    learning = np.ones(len(x), dtype=bool) if learn_mask is None else np.asarray(learn_mask)
    if learning.shape != y.shape or learning.dtype != np.bool_:
        raise ValueError("learn_mask must be a boolean vector with one value per trial")
    # Validate the whole schedule before mutating the model.
    for row in x:
        model._stimulus(row)
    probabilities, log_probabilities, traces = [], [], []
    generated = []
    rng = np.random.default_rng(seed)
    for i, (row, target) in enumerate(zip(x, y)):
        lambdas_before = model.lambdas.copy()
        clusters_before = len(model.clusters)
        result = model.step(row, int(target), learn=bool(learning[i]))
        prediction = result.prediction
        probabilities.append(prediction.probabilities)
        log_probabilities.append(prediction.log_probabilities)
        if sample_choices:
            generated.append(int(rng.choice(model.n_categories, p=prediction.probabilities)))
        if trace:
            traces.append(dict(
                trial=i + 1, prediction_winner=prediction.winner,
                learning_winner=result.learning_winner, recruited=result.recruited,
                learned=bool(learning[i]), n_clusters_before=clusters_before,
                lambdas_before=lambdas_before.tolist(),
                post_feedback=model.state_dict(),
            ))
    probs, log_probs = np.asarray(probabilities), np.asarray(log_probabilities)
    nll = None if observed is None else -float(log_probs[np.arange(len(x)), observed.astype(int)].sum())
    output_choices = np.asarray(generated, dtype=int) if sample_choices else observed
    return SequenceResult(probs, log_probs, output_choices, nll, traces)
