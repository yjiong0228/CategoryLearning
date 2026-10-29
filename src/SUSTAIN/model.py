"""Classification and unsupervised SUSTAIN (Love, Medin & Gureckis, 2004).

Equations 4--14 specify the calculations. ``prediction_error`` follows the
paper's output-based recruitment; ``cluster_label`` follows Gureckis's public
Python demonstration. See README.md for the differences and supported tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Parameters:
    """Four learning/decision parameters and an unsupervised threshold."""

    r: float = 9.01245
    beta: float = 1.252233
    d: float = 16.924073
    eta: float = 0.092327
    tau: float = 0.5

    def __post_init__(self) -> None:
        for name in ("r", "beta", "d", "eta", "tau"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.eta > 1 or self.tau > 1:
            raise ValueError("eta and tau must lie in [0, 1]")


@dataclass(frozen=True)
class Prediction:
    """State readout before the current feedback; cluster indices are zero-based."""

    probabilities: Array
    log_probabilities: Array
    activations: Array
    winner: int | None
    winner_output: float
    category_outputs: Array


@dataclass(frozen=True)
class TrialResult:
    prediction: Prediction
    recruited: bool
    learning_winner: int | None


class SUSTAIN:
    """Incremental learner for nominal or [0, 1] continuous stimulus dimensions.

    ``dimensions`` gives encoded widths: 1 for a continuous scalar, >=2 for
    a nominal one-hot vector. Categories are separate zero-based integers.
    ``n_categories=None`` selects unsupervised clustering. ``predict`` never
    receives the current correct category. ``step`` predicts, then learns.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
        n_categories: int | None = 2,
        parameters: Parameters | None = None,
        *,
        initial_lambdas: ArrayLike | None = None,
        recruitment: str = "prediction_error",
        ties: str = "first",
        seed: int = 0,
    ) -> None:
        dimensions = tuple(dimensions)
        if not dimensions or any(
            isinstance(d, (bool, np.bool_)) or int(d) != d or d < 1 for d in dimensions
        ):
            raise ValueError("dimensions must contain positive integer widths")
        if n_categories is not None and (
            isinstance(n_categories, (bool, np.bool_))
            or int(n_categories) != n_categories or n_categories < 2
        ):
            raise ValueError("n_categories must be at least 2, or None for unsupervised learning")
        if recruitment not in ("prediction_error", "cluster_label"):
            raise ValueError("recruitment must be prediction_error or cluster_label")
        if ties not in ("first", "random"):
            raise ValueError("ties must be first or random")
        self.dimensions = tuple(int(d) for d in dimensions)
        self.n_categories = None if n_categories is None else int(n_categories)
        self.parameters = parameters or Parameters()
        self.recruitment = recruitment
        self.ties = ties
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        edges = np.r_[0, np.cumsum(self.dimensions)]
        self._slices = tuple(slice(a, b) for a, b in zip(edges[:-1], edges[1:]))
        self._initial_lambdas = (
            np.ones(len(self.dimensions)) if initial_lambdas is None
            else np.array(initial_lambdas, dtype=float, copy=True)
        )
        if (self._initial_lambdas.shape != (len(self.dimensions),)
                or not np.isfinite(self._initial_lambdas).all()
                or np.any(self._initial_lambdas < 1)):
            raise ValueError("initial_lambdas must have one finite value >= 1 per dimension")
        self.reset()

    def reset(self) -> None:
        """Reset the learning history and tie-breaking random generator."""
        self.lambdas = self._initial_lambdas.copy()
        self.clusters = np.empty((0, sum(self.dimensions)), dtype=float)
        self.category_positions = np.empty((0, self.n_categories or 0), dtype=float)
        self.weights = np.empty((0, self.n_categories or 0), dtype=float)
        self._rng = np.random.default_rng(self.seed)

    def _stimulus(self, stimulus: ArrayLike) -> Array:
        x = np.asarray(stimulus, dtype=float)
        if x.shape != (sum(self.dimensions),) or not np.isfinite(x).all():
            raise ValueError("stimulus must be a finite vector of the configured encoded width")
        if np.any((x < 0) | (x > 1)):
            raise ValueError("stimulus coordinates must lie in [0, 1]")
        for width, section in zip(self.dimensions, self._slices):
            if width > 1 and (not np.isin(x[section], (0, 1)).all() or x[section].sum() != 1):
                raise ValueError("each nominal dimension must be one-hot encoded")
        return x

    def _target(self, target: int | None) -> int | None:
        if self.n_categories is None:
            if target is not None:
                raise ValueError("unsupervised learning takes no category target")
        elif (target is None or not np.isfinite(target) or isinstance(target, (bool, np.bool_))
              or int(target) != target or not 0 <= target < self.n_categories):
            raise ValueError("supervised learning needs a zero-based category target")
        return None if target is None else int(target)

    def _distances(self, x: Array) -> Array:
        # Eq. 4 is half the L1 distance for nominal probability vectors.
        # Continuous coordinates use absolute distance on an explicit [0, 1] scale.
        return np.column_stack([
            np.abs(self.clusters[:, section] - x[section]).sum(axis=1)
            * (0.5 if width > 1 else 1.0)
            for width, section in zip(self.dimensions, self._slices)
        ])

    def _forward(self, x: Array, forced_winner: int | None = None) -> tuple[Prediction, Array]:
        mu = self._distances(x)
        n = len(self.clusters)
        if n:
            # Log-domain forms of Eqs. 5--6 retain cluster competition when
            # all unscaled activations underflow together.
            log_attention = self.parameters.r * np.log(self.lambdas)
            log_attention -= np.logaddexp.reduce(log_attention)
            log_activations = np.logaddexp.reduce(
                log_attention - mu * self.lambdas, axis=1
            )
            activations = np.exp(log_activations)
            candidates = np.flatnonzero(log_activations == log_activations.max())
            winner = (int(candidates[0]) if self.ties == "first"
                      else int(self._rng.choice(candidates))) if forced_winner is None else forced_winner
            competition = self.parameters.beta * (log_activations - log_activations.max())
            output = float(np.exp(
                log_activations[winner] + competition[winner] - np.logaddexp.reduce(competition)
            ))
            category_outputs = output * self.weights[winner]
        else:
            activations = np.empty(0)
            winner, output = None, 0.0
            category_outputs = np.zeros(self.n_categories or 0)
        if self.n_categories is None:
            probabilities = log_probabilities = np.empty(0)
        else:
            logits = self.parameters.d * category_outputs
            logits -= logits.max()
            log_probabilities = logits - np.logaddexp.reduce(logits)
            probabilities = np.exp(log_probabilities)
        return Prediction(probabilities, log_probabilities, activations, winner, output, category_outputs), mu

    def predict(self, stimulus: ArrayLike) -> Prediction:
        """Predict before feedback; random ties consume only the model's tie RNG."""
        return self._forward(self._stimulus(stimulus))[0]

    def step(self, stimulus: ArrayLike, target: int | None = None, *, learn: bool = True) -> TrialResult:
        """Return pre-feedback probabilities and then apply one feedback update.

        With ``learn=False``, no target is needed and the learned state is frozen.
        In supervised learning ``target`` is feedback identifying the category.
        A sampled or observed response never triggers cluster recruitment itself.
        """
        x = self._stimulus(stimulus)
        target = self._target(target) if learn else target
        before, mu = self._forward(x)
        if not learn:
            return TrialResult(before, False, None)
        recruit = before.winner is None
        if before.winner is not None:
            if self.n_categories is None:
                recruit = bool(before.activations[before.winner] < self.parameters.tau)
            elif self.recruitment == "cluster_label":
                recruit = not np.array_equal(
                    self.category_positions[before.winner], np.eye(self.n_categories)[target]
                )
            else:
                # Eq. 10: a unique maximal output must identify the feedback
                # category. Output ties count as an unresolved prediction.
                values = before.category_outputs
                recruit = bool(values[target] != values.max() or (values == values.max()).sum() != 1)
        learning = before
        if recruit:
            self.clusters = np.vstack((self.clusters, x))
            target_position = np.empty(0) if target is None else np.eye(self.n_categories)[target]
            self.category_positions = np.vstack((self.category_positions, target_position))
            self.weights = np.vstack((self.weights, np.zeros(self.n_categories or 0)))
            # The paper assigns the newly recruited unit to the current item.
            # The author demo instead repeats winner selection, including ties.
            forced = len(self.clusters) - 1 if self.recruitment == "prediction_error" else None
            learning, mu = self._forward(x, forced_winner=forced)
        win = learning.winner
        assert win is not None
        eta = self.parameters.eta
        if self.n_categories is not None:
            target_vector = np.eye(self.n_categories)[target]
            # Eq. 9: the humble teacher gives no penalty for a correct output
            # already beyond its target. Eq. 14 updates the learning winner.
            error = np.where(target_vector == 1,
                             np.maximum(1 - learning.category_outputs, 0),
                             np.minimum(-learning.category_outputs, 0))
            self.weights[win] += eta * error * learning.winner_output
            self.category_positions[win] += eta * (target_vector - self.category_positions[win])
        self.clusters[win] += eta * (x - self.clusters[win])  # Eq. 12
        scaled_distance = self.lambdas * mu[win]
        self.lambdas += eta * np.exp(-scaled_distance) * (1 - scaled_distance)  # Eq. 13
        return TrialResult(before, recruit, win)

    def state_dict(self) -> dict:
        """Copy the post-feedback state into JSON-compatible arrays."""
        return dict(lambdas=self.lambdas.tolist(), clusters=self.clusters.tolist(),
                    category_positions=self.category_positions.tolist(), weights=self.weights.tolist())
