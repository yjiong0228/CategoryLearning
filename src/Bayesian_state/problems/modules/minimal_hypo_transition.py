"""Minimal feedback-driven, fixed-capacity hypothesis transition.

This module implements the ``swap-one`` mechanism specified in
``manuscript/model_newplan.tex``.  It intentionally does not inherit from the
large strategy-controller implementation: the only dynamic parameter is
``theta``, the probability scale for replacing one hypothesis after imperfect
feedback.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .base_module import BaseModule


class FeedbackSwapHypothesisModule(BaseModule):
    """Maintain exactly ``capacity`` active hypotheses.

    On trial ``t > 1`` the module samples

    ``Z_t ~ Bernoulli(theta * (1 - feedback_{t-1}))``.

    If ``Z_t = 1``, one minimum-posterior active hypothesis is dropped and one
    uniformly sampled inactive hypothesis enters.  The newcomer receives prior
    mass ``1 / capacity`` and survivors share the remaining mass in proportion
    to the previous posterior.

    Three random uniforms are consumed on every trial, including trials where
    swapping is impossible.  This keeps paired random-number streams aligned
    across different values of ``theta``.
    """

    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)

        self.total_hypo = int(getattr(engine, "set_size", 0))
        if self.total_hypo <= 0:
            raise ValueError("FeedbackSwapHypothesisModule requires engine.set_size > 0.")

        capacity_raw = kwargs.get(
            "capacity",
            kwargs.get("max_active_hypotheses", kwargs.get("init_num", 5)),
        )
        self.capacity = self._validate_int(capacity_raw, "capacity")
        if not 1 <= self.capacity <= self.total_hypo:
            raise ValueError(
                "capacity must be between 1 and the hypothesis-space size, "
                f"got {self.capacity} for K={self.total_hypo}."
            )
        for alias in ("init_num", "max_active_hypotheses"):
            if alias in kwargs and self._validate_int(kwargs[alias], alias) != self.capacity:
                raise ValueError(f"{alias} must equal capacity for fixed-capacity swap-one.")

        self.theta = self._validate_probability(kwargs.get("theta", 0.0), "theta")
        if self.capacity == self.total_hypo and self.theta > 0.0:
            raise ValueError("theta must be 0 when capacity equals the full hypothesis space.")

        self.tie_atol = float(kwargs.get("tie_atol", 1e-12))
        if not np.isfinite(self.tie_atol) or self.tie_atol < 0.0:
            raise ValueError(f"tie_atol must be a finite non-negative float, got {self.tie_atol!r}.")

        self.module_seed = kwargs.get("module_seed")
        if self.module_seed is not None:
            self.module_seed = int(self.module_seed)
        seed_sequence = np.random.SeedSequence(self.module_seed)
        init_seed, trial_seed = seed_sequence.spawn(2)
        self.init_rng = np.random.default_rng(init_seed)
        self.trial_rng = np.random.default_rng(trial_seed)

        self.full_indices = np.arange(self.total_hypo, dtype=int)
        self.active = self._initialize_active_set(kwargs.get("init_hypotheses"))
        self.old_active = self.active.copy()
        self.previous_feedback: float | None = None
        self.trial_index = 0

        self.transition_log: list[Dict[str, Any]] = []
        # Compatibility with the existing result serializer.
        self.strategy_counts_log = self.transition_log

        self._apply_mask()
        initial_prior = np.zeros(self.total_hypo, dtype=float)
        initial_prior[self.active] = 1.0 / float(self.capacity)
        self.engine.prior = initial_prior

    @staticmethod
    def _validate_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        parsed = int(value)
        if float(value) != float(parsed):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        return parsed

    @staticmethod
    def _validate_probability(value: Any, name: str) -> float:
        parsed = float(value)
        if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{name} must be a finite float in [0, 1], got {value!r}.")
        return parsed

    def _initialize_active_set(self, raw: Sequence[int] | np.ndarray | None) -> np.ndarray:
        if raw is None:
            selected = self.init_rng.choice(
                self.full_indices,
                size=self.capacity,
                replace=False,
            )
            return np.sort(np.asarray(selected, dtype=int))

        selected = np.asarray(raw, dtype=int).reshape(-1)
        if selected.size != self.capacity:
            raise ValueError(
                "init_hypotheses must contain exactly capacity indices, "
                f"got {selected.size} for capacity={self.capacity}."
            )
        if np.unique(selected).size != selected.size:
            raise ValueError("init_hypotheses cannot contain duplicate indices.")
        if np.any(selected < 0) or np.any(selected >= self.total_hypo):
            raise ValueError("init_hypotheses contains indices outside the hypothesis space.")
        return np.sort(selected)

    def _apply_mask(self) -> None:
        mask = np.zeros(self.total_hypo, dtype=float)
        mask[self.active] = 1.0
        self.engine.hypotheses_mask = mask

    def _posterior_for_transition(self) -> np.ndarray:
        raw = getattr(self.engine, "posterior", None)
        if raw is None:
            raw = getattr(self.engine, "prior", None)
        posterior = np.asarray(raw, dtype=float).reshape(-1)
        if posterior.shape[0] != self.total_hypo:
            raise ValueError(
                "Transition posterior width does not match hypothesis space: "
                f"{posterior.shape[0]} vs {self.total_hypo}."
            )
        if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
            raise ValueError("Transition posterior must contain finite non-negative values.")
        active_mass = float(np.sum(posterior[self.active]))
        if active_mass <= 0.0:
            raise ValueError("Transition posterior has zero mass on the active set.")
        normalized = np.zeros(self.total_hypo, dtype=float)
        normalized[self.active] = posterior[self.active] / active_mass
        return normalized

    @staticmethod
    def _uniform_index(uniform: float, size: int) -> int:
        if size <= 0:
            raise ValueError("Cannot sample uniformly from an empty set.")
        return min(int(float(uniform) * int(size)), int(size) - 1)

    def _set_previous_feedback(self, raw_value: Any) -> float:
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Feedback must be a finite value in [0, 1], got {raw_value!r}.")
        self.previous_feedback = value
        return value

    def _record_current_feedback(self) -> float:
        observation = getattr(self.engine, "observation", None)
        if observation is None or len(observation) < 3:
            raise ValueError("FeedbackSwapHypothesisModule requires observation=(stimulus, choice, feedback).")
        return self._set_previous_feedback(observation[2])

    def record_outcome_feedback(self, feedback: float) -> None:
        """Replace the provisional current feedback after autonomous choice generation.

        The ordinary conditional-fitting path never needs this method because
        the observed outcome is already present when ``process`` runs.  A
        generate-then-update loop must prepare the active set before the
        current choice and can call this method once the task has produced the
        actual feedback.
        """
        value = self._set_previous_feedback(feedback)
        if self.transition_log:
            self.transition_log[-1]["current_feedback_recorded"] = float(value)

    def reseed_future(self, module_seed: int) -> None:
        """Assign an independent future transition stream after state resampling."""

        self.module_seed = int(module_seed)
        self.trial_rng = np.random.default_rng(self.module_seed)

    def _initialize_newcomer_beta(self, newcomer: int | None) -> None:
        if newcomer is None:
            return
        beta_mod = getattr(self.engine, "modules", {}).get("beta_mod")
        if beta_mod is None or not hasattr(beta_mod, "initialize_beta_for_hypotheses"):
            return
        # Passing no prior explicitly enforces beta_init even if a legacy
        # configuration accidentally leaves prior scaling enabled.
        beta_mod.initialize_beta_for_hypotheses(
            np.asarray([int(newcomer)], dtype=int),
            priors=None,
        )

    def process(self, **kwargs) -> None:
        del kwargs
        active_before = self.active.copy()
        self.old_active = active_before.copy()
        posterior = self._posterior_for_transition()

        # Consume a fixed number of random variates per trial for paired CRN.
        u_swap, u_drop, u_new = self.trial_rng.random(3)

        feedback_used = self.previous_feedback
        if feedback_used is None:
            swap_probability = 0.0
        else:
            swap_probability = float(np.clip(self.theta * (1.0 - feedback_used), 0.0, 1.0))
        swap_event = bool(feedback_used is not None and u_swap < swap_probability)

        dropped: int | None = None
        newcomer: int | None = None
        tied_minimum: list[int] = []
        fallback_reason: str | None = None

        if swap_event:
            inactive = self.full_indices[~np.isin(self.full_indices, active_before)]
            if inactive.size == 0:
                raise RuntimeError("swap_event occurred with an empty inactive pool.")

            active_posterior = posterior[active_before]
            minimum = float(np.min(active_posterior))
            tie_mask = np.isclose(
                active_posterior,
                minimum,
                rtol=0.0,
                atol=self.tie_atol,
            )
            tied = active_before[tie_mask]
            tied_minimum = [int(value) for value in tied]
            dropped = int(tied[self._uniform_index(u_drop, tied.size)])
            newcomer = int(inactive[self._uniform_index(u_new, inactive.size)])

            survivors = active_before[active_before != dropped]
            self.active = np.sort(
                np.concatenate([survivors, np.asarray([newcomer], dtype=int)])
            )
            new_prior = np.zeros(self.total_hypo, dtype=float)
            newcomer_mass = 1.0 / float(self.capacity)
            new_prior[newcomer] = newcomer_mass
            if survivors.size:
                survivor_values = posterior[survivors]
                survivor_total = float(np.sum(survivor_values))
                if survivor_total <= 0.0:
                    new_prior[survivors] = (1.0 - newcomer_mass) / float(survivors.size)
                    fallback_reason = "zero_survivor_mass"
                else:
                    new_prior[survivors] = (
                        (1.0 - newcomer_mass)
                        * survivor_values
                        / survivor_total
                    )
        else:
            self.active = active_before.copy()
            new_prior = posterior

        if self.active.size != self.capacity or np.unique(self.active).size != self.capacity:
            raise RuntimeError("swap-one violated the fixed-capacity active-set invariant.")
        if not np.isclose(float(np.sum(new_prior)), 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError("swap-one prior is not normalized.")

        self.engine.prior = np.asarray(new_prior, dtype=float)
        self._apply_mask()
        self._initialize_newcomer_beta(newcomer)
        current_feedback = self._record_current_feedback()

        log_item: Dict[str, Any] = {
            "trial_index": int(self.trial_index),
            "theta": float(self.theta),
            "feedback_used": None if feedback_used is None else float(feedback_used),
            "current_feedback_recorded": float(current_feedback),
            "swap_probability": float(swap_probability),
            "swap_event": bool(swap_event),
            "random_uniform_swap": float(u_swap),
            "active_before": [int(value) for value in active_before],
            "tied_minimum": tied_minimum,
            "dropped_hypothesis": dropped,
            "new_hypothesis": newcomer,
            "active_after": [int(value) for value in self.active],
            "active_total": int(self.active.size),
            "newcomer_prior_mass": (
                float(new_prior[newcomer]) if newcomer is not None else 0.0
            ),
            "prior_sum": float(np.sum(new_prior)),
            "fallback": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "strategies": [],
        }
        self.transition_log.append(log_item)
        self.trial_index += 1
