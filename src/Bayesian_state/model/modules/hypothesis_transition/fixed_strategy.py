"""Subject-fixed hypothesis-transition strategies.

"Static" means that a subject uses one fixed selection rule and one fixed
prior-assignment rule throughout the experiment.  Those rules may react to the
current posterior, entropy, confidence, or feedback history; their identity
and fitted parameters do not change across trials.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..base_module import BaseModule, ModuleRole
from .contracts import (
    HypothesisSelection,
    TransitionContext,
    TwoStepHypothesisTransitionMixin,
)
from .execution import WorkspaceTransitionExecutionMixin
from .workspace import AdaptiveWorkspaceController
from .selection import HypothesisSelectionPolicy


@dataclass(frozen=True)
class FixedHypothesisStrategySpace:
    """Named strategy families currently implemented by the model layer."""

    selection_families: tuple[str, ...] = (
        "strategy_chain",
        "bounded_workspace",
        "feedback_swap_one",
    )
    amount_families: tuple[str, ...] = (
        "fixed",
        "entropy",
        "opposite_entropy",
        "normalized_entropy",
        "opposite_normalized_entropy",
        "confidence",
        "opposite_confidence",
        "recent_accuracy_inverse",
        "accuracy_delta",
        "opposite_accuracy_delta",
        "random",
        "post_error_explore",
    )
    hypothesis_selectors: tuple[str, ...] = (
        "top_posterior",
        "random_posterior",
        "random",
        "ksimilar_centers",
        "epsilon_posterior",
        "temperature_posterior",
        "low_posterior",
    )
    prior_assignments: tuple[str, ...] = (
        "similarity_novelty",
        "conservative_carryover",
        "error_boost_newcomers",
        "stochastic_reset",
        "pairwise_mass_transfer",
        "fixed_newcomer_mass",
    )


FIXED_STRATEGY_SPACE = FixedHypothesisStrategySpace()


def _normalize_prior_assignment(spec: Mapping[str, Any]) -> dict[str, Any]:
    method = str(spec.get("method", ""))
    if method not in FIXED_STRATEGY_SPACE.prior_assignments:
        raise ValueError(
            "prior_assignment.method must be one of "
            f"{FIXED_STRATEGY_SPACE.prior_assignments}, got {method!r}."
        )
    params = spec.get("params", {}) or {}
    if not isinstance(params, Mapping):
        raise ValueError("prior_assignment.params must be a mapping.")
    config = dict(params)
    config["method"] = method
    for key, value in spec.items():
        if key not in {"method", "params"}:
            config[key] = value
    return config


def _strategy_chain_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    method = str(spec.get("method", "strategy_chain"))
    if method != "strategy_chain":
        raise ValueError(
            "FixedStrategyHypothesisTransitionModule requires "
            "selection_strategy.method='strategy_chain'. Use "
            "FixedWorkspaceHypothesisTransitionModule for "
            "selection_strategy.method='bounded_workspace'."
        )
    strategies = spec.get("strategies", spec.get("steps"))
    if not isinstance(strategies, list) or not strategies:
        raise ValueError(
            "strategy_chain selection requires a non-empty strategies/steps list."
        )
    translated = {"strategies": strategies}
    for key in (
        "init_num",
        "init_hypotheses",
        "max_active_hypotheses",
    ):
        if key in spec:
            translated[key] = spec[key]
    return translated


class FixedStrategyHypothesisTransitionModule(
    TwoStepHypothesisTransitionMixin,
    HypothesisSelectionPolicy,
):
    """Execute one subject-fixed strategy chain through the common two steps."""

    strategy_mode = "static"

    def __init__(self, engine, **kwargs):
        resolved = dict(kwargs)
        if "module_seed" not in resolved and "random_seed" in resolved:
            resolved["module_seed"] = resolved["random_seed"]
        if "state_controller" in resolved:
            raise ValueError(
                "FixedStrategyHypothesisTransitionModule does not accept a state "
                "controller. Use DynamicDiscreteStrategyHypothesisTransitionModule."
            )

        selection_spec = resolved.pop("selection_strategy", None)
        if selection_spec is not None:
            if not isinstance(selection_spec, Mapping):
                raise ValueError("selection_strategy must be a mapping.")
            resolved.update(_strategy_chain_kwargs(selection_spec))

        prior_spec = resolved.pop("prior_assignment", None)
        if prior_spec is not None:
            if not isinstance(prior_spec, Mapping):
                raise ValueError("prior_assignment must be a mapping.")
            prior_config = _normalize_prior_assignment(prior_spec)
            if prior_config["method"] == "pairwise_mass_transfer":
                raise ValueError(
                    "pairwise_mass_transfer requires a selection strategy that "
                    "returns dropped-newcomer pairs; use the bounded-workspace policy."
                )
            resolved["post_to_prior"] = prior_config

        super().__init__(engine, **resolved)
        # Keep the legacy log object while exposing the common name consumed
        # by inference backends.  Both attributes intentionally reference the
        # same list.
        self.transition_log = self.strategy_counts_log

    def select_hypotheses(
        self,
        context: TransitionContext,
        **kwargs,
    ) -> HypothesisSelection:
        del context
        self._transition(**kwargs)
        return HypothesisSelection.from_active_sets(
            self.old_active,
            self.active,
            diagnostics={"strategy_mode": self.strategy_mode},
        )

    def assign_prior(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        **kwargs,
    ) -> np.ndarray:
        del context, selection, kwargs
        self._posterior_to_prior_transition()
        return np.asarray(self.engine.prior, dtype=float)

    def _finish_hypothesis_transition(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        prior: np.ndarray,
        **kwargs,
    ) -> Mapping[str, Any]:
        del context, selection, prior, kwargs
        latest = self.strategy_counts_log[-1] if self.strategy_counts_log else {}
        # The strategy-chain implementation predates the particle-filter
        # diagnostics contract.  Expose the realized transition through the
        # same descriptive fields used by newer workspace controllers.  These
        # values only annotate an already completed transition; they do not
        # affect selection, prior assignment, or choice probabilities.
        strategies = list(latest.get("strategies", ()))
        retained_count = int(
            sum(
                int(step.get("selected_count", 0))
                for step in strategies
                if str(step.get("pool")) == "active"
            )
        )
        explored_count = int(
            sum(
                int(step.get("selected_count", 0))
                for step in strategies
                if str(step.get("pool")) == "inactive"
            )
        )
        active_before = np.asarray(self.old_active, dtype=int).reshape(-1)
        active_after = np.asarray(self.active, dtype=int).reshape(-1)
        newcomer_count = int(np.sum(~np.isin(active_after, active_before)))
        changed = not np.array_equal(np.sort(active_before), np.sort(active_after))
        denominator = max(int(active_before.size), 1)
        replacement_fraction = float(newcomer_count / denominator)
        realized_indicator = float(changed)
        latest.update(
            {
                "swap_probability": realized_indicator,
                "swap_event": bool(changed),
                "predictive_m": replacement_fraction,
                "predictive_g": float(explored_count > 0),
                "replacement_count": newcomer_count,
                "replacement_fraction": replacement_fraction,
                "retained_count": retained_count,
                "explored_count": explored_count,
                "diagnostic_probability_semantics": "realized_particle_indicator",
            }
        )
        latest["strategy_mode"] = self.strategy_mode
        return latest


class FixedWorkspaceHypothesisTransitionModule(
    WorkspaceTransitionExecutionMixin,
    AdaptiveWorkspaceController,
):
    """Fixed-capacity strategy with subject-fixed ``m`` and ``g`` controls."""

    strategy_mode = "static"

    def __init__(self, engine, **kwargs):
        resolved = dict(kwargs)
        if "module_seed" not in resolved and "random_seed" in resolved:
            resolved["module_seed"] = resolved["random_seed"]
        selection_spec = resolved.pop("selection_strategy", None)
        if selection_spec is not None:
            if not isinstance(selection_spec, Mapping):
                raise ValueError("selection_strategy must be a mapping.")
            method = str(selection_spec.get("method", ""))
            if method != "bounded_workspace":
                raise ValueError(
                    "FixedWorkspaceHypothesisTransitionModule requires "
                    "selection_strategy.method='bounded_workspace'."
                )
            for key, value in selection_spec.items():
                if key != "method":
                    resolved[key] = value

        prior_spec = resolved.get("prior_assignment")
        if prior_spec is not None:
            if not isinstance(prior_spec, Mapping):
                raise ValueError("prior_assignment must be a mapping.")
            method = str(prior_spec.get("method", ""))
            if method not in self.VALID_PRIOR_ASSIGNMENTS:
                raise ValueError(
                    "The bounded-workspace strategy requires a supported "
                    "bounded-workspace prior_assignment method."
                )

        super().__init__(engine, **resolved)
        if self.dynamic_rate or self.dynamic_range:
            raise ValueError(
                "FixedWorkspaceHypothesisTransitionModule requires fixed "
                "m and g. Use DynamicAdaptiveControlHypothesisTransitionModule when "
                "m_t or g_t changes across trials."
            )
        self._pending_transition: dict[str, Any] | None = None


class FixedFeedbackSwapHypothesisTransitionModule(
    TwoStepHypothesisTransitionMixin,
    BaseModule,
):
    """Fixed feedback-gated swap-one strategy used by active-set analyses.

    ``theta`` is a subject-fixed parameter.  Feedback changes the transition
    outcome, but there is no trial-level strategy state or continuous controller
    state, so this is a static reactive strategy under the package's formal
    mode definition.
    """

    strategy_mode = "static"

    def __init__(self, engine, **kwargs):
        resolved = dict(kwargs)
        if "module_seed" not in resolved and "random_seed" in resolved:
            resolved["module_seed"] = resolved["random_seed"]
        selection_spec = resolved.pop("selection_strategy", None)
        if selection_spec is not None:
            if not isinstance(selection_spec, Mapping):
                raise ValueError("selection_strategy must be a mapping.")
            if str(selection_spec.get("method", "")) != "feedback_swap_one":
                raise ValueError(
                    "FixedFeedbackSwapHypothesisTransitionModule requires "
                    "selection_strategy.method='feedback_swap_one'."
                )
            for key, value in selection_spec.items():
                if key != "method":
                    resolved[key] = value
        prior_spec = resolved.pop("prior_assignment", None)
        if prior_spec is not None:
            if not isinstance(prior_spec, Mapping):
                raise ValueError("prior_assignment must be a mapping.")
            if str(prior_spec.get("method", "")) != "fixed_newcomer_mass":
                raise ValueError(
                    "feedback_swap_one currently requires "
                    "prior_assignment.method='fixed_newcomer_mass'."
                )

        super().__init__(engine, **resolved)
        self.total_hypo = int(getattr(engine, "set_size", 0))
        if self.total_hypo <= 0:
            raise ValueError(
                "FixedFeedbackSwapHypothesisTransitionModule requires "
                "engine.set_size > 0."
            )
        capacity_raw = resolved.get(
            "capacity",
            resolved.get("max_active_hypotheses", resolved.get("init_num", 5)),
        )
        self.capacity = self._validate_int(capacity_raw, "capacity")
        if not 1 <= self.capacity <= self.total_hypo:
            raise ValueError(
                "capacity must be between 1 and the hypothesis-space size, "
                f"got {self.capacity} for K={self.total_hypo}."
            )
        for alias in ("init_num", "max_active_hypotheses"):
            if alias in resolved and self._validate_int(resolved[alias], alias) != self.capacity:
                raise ValueError(f"{alias} must equal capacity for feedback_swap_one.")

        self.theta = self._validate_probability(resolved.get("theta", 0.0), "theta")
        if self.capacity == self.total_hypo and self.theta > 0.0:
            raise ValueError("theta must be 0 when capacity equals the full hypothesis space.")
        self.tie_atol = float(resolved.get("tie_atol", 1e-12))
        if not np.isfinite(self.tie_atol) or self.tie_atol < 0.0:
            raise ValueError("tie_atol must be finite and non-negative.")

        self.module_seed = resolved.get("module_seed")
        if self.module_seed is not None:
            self.module_seed = int(self.module_seed)
        seed_sequence = np.random.SeedSequence(self.module_seed)
        init_seed, trial_seed = seed_sequence.spawn(2)
        self.init_rng = np.random.default_rng(init_seed)
        self.trial_rng = np.random.default_rng(trial_seed)

        self.full_indices = np.arange(self.total_hypo, dtype=int)
        self.active = self._initialize_active_set(resolved.get("init_hypotheses"))
        self.old_active = self.active.copy()
        self.previous_feedback: float | None = None
        self.trial_index = 0
        self.transition_log: list[dict[str, Any]] = []
        self.strategy_counts_log = self.transition_log
        self._pending_transition: dict[str, Any] | None = None

        initial_prior = np.zeros(self.total_hypo, dtype=float)
        initial_prior[self.active] = 1.0 / float(self.capacity)
        self.engine.prior = initial_prior
        self._apply_mask()

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
            raise ValueError(f"{name} must be a finite float in [0, 1].")
        return parsed

    def _initialize_active_set(
        self,
        raw: Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        if raw is None:
            selected = self.init_rng.choice(
                self.full_indices,
                size=self.capacity,
                replace=False,
            )
            return np.sort(np.asarray(selected, dtype=int))
        selected = np.asarray(raw, dtype=int).reshape(-1)
        if selected.size != self.capacity:
            raise ValueError("init_hypotheses must contain exactly capacity indices.")
        if np.unique(selected).size != selected.size:
            raise ValueError("init_hypotheses cannot contain duplicate indices.")
        if np.any(selected < 0) or np.any(selected >= self.total_hypo):
            raise ValueError("init_hypotheses contains out-of-range indices.")
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
        if posterior.shape != (self.total_hypo,):
            raise ValueError("transition posterior width does not match hypothesis space.")
        if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
            raise ValueError("transition posterior must be finite and non-negative.")
        active_mass = float(np.sum(posterior[self.active]))
        if active_mass <= 0.0:
            raise ValueError("transition posterior has zero mass on the active set.")
        normalized = np.zeros(self.total_hypo, dtype=float)
        normalized[self.active] = posterior[self.active] / active_mass
        return normalized

    @staticmethod
    def _uniform_index(uniform: float, size: int) -> int:
        if size <= 0:
            raise ValueError("cannot sample uniformly from an empty set.")
        return min(int(float(uniform) * int(size)), int(size) - 1)

    def select_hypotheses(
        self,
        context: TransitionContext,
        **kwargs,
    ) -> HypothesisSelection:
        del kwargs
        posterior = self._posterior_for_transition()
        u_swap, u_drop, u_new = self.trial_rng.random(3)
        feedback_used = self.previous_feedback
        swap_probability = (
            0.0
            if feedback_used is None
            else float(np.clip(self.theta * (1.0 - feedback_used), 0.0, 1.0))
        )
        swap_event = bool(feedback_used is not None and u_swap < swap_probability)

        active_after = context.active_before.copy()
        dropped: int | None = None
        newcomer: int | None = None
        tied_minimum: list[int] = []
        if swap_event:
            inactive = self.full_indices[
                ~np.isin(self.full_indices, context.active_before)
            ]
            if inactive.size == 0:
                raise RuntimeError("swap event occurred with an empty inactive pool.")
            active_posterior = posterior[context.active_before]
            minimum = float(np.min(active_posterior))
            tied = context.active_before[
                np.isclose(active_posterior, minimum, rtol=0.0, atol=self.tie_atol)
            ]
            tied_minimum = tied.astype(int).tolist()
            dropped = int(tied[self._uniform_index(u_drop, tied.size)])
            newcomer = int(inactive[self._uniform_index(u_new, inactive.size)])
            survivors = context.active_before[context.active_before != dropped]
            active_after = np.sort(
                np.concatenate([survivors, np.asarray([newcomer], dtype=int)])
            )

        pairs = () if dropped is None else ((dropped, int(newcomer)),)
        self._pending_transition = {
            "posterior": posterior,
            "feedback_used": feedback_used,
            "swap_probability": swap_probability,
            "swap_event": swap_event,
            "u_swap": float(u_swap),
            "tied_minimum": tied_minimum,
            "dropped": dropped,
            "newcomer": newcomer,
            "fallback_reason": None,
        }
        return HypothesisSelection.from_active_sets(
            context.active_before,
            active_after,
            replacement_pairs=pairs,
            diagnostics={"strategy_mode": self.strategy_mode},
        )

    def assign_prior(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        **kwargs,
    ) -> np.ndarray:
        del context, kwargs
        if self._pending_transition is None:
            raise RuntimeError("feedback swap has no pending selection result.")
        posterior = np.asarray(self._pending_transition["posterior"], dtype=float)
        newcomer = self._pending_transition["newcomer"]
        if newcomer is None:
            return posterior.copy()

        new_prior = np.zeros(self.total_hypo, dtype=float)
        newcomer_mass = 1.0 / float(self.capacity)
        new_prior[int(newcomer)] = newcomer_mass
        survivors = selection.survivors
        if survivors.size:
            survivor_values = posterior[survivors]
            survivor_total = float(np.sum(survivor_values))
            if survivor_total <= 0.0:
                new_prior[survivors] = (1.0 - newcomer_mass) / float(survivors.size)
                self._pending_transition["fallback_reason"] = "zero_survivor_mass"
            else:
                new_prior[survivors] = (
                    (1.0 - newcomer_mass) * survivor_values / survivor_total
                )
        return new_prior

    def _set_previous_feedback(self, raw_value: Any) -> float:
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("feedback must be a finite value in [0, 1].")
        self.previous_feedback = value
        return value

    def record_outcome_feedback(self, feedback: float) -> None:
        value = self._set_previous_feedback(feedback)
        if self.transition_log:
            self.transition_log[-1]["current_feedback_recorded"] = value

    def record_outcome(
        self,
        observation: tuple[np.ndarray, int, float],
    ) -> None:
        """Record the completed outcome after an autonomous choice."""

        if observation is None or len(observation) < 3:
            raise ValueError(
                "feedback_swap_one requires observation=(stimulus, choice, feedback)."
            )
        self.record_outcome_feedback(float(observation[2]))

    def _initialize_newcomer_beta(self, newcomer: int | None) -> None:
        if newcomer is None:
            return
        beta_mod = self.engine.get_module(ModuleRole.BETA)
        if beta_mod is not None and hasattr(beta_mod, "initialize_beta_for_hypotheses"):
            beta_mod.initialize_beta_for_hypotheses(
                np.asarray([int(newcomer)], dtype=int),
                priors=None,
            )

    def _finish_hypothesis_transition(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        prior: np.ndarray,
        **kwargs,
    ) -> Mapping[str, Any]:
        if self._pending_transition is None:
            raise RuntimeError("feedback swap has no pending state to log.")
        pending = self._pending_transition
        newcomer = pending["newcomer"]
        self._initialize_newcomer_beta(newcomer)
        fallback_reason = pending["fallback_reason"]
        event: dict[str, Any] = {
            "trial_index": int(context.trial_index),
            "strategy_mode": self.strategy_mode,
            "transition_method": "feedback_swap_one",
            "theta": float(self.theta),
            "feedback_used": pending["feedback_used"],
            "current_feedback_recorded": None,
            "swap_probability": float(pending["swap_probability"]),
            "swap_event": bool(pending["swap_event"]),
            "random_uniform_swap": float(pending["u_swap"]),
            "active_before": selection.active_before.astype(int).tolist(),
            "tied_minimum": list(pending["tied_minimum"]),
            "dropped_hypothesis": pending["dropped"],
            "new_hypothesis": newcomer,
            "dropped_hypotheses": selection.dropped.astype(int).tolist(),
            "new_hypotheses": selection.newcomers.astype(int).tolist(),
            "active_after": selection.active_after.astype(int).tolist(),
            "active_total": int(selection.active_after.size),
            "newcomer_prior_mass": (
                float(prior[int(newcomer)]) if newcomer is not None else 0.0
            ),
            "prior_sum": float(np.sum(prior)),
            "fallback": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "strategies": [],
        }
        self.transition_log.append(event)
        self.trial_index += 1
        self._pending_transition = None
        return event

    def reseed_future(self, module_seed: int) -> None:
        self.module_seed = int(module_seed)
        self.trial_rng = np.random.default_rng(self.module_seed)

    def state_dict(self) -> dict[str, Any]:
        return {
            "active": self.active.copy(),
            "old_active": self.old_active.copy(),
            "previous_feedback": self.previous_feedback,
            "trial_index": int(self.trial_index),
            "trial_rng_state": deepcopy(self.trial_rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.active = np.asarray(state["active"], dtype=int).copy()
        self.old_active = np.asarray(state["old_active"], dtype=int).copy()
        previous_feedback = state.get("previous_feedback")
        self.previous_feedback = (
            None if previous_feedback is None else float(previous_feedback)
        )
        self.trial_index = int(state["trial_index"])
        rng_state = state.get("trial_rng_state")
        if rng_state is not None:
            self.trial_rng.bit_generator.state = deepcopy(rng_state)
        self._apply_mask()

    def clear_logs(self) -> None:
        self.transition_log.clear()


# Historical class name, now located in the static strategy module.
FeedbackSwapHypothesisModule = FixedFeedbackSwapHypothesisTransitionModule


__all__ = [
    "FIXED_STRATEGY_SPACE",
    "FeedbackSwapHypothesisModule",
    "FixedFeedbackSwapHypothesisTransitionModule",
    "FixedWorkspaceHypothesisTransitionModule",
    "FixedHypothesisStrategySpace",
    "FixedStrategyHypothesisTransitionModule",
]
