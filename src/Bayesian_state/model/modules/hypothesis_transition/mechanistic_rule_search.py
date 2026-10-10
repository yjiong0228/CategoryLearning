"""Executable Model 0923 full model: variable candidates and two local proposal routes."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

import numpy as np
from scipy.special import expit

from ....hypothesis_space.rule_programs import GRAMMAR_VERSION, build_rule_edit_costs
from ..base_module import BaseModule, ModuleRole
from ..pairing_memory import HierarchicalPairingMemoryModule
from .contracts import HypothesisSelection, TransitionContext, TwoStepHypothesisTransitionMixin


def _number(value: Any, name: str, *, low: float = 0, high: float = np.inf,
            open_low: bool = False, open_high: bool = False) -> float:
    number = float(value)
    if (not np.isfinite(number) or number < low or number > high
            or (open_low and number == low) or (open_high and number == high)):
        raise ValueError(f"{name} is outside its finite mathematical range")
    return number


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.size == 0 or np.any(np.isnan(logits)) or not np.isfinite(logits.max()):
        raise ValueError("proposal or execution weights have no finite support")
    values = np.exp(logits - logits.max())
    return values / values.sum()


class MechanisticRuleSearchHypothesisTransitionModule(TwoStepHypothesisTransitionMixin, BaseModule):
    """Implement specification sections 14–15 using the shared trial lifecycle.

    Initialization, exit, capacity competition and execution protection remain
    explicit model assumptions. Parameters are not fitted by constructing 0923 full model.
    """

    MODE = "mechanistic_rule_search_0923_v1"
    prior_assignment_method = "relation_driven_transport_0923_v1"
    persistent_execution_enabled = True
    uses_outcome_feedback_controller = True
    feedback_interpretation = "graded"
    _STATE_FIELDS = (
        "active", "old_active", "trial_index", "failure_pressure", "previous_feedback",
        "outcome_pending", "current_event_probability", "current_g", "executed_hypothesis",
        "execution_dwell_trials", "execution_switch_count", "current_execution_switch_probability",
        "current_execution_switch_event", "execution_probabilities",
    )

    def __init__(self, engine: Any, **kwargs: Any) -> None:
        super().__init__(engine, **kwargs)
        fields = {"max_capacity", "exit_rate", "local_structure_weight", "transfer_scale",
                  "execution_inertia", "tau_proposal", "edit_costs", "search_controller",
                  "relation_samples", "relation_seed", "module_seed"}
        if set(kwargs) - fields:
            raise ValueError(f"0923 full model has unsupported transition keys: {sorted(set(kwargs)-fields)}")
        self.total_hypo = int(engine.set_size)
        raw_capacity = kwargs.get("max_capacity", 3)
        self.capacity = int(raw_capacity)
        if isinstance(raw_capacity, bool) or float(raw_capacity) != self.capacity or not 2 <= self.capacity < self.total_hypo:
            raise ValueError("max_capacity must be an integer in [2, hypothesis_count)")
        self.exit_rate = _number(kwargs.get("exit_rate", 0), "exit_rate", high=1)
        self.local_structure_weight = _number(kwargs.get("local_structure_weight", 0.5),
                                              "local_structure_weight", high=1)
        self.transfer_scale = _number(kwargs.get("transfer_scale", 0.1),
                                      "transfer_scale", open_low=True)
        self.execution_inertia = _number(kwargs.get("execution_inertia", 0), "execution_inertia")
        self.tau_proposal = _number(kwargs.get("tau_proposal", 0.1), "tau_proposal", open_low=True)
        if self.tau_proposal != 0.1:
            raise ValueError("0923 full model v1 fixes tau_proposal=0.10")
        controller = kwargs.get("search_controller")
        expected = {"baseline_probability", "error_gain", "failure_decay",
                    "global_search", "global_search_failure_gain"}
        if not isinstance(controller, Mapping) or set(controller) != expected:
            raise ValueError(f"0923 full model search_controller requires exactly {sorted(expected)}")
        self.baseline_probability = _number(controller["baseline_probability"], "baseline_probability",
                                            high=1, open_low=True, open_high=True)
        self.error_gain = _number(controller["error_gain"], "error_gain")
        self.failure_decay = _number(controller["failure_decay"], "failure_decay",
                                    high=1, open_high=True)
        self.global_search = _number(controller["global_search"], "global_search", high=1)
        self.global_search_failure_gain = _number(controller["global_search_failure_gain"],
                                                  "global_search_failure_gain", high=1)
        samples = kwargs.get("relation_samples", 100_000)
        if isinstance(samples, bool) or int(samples) != samples or samples <= 0:
            raise ValueError("relation_samples must be a positive integer")
        seed = kwargs.get("relation_seed", 0)
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError("relation_seed must be a nonnegative integer")
        self.relation_samples, self.relation_seed = int(samples), int(seed)
        if engine.distance_mode != "boundary":
            raise ValueError("0923 full model v1 uses boundary functional relations")
        self.rule_programs, self.edit_distances = build_rule_edit_costs(
            engine.partition.hypothesis_space, kwargs.get("edit_costs"))
        self.confusion_tensor = engine.partition.similarity.get_confusion_tensor(
            distance_mode="boundary", n_samples=self.relation_samples,
            random_state=self.relation_seed)
        self.functional_similarity = np.trace(self.confusion_tensor, axis1=2, axis2=3)
        self.functional_similarity.setflags(write=False)
        self._distance_matrix = np.clip(1-self.functional_similarity, 0, 1)
        self._distance_matrix.setflags(write=False)
        self.base_prior = np.asarray(engine.prior, dtype=float).copy()
        if (self.base_prior.shape != (self.total_hypo,) or not np.isfinite(self.base_prior).all()
                or np.any(self.base_prior <= 0)):
            raise ValueError("0923 full model base prior must be finite and strictly positive on the catalogue")
        self.base_prior /= self.base_prior.sum()
        settings = (self.MODE, GRAMMAR_VERSION, self.capacity, self.exit_rate,
                    self.local_structure_weight, self.transfer_scale, self.execution_inertia,
                    self.tau_proposal, self.baseline_probability, self.error_gain,
                    self.failure_decay, self.global_search, self.global_search_failure_gain,
                    self.relation_samples, self.relation_seed)
        self.configuration_signature = sha256(
            repr(settings).encode() + self.edit_distances.tobytes()
            + self._distance_matrix.tobytes() + self.base_prior.tobytes()).hexdigest()
        self.module_seed = kwargs.get("module_seed")
        seeds = np.random.SeedSequence(self.module_seed).spawn(3)
        self.init_rng, self.trial_rng, self.execution_rng = [np.random.default_rng(seed) for seed in seeds]
        first = int(self.init_rng.choice(self.total_hypo, p=self.base_prior))
        self.active = np.array([first], dtype=int)
        self.old_active = self.active.copy()
        self.executed_hypothesis = first
        self.trial_index = 0
        self.failure_pressure = 0.0
        self.previous_feedback: float | None = None
        self.outcome_pending = False
        self.current_event_probability = self.baseline_probability
        self.current_g = self.global_search
        self.current_execution_switch_probability = 0.0
        self.current_execution_switch_event = False
        self.execution_dwell_trials = 0
        self.execution_switch_count = 0
        self.execution_probabilities = np.zeros(self.total_hypo)
        self.execution_probabilities[first] = 1
        self.transition_log: list[dict[str, Any]] = []
        self.strategy_counts_log = self.transition_log
        self.transition_rate_log: list[dict[str, Any]] = []
        self.engine.prior = self.execution_probabilities.copy()
        self._apply_mask()

    def _apply_mask(self) -> None:
        mask = np.zeros(self.total_hypo)
        mask[self.active] = 1
        self.engine.hypotheses_mask = mask

    def record_outcome(self, observation: Any) -> None:
        if self.outcome_pending:
            raise RuntimeError("completed feedback has not yet been consumed")
        feedback = float(observation[2])
        if feedback not in (0, 0.5, 1):
            raise ValueError("0923 full model feedback must be 0, 0.5 or 1")
        self.previous_feedback = feedback
        self.outcome_pending = True

    def _prepare_hypothesis_transition(self, **kwargs: Any) -> None:
        if self.outcome_pending:
            self.failure_pressure = (self.failure_decay * self.failure_pressure
                                     + (1-self.failure_decay) * (1-self.previous_feedback))
            logit = np.log(self.baseline_probability) - np.log1p(-self.baseline_probability)
            self.current_event_probability = float(expit(logit + self.error_gain * self.failure_pressure))
            self.current_g = (self.global_search + (1-self.global_search)
                              * self.global_search_failure_gain * self.failure_pressure)
            self.outcome_pending = False

    def _proposal_kernels(self, remaining: np.ndarray, inactive: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        functional, structural, fallbacks = [], [], []
        for h in remaining:
            f = _softmax(np.log(self.base_prior[inactive]) - self._distance_matrix[h, inactive]/self.tau_proposal)
            costs = self.edit_distances[h, inactive]
            available = np.isfinite(costs)
            fallback = not available.any()
            s = f.copy() if fallback else np.zeros(len(inactive))
            if not fallback:
                s[available] = _softmax(-costs[available])
            functional.append(f)
            structural.append(s)
            fallbacks.append(fallback)
        return np.asarray(functional), np.asarray(structural), np.asarray(fallbacks)

    def _temporary_weights(self, remaining: np.ndarray, gamma: np.ndarray, newcomer: int) -> np.ndarray:
        affinity = np.exp(-self._distance_matrix[remaining, newcomer]/self.transfer_scale)
        old = gamma/(1+affinity)
        return np.r_[old, np.sum(gamma*affinity/(1+affinity))]

    def _admission_probability(self, remaining: np.ndarray, gamma: np.ndarray,
                               inactive: np.ndarray, proposal: np.ndarray) -> float:
        if len(remaining) < self.capacity:
            return 1.0
        probabilities = []
        eligible_old = remaining != self.executed_hypothesis
        for u in inactive:
            w = self._temporary_weights(remaining, gamma, int(u))
            eligible = np.r_[w[:-1][eligible_old], w[-1]]
            minimum = eligible.min()
            tied = eligible == minimum
            probabilities.append(1.0 - float(tied[-1])/int(tied.sum()))
        return float(proposal @ probabilities)

    def select_hypotheses(self, context: TransitionContext, **kwargs: Any) -> HypothesisSelection:
        before = context.active_before
        diagnostics: dict[str, Any] = {"search_event": False, "proposal_source": None,
                                     "proposal_parent": None, "proposed_hypothesis": None,
                                     "structural_fallback": False, "capacity_dropped": None,
                                     "autonomous_exits": [], "remaining": before.copy(),
                                     "gamma": context.posterior[before].copy(),
                                     "admission_probability_given_search": 0.0,
                                     "zero_support_fallback": False}
        if context.trial_index == 0:
            return HypothesisSelection.from_active_sets(before, before, diagnostics=diagnostics)
        exits = self.trial_rng.random(len(before)) < self.exit_rate*(1-context.posterior[before])
        exits[before == self.executed_hypothesis] = False
        remaining = before[~exits]
        gamma = context.posterior[remaining].copy()
        if gamma.sum() == 0:
            # An explicit support-recovery convention for an otherwise undefined
            # conditioning event; never silently inject a fixed newcomer budget.
            gamma = self.base_prior[remaining].copy()
            diagnostics["zero_support_fallback"] = True
        gamma /= gamma.sum()
        inactive = np.setdiff1d(np.arange(self.total_hypo), before)
        functional, structural, fallback = self._proposal_kernels(remaining, inactive)
        global_q = self.base_prior[inactive]/self.base_prior[inactive].sum()
        local_q = gamma @ ((1-self.local_structure_weight)*functional + self.local_structure_weight*structural)
        proposal = (1-self.current_g)*local_q + self.current_g*global_q
        diagnostics.update(remaining=remaining.copy(), gamma=gamma.copy(),
                           autonomous_exits=before[exits].tolist(),
                           admission_probability_given_search=self._admission_probability(remaining, gamma, inactive, proposal))
        after = remaining.copy()
        if self.trial_rng.random() < self.current_event_probability:
            diagnostics["search_event"] = True
            if self.trial_rng.random() < self.current_g:
                q, source = global_q, "global"
            else:
                parent = int(self.trial_rng.choice(len(remaining), p=gamma))
                diagnostics["proposal_parent"] = int(remaining[parent])
                if self.trial_rng.random() < self.local_structure_weight:
                    q = structural[parent]
                    source = "functional" if fallback[parent] else "structural"
                    diagnostics["structural_fallback"] = bool(fallback[parent])
                else:
                    q, source = functional[parent], "functional"
            u = int(self.trial_rng.choice(inactive, p=q))
            diagnostics.update(proposal_source=source, proposed_hypothesis=u)
            after = np.r_[remaining, u]
            if len(remaining) == self.capacity:
                weights = self._temporary_weights(remaining, gamma, u)
                eligible = np.flatnonzero(after != self.executed_hypothesis)
                tied = eligible[weights[eligible] == weights[eligible].min()]
                index = int(self.trial_rng.choice(tied))
                if after[index] != u:
                    diagnostics["capacity_dropped"] = int(after[index])
                after = np.delete(after, index)
        return HypothesisSelection.from_active_sets(before, after, diagnostics=diagnostics)

    def assign_prior(self, context: TransitionContext, selection: HypothesisSelection, **kwargs: Any) -> np.ndarray:
        remaining = np.asarray(selection.diagnostics["remaining"], dtype=int)
        gamma = np.asarray(selection.diagnostics["gamma"], dtype=float)
        final = selection.active_after
        accepted = selection.newcomers.size > 0
        u = int(selection.newcomers[0]) if accepted else None
        prior = np.zeros(self.total_hypo)
        for h, mass in zip(remaining, gamma):
            if h in final:
                if accepted:
                    affinity = float(np.exp(-self._distance_matrix[h, u]/self.transfer_scale))
                    prior[h] += mass/(1+affinity)
                    prior[u] += mass*affinity/(1+affinity)
                else:
                    prior[h] += mass
            else:
                distances = self._distance_matrix[h, final]
                weights = _softmax(-(distances-distances.min())/self.transfer_scale)
                prior[final] += mass*weights
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            memory.assign_relation_prior(prior, selection)
        return prior

    def _finish_hypothesis_transition(self, context: TransitionContext, selection: HypothesisSelection,
                                     prior: np.ndarray, **kwargs: Any) -> Mapping[str, Any]:
        beta = self.engine.get_module(ModuleRole.BETA)
        if beta is not None:
            beta.initialize_beta_for_hypotheses(selection.newcomers, priors=prior)
            beta._zero_inactive_beta(selection.active_after)
            self.engine.beta = beta.beta.copy()
        previous = self.executed_hypothesis
        logits = np.full(self.total_hypo, -np.inf)
        positive = prior > 0
        logits[positive] = np.log(prior[positive])
        logits[previous] += self.execution_inertia
        self.execution_probabilities = _softmax(logits)
        self.executed_hypothesis = int(self.execution_rng.choice(self.total_hypo, p=self.execution_probabilities))
        self.current_execution_switch_probability = float(1-self.execution_probabilities[previous])
        self.current_execution_switch_event = self.executed_hypothesis != previous
        self.execution_switch_count += int(self.current_execution_switch_event)
        self.execution_dwell_trials = 1 if self.current_execution_switch_event else self.execution_dwell_trials+1
        attempted = bool(selection.diagnostics["search_event"])
        admitted = bool(selection.newcomers.size)
        capacity_drop = selection.diagnostics["capacity_dropped"]
        event = {key: value for key, value in selection.diagnostics.items() if key not in {"gamma", "remaining"}}
        newcomer_distance = 0.0
        if admitted:
            sources = selection.diagnostics["remaining"]
            source_weights = selection.diagnostics["gamma"]
            newcomer_distance = float(source_weights @ self._distance_matrix[sources, selection.newcomers[0]])
        event.update({
            "transition_method": self.MODE, "trial": int(context.trial_index),
            "grammar_version": GRAMMAR_VERSION,
            "active_count": len(self.active), "capacity": self.capacity,
            "active_before": context.active_before.tolist(), "active_after": self.active.tolist(),
            "search_probability": 0.0 if context.trial_index == 0 else self.current_event_probability,
            "swap_probability": (0.0 if context.trial_index == 0 else self.current_event_probability
                                 * selection.diagnostics["admission_probability_given_search"]),
            "swap_event": admitted, "search_event": attempted, "admission_event": admitted,
            "replacement_count": int(capacity_drop is not None),
            "replacement_fraction": int(capacity_drop is not None)/len(context.active_before),
            "newcomer_count": int(admitted),
            "removed_mass": float(context.posterior[selection.dropped].sum()),
            "newcomer_distance": newcomer_distance,
            "newcomer_mass": float(prior[selection.newcomers].sum()),
            "predictive_m": self.current_event_probability, "predictive_g": self.current_g,
            "g": self.current_g, "failure_pressure": self.failure_pressure,
            "mastery_evidence": 1-self.failure_pressure, "peak_mastery_evidence": np.nan,
            "exploration_target": self.current_event_probability, "global_target": self.current_g,
            "persistent_execution_enabled": True, "executed_hypothesis": self.executed_hypothesis,
            "executed_hypothesis_previous": previous,
            "execution_switch_probability": self.current_execution_switch_probability,
            "execution_switch_event": self.current_execution_switch_event,
            "execution_dwell_trials": self.execution_dwell_trials,
        })
        self.transition_log.append(event)
        self.transition_rate_log.append(event.copy())
        self.trial_index += 1
        return event

    def reseed_future(self, module_seed: int) -> None:
        self.module_seed = int(module_seed)
        seeds = np.random.SeedSequence([self.module_seed, 0x4D31]).spawn(2)
        self.trial_rng, self.execution_rng = [np.random.default_rng(seed) for seed in seeds]

    def state_dict(self) -> dict[str, Any]:
        return {name: deepcopy(getattr(self, name)) for name in self._STATE_FIELDS} | {
                "mode": self.MODE, "capacity": self.capacity,
                "configuration_signature": self.configuration_signature,
                "trial_rng": deepcopy(self.trial_rng.bit_generator.state),
                "execution_rng": deepcopy(self.execution_rng.bit_generator.state)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = set(self._STATE_FIELDS) | {
            "mode", "capacity", "configuration_signature", "trial_rng", "execution_rng"}
        if set(state) != expected:
            raise ValueError("0923 full model snapshot has incomplete or unsupported state fields")
        if (state["mode"] != self.MODE or state["capacity"] != self.capacity
                or state["configuration_signature"] != self.configuration_signature):
            raise ValueError("0923 full model snapshot does not match the configured mechanism/capacity")
        active = np.asarray(state["active"], dtype=int)
        if (active.ndim != 1 or not 1 <= len(active) <= self.capacity
                or len(np.unique(active)) != len(active) or np.any(active < 0)
                or np.any(active >= self.total_hypo) or state["executed_hypothesis"] not in active):
            raise ValueError("0923 full model snapshot has an invalid active/executed rule")
        _number(state["failure_pressure"], "saved failure_pressure", high=1)
        _number(state["current_event_probability"], "saved search probability", high=1)
        _number(state["current_g"], "saved global fraction", high=1)
        _number(state["current_execution_switch_probability"], "saved execution switch probability", high=1)
        for name in ("trial_index", "execution_dwell_trials", "execution_switch_count"):
            value = state[name]
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError("0923 full model snapshot counters must be nonnegative integers")
        if state["previous_feedback"] not in (None, 0., .5, 1.):
            raise ValueError("0923 full model snapshot has invalid completed feedback")
        if (not isinstance(state["outcome_pending"], bool)
                or not isinstance(state["current_execution_switch_event"], bool)
                or state["outcome_pending"] and state["previous_feedback"] is None):
            raise ValueError("0923 full model snapshot has invalid outcome/execution flags")
        probability = np.asarray(state["execution_probabilities"], dtype=float)
        if (probability.shape != (self.total_hypo,) or not np.isfinite(probability).all()
                or np.any(probability < 0) or not np.isclose(probability.sum(), 1)
                or np.any(probability[np.setdiff1d(np.arange(self.total_hypo), active)] != 0)):
            raise ValueError("0923 full model snapshot has invalid execution probabilities")
        trial_rng, execution_rng = deepcopy(self.trial_rng), deepcopy(self.execution_rng)
        trial_rng.bit_generator.state = deepcopy(state["trial_rng"])
        execution_rng.bit_generator.state = deepcopy(state["execution_rng"])
        for name in self._STATE_FIELDS:
            setattr(self, name, deepcopy(state[name]))
        self.trial_rng, self.execution_rng = trial_rng, execution_rng
        self._apply_mask()
        self.last_transition_result = None

    def clear_logs(self) -> None:
        self.transition_log.clear()
        self.transition_rate_log.clear()


__all__ = ["MechanisticRuleSearchHypothesisTransitionModule"]
