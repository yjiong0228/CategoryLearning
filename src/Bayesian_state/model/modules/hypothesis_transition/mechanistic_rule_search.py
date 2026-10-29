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
from .candidate_retention import sample_candidate_set, subset_distribution


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
    """Manage candidate rules and their contribution to the next choice.

    ``0923-full-v3`` implements unprotected exits, capacity release before
    support transfer, and continuous choice weights. The default ``legacy``
    preserves the saved full-v1/v2 transition and execution mechanism.
    """

    MODE = "mechanistic_rule_search_0923_v1"
    prior_assignment_method = "relation_driven_transport_0923_v1"
    persistent_execution_enabled = True
    continuous_choice_enabled = False
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
                  "relation_samples", "relation_seed", "module_seed", "mechanism_version",
                  "intervention"}
        if set(kwargs) - fields:
            raise ValueError(f"0923 full model has unsupported transition keys: {sorted(set(kwargs)-fields)}")
        self.mechanism_version = kwargs.get("mechanism_version", "legacy")
        if self.mechanism_version not in ("legacy", "0923-full-v3"):
            raise ValueError("mechanism_version must be legacy or 0923-full-v3")
        self.continuous_choice_enabled = self.mechanism_version == "0923-full-v3"
        self.intervention = deepcopy(kwargs.get("intervention") or {"kind": "B"})
        self._validate_intervention()
        self.intervention_kind = self.intervention["kind"]
        self._STATE_FIELDS = tuple(type(self)._STATE_FIELDS)
        if self.intervention_kind == "S1" or self.intervention.get('search') == 'accumulate':
            self._STATE_FIELDS += ("accumulated_failure",)
        if self.intervention_kind == "R1":
            self._STATE_FIELDS += ("recent_log_evidence", "recent_evidence_count")
        elif self.intervention_kind in ("E", "J"):
            self._STATE_FIELDS += ("retention_evidence", "retention_exposure")
        self.persistent_execution_enabled = not self.continuous_choice_enabled
        if self.continuous_choice_enabled:
            self.MODE = "mechanistic_rule_search_0923_v3"
            self.prior_assignment_method = "retained_rule_affinity_transfer_0923_v3"
        self.total_hypo = int(engine.set_size)
        if self.intervention_kind == "S1" or self.intervention.get('search') == 'accumulate':
            self.accumulated_failure = 0.
        if self.intervention_kind == "R1":
            self.recent_log_evidence = np.zeros((self.total_hypo, self.intervention["window"]))
            self.recent_evidence_count = np.zeros(self.total_hypo, dtype=int)
        elif self.intervention_kind in ("E", "J"):
            self.retention_evidence = np.zeros(self.total_hypo)
            self.retention_exposure = np.zeros(self.total_hypo)
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
        discrete_resources = getattr(engine.partition, "rule_search_resources", None)
        grammar_version = GRAMMAR_VERSION
        if engine.distance_mode == "rule" and callable(discrete_resources):
            (grammar_version, self.rule_programs, self.edit_distances,
             self.confusion_tensor) = discrete_resources(
                kwargs.get("edit_costs"), distance_mode=engine.distance_mode)
            relation_owner = engine.partition
        else:
            if engine.distance_mode != "boundary":
                raise ValueError("0923 full model v1 uses boundary functional relations")
            self.rule_programs, self.edit_distances = build_rule_edit_costs(
                engine.partition.hypothesis_space, kwargs.get("edit_costs"))
            self.confusion_tensor = engine.partition.similarity.get_confusion_tensor(
                distance_mode="boundary", n_samples=self.relation_samples,
                random_state=self.relation_seed)
            relation_owner = engine.partition.similarity
        resources = getattr(relation_owner, "_0923_relation_resources", None)
        if resources is None or not np.shares_memory(resources[0], self.confusion_tensor):
            agreement = np.trace(self.confusion_tensor, axis1=2, axis2=3)
            distance = np.clip(1-agreement, 0, 1)
            # These fixed catalogue relations are shared by the particles.
            # Bytes-backed buffers prevent changes through a writable view.
            agreement = np.frombuffer(agreement.tobytes(), dtype=float).reshape(agreement.shape)
            distance = np.frombuffer(distance.tobytes(), dtype=float).reshape(distance.shape)
            resources = self.confusion_tensor, agreement, distance
            relation_owner._0923_relation_resources = resources
        self.functional_similarity, self._distance_matrix = [array.view() for array in resources[1:]]
        self.base_prior = np.asarray(engine.prior, dtype=float).copy()
        if (self.base_prior.shape != (self.total_hypo,) or not np.isfinite(self.base_prior).all()
                or np.any(self.base_prior <= 0)):
            raise ValueError("0923 full model base prior must be finite and strictly positive on the catalogue")
        self.base_prior /= self.base_prior.sum()
        settings = (self.MODE, grammar_version, self.capacity, self.exit_rate,
                    self.local_structure_weight, self.transfer_scale, self.execution_inertia,
                    self.tau_proposal, self.baseline_probability, self.error_gain,
                    self.failure_decay, self.global_search, self.global_search_failure_gain,
                    self.relation_samples, self.relation_seed)
        if self.intervention_kind != "B":
            settings += (tuple(sorted(self.intervention.items())),)
        if self.intervention_kind in ('E','J'):
            settings += ('continuous_retention_v2',)
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

    def _validate_intervention(self) -> None:
        """Require one explicit experimental change; B keeps saved v3 snapshots."""
        if not isinstance(self.intervention, Mapping):
            raise ValueError("intervention must be a mapping")
        kind = self.intervention.get("kind")
        fields = {"B": {"kind"}, "S_control": {"kind", "factor"}, "S1": {"kind"},
                  "R_control": {"kind", "factor"}, "R1": {"kind", "window"}, "C1": {"kind"},
                  "E": {"kind", "decay", "capacity"}, "J": {"kind", "decay", "cost", "refresh"}}
        optional = {'search'} if kind == 'E' else {'search','stay_bias','score'} if kind == 'J' else set()
        if kind not in fields or not fields[kind] <= set(self.intervention) or set(self.intervention)-fields[kind]-optional:
            raise ValueError("intervention kind or settings are unsupported")
        if kind != "B" and not self.continuous_choice_enabled:
            raise ValueError("interventions require 0923-full-v3")
        if kind in ("S_control", "R_control"):
            value = self.intervention["factor"]
            if isinstance(value, bool):
                raise ValueError("intervention factor must be numeric")
            _number(value, "intervention factor", open_low=True,
                    high=1 if kind == "R_control" else np.inf)
        if kind == "R1":
            window = self.intervention["window"]
            if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
                raise ValueError("intervention window must be a positive integer")
        if kind in ("E", "J"):
            decay = self.intervention["decay"]
            if decay != "memory":
                if isinstance(decay, (bool, str)):
                    raise ValueError("evidence decay must be memory or a number in [0, 1]")
                _number(decay, "evidence decay", high=1)
            if self.intervention.get('search','original') not in ('original','accumulate'):
                raise ValueError('Search controller must be original or accumulate')
            if kind == "E" and self.intervention["capacity"] not in ("posterior", "evidence", "evidence_mean"):
                raise ValueError("E capacity policy must be posterior, evidence or evidence_mean")
            if kind == "J":
                _number(self.intervention["cost"], "retention cost", low=-np.inf)
                if self.intervention["refresh"] not in ("every_trial", "search"):
                    raise ValueError("J refresh must be every_trial or search")
                _number(self.intervention.get('stay_bias',0.),'incumbent retention bias')
                if self.intervention.get('score','sum') not in ('sum','mean'):
                    raise ValueError('Joint score must be sum or mean')

    def _previous_log_evidence(self) -> np.ndarray:
        """Score completed feedback using its pre-learning likelihood."""
        likelihood = np.asarray(self.engine.likelihood, dtype=float)[self.active]
        if not np.isfinite(likelihood).all() or np.any(likelihood < 0) or np.any(likelihood > 1):
            raise ValueError("Retention needs valid completed feedback likelihoods")
        n_cats = self.engine.partition.n_cats
        pairing = isinstance(self.engine.get_module(ModuleRole.MEMORY), HierarchicalPairingMemoryModule)
        if pairing:
            neutral = .5 if self.previous_feedback == 0 else .25
        else:
            if self.previous_feedback == .5:
                raise ValueError("Binary-feedback retention does not accept half credit")
            neutral = 1/n_cats if self.previous_feedback == 1 else 1-1/n_cats
        return np.log(np.maximum(likelihood, np.finfo(float).tiny)) - np.log(neutral)

    @property
    def retention_decay(self) -> float:
        """Read shared memory decay after assembly of all modules."""
        decay = self.intervention["decay"]
        return float(self.engine.get_module(ModuleRole.MEMORY).gamma if decay == "memory" else decay)

    def _consume_previous_evidence(self) -> None:
        """At t+1, retain the likelihood computed before the beta update at t.

        engine.begin_trial changes the stimulus and prior but retains the last
        completed likelihood. Hierarchical likelihood already marginalizes over
        the model's forgotten pairing belief. Inactive histories are cleared on
        selection, so revisiting a rule starts a new evidence window.
        """
        log_ratio = self._previous_log_evidence()
        history = self.recent_log_evidence[self.active]
        history[:, :-1] = history[:, 1:]
        history[:, -1] = log_ratio
        self.recent_log_evidence[self.active] = history
        self.recent_evidence_count[self.active] = np.minimum(
            self.recent_evidence_count[self.active]+1, self.intervention["window"])

    @property
    def dominant_hypothesis(self) -> int:
        """Return the current dominant rule used for next-trial inertia."""
        return self.executed_hypothesis

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
            if self.intervention_kind == "R1":
                self._consume_previous_evidence()
            elif self.intervention_kind in ("E", "J"):
                self.retention_evidence[self.active] = (self.retention_decay*self.retention_evidence[self.active]
                                                        + self._previous_log_evidence())
                self.retention_exposure[self.active] = self.retention_decay*self.retention_exposure[self.active]+1.
            self.failure_pressure = (self.failure_decay * self.failure_pressure
                                     + (1-self.failure_decay) * (1-self.previous_feedback))
            logit = np.log(self.baseline_probability) - np.log1p(-self.baseline_probability)
            pressure = self.failure_pressure
            if self.intervention_kind == "S1" or self.intervention.get('search') == 'accumulate':
                if self.previous_feedback == 1:
                    self.accumulated_failure *= self.failure_decay
                else:
                    self.accumulated_failure += 1-self.previous_feedback
                pressure = self.accumulated_failure
            if self.intervention_kind == "S_control":
                logit += np.log(self.intervention["factor"])
            self.current_event_probability = float(expit(logit + self.error_gain * pressure))
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
        if self.intervention_kind == "J":
            return self._select_joint(context)
        if self.continuous_choice_enabled:
            return self._select_v3(context)
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

    def _select_v3(self, context: TransitionContext) -> HypothesisSelection:
        """Exit old rules, propose one newcomer, then release capacity."""
        before = context.active_before
        diagnostics: dict[str, Any] = {
            "search_event": False, "proposal_source": None, "proposal_parent": None,
            "proposed_hypothesis": None, "structural_fallback": False,
            "capacity_dropped": None, "autonomous_exits": [],
            "remaining": before.copy(), "gamma": context.posterior[before].copy(),
            "admission_probability_given_search": 0.0, "zero_support_fallback": False,
        }
        if context.trial_index == 0:
            return HypothesisSelection.from_active_sets(before, before, diagnostics=diagnostics)
        mass = context.posterior[before]
        exit_probability = self.exit_rate * (1 - mass)
        if self.intervention_kind == "R_control":
            exit_probability *= self.intervention["factor"]
        elif self.intervention_kind == "R1":
            exit_probability *= self.recent_log_evidence[before].sum(axis=1) < 0
        elif self.intervention_kind == "E":
            exit_probability *= self.retention_evidence[before] < 0
        exits = self.trial_rng.random(len(before)) < exit_probability
        if not np.any(~exits) or mass[~exits].sum() == 0:
            # Recover a highest-belief old rule before conditioning on retention.
            tied = np.flatnonzero(mass == mass.max())
            exits[int(self.trial_rng.choice(tied))] = False
            diagnostics["zero_support_fallback"] = True
        remaining = before[~exits]
        gamma = context.posterior[remaining].copy()
        gamma /= gamma.sum()
        diagnostics.update(remaining=remaining.copy(), gamma=gamma.copy(),
                           autonomous_exits=before[exits].tolist(),
                           admission_probability_given_search=1.0)
        after = remaining.copy()
        if self.intervention_kind == "C1" and len(remaining) == self.capacity:
            inactive = np.setdiff1d(np.arange(self.total_hypo), before)
            functional, structural, _ = self._proposal_kernels(remaining, inactive)
            local = gamma @ (self.local_structure_weight*structural + (1-self.local_structure_weight)*functional)
            global_q = self.base_prior[inactive]/self.base_prior[inactive].sum()
            proposal = self.current_g*global_q + (1-self.current_g)*local
            affinity = np.exp(-self._distance_matrix[np.ix_(remaining, inactive)]/self.transfer_scale)
            old_weights = gamma[:, None]/(1+affinity)
            weights = np.vstack((old_weights, (gamma[:, None]*affinity/(1+affinity)).sum(axis=0)))
            ties = weights == weights.min(axis=0)
            rejection = ties[-1]/ties.sum(axis=0)
            diagnostics["admission_probability_given_search"] = float(proposal @ (1-rejection))
        if self.trial_rng.random() < self.current_event_probability:
            diagnostics["search_event"] = True
            newcomer = self._draw_v3_proposal(before, remaining, gamma, diagnostics)
            if len(remaining) == self.capacity:
                weights = (self._temporary_weights(remaining, gamma, newcomer)
                           if self.intervention_kind == "C1" else gamma)
                if self.intervention_kind == "E" and self.intervention["capacity"] in ('evidence','evidence_mean'):
                    weights = self._retention_scores(remaining,self.intervention['capacity']=='evidence_mean')
                tied = np.flatnonzero(weights == weights.min())
                drop_index = int(self.trial_rng.choice(tied))
                if drop_index == len(remaining):
                    diagnostics["newcomer_rejected"] = True
                    return HypothesisSelection.from_active_sets(before, after, diagnostics=diagnostics)
                diagnostics["capacity_dropped"] = int(remaining[drop_index])
                after = np.delete(remaining, drop_index)
            after = np.r_[after, newcomer]
        return HypothesisSelection.from_active_sets(before, after, diagnostics=diagnostics)

    def _draw_v3_proposal(self, before: np.ndarray, remaining: np.ndarray,
                          gamma: np.ndarray, diagnostics: dict) -> int:
        """Use the existing kernel, excluding all previously active rules."""
        # The pool excludes every rule considered on the preceding trial,
        # including rules that have just exited.
        inactive = np.setdiff1d(np.arange(self.total_hypo), before)
        if self.trial_rng.random() < self.current_g:
            q = self.base_prior[inactive] / self.base_prior[inactive].sum()
            source = "global"
        else:
            parent = int(self.trial_rng.choice(len(remaining), p=gamma))
            diagnostics["proposal_parent"] = int(remaining[parent])
            functional, structural, fallback = self._proposal_kernels(remaining[parent:parent+1], inactive)
            if self.trial_rng.random() < self.local_structure_weight:
                q = structural[0]
                source = "functional" if fallback[0] else "structural"
                diagnostics["structural_fallback"] = bool(fallback[0])
            else:
                q, source = functional[0], "functional"
        newcomer = int(self.trial_rng.choice(inactive, p=q))
        diagnostics.update(proposal_source=source, proposed_hypothesis=newcomer)
        return newcomer

    def _retention_scores(self, rules: np.ndarray, mean: bool = False) -> np.ndarray:
        values = self.retention_evidence[rules].copy()
        if mean:
            np.divide(values,self.retention_exposure[rules],out=values,where=self.retention_exposure[rules]>0)
        return values

    def _select_joint(self, context: TransitionContext) -> HypothesisSelection:
        """Propose first, then choose a nonempty set using one retention score."""
        before = context.active_before
        gamma = context.posterior[before].copy()
        gamma /= gamma.sum()
        diagnostics = dict(search_event=False, proposal_source=None, proposal_parent=None,
            proposed_hypothesis=None, structural_fallback=False, capacity_dropped=None,
            autonomous_exits=[], joint_dropped=[], remaining=before.copy(), gamma=gamma,
            admission_probability_given_search=0., zero_support_fallback=False)
        if context.trial_index == 0:
            return HypothesisSelection.from_active_sets(before, before, diagnostics=diagnostics)
        # All inactive rules start with zero evidence. Their hypothetical
        # admission marginal is therefore independent of the proposed identity.
        old_scores = (self._retention_scores(before,self.intervention.get('score','sum')=='mean')
                      - self.intervention['cost'] + self.intervention.get('stay_bias',0.))
        _, _, marginal = subset_distribution(np.r_[-self.intervention['cost'],old_scores],self.capacity)
        diagnostics['admission_probability_given_search'] = marginal
        search = self.trial_rng.random() < self.current_event_probability
        diagnostics['search_event'] = bool(search)
        if not search and self.intervention['refresh'] == 'search':
            return HypothesisSelection.from_active_sets(before, before, diagnostics=diagnostics)
        newcomer = self._draw_v3_proposal(before, before, gamma, diagnostics) if search else None
        pool = np.r_[newcomer, before].astype(int) if search else before
        scores = self._retention_scores(pool,self.intervention.get('score','sum')=='mean')-self.intervention['cost']
        scores[np.isin(pool,before)] += self.intervention.get('stay_bias',0.)
        selected, marginal = sample_candidate_set(scores, self.capacity, self.trial_rng)
        chosen = pool[selected]
        # Preserve old rule order, then append a surviving newcomer.
        after = np.r_[before[np.isin(before, chosen)], [newcomer] if search and newcomer in chosen else []].astype(int)
        diagnostics['joint_dropped'] = before[~np.isin(before, after)].tolist()
        if search:
            diagnostics['admission_probability_given_search'] = marginal
            diagnostics['newcomer_rejected'] = newcomer not in chosen
        return HypothesisSelection.from_active_sets(before, after, diagnostics=diagnostics)

    def assign_prior(self, context: TransitionContext, selection: HypothesisSelection, **kwargs: Any) -> np.ndarray:
        if self.continuous_choice_enabled:
            return self._assign_v3_prior(context, selection)
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

    def _assign_v3_prior(self, context: TransitionContext, selection: HypothesisSelection) -> np.ndarray:
        """Normalize retained old support, then split it once by affinity."""
        retained = selection.survivors
        support = context.posterior[retained].copy()
        prior = np.zeros(self.total_hypo)
        if not len(retained):
            # Joint selection may keep only the newcomer. Pairing transfer
            # continues to use the complete old joint state.
            prior[selection.newcomers] = 1.
            memory = self.engine.get_module(ModuleRole.MEMORY)
            if isinstance(memory, HierarchicalPairingMemoryModule):
                memory.assign_relation_prior(prior, selection)
            return prior
        if self.intervention_kind == "J" and support.sum() == 0:
            support = self.base_prior[retained].copy()
            selection.diagnostics['zero_support_fallback'] = True
        support /= support.sum()
        if selection.newcomers.size:
            newcomer = int(selection.newcomers[0])
            split = self._temporary_weights(retained, support, newcomer)
            prior[retained], prior[newcomer] = split[:-1], split[-1]
        else:
            prior[retained] = support
        memory = self.engine.get_module(ModuleRole.MEMORY)
        if isinstance(memory, HierarchicalPairingMemoryModule):
            # Its joint state still contains all pre-exit rules, so the new
            # rule inherits their complete posterior pairing marginal.
            memory.assign_relation_prior(prior, selection)
        return prior

    def _finish_hypothesis_transition(self, context: TransitionContext, selection: HypothesisSelection,
                                     prior: np.ndarray, **kwargs: Any) -> Mapping[str, Any]:
        if self.intervention_kind == "R1":
            cleared = np.r_[selection.dropped, selection.newcomers]
            self.recent_log_evidence[cleared] = 0
            self.recent_evidence_count[cleared] = 0
        elif self.intervention_kind in ("E", "J"):
            self.retention_evidence[np.r_[selection.dropped, selection.newcomers]] = 0.
            self.retention_exposure[np.r_[selection.dropped, selection.newcomers]] = 0.
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
        if self.continuous_choice_enabled:
            tied = np.flatnonzero(logits == logits.max())
            if previous in tied:
                self.executed_hypothesis = previous
            else:
                self.executed_hypothesis = int(tied[0] if len(tied) == 1 else self.execution_rng.choice(tied))
            self.current_execution_switch_probability = float(self.executed_hypothesis != previous)
        else:
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
        if self.continuous_choice_enabled:
            event.pop("executed_hypothesis")
            event.pop("executed_hypothesis_previous")
            event.update(
                persistent_execution_enabled=False, continuous_choice_enabled=True,
                mechanism_version=self.mechanism_version,
                dominant_hypothesis=self.dominant_hypothesis,
                dominant_hypothesis_previous=previous,
                choice_weights=self.execution_probabilities.tolist(),
                effective_participation=float(1 / np.sum(self.execution_probabilities ** 2)),
            )
        if self.intervention_kind != "B":
            event["intervention"] = deepcopy(self.intervention)
        if self.intervention_kind == "S1" or self.intervention.get('search') == 'accumulate':
            event["accumulated_failure"] = self.accumulated_failure
        if self.intervention_kind == "R1":
            event["recent_log_evidence"] = self.recent_log_evidence[self.active].sum(axis=1).tolist()
            event["recent_evidence_count"] = self.recent_evidence_count[self.active].tolist()
        elif self.intervention_kind in ("E", "J"):
            event['retention_evidence'] = self.retention_evidence[self.active].tolist()
            event['retention_exposure'] = self.retention_exposure[self.active].tolist()
        self.transition_log.append(event)
        self.transition_rate_log.append(event.copy())
        self.trial_index += 1
        return event

    def reseed_future(self, module_seed: int) -> None:
        self.module_seed = int(module_seed)
        seeds = np.random.SeedSequence([self.module_seed, 0x4D31]).spawn(2)
        self.trial_rng, self.execution_rng = [np.random.default_rng(seed) for seed in seeds]

    def state_dict(self) -> dict[str, Any]:
        state = {name: deepcopy(getattr(self, name)) for name in self._STATE_FIELDS} | {
                "mode": self.MODE, "capacity": self.capacity,
                "configuration_signature": self.configuration_signature,
                "trial_rng": deepcopy(self.trial_rng.bit_generator.state),
                "execution_rng": deepcopy(self.execution_rng.bit_generator.state)}
        if self.continuous_choice_enabled:
            state["mechanism_version"] = self.mechanism_version
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = set(self._STATE_FIELDS) | {
            "mode", "capacity", "configuration_signature", "trial_rng", "execution_rng"}
        if state.get("mechanism_version", "legacy") != self.mechanism_version:
            raise ValueError("0923 full model snapshot does not match the configured mechanism version")
        if self.continuous_choice_enabled:
            expected.add("mechanism_version")
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
        if self.intervention_kind == "S1" or self.intervention.get('search') == 'accumulate':
            _number(state["accumulated_failure"], "saved accumulated failure")
        if self.intervention_kind == "R1":
            evidence = np.asarray(state["recent_log_evidence"])
            count = np.asarray(state["recent_evidence_count"])
            inactive = np.setdiff1d(np.arange(self.total_hypo), active)
            if (evidence.shape != self.recent_log_evidence.shape or not np.isfinite(evidence).all()
                    or count.shape != (self.total_hypo,) or not np.issubdtype(count.dtype, np.integer)
                    or np.any(count < 0) or np.any(count > self.intervention["window"])
                    or np.any(evidence[inactive] != 0) or np.any(count[inactive] != 0)):
                raise ValueError("invalid R1 feedback history snapshot")
        elif self.intervention_kind in ("E", "J"):
            evidence = np.asarray(state['retention_evidence'])
            exposure = np.asarray(state['retention_exposure'])
            inactive = np.setdiff1d(np.arange(self.total_hypo), active)
            if evidence.shape != (self.total_hypo,) or not np.isfinite(evidence).all() or np.any(evidence[inactive] != 0):
                raise ValueError('Invalid continuous retention evidence snapshot')
            if exposure.shape != (self.total_hypo,) or not np.isfinite(exposure).all() or np.any(exposure<0) or np.any(exposure[inactive]!=0):
                raise ValueError('Invalid continuous retention exposure snapshot')
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
        if self.continuous_choice_enabled and probability[int(state["executed_hypothesis"])] != probability.max():
            raise ValueError("0923 full-v3 snapshot dominant rule must maximize choice weights")
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
