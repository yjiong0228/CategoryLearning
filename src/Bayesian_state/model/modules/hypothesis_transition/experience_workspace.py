"""Model 1031: one workspace for examples, local generalizations and rules.

Births use completed experiences only. A generalization drops one constraint;
subsequent feedback evaluates it through the ordinary shared belief update.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
from scipy.special import expit

from ....hypothesis_space.observation_model.experience_workspace import ExperienceWorkspacePartition
from ..base_module import BaseModule, ModulePhase, ModuleRole
from ..beta import BetaModule
from ..memory import DualMemoryModule
from .belief_error_entry import newcomer_share
from .belief_outcome_retention import effective_retention


class ExperienceWorkspaceTransition(BaseModule):
    """Search and finite-capacity belief allocation, with trajectory-local contents."""

    phase = ModulePhase.PRE_CHOICE
    role = ModuleRole.HYPOTHESIS_TRANSITION
    continuous_choice_enabled = True  # Shared readout receives one-hot execution weights.
    persistent_execution_enabled = False
    _STATE_FIELDS = (
        "trial_index", "next_uid", "pending_observation", "pending_beta", "feedback_committed",
        "accumulated_failure", "current_event_probability", "executed_hypothesis", "executed_uid",
        "execution_probabilities", "execution_selection_probabilities", "execution_dwell_trials",
        "current_execution_switch_event", "current_execution_switch_probability")

    def __init__(self, engine, *, baseline_probability=.15, error_gain=1., failure_decay=.8,
                 global_search=.2, execution_inertia=1., module_seed=1031):
        super().__init__(engine)
        self.partition = engine.partition
        if (not isinstance(self.partition, ExperienceWorkspacePartition)
                or engine.context.condition not in (1, 2)
                or self.partition.n_cats != (2 if engine.context.condition == 1 else 4)):
            raise ValueError("Model 1031 v1 supports condition 1 (2 labels) and condition 2 (4 labels)")
        values = dict(baseline_probability=baseline_probability, error_gain=error_gain,
                      failure_decay=failure_decay, global_search=global_search, execution_inertia=execution_inertia)
        if (any(not np.isfinite(v) or isinstance(v, bool) for v in values.values())
                or not 0 < baseline_probability < 1 or error_gain < 0
                or not 0 <= failure_decay <= 1 or not 0 <= global_search <= 1 or execution_inertia < 0):
            raise ValueError("Invalid search/retention/execution coordinates")
        for key, value in values.items():
            setattr(self, key, float(value))
        self.capacity = self.partition.length
        self.configuration_signature = sha256(json.dumps(dict(values, capacity=self.capacity,
            n_cats=self.partition.n_cats, width=self.partition.similarity_width,
            version="1031-experience-v1"), sort_keys=True).encode()).hexdigest()
        self.reseed_future(module_seed)
        self.partition.nodes[0] = dict(kind="uncommitted", uid=0, parent_uid=-1, born_trial=0)
        self.engine.prior = np.zeros(self.capacity)
        self.engine.prior[0] = 1.
        self.engine.hypotheses_mask = (self.engine.prior > 0).astype(float)
        self.trial_index, self.next_uid = 0, 1
        self.pending_observation, self.pending_beta = None, None
        self.feedback_committed = False
        self.accumulated_failure = 0.
        self.current_event_probability = 0.
        self.executed_hypothesis, self.executed_uid = 0, 0
        self.execution_probabilities = self.engine.prior.copy()
        self.execution_selection_probabilities = self.engine.prior.copy()
        self.execution_dwell_trials = 0
        self.current_execution_switch_event = False
        self.current_execution_switch_probability = 0.
        self.transition_log: list[dict] = []

    @property
    def active(self) -> np.ndarray:
        return np.asarray([i for i, node in enumerate(self.partition.nodes) if node is not None], int)

    @property
    def dominant_hypothesis(self) -> int:
        return self.executed_hypothesis

    def reseed_future(self, module_seed: int) -> None:
        seeds = np.random.SeedSequence([int(module_seed), 1031]).spawn(2)
        self.trial_rng, self.execution_rng = [np.random.default_rng(seed) for seed in seeds]

    def record_outcome(self, observation) -> None:
        x, choice, feedback = observation
        if feedback not in (0., 1.) or choice != int(choice) or not 1 <= choice <= self.partition.n_cats:
            raise ValueError("Model 1031 requires a valid choice and binary feedback")
        self.partition.stimuli([x])
        self.pending_observation = (np.asarray(x).copy(), int(choice), float(feedback))
        self.pending_beta = self.engine.beta.copy()
        self.feedback_committed = False

    def commit_feedback(self) -> None:
        # Runs after both memory and precision learning. Label learning uses the
        # saved predictive beta, so feedback never enters its own emission twice.
        self.partition.update_template_labels(self.pending_observation, self.pending_beta)
        self.feedback_committed = True

    def _new_example(self) -> dict:
        x, choice, feedback = self.pending_observation
        allowed = np.arange(self.partition.n_cats) == choice - 1
        if feedback == 0:
            allowed = ~allowed
        return dict(kind="template", anchor=x.tolist(), mask=[True] * 4,
                    category_belief=(allowed / allowed.sum()).tolist(), parent_uid=-1,
                    source_trial=self.trial_index - 1)

    def _select_parent(self, parents: list[int], posterior: np.ndarray) -> int:
        weights = posterior[parents]
        weights = np.ones(len(parents)) if weights.sum() == 0 else weights
        return int(self.trial_rng.choice(parents, p=weights / weights.sum()))

    def _propose(self, posterior: np.ndarray) -> tuple[dict, str]:
        nodes = self.partition.nodes
        templates = [i for i in self.active if nodes[i]["kind"] == "template"]
        generalized = [i for i in templates if self.partition.compatible_rule_candidates(nodes[i]).size]
        if generalized and self.trial_rng.random() < self.global_search:
            parent_slot = self._select_parent(generalized, posterior)
            parent = nodes[parent_slot]
            occupied = [n["rule_id"] for n in nodes if n is not None and n["kind"] == "rule"]
            rules = np.setdiff1d(self.partition.compatible_rule_candidates(parent), occupied)
            if rules.size:
                weights = self.partition.rule_proposal_weights(parent, rules, float(self.engine.beta[parent_slot]))
                return dict(kind="rule", rule_id=int(self.trial_rng.choice(rules, p=weights)),
                            parent_uid=parent["uid"]), "rule_from_generalization"
        reducible = [i for i in templates if sum(nodes[i]["mask"]) > 1]
        # Fixed common policy: divide local search equally between encoding a
        # completed example and dropping one condition. Success permits both.
        if reducible and self.trial_rng.random() < .5:
            parent = nodes[self._select_parent(reducible, posterior)]
            child = deepcopy(parent)
            dimension = int(self.trial_rng.choice(np.flatnonzero(child["mask"])))
            child["mask"][dimension] = False
            child["parent_uid"] = parent["uid"]
            child["dropped_dimension"] = dimension
            return child, "drop_feature"
        return self._new_example(), "encode_example"

    def _duplicate(self, candidate: dict) -> bool:
        for node in self.partition.nodes:
            if node is None or node["kind"] != candidate["kind"]:
                continue
            if node["kind"] == "rule" and node["rule_id"] == candidate["rule_id"]:
                return True
            if (node["kind"] == "template" and node["mask"] == candidate["mask"]
                    and node["category_belief"] == candidate["category_belief"]):
                mask = np.asarray(node["mask"], bool)
                if np.array_equal(np.asarray(node["anchor"])[mask], np.asarray(candidate["anchor"])[mask]):
                    return True
        return False

    def _admit(self, candidate: dict, posterior: np.ndarray) -> tuple[np.ndarray, dict]:
        active = self.active
        initial = self.partition.nodes[0]["kind"] == "uncommitted"
        full = len(active) == self.capacity
        if initial:
            slot = 0
        elif full:
            least = active[posterior[active] == posterior[active].min()]
            slot = int(self.trial_rng.choice(least))
        else:
            slot = int(np.flatnonzero([n is None for n in self.partition.nodes])[0])
        removed = self.partition.nodes[slot]
        inherited_beta = float(posterior @ self.engine.beta)
        candidate = dict(candidate, uid=self.next_uid, born_trial=self.trial_index)
        self.next_uid += 1
        self.partition.nodes[slot] = candidate
        prior = posterior.copy()
        mass = 1. if initial else (newcomer_share(float(posterior[slot]), self.capacity,
                                                self.pending_observation[2]) if full else 1 / (len(active) + 1))
        prior[slot] = 0.
        if prior.sum() > 0:
            prior *= (1 - mass) / prior.sum()
        prior[slot] = mass
        beta = self.engine.get_module(ModuleRole.BETA, required=True)
        beta.beta[slot] = np.clip(inherited_beta, beta.beta_min, beta.beta_max)
        self.engine.beta = beta.beta.copy()
        return prior, dict(newcomer_slot=slot, newcomer_uid=candidate["uid"], newcomer_mass=mass,
            replacement_count=int(removed is not None and not initial),
            removed_uid=None if removed is None else removed["uid"],
            removed_mass=float(posterior[slot]) if removed is not None else 0.)

    def process(self, **kwargs) -> None:
        del kwargs
        posterior = np.asarray(self.engine.posterior if self.engine.posterior is not None else self.engine.prior).copy()
        prior = posterior.copy()
        event = dict(trial=self.trial_index, search_event=False, swap_event=False,
                     swap_probability=0., search_probability=0., replacement_count=0, proposal="none")
        if self.feedback_committed:
            feedback = self.pending_observation[2]
            self.accumulated_failure = (self.accumulated_failure * self.failure_decay if feedback == 1
                                        else self.accumulated_failure + 1)
            p = self.baseline_probability
            self.current_event_probability = float(expit(np.log(p) - np.log1p(-p) + self.error_gain * self.accumulated_failure))
            initial = self.partition.nodes[0]["kind"] == "uncommitted"
            searched = initial or self.trial_rng.random() < self.current_event_probability
            event.update(search_event=bool(searched), search_probability=1. if initial else self.current_event_probability)
            if searched:
                candidate, proposal = (self._new_example(), "first_example") if initial else self._propose(posterior)
                event["proposal"] = proposal
                if not self._duplicate(candidate):
                    prior, admission = self._admit(candidate, posterior)
                    event.update(admission, swap_event=True)
            self.feedback_committed = False
        self.engine.hypotheses_mask = np.asarray([node is not None for node in self.partition.nodes], float)
        self.engine.prior = prior
        beta = self.engine.get_module(ModuleRole.BETA, required=True)
        beta._zero_inactive_beta(self.active)
        self.engine.beta = beta.beta.copy()
        previous_uid = self.executed_uid
        previous = next((i for i, node in enumerate(self.partition.nodes)
                         if node is not None and node["uid"] == previous_uid), None)
        logits = np.full(self.capacity, -np.inf)
        positive = prior > 0
        logits[positive] = np.log(prior[positive])
        if previous is not None:
            logits[previous] += self.execution_inertia
        distribution = np.exp(logits - logits.max())
        distribution /= distribution.sum()
        selected = int(self.execution_rng.choice(self.capacity, p=distribution))
        self.executed_hypothesis = selected
        self.executed_uid = self.partition.nodes[selected]["uid"]
        self.execution_selection_probabilities = distribution
        self.execution_probabilities = np.zeros(self.capacity)
        self.execution_probabilities[selected] = 1.
        self.current_execution_switch_event = self.executed_uid != previous_uid
        self.current_execution_switch_probability = 1. if previous is None else float(1 - distribution[previous])
        self.execution_dwell_trials = 1 if self.current_execution_switch_event else self.execution_dwell_trials + 1
        # Averaging realized admission indicators across particles estimates
        # admission probability, including duplicate proposals. Search hazard
        # is reported separately; it is not an admission probability.
        event["swap_probability"] = float(event["swap_event"])
        event.update(executed_hypothesis=selected, executed_uid=self.executed_uid,
            execution_switch_event=self.current_execution_switch_event,
            execution_switch_probability=self.current_execution_switch_probability,
            execution_dwell_trials=self.execution_dwell_trials,
            execution_selection_probabilities=distribution.tolist(),
            predictive_m=event["search_probability"], predictive_g=self.global_search,
            failure_pressure=self.accumulated_failure, active_count=len(self.active),
            representation_kinds=[None if n is None else n["kind"] for n in self.partition.nodes])
        self.transition_log.append(event)
        self.trial_index += 1

    def state_dict(self) -> dict:
        return dict({name: deepcopy(getattr(self, name)) for name in self._STATE_FIELDS},
                    configuration_signature=self.configuration_signature,
                    representations=self.partition.export_representations(),
                    trial_rng=deepcopy(self.trial_rng.bit_generator.state),
                    execution_rng=deepcopy(self.execution_rng.bit_generator.state))

    def load_state_dict(self, state) -> None:
        expected = set(self._STATE_FIELDS) | {"configuration_signature", "representations", "trial_rng", "execution_rng"}
        if set(state) != expected or state["configuration_signature"] != self.configuration_signature:
            raise ValueError("Workspace snapshot/configuration mismatch")
        self.partition.restore_representations(state["representations"])
        for name in self._STATE_FIELDS:
            setattr(self, name, deepcopy(state[name]))
        self.trial_rng.bit_generator.state = deepcopy(state["trial_rng"])
        self.execution_rng.bit_generator.state = deepcopy(state["execution_rng"])
        if self.partition.nodes[self.executed_hypothesis]["uid"] != self.executed_uid:
            raise ValueError("Saved execution identity does not match its slot content")

    def clear_logs(self) -> None:
        self.transition_log.clear()


class ExperienceWorkspaceMemory(DualMemoryModule):
    """Reuse Model 1024's outcome-dependent normalized power update."""

    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)
        if self.forgetting_mode != "exact_power":
            raise ValueError("Experience workspace requires exact_power belief memory")

    def process(self, **kwargs):
        original = self.gamma
        self.gamma = effective_retention(original, self.engine.observation[2])
        try:
            return super().process(**kwargs)
        finally:
            self.gamma = original


class ExperienceWorkspaceBeta(BetaModule):
    """Shared feedback derivative, ADD10 step, then conditional label learning."""

    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)
        if self.beta_update_mode != "feedback_gradient":
            raise ValueError("Experience workspace requires feedback_gradient precision")

    def _update_beta_feedback_gradient(self, stimulus, choice, feedback, indices):
        original = self.learning_rate
        self.learning_rate = 10 * original
        try:
            return super()._update_beta_feedback_gradient(stimulus, choice, feedback, indices)
        finally:
            self.learning_rate = original

    def process(self, **kwargs):
        super().process(**kwargs)
        self.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION, required=True).commit_feedback()
