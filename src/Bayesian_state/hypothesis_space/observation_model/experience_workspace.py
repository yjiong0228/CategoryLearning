"""Particle-local experiences and generalizations beside an immutable rule catalogue.

Indices address reusable workspace slots. Contents are saved with transition
state; slot numbers are never cross-particle psychological identities.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np

from .continuous_partition import ContinuousPartition
from .likelihood import ObservationLikelihood
from .pairing import pairing_feedback_kernel


class ExperienceWorkspacePartition:
    """Finite slots with local template emissions and existing rule geometry."""

    VALID_DISTANCE_MODES = ("boundary",)
    adaptive_representations = True

    def __init__(self, *, n_cats: int, capacity: int = 4, similarity_width: float = .25,
                 catalogue: ContinuousPartition | None = None,
                 conditional_pairing: bool = False) -> None:
        if n_cats not in (2, 4) or isinstance(n_cats, bool):
            raise ValueError("Experience workspace requires two or four categories")
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not np.isfinite(similarity_width) or similarity_width <= 0:
            raise ValueError("similarity_width must be finite and positive")
        self.n_dims, self.n_cats = 4, int(n_cats)
        self.length = int(capacity)
        self.similarity_width = float(similarity_width)
        if not isinstance(conditional_pairing, bool) or (conditional_pairing and n_cats != 4):
            raise ValueError("Conditional templates require four categories")
        self.conditional_pairing = conditional_pairing
        self.pairing_weights = np.full((self.length, 3), 1 / 3)
        self.catalogue = catalogue if catalogue is not None else ContinuousPartition(
            4, self.n_cats, pairwise_similarity_tolerance=.10,
            structural_extension="relational_152_0923" if n_cats == 4 else None)
        if (self.catalogue.n_dims, self.catalogue.n_cats) != (4, self.n_cats):
            raise ValueError("Catalogue and workspace dimensions disagree")
        self.nodes: list[dict[str, Any] | None] = [None] * self.length

    def fork_for_model(self) -> "ExperienceWorkspacePartition":
        """Create private empty slots sharing only the catalogue/geometry."""
        return type(self)(n_cats=self.n_cats, capacity=self.length,
                          similarity_width=self.similarity_width, catalogue=self.catalogue,
                          conditional_pairing=self.conditional_pairing)

    def category_belief(self, node: dict) -> np.ndarray:
        """Return the category marginal; conditional labels stay in the node."""
        theta = np.asarray(node["category_belief"], float)
        if not self.conditional_pairing:
            return theta
        slot = next(i for i, own in enumerate(self.nodes) if own is node)
        return self.pairing_weights[slot] @ theta

    def conditional_category_probabilities(self, hypo, data, beta) -> np.ndarray:
        """Return pairing × category × stimulus probabilities before feedback."""
        node = self.nodes[int(hypo)]
        x = self.stimuli(data[0])
        if node is None or node["kind"] == "uncommitted":
            return np.full((3, self.n_cats, len(x)), 1 / self.n_cats)
        if node["kind"] == "rule":
            p = self.catalogue.get_category_probabilities(node["rule_id"], data, beta,
                                                         distance_mode="boundary")
            return np.broadcast_to(p, (3, *p.shape)).copy()
        gain, _ = self.template_gain(node, x, beta)
        theta = np.asarray(node["category_belief"], float)
        return (1 - gain[None, None, :]) / self.n_cats + gain[None, None, :] * theta[:, :, None]

    def export_representations(self) -> list[dict[str, Any] | None]:
        return deepcopy(self.nodes)

    @staticmethod
    def stimuli(values) -> np.ndarray:
        x = np.asarray(values, float)
        if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("Workspace stimuli must have shape (trials, 4) in [0, 1]")
        return x

    def affinity(self, node: dict, x: np.ndarray) -> np.ndarray:
        """Similarity on retained dimensions; dropping one broadens coverage."""
        delta = (self.stimuli(x) - node["anchor"])[:, np.asarray(node["mask"], bool)]
        return np.exp(-np.sum(delta * delta, axis=1) / (2 * self.similarity_width ** 2))

    def template_gain(self, node: dict, x: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
        # Marginalize a softmax preference for each possible associated label.
        e = np.exp(-beta)
        denominator = 1 + (self.n_cats - 1) * e
        a = self.affinity(node, x)
        return a * (1 - e) / denominator, a * self.n_cats * e / denominator ** 2

    def get_category_probabilities(self, hypo, data, beta, *, distance_mode="boundary"):
        if distance_mode != "boundary" or not np.isfinite(beta) or beta < 0:
            raise ValueError("Workspace requires boundary mode and finite nonnegative precision")
        if isinstance(hypo, bool) or int(hypo) != hypo or not 0 <= int(hypo) < self.length:
            raise ValueError("Invalid workspace slot")
        if self.conditional_pairing:
            return np.einsum("g,gct->ct", self.pairing_weights[int(hypo)],
                             self.conditional_category_probabilities(hypo, data, beta))
        x = self.stimuli(data[0])
        node = self.nodes[int(hypo)]
        if node is None or node["kind"] == "uncommitted":
            return np.full((self.n_cats, len(x)), 1 / self.n_cats)
        if node["kind"] == "rule":
            return self.catalogue.get_category_probabilities(
                node["rule_id"], data, beta, distance_mode=distance_mode)
        gain, _ = self.template_gain(node, x, beta)
        theta = np.asarray(node["category_belief"], float)
        return ((1 - gain[:, None]) / self.n_cats + gain[:, None] * theta).T

    def calc_likelihood(self, *args, **kwargs):
        raise ValueError("Experience workspace uses exact feedback-gradient likelihood")

    def update_template_labels(self, observation, beta: np.ndarray) -> None:
        """Condition label uncertainty using pre-feedback precision and content."""
        x, choice, feedback = observation
        if self.conditional_pairing:
            allowed = pairing_feedback_kernel(np.eye(4), choice, feedback).T
        else:
            allowed = np.arange(self.n_cats) == int(choice) - 1
            if feedback == 0:
                allowed = ~allowed
        for slot, node in enumerate(self.nodes):
            if node is None or node["kind"] != "template":
                continue
            gain, _ = self.template_gain(node, [x], float(beta[slot]))
            chance = allowed.mean(axis=-1, keepdims=True) if self.conditional_pairing else allowed.mean()
            likelihood = (1 - gain[0]) * chance + gain[0] * allowed
            weights = np.asarray(node["category_belief"]) * likelihood
            normalizer = weights.sum(axis=-1, keepdims=True) if self.conditional_pairing else weights.sum()
            node["category_belief"] = (weights / normalizer).tolist()

    def compatible_rule_candidates(self, parent: dict) -> np.ndarray:
        """A partial template opens the full catalogue, including relational rules."""
        if parent["kind"] != "template":
            return np.empty(0, int)
        mask = np.asarray(parent["mask"], bool)
        if mask.sum() == 4:
            return np.empty(0, int)
        return np.arange(self.catalogue.length)

    def rule_proposal_weights(self, parent: dict, rules: np.ndarray, beta: float) -> np.ndarray:
        """Compare rule predictions with the parent over ignored-feature endpoints.

        This fixed approximation favors agreement across the generalization's
        range while preserving positive proposal support for the full catalogue.
        Probes determine proposal weights only; they are not feedback events.
        """
        ignored = np.flatnonzero(~np.asarray(parent["mask"], bool))
        probes = np.tile(parent["anchor"], (2 ** len(ignored), 1))
        probes[:, ignored] = np.asarray(list(product((0., 1.), repeat=len(ignored))))
        data = (probes, np.ones(len(probes)), np.ones(len(probes)))
        theta = self.category_belief(parent)
        scores = np.asarray([
            np.mean(np.log(np.maximum(theta @ self.catalogue.get_category_probabilities(
                int(rule), data, beta, distance_mode="boundary"), np.finfo(float).tiny)))
            for rule in rules])
        weights = np.exp(scores - scores.max())
        return weights / weights.sum()

    def observer_state(self) -> dict[str, np.ndarray]:
        """Numeric contents accompanying particle weights in the filter observer."""
        k = self.length
        state = dict(representation_kind=np.full(k, -1, int), representation_uid=np.full(k, -1, int),
                     representation_parent=np.full(k, -1, int), catalogue_rule=np.full(k, -1, int),
                     template_mask=np.zeros((k, 4), bool), template_anchor=np.full((k, 4), np.nan),
                     template_category_belief=np.full((k, self.n_cats), np.nan))
        if self.conditional_pairing:
            state["template_conditional_category_belief"] = np.full((k, 3, 4), np.nan)
            state["content_pairing_belief"] = self.pairing_weights.copy()
        for i, node in enumerate(self.nodes):
            if node is None:
                continue
            state["representation_kind"][i] = {"uncommitted": 0, "template": 1, "rule": 2}[node["kind"]]
            state["representation_uid"][i] = node["uid"]
            state["representation_parent"][i] = node["parent_uid"]
            if node["kind"] == "rule":
                state["catalogue_rule"][i] = node["rule_id"]
            elif node["kind"] == "template":
                state["template_mask"][i] = node["mask"]
                state["template_anchor"][i] = node["anchor"]
                state["template_category_belief"][i] = self.category_belief(node)
                if self.conditional_pairing:
                    state["template_conditional_category_belief"][i] = node["category_belief"]
        return state

    def restore_representations(self, nodes) -> None:
        if not isinstance(nodes, list) or len(nodes) != self.length:
            raise ValueError("Invalid representation snapshot width")
        uids = []
        for node in nodes:
            if node is None:
                continue
            if node.get("kind") not in ("uncommitted", "template", "rule"):
                raise ValueError("Unknown representation kind")
            if not isinstance(node.get("uid"), int) or node["uid"] < 0:
                raise ValueError("Invalid representation identity")
            uids.append(node["uid"])
            if node["kind"] == "template":
                self.stimuli([node["anchor"]])
                mask = np.asarray(node["mask"])
                theta = np.asarray(node["category_belief"], float)
                if (mask.shape != (4,) or mask.dtype != bool or not mask.any()
                        or theta.shape != ((3, 4) if self.conditional_pairing else (self.n_cats,))
                        or not np.isfinite(theta).all() or np.any(theta < 0)
                        or not np.allclose(theta.sum(axis=-1), 1.)):
                    raise ValueError("Invalid template mask or category belief")
            if node["kind"] == "rule" and not 0 <= node["rule_id"] < self.catalogue.length:
                raise ValueError("Invalid catalogue rule identity")
        if len(set(uids)) != len(uids):
            raise ValueError("Representation identities must be unique within a path")
        self.nodes = deepcopy(nodes)


class ExperienceWorkspaceLikelihood(ObservationLikelihood):
    """Exact binary feedback and the derivative of that same choice emission."""

    def __init__(self, partition, **kwargs):
        super().__init__(partition, **kwargs)
        mode = "hierarchical_pairing" if getattr(partition, "conditional_pairing", False) else "category_feedback"
        if (not isinstance(partition, ExperienceWorkspacePartition) or self.beta_source != "action"
                or self.distance_mode != "boundary" or self.feedback_likelihood_mode != mode
                or self.feedback_lapse != 0):
            raise ValueError("Experience workspace feedback mode must match its conditional labels")
        self.rule_likelihood = ObservationLikelihood(partition.catalogue, distance_mode="boundary",
                                                    feedback_likelihood_mode=mode)

    def pairing_kernel(self, observation, hypotheses, beta=None):
        if not self.partition.conditional_pairing:
            raise ValueError("Conditional templates are required for ternary feedback")
        x, choice, feedback = observation
        betas = np.broadcast_to(np.asarray(beta, float), (len(hypotheses),))
        return np.asarray([np.diag(pairing_feedback_kernel(
            self.partition.conditional_category_probabilities(h, ([x], [], []), b)[:, :, 0],
            choice, feedback)) for h, b in zip(hypotheses, betas)])

    def feedback_precision_gradient(self, observation, hypotheses, beta, *, pairing_weights=None):
        if self.partition.conditional_pairing:
            return self._conditional_precision_gradient(observation, hypotheses, beta, pairing_weights)
        if pairing_weights is not None:
            raise ValueError("Binary templates have no conditional pairing")
        evidence = self.binary_feedback_likelihood(observation, hypotheses, beta)
        x, choice, feedback = observation
        allowed = np.arange(self.partition.n_cats) == int(choice) - 1
        if feedback == 0:
            allowed = ~allowed
        betas = np.broadcast_to(np.asarray(beta, float), (len(hypotheses),))
        derivative = np.zeros(len(hypotheses))
        for i, (slot, b) in enumerate(zip(hypotheses, betas)):
            node = self.partition.nodes[int(slot)]
            if node is None or node["kind"] == "uncommitted":
                continue
            if node["kind"] == "rule":
                derivative[i] = self.rule_likelihood.feedback_precision_gradient(
                    observation, [node["rule_id"]], [b])[0]
            else:
                _, gain_derivative = self.partition.template_gain(node, [x], float(b))
                theta_mass = np.asarray(node["category_belief"])[allowed].sum()
                derivative[i] = gain_derivative[0] * (theta_mass - allowed.mean()) / evidence[i]
        return derivative

    def _conditional_precision_gradient(self, observation, hypotheses, beta, pairing_weights):
        x, choice, feedback = observation
        weights = np.asarray(pairing_weights, float)
        if (weights.shape != (len(hypotheses), 3) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1.)):
            raise ValueError("Conditional precision requires normalized pre-feedback pairing weights")
        allowed = pairing_feedback_kernel(np.eye(4), choice, feedback).T
        chance = allowed.mean(axis=1)
        betas = np.broadcast_to(np.asarray(beta, float), (len(hypotheses),))
        evidence = (weights * self.pairing_kernel(observation, hypotheses, beta=betas)).sum(axis=1)
        derivative = np.zeros(len(hypotheses))
        for i, (slot, b) in enumerate(zip(hypotheses, betas)):
            node = self.partition.nodes[int(slot)]
            if node is None or node["kind"] == "uncommitted":
                continue
            if node["kind"] == "rule":
                derivative[i] = self.rule_likelihood.feedback_precision_gradient(
                    observation, [node["rule_id"]], [b], pairing_weights=weights[i:i+1])[0]
            else:
                _, gain_derivative = self.partition.template_gain(node, [x], float(b))
                mass = (np.asarray(node["category_belief"]) * allowed).sum(axis=1)
                derivative[i] = gain_derivative[0] * (weights[i] @ (mass - chance)) / evidence[i]
        return derivative
