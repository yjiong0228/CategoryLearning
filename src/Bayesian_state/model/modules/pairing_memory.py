"""Model 0826 joint fading memory for rules and unknown response pairings.

Pairing probabilities are learned state, not three extra fitted parameters.
They follow rules through workspace transport and particle resampling.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base_module import BaseModule, ModulePhase, ModuleRole


def _normalize_joint(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if not np.all(np.isfinite(values)) or np.any(values < 0.) or total <= 0.:
        raise ValueError("joint belief must have finite non-negative values and positive mass.")
    return values / total


def transport_joint_belief(
    joint: np.ndarray, selection: Any, *, method: str,
    local_kernel: np.ndarray | None, base_prior: np.ndarray,
    global_fraction: float,
) -> tuple[np.ndarray, bool]:
    """Lift the existing scalar prior assignment to rules × pairings.

    Local projections normalize over both axes together. Normalizing each
    pairing separately would erase its evidence and change the rule marginal.
    The returned flag records a zero-local-mass fallback to global projection.
    """
    old = _normalize_joint(joint)
    after, survivors, newcomers = selection.active_after, selection.survivors, selection.newcomers
    out = np.zeros_like(old)
    if newcomers.size == 0:
        out[after] = old[after]
        return _normalize_joint(out), False
    if method == "pairwise_mass_transfer":
        out = old.copy()
        for dropped, newcomer in selection.replacement_pairs:
            out[newcomer] = out[dropped]
            out[dropped] = 0.
        return _normalize_joint(out), False
    if method not in ("similarity_transport", "mass_preserving_similarity_transport"):
        raise ValueError(f"unsupported joint prior assignment: {method!r}.")
    if not np.isfinite(global_fraction) or not 0. <= global_fraction <= 1.:
        raise ValueError("global_fraction must lie in [0, 1].")
    target = after if method == "similarity_transport" else newcomers
    global_weights = np.asarray(base_prior, dtype=float)[target]
    global_joint = global_weights[:, None] / global_weights.sum() * old.sum(axis=0)
    local = (np.asarray(local_kernel, dtype=float).T @ old)[target]
    fallback = float(local.sum()) <= 0.
    local = global_joint if fallback else _normalize_joint(local)
    projection = (1. - global_fraction) * local + global_fraction * global_joint
    if method == "similarity_transport":
        fraction = newcomers.size / selection.active_after.size
        if survivors.size:
            out[survivors] = (1. - fraction) * _normalize_joint(old[survivors])
        out[target] += fraction * projection
    else:
        out[survivors] = old[survivors]
        out[target] = old[selection.dropped].sum() * projection
    return _normalize_joint(out), fallback


class HierarchicalPairingMemoryModule(BaseModule):
    """Single joint power update, using the existing subject memory parameter."""

    phase = ModulePhase.POST_CHOICE
    role = ModuleRole.MEMORY

    def __init__(self, engine, *, gamma: float = .8, w0: float = 0.,
                 feedback_gain: float = 1., **kwargs) -> None:
        super().__init__(engine, **kwargs)
        self.gamma, self.w0, self.feedback_gain = float(gamma), float(w0), float(feedback_gain)
        if not np.isfinite(self.gamma) or not 0. <= self.gamma <= 1.:
            raise ValueError("gamma must be finite and lie in [0, 1].")
        if self.w0 != 0. or self.feedback_gain != 1.:
            raise ValueError("condition 3 joint memory requires w0=0 and feedback_gain=1.")
        self.joint = np.repeat(np.asarray(engine.prior, dtype=float)[:, None]/3., 3, axis=1)
        self.feedback_evidence: np.ndarray | None = None
        self._pending_joint: np.ndarray | None = None
        self.local_projection_fallback = False

    def initialize_prior(self) -> None:
        """Initialize after all modules have installed the finite workspace."""
        self.joint = np.repeat(np.asarray(self.engine.prior)[:, None]/3., 3, axis=1)

    def pairing_marginal(self) -> np.ndarray:
        return self.joint.sum(axis=0).copy()

    def conditional_pairing(self) -> np.ndarray:
        marginal = self.joint.sum(axis=1, keepdims=True)
        return np.divide(self.joint, marginal, out=np.full_like(self.joint, 1/3), where=marginal > 0.)

    def prepare_feedback(self, kernel: np.ndarray) -> np.ndarray:
        """Cache evidence from the effective pre-feedback joint distribution."""
        if self._pending_joint is not None:
            raise RuntimeError("previous joint feedback update has not been consumed.")
        kernel = np.asarray(kernel, dtype=float)
        if (kernel.shape != self.joint.shape or not np.all(np.isfinite(kernel))
                or np.any(kernel < 0.) or np.any(kernel > 1.)):
            raise ValueError("feedback kernel must match joint state and lie in [0, 1].")
        prior = np.asarray(self.engine.prior, dtype=float)
        if not np.allclose(self.joint.sum(axis=1), prior, rtol=1e-9, atol=1e-12):
            raise RuntimeError("joint rule marginal disagrees with the pre-choice prior.")
        mask = getattr(self.engine, "hypotheses_mask", None)
        active = prior > 0. if mask is None else np.asarray(mask) > 0.
        effective = np.zeros_like(self.joint)
        # The gamma=0 boundary includes zero cells in the active workspace.
        effective[active] = 1. if self.gamma == 0. else self.joint[active]**self.gamma
        effective = _normalize_joint(effective)
        marginal = effective.sum(axis=1, keepdims=True)
        omega = np.divide(effective, marginal, out=np.full_like(effective, 1/3), where=marginal > 0.)
        self.feedback_evidence = (omega * kernel).sum(axis=1)
        self._pending_joint = _normalize_joint(effective * kernel)
        return self.feedback_evidence.copy()

    def process(self, **kwargs) -> None:
        if self._pending_joint is None:
            raise RuntimeError("prepare_feedback() must precede the joint memory update.")
        self.joint = self._pending_joint.copy()
        self._pending_joint = None
        self.engine.posterior = self.joint.sum(axis=1)

    def assign_transition_prior(self, transition, selection) -> tuple[np.ndarray, dict[str, Any]]:
        """Carry the complete learned state through the realized search event."""
        method = transition.prior_assignment_method
        if selection.newcomers.size and method != "pairwise_mass_transfer":
            transition._ensure_geometry()
        joint, fallback = transport_joint_belief(
            self.joint, selection, method=method,
            local_kernel=getattr(transition, "_local_kernel", None),
            base_prior=transition.base_prior, global_fraction=transition.current_g,
        )
        self.joint = joint
        self.local_projection_fallback = fallback
        fraction = selection.newcomers.size / selection.active_after.size
        newcomer_mass = float(joint[selection.newcomers].sum())
        semantic_mass = (newcomer_mass / fraction if fraction else 0.)
        if method == "mass_preserving_similarity_transport" and fraction:
            semantic_mass = 1.
        elif method == "pairwise_mass_transfer":
            semantic_mass = float("nan")
        return joint.sum(axis=1), {
            "prior_assignment_method": method,
            "prior_transport_fraction": float(fraction),
            "semantic_newcomer_mass": semantic_mass,
            "pairing_local_projection_fallback": fallback,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint.copy(),
            "feedback_evidence": None if self.feedback_evidence is None else self.feedback_evidence.copy(),
            "pending_joint": None if self._pending_joint is None else self._pending_joint.copy(),
            "local_projection_fallback": self.local_projection_fallback,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        joint = np.asarray(state["joint"], dtype=float)
        if joint.shape != (self.engine.set_size, 3):
            raise ValueError("snapshot joint belief has the wrong shape.")
        _normalize_joint(joint)
        if not np.isclose(joint.sum(), 1., rtol=1e-10, atol=1e-12):
            raise ValueError("snapshot joint belief must be normalized.")
        evidence = state.get("feedback_evidence")
        pending = state.get("pending_joint")
        if evidence is not None:
            evidence = np.asarray(evidence, dtype=float)
            if (evidence.shape != (self.engine.set_size,) or not np.all(np.isfinite(evidence))
                    or np.any(evidence < 0.) or np.any(evidence > 1.)):
                raise ValueError("snapshot feedback evidence must be a valid per-rule probability vector.")
        if pending is not None:
            pending = np.asarray(pending, dtype=float)
            if pending.shape != joint.shape:
                raise ValueError("snapshot pending joint belief has the wrong shape.")
            _normalize_joint(pending)
            if not np.isclose(pending.sum(), 1., rtol=1e-10, atol=1e-12):
                raise ValueError("snapshot pending joint belief must be normalized.")
        # Validate the complete payload before mutating the live particle.
        self.joint = joint.copy()
        self.feedback_evidence = None if evidence is None else np.asarray(evidence, dtype=float).copy()
        self._pending_joint = None if pending is None else np.asarray(pending, dtype=float).copy()
        self.local_projection_fallback = bool(state.get("local_projection_fallback", False))
