"""
Beta Module: Per-Hypothesis Beta Evolution

This module manages hypothesis-specific beta (inverse temperature) parameters
with dynamic evolution rules that reflect learning behavior:
- Beta is positively correlated with posterior (better hypotheses have higher beta)
- Correct choices lead to small beta increases for consistent hypotheses
- Incorrect choices lead to sharp beta decreases for inconsistent hypotheses
- Updates can target either every active hypothesis or only the overtly executed
  hypothesis, so confidence learning can be rule-specific.
"""

from __future__ import annotations
from typing import Any, Mapping, Optional, List
import numpy as np
from .base_module import BaseModule, ModulePhase, ModuleRole


class BetaModule(BaseModule):
    """
    Manages per-hypothesis beta values with dynamic evolution.
    
    Beta controls the sharpness of category decisions:
    - Low beta (~0-5): soft/noisy decisions
    - High beta (>30): sharp/deterministic decisions
    
    Evolution rules:
    1. Correct choice: beta increases slightly for consistent hypotheses
    2. Incorrect choice: beta decreases sharply for inconsistent hypotheses
    3. New hypotheses start with low beta, scaled by initial prior
    """
    
    # Beta bounds
    BETA_MIN = 0.1
    BETA_MAX = 100.0
    BETA_DEFAULT = 3.0
    phase = ModulePhase.POST_CHOICE
    role = ModuleRole.BETA
    
    def __init__(self, engine, **kwargs):
        """
        Initialize BetaModule.
        
        Parameters
        ----------
        engine : BayesianStateEngine
            The Bayesian engine instance.
        **kwargs :
            - beta_init: Initial beta value for new hypotheses (default: 3.0)
            - beta_min: Minimum beta value (default: 0.1)
            - beta_max: Maximum beta value (default: 100.0)
            - decrease_rate: Multiplicative factor for incorrect responses (default: 0.3)
            - correct_additive: Additive bonus for correct responses (default: 0.5)
            - update_scope: ``active_hypotheses`` (legacy default) or
              ``executed_hypothesis`` for rule-specific confidence learning
            - use_prior_scaling: Whether to scale initial beta by prior (default: True)
            - prior_beta_scale: Scaling factor for prior-based initialization (default: 10.0)
        """
        super().__init__(engine, **kwargs)
        
        # Configuration
        self.beta_init = float(kwargs.get("beta_init", self.BETA_DEFAULT))
        self.beta_min = float(kwargs.get("beta_min", self.BETA_MIN))
        self.beta_max = float(kwargs.get("beta_max", self.BETA_MAX))
        
        # Evolution parameters (nonlinear dynamics)
        self.decrease_rate = float(kwargs.get("decrease_rate", 0.3))   # Multiplicative (sharp drop)
        self.correct_additive = float(kwargs.get("correct_additive", 0.5))  # Small additive bonus
        self.beta_update_mode = self._resolve_beta_update_mode(
            kwargs.get("beta_update_mode", "inferred_correct_category")
        )
        self.update_scope = self._resolve_update_scope(
            kwargs.get("update_scope", "active_hypotheses")
        )
        self.probabilistic_feedback_lapse = float(
            kwargs.get("probabilistic_feedback_lapse", kwargs.get("feedback_lapse", 0.0))
        )
        if (
            not np.isfinite(self.probabilistic_feedback_lapse)
            or self.probabilistic_feedback_lapse < 0.0
            or self.probabilistic_feedback_lapse >= 1.0
        ):
            raise ValueError(
                "probabilistic_feedback_lapse must be in [0, 1), "
                f"got {self.probabilistic_feedback_lapse!r}."
            )

        # Prior-based initialization
        self.use_prior_scaling = bool(kwargs.get("use_prior_scaling", True))
        self.prior_beta_scale = float(kwargs.get("prior_beta_scale", 10.0))
        
        # Initialize beta array
        set_size = getattr(engine, "set_size", 0)
        if set_size <= 0:
            raise ValueError("BetaModule requires positive engine.set_size")
        
        self.beta = np.full(set_size, self.beta_init, dtype=float)
        
        # Register beta array with engine
        self.engine.beta = self.beta
        
        # Track beta history for visualization
        self.beta_log: List[np.ndarray] = []

    @staticmethod
    def _resolve_beta_update_mode(mode: str) -> str:
        mode = str(mode).strip().lower()
        aliases = {
            "legacy": "inferred_correct_category",
            "inferred": "inferred_correct_category",
            "hard": "inferred_correct_category",
            "category": "inferred_correct_category",
            "probabilistic": "probabilistic_feedback",
            "probability": "probabilistic_feedback",
            "soft": "probabilistic_feedback",
            "bernoulli": "probabilistic_feedback",
        }
        resolved = aliases.get(mode, mode)
        valid = {"inferred_correct_category", "probabilistic_feedback"}
        if resolved not in valid:
            raise ValueError(
                f"Unsupported beta_update_mode '{mode}'. "
                f"Expected one of: {sorted(valid)}."
            )
        return resolved

    @staticmethod
    def _resolve_update_scope(scope: str) -> str:
        scope = str(scope).strip().lower()
        aliases = {
            "active": "active_hypotheses",
            "all_active": "active_hypotheses",
            "workspace": "active_hypotheses",
            "executed": "executed_hypothesis",
            "current_executed": "executed_hypothesis",
            "overt": "executed_hypothesis",
        }
        resolved = aliases.get(scope, scope)
        valid = {"active_hypotheses", "executed_hypothesis"}
        if resolved not in valid:
            raise ValueError(
                f"Unsupported beta update_scope '{scope}'. "
                f"Expected one of: {sorted(valid)}."
            )
        return resolved

    def _resolve_update_indices(self, active_indices: np.ndarray) -> np.ndarray:
        """Return the active rules whose confidence consumes this outcome."""
        active_indices = np.asarray(active_indices, dtype=int)
        if self.update_scope == "active_hypotheses":
            return active_indices

        transition = self.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        if transition is None or not bool(
            getattr(transition, "persistent_execution_enabled", False)
        ):
            raise RuntimeError(
                "beta update_scope='executed_hypothesis' requires the "
                "hypothesis-transition persistent execution controller."
            )
        executed_hypothesis = getattr(transition, "executed_hypothesis", None)
        if executed_hypothesis is None:
            raise RuntimeError(
                "The persistent execution controller has no executed_hypothesis."
            )
        executed_hypothesis = int(executed_hypothesis)
        if executed_hypothesis not in set(active_indices.tolist()):
            raise RuntimeError(
                "The executed hypothesis must remain in the active workspace "
                "during beta updating."
            )
        return np.asarray([executed_hypothesis], dtype=int)
        
    def _get_stimulus_category(self, stimulus: np.ndarray, hypo: int) -> int:
        """
        Determine which category a stimulus belongs to under a given hypothesis.
        
        Uses partition's unified assignment entry so the behavior is aligned
        with the configured likelihood distance mode.
        """
        partition = getattr(self.engine, "partition", None)
        if partition is None or not hasattr(partition, "get_category_assignment"):
            return 0
        distance_mode = getattr(self.engine, "distance_mode", "prototype")
        return partition.get_category_assignment(
            hypo=hypo,
            stimulus=np.asarray(stimulus, dtype=float),
            distance_mode=distance_mode,
            beta=1.0,
        )
    
    def initialize_beta_for_hypotheses(self, 
                                       indices: np.ndarray,
                                       priors: Optional[np.ndarray] = None) -> None:
        """
        Initialize beta values for newly sampled hypotheses.
        
        Parameters
        ----------
        indices : np.ndarray
            Indices of hypotheses to initialize.
        priors : np.ndarray, optional
            Prior probabilities for these hypotheses (used for scaling).
        """
        if len(indices) == 0:
            return
            
        if self.use_prior_scaling and priors is not None and len(priors) > 0:
            # Scale initial beta by relative prior magnitude
            # Higher prior -> higher initial beta
            prior_vals = priors[indices] if len(priors) > max(indices) else np.ones(len(indices))
            prior_max = np.max(prior_vals) if np.max(prior_vals) > 0 else 1.0
            prior_normalized = prior_vals / prior_max
            
            # Beta initialization: base + scale * normalized_prior
            # Range: [beta_init, beta_init + prior_beta_scale]
            init_vals = self.beta_init + self.prior_beta_scale * prior_normalized
        else:
            init_vals = np.full(len(indices), self.beta_init)
        
        self.beta[indices] = np.clip(init_vals, self.beta_min, self.beta_max)

    def _zero_inactive_beta(self, active_indices: np.ndarray) -> None:
        active_mask = np.zeros(len(self.beta), dtype=bool)
        active_mask[np.asarray(active_indices, dtype=int)] = True
        self.beta[~active_mask] = 0.0

    def _choice_probability_under_hypothesis(
        self,
        stimulus: np.ndarray,
        choice: int,
        hypo: int,
        beta: float,
    ) -> float:
        partition = getattr(self.engine, "partition", None)
        if partition is None or not hasattr(partition, "get_category_probabilities"):
            return 0.5
        distance_mode = getattr(self.engine, "distance_mode", "prototype")
        trial_data = ([np.asarray(stimulus, dtype=float)], [int(choice)], [1.0])
        prob = partition.get_category_probabilities(
            hypo=int(hypo),
            data=trial_data,
            beta=float(max(beta, self.beta_min)),
            distance_mode=distance_mode,
        )
        if prob.ndim == 1:
            prob = prob.reshape(-1, 1)
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= prob.shape[0]:
            return 0.0
        p_choice = float(prob[choice_idx, 0])
        n_cats = max(1, int(getattr(partition, "n_cats", prob.shape[0])))
        chance = 1.0 / float(n_cats)
        lapse = self.probabilistic_feedback_lapse
        return float(np.clip((1.0 - lapse) * p_choice + lapse * chance, 1e-12, 1.0 - 1e-12))

    def _update_beta_probabilistic_feedback(
        self,
        stimulus: np.ndarray,
        choice: int,
        feedback: float,
        update_indices: np.ndarray,
    ) -> None:
        feedback_value = float(np.clip(feedback, 0.0, 1.0))
        for hypo_idx in update_indices:
            current_beta = self.beta[hypo_idx]
            p_choice = self._choice_probability_under_hypothesis(
                stimulus=stimulus,
                choice=choice,
                hypo=int(hypo_idx),
                beta=current_beta,
            )
            evidence = feedback_value * p_choice + (1.0 - feedback_value) * (1.0 - p_choice)
            centered = 2.0 * (evidence - 0.5)
            if centered >= 0:
                headroom = self.beta_max - current_beta
                increment = self.correct_additive * centered * (headroom / self.beta_max)
                self.beta[hypo_idx] = min(current_beta + increment, self.beta_max)
            else:
                penalty = self.decrease_rate * current_beta * min(1.0, -centered)
                self.beta[hypo_idx] = max(current_beta - penalty, self.beta_min)

    def update_beta(self, 
                    stimulus: np.ndarray,
                    choice: int,
                    feedback: float,
                    active_mask: Optional[np.ndarray] = None) -> None:
        """
        Update beta values based on trial outcome.
        
        NEW Evolution rules (based on ground truth, not subject's choice):
        - We infer the correct category from feedback:
          - If feedback=1 (correct), correct_category = choice
          - If feedback=0 (wrong), correct_category = other category
        - For each hypothesis:
          - If hypothesis predicts correct_category: beta INCREASES
          - If hypothesis predicts wrong category: beta DECREASES
        
        This ensures GT hypothesis always gets rewarded when trial outcome is known.
        
        Parameters
        ----------
        stimulus : np.ndarray
            The stimulus presented in this trial.
        choice : int
            The category chosen by the subject (1-indexed).
        feedback : float
            Response correctness (1.0=correct, 0.5=family-correct, 0.0=wrong).
        active_mask : np.ndarray, optional
            Mask of currently active hypotheses.
        """
        if active_mask is None:
            active_mask = getattr(self.engine, "hypotheses_mask", None)
        if active_mask is None:
            active_mask = np.ones(len(self.beta), dtype=float)
        
        active_indices = np.where(active_mask > 0)[0]
        update_indices = self._resolve_update_indices(active_indices)
        if self.beta_update_mode == "probabilistic_feedback":
            self._update_beta_probabilistic_feedback(
                stimulus,
                choice,
                feedback,
                update_indices,
            )
            self._zero_inactive_beta(active_indices)
            self.engine.beta = self.beta
            return

        choice_0idx = int(choice) - 1  # Convert to 0-indexed
        
        partition = getattr(self.engine, "partition", None)
        if partition is None:
            return
        
        # Infer the correct category from feedback
        # For 2-category case: if choice was wrong, correct = 1 - choice
        n_cats = getattr(partition, "n_cats", 2)
        if feedback >= 1.0:
            correct_category = choice_0idx
        else:
            # Subject was wrong, so correct category is the other one
            # For 2 categories: correct = 1 - choice
            # For >2 categories: we can't know for sure, use choice anyway
            if n_cats == 2:
                correct_category = 1 - choice_0idx
            else:
                # Can't determine correct category with >2 categories
                # Fall back to penalizing choice-consistent hypotheses
                correct_category = None
        
        for hypo_idx in update_indices:
            # Determine which category the stimulus belongs to under this hypothesis
            stim_category = self._get_stimulus_category(stimulus, hypo_idx)
            
            if correct_category is not None:
                # We know the correct category
                hypo_is_correct = (stim_category == correct_category)
                current_beta = self.beta[hypo_idx]
                
                if hypo_is_correct:
                    # Hypothesis predicts the CORRECT category -> reward
                    # Use additive increase for stability: β_new = β + increment
                    # Increment scales with how far from beta_max we are (diminishing returns)
                    headroom = self.beta_max - current_beta
                    increment = self.correct_additive * (headroom / self.beta_max)
                    new_beta = current_beta + increment
                    self.beta[hypo_idx] = min(new_beta, self.beta_max)
                else:
                    # Hypothesis predicts WRONG category -> penalize
                    # Use additive decrease (gentler than multiplicative)
                    # Penalty proportional to current beta (higher beta = more confident = bigger penalty)
                    penalty = self.decrease_rate * current_beta
                    new_beta = current_beta - penalty
                    self.beta[hypo_idx] = max(new_beta, self.beta_min)
            else:
                # >2 categories and subject was wrong: use old logic
                hypo_predicts_choice = (stim_category == choice_0idx)
                if hypo_predicts_choice:
                    # Hypothesis agreed with wrong choice -> penalize
                    self.beta[hypo_idx] = max(
                        self.beta[hypo_idx] * (1 - self.decrease_rate),
                        self.beta_min
                    )
        # for other hypos, set beta to zero
        self._zero_inactive_beta(active_indices)
        
        # Ensure engine.beta reference is updated
        self.engine.beta = self.beta
    
    def process(self, **kwargs) -> None:
        """
        Process the current observation and update beta values.
        
        This should be called AFTER likelihood computation but can be placed
        at any point in the agenda where observation data is available.

        ``beta_log[t]`` records the pre-feedback beta that was actually used by
        the likelihood and choice prediction on trial ``t``.  The update based
        on the current outcome becomes available on the following trial.
        """
        observation = getattr(self.engine, "observation", None)
        if observation is None:
            return
        
        # Extract observation components
        stimulus = observation[0]
        choice = observation[1]
        feedback = observation[2]
        
        # Get active mask
        active_mask = getattr(self.engine, "hypotheses_mask", None)

        # Log before consuming the current outcome. Prediction metrics aligned
        # to prior_t must not use a beta that has already seen feedback_t.
        self.beta_log.append(self.beta.copy())

        # Update beta based on trial outcome
        self.update_beta(stimulus, choice, feedback, active_mask)
    
    def get_beta_for_hypotheses(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get beta values for specified hypotheses (or all active).
        
        Parameters
        ----------
        indices : np.ndarray, optional
            Hypothesis indices. If None, returns all beta values.
            
        Returns
        -------
        np.ndarray
            Beta values for the specified hypotheses.
        """
        if indices is None:
            return self.beta.copy()
        return self.beta[indices].copy()

    def state_dict(self) -> dict[str, Any]:
        return {"beta": self.beta.copy()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        beta = np.asarray(state["beta"], dtype=float).copy()
        if beta.shape != self.beta.shape:
            raise ValueError(
                f"beta state shape mismatch: {beta.shape} vs {self.beta.shape}."
            )
        self.beta = beta
        self.engine.beta = self.beta

    def clear_logs(self) -> None:
        self.beta_log.clear()
    
    def reset(self) -> None:
        """Reset all beta values to initial state."""
        self.beta.fill(self.beta_init)
        self.engine.beta = self.beta
        self.beta_log.clear()
