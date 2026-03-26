"""
Beta Module: Per-Hypothesis Beta Evolution

This module manages hypothesis-specific beta (inverse temperature) parameters
with dynamic evolution rules that reflect learning behavior:
- Beta is positively correlated with posterior (better hypotheses have higher beta)
- Correct choices lead to small beta increases for consistent hypotheses
- Incorrect choices lead to sharp beta decreases for inconsistent hypotheses
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
from .base_module import BaseModule


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
    
    def __init__(self, engine, **kwargs):
        """
        Initialize BetaModule.
        
        Parameters
        ----------
        engine : BaseEngine
            The Bayesian engine instance.
        **kwargs :
            - beta_init: Initial beta value for new hypotheses (default: 3.0)
            - beta_min: Minimum beta value (default: 0.1)
            - beta_max: Maximum beta value (default: 100.0)
            - decrease_rate: Multiplicative factor for incorrect responses (default: 0.3)
            - correct_additive: Additive bonus for correct responses (default: 0.5)
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
        
    def _get_stimulus_category(self, stimulus: np.ndarray, hypo: int) -> int:
        """
        Determine which category a stimulus belongs to under a given hypothesis.
        
        Uses the partition model's **prototype-based** method (aligned with the
        likelihood calculation) to compute which category is closest to the
        stimulus.
        """
        partition = getattr(self.engine, "partition", None)
        if partition is None or not hasattr(partition, "prototypes_np"):
            return 0

        protos = partition.prototypes_np
        if hypo >= len(protos):
            return 0

        stimulus_vec = np.asarray(stimulus, dtype=float).flatten()
        proto_block = protos[hypo]
        if proto_block.ndim == 2:
            proto_block = proto_block[np.newaxis, ...]
        elif proto_block.ndim != 3:
            return 0

        distances = np.linalg.norm(proto_block - stimulus_vec, axis=-1)  # [n_protos, n_cats]
        typical = np.min(distances, axis=0)  # [n_cats]
        return int(np.argmin(typical))
    
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
        
        for hypo_idx in active_indices:
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
        for hypo_idx in range(len(self.beta)):
            if hypo_idx not in active_indices:
                self.beta[hypo_idx] = 0.
        
        # Ensure engine.beta reference is updated
        self.engine.beta = self.beta
    
    def process(self, **kwargs) -> None:
        """
        Process the current observation and update beta values.
        
        This should be called AFTER likelihood computation but can be placed
        at any point in the agenda where observation data is available.
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
        
        # Update beta based on trial outcome
        self.update_beta(stimulus, choice, feedback, active_mask)
        
        # Log current beta state (copy to avoid reference issues)
        self.beta_log.append(self.beta.copy())
    
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
    
    def reset(self) -> None:
        """Reset all beta values to initial state."""
        self.beta.fill(self.beta_init)
        self.engine.beta = self.beta
        self.beta_log.clear()
