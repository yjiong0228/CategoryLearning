"""
Likelihood Module
"""
import numpy as np
from .base_module import BaseModule


class LikelihoodModule(BaseModule):
    """
    A module responsible for calculating the likelihood of an observation
    based on a partition model.
    
    Supports both global beta and per-hypothesis beta arrays:
    - If engine.beta is available (set by BetaModule), uses per-hypothesis beta
    - Otherwise, falls back to a global beta from kwargs
    """

    def __init__(self, engine, **kwargs):
        """
        Initializes the LikelihoodModule.

        Parameters
        ----------
        engine : BaseEngine
            The Bayesian engine instance.
        partition : BasePartition
            The partition model used for likelihood calculations.
        **kwargs:
            Additional keyword arguments, e.g., `beta` (fallback global beta).
        """
        super().__init__(engine)
        self.partition = kwargs.get('partition', self.engine.partition)
        # Get the list of hypothesis indices from the engine's hypothesis set
        self.h_indices = list(self.engine.hypotheses_set)
        self.kwargs = kwargs
        
        # Default global beta (used as fallback if no BetaModule is present)
        self.default_beta = float(kwargs.get('beta', 10.0))

    def _get_beta_values(self) -> np.ndarray | float:
        """
        Get beta values for likelihood calculation.
        
        Returns per-hypothesis beta array if available from engine,
        otherwise returns the default global beta.
        """
        # Check if engine has per-hypothesis beta from BetaModule
        engine_beta = getattr(self.engine, "beta", None)
        if engine_beta is not None and isinstance(engine_beta, np.ndarray):
            # Return beta values for active hypotheses
            return engine_beta[self.h_indices]
        
        # Fallback to global beta
        return self.default_beta

    def process(self, **kwargs):
        """
        Calculates the likelihood for the current observation in the engine
        and updates the engine's likelihood state.

        The observation is expected to be a single trial data point.
        """
        observation = self.engine.observation
        if observation is None:
            raise ValueError("Engine's observation is not set before processing.")

        # The partition.calc_likelihood expects data where each element
        # (stimulus, choices, responses) is a list/array.
        # For a single observation, we wrap it to match the expected format.
        # Assuming observation is a tuple: (stimulus, choice, response)
        single_trial_data = (
            [observation[0]],  # stimulus
            [observation[1]],  # choice
            [observation[2]]   # response
        )

        # Get beta values (per-hypothesis array or global scalar)
        beta = self._get_beta_values()

        # `calc_likelihood` returns shape [n_trials, n_hypos].
        # For a single trial, this will be [1, n_hypos].
        likelihood_matrix = self.partition.calc_likelihood(
            hypos=self.h_indices,
            data=single_trial_data,
            beta=beta,
            use_cached_dist=kwargs.get('use_cached_dist', False),
            normalized=True,
            **{k: v for k, v in self.kwargs.items() if k != 'beta'}
        )

        # Squeeze the result to a 1D array of shape (n_hypos,)
        likelihood_vector = likelihood_matrix.squeeze()

        # Update the engine's likelihood attribute
        self.engine.likelihood = likelihood_vector
