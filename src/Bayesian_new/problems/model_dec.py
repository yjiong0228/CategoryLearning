"""
Model with blurry decision.
"""

import numpy as np
from .model import SingleRationalModel, ModelParams


class SingleRationalModelDec(SingleRationalModel):
    """
    With Blurry Decision
    """

    # def __init__(self, config:Dict, phi:float, **kwargs):
    #     """

    #     """

    def predict_choice(self, params: ModelParams, x: np.ndarray,
                       condition: int):
        """
        choice
        """
        raise NotImplementedError


class MemoryModel(SingleRationalModel):
    """
    """

    def preprocess(self, data):
        """
        """

        self.rational_results = super().fit_trial_by_trial(data)

    def fit_trial_by_trial(self, data):
        """
        """
        self.preprocess(data)
