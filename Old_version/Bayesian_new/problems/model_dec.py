"""
Model with blurry decision.
"""

import numpy as np
from typing import Dict, List, Tuple
from .model import SingleRationalModel, ModelParams


class SingleRationalModelDec(SingleRationalModel):
    """
    With Blurry Decision
    """

    # def __init__(self, config:Dict, phi:float, **kwargs):
    #     """

    #     """
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)

        # 初始化参数搜索空间
        self.gamma_values = np.arange(0.1, 1.1, 0.1)
        self.w0_values = np.arange(0.1, 1.1, 0.1)

        # 初始化记忆权重缓存
        self.memory_weights_cache: Dict = {}

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
