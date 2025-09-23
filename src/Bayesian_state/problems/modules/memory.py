"""
Module: Memory Mechanism
"""

from abc import ABC
from dataclasses import dataclass
from collections.abc import Callable
from typing import List, Tuple, Dict, Set, Sequence
import numpy as np
from scipy.optimize import minimize
from .base_module import BaseModule
from .base_module import (cdist, softmax, BaseSet, entropy)
from ..model import BaseModelParams

class BaseMemory(BaseModule):
    """
    Base Memory
    """

    def __init__(self, model, **kwargs):
        """
        Initialize
        """
        super().__init__(model, **kwargs)

        # TODO：增加其他参数？

        personal_memory_range = kwargs.pop("personal_memory_range", {"gamma": (0.05, 1.0), "w0": (0.075, 0.15)})
        

        param_resolution = kwargs.pop("param_resolution", 20)

        # 初始化参数搜索空间
        self.gamma_values = np.linspace(*personal_memory_range["gamma"], param_resolution, endpoint=True)
        #self.w0_values = np.linspace(*personal_memory_range["w0"], param_resolution, endpoint=True)
        self.w0_values = [personal_memory_range["w0"][1] / (i + 1) for i in range(param_resolution)]
    
    @property
    def params_dict(self) -> Dict[str, type]:
        """
        Returns a dictionary of parameters for the model.
        """
        return {
            "gamma": float,
            "w0": float,
        }

    @property
    def optimize_params_dict(self) -> Dict[str, Sequence]:
        """
        Returns a dictionary of parameters and their values for optimization.
        """
        return {
            "gamma": self.gamma_values,
            "w0": self.w0_values,
        }