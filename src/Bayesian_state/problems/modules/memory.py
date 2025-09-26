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

        personal_memory_range = kwargs.pop("personal_memory_range", {
            "gamma": (0.05, 1.0),
            "w0": (0.075, 0.15)
        })

        param_resolution = kwargs.pop("param_resolution", 20)

        # 初始化参数搜索空间
        self.gamma_values = np.linspace(*personal_memory_range["gamma"],
                                        param_resolution,
                                        endpoint=True)
        #self.w0_values = np.linspace(*personal_memory_range["w0"], param_resolution, endpoint=True)
        self.w0_values = [
            personal_memory_range["w0"][1] / (i + 1)
            for i in range(param_resolution)
        ]

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
        """h
        Returns a dictionary of parameters and their values for optimization.
        """
        return {
            "gamma": self.gamma_values,
            "w0": self.w0_values,
        }


class DualStateMemory(BaseMemory):
    """
    w0 \\sum\\gamma^(\\tau-t)\\log p(x_t|h) +(1-w0) +\\sum \\log p(x_t|h)
    """

    def __init__(self, model, **kwargs):
        """
        """
        super().__init__(model, **kwargs)
        self.state = kwargs.pop("default_state_init", {
            "fade": None,
            "static": None
        })
        self.engine = kwargs.get("engine")
        self.infer = self.engine.infer
        self.mask = np.zeros(self.engine.hypotheses_set.length)
        self.gamma = kwargs.get("gamma")
        self.w0 = kwargs.get("w0")

    def state_evolution(self):
        """
        state evolution
        """
        if "fade" in self.state:
            self.state["fade"] = self.gamma

    def infer_single(self, observation, **kwargs) -> float:
        """
        """
        likelihood_row = self.engine.likelihood.get_likelihood(observation, **kwargs)
        self.state["fade"] = self.state["fade"]*self.gamma+np.log(likelihood_row)*self.mask
        

    def set_prior(self, prior: dict | np.ndarray | List):
        """
        """
        self.old_mask = self.mask
        self.mask = np.zeros_like(self.old_mask)

        match prior:
            case dict():
                for i in prior:
                    self.mask[self.engine.hypotheses_set.inv[i]] = 1
            case list():
                self.mask = np.array([0 if x < 0 else 1 for x in prior])
            case np.array():
                self.mask[prior < -1] = 1
            case _:
                raise Exception

        self.engine.prior = np.array(prior)
        diff_mask = (1 - self.old_mask) * self.mask
        for key in self.state:
            self.state[key][diff_mask] = np.log(prior[diff_mask])
            self.state[key] *= self.mask


