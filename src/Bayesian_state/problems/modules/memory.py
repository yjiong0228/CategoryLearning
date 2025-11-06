"""
Module: Memory Mechanism
"""

from abc import ABC
from dataclasses import dataclass
from collections.abc import Callable
from operator import ge
from typing import List, Tuple, Dict, Set, Sequence
import numpy as np
from scipy.optimize import minimize

#from CategoryLearning.Old_version.Bayesian_new.inference_engine.bayesian_engine import BaseEngine
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
        self.mask = getattr(self.engine, "partition", None)
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




###################### NEW Memory Module ######################

class DualMemoryModule(BaseModule):

    upper_numerical_bound = 1e15
    lower_numerical_bound = 1e-15

    """
    Memory Module
    """

    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)
        
        self.engine.state = kwargs.pop("default_state_init", {
            "fade": None,
            "static": None
        })
        self.state = self.engine.state

        self.gamma = kwargs.get("gamma", 0.9)
        self.w0 = kwargs.get("w0", 0.1)


        ##### For parameter optimization #####
        personal_memory_range = kwargs.get("personal_memory_range", {
            "gamma": (0.05, 1.0),
            "w0": (0.075, 0.15),
        })
        param_resolution = max(1, int(kwargs.get("param_resolution", 20)))

        gamma_grid = kwargs.get("gamma_grid")
        if gamma_grid is not None:
            self.gamma_grid = np.asarray(gamma_grid, dtype=float)
        else:
            gamma_range = personal_memory_range.get("gamma", (0.05, 1.0))
            gamma_start = float(gamma_range[0])
            gamma_stop = float(gamma_range[1])
            self.gamma_grid = np.linspace(gamma_start,
                                          gamma_stop,
                                          param_resolution,
                                          endpoint=True)

        w0_grid = kwargs.get("w0_grid")
        if w0_grid is not None:
            self.w0_grid = np.asarray(w0_grid, dtype=float)
        else:
            w0_range = personal_memory_range.get("w0", (0.075, 0.15))
            upper = float(w0_range[1])
            self.w0_grid = np.array(
                [upper / (i + 1) for i in range(param_resolution)],
                dtype=float,
            )
        ####################################

        # Ensure we always work with a numeric mask array
        # Default to an all-one mask when the engine has not installed a hypothesis mask yet
        mask = getattr(self.engine, "hypotheses_mask", None)
        if mask is None:
            set_size = int(getattr(self.engine, "set_size", 0))
            if set_size <= 0:
                raise ValueError("DualMemoryModule requires a positive engine set_size to initialise the mask.")
            mask = np.ones(set_size, dtype=float)
        self.mask = np.asarray(mask, dtype=float)
        if np.sum(self.mask) <= 0:
            self.mask = np.ones_like(self.mask, dtype=float)
        # state 初始化为 prior
        self.prior = getattr(engine, "prior", np.ones_like(self.mask) / np.sum(self.mask)).copy()
        for key in self.state:
            self.state[key] = self.translate_to_log(self.prior, mask=self.mask)


    @staticmethod
    def translate_from_log(log: np.ndarray, mask=None) -> np.ndarray:
        log -= np.max(log)
        exp = np.exp(log)
        if mask is not None:
            exp *= mask
        return exp / np.sum(exp)

    @staticmethod
    def translate_to_log(exp: np.ndarray, mask=None) -> np.ndarray:
        clipped = np.clip(exp, DualMemoryModule.lower_numerical_bound, DualMemoryModule.upper_numerical_bound)
        if mask is not None:
            clipped *= mask
        return np.log(clipped)

    def state_update(self, likelihood):
        """
        Update the memory state with new observation likelihoods

        Args:
            likelihood (np.ndarray): Likelihoods of the new observation for each hypothesis
        """
        if "fade" in self.state:
            self.state["fade"] = self.state["fade"] * self.gamma + self.translate_to_log(likelihood, mask=self.mask)
        if "static" in self.state:
            self.state["static"] = self.state["static"] + self.translate_to_log(likelihood, mask=self.mask)

    def process(self, **kwargs):
        """
        Process the likelihoods with memory mechanism
        """

        likelihood = kwargs.get("likelihood", self.engine.likelihood)

        new_mask = getattr(self.engine, "hypotheses_mask", None)
        if new_mask is None:
            new_mask = np.ones_like(self.mask, dtype=float)
        self.mask = np.asarray(new_mask, dtype=float)
        if np.sum(self.mask) <= 0:
            self.mask = np.ones_like(self.mask, dtype=float)
        self.state_update(likelihood)
        
        log_posterior = self.w0 * self.state["fade"] + (1 - self.w0) * self.state["static"]
        posterior = self.translate_from_log(log_posterior, mask=self.mask)
        self.engine.posterior = posterior

    @property
    def optimize_params_dict(self) -> Dict[str, np.ndarray]:
        return {
            "gamma": np.asarray(self.gamma_grid, dtype=float),
            "w0": np.asarray(self.w0_grid, dtype=float),
        }

    @property
    def params_dict(self) -> Dict[str, type]:
        return {
            "gamma": float,
            "w0": float,
        }


