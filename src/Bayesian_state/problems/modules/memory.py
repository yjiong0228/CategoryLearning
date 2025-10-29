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
        
        self.state = kwargs.pop("default_state_init", {
            "fade": None,
            "static": None
        })
        self.gamma = kwargs.get("gamma", 0.9)
        self.w0 = kwargs.get("w0", 0.1)
        # Ensure we always work with a numeric mask array
        raw_mask = getattr(self.engine, "hypotheses_mask", None)
        if raw_mask is None:
            raise ValueError("Engine must have 'hypotheses_mask' attribute for DualMemoryModule.")
        self.mask = np.asarray(raw_mask, dtype=float)
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

        self.state_transition()
        self.state_update(likelihood)
        
        log_posterior = self.w0 * self.state["fade"] + (1 - self.w0) * self.state["static"]
        posterior = self.translate_from_log(log_posterior, mask=self.mask)
        self.engine.posterior = posterior

    def state_transition(self):
        """
        State transition from posterior_t to prior_{t+1}
        ## mask applied here ##
        """
        old_mask_bool = np.asarray(self.mask, dtype=bool)

        new_mask_raw = getattr(self.engine, "hypotheses_mask", None)
        if new_mask_raw is None:
            raise ValueError("Engine must have 'hypotheses_mask' attribute for DualMemoryModule.")
        new_mask = np.asarray(new_mask_raw, dtype=float)
        new_mask_bool = new_mask.astype(bool)

        # 比对较旧的 mask 和当前的 mask，找出新增和移除的 hypotheses
        added = np.logical_and(new_mask_bool, np.logical_not(old_mask_bool))
        removed = np.logical_and(np.logical_not(new_mask_bool), old_mask_bool)
        for key in self.state:
            # 对于被移除的 hypotheses，设定为 log(0)
            if np.any(removed):
                self.state[key][removed] = -np.inf
            # 对于留下的 hypotheses，其概率之和scale到 (留下总数/(留下+新增))
            if np.any(removed) or np.any(added):
                current_probs = self.translate_from_log(self.state[key], mask=new_mask)
                scale_factor = np.sum(new_mask) / np.sum(old_mask_bool)
                scaled_probs = current_probs * scale_factor
                self.state[key] = self.translate_to_log(scaled_probs, mask=new_mask)
            # 对于新增的 hypotheses，设定为平均值
            if np.any(added):
                self.state[key][added] = self.translate_to_log(np.ones(np.sum(added)) / np.sum(new_mask))
        self.mask = new_mask
