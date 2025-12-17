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
        
        # Initialize baseline state (tracks a hypothetical hypothesis with uniform likelihood)
        # Initial value corresponds to log(1/N)
        n_init = np.sum(self.mask)
        init_log_val = np.log(1.0 / n_init) if n_init > 0 else np.log(1.0 / len(self.mask))
        self.baseline_state = {
            "fade": init_log_val,
            "static": init_log_val
        }


    @staticmethod
    def translate_from_log(log: np.ndarray, mask=None) -> np.ndarray:
        log -= np.max(log)
        exp = np.exp(log)
        if mask is not None:
            exp *= mask
        # 归一化
        return exp / np.sum(exp)

    @staticmethod
    def translate_to_log(exp: np.ndarray, mask=None) -> np.ndarray:
        clipped = np.clip(exp, DualMemoryModule.lower_numerical_bound, DualMemoryModule.upper_numerical_bound)
        if mask is not None:
            clipped *= mask
        return np.log(clipped)

    def _state_transition(self, new_mask: np.ndarray, force_sync: bool = False) -> None:
        """
        State transition from posterior_t to prior_{t+1}
        Adjusts state so that exp(w0*static + (1-w0)*fade) is proportional to engine.prior
        """
        old_mask_bool = self.mask.astype(bool)
        new_mask_bool = new_mask.astype(bool)
        
        # If masks are identical, no transition needed unless force_sync is True
        if np.array_equal(old_mask_bool, new_mask_bool) and not force_sync:
            return
        # Get the new prior (prior_{t+1}) from engine
        prior_new = getattr(self.engine, "prior", None)
        if prior_new is None:
            # Fallback: uniform on new mask
            n_new = np.sum(new_mask)
            prior_new = np.zeros_like(new_mask)
            if n_new > 0:
                prior_new[new_mask_bool] = 1.0 / n_new
        # Target log probability
        target_log = self.translate_to_log(prior_new, mask=new_mask)

        # Masks
        added_mask = new_mask_bool & (~old_mask_bool)
        survivor_mask = new_mask_bool & old_mask_bool
        removed_mask = old_mask_bool & (~new_mask_bool)

        # Removed Hypotheses -> -inf
        for key in self.state:
            if np.any(removed_mask):
                self.state[key][removed_mask] = -np.inf

        # Calculate shift using baseline state
        b_static = self.baseline_state["static"]
        b_fade = self.baseline_state["fade"]
        b_combined = self.w0 * b_static + (1 - self.w0) * b_fade
        
        # Offset between static and fade from baseline
        offset = b_static - b_fade
        
        # Shift to align target_log (normalized) with state (unnormalized)
        # We assume baseline corresponds to uniform probability 1/N_active
        n_active = np.sum(new_mask)
        log_uniform = np.log(1.0 / n_active) if n_active > 0 else 0.0
        
        # state[new] = target_log + (B_combined - log(1/N))
        shift = b_combined - log_uniform

        # Determine which hypotheses to update
        if force_sync:
            # Update ALL active hypotheses (added + survivors)
            update_mask = new_mask_bool
        else:
            # Update only ADDED hypotheses
            update_mask = added_mask

        # Update Hypotheses
        if np.any(update_mask):
            target_val = target_log[update_mask] + shift
            
            if "static" in self.state and "fade" in self.state:
                # w0 * static + (1-w0) * fade = target
                # static - fade = offset
                self.state["fade"][update_mask] = target_val - self.w0 * offset
                self.state["static"][update_mask] = target_val + (1 - self.w0) * offset
            elif "static" in self.state:
                self.state["static"][update_mask] = target_val
            elif "fade" in self.state:
                self.state["fade"][update_mask] = target_val



    def state_update(self, likelihood):
        """
        Update the memory state with new observation likelihoods

        Args:
            likelihood (np.ndarray): Likelihoods of the new observation for each hypothesis
        """
        # NEW: Update baseline state with fake likelihood
        n_total = len(self.mask)
        # n_total = np.sum(self.mask)
        log_fake_likelihood = np.log(1.0 / n_total) if n_total > 0 else -np.inf
        # Clip to avoid numerical issues if needed, though 1/N is usually safe
        log_fake_likelihood = np.clip(log_fake_likelihood, np.log(DualMemoryModule.lower_numerical_bound), np.log(DualMemoryModule.upper_numerical_bound))

        if "fade" in self.baseline_state:
            self.baseline_state["fade"] = self.baseline_state["fade"] * self.gamma + log_fake_likelihood
        if "static" in self.baseline_state:
            self.baseline_state["static"] = self.baseline_state["static"] + log_fake_likelihood

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
        
        # FIXME: 这是实现比较粗糙，为了解决没有 hypothesis 时的 prior 更新问题。
        # Check if a hypothesis transition module exists
        hypo_module_exists = False
        if hasattr(self.engine, "modules"):
            for module in self.engine.modules.values():
                if "Hypothesis" in module.__class__.__name__:
                    hypo_module_exists = True
                    break
        
        # Perform state transition before updating mask and state
        # If Hypo module exists, we trust engine.prior and force sync
        self._state_transition(np.asarray(new_mask, dtype=float), force_sync=hypo_module_exists)

        self.mask = np.asarray(new_mask, dtype=float)

        if not hypo_module_exists:
            # 没有 H 模块就自己更新 prior
            if self.engine.posterior is not None:
                self.engine.prior = self.engine.posterior.copy()
        
        self.state_update(likelihood)
        
        log_posterior = self.w0 * self.state["static"] + (1 - self.w0) * self.state["fade"]
        # Safety check for nan/inf
        log_posterior = np.nan_to_num(log_posterior, nan=-np.inf, posinf=1e15, neginf=-1e15)
        
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
