"""
Module: Memory Mechanism
"""

from typing import Any, Dict, Mapping
import numpy as np

from .base_module import BaseModule, ModulePhase, ModuleRole


class BayesianMemoryModule(BaseModule):
    """Standard one-trial Bayesian update without additional memory state."""

    phase = ModulePhase.POST_CHOICE
    role = ModuleRole.MEMORY

    def process(self, **kwargs) -> None:
        del kwargs
        self.engine.process()


class DualMemoryModule(BaseModule):

    phase = ModulePhase.POST_CHOICE
    role = ModuleRole.MEMORY

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

        self.gamma = float(kwargs.get("gamma", 0.9))
        self.w0 = float(kwargs.get("w0", 0.1))
        # Subject-level feedback sensitivity is implemented as a gain on the
        # hypothesis-specific log likelihood written into memory.  The raw
        # likelihood remains unchanged for the beta module, which keeps
        # evidence updating distinct from rule-representation plasticity.
        self.feedback_gain = float(kwargs.get("feedback_gain", 1.0))
        if not np.isfinite(self.gamma) or self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError(
                f"gamma must be a finite float in [0, 1], got {self.gamma!r}."
            )
        if not np.isfinite(self.w0) or self.w0 < 0.0 or self.w0 > 1.0:
            raise ValueError(f"w0 must be a finite float in [0, 1], got {self.w0!r}.")
        if not np.isfinite(self.feedback_gain) or self.feedback_gain < 0.0:
            raise ValueError(
                "feedback_gain must be a finite non-negative float, "
                f"got {self.feedback_gain!r}."
            )


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
        """
        Translate probabilities to log-space with optional hypothesis masking.

        Inactive hypotheses (mask == 0) are explicitly set to -inf instead of
        relying on log(0), which avoids runtime warnings while preserving the
        intended memory semantics.
        """
        exp_arr = np.asarray(exp, dtype=float)
        clipped = np.clip(
            exp_arr,
            DualMemoryModule.lower_numerical_bound,
            DualMemoryModule.upper_numerical_bound,
        )
        if mask is None:
            return np.log(clipped)

        mask_arr = np.asarray(mask)
        if mask_arr.shape != clipped.shape:
            raise ValueError(
                f"Mask shape {mask_arr.shape} is incompatible with exp shape {clipped.shape}."
            )

        active = mask_arr.astype(bool)
        out = np.full(clipped.shape, -np.inf, dtype=float)
        if np.any(active):
            out[active] = np.log(clipped[active])
        return out

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
        
        # Preserve each survivor's long-vs-short memory disagreement.  A common
        # baseline offset is used only for hypotheses without an existing
        # finite state (normally newcomers).
        baseline_offset = b_static - b_fade
        offset = np.full_like(target_log, baseline_offset, dtype=float)
        if "static" in self.state and "fade" in self.state:
            state_static = np.asarray(self.state["static"], dtype=float)
            state_fade = np.asarray(self.state["fade"], dtype=float)
            finite_offset = np.isfinite(state_static) & np.isfinite(state_fade)
            offset[finite_offset] = state_static[finite_offset] - state_fade[finite_offset]
        
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
                # static - fade = per-hypothesis offset
                update_offset = offset[update_mask]
                self.state["fade"][update_mask] = target_val - self.w0 * update_offset
                self.state["static"][update_mask] = (
                    target_val + (1 - self.w0) * update_offset
                )
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
        log_fake_likelihood *= self.feedback_gain

        log_likelihood = self.translate_to_log(likelihood, mask=self.mask)
        finite_likelihood = np.isfinite(log_likelihood)
        log_likelihood[finite_likelihood] *= self.feedback_gain

        if "fade" in self.baseline_state:
            self.baseline_state["fade"] = self.baseline_state["fade"] * self.gamma + log_fake_likelihood
        if "static" in self.baseline_state:
            self.baseline_state["static"] = self.baseline_state["static"] + log_fake_likelihood

        if "fade" in self.state:
            self.state["fade"] = self.state["fade"] * self.gamma + log_likelihood
        if "static" in self.state:
            self.state["static"] = self.state["static"] + log_likelihood

    def prepare_for_process(self, **kwargs) -> None:
        """在 engine 调度 memory 前，对齐 active mask 和当前 prior。"""

        del kwargs
        new_mask = getattr(self.engine, "hypotheses_mask", None)
        if new_mask is None:
            new_mask = np.ones_like(self.mask, dtype=float)
        self._state_transition(np.asarray(new_mask, dtype=float), force_sync=True)
        self.mask = np.asarray(new_mask, dtype=float)

    def process(self, **kwargs):
        """Process the likelihoods with memory mechanism."""

        likelihood = kwargs.get("likelihood", self.engine.likelihood)
        
        self.state_update(likelihood)

        # Avoid 0 * (-inf) when w0 is at boundaries; memory semantics remain unchanged.
        w0 = float(self.w0)
        if np.isclose(w0, 1.0):
            log_posterior = np.array(self.state["static"], copy=True)
        elif np.isclose(w0, 0.0):
            log_posterior = np.array(self.state["fade"], copy=True)
        else:
            static = np.asarray(self.state["static"], dtype=float)
            fade = np.asarray(self.state["fade"], dtype=float)
            log_posterior = np.full(static.shape, -np.inf, dtype=float)

            static_finite = np.isfinite(static)
            fade_finite = np.isfinite(fade)

            both_finite = static_finite & fade_finite
            static_only = static_finite & (~fade_finite)
            fade_only = (~static_finite) & fade_finite

            if np.any(both_finite):
                log_posterior[both_finite] = w0 * static[both_finite] + (1.0 - w0) * fade[both_finite]
            if np.any(static_only):
                log_posterior[static_only] = w0 * static[static_only]
            if np.any(fade_only):
                log_posterior[fade_only] = (1.0 - w0) * fade[fade_only]

        # Safety check for nan/inf
        log_posterior = np.nan_to_num(log_posterior, nan=-np.inf, posinf=1e15, neginf=-1e15)
        
        posterior = self.translate_from_log(log_posterior, mask=self.mask)
        self.engine.posterior = posterior

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state": {
                str(key): np.asarray(value, dtype=float).copy()
                for key, value in self.state.items()
            },
            "baseline_state": {
                str(key): float(value)
                for key, value in self.baseline_state.items()
            },
            "mask": self.mask.copy(),
            "prior": np.asarray(self.prior, dtype=float).copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.state = {
            str(key): np.asarray(value, dtype=float).copy()
            for key, value in state["state"].items()
        }
        self.engine.state = self.state
        self.baseline_state = {
            str(key): float(value)
            for key, value in state["baseline_state"].items()
        }
        self.mask = np.asarray(state["mask"], dtype=float).copy()
        self.prior = np.asarray(state["prior"], dtype=float).copy()

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
            "feedback_gain": float,
        }
