"""Simple fixed-number hypothesis module for the state-based engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from ...utils import print

from .base_module import BaseModule
import numpy as np

@dataclass
class _Config:
    fixed_hypo_num: int = 7
    init_strategy: str = "random"
    transition_mode: str = "stable"
    throw_num: int = 1
    random_seed: int | None = None


class FixedNumHypothesisModule(BaseModule):
    """
    Maintains a fixed-size hypothesis mask with simple strategies.
    
    At each processing step, the module throws out a fixed number of hypotheses and resamples new ones.
    """
    upper_numerical_bound = 1e15
    lower_numerical_bound = 1e-15
    @staticmethod
    def translate_from_log(log: np.ndarray, mask=None) -> np.ndarray:
        log -= np.max(log)
        exp = np.exp(log)
        if mask is not None:
            exp *= mask
        return exp / np.sum(exp)
    @staticmethod
    def translate_to_log(exp: np.ndarray, mask=None) -> np.ndarray:
        clipped = np.clip(exp, FixedNumHypothesisModule.lower_numerical_bound, FixedNumHypothesisModule.upper_numerical_bound)
        if mask is not None:
            clipped *= mask
        return np.log(clipped)

    def __init__(self, engine, **kwargs):
        """INITIALIZE BEFORE MEMORY MODULE"""
        super().__init__(engine, **kwargs)

        total_hypo = self.engine.set_size
        
        cfg = _Config(
            fixed_hypo_num=int(kwargs.get("fixed_hypo_num", 1)),
            init_strategy=str(kwargs.get("init_strategy", "random")),
            transition_mode=str(kwargs.get("transition_mode", "stable")),
            throw_num=int(kwargs.get("throw_num", 1)),
            random_seed=kwargs.get("random_seed", None),
        )

        if not (0 < cfg.fixed_hypo_num <= total_hypo):
            raise ValueError("fixed_hypo_num must be between 1 and the total number of hypotheses.")

        self.cfg = cfg
        self.total_hypo = total_hypo
        self.full_indices = np.arange(total_hypo, dtype=int)
        self.rng = np.random.default_rng(cfg.random_seed)
        self.active: np.ndarray | None = None
        self.old_active: np.ndarray | None = None
        self._init_mask()
        self.debug = kwargs.get("hypothesis_debug", False)

    def process(self, **_: object) -> None:
        self._transition()
        self._apply_mask()
        self._state_transition()

    def _init_mask(self) -> None:
        if self.cfg.init_strategy == "stable":
            selection = self.full_indices[: self.cfg.fixed_hypo_num]
        elif self.cfg.init_strategy == "random":
            selection = self._sample_from_pool(self.full_indices, self.cfg.fixed_hypo_num)
        else:
            raise ValueError(f"Unknown init_strategy: {self.cfg.init_strategy}")
        self.active = np.sort(np.array(selection, dtype=int))
        self._apply_mask()

    def _transition(self) -> None:
        """Update the active hypothesis set based on the transition mode."""
        if self.cfg.transition_mode == "stable" or self.cfg.throw_num <= 0:
            return

        current = np.array(self.active, copy=True)
        throw_count = int(min(self.cfg.throw_num, current.size))
        if throw_count <= 0:
            return

        if self.cfg.transition_mode == "random":
            to_drop = self._sample_from_pool(current, throw_count)
        elif self.cfg.transition_mode == "top_posterior":
            to_drop = self._drop_lowest_posterior(current, throw_count)
        else:
            raise ValueError(f"Unknown transition_mode: {self.cfg.transition_mode}")
        
        # DEBUG
        if self.debug:
            print("Hypotheses now:", current, "Dropping hypotheses:", to_drop, s=1)

        survivors = current[~np.isin(current, to_drop)]
        self.old_active = self.active.copy()
        self.active = self._resample_deficit(survivors, to_drop)

    def _drop_lowest_posterior(self, active: np.ndarray, count: int) -> np.ndarray:
        source = self._get_posterior_like()
        if source is None:
            return self._sample_from_pool(active, count)

        scores = source[active]
        order = np.argsort(scores)
        return active[order[:count]]

    def _resample_deficit(self, survivors: np.ndarray, to_drop: np.ndarray) -> np.ndarray:
        """Uniformly resample new hypotheses to maintain fixed size after dropping some."""
        need = self.cfg.fixed_hypo_num - survivors.size
        if need <= 0:
            return np.sort(survivors)

        available = self._exclude(self.full_indices, survivors)
        if available.size < need:
            available = np.unique(np.concatenate([available, to_drop]))

        newcomers = self._sample_from_pool(available, need)
        return np.sort(np.concatenate([survivors, newcomers]))

    def _apply_mask(self) -> None:
        """Apply the current active hypothesis mask."""
        if self.active is None:
            return
        mask = np.zeros(self.total_hypo, dtype=float)
        mask[self.active] = 1.0
        self.engine.hypotheses_mask = mask

    def _get_posterior_like(self) -> np.ndarray | None:
        posterior = getattr(self.engine, "posterior", None)
        prior = getattr(self.engine, "prior", None)

        if posterior is not None and len(posterior) == self.total_hypo:
            return np.asarray(posterior, dtype=float)
        if prior is not None and len(prior) == self.total_hypo:
            return np.asarray(prior, dtype=float)
        return None

    def _sample_from_pool(self, pool: Sequence[int] | np.ndarray, size: int) -> np.ndarray:
        """Uniformly sample 'size' elements from 'pool' without replacement.
        
        Args:
            pool: Sequence or array of integers to sample from.
            size: Number of elements to sample.
        """
        if size <= 0:
            return np.empty(0, dtype=int)

        pool_array = np.asarray(pool, dtype=int)
        if pool_array.size <= size:
            shuffled = np.array(pool_array, copy=True)
            self.rng.shuffle(shuffled)
            return shuffled[:size]

        indices = self.rng.choice(pool_array.size, size=size, replace=False)
        return pool_array[indices]

    def _exclude(self, pool: np.ndarray, used: np.ndarray) -> np.ndarray:
        """Return elements in 'pool' that are not in 'used'."""
        mask = ~np.isin(pool, used)
        return pool[mask]
    
    def _state_transition(self):
        """
        State transition from posterior_t to prior_{t+1}
        ## mask applied here ##
        """
        # 比对 old_active 和 active，找出 removed 和 added 的 hypotheses
        if self.old_active is None:
            return  # 初始时没有 old_active，跳过
        old_mask_bool = np.zeros(self.total_hypo, dtype=bool)
        old_mask_bool[self.old_active] = True
        new_mask = np.zeros(self.total_hypo, dtype=float)
        new_mask[self.active] = 1.0
        removed = old_mask_bool & (~new_mask.astype(bool))
        added = (~old_mask_bool) & new_mask.astype(bool)
        for key in self.engine.state:
            # 对于被移除的 hypotheses，设定为 log(0)
            if np.any(removed):
                self.engine.state[key][removed] = -np.inf
            # 对于留下的 hypotheses，其概率之和scale到 (留下总数/(留下+新增))
            if np.any(removed) or np.any(added):
                current_probs = self.translate_from_log(self.engine.state[key], mask=new_mask)
                scale_factor = np.sum(new_mask) / np.sum(old_mask_bool)
                scaled_probs = current_probs * scale_factor
                self.engine.state[key] = self.translate_to_log(scaled_probs, mask=new_mask)
            # 对于新增的 hypotheses，设定为平均值
            if np.any(added):
                self.engine.state[key][added] = self.translate_to_log(np.ones(np.sum(added)) / np.sum(new_mask))




