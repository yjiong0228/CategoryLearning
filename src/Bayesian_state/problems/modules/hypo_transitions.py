"""Simple fixed-number hypothesis module for the state-based engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Dict, Set, Any, Tuple
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
        self._posterior_to_prior_transition()

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
    
    def _posterior_to_prior_transition(self):
        """
        Update engine.prior based on the transition from old_active to active hypotheses.
        Uses similarity-weighted sum of old posteriors to initialize new hypotheses.
        """
        if self.old_active is None or self.active is None:
            return

        # Get current posterior (from previous step)
        # If posterior is not available, fallback to prior or uniform
        current_posterior = None
        if hasattr(self.engine, "posterior") and self.engine.posterior is not None:
            current_posterior = self.engine.posterior.copy()
        
        if current_posterior is None:
            # If no posterior, just ensure prior is uniform on active set
            mask = np.zeros(self.total_hypo, dtype=float)
            mask[self.active] = 1.0
            self.engine.prior = mask / mask.sum()
            return

        # Create boolean masks
        old_mask_bool = np.zeros(self.total_hypo, dtype=bool)
        old_mask_bool[self.old_active] = True
        
        new_mask_bool = np.zeros(self.total_hypo, dtype=bool)
        new_mask_bool[self.active] = True

        # Identify added hypotheses
        added_mask = new_mask_bool & (~old_mask_bool)
        added_indices = np.where(added_mask)[0]
        old_indices = self.old_active

        # Initialize new prior with current posterior
        new_prior = current_posterior.copy()

        
        if np.any(old_mask_bool & new_mask_bool): # at least one survivor
            # Get similarity matrix
            partition = getattr(self.engine, "partition", None)
            similarity_matrix = getattr(partition, "similarity_matrix", None)

            # Calculate prior for ADDED hypotheses based on similarity
            if np.any(added_mask) and similarity_matrix is not None:
                # S[added, old]
                sim_sub = similarity_matrix[np.ix_(added_indices, old_indices)]
                
                # Normalize rows to sum to 1 (weights)
                row_sums = sim_sub.sum(axis=1, keepdims=True)
                # Avoid division by zero
                row_sums[row_sums == 0] = 1.0 
                weights = sim_sub / row_sums
                
                # Weighted sum of old posteriors (normalized on old active set)
                old_probs_active = current_posterior[old_indices]
                # Ensure old probs sum to 1 for correct weighting (though they should be close)
                if old_probs_active.sum() > 0:
                    old_probs_active /= old_probs_active.sum()
                
                added_probs = weights @ old_probs_active
                new_prior[added_indices] = added_probs
            elif np.any(added_mask):
                # Fallback if no similarity matrix: uniform distribution for added
                # This is a rough heuristic; ideally similarity should be used.
                # We set them to average of survivors or just uniform 1/N
                new_prior[added_indices] = 1.0 / len(self.active)

            # Apply new mask (sets removed to 0)
            new_mask_float = new_mask_bool.astype(float)
            new_prior *= new_mask_float
            
            # Normalize on new active set
            total_prob = new_prior.sum()
            if total_prob > 0:
                new_prior /= total_prob
            else:
                # Fallback: uniform on new mask
                new_prior[new_mask_bool] = 1.0 / new_mask_bool.sum()
        else:
            # No survivors, uniform on new active set
            new_prior = np.zeros(self.total_hypo, dtype=float)
            new_prior[new_mask_bool] = 1.0 / new_mask_bool.sum()

        # Update engine.prior
        self.engine.prior = new_prior



# TODO: Requires testing
class DynamicHypothesisModule(BaseModule):
    """
    Maintains a dynamic-size hypothesis mask based on entropy and other strategies.
    
    This module allows for a variable number of hypotheses to be active at any given time,
    determined by strategies such as posterior entropy.
    """
    def __init__(self, engine, **kwargs):
        """INITIALIZE BEFORE MEMORY MODULE"""
        super().__init__(engine, **kwargs)

        self.total_hypo = self.engine.set_size
        self.full_indices = np.arange(self.total_hypo, dtype=int)
        self.rng = np.random.default_rng(kwargs.get("random_seed", None))
        
        # Config: strategies is a list of dicts
        # Example: [{"amount": "entropy", "method": "top_posterior", "min": 3, "max": 7}, ...]
        self.strategies = kwargs.get("strategies", [
            {"amount": "entropy", "method": "top_posterior", "min": 3, "max": 7},
            {"amount": "fixed", "method": "random", "value": 1}
        ])
        self.init_num = int(kwargs.get("init_num", 5))
        
        self.active: np.ndarray | None = None
        self.old_active: np.ndarray | None = None
        self._init_mask()
        self.debug = kwargs.get("hypothesis_debug", False)

    def process(self, **_: object) -> None:
        self._transition()
        self._apply_mask()

    def _init_mask(self) -> None:
        # Simple random init
        selection = self._sample_from_pool(self.full_indices, self.init_num)
        self.active = np.sort(np.array(selection, dtype=int))
        self._apply_mask()

    def _transition(self) -> None:
        self.old_active = self.active.copy() if self.active is not None else None
        
        posterior = self._get_posterior_like()
        # If posterior is None (e.g. first step), use uniform
        if posterior is None:
            posterior = np.ones(self.total_hypo) / self.total_hypo

        new_active_set = set()
        
        for strat in self.strategies:
            amount_type = strat.get("amount", "fixed")
            method_type = strat.get("method", "random")
            
            # 1. Calculate Amount
            count = 0
            if amount_type == "fixed":
                count = int(strat.get("value", 1))
            elif amount_type == "entropy":
                min_n = int(strat.get("min", 1))
                max_n = int(strat.get("max", 5))
                count = self._calc_amount_entropy(posterior, min_n, max_n)
            
            # 2. Select
            if count > 0:
                selected = self._select_hypotheses(method_type, count, posterior, exclude=new_active_set)
                new_active_set.update(selected)
        
        if not new_active_set:
            # Fallback if empty
            new_active_set.update(self._sample_from_pool(self.full_indices, 1))

        self.active = np.sort(list(new_active_set))
        if self.debug:
            print(f"DynamicHypothesis: {len(self.old_active) if self.old_active is not None else 0} -> {len(self.active)} hypos")

    def _calc_amount_entropy(self, posterior: np.ndarray, min_n: int, max_n: int) -> int:
        # Calculate entropy of the ACTIVE hypotheses
        if self.active is None:
            return max_n
            
        # Extract posterior of currently active hypotheses
        active_post = posterior[self.active]
        # Normalize
        if active_post.sum() == 0:
            return max_n
        p = active_post / active_post.sum()
        
        # Entropy
        # H = -sum(p * log(p))
        # Avoid log(0)
        p = p[p > 0]
        ent = -np.sum(p * np.log(p))
        
        # Max possible entropy for N items is log(N)
        max_ent = np.log(len(self.active)) if len(self.active) > 1 else 1.0
        norm_ent = ent / max_ent
        
        # Map norm_ent [0, 1] to [min_n, max_n]
        # High entropy (uncertain) -> More hypotheses
        scaled = min_n + (max_n - min_n) * norm_ent
        return int(np.round(scaled))

    def _select_hypotheses(self, method: str, count: int, posterior: np.ndarray, exclude: Set[int]) -> List[int]:
        if count <= 0:
            return []
            
        if method == "top_posterior":
            # Select from currently active ones based on posterior
            if self.active is None:
                return []
            
            # Filter out already selected
            candidates = [x for x in self.active if x not in exclude]
            if not candidates:
                return []
                
            cand_indices = np.array(candidates)
            scores = posterior[cand_indices]
            
            # Top K
            if len(scores) <= count:
                return candidates
            
            top_args = np.argsort(scores)[-count:]
            return cand_indices[top_args].tolist()

        elif method == "random":
            # Select from ALL available (exploration)
            # Exclude 'exclude' set
            pool = self._exclude(self.full_indices, np.array(list(exclude)))
            return self._sample_from_pool(pool, count).tolist()
            
        elif method == "random_posterior":
             # Softmax sampling from active
            if self.active is None:
                return []
            candidates = [x for x in self.active if x not in exclude]
            if not candidates:
                return []
            
            cand_indices = np.array(candidates)
            scores = posterior[cand_indices]
            if scores.sum() == 0:
                probs = None
            else:
                probs = scores / scores.sum()
                
            return self.rng.choice(cand_indices, size=min(count, len(cand_indices)), replace=False, p=probs).tolist()

        return []

    def _apply_mask(self) -> None:
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
        if size <= 0: return np.empty(0, dtype=int)
        pool_array = np.asarray(pool, dtype=int)
        if pool_array.size <= size: return pool_array
        indices = self.rng.choice(pool_array.size, size=size, replace=False)
        return pool_array[indices]

    def _exclude(self, pool: np.ndarray, used: np.ndarray) -> np.ndarray:
        mask = ~np.isin(pool, used)
        return pool[mask]
    
    # def _state_transition(self):
    #     # FIXME: 交给memory做
    #     """
    #     State transition from posterior_t to prior_{t+1}
    #     ## mask applied here ##
    #     """
    #     # 比对 old_active 和 active，找出 removed 和 added 的 hypotheses
    #     if self.old_active is None:
    #         return  # 初始时没有 old_active，跳过
    #     old_mask_bool = np.zeros(self.total_hypo, dtype=bool)
    #     old_mask_bool[self.old_active] = True
    #     new_mask = np.zeros(self.total_hypo, dtype=float)
    #     new_mask[self.active] = 1.0
    #     removed = old_mask_bool & (~new_mask.astype(bool))
    #     added = (~old_mask_bool) & new_mask.astype(bool)
    #     for key in self.engine.state:
    #         # 对于被移除的 hypotheses，设定为 log(0)
    #         if np.any(removed):
    #             self.engine.state[key][removed] = -np.inf
    #         # 对于留下的 hypotheses，其概率之和scale到 (留下总数/(留下+新增))
    #         if np.any(removed) or np.any(added):
    #             current_probs = self.translate_from_log(self.engine.state[key], mask=new_mask)
    #             scale_factor = np.sum(new_mask) / np.sum(old_mask_bool)
    #             scaled_probs = current_probs * scale_factor
    #             self.engine.state[key] = self.translate_to_log(scaled_probs, mask=new_mask)
    #         # 对于新增的 hypotheses，设定为平均值
    #         if np.any(added):
    #             self.engine.state[key][added] = self.translate_to_log(np.ones(np.sum(added)) / np.sum(new_mask))




