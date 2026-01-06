"""Simple fixed-number hypothesis module for the state-based engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Dict, Set, Any, Tuple, Callable
from scipy.spatial.distance import cdist
from ...utils import print, entropy, softmax

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
        # Track how many hypotheses each strategy selects per transition step
        self.strategy_counts_log: List[Dict[str, int]] = []

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
            self.old_active = self.active.copy()
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




class DynamicHypothesisModule(BaseModule):
    """
    Maintains a dynamic-size hypothesis mask based on entropy and other strategies.
    
    This module allows for a variable number of hypotheses to be active at any given time,
    determined by strategies such as posterior entropy.
    """
    
    amount_evaluators = {}

    def __init__(self, engine, **kwargs):
        """INITIALIZE BEFORE MEMORY MODULE"""
        super().__init__(engine, **kwargs)

        self.total_hypo = self.engine.set_size
        self.full_indices = np.arange(self.total_hypo, dtype=int)
        self.rng = np.random.default_rng(kwargs.get("random_seed", None))
        
        # Config: strategies is a list of dicts
        # Example: [{"amount": "entropy", "method": "top_posterior", "min": 3, "max": 7}, ...]
        strategies_input = kwargs.get("strategies", None)
        if strategies_input == "original_strategies":
            self.strategies = self.get_original_strategy_config_b() # Default to B (Cond 2/3) for compat
        elif strategies_input == "original_strategies_a":
            self.strategies = self.get_original_strategy_config_a()
        elif strategies_input == "original_strategies_b":
            self.strategies = self.get_original_strategy_config_b()
        elif strategies_input is None:
            self.strategies = [
                {"amount": "entropy", "method": "top_posterior", "min": 3, "max": 7},
                {"amount": "fixed", "method": "random", "value": 1}
            ]
        else:
            self.strategies = strategies_input
            
        # Global parameters for strategies
        self.strategy_params = {
            "beta": kwargs.get("beta", 5.0) # Default to 5.0 to match likelihood
        }
        self.init_num = int(kwargs.get("init_num", 5))
        
        self.active: np.ndarray | None = None
        self.old_active: np.ndarray | None = None
        self._init_mask()
        self.debug = kwargs.get("hypothesis_debug", False)
        # Track how many hypotheses each strategy selects per transition step (for plotting)
        self.strategy_counts_log: List[Dict[str, int]] = []

        self.cached_dist: Dict[Tuple, float] = {}

        # Register amount evaluators
        self.amount_evaluators = {
            "entropy": self._amount_entropy_gen(3),
            "entropy_1": self._amount_entropy_gen(1),
            "entropy_2": self._amount_entropy_gen(2),
            "entropy_3": self._amount_entropy_gen(3),
            "entropy_4": self._amount_entropy_gen(4),
            "entropy_5": self._amount_entropy_gen(5),
            "entropy_6": self._amount_entropy_gen(6),
            "entropy_7": self._amount_entropy_gen(7),
            "opp_entropy_2": self._amount_opposite_entropy_gen(2),
            "opp_entropy_4": self._amount_opposite_entropy_gen(4),
            "opp_entropy_7": self._amount_opposite_entropy_gen(7),
            "max_1": self._amount_max_gen(1),
            "max_2": self._amount_max_gen(2),
            "max_3": self._amount_max_gen(3),
            "max_4": self._amount_max_gen(4),
            "max_5": self._amount_max_gen(5),
            "max_6": self._amount_max_gen(6),
            "max_7": self._amount_max_gen(7),
            "random_1": self._amount_random_gen(1),
            "random_2": self._amount_random_gen(2),
            "random_3": self._amount_random_gen(3),
            "random_4": self._amount_random_gen(4),
            "random_5": self._amount_random_gen(5),
            "random_6": self._amount_random_gen(6),
            "random_7": self._amount_random_gen(7),
            "random_9": self._amount_random_gen(9),
            "opp_random_2": self._amount_opposite_random_gen(2),
            "opp_random_4": self._amount_opposite_random_gen(4),
            "opp_random_7": self._amount_opposite_random_gen(7),
            "confidence_7": self._amount_confidence_gen(7),
            "opp_confidence_7": self._amount_opposite_confidence_gen(7),
        }

    @classmethod
    def get_original_strategy_config_a(cls) -> List[Dict]:
        """
        Returns the strategy configuration for Condition 1 (Sub 1, 4, 7...).
        Ref: M7 config in fit_config.py for sub_cond1.
        Features: 
        - Max 4 hypotheses
        - Top Posterior for exploitation (confident)
        - Random for exploration (uncertainty)
        - No association (ksimilar)
        """
        return [
            # 1. Exploitation: entropy-based retention (Low Entropy -> Retain more)
            # using top_posterior as per old M7 Cond 1
            {"amount": "entropy_4", "method": "top_posterior", "top_p": 0.0},
            # 2. Exploration: entropy complement (High Entropy -> Explore more)
            {"amount": "opp_entropy_4", "method": "random"},
        ]

    @classmethod
    def get_original_strategy_config_b(cls) -> List[Dict]:
        """
        Returns the strategy configuration for Condition 2 & 3 (Sub 2, 3, 5, 6...).
        Ref: M7 config in fit_config.py for sub_cond2 + sub_cond3.
        Features:
        - Max 7 hypotheses
        - Random Posterior for exploitation (confident)
        - Random for exploration (uncertainty)
        - Association (ksimilar) included (1 neighbor)
        """
        return [
            # 1. Exploitation: entropy-based retention (higher entropy -> more)
            {"amount": "entropy_7", "method": "random_posterior"},
            # 2. Exploration: entropy complement (entropy low -> fewer random)
            {"amount": "opp_entropy_7", "method": "random"},
            # 3. Association: Add similar hypotheses
            {"amount": "fixed", "method": "ksimilar_centers", "value": 1, 
             "proto_hypo_amount": 1, "proto_hypo_method": "top", "cluster_hypo_method": "top"}
        ]
    
    @classmethod
    def get_original_strategy_config(cls) -> List[Dict]:
        """
        Deprecated. Alias for get_original_strategy_config_b.
        """
        return cls.get_original_strategy_config_b()

    def adaptive_amount_evaluator(self, amount: float | str | Callable, **kwargs) -> int:
        """
        Adaptively deal with evaluator / number format of amount.
        """
        if isinstance(amount, int):
            return amount
        elif callable(amount):
            return amount(**kwargs)
        elif isinstance(amount, str):
            if amount in self.amount_evaluators:
                return self.amount_evaluators[amount](**kwargs)
            else:
                return 1
        else:
            raise TypeError(f"Unexpected amount type. {amount}")

    @classmethod
    def _amount_entropy_gen(cls, max_amount=3):
        def _amount_entropy_based(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            # posterior is array
            p_entropy = entropy(posterior)
            return max(0, int(max_amount - min(np.exp(p_entropy), max_amount + 30)) + 2)
        return _amount_entropy_based

    @classmethod
    def _amount_opposite_entropy_gen(cls, max_amount=3):
        def _amount_opposite_entropy_based(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            p_entropy = entropy(posterior)
            n_hypos = len(posterior)
            max_possible_entropy = np.log(n_hypos) if n_hypos > 1 else 1.0
            normalized_entropy = p_entropy / max_possible_entropy
            scaled_amount = 1 + int(normalized_entropy * (max_amount - 1))
            return min(scaled_amount, max_amount)
        return _amount_opposite_entropy_based

    @classmethod
    def _amount_max_gen(cls, max_amount=3):
        def _amount_max_based(posterior: np.ndarray, max_amount=max_amount, **kwargs):
            max_post = np.max(posterior)
            if max_post <= 0: return max_amount # Avoid div by zero
            return 0 if 3. / max_post > max_amount else int(3. / max_post)
        return _amount_max_based

    @classmethod
    def _amount_random_gen(cls, max_amount=3):
        def _amount_random_based(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            max_post = np.max(posterior)
            # p=[1 - max_post] + [max_post / max_amount] * max_amount
            # This assumes max_post <= 1.
            # And sum is (1-max) + max = 1.
            return np.random.choice(max_amount + 1,
                                    p=[1 - max_post] +
                                    [max_post / max_amount] * max_amount)
        return _amount_random_based

    @classmethod
    def _amount_accuracy_gen(cls, amount_function: Callable, max_amount=3, static=True):
        def _amount_accuracy_static(feedbacks: List[float], amount_function: Callable = amount_function, **kwargs) -> int:
            if not feedbacks: return max_amount
            feedbacks = [int(f) for f in feedbacks]
            accuracy = np.sum(feedbacks) / len(feedbacks)
            amount = amount_function(accuracy)
            match amount:
                case int():
                    return amount if amount < max_amount else max_amount
                case Callable():
                    return amount(**kwargs)
                case _: return max_amount

        def _amount_accuracy_delta(feedbacks: List[float], amount_function: Callable = amount_function, **kwargs) -> int:
            if not feedbacks: return max_amount
            feedbacks = [int(f) for f in feedbacks]
            length = 8
            if len(feedbacks) < length: return max_amount # Not enough data
            old_acc = np.sum(feedbacks[:length]) / length
            new_acc = np.sum(feedbacks[length:]) / length
            delta_acc = new_acc - old_acc
            amount = amount_function(delta_acc)
            match amount:
                case int():
                    return amount if amount < max_amount else max_amount
                case Callable():
                    return amount(**kwargs)
                case _: return max_amount

        return _amount_accuracy_static if static else _amount_accuracy_delta

    @classmethod
    def _amount_opposite_random_gen(cls, max_amount=7):
        base_rand = cls._amount_random_gen(max_amount)
        def _opposite_random(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            return max_amount - base_rand(posterior, max_amount=max_amount, **kwargs)
        return _opposite_random

    @classmethod
    def _amount_confidence_gen(cls, max_amount=7, threshold_min=0.2):
        """
        Generates amount based on confidence (max posterior).
        Mimics the original step function:
        <= 0.2 -> 0
        0.2-0.3 -> 1
        ...
        >= 0.8 -> 7 (if max_amount=7)
        
        MODIFIED: Returns at least 1 to prevent total memory loss in large hypothesis spaces.
        """
        def _amount_confidence(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            max_post = np.max(posterior)
            if max_post <= threshold_min:
                # Return 1 instead of 0 to ensure we keep at least one hypothesis
                # (the best one or a lucky weighted one) even when confidence is low.
                return 1
            
            # Step function: (max_post - 0.2) * 10 + 1
            val = int((max_post - threshold_min) * 10) + 1
            return max(1, min(val, max_amount))
        return _amount_confidence

    @classmethod
    def _amount_opposite_confidence_gen(cls, max_amount=7, threshold_min=0.2):
        base_func = cls._amount_confidence_gen(max_amount, threshold_min)
        def _amount_opp_confidence(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            conf_amount = base_func(posterior, max_amount, **kwargs)
            return max(0, max_amount - conf_amount)
        return _amount_opp_confidence

    
    def _cluster_strategy_ksimilar_centers(self,
                                           amount: int,
                                           exclude: Set[int],
                                           posterior: np.ndarray,
                                           strategy_config: Dict,
                                           **kwargs):
        """
        Cluster strategy: ksimilar distance version
        """
        # 0. Check prerequisites
        if not hasattr(self.engine, "partition") or self.engine.partition is None:
            return []
        
        # 1. Prepare available hypotheses
        available_hypos = self._exclude(self.full_indices, np.array(list(exclude)))
        if len(available_hypos) == 0:
            return []
        
        # 2. Get stimulus
        stimulus = None
        if self.engine.observation is not None:
             # Assuming observation[0] is stimulus
             try:
                stimulus = np.array(self.engine.observation[0])
             except:
                pass
        
        if stimulus is None:
             return self._sample_from_pool(available_hypos, amount).tolist()

        # 3. Prepare reference hypotheses
        # Use currently active hypotheses as reference
        if self.active is None or len(self.active) == 0:
             return self._sample_from_pool(available_hypos, amount).tolist()
        
        active_indices = self.active
        active_probs = posterior[active_indices]
        
        # Sort by posterior
        ref_hypos = sorted(zip(active_indices, active_probs), key=lambda x: x[1], reverse=True)
        
        proto_hypo_amount = strategy_config.get("proto_hypo_amount", 1)
        proto_hypo_method = strategy_config.get("proto_hypo_method", "top")
        
        if proto_hypo_method == "top":
            ref_hypos = ref_hypos[:proto_hypo_amount]
        elif proto_hypo_method == "random":
            # Weighted sample
            probs = np.array([x[1] for x in ref_hypos])
            if probs.sum() > 0:
                probs /= probs.sum()
                indices = np.random.choice(len(ref_hypos), size=min(len(ref_hypos), proto_hypo_amount), p=probs, replace=False)
                ref_hypos = [ref_hypos[i] for i in indices]
            else:
                ref_hypos = ref_hypos[:proto_hypo_amount]
        else:
            ref_hypos = ref_hypos[:proto_hypo_amount]

        proto_hypo_amount = len(ref_hypos)
        if proto_hypo_amount == 0:
             return self._sample_from_pool(available_hypos, amount).tolist()

        ref_hypos_index = np.array([x[0] for x in ref_hypos])
        ref_hypos_post = np.array([x[1] for x in ref_hypos])
        # Assume beta is constant or passed in kwargs, default 1.0
        # In old code, beta was part of posterior. Here we don't have it.
        # Use self.strategy_params or kwargs
        beta_val = kwargs.get("beta", self.strategy_params.get("beta", 1.0))
        ref_hypos_beta = np.array([beta_val] * proto_hypo_amount)

        # ref_full_centers: shape (proto_hypo_amount, n_cats, n_dims)
        # self.engine.partition.centers[k] is (split_type, {cat_idx: center_tuple})
        try:
            ref_full_centers = np.array([
                list(self.engine.partition.centers[k][1].values())
                for k in ref_hypos_index
            ])
        except Exception as e:
            if self.debug: print(f"Error getting centers: {e}")
            return self._sample_from_pool(available_hypos, amount).tolist()

        n_dims = self.engine.partition.n_dims
        n_cats = self.engine.partition.n_cats

        # Calculate distance from stimulus to all centers of reference hypos
        # stimulus: (n_dims,) -> (1, n_dims)
        # ref_full_centers: (proto, n_cats, n_dims) -> (proto*n_cats, n_dims)
        ref_dist = cdist(
            stimulus.reshape(1, -1),
            ref_full_centers.reshape(-1, n_dims)
        ) # shape (1, proto*n_cats)
        
        # Softmax to get choice probabilities for each reference hypo
        # ref_dist reshape -> (proto, n_cats)
        # beta reshape -> (proto, 1)
        ref_dist_reshaped = ref_dist.reshape(-1, n_cats)
        ref_probs = softmax(ref_dist_reshaped, beta=-ref_hypos_beta.reshape(-1, 1), axis=1)
        
        # Sample choice for each reference hypo
        ref_choices = [
            np.random.choice(n_cats, p=prob)
            for prob in ref_probs
        ]
        
        # Get the chosen center for each reference hypo
        # ref_hypos_center shape: (proto, n_dims)
        ref_hypos_center = ref_full_centers[range(proto_hypo_amount), ref_choices]

        # Prepare candidate hypos
        candidate_hypos_index = available_hypos # already excluded
        
        # candidate_full_center: (n_candidates, n_cats, n_dims)
        candidate_full_center = np.array([
            list(self.engine.partition.centers[k][1].values())
            for k in candidate_hypos_index
        ])
        
        # Calculate similarity score
        # For each candidate, calculate distance of its center (for the SAME choice as ref) to ref center
        # But wait, "same choice"?
        # Old code:
        # exp_dist = np.exp([[
        #    -1 * self.center_dist(ref_hypos_center[i],
        #                          candidate_full_center[j, ref_choices[i]])
        #    for i, _ in enumerate(ref_hypos_index)
        # ] for j, _ in enumerate(candidate_hypos_index)])
        
        # It compares ref_center[i] (which is center of choice C_i) 
        # with candidate_center[j][C_i].
        # So it assumes the candidate would make the SAME choice?
        # Or it measures how similar the candidate's center for that choice is.
        
        scores = np.zeros(len(candidate_hypos_index))
        
        for j, cand_idx in enumerate(candidate_hypos_index):
            # For this candidate, sum similarity over all reference hypos
            sim_sum = 0.0
            for i in range(proto_hypo_amount):
                ref_c = ref_hypos_center[i] # (n_dims,)
                # Candidate center for the choice made by ref hypo i
                cand_c = candidate_full_center[j, ref_choices[i]] # (n_dims,)
                
                dist = self.center_dist(tuple(ref_c), tuple(cand_c))
                sim = np.exp(-dist) # Similarity
                sim_sum += sim * ref_hypos_post[i]
            scores[j] = sim_sum

        # Select based on scores
        cluster_hypo_method = strategy_config.get("cluster_hypo_method", "top")
        
        if cluster_hypo_method == "top":
            argscore = np.argsort(scores)[-amount:]
            ret_val = candidate_hypos_index[argscore]
        elif cluster_hypo_method == "random":
            if scores.sum() > 0:
                probs = scores / scores.sum()
                ret_val = np.random.choice(candidate_hypos_index, size=min(amount, len(candidate_hypos_index)), p=probs, replace=False)
            else:
                ret_val = self._sample_from_pool(candidate_hypos_index, amount)
        else:
            argscore = np.argsort(scores)[-amount:]
            ret_val = candidate_hypos_index[argscore]
            
        return ret_val.tolist()

    def _calc_cached_dist(self):
        """
        Calculate Cached diatances
        """
        if not hasattr(self.engine, "partition") or self.engine.partition is None:
            # If partition is not ready, skip
            return

        # Check if centers are available
        if not hasattr(self.engine.partition, "centers"):
             return

        self.cached_dist = {}
        # self.engine.partition.centers is a list of (split_type, {cat_idx: center_tuple})
        # We iterate over all pairs of hypotheses
        # This might be expensive if total_hypo is large. 
        # But usually it is done once or lazily.
        # Here we do it lazily in center_dist if needed, or precompute?
        # The old code precomputed it.
        
        # Optimization: Only compute when needed.
        pass

    def center_dist(self, this, other) -> float:
        """
        Read out center distances between two category centers (tuples).
        """
        key = (*this, *other)
        if key in self.cached_dist:
            return self.cached_dist[key]
        
        inv = (*other, *this)
        if inv in self.cached_dist:
            return self.cached_dist[inv]

        dist = np.sum((np.array(this) - np.array(other))**2)**0.5
        self.cached_dist[key] = dist
        self.cached_dist[inv] = dist
        return dist

    def process(self, **kwargs) -> None:
        # Pass kwargs to transition (e.g. feedbacks)
        self._transition(**kwargs)
        self._apply_mask()
        self._posterior_to_prior_transition()

    def _init_mask(self) -> None:
        # Simple random init
        selection = self._sample_from_pool(self.full_indices, self.init_num)
        self.active = np.sort(np.array(selection, dtype=int))
        self._apply_mask()

    def _transition(self, **kwargs) -> None:
        self.old_active = self.active.copy() if self.active is not None else None
        
        posterior = self._get_posterior_like()
        
        if self.debug:
            max_post = np.max(posterior)
            print(f"Transition Debug: Max Posterior = {max_post:.4f}")
            beta_debug = kwargs.get("beta", self.strategy_params.get("beta", "N/A"))
            print(f"Transition Debug: Beta = {beta_debug}")

        new_active_set = set()
        # Track counts for this step
        step_counts: Dict[str, int] = {}
        
        for strat in self.strategies:
            amount_type = strat.get("amount", "fixed")
            method_type = strat.get("method", "random")
            
            # 1. Calculate Amount
            count = 0
            if amount_type == "fixed":
                count = int(strat.get("value", 1))
            else:
                # Use adaptive evaluator
                # Pass posterior and kwargs (feedbacks etc)
                try:
                    count = self.adaptive_amount_evaluator(amount_type, posterior=posterior, **kwargs)
                except Exception as e:
                    if self.debug: print(f"Error in amount evaluator {amount_type}: {e}")
                    count = 1 # Fallback
            
            # 2. Select
            # If top_p is specified in strat, we might ignore count or use it as limit
            if "top_p" in strat and method_type == "top_posterior":
                 # Special handling for top_p
                 pass 

            if self.debug:
                print(f"  Strategy {method_type}: amount={count}")

            selected: List[int] = []
            if count > 0 or ("top_p" in strat and method_type == "top_posterior"):
                selected = self._select_hypotheses(method_type, count, posterior, exclude=new_active_set, strategy_config=strat, **kwargs)
                if self.debug:
                    print(f"    Selected: {selected}")
                new_active_set.update(selected)
            # Record per-strategy count
            step_counts[f"{method_type}"] = step_counts.get(f"{method_type}", 0) + len(selected)
        
        if not new_active_set:
            # Fallback if empty
            if self.debug:
                print("  No hypotheses selected! Fallback to 1 random.")
            new_active_set.update(self._sample_from_pool(self.full_indices, 1))

        self.active = np.sort(list(new_active_set))
        # Record totals for plotting/logging
        step_counts["active_total"] = len(self.active)
        # Defensive: ensure log list exists even if older instances skip __init__ field
        if not hasattr(self, "strategy_counts_log"):
            self.strategy_counts_log = []
        self.strategy_counts_log.append(step_counts)
        if self.debug:
            print(f"DynamicHypothesis: {len(self.old_active) if self.old_active is not None else 0} -> {len(self.active)} hypos")
            if 42 in self.active:
                print(f"  Hypothesis 42 is ACTIVE. Post: {posterior[42]:.4f}")
            else:
                print(f"  Hypothesis 42 is INACTIVE. Post: {posterior[42]:.4f}")

    def _cluster_strategy_random_post(self, amount: int, exclude: Set[int], posterior: np.ndarray, **kwargs) -> List[int]:
        """
        Cluster strategy: random n from posterior (Weighted Sampling from Active set)
        """
        if self.active is None:
            return []
            
        # Candidates: Active hypotheses that are not in the exclude set
        candidates = [x for x in self.active if x not in exclude]
        if not candidates:
            return []
            
        cand_indices = np.array(candidates)
        # Get weights (probabilities)
        # posterior is expected to be an array of size total_hypo
        raw_w = posterior[cand_indices]
        
        # Safety check for zero probabilities
        if raw_w.sum() == 0:
             # Fallback to uniform
             return self._sample_from_pool(cand_indices, min(amount, len(cand_indices))).tolist()

        # Normalize
        prob = raw_w / raw_w.sum()
        
        # Weighted Sample
        actual_amount = min(amount, len(cand_indices))
        
        # Handle case where non-zero probs are fewer than amount
        n_nonzero = (prob > 0).sum()
        if n_nonzero < actual_amount:
             # 1. Pick all non-zero
             non_zero_indices = cand_indices[prob > 0]
             
             # 2. Pick remainder from zero-prob uniformly
             zero_indices = cand_indices[prob == 0]
             remainder = actual_amount - len(non_zero_indices)
             
             chosen_zeros = self._sample_from_pool(zero_indices, remainder)
             return np.concatenate([non_zero_indices, chosen_zeros]).tolist()

        chosen = self.rng.choice(cand_indices, size=actual_amount, replace=False, p=prob)
        return chosen.tolist()

    def _select_hypotheses(self, method: str, count: int, posterior: np.ndarray, exclude: Set[int], strategy_config: Dict = None, **kwargs) -> List[int]:
        strategy_config = strategy_config or {}
        
        if method == "ksimilar_centers":
            return self._cluster_strategy_ksimilar_centers(count, exclude, posterior, strategy_config, **kwargs)
        
        if method == "random_posterior":
            return self._cluster_strategy_random_post(count, exclude, posterior, **kwargs)

        if count <= 0 and "top_p" not in strategy_config:
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
            
            # Top P
            if (prob := strategy_config.get("top_p", 0.)) > 0:
                # Sort by score descending
                sorted_indices = np.argsort(scores)[::-1]
                sorted_scores = scores[sorted_indices]
                
                cum_prob = 0.0
                selected_indices = []
                for idx, score in zip(sorted_indices, sorted_scores):
                    selected_indices.append(cand_indices[idx])
                    cum_prob += score
                    if cum_prob > prob:
                        break
                return selected_indices

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
    
    def _posterior_to_prior_transition(self):
        """
        Update engine.prior based on the transition from old_active to active hypotheses.
        Uses similarity-weighted sum of old posteriors to initialize new hypotheses.
        """
        # TODO: Prior estimation with "confidence"

        if self.old_active is None or self.active is None:
            return
        
        # get prior on old active set
        current_prior = self.engine.prior

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
        new_prior = current_prior.copy()
        
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
                old_probs_active = current_prior[old_indices]
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

# TODO: 现在有了similarity matrix，能不能简化dynamic hypothesis module的transition逻辑？

