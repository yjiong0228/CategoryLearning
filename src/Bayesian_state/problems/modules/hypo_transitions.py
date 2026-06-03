"""Simple fixed-number hypothesis module for the state-based engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import inspect
from typing import Sequence, List, Dict, Set, Tuple, Callable, Any
from scipy.spatial.distance import cdist
from ...utils import print, entropy, softmax

from .base_module import BaseModule
import numpy as np


class DynamicHypothesisModule(BaseModule):
    """
    Maintains a dynamic-size hypothesis mask based on entropy and other strategies.
    
    This module allows for a variable number of hypotheses to be active at any given time,
    determined by strategies such as posterior entropy.
    """
    
    POOL_ACTIVE = "active"
    POOL_INACTIVE = "inactive"
    POOL_ALL_UNSELECTED = "all_unselected"
    VALID_POOLS = (POOL_ACTIVE, POOL_INACTIVE, POOL_ALL_UNSELECTED)
    VALID_METHODS = (
        "top_posterior",
        "random_posterior",
        "random",
        "ksimilar_centers",
        "epsilon_posterior",
        "temperature_posterior",
        "diverse_posterior",
    )
    VALID_TOP_P_SCOPES = ("global", "pool")
    VALID_FEEDBACK_MODES = ("graded", "exact")
    VALID_PADDING_MODES = ("chance", "zero", "one")
    VALID_SIMILARITY_SOURCES = ("partition",)
    DEFAULT_POOL_BY_METHOD = {
        "top_posterior": POOL_ACTIVE,
        "random_posterior": POOL_ACTIVE,
        "epsilon_posterior": POOL_ACTIVE,
        "temperature_posterior": POOL_ACTIVE,
        "diverse_posterior": POOL_ACTIVE,
        "random": POOL_ALL_UNSELECTED,
        "ksimilar_centers": POOL_ALL_UNSELECTED,
    }
    amount_evaluators = {}

    def __init__(self, engine, **kwargs):
        """INITIALIZE BEFORE MEMORY MODULE"""
        super().__init__(engine, **kwargs)

        self.total_hypo = self.engine.set_size
        self.full_indices = np.arange(self.total_hypo, dtype=int)
        self.module_seed = kwargs.get("module_seed", None)
        self.rng = np.random.default_rng(self.module_seed)
        
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
            raise ValueError("Strategies configuration is required. Set strategies to 'original_strategies', 'original_strategies_a', or 'original_strategies_b', or provide a custom list of strategy dicts.")
        else:
            self.strategies = strategies_input

        # Hard cap for total active hypotheses after each transition.
        # Defaults are aligned with legacy condition-specific strategy presets.
        self.max_active_hypotheses: int | None = kwargs.get("max_active_hypotheses", None)
        if self.max_active_hypotheses is None:
            if strategies_input == "original_strategies_a":
                self.max_active_hypotheses = 4
            elif strategies_input in ("original_strategies", "original_strategies_b"):
                self.max_active_hypotheses = 7
        if self.max_active_hypotheses is not None:
            self.max_active_hypotheses = max(1, int(self.max_active_hypotheses))
            
        # Global parameters for strategies
        self.strategy_params = {
            "beta": kwargs.get("beta", 5.0) # Default to 5.0 to match likelihood
        }
        self.init_num = int(kwargs.get("init_num", 5))
        
        self.debug = kwargs.get("hypothesis_debug", False)
        # Track how many hypotheses each strategy selects per transition step (for plotting)
        self.strategy_counts_log: List[Dict[str, Any]] = []
        self.feedback_history = deque(maxlen=max(1, int(kwargs.get("history_maxlen", 50))))

        self.cached_dist: Dict[Tuple, float] = {}

        # Register amount evaluators. The strategy runner treats this as the
        # single supported amount registry; unknown strings fail fast.
        self.amount_evaluators = {
            "entropy": self._amount_entropy_gen(3),
        }
        for amount in range(1, 10):
            self.amount_evaluators[f"entropy_{amount}"] = self._amount_entropy_gen(amount)
            self.amount_evaluators[f"opp_entropy_{amount}"] = self._amount_opposite_entropy_gen(amount)
            self.amount_evaluators[f"entropy_norm_{amount}"] = self._amount_entropy_norm_gen(amount)
            self.amount_evaluators[f"opp_entropy_norm_{amount}"] = self._amount_opposite_entropy_norm_gen(amount)
            self.amount_evaluators[f"max_{amount}"] = self._amount_max_gen(amount)
            self.amount_evaluators[f"random_{amount}"] = self._amount_random_gen(amount)
            self.amount_evaluators[f"opp_random_{amount}"] = self._amount_opposite_random_gen(amount)
            self.amount_evaluators[f"confidence_{amount}"] = self._amount_confidence_gen(amount)
            self.amount_evaluators[f"opp_confidence_{amount}"] = self._amount_opposite_confidence_gen(amount)
            self.amount_evaluators[f"recent_accuracy_inverse_{amount}"] = self._amount_recent_accuracy_inverse_gen(amount)
        self.method_selectors = {
            "top_posterior": self._select_top_posterior,
            "random_posterior": self._select_random_posterior,
            "random": self._select_random,
            "ksimilar_centers": self._cluster_strategy_ksimilar_centers,
            "epsilon_posterior": self._select_epsilon_posterior,
            "temperature_posterior": self._select_temperature_posterior,
            "diverse_posterior": self._select_diverse_posterior,
        }
        self.strategies = self._validate_strategies(self.strategies)
        history_maxlen = kwargs.get("history_maxlen", None)
        required_history = self._required_history_maxlen(self.strategies)
        self.uses_feedback_history = required_history > 0
        if history_maxlen is None:
            history_maxlen = max(required_history, 1)
        history_maxlen = self._validate_count(history_maxlen, context="history_maxlen")
        if history_maxlen <= 0:
            raise ValueError("history_maxlen must be positive.")
        if required_history > history_maxlen:
            raise ValueError(
                f"history_maxlen={history_maxlen} is smaller than required history window {required_history}."
            )
        self.feedback_history = deque(maxlen=history_maxlen)
        self._validate_history_feedback_modes()
        
        self.active: np.ndarray | None = None
        self.old_active: np.ndarray | None = None
        self._init_mask()

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
            {"amount": "random_4", "method": "random_posterior", "pool": "active", "top_p": 0.0},
            # 2. Exploration: entropy complement (High Entropy -> Explore more)
            {"amount": "opp_random_4", "method": "random", "pool": "all_unselected"},
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
            {"amount": "entropy_7", "method": "random_posterior", "pool": "active"},
            # 2. Exploration: entropy complement (entropy low -> fewer random)
            {"amount": "opp_entropy_7", "method": "random", "pool": "all_unselected"},
            # 3. Association: Add similar hypotheses
            {"amount": "fixed", "method": "ksimilar_centers", "pool": "all_unselected", "value": 1, 
             "proto_hypo_amount": 1, "proto_hypo_method": "top", "cluster_hypo_method": "top"}
        ]
    
    @classmethod
    def get_original_strategy_config(cls) -> List[Dict]:
        """
        Deprecated. Alias for get_original_strategy_config_b.
        """
        return cls.get_original_strategy_config_b()

    def _validate_strategies(self, strategies: Any) -> List[Dict[str, Any]]:
        if not isinstance(strategies, list) or not strategies:
            raise ValueError("strategies must be a non-empty list of strategy dictionaries.")

        validated: List[Dict[str, Any]] = []
        for idx, raw in enumerate(strategies):
            if not isinstance(raw, dict):
                raise ValueError(f"Strategy #{idx} must be a dict, got {type(raw).__name__}.")
            missing = [key for key in ("amount", "method") if key not in raw]
            if missing:
                raise ValueError(
                    f"Strategy #{idx} is missing required key(s): {', '.join(missing)}. "
                    "Each strategy must set amount and method."
                )

            strat = dict(raw)
            method = str(strat["method"])
            if method not in self.method_selectors:
                raise ValueError(
                    f"Strategy #{idx} has unsupported method '{method}'. "
                    f"Supported methods: {', '.join(self.VALID_METHODS)}."
                )
            if "pool" not in strat:
                strat["pool"] = self.DEFAULT_POOL_BY_METHOD.get(method)
            pool = str(strat["pool"])
            if pool not in self.VALID_POOLS:
                raise ValueError(
                    f"Strategy #{idx} has unsupported pool '{pool}'. "
                    f"Supported pools: {', '.join(self.VALID_POOLS)}."
                )
            if method == "ksimilar_centers":
                self._validate_ksimilar_partition()
            # Validate amount names up front when possible.
            amount = strat["amount"]
            if isinstance(amount, str) and amount != "fixed" and amount not in self.amount_evaluators:
                raise ValueError(
                    f"Strategy #{idx} has unsupported amount '{amount}'. "
                    "Use 'fixed' or a registered amount evaluator."
                )
            if amount == "fixed" and "value" not in strat:
                raise ValueError(f"Strategy #{idx} uses fixed amount but does not define value.")
            if amount == "fixed":
                self._validate_count(strat["value"], context=f"Strategy #{idx} fixed amount")
            self._validate_strategy_parameters(idx, strat)
            validated.append(strat)
        return validated

    def _validate_strategy_parameters(self, idx: int, strat: Dict[str, Any]) -> None:
        method = str(strat["method"])
        amount = strat["amount"]

        if "label" in strat and not str(strat["label"]).strip():
            raise ValueError(f"Strategy #{idx} label must be non-empty when provided.")

        if method == "top_posterior":
            self._validate_float_range(strat.get("top_p", 0.0), "top_p", 0.0, 1.0)
            top_p_scope = str(strat.get("top_p_scope", "global"))
            if top_p_scope not in self.VALID_TOP_P_SCOPES:
                raise ValueError(
                    f"Strategy #{idx} has unsupported top_p_scope '{top_p_scope}'. "
                    f"Supported values: {', '.join(self.VALID_TOP_P_SCOPES)}."
                )

        if method == "epsilon_posterior":
            self._validate_float_range(strat.get("epsilon", 0.25), "epsilon", 0.0, 1.0)

        if method == "temperature_posterior":
            self._validate_positive_float(strat.get("temperature", 1.0), "temperature")
            self._validate_positive_float(strat.get("weight_floor", 1e-12), "weight_floor")

        if method == "diverse_posterior":
            self._validate_float_range(strat.get("diversity_lambda", 0.6), "diversity_lambda", 0.0, 1.0)
            similarity_source = str(strat.get("similarity_source", "partition"))
            if similarity_source not in self.VALID_SIMILARITY_SOURCES:
                raise ValueError(
                    f"Strategy #{idx} has unsupported similarity_source '{similarity_source}'. "
                    f"Supported values: {', '.join(self.VALID_SIMILARITY_SOURCES)}."
                )
            self._validate_similarity_interface()

        if method == "ksimilar_centers":
            proto_method = str(strat.get("proto_hypo_method", "top"))
            cluster_method = str(strat.get("cluster_hypo_method", "top"))
            if proto_method not in ("top", "random"):
                raise ValueError(f"Strategy #{idx} has unsupported proto_hypo_method '{proto_method}'.")
            if cluster_method not in ("top", "random"):
                raise ValueError(f"Strategy #{idx} has unsupported cluster_hypo_method '{cluster_method}'.")
            self._validate_count(strat.get("proto_hypo_amount", 1), context=f"Strategy #{idx} proto_hypo_amount")

        if isinstance(amount, str) and amount.startswith(("confidence_", "opp_confidence_")):
            self._validate_float_range(strat.get("threshold_min", 0.2), "threshold_min", 0.0, 1.0)
            self._validate_positive_float(strat.get("scale", 10.0), "scale")
            self._validate_count(strat.get("min_count", 1), context=f"Strategy #{idx} min_count")

        if isinstance(amount, str) and amount.startswith("max_"):
            self._validate_positive_float(strat.get("reference_mass", 3.0), "reference_mass")

        if isinstance(amount, str) and amount.startswith(("entropy_norm_", "opp_entropy_norm_")):
            self._validate_count(strat.get("min_count", 0), context=f"Strategy #{idx} min_count")

        if isinstance(amount, str) and amount.startswith("recent_accuracy_inverse_"):
            self._validate_history_strategy(idx, strat)

    def _validate_history_strategy(self, idx: int, strat: Dict[str, Any]) -> None:
        window = self._validate_count(strat.get("window", 10), context=f"Strategy #{idx} window")
        if window <= 0:
            raise ValueError(f"Strategy #{idx} window must be positive.")
        min_count = self._validate_count(strat.get("min_count", 1), context=f"Strategy #{idx} min_count")
        if min_count < 0:
            raise ValueError(f"Strategy #{idx} min_count must be non-negative.")
        self._validate_positive_float(strat.get("gamma", 1.0), "gamma")
        feedback_mode = str(strat.get("feedback_mode", "graded"))
        if feedback_mode not in self.VALID_FEEDBACK_MODES:
            raise ValueError(
                f"Strategy #{idx} has unsupported feedback_mode '{feedback_mode}'. "
                f"Supported values: {', '.join(self.VALID_FEEDBACK_MODES)}."
            )
        padding = strat.get("padding", "chance")
        if isinstance(padding, str):
            if padding not in self.VALID_PADDING_MODES:
                raise ValueError(
                    f"Strategy #{idx} has unsupported padding '{padding}'. "
                    f"Supported values: {', '.join(self.VALID_PADDING_MODES)} or a numeric value."
                )
            if padding == "chance":
                self._chance_padding_value(require=True)
        else:
            self._validate_float_range(padding, "padding", 0.0, 1.0)

    @staticmethod
    def _validate_float_range(value: Any, name: str, low: float, high: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number in [{low}, {high}], got boolean.")
        val = float(value)
        if not np.isfinite(val) or val < low or val > high:
            raise ValueError(f"{name} must be in [{low}, {high}], got {value!r}.")
        return val

    @staticmethod
    def _validate_positive_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite number, got boolean.")
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
        return val

    def _required_history_maxlen(self, strategies: Sequence[Dict[str, Any]]) -> int:
        required = 0
        for strat in strategies:
            amount = strat.get("amount")
            if isinstance(amount, str) and amount.startswith("recent_accuracy_inverse_"):
                required = max(required, self._validate_count(strat.get("window", 10), context="history window"))
        return required

    def _validate_history_feedback_modes(self) -> None:
        self._history_feedback_mode()

    def _validate_ksimilar_partition(self) -> None:
        partition = getattr(self.engine, "partition", None)
        missing = [
            name for name in ("prototypes", "n_dims", "n_cats")
            if partition is None or not hasattr(partition, name)
        ]
        if missing:
            raise ValueError(
                "ksimilar_centers requires a prototype-backed partition with "
                f"attributes prototypes, n_dims, and n_cats. Missing: {', '.join(missing)}."
            )

    @staticmethod
    def _validate_count(count: Any, *, context: str) -> int:
        if isinstance(count, bool):
            raise ValueError(f"{context} produced a boolean amount, expected a non-negative integer.")
        if not np.isfinite(count):
            raise ValueError(f"{context} produced a non-finite amount: {count!r}.")
        int_count = int(count)
        if int_count != count:
            raise ValueError(f"{context} produced a non-integer amount: {count!r}.")
        if int_count < 0:
            raise ValueError(f"{context} produced a negative amount: {int_count}.")
        return int_count

    @staticmethod
    def _validate_probability_vector(prob: np.ndarray, *, context: str) -> np.ndarray:
        prob = np.asarray(prob, dtype=float)
        if prob.ndim != 1:
            raise ValueError(f"{context} probabilities must be 1-D, got shape {prob.shape}.")
        if not np.all(np.isfinite(prob)):
            raise ValueError(f"{context} probabilities contain non-finite values.")
        if np.any(prob < 0):
            raise ValueError(f"{context} probabilities contain negative values.")
        total = float(prob.sum())
        if total <= 0:
            raise ValueError(f"{context} probabilities sum to zero.")
        return prob / total

    def _candidate_relevance_scores(self, scores: np.ndarray, *, context: str) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError(f"{context} scores must be 1-D, got shape {scores.shape}.")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{context} scores contain non-finite values.")
        if np.any(scores < 0):
            raise ValueError(f"{context} scores contain negative values.")
        total = float(scores.sum())
        if total > 0.0:
            return scores / total
        if scores.size == 0:
            return scores
        return np.full(scores.shape, 1.0 / scores.size, dtype=float)

    def adaptive_amount_evaluator(self, amount: float | str | Callable, **kwargs) -> int:
        """
        Adaptively deal with evaluator / number format of amount.
        """
        if isinstance(amount, int):
            return self._validate_count(amount, context="integer amount")
        elif callable(amount):
            kwargs.setdefault("rng", self.rng)
            return self._validate_count(amount(**kwargs), context="callable amount")
        elif isinstance(amount, str):
            if amount in self.amount_evaluators:
                kwargs.setdefault("rng", self.rng)
                return self._validate_count(
                    self.amount_evaluators[amount](**kwargs),
                    context=f"amount evaluator '{amount}'",
                )
            raise ValueError(f"Unsupported amount evaluator '{amount}'.")
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
    def _amount_entropy_norm_gen(cls, max_amount=3):
        def _amount_entropy_norm(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            min_count = int(strategy_config.get("min_count", 0))
            p_entropy = entropy(posterior)
            n_hypos = len(posterior)
            max_possible_entropy = np.log(n_hypos) if n_hypos > 1 else 1.0
            normalized_entropy = np.clip(p_entropy / max_possible_entropy, 0.0, 1.0)
            return int(min(max_amount, max(min_count, round(max_amount * (1.0 - normalized_entropy)))))
        return _amount_entropy_norm

    @classmethod
    def _amount_opposite_entropy_norm_gen(cls, max_amount=3):
        def _amount_opposite_entropy_norm(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            min_count = int(strategy_config.get("min_count", 0))
            p_entropy = entropy(posterior)
            n_hypos = len(posterior)
            max_possible_entropy = np.log(n_hypos) if n_hypos > 1 else 1.0
            normalized_entropy = np.clip(p_entropy / max_possible_entropy, 0.0, 1.0)
            return int(min(max_amount, max(min_count, round(max_amount * normalized_entropy))))
        return _amount_opposite_entropy_norm

    @classmethod
    def _amount_max_gen(cls, max_amount=3):
        def _amount_max_based(posterior: np.ndarray, max_amount=max_amount, **kwargs):
            strategy_config = kwargs.get("strategy_config", {}) or {}
            reference_mass = float(strategy_config.get("reference_mass", 3.0))
            max_post = np.max(posterior)
            if max_post <= 0: return max_amount # Avoid div by zero
            return 0 if reference_mass / max_post > max_amount else int(reference_mass / max_post)
        return _amount_max_based

    @classmethod
    def _amount_random_gen(cls, max_amount=3):
        def _amount_random_based(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            max_post = np.max(posterior)
            if not np.isfinite(max_post) or max_post < 0.0 or max_post > 1.0:
                raise ValueError(f"random amount requires max posterior in [0, 1], got {max_post!r}.")
            rng = kwargs.get("rng")
            if rng is None:
                raise ValueError("random amount evaluator requires an rng.")
            # p=[1 - max_post] + [max_post / max_amount] * max_amount
            # This assumes max_post <= 1.
            # And sum is (1-max) + max = 1.
            probs = np.array([1 - max_post] + [max_post / max_amount] * max_amount, dtype=float)
            probs = probs / probs.sum()
            return rng.choice(max_amount + 1, p=probs)
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
            strategy_config = kwargs.get("strategy_config", {}) or {}
            threshold = float(strategy_config.get("threshold_min", threshold_min))
            scale = float(strategy_config.get("scale", 10.0))
            min_count = int(strategy_config.get("min_count", 1))
            max_post = np.max(posterior)
            if max_post <= threshold:
                # Return 1 instead of 0 to ensure we keep at least one hypothesis
                # (the best one or a lucky weighted one) even when confidence is low.
                return min(min_count, max_amount)
            
            # Step function: (max_post - 0.2) * 10 + 1
            val = int((max_post - threshold) * scale) + min_count
            return min(max_amount, max(min_count, min(val, max_amount)))
        return _amount_confidence

    @classmethod
    def _amount_opposite_confidence_gen(cls, max_amount=7, threshold_min=0.2):
        base_func = cls._amount_confidence_gen(max_amount, threshold_min)
        def _amount_opp_confidence(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            conf_amount = base_func(posterior, max_amount=max_amount, **kwargs)
            return max(0, max_amount - conf_amount)
        return _amount_opp_confidence

    def _amount_recent_accuracy_inverse_gen(self, max_amount=7):
        def _amount_recent_accuracy_inverse(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            window = self._validate_count(strategy_config.get("window", 10), context="recent accuracy window")
            min_count = self._validate_count(strategy_config.get("min_count", 1), context="recent accuracy min_count")
            gamma = float(strategy_config.get("gamma", 1.0))
            if window <= 0:
                raise ValueError("recent_accuracy_inverse window must be positive.")
            if gamma <= 0 or not np.isfinite(gamma):
                raise ValueError(f"recent_accuracy_inverse gamma must be positive, got {gamma!r}.")
            if min_count > max_amount:
                return max_amount
            acc = self._recent_accuracy(window, strategy_config)
            amount = min_count + round(((1.0 - acc) ** gamma) * (max_amount - min_count))
            return int(max(min_count, min(amount, max_amount)))
        return _amount_recent_accuracy_inverse

    def _recent_accuracy(self, window: int, strategy_config: Dict[str, Any]) -> float:
        recent = list(self.feedback_history)[-window:]
        padding = strategy_config.get("padding", "chance")
        if len(recent) >= window:
            values = recent
        else:
            missing = window - len(recent)
            pad_value = self._resolve_padding_value(padding)
            values = recent + [pad_value] * missing
        if not values:
            return self._resolve_padding_value(padding)
        accuracy = float(np.mean(values))
        if not np.isfinite(accuracy):
            raise ValueError("recent accuracy is non-finite.")
        return float(np.clip(accuracy, 0.0, 1.0))

    def _resolve_padding_value(self, padding: Any) -> float:
        if isinstance(padding, str):
            if padding == "chance":
                return self._chance_padding_value(require=True)
            if padding == "zero":
                return 0.0
            if padding == "one":
                return 1.0
            raise ValueError(f"Unsupported padding '{padding}'.")
        return self._validate_float_range(padding, "padding", 0.0, 1.0)

    def _chance_padding_value(self, *, require: bool) -> float:
        partition = getattr(self.engine, "partition", None)
        n_cats = getattr(partition, "n_cats", None)
        if n_cats is None:
            if require:
                raise ValueError("padding='chance' requires engine.partition.n_cats; use numeric padding otherwise.")
            return 0.0
        n_cats = self._validate_count(n_cats, context="partition.n_cats")
        if n_cats <= 0:
            raise ValueError(f"partition.n_cats must be positive for chance padding, got {n_cats}.")
        return 1.0 / float(n_cats)

    def _record_feedback_from_observation(self) -> None:
        if not self.uses_feedback_history:
            return
        observation = getattr(self.engine, "observation", None)
        if observation is None or len(observation) < 3:
            return
        feedback = observation[2]
        try:
            value = float(feedback)
        except (TypeError, ValueError):
            raise ValueError(f"Feedback must be numeric to record transition history, got {feedback!r}.")
        if not np.isfinite(value):
            raise ValueError(f"Feedback must be finite to record transition history, got {feedback!r}.")
        mode = self._history_feedback_mode()
        if mode == "exact":
            value = 1.0 if value == 1.0 else 0.0
        else:
            value = float(np.clip(value, 0.0, 1.0))
        self.feedback_history.append(value)

    def _history_feedback_mode(self) -> str:
        modes = {
            str(strat.get("feedback_mode", "graded"))
            for strat in self.strategies
            if isinstance(strat.get("amount"), str) and str(strat.get("amount")).startswith("recent_accuracy_inverse_")
        }
        if not modes:
            return "graded"
        if len(modes) > 1:
            raise ValueError("History-based strategies must use a single feedback_mode within one module.")
        return next(iter(modes))

    
    def _cluster_strategy_ksimilar_centers(self,
                                           amount: int,
                                           candidates: Sequence[int] | np.ndarray,
                                           posterior: np.ndarray,
                                           strategy_config: Dict,
                                           **kwargs):
        """
        Cluster strategy: ksimilar distance version
        """
        self._validate_ksimilar_partition()
        amount = self._validate_count(amount, context="ksimilar amount")
        if amount <= 0:
            return []

        # 1. Prepare available hypotheses
        candidate_hypos_index = np.asarray(candidates, dtype=int)
        if len(candidate_hypos_index) == 0:
            return []
        
        # 2. Get stimulus
        if self.engine.observation is None:
            raise ValueError("ksimilar_centers requires engine.observation to be set.")
        stimulus = np.asarray(self.engine.observation[0], dtype=float)

        # 3. Prepare reference hypotheses
        # Use currently active hypotheses as reference
        if self.active is None or len(self.active) == 0:
            raise ValueError("ksimilar_centers requires a non-empty active hypothesis set.")
        
        active_indices = self.active
        active_probs = posterior[active_indices]
        
        # Sort by posterior
        ref_hypos = sorted(zip(active_indices, active_probs), key=lambda x: x[1], reverse=True)
        
        proto_hypo_amount = strategy_config.get("proto_hypo_amount", 1)
        proto_hypo_method = strategy_config.get("proto_hypo_method", "top")
        proto_hypo_amount = self._validate_count(proto_hypo_amount, context="proto_hypo_amount")
        
        if proto_hypo_method == "top":
            ref_hypos = ref_hypos[:proto_hypo_amount]
        elif proto_hypo_method == "random":
            # Weighted sample
            probs = np.array([x[1] for x in ref_hypos])
            probs = self._validate_probability_vector(probs, context="ksimilar reference")
            indices = self.rng.choice(
                len(ref_hypos),
                size=min(len(ref_hypos), proto_hypo_amount),
                p=probs,
                replace=False,
            )
            ref_hypos = [ref_hypos[i] for i in indices]
        else:
            raise ValueError(f"Unsupported proto_hypo_method '{proto_hypo_method}'.")

        proto_hypo_amount = len(ref_hypos)
        if proto_hypo_amount == 0:
            return []

        ref_hypos_index = np.array([x[0] for x in ref_hypos])
        ref_hypos_post = np.array([x[1] for x in ref_hypos])
        # Assume beta is constant or passed in kwargs, default 1.0
        # In old code, beta was part of posterior. Here we don't have it.
        # Use self.strategy_params or kwargs
        beta_val = kwargs.get("beta", self.strategy_params.get("beta", 1.0))
        ref_hypos_beta = np.array([beta_val] * proto_hypo_amount)

        # ref_full_centers: shape (proto_hypo_amount, n_cats, n_dims)
        # self.engine.partition.prototypes[k, 0] has shape [n_cats, n_dims]
        ref_full_centers = np.asarray(
            self.engine.partition.prototypes[ref_hypos_index, 0],
            dtype=float,
        )

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
            self.rng.choice(n_cats, p=self._validate_probability_vector(prob, context="ksimilar category"))
            for prob in ref_probs
        ]
        
        # Get the chosen center for each reference hypo
        # ref_hypos_center shape: (proto, n_dims)
        ref_hypos_center = ref_full_centers[range(proto_hypo_amount), ref_choices]

        # Prepare candidate hypos
        # candidate_full_center: (n_candidates, n_cats, n_dims)
        candidate_full_center = np.asarray(
            self.engine.partition.prototypes[candidate_hypos_index, 0],
            dtype=float,
        )
        
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
                probs = self._validate_probability_vector(scores, context="ksimilar candidate")
                ret_val = self.rng.choice(
                    candidate_hypos_index,
                    size=min(amount, len(candidate_hypos_index)),
                    p=probs,
                    replace=False,
                )
            else:
                ret_val = self._sample_from_pool(candidate_hypos_index, amount)
        else:
            raise ValueError(f"Unsupported cluster_hypo_method '{cluster_hypo_method}'.")
            
        return ret_val.tolist()

    def _calc_cached_dist(self):
        """
        Calculate Cached diatances
        """
        if not hasattr(self.engine, "partition") or self.engine.partition is None:
            # If partition is not ready, skip
            return

        # Check if prototypes are available.
        if not hasattr(self.engine.partition, "prototypes"):
             return

        self.cached_dist = {}
        # self.engine.partition.prototypes has shape [n_hypo, n_proto, n_cat, n_dim].
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
        self._record_feedback_from_observation()

    def _init_mask(self) -> None:
        # Simple random init
        selection = self._sample_from_pool(self.full_indices, self.init_num)
        self.active = np.sort(np.array(selection, dtype=int))
        self._apply_mask()

    def _transition(self, **kwargs) -> None:
        self.old_active = self.active.copy() if self.active is not None else None
        
        posterior = self._get_posterior_like()
        if posterior is None:
            posterior = np.ones(self.total_hypo, dtype=float)
            posterior /= posterior.sum()
        posterior = self._validate_probability_vector(posterior, context="transition posterior")
        
        if self.debug:
            max_post = np.max(posterior)
            print(f"Transition Debug: Max Posterior = {max_post:.4f}")
            beta_debug = kwargs.get("beta", self.strategy_params.get("beta", "N/A"))
            print(f"Transition Debug: Beta = {beta_debug}")

        new_active_set = set()
        # Track counts for this step. Keep method-level aggregate keys for old
        # plotting code and add structured details for strategy-level debugging.
        step_counts: Dict[str, Any] = {"strategies": []}
        
        for strat in self.strategies:
            amount_type = strat["amount"]
            method_type = strat["method"]
            pool_type = strat["pool"]
            
            # 1. Calculate Amount
            if amount_type == "fixed":
                count = self._validate_count(strat["value"], context="fixed amount")
            else:
                count = self.adaptive_amount_evaluator(
                    amount_type,
                    posterior=posterior,
                    rng=self.rng,
                    strategy_config=strat,
                    **kwargs,
                )
            requested_count = count

            # Enforce a global budget for active hypotheses.
            remaining_budget = None
            if self.max_active_hypotheses is not None:
                remaining_budget = self.max_active_hypotheses - len(new_active_set)
                if remaining_budget <= 0:
                    break
                count = min(count, remaining_budget)

            top_p_val = 0.0
            if method_type == "top_posterior":
                top_p_val = float(strat.get("top_p", 0.0))
                if not np.isfinite(top_p_val) or top_p_val < 0.0 or top_p_val > 1.0:
                    raise ValueError(f"top_p must be in [0, 1], got {top_p_val!r}.")
            has_positive_top_p = method_type == "top_posterior" and top_p_val > 0.0

            if self.debug:
                print(f"  Strategy {method_type}: pool={pool_type}, amount={count}")

            selected: List[int] = []
            if count > 0 or has_positive_top_p:
                candidates = self._resolve_pool(pool_type, new_active_set)
                selected = self._select_hypotheses(
                    method_type,
                    count,
                    candidates,
                    posterior,
                    strategy_config=strat,
                    **kwargs,
                )
                if remaining_budget is not None and len(selected) > remaining_budget:
                    selected_arr = np.array(selected, dtype=int)
                    keep_args = np.argsort(posterior[selected_arr])[-remaining_budget:]
                    selected = selected_arr[keep_args].tolist()
                if self.debug:
                    print(f"    Selected: {selected}")
                new_active_set.update(selected)
            selected = [int(x) for x in selected]
            label = str(strat.get("label", f"strategy_{len(step_counts['strategies'])}"))
            step_counts["strategies"].append({
                "label": label,
                "amount": amount_type,
                "method": method_type,
                "pool": pool_type,
                "requested_count": int(requested_count),
                "selected_count": len(selected),
                "selected": selected,
            })
            # Backward-compatible aggregate count by method for plotting.
            step_counts[f"{method_type}"] = step_counts.get(f"{method_type}", 0) + len(selected)
        
        if not new_active_set:
            fallback_idx, fallback_pool = self._fallback_best_posterior(posterior)
            new_active_set.add(fallback_idx)
            step_counts["strategies"].append({
                "label": "fallback_best_posterior",
                "amount": "fallback",
                "method": "top_posterior",
                "pool": fallback_pool,
                "requested_count": 1,
                "selected_count": 1,
                "selected": [fallback_idx],
            })
            step_counts["fallback"] = step_counts.get("fallback", 0) + 1

        self.active = np.sort(list(new_active_set))
        if self.max_active_hypotheses is not None and len(self.active) > self.max_active_hypotheses:
            keep_args = np.argsort(posterior[self.active])[-self.max_active_hypotheses:]
            self.active = np.sort(self.active[keep_args])
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

    def _resolve_pool(self, pool: str, selected: Set[int]) -> np.ndarray:
        selected_arr = np.array(list(selected), dtype=int)
        if pool == self.POOL_ALL_UNSELECTED:
            return self._exclude(self.full_indices, selected_arr)
        if self.old_active is None:
            active = np.empty(0, dtype=int)
        else:
            active = np.asarray(self.old_active, dtype=int)
        if pool == self.POOL_ACTIVE:
            return self._exclude(active, selected_arr)
        if pool == self.POOL_INACTIVE:
            inactive = self._exclude(self.full_indices, active)
            return self._exclude(inactive, selected_arr)
        raise ValueError(f"Unsupported pool '{pool}'.")

    def _fallback_best_posterior(self, posterior: np.ndarray) -> Tuple[int, str]:
        if self.old_active is not None and len(self.old_active) > 0:
            candidates = np.asarray(self.old_active, dtype=int)
            pool_label = self.POOL_ACTIVE
        else:
            candidates = self.full_indices
            pool_label = "full"

        candidates = candidates[(candidates >= 0) & (candidates < self.total_hypo)]
        if candidates.size == 0:
            candidates = self.full_indices
            pool_label = "full"

        best_arg = int(np.argmax(posterior[candidates]))
        return int(candidates[best_arg]), pool_label

    def _select_random_posterior(self, amount: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, **kwargs) -> List[int]:
        """
        Cluster strategy: random n from posterior (Weighted Sampling from Active set)
        """
        if amount <= 0:
            return []
            
        cand_indices = np.asarray(candidates, dtype=int)
        if cand_indices.size == 0:
            return []
        # Get weights (probabilities)
        # posterior is expected to be an array of size total_hypo
        raw_w = posterior[cand_indices]
        if not np.all(np.isfinite(raw_w)) or np.any(raw_w < 0):
            raise ValueError("random_posterior probabilities contain invalid values.")

        # Normalize
        actual_amount = min(amount, len(cand_indices))
        if float(raw_w.sum()) == 0.0:
            return self._sample_from_pool(cand_indices, actual_amount).tolist()

        prob = self._validate_probability_vector(raw_w, context="random_posterior")
        
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

    def _select_hypotheses(self, method: str, count: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, strategy_config: Dict | None = None, **kwargs) -> List[int]:
        strategy_config = strategy_config or {}
        selector = self.method_selectors.get(method)
        if selector is None:
            raise ValueError(f"Unsupported selection method '{method}'.")
        return selector(count, candidates, posterior, strategy_config=strategy_config, **kwargs)

    def _select_top_posterior(self, amount: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, strategy_config: Dict | None = None, **kwargs) -> List[int]:
        strategy_config = strategy_config or {}
        cand_indices = np.asarray(candidates, dtype=int)
        if cand_indices.size == 0:
            return []
        scores = posterior[cand_indices]

        if (prob := float(strategy_config.get("top_p", 0.0))) > 0:
            if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
                raise ValueError(f"top_p must be in [0, 1], got {prob!r}.")
            top_p_scope = str(strategy_config.get("top_p_scope", "global"))
            if top_p_scope == "pool":
                sorted_scores = self._validate_probability_vector(scores, context="top_posterior pool")
            elif top_p_scope == "global":
                sorted_scores = scores
            else:
                raise ValueError(f"Unsupported top_p_scope '{top_p_scope}'.")
            sorted_indices = np.argsort(scores)[::-1]
            sorted_scores = sorted_scores[sorted_indices]
            
            cum_prob = 0.0
            selected_indices = []
            for idx, score in zip(sorted_indices, sorted_scores):
                selected_indices.append(cand_indices[idx])
                cum_prob += score
                if cum_prob > prob:
                    break
            return [int(x) for x in selected_indices]

        if amount <= 0:
            return []
        if len(scores) <= amount:
            return cand_indices.tolist()
        
        top_args = np.argsort(scores)[-amount:]
        return cand_indices[top_args].tolist()

    def _select_random(self, amount: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, **kwargs) -> List[int]:
        if amount <= 0:
            return []
        return self._sample_from_pool(candidates, amount).tolist()

    def _select_epsilon_posterior(self, amount: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, strategy_config: Dict | None = None, **kwargs) -> List[int]:
        strategy_config = strategy_config or {}
        if amount <= 0:
            return []
        cand_indices = np.asarray(candidates, dtype=int)
        if cand_indices.size == 0:
            return []
        actual_amount = min(amount, len(cand_indices))
        epsilon = float(strategy_config.get("epsilon", 0.25))
        if not np.isfinite(epsilon) or epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}.")
        raw = np.asarray(posterior[cand_indices], dtype=float)
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError("epsilon_posterior candidate posterior contains invalid values.")
        posterior_mass = float(raw.sum())
        posterior_part = raw / posterior_mass if posterior_mass > 0.0 else np.zeros_like(raw)
        uniform_part = np.full_like(raw, 1.0 / len(raw), dtype=float)
        prob = (1.0 - epsilon) * posterior_part + epsilon * uniform_part
        prob = self._validate_probability_vector(prob, context="epsilon_posterior")
        chosen = self.rng.choice(cand_indices, size=actual_amount, replace=False, p=prob)
        return chosen.tolist()

    def _select_temperature_posterior(self, amount: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, strategy_config: Dict | None = None, **kwargs) -> List[int]:
        strategy_config = strategy_config or {}
        if amount <= 0:
            return []
        cand_indices = np.asarray(candidates, dtype=int)
        if cand_indices.size == 0:
            return []
        actual_amount = min(amount, len(cand_indices))
        temperature = float(strategy_config.get("temperature", 1.0))
        weight_floor = float(strategy_config.get("weight_floor", 1e-12))
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature!r}.")
        if not np.isfinite(weight_floor) or weight_floor <= 0.0:
            raise ValueError(f"weight_floor must be positive, got {weight_floor!r}.")
        raw = np.asarray(posterior[cand_indices], dtype=float)
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError("temperature_posterior candidate posterior contains invalid values.")
        weights = np.power(raw + weight_floor, 1.0 / temperature)
        prob = self._validate_probability_vector(weights, context="temperature_posterior")
        chosen = self.rng.choice(cand_indices, size=actual_amount, replace=False, p=prob)
        return chosen.tolist()

    def _select_diverse_posterior(self, amount: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, strategy_config: Dict | None = None, **kwargs) -> List[int]:
        strategy_config = strategy_config or {}
        if amount <= 0:
            return []
        cand_indices = np.asarray(candidates, dtype=int)
        if cand_indices.size == 0:
            return []
        actual_amount = min(amount, len(cand_indices))
        lambda_val = float(strategy_config.get("diversity_lambda", 0.6))
        if not np.isfinite(lambda_val) or lambda_val < 0.0 or lambda_val > 1.0:
            raise ValueError(f"diversity_lambda must be in [0, 1], got {lambda_val!r}.")
        sim = self._get_similarity_matrix(require=True)
        posterior_scores = posterior[cand_indices]
        posterior_scores = self._candidate_relevance_scores(posterior_scores, context="diverse_posterior")

        selected: List[int] = []
        remaining = cand_indices.tolist()
        score_lookup = {int(idx): float(score) for idx, score in zip(cand_indices, posterior_scores)}
        while remaining and len(selected) < actual_amount:
            best_idx = None
            best_score = -np.inf
            for idx in remaining:
                if selected:
                    similarity_penalty = float(np.max(sim[int(idx), selected]))
                else:
                    similarity_penalty = 0.0
                score = lambda_val * score_lookup[int(idx)] - (1.0 - lambda_val) * similarity_penalty
                if score > best_score:
                    best_score = score
                    best_idx = int(idx)
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    def _get_similarity_matrix(self, *, require: bool) -> np.ndarray | None:
        partition = getattr(self.engine, "partition", None)
        matrix = getattr(partition, "similarity_matrix", None)
        if matrix is None and partition is not None and hasattr(partition, "get_similarity_matrix"):
            matrix = partition.get_similarity_matrix()
        if matrix is None:
            if require:
                raise ValueError("diverse_posterior requires partition.similarity_matrix or get_similarity_matrix().")
            return None
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (self.total_hypo, self.total_hypo):
            raise ValueError(
                f"similarity matrix must have shape {(self.total_hypo, self.total_hypo)}, got {matrix.shape}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("similarity matrix contains non-finite values.")
        return matrix

    def _validate_similarity_interface(self) -> None:
        partition = getattr(self.engine, "partition", None)
        if partition is None:
            raise ValueError("diverse_posterior requires engine.partition.")
        has_matrix = inspect.getattr_static(partition, "similarity_matrix", None) is not None
        has_getter = inspect.getattr_static(partition, "get_similarity_matrix", None) is not None
        if not has_matrix and not has_getter:
            raise ValueError("diverse_posterior requires partition.similarity_matrix or get_similarity_matrix().")

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
        Uses a heuristic mixture of Similarity (Exploitation) and Novelty (Exploration)
        to initialize new hypotheses.
        """
        if self.old_active is None or self.active is None:
            return
        
        # 1. Get current posterior (from previous step)
        current_posterior = None
        if hasattr(self.engine, "posterior") and self.engine.posterior is not None:
            current_posterior = self.engine.posterior.copy()
        
        if current_posterior is None:
            # Fallback: uniform
            mask = np.zeros(self.total_hypo, dtype=float)
            mask[self.active] = 1.0
            self.engine.prior = mask / mask.sum()
            return
            
        # 2. Identify Survivors vs Newcomers
        old_indices = self.old_active
        active_indices = self.active
        
        # Boolean masks
        is_survivor = np.isin(active_indices, old_indices)
        survivor_indices = active_indices[is_survivor]
        newcomer_indices = active_indices[~is_survivor]

        # 3. Calculate Confidence of the previous step
        # using max posterior probability of the *original* global posterior
        confidence = np.max(current_posterior) if len(current_posterior) > 0 else 0.0

        # Prepare normalized old values for matrix product
        old_posterior_values = current_posterior[old_indices].copy()
        if old_posterior_values.sum() > 0:
            old_posterior_values /= old_posterior_values.sum()
        
        # 4. Initialize new prior
        # Survivors carry over their previous posterior (raw values from engine.posterior)
        new_prior = np.zeros(self.total_hypo, dtype=float)
        new_prior[survivor_indices] = current_posterior[survivor_indices]

        # 5. Calculate Prior for Newcomers
        if len(newcomer_indices) > 0:
            partition = getattr(self.engine, "partition", None)
            similarity_matrix = getattr(partition, "similarity_matrix", None)

            if similarity_matrix is not None:
                # S[new, old]
                sim_sub = similarity_matrix[np.ix_(newcomer_indices, old_indices)]

                # Component A: Similarity-based (Likelihood propagation)
                # "Similar to good is good" (Exploitation)
                p_sim = sim_sub @ old_posterior_values

                # Component B: Novelty-based (Repulsion)
                # "Dissimilar to bad is good" (Exploration)
                max_sim_to_old = np.max(sim_sub, axis=1) # (n_new,)
                p_nov = 1.0 - max_sim_to_old
                p_nov = np.clip(p_nov, 0, 1)

                # Mixture based on Confidence
                # High Confidence -> Trust similarity (Exploit)
                # Low Confidence -> Trust novelty (Explore)
                raw_score = confidence * p_sim + (1.0 - confidence) * p_nov
                
                # SCALING Factor applied to Newcomers
                # If Confident, new hypotheses should have low prior mass to preserve survivor dominance.
                # If Uncertain, new hypotheses should have higher mass to encourage replacement.
                # min_scale ensures we don't zero out completely (keeps 5% budget for pure exploration).
                scale_factor = max(1.0 - confidence, 0.05)
                
                prior_newcomers = scale_factor * raw_score
                new_prior[newcomer_indices] = prior_newcomers
            else:
                # Fallback if no similarity matrix: uniform scaled by uncertainty
                scale_factor = max(1.0 - confidence, 0.05)
                # Distribute scale_factor * avg_survivor_mass? 
                # Or just assign relative to current survivors.
                # Heuristic: Assign small value relative to max.
                fill_val = (1.0 / len(self.active)) * scale_factor
                new_prior[newcomer_indices] = fill_val

        # 6. Normalize
        total_mass = new_prior.sum()
        if total_mass > 0:
            new_prior /= total_mass
        else:
            new_prior[self.active] = 1.0 / len(self.active)

        self.engine.prior = new_prior
        
        # 7. Initialize beta for newcomers via BetaModule
        self._initialize_beta_for_newcomers(newcomer_indices, new_prior)

    def _initialize_beta_for_newcomers(self, newcomer_indices: np.ndarray, prior: np.ndarray) -> None:
        """
        Initialize beta values for newly added hypotheses using BetaModule.
        
        Parameters
        ----------
        newcomer_indices : np.ndarray
            Indices of newly added hypotheses.
        prior : np.ndarray
            Prior probability distribution.
        """
        if len(newcomer_indices) == 0:
            return
            
        # Check if BetaModule exists
        beta_mod = self.engine.modules.get("beta_mod", None)
        if beta_mod is not None and hasattr(beta_mod, "initialize_beta_for_hypotheses"):
            beta_mod.initialize_beta_for_hypotheses(newcomer_indices, prior)


# TODO: 现在有了similarity matrix，能不能简化dynamic hypothesis module的transition逻辑？
