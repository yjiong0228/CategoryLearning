"""Simple fixed-number hypothesis module for the state-based engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
    VALID_POOLS = (POOL_ACTIVE, POOL_INACTIVE)
    VALID_METHODS = (
        "top_posterior",
        "random_posterior",
        "random",
        "ksimilar_centers",
        "epsilon_posterior",
        "temperature_posterior",
        "low_posterior",
    )
    VALID_TOP_P_SCOPES = ("global", "pool")
    VALID_FEEDBACK_MODES = ("graded", "exact")
    VALID_PADDING_MODES = ("chance", "zero", "one")
    VALID_POST_TO_PRIOR_METHODS = (
        "similarity_novelty",
        "conservative_carryover",
        "error_boost_newcomers",
        "stochastic_reset",
    )
    VALID_POST_TO_PRIOR_CONFIDENCE_SOURCES = (
        "max_posterior",
        "entropy",
        "recent_accuracy",
        "latent_volatility",
    )
    VALID_PRIOR_RESET_TARGETS = (
        "uniform_active",
        "newcomer_boost",
        "sampled_active",
        "sampled_newcomer",
    )
    VALID_PRIOR_RESET_SOURCES = ("feedback", "latent_volatility")
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
        if strategies_input is None:
            raise ValueError("Strategies configuration is required. Provide a non-empty list of strategy dicts.")
        if isinstance(strategies_input, str):
            raise ValueError(
                "String strategy shortcuts are no longer supported. "
                "Provide an explicit list of strategy dicts with amount, method, and pool."
            )
        self.strategies = strategies_input

        # Hard cap for total active hypotheses after each transition.
        self.max_active_hypotheses: int | None = kwargs.get("max_active_hypotheses", None)
        if self.max_active_hypotheses is not None:
            self.max_active_hypotheses = max(1, int(self.max_active_hypotheses))
            
        # Global parameters for strategies
        self.strategy_params = {
            "beta": kwargs.get("beta", 5.0) # Default to 5.0 to match likelihood
        }
        self.init_num = int(kwargs.get("init_num", 5))
        self.latent_volatility_base = self._validate_float_range(
            kwargs.get("latent_volatility_base", 0.0),
            "latent_volatility_base",
            0.0,
            1.0,
        )
        self.latent_volatility_error_gain = self._validate_float_range(
            kwargs.get("latent_volatility_error_gain", 0.0),
            "latent_volatility_error_gain",
            0.0,
            1.0,
        )
        self.latent_volatility_low_accuracy_gain = self._validate_float_range(
            kwargs.get("latent_volatility_low_accuracy_gain", 0.0),
            "latent_volatility_low_accuracy_gain",
            0.0,
            1.0,
        )
        self.latent_volatility_threshold = self._validate_float_range(
            kwargs.get("latent_volatility_threshold", 0.70),
            "latent_volatility_threshold",
            1e-9,
            1.0,
        )
        self.latent_volatility_window = self._validate_count(
            kwargs.get("latent_volatility_window", 6),
            context="latent_volatility_window",
        )
        if self.latent_volatility_window <= 0:
            raise ValueError("latent_volatility_window must be positive.")
        self.latent_volatility_decay = self._validate_float_range(
            kwargs.get("latent_volatility_decay", 0.0),
            "latent_volatility_decay",
            0.0,
            1.0,
        )
        self.latent_volatility_max = self._validate_float_range(
            kwargs.get("latent_volatility_max", 1.0),
            "latent_volatility_max",
            0.0,
            1.0,
        )
        if self.latent_volatility_max < self.latent_volatility_base:
            raise ValueError(
                "latent_volatility_max must be >= latent_volatility_base, "
                f"got {self.latent_volatility_max!r} < {self.latent_volatility_base!r}."
            )
        self.latent_volatility_feedback_mode = str(
            kwargs.get("latent_volatility_feedback_mode", "exact")
        )
        if self.latent_volatility_feedback_mode not in self.VALID_FEEDBACK_MODES:
            raise ValueError(
                "latent_volatility_feedback_mode must be one of "
                f"{self.VALID_FEEDBACK_MODES}, got {self.latent_volatility_feedback_mode!r}."
            )
        self.latent_volatility_enabled = (
            self.latent_volatility_max > 0.0
            and (
                self.latent_volatility_base > 0.0
                or self.latent_volatility_error_gain > 0.0
                or self.latent_volatility_low_accuracy_gain > 0.0
            )
        )
        self.latent_volatility_state = 0.0
        self.latent_volatility_log: List[Dict[str, Any]] = []
        self.prior_reset_base = self._validate_float_range(
            kwargs.get("prior_reset_base", 0.0),
            "prior_reset_base",
            0.0,
            1.0,
        )
        self.prior_reset_post_error = self._validate_float_range(
            kwargs.get("prior_reset_post_error", 0.0),
            "prior_reset_post_error",
            0.0,
            1.0,
        )
        self.prior_reset_low_accuracy = self._validate_float_range(
            kwargs.get("prior_reset_low_accuracy", 0.0),
            "prior_reset_low_accuracy",
            0.0,
            1.0,
        )
        self.prior_reset_threshold = self._validate_float_range(
            kwargs.get("prior_reset_threshold", 0.70),
            "prior_reset_threshold",
            1e-9,
            1.0,
        )
        self.prior_reset_window = self._validate_count(
            kwargs.get("prior_reset_window", 8),
            context="prior_reset_window",
        )
        if self.prior_reset_window <= 0:
            raise ValueError("prior_reset_window must be positive.")
        self.prior_reset_decay = self._validate_float_range(
            kwargs.get("prior_reset_decay", 0.0),
            "prior_reset_decay",
            0.0,
            1.0,
        )
        self.prior_reset_max = self._validate_float_range(
            kwargs.get("prior_reset_max", 0.50),
            "prior_reset_max",
            0.0,
            1.0,
        )
        if self.prior_reset_max < self.prior_reset_base:
            raise ValueError(
                "prior_reset_max must be >= prior_reset_base, "
                f"got {self.prior_reset_max!r} < {self.prior_reset_base!r}."
            )
        self.prior_reset_target = str(kwargs.get("prior_reset_target", "uniform_active"))
        if self.prior_reset_target not in self.VALID_PRIOR_RESET_TARGETS:
            raise ValueError(
                "prior_reset_target must be one of "
                f"{self.VALID_PRIOR_RESET_TARGETS}, got {self.prior_reset_target!r}."
            )
        self.prior_reset_source = str(kwargs.get("prior_reset_source", "feedback"))
        if self.prior_reset_source not in self.VALID_PRIOR_RESET_SOURCES:
            raise ValueError(
                "prior_reset_source must be one of "
                f"{self.VALID_PRIOR_RESET_SOURCES}, got {self.prior_reset_source!r}."
            )
        self.prior_reset_volatility_gain = self._validate_float_range(
            kwargs.get("prior_reset_volatility_gain", 0.0),
            "prior_reset_volatility_gain",
            0.0,
            1.0,
        )
        self.prior_reset_enabled = (
            self.prior_reset_max > 0.0
            and (
                (
                    self.prior_reset_source == "latent_volatility"
                    and (
                        self.prior_reset_base > 0.0
                        or self.prior_reset_volatility_gain > 0.0
                    )
                )
                or (
                    self.prior_reset_source == "feedback"
                    and (
                        self.prior_reset_base > 0.0
                        or self.prior_reset_post_error > 0.0
                        or self.prior_reset_low_accuracy > 0.0
                    )
                )
            )
        )
        self._prior_reset_state = 0.0
        self.prior_reset_log: List[Dict[str, Any]] = []
        self.post_to_prior_config = self._validate_post_to_prior_config(
            kwargs.get("post_to_prior", {"method": "similarity_novelty"})
        )
        
        self.debug = kwargs.get("hypothesis_debug", False)
        # Track how many hypotheses each strategy selects per transition step (for plotting)
        self.strategy_counts_log: List[Dict[str, Any]] = []

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
            self.amount_evaluators[f"acc_{amount}"] = self._amount_accuracy_static_gen(amount)
            self.amount_evaluators[f"accuracy_static_{amount}"] = self._amount_accuracy_static_gen(amount)
            self.amount_evaluators[f"opp_acc_{amount}"] = self._amount_opposite_accuracy_static_gen(amount)
            self.amount_evaluators[f"opp_accuracy_static_{amount}"] = self._amount_opposite_accuracy_static_gen(amount)
            self.amount_evaluators[f"accuracy_delta_{amount}"] = self._amount_accuracy_delta_gen(amount)
            self.amount_evaluators[f"opp_accuracy_delta_{amount}"] = self._amount_opposite_accuracy_delta_gen(amount)
            self.amount_evaluators[f"latent_volatility_{amount}"] = self._amount_latent_volatility_gen(amount)
            self.amount_evaluators[f"opp_latent_volatility_{amount}"] = self._amount_opposite_latent_volatility_gen(amount)
            self.amount_evaluators[f"post_error_explore_{amount}"] = self._amount_post_error_explore_gen(amount)
        self.method_selectors = {
            "top_posterior": self._select_top_posterior,
            "random_posterior": self._select_random_posterior,
            "random": self._select_random,
            "ksimilar_centers": self._cluster_strategy_ksimilar_centers,
            "epsilon_posterior": self._select_epsilon_posterior,
            "temperature_posterior": self._select_temperature_posterior,
            "low_posterior": self._select_low_posterior,
        }
        self.strategies = self._validate_strategies(self.strategies)
        if "history_maxlen" in kwargs:
            raise ValueError("history_maxlen is no longer a supported config key; set window on history-based strategies.")
        required_history = self._required_history_length(self.strategies)
        if self.latent_volatility_enabled or self.prior_reset_enabled:
            required_history = max(required_history, self.prior_reset_window, 1)
        if self.latent_volatility_enabled:
            required_history = max(required_history, self.latent_volatility_window, 1)
        required_history = max(required_history, self._post_to_prior_required_history())
        self.uses_feedback_history = required_history > 0
        self.feedback_history = deque(maxlen=max(required_history, 1))
        self._validate_history_feedback_modes()
        
        self.active: np.ndarray | None = None
        self.old_active: np.ndarray | None = None
        self._init_mask()

    def _validate_strategies(self, strategies: Any) -> List[Dict[str, Any]]:
        if not isinstance(strategies, list) or not strategies:
            raise ValueError("strategies must be a non-empty list of strategy dictionaries.")

        validated: List[Dict[str, Any]] = []
        for idx, raw in enumerate(strategies):
            if not isinstance(raw, dict):
                raise ValueError(f"Strategy #{idx} must be a dict, got {type(raw).__name__}.")
            missing = [key for key in ("amount", "method", "pool") if key not in raw]
            if missing:
                raise ValueError(
                    f"Strategy #{idx} is missing required key(s): {', '.join(missing)}. "
                    "Each strategy must set amount, method, and pool."
                )

            strat = dict(raw)
            method = str(strat["method"])
            if method not in self.method_selectors:
                raise ValueError(
                    f"Strategy #{idx} has unsupported method '{method}'. "
                    f"Supported methods: {', '.join(self.VALID_METHODS)}."
                )
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

        if self._is_history_amount(amount):
            self._validate_history_strategy(idx, strat)

    def _validate_history_strategy(self, idx: int, strat: Dict[str, Any]) -> None:
        amount = str(strat["amount"])
        window = self._validate_count(strat.get("window", self._default_history_window(amount)), context=f"Strategy #{idx} window")
        if window <= 0:
            raise ValueError(f"Strategy #{idx} window must be positive.")
        feedback_mode = str(strat.get("feedback_mode", self._default_feedback_mode_for_amount(amount)))
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
        if amount.startswith("recent_accuracy_inverse_"):
            min_count = self._validate_count(strat.get("min_count", 1), context=f"Strategy #{idx} min_count")
            if min_count < 0:
                raise ValueError(f"Strategy #{idx} min_count must be non-negative.")
            self._validate_positive_float(strat.get("gamma", 1.0), "gamma")
        if self._is_accuracy_static_amount(amount):
            self._validate_float_range(strat.get("threshold_min", 0.2), "threshold_min", 0.0, 1.0)
            self._validate_positive_float(strat.get("scale", 10.0), "scale")
        if self._is_accuracy_delta_amount(amount):
            self._validate_float_range(strat.get("threshold", 0.0), "threshold", 0.0, 1.0)
            self._validate_positive_float(strat.get("scale", 0.5), "scale")
        if self._is_latent_volatility_amount(amount):
            min_count = self._validate_count(strat.get("min_count", 0), context=f"Strategy #{idx} min_count")
            if min_count < 0:
                raise ValueError(f"Strategy #{idx} min_count must be non-negative.")
            self._validate_float_range(strat.get("threshold", 0.0), "threshold", 0.0, 1.0)
            self._validate_positive_float(strat.get("power", 1.0), "power")
        if self._is_post_error_explore_amount(amount):
            min_count = self._validate_count(strat.get("min_count", 0), context=f"Strategy #{idx} min_count")
            if min_count < 0:
                raise ValueError(f"Strategy #{idx} min_count must be non-negative.")
            if min_count > self._amount_suffix(amount):
                raise ValueError(f"Strategy #{idx} min_count cannot exceed amount maximum.")
            self._validate_positive_float(strat.get("gamma", 1.0), "gamma")

    def _validate_post_to_prior_config(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            raw = {"method": "similarity_novelty"}
        if not isinstance(raw, dict):
            raise ValueError("post_to_prior must be a dictionary when provided.")
        config = dict(raw)
        method = str(config.get("method", "similarity_novelty"))
        if method not in self.VALID_POST_TO_PRIOR_METHODS:
            raise ValueError(
                f"Unsupported post_to_prior method '{method}'. "
                f"Supported methods: {', '.join(self.VALID_POST_TO_PRIOR_METHODS)}."
            )
        config["method"] = method
        if "label" in config and not str(config["label"]).strip():
            raise ValueError("post_to_prior label must be non-empty when provided.")

        confidence_source = str(config.get("confidence_source", "max_posterior"))
        if confidence_source not in self.VALID_POST_TO_PRIOR_CONFIDENCE_SOURCES:
            raise ValueError(
                "post_to_prior confidence_source must be one of "
                f"{self.VALID_POST_TO_PRIOR_CONFIDENCE_SOURCES}, got {confidence_source!r}."
            )
        config["confidence_source"] = confidence_source
        if confidence_source == "recent_accuracy":
            self._validate_post_to_prior_history_config(config, context="post_to_prior")

        if method == "similarity_novelty":
            self._validate_float_range(config.get("min_newcomer_scale", 0.05), "post_to_prior.min_newcomer_scale", 0.0, 1.0)
        elif method == "conservative_carryover":
            self._validate_float_range(config.get("newcomer_mass", 0.05), "post_to_prior.newcomer_mass", 0.0, 1.0)
        elif method == "error_boost_newcomers":
            self._validate_post_to_prior_history_config(config, context="post_to_prior")
            base_mass = self._validate_float_range(
                config.get("base_newcomer_mass", 0.05),
                "post_to_prior.base_newcomer_mass",
                0.0,
                1.0,
            )
            max_mass = self._validate_float_range(
                config.get("max_newcomer_mass", 0.65),
                "post_to_prior.max_newcomer_mass",
                0.0,
                1.0,
            )
            if max_mass < base_mass:
                raise ValueError("post_to_prior.max_newcomer_mass must be >= base_newcomer_mass.")
            self._validate_float_range(config.get("volatility_gain", 0.0), "post_to_prior.volatility_gain", 0.0, 1.0)
        elif method == "stochastic_reset":
            self._validate_float_range(config.get("reset_probability", 0.25), "post_to_prior.reset_probability", 0.0, 1.0)
            self._validate_float_range(config.get("newcomer_mass", 0.50), "post_to_prior.newcomer_mass", 0.0, 1.0)
            self._validate_positive_float(config.get("concentration", 1.0), "post_to_prior.concentration")
        return config

    def _validate_post_to_prior_history_config(self, config: Dict[str, Any], *, context: str) -> None:
        window = self._validate_count(config.get("window", 8), context=f"{context}.window")
        if window <= 0:
            raise ValueError(f"{context}.window must be positive.")
        feedback_mode = str(config.get("feedback_mode", "exact"))
        if feedback_mode not in self.VALID_FEEDBACK_MODES:
            raise ValueError(
                f"{context}.feedback_mode must be one of "
                f"{self.VALID_FEEDBACK_MODES}, got {feedback_mode!r}."
            )
        padding = config.get("padding", "chance")
        if isinstance(padding, str):
            if padding not in self.VALID_PADDING_MODES:
                raise ValueError(
                    f"{context}.padding must be one of "
                    f"{self.VALID_PADDING_MODES} or a numeric value, got {padding!r}."
                )
            if padding == "chance":
                self._chance_padding_value(require=True)
        else:
            self._validate_float_range(padding, f"{context}.padding", 0.0, 1.0)

    def _post_to_prior_required_history(self) -> int:
        required = 0
        if self.post_to_prior_config.get("method") == "error_boost_newcomers":
            required = max(
                required,
                self._validate_count(self.post_to_prior_config.get("window", 8), context="post_to_prior.window"),
            )
        if self.post_to_prior_config.get("confidence_source") == "recent_accuracy":
            required = max(
                required,
                self._validate_count(self.post_to_prior_config.get("window", 8), context="post_to_prior.window"),
            )
        return required

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

    def _required_history_length(self, strategies: Sequence[Dict[str, Any]]) -> int:
        required = 0
        for strat in strategies:
            amount = strat.get("amount")
            if not self._is_history_amount(amount):
                continue
            amount_str = str(amount)
            window = self._validate_count(
                strat.get("window", self._default_history_window(amount_str)),
                context="history window",
            )
            if self._is_accuracy_delta_amount(amount_str):
                window *= 2
            required = max(required, window)
        return required

    def _validate_history_feedback_modes(self) -> None:
        self._history_feedback_mode()

    @classmethod
    def _is_accuracy_static_amount(cls, amount: Any) -> bool:
        return isinstance(amount, str) and amount.startswith(
            ("acc_", "accuracy_static_", "opp_acc_", "opp_accuracy_static_")
        )

    @classmethod
    def _is_accuracy_delta_amount(cls, amount: Any) -> bool:
        return isinstance(amount, str) and amount.startswith(("accuracy_delta_", "opp_accuracy_delta_"))

    @classmethod
    def _is_latent_volatility_amount(cls, amount: Any) -> bool:
        return isinstance(amount, str) and amount.startswith(
            ("latent_volatility_", "opp_latent_volatility_")
        )

    @classmethod
    def _is_post_error_explore_amount(cls, amount: Any) -> bool:
        return isinstance(amount, str) and amount.startswith("post_error_explore_")

    @classmethod
    def _is_history_amount(cls, amount: Any) -> bool:
        return (
            isinstance(amount, str)
            and (
                amount.startswith("recent_accuracy_inverse_")
                or cls._is_accuracy_static_amount(amount)
                or cls._is_accuracy_delta_amount(amount)
                or cls._is_latent_volatility_amount(amount)
                or cls._is_post_error_explore_amount(amount)
            )
        )

    @classmethod
    def _default_history_window(cls, amount: str) -> int:
        if cls._is_post_error_explore_amount(amount):
            return 1
        if cls._is_accuracy_delta_amount(amount):
            return 8
        if cls._is_accuracy_static_amount(amount):
            return 16
        if cls._is_latent_volatility_amount(amount):
            return 6
        return 10

    @classmethod
    def _default_feedback_mode_for_amount(cls, amount: str) -> str:
        if (
            cls._is_accuracy_static_amount(amount)
            or cls._is_accuracy_delta_amount(amount)
            or cls._is_latent_volatility_amount(amount)
            or cls._is_post_error_explore_amount(amount)
        ):
            return "exact"
        return "graded"

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
    def _amount_suffix(amount: Any) -> int:
        text = str(amount)
        try:
            return int(text.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Amount evaluator '{amount}' does not end with an integer maximum.") from exc

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

    def _amount_accuracy_static_gen(self, max_amount=7):
        def _amount_accuracy_static(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            window = self._validate_count(strategy_config.get("window", 16), context="accuracy static window")
            if window <= 0:
                raise ValueError("accuracy static window must be positive.")
            threshold = float(strategy_config.get("threshold_min", 0.2))
            scale = float(strategy_config.get("scale", 10.0))
            acc = self._recent_accuracy(window, strategy_config)
            if acc <= threshold:
                return 0
            amount = int(scale * (acc - threshold)) + 1
            return int(max(0, min(amount, max_amount)))
        return _amount_accuracy_static

    def _amount_opposite_accuracy_static_gen(self, max_amount=7):
        base_func = self._amount_accuracy_static_gen(max_amount)
        def _amount_opposite_accuracy_static(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            acc_amount = base_func(posterior, max_amount=max_amount, **kwargs)
            return int(max(0, max_amount - acc_amount))
        return _amount_opposite_accuracy_static

    def _amount_accuracy_delta_gen(self, max_amount=7):
        def _amount_accuracy_delta(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            delta = self._recent_accuracy_delta(strategy_config)
            threshold = float(strategy_config.get("threshold", 0.0))
            scale = float(strategy_config.get("scale", 0.5))
            normalized = np.clip((delta - threshold) / scale, 0.0, 1.0)
            return int(max(0, min(round(max_amount * normalized), max_amount)))
        return _amount_accuracy_delta

    def _amount_opposite_accuracy_delta_gen(self, max_amount=7):
        def _amount_opposite_accuracy_delta(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            delta = self._recent_accuracy_delta(strategy_config)
            threshold = float(strategy_config.get("threshold", 0.0))
            scale = float(strategy_config.get("scale", 0.5))
            normalized = np.clip((-delta - threshold) / scale, 0.0, 1.0)
            return int(max(0, min(round(max_amount * normalized), max_amount)))
        return _amount_opposite_accuracy_delta

    def _amount_latent_volatility_gen(self, max_amount=7):
        def _amount_latent_volatility(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            min_count = self._validate_count(strategy_config.get("min_count", 0), context="latent volatility min_count")
            if min_count > max_amount:
                return max_amount
            threshold = float(strategy_config.get("threshold", 0.0))
            power = float(strategy_config.get("power", 1.0))
            denom = max(self.latent_volatility_max - threshold, 1e-12)
            normalized = np.clip((self.latent_volatility_state - threshold) / denom, 0.0, 1.0)
            amount = min_count + round((normalized ** power) * (max_amount - min_count))
            return int(max(min_count, min(amount, max_amount)))
        return _amount_latent_volatility

    def _amount_opposite_latent_volatility_gen(self, max_amount=7):
        base_func = self._amount_latent_volatility_gen(max_amount)
        def _amount_opposite_latent_volatility(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            vol_amount = base_func(posterior, max_amount=max_amount, **kwargs)
            return int(max(0, max_amount - vol_amount))
        return _amount_opposite_latent_volatility

    def _amount_post_error_explore_gen(self, max_amount=7):
        def _amount_post_error_explore(posterior: np.ndarray, max_amount=max_amount, **kwargs) -> int:
            strategy_config = kwargs.get("strategy_config", {}) or {}
            min_count = self._validate_count(strategy_config.get("min_count", 0), context="post_error_explore min_count")
            if min_count > max_amount:
                return max_amount
            gamma = float(strategy_config.get("gamma", 1.0))
            if not np.isfinite(gamma) or gamma <= 0.0:
                raise ValueError(f"post_error_explore gamma must be positive, got {gamma!r}.")
            last_accuracy = self._recent_accuracy(1, strategy_config)
            error_severity = float(np.clip(1.0 - last_accuracy, 0.0, 1.0))
            amount = min_count + round((error_severity ** gamma) * (max_amount - min_count))
            return int(max(min_count, min(amount, max_amount)))
        return _amount_post_error_explore

    def _recent_accuracy(self, window: int, strategy_config: Dict[str, Any]) -> float:
        values = self._padded_history_values(window, strategy_config)
        accuracy = float(np.mean(values))
        if not np.isfinite(accuracy):
            raise ValueError("recent accuracy is non-finite.")
        return float(np.clip(accuracy, 0.0, 1.0))

    def _recent_accuracy_delta(self, strategy_config: Dict[str, Any]) -> float:
        window = self._validate_count(strategy_config.get("window", 8), context="accuracy delta window")
        if window <= 0:
            raise ValueError("accuracy delta window must be positive.")
        values = self._padded_history_values(window * 2, strategy_config)
        old_acc = float(np.mean(values[:window]))
        new_acc = float(np.mean(values[window:]))
        delta = new_acc - old_acc
        if not np.isfinite(delta):
            raise ValueError("recent accuracy delta is non-finite.")
        return float(np.clip(delta, -1.0, 1.0))

    def _padded_history_values(self, length: int, strategy_config: Dict[str, Any]) -> List[float]:
        length = self._validate_count(length, context="history length")
        if length <= 0:
            raise ValueError("history length must be positive.")
        recent = list(self.feedback_history)[-length:]
        padding = strategy_config.get("padding", "chance")
        if len(recent) >= length:
            return [float(x) for x in recent]
        missing = length - len(recent)
        pad_value = self._resolve_padding_value(padding)
        return [pad_value] * missing + [float(x) for x in recent]

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
            str(strat.get("feedback_mode", self._default_feedback_mode_for_amount(str(strat.get("amount")))))
            for strat in self.strategies
            if self._is_history_amount(strat.get("amount"))
        }
        if self.latent_volatility_enabled or self.prior_reset_enabled:
            modes.add(self.latent_volatility_feedback_mode)
        if (
            self.post_to_prior_config.get("method") == "error_boost_newcomers"
            or self.post_to_prior_config.get("confidence_source") == "recent_accuracy"
        ):
            modes.add(str(self.post_to_prior_config.get("feedback_mode", "exact")))
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
        self._update_latent_volatility_state()
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
        step_counts: Dict[str, Any] = {
            "strategies": [],
            "latent_volatility_state": float(self.latent_volatility_state),
        }
        if self.latent_volatility_log:
            latest_volatility = self.latent_volatility_log[-1]
            step_counts["latent_volatility_recent_accuracy"] = latest_volatility.get("recent_accuracy")
            step_counts["latent_volatility_error_severity"] = latest_volatility.get("error_severity")
        
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

    def _select_low_posterior(self, amount: int, candidates: Sequence[int] | np.ndarray, posterior: np.ndarray, strategy_config: Dict | None = None, **kwargs) -> List[int]:
        strategy_config = strategy_config or {}
        if amount <= 0:
            return []
        cand_indices = np.asarray(candidates, dtype=int)
        if cand_indices.size == 0:
            return []
        actual_amount = min(amount, len(cand_indices))
        raw = np.asarray(posterior[cand_indices], dtype=float)
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError("low_posterior candidate posterior contains invalid values.")
        if cand_indices.size <= actual_amount:
            return cand_indices.tolist()
        low_args = np.argsort(raw)[:actual_amount]
        return cand_indices[low_args].tolist()

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

    def _update_latent_volatility_state(self) -> None:
        if not self.latent_volatility_enabled:
            self.latent_volatility_state = 0.0
            self.latent_volatility_log.append({
                "state": 0.0,
                "base": 0.0,
                "error": 0.0,
                "low_accuracy": 0.0,
                "recent_accuracy": 1.0,
                "error_severity": 0.0,
            })
            return

        previous_feedback = 1.0
        if self.feedback_history:
            previous_feedback = float(self.feedback_history[-1])
        error_severity = float(np.clip(1.0 - previous_feedback, 0.0, 1.0))
        recent_accuracy = self._recent_accuracy(
            self.latent_volatility_window,
            {"padding": "chance"},
        )
        low_accuracy_scale = max(
            0.0,
            self.latent_volatility_threshold - recent_accuracy,
        ) / max(self.latent_volatility_threshold, 1e-12)
        error_component = self.latent_volatility_error_gain * error_severity
        low_accuracy_component = self.latent_volatility_low_accuracy_gain * low_accuracy_scale
        state = (
            self.latent_volatility_decay * self.latent_volatility_state
            + self.latent_volatility_base
            + error_component
            + low_accuracy_component
        )
        self.latent_volatility_state = float(np.clip(state, 0.0, self.latent_volatility_max))
        self.latent_volatility_log.append({
            "state": float(self.latent_volatility_state),
            "base": float(self.latent_volatility_base),
            "error": float(error_component),
            "low_accuracy": float(low_accuracy_component),
            "recent_accuracy": float(recent_accuracy),
            "error_severity": float(error_severity),
        })

    def _compute_prior_reset_strength(self) -> Tuple[float, Dict[str, float]]:
        if not self.prior_reset_enabled:
            return 0.0, {
                "base": 0.0,
                "post_error_state": 0.0,
                "low_accuracy": 0.0,
                "recent_accuracy": 1.0,
                "error_severity": 0.0,
                "latent_volatility": float(self.latent_volatility_state),
            }

        if self.prior_reset_source == "latent_volatility":
            reset_strength = self.prior_reset_base + (
                self.prior_reset_volatility_gain * self.latent_volatility_state
            )
            reset_strength = float(np.clip(reset_strength, 0.0, self.prior_reset_max))
            recent_accuracy = self._recent_accuracy(
                self.latent_volatility_window,
                {"padding": "chance"},
            )
            previous_feedback = float(self.feedback_history[-1]) if self.feedback_history else 1.0
            return reset_strength, {
                "base": float(self.prior_reset_base),
                "post_error_state": 0.0,
                "low_accuracy": 0.0,
                "recent_accuracy": float(recent_accuracy),
                "error_severity": float(np.clip(1.0 - previous_feedback, 0.0, 1.0)),
                "latent_volatility": float(self.latent_volatility_state),
            }

        previous_feedback = 1.0
        if self.feedback_history:
            previous_feedback = float(self.feedback_history[-1])
        error_severity = float(np.clip(1.0 - previous_feedback, 0.0, 1.0))
        self._prior_reset_state = (
            self.prior_reset_decay * self._prior_reset_state
            + self.prior_reset_post_error * error_severity
        )

        recent_accuracy = self._recent_accuracy(
            self.prior_reset_window,
            {"padding": "chance"},
        )
        low_accuracy_scale = max(0.0, self.prior_reset_threshold - recent_accuracy) / max(
            self.prior_reset_threshold,
            1e-12,
        )
        low_accuracy_reset = self.prior_reset_low_accuracy * low_accuracy_scale
        reset_strength = self.prior_reset_base + self._prior_reset_state + low_accuracy_reset
        reset_strength = float(np.clip(reset_strength, 0.0, self.prior_reset_max))

        return reset_strength, {
            "base": float(self.prior_reset_base),
            "post_error_state": float(self._prior_reset_state),
            "low_accuracy": float(low_accuracy_reset),
            "recent_accuracy": float(recent_accuracy),
            "error_severity": float(error_severity),
            "latent_volatility": float(self.latent_volatility_state),
        }

    def _prior_reset_distribution(
        self,
        active_indices: np.ndarray,
        newcomer_indices: np.ndarray,
    ) -> np.ndarray:
        target = np.zeros(self.total_hypo, dtype=float)
        active_indices = np.asarray(active_indices, dtype=int)
        newcomer_indices = np.asarray(newcomer_indices, dtype=int)

        def fill_uniform(indices: np.ndarray) -> np.ndarray:
            out = np.zeros(self.total_hypo, dtype=float)
            if indices.size:
                out[indices] = 1.0 / float(indices.size)
            return out

        if self.prior_reset_target == "newcomer_boost":
            target = fill_uniform(newcomer_indices if newcomer_indices.size else active_indices)
        elif self.prior_reset_target == "sampled_active":
            if active_indices.size:
                target[int(self.rng.choice(active_indices))] = 1.0
        elif self.prior_reset_target == "sampled_newcomer":
            pool = newcomer_indices if newcomer_indices.size else active_indices
            if pool.size:
                target[int(self.rng.choice(pool))] = 1.0
        else:
            target = fill_uniform(active_indices)

        total = float(target.sum())
        if total <= 0.0 and active_indices.size:
            target[active_indices] = 1.0 / float(active_indices.size)
        elif total > 0.0:
            target /= total
        return target

    def _apply_prior_reset(
        self,
        prior: np.ndarray,
        active_indices: np.ndarray,
        newcomer_indices: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        reset_strength, components = self._compute_prior_reset_strength()
        if reset_strength <= 0.0:
            log_item = {
                "strength": 0.0,
                "target": self.prior_reset_target,
                "source": self.prior_reset_source,
                **components,
            }
            self.prior_reset_log.append(log_item)
            if self.strategy_counts_log:
                self.strategy_counts_log[-1]["prior_reset_strength"] = 0.0
                self.strategy_counts_log[-1]["prior_reset_target"] = self.prior_reset_target
                self.strategy_counts_log[-1]["prior_reset_source"] = self.prior_reset_source
            return prior, 0.0

        target = self._prior_reset_distribution(active_indices, newcomer_indices)
        mixed = (1.0 - reset_strength) * prior + reset_strength * target
        total = float(mixed.sum())
        if total > 0.0:
            mixed /= total

        log_item = {
            "strength": float(reset_strength),
            "target": self.prior_reset_target,
            "source": self.prior_reset_source,
            **components,
        }
        self.prior_reset_log.append(log_item)
        if self.strategy_counts_log:
            self.strategy_counts_log[-1]["prior_reset_strength"] = float(reset_strength)
            self.strategy_counts_log[-1]["prior_reset_target"] = self.prior_reset_target
            self.strategy_counts_log[-1]["prior_reset_source"] = self.prior_reset_source
            self.strategy_counts_log[-1]["prior_reset_recent_accuracy"] = components["recent_accuracy"]
            self.strategy_counts_log[-1]["prior_reset_error_severity"] = components["error_severity"]
            self.strategy_counts_log[-1]["prior_reset_latent_volatility"] = components["latent_volatility"]
        return mixed, float(reset_strength)
    
    def _post_to_prior_confidence(
        self,
        current_posterior: np.ndarray,
        config: Dict[str, Any],
    ) -> float:
        source = str(config.get("confidence_source", "max_posterior"))
        if source == "max_posterior":
            confidence = float(np.max(current_posterior)) if current_posterior.size else 0.0
        elif source == "entropy":
            p_entropy = entropy(current_posterior)
            max_entropy = np.log(len(current_posterior)) if len(current_posterior) > 1 else 1.0
            confidence = 1.0 - float(np.clip(p_entropy / max_entropy, 0.0, 1.0))
        elif source == "recent_accuracy":
            window = self._validate_count(config.get("window", 8), context="post_to_prior confidence window")
            confidence = self._recent_accuracy(window, config)
        elif source == "latent_volatility":
            denom = max(self.latent_volatility_max, 1e-12)
            confidence = 1.0 - float(np.clip(self.latent_volatility_state / denom, 0.0, 1.0))
        else:
            raise ValueError(f"Unsupported post_to_prior confidence_source '{source}'.")
        if not np.isfinite(confidence):
            raise ValueError("post_to_prior confidence is non-finite.")
        return float(np.clip(confidence, 0.0, 1.0))

    def _normalize_weights_or_uniform(self, values: np.ndarray, size: int) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if size <= 0:
            return np.empty(0, dtype=float)
        if values.size != size or not np.all(np.isfinite(values)) or np.any(values < 0):
            return np.full(size, 1.0 / float(size), dtype=float)
        total = float(values.sum())
        if total <= 0.0:
            return np.full(size, 1.0 / float(size), dtype=float)
        return values / total

    def _allocate_prior_between_survivors_and_newcomers(
        self,
        survivor_indices: np.ndarray,
        newcomer_indices: np.ndarray,
        survivor_values: np.ndarray,
        newcomer_values: np.ndarray,
        newcomer_mass: float,
    ) -> np.ndarray:
        new_prior = np.zeros(self.total_hypo, dtype=float)
        if len(survivor_indices) == 0 and len(newcomer_indices) == 0:
            return new_prior
        if len(survivor_indices) == 0:
            new_prior[newcomer_indices] = self._normalize_weights_or_uniform(newcomer_values, len(newcomer_indices))
            return new_prior
        if len(newcomer_indices) == 0:
            new_prior[survivor_indices] = self._normalize_weights_or_uniform(survivor_values, len(survivor_indices))
            return new_prior

        newcomer_mass = float(np.clip(newcomer_mass, 0.0, 1.0))
        survivor_mass = 1.0 - newcomer_mass
        new_prior[survivor_indices] = (
            survivor_mass * self._normalize_weights_or_uniform(survivor_values, len(survivor_indices))
        )
        new_prior[newcomer_indices] = (
            newcomer_mass * self._normalize_weights_or_uniform(newcomer_values, len(newcomer_indices))
        )
        return new_prior

    def _similarity_novelty_newcomer_scores(
        self,
        current_posterior: np.ndarray,
        old_indices: np.ndarray,
        newcomer_indices: np.ndarray,
        confidence: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if len(newcomer_indices) == 0:
            return np.empty(0, dtype=float), {"similarity_available": False, "fallback": False}
        partition = getattr(self.engine, "partition", None)
        similarity_matrix = getattr(partition, "similarity_matrix", None)
        if similarity_matrix is None or len(old_indices) == 0:
            return (
                np.ones(len(newcomer_indices), dtype=float),
                {"similarity_available": False, "fallback": True},
            )
        similarity_matrix = np.asarray(similarity_matrix, dtype=float)
        if (
            similarity_matrix.ndim != 2
            or similarity_matrix.shape[0] < self.total_hypo
            or similarity_matrix.shape[1] < self.total_hypo
        ):
            raise ValueError(
                "partition.similarity_matrix must cover all hypotheses for post_to_prior similarity_novelty."
            )
        sim_sub = similarity_matrix[np.ix_(newcomer_indices, old_indices)]
        if not np.all(np.isfinite(sim_sub)):
            raise ValueError("post_to_prior similarity matrix contains non-finite values.")

        old_posterior_values = current_posterior[old_indices].copy()
        old_total = float(old_posterior_values.sum())
        if old_total > 0.0:
            old_posterior_values /= old_total
        p_sim = sim_sub @ old_posterior_values
        max_sim_to_old = np.max(sim_sub, axis=1)
        p_nov = np.clip(1.0 - max_sim_to_old, 0.0, 1.0)
        raw_score = confidence * p_sim + (1.0 - confidence) * p_nov
        return raw_score, {
            "similarity_available": True,
            "fallback": False,
            "mean_similarity_score": float(np.mean(p_sim)) if p_sim.size else float("nan"),
            "mean_novelty_score": float(np.mean(p_nov)) if p_nov.size else float("nan"),
        }

    def _post_to_prior_similarity_novelty(
        self,
        current_posterior: np.ndarray,
        old_indices: np.ndarray,
        active_indices: np.ndarray,
        survivor_indices: np.ndarray,
        newcomer_indices: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        confidence = self._post_to_prior_confidence(current_posterior, config)
        min_scale = float(config.get("min_newcomer_scale", 0.05))
        new_prior = np.zeros(self.total_hypo, dtype=float)
        new_prior[survivor_indices] = current_posterior[survivor_indices]
        raw_score, score_log = self._similarity_novelty_newcomer_scores(
            current_posterior,
            old_indices,
            newcomer_indices,
            confidence,
        )
        scale_factor = max(1.0 - confidence, min_scale)
        if len(newcomer_indices) > 0:
            if score_log.get("similarity_available", False):
                new_prior[newcomer_indices] = scale_factor * raw_score
            else:
                new_prior[newcomer_indices] = (scale_factor / float(len(active_indices)))
        total_mass = float(new_prior.sum())
        if total_mass > 0.0:
            new_prior /= total_mass
        elif len(active_indices) > 0:
            new_prior[active_indices] = 1.0 / float(len(active_indices))
        log = {
            "method": str(config.get("method", "similarity_novelty")),
            "confidence": float(confidence),
            "newcomer_scale": float(scale_factor),
            **score_log,
        }
        return new_prior, log

    def _post_to_prior_conservative_carryover(
        self,
        current_posterior: np.ndarray,
        survivor_indices: np.ndarray,
        newcomer_indices: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        newcomer_mass = float(config.get("newcomer_mass", 0.05)) if len(newcomer_indices) > 0 else 0.0
        new_prior = self._allocate_prior_between_survivors_and_newcomers(
            survivor_indices,
            newcomer_indices,
            current_posterior[survivor_indices],
            np.ones(len(newcomer_indices), dtype=float),
            newcomer_mass,
        )
        return new_prior, {
            "method": "conservative_carryover",
            "configured_newcomer_mass": float(newcomer_mass),
            "similarity_available": False,
            "fallback": False,
        }

    def _post_to_prior_error_boost_newcomers(
        self,
        current_posterior: np.ndarray,
        old_indices: np.ndarray,
        survivor_indices: np.ndarray,
        newcomer_indices: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        confidence = self._post_to_prior_confidence(current_posterior, config)
        raw_score, score_log = self._similarity_novelty_newcomer_scores(
            current_posterior,
            old_indices,
            newcomer_indices,
            confidence,
        )
        window = self._validate_count(config.get("window", 8), context="post_to_prior.window")
        recent_accuracy = self._recent_accuracy(window, config)
        error_severity = float(np.clip(1.0 - recent_accuracy, 0.0, 1.0))
        volatility_gain = float(config.get("volatility_gain", 0.0))
        volatility_component = volatility_gain * float(np.clip(self.latent_volatility_state, 0.0, 1.0))
        boost = float(np.clip(error_severity + volatility_component, 0.0, 1.0))
        base_mass = float(config.get("base_newcomer_mass", 0.05))
        max_mass = float(config.get("max_newcomer_mass", 0.65))
        newcomer_mass = base_mass + (max_mass - base_mass) * boost
        if len(newcomer_indices) == 0:
            newcomer_mass = 0.0
        new_prior = self._allocate_prior_between_survivors_and_newcomers(
            survivor_indices,
            newcomer_indices,
            current_posterior[survivor_indices],
            raw_score,
            newcomer_mass,
        )
        return new_prior, {
            "method": "error_boost_newcomers",
            "confidence": float(confidence),
            "recent_accuracy": float(recent_accuracy),
            "error_severity": float(error_severity),
            "latent_volatility": float(self.latent_volatility_state),
            "newcomer_mass": float(newcomer_mass),
            **score_log,
        }

    def _post_to_prior_stochastic_reset(
        self,
        current_posterior: np.ndarray,
        old_indices: np.ndarray,
        active_indices: np.ndarray,
        survivor_indices: np.ndarray,
        newcomer_indices: np.ndarray,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        reset_probability = float(config.get("reset_probability", 0.25))
        reset_applied = bool(self.rng.random() < reset_probability)
        if not reset_applied:
            delegated_config = dict(config)
            delegated_config["method"] = "similarity_novelty"
            new_prior, log = self._post_to_prior_similarity_novelty(
                current_posterior,
                old_indices,
                active_indices,
                survivor_indices,
                newcomer_indices,
                delegated_config,
            )
            log.update({
                "method": "stochastic_reset",
                "reset_applied": False,
                "reset_probability": float(reset_probability),
            })
            return new_prior, log

        concentration = float(config.get("concentration", 1.0))
        random_active_weights = self.rng.gamma(
            shape=concentration,
            scale=1.0,
            size=len(active_indices),
        )
        if not np.all(np.isfinite(random_active_weights)) or float(random_active_weights.sum()) <= 0.0:
            random_active_weights = np.ones(len(active_indices), dtype=float)
        random_active_weights = random_active_weights / float(random_active_weights.sum())
        new_prior = np.zeros(self.total_hypo, dtype=float)
        new_prior[active_indices] = random_active_weights
        if len(survivor_indices) > 0 and len(newcomer_indices) > 0:
            newcomer_mass = float(config.get("newcomer_mass", 0.50))
            survivor_weights = new_prior[survivor_indices]
            newcomer_weights = new_prior[newcomer_indices]
            new_prior[survivor_indices] = (
                (1.0 - newcomer_mass)
                * self._normalize_weights_or_uniform(survivor_weights, len(survivor_indices))
            )
            new_prior[newcomer_indices] = (
                newcomer_mass
                * self._normalize_weights_or_uniform(newcomer_weights, len(newcomer_indices))
            )
        return new_prior, {
            "method": "stochastic_reset",
            "reset_applied": True,
            "reset_probability": float(reset_probability),
            "newcomer_mass": (
                float(np.sum(new_prior[newcomer_indices])) if len(newcomer_indices) > 0 else 0.0
            ),
            "similarity_available": False,
            "fallback": False,
        }

    def _posterior_to_prior_transition(self):
        """
        Update engine.prior based on the transition from old_active to active hypotheses.
        """
        if self.old_active is None or self.active is None:
            return

        active_indices = np.asarray(self.active, dtype=int)
        old_indices = np.asarray(self.old_active, dtype=int)
        if len(active_indices) == 0:
            return
        if not hasattr(self, "strategy_counts_log"):
            self.strategy_counts_log = []
        if not self.strategy_counts_log:
            self.strategy_counts_log.append({"strategies": [], "active_total": int(len(active_indices))})

        current_posterior = None
        if hasattr(self.engine, "posterior") and self.engine.posterior is not None:
            current_posterior = np.asarray(self.engine.posterior, dtype=float).copy()

        if current_posterior is None:
            new_prior = np.zeros(self.total_hypo, dtype=float)
            new_prior[active_indices] = 1.0 / float(len(active_indices))
            self.engine.prior = new_prior
            if self.strategy_counts_log:
                self.strategy_counts_log[-1]["post_to_prior"] = {
                    "method": self.post_to_prior_config.get("method", "similarity_novelty"),
                    "fallback": True,
                    "fallback_reason": "missing_posterior",
                    "survivor_mass": 0.0,
                    "newcomer_mass": 1.0,
                }
            return

        if current_posterior.shape[0] != self.total_hypo:
            raise ValueError(
                "posterior length does not match hypothesis space in post_to_prior transition: "
                f"{current_posterior.shape[0]} vs {self.total_hypo}."
            )
        if not np.all(np.isfinite(current_posterior)) or np.any(current_posterior < 0):
            raise ValueError("posterior contains invalid values in post_to_prior transition.")

        is_survivor = np.isin(active_indices, old_indices)
        survivor_indices = active_indices[is_survivor]
        newcomer_indices = active_indices[~is_survivor]

        config = self.post_to_prior_config
        method = str(config.get("method", "similarity_novelty"))
        if method == "similarity_novelty":
            new_prior, log = self._post_to_prior_similarity_novelty(
                current_posterior,
                old_indices,
                active_indices,
                survivor_indices,
                newcomer_indices,
                config,
            )
        elif method == "conservative_carryover":
            new_prior, log = self._post_to_prior_conservative_carryover(
                current_posterior,
                survivor_indices,
                newcomer_indices,
                config,
            )
        elif method == "error_boost_newcomers":
            new_prior, log = self._post_to_prior_error_boost_newcomers(
                current_posterior,
                old_indices,
                survivor_indices,
                newcomer_indices,
                config,
            )
        elif method == "stochastic_reset":
            new_prior, log = self._post_to_prior_stochastic_reset(
                current_posterior,
                old_indices,
                active_indices,
                survivor_indices,
                newcomer_indices,
                config,
            )
        else:
            raise ValueError(f"Unsupported post_to_prior method '{method}'.")

        total_mass = float(new_prior.sum())
        if total_mass > 0.0:
            new_prior /= total_mass
        else:
            new_prior[active_indices] = 1.0 / float(len(active_indices))
            log["fallback"] = True
            log["fallback_reason"] = "zero_prior_mass"

        pre_reset_survivor_mass = float(np.sum(new_prior[survivor_indices])) if len(survivor_indices) else 0.0
        pre_reset_newcomer_mass = float(np.sum(new_prior[newcomer_indices])) if len(newcomer_indices) else 0.0
        new_prior, reset_strength = self._apply_prior_reset(new_prior, active_indices, newcomer_indices)

        log.update({
            "survivor_count": int(len(survivor_indices)),
            "newcomer_count": int(len(newcomer_indices)),
            "survivor_mass": float(pre_reset_survivor_mass),
            "newcomer_mass": float(pre_reset_newcomer_mass),
            "post_reset_survivor_mass": float(np.sum(new_prior[survivor_indices])) if len(survivor_indices) else 0.0,
            "post_reset_newcomer_mass": float(np.sum(new_prior[newcomer_indices])) if len(newcomer_indices) else 0.0,
            "prior_reset_strength": float(reset_strength),
        })
        if self.strategy_counts_log:
            self.strategy_counts_log[-1]["post_to_prior"] = log

        self.engine.prior = new_prior

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
