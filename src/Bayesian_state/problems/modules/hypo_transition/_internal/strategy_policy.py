"""Private strategy-policy implementation shared by public H modes.

This is an internal strategy library, not a complete H module.  Public modes
provide the lifecycle in :mod:`static` and :mod:`dynamic_discrete`.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence, List, Dict, Set, Tuple, Callable, Any, Mapping
from scipy.spatial.distance import cdist
from .....utils.basic_stat import entropy, softmax
from .....utils.console_styles import print

from ...base_module import BaseModule
import numpy as np


class StrategyPolicyRuntime(BaseModule):
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
    VALID_LATENT_VOLATILITY_SIGNALS = ("error", "confidence_weighted_error")
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
    VALID_STATE_POLICY_METHODS = (
        "conservative",
        "stable",
        "aggressive",
        "stubborn",
    )
    VALID_NEWCOMER_SCORES = ("random", "recent_choice", "recent_error_choice")
    VALID_INIT_POOLS = ("all", "base", "label_permuted")
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
        
        # Config: strategies is a list of dicts. A state_controller may
        # provide per-trial states instead of a single static strategy list.
        strategies_input = kwargs.get("strategies", None)
        self.state_controller_raw = kwargs.get("state_controller", None)
        if strategies_input is None and self.state_controller_raw is None:
            raise ValueError("Strategies configuration is required. Provide a non-empty list of strategy dicts.")
        if isinstance(strategies_input, str):
            raise ValueError(
                "String strategy shortcuts are no longer supported. "
                "Provide an explicit list of strategy dicts with amount, method, and pool."
            )
        self.strategies = strategies_input if strategies_input is not None else []

        # Hard cap for total active hypotheses after each transition.
        self.max_active_hypotheses: int | None = kwargs.get("max_active_hypotheses", None)
        if self.max_active_hypotheses is not None:
            self.max_active_hypotheses = max(1, int(self.max_active_hypotheses))
            
        # Global parameters for strategies
        self.strategy_params = {
            "beta": kwargs.get("beta", 5.0) # Default to 5.0 to match likelihood
        }
        self.init_num = int(kwargs.get("init_num", 5))
        raw_init_hypotheses = kwargs.get("init_hypotheses", None)
        self.init_hypotheses: np.ndarray | None = None
        if raw_init_hypotheses is not None:
            if not isinstance(raw_init_hypotheses, (list, tuple, np.ndarray)):
                raise ValueError("init_hypotheses must be a list of hypothesis indices when provided.")
            init_arr = np.asarray(raw_init_hypotheses, dtype=int).reshape(-1)
            if init_arr.size == 0:
                raise ValueError("init_hypotheses cannot be empty when provided.")
            if np.unique(init_arr).size != init_arr.size:
                raise ValueError("init_hypotheses cannot contain duplicate indices.")
            if np.any(init_arr < 0) or np.any(init_arr >= self.total_hypo):
                raise ValueError("init_hypotheses contains indices outside the hypothesis space.")
            self.init_hypotheses = init_arr
        self.init_pool = str(kwargs.get("init_pool", "all"))
        if self.init_pool not in self.VALID_INIT_POOLS:
            raise ValueError(
                "init_pool must be one of "
                f"{self.VALID_INIT_POOLS}, got {self.init_pool!r}."
            )
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
        self.latent_volatility_signal = str(
            kwargs.get("latent_volatility_signal", "error")
        )
        if self.latent_volatility_signal not in self.VALID_LATENT_VOLATILITY_SIGNALS:
            raise ValueError(
                "latent_volatility_signal must be one of "
                f"{self.VALID_LATENT_VOLATILITY_SIGNALS}, "
                f"got {self.latent_volatility_signal!r}."
            )
        self.latent_volatility_pressure_slope = self._validate_positive_float(
            kwargs.get("latent_volatility_pressure_slope", 8.0),
            "latent_volatility_pressure_slope",
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
        self._current_post_to_prior_config = self.post_to_prior_config
        self._post_to_prior_override: Dict[str, Any] | None = None
        
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
        self.strategies = self._validate_strategies(self.strategies) if self.strategies else []
        self.state_controller = self._validate_state_controller(self.state_controller_raw)
        if "history_maxlen" in kwargs:
            raise ValueError("history_maxlen is no longer a supported config key; set window on history-based strategies.")
        required_history = self._required_history_length(self._all_configured_strategies())
        required_history = max(required_history, self._state_controller_required_history())
        if self.latent_volatility_enabled or self.prior_reset_enabled:
            required_history = max(required_history, self.prior_reset_window, 1)
        if self.latent_volatility_enabled:
            required_history = max(required_history, self.latent_volatility_window, 1)
        required_history = max(
            required_history,
            max(self._post_to_prior_required_history(config) for config in self._all_post_to_prior_configs()),
        )
        self.uses_feedback_history = required_history > 0
        self.feedback_history = deque(maxlen=max(required_history, 1))
        self._validate_history_feedback_modes()
        
        self.active: np.ndarray | None = None
        self.old_active: np.ndarray | None = None
        self.previous_observation: tuple[np.ndarray, int, float] | None = None
        self.observation_history = deque(maxlen=max(required_history, 1))
        self._init_mask()

    def _all_configured_strategies(self) -> List[Dict[str, Any]]:
        strategies = list(self.strategies)
        if isinstance(getattr(self, "state_controller", None), dict):
            for state in self.state_controller.get("states", []):
                strategies.extend(state.get("strategies", []))
        return strategies

    def _all_post_to_prior_configs(self) -> List[Dict[str, Any]]:
        configs = [self.post_to_prior_config]
        if isinstance(getattr(self, "state_controller", None), dict):
            for state in self.state_controller.get("states", []):
                configs.append(state.get("post_to_prior", self.post_to_prior_config))
        return configs

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

    def _validate_state_controller(self, raw: Any) -> Dict[str, Any] | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("state_controller must be a dictionary when provided.")
        config = dict(raw)
        method = str(config.get("method", "feedback_gated_softmax"))
        if method != "feedback_gated_softmax":
            raise ValueError("state_controller.method must be 'feedback_gated_softmax'.")
        states = config.get("states")
        if not isinstance(states, list) or not states:
            raise ValueError("state_controller.states must be a non-empty list.")

        activation = config.get("activation", {}) or {}
        if not isinstance(activation, dict):
            raise ValueError("state_controller.activation must be a dictionary when provided.")
        temperature = self._validate_positive_float(
            activation.get("temperature", 1.0),
            "state_controller.activation.temperature",
        )
        weights_by_state = activation.get("weights", {}) or {}
        if not isinstance(weights_by_state, dict):
            raise ValueError("state_controller.activation.weights must be a mapping when provided.")

        features = config.get("features", {}) or {}
        if not isinstance(features, dict):
            raise ValueError("state_controller.features must be a dictionary when provided.")
        recent_window = self._validate_count(
            features.get("recent_accuracy_window", 8),
            context="state_controller.features.recent_accuracy_window",
        )
        delta_window = self._validate_count(
            features.get("accuracy_delta_window", recent_window),
            context="state_controller.features.accuracy_delta_window",
        )
        if recent_window <= 0 or delta_window <= 0:
            raise ValueError("state_controller history windows must be positive.")
        feedback_mode = str(features.get("feedback_mode", "exact"))
        if feedback_mode not in self.VALID_FEEDBACK_MODES:
            raise ValueError(
                "state_controller.features.feedback_mode must be one of "
                f"{self.VALID_FEEDBACK_MODES}, got {feedback_mode!r}."
            )
        padding = features.get("padding", "chance")
        if isinstance(padding, str):
            if padding not in self.VALID_PADDING_MODES:
                raise ValueError(
                    "state_controller.features.padding must be one of "
                    f"{self.VALID_PADDING_MODES} or a numeric value, got {padding!r}."
                )
            if padding == "chance":
                self._chance_padding_value(require=True)
        else:
            self._validate_float_range(padding, "state_controller.features.padding", 0.0, 1.0)

        state_ids: Set[str] = set()
        validated_states: List[Dict[str, Any]] = []
        for idx, raw_state in enumerate(states):
            if not isinstance(raw_state, dict):
                raise ValueError(f"state_controller.states[{idx}] must be a dictionary.")
            state = dict(raw_state)
            state_id = str(state.get("id", "")).strip()
            if not state_id:
                raise ValueError(f"state_controller.states[{idx}].id must be non-empty.")
            if state_id in state_ids:
                raise ValueError(f"Duplicate state_controller state id: {state_id!r}.")
            state_ids.add(state_id)
            state["id"] = state_id
            policy_method = state.get("policy_method")
            if policy_method is not None:
                policy_method = str(policy_method)
                if policy_method not in self.VALID_STATE_POLICY_METHODS:
                    raise ValueError(
                        f"state_controller state {state_id!r} has unsupported policy_method "
                        f"{policy_method!r}. Supported values: {', '.join(self.VALID_STATE_POLICY_METHODS)}."
                    )
                if "strategies" in state:
                    raise ValueError(
                        f"state_controller state {state_id!r} cannot define both "
                        "policy_method and strategies."
                    )
                state["policy_method"] = policy_method
                state["strategies"] = []
                self._validate_state_policy_parameters(state)
            else:
                state["strategies"] = self._validate_strategies(state.get("strategies"))
            state["post_to_prior"] = self._validate_post_to_prior_config(
                state.get("post_to_prior", self.post_to_prior_config)
            )
            state_activation = dict(weights_by_state.get(state_id, {}) or {})
            if "activation" in state:
                if not isinstance(state["activation"], dict):
                    raise ValueError(f"state_controller state {state_id!r} activation must be a dictionary.")
                state_activation.update(dict(state["activation"]))
            for name, value in state_activation.items():
                self._validate_finite_float(value, f"state_controller state {state_id}.{name}")
            state["activation"] = state_activation
            validated_states.append(state)

        return {
            "method": method,
            "states": validated_states,
            "activation": {"temperature": temperature},
            "features": {
                "recent_accuracy_window": recent_window,
                "accuracy_delta_window": delta_window,
                "feedback_mode": feedback_mode,
                "padding": padding,
                "trial_progress_scale": self._validate_positive_float(
                    features.get("trial_progress_scale", 100.0),
                    "state_controller.features.trial_progress_scale",
                ),
                "mid_phase_center": self._validate_float_range(
                    features.get("mid_phase_center", 0.35),
                    "state_controller.features.mid_phase_center",
                    0.0,
                    1.0,
                ),
                "mid_phase_width": self._validate_positive_float(
                    features.get("mid_phase_width", 0.12),
                    "state_controller.features.mid_phase_width",
                ),
            },
        }

    def _validate_state_policy_parameters(self, state: Dict[str, Any]) -> None:
        state_id = str(state.get("id", "<unknown>"))
        prefix = f"state_controller state {state_id!r}"
        active_limit = self._validate_count(state.get("active_limit", 5), context=f"{prefix} active_limit")
        if active_limit <= 0:
            raise ValueError(f"{prefix} active_limit must be positive.")
        if active_limit > self.total_hypo:
            raise ValueError(f"{prefix} active_limit cannot exceed hypothesis space size.")

        policy_method = str(state["policy_method"])
        if policy_method == "stable":
            self._validate_positive_float(state.get("retain_temperature", 1.0), f"{prefix} retain_temperature")
            explore_count = self._validate_count(state.get("explore_count", 1), context=f"{prefix} explore_count")
            if explore_count < 0:
                raise ValueError(f"{prefix} explore_count must be non-negative.")
        elif policy_method == "aggressive":
            max_newcomers = self._validate_count(state.get("max_newcomers", active_limit - 1), context=f"{prefix} max_newcomers")
            min_newcomers = self._validate_count(state.get("min_newcomers", 0), context=f"{prefix} min_newcomers")
            if min_newcomers < 0 or max_newcomers < 0 or min_newcomers > max_newcomers:
                raise ValueError(f"{prefix} newcomer bounds must satisfy 0 <= min_newcomers <= max_newcomers.")
        elif policy_method == "stubborn":
            retain_count = self._validate_count(state.get("retain_count", 2), context=f"{prefix} retain_count")
            if retain_count <= 0:
                raise ValueError(f"{prefix} retain_count must be positive.")
            self._validate_float_range(state.get("base_explore_prob", 0.02), f"{prefix} base_explore_prob", 0.0, 1.0)
            self._validate_float_range(state.get("post_correct_explore_prob", 0.20), f"{prefix} post_correct_explore_prob", 0.0, 1.0)
            self._validate_float_range(state.get("post_error_explore_prob", 0.0), f"{prefix} post_error_explore_prob", 0.0, 1.0)
            self._validate_float_range(state.get("newcomer_mass", 0.02), f"{prefix} newcomer_mass", 0.0, 1.0)
        survivor_score = str(state.get("survivor_score", "posterior"))
        if survivor_score not in ("posterior", "posterior_choice"):
            raise ValueError(f"{prefix} survivor_score must be 'posterior' or 'posterior_choice'.")
        self._validate_float_range(
            state.get("survivor_posterior_weight", 1.0),
            f"{prefix} survivor_posterior_weight",
            0.0,
            10.0,
        )
        self._validate_float_range(
            state.get("survivor_choice_weight", 0.0),
            f"{prefix} survivor_choice_weight",
            0.0,
            10.0,
        )
        self._validate_positive_float(
            state.get("survivor_choice_floor", 1e-9),
            f"{prefix} survivor_choice_floor",
        )
        if "survivor_choice_beta" in state:
            self._validate_positive_float(state["survivor_choice_beta"], f"{prefix} survivor_choice_beta")
        newcomer_score = str(state.get("newcomer_score", "random"))
        if newcomer_score not in self.VALID_NEWCOMER_SCORES:
            raise ValueError(
                f"{prefix} newcomer_score must be one of {self.VALID_NEWCOMER_SCORES}, got {newcomer_score!r}."
            )
        newcomer_choice_window = self._validate_count(
            state.get("newcomer_choice_window", 8),
            context=f"{prefix} newcomer_choice_window",
        )
        if newcomer_choice_window <= 0:
            raise ValueError(f"{prefix} newcomer_choice_window must be positive.")
        self._validate_float_range(
            state.get("newcomer_choice_weight", 1.0),
            f"{prefix} newcomer_choice_weight",
            0.0,
            10.0,
        )
        self._validate_positive_float(
            state.get("newcomer_choice_floor", 1e-9),
            f"{prefix} newcomer_choice_floor",
        )
        self._validate_positive_float(
            state.get("newcomer_choice_temperature", 1.0),
            f"{prefix} newcomer_choice_temperature",
        )
        if "newcomer_choice_beta" in state:
            self._validate_positive_float(state["newcomer_choice_beta"], f"{prefix} newcomer_choice_beta")

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

    def _post_to_prior_required_history(self, config: Dict[str, Any] | None = None) -> int:
        config = config or self.post_to_prior_config
        required = 0
        if config.get("method") == "error_boost_newcomers":
            required = max(
                required,
                self._validate_count(config.get("window", 8), context="post_to_prior.window"),
            )
        if config.get("confidence_source") == "recent_accuracy":
            required = max(
                required,
                self._validate_count(config.get("window", 8), context="post_to_prior.window"),
            )
        return required

    def _state_controller_required_history(self) -> int:
        if not isinstance(getattr(self, "state_controller", None), dict):
            return 0
        features = self.state_controller["features"]
        required = max(
            int(features.get("recent_accuracy_window", 0)),
            2 * int(features.get("accuracy_delta_window", 0)),
            1,
        )
        for state in self.state_controller.get("states", []):
            if str(state.get("newcomer_score", "random")) in ("recent_choice", "recent_error_choice"):
                required = max(required, int(state.get("newcomer_choice_window", 8)))
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

    @staticmethod
    def _validate_finite_float(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number, got boolean.")
        val = float(value)
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {value!r}.")
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

    def _record_feedback_from_observation(
        self,
        observation: tuple[np.ndarray, int, float] | None = None,
    ) -> None:
        if not self.uses_feedback_history:
            return
        if observation is None:
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
            for strat in self._all_configured_strategies()
            if self._is_history_amount(strat.get("amount"))
        }
        if self.latent_volatility_enabled or self.prior_reset_enabled:
            modes.add(self.latent_volatility_feedback_mode)
        for config in self._all_post_to_prior_configs():
            if (
                config.get("method") == "error_boost_newcomers"
                or config.get("confidence_source") == "recent_accuracy"
            ):
                modes.add(str(config.get("feedback_mode", "exact")))
        if isinstance(getattr(self, "state_controller", None), dict):
            modes.add(str(self.state_controller["features"].get("feedback_mode", "exact")))
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

    def _controller_history_config(self) -> Dict[str, Any]:
        if not isinstance(getattr(self, "state_controller", None), dict):
            return {}
        features = self.state_controller["features"]
        return {
            "padding": features.get("padding", "chance"),
            "feedback_mode": features.get("feedback_mode", "exact"),
        }

    def _controller_activation_features(self, posterior: np.ndarray) -> Dict[str, float]:
        config = self._controller_history_config()
        features_cfg = self.state_controller["features"]
        recent_window = int(features_cfg["recent_accuracy_window"])
        delta_window = int(features_cfg["accuracy_delta_window"])
        recent_accuracy = self._recent_accuracy(recent_window, config)
        values = self._padded_history_values(delta_window * 2, config)
        old_acc = float(np.mean(values[:delta_window]))
        new_acc = float(np.mean(values[delta_window:]))
        p_entropy = entropy(posterior)
        max_entropy = np.log(len(posterior)) if len(posterior) > 1 else 1.0
        entropy_norm = float(np.clip(p_entropy / max(max_entropy, 1e-12), 0.0, 1.0))
        last_feedback = float(self.feedback_history[-1]) if self.feedback_history else self._resolve_padding_value(config["padding"])
        progress_scale = float(features_cfg["trial_progress_scale"])
        trial_progress = float(np.clip(len(self.strategy_counts_log) / max(progress_scale, 1e-12), 0.0, 1.0))
        mid_center = float(features_cfg.get("mid_phase_center", 0.35))
        mid_width = float(features_cfg.get("mid_phase_width", 0.12))
        mid_phase = float(np.exp(-0.5 * ((trial_progress - mid_center) / max(mid_width, 1e-12)) ** 2))
        latent_denom = max(float(self.latent_volatility_max), 1e-12)
        latent_normalized = float(
            np.clip(self.latent_volatility_state / latent_denom, 0.0, 1.0)
        )
        latent_threshold = float(
            np.clip(self.latent_volatility_threshold / latent_denom, 0.0, 1.0)
        )
        pressure_argument = self.latent_volatility_pressure_slope * (
            latent_normalized - latent_threshold
        )
        pressure_argument = float(np.clip(pressure_argument, -60.0, 60.0))
        latent_pressure = float(1.0 / (1.0 + np.exp(-pressure_argument)))
        return {
            "bias": 1.0,
            "last_error": float(np.clip(1.0 - last_feedback, 0.0, 1.0)),
            "recent_accuracy": float(recent_accuracy),
            "recent_error": float(np.clip(1.0 - recent_accuracy, 0.0, 1.0)),
            "accuracy_delta": float(np.clip(new_acc - old_acc, -1.0, 1.0)),
            "posterior_entropy": entropy_norm,
            "posterior_confidence": float(1.0 - entropy_norm),
            "latent_volatility": latent_normalized,
            "latent_volatility_pressure": latent_pressure,
            "trial_progress": trial_progress,
            "mid_phase": mid_phase,
        }

    def _select_strategy_state(self, posterior: np.ndarray) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
        if not isinstance(getattr(self, "state_controller", None), dict):
            return None, {}
        features = self._controller_activation_features(posterior)
        states = self.state_controller["states"]
        logits = []
        for state in states:
            weights = state.get("activation", {}) or {}
            logit = 0.0
            for name, weight in weights.items():
                if name not in features:
                    raise ValueError(
                        f"state_controller state {state['id']!r} references unknown feature {name!r}."
                    )
                logit += float(weight) * float(features[name])
            logits.append(logit)
        logits_arr = np.asarray(logits, dtype=float)
        if not np.all(np.isfinite(logits_arr)):
            raise ValueError("state_controller activation logits contain non-finite values.")
        temperature = float(self.state_controller["activation"]["temperature"])
        scaled = logits_arr / temperature
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs = self._validate_probability_vector(probs, context="state_controller state")
        chosen_idx = int(self.rng.choice(len(states), p=probs))
        probabilities = {
            str(state["id"]): float(prob)
            for state, prob in zip(states, probs)
        }
        state_policy_methods = {
            str(state["id"]): str(state.get("policy_method", state["id"]))
            for state in states
        }
        policy_probabilities: Dict[str, float] = {}
        for state, prob in zip(states, probs):
            policy_method = str(state.get("policy_method", state["id"]))
            policy_probabilities[policy_method] = (
                policy_probabilities.get(policy_method, 0.0) + float(prob)
            )
        selected_policy_method = str(
            states[chosen_idx].get("policy_method", states[chosen_idx]["id"])
        )
        return states[chosen_idx], {
            "method": self.state_controller["method"],
            "features": {key: float(value) for key, value in features.items()},
            "state_logits": {
                str(state["id"]): float(logit)
                for state, logit in zip(states, logits_arr)
            },
            "state_probabilities": probabilities,
            "state_policy_methods": state_policy_methods,
            "policy_probabilities": policy_probabilities,
            "selected_state": str(states[chosen_idx]["id"]),
            "selected_policy_method": selected_policy_method,
        }

    def _run_strategy_chain(
        self,
        strategies: Sequence[Dict[str, Any]],
        posterior: np.ndarray,
        step_counts: Dict[str, Any],
        **kwargs,
    ) -> Set[int]:
        new_active_set: Set[int] = set()
        for strat in strategies:
            amount_type = strat["amount"]
            method_type = strat["method"]
            pool_type = strat["pool"]
            
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
            step_counts[f"{method_type}"] = step_counts.get(f"{method_type}", 0) + len(selected)
        return new_active_set

    def _state_active_limit(self, state: Dict[str, Any]) -> int:
        configured = int(state.get("active_limit", 5))
        if self.max_active_hypotheses is not None:
            configured = min(configured, int(self.max_active_hypotheses))
        return max(1, min(configured, self.total_hypo))

    def _posterior_weighted_sample(
        self,
        candidates: Sequence[int] | np.ndarray,
        count: int,
        posterior: np.ndarray,
        *,
        temperature: float = 1.0,
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if count <= 0 or cand.size == 0:
            return np.empty(0, dtype=int)
        actual = min(int(count), int(cand.size))
        raw = np.asarray(posterior[cand], dtype=float)
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError("state policy posterior weights contain invalid values.")
        if float(raw.sum()) <= 0.0:
            return self._sample_from_pool(cand, actual)
        weights = np.power(raw + 1e-12, 1.0 / max(float(temperature), 1e-12))
        prob = self._validate_probability_vector(weights, context="state policy")
        return self.rng.choice(cand, size=actual, replace=False, p=prob)

    def _current_beta_for_hypothesis(self, hypo: int, state: Dict[str, Any]) -> float:
        if "survivor_choice_beta" in state:
            return float(state["survivor_choice_beta"])
        beta = getattr(self.engine, "beta", 1.0)
        arr = np.asarray(beta, dtype=float).reshape(-1)
        if arr.size == self.total_hypo:
            return float(arr[int(hypo)])
        if arr.size > 0:
            return float(arr[0])
        return 1.0

    def _previous_choice_likelihood(
        self,
        candidates: Sequence[int] | np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if cand.size == 0:
            return np.empty(0, dtype=float)
        if self.previous_observation is None:
            return np.ones(cand.size, dtype=float)
        partition = getattr(self.engine, "partition", None)
        if partition is None or not hasattr(partition, "get_category_probabilities"):
            return np.ones(cand.size, dtype=float)
        stimulus, choice, _feedback = self.previous_observation
        choice_idx = int(choice) - 1
        n_cats = int(getattr(partition, "n_cats", 0))
        if choice_idx < 0 or (n_cats > 0 and choice_idx >= n_cats):
            return np.ones(cand.size, dtype=float)
        likelihood = np.ones(cand.size, dtype=float)
        for pos, hypo in enumerate(cand):
            prob = partition.get_category_probabilities(
                int(hypo),
                ([np.asarray(stimulus, dtype=float)], [int(choice)], [1.0]),
                beta=self._current_beta_for_hypothesis(int(hypo), state),
                distance_mode=getattr(self.engine, "distance_mode", "prototype"),
            )
            prob = np.asarray(prob, dtype=float)
            if prob.ndim == 1:
                prob = prob.reshape(-1, 1)
            likelihood[pos] = float(prob[choice_idx, 0])
        floor = float(state.get("survivor_choice_floor", 1e-9))
        likelihood = np.clip(likelihood, floor, 1.0)
        if not np.all(np.isfinite(likelihood)):
            raise ValueError("survivor choice likelihood contains non-finite values.")
        return likelihood

    def _recent_choice_likelihood(
        self,
        candidates: Sequence[int] | np.ndarray,
        state: Dict[str, Any],
        *,
        errors_only: bool = False,
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if cand.size == 0:
            return np.empty(0, dtype=float)
        history = list(self.observation_history)
        if not history:
            return np.ones(cand.size, dtype=float)
        window = int(state.get("newcomer_choice_window", 8))
        recent = history[-window:]
        if errors_only:
            recent = [obs for obs in recent if float(obs[2]) < 1.0]
            if not recent:
                return np.ones(cand.size, dtype=float)
        partition = getattr(self.engine, "partition", None)
        if partition is None or not hasattr(partition, "get_category_probabilities"):
            return np.ones(cand.size, dtype=float)
        n_cats = int(getattr(partition, "n_cats", 0))
        floor = float(state.get("newcomer_choice_floor", 1e-9))
        beta_override = state.get("newcomer_choice_beta", None)
        log_likelihood = np.zeros(cand.size, dtype=float)
        count = 0
        for stimulus, choice, _feedback in recent:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or (n_cats > 0 and choice_idx >= n_cats):
                continue
            count += 1
            for pos, hypo in enumerate(cand):
                beta = float(beta_override) if beta_override is not None else self._current_beta_for_hypothesis(int(hypo), state)
                prob = partition.get_category_probabilities(
                    int(hypo),
                    ([np.asarray(stimulus, dtype=float)], [int(choice)], [1.0]),
                    beta=beta,
                    distance_mode=getattr(self.engine, "distance_mode", "prototype"),
                )
                prob = np.asarray(prob, dtype=float)
                if prob.ndim == 1:
                    prob = prob.reshape(-1, 1)
                log_likelihood[pos] += np.log(float(np.clip(prob[choice_idx, 0], floor, 1.0)))
        if count == 0:
            return np.ones(cand.size, dtype=float)
        log_likelihood /= float(count)
        log_likelihood -= float(np.max(log_likelihood))
        likelihood = np.exp(log_likelihood)
        if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0):
            raise ValueError("newcomer recent-choice likelihood contains invalid values.")
        return np.clip(likelihood, floor, 1.0)

    def _newcomer_scores(
        self,
        candidates: Sequence[int] | np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if cand.size == 0:
            return np.empty(0, dtype=float)
        newcomer_score = str(state.get("newcomer_score", "random"))
        if newcomer_score == "random":
            return np.ones(cand.size, dtype=float)
        floor = float(state.get("newcomer_choice_floor", 1e-9))
        weight = float(state.get("newcomer_choice_weight", 1.0))
        likelihood = self._recent_choice_likelihood(
            cand,
            state,
            errors_only=(newcomer_score == "recent_error_choice"),
        )
        scores = np.power(np.clip(likelihood, floor, 1.0), weight)
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError("newcomer scores contain invalid values.")
        return scores

    def _newcomer_sample(
        self,
        candidates: Sequence[int] | np.ndarray,
        count: int,
        state: Dict[str, Any],
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if count <= 0 or cand.size == 0:
            return np.empty(0, dtype=int)
        actual = min(int(count), int(cand.size))
        raw = self._newcomer_scores(cand, state)
        if float(raw.sum()) <= 0.0:
            return self._sample_from_pool(cand, actual)
        temperature = float(state.get("newcomer_choice_temperature", 1.0))
        weights = np.power(raw + 1e-12, 1.0 / max(temperature, 1e-12))
        prob = self._validate_probability_vector(weights, context="state newcomer policy")
        return self.rng.choice(cand, size=actual, replace=False, p=prob)

    def _survivor_scores(
        self,
        candidates: Sequence[int] | np.ndarray,
        posterior: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if cand.size == 0:
            return np.empty(0, dtype=float)
        raw_post = np.asarray(posterior[cand], dtype=float)
        if not np.all(np.isfinite(raw_post)) or np.any(raw_post < 0):
            raise ValueError("survivor posterior scores contain invalid values.")
        if str(state.get("survivor_score", "posterior")) == "posterior":
            return raw_post
        floor = float(state.get("survivor_choice_floor", 1e-9))
        post_weight = float(state.get("survivor_posterior_weight", 1.0))
        choice_weight = float(state.get("survivor_choice_weight", 1.0))
        choice_like = self._previous_choice_likelihood(cand, state)
        log_score = (
            post_weight * np.log(np.clip(raw_post, floor, 1.0))
            + choice_weight * np.log(np.clip(choice_like, floor, 1.0))
        )
        log_score -= np.max(log_score)
        score = np.exp(log_score)
        if not np.all(np.isfinite(score)) or np.any(score < 0):
            raise ValueError("survivor combined scores contain invalid values.")
        return score

    def _survivor_weighted_sample(
        self,
        candidates: Sequence[int] | np.ndarray,
        count: int,
        posterior: np.ndarray,
        state: Dict[str, Any],
        *,
        temperature: float = 1.0,
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if count <= 0 or cand.size == 0:
            return np.empty(0, dtype=int)
        actual = min(int(count), int(cand.size))
        raw = self._survivor_scores(cand, posterior, state)
        if float(raw.sum()) <= 0.0:
            return self._sample_from_pool(cand, actual)
        weights = np.power(raw + 1e-12, 1.0 / max(float(temperature), 1e-12))
        prob = self._validate_probability_vector(weights, context="state survivor policy")
        return self.rng.choice(cand, size=actual, replace=False, p=prob)

    def _top_survivor_indices(
        self,
        candidates: Sequence[int] | np.ndarray,
        count: int,
        posterior: np.ndarray,
        state: Dict[str, Any],
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if count <= 0 or cand.size == 0:
            return np.empty(0, dtype=int)
        actual = min(int(count), int(cand.size))
        score = self._survivor_scores(cand, posterior, state)
        order = np.argsort(score)[-actual:]
        return np.sort(cand[order])

    def _top_posterior_indices(
        self,
        candidates: Sequence[int] | np.ndarray,
        count: int,
        posterior: np.ndarray,
    ) -> np.ndarray:
        cand = np.asarray(candidates, dtype=int)
        if count <= 0 or cand.size == 0:
            return np.empty(0, dtype=int)
        actual = min(int(count), int(cand.size))
        order = np.argsort(posterior[cand])[-actual:]
        return np.sort(cand[order])

    def _policy_inactive_candidates(self, retained: Sequence[int] | np.ndarray) -> np.ndarray:
        retained_arr = np.asarray(retained, dtype=int)
        old_active = np.asarray(self.old_active, dtype=int) if self.old_active is not None else np.empty(0, dtype=int)
        inactive = self._exclude(self.full_indices, old_active)
        return self._exclude(inactive, retained_arr)

    def _policy_override_prior(
        self,
        active_indices: Sequence[int] | np.ndarray,
        survivor_indices: Sequence[int] | np.ndarray,
        newcomer_indices: Sequence[int] | np.ndarray,
        survivor_values: Sequence[float] | np.ndarray,
        newcomer_values: Sequence[float] | np.ndarray,
        newcomer_mass: float,
        log: Dict[str, Any],
    ) -> None:
        active_arr = np.asarray(active_indices, dtype=int)
        survivor_arr = np.asarray(survivor_indices, dtype=int)
        newcomer_arr = np.asarray(newcomer_indices, dtype=int)
        prior = self._allocate_prior_between_survivors_and_newcomers(
            survivor_arr,
            newcomer_arr,
            np.asarray(survivor_values, dtype=float),
            np.asarray(newcomer_values, dtype=float),
            float(newcomer_mass),
        )
        if float(prior.sum()) <= 0.0 and active_arr.size > 0:
            prior[active_arr] = 1.0 / float(active_arr.size)
        self._post_to_prior_override = {"prior": prior, "log": dict(log)}

    def _run_state_policy(
        self,
        state: Dict[str, Any],
        posterior: np.ndarray,
        step_counts: Dict[str, Any],
    ) -> Set[int]:
        method = str(state["policy_method"])
        limit = self._state_active_limit(state)
        old_active = (
            np.asarray(self.old_active, dtype=int)
            if self.old_active is not None
            else np.empty(0, dtype=int)
        )
        old_active = old_active[(old_active >= 0) & (old_active < self.total_hypo)]
        self._post_to_prior_override = None

        if method == "conservative":
            active = old_active[:limit]
            if active.size == 0:
                active = self._top_survivor_indices(self.full_indices, 1, posterior, state)
            step_counts["state_policy"] = {
                "policy_method": method,
                "survivor_score": str(state.get("survivor_score", "posterior")),
                "newcomer_score": str(state.get("newcomer_score", "random")),
                "retained_count": int(active.size),
                "dropped_count": int(max(0, old_active.size - active.size)),
                "newcomer_count": 0,
                "newcomer_mass": 0.0,
            }
            return {int(x) for x in active}

        if method == "stable":
            explore_count = min(int(state.get("explore_count", 1)), max(0, limit - 1))
            has_inactive = self._policy_inactive_candidates(old_active).size > 0
            reserved_for_new = explore_count if has_inactive else 0
            retain_target = min(int(old_active.size), max(0, limit - reserved_for_new))
            if old_active.size >= limit and has_inactive:
                retain_target = min(retain_target, max(0, limit - 1))
                reserved_for_new = max(1, min(explore_count or 1, limit - retain_target))
            retained = self._survivor_weighted_sample(
                old_active,
                retain_target,
                posterior,
                state,
                temperature=float(state.get("retain_temperature", 1.0)),
            )
            inactive = self._policy_inactive_candidates(retained)
            newcomers = self._newcomer_sample(inactive, min(reserved_for_new, inactive.size), state)
            active = np.sort(np.concatenate([retained, newcomers]).astype(int))
            step_counts["state_policy"] = {
                "policy_method": method,
                "survivor_score": str(state.get("survivor_score", "posterior")),
                "newcomer_score": str(state.get("newcomer_score", "random")),
                "retained_count": int(retained.size),
                "dropped_count": int(max(0, old_active.size - retained.size)),
                "newcomer_count": int(newcomers.size),
                "newcomer_mass": None,
                "p2p": "similarity_novelty",
            }
            return {int(x) for x in active}

        if method == "aggressive":
            source_pool = old_active if old_active.size > 0 else self.full_indices
            top = self._top_survivor_indices(source_pool, 1, posterior, state)
            p_top = float(posterior[int(top[0])]) if top.size else 0.0
            max_newcomers = min(int(state.get("max_newcomers", limit - 1)), max(0, limit - 1))
            min_newcomers = min(int(state.get("min_newcomers", 0)), max_newcomers)
            requested = int(round((1.0 - p_top) * float(max_newcomers)))
            requested = max(min_newcomers, min(max_newcomers, requested))
            inactive = self._policy_inactive_candidates(top)
            newcomers = self._newcomer_sample(inactive, min(requested, inactive.size), state)
            active = np.sort(np.concatenate([top, newcomers]).astype(int))
            newcomer_mass = (1.0 - p_top) if newcomers.size > 0 else 0.0
            self._policy_override_prior(
                active,
                top,
                newcomers,
                np.asarray([max(p_top, 0.0)], dtype=float),
                np.ones(newcomers.size, dtype=float),
                newcomer_mass,
                {
                    "method": "policy_aggressive",
                    "policy_method": method,
                    "top_hypothesis": int(top[0]) if top.size else None,
                    "top_posterior": float(p_top),
                    "newcomer_score": str(state.get("newcomer_score", "random")),
                    "newcomer_mass": float(newcomer_mass),
                },
            )
            step_counts["state_policy"] = {
                "policy_method": method,
                "survivor_score": str(state.get("survivor_score", "posterior")),
                "newcomer_score": str(state.get("newcomer_score", "random")),
                "retained_count": int(top.size),
                "dropped_count": int(max(0, old_active.size - top.size)),
                "newcomer_count": int(newcomers.size),
                "newcomer_mass": float(newcomer_mass),
            }
            return {int(x) for x in active}

        if method == "stubborn":
            retain_count = min(int(state.get("retain_count", 2)), limit)
            source_pool = old_active if old_active.size > 0 else self.full_indices
            retained = self._top_survivor_indices(source_pool, retain_count, posterior, state)
            previous_feedback = float(self.feedback_history[-1]) if self.feedback_history else 1.0
            last_error = float(np.clip(1.0 - previous_feedback, 0.0, 1.0))
            explore_prob = (
                float(state.get("base_explore_prob", 0.02))
                + float(state.get("post_correct_explore_prob", 0.20)) * (1.0 - last_error)
                + float(state.get("post_error_explore_prob", 0.0)) * last_error
            )
            explore_prob = float(np.clip(explore_prob, 0.0, 1.0))
            inactive = self._policy_inactive_candidates(retained)
            can_add = retained.size < limit and inactive.size > 0
            newcomers = (
                self._newcomer_sample(inactive, 1, state)
                if can_add and bool(self.rng.random() < explore_prob)
                else np.empty(0, dtype=int)
            )
            active = np.sort(np.concatenate([retained, newcomers]).astype(int))
            newcomer_mass = float(state.get("newcomer_mass", 0.02)) if newcomers.size > 0 else 0.0
            self._policy_override_prior(
                active,
                retained,
                newcomers,
                posterior[retained] if retained.size > 0 else np.empty(0, dtype=float),
                np.ones(newcomers.size, dtype=float),
                newcomer_mass,
                {
                    "method": "policy_stubborn",
                    "policy_method": method,
                    "last_error": float(last_error),
                    "explore_probability": float(explore_prob),
                    "newcomer_score": str(state.get("newcomer_score", "random")),
                    "newcomer_mass": float(newcomer_mass),
                },
            )
            step_counts["state_policy"] = {
                "policy_method": method,
                "survivor_score": str(state.get("survivor_score", "posterior")),
                "newcomer_score": str(state.get("newcomer_score", "random")),
                "retained_count": int(retained.size),
                "dropped_count": int(max(0, old_active.size - retained.size)),
                "newcomer_count": int(newcomers.size),
                "newcomer_mass": float(newcomer_mass),
                "explore_probability": float(explore_prob),
            }
            return {int(x) for x in active}

        raise ValueError(f"Unsupported state policy_method '{method}'.")

    def _record_previous_observation(
        self,
        observation: tuple[np.ndarray, int, float] | None = None,
    ) -> None:
        if observation is None:
            observation = getattr(self.engine, "observation", None)
        if observation is None or len(observation) < 3:
            return
        self.previous_observation = (
            np.asarray(observation[0], dtype=float).copy(),
            int(observation[1]),
            float(observation[2]),
        )
        self.observation_history.append(self.previous_observation)

    def record_outcome(
        self,
        observation: tuple[np.ndarray, int, float],
    ) -> None:
        """Record one completed trial for the next pre-choice transition."""

        self._record_feedback_from_observation(observation)
        self._record_previous_observation(observation)

    def _init_pool_indices(self) -> np.ndarray:
        if self.init_pool == "all":
            return self.full_indices

        partition = getattr(self.engine, "partition", None)
        metadata = getattr(partition, "hypothesis_metadata", None)
        if metadata is None:
            raise ValueError(
                f"init_pool={self.init_pool!r} requires partition.hypothesis_metadata "
                "with is_label_permuted flags."
            )
        if len(metadata) != self.total_hypo:
            raise ValueError(
                "partition.hypothesis_metadata length does not match hypothesis space "
                f"for init_pool={self.init_pool!r}: {len(metadata)} vs {self.total_hypo}."
            )
        mask = np.asarray(
            [bool(item.get("is_label_permuted", False)) for item in metadata],
            dtype=bool,
        )
        if self.init_pool == "base":
            indices = self.full_indices[~mask]
        elif self.init_pool == "label_permuted":
            indices = self.full_indices[mask]
        else:
            raise ValueError(f"Unsupported init_pool {self.init_pool!r}.")
        if indices.size == 0:
            raise ValueError(f"init_pool={self.init_pool!r} resolved to an empty candidate set.")
        return indices

    def _init_mask(self) -> None:
        # Simple random init
        if self.init_hypotheses is not None:
            forced = np.asarray(self.init_hypotheses, dtype=int)
            if forced.size >= self.init_num:
                selection = forced[: self.init_num]
            else:
                pool = self._exclude(self._init_pool_indices(), forced)
                fill = self._sample_from_pool(pool, self.init_num - int(forced.size))
                selection = np.concatenate([forced, np.asarray(fill, dtype=int)])
        else:
            selection = self._sample_from_pool(self._init_pool_indices(), self.init_num)
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
            step_counts["latent_volatility_confidence"] = latest_volatility.get("confidence")
            step_counts["latent_volatility_signal"] = latest_volatility.get("signal")

        state, controller_log = self._select_strategy_state(posterior)
        if state is None:
            active_strategies = self.strategies
            self._current_post_to_prior_config = self.post_to_prior_config
            self._post_to_prior_override = None
        else:
            active_strategies = state["strategies"]
            self._current_post_to_prior_config = state.get("post_to_prior", self.post_to_prior_config)
            self._post_to_prior_override = None
            step_counts["state_controller"] = controller_log
            step_counts["selected_state"] = controller_log.get("selected_state")
            step_counts["state_probabilities"] = controller_log.get("state_probabilities", {})
            step_counts["selected_policy_method"] = controller_log.get("selected_policy_method")
            step_counts["policy_probabilities"] = controller_log.get("policy_probabilities", {})
        if state is not None and "policy_method" in state:
            new_active_set = self._run_state_policy(state, posterior, step_counts)
        else:
            new_active_set = self._run_strategy_chain(active_strategies, posterior, step_counts, **kwargs)
        
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
            print(f"StrategyChain: {len(self.old_active) if self.old_active is not None else 0} -> {len(self.active)} hypos")
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

    def _previous_controller_confidence(self) -> float:
        """Return pre-feedback confidence recorded at the previous transition."""
        if not self.strategy_counts_log:
            return 1.0
        controller = self.strategy_counts_log[-1].get("state_controller")
        if not isinstance(controller, dict):
            return 1.0
        features = controller.get("features")
        if not isinstance(features, dict):
            return 1.0
        try:
            confidence = float(features.get("posterior_confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        if not np.isfinite(confidence):
            confidence = 1.0
        return float(np.clip(confidence, 0.0, 1.0))

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
                "confidence": 1.0,
                "signal": self.latent_volatility_signal,
            })
            return

        previous_feedback = 1.0
        if self.feedback_history:
            previous_feedback = float(self.feedback_history[-1])
        raw_error_severity = float(np.clip(1.0 - previous_feedback, 0.0, 1.0))
        confidence = self._previous_controller_confidence()
        if self.latent_volatility_signal == "confidence_weighted_error":
            error_severity = raw_error_severity * confidence
        else:
            error_severity = raw_error_severity
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
            "raw_error_severity": float(raw_error_severity),
            "confidence": float(confidence),
            "signal": self.latent_volatility_signal,
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
                    "method": getattr(self, "_current_post_to_prior_config", self.post_to_prior_config).get("method", "similarity_novelty"),
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

        override = getattr(self, "_post_to_prior_override", None)
        if isinstance(override, dict):
            new_prior = np.asarray(override.get("prior"), dtype=float).copy()
            if new_prior.shape[0] != self.total_hypo:
                raise ValueError(
                    "state policy prior override length does not match hypothesis space: "
                    f"{new_prior.shape[0]} vs {self.total_hypo}."
                )
            if not np.all(np.isfinite(new_prior)) or np.any(new_prior < 0):
                raise ValueError("state policy prior override contains invalid values.")
            log = dict(override.get("log", {}) or {})
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
            return

        config = getattr(self, "_current_post_to_prior_config", self.post_to_prior_config)
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

    def state_dict(self) -> Dict[str, Any]:
        """Return dynamic controller state for generic particle resampling."""

        return {
            "active": None if self.active is None else self.active.copy(),
            "old_active": None if self.old_active is None else self.old_active.copy(),
            "previous_observation": deepcopy(self.previous_observation),
            "feedback_history": list(self.feedback_history),
            "observation_history": deepcopy(list(self.observation_history)),
            "latent_volatility_state": float(self.latent_volatility_state),
            "prior_reset_state": float(self._prior_reset_state),
            "current_post_to_prior_config": deepcopy(self._current_post_to_prior_config),
            "post_to_prior_override": deepcopy(self._post_to_prior_override),
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        active = state.get("active")
        old_active = state.get("old_active")
        self.active = None if active is None else np.asarray(active, dtype=int).copy()
        self.old_active = (
            None if old_active is None else np.asarray(old_active, dtype=int).copy()
        )
        self.previous_observation = deepcopy(state.get("previous_observation"))
        self.feedback_history = deque(
            state.get("feedback_history", []),
            maxlen=self.feedback_history.maxlen,
        )
        self.observation_history = deque(
            deepcopy(state.get("observation_history", [])),
            maxlen=self.observation_history.maxlen,
        )
        self.latent_volatility_state = float(
            state.get("latent_volatility_state", 0.0)
        )
        self._prior_reset_state = float(state.get("prior_reset_state", 0.0))
        self._current_post_to_prior_config = deepcopy(
            state.get("current_post_to_prior_config", self.post_to_prior_config)
        )
        self._post_to_prior_override = deepcopy(state.get("post_to_prior_override"))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = deepcopy(rng_state)
        self._apply_mask()

    def clear_logs(self) -> None:
        self.strategy_counts_log.clear()
        self.latent_volatility_log.clear()
        self.prior_reset_log.clear()

    def reseed_future(self, module_seed: int) -> None:
        self.module_seed = int(module_seed)
        self.rng = np.random.default_rng(self.module_seed)


__all__ = ["StrategyPolicyRuntime"]
