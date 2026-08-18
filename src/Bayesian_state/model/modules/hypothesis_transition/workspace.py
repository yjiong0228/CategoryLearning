"""Private bounded-workspace controller used by public H modes."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ..base_module import ModuleRole

from ..base_module import BaseModule


class AdaptiveWorkspaceController(BaseModule):
    """Maintain a bounded workspace with adaptive binomial replacement.

    Before trial ``t > 0`` the module computes feedback surprise and normalized
    rule uncertainty from trial ``t - 1`` and can update both replacement rate
    ``m_t`` and global-search range ``g_t``:

    ``logit(m_t) = logit(m_0) + phi * (logit(m_{t-1}) - logit(m_0))
                    + beta_s * z(surprise) + beta_u * z(uncertainty)``.

    ``g_t`` follows the same mean-reverting form with its own ``g_phi`` and
    ``g_beta_*`` coefficients.  The range controller is disabled by default.

    It then samples ``K_t ~ Binomial(capacity, m_t)``, drops ``K_t`` active
    hypotheses with weights proportional to ``1 - posterior``, and samples the
    same number of inactive newcomers from a local/global proposal mixture.
    Dropped posterior mass is transferred pairwise to the newcomers.
    """

    LEGACY_CONTROLLER_MODE = "legacy"
    FAILURE_ACCUMULATOR_MODE = "failure_accumulator_v2"
    PAIRWISE_PRIOR_ASSIGNMENT = "pairwise_mass_transfer"
    SIMILARITY_TRANSPORT_PRIOR_ASSIGNMENT = "similarity_transport"
    VALID_PRIOR_ASSIGNMENTS = {
        PAIRWISE_PRIOR_ASSIGNMENT,
        SIMILARITY_TRANSPORT_PRIOR_ASSIGNMENT,
    }

    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)

        self.total_hypo = int(getattr(engine, "set_size", 0))
        if self.total_hypo <= 0:
            raise ValueError(
                "bounded-workspace transition requires engine.set_size > 0."
            )

        capacity_raw = kwargs.get(
            "capacity",
            kwargs.get("max_active_hypotheses", kwargs.get("init_num", 3)),
        )
        self.capacity = self._validate_int(capacity_raw, "capacity")
        if not 1 <= self.capacity <= self.total_hypo:
            raise ValueError(
                "capacity must be between 1 and the hypothesis-space size, "
                f"got {self.capacity} for K={self.total_hypo}."
            )
        if self.capacity > self.total_hypo - self.capacity:
            raise ValueError(
                "binomial replacement requires at least capacity inactive hypotheses; "
                f"got capacity={self.capacity}, K={self.total_hypo}."
            )
        for alias in ("init_num", "max_active_hypotheses"):
            if alias in kwargs and self._validate_int(kwargs[alias], alias) != self.capacity:
                raise ValueError(f"{alias} must equal capacity for a fixed workspace.")

        self.prior_assignment_method = self._parse_prior_assignment(
            kwargs.get("prior_assignment")
        )

        controller = kwargs.get("rate_controller", {}) or {}
        if not isinstance(controller, Mapping):
            raise ValueError("rate_controller must be a mapping when provided.")

        def setting(name: str, default: Any) -> Any:
            return controller.get(name, kwargs.get(name, default))

        self.m = self._validate_probability(setting("m", 0.15), "m")
        self.m_phi = self._validate_open_unit(setting("m_phi", setting("phi", 0.0)), "m_phi")
        self.m_beta_surprise = self._validate_nonnegative(
            setting("m_beta_surprise", setting("beta_surprise", 0.0)),
            "m_beta_surprise",
        )
        self.m_beta_uncertainty = self._validate_nonnegative(
            setting("m_beta_uncertainty", setting("beta_uncertainty", 0.0)),
            "m_beta_uncertainty",
        )
        self.surprise_center = self._validate_finite(
            setting("surprise_center", 0.0), "surprise_center"
        )
        self.surprise_scale = self._validate_positive(
            setting("surprise_scale", 1.0), "surprise_scale"
        )
        self.uncertainty_center = self._validate_finite(
            setting("uncertainty_center", 0.0), "uncertainty_center"
        )
        self.uncertainty_scale = self._validate_positive(
            setting("uncertainty_scale", 1.0), "uncertainty_scale"
        )
        self.dynamic_rate = bool(
            self.m_beta_surprise > 0.0 or self.m_beta_uncertainty > 0.0
        )
        if self.dynamic_rate and not 0.0 < self.m < 1.0:
            raise ValueError("a dynamic replacement rate requires baseline m in (0, 1).")

        range_controller = kwargs.get("range_controller", {}) or {}
        if not isinstance(range_controller, Mapping):
            raise ValueError("range_controller must be a mapping when provided.")

        def range_setting(name: str, default: Any) -> Any:
            return range_controller.get(name, kwargs.get(name, default))

        self.g = self._validate_probability(range_setting("g", 0.35), "g")
        self.g_phi = self._validate_open_unit(
            range_setting("g_phi", range_controller.get("phi", 0.0)),
            "g_phi",
        )
        self.g_beta_surprise = self._validate_nonnegative(
            range_setting(
                "g_beta_surprise", range_controller.get("beta_surprise", 0.0)
            ),
            "g_beta_surprise",
        )
        self.g_beta_uncertainty = self._validate_nonnegative(
            range_setting(
                "g_beta_uncertainty", range_controller.get("beta_uncertainty", 0.0)
            ),
            "g_beta_uncertainty",
        )
        self.g_surprise_center = self._validate_finite(
            range_setting("g_surprise_center", self.surprise_center),
            "g_surprise_center",
        )
        self.g_surprise_scale = self._validate_positive(
            range_setting("g_surprise_scale", self.surprise_scale),
            "g_surprise_scale",
        )
        self.g_uncertainty_center = self._validate_finite(
            range_setting("g_uncertainty_center", self.uncertainty_center),
            "g_uncertainty_center",
        )
        self.g_uncertainty_scale = self._validate_positive(
            range_setting("g_uncertainty_scale", self.uncertainty_scale),
            "g_uncertainty_scale",
        )
        self.dynamic_range = bool(
            self.g_beta_surprise > 0.0 or self.g_beta_uncertainty > 0.0
        )
        if self.dynamic_range and not 0.0 < self.g < 1.0:
            raise ValueError("a dynamic search range requires baseline g in (0, 1).")

        self.controller_mode = str(
            kwargs.get("controller_mode", self.LEGACY_CONTROLLER_MODE)
        )
        if self.controller_mode not in {
            self.LEGACY_CONTROLLER_MODE,
            self.FAILURE_ACCUMULATOR_MODE,
        }:
            raise ValueError(
                "controller_mode must be 'legacy' or 'failure_accumulator_v2', "
                f"got {self.controller_mode!r}."
            )
        self.failure_accumulator_enabled = bool(
            self.controller_mode == self.FAILURE_ACCUMULATOR_MODE
        )
        self.uses_outcome_feedback_controller = self.failure_accumulator_enabled
        self._configure_failure_accumulator_controller(
            kwargs.get("failure_accumulator_controller", {})
        )
        if (
            self.prior_assignment_method
            == self.SIMILARITY_TRANSPORT_PRIOR_ASSIGNMENT
            and self.prior_reset_max_strength > 0.0
        ):
            raise ValueError(
                "similarity_transport already defines the transition prior and "
                "cannot be combined with failure-accumulator prior_reset."
            )
        if self.failure_accumulator_enabled:
            self.dynamic_rate = True
            self.dynamic_range = bool(self.global_max > self.global_min)

        tau_local = kwargs.get("tau_local")
        self.tau_local = (
            None if tau_local is None else self._validate_positive(tau_local, "tau_local")
        )
        self.epsilon = self._validate_positive(kwargs.get("epsilon", 1e-12), "epsilon")

        self.module_seed = kwargs.get("module_seed", kwargs.get("random_seed"))
        if self.module_seed is not None:
            self.module_seed = int(self.module_seed)
        seed_sequence = np.random.SeedSequence(self.module_seed)
        init_seed, trial_seed, execution_seed = seed_sequence.spawn(3)
        self.init_rng = np.random.default_rng(init_seed)
        self.trial_rng = np.random.default_rng(trial_seed)
        self.execution_rng = np.random.default_rng(execution_seed)

        raw_prior = np.asarray(getattr(engine, "prior", None), dtype=float).reshape(-1)
        self.base_prior = self._normalize(raw_prior, "initial engine prior")
        if self.base_prior.shape[0] != self.total_hypo:
            raise ValueError(
                "initial engine prior width does not match hypothesis space: "
                f"{self.base_prior.shape[0]} vs {self.total_hypo}."
            )

        self.full_indices = np.arange(self.total_hypo, dtype=int)
        self.active = self._initialize_active_set(kwargs.get("init_hypotheses"))
        self.old_active = self.active.copy()
        self._configure_persistent_execution(kwargs.get("persistent_execution", {}))
        self.executed_hypothesis = self._initialize_executed_hypothesis()
        self.execution_dwell_trials = 0
        self.execution_switch_count = 0
        self.current_execution_switch_probability = 0.0
        self.current_execution_switch_event = False
        # Choice compatibility is updated only after a completed trial and is
        # therefore a strictly history-only signal when trial t is selected.
        self.choice_compatibility = np.full(self.total_hypo, 0.50, dtype=float)
        self.choice_compatibility_observations = 0
        self.misconception_capture_hold_remaining = 0
        self.current_misconception_capture_eligible = False
        self.current_misconception_capture_switch_event = False
        self.current_executed_choice_compatibility = float("nan")
        self.current_best_alternative_choice_compatibility = float("nan")
        self.current_misconception_capture_advantage = float("nan")
        self.rule_commitment_active = False
        self.rule_commitment_age = 0
        self.rule_commitment_disconfirmation = 0.0
        self.rule_commitment_cooldown_remaining = 0
        self.current_rule_commitment_eligible = False
        self.current_rule_commitment_entry_event = False
        self.current_rule_commitment_exit_event = False
        self.current_rule_commitment_candidate = None
        self.current_rule_commitment_candidate_compatibility = float("nan")
        self.current_rule_commitment_runner_up_compatibility = float("nan")
        self.current_rule_commitment_margin = float("nan")
        self.trial_index = 0

        self.baseline_logit = self._logit(self.m) if 0.0 < self.m < 1.0 else 0.0
        self.control_logit = float(self.baseline_logit)
        self.current_m = float(self.m)
        self.g_baseline_logit = self._logit(self.g) if 0.0 < self.g < 1.0 else 0.0
        self.g_control_logit = float(self.g_baseline_logit)
        self.current_g = float(self.g)
        if self.failure_accumulator_enabled:
            self.current_event_probability = float(self.event_min)
            self.current_m = self._event_probability_to_slot_rate(
                self.current_event_probability
            )
            self.current_g = float(self.global_min)
            self.control_logit = self._safe_logit(self.current_m)
            self.g_control_logit = self._safe_logit(self.current_g)
        else:
            self.current_event_probability = self._slot_rate_to_event_probability(
                self.current_m
            )
        self.failure_pressure = float(self.initial_failure)
        self.mastery_evidence = float(self.initial_mastery)
        self.peak_mastery_evidence = float(self.initial_mastery)
        self.previous_feedback = float("nan")
        self.outcome_pending = False
        self.exploration_target = float(self.current_event_probability)
        self.global_target = float(self.current_g)
        self.current_prior_reset_strength = 0.0
        self.predictive_prior: np.ndarray | None = None
        self.feedback_surprise = float("nan")
        self.feedback_uncertainty = float("nan")

        self._distance_matrix: np.ndarray | None = None
        self._local_kernel: np.ndarray | None = None

        self.transition_log: list[Dict[str, Any]] = []
        self.strategy_counts_log = self.transition_log
        self.transition_rate_log: list[Dict[str, Any]] = []

        initial_prior = np.zeros(self.total_hypo, dtype=float)
        initial_prior[self.active] = self.base_prior[self.active]
        initial_prior = self._normalize(initial_prior, "initial active-set prior")
        self.engine.prior = initial_prior
        self._apply_mask()

    @staticmethod
    def _validate_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        parsed = int(value)
        if float(value) != float(parsed):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
        return parsed

    @classmethod
    def _parse_prior_assignment(cls, raw: Any) -> str:
        """Resolve the bounded-workspace posterior-to-prior mapping.

        ``similarity_transport`` deliberately has no method-specific fitted
        parameters.  Its transport strength is the realized replacement
        fraction, and it reuses the already configured local/global kernel.
        """

        if raw is None:
            return cls.PAIRWISE_PRIOR_ASSIGNMENT
        if not isinstance(raw, Mapping):
            raise ValueError("prior_assignment must be a mapping.")
        unknown = set(raw) - {"method"}
        if unknown:
            raise ValueError(
                "bounded-workspace prior_assignment supports only the method "
                f"key; got unsupported keys {sorted(unknown)}."
            )
        method = str(raw.get("method", ""))
        if method not in cls.VALID_PRIOR_ASSIGNMENTS:
            raise ValueError(
                "bounded-workspace prior_assignment.method must be one of "
                f"{sorted(cls.VALID_PRIOR_ASSIGNMENTS)}, got {method!r}."
            )
        return method

    @staticmethod
    def _validate_finite(value: Any, name: str) -> float:
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return parsed

    @classmethod
    def _validate_probability(cls, value: Any, name: str) -> float:
        parsed = cls._validate_finite(value, name)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {parsed!r}.")
        return parsed

    @classmethod
    def _validate_open_unit(cls, value: Any, name: str) -> float:
        parsed = cls._validate_finite(value, name)
        if not -1.0 < parsed < 1.0:
            raise ValueError(f"{name} must lie in (-1, 1), got {parsed!r}.")
        return parsed

    @classmethod
    def _validate_nonnegative(cls, value: Any, name: str) -> float:
        parsed = cls._validate_finite(value, name)
        if parsed < 0.0:
            raise ValueError(f"{name} must be non-negative, got {parsed!r}.")
        return parsed

    @classmethod
    def _validate_positive(cls, value: Any, name: str) -> float:
        parsed = cls._validate_finite(value, name)
        if parsed <= 0.0:
            raise ValueError(f"{name} must be positive, got {parsed!r}.")
        return parsed

    def _configure_failure_accumulator_controller(self, raw: Any) -> None:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("failure_accumulator_controller must be a mapping.")
        if not self.failure_accumulator_enabled:
            if raw:
                raise ValueError(
                    "failure_accumulator_controller requires "
                    "controller_mode='failure_accumulator_v2'."
                )
            raw = {}
        unknown = set(raw) - {"state", "exploration", "range", "prior_reset"}
        if unknown:
            raise ValueError(
                "failure_accumulator_controller has unsupported keys: "
                f"{sorted(unknown)}."
            )

        state = raw.get("state", {}) or {}
        exploration = raw.get("exploration", {}) or {}
        search_range = raw.get("range", {}) or {}
        prior_reset = raw.get("prior_reset", {}) or {}
        for name, value in (
            ("state", state),
            ("exploration", exploration),
            ("range", search_range),
            ("prior_reset", prior_reset),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"failure_accumulator_controller.{name} must be a mapping.")

        state_keys = {
            "failure_decay",
            "mastery_decay",
            "initial_failure",
            "initial_mastery",
        }
        exploration_keys = {
            "event_min",
            "event_max",
            "failure_threshold",
            "failure_gain",
            "uncertainty_weight",
            "mastery_weight",
            "surprise_weight",
            "rise_rate",
            "recovery_rate",
        }
        range_keys = {
            "global_min",
            "global_max",
            "failure_threshold",
            "failure_gain",
            "uncertainty_weight",
            "mastery_weight",
            "surprise_weight",
            "rise_rate",
            "recovery_rate",
        }
        prior_reset_keys = {"max_strength"}
        for name, value, allowed in (
            ("state", state, state_keys),
            ("exploration", exploration, exploration_keys),
            ("range", search_range, range_keys),
            ("prior_reset", prior_reset, prior_reset_keys),
        ):
            extra = set(value) - allowed
            if extra:
                raise ValueError(
                    f"failure_accumulator_controller.{name} has unsupported keys: "
                    f"{sorted(extra)}."
                )

        self.failure_decay = self._validate_probability(
            state.get("failure_decay", 0.60),
            "failure_decay",
        )
        self.mastery_decay = self._validate_probability(
            state.get("mastery_decay", 0.90),
            "mastery_decay",
        )
        if self.failure_accumulator_enabled and (
            self.failure_decay >= 1.0 or self.mastery_decay >= 1.0
        ):
            raise ValueError("failure_decay and mastery_decay must be smaller than 1.")
        self.initial_failure = self._validate_probability(
            state.get("initial_failure", 0.0),
            "initial_failure",
        )
        self.initial_mastery = self._validate_probability(
            state.get("initial_mastery", 0.50),
            "initial_mastery",
        )

        self.event_min = self._validate_probability(
            exploration.get("event_min", 0.05),
            "event_min",
        )
        self.event_max = self._validate_probability(
            exploration.get("event_max", 0.65),
            "event_max",
        )
        if self.event_max <= self.event_min:
            raise ValueError("event_max must be greater than event_min.")
        self.exploration_failure_threshold = self._validate_probability(
            exploration.get("failure_threshold", 0.55),
            "exploration.failure_threshold",
        )
        self.exploration_failure_gain = self._validate_positive(
            exploration.get("failure_gain", 10.0),
            "exploration.failure_gain",
        )
        self.exploration_uncertainty_weight = self._validate_nonnegative(
            exploration.get("uncertainty_weight", 0.0),
            "exploration.uncertainty_weight",
        )
        self.exploration_mastery_weight = self._validate_nonnegative(
            exploration.get("mastery_weight", 1.0),
            "exploration.mastery_weight",
        )
        self.exploration_surprise_weight = self._validate_nonnegative(
            exploration.get("surprise_weight", 0.0),
            "exploration.surprise_weight",
        )
        self.exploration_rise_rate = self._validate_probability(
            exploration.get("rise_rate", 0.80),
            "exploration.rise_rate",
        )
        self.exploration_recovery_rate = self._validate_probability(
            exploration.get("recovery_rate", 0.20),
            "exploration.recovery_rate",
        )

        self.global_min = self._validate_probability(
            search_range.get("global_min", 0.05),
            "range.global_min",
        )
        self.global_max = self._validate_probability(
            search_range.get("global_max", 0.80),
            "range.global_max",
        )
        if self.global_max < self.global_min:
            raise ValueError("global_max must be greater than or equal to global_min.")
        self.global_failure_threshold = self._validate_probability(
            search_range.get("failure_threshold", 0.75),
            "range.failure_threshold",
        )
        self.global_failure_gain = self._validate_positive(
            search_range.get("failure_gain", 12.0),
            "range.failure_gain",
        )
        self.global_uncertainty_weight = self._validate_nonnegative(
            search_range.get("uncertainty_weight", 0.0),
            "range.uncertainty_weight",
        )
        self.global_mastery_weight = self._validate_nonnegative(
            search_range.get("mastery_weight", 0.0),
            "range.mastery_weight",
        )
        self.global_surprise_weight = self._validate_nonnegative(
            search_range.get("surprise_weight", 0.0),
            "range.surprise_weight",
        )
        self.global_rise_rate = self._validate_probability(
            search_range.get("rise_rate", 0.80),
            "range.rise_rate",
        )
        self.global_recovery_rate = self._validate_probability(
            search_range.get("recovery_rate", 0.20),
            "range.recovery_rate",
        )
        self.prior_reset_max_strength = self._validate_probability(
            prior_reset.get("max_strength", 0.0),
            "prior_reset.max_strength",
        )
        if self.failure_accumulator_enabled and (
            self.exploration_rise_rate <= 0.0
            or self.exploration_recovery_rate <= 0.0
            or self.global_rise_rate <= 0.0
            or self.global_recovery_rate <= 0.0
        ):
            raise ValueError("v2 rise/recovery rates must be positive.")

    def _configure_persistent_execution(self, raw: Any) -> None:
        """Configure the overt rule maintained while alternatives are searched."""
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("persistent_execution must be a mapping.")
        unknown = set(raw) - {
            "enabled",
            "switch_scale",
            "misconception_capture",
            "rule_commitment",
        }
        if unknown:
            raise ValueError(
                "persistent_execution has unsupported keys: "
                f"{sorted(unknown)}."
            )
        self.persistent_execution_enabled = bool(raw.get("enabled", False))
        self.execution_switch_scale = self._validate_probability(
            raw.get("switch_scale", 0.20),
            "persistent_execution.switch_scale",
        )

        capture = raw.get("misconception_capture", {}) or {}
        if not isinstance(capture, Mapping):
            raise ValueError(
                "persistent_execution.misconception_capture must be a mapping."
            )
        capture_keys = {
            "enabled",
            "choice_decay",
            "failure_threshold",
            "min_evidence_trials",
            "min_advantage",
            "min_choice_compatibility",
            "min_dwell_trials",
        }
        capture_unknown = set(capture) - capture_keys
        if capture_unknown:
            raise ValueError(
                "persistent_execution.misconception_capture has unsupported keys: "
                f"{sorted(capture_unknown)}."
            )
        self.misconception_capture_enabled = bool(capture.get("enabled", False))
        self.misconception_choice_decay = self._validate_probability(
            capture.get("choice_decay", 0.85),
            "persistent_execution.misconception_capture.choice_decay",
        )
        if self.misconception_choice_decay >= 1.0:
            raise ValueError(
                "misconception_capture.choice_decay must be smaller than 1."
            )
        self.misconception_failure_threshold = self._validate_probability(
            capture.get("failure_threshold", 0.55),
            "persistent_execution.misconception_capture.failure_threshold",
        )
        self.misconception_min_evidence_trials = self._validate_int(
            capture.get("min_evidence_trials", 6),
            "persistent_execution.misconception_capture.min_evidence_trials",
        )
        if self.misconception_min_evidence_trials < 1:
            raise ValueError("misconception_capture.min_evidence_trials must be >= 1.")
        self.misconception_min_advantage = self._validate_probability(
            capture.get("min_advantage", 0.05),
            "persistent_execution.misconception_capture.min_advantage",
        )
        self.misconception_min_choice_compatibility = self._validate_probability(
            capture.get("min_choice_compatibility", 0.0),
            "persistent_execution.misconception_capture.min_choice_compatibility",
        )
        self.misconception_min_dwell_trials = self._validate_int(
            capture.get("min_dwell_trials", 8),
            "persistent_execution.misconception_capture.min_dwell_trials",
        )
        if self.misconception_min_dwell_trials < 1:
            raise ValueError("misconception_capture.min_dwell_trials must be >= 1.")

        commitment = raw.get("rule_commitment", {}) or {}
        if not isinstance(commitment, Mapping):
            raise ValueError(
                "persistent_execution.rule_commitment must be a mapping."
            )
        commitment_keys = {
            "enabled",
            "choice_decay",
            "failure_threshold",
            "min_evidence_trials",
            "min_prior_mastery",
            "min_choice_compatibility",
            "min_runner_up_margin",
            "entry_probability",
            "min_dwell_trials",
            "min_hold_choice_compatibility",
            "disconfirmation_decay",
            "recovery_threshold",
            "reentry_cooldown_trials",
        }
        commitment_unknown = set(commitment) - commitment_keys
        if commitment_unknown:
            raise ValueError(
                "persistent_execution.rule_commitment has unsupported keys: "
                f"{sorted(commitment_unknown)}."
            )
        self.rule_commitment_enabled = bool(commitment.get("enabled", False))
        self.rule_commitment_choice_decay = self._validate_probability(
            commitment.get("choice_decay", 0.875),
            "persistent_execution.rule_commitment.choice_decay",
        )
        if self.rule_commitment_choice_decay >= 1.0:
            raise ValueError("rule_commitment.choice_decay must be smaller than 1.")
        self.rule_commitment_failure_threshold = self._validate_probability(
            commitment.get("failure_threshold", 0.60),
            "persistent_execution.rule_commitment.failure_threshold",
        )
        self.rule_commitment_min_evidence_trials = self._validate_int(
            commitment.get("min_evidence_trials", 8),
            "persistent_execution.rule_commitment.min_evidence_trials",
        )
        if self.rule_commitment_min_evidence_trials < 1:
            raise ValueError("rule_commitment.min_evidence_trials must be >= 1.")
        self.rule_commitment_min_prior_mastery = self._validate_probability(
            commitment.get("min_prior_mastery", 0.0),
            "persistent_execution.rule_commitment.min_prior_mastery",
        )
        self.rule_commitment_min_choice_compatibility = self._validate_probability(
            commitment.get("min_choice_compatibility", 0.75),
            "persistent_execution.rule_commitment.min_choice_compatibility",
        )
        self.rule_commitment_min_runner_up_margin = self._validate_probability(
            commitment.get("min_runner_up_margin", 0.05),
            "persistent_execution.rule_commitment.min_runner_up_margin",
        )
        self.rule_commitment_entry_probability = self._validate_probability(
            commitment.get("entry_probability", 1.0),
            "persistent_execution.rule_commitment.entry_probability",
        )
        self.rule_commitment_min_dwell_trials = self._validate_int(
            commitment.get("min_dwell_trials", 12),
            "persistent_execution.rule_commitment.min_dwell_trials",
        )
        if self.rule_commitment_min_dwell_trials < 1:
            raise ValueError("rule_commitment.min_dwell_trials must be >= 1.")
        self.rule_commitment_min_hold_choice_compatibility = (
            self._validate_probability(
                commitment.get("min_hold_choice_compatibility", 0.0),
                "persistent_execution.rule_commitment."
                "min_hold_choice_compatibility",
            )
        )
        self.rule_commitment_disconfirmation_decay = self._validate_probability(
            commitment.get("disconfirmation_decay", 0.95),
            "persistent_execution.rule_commitment.disconfirmation_decay",
        )
        if self.rule_commitment_disconfirmation_decay >= 1.0:
            raise ValueError(
                "rule_commitment.disconfirmation_decay must be smaller than 1."
            )
        self.rule_commitment_recovery_threshold = self._validate_positive(
            commitment.get("recovery_threshold", 6.0),
            "persistent_execution.rule_commitment.recovery_threshold",
        )
        self.rule_commitment_reentry_cooldown_trials = self._validate_int(
            commitment.get("reentry_cooldown_trials", 8),
            "persistent_execution.rule_commitment.reentry_cooldown_trials",
        )
        if self.rule_commitment_reentry_cooldown_trials < 0:
            raise ValueError("rule_commitment.reentry_cooldown_trials must be >= 0.")

        if self.misconception_capture_enabled and self.rule_commitment_enabled:
            raise ValueError(
                "Configure misconception_capture or rule_commitment, not both."
            )
        if (
            self.persistent_execution_enabled
            and not self.failure_accumulator_enabled
            and not bool(
                getattr(self, "allows_legacy_persistent_execution", False)
            )
        ):
            raise ValueError(
                "persistent execution currently requires "
                "controller_mode='failure_accumulator_v2'."
            )
        if self.persistent_execution_enabled and self.capacity < 2:
            raise ValueError(
                "persistent execution requires capacity >= 2 so one rule can be "
                "executed while at least one alternative is searched."
            )
        if (
            self.misconception_capture_enabled or self.rule_commitment_enabled
        ) and not self.persistent_execution_enabled:
            raise ValueError(
                "misconception capture and rule commitment require "
                "persistent_execution.enabled=true."
            )

    def _initialize_executed_hypothesis(self) -> int | None:
        if not self.persistent_execution_enabled:
            return None
        weights = self._normalize(
            self.base_prior[self.active],
            "initial executed-hypothesis weights",
        )
        return int(self.execution_rng.choice(self.active, p=weights))

    @classmethod
    def _safe_logit(cls, probability: float) -> float:
        clipped = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
        return cls._logit(clipped)

    def _slot_rate_to_event_probability(self, rate: float) -> float:
        return float(1.0 - (1.0 - float(rate)) ** self.capacity)

    def _event_probability_to_slot_rate(self, probability: float) -> float:
        return float(1.0 - (1.0 - float(probability)) ** (1.0 / self.capacity))

    @staticmethod
    def _event_probability_to_rate_for_slots(
        probability: float,
        slot_count: int,
    ) -> float:
        slots = int(slot_count)
        if slots <= 0:
            raise ValueError("slot_count must be positive.")
        return float(1.0 - (1.0 - float(probability)) ** (1.0 / slots))

    @staticmethod
    def _asymmetric_update(
        current: float,
        target: float,
        *,
        rise_rate: float,
        recovery_rate: float,
    ) -> float:
        rate = float(rise_rate if target > current else recovery_rate)
        return float(current + rate * (target - current))

    @staticmethod
    def _normalize(values: np.ndarray, context: str) -> np.ndarray:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{context} must be finite and non-negative.")
        total = float(np.sum(array))
        if total <= 0.0:
            raise ValueError(f"{context} has zero mass.")
        return array / total

    @staticmethod
    def _logit(probability: float) -> float:
        return math.log(float(probability) / (1.0 - float(probability)))

    @staticmethod
    def _expit(value: float) -> float:
        clipped = float(np.clip(value, -30.0, 30.0))
        return 1.0 / (1.0 + math.exp(-clipped))

    def _initialize_active_set(self, raw: Sequence[int] | np.ndarray | None) -> np.ndarray:
        if raw is None:
            selected = self.init_rng.choice(
                self.full_indices,
                size=self.capacity,
                replace=False,
                p=self.base_prior,
            )
            return np.sort(np.asarray(selected, dtype=int))
        selected = np.asarray(raw, dtype=int).reshape(-1)
        if selected.size != self.capacity:
            raise ValueError(
                "init_hypotheses must contain exactly capacity indices, "
                f"got {selected.size} for capacity={self.capacity}."
            )
        if np.unique(selected).size != selected.size:
            raise ValueError("init_hypotheses cannot contain duplicate indices.")
        if np.any(selected < 0) or np.any(selected >= self.total_hypo):
            raise ValueError("init_hypotheses contains indices outside the hypothesis space.")
        return np.sort(selected)

    def _apply_mask(self) -> None:
        mask = np.zeros(self.total_hypo, dtype=float)
        mask[self.active] = 1.0
        self.engine.hypotheses_mask = mask

    def _posterior_for_transition(self) -> np.ndarray:
        raw = getattr(self.engine, "posterior", None)
        if raw is None:
            raw = getattr(self.engine, "prior", None)
        posterior = np.asarray(raw, dtype=float).reshape(-1)
        if posterior.shape[0] != self.total_hypo:
            raise ValueError(
                "transition posterior width does not match hypothesis space: "
                f"{posterior.shape[0]} vs {self.total_hypo}."
            )
        active_values = np.zeros(self.total_hypo, dtype=float)
        active_values[self.active] = posterior[self.active]
        return self._normalize(active_values, "active transition posterior")

    def _previous_feedback_signals(self) -> tuple[float, float]:
        if self.predictive_prior is None:
            return float("nan"), float("nan")
        likelihood = getattr(self.engine, "likelihood", None)
        posterior = getattr(self.engine, "posterior", None)
        if likelihood is None or posterior is None:
            return float("nan"), float("nan")
        likelihood_array = np.asarray(likelihood, dtype=float).reshape(-1)
        posterior_array = np.asarray(posterior, dtype=float).reshape(-1)
        if (
            likelihood_array.shape[0] != self.total_hypo
            or posterior_array.shape[0] != self.total_hypo
        ):
            raise ValueError("previous likelihood/posterior width does not match hypothesis space.")
        if not np.all(np.isfinite(likelihood_array)) or np.any(likelihood_array < 0.0):
            raise ValueError("previous likelihood must be finite and non-negative.")
        feedback_probability = float(np.sum(self.predictive_prior * likelihood_array))
        surprise = -math.log(max(feedback_probability, self.epsilon))

        active_values = posterior_array[self.active]
        active_values = self._normalize(active_values, "previous active posterior")
        if self.capacity <= 1:
            uncertainty = 0.0
        else:
            uncertainty = float(
                -np.sum(active_values * np.log(np.clip(active_values, self.epsilon, 1.0)))
                / math.log(self.capacity)
            )
        return float(surprise), float(uncertainty)

    def _update_failure_accumulator_controls(
        self,
        *,
        z_surprise: float,
        z_uncertainty: float,
    ) -> None:
        if self.outcome_pending and np.isfinite(self.previous_feedback):
            feedback = float(np.clip(self.previous_feedback, 0.0, 1.0))
            error = 1.0 - feedback
            self.failure_pressure = float(
                self.failure_decay * self.failure_pressure
                + (1.0 - self.failure_decay) * error
            )
            self.mastery_evidence = float(
                self.mastery_decay * self.mastery_evidence
                + (1.0 - self.mastery_decay) * feedback
            )
            self.peak_mastery_evidence = float(
                max(self.peak_mastery_evidence, self.mastery_evidence)
            )
            self.outcome_pending = False

        safe_surprise = float(z_surprise) if np.isfinite(z_surprise) else 0.0
        safe_uncertainty = float(z_uncertainty) if np.isfinite(z_uncertainty) else 0.0
        exploration_drive = (
            self.exploration_failure_gain
            * (self.failure_pressure - self.exploration_failure_threshold)
            + self.exploration_uncertainty_weight * safe_uncertainty
            - self.exploration_mastery_weight * self.mastery_evidence
            + self.exploration_surprise_weight * max(safe_surprise, 0.0)
        )
        exploration_gate = self._expit(exploration_drive)
        self.exploration_target = float(
            self.event_min
            + (self.event_max - self.event_min) * exploration_gate
        )
        self.current_event_probability = self._asymmetric_update(
            self.current_event_probability,
            self.exploration_target,
            rise_rate=self.exploration_rise_rate,
            recovery_rate=self.exploration_recovery_rate,
        )
        self.current_m = self._event_probability_to_slot_rate(
            self.current_event_probability
        )
        self.control_logit = self._safe_logit(self.current_m)

        global_drive = (
            self.global_failure_gain
            * (self.failure_pressure - self.global_failure_threshold)
            + self.global_uncertainty_weight * safe_uncertainty
            - self.global_mastery_weight * self.mastery_evidence
            + self.global_surprise_weight * max(safe_surprise, 0.0)
        )
        global_gate = self._expit(global_drive)
        self.global_target = float(
            self.global_min + (self.global_max - self.global_min) * global_gate
        )
        self.current_g = self._asymmetric_update(
            self.current_g,
            self.global_target,
            rise_rate=self.global_rise_rate,
            recovery_rate=self.global_recovery_rate,
        )
        self.g_control_logit = self._safe_logit(self.current_g)
        if self.global_max > self.global_min:
            normalized_global_search = float(
                np.clip(
                    (self.current_g - self.global_min)
                    / (self.global_max - self.global_min),
                    0.0,
                    1.0,
                )
            )
        else:
            normalized_global_search = 0.0
        self.current_prior_reset_strength = float(
            self.prior_reset_max_strength * normalized_global_search
        )

    def _update_transition_controls(self) -> tuple[float, float]:
        surprise, uncertainty = self._previous_feedback_signals()
        self.feedback_surprise = float(surprise)
        self.feedback_uncertainty = float(uncertainty)
        signals_available = bool(np.isfinite(surprise) and np.isfinite(uncertainty))
        if signals_available:
            z_surprise = (surprise - self.surprise_center) / self.surprise_scale
            z_uncertainty = (
                uncertainty - self.uncertainty_center
            ) / self.uncertainty_scale
        else:
            z_surprise = float("nan")
            z_uncertainty = float("nan")

        if self.failure_accumulator_enabled:
            self._update_failure_accumulator_controls(
                z_surprise=z_surprise,
                z_uncertainty=z_uncertainty,
            )
            return float(z_surprise), float(z_uncertainty)

        if self.dynamic_rate and signals_available:
            self.control_logit = (
                self.baseline_logit
                + self.m_phi * (self.control_logit - self.baseline_logit)
                + self.m_beta_surprise * z_surprise
                + self.m_beta_uncertainty * z_uncertainty
            )
            self.current_m = self._expit(self.control_logit)
        else:
            self.current_m = float(self.m)

        if self.dynamic_range and signals_available:
            g_z_surprise = (
                surprise - self.g_surprise_center
            ) / self.g_surprise_scale
            g_z_uncertainty = (
                uncertainty - self.g_uncertainty_center
            ) / self.g_uncertainty_scale
            self.g_control_logit = (
                self.g_baseline_logit
                + self.g_phi * (self.g_control_logit - self.g_baseline_logit)
                + self.g_beta_surprise * g_z_surprise
                + self.g_beta_uncertainty * g_z_uncertainty
            )
            self.current_g = self._expit(self.g_control_logit)
        else:
            self.current_g = float(self.g)
        return float(z_surprise), float(z_uncertainty)

    def record_outcome(
        self,
        observation: tuple[np.ndarray, int, float],
    ) -> None:
        """Store completed feedback for the next causal controller update."""
        if not self.failure_accumulator_enabled:
            return
        if observation is None or len(observation) < 3:
            raise ValueError(
                "failure_accumulator_v2 requires observation=(stimulus, choice, feedback)."
            )
        feedback = self._validate_probability(observation[2], "feedback")
        self.previous_feedback = float(feedback)
        self.outcome_pending = True

        if self.rule_commitment_enabled and self.rule_commitment_active:
            error = 1.0 - float(feedback)
            self.rule_commitment_disconfirmation = float(
                self.rule_commitment_disconfirmation_decay
                * self.rule_commitment_disconfirmation
                + error
            )

        compatibility_enabled = bool(
            self.misconception_capture_enabled or self.rule_commitment_enabled
        )
        if not compatibility_enabled:
            return
        partition = getattr(self.engine, "partition", None)
        if partition is None or not hasattr(partition, "get_category_assignment"):
            raise ValueError(
                "choice-supported execution requires "
                "partition.get_category_assignment."
            )
        choice = self._validate_int(observation[1], "choice") - 1
        n_cats = int(getattr(partition, "n_cats", 2))
        if not 0 <= choice < n_cats:
            raise ValueError(
                f"choice must be 1-based and lie in [1, {n_cats}]."
            )
        stimulus = np.asarray(observation[0], dtype=float)
        distance_mode = getattr(self.engine, "distance_mode", "prototype")
        supports = np.fromiter(
            (
                float(
                    partition.get_category_assignment(
                        hypo=hypothesis,
                        stimulus=stimulus,
                        distance_mode=distance_mode,
                        beta=1.0,
                    )
                    == choice
                )
                for hypothesis in range(self.total_hypo)
            ),
            dtype=float,
            count=self.total_hypo,
        )
        decay = float(
            self.rule_commitment_choice_decay
            if self.rule_commitment_enabled
            else self.misconception_choice_decay
        )
        self.choice_compatibility = (
            decay * self.choice_compatibility + (1.0 - decay) * supports
        )
        self.choice_compatibility_observations += 1

    def _misconception_search_is_eligible(self) -> bool:
        return bool(
            self.misconception_capture_enabled
            and self.choice_compatibility_observations
            >= self.misconception_min_evidence_trials
            and self.failure_pressure >= self.misconception_failure_threshold
        )

    def _misconception_capture_target(
        self,
        alternatives: np.ndarray,
    ) -> tuple[int | None, float, float, float, bool]:
        """Resolve a history-supported alternative without inspecting trial-t choice."""
        current = self.executed_hypothesis
        if current is None or alternatives.size == 0:
            return None, float("nan"), float("nan"), float("nan"), False
        current_support = float(self.choice_compatibility[int(current)])
        alternative_supports = self.choice_compatibility[alternatives]
        best_support = float(np.max(alternative_supports))
        tied = alternatives[
            np.isclose(alternative_supports, best_support, rtol=0.0, atol=1e-12)
        ]
        target = int(tied[int(np.argmax(self.base_prior[tied]))])
        advantage = float(best_support - current_support)
        eligible = bool(
            self._misconception_search_is_eligible()
            and advantage >= self.misconception_min_advantage
            and best_support >= self.misconception_min_choice_compatibility
        )
        return target, current_support, best_support, advantage, eligible

    def _rule_commitment_candidate(
        self,
    ) -> tuple[int | None, float, float, float, bool]:
        """Choose the strongest history-supported rule from the full space."""
        if not self.rule_commitment_enabled or self.total_hypo < 2:
            return None, float("nan"), float("nan"), float("nan"), False
        supports = np.asarray(self.choice_compatibility, dtype=float)
        best_support = float(np.max(supports))
        tied = self.full_indices[
            np.isclose(supports, best_support, rtol=0.0, atol=1e-12)
        ]
        target = int(tied[int(np.argmax(self.base_prior[tied]))])
        runner_up_support = float(np.max(np.delete(supports, target)))
        margin = float(best_support - runner_up_support)
        eligible = bool(
            self.choice_compatibility_observations
            >= self.rule_commitment_min_evidence_trials
            and self.peak_mastery_evidence
            >= self.rule_commitment_min_prior_mastery
            and self.failure_pressure >= self.rule_commitment_failure_threshold
            and best_support >= self.rule_commitment_min_choice_compatibility
            and margin >= self.rule_commitment_min_runner_up_margin
        )
        return target, best_support, runner_up_support, margin, eligible

    def _prepare_rule_commitment(self) -> int | None:
        """Resolve a history-only commitment entry request before trial-t choice."""
        self.current_rule_commitment_eligible = False
        self.current_rule_commitment_entry_event = False
        self.current_rule_commitment_exit_event = False
        self.current_rule_commitment_candidate = None
        self.current_rule_commitment_candidate_compatibility = float("nan")
        self.current_rule_commitment_runner_up_compatibility = float("nan")
        self.current_rule_commitment_margin = float("nan")
        if not self.rule_commitment_enabled:
            return None

        if self.rule_commitment_active:
            current = int(self.executed_hypothesis)
            support = float(self.choice_compatibility[current])
            others = np.delete(self.choice_compatibility, current)
            runner_up = float(np.max(others)) if others.size else float("nan")
            self.current_rule_commitment_candidate = current
            self.current_rule_commitment_candidate_compatibility = support
            self.current_rule_commitment_runner_up_compatibility = runner_up
            self.current_rule_commitment_margin = float(
                support - runner_up if np.isfinite(runner_up) else float("nan")
            )
            can_recover = bool(
                self.rule_commitment_age >= self.rule_commitment_min_dwell_trials
                and (
                    self.rule_commitment_disconfirmation
                    >= self.rule_commitment_recovery_threshold
                    or (
                        self.rule_commitment_min_hold_choice_compatibility > 0.0
                        and support
                        < self.rule_commitment_min_hold_choice_compatibility
                    )
                )
            )
            if can_recover:
                self.rule_commitment_active = False
                self.rule_commitment_age = 0
                self.rule_commitment_cooldown_remaining = int(
                    self.rule_commitment_reentry_cooldown_trials
                )
                self.current_rule_commitment_exit_event = True
            return None

        if self.rule_commitment_cooldown_remaining > 0:
            self.rule_commitment_cooldown_remaining -= 1
            return None

        target, best, runner_up, margin, eligible = (
            self._rule_commitment_candidate()
        )
        self.current_rule_commitment_candidate = target
        self.current_rule_commitment_candidate_compatibility = float(best)
        self.current_rule_commitment_runner_up_compatibility = float(runner_up)
        self.current_rule_commitment_margin = float(margin)
        self.current_rule_commitment_eligible = bool(eligible)
        if not eligible or target is None:
            return None
        entry_draw = float(self.execution_rng.random())
        return (
            int(target)
            if entry_draw < self.rule_commitment_entry_probability
            else None
        )

    def _ensure_geometry(self) -> None:
        if self._distance_matrix is not None and self._local_kernel is not None:
            return
        partition = getattr(self.engine, "partition", None)
        similarity = getattr(partition, "similarity_matrix", None)
        if similarity is None:
            raise ValueError(
                "adaptive bounded-workspace transitions require partition.similarity_matrix."
            )
        similarity_array = np.asarray(similarity, dtype=float)
        expected = (self.total_hypo, self.total_hypo)
        if similarity_array.shape != expected:
            raise ValueError(
                "partition similarity matrix has the wrong shape: "
                f"{similarity_array.shape} vs {expected}."
            )
        distance = np.clip(1.0 - similarity_array, 0.0, 1.0)
        nonself = distance.copy()
        np.fill_diagonal(nonself, np.inf)
        tau = self.tau_local
        if tau is None:
            tau = float(np.median(np.min(nonself, axis=1)))
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("resolved tau_local must be finite and positive.")
        local = np.zeros_like(distance)
        for source in range(self.total_hypo):
            weights = self.base_prior * np.exp(-distance[source] / float(tau))
            weights[source] = 0.0
            local[source] = self._normalize(weights, f"local kernel row {source}")
        self.tau_local = float(tau)
        self._distance_matrix = distance
        self._local_kernel = local

    def _weighted_sample_without_replacement(
        self,
        pool: np.ndarray,
        weights: np.ndarray,
        count: int,
    ) -> np.ndarray:
        count = int(count)
        pool_array = np.asarray(pool, dtype=int).reshape(-1)
        if count == 0:
            return np.empty(0, dtype=int)
        if count < 0 or count > pool_array.size:
            raise ValueError(
                f"cannot sample {count} values without replacement from {pool_array.size}."
            )
        probability = self._normalize(np.asarray(weights, dtype=float), "sampling weights")
        if int(np.count_nonzero(probability > 0.0)) < count:
            probability = self._normalize(probability + self.epsilon, "regularized weights")
        selected = self.trial_rng.choice(
            pool_array,
            size=count,
            replace=False,
            p=probability,
        )
        return np.asarray(selected, dtype=int)

    def _newcomer_proposal(
        self,
        posterior: np.ndarray,
        inactive: np.ndarray,
    ) -> np.ndarray:
        global_weights = self._normalize(
            self.base_prior[inactive], "inactive global proposal"
        )
        if self.current_g >= 1.0:
            proposal = global_weights
        else:
            self._ensure_geometry()
            assert self._local_kernel is not None
            local_full = np.asarray(posterior @ self._local_kernel, dtype=float)
            local = self._normalize(local_full[inactive], "inactive local proposal")
            proposal = self._normalize(
                (1.0 - self.current_g) * local + self.current_g * global_weights,
                "newcomer mixture",
            )
        if self._misconception_search_is_eligible():
            proposal = self._normalize(
                proposal
                * np.clip(self.choice_compatibility[inactive], self.epsilon, 1.0),
                "choice-compatible newcomer proposal",
            )
        return proposal

    def _newcomer_distance(
        self,
        posterior: np.ndarray,
        active_before: np.ndarray,
        newcomers: np.ndarray,
    ) -> float:
        if newcomers.size == 0:
            return 0.0
        self._ensure_geometry()
        assert self._distance_matrix is not None
        total = 0.0
        for newcomer in newcomers:
            total += float(
                np.sum(
                    posterior[active_before]
                    * self._distance_matrix[active_before, int(newcomer)]
                )
            )
        return total / float(newcomers.size)

    def _initialize_newcomer_beta(self, newcomers: np.ndarray) -> None:
        if newcomers.size == 0:
            return
        beta_mod = self.engine.get_module(ModuleRole.BETA)
        if beta_mod is not None and hasattr(beta_mod, "initialize_beta_for_hypotheses"):
            beta_mod.initialize_beta_for_hypotheses(newcomers, priors=None)

    def _initialize_newcomer_mapping(self, newcomers: np.ndarray) -> None:
        """Reset label orientation when a geometry enters the workspace."""

        if newcomers.size == 0:
            return
        try:
            mapping_mod = self.engine.get_module(ModuleRole.MAPPING)
        except KeyError:
            # Lightweight test/legacy engines may expose only the original
            # module roles.  Mapping is an optional M1 extension.
            mapping_mod = None
        if mapping_mod is not None and hasattr(
            mapping_mod, "initialize_orientation_for_hypotheses"
        ):
            mapping_mod.initialize_orientation_for_hypotheses(newcomers)

    def reseed_future(self, module_seed: int) -> None:
        self.module_seed = int(module_seed)
        self.trial_rng = np.random.default_rng(self.module_seed)
        self.execution_rng = np.random.default_rng(
            np.random.SeedSequence([self.module_seed, 0x45584543])
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active.copy(),
            "old_active": self.old_active.copy(),
            "trial_index": int(self.trial_index),
            "control_logit": float(self.control_logit),
            "current_m": float(self.current_m),
            "g_control_logit": float(self.g_control_logit),
            "current_g": float(self.current_g),
            "controller_mode": str(self.controller_mode),
            "current_event_probability": float(self.current_event_probability),
            "failure_pressure": float(self.failure_pressure),
            "mastery_evidence": float(self.mastery_evidence),
            "peak_mastery_evidence": float(self.peak_mastery_evidence),
            "previous_feedback": float(self.previous_feedback),
            "outcome_pending": bool(self.outcome_pending),
            "exploration_target": float(self.exploration_target),
            "global_target": float(self.global_target),
            "current_prior_reset_strength": float(
                self.current_prior_reset_strength
            ),
            "predictive_prior": (
                None if self.predictive_prior is None else self.predictive_prior.copy()
            ),
            "feedback_surprise": float(self.feedback_surprise),
            "feedback_uncertainty": float(self.feedback_uncertainty),
            "trial_rng_state": deepcopy(self.trial_rng.bit_generator.state),
            "persistent_execution_enabled": bool(
                self.persistent_execution_enabled
            ),
            "executed_hypothesis": self.executed_hypothesis,
            "execution_dwell_trials": int(self.execution_dwell_trials),
            "execution_switch_count": int(self.execution_switch_count),
            "current_execution_switch_probability": float(
                self.current_execution_switch_probability
            ),
            "current_execution_switch_event": bool(
                self.current_execution_switch_event
            ),
            "misconception_capture_enabled": bool(
                self.misconception_capture_enabled
            ),
            "choice_compatibility": self.choice_compatibility.copy(),
            "choice_compatibility_observations": int(
                self.choice_compatibility_observations
            ),
            "misconception_capture_hold_remaining": int(
                self.misconception_capture_hold_remaining
            ),
            "current_misconception_capture_eligible": bool(
                self.current_misconception_capture_eligible
            ),
            "current_misconception_capture_switch_event": bool(
                self.current_misconception_capture_switch_event
            ),
            "current_executed_choice_compatibility": float(
                self.current_executed_choice_compatibility
            ),
            "current_best_alternative_choice_compatibility": float(
                self.current_best_alternative_choice_compatibility
            ),
            "current_misconception_capture_advantage": float(
                self.current_misconception_capture_advantage
            ),
            "rule_commitment_enabled": bool(self.rule_commitment_enabled),
            "rule_commitment_active": bool(self.rule_commitment_active),
            "rule_commitment_age": int(self.rule_commitment_age),
            "rule_commitment_disconfirmation": float(
                self.rule_commitment_disconfirmation
            ),
            "rule_commitment_cooldown_remaining": int(
                self.rule_commitment_cooldown_remaining
            ),
            "current_rule_commitment_eligible": bool(
                self.current_rule_commitment_eligible
            ),
            "current_rule_commitment_entry_event": bool(
                self.current_rule_commitment_entry_event
            ),
            "current_rule_commitment_exit_event": bool(
                self.current_rule_commitment_exit_event
            ),
            "current_rule_commitment_candidate": (
                None
                if self.current_rule_commitment_candidate is None
                else int(self.current_rule_commitment_candidate)
            ),
            "current_rule_commitment_candidate_compatibility": float(
                self.current_rule_commitment_candidate_compatibility
            ),
            "current_rule_commitment_runner_up_compatibility": float(
                self.current_rule_commitment_runner_up_compatibility
            ),
            "current_rule_commitment_margin": float(
                self.current_rule_commitment_margin
            ),
            "execution_rng_state": deepcopy(
                self.execution_rng.bit_generator.state
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.active = np.asarray(state["active"], dtype=int).copy()
        self.old_active = np.asarray(state["old_active"], dtype=int).copy()
        self.trial_index = int(state["trial_index"])
        self.control_logit = float(state["control_logit"])
        self.current_m = float(state["current_m"])
        self.g_control_logit = float(
            state.get("g_control_logit", self.g_baseline_logit)
        )
        self.current_g = float(state.get("current_g", self.g))
        saved_mode = str(state.get("controller_mode", self.controller_mode))
        if saved_mode != self.controller_mode:
            raise ValueError(
                "controller state mode does not match configured mode: "
                f"{saved_mode!r} vs {self.controller_mode!r}."
            )
        self.current_event_probability = float(
            state.get(
                "current_event_probability",
                self._slot_rate_to_event_probability(self.current_m),
            )
        )
        self.failure_pressure = float(
            state.get("failure_pressure", self.initial_failure)
        )
        self.mastery_evidence = float(
            state.get("mastery_evidence", self.initial_mastery)
        )
        self.peak_mastery_evidence = float(
            state.get("peak_mastery_evidence", self.mastery_evidence)
        )
        if not 0.0 <= self.peak_mastery_evidence <= 1.0:
            raise ValueError("restored peak_mastery_evidence must lie in [0, 1].")
        self.previous_feedback = float(
            state.get("previous_feedback", float("nan"))
        )
        self.outcome_pending = bool(state.get("outcome_pending", False))
        self.exploration_target = float(
            state.get("exploration_target", self.current_event_probability)
        )
        self.global_target = float(state.get("global_target", self.current_g))
        self.current_prior_reset_strength = float(
            state.get("current_prior_reset_strength", 0.0)
        )
        predictive_prior = state.get("predictive_prior")
        self.predictive_prior = (
            None
            if predictive_prior is None
            else np.asarray(predictive_prior, dtype=float).copy()
        )
        self.feedback_surprise = float(state.get("feedback_surprise", float("nan")))
        self.feedback_uncertainty = float(
            state.get("feedback_uncertainty", float("nan"))
        )
        rng_state = state.get("trial_rng_state")
        if rng_state is not None:
            self.trial_rng.bit_generator.state = deepcopy(rng_state)
        saved_execution_enabled = bool(
            state.get(
                "persistent_execution_enabled",
                self.persistent_execution_enabled,
            )
        )
        if saved_execution_enabled != self.persistent_execution_enabled:
            raise ValueError(
                "persistent-execution state does not match configured enabled flag."
            )
        executed = state.get("executed_hypothesis", self.executed_hypothesis)
        self.executed_hypothesis = None if executed is None else int(executed)
        self.execution_dwell_trials = int(state.get("execution_dwell_trials", 0))
        self.execution_switch_count = int(state.get("execution_switch_count", 0))
        self.current_execution_switch_probability = float(
            state.get("current_execution_switch_probability", 0.0)
        )
        self.current_execution_switch_event = bool(
            state.get("current_execution_switch_event", False)
        )
        saved_capture_enabled = bool(
            state.get(
                "misconception_capture_enabled",
                self.misconception_capture_enabled,
            )
        )
        if saved_capture_enabled != self.misconception_capture_enabled:
            raise ValueError(
                "misconception-capture state does not match configured enabled flag."
            )
        choice_compatibility = np.asarray(
            state.get(
                "choice_compatibility",
                np.full(self.total_hypo, 0.50, dtype=float),
            ),
            dtype=float,
        ).reshape(-1)
        if (
            choice_compatibility.shape[0] != self.total_hypo
            or not np.all(np.isfinite(choice_compatibility))
            or np.any(choice_compatibility < 0.0)
            or np.any(choice_compatibility > 1.0)
        ):
            raise ValueError(
                "restored choice_compatibility must be a finite [0, 1] vector "
                "matching the hypothesis-space size."
            )
        self.choice_compatibility = choice_compatibility.copy()
        self.choice_compatibility_observations = int(
            state.get("choice_compatibility_observations", 0)
        )
        self.misconception_capture_hold_remaining = int(
            state.get("misconception_capture_hold_remaining", 0)
        )
        if (
            self.choice_compatibility_observations < 0
            or self.misconception_capture_hold_remaining < 0
        ):
            raise ValueError(
                "restored misconception-capture counters must be non-negative."
            )
        self.current_misconception_capture_eligible = bool(
            state.get("current_misconception_capture_eligible", False)
        )
        self.current_misconception_capture_switch_event = bool(
            state.get("current_misconception_capture_switch_event", False)
        )
        self.current_executed_choice_compatibility = float(
            state.get("current_executed_choice_compatibility", float("nan"))
        )
        self.current_best_alternative_choice_compatibility = float(
            state.get(
                "current_best_alternative_choice_compatibility",
                float("nan"),
            )
        )
        self.current_misconception_capture_advantage = float(
            state.get("current_misconception_capture_advantage", float("nan"))
        )
        saved_commitment_enabled = bool(
            state.get("rule_commitment_enabled", self.rule_commitment_enabled)
        )
        if saved_commitment_enabled != self.rule_commitment_enabled:
            raise ValueError(
                "rule-commitment state does not match configured enabled flag."
            )
        self.rule_commitment_active = bool(
            state.get("rule_commitment_active", False)
        )
        self.rule_commitment_age = int(state.get("rule_commitment_age", 0))
        self.rule_commitment_disconfirmation = float(
            state.get("rule_commitment_disconfirmation", 0.0)
        )
        self.rule_commitment_cooldown_remaining = int(
            state.get("rule_commitment_cooldown_remaining", 0)
        )
        self.current_rule_commitment_eligible = bool(
            state.get("current_rule_commitment_eligible", False)
        )
        self.current_rule_commitment_entry_event = bool(
            state.get("current_rule_commitment_entry_event", False)
        )
        self.current_rule_commitment_exit_event = bool(
            state.get("current_rule_commitment_exit_event", False)
        )
        candidate = state.get("current_rule_commitment_candidate")
        self.current_rule_commitment_candidate = (
            None if candidate is None else int(candidate)
        )
        self.current_rule_commitment_candidate_compatibility = float(
            state.get(
                "current_rule_commitment_candidate_compatibility",
                float("nan"),
            )
        )
        self.current_rule_commitment_runner_up_compatibility = float(
            state.get(
                "current_rule_commitment_runner_up_compatibility",
                float("nan"),
            )
        )
        self.current_rule_commitment_margin = float(
            state.get("current_rule_commitment_margin", float("nan"))
        )
        if (
            self.rule_commitment_age < 0
            or self.rule_commitment_disconfirmation < 0.0
            or self.rule_commitment_cooldown_remaining < 0
        ):
            raise ValueError(
                "restored rule-commitment counters must be non-negative."
            )
        if self.rule_commitment_active and self.executed_hypothesis is None:
            raise ValueError(
                "active rule commitment requires an executed_hypothesis."
            )
        execution_rng_state = state.get("execution_rng_state")
        if execution_rng_state is not None:
            self.execution_rng.bit_generator.state = deepcopy(execution_rng_state)
        if self.persistent_execution_enabled and self.executed_hypothesis not in set(
            self.active.tolist()
        ):
            raise ValueError(
                "restored executed_hypothesis must remain in the active workspace."
            )
        self._apply_mask()

    def clear_logs(self) -> None:
        self.transition_log.clear()
        self.transition_rate_log.clear()



__all__ = ["AdaptiveWorkspaceController"]
