"""Private bounded-workspace policy implementation used by public H modes.

The runtime stores workspace geometry and controller state.  The mixin below
implements the same public two-step contract used by every H mode.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ...base_module import BaseModule
from ..process import (
    HypothesisSelection,
    TransitionContext,
    TwoStepHypothesisTransitionMixin,
)


class BoundedWorkspacePolicyRuntime(BaseModule):
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
        tau_local = kwargs.get("tau_local")
        self.tau_local = (
            None if tau_local is None else self._validate_positive(tau_local, "tau_local")
        )
        self.epsilon = self._validate_positive(kwargs.get("epsilon", 1e-12), "epsilon")

        self.module_seed = kwargs.get("module_seed", kwargs.get("random_seed"))
        if self.module_seed is not None:
            self.module_seed = int(self.module_seed)
        seed_sequence = np.random.SeedSequence(self.module_seed)
        init_seed, trial_seed = seed_sequence.spawn(2)
        self.init_rng = np.random.default_rng(init_seed)
        self.trial_rng = np.random.default_rng(trial_seed)

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
        self.trial_index = 0

        self.baseline_logit = self._logit(self.m) if 0.0 < self.m < 1.0 else 0.0
        self.control_logit = float(self.baseline_logit)
        self.current_m = float(self.m)
        self.g_baseline_logit = self._logit(self.g) if 0.0 < self.g < 1.0 else 0.0
        self.g_control_logit = float(self.g_baseline_logit)
        self.current_g = float(self.g)
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
            return global_weights
        self._ensure_geometry()
        assert self._local_kernel is not None
        local_full = np.asarray(posterior @ self._local_kernel, dtype=float)
        local = self._normalize(local_full[inactive], "inactive local proposal")
        return self._normalize(
            (1.0 - self.current_g) * local + self.current_g * global_weights,
            "newcomer mixture",
        )

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
        beta_mod = getattr(self.engine, "modules", {}).get("beta_mod")
        if beta_mod is not None and hasattr(beta_mod, "initialize_beta_for_hypotheses"):
            beta_mod.initialize_beta_for_hypotheses(newcomers, priors=None)

    def reseed_future(self, module_seed: int) -> None:
        self.module_seed = int(module_seed)
        self.trial_rng = np.random.default_rng(self.module_seed)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active.copy(),
            "old_active": self.old_active.copy(),
            "trial_index": int(self.trial_index),
            "control_logit": float(self.control_logit),
            "current_m": float(self.current_m),
            "g_control_logit": float(self.g_control_logit),
            "current_g": float(self.current_g),
            "predictive_prior": (
                None if self.predictive_prior is None else self.predictive_prior.copy()
            ),
            "feedback_surprise": float(self.feedback_surprise),
            "feedback_uncertainty": float(self.feedback_uncertainty),
            "trial_rng_state": deepcopy(self.trial_rng.bit_generator.state),
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
        self._apply_mask()

    def clear_logs(self) -> None:
        self.transition_log.clear()
        self.transition_rate_log.clear()


class BoundedWorkspaceTransitionMixin(TwoStepHypothesisTransitionMixin):
    """Two-step bounded-workspace lifecycle shared by public modes."""

    dynamic_controls = False

    def _transition_signals(self) -> Mapping[str, Any]:
        return {
            "m_previous": float(self.current_m),
            "g_previous": float(self.current_g),
            "feedback_surprise_previous": float(self.feedback_surprise),
            "feedback_uncertainty_previous": float(self.feedback_uncertainty),
        }

    def select_hypotheses(
        self,
        context: TransitionContext,
        **kwargs,
    ) -> HypothesisSelection:
        del kwargs
        active_before = self.active.copy()
        posterior = self._posterior_for_transition()

        if self.trial_index == 0:
            z_surprise = float("nan")
            z_uncertainty = float("nan")
            replacement_count = 0
        else:
            if self.dynamic_controls:
                z_surprise, z_uncertainty = self._update_transition_controls()
            else:
                z_surprise = float("nan")
                z_uncertainty = float("nan")
            replacement_count = int(
                self.trial_rng.binomial(self.capacity, self.current_m)
            )

        dropped = np.empty(0, dtype=int)
        newcomers = np.empty(0, dtype=int)
        active_after = active_before.copy()
        if replacement_count > 0:
            drop_weights = 1.0 - posterior[active_before] + self.epsilon
            dropped = self._weighted_sample_without_replacement(
                active_before,
                drop_weights,
                replacement_count,
            )
            inactive = self.full_indices[~np.isin(self.full_indices, active_before)]
            proposal = self._newcomer_proposal(posterior, inactive)
            newcomers = self._weighted_sample_without_replacement(
                inactive,
                proposal,
                replacement_count,
            )
            survivors = active_before[~np.isin(active_before, dropped)]
            active_after = np.sort(np.concatenate([survivors, newcomers]))

        if active_after.size != self.capacity or np.unique(active_after).size != self.capacity:
            raise RuntimeError("bounded-workspace transition violated fixed capacity.")

        replacement_pairs = tuple(
            (int(dropped_hypothesis), int(newcomer))
            for dropped_hypothesis, newcomer in zip(dropped, newcomers)
        )
        self._pending_transition = {
            "posterior": posterior,
            "z_surprise": float(z_surprise),
            "z_uncertainty": float(z_uncertainty),
            "replacement_count": int(replacement_count),
            "dropped": dropped,
            "newcomers": newcomers,
        }
        return HypothesisSelection.from_active_sets(
            context.active_before,
            active_after,
            replacement_pairs=replacement_pairs,
            diagnostics={
                "strategy_mode": self.strategy_mode,
                "predictive_m": float(self.current_m),
                "predictive_g": float(self.current_g),
            },
        )

    def assign_prior(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        **kwargs,
    ) -> np.ndarray:
        del context, kwargs
        if self._pending_transition is None:
            raise RuntimeError("bounded-workspace transition has no pending selection.")
        posterior = np.asarray(self._pending_transition["posterior"], dtype=float)
        prior = posterior.copy()
        for dropped_hypothesis, newcomer in selection.replacement_pairs:
            prior[newcomer] = prior[dropped_hypothesis]
            prior[dropped_hypothesis] = 0.0
        return self._normalize(prior, "post-transition prior")

    def _finish_hypothesis_transition(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        prior: np.ndarray,
        **kwargs,
    ) -> Mapping[str, Any]:
        del kwargs
        if self._pending_transition is None:
            raise RuntimeError("bounded-workspace transition has no pending state.")
        pending = self._pending_transition
        posterior = np.asarray(pending["posterior"], dtype=float)
        dropped = np.asarray(pending["dropped"], dtype=int)
        newcomers = np.asarray(pending["newcomers"], dtype=int)
        replacement_count = int(pending["replacement_count"])

        newcomer_distance = self._newcomer_distance(
            posterior,
            selection.active_before,
            newcomers,
        )
        removed_mass = float(np.sum(posterior[dropped])) if dropped.size else 0.0
        self.predictive_prior = prior.copy()
        self._initialize_newcomer_beta(newcomers)

        probability_any_replacement = 1.0 - (1.0 - self.current_m) ** self.capacity
        event: Dict[str, Any] = {
            "trial_index": int(context.trial_index),
            "strategy_mode": self.strategy_mode,
            "transition_method": (
                "adaptive_binomial_replacement"
                if self.dynamic_controls
                else "fixed_binomial_replacement"
            ),
            "m": float(self.m),
            "predictive_m": float(self.current_m),
            "control_logit": float(self.control_logit),
            "g": float(self.g),
            "predictive_g": float(self.current_g),
            "g_control_logit": float(self.g_control_logit),
            "feedback_surprise": float(self.feedback_surprise),
            "feedback_uncertainty": float(self.feedback_uncertainty),
            "standardized_surprise": float(pending["z_surprise"]),
            "standardized_uncertainty": float(pending["z_uncertainty"]),
            "replacement_count": replacement_count,
            "replacement_fraction": float(replacement_count) / float(self.capacity),
            "removed_mass": removed_mass,
            "newcomer_distance": float(newcomer_distance),
            "dropped_hypotheses": dropped.astype(int).tolist(),
            "new_hypotheses": newcomers.astype(int).tolist(),
            "active_before": selection.active_before.astype(int).tolist(),
            "active_after": selection.active_after.astype(int).tolist(),
            "active_total": int(selection.active_after.size),
            "prior_sum": float(np.sum(prior)),
            "swap_probability": float(probability_any_replacement),
            "swap_event": bool(replacement_count > 0),
            "strategies": [],
        }
        self.transition_log.append(event)
        self.transition_rate_log.append(
            {
                key: event[key]
                for key in (
                    "trial_index",
                    "predictive_m",
                    "control_logit",
                    "predictive_g",
                    "g_control_logit",
                    "feedback_surprise",
                    "feedback_uncertainty",
                    "replacement_count",
                    "replacement_fraction",
                )
            }
        )
        self.trial_index += 1
        self._pending_transition = None
        return event


__all__ = ["BoundedWorkspacePolicyRuntime", "BoundedWorkspaceTransitionMixin"]
