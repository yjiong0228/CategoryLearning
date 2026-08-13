"""Backend-neutral inference results produced before optimization metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass
class InferenceResult:
    """Common result contract shared by every inference backend.

    Backends place their outputs in five stable namespaces.  Consumers can use
    these mappings without importing a backend-specific result class:

    - ``observation_probabilities``: observable predictions keyed by timing;
    - ``state_probabilities``: hypothesis/active-state probability summaries;
    - ``latent_summaries``: transition and controller summaries;
    - ``diagnostics``: numerical or trial-level diagnostics;
    - ``artifacts``: optional live objects and heavyweight backend artifacts.

    The compatibility properties below preserve the historical trajectory and
    particle-filter attribute names while callers migrate to the common maps.
    """

    backend: str
    observation_probabilities: dict[str, Any] = field(default_factory=dict)
    state_probabilities: dict[str, Any] = field(default_factory=dict)
    latent_summaries: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_backend(self, expected: str) -> "InferenceResult":
        if self.backend != expected:
            raise TypeError(
                f"Expected inference backend {expected!r}, got {self.backend!r}."
            )
        return self

    # Trajectory compatibility properties.
    @property
    def model(self) -> Any:
        return self.artifacts.get("model")

    @property
    def posterior_log(self) -> Any:
        return self.state_probabilities.get("hypothesis_posterior")

    @property
    def prior_log(self) -> Any:
        return self.state_probabilities.get("hypothesis_prior")

    @property
    def step_log(self) -> Sequence[dict[str, Any]]:
        return self.diagnostics.get("step_log", ())

    @property
    def beta_log(self) -> Optional[Any]:
        return self.latent_summaries.get("beta")

    @property
    def transition_counts(self) -> Optional[Any]:
        return self.latent_summaries.get("transition_events")

    @property
    def latent_volatility_log(self) -> Optional[Any]:
        return self.latent_summaries.get("latent_volatility")

    @property
    def module_seed(self) -> Optional[int]:
        value = self.metadata.get("module_seed")
        return None if value is None else int(value)

    # Particle-filter compatibility properties.
    @property
    def marginal_probabilities(self) -> Any:
        return self.observation_probabilities.get("prior_t")

    @property
    def marginal_hypothesis_prior(self) -> Any:
        return self.state_probabilities.get("hypothesis_prior")

    @property
    def marginal_active_probability(self) -> Any:
        return self.state_probabilities.get("active_probability")

    @property
    def pre_choice_ess(self) -> Any:
        return self.diagnostics.get("pre_choice_ess")

    @property
    def post_choice_ess(self) -> Any:
        return self.diagnostics.get("post_choice_ess")

    @property
    def resampled(self) -> Any:
        return self.diagnostics.get("resampled")

    @property
    def resampling_unique_ancestors(self) -> Any:
        return self.diagnostics.get("resampling_unique_ancestors")

    @property
    def filtered_swap_probability(self) -> Any:
        return self.latent_summaries.get("swap_probability")

    @property
    def filtered_swap_event_probability(self) -> Any:
        return self.latent_summaries.get("swap_event_probability")

    @property
    def filtered_transition_rate(self) -> Any:
        return self.latent_summaries.get("transition_rate")

    @property
    def filtered_search_range(self) -> Any:
        return self.latent_summaries.get("search_range")

    @property
    def filtered_replacement_count(self) -> Any:
        return self.latent_summaries.get("replacement_count")

    @property
    def filtered_replacement_fraction(self) -> Any:
        return self.latent_summaries.get("replacement_fraction")

    @property
    def filtered_removed_mass(self) -> Any:
        return self.latent_summaries.get("removed_mass")

    @property
    def filtered_newcomer_distance(self) -> Any:
        return self.latent_summaries.get("newcomer_distance")

    @property
    def filtered_feedback_surprise(self) -> Any:
        return self.latent_summaries.get("feedback_surprise")

    @property
    def filtered_feedback_uncertainty(self) -> Any:
        return self.latent_summaries.get("feedback_uncertainty")

    @property
    def predictive_swap_probability(self) -> Any:
        return self.latent_summaries.get("predictive_swap_probability")

    @property
    def predictive_swap_event_probability(self) -> Any:
        return self.latent_summaries.get("predictive_swap_event_probability")

    @property
    def predictive_transition_rate(self) -> Any:
        return self.latent_summaries.get("predictive_transition_rate")

    @property
    def predictive_search_range(self) -> Any:
        return self.latent_summaries.get("predictive_search_range")

    @property
    def predictive_replacement_fraction(self) -> Any:
        return self.latent_summaries.get("predictive_replacement_fraction")

    @property
    def predictive_newcomer_distance(self) -> Any:
        return self.latent_summaries.get("predictive_newcomer_distance")

    @property
    def predictive_strategy_exploit(self) -> Any:
        return self.latent_summaries.get("predictive_strategy_exploit")

    @property
    def predictive_strategy_local_explore(self) -> Any:
        return self.latent_summaries.get("predictive_strategy_local_explore")

    @property
    def predictive_strategy_global_explore(self) -> Any:
        return self.latent_summaries.get("predictive_strategy_global_explore")

    @property
    def final_weights(self) -> Any:
        return self.artifacts.get("final_weights")

    @property
    def particle_swap_counts(self) -> Any:
        return self.artifacts.get("particle_swap_counts")

    @property
    def resampling_log(self) -> Any:
        return self.artifacts.get("resampling_log")

    @property
    def particle_count(self) -> Optional[int]:
        value = self.metadata.get("particle_count")
        return None if value is None else int(value)

    @property
    def resample_threshold_fraction(self) -> Optional[float]:
        value = self.metadata.get("resample_threshold_fraction")
        return None if value is None else float(value)

    @property
    def filter_seed(self) -> Optional[int]:
        value = self.metadata.get("filter_seed")
        return None if value is None else int(value)


class TrajectoryInferenceResult(InferenceResult):
    """Compatibility constructor for one realized latent trajectory."""

    def __init__(
        self,
        *,
        model: Any,
        posterior_log: Any,
        prior_log: Any,
        step_log: Sequence[dict[str, Any]],
        beta_log: Optional[Any],
        transition_counts: Optional[Any],
        latent_volatility_log: Optional[Any],
        module_seed: Optional[int],
    ) -> None:
        super().__init__(
            backend="trajectory",
            state_probabilities={
                "hypothesis_prior": prior_log,
                "hypothesis_posterior": posterior_log,
            },
            latent_summaries={
                "beta": beta_log,
                "transition_events": transition_counts,
                "latent_volatility": latent_volatility_log,
            },
            diagnostics={"step_log": step_log},
            artifacts={"model": model},
            metadata={"module_seed": module_seed},
        )


class ParticleFilterResult(InferenceResult):
    """Compatibility constructor for bootstrap particle-filter summaries."""

    def __init__(
        self,
        *,
        marginal_probabilities: Any,
        marginal_hypothesis_prior: Any,
        marginal_active_probability: Any,
        pre_choice_ess: Any,
        post_choice_ess: Any,
        resampled: Any,
        resampling_unique_ancestors: Any,
        filtered_swap_probability: Any,
        filtered_swap_event_probability: Any,
        filtered_transition_rate: Any,
        filtered_replacement_count: Any,
        filtered_replacement_fraction: Any,
        filtered_removed_mass: Any,
        filtered_newcomer_distance: Any,
        filtered_feedback_surprise: Any,
        filtered_feedback_uncertainty: Any,
        final_weights: Any,
        particle_swap_counts: Any,
        resampling_log: Any,
        particle_count: int,
        resample_threshold_fraction: float,
        filter_seed: int,
        filtered_search_range: Any = None,
        predictive_swap_probability: Any = None,
        predictive_swap_event_probability: Any = None,
        predictive_transition_rate: Any = None,
        predictive_search_range: Any = None,
        predictive_replacement_fraction: Any = None,
        predictive_newcomer_distance: Any = None,
        predictive_strategy_exploit: Any = None,
        predictive_strategy_local_explore: Any = None,
        predictive_strategy_global_explore: Any = None,
        predictive_failure_pressure: Any = None,
        predictive_mastery_evidence: Any = None,
        predictive_peak_mastery_evidence: Any = None,
        predictive_choice_confidence_signal: Any = None,
        predictive_strategy_choice_precision: Any = None,
        predictive_exploration_target: Any = None,
        predictive_global_target: Any = None,
        predictive_prior_reset_strength: Any = None,
        predictive_prior_reset_mass_shift: Any = None,
        audit_hypothesis_map: Any = None,
        audit_adaptive_sharpening: Any = None,
        audit_exploration_lapse: Any = None,
        audit_unsharpened_expectation: Any = None,
        audit_sharpened_no_lapse: Any = None,
        audit_strategy_confidence_no_lapse: Any = None,
        audit_correct_predicting_available_probability: Any = None,
        audit_correct_predicting_prior_mass: Any = None,
        audit_best_active_correct_probability: Any = None,
        audit_particle_correct_q10: Any = None,
        audit_particle_correct_q50: Any = None,
        audit_particle_correct_q90: Any = None,
        audit_ancestral_paths: Mapping[str, Any] | None = None,
        marginal_executed_probability: Any = None,
        filtered_executed_probability: Any = None,
        predictive_execution_switch_probability: Any = None,
        predictive_execution_switch_event_probability: Any = None,
        predictive_execution_dwell_trials: Any = None,
        predictive_misconception_capture_eligible_probability: Any = None,
        predictive_misconception_capture_hold_probability: Any = None,
        predictive_misconception_capture_switch_event_probability: Any = None,
        predictive_rule_commitment_probability: Any = None,
        predictive_rule_commitment_eligible_probability: Any = None,
        predictive_rule_commitment_entry_event_probability: Any = None,
        predictive_rule_commitment_exit_event_probability: Any = None,
        predictive_rule_commitment_age: Any = None,
        predictive_rule_commitment_disconfirmation: Any = None,
        predictive_rule_commitment_margin: Any = None,
        predictive_rule_commitment_confidence_signal: Any = None,
        predictive_rule_commitment_choice_precision: Any = None,
        predictive_executed_choice_compatibility: Any = None,
        predictive_best_alternative_choice_compatibility: Any = None,
        predictive_executed_beta: Any = None,
        filtered_executed_beta: Any = None,
        filtered_execution_switch_event_probability: Any = None,
        filtered_execution_dwell_trials: Any = None,
        audit_persistent_execution_no_lapse: Any = None,
    ) -> None:
        observation_probabilities = {"prior_t": marginal_probabilities}
        for key, value in (
            ("audit_hypothesis_map", audit_hypothesis_map),
            ("audit_adaptive_sharpening", audit_adaptive_sharpening),
            ("audit_exploration_lapse", audit_exploration_lapse),
            ("audit_unsharpened_expectation", audit_unsharpened_expectation),
            ("audit_sharpened_no_lapse", audit_sharpened_no_lapse),
            (
                "audit_strategy_confidence_no_lapse",
                audit_strategy_confidence_no_lapse,
            ),
            (
                "audit_persistent_execution_no_lapse",
                audit_persistent_execution_no_lapse,
            ),
        ):
            if value is not None:
                observation_probabilities[key] = value
        audit_diagnostics = {}
        for key, value in (
            ("audit_particle_correct_q10", audit_particle_correct_q10),
            ("audit_particle_correct_q50", audit_particle_correct_q50),
            ("audit_particle_correct_q90", audit_particle_correct_q90),
            (
                "audit_correct_predicting_available_probability",
                audit_correct_predicting_available_probability,
            ),
            (
                "audit_correct_predicting_prior_mass",
                audit_correct_predicting_prior_mass,
            ),
            (
                "audit_best_active_correct_probability",
                audit_best_active_correct_probability,
            ),
        ):
            if value is not None:
                audit_diagnostics[key] = value
        super().__init__(
            backend="particle_filter",
            observation_probabilities=observation_probabilities,
            state_probabilities={
                "hypothesis_prior": marginal_hypothesis_prior,
                "active_probability": marginal_active_probability,
                **(
                    {"executed_probability": marginal_executed_probability}
                    if marginal_executed_probability is not None
                    else {}
                ),
                **(
                    {
                        "filtered_executed_probability": (
                            filtered_executed_probability
                        )
                    }
                    if filtered_executed_probability is not None
                    else {}
                ),
            },
            latent_summaries={
                "swap_probability": filtered_swap_probability,
                "swap_event_probability": filtered_swap_event_probability,
                "transition_rate": filtered_transition_rate,
                "search_range": filtered_search_range,
                "replacement_count": filtered_replacement_count,
                "replacement_fraction": filtered_replacement_fraction,
                "removed_mass": filtered_removed_mass,
                "newcomer_distance": filtered_newcomer_distance,
                "feedback_surprise": filtered_feedback_surprise,
                "feedback_uncertainty": filtered_feedback_uncertainty,
                "predictive_swap_probability": predictive_swap_probability,
                "predictive_swap_event_probability": predictive_swap_event_probability,
                "predictive_transition_rate": predictive_transition_rate,
                "predictive_search_range": predictive_search_range,
                "predictive_replacement_fraction": predictive_replacement_fraction,
                "predictive_newcomer_distance": predictive_newcomer_distance,
                "predictive_strategy_exploit": predictive_strategy_exploit,
                "predictive_strategy_local_explore": predictive_strategy_local_explore,
                "predictive_strategy_global_explore": predictive_strategy_global_explore,
                "predictive_failure_pressure": predictive_failure_pressure,
                "predictive_mastery_evidence": predictive_mastery_evidence,
                "predictive_peak_mastery_evidence": (
                    predictive_peak_mastery_evidence
                ),
                "predictive_choice_confidence_signal": (
                    predictive_choice_confidence_signal
                ),
                "predictive_strategy_choice_precision": (
                    predictive_strategy_choice_precision
                ),
                "predictive_exploration_target": predictive_exploration_target,
                "predictive_global_target": predictive_global_target,
                "predictive_prior_reset_strength": predictive_prior_reset_strength,
                "predictive_prior_reset_mass_shift": predictive_prior_reset_mass_shift,
                "predictive_execution_switch_probability": (
                    predictive_execution_switch_probability
                ),
                "predictive_execution_switch_event_probability": (
                    predictive_execution_switch_event_probability
                ),
                "predictive_execution_dwell_trials": (
                    predictive_execution_dwell_trials
                ),
                "predictive_misconception_capture_eligible_probability": (
                    predictive_misconception_capture_eligible_probability
                ),
                "predictive_misconception_capture_hold_probability": (
                    predictive_misconception_capture_hold_probability
                ),
                "predictive_misconception_capture_switch_event_probability": (
                    predictive_misconception_capture_switch_event_probability
                ),
                "predictive_rule_commitment_probability": (
                    predictive_rule_commitment_probability
                ),
                "predictive_rule_commitment_eligible_probability": (
                    predictive_rule_commitment_eligible_probability
                ),
                "predictive_rule_commitment_entry_event_probability": (
                    predictive_rule_commitment_entry_event_probability
                ),
                "predictive_rule_commitment_exit_event_probability": (
                    predictive_rule_commitment_exit_event_probability
                ),
                "predictive_rule_commitment_age": (
                    predictive_rule_commitment_age
                ),
                "predictive_rule_commitment_disconfirmation": (
                    predictive_rule_commitment_disconfirmation
                ),
                "predictive_rule_commitment_margin": (
                    predictive_rule_commitment_margin
                ),
                "predictive_rule_commitment_confidence_signal": (
                    predictive_rule_commitment_confidence_signal
                ),
                "predictive_rule_commitment_choice_precision": (
                    predictive_rule_commitment_choice_precision
                ),
                "predictive_executed_choice_compatibility": (
                    predictive_executed_choice_compatibility
                ),
                "predictive_best_alternative_choice_compatibility": (
                    predictive_best_alternative_choice_compatibility
                ),
                "predictive_executed_beta": predictive_executed_beta,
                "filtered_executed_beta": filtered_executed_beta,
                "execution_switch_event_probability": (
                    filtered_execution_switch_event_probability
                ),
                "execution_dwell_trials": filtered_execution_dwell_trials,
            },
            diagnostics={
                "pre_choice_ess": pre_choice_ess,
                "post_choice_ess": post_choice_ess,
                "resampled": resampled,
                "resampling_unique_ancestors": resampling_unique_ancestors,
                **audit_diagnostics,
            },
            artifacts={
                "final_weights": final_weights,
                "particle_swap_counts": particle_swap_counts,
                "resampling_log": resampling_log,
                **(
                    {"audit_ancestral_paths": dict(audit_ancestral_paths)}
                    if audit_ancestral_paths is not None
                    else {}
                ),
            },
            metadata={
                "particle_count": int(particle_count),
                "resample_threshold_fraction": float(resample_threshold_fraction),
                "filter_seed": int(filter_seed),
            },
        )


def ensure_inference_result(
    result: InferenceResult,
    *,
    backend: str | None = None,
) -> InferenceResult:
    """Validate the public result contract at a backend boundary."""

    if not isinstance(result, InferenceResult):
        raise TypeError(
            "Inference backends must return InferenceResult, "
            f"got {type(result).__name__}."
        )
    if backend is not None:
        result.require_backend(backend)
    if not isinstance(result.observation_probabilities, Mapping):
        raise TypeError("InferenceResult.observation_probabilities must be a mapping.")
    if not isinstance(result.state_probabilities, Mapping):
        raise TypeError("InferenceResult.state_probabilities must be a mapping.")
    return result


__all__ = [
    "InferenceResult",
    "ParticleFilterResult",
    "TrajectoryInferenceResult",
    "ensure_inference_result",
]
