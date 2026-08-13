"""Private bounded-workspace execution process used by public H modes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from .contracts import (
    HypothesisSelection,
    TransitionContext,
    TwoStepHypothesisTransitionMixin,
)


class WorkspaceTransitionExecutionMixin(TwoStepHypothesisTransitionMixin):
    """Two-step bounded-workspace lifecycle shared by public modes."""

    dynamic_controls = False

    def _transition_signals(self) -> Mapping[str, Any]:
        return {
            "controller_mode": str(self.controller_mode),
            "m_previous": float(self.current_m),
            "g_previous": float(self.current_g),
            "event_probability_previous": float(self.current_event_probability),
            "prior_reset_strength_previous": float(
                self.current_prior_reset_strength
            ),
            "failure_pressure_previous": float(self.failure_pressure),
            "mastery_evidence_previous": float(self.mastery_evidence),
            "peak_mastery_evidence_previous": float(
                self.peak_mastery_evidence
            ),
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
        search_slot_count = int(self.capacity)
        search_slot_rate = float(self.current_m)

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
            search_slot_rate = float(self.current_m)
            if self.persistent_execution_enabled:
                if self.executed_hypothesis not in set(active_before.tolist()):
                    raise RuntimeError(
                        "executed hypothesis left the active workspace before selection."
                    )
                search_slot_count = int(self.capacity - 1)
                search_slot_rate = self._event_probability_to_rate_for_slots(
                    self.current_event_probability,
                    search_slot_count,
                )
            replacement_count = int(
                self.trial_rng.binomial(search_slot_count, search_slot_rate)
            )

        commitment_target = self._prepare_rule_commitment()
        commitment_entry_requested = commitment_target is not None
        forced_newcomer = bool(
            commitment_entry_requested
            and int(commitment_target) not in set(active_before.tolist())
        )
        if forced_newcomer:
            replacement_count = max(1, replacement_count)

        capture_hold_active = bool(
            self.misconception_capture_enabled
            and self.misconception_capture_hold_remaining > 0
        )
        commitment_hold_active = bool(
            self.rule_commitment_active or commitment_entry_requested
        )
        ordinary_switch_allowed = bool(
            not capture_hold_active and not commitment_hold_active
        )
        switch_probability = (
            float(self.current_event_probability * self.execution_switch_scale)
            if self.persistent_execution_enabled
            and self.trial_index > 0
            and ordinary_switch_allowed
            else 0.0
        )
        switch_requested = bool(
            self.persistent_execution_enabled
            and replacement_count > 0
            and ordinary_switch_allowed
            and self.execution_rng.random() < self.execution_switch_scale
        )
        dropped = np.empty(0, dtype=int)
        newcomers = np.empty(0, dtype=int)
        active_after = active_before.copy()
        if replacement_count > 0:
            drop_pool = active_before
            if self.persistent_execution_enabled:
                protected = {int(self.executed_hypothesis)}
                if commitment_entry_requested and int(commitment_target) in set(
                    active_before.tolist()
                ):
                    protected.add(int(commitment_target))
                drop_pool = active_before[
                    ~np.isin(active_before, np.asarray(sorted(protected), dtype=int))
                ]
            replacement_count = min(replacement_count, int(drop_pool.size))
            drop_weights = 1.0 - posterior[drop_pool] + self.epsilon
            dropped = self._weighted_sample_without_replacement(
                drop_pool,
                drop_weights,
                replacement_count,
            )
            inactive = self.full_indices[~np.isin(self.full_indices, active_before)]
            remaining_count = int(replacement_count)
            newcomer_parts: list[np.ndarray] = []
            if forced_newcomer:
                forced = np.asarray([int(commitment_target)], dtype=int)
                newcomer_parts.append(forced)
                inactive = inactive[inactive != int(commitment_target)]
                remaining_count -= 1
            if remaining_count > 0:
                proposal = self._newcomer_proposal(posterior, inactive)
                newcomer_parts.append(
                    self._weighted_sample_without_replacement(
                        inactive,
                        proposal,
                        remaining_count,
                    )
                )
            newcomers = (
                np.concatenate(newcomer_parts).astype(int)
                if newcomer_parts
                else np.empty(0, dtype=int)
            )
            survivors = active_before[~np.isin(active_before, dropped)]
            active_after = np.sort(np.concatenate([survivors, newcomers]))

        if active_after.size != self.capacity or np.unique(active_after).size != self.capacity:
            raise RuntimeError("bounded-workspace transition violated fixed capacity.")
        if commitment_entry_requested and int(commitment_target) not in set(
            active_after.tolist()
        ):
            raise RuntimeError(
                "guided rule-commitment target was not installed in the workspace."
            )

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
            "execution_search_slot_count": int(search_slot_count),
            "execution_search_slot_rate": float(search_slot_rate),
            "execution_switch_probability": float(switch_probability),
            "execution_switch_requested": bool(switch_requested),
            "rule_commitment_entry_requested": bool(commitment_entry_requested),
            "rule_commitment_target": (
                None if commitment_target is None else int(commitment_target)
            ),
            "rule_commitment_forced_newcomer": bool(forced_newcomer),
            "misconception_capture_search_bias": bool(
                self._misconception_search_is_eligible()
            ),
            "misconception_capture_hold_before": int(
                self.misconception_capture_hold_remaining
            ),
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
        pairwise_prior = self._normalize(prior, "pairwise post-transition prior")

        reset_strength = 0.0
        reset_mass_shift = 0.0
        if selection.replacement_pairs and self.current_prior_reset_strength > 0.0:
            reset_strength = float(self.current_prior_reset_strength)
            broad_prior = np.zeros(self.total_hypo, dtype=float)
            broad_prior[selection.active_after] = self.base_prior[
                selection.active_after
            ]
            broad_prior = self._normalize(
                broad_prior,
                "active-set base prior for global reset",
            )
            prior = self._normalize(
                (1.0 - reset_strength) * pairwise_prior
                + reset_strength * broad_prior,
                "globally reset post-transition prior",
            )
            reset_mass_shift = float(0.5 * np.sum(np.abs(prior - pairwise_prior)))
        else:
            prior = pairwise_prior

        self._pending_transition["prior_reset_strength"] = reset_strength
        self._pending_transition["prior_reset_mass_shift"] = reset_mass_shift
        return prior

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
        prior_reset_strength = float(pending.get("prior_reset_strength", 0.0))
        prior_reset_mass_shift = float(pending.get("prior_reset_mass_shift", 0.0))
        previous_executed_hypothesis = self.executed_hypothesis
        execution_switched = False
        capture_switched = False
        capture_target = None
        current_choice_compatibility = float("nan")
        best_alternative_choice_compatibility = float("nan")
        capture_advantage = float("nan")
        capture_eligible = False
        commitment_entry_requested = bool(
            pending.get("rule_commitment_entry_requested", False)
        )
        commitment_target = pending.get("rule_commitment_target")
        if self.persistent_execution_enabled:
            if previous_executed_hypothesis not in set(selection.active_after.tolist()):
                raise RuntimeError(
                    "persistent execution failed to protect the executed hypothesis."
                )
            if commitment_entry_requested:
                if commitment_target is None or int(commitment_target) not in set(
                    selection.active_after.tolist()
                ):
                    raise RuntimeError(
                        "rule-commitment entry requires an active target hypothesis."
                    )
                self.executed_hypothesis = int(commitment_target)
                execution_switched = bool(
                    int(self.executed_hypothesis)
                    != int(previous_executed_hypothesis)
                )
                if execution_switched:
                    self.execution_switch_count += 1
                self.execution_dwell_trials = 1
                self.rule_commitment_active = True
                self.rule_commitment_age = 1
                self.rule_commitment_disconfirmation = 0.0
                self.current_rule_commitment_entry_event = True
                self.misconception_capture_hold_remaining = 0
            elif self.rule_commitment_active:
                self.execution_dwell_trials += 1
                self.rule_commitment_age += 1
            elif bool(pending.get("execution_switch_requested", False)):
                alternatives = selection.active_after[
                    selection.active_after != int(previous_executed_hypothesis)
                ]
                (
                    capture_target,
                    current_choice_compatibility,
                    best_alternative_choice_compatibility,
                    capture_advantage,
                    capture_eligible,
                ) = self._misconception_capture_target(alternatives)
                # Consume the ordinary-target draw on every overt switch.
                # Capture may override its value, but threshold variants then
                # retain a common future RNG stream for paired comparisons.
                alternative_weights = self._normalize(
                    prior[alternatives],
                    "executed-hypothesis switch weights",
                )
                ordinary_target = int(
                    self.execution_rng.choice(
                        alternatives,
                        p=alternative_weights,
                    )
                )
                if capture_eligible:
                    self.executed_hypothesis = int(capture_target)
                    capture_switched = True
                else:
                    self.executed_hypothesis = ordinary_target
                execution_switched = True
                self.execution_switch_count += 1
                self.execution_dwell_trials = 1
                self.misconception_capture_hold_remaining = (
                    self.misconception_min_dwell_trials - 1
                    if capture_switched
                    else 0
                )
            else:
                self.execution_dwell_trials += 1
                if self.misconception_capture_hold_remaining > 0:
                    self.misconception_capture_hold_remaining -= 1
                alternatives = selection.active_after[
                    selection.active_after != int(previous_executed_hypothesis)
                ]
                (
                    capture_target,
                    current_choice_compatibility,
                    best_alternative_choice_compatibility,
                    capture_advantage,
                    capture_eligible,
                ) = self._misconception_capture_target(alternatives)

            if self.rule_commitment_enabled:
                executed_index = int(self.executed_hypothesis)
                current_choice_compatibility = float(
                    self.choice_compatibility[executed_index]
                )
                other_indices = self.full_indices[
                    self.full_indices != executed_index
                ]
                best_alternative_choice_compatibility = float(
                    np.max(self.choice_compatibility[other_indices])
                )
            self.current_execution_switch_probability = float(
                self.rule_commitment_entry_probability
                if commitment_entry_requested and execution_switched
                else pending.get("execution_switch_probability", 0.0)
            )
            self.current_execution_switch_event = bool(execution_switched)
            self.current_misconception_capture_eligible = bool(capture_eligible)
            self.current_misconception_capture_switch_event = bool(capture_switched)
            self.current_executed_choice_compatibility = float(
                current_choice_compatibility
            )
            self.current_best_alternative_choice_compatibility = float(
                best_alternative_choice_compatibility
            )
            self.current_misconception_capture_advantage = float(capture_advantage)
        else:
            self.current_execution_switch_probability = 0.0
            self.current_execution_switch_event = False
            self.current_misconception_capture_eligible = False
            self.current_misconception_capture_switch_event = False

        newcomer_distance = self._newcomer_distance(
            posterior,
            selection.active_before,
            newcomers,
        )
        removed_mass = float(np.sum(posterior[dropped])) if dropped.size else 0.0
        self.predictive_prior = prior.copy()
        self._initialize_newcomer_beta(newcomers)

        # Trial 0 initializes the workspace and never executes the binomial
        # replacement draw.  Its logged event probability must therefore match
        # the realized transition policy rather than the later-trial formula.
        probability_any_replacement = (
            0.0
            if self.trial_index == 0
            else (
                float(self.current_event_probability)
                if self.persistent_execution_enabled
                else self._slot_rate_to_event_probability(self.current_m)
            )
        )
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
            "controller_mode": str(self.controller_mode),
            "failure_pressure": float(self.failure_pressure),
            "mastery_evidence": float(self.mastery_evidence),
            "peak_mastery_evidence": float(self.peak_mastery_evidence),
            "previous_feedback": float(self.previous_feedback),
            "exploration_target": float(self.exploration_target),
            "global_target": float(self.global_target),
            "prior_reset_max_strength": float(self.prior_reset_max_strength),
            "prior_reset_controller_strength": float(
                self.current_prior_reset_strength
            ),
            "prior_reset_strength": prior_reset_strength,
            "prior_reset_mass_shift": prior_reset_mass_shift,
            "prior_reset_applied": bool(prior_reset_strength > 0.0),
            "feedback_surprise": float(self.feedback_surprise),
            "feedback_uncertainty": float(self.feedback_uncertainty),
            "standardized_surprise": float(pending["z_surprise"]),
            "standardized_uncertainty": float(pending["z_uncertainty"]),
            "replacement_count": replacement_count,
            "replacement_fraction": float(replacement_count) / float(self.capacity),
            "persistent_execution_enabled": bool(
                self.persistent_execution_enabled
            ),
            "executed_hypothesis_previous": previous_executed_hypothesis,
            "executed_hypothesis": self.executed_hypothesis,
            "execution_switch_scale": float(self.execution_switch_scale),
            "execution_switch_probability": float(
                self.current_execution_switch_probability
            ),
            "execution_switch_event": bool(self.current_execution_switch_event),
            "execution_dwell_trials": int(self.execution_dwell_trials),
            "misconception_capture_enabled": bool(
                self.misconception_capture_enabled
            ),
            "misconception_capture_search_bias": bool(
                pending.get("misconception_capture_search_bias", False)
            ),
            "misconception_capture_eligible": bool(
                self.current_misconception_capture_eligible
            ),
            "misconception_capture_switch_event": bool(
                self.current_misconception_capture_switch_event
            ),
            "misconception_capture_hold_remaining": int(
                self.misconception_capture_hold_remaining
            ),
            "choice_compatibility_observations": int(
                self.choice_compatibility_observations
            ),
            "executed_choice_compatibility": float(
                self.current_executed_choice_compatibility
            ),
            "best_alternative_choice_compatibility": float(
                self.current_best_alternative_choice_compatibility
            ),
            "misconception_capture_advantage": float(
                self.current_misconception_capture_advantage
            ),
            "misconception_min_choice_compatibility": float(
                self.misconception_min_choice_compatibility
            ),
            "misconception_capture_target_hypothesis": capture_target,
            "rule_commitment_enabled": bool(self.rule_commitment_enabled),
            "rule_commitment_eligible": bool(
                self.current_rule_commitment_eligible
            ),
            "rule_commitment_active": bool(self.rule_commitment_active),
            "rule_commitment_entry_event": bool(
                self.current_rule_commitment_entry_event
            ),
            "rule_commitment_exit_event": bool(
                self.current_rule_commitment_exit_event
            ),
            "rule_commitment_age": int(self.rule_commitment_age),
            "rule_commitment_disconfirmation": float(
                self.rule_commitment_disconfirmation
            ),
            "rule_commitment_recovery_threshold": float(
                self.rule_commitment_recovery_threshold
            ),
            "rule_commitment_min_prior_mastery": float(
                self.rule_commitment_min_prior_mastery
            ),
            "rule_commitment_min_hold_choice_compatibility": float(
                self.rule_commitment_min_hold_choice_compatibility
            ),
            "rule_commitment_cooldown_remaining": int(
                self.rule_commitment_cooldown_remaining
            ),
            "rule_commitment_candidate_hypothesis": (
                None
                if self.current_rule_commitment_candidate is None
                else int(self.current_rule_commitment_candidate)
            ),
            "rule_commitment_candidate_compatibility": float(
                self.current_rule_commitment_candidate_compatibility
            ),
            "rule_commitment_runner_up_compatibility": float(
                self.current_rule_commitment_runner_up_compatibility
            ),
            "rule_commitment_margin": float(
                self.current_rule_commitment_margin
            ),
            "rule_commitment_forced_newcomer": bool(
                pending.get("rule_commitment_forced_newcomer", False)
            ),
            "execution_search_slot_count": int(
                pending.get("execution_search_slot_count", self.capacity)
            ),
            "execution_search_slot_rate": float(
                pending.get("execution_search_slot_rate", self.current_m)
            ),
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
                    "controller_mode",
                    "failure_pressure",
                    "mastery_evidence",
                    "peak_mastery_evidence",
                    "previous_feedback",
                    "exploration_target",
                    "global_target",
                    "prior_reset_controller_strength",
                    "prior_reset_strength",
                    "prior_reset_mass_shift",
                    "feedback_surprise",
                    "feedback_uncertainty",
                    "replacement_count",
                    "replacement_fraction",
                    "executed_hypothesis",
                    "execution_switch_probability",
                    "execution_switch_event",
                    "execution_dwell_trials",
                    "misconception_capture_search_bias",
                    "misconception_capture_eligible",
                    "misconception_capture_switch_event",
                    "misconception_capture_hold_remaining",
                    "choice_compatibility_observations",
                    "executed_choice_compatibility",
                    "best_alternative_choice_compatibility",
                    "misconception_capture_advantage",
                    "rule_commitment_eligible",
                    "rule_commitment_active",
                    "rule_commitment_entry_event",
                    "rule_commitment_exit_event",
                    "rule_commitment_age",
                    "rule_commitment_disconfirmation",
                    "rule_commitment_cooldown_remaining",
                    "rule_commitment_candidate_compatibility",
                    "rule_commitment_runner_up_compatibility",
                    "rule_commitment_margin",
                )
            }
        )
        self.trial_index += 1
        self._pending_transition = None
        return event


__all__ = ["WorkspaceTransitionExecutionMixin"]
