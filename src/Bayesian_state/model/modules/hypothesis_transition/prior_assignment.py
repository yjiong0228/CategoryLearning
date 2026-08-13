"""Private strategy-policy implementation shared by public H modes.

This is an internal strategy library, not a complete H module. Public modes
provide the lifecycle in ``fixed_strategy`` and ``dynamic_discrete_strategy``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..base_module import ModuleRole


class PriorAssignmentPolicyMixin:
    """Assign and optionally reset prior mass after hypothesis selection."""

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
        beta_mod = self.engine.get_module(ModuleRole.BETA)
        if beta_mod is not None and hasattr(beta_mod, "initialize_beta_for_hypotheses"):
            beta_mod.initialize_beta_for_hypotheses(newcomer_indices, prior)


__all__ = ["PriorAssignmentPolicyMixin"]
