"""Trial-varying continuous-control hypothesis-transition strategies.

The selection and prior-assignment mechanisms remain fixed for a subject.
Explicit continuous cognitive controls, currently replacement rate ``m_t`` and
search range ``g_t``, evolve across trials and modulate those mechanisms.
"""

from __future__ import annotations

from typing import Any, Mapping

from .execution import WorkspaceTransitionExecutionMixin
from .workspace import AdaptiveWorkspaceController


class DynamicAdaptiveControlHypothesisTransitionModule(
    WorkspaceTransitionExecutionMixin,
    AdaptiveWorkspaceController,
):
    """Bounded-workspace policy driven by trial-varying ``m_t``/``g_t``."""

    strategy_mode = "dynamic_continuous"
    dynamic_controls = True

    def __init__(self, engine, **kwargs):
        resolved = dict(kwargs)
        continuous = resolved.pop("continuous_controller", None)
        if continuous is not None:
            if not isinstance(continuous, Mapping):
                raise ValueError("continuous_controller must be a mapping.")
            mode = str(continuous.get("mode", "legacy"))
            if mode == "legacy":
                unknown = set(continuous) - {"mode", "rate", "range"}
            elif mode == "failure_accumulator_v2":
                unknown = set(continuous) - {
                    "mode",
                    "state",
                    "exploration",
                    "range",
                    "prior_reset",
                    "execution",
                }
            else:
                raise ValueError(
                    "continuous_controller.mode must be 'legacy' or "
                    f"'failure_accumulator_v2', got {mode!r}."
                )
            if unknown:
                raise ValueError(
                    f"continuous_controller mode {mode!r} has unsupported keys; "
                    f"got {sorted(unknown)}."
                )
            if mode == "legacy":
                if "rate" in continuous:
                    if "rate_controller" in resolved:
                        raise ValueError(
                            "Configure continuous_controller.rate or rate_controller, not both."
                        )
                    resolved["rate_controller"] = continuous["rate"]
                if "range" in continuous:
                    if "range_controller" in resolved:
                        raise ValueError(
                            "Configure continuous_controller.range or range_controller, not both."
                        )
                    resolved["range_controller"] = continuous["range"]
            else:
                legacy_controller_keys = {
                    "rate_controller",
                    "range_controller",
                    "m_phi",
                    "m_beta_surprise",
                    "m_beta_uncertainty",
                    "g_phi",
                    "g_beta_surprise",
                    "g_beta_uncertainty",
                }
                configured_legacy = sorted(legacy_controller_keys.intersection(resolved))
                if configured_legacy:
                    raise ValueError(
                        "failure_accumulator_v2 cannot be combined with legacy "
                        f"controller settings: {configured_legacy}."
                    )
                resolved["controller_mode"] = mode
                resolved["failure_accumulator_controller"] = {
                    "state": continuous.get("state", {}),
                    "exploration": continuous.get("exploration", {}),
                    "range": continuous.get("range", {}),
                    "prior_reset": continuous.get("prior_reset", {}),
                }
                resolved["persistent_execution"] = continuous.get("execution", {})

        prior_spec = resolved.get("prior_assignment")
        if prior_spec is not None:
            if not isinstance(prior_spec, Mapping):
                raise ValueError("prior_assignment must be a mapping.")
            if (
                str(prior_spec.get("method", ""))
                not in self.VALID_PRIOR_ASSIGNMENTS
            ):
                raise ValueError(
                    "Continuous bounded-workspace control requires a supported "
                    "bounded-workspace prior_assignment method."
                )

        selection_spec = resolved.pop("selection_strategy", None)
        if selection_spec is not None:
            if not isinstance(selection_spec, Mapping):
                raise ValueError("selection_strategy must be a mapping.")
            if str(selection_spec.get("method", "")) != "bounded_workspace":
                raise ValueError(
                    "DynamicAdaptiveControlHypothesisTransitionModule requires "
                    "selection_strategy.method='bounded_workspace'."
                )
            for key, value in selection_spec.items():
                if key != "method":
                    resolved[key] = value

        super().__init__(engine, **resolved)
        if not (self.dynamic_rate or self.dynamic_range):
            raise ValueError(
                "DynamicAdaptiveControlHypothesisTransitionModule requires at least "
                "one trial-varying control. Set a non-zero m_beta_* or g_beta_* "
                "coefficient, or use FixedWorkspaceHypothesisTransitionModule."
            )
        self._pending_transition: dict[str, Any] | None = None


__all__ = ["DynamicAdaptiveControlHypothesisTransitionModule"]
