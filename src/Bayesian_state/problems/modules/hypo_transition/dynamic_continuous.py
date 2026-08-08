"""Trial-varying continuous-control hypothesis-transition strategies.

The selection and prior-assignment mechanisms remain fixed for a subject.
Explicit continuous cognitive controls, currently replacement rate ``m_t`` and
search range ``g_t``, evolve across trials and modulate those mechanisms.
"""

from __future__ import annotations

from typing import Any, Mapping

from ._internal.workspace_policy import (
    BoundedWorkspacePolicyRuntime,
    BoundedWorkspaceTransitionMixin,
)


class DynamicContinuousHypothesisTransitionModule(
    BoundedWorkspaceTransitionMixin,
    BoundedWorkspacePolicyRuntime,
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
            unknown = set(continuous) - {"rate", "range"}
            if unknown:
                raise ValueError(
                    "continuous_controller supports only 'rate' and 'range'; "
                    f"got {sorted(unknown)}."
                )
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

        prior_spec = resolved.pop("prior_assignment", None)
        if prior_spec is not None:
            if not isinstance(prior_spec, Mapping):
                raise ValueError("prior_assignment must be a mapping.")
            if str(prior_spec.get("method", "")) != "pairwise_mass_transfer":
                raise ValueError(
                    "Continuous bounded-workspace control currently supports only "
                    "prior_assignment.method='pairwise_mass_transfer'."
                )

        selection_spec = resolved.pop("selection_strategy", None)
        if selection_spec is not None:
            if not isinstance(selection_spec, Mapping):
                raise ValueError("selection_strategy must be a mapping.")
            if str(selection_spec.get("method", "")) != "bounded_workspace":
                raise ValueError(
                    "DynamicContinuousHypothesisTransitionModule requires "
                    "selection_strategy.method='bounded_workspace'."
                )
            for key, value in selection_spec.items():
                if key != "method":
                    resolved[key] = value

        super().__init__(engine, **resolved)
        if not (self.dynamic_rate or self.dynamic_range):
            raise ValueError(
                "DynamicContinuousHypothesisTransitionModule requires at least "
                "one trial-varying control. Set a non-zero m_beta_* or g_beta_* "
                "coefficient, or use StaticWorkspaceHypothesisTransitionModule."
            )
        self._pending_transition: dict[str, Any] | None = None


__all__ = ["DynamicContinuousHypothesisTransitionModule"]
