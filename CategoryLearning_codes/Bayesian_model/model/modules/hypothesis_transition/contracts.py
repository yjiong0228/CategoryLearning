"""Common two-step process for hypothesis transitions.

A hypothesis transition is deliberately defined as two cognitive operations:

1. select the hypotheses that will be active on the next trial;
2. assign a prior distribution to that newly selected active set.

Static, discrete-state, and continuous-control models differ in how they
produce the policies or control state used by these operations.  They do not
reimplement the surrounding transition lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..base_module import ModulePhase, ModuleRole


@dataclass(frozen=True)
class TransitionContext:
    """Information available before the current trial's transition.

    ``posterior`` is normally the posterior produced by the preceding trial.
    Dynamic implementations may add causal history-derived values to
    ``signals``; current-trial feedback must not be inserted there before the
    transition has run.
    """

    trial_index: int
    posterior: np.ndarray
    active_before: np.ndarray
    signals: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HypothesisSelection:
    """Result of step 1: selecting the next active hypothesis set."""

    active_before: np.ndarray
    active_after: np.ndarray
    survivors: np.ndarray
    dropped: np.ndarray
    newcomers: np.ndarray
    replacement_pairs: tuple[tuple[int, int], ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_active_sets(
        cls,
        active_before: Sequence[int] | np.ndarray,
        active_after: Sequence[int] | np.ndarray,
        *,
        replacement_pairs: Sequence[tuple[int, int]] = (),
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "HypothesisSelection":
        """Build the survivor/drop/newcomer partition from two active sets."""

        before = np.asarray(active_before, dtype=int).reshape(-1).copy()
        after = np.asarray(active_after, dtype=int).reshape(-1).copy()
        survivors = after[np.isin(after, before)]
        dropped = before[~np.isin(before, after)]
        newcomers = after[~np.isin(after, before)]
        return cls(
            active_before=before,
            active_after=after,
            survivors=survivors,
            dropped=dropped,
            newcomers=newcomers,
            replacement_pairs=tuple(
                (int(dropped_hypothesis), int(newcomer))
                for dropped_hypothesis, newcomer in replacement_pairs
            ),
            diagnostics=dict(diagnostics or {}),
        )


@dataclass(frozen=True)
class HypothesisTransitionResult:
    """Completed two-step transition and its trial-level diagnostics."""

    context: TransitionContext
    selection: HypothesisSelection
    prior_after: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class TwoStepHypothesisTransitionMixin(ABC):
    """Reusable lifecycle for the two-step hypothesis-transition process.

    Concrete modes implement :meth:`select_hypotheses` and
    :meth:`assign_prior`.  Optional hooks update controller state before the
    selection and write mode-specific logs after the prior has been committed.
    """

    last_transition_result: HypothesisTransitionResult | None = None
    phase = ModulePhase.PRE_CHOICE
    role = ModuleRole.HYPOTHESIS_TRANSITION

    def process(self, **kwargs) -> None:
        self._prepare_hypothesis_transition(**kwargs)
        context = self._build_transition_context()

        selection = self.select_hypotheses(context, **kwargs)
        self._validate_selection(selection)
        self._commit_selection(selection)

        prior = np.asarray(
            self.assign_prior(context, selection, **kwargs), dtype=float
        ).reshape(-1)
        self._validate_prior_assignment(prior, selection)
        self.engine.prior = prior.copy()

        diagnostics = self._finish_hypothesis_transition(
            context,
            selection,
            prior,
            **kwargs,
        )
        self.last_transition_result = HypothesisTransitionResult(
            context=context,
            selection=selection,
            prior_after=prior.copy(),
            diagnostics=dict(diagnostics or {}),
        )

    def _prepare_hypothesis_transition(self, **kwargs) -> None:
        """Update causal controller state before step 1, when applicable."""

        del kwargs

    def _build_transition_context(self) -> TransitionContext:
        total = int(getattr(self, "total_hypo", getattr(self.engine, "set_size", 0)))
        if total <= 0:
            raise ValueError("hypothesis transition requires a non-empty hypothesis space.")

        raw = getattr(self.engine, "posterior", None)
        if raw is None:
            raw = getattr(self.engine, "prior", None)
        posterior = np.asarray(raw, dtype=float).reshape(-1)
        if posterior.shape != (total,):
            raise ValueError(
                "transition posterior width does not match hypothesis space: "
                f"{posterior.shape[0]} vs {total}."
            )
        if not np.all(np.isfinite(posterior)) or np.any(posterior < 0.0):
            raise ValueError("transition posterior must be finite and non-negative.")
        posterior_total = float(np.sum(posterior))
        if posterior_total <= 0.0:
            raise ValueError("transition posterior must have positive total mass.")
        posterior = posterior / posterior_total

        active = getattr(self, "active", None)
        if active is None:
            mask = getattr(self.engine, "hypotheses_mask", None)
            if mask is None:
                active = np.arange(total, dtype=int)
            else:
                active = np.flatnonzero(np.asarray(mask, dtype=float) > 0.0)
        active_before = np.asarray(active, dtype=int).reshape(-1).copy()

        trial_index = getattr(self, "trial_index", None)
        if trial_index is None:
            log = getattr(self, "transition_log", None)
            if log is None:
                log = getattr(self, "strategy_counts_log", ())
            trial_index = len(log)

        return TransitionContext(
            trial_index=int(trial_index),
            posterior=posterior.copy(),
            active_before=active_before,
            signals=dict(self._transition_signals()),
        )

    def _transition_signals(self) -> Mapping[str, Any]:
        return {}

    @abstractmethod
    def select_hypotheses(
        self,
        context: TransitionContext,
        **kwargs,
    ) -> HypothesisSelection:
        """Perform step 1 and return an explicit active-set decision."""

        raise NotImplementedError

    @abstractmethod
    def assign_prior(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        **kwargs,
    ) -> np.ndarray:
        """Perform step 2 and return prior mass on ``selection.active_after``."""

        raise NotImplementedError

    def _commit_selection(self, selection: HypothesisSelection) -> None:
        self.old_active = selection.active_before.copy()
        self.active = selection.active_after.copy()
        apply_mask = getattr(self, "_apply_mask", None)
        if callable(apply_mask):
            apply_mask()
            return

        total = int(getattr(self, "total_hypo", getattr(self.engine, "set_size", 0)))
        mask = np.zeros(total, dtype=float)
        mask[self.active] = 1.0
        self.engine.hypotheses_mask = mask

    def _finish_hypothesis_transition(
        self,
        context: TransitionContext,
        selection: HypothesisSelection,
        prior: np.ndarray,
        **kwargs,
    ) -> Mapping[str, Any]:
        del context, selection, prior, kwargs
        return {}

    def _validate_selection(self, selection: HypothesisSelection) -> None:
        if not isinstance(selection, HypothesisSelection):
            raise TypeError("select_hypotheses() must return HypothesisSelection.")
        total = int(getattr(self, "total_hypo", getattr(self.engine, "set_size", 0)))
        active = np.asarray(selection.active_after, dtype=int).reshape(-1)
        if active.size == 0:
            raise ValueError("hypothesis selection cannot produce an empty active set.")
        if np.unique(active).size != active.size:
            raise ValueError("hypothesis selection contains duplicate active indices.")
        if np.any(active < 0) or np.any(active >= total):
            raise ValueError("hypothesis selection contains out-of-range indices.")

        expected = HypothesisSelection.from_active_sets(
            selection.active_before,
            selection.active_after,
        )
        for name in ("survivors", "dropped", "newcomers"):
            actual_values = np.sort(np.asarray(getattr(selection, name), dtype=int))
            expected_values = np.sort(np.asarray(getattr(expected, name), dtype=int))
            if not np.array_equal(actual_values, expected_values):
                raise ValueError(f"selection.{name} is inconsistent with the active sets.")

        for dropped_hypothesis, newcomer in selection.replacement_pairs:
            if dropped_hypothesis not in set(selection.dropped.tolist()):
                raise ValueError("replacement pair references a hypothesis that was not dropped.")
            if newcomer not in set(selection.newcomers.tolist()):
                raise ValueError("replacement pair references a hypothesis that is not new.")

    def _validate_prior_assignment(
        self,
        prior: np.ndarray,
        selection: HypothesisSelection,
    ) -> None:
        total = int(getattr(self, "total_hypo", getattr(self.engine, "set_size", 0)))
        if prior.shape != (total,):
            raise ValueError(
                "assigned prior width does not match hypothesis space: "
                f"{prior.shape[0]} vs {total}."
            )
        if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
            raise ValueError("assigned prior must be finite and non-negative.")
        if not np.isclose(float(np.sum(prior)), 1.0, rtol=1e-9, atol=1e-12):
            raise ValueError("assigned prior must sum to one.")

        inactive = np.ones(total, dtype=bool)
        inactive[selection.active_after] = False
        if np.any(prior[inactive] > 1e-12):
            raise ValueError("assigned prior has mass outside the selected active set.")


__all__ = [
    "HypothesisSelection",
    "HypothesisTransitionResult",
    "TransitionContext",
    "TwoStepHypothesisTransitionMixin",
]
