"""Validate partial oral constraints against local category components.

Support is evidence about represented content, not a report likelihood. The
existing encoder defines numerical semantics; symbolic body and half references remain
unresolved. No fitted state, current stimulus, feedback or correct rule is used
to reinterpret a report. Explicit-facet matches describe geometric predicates,
not the order of cognitive operations in a decision program.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import warnings
from typing import Sequence

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

from ...model.modules.readout.report_space import component_tokens, facet_token

VERSION = "oral-partial-constraints-v1"
LABELS = ("support", "conflict", "undetermined")


class InconsistentReportError(ValueError):
    """Encoded report constraints cannot jointly describe a local region."""


@dataclass(frozen=True)
class ConstraintMatch:
    """Classify one complete conjunction against one local rule component."""

    structural: str
    functional: str
    available: bool
    explicit_predicates: int
    report_predicates: int
    symbolic_predicates: int


def _solve(c: np.ndarray, A: np.ndarray, b: np.ndarray, bounds: list):
    # SciPy passes this HiGHS option through. One solver thread prevents a
    # second parallel layer inside the workflow's independent query workers.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*",
                                category=OptimizeWarning)
        result = linprog(c, A_ub=A if len(A) else None,
                         b_ub=b if len(A) else None, bounds=bounds,
                         method="highs", options={"threads": 1})
    if result.status not in (0, 2):
        raise RuntimeError(f"Constraint solver failed: {result.message}")
    return result


def _interior(A: np.ndarray, b: np.ndarray, n_dims: int) -> float:
    """Radius of an interior ball, including the feature-cube boundaries."""
    cube = np.vstack((np.eye(n_dims), -np.eye(n_dims)))
    all_A = np.vstack((A, cube))
    all_b = np.concatenate((b, np.ones(n_dims), np.zeros(n_dims)))
    result = _solve(np.r_[np.zeros(n_dims), -1.],
                    np.column_stack((all_A, np.linalg.norm(all_A, axis=1))),
                    all_b, [(None, None)] * n_dims + [(0., None)])
    return float(result.x[-1]) if result.success else 0.


@lru_cache(maxsize=8192)
def _report_rows(tokens: tuple[str, ...], n_dims: int, tolerance: float):
    rows, symbolic = [], 0
    for token in tokens:
        if token.startswith(("body:", "half_unknown:")):
            fields = token.split(":")
            if (len(fields) != 3 or not fields[1].isdigit()
                    or int(fields[1]) >= n_dims or fields[2] not in {"lt", "gt"}):
                raise ValueError("Invalid symbolic reference")
            symbolic += 1
        elif token.startswith("facet:"):
            values = np.asarray(json.loads(token[6:]), dtype=float)
            if (values.ndim != 2 or values.shape[1] != n_dims + 1 or not len(values)
                    or not np.isfinite(values).all()
                    or np.any(np.max(np.abs(values[:, :-1]), axis=1) == 0)):
                raise ValueError("Invalid report halfspaces")
            rows.extend(values.tolist())
        else:
            raise ValueError("Unrecognized report token")
    values = np.asarray(rows, dtype=float).reshape(-1, n_dims + 1)
    A, b = values[:, :-1], values[:, -1]
    for kind in ("body", "half_unknown"):
        if any(f"{kind}:{i}:lt" in tokens and f"{kind}:{i}:gt" in tokens for i in range(n_dims)):
            raise InconsistentReportError(f"Report has opposing symbolic {kind} directions")
    if len(rows) and _interior(A, b, n_dims) <= tolerance:
        raise InconsistentReportError("Report has no positive-volume numerical interpretation")
    return A, b, symbolic


def validate_report_constraints(report_tokens: Sequence[str], n_dims: int = 4,
                                *, tolerance: float = 1e-9) -> None:
    """Check joint content before evaluating it against model states.

    Inconsistency is an annotation issue to retain separately from model
    conflict. Symbolic directions are checked within each reference type without
    a numerical anchor. Bare half references share an unknown within-report referent.
    """
    if (not isinstance(n_dims, int) or n_dims < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Expected positive dimensions and finite positive tolerance")
    if isinstance(report_tokens, str) or any(not isinstance(x, str) for x in report_tokens):
        raise ValueError("report_tokens must be a sequence of strings")
    _report_rows(tuple(sorted(set(report_tokens))), n_dims, float(tolerance))


@lru_cache(maxsize=8192)
def _rule_rows(key: str, n_dims: int, tolerance: float):
    values = np.asarray(json.loads(key), dtype=float).reshape(-1, n_dims + 1)
    A, b = values[:, :-1], values[:, -1]
    if _interior(A, b, n_dims) <= tolerance:
        raise ValueError("Rule component must have positive volume in the feature cube")
    predicates = component_tokens(A, b) if len(A) else ()
    return A, b, predicates


@lru_cache(maxsize=131072)
def _match(tokens: tuple[str, ...], key: str, n_dims: int,
           tolerance: float) -> ConstraintMatch:
    rule_A, rule_b, predicates = _rule_rows(key, n_dims, tolerance)
    oral_A, oral_b, symbolic = _report_rows(tokens, n_dims, tolerance)
    count = len(tokens)
    if not count:
        return ConstraintMatch("undetermined", "undetermined", False, 0, 0, 0)
    explicit = len(set(tokens) & set(predicates))
    if explicit == count:
        return ConstraintMatch("support", "support", True, count, count, 0)
    conflict = bool(len(oral_A) and _interior(
        np.vstack((rule_A, oral_A)), np.r_[rule_b, oral_b], n_dims) <= tolerance)
    if conflict:
        return ConstraintMatch("conflict", "conflict", True, explicit, count, symbolic)
    entailed = bool(len(oral_A)) and not symbolic
    if entailed:
        for row, bound in zip(oral_A, oral_b):
            result = _solve(-row, rule_A, rule_b, [(0., 1.)] * n_dims)
            if not result.success:
                raise RuntimeError("Validated rule component became infeasible")
            if -result.fun > bound + tolerance:
                entailed = False
                break
    return ConstraintMatch("undetermined", "support" if entailed else "undetermined",
                           True, explicit, count, symbolic)


def evaluate_component(report_tokens: Sequence[str], rule_A: np.ndarray,
                       rule_b: np.ndarray, *, tolerance: float = 1e-9) -> ConstraintMatch:
    """Evaluate observed content without constructing a language outcome space.

    ``structural=support`` requires every reported predicate to occur explicitly
    in this component. ``functional=support`` requires the component to imply
    the whole reported conjunction. Conflict means their intersection has no
    positive volume; a shared boundary is ignored under a continuous-input
    interpretation. Other combinations remain undetermined. Unspoken rule
    predicates incur no penalty. The caller records whether tokens describe a
    complete report, a parsed fragment, or an independently annotated core.
    """
    A, b = np.asarray(rule_A, dtype=float), np.asarray(rule_b, dtype=float)
    if (A.ndim != 2 or A.shape[1] < 1 or b.shape != (len(A),)
            or not np.isfinite(A).all() or not np.isfinite(b).all()
            or (len(A) and np.any(np.max(np.abs(A), axis=1) == 0))):
        raise ValueError("Expected finite, nonzero-normal rule constraints A[m,d], b[m]")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if isinstance(report_tokens, str) or any(not isinstance(x, str) for x in report_tokens):
        raise ValueError("report_tokens must be a sequence of strings")
    tokens = tuple(sorted(set(report_tokens)))
    key = facet_token(A, b)[6:] if len(A) else "[]"
    return _match(tokens, key, A.shape[1], float(tolerance))


def summarize_matches(weights: np.ndarray, matches: Sequence[ConstraintMatch]) -> dict[str, float]:
    """Aggregate already choice-conditioned state/component weights.

    Weights must be specified independently of the report, include all states,
    and sum to one. Separate components require independently supplied weights;
    choosing the best-matching component is not a valid state aggregation.
    """
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(matches),) or not len(matches) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("State weights must be finite, nonnegative and normalized")
    if not all(match.available for match in matches):
        raise ValueError("Missing report content has no alignment score")
    result = {}
    for domain in ("structural", "functional"):
        labels = np.array([getattr(match, domain) for match in matches])
        if not np.isin(labels, LABELS).all():
            raise ValueError("Unrecognized match label")
        for label in LABELS:
            result[f"{domain}_{label}"] = float(weights[labels == label].sum())
    return result


def choice_conditioned_weights(particle_weights: np.ndarray, contribution_weights: np.ndarray,
                               choice_likelihoods: np.ndarray) -> np.ndarray:
    """Condition the joint particle/rule mass once on the actual choice.

    This evaluator accepts saved per-particle values, not products of marginal
    averages. Inputs concern the current choice before current feedback.
    """
    p, w, likelihood = (np.asarray(x, dtype=float) for x in
                        (particle_weights, contribution_weights, choice_likelihoods))
    if (p.ndim != 1 or not len(p) or w.ndim != 2 or w.shape != likelihood.shape
            or w.shape[0] != len(p) or not w.shape[1]
            or any(not np.isfinite(x).all() or np.any(x < 0) for x in (p, w, likelihood))
            or np.any(likelihood > 1)
            or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0)
            or not np.allclose(w.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError("Invalid particle weights, contribution weights or choice likelihoods")
    joint = p[:, None] * w * likelihood
    if joint.sum() <= 0:
        raise ValueError("Cannot condition on an impossible choice")
    return joint / joint.sum()
