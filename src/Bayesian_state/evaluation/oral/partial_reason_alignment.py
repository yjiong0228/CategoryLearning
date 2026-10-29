"""Bidirectional content coverage for provisional reports of choice reasons.

Report features and predicates retain equal descriptive weight. Extra reasons
stay in the denominator; unspoken rule content is measured separately. Local
category content is a geometric description, not a causal attention measure.
"""
from __future__ import annotations

from functools import lru_cache
import json
import re

import numpy as np

from .constraint_alignment import evaluate_component
from ...model.modules.readout.report_space import component_tokens

FEATURE_METRICS = ('reason_feature_coverage', 'basis_feature_coverage', 'exact_feature_set')
PREDICATE_METRICS = ('predicate_supported', 'predicate_conflicting', 'predicate_unexplained',
                     'basis_predicates_explicit')


def unresolved_reason_role(text: str) -> bool:
    """Keep negative/concessive attribution out of a positive-reason score."""
    return bool(re.search(r'不看|不用|没看|无关|不是|并非|排除|虽然|但是|然而|但|猜|随便|随机', text))


def feature_coverage(reported: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return component × three feature-set scores, with no best-match step."""
    report = np.asarray(reported, bool)
    masks = np.asarray(basis, bool)
    if report.ndim != 1 or masks.shape[-1] != len(report) or not report.any():
        raise ValueError('Need a nonempty report and aligned feature masks')
    shared = np.sum(masks & report, axis=-1)
    size = masks.sum(axis=-1)
    if np.any(size == 0):
        raise ValueError('Only represented, nonempty local bases may be scored')
    return np.stack((shared/report.sum(), shared/size,
                     np.all(masks == report, axis=-1).astype(float)), axis=-1)


@lru_cache(maxsize=65536)
def _predicate_verdict(token: str, geometry: str) -> tuple[float, float, float]:
    rows = np.asarray(json.loads(geometry), float)
    verdict = evaluate_component((token,), rows[:, :-1], rows[:, -1]).functional
    return tuple(float(verdict == name) for name in ('support', 'conflict', 'undetermined'))


def predicate_coverage(tokens: tuple[str, ...], A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean atomic verdicts plus directly mentioned local rule predicates.

The caller separately requires a coherent encoded report and carries its
unparsed fragments, symbolic references and threshold conventions. Predicate
fractions are content summaries, not probabilities of producing a report.
"""
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError('Need unique nonempty encoded predicates')
    geometry = json.dumps(np.column_stack((A, b)).tolist(), separators=(',', ':'))
    fractions = np.mean([_predicate_verdict(token, geometry) for token in tokens], axis=0)
    basis = set(component_tokens(A, b))
    if not basis:
        raise ValueError('Local rule needs at least one represented predicate')
    return np.r_[fractions, len(basis.intersection(tokens))/len(basis)]


def component_geometry(space, choice: int) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Return valid local regions, their feature masks and complete-rule masks."""
    if choice not in range(1, space.n_cats+1):
        raise ValueError('Choice must use the catalogue category coordinate')
    entries, local, whole = [], [], []
    for h, rule in enumerate(space):
        complete_mask = np.any(np.vstack([c.A for region in rule.categories
                                          for c in region.components]) != 0, axis=0)
        for m, region in enumerate(rule.categories[choice-1].components):
            entries.append((h, m, region.A, region.b))
            local.append(np.any(np.abs(region.A) > 1e-12, axis=0))
            whole.append(complete_mask)
    return entries, np.asarray(local), np.asarray(whole)


def stimulus_choice_reference(stimuli: np.ndarray, entries: list[tuple], rules: int,
                              components: int, tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Uniform plausible rules given the physical stimulus and chosen category.

Each compatible rule receives the same mass. Its containing components share
that mass. This is a descriptive no-history benchmark, not a fitted choice
model; impossible stimulus/choice configurations retain NaN and an explicit mask.
"""
    x = np.asarray(stimuli, float)
    inside = np.zeros((len(x), rules, components), bool)
    for h, m, A, b in entries:
        inside[:, h, m] = np.all(x @ np.asarray(A).T <= np.asarray(b)+tolerance, axis=1)
    count = inside.sum(axis=2)
    compatible = count > 0
    n_rules = compatible.sum(axis=1)
    available = n_rules > 0
    mass = np.full(inside.shape, np.nan, float)
    mass[available] = (inside[available]/np.maximum(count[available, :, None], 1)
                       /n_rules[available, None, None])
    np.testing.assert_allclose(mass[available].sum((1, 2)), 1., atol=1e-12, rtol=0)
    return mass, available


def project(mass: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Average scores under unchanged behavior-derived component probabilities."""
    mass, values = np.asarray(mass, float), np.asarray(values, float)
    if mass.shape != values.shape[:-1] or not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('Scores and component probabilities must align')
    np.testing.assert_allclose(mass.sum(), 1., atol=1e-10, rtol=0)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Content scores must lie in the unit interval')
    return np.einsum('hm,hmk->k', mass, values)
