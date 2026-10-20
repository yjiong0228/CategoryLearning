"""Capture pre-feedback region associations inside every finite-H particle.

The context wraps StateModel.begin_trial to copy active A and q after the H
transition. The existing particle-filter observer supplies decision weights and
the predictive/filtered particle weights. Recorded q*A must reproduce the actual
per-rule predictions. No observations, random streams or learning are changed.

Sparse slots preserve particle identity and jointly available rules. Padding is
rule ID -1 with zero numerical values. Both actual filter weights (which use the
core's 1e-12 choice floor) and unfloored marginal choice probabilities are saved.
The arrays support later geometric reason projection without reading oral data.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from ...model.state_model import StateModel
from ...model.modules.base_module import ModuleRole
from ...model.modules.hypothesis_transition.finite_region_learning import ActiveAssociationLikelihood

VERSION = 1
SLOT_FIELDS = ('choice_weights', 'geometry', 'associations', 'active_beta', 'active_belief')
PARTICLE_FIELDS = ('perceived_stimulus', 'particle_prediction', 'prior_weights', 'filtered_weights',
                   'choice_floor_active')


class FiniteRegionStates:
    """One complete filter's report-independent, sparse decision records."""

    def __init__(self, choices: np.ndarray, particle_count: int) -> None:
        choices = np.asarray(choices)
        if (choices.ndim != 1 or not len(choices) or not np.isin(choices, [1, 2, 3, 4]).all()
                or type(particle_count) is not int or particle_count < 1):
            raise ValueError('Require valid choices and a positive particle count')
        self.choices = choices.astype(int, copy=True)
        self.particle_count = particle_count
        self.models = []
        self.pending = []
        self.rows = []
        self.current = None
        self.total_rules = None

    def capture(self, model, prepared) -> None:
        """Copy the association state at the already completed pre-choice transition."""
        index = len(self.pending)
        if self.current is not None or index >= self.particle_count or len(self.rows) >= len(self.choices):
            raise RuntimeError('Unexpected trial/observer lifecycle during region capture')
        if len(self.models) < self.particle_count:
            if any(m is model for m in self.models):
                raise RuntimeError('Repeated particle before its observer')
            self.models.append(model)
        elif self.models[index] is not model:
            raise RuntimeError('Particle order changed during region capture')
        evaluator = model.engine.observation_likelihood
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        if model.engine.context.condition != 2 or not isinstance(evaluator, ActiveAssociationLikelihood):
            raise ValueError('Finite-region capture requires condition-2 association particles')
        if self.total_rules is None:
            self.total_rules = h.total_hypo
        elif self.total_rules != h.total_hypo:
            raise ValueError('Particle catalogues differ')
        ids = np.sort(h.active.copy())
        beta = model.engine.beta[ids].copy()
        perceived = prepared.perceived_stimulus.copy()
        geometry = np.stack([model.partition_model.get_category_probabilities(
            int(rule), ([perceived],), float(value), distance_mode='boundary')[:, 0]
            for rule, value in zip(ids, beta)])
        self.pending.append(dict(rule_ids=ids, active_beta=beta, geometry=geometry,
            associations=h.region_associations.associations(ids), perceived_stimulus=perceived))

    def __call__(self, trial: int, phase: str, snapshot) -> None:
        if trial != len(self.rows) or len(self.pending) != self.particle_count:
            raise RuntimeError('Incomplete or out-of-order particle region state')
        if phase == 'predictive':
            if self.current is not None:
                raise RuntimeError('Repeated predictive observer')
            n, width = self.particle_count, max(len(p['rule_ids']) for p in self.pending)
            row = dict(rule_ids=np.full((n, width), -1, dtype=np.int32),
                geometry=np.zeros((n, width, 4)), associations=np.zeros((n, width, 4, 4)),
                active_beta=np.zeros((n, width)), active_belief=np.zeros((n, width)),
                choice_weights=np.zeros((n, width)), perceived_stimulus=np.empty((n, 4)),
                prior_weights=np.array(snapshot['weights'], copy=True),
                particle_prediction=np.array(snapshot['prediction'], copy=True))
            for i, saved in enumerate(self.pending):
                ids = saved['rule_ids']; size = len(ids)
                np.testing.assert_array_equal(ids, np.flatnonzero(snapshot['active'][i]))
                for name in ('rule_ids', 'geometry', 'associations', 'active_beta'):
                    row[name][i, :size] = saved[name]
                row['perceived_stimulus'][i] = saved['perceived_stimulus']
                row['choice_weights'][i, :size] = snapshot['choice_weights'][i, ids]
                row['active_belief'][i, :size] = snapshot['belief'][i, ids]
                np.testing.assert_array_equal(saved['active_beta'], snapshot['beta'][i, ids])
                np.testing.assert_array_equal(saved['perceived_stimulus'], snapshot['perceived_stimulus'][i])
                per_rule = np.einsum('kg,kgc->kc', saved['geometry'], saved['associations'])
                np.testing.assert_allclose(per_rule, snapshot['rule_choice_probabilities'][i, ids],
                                           atol=1e-12, rtol=0)
            np.testing.assert_allclose(row['choice_weights'].sum(1), 1., atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.einsum('pk,pkg,pkgc->pc', row['choice_weights'],
                row['geometry'], row['associations']), row['particle_prediction'], atol=1e-12, rtol=0)
            row['prediction'] = row['prior_weights'] @ row['particle_prediction']
            row['choice_probability'] = np.asarray(row['prediction'][self.choices[trial]-1])
            self.current = row
        elif phase == 'filtered':
            if self.current is None:
                raise RuntimeError('Filtered observer must follow predictive observer')
            row = self.current
            selected = row['particle_prediction'][:, self.choices[trial]-1]
            expected = row['prior_weights'] * np.clip(selected, 1e-12, 1.)
            expected /= expected.sum()
            np.testing.assert_allclose(snapshot['weights'], expected, atol=1e-12, rtol=0)
            row['filtered_weights'] = np.array(snapshot['weights'], copy=True)
            row['choice_floor_active'] = selected < 1e-12
            self.rows.append(row)
            self.pending = []
            self.current = None
        else:
            raise ValueError('Unknown particle observer phase')

    def arrays(self) -> dict[str, np.ndarray]:
        """Return detached sparse arrays after the complete filter replay."""
        if len(self.rows) != len(self.choices) or self.pending or self.current is not None:
            raise ValueError('Incomplete finite-region state capture')
        length, n = len(self.rows), self.particle_count
        width = max(r['rule_ids'].shape[1] for r in self.rows)
        first = self.rows[0]
        result = dict(rule_ids=np.full((length, n, width), -1, dtype=np.int32))
        for name in SLOT_FIELDS:
            result[name] = np.zeros((length, n, width, *first[name].shape[2:]), dtype=first[name].dtype)
        for trial, row in enumerate(self.rows):
            used = row['rule_ids'].shape[1]
            for name in ('rule_ids', *SLOT_FIELDS):
                result[name][trial, :, :used] = row[name]
        for name in (*PARTICLE_FIELDS, 'prediction', 'choice_probability'):
            result[name] = np.stack([r[name] for r in self.rows])
        result.update(choices=self.choices.copy(), total_rules=np.array(self.total_rules, dtype=np.int32),
                      format_version=np.array(VERSION, dtype=np.int16))
        return result


@contextmanager
def capture_finite_region_states(*, choices: np.ndarray, particle_count: int):
    """Yield the existing PF observer callback; install for exactly one filter."""
    original = StateModel.begin_trial
    if getattr(original, '_finite_region_capture', False):
        raise ValueError('Finite-region capture must be installed once')
    collector = FiniteRegionStates(choices, particle_count)

    def begin(model, *args, **kwargs):
        prepared = original(model, *args, **kwargs)
        collector.capture(model, prepared)
        return prepared

    begin._finite_region_capture = True
    StateModel.begin_trial = begin
    try:
        yield collector
    finally:
        StateModel.begin_trial = original
