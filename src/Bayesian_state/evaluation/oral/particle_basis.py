"""Retain sparse, report-independent decision snapshots inside each particle.

The existing collector remains responsible for the established marginal arrays.
``with_particle_basis`` extends that collector with active rule identities,
actual response contributions, category probabilities, precision, active-rule
beliefs, perceived inputs and current-choice component weights. Padding uses
rule ID -1 and zero numerical values. These are pre-feedback decision states;
dormant precision caches, historical evidence and future particle ancestry are
outside this readout snapshot. No oral content or random draws are used.
"""
from __future__ import annotations

import numpy as np

VERSION = 1
PREFIX = 'particle_basis_'
SLOT_FIELDS = ('choice_weights', 'category_probabilities', 'active_beta',
               'active_belief', 'component_weights')
PARTICLE_FIELDS = ('prior_weights', 'perceived_stimulus', 'prediction', 'pairing_marginal')


class _ParticleBasisMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.particle_rows: list[dict[str, np.ndarray]] = []
        self.particle_count: int | None = None

    def __call__(self, trial: int, phase: str, snapshot) -> None:
        super().__call__(trial, phase, snapshot)
        if phase == 'filtered':
            return
        active = np.asarray(snapshot['active'], bool)
        n, h = active.shape
        if self.particle_count is not None and self.particle_count != n:
            raise ValueError('Particle count must remain fixed within one replay')
        self.particle_count = n
        counts = active.sum(axis=1)
        if not counts.min() > 0:
            raise ValueError('Each predictive particle must retain an active rule')
        width = int(counts.max())
        particles, rules = np.nonzero(active)
        slots = np.cumsum(active, axis=1)[particles, rules] - 1
        ids = np.full((n, width), -1, dtype=np.int32)
        ids[particles, slots] = rules
        row = dict(rule_ids=ids)
        fields = dict(choice_weights='choice_weights', category_probabilities='rule_choice_probabilities',
                      active_beta='beta', active_belief='belief')
        for name, source in fields.items():
            values = np.asarray(snapshot[source])
            if values.shape[:2] != (n, h) or not np.isfinite(values[active]).all() or (values[active] < 0).any():
                raise ValueError('Invalid active decision state: ' + source)
            saved = np.zeros((n, width, *values.shape[2:]), dtype=values.dtype)
            saved[particles, slots] = values[particles, rules]
            row[name] = saved
        for name, source in (('prior_weights', 'weights'), ('perceived_stimulus', 'perceived_stimulus'),
                             ('prediction', 'prediction')):
            row[name] = np.array(snapshot[source], copy=True)
        pairing = snapshot.get('pairing')
        row['pairing_marginal'] = (np.empty((n, 0)) if pairing is None else np.array(pairing, copy=True))
        if (row['pairing_marginal'].ndim != 2 or row['pairing_marginal'].shape[0] != n
                or not np.isfinite(row['pairing_marginal']).all() or (row['pairing_marginal'] < 0).any()):
            raise ValueError('Invalid optional particle pairing marginal')
        if pairing is not None:
            np.testing.assert_allclose(row['pairing_marginal'].sum(axis=1), 1., atol=1e-12, rtol=0)
        probabilities = row['category_probabilities']
        np.testing.assert_allclose(probabilities[ids >= 0].sum(axis=-1), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(row['choice_weights'].sum(axis=1), 1., atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.einsum('pk,pkc->pc', row['choice_weights'], probabilities),
                                   row['prediction'], atol=1e-12, rtol=0)
        choice = int(self.choices[trial]) - 1
        rho = np.zeros((n, width, self.max_components))
        for rule in np.unique(ids[row['choice_weights'] > 0]):
            use_p, use_k = np.nonzero((ids == rule) & (row['choice_weights'] > 0))
            components = self.space[int(rule)].categories[choice].components
            if len(components) == 1:
                rho[use_p, use_k, 0] = 1.
                continue
            unique, inverse = np.unique(row['perceived_stimulus'][use_p], axis=0, return_inverse=True)
            distances = np.column_stack([self.geometry.distances_to_polytope(unique, part) for part in components])
            nearest = np.isclose(distances, distances.min(axis=1, keepdims=True),
                                 atol=self.geometry.tolerance, rtol=0)
            rho[use_p, use_k, :len(components)] = (nearest / nearest.sum(axis=1, keepdims=True))[inverse]
        row['component_weights'] = rho
        np.testing.assert_allclose(rho.sum(axis=-1), (row['choice_weights'] > 0).astype(float), atol=1e-12, rtol=0)
        prior = row['prior_weights'][:, None] * row['choice_weights']
        joint = prior * probabilities[:, :, choice]
        for values, name in ((prior, 'prior_component_mass'), (joint, 'choice_component_joint_mass')):
            reconstructed = np.zeros((h, self.max_components))
            np.add.at(reconstructed, rules, (values[..., None] * rho)[particles, slots])
            np.testing.assert_allclose(reconstructed, self.rows[-1][name], atol=1e-12, rtol=0)
        self.particle_rows.append(row)

    def arrays(self) -> dict[str, np.ndarray]:
        original = super().arrays()
        if len(self.particle_rows) != len(self.choices):
            raise ValueError('Incomplete particle-level decision snapshots')
        width = max(row['rule_ids'].shape[1] for row in self.particle_rows)
        length, n = len(self.particle_rows), self.particle_count
        ids = np.full((length, n, width), -1, dtype=np.int32)
        extra = dict(rule_ids=ids)
        first = self.particle_rows[0]
        for name in SLOT_FIELDS:
            extra[name] = np.zeros((length, n, width, *first[name].shape[2:]), dtype=first[name].dtype)
        for trial, row in enumerate(self.particle_rows):
            used = row['rule_ids'].shape[1]
            for name in ('rule_ids', *SLOT_FIELDS):
                extra[name][trial, :, :used] = row[name]
        for name in PARTICLE_FIELDS:
            extra[name] = np.stack([row[name] for row in self.particle_rows])
        extra['format_version'] = np.array(VERSION, dtype=np.int16)
        return dict(original, **{PREFIX + name: value for name, value in extra.items()})


def with_particle_basis(base_collector: type) -> type:
    """Extend the caller's existing collector, preserving its loaded core.

    Passing the class explicitly allows archived-runtime workers to use their
    already selected collector without importing a second cognitive package.
    """
    if issubclass(base_collector, _ParticleBasisMixin):
        raise ValueError('Particle-basis collection must be installed once')

    class ParticleBasisCollector(_ParticleBasisMixin, base_collector):
        pass

    return ParticleBasisCollector
