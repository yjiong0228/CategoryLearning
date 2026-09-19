"""Explicit support extensions and boundary diagnostics for observed PMH fits.

The frozen recovery support stays immutable. Extensions change the estimation
space, not cognitive equations; artificial edges always remain review flags.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .model_0826 import _anchor_point, _profile_spaces, extract_model_0826_parameters


def adaptive_support(parameter_space: Mapping[str, Any], extensions: Mapping[str, list]) -> tuple[dict, dict, dict]:
    """Expand declared fine grids within model domains, preserving exact zeros."""
    config = deepcopy(parameter_space)
    parameters = config['subject_parameters']
    for name, values in extensions.items():
        if name not in parameters or not isinstance(values, list) or not values:
            raise ValueError(f'Invalid boundary extension: {name}')
        spec = parameters[name]
        if name == 'workspace_execution':
            for value in values:
                if not isinstance(value, dict) or set(value) != {'M', 'chi'}:
                    raise ValueError('workspace extensions require M and chi')
                m, chi = value['M'], value['chi']
                if type(m) is not int or type(chi) is not int or not 1 <= m <= 14 or chi not in (0, 1) or (m == 1 and chi):
                    raise ValueError('Invalid workspace extension; M in 1..14, chi in 0..1, M=1 requires chi=0')
            cells = {(p['M'], p['chi']) for p in spec['fine_candidates'] + values}
            spec['fine_candidates'] = [dict(M=m, chi=chi) for m, chi in sorted(cells)]
            continue
        domain = spec['theoretical_domain']
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'Nonfinite or nonnumeric extension: {name}')
            for side, compare in [('lower', lambda a, b: a < b), ('upper', lambda a, b: a > b)]:
                bound = domain[side]
                if bound is not None and (compare(value, bound) or (value == bound and not domain[f'{side}_closed'])):
                    raise ValueError(f'Extension violates {name} domain')
            if name in ('eta_plus', 'eta_minus') and value == 0:
                raise ValueError('Zero beta update rates need a separately declared ablation')
        key = 'fine_positive_values' if 'zero_value' in spec else 'fine_values'
        positive = [float(v) for v in values if v != 0 or key != 'fine_positive_values']
        spec[key] = sorted(set(spec[key] + positive))
    free = config['architecture_cells']['PMH']['free_parameters']
    space = {key: value['values'] for key, value in _profile_spaces(config, free, 'fine').items()}
    # Extremely large logit increments can round E_E to exactly one.
    for key, values in space.items():
        for value in values:
            named = extract_model_0826_parameters({key: value})
            if 'E_E' in named and not 0 < named['E_C'] <= named['E_E'] < 1:
                raise ValueError('Extension saturates E_E; choose finite nonsaturating increments')
    anchor = _anchor_point(config, free, parameters['workspace_execution']['start_candidates'][0])
    return space, anchor, config


def boundary_report(rows: list[dict], support: dict, tolerance: float, max_candidates: int) -> dict:
    """Check the representative and near-scoring alternatives; chi is categorical."""
    ranked = sorted(rows, key=lambda row: (row['mean_nll'], row['id']))
    limits: dict[str, tuple[float, float]] = {}
    for name, spec in support['subject_parameters'].items():
        values = ([v['M'] for v in spec['fine_candidates']] if name == 'workspace_execution'
                  else spec.get('fine_values', [spec.get('zero_value', 0), *spec.get('fine_positive_values', [])]))
        limits['M' if name == 'workspace_execution' else name] = (min(values), max(values))
    checked = [r for r in ranked if r['mean_nll'] <= ranked[0]['mean_nll'] + tolerance][:max_candidates]
    hits = []
    for row in checked:
        for name, value in extract_model_0826_parameters(row['hyperparams']).items():
            if name not in limits:
                continue
            for side, bound in zip(('lower', 'upper'), limits[name]):
                if not np.isclose(value, bound, rtol=0, atol=1e-9):
                    continue
                if name == 'M':
                    natural = side == 'lower' and value == 1
                else:
                    domain = support['subject_parameters'][name]['theoretical_domain']
                    natural = domain[side] is not None and domain[f'{side}_closed'] and value == domain[side]
                hits.append({'candidate': row['id'], 'parameter': name, 'value': value,
                             'side': side, 'kind': 'mechanism_boundary' if natural else 'artificial_boundary'})
    return {'checked_candidates': [r['id'] for r in checked], 'hits': hits,
            'review_required': any(h['kind'] == 'artificial_boundary' for h in hits),
            'scope': 'Finite grid edges, not a population boundary-hit rate or identifiability test.'}
