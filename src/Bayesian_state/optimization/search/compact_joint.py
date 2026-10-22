"""Bounded population proposals on the existing COMPACT parameter support.

The population combines coordinate rays, balanced coordinate-pair changes,
changes to three or more coordinates, and new global starts. Selection uses
behavioral likelihood only. Known zero-effect failure-decay duplicates share
one search identity, while saved parameter points retain their original values.
No cognitive equations, parameter supports or likelihood definitions change.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import numpy as np

from .adaptive_proposals import candidate, point_id, ray_proposals, select_elites

CONTROL = 'engine.modules.hypo_transitions_mod.kwargs.search_controller.'
CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'


def behavior_key(point: dict) -> str:
    """Collapse a proven inactive coordinate, never a merely weak effect."""
    normalized = deepcopy(point)
    if (point.get(CONTROL + 'error_gain') == 0
            and point.get(CONTROL + 'global_search_failure_gain') == 0):
        normalized[CONTROL + 'failure_decay'] = 0.
    return point_id(normalized)


def validate_point(point: dict, space: dict) -> None:
    if set(point) != set(space) or any(point[k] not in v for k, v in space.items()):
        raise ValueError('Candidate is outside the frozen parameter support')


def initial_bank(space: dict, anchors: list[dict], count: int, seed: int) -> list[dict]:
    """Preserve behavior-only anchors and stratify the remaining coordinates."""
    if count < len(anchors) or count < 1:
        raise ValueError('Invalid initial candidate budget')
    rng = np.random.default_rng(seed)
    bank = {}
    for point in anchors:
        validate_point(point, space)
        bank.setdefault(behavior_key(point), candidate(point, 'previous_training_fit'))
    # A support-midpoint point and capacity coverage are independent of truth.
    middle = {k: v[len(v) // 2] for k, v in space.items()}
    for capacity in space[CAPACITY]:
        point = dict(middle, **{CAPACITY: capacity})
        if len(bank) < count:
            bank.setdefault(behavior_key(point), candidate(point, 'support_only_start'))
    for _ in range(20):
        width = max(2 * count, 32)
        indices = {k: ((rng.permutation(width) + .5) / width * len(v)).astype(int)
                   for k, v in space.items()}
        for i in range(width):
            if len(bank) == count:
                return list(bank.values())
            point = {k: deepcopy(v[indices[k][i]]) for k, v in space.items()}
            bank.setdefault(behavior_key(point), candidate(point, 'stratified_global_start'))
    raise ValueError('Initial candidate budget exceeds distinct behavior support')


def population_round(rows: list[dict], space: dict, seen: set[str], *, seed: int,
                     elites: int, rays: int, pairs: int, multi: int,
                     global_starts: int) -> list[dict]:
    """Construct fixed quotas without observing future behavior or truths."""
    rng = np.random.default_rng(seed)
    selected = select_elites(rows, elites, exclude_key=CAPACITY)
    if len(selected) != elites:
        raise ValueError('Insufficient distinct elites')
    keys = [k for k, values in space.items() if len(values) > 1]
    if len(keys) < 3:
        raise ValueError('Joint population search requires three variable coordinates')
    combinations_list = list(combinations(keys, 2))
    pair_order = rng.permutation(len(combinations_list))
    bank = {}

    def add(point, source, **origin):
        validate_point(point, space)
        key = behavior_key(point)
        if key in seen or key in bank:
            return False
        bank[key] = candidate(point, source, **origin)
        return True

    for number, elite in enumerate(selected):
        anchor = elite['hyperparams']
        ray_bank = ray_proposals([elite], space, set(), sum(map(len, space.values())),
                                 seed + number, 'coordinate_ray')
        added = 0
        for row in ray_bank:
            if added == rays:
                break
            added += add(row['hyperparams'], 'coordinate_ray', anchor=elite['id'])
        if added != rays:
            raise ValueError('Coordinate proposal quota exhausted')
        for kind, quota in [('coordinate_pair', pairs), ('multi_coordinate', multi)]:
            added = 0
            for attempt in range(max(200, quota * 100)):
                if added == quota:
                    break
                point = deepcopy(anchor)
                if kind == 'coordinate_pair':
                    index = pair_order[(number * pairs + attempt) % len(pair_order)]
                    changed = combinations_list[index]
                else:
                    changed = rng.choice(keys, int(rng.integers(3, len(keys) + 1)), replace=False)
                for key in changed:
                    values = [v for v in space[key] if v != anchor[key]]
                    donors = [r['hyperparams'][key] for r in selected
                              if r['hyperparams'][key] != anchor[key]]
                    choices = donors if kind == 'multi_coordinate' and donors and rng.random() < .5 else values
                    point[key] = deepcopy(choices[int(rng.integers(len(choices)))])
                added += add(point, kind, anchor=elite['id'], changed=list(changed))
            if added != quota:
                raise ValueError('Joint proposal quota exhausted')
    added = 0
    for attempt in range(max(200, global_starts * 100)):
        if added == global_starts:
            break
        point = {k: deepcopy(v[int(rng.integers(len(v)))]) for k, v in space.items()}
        added += add(point, 'global_restart')
    if added != global_starts:
        raise ValueError('Global proposal quota exhausted')
    return list(bank.values())


def diverse_shortlist(rows: list[dict], required: dict, count: int, top: int) -> list[dict]:
    """Keep the reference, strongest scores and parameter-diverse contenders."""
    ordered = sorted(rows, key=lambda r: (r['mean_nll'], r['id']))
    selected = {behavior_key(required['hyperparams']): required}
    for row in ordered[:top]:
        selected.setdefault(behavior_key(row['hyperparams']), row)
    if len(selected) > count:
        raise ValueError('Top quota exceeds shortlist budget')
    for row in select_elites(ordered, len(ordered), exclude_key=CAPACITY):
        if len(selected) == count:
            break
        selected.setdefault(behavior_key(row['hyperparams']), row)
    if len(selected) != count:
        raise ValueError('Insufficient unique behavioral contenders')
    return list(selected.values())


def stage_archive_shortlist(history: list[list[dict]], required: dict, count: int,
                            per_stage: int) -> list[dict]:
    """Recheck leaders from every stage under one fresh high-precision bank."""
    bank = {behavior_key(required['hyperparams']): required}
    for rows in reversed(history):
        for row in sorted(rows, key=lambda r: (r['mean_nll'], r['id']))[:per_stage]:
            bank.setdefault(behavior_key(row['hyperparams']), row)
    if len(bank) > count:
        raise ValueError('Stage archive exceeds independent recheck budget')
    for rows in reversed(history):
        for row in sorted(rows, key=lambda r: (r['mean_nll'], r['id'])):
            if len(bank) == count:
                return list(bank.values())
            bank.setdefault(behavior_key(row['hyperparams']), row)
    raise ValueError('Insufficient stage archive candidates')
