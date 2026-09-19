"""Bounded, diverse Model 0826 proposals shared by fitting and historical pilots.

Promoted unchanged from the validated pilots; no model equations live here.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import numpy as np
from .cd_v2 import canonical_point_key
from ..model_0826 import WORKSPACE_PROFILE_KEY, extract_model_0826_parameters

def point_id(point: dict) -> str:
    return hashlib.sha256(canonical_point_key(point).encode()).hexdigest()[:16]


def initial_points(space: dict, anchor: dict, count: int, seed: int) -> list[dict]:
    """Cover all workspace cells at the standard anchor, then stratify all blocks."""
    cells = space[WORKSPACE_PROFILE_KEY]
    if count < len(cells):
        raise ValueError('initial_count must cover every workspace cell')
    points = [dict(deepcopy(anchor), **{WORKSPACE_PROFILE_KEY: deepcopy(cell)}) for cell in cells]
    n = count - len(points)
    rng = np.random.default_rng(seed)
    draws = {key: np.minimum(((rng.permutation(n) + .5) / max(n, 1) * len(values)).astype(int),
                             len(values)-1) for key, values in space.items()}
    points.extend({key: deepcopy(values[draws[key][i]]) for key, values in space.items()} for i in range(n))
    unique = {point_id(p): p for p in points}
    if len(unique) != count:
        raise ValueError('Stratified starts collided; choose another proposal seed')
    return list(unique.values())


def select_elites(rows: list[dict], count: int) -> list[dict]:
    """Require two non-workspace blocks to differ when sufficient points exist."""
    ordered = sorted(rows, key=lambda r: (r['mean_nll'], r['id']))
    selected = []
    for row in ordered:
        p = row['hyperparams']
        if all(sum(point_id({k: p[k]}) != point_id({k: old['hyperparams'][k]})
                   for k in p if k != WORKSPACE_PROFILE_KEY) >= 2 for old in selected):
            selected.append(row)
        if len(selected) == count:
            return selected
    used = {r['id'] for r in selected}
    return (selected + [r for r in ordered if r['id'] not in used])[:count]


def neighbor_values(key: str, value: object, values: list) -> list:
    """One ordinal step in one primitive coordinate, including profile blocks."""
    if not isinstance(value, dict):
        i = values.index(value)
        return [values[j] for j in (i-1, i+1) if 0 <= j < len(values)]
    named = [extract_model_0826_parameters({key: v}) for v in values]
    current = extract_model_0826_parameters({key: value})
    names = [name for name in current if name != 'E_E']  # Derived, not an extra coordinate.
    vectors = np.array([[round(p[name], 10) for name in names] for p in named])
    target = np.array([round(current[name], 10) for name in names])
    distance = np.zeros(len(values), dtype=int)
    for j in range(len(names)):
        levels = np.unique(vectors[:, j])
        distance += np.abs(np.searchsorted(levels, vectors[:, j])-np.searchsorted(levels, target[j]))
    return [deepcopy(v) for v, d in zip(values, distance) if d == 1]


def propose_round(elites: list[dict], space: dict, seen: set[str], per_elite: int,
                  seed: int, local_fraction: float, jump_fraction: float) -> list[dict]:
    """Balance local moves, whole-block jumps and joint two-block moves."""
    rng = np.random.default_rng(seed)
    pending = {}
    keys = list(space)
    for elite in elites:
        anchor = elite['hyperparams']
        pools = [[], [], []]
        for key, values in space.items():
            for value in neighbor_values(key, anchor[key], values):
                pools[0].append({**deepcopy(anchor), key: deepcopy(value)})
            for value in values:
                if value != anchor[key]:
                    pools[1].append({**deepcopy(anchor), key: deepcopy(value)})
        for _ in range(max(100, per_elite * 10)):
            a, b = rng.choice(keys, 2, replace=False)
            candidate = deepcopy(anchor)
            for key in (a, b):
                local = neighbor_values(key, anchor[key], space[key])
                values = local if local and rng.random() < .5 else space[key]
                values = [v for v in values if v != anchor[key]]
                candidate[key] = deepcopy(values[int(rng.integers(len(values)))])
            pools[2].append(candidate)
        quotas = [int(per_elite * local_fraction), int(per_elite * jump_fraction)]
        quotas.append(per_elite - sum(quotas))
        selected = 0
        for kind, pool, quota in zip(('local', 'jump', 'joint'), pools, quotas):
            if quota <= 0:
                continue
            for index in rng.permutation(len(pool)):
                candidate = pool[index]
                pid = point_id(candidate)
                if pid in seen or pid in pending:
                    continue
                pending[pid] = {'id': pid, 'hyperparams': candidate, 'sources': ['new_search'],
                                'origin': {'anchor': elite['id'], 'kind': kind}}
                selected += 1
                quota -= 1
                if quota == 0:
                    break
            if selected >= per_elite:
                break
    return list(pending.values())


def candidate(point: dict, source: str, **origin: object) -> dict:
    return {'id': point_id(point), 'hyperparams': deepcopy(point),
            'sources': [source], 'origin': origin}


def merge_bank(groups: list[tuple[str, list[dict]]]) -> list[dict]:
    bank = {}
    for source, rows in groups:
        for row in rows:
            pid = row['id']
            item = bank.setdefault(pid, candidate(row['hyperparams'], source))
            if source not in item['sources']:
                item['sources'].append(source)
    return list(bank.values())


def axis_values(key: str, current: object, values: list) -> list[list]:
    """Primitive-coordinate rays; derived E_E moves with E_C / delta_E."""
    if not isinstance(current, dict) or key == WORKSPACE_PROFILE_KEY:
        return [values]
    named = [extract_model_0826_parameters({key: v}) for v in values]
    anchor = extract_model_0826_parameters({key: current})
    names = [k for k in anchor if k != 'E_E']
    return [[v for v, p in zip(values, named)
             if all(abs(p[k]-anchor[k]) < 1e-9 for k in names if k != axis)]
            for axis in names]


def ray_proposals(elites: list[dict], space: dict, seen: set[str],
                  per_elite: int, seed: int, source: str) -> list[dict]:
    """Round-robin axes with both long jumps and neighbors, without dense grids.

    Every primitive axis gets opportunities before extra points on any axis.
    This avoids spending most proposals in a large flattened joint profile.
    """
    rng = np.random.default_rng(seed)
    pending = {}
    for elite in elites:
        anchor = elite['hyperparams']
        pools = []
        for key, values in space.items():
            for ray in axis_values(key, anchor[key], values):
                ordered = [ray[int(i)] for i in np.unique(np.linspace(0, len(ray)-1, 3).astype(int))]
                ordered += [v for v in neighbor_values(key, anchor[key], values) if v in ray]
                ordered += [ray[int(i)] for i in rng.permutation(len(ray))]
                pools.append([{**deepcopy(anchor), key: deepcopy(v)} for v in ordered])
        added = 0
        while added < per_elite and any(pools):
            for pool in pools:
                while pool:
                    point = pool.pop(0)
                    pid = point_id(point)
                    if pid in seen or pid in pending:
                        continue
                    pending[pid] = candidate(point, source, kind='primitive_ray', anchor=elite['id'])
                    added += 1
                    break
                if added >= per_elite:
                    break
    return list(pending.values())


def ranking(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (r['mean_nll'], r['id']))
