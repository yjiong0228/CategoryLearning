"""Reproducible compact PF scoring for Model 0826 adaptive fitting.

One process layer spans subjects, candidates and seeds. Each complete seed is
published atomically, so interrupted batches can be resumed without discarding
research outputs. Only the shared simulation evaluator runs the model.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any

from joblib import Parallel, delayed
import numpy as np

from .diagnostics.decision_precision import mixture_nll
from .search.adaptive_proposals import point_id
from ..simulation.execution import evaluate_state_model_run
from ..simulation.parameters import apply_fixed_hyperparams_to_engine_config
from ..utils.parallel import parallel_job_count, single_threaded_processes
from ..utils.paths import ROOT_DIR
from ..utils.seeding import stable_seed


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def publish(path: Path, writer: Any) -> None:
    """Publish without replacing an existing artifact, including concurrent races."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.publishing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing artifact differs: {path}')
    else:
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        publish(path, lambda handle: handle.write(encoded))


def run_manifest(config: dict, config_path: Path, subjects: list[dict], smoke: bool) -> dict:
    files = {config_path, Path(config['parameter_space'])}
    files.update(Path(config['processed_dir']).glob('*.csv'))
    files.update(ROOT_DIR.joinpath('src').rglob('*.py'))
    files.update(ROOT_DIR.joinpath('src/Bayesian_state/hypothesis_space').rglob('*.npy'))
    files.update(Path(s['engine']) for s in subjects)
    versions = {'python': sys.version}
    for name in ('numpy', 'scipy', 'pandas', 'joblib', 'numba', 'PyYAML', 'threadpoolctl'):
        versions[name] = importlib.metadata.version(name)
    return {'config': config, 'subjects': subjects, 'smoke': smoke, 'versions': versions,
            'input_sha256': {str(path.resolve()): digest(path) for path in sorted(files)},
            'scope': 'Observed exp123 PMH choice fitting. All trials recurse; first trial excluded from default NLL. No final state precision or parameter-recovery claim.'}


def verify_manifest(manifest: dict) -> None:
    for path, expected in manifest['input_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Input changed during run: {path}')


def phase_seeds(base: int, subject: int, family: str, count: int) -> list[int]:
    seeds = [stable_seed({'role': 'model0826_adaptive_fit_v1', 'base': base,
                          'subject': subject, 'family': family, 'repeat': i}) for i in range(count)]
    if len(set(seeds)) != count:
        raise ValueError('Seed collision')
    return seeds


def validate_arrays(data: dict, particles: int, seed: int, pid: str) -> None:
    if int(data['particles']) != particles or int(data['seed']) != seed or str(data['point_id']) != pid:
        raise ValueError('Cached PF identity differs')
    p, y, mask = data['probabilities'], data['observed'], data['mask']
    if (p.ndim != 2 or y.shape != (len(p),) or mask.shape != y.shape or
            mask.dtype != np.bool_ or not np.issubdtype(y.dtype, np.integer) or
            not mask.any() or not np.isfinite(p).all() or (p < 0).any() or
            (y < 0).any() or (y >= p.shape[1]).any()):
        raise ValueError('Invalid cached probability/choice/mask arrays')
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-10)


def cache_path(root: Path, subject: int, pid: str, particles: int, seed: int) -> Path:
    return root/'cache'/str(subject)/f'R{particles}_seed{seed}'/f'{pid}.npz'


def read_seed(path: Path, particles: int, seed: int, pid: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    validate_arrays(data, particles, seed, pid)
    return data


def score_seed(context: dict, point: dict, particles: int, seed: int, path: Path) -> dict:
    pid = point_id(point)
    if path.exists():
        read_seed(path, particles, seed, pid)
        return {'computed': False}
    engine = apply_fixed_hyperparams_to_engine_config(context['engine'], point)
    engine['inference']['particle_count'] = particles
    began = perf_counter()
    result = evaluate_state_model_run(
        subject_id=context['subject'], condition=context['condition'], arrays=context['arrays'],
        params={}, engine_config_template=engine, processed_data_dir=context['processed_dir'],
        window_size=16, keep_logs=False, prediction_mode='prior_t',
        selection_prediction_mode='prior_t', loss_metric='choice_nll', trajectory_seed=seed)
    metrics = result.metrics_by_mode['prior_t']
    data = {'probabilities': np.asarray(metrics['pred_category_probs']),
            'observed': np.asarray(metrics['observed_choice_index']),
            'mask': np.asarray(metrics['valid_trial_mask'], dtype=bool),
            'particles': particles, 'seed': seed, 'point_id': pid,
            'seconds': perf_counter()-began}
    validate_arrays(data, particles, seed, pid)
    publish(path, lambda handle: np.savez_compressed(handle, **data))
    return {'computed': True}


class Scorer:
    def __init__(self, root: Path, config: dict, contexts: dict[int, dict]):
        self.root, self.config, self.contexts = root, config, contexts

    def arrays(self, subject: int, row: dict, budget: dict, family: str) -> dict:
        particles = budget['particle_count']
        seeds = phase_seeds(self.config['base_seed'], subject, family, budget['filter_seed_count'])
        runs = [read_seed(cache_path(self.root, subject, row['id'], particles, seed), particles, seed, row['id']) for seed in seeds]
        for run in runs[1:]:
            np.testing.assert_array_equal(run['observed'], runs[0]['observed'])
            np.testing.assert_array_equal(run['mask'], runs[0]['mask'])
        return {'probabilities': np.stack([run['probabilities'] for run in runs]),
                'observed': runs[0]['observed'], 'mask': runs[0]['mask'], 'seeds': np.array(seeds)}

    def batch(self, banks: dict[int, list[dict]], budget: dict, family: str, name: str) -> dict[int, list[dict]]:
        banks = {sid: rows for sid, rows in banks.items() if rows}
        if not banks:
            return {}
        plan = {'banks': {str(sid): rows for sid, rows in banks.items()}, 'budget': budget, 'family': family}
        directory = self.root/'batches'/name
        freeze_json(directory/'plan.json', plan)
        receipt = directory/'scores.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            for path, expected in saved['cache_sha256'].items():
                if digest(self.root/path) != expected:
                    raise ValueError(f'Completed PF cache changed: {path}')
            return {int(sid): rows for sid, rows in saved['rows'].items()}
        particles = budget['particle_count']
        tasks = []
        for sid, rows in banks.items():
            if len({r['id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate candidates in a batch')
            for row in rows:
                if point_id(row['hyperparams']) != row['id']:
                    raise ValueError('Candidate ID mismatch')
                for seed in phase_seeds(self.config['base_seed'], sid, family, budget['filter_seed_count']):
                    path = cache_path(self.root, sid, row['id'], particles, seed)
                    tasks.append((self.contexts[sid], row['hyperparams'], particles, seed, path))
        # Completed seeds stay cached, and do not consume pool slots on resume.
        pending = [task for task in tasks if not task[-1].exists()]
        began = perf_counter()
        workers = parallel_job_count(self.config['parallel_budget'], len(pending)) if pending else 0
        if pending:
            with single_threaded_processes():
                Parallel(n_jobs=workers)(delayed(score_seed)(*task) for task in pending)
        rows_by_subject = {}
        for sid, rows in banks.items():
            output = []
            first = None
            for row in rows:
                arrays = self.arrays(sid, row, budget, family)
                if first is not None:
                    for key in ('observed', 'mask', 'seeds'):
                        np.testing.assert_array_equal(first[key], arrays[key])
                first = arrays
                output.append({**row, 'mean_nll': mixture_nll(arrays['probabilities'], arrays['observed'], arrays['mask'])})
            rows_by_subject[sid] = output
        result = {'rows': {str(sid): rows for sid, rows in rows_by_subject.items()},
                  'workers': workers, 'computed_seed_runs': len(pending), 'seconds': perf_counter()-began,
                  'cache_sha256': {str(task[-1].relative_to(self.root)): digest(task[-1]) for task in tasks}}
        freeze_json(receipt, result)
        print(json.dumps({'batch': name, 'workers': workers, 'new_seed_runs': len(pending),
                          'seconds': result['seconds']}), flush=True)
        return rows_by_subject
