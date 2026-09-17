"""Verify full CPU budgets, worker thread caps and nested-pool ownership."""
import os

from joblib import Parallel, delayed
import pytest
from threadpoolctl import threadpool_info

from src.Bayesian_state.utils import parallel as runtime


def _worker_probe():
    with runtime.single_threaded_processes():
        nested_jobs = runtime.parallel_job_count(128, 128)
        child_pids = Parallel(n_jobs=nested_jobs)(delayed(os.getpid)() for _ in range(2))
    return {
        'pid': os.getpid(), 'nested_jobs': nested_jobs, 'child_pids': child_pids,
        'env': {name: os.environ.get(name) for name in (
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')},
        'threads': [pool['num_threads'] for pool in threadpool_info()],
    }


def test_full_budget_without_reserved_cores_and_respects_task_affinity_limits(monkeypatch):
    monkeypatch.setattr(runtime, 'cpu_count', lambda: 128)
    with runtime.single_threaded_processes():
        assert runtime.parallel_job_count(128, 256) == 128
        assert runtime.parallel_job_count(128, 17) == 17
        assert runtime.parallel_job_count(4, 17) == 4
        monkeypatch.setattr(runtime, 'cpu_count', lambda: 2)
        assert runtime.parallel_job_count(128, 256) == 2
    with pytest.raises(ValueError, match='zero'):
        runtime.parallel_job_count(0, 2)
    with pytest.raises(ValueError, match='positive'):
        runtime.parallel_job_count(2, 0)


def test_child_limits_override_inherited_environment_and_no_nested_processes(monkeypatch):
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        monkeypatch.setenv(name, '64')
    before = [(pool['filepath'], pool['num_threads']) for pool in threadpool_info()]
    with runtime.single_threaded_processes():
        assert all(pool['num_threads'] == 1 for pool in threadpool_info())
        rows = Parallel(n_jobs=2)(delayed(_worker_probe)() for _ in range(4))
    for row in rows:
        assert row['pid'] != os.getpid()
        assert row['nested_jobs'] == 1
        assert row['child_pids'] == [row['pid'], row['pid']]
        assert set(row['env'].values()) == {'1'}
        assert set(row['threads']) == {1}
    assert [(pool['filepath'], pool['num_threads']) for pool in threadpool_info()] == before


def test_model_0826_generator_defaults_to_full_budget_but_keeps_explicit_override(tmp_path):
    from src.Bayesian_state.optimization.model_0826 import build_model_0826_hyper_config
    from src.Bayesian_state.optimization.parameter_space import load_model_parameter_space
    space = load_model_parameter_space('configs/exp123/specific_models/model_0826_cond1_parameter_space.yaml',
                                      expected_model_id='model_0826')
    analysis = dict(analysis_id='parallel-test', subjects=[101], hyper_base_seed=9)
    budgets = {'coarse': {'particle_count': 16, 'filter_seed_count': 4},
               'fine': {'particle_count': 64, 'filter_seed_count': 8},
               'final_rescore': {'particle_count': 128, 'filter_seed_count': 16, 'seed_family': 'parallel-test'}}
    full = build_model_0826_hyper_config(analysis, space, 'PMH', tmp_path/'base.yaml', tmp_path/'out', budgets)
    assert full['cd']['parallel_budget'] == 128
    analysis['cd'] = {'parallel_budget': 2}
    small = build_model_0826_hyper_config(analysis, space, 'PMH', tmp_path/'base.yaml', tmp_path/'out', budgets)
    assert small['cd']['parallel_budget'] == 2
    full['cd']['parallel_budget'] = 2
    assert full == small
