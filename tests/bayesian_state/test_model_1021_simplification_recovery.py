"""Neutral recovery starts and a shared bound on all numerical case workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import time
from types import SimpleNamespace

import yaml

from src.Bayesian_state.workflows.analysis import model_1021_simplification_recovery as study


def test_all_variable_coordinates_have_support_only_starts():
    settings = yaml.safe_load(Path('configs/shared/model1021_simplification.yaml').read_text())
    for condition, variables in [(2, 8), (3, 9)]:
        space = study.driver.model_space(settings, 'COMPACT', condition)
        rows = study.neutral_starts(space, 40, 20261021)
        assert len(rows) == len({r['id'] for r in rows}) == 40
        assert sum(len(values)>1 for values in space.values()) == variables
        assert rows[0]['hyperparams'] == {k: values[len(values)//2] for k, values in space.items()}
        assert {r['hyperparams'][study.driver.H+'max_capacity'] for r in rows} == set(space[study.driver.H+'max_capacity'])
        assert study.neutral_starts(space, 40, 20261021) == rows


def test_case_controllers_share_one_worker_limit(monkeypatch, tmp_path):
    counts = dict(active=0, maximum=0, finished=0)
    lock = Lock()

    def mock_run(*args, **kwargs):
        assert all(kwargs['env'][name] == '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'))
        with lock:
            counts['active'] += 1
            counts['maximum'] = max(counts['maximum'], counts['active'])
        time.sleep(.02)
        with lock:
            counts['active'] -= 1
            counts['finished'] += 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(study.subprocess, 'run', mock_run)
    old_execute, old_hashes = study.driver.base.execute, study.driver.base.source_hashes
    try:
        with ThreadPoolExecutor(max_workers=3) as numerical:
            study.configure_executor(numerical)
            with ThreadPoolExecutor(max_workers=2) as cases:
                futures = []
                for case in range(2):
                    folder = tmp_path/str(case)
                    folder.mkdir()
                    jobs = [dict(name=f'job{i}', model='COMPACT', phase='training', runtime='unused',
                                 mechanism=dict(capacity='evidence_mean')) for i in range(8)]
                    futures.append(cases.submit(study.driver.base.execute, jobs, folder, 128))
                for future in futures:
                    future.result()
    finally:
        study.driver.base.execute, study.driver.base.source_hashes = old_execute, old_hashes
    assert counts == dict(active=0, maximum=3, finished=16)
