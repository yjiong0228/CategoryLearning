"""Generation readiness and one shared limit across evaluation and recovery."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import time

import pytest

from src.Bayesian_state.workflows.analysis.model_1021_simplification_overlap import EvaluationDispatch


def test_recovery_can_start_while_conditional_filter_runs_with_shared_limit(tmp_path):
    release = Event()
    lock = Lock()
    counts = dict(active=0, maximum=0, finished=0)

    def work(name):
        with lock:
            counts['active'] += 1
            counts['maximum'] = max(counts['maximum'], counts['active'])
        if name == 'conditional':
            assert release.wait(10)
        else:
            time.sleep(.02)
        with lock:
            counts['active'] -= 1
            counts['finished'] += 1

    def launch(jobs, folder, budget):
        assert len(jobs) == budget == 1
        work(jobs[0]['name'])

    with ThreadPoolExecutor(max_workers=3) as numerical:
        dispatch = EvaluationDispatch(numerical, launch)
        jobs = [dict(name='conditional', phase='evaluate'),
                dict(name='path0', phase='autonomous'), dict(name='path1', phase='autonomous')]
        with ThreadPoolExecutor(max_workers=1) as control:
            evaluation = control.submit(dispatch, jobs, tmp_path, 3)
            try:
                assert dispatch.generation_ready.wait(10)
                assert not evaluation.done()
                recovery_jobs = [numerical.submit(work, f'recovery{i}') for i in range(4)]
                for job in recovery_jobs:
                    job.result()
                assert counts['finished'] == 6
            finally:
                release.set()
            evaluation.result()
        assert counts == dict(active=0, maximum=3, finished=7)


def test_generation_failure_unblocks_controller_and_records_error(tmp_path):
    def fail(*args):
        raise RuntimeError('failed path')

    with ThreadPoolExecutor(max_workers=1) as numerical:
        dispatch = EvaluationDispatch(numerical, fail)
        with pytest.raises(RuntimeError, match='failed path'):
            dispatch([dict(name='path', phase='autonomous')], tmp_path, 1)
        assert dispatch.generation_ready.is_set()
        assert isinstance(dispatch.error, RuntimeError)
