"""Coordinate scientific queues without exceeding their shared CPU budget."""
import json
from threading import Event, Thread

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_episodic_adaptive_task import SharedWorkerBudget


def test_reserved_workers_include_unlaunched_predecessor_tasks(tmp_path):
    (tmp_path / 'states').mkdir()
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(workers=4, jobs=8)))
    budget = SharedWorkerBudget(4, tmp_path)
    assert budget.reserved() == 4
    for i in range(5):
        (tmp_path / 'states' / f'{i}.json').write_text('{}')
    assert budget.reserved() == 3
    with pytest.raises(RuntimeError):
        with budget.slot():
            assert budget.active == 1
            raise RuntimeError('Worker failed')
    assert budget.active == 0
    (tmp_path / 'completion.json').write_text('{}')
    assert budget.reserved() == 0


def test_waiting_worker_enters_only_after_predecessor_releases_slot(tmp_path):
    (tmp_path / 'states').mkdir()
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(workers=1, jobs=1)))
    budget = SharedWorkerBudget(1, tmp_path)
    waiting, entered = Event(), Event()
    def work():
        waiting.set()
        with budget.slot():
            entered.set()
    thread = Thread(target=work)
    thread.start()
    assert waiting.wait(1)
    assert not entered.wait(.05)
    (tmp_path / 'states' / 'done.json').write_text('{}')
    with budget.condition:
        budget.condition.notify_all()
    thread.join(timeout=5)
    assert not thread.is_alive() and entered.is_set() and budget.active == 0
