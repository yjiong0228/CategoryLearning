"""Evaluation keeps the pending recovery filters inside the CPU allocation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import json
import time

from src.Bayesian_state.workflows.analysis.model_1022_compact_evaluate import ResourceGate, write


def test_resource_gate_expands_only_after_recovery_completion(tmp_path):
    gate = ResourceGate(tmp_path, maximum=3, remaining=1)
    release = Event()

    def hold():
        gate.enter()
        try:
            release.wait(timeout=5)
        finally:
            gate.leave()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(hold) for _ in range(3)]
        deadline = time.monotonic()+3
        while gate.active < 2 and time.monotonic()<deadline:
            time.sleep(.005)
        assert gate.active == gate.peak == 2
        assert gate.limit() == 2
        write(tmp_path/'completion.json', dict(status='complete'))
        with gate.condition:
            gate.condition.notify_all()
        deadline = time.monotonic()+3
        while gate.active < 3 and time.monotonic()<deadline:
            time.sleep(.005)
        assert gate.active == gate.peak == 3
        release.set()
        for future in futures:
            future.result()
    assert gate.active == 0
    assert [row['evaluation_limit'] for row in gate.events] == [2, 3]


def test_completed_truth_groups_release_only_their_declared_slots(tmp_path):
    first,second=tmp_path/'first.json',tmp_path/'second.json'
    gate=ResourceGate(tmp_path,maximum=4,remaining=3,pending_stages={first:2,second:1})
    assert gate.limit()==1
    # A partial file and a nonterminal stage do not release numerical workers.
    first.write_text('{')
    write(second,dict(status='running',stage='confirmation'))
    assert gate.limit()==1
    first.write_text(json.dumps(dict(status='complete',stage='confirmation')))
    assert gate.limit()==3
    second.write_text(json.dumps(dict(status='complete',stage='selection')))
    assert gate.limit()==3
    second.write_text(json.dumps(dict(status='complete',stage='confirmation')))
    assert gate.limit()==4
