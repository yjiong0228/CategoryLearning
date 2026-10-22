"""Verify shared leases and the diagnostic's decreasing capacity reservation."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

from src.Bayesian_state.workflows.analysis.model_1023_compute_budget import (
    SharedBudget, create_budget, summarize_budget,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_independent_clients_share_one_strict_limit(tmp_path):
    create_budget(tmp_path/'budget', 2)
    clients = [SharedBudget(tmp_path/'budget', name) for name in ('validation', 'human')]

    def task(i):
        with clients[i % 2].lease(str(i)):
            time.sleep(.01)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(task, range(12)))
    done = summarize_budget(tmp_path/'budget')
    assert done['peak_new_processes'] == 2
    assert done['peak_including_legacy_reservation'] == 2
    assert done['completed_by_phase'] == {'validation': 6, 'human': 6}


def test_reservation_follows_remaining_diagnostic_stages(tmp_path):
    legacy = tmp_path/'diagnostic'
    settings = dict(confirmation_candidates=6, audit_candidates=3, initial_candidates=64,
                    selection_candidates=16, rounds=3, carry_candidates=16,
                    proposals=dict(elites=4, rays=4, pairs=8, multi=4, global_starts=16),
                    budgets={stage: dict(particles=128, seeds=2) for stage in
                             ('screen', 'refine_1', 'refine_2', 'refine_3')})
    settings['budgets'].update(selection=dict(particles=512, seeds=4),
                               confirmation=dict(particles=1024, seeds=8))
    cases = [dict(output=str(legacy/str(i)), settings=settings) for i in range(2)]
    save(legacy/'cases.json', {'cases': cases})
    save(legacy/'protocol.json', {'phase': 'diagnostic'})
    create_budget(tmp_path/'budget', 128, legacy)
    client = SharedBudget(tmp_path/'budget', 'validation')
    assert client.reservation() == 128
    for i in range(2):
        save(legacy/str(i)/'selection_completion.json', {'status': 'complete'})
    assert client.reservation() == 96
    save(legacy/'resource_snapshots/000001.json', {'completed': 1572})
    client._checked = 0
    assert client.reservation() == 60
    save(legacy/'resource_snapshots/000002.json', {'completed': 1612})
    client._checked = 0
    assert client.reservation() == 48  # Retain the next audit stage's peak.
    save(legacy/'selected_nominees.json', {'status': 'complete'})
    assert client.reservation() == 48
    save(legacy/'resource_snapshots/000003.json', {'completed': 1670})
    client._checked = 0
    assert client.reservation() == 16  # Retain the later generating-point peak.
    save(legacy/'audit_fit/completion.json', {'status': 'complete'})
    assert client.reservation() == 16
    save(legacy/'resource_snapshots/000004.json', {'completed': 1692})
    client._checked = 0
    assert client.reservation() == 4
    save(legacy/'completion.json', {'status': 'complete'})
    assert client.reservation() == 0
