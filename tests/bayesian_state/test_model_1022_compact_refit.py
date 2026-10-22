"""Scientific guards for common-start refits and the shared CPU allocation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import time

import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_1022_compact_refit as study


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_common_starts_and_proposals_preserve_support_and_budgets(condition):
    original = yaml.safe_load(Path('configs/shared/model1021_simplification.yaml').read_text())
    policy = yaml.safe_load(Path('configs/shared/model1022_compact_refit.yaml').read_text())
    space = study.driver.model_space(original, 'COMPACT', condition)
    anchor = {k: values[len(values)//2] for k, values in space.items()}
    starts = study.starting_bank(space, [anchor], 40, 20261022)
    assert starts == study.starting_bank(space, [anchor], 40, 20261022)
    assert len(starts) == len({r['id'] for r in starts}) == 40
    assert {r['hyperparams'][study.driver.H+'max_capacity'] for r in starts} == set(space[study.driver.H+'max_capacity'])
    rows = [dict(r, mean_nll=i) for i, r in enumerate(starts)]
    seen = {r['id'] for r in starts}
    frozen = deepcopy(starts)
    settings = study.configure_settings(original, policy, seed=17, variant='COMPACT', case_id='A')
    other = study.configure_settings(original, policy, seed=17, variant='TRANSFER', case_id='B')
    for turn in (1, 2):
        proposals = study.refine_bank(rows, space, seen, settings, turn, False)
        assert proposals == study.refine_bank(rows, space, seen, other, turn, False)
        assert len(proposals) == 64
        assert not seen & {r['id'] for r in proposals}
        for row in proposals:
            assert set(row['hyperparams']) == set(space)
            assert all(row['hyperparams'][k] in values for k, values in space.items())
        seen.update(r['id'] for r in proposals)
        rows = [dict(r, mean_nll=i) for i, r in enumerate(proposals)]
    assert len(seen) == 168
    assert starts == frozen


def test_shared_queue_caps_processes_and_records_variant(monkeypatch, tmp_path):
    counts = dict(active=0, peak=0, finished=0)
    lock = Lock()

    def run(*args, **kwargs):
        assert all(kwargs['env'][key]=='1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'))
        with lock:
            counts['active'] += 1
            counts['peak'] = max(counts['peak'], counts['active'])
        time.sleep(.02)
        with lock:
            counts['active'] -= 1
            counts['finished'] += 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(study.subprocess, 'run', run)
    old = study.driver.base.execute, study.driver.base.source_hashes
    try:
        with ThreadPoolExecutor(max_workers=3) as numerical:
            study.configure_executor(numerical)
            with ThreadPoolExecutor(max_workers=2) as controls:
                futures = []
                for variant in ('COMPACT','TRANSFER'):
                    folder = tmp_path/variant; folder.mkdir()
                    study.write(folder/'protocol.json', dict(settings=dict(compact_variant=variant,case_id=variant)))
                    jobs = [dict(name=f'job{i}',model='COMPACT',phase='training',runtime='unused',seeds=[i],
                                 mechanism=dict(capacity='evidence_mean')) for i in range(6)]
                    futures.append(controls.submit(study.driver.base.execute,jobs,folder,128))
                for future in futures:
                    future.result()
    finally:
        study.driver.base.execute, study.driver.base.source_hashes = old
    assert counts == dict(active=0,peak=3,finished=12)
    transferred = study.read(tmp_path/'TRANSFER/jobs/job0.json')
    assert transferred['entry_control'] == 'PRECISION'
    assert transferred['mechanism']['capacity'] == 'evidence'
    assert 'entry_control' not in study.read(tmp_path/'COMPACT/jobs/job0.json')
