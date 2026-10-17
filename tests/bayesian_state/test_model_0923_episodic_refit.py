"""Common supports, exact warm starts, shared exchange and data boundaries."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_0923_episodic_refit as w


@pytest.fixture
def settings():
    return yaml.safe_load(Path('configs/shared/model0923_episodic_refit.yaml').read_text())


def fake_nominees(settings):
    space = w.base.model_space(settings, 'BASE')
    point = {key: values[len(values)//2] for key, values in space.items()}
    rows = []
    for subject in settings['subjects']:
        for model, rate in [('ADD1', .6), ('ADD10', .3)]:
            rows.append(dict(subject=subject, model=model,
                             candidate=dict(hyperparams=dict(point, **{w.base.RATE: rate}))))
    return dict(nominees=rows)


def test_common_banks_cover_capacity_and_preserve_warm_rates(settings):
    banks, required = w.starting_banks(settings, fake_nominees(settings), False)
    assert banks == w.starting_banks(settings, fake_nominees(settings), False)[0]
    for subject in settings['subjects']:
        rows = banks[subject, 'BASE']
        assert len(rows) == settings['initial_candidates']
        assert len({row['id'] for row in rows}) == len(rows)
        assert rows[0]['hyperparams'][w.base.RATE] == .3
        assert rows[1]['hyperparams'][w.base.RATE] == .06
        capacity = w.base.validation().H_PATH+'max_capacity'
        assert set(w.base.model_space(settings, 'BASE')[capacity]) <= {r['hyperparams'][capacity] for r in rows}
        for arm in settings['arms']:
            assert rows == banks[subject, arm]
            assert required[subject, arm]['id'] == rows[0]['id']
            assert all(len(r['hyperparams']) == 14 for r in rows)


def test_exchange_compares_same_union_and_retains_reference(settings):
    banks, required = w.starting_banks(settings, fake_nominees(settings), False)
    ranked = {key: list(reversed(rows)) if key[1] == 'POOL' else rows for key, rows in banks.items()}
    exchanged = w.exchange_banks(ranked, required, 2)
    for subject in settings['subjects']:
        ids = {r['id'] for r in exchanged[subject, 'BASE']}
        assert len(ids) == 4
        assert required[subject, 'BASE']['id'] in ids
        assert all({r['id'] for r in exchanged[subject, arm]} == ids for arm in settings['arms'])
        assert all('mean_nll' not in r for r in exchanged[subject, 'BASE'])


def test_incompatible_support_rejected(settings):
    frozen = fake_nominees(settings)
    settings = deepcopy(settings)
    settings['arms']['NEW'] = dict(settings['arms']['NEW'], rates=[0.])
    with pytest.raises(ValueError, match='identical parameter supports'):
        w.starting_banks(settings, frozen, False)


def test_training_stage_never_requests_evaluation_arrays(settings, monkeypatch, tmp_path):
    v = w.base.validation()
    monkeypatch.setattr(v, 'read', lambda path: {'settings': {'source_fit': 'archive'}})
    monkeypatch.setattr(v, 'source_hashes', lambda: {})
    for stage in ('screen', 'refine', 'exchange', 'selection', 'confirmation'):
        jobs = v.score_jobs(tmp_path, tmp_path/'output', w.base.settings_for_scoring(settings),
                            {(107, 'POOL'): [dict(id='point')]}, stage, False)
        assert jobs[0]['phase'] == 'training'
        assert jobs[0]['particles'] == settings['budgets'][stage]['particles']
    seeds = [v.score_jobs(tmp_path, tmp_path/'output', w.base.settings_for_scoring(settings),
                        {(107, 'BASE'): [dict(id='point')], (107, 'DRAW'): [dict(id='point')]},
                        stage, False) for stage in ('selection', 'confirmation', 'evaluate')]
    for jobs in seeds: assert jobs[0]['seeds'] == jobs[1]['seeds']
    assert not set(seeds[0][0]['seeds']) & set(seeds[1][0]['seeds'])
    assert not set(seeds[1][0]['seeds']) & set(seeds[2][0]['seeds'])
