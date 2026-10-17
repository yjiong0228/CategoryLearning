"""Equal starts, complete parameter rays and independent confirmation seeds."""
from pathlib import Path

import yaml

from src.Bayesian_state.workflows.analysis import model_0923_recalled_search_refit as workflow


def test_common_banks_preserve_all_declared_rays():
    settings = yaml.safe_load(Path('configs/shared/model0923_recalled_search_refit.yaml').read_text())
    space = workflow.base.model_space(settings, 'POOL')
    anchor = {key: levels[len(levels)//2] for key, levels in space.items()}
    frozen = dict(nominees=[dict(subject=subject, model=arm, candidate=dict(hyperparams=anchor))
        for subject in settings['subjects'] for arm in ('BASE', 'POOL', 'NEW', 'DRAW')])
    banks, required = workflow.starting_banks(settings, frozen, False)
    assert banks == workflow.starting_banks(settings, frozen, False)[0]
    for subject in settings['subjects']:
        rows = banks[subject, 'POOL']
        assert rows == banks[subject, 'SEARCH']
        assert len(rows) == len({row['id'] for row in rows}) == settings['initial_candidates']
        assert required[subject, 'SEARCH']['hyperparams'] == anchor
        for coordinate in settings['initial_ray_coordinates']:
            rays = [r['hyperparams'] for r in rows
                    if all(r['hyperparams'][k] == v for k, v in anchor.items() if k != coordinate)]
            assert {r[coordinate] for r in rays} == set(space[coordinate])


def test_training_jobs_exclude_test_arrays_and_share_independent_stage_seeds(monkeypatch, tmp_path):
    settings = yaml.safe_load(Path('configs/shared/model0923_recalled_search_refit.yaml').read_text())
    v = workflow.base.validation()
    monkeypatch.setattr(v, 'read', lambda path: {'settings': {'source_fit': 'archive'}})
    monkeypatch.setattr(v, 'source_hashes', lambda: {})
    seed_banks = []
    for stage in ('screen', 'refine', 'exchange', 'selection', 'confirmation', 'evaluate'):
        jobs = v.score_jobs(tmp_path, tmp_path / 'output', workflow.base.settings_for_scoring(settings),
            {(207, arm): [dict(id='point')] for arm in settings['arms']}, stage, False)
        assert all(job['phase'] == ('evaluate' if stage == 'evaluate' else 'training') for job in jobs)
        assert jobs[0]['seeds'] == jobs[1]['seeds']
        seeds = set(jobs[0]['seeds'])
        assert not any(seeds & previous for previous in seed_banks)
        seed_banks.append(seeds)
