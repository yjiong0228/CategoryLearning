"""Preserve fitting budgets and identify the retained-precision intervention."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_0923_reentry_precision_refit as workflow


def settings():
    original = yaml.safe_load(Path('configs/shared/model0923_recalled_search_refit.yaml').read_text())
    revised = yaml.safe_load(Path('configs/shared/model0923_reentry_precision_refit_s208.yaml').read_text())
    return revised, original


def test_retention_keeps_original_seed_namespace_and_scientific_budgets():
    revised, original = settings()
    workflow.validate_settings(revised, original)
    assert revised['protocol'] == original['protocol']
    assert revised['budgets'] == original['budgets']
    assert revised['subjects'] == [208]


@pytest.mark.parametrize('field', ['protocol', 'seed', 'budgets', 'data', 'arms',
                                  'parameter_support', 'initial_candidates', 'initial_ray_coordinates'])
def test_changed_scientific_setting_rejected(field):
    revised, original = settings()
    revised[field] = 'changed'
    with pytest.raises(ValueError, match='supports, budgets, seeds or task'):
        workflow.validate_settings(revised, original)


@pytest.mark.parametrize('subjects', [[], [208, 208], [306]])
def test_new_or_duplicate_subjects_rejected(subjects):
    revised, original = settings()
    revised['subjects'] = subjects
    with pytest.raises(ValueError, match='subjects'):
        workflow.validate_settings(revised, original)


def test_explicit_intervention_required():
    revised, original = settings()
    revised.pop('structural_intervention')
    with pytest.raises(ValueError, match='intervention'):
        workflow.validate_settings(revised, original)


@pytest.mark.parametrize('changed', ['nomination', 'executing_source'])
def test_reference_binding_rejects_implementation_changes(changed):
    revised, original = settings()
    protocol = dict(settings=original, source_sha256={'core.py': 'original'})
    frozen = dict(settings=original, source_sha256={'core.py': 'original'},
                  smoke=False, status='frozen_before_test')
    if changed == 'nomination':
        frozen['source_sha256']['core.py'] = 'different'
    files = {'protocol.json': protocol, 'nominees.json': frozen,
             'completion.json': dict(status='complete', smoke=False, nomination_sha256='nomination')}
    v = SimpleNamespace(read=lambda p: files[p.name],
                        digest=lambda p: 'nomination' if p.name == 'nominees.json' else 'changed')
    recalled = SimpleNamespace(base=SimpleNamespace(validation=lambda: v))
    with pytest.raises(ValueError, match='implementation'):
        workflow.reference_inputs(recalled, revised)


def test_wrapper_tags_both_arms_without_changing_candidates_or_caller_jobs():
    calls = []
    base = SimpleNamespace(source_hashes=lambda v: {'old': 'hash'}, archive=lambda *a: None)
    recalled = SimpleNamespace(base=base, configure=lambda: None,
                               execute=lambda jobs, output, budget: calls.append((jobs, output, budget)))
    workflow.configure(recalled, {'reference.json': 'sha'})
    jobs = [dict(model=arm, phase=phase, candidate={'id': 'same'}, seeds=[1, 2])
            for arm in ['POOL', 'SEARCH'] for phase in ['training', 'evaluate', 'autonomous']]
    before = deepcopy(jobs)
    base.execute(jobs, Path('output'), 64)
    assert jobs == before
    tagged, output, budget = calls[0]
    assert (output, budget) == (Path('output'), 64)
    for old, new in zip(jobs, tagged):
        assert {key: new[key] for key in old} == old
        assert new['retain_precision'] is True
        assert new['structural_intervention'] == workflow.INTERVENTION
        assert new['retention_reference_inputs'] == {'reference.json': 'sha'}
    v = SimpleNamespace(digest=lambda path: 'checked')
    assert set([workflow.ENTRY, workflow.RECALLED, workflow.MECHANISM]) <= set(base.source_hashes(v))
    assert recalled.ENTRY == workflow.ENTRY
