"""Bind refitted oral states to the actual behavior-only SEPARATE evaluation."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import GAMMA, PREFIX, TRANSFER
from src.Bayesian_state.workflows.analysis import model_0923_separate_belief_refit as refit
from src.Bayesian_state.workflows.analysis import model_0923_separate_belief_refit_states as states
from src.Bayesian_state.utils.seeding import stable_seed


def execution(arm='POOL'):
    point = {GAMMA: .95, PREFIX + 'intervention.decay': .3, TRANSFER: .01}
    planned = dict(phase='evaluate', model=arm, candidate=dict(id=arm, hyperparams=point), particles=4, seeds=[12])
    fixed = {TRANSFER: .01}
    actual = dict(planned, structural_intervention='retained_rule_precision', retain_precision=True,
        retention_reference_inputs={}, belief_intervention=states.INTERVENTION,
        online_belief=False, probe_mechanism='SEPARATE', fixed_coordinates=fixed)
    record = {k: deepcopy(actual[k]) for k in ('model', 'candidate', 'particles', 'seeds')}; record['trials'] = 8
    precision = dict(structural_intervention='retained_rule_precision', retained_precision=True, transitions=32)
    belief = dict(enabled=True, belief_intervention=states.INTERVENTION, gamma=.95, delta=[.3], updates=28,
                  completed_feedback_timing_exact=True, belief_matches_saved_state=True)
    return planned, actual, {}, fixed, record, precision, belief


@pytest.mark.parametrize('arm', ['POOL', 'SEARCH'])
def test_validated_execution_for_both_proposal_arms(arm):
    states.checked_execution(*execution(arm))


@pytest.mark.parametrize('index,key,value', [(1, 'online_belief', True), (1, 'probe_mechanism', 'ONLINE'),
    (1, 'retain_precision', False), (1, 'phase', 'training'), (4, 'particles', 8),
    (4, 'seeds', [13]), (5, 'transitions', 28), (6, 'updates', 32),
    (6, 'gamma', .3), (6, 'delta', [.95]), (6, 'belief_matches_saved_state', False)])
def test_rejects_changed_execution(index, key, value):
    args = deepcopy(execution()); args[index][key] = value
    with pytest.raises(ValueError): states.checked_execution(*args)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    fit, source, data = tmp_path / 'fit', tmp_path / 'evaluation', tmp_path / 'data'
    folder = data / 'subject_208'; write(folder / 'prepared.json', dict(subject=208))
    hashes = {name: hashlib.sha256((states.ROOT / name).read_bytes()).hexdigest()
              for name in (states.REFIT, refit.PROBE, refit.BELIEF)}
    fixed = {TRANSFER: .01}
    settings = dict(belief_intervention=states.INTERVENTION, fixed_coordinates=fixed, subjects=[208],
        arms={'POOL': {'precision': 'fast'}, 'SEARCH': {'precision': 'fast'}}, mechanism={'kind': 'E'},
        protocol='fixture', seed=19, particle_overrides={}, budgets={'evaluate': {'seeds': 8, 'particles': 4}})
    nominees = [dict(subject=208, model=arm, candidate=execution(arm)[0]['candidate']) for arm in settings['arms']]
    common = dict(settings=settings, source_sha256=hashes, smoke=False)
    write(fit / 'protocol.json', common)
    write(fit / 'nominees.json', dict(common, status='frozen_before_test', data=str(data), nominees=nominees))
    nomination = hashlib.sha256((fit / 'nominees.json').read_bytes()).hexdigest()
    write(fit / 'completion.json', dict(status='complete', test_choices_used=False, smoke=False, nomination_sha256=nomination))
    write(fit / 'separate_refit_protocol.json', dict(belief_intervention=states.INTERVENTION,
                                                   input_sha256={}, fixed_coordinates=fixed))
    plans = []
    for nominee in nominees:
        arm = nominee['model']
        for i in range(8):
            planned, actual, _, _, record, precision, belief = execution(arm)
            name = f'evaluate_S208_{arm}_{arm}_C{i:03d}'
            seed = stable_seed(dict(protocol='fixture', base=19, stage='evaluate', subject=208, repeat=i))
            target = source / 'conditional' / name
            planned.update(name=name, smoke=False, seeds=[seed], folder=str(folder), target=str(target),
                arm_settings=settings['arms'][arm], mechanism=settings['mechanism'], source_sha256=hashes)
            actual.update(planned); record.update(seeds=[seed])
            write(source / 'jobs' / (name + '.json'), actual)
            for suffix, content in [('.json', record), ('.precision.json', precision), ('.belief.json', belief)]:
                write(target.with_suffix(suffix), content)
            plans.append(planned)
    write(source / 'protocol.json', dict(fit=str(fit), source_sha256=hashes, nomination_sha256=nomination, jobs=plans))
    write(source / 'completion.json', dict(status='complete', jobs=16, nomination_sha256=nomination))
    original_helper = states.helper
    monkeypatch.setattr(refit, 'reference_inputs', lambda *args: {})
    monkeypatch.setattr(states, 'helper', lambda name, relative:
                        refit if relative == states.REFIT else original_helper(name, relative))
    return dict(subjects=[208], arms=['POOL', 'SEARCH'], seeds_per_arm=8, sample_role='development', workers=16), source, fit


def test_preserves_all_evaluation_seeds_and_both_smoke_arms(tmp_path, monkeypatch):
    settings, source, _ = fixture(tmp_path, monkeypatch)
    files, metadata = states.source_jobs(settings, source, False)
    assert len(files) == 16 and metadata['scientific_scope'] is True
    assert metadata['parameter_estimation_scope'] == 'prefix' and len(metadata['selected_parameters']) == 2
    assert len(states.source_jobs(settings, source, True)[0]) == 2
    context, note = states.estimation_context(metadata)
    assert context['window_definition'] == 'original_chronological_boundary' and 'SEPARATE' in note


@pytest.mark.parametrize('change', ['later_fit', 'missing_seed', 'source_hash', 'wrong_nomination'])
def test_rejects_invalid_selection_provenance(tmp_path, monkeypatch, change):
    settings, source, fit = fixture(tmp_path, monkeypatch)
    if change == 'missing_seed':
        next((source / 'jobs').glob('*.json')).unlink()
    else:
        path = fit / ('completion.json' if change == 'later_fit' else 'nominees.json')
        value = json.loads(path.read_text())
        if change == 'later_fit': value['test_choices_used'] = True
        elif change == 'source_hash': value['source_sha256'][refit.BELIEF] = 'wrong'
        else: value['nominees'][0]['candidate']['id'] = 'changed'
        write(path, value)
    with pytest.raises(ValueError): states.source_jobs(settings, source, False)


def test_short_fixture_cannot_be_presented_as_scientific_oral_evidence(tmp_path, monkeypatch):
    settings, source, _ = fixture(tmp_path, monkeypatch)
    _, metadata = states.source_jobs(settings, source, False)
    metadata['scientific_scope'] = False
    with pytest.raises(ValueError): states.estimation_context(metadata)
