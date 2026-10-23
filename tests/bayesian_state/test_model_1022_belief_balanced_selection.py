"""Training eligibility and numerical uncertainty determine the frozen decision."""
import numpy as np
import pytest
from copy import deepcopy
import yaml

from src.Bayesian_state.workflows.analysis import model_1022_belief_balanced_selection as selection


def test_generation_choice_respects_likelihood_tolerance():
    rows = [dict(candidate='nll', confirmation_nll=.5, training_energy=.2),
            dict(candidate='balanced', confirmation_nll=.515, training_energy=.1),
            dict(candidate='outside', confirmation_nll=.53, training_energy=.01)]
    assert selection.choose_energy(rows, .02)['candidate'] == 'balanced'
    assert selection.choose_energy(rows, .01)['candidate'] == 'nll'


def test_energy_ties_use_likelihood_then_stable_identity():
    rows = [dict(candidate='b', confirmation_nll=.5, training_energy=.1),
            dict(candidate='a', confirmation_nll=.5, training_energy=.1),
            dict(candidate='c', confirmation_nll=.51, training_energy=.1)]
    assert selection.choose_energy(rows, .02)['candidate'] == 'a'


def test_identical_points_pass_with_zero_paired_uncertainty():
    p = np.array([[.6, .8], [.7, .9], [.5, .7], [.65, .85]])
    decision = selection.likelihood_guard(p, p.copy(), .02, 2.)
    assert decision['accepted']
    assert decision['nll_difference'] == decision['difference_mcse'] == 0


def test_numerical_uncertainty_can_reject_an_equal_point_estimate():
    original = np.full((2, 3), .8)
    proposed = np.array([[.6, .8, .8], [1., .8, .8]])
    decision = selection.likelihood_guard(original, proposed, .02, 2.)
    assert decision['nll_difference'] == pytest.approx(0, abs=1e-14)
    assert decision['difference_mcse'] > .01
    assert not decision['accepted']


def test_guard_accepts_small_stable_loss_and_rejects_large_loss():
    original = np.full((4, 3), .8)
    accepted = selection.likelihood_guard(original, np.full((4, 3), .79), .02, 2.)
    rejected = selection.likelihood_guard(original, np.full((4, 3), .7), .02, 2.)
    assert accepted['nll_difference'] == pytest.approx(np.log(.8/.79))
    assert accepted['accepted'] and not rejected['accepted']


@pytest.mark.parametrize('proposed', [np.ones((3, 3)), np.full((4, 3), 1.1), np.full((4, 3), np.nan)])
def test_invalid_guard_arrays_fail(proposed):
    with pytest.raises(ValueError):
        selection.likelihood_guard(np.full((4, 3), .8), proposed, .02, 2.)


@pytest.mark.parametrize('cohort', ['development', 'extension'])
def test_selection_scope_matches_cohort(cohort):
    policy = dict(cohort=cohort, scope=f'{cohort}_training_likelihood_constrained_generation_selection')
    assert selection.valid_scope(policy)
    policy['cohort'] = 'unknown'
    assert not selection.valid_scope(policy)


def test_guard_settings_cannot_override_inherited_cohort(tmp_path):
    policy = dict(generation_guard=dict(cohort='extension'))
    with pytest.raises(ValueError, match='exactly'):
        selection.generation_guard_policy(policy, tmp_path)
    assert selection.generation_guard_policy({}, tmp_path) is None


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('cohort', ['development', 'extension'])
def test_generation_guard_finishes_before_freezing_and_evaluation(tmp_path, monkeypatch, enabled, cohort):
    """Exercise real selection orchestration with synthetic numerical stage outputs."""
    sid = 108 if cohort == 'development' else 110
    arms = ['SUPPORT', 'SLOT', 'ADAPT']
    original = {}
    scores = []
    for arm in arms:
        points = [dict(id=name, hyperparams=dict(theta=i), mean_nll=.2+.01*i)
                  for i, name in enumerate(('old', 'new'))]
        original[sid, arm] = dict(subject=sid, arm=arm, candidate=points[0], finalists=points)
        scores.extend(dict(subject=sid, arm=arm, candidate=p['id'], confirmation_nll=p['mean_nll'],
                           training_energy=.2 if p['id']=='old' else .1) for p in points)
    source = tmp_path/'study'
    source.mkdir()
    selection.write(source/'protocol.json', dict(settings=dict(data='prepared')))
    selection.write(source/'nominees.json', dict(nominees=list(original.values())))
    templates = [dict(subject=sid, model=arm, folder='prepared', seeds=[20, 21]) for arm in arms]
    selection.write(source/'autonomous_jobs.json', dict(jobs=templates))
    generated = tmp_path/'generation'
    generated.mkdir()
    selection.write(generated/'cases.json', dict(cases=[]))
    policy = dict(protocol='test', seed=7, source_study='study', subjects=[sid], variants=arms,
        fallback='retain_original_nll_nominee', cohort=cohort,
        scope=f'{cohort}_training_likelihood_constrained_generation_selection', parallel_budget=1,
        budgets=dict(guard=dict(particles=2, seeds=2), evaluate=dict(particles=2, seeds=2)),
        autonomous_paths=4, autonomous_chunk=4, features=dict(block_width=32, temporal_weight=.5),
        training_nll_tolerance=.02, guard_mcse_multiplier=2.)
    if enabled:
        policy['generation_guard'] = dict(protocol='test_energy', seed=9, paths=4, chunk=4,
            guard_mcse_multiplier=2., maximum_confirmation_batches=1,
            acceptance='energy_difference_plus_mcse_strictly_below_zero')
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(policy))
    monkeypatch.setattr(selection, 'ROOT', tmp_path)
    monkeypatch.setattr(selection, 'sources', lambda _: {})
    monkeypatch.setattr(selection, 'inspect_generation', lambda *args: (scores, original, {}))
    monkeypatch.setattr(selection.pilot.driver.base, 'archive', lambda *args: None)
    for name in ('source_hashes', 'execute'):
        monkeypatch.setattr(selection.pilot.driver.base, name, getattr(selection.pilot.driver.base, name))
    order = []

    class Evaluated(Exception):
        pass

    def confirm(config, output, **kwargs):
        assert enabled and kwargs['before_evaluation']
        assert kwargs['policy']['cohort'] == cohort
        assert not (output.parent/'nominees.json').exists()
        assert (output.parent/'likelihood_decisions.json').exists()
        order.append('energy')
        output.mkdir()
        selection.write(output/'completion.json', dict(status='complete'))
        (output/'decisions.csv').write_text('unit-test artifact\n')
        return [dict(subject=sid, arm=arm, original='old', proposed='new',
                     energy_accepted=arm=='SUPPORT', accepted=arm=='SUPPORT',
                     selected='new' if arm=='SUPPORT' else 'old', guard_value=-.1 if arm=='SUPPORT' else .1)
                for arm in arms]

    monkeypatch.setattr(selection.energy_guard, 'confirm', confirm)

    def stage(settings, data, output, banks, phase, smoke):
        order.append(phase)
        if phase == 'guard':
            checked = {}
            for key, bank in banks.items():
                checked[key] = []
                for point in bank:
                    target = output/'guard'/key[1]/point['id']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    value = .8 if point['id']=='old' else .79
                    np.savez_compressed(target.with_suffix('.npz'), observed_probability=np.full((2, 4), value), seeds=[11, 12])
                    selection.write(target.with_suffix('.json'), dict(subject=sid, candidate=point,
                        predictions_sha256=selection.digest(target.with_suffix('.npz'))))
                    checked[key].append(dict(deepcopy(point), score_file=str(target.with_suffix('.json'))))
            return checked
        assert phase == 'evaluate'
        assert order == (['guard', 'energy', 'evaluate'] if enabled else ['guard', 'evaluate'])
        nomination = selection.read(output/'nominees.json')
        assert nomination['status']=='frozen_before_test' and not nomination['later_choices_used']
        for row in nomination['nominees']:
            expected = 'old' if enabled and row['arm']!='SUPPORT' else 'new'
            assert row['candidate']['id']==expected
            assert banks[sid, row['arm']][0]['id']==expected
        raise Evaluated

    monkeypatch.setattr(selection.pilot.driver.base, 'stage', stage)
    with pytest.raises(Evaluated):
        selection.run(config, generated, tmp_path/'output', False)
