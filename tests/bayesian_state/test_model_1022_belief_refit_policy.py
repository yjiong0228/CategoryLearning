"""Warm anchors and named mechanisms remain explicit in independent refits."""
from pathlib import Path
import json
from copy import deepcopy

import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_1022_belief_comparison as pilot


def warm_source(root: Path, name: str, subject: int) -> None:
    p = root / name
    p.mkdir()
    pilot.write(p/'nominees.json', dict(smoke=False, status='frozen_before_test', later_choices_used=False,
                                      oral_used=False, nominees=[dict(subject=subject, arm='SUPPORT')]))
    pilot.write(p/'protocol.json', dict(settings=dict(data='training_data')))
    pilot.write(p/'completion.json', dict(status='complete'))
    (p/'summary').mkdir()
    pilot.write(p/'summary/completion.json', dict(nominee_sha256=pilot.digest(p/'nominees.json')))


def test_multiple_sources_preserve_order_and_legacy_single_source(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    warm_source(tmp_path, 'first', 108)
    warm_source(tmp_path, 'second', 306)
    a, _ = pilot.load_warm_nominees(dict(warm_comparison='first'), 'training_data')
    b, _ = pilot.load_warm_nominees(dict(warm_comparisons=['first']), 'training_data')
    assert a == b
    both, inputs = pilot.load_warm_nominees(dict(warm_comparisons=['first', 'second']), 'training_data')
    assert [r['subject'] for r in both['nominees']] == [108, 306]
    assert len(inputs) == 8
    assert pilot.load_warm_nominees({}, 'training_data') == (None, {})


def test_warm_source_mismatch_and_changed_nomination_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    warm_source(tmp_path, 'first', 108)
    policy = dict(warm_comparison='first')
    with pytest.raises(ValueError, match='training-only'):
        pilot.load_warm_nominees(policy, 'different_data')
    p = tmp_path/'first/nominees.json'
    value = pilot.read(p)
    value['nominees'][0]['subject'] = 306
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='intact'):
        pilot.load_warm_nominees(policy, 'training_data')


@pytest.mark.parametrize('policy', [
    dict(warm_comparison='a', warm_comparisons=['b']),
    dict(warm_comparisons=['a', 'a']), dict(warm_comparisons='a'),
])
def test_ambiguous_warm_configuration_fails(policy):
    with pytest.raises(ValueError):
        pilot.load_warm_nominees(policy, 'training_data')


@pytest.mark.parametrize('arms, expected', [
    (['SUPPORT', 'BELIEF'], True), (['SUPPORT', 'BELIEF', 'SLOT'], True),
    (['SUPPORT', 'SLOT', 'ADAPT'], True), (['SUPPORT', 'ENTRY', 'OUTCOME'], True),
    (['SLOT', 'ADAPT'], False), (['SUPPORT', 'SUPPORT'], False),
    (['SUPPORT', 'UNKNOWN'], False), ([], False), (None, False),
])
def test_registered_model_names_have_one_support_reference(arms, expected):
    assert pilot.valid_variants(arms) is expected


@pytest.mark.parametrize('confirm_anchor', [None, False, True])
def test_original_anchor_reaches_confirmation_after_losing_screen(tmp_path, monkeypatch, confirm_anchor):
    """A forced anchor survives selection even without an additional warm study."""
    from src.Bayesian_state.optimization.search import adaptive_proposals
    monkeypatch.setattr(pilot, 'ROOT', tmp_path)
    monkeypatch.setattr(pilot, 'sources', lambda _: {})
    monkeypatch.setattr(pilot.driver.base, 'source_hashes', pilot.driver.base.source_hashes)
    monkeypatch.setattr(pilot.driver.base, 'execute', pilot.driver.base.execute)
    monkeypatch.setattr(pilot.driver.base, 'archive', lambda *args: None)
    monkeypatch.setattr(pilot.driver, 'model_space', lambda *args: {'theta': [1, 2]})
    monkeypatch.setattr(adaptive_proposals, 'initial_points', lambda *args, **kwargs: [{'theta': 1}, {'theta': 2}])
    source = tmp_path/'fit/S108_TRANSFER'
    source.mkdir(parents=True)
    pilot.write(source/'nominees.json', dict(smoke=False, settings={}, nominee=dict(candidate=dict(hyperparams={'theta': 1}))))
    pilot.write(source/'completion.json', dict(status='complete', test_choices_used=False,
                                             nomination_sha256=pilot.digest(source/'nominees.json')))
    (tmp_path/'data/subject_108').mkdir(parents=True)
    pilot.write(tmp_path/'data/subject_108/prepared.json', dict(condition=1))
    (tmp_path/'base.yaml').write_text(yaml.safe_dump(dict(arms=dict(COMPACT={}), data='data')))
    policy = dict(protocol='test', seed=1, subjects=[108], variants=['SUPPORT', 'SLOT', 'ADAPT'],
                  entry_belief='by_variant', base_config='base.yaml', warm_fit='fit', parallel_budget=1,
                  initial_candidates=2, refinement_rounds=0, finalists_per_arm=1, budgets={},
                  autonomous_paths=4, autonomous_chunk=4)
    if confirm_anchor is not None:
        policy['confirm_training_anchors'] = confirm_anchor
    (tmp_path/'policy.yaml').write_text(yaml.safe_dump(policy))

    class StopAfterConfirmation(Exception):
        pass

    def stage(settings, data, output, banks, name, smoke):
        if name == 'screen':
            return {key: sorted([dict(deepcopy(point), mean_nll=3-point['hyperparams']['theta'])
                                  for point in points], key=lambda point: point['mean_nll'])
                    for key, points in banks.items()}
        assert name == 'confirmation'
        for points in banks.values():
            assert any(point['hyperparams']['theta'] == 2 for point in points)
            assert any(point['hyperparams']['theta'] == 1 for point in points) is bool(confirm_anchor)
        raise StopAfterConfirmation

    monkeypatch.setattr(pilot.driver.base, 'stage', stage)
    with pytest.raises(StopAfterConfirmation):
        pilot.run(tmp_path/'policy.yaml', tmp_path/'output', False)
