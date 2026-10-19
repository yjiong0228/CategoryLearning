"""Independent initialization, synthetic-data isolation and joint recovery scores."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_joint_parameter_recovery as recovery
from src.Bayesian_state.workflows.analysis.model_0923_h_objective_recovery import fixture


def space():
    return {recovery.CAPACITY: [1,2,4,8,16], recovery.TRANSFER: [.01],
            **{f'parameter_{i}': [0., .5, 1.] for i in range(12)}}


def test_small_smoke_bank_avoids_single_stratum_midpoint_collision():
    points = recovery.independent_starts(space(), 6, 12)
    assert len({p['id'] for p in points}) == 6
    assert points[0]['hyperparams'][recovery.CAPACITY] == 4
    assert set(p['hyperparams'][recovery.CAPACITY] for p in points) == {1,2,4,8,16}


def test_full_starts_reproducible_and_cover_thirteen_coordinates():
    a = recovery.independent_starts(space(), 40, 12)
    assert a == recovery.independent_starts(space(), 40, 12)
    assert a != recovery.independent_starts(space(), 40, 13)
    for key, levels in space().items():
        assert set(p['hyperparams'][key] for p in a) == set(levels)


@pytest.mark.parametrize('mutation', ['fewer_free', 'free_transfer', 'duplicate'])
def test_invalid_joint_support_rejected(mutation):
    s = space()
    if mutation == 'fewer_free': s['parameter_0'] = [.5]
    if mutation == 'free_transfer': s[recovery.TRANSFER] = [.01,.02]
    if mutation == 'duplicate': s['parameter_0'] = [0.,0.,1.]
    with pytest.raises(ValueError, match='thirteen'):
        recovery.independent_starts(s, 40, 12)


def test_seed_selection_independent_of_behavior_and_input_order():
    seeds = np.array([90,10,50,30])
    assert recovery.selected_path_indices(seeds,2) == [1,3]
    reversed_seeds = seeds[::-1]
    np.testing.assert_array_equal(reversed_seeds[recovery.selected_path_indices(reversed_seeds,2)], [10,30])
    with pytest.raises(ValueError):
        recovery.selected_path_indices(np.array([1,1,2]),2)


def test_fixture_seals_no_human_answers_or_future_trials(tmp_path):
    original = tmp_path/'original'; original.mkdir()
    (original/'prepared.json').write_text(json.dumps({'subject':107,'training_trials':2,'test_trials':1,'trials':3}))
    (original/'engine.json').write_text(json.dumps({'engine':'physical configuration'}))
    arrays = {'stimulus':np.array([[1.,2.],[3.,4.]]),'categories':np.array([1,2]),
              'choices':np.array([2,2]),'feedback':np.array([0.,1.])}
    fixture(original,tmp_path/'synthetic',arrays)
    meta = json.loads((tmp_path/'synthetic/prepared.json').read_text())
    assert meta['test_trials']==0 and meta['data_role']=='synthetic_training_only'
    with np.load(tmp_path/'synthetic/training.npz') as saved, np.load(tmp_path/'synthetic/sealed_test.npz') as sealed:
        for key,value in arrays.items():
            np.testing.assert_array_equal(saved[key],value)
            assert len(sealed[key])==0


def test_optimizer_ignores_reference_nominees(monkeypatch):
    from types import SimpleNamespace
    s=space()
    base=SimpleNamespace(archive=lambda *args:None, source_hashes=lambda _: {},execute=lambda *args:None,
                         model_space=lambda *_:deepcopy(s))
    recalled=SimpleNamespace(base=base)
    cohort=SimpleNamespace(helpers=lambda:(None,None,None,recalled),configure=lambda *args:None)
    monkeypatch.setattr(recovery,'helpers',lambda:cohort)
    settings={'subjects':[107],'seed':23,'initial_candidates':40,'arms':{'POOL':{},'SEARCH':{}}}
    module=recovery.configure(settings,{})
    banks,anchors=module.starting_banks(settings,{'nominees':'invalid and intentionally unread'},False)
    other,_=module.starting_banks(settings,{},False)
    assert banks==other and banks[107,'POOL']==banks[107,'SEARCH']
    assert anchors[107,'SEARCH']==banks[107,'SEARCH'][0]


def test_generation_seeds_cannot_score_their_own_path(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    base=SimpleNamespace(archive=lambda *args:None, source_hashes=lambda _: {},execute=lambda *args:calls.append(args))
    recalled=SimpleNamespace(base=base)
    cohort=SimpleNamespace(helpers=lambda:(None,None,None,recalled),configure=lambda *args:None)
    monkeypatch.setattr(recovery,'helpers',lambda:cohort)
    module=recovery.configure({'recovery_generation_seeds':[9,12]}, {})
    with pytest.raises(ValueError,match='overlap'):
        module.base.execute([{'seeds':[9]}],Path('unused'),1)
    module.base.execute([{'seeds':[5]}],Path('unused'),1)
    assert len(calls)==1


def test_error_measures_keep_numeric_and_ordinal_distance_distinct():
    from src.Bayesian_state.evaluation.model_0923_joint_parameter_recovery import parameter_errors
    rows=parameter_errors({'a':.001,'fixed':.01},{'a':.1,'fixed':.01},{'a':[0.,.001,.1,1.],'fixed':[.01]})
    assert rows[0]['absolute_error']==pytest.approx(.099)
    assert rows[0]['normalized_absolute_error']==pytest.approx(.099)
    assert rows[0]['grid_absolute_error']==1
    assert rows[1]['variable'] is False and rows[1]['estimate_at_boundary'] is False
    with pytest.raises(ValueError):
        parameter_errors({'a':0.},{'a':.3},{'a':[0.,1.]})


def test_filter_difference_sign_and_common_seed_pairing():
    from src.Bayesian_state.evaluation.model_0923_joint_parameter_recovery import paired_scores
    from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_scores,jackknife_se
    truth={'seeds':np.array([11,12,13]),'observed_probability':np.array([[.8,.7],[.7,.8],[.9,.6]])}
    estimate={'seeds':truth['seeds'].copy(),'observed_probability':truth['observed_probability']*.8}
    result=paired_scores(estimate,truth)
    assert result['estimate_minus_truth_nll']==pytest.approx(-np.log(.8))
    assert result['difference_mc_se']==pytest.approx(0.,abs=1e-14)
    assert result['estimate_worse_than_truth_by_two_mc_se']
    estimate['observed_probability']=np.array([[.7,.6],[.5,.7],[.9,.5]])
    expected=jackknife_se(jackknife_scores(estimate['observed_probability'])-jackknife_scores(truth['observed_probability']))
    assert paired_scores(estimate,truth)['difference_mc_se']==pytest.approx(expected)
    estimate['seeds']=np.array([12,11,13])
    with pytest.raises(AssertionError): paired_scores(estimate,truth)


@pytest.mark.parametrize('invalid',[np.nan,1.1,-.1])
def test_invalid_scoring_probabilities_rejected(invalid):
    from src.Bayesian_state.evaluation.model_0923_joint_parameter_recovery import paired_scores
    item={'seeds':np.array([11,12]),'observed_probability':np.array([[.8,.7],[.7,.8]])}
    other=deepcopy(item);other['observed_probability'][0,0]=invalid
    with pytest.raises(ValueError):paired_scores(other,item)


def test_search_inputs_do_not_open_evaluation_truths(tmp_path):
    import yaml
    from src.Bayesian_state.evaluation.model_0923_h_validation import digest
    folder=tmp_path/'data/subject_107';folder.mkdir(parents=True)
    files=['prepared.json','engine.json','training.npz','sealed_test.npz']
    for name in files:(folder/name).write_text('fixture '+name)
    config=tmp_path/'fit.yaml';config.write_text(yaml.safe_dump({'data':str(tmp_path/'data')}))
    case={'replicate':0,'subject':107,'fit_config':str(config),'fit_config_sha256':digest(config),
          'prepared_sha256':{name:digest(folder/name) for name in files}}
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'cases':[case]}))
    (tmp_path/'completion.json').write_text(json.dumps({'status':'complete','protocol_sha256':digest(protocol),
                                                       'truths_sha256':'intentionally absent'}))
    settings,inputs=recovery.case_settings(tmp_path,0)
    assert settings['data']==str(tmp_path/'data')
    assert all(not key.endswith('truths.json') for key in inputs)
    (folder/'training.npz').write_text('changed data')
    with pytest.raises(ValueError,match='arrays changed'):recovery.case_settings(tmp_path,0)


def test_truth_score_refuses_unfrozen_fit_before_opening_truths(tmp_path,monkeypatch):
    fit=tmp_path/'fit';fit.mkdir()
    (fit/'nominees.json').write_text('{}')
    (fit/'completion.json').write_text(json.dumps({'status':'running'}))
    monkeypatch.setattr(recovery,'case_settings',lambda *_:({},{}))
    with pytest.raises(ValueError,match='only after'):
        recovery.score_truth(tmp_path/'missing_preparation',0,fit,tmp_path/'unused_output')
    assert not (tmp_path/'unused_output').exists()
