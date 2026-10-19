"""Bounds must retain unweighted candidates and exact choice conditioning."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.content_support import content_support_profile


def fixture():
    return dict(rule_ids=np.array([[0, 1, -1], [0, 2, -1]]),
        labels=np.array([[0, -1], [0, 1], [1, -1]]), component_counts=np.array([1, 2, 1]),
        probabilities=np.array([[[.8, .2], [.2, .8], [0, 0]], [[.1, .9], [.9, .1], [0, 0]]]),
        belief=np.array([[.4, .6, 0], [.2, .8, 0]]),
        contribution=np.array([[.5, .5, 0], [.5, .5, 0]]),
        component_weights=np.array([[[1., 0], [.25, .75], [0, 0]], [[1., 0], [1., 0], [0, 0]]]),
        prior_weights=np.array([.25, .75]), choice=1)


def test_joint_choice_conditioning_and_component_bounds():
    data = fixture(); r = content_support_profile(**data)
    pre, post = r['scores']['prior'], r['scores']['choice_conditioned']
    assert r['choice_probability'] == pytest.approx(.5)
    assert pre['support_belief_lower'] == pytest.approx(.25)
    assert pre['support_belief_upper'] == pytest.approx(.4)
    assert pre['support_decision_mass'] == pytest.approx(.53125)
    # All particles have equal predictive choice probability. Their posterior
    # weights stay unchanged while the latent rule weights change substantially.
    assert post['support_decision_mass'] == pytest.approx(.2875)
    assert post['support_decision_mass'] != pytest.approx(pre['support_decision_mass'])
    for weighting in r['scores'].values():
        assert weighting['support_availability_lower'] == 1
        assert weighting['support_availability_upper'] == 1


def test_zero_output_candidate_keeps_belief_bounds_without_invented_components():
    data = fixture()
    data['contribution'][:] = [1., 0, 0]
    data['component_weights'][:, 1] = 0
    r = content_support_profile(**data)
    for row in r['scores'].values():
        assert row['support_decision_mass'] == pytest.approx(1.)
        assert row['conflict_decision_mass'] == 0
        assert row['conflict_belief_upper'] > row['conflict_belief_lower']


def test_single_component_categories_produce_equal_bounds():
    data = fixture(); data['labels'][1] = [2, -1]; data['component_counts'][1] = 1
    data['component_weights'][0, 1] = [1, 0]
    r = content_support_profile(**data)
    for row in r['scores'].values():
        for label in ('support', 'conflict', 'undetermined'):
            for measure in ('availability', 'belief'):
                assert row[f'{label}_{measure}_lower'] == row[f'{label}_{measure}_upper']


def test_same_weighted_means_do_not_replace_particle_joint_state():
    data = fixture(); baseline = content_support_profile(**data)
    data['probabilities'][0, :2] = data['probabilities'][0, :2][::-1]
    changed = content_support_profile(**data)
    assert baseline['choice_probability'] == changed['choice_probability']
    assert baseline['scores']['prior'] == changed['scores']['prior']
    assert baseline['scores']['choice_conditioned']['support_decision_mass'] != changed['scores']['choice_conditioned']['support_decision_mass']


@pytest.mark.parametrize('fault', ['missing_weight', 'padded_component', 'bad_label', 'bad_count',
                                  'nan', 'duplicate_rule', 'impossible_choice'])
def test_invalid_components_or_states_rejected(fault):
    data = fixture()
    if fault == 'missing_weight': data['component_weights'][0, 0] = 0
    elif fault == 'padded_component': data['component_weights'][0, 0] = [.5, .5]
    elif fault == 'bad_label': data['labels'][0, 0] = 3
    elif fault == 'bad_count': data['component_counts'][0] = 0
    elif fault == 'nan': data['component_weights'][0, 0, 0] = np.nan
    elif fault == 'duplicate_rule': data['rule_ids'][0, 1] = 0
    else:
        active = data['rule_ids'] >= 0
        data['probabilities'][active] = [0., 1.]
    with pytest.raises((ValueError, AssertionError)): content_support_profile(**data)


@pytest.mark.parametrize('text,origin,whole,relative', [
    ('头长', 'original_complete', True, False),
    ('腿长但是脖子短', 'new_complete', True, True),
    ('腿长但是脖子短，头相对长一些', 'parsed_fragment', False, False),
    ('头长但是头短', 'parsed_fragment', False, False),
    ('', 'parsed_fragment', False, False)])
def test_target_scope_is_fixed_by_text_before_model_scoring(text, origin, whole, relative):
    from src.Bayesian_state.workflows.analysis.model_0923_oral_discourse_encoding import encode_text
    from src.Bayesian_state.evaluation.model_0923_discourse_content import content_targets
    report = encode_text((text, ('neck', 'tail', 'head', 'leg')))
    targets = content_targets(report)
    assert all(t['origin'] == origin for t in targets)
    assert any(t['target'] == 'whole_content' for t in targets) == whole
    assert any(t['reading'] == 'relative' for t in targets) == relative
    report['model_agreement'] = .999; report['stimulus_match'] = False
    assert content_targets(report) == targets
