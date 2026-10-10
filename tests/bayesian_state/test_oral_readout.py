"""Analytic report probabilities and causal, behavior-preserving integration."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pickle

import numpy as np
import pytest
import yaml

from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_axis_pair_overlap_space
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.config import ModelConfig
from src.Bayesian_state.model.modules.readout.oral import OralReadout, OralReadoutParameters
from src.Bayesian_state.model.modules.readout.report_space import EMPTY_REPORT, OTHER_REPORT, component_tokens, report_code
from src.Bayesian_state.model.modules.readout.choice import (
    CHOICE_READOUT_SAMPLE, CHOICE_READOUT_MAP, CHOICE_READOUT_EXPECTATION,
    resolve_choice_readout_config, resolve_output_noise_config,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ('neck', 'leg', 'tail', 'head')
X = np.array([.8, .3, .2, .4])


def _parameters(**changes):
    base = OralReadoutParameters(.8, 0., 0., .75, .1, .2)
    return replace(base, **changes)


@pytest.fixture(scope='module')
def reader():
    return OralReadout(build_axis_pair_overlap_space(), _parameters())


def _code(text):
    report = encode_report(text, FEATURES)
    assert report.status == 'coded'
    return report_code(report.tokens)


def test_omission_and_one_extra_match_hand_calculated_probabilities(reader):
    probabilities, extras = reader.distribution(117, 4, X)
    expected = {
        '脖子长，且脖子长于尾巴。': .8 * .8 * .8,
        '脖子长。': .8 * .2 * .8,
        '脖子长，且腿短。': .8 * .2 * .2 / 2,
        '脖子长，且脖子长于尾巴，且腿短。': .8 * .8 * .2 / 2,
    }
    for text, mass in expected.items():
        assert probabilities[reader.codes.index(_code(text))] == pytest.approx(mass)
    assert probabilities[reader.codes.index(EMPTY_REPORT)] == pytest.approx(.2 * .2 * .8)
    assert extras.sum() == pytest.approx(.2)
    assert np.all(extras <= probabilities + 1e-15)
    assert probabilities.sum() == pytest.approx(1.)
    # Every supplementary phrase concerns just one unused dimension.
    assert probabilities[reader.codes.index(_code('脖子长，且腿长。'))] == 0
    double_extra = _code('脖子长，且腿短，且头短。')
    assert dict(zip(reader.codes, probabilities)).get(double_extra, 0.) == 0.


def test_extras_follow_same_perception_without_changing_vocab_or_core(reader):
    first, _ = reader.distribution(117, 4, X)
    changed = X.copy(); changed[1] = .9
    second, _ = reader.distribution(117, 4, changed)
    core = reader.codes.index(_code('脖子长，且脖子长于尾巴。'))
    short = reader.codes.index(_code('脖子长，且腿短。'))
    long = reader.codes.index(_code('脖子长，且腿长。'))
    assert first[core] == second[core]
    assert first[short] == second[long] > 0
    assert first[long] == second[short] == 0


@pytest.mark.parametrize('mention,extra,other', [(0., 0., 0.), (0., 1., 0.), (1., 1., 0.), (1., 0., 1.), (.8, .2, .02)])
def test_parameter_boundaries_conserve_mass_and_keep_a_common_vocabulary(reader, mention, extra, other):
    parameters = _parameters(mention_probability=mention, extra_probability=extra, other_probability=other)
    instance = OralReadout(reader.space, parameters)
    assert instance.codes == reader.codes
    for h in (0, 100, 117, 127):
        for choice in range(1, 5):
            mass, additions = instance.distribution(h, choice, X)
            assert np.isfinite(mass).all() and np.all(mass >= 0)
            np.testing.assert_allclose(mass.sum(), 1., rtol=0, atol=1e-12)
            assert np.all(additions <= mass + 1e-15)
            assert mass[instance.codes.index(OTHER_REPORT)] == other
    # Exactly at .5, unused features generate no long/short assertion.
    _, additions = instance.distribution(117, 4, np.array([.8, .5, .2, .5]))
    assert additions.sum() == 0


def test_nearest_disconnected_component_and_equal_distance_tie():
    space = build_continuous_hypothesis_space(4, 2)
    h = next(h for h in space if h.family == 'univariate_center_band')
    category = next(i for i, c in enumerate(h.categories) if len(c.components) == 2)
    r = OralReadout(space, _parameters(mention_probability=1., extra_probability=0.))
    codes = [report_code(component_tokens(p.A, p.b)) for p in h.categories[category].components]
    dim = int(np.flatnonzero(h.categories[category].components[0].A.any(axis=0))[0])
    for value in (.1, .9):
        x = np.full(4, .5); x[dim] = value
        p, _ = r.distribution(h.index, category + 1, x)
        truth = [bool(np.all(part.A @ x <= part.b)) for part in h.categories[category].components]
        np.testing.assert_array_equal(p[[r.codes.index(code) for code in codes]], truth)
    p, _ = r.distribution(h.index, category + 1, np.full(4, .5))
    np.testing.assert_allclose(p[[r.codes.index(code) for code in codes]], [.5, .5])


def test_joint_rule_source_conditions_on_choice_and_unknown_text_is_not_other():
    space = build_continuous_hypothesis_space(4, 2)
    r = OralReadout(space, _parameters(mention_probability=1., extra_probability=0.))
    result = r.predict(hypothesis_indices=(0, 1), rule_weights=np.array([.6, .4]),
                       category_probabilities=np.array([[.9, .1], [.2, .8]]),
                       choice=1, perceived_stimulus=X)
    np.testing.assert_allclose(result.rule_weights_given_choice, [.54 / .62, .08 / .62])
    assert result.choice_probability == pytest.approx(.62)
    assert result.probability('an unparsed weighted-sum sentence') == 0
    with pytest.raises(ValueError, match='zero-probability'):
        result.extra_given_report('unknown')
    assert result.sample(np.random.default_rng(3)) == result.sample(np.random.default_rng(3))


def _config(condition, tmp_path):
    cfg = yaml.safe_load((ROOT / f'configs/exp123/model_struct/model_0923_cond{condition}_B0.yaml').read_text())
    cfg['partition']['kwargs'].update(similarity_n_samples=64, similarity_cache_dir=str(tmp_path))
    cfg['modules']['perception_mod']['kwargs'] = {'features': 4, 'mean': [0.] * 4, 'std': [0.] * 4}
    cfg['modules']['hypo_transitions_mod']['kwargs']['module_seed'] = 0
    return cfg


def _choose(model, rng, method=None):
    config = resolve_choice_readout_config(None, model.engine_config)
    if method is not None:
        config['method'] = method
    return model.predict_choice(choices=[], feedback=[], choice_readout_config=config,
                                output_noise_config=resolve_output_noise_config(None, model.engine_config),
                                rng=rng, sticky_state={}, post_error_lapse_state=0.)


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_live_readout_observes_timing_and_preserves_behavior_and_learning(condition, tmp_path):
    config = _config(condition, tmp_path)
    plain = deepcopy(config); del plain['oral_readout']
    model = StateModel(config, context=ModelContext(condition=condition))
    control = StateModel(plain, context=ModelContext(condition=condition))
    with pytest.raises(RuntimeError, match='pending trial'):
        model.predict_oral(1)
    for trial in range(3):
        model.begin_trial(X); control.begin_trial(X)
        with pytest.raises(RuntimeError, match='predict_choice'):
            model.predict_oral(1)
        decision = _choose(model, np.random.default_rng(trial))
        expected = _choose(control, np.random.default_rng(trial))
        np.testing.assert_array_equal(decision.observed_probabilities, expected.observed_probabilities)
        before = pickle.dumps({k: v.state_dict() for k, v in model.engine.modules.items()})
        prior = model.engine.prior.copy()
        report = model.predict_oral(1)
        repeated = model.predict_oral(1)
        np.testing.assert_array_equal(report.probabilities, repeated.probabilities)
        assert report.choice_probability == pytest.approx(decision.observed_probabilities[0])
        np.testing.assert_allclose(report.probabilities.sum(), 1.)
        np.testing.assert_array_equal(model.engine.prior, prior)
        assert pickle.dumps({k: v.state_dict() for k, v in model.engine.modules.items()}) == before
        with pytest.raises(RuntimeError, match='already fixed'):
            _choose(model, np.random.default_rng(trial))
        with pytest.raises(ValueError, match='different reported choice'):
            model.predict_oral(2)
        with pytest.raises(ValueError, match='feedback choice differs'):
            model.complete_trial(2, 1.)
        feedback = .5 if condition == 3 and trial == 0 else float(trial % 2)
        actual_post, _, log = model.complete_trial(1, feedback)
        expected_post, _, _ = control.complete_trial(1, feedback)
        np.testing.assert_array_equal(actual_post, expected_post)
        assert log['oral_readout']['timing'] == 'after_choice_before_feedback'
        with pytest.raises(RuntimeError, match='before feedback'):
            model.predict_oral(1)


@pytest.mark.parametrize('method', [CHOICE_READOUT_SAMPLE, CHOICE_READOUT_MAP, CHOICE_READOUT_EXPECTATION])
def test_oral_uses_actual_decision_weights_including_sampled_rule(method, tmp_path):
    model = StateModel(_config(2, tmp_path), context=ModelContext(condition=2))
    model.begin_trial(X)
    decision = _choose(model, np.random.default_rng(42), method)
    snapshot = decision.rule_snapshot
    report = model.predict_oral(1)
    mass = snapshot.rule_weights * snapshot.category_probabilities[:, 0]
    np.testing.assert_allclose(report.rule_weights_given_choice, mass / mass.sum())
    if method in (CHOICE_READOUT_SAMPLE, CHOICE_READOUT_MAP):
        assert np.count_nonzero(report.rule_weights_given_choice) == 1


@pytest.mark.parametrize('change', ['unknown_method', 'invalid_extra', 'unknown_parameter'])
def test_invalid_oral_config_is_rejected(change, tmp_path):
    config = _config(2, tmp_path)
    if change == 'unknown_method': config['oral_readout']['method'] = 'post_feedback'
    elif change == 'invalid_extra': config['oral_readout']['kwargs']['extra_probability'] = 1.1
    else: config['oral_readout']['kwargs']['individual_report_stage'] = True
    with pytest.raises((ValueError, TypeError)):
        ModelConfig.from_mapping(config)


@pytest.mark.parametrize('choice', [0, 5, 1.1, True])
def test_invalid_choice_rejected(reader, choice):
    with pytest.raises(ValueError, match='one-based integer'):
        reader.distribution(117, choice, X)


def test_body_naming_keeps_direction_and_does_not_change_rule_geometry(reader):
    parameters = _parameters(mention_probability=1., extra_probability=0.,
                             body_naming_probability=.5, body_reference_width=.1)
    named = OralReadout(reader.space, parameters)
    before = [part.b.copy() for part in reader.space[117].categories[3].components]
    p, _ = named.distribution(117, 4, X)
    body = _code('躯干短于脖子，且脖子长于尾巴。')
    plain = _code('脖子长，且脖子长于尾巴。')
    naming_chance = .5 * np.exp(-.5 * ((.5 - .75) / .1) ** 2)
    assert p[named.codes.index(body)] == pytest.approx(naming_chance)
    assert p[named.codes.index(plain)] == pytest.approx(1 - naming_chance)
    wrong = _code('脖子短于躯干，且脖子长于尾巴。')
    assert p[named.codes.index(wrong)] == 0
    for expected, part in zip(before, reader.space[117].categories[3].components):
        np.testing.assert_array_equal(part.b, expected)


def test_output_lapse_requires_explicit_joint_model(tmp_path):
    model = StateModel(_config(2, tmp_path), context=ModelContext(condition=2))
    model.begin_trial(X)
    decision = _choose(model, np.random.default_rng(0))
    # Exercise the guard independently of output-noise policy configuration.
    model._oral_choice_prediction = replace(decision, output_lapse=.1)
    with pytest.raises(ValueError, match='explicit joint observation model'):
        model.predict_oral(1)
