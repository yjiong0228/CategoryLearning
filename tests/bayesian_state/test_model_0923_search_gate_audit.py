"""Search-gate flags and actual engine invariance at exactly closed channels."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_0923_search_gate_audit as audit


def point(global_search=1., error_gain=0., global_gain=.5):
    return {audit.C + 'global_search': global_search, audit.C + 'error_gain': error_gain,
            audit.C + 'global_search_failure_gain': global_gain, audit.C + 'failure_decay': .3,
            audit.H + 'local_structure_weight': .25}


def test_exact_gates_and_distinct_error_gain_channels():
    assert set(audit.closed_channels(point())) == {
        audit.C + 'failure_decay', audit.C + 'global_search_failure_gain', audit.H + 'local_structure_weight'}
    assert set(audit.closed_channels(point(error_gain=1.))) == {
        audit.C + 'global_search_failure_gain', audit.H + 'local_structure_weight'}
    assert audit.closed_channels(point(global_search=.8)) == {}
    assert set(audit.closed_channels(point(global_search=.8, global_gain=0.))) == {audit.C + 'failure_decay'}
    assert audit.closed_channels(point(global_search=np.nextafter(1., 0.))) == {}
    assert audit.closed_channels(point(global_search=.8, global_gain=0., error_gain=1e-100)) == {}
    with pytest.raises(ValueError, match='finite'):
        audit.closed_channels(point(error_gain=np.nan))


def test_completed_provenance_reader_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    for name in audit.REQUIRED_SOURCES:
        file = tmp_path / name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(name)
    fit = tmp_path / 'fit'; fit.mkdir()
    frozen = dict(status='frozen_before_test', smoke=False,
        source_sha256={name: audit.digest(tmp_path / name) for name in audit.REQUIRED_SOURCES},
        settings=dict(arms={'POOL': {}, 'SEARCH': {}},
                      mechanism=dict(kind='E', capacity='evidence_mean', search='accumulate')),
        nominees=[dict(subject=308, model='SEARCH', candidate=dict(id='point', hyperparams=point()))])
    file = fit / 'nominees.json'; file.write_text(json.dumps(frozen))
    completion = fit / 'completion.json'
    completion.write_text(json.dumps(dict(status='complete', smoke=False, nomination_sha256=audit.digest(file))))
    rows, summary, hashes = audit.nomination_rows(fit)
    assert len(rows) == 5 and summary[0]['closed_search_coordinates'] == 3 and len(hashes) == 5
    frozen['smoke'] = True; file.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='complete scientific'):
        audit.nomination_rows(fit)
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(FileExistsError):
        audit.run([fit], existing)


@pytest.mark.parametrize('channel', ['occurrence', 'global_fraction'])
def test_failure_decay_reaches_either_open_controller_channel(channel):
    from types import SimpleNamespace
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
        MechanisticRuleSearchHypothesisTransitionModule as Transition)
    outputs = []
    for decay in (0., .99):
        state = SimpleNamespace(outcome_pending=True, intervention_kind='B',
            intervention={'search': 'accumulate'}, failure_pressure=.7, accumulated_failure=2.,
            baseline_probability=.2, previous_feedback=1., failure_decay=decay,
            error_gain=2. if channel == 'occurrence' else 0.,
            global_search=1. if channel == 'occurrence' else .2, global_search_failure_gain=.5)
        Transition._prepare_hypothesis_transition(state)
        outputs.append(state.current_event_probability if channel == 'occurrence' else state.current_g)
    assert outputs[0] != outputs[1]


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('proposal', [False, True])
@pytest.mark.parametrize('global_search', [.6, 1.])
def test_real_engine_closed_channels_preserve_predictions_and_rule_paths(
        condition, proposal, global_search, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
        MechanisticRuleSearchHypothesisTransitionModule as Transition)
    from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import episodic_admission
    from src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal import episodic_proposal
    from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = yaml.safe_load(Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    cfg['oral_readout'] = None
    cfg['modules']['perception_mod']['kwargs'] = dict(features=4, mean=[0.]*4, std=[0.]*4, module_seed=2)
    h = cfg['modules']['hypo_transitions_mod']['kwargs']
    h.update(intervention=dict(kind='E', decay=.8, capacity='evidence_mean', search='accumulate'),
             relation_samples=128, module_seed=3, max_capacity=2, exit_rate=0., local_structure_weight=.1)
    h['search_controller'].update(baseline_probability=.8, error_gain=0., failure_decay=0.,
                                  global_search=global_search, global_search_failure_gain=0.)
    cfg['provenance']['hypothesis_similarity']['sample_count'] = 128
    changed = deepcopy(cfg); other = changed['modules']['hypo_transitions_mod']['kwargs']
    other['search_controller']['failure_decay'] = .99
    if global_search == 1.:
        other['local_structure_weight'] = .9
        other['search_controller']['global_search_failure_gain'] = 1.
    with episodic_admission(Transition, mode='POOL'), episodic_proposal(Transition, proposal=proposal):
        models = [StateModel(c, context=ModelContext(condition=condition)) for c in (cfg, changed)]
        transitions = [m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in models]
        searches = 0; feedback_state_differs = False
        for trial in range(16):
            stimulus = [.8, .3, .2, .4] if trial % 2 else [.2, .7, .8, .6]
            predictions = [m.begin_trial(stimulus) for m in models]
            for field in ('prior', 'beta'):
                np.testing.assert_array_equal(getattr(predictions[0], field), getattr(predictions[1], field))
            choices = [read_choice_probabilities_from_model(m, stimulus) for m in models]
            np.testing.assert_array_equal(*choices)
            for field in ('active', 'execution_probabilities'):
                np.testing.assert_array_equal(getattr(transitions[0], field), getattr(transitions[1], field))
            assert transitions[0].trial_rng.bit_generator.state == transitions[1].trial_rng.bit_generator.state
            assert transitions[0].current_g == transitions[1].current_g
            assert transitions[0].current_event_probability == transitions[1].current_event_probability
            searches += transitions[0].last_transition_result.selection.diagnostics['search_event']
            feedback_state_differs |= transitions[0].failure_pressure != transitions[1].failure_pressure
            for model in models:
                model.complete_trial(1, .5 if condition == 3 and trial % 3 == 0 else float(trial % 2))
        assert searches > 0 and feedback_state_differs
