"""Audit local-reason interpretation with explicit synthetic counterexamples.

Run --output NEW_DIRECTORY. The existing state collector and constraint
definitions are exercised unchanged. These examples distinguish joint use
inside a state from uncertainty across states, and stimulus compatibility
from an effect on choice. They do not introduce a fitted oral score, modify
the catalogue, or estimate participant parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.oral.constraint_alignment import evaluate_component
from src.Bayesian_state.evaluation.oral.decision_basis import DecisionBasisCollector
from src.Bayesian_state.evaluation.oral.predicate_alignment import predicate_fractions
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.model.modules.readout.report_space import component_tokens

ROOT = Path(__file__).resolve().parents[4]


def contribution_example(mode: str) -> tuple[dict, dict, dict]:
    """Return equal-marginal snapshots with different within-state rule use.

    Both rules and both particles are retained in each example. Rule 0 uses
    dimension 0; rule 2 uses dimension 2. At the current stimulus both assign
    probability 0.9 to choice 1. The synthetic counterfactual table assigns
    0.1 when the corresponding dimension crosses its threshold. All other
    quantities are held fixed; this is a readout example, not a learning run.
    """
    if mode not in ('uncertain_single_rule', 'simultaneous_mixture'):
        raise ValueError('Unknown synthetic use pattern')
    space = build_continuous_hypothesis_space(4, 2)
    rules = [0, 2]
    active = np.zeros((2, len(space)))
    active[:, rules] = 1.
    weights = active / 2
    if mode == 'uncertain_single_rule':
        weights = np.zeros_like(active)
        weights[0, rules[0]] = weights[1, rules[1]] = 1.
    likelihood = np.zeros((2, len(space), 2))
    likelihood[:, rules, :] = [.9, .1]
    snapshot = dict(weights=np.array([.5, .5]), choice_weights=weights,
        rule_choice_probabilities=likelihood, perceived_stimulus=np.full((2, 4), .2),
        active=active, prediction=np.einsum('ph,phc->pc', weights, likelihood),
        belief=active / 2, beta=np.ones_like(active), search=np.zeros(2), admission_event=np.zeros(2))
    collector = DecisionBasisCollector(space, np.array([1]))
    collector(0, 'predictive', snapshot)
    arrays = collector.arrays()
    tokens = tuple(token for rule in rules for token in
                   component_tokens(space[rule].categories[0].components[0].A,
                                    space[rule].categories[0].components[0].b))
    whole, partial = [], []
    for rule in rules:
        component = space[rule].categories[0].components[0]
        whole.append(evaluate_component(tokens, component.A, component.b).functional == 'support')
        partial.append(predicate_fractions(tokens, component.A, component.b)[1, 0])
    joint = arrays['choice_component_joint_mass'][0, rules, 0]
    posterior_rule_mass = joint / arrays['choice_probability'][0]
    # The rule-specific counterfactual changes are known by construction.
    # Averaging them first erases whether both influences occur in one state.
    effects = .8 * weights[:, rules]
    summary = dict(example=mode, choice_probability=float(arrays['choice_probability'][0]),
        whole_functional_support=float(posterior_rule_mass @ whole),
        mean_predicate_functional_support=float(posterior_rule_mass @ partial),
        marginal_effect_dimension0=float(snapshot['weights'] @ effects[:, 0]),
        marginal_effect_dimension2=float(snapshot['weights'] @ effects[:, 1]),
        mean_effective_rule_count=float(snapshot['weights'] @ (1 / np.square(weights).sum(axis=1))),
        probability_both_effects_present_within_state=float(snapshot['weights'] @ np.all(effects > 0, axis=1)),
        both_rules_active_probability=1.)
    return arrays, snapshot, summary


def logical_examples() -> pd.DataFrame:
    """Retain scope and relevance failures of two geometric shortcuts."""
    # Synthetic predicates have explicit numerical meaning. No participant's
    # ambiguous long/short wording is reinterpreted by this method check.
    neck = np.array([[1., 0., 0., 0.]])
    head = np.array([[0., 0., 1., 0.]])
    examples = [
        ('single_reported_reason', neck, [.5], neck, [.5]),
        ('unused_extra_reported_reason', neck, [.5], np.vstack((neck, head)), [.5, .5]),
        ('joint_rule_with_two_reasons', np.vstack((neck, head)), [.5, .5],
         np.vstack((neck, head)), [.5, .5]),
        ('narrower_current_description', neck, [.5], neck, [.25]),
    ]
    rows = []
    x = np.full(4, .2)
    for name, rule_A, rule_b, oral_A, oral_b in examples:
        rule_b, oral_b = np.array(rule_b), np.array(oral_b)
        oral_tokens = component_tokens(oral_A, oral_b)
        rule_tokens = component_tokens(rule_A, rule_b)
        direct = evaluate_component(oral_tokens, rule_A, rule_b)
        reverse = evaluate_component(rule_tokens, oral_A, oral_b)
        def predict(point):
            return .1 + .8 * bool(np.all(rule_A @ point <= rule_b))
        effects = []
        for dimension in (0, 2):
            flipped = x.copy(); flipped[dimension] = .8
            effects.append(predict(x) - predict(flipped))
        rows.append(dict(example=name, rule_implies_report=direct.functional,
            report_implies_rule=reverse.functional, report_true_at_current_stimulus=bool(np.all(oral_A @ x <= oral_b)),
            rule_true_at_current_stimulus=bool(np.all(rule_A @ x <= rule_b)),
            dimension0_flip_effect=effects[0], dimension2_flip_effect=effects[1]))
    return pd.DataFrame(rows)


def redundant_reasons() -> dict:
    """Show why single-feature removal alone cannot define relevance."""
    # Rows are whether each reason holds. A noisy OR retains high choice
    # probability when either reason remains; changing both lowers it.
    probabilities = {(False, False): .1, (False, True): .9,
                     (True, False): .9, (True, True): .9}
    current = probabilities[True, True]
    return dict(example='redundant_or_reasons', current_probability=current,
        remove_first_effect=current - probabilities[False, True],
        remove_second_effect=current - probabilities[True, False],
        remove_both_effect=current - probabilities[False, False])


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new method-audit directory')
    original, selected_state, first = contribution_example('uncertain_single_rule')
    mixture, mixture_state, second = contribution_example('simultaneous_mixture')
    assert set(original) == set(mixture)
    for key in original:
        np.testing.assert_array_equal(original[key], mixture[key])
    assert not np.array_equal(selected_state['choice_weights'], mixture_state['choice_weights'])
    assert first['probability_both_effects_present_within_state'] == 0.
    assert second['probability_both_effects_present_within_state'] == 1.
    assert first['whole_functional_support'] == second['whole_functional_support'] == 0.
    assert first['mean_predicate_functional_support'] == second['mean_predicate_functional_support'] == .5
    logic, redundant = logical_examples(), redundant_reasons()
    assert list(logic.rule_implies_report) == ['support', 'undetermined', 'support', 'undetermined']
    assert logic.report_implies_rule.eq('support').all()
    assert redundant['remove_first_effect'] == redundant['remove_second_effect'] == 0.
    assert np.isclose(redundant['remove_both_effect'], .8)
    output.mkdir(parents=True)
    pd.DataFrame([first, second]).to_csv(output / 'state_mixture_examples.csv', index=False)
    logic.to_csv(output / 'logical_examples.csv', index=False)
    pd.DataFrame([redundant]).to_csv(output / 'redundancy_example.csv', index=False)
    np.savez_compressed(output / 'counterexample_arrays.npz', **original,
        single_state_choice_weights=selected_state['choice_weights'],
        mixture_state_choice_weights=mixture_state['choice_weights'])
    sources = [Path(__file__), ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis.py',
        ROOT / 'src/Bayesian_state/evaluation/oral/constraint_alignment.py',
        ROOT / 'src/Bayesian_state/evaluation/oral/predicate_alignment.py',
        ROOT / 'src/Bayesian_state/model/modules/readout/report_space.py',
        ROOT / 'src/Bayesian_state/hypothesis_space/spaces/continuous.py']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(status='complete', scientific_scope='synthetic_method_diagnostic',
        human_reports_used=False, parameters_fitted=False, new_scientific_score=False,
        identical_saved_arrays=sorted(original), identical_array_elements=sum(x.size for x in original.values()),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        artifacts={p.name: digest(p) for p in output.iterdir()},
        implications=[
            'Marginal local-basis mass does not identify simultaneous rule use inside a particle.',
            'Whole single-component support and equal-predicate coverage do not recover that lost association.',
            'Reverse implication and factual stimulus compatibility can credit an unused reported feature.',
            'Single-feature removal can miss redundant reasons; local effects require a declared contrast.',
            'Future joint-reason readouts require particle-specific contributions before marginalization.'])
    (output / 'completion.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    result = run(parser.parse_args().output)
    print({key: result[key] for key in ('status', 'scientific_scope', 'identical_array_elements')})
