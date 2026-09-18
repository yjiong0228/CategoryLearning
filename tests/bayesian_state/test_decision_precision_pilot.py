"""Decision, seed independence and protected-batch contracts of the new pilot."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import pilot_model_0826_decision_precision as pilot


def arrays(q, seeds=None):
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = np.repeat(q[:, None], 6, axis=1)
    return {'probabilities': np.stack([q, 1-q], axis=-1),
            'seeds': np.arange(len(q)) if seeds is None else np.asarray(seeds),
            'observed': np.zeros(q.shape[1], dtype=int),
            'mask': np.array([False]+[True]*(q.shape[1]-1))}


def diagnostic(bank, primary='best'):
    return pilot.decision_diagnostics(bank, primary, .005, .05/3, 3000, 91)


def test_noisy_clearly_poor_candidates_do_not_block_good_selection():
    bank = {'best': arrays([.8]*16),
            'bad_a': arrays([.05]*8+[.25]*8),
            'bad_b': arrays([.3]*8+[.06]*8)}
    result = diagnostic(bank)
    assert result['status'] == 'acceptable_within_bank'
    assert result['regret_upper'] == pytest.approx(0.)
    assert not result['old_score_gate_pass']


def test_selected_candidate_is_not_reselected_on_validation_scores():
    bank = {'best': arrays([.7]*16), 'challenger': arrays([.8]*16)}
    result = diagnostic(bank)
    assert result['selected'] == 'best'
    assert result['status'] == 'selected_point_inferior'
    assert result['regret_lower'] == pytest.approx(np.log(.8/.7))


def test_equivalent_candidates_and_common_seed_noise_allow_selection():
    a = arrays([.3, .4, .7, .8]*4)
    result = diagnostic({'best': a, 'tie': deepcopy(a)})
    assert result['status'] == 'acceptable_within_bank'
    assert result['regret_upper'] == pytest.approx(0.)
    # A noisy close challenger must remain unresolved instead of being dropped.
    result = diagnostic({'best': arrays([.6]*16), 'noisy': arrays([.3]*8+[.9]*8)})
    assert result['status'] == 'unresolved'


def test_mixture_score_averages_probabilities_before_log_and_excludes_first_trial():
    q = np.array([[.001, .2, .4], [.999, .8, .6]])
    result = diagnostic({'best': arrays(q)})
    assert result['scores']['best'] == pytest.approx(-np.log(.5))


def test_seed_alignment_and_unique_seeds_are_required():
    a = arrays([.6]*4)
    b = arrays([.6]*4, seeds=[4, 5, 6, 7])
    with pytest.raises(AssertionError):
        diagnostic({'best': a, 'wrong_family': b})
    a['seeds'][:] = 1
    with pytest.raises(ValueError, match='distinct seeds'):
        diagnostic({'best': a})


def test_unpaired_sensitivity_rejects_shared_seeds():
    a, b = arrays([.6]*4), arrays([.5]*4, seeds=[4, 5, 6, 7])
    result = pilot.independent_budget_difference(a, b, 100, 12)
    assert result['audit_minus_decision_nll'] == pytest.approx(np.log(.5/.6))
    with pytest.raises(ValueError, match='shared seeds'):
        pilot.independent_budget_difference(a, a, 100, 12)


def test_plateau_resets_after_late_improvement_and_cap_is_not_convergence():
    count, stopped = pilot.advance_plateau(.5, .5, 0, .0001, 2)
    assert (count, stopped) == (1, False)
    count, stopped = pilot.advance_plateau(.5, .49, count, .0001, 2)
    assert (count, stopped) == (0, False)
    assert pilot.advance_plateau(.49, .49, 1, .0001, 2) == (2, True)
    assert pilot.challenge_verdict(False, 'acceptable_within_bank') == 'budget_cap_without_plateau'
    assert pilot.challenge_verdict(True, 'unresolved') == 'plateau_challenge_inconclusive'
    assert pilot.challenge_verdict(True, 'selected_point_inferior') == 'plateau_selection_falsified_in_audit'


def test_protected_batch_matches_plan_and_refuses_partial_overwrite(tmp_path, monkeypatch):
    def fake_score(ctx, point, particles, seed):
        a = arrays([.6]*2)
        return {'probabilities': a['probabilities'][0], 'observed': a['observed'],
                'mask': a['mask'], 'seconds': 0.}
    monkeypatch.setattr(pilot, 'compact_one', fake_score)
    bank = [{'id': 'x', 'hyperparams': {'a': 1}}]
    specs = [({'subject': 103}, bank)]
    budget = {'particle_count': 2, 'filter_seed_count': 2}
    first = pilot.score_group(specs, budget, 'test', 17, 1, tmp_path/'complete')
    assert pilot.score_group(specs, budget, 'test', 17, 1, tmp_path/'complete') == first
    with pytest.raises(ValueError, match='configuration differs'):
        pilot.score_group(specs, budget, 'changed', 17, 1, tmp_path/'complete')
    (tmp_path/'partial').mkdir()
    with pytest.raises(FileExistsError):
        pilot.score_group(specs, budget, 'test', 17, 1, tmp_path/'partial')


def test_frozen_plan_never_overwrites_existing_decision(tmp_path):
    path = tmp_path/'plan.json'
    pilot.freeze_json(path, {'selected': 'a'})
    pilot.freeze_json(path, {'selected': 'a'})
    with pytest.raises(ValueError, match='Frozen plan differs'):
        pilot.freeze_json(path, {'selected': 'b'})
