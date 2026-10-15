"""Independent selection and confirmation rules for the bounded continuation."""
import numpy as np

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_0923_h_training_refinement import (
    audit_bank, paired_comparison, proposal_bank,
)
from src.Bayesian_state.workflows.analysis.model_0923_h_validation import digest, write


def test_audit_always_preserves_original_and_uses_training_rank():
    anchor = dict(id='old', mean_nll=.9)
    rows = [dict(id='b', mean_nll=.3), anchor, dict(id='a', mean_nll=.2), dict(id='c', mean_nll=.4)]
    assert [row['id'] for row in audit_bank(rows, anchor, 3)] == ['old', 'a', 'b']


def test_proposals_are_reproducible_unique_and_keep_equal_budget():
    space = {f'coordinate{i}': list(range(6)) for i in range(5)}
    anchors = [dict(candidate({key: value for key in space}, 'test'), mean_nll=.1 + value/10) for value in (1, 3)]
    row = dict(candidate=anchors[0], finalists=anchors)
    settings = dict(archived_finalists=2, elites=2, rays_per_elite=3, joint_per_elite=2)
    bank = proposal_bank(row, space, settings, 123)
    assert bank == proposal_bank(row, space, settings, 123)
    assert len(bank) == len({item['id'] for item in bank}) == 12
    assert bank[:2] == anchors


def save_score(tmp_path, name, probability):
    file = tmp_path / (name + '.npz')
    np.savez(file, seeds=np.arange(len(probability)), observed_probability=probability)
    write(file.with_suffix('.json'), dict(predictions_sha256=digest(file)))
    return dict(id=name, score_file=str(file.with_suffix('.json')))


def test_confirmation_accepts_clear_improvement_but_rejects_worse_and_ties(tmp_path):
    anchor = save_score(tmp_path, 'anchor', np.full((4, 10), .5))
    better = save_score(tmp_path, 'better', np.full((4, 10), .6))
    assert paired_comparison(anchor, better, 2)['accepted']
    assert not paired_comparison(better, anchor, 2)['accepted']
    assert not paired_comparison(anchor, anchor, 2)['accepted']


def test_confirmation_retains_seed_pairing_and_rejects_uncertain_gain(tmp_path):
    anchor = save_score(tmp_path, 'anchor', np.full((4, 10), .5))
    challenger = save_score(tmp_path, 'challenger', np.repeat([[.2], [.4], [.6], [.84]], 10, axis=1))
    result = paired_comparison(anchor, challenger, 2)
    assert result['delta_mean_nll'] < 0
    assert result['paired_mc_se'] > 0
    assert not result['accepted']
