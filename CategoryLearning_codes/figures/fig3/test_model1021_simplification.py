"""Check paired candidate scores against a hand-calculable filter bank."""
import json

import numpy as np
import pytest

from CategoryLearning_codes.figures.fig3.model1021_simplification import selection_candidates
from src.Bayesian_state.evaluation.model_0923_h_validation import digest, nll


def test_candidates_use_confirmed_nominee_and_paired_seed_error(tmp_path):
    # The confirmed point can rank second on the earlier selection seeds.
    # Multiplying every probability by 1/2 shifts NLL by log(2), while
    # preserving the same seed fluctuation; its paired difference has zero SE.
    probability = np.array([[.2, .5], [.8, .7]])
    points = []
    for name, values in [('earlier_best', probability), ('confirmed', probability / 2)]:
        path = tmp_path / (name + '.npz')
        np.savez(path, observed_probability=values, seeds=np.array([21, 22]))
        point = dict(id=name, hyperparams={'support_decay': .8 if name == 'confirmed' else .6},
                     mean_nll=nll(values), score_file=str(path.with_suffix('.json')))
        path.with_suffix('.json').write_text(json.dumps(dict(subject=108, model='COMPACT', candidate=point,
            particles=2, trials=2, seeds=[21, 22], mean_nll=point['mean_nll'], predictions_sha256=digest(path))))
        points.append(point)
    (tmp_path / 'subject_108').mkdir()
    (tmp_path / 'subject_108/prepared.json').write_text(json.dumps(dict(training_trials=2)))
    (tmp_path / 'selection_completion.json').write_text(json.dumps(dict(status='complete', stage='selection',
        groups=[dict(subject=108, model='COMPACT', rows=points)])))
    frozen = dict(data=str(tmp_path), settings=dict(selection_candidates=2, budgets=dict(selection=dict(particles=2, seeds=2))),
        nominees=[dict(subject=108, model='COMPACT', candidate=points[1], reference_id='confirmed')])

    def bind(path, expected=None):
        if expected is not None and digest(path) != expected:
            raise ValueError('Input changed')
        return path

    scores, parameters = selection_candidates(tmp_path, frozen, bind)
    best, chosen = scores.set_index('candidate').loc[['earlier_best', 'confirmed']].itertuples()
    assert chosen.selected and chosen.selection_rank == 2 and chosen.warm_start
    assert not best.selected and best.selection_rank == 1
    np.testing.assert_allclose([best.delta_nll, chosen.delta_nll], [-np.log(2), 0], atol=1e-14)
    np.testing.assert_allclose(scores.paired_mc_se, 0, atol=1e-14)
    assert len(parameters) == 2

    # A permuted seed array has the same marginal score but cannot be paired.
    path = tmp_path / 'confirmed.npz'
    np.savez(path, observed_probability=probability / 2, seeds=np.array([22, 21]))
    record = json.loads(path.with_suffix('.json').read_text())
    record.update(seeds=[22, 21], predictions_sha256=digest(path))
    path.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        selection_candidates(tmp_path, frozen, bind)
