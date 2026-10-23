"""Independent checks of pooled scores and repeated-label aggregation."""
import hashlib
import json

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1022_belief_summary import load_case, nll


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def save(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nll_pools_probabilities_before_logarithm():
    probabilities = np.array([[.1, .8], [.9, .2]])
    assert nll(probabilities) == pytest.approx(np.log(2))
    assert nll(probabilities) != pytest.approx(-np.log(probabilities).mean())


@pytest.mark.parametrize('repeated_labels', [False, True])
def test_saved_prediction_identity_and_repeated_chunk_labels(tmp_path, repeated_labels):
    folder, study = tmp_path/'data', tmp_path/'study'
    choices = np.array([1, 2, 1, 2])
    categories = np.array([1, 1, 2, 2])
    hashes = {}
    for name, sl in [('training.npz', slice(0, 2)), ('sealed_test.npz', slice(2, 4))]:
        hashes[name] = save(folder/name, choices=choices[sl], categories=categories[sl],
                            feedback=(choices[sl] == categories[sl]).astype(float))
    dump(folder/'prepared.json', dict(trials=4, training_trials=2, test_trials=2, condition=1, array_sha256=hashes))
    p = np.array([[[.8, .2], [.6, .4], [.3, .7], [.2, .8]],
                  [[.6, .4], [.4, .6], [.1, .9], [.4, .6]]])
    target = study/'evaluate/subject_108/BELIEF/frozen'
    observed_p, correct_p = p[:, np.arange(4), choices-1], p[:, np.arange(4), categories-1]
    prediction = dict(probabilities=p, choices=np.tile(choices, 2 if repeated_labels else 1),
                      categories=np.tile(categories, 2 if repeated_labels else 1),
                      observed_probability=observed_p, correct_probability=correct_p, seeds=[10, 11])
    sha = save(target.with_suffix('.npz'), **prediction)
    dump(target.with_suffix('.json'), dict(predictions_sha256=sha))
    auto = study/'autonomous/case'
    auto_sha = save(auto.with_suffix('.npz'), categories=categories, seeds=[20, 21],
                    choices=np.tile(choices, (2, 1)), probabilities=p,
                    feedback=np.tile(choices == categories, (2, 1)).astype(float))
    dump(auto.with_suffix('.json'), dict(predictions_sha256=auto_sha))
    nominee = dict(subject=108, arm='BELIEF', candidate=dict(id='frozen'))
    jobs = [dict(subject=108, model='BELIEF', folder=str(folder), target=str(auto), seeds=[20, 21])]
    result = load_case(study, nominee, jobs)
    np.testing.assert_array_equal(result['observed']['choices'], choices)
    # Corrupt an identity while preserving the artifact hash: the numerical
    # identity audit must detect it independently of checksum verification.
    prediction['choices'][0] = 2
    sha = save(target.with_suffix('.npz'), **prediction)
    dump(target.with_suffix('.json'), dict(predictions_sha256=sha))
    with pytest.raises(AssertionError):
        load_case(study, nominee, jobs)


def test_each_belief_variant_is_compared_with_support(tmp_path, monkeypatch):
    import pandas as pd
    from src.Bayesian_state.workflows.analysis import model_1022_belief_summary as summary
    study = tmp_path/'study'
    arms = {'SUPPORT': .8, 'BELIEF': .7, 'SLOT': .9}
    dump(study/'nominees.json', dict(status='frozen_before_test', smoke=False,
        nominees=[dict(subject=108, arm=arm, candidate={'id': arm}) for arm in arms]))
    dump(study/'autonomous_jobs.json', {'jobs': []})
    categories = np.ones(64, dtype=int)

    def case(study, nominee, jobs):
        p = arms[nominee['arm']]
        return dict(meta=dict(training_trials=32, test_trials=32, condition=1),
            observed=dict(choices=categories, categories=categories),
            prediction=dict(observed_probability=np.full((3, 64), p), seeds=np.arange(3)),
            autonomous=dict(choices=np.tile(categories, (4, 1)), seeds=np.arange(4),
                            present=np.ones((4, 64)), contribution=np.full((4, 64), p)))

    monkeypatch.setattr(summary, 'load_case', case)
    summary.run(study, tmp_path/'summary')
    comparisons = pd.read_csv(tmp_path/'summary/comparison.csv').set_index('arm')
    assert set(comparisons.index) == {'BELIEF', 'SLOT'}
    for arm in comparisons.index:
        assert comparisons.loc[arm, 'test_nll_difference'] == pytest.approx(np.log(.8/arms[arm]))
