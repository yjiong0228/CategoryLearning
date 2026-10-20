"""Check likelihood interpretation, paired filters and complete matrix boundaries."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation import model_0923_finalization_parameter_support as support


def bank(probability, seeds=(11, 22)):
    return dict(observed_probability=np.asarray(probability), seeds=np.asarray(seeds))


def test_sequence_likelihood_ratio_pools_filters_before_taking_logs():
    result = support.rank_bank({"better": bank([[.8, .6], [.4, .8]]),
                                "worse": bank([[.3, .2], [.7, .6]])})
    rows = {r["point"]: r for r in result}
    assert rows["better"]["training_nll"] == pytest.approx(-np.log(.6 * .7) / 2)
    assert rows["worse"]["relative_point_likelihood"] == pytest.approx((.5 * .4) / (.6 * .7))
    # Two leave-one-filter differences: the paired SE is half their separation.
    deletions = [-np.log(.3 * .2 / (.8 * .6)), -np.log(.7 * .6 / (.4 * .8))]
    assert rows["worse"]["delta_total_nll_mcse"] == pytest.approx(abs(deletions[0] - deletions[1]) / 2)
    assert rows["better"]["relative_point_likelihood"] == 1.
    assert rows["better"]["delta_total_nll_mcse"] == 0.


def test_filter_reordering_preserves_pairing_and_identical_points():
    result = support.rank_bank({"b": bank([[.8, .6], [.4, .8]]),
                                "a": bank([[.4, .8], [.8, .6]], seeds=(22, 11))})
    assert all(r["relative_point_likelihood"] == 1. for r in result)
    assert all(r["delta_total_nll_mcse"] == 0. for r in result)


@pytest.mark.parametrize("fault", ["zero", "nan", "duplicate", "one_filter", "other_seeds", "other_trials"])
def test_invalid_probabilities_or_unpaired_banks_fail(fault):
    first = bank([[.6, .7], [.7, .8]])
    second = deepcopy(first)
    if fault == "zero":
        second["observed_probability"][0, 0] = 0.
    elif fault == "nan":
        second["observed_probability"][0, 0] = np.nan
    elif fault == "duplicate":
        second["seeds"][:] = 11
    elif fault == "one_filter":
        second = bank([[.7, .8]], seeds=(11,))
    elif fault == "other_seeds":
        second["seeds"][1] = 33
    else:
        second["observed_probability"] = second["observed_probability"][:, :1]
    with pytest.raises((ValueError, AssertionError)):
        support.rank_bank({"a": first, "b": second})


def scope(subjects):
    manifest = dict(status="complete", subjects=subjects, registered_subjects=[108, 205, 306],
                    complete_primary_matrix=subjects == [108, 205, 306])
    cells = pd.DataFrame([dict(subject=s, candidate=c, precision_family=f, arm=a)
        for s in subjects for c, (f, a) in support.CANDIDATES.items()])
    return manifest, cells


def test_partial_requires_explicit_flag_and_all_three_candidates():
    manifest, cells = scope([108])
    with pytest.raises(ValueError):
        support.check_scope(manifest, cells, False)
    support.check_scope(manifest, cells, True)
    with pytest.raises(ValueError):
        support.check_scope(manifest, cells.iloc[:2], True)
    manifest, cells = scope([108, 205, 306])
    support.check_scope(manifest, cells, False)
    cells.loc[0, "arm"] = "POOL"
    with pytest.raises(ValueError):
        support.check_scope(manifest, cells, False)


def test_online_channel_closures_keep_evidence_decay_active_when_exit_is_zero():
    p = {k: .5 for k in support.COORDINATES}
    p.update({support.H + "search_controller.global_search": 1.,
              support.H + "search_controller.error_gain": 0., support.H + "exit_rate": 0.})
    frozen = dict(settings=dict(arms={"SEARCH": {}}, belief_intervention="online_retention_evidence",
                                mechanism=dict(kind="E", capacity="posterior")),
        nominees=[dict(subject=108, model="SEARCH", candidate=dict(hyperparams=p))])
    closed = support.point_channels(frozen, 108, "SEARCH")
    assert set(closed) == {support.H + s for s in ("search_controller.global_search_failure_gain",
        "local_structure_weight", "search_controller.failure_decay", "transfer_scale")}
    assert support.H + "intervention.decay" not in closed
