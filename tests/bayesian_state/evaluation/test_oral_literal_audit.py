"""Measurement-audit guards: fixed semantics and original trial identities."""
import numpy as np
import pytest

from src.Bayesian_state.model.modules.readout.report_space import facet_token
from src.Bayesian_state.evaluation.oral.constraint_alignment import InconsistentReportError
from src.Bayesian_state.workflows.analysis.model_0923_oral_literal_audit import (
    literal_check, neighboring_keys,
)


def test_literal_threshold_uses_existing_halfspace():
    token = facet_token(np.array([[-1., 0, 0, 0]]), np.array([-.5]))
    inside = literal_check([token], np.array([.8, .2, .3, .4]))
    outside = literal_check([token], np.array([.2, .2, .3, .4]))
    assert inside['numeric_holds'] and not outside['numeric_holds']
    assert inside['max_signed_halfspace_violation'] == pytest.approx(-.3)
    assert outside['max_signed_halfspace_violation'] == pytest.approx(.3)
    assert outside['violated_tokens'] == [token]


def test_relative_violation_is_normalized_by_halfspace_normal():
    token = facet_token(np.array([[1., -1, 0, 0]]), np.array([0.]))
    result = literal_check([token], np.array([.8, .2, .3, .4]))
    assert result['max_signed_halfspace_violation'] == pytest.approx(.6/np.sqrt(2))


def test_symbolic_references_keep_unknown_numerical_meaning():
    result = literal_check(['body:0:gt', 'half_unknown:1:lt'], np.array([.8, .2, .3, .4]))
    assert result['numeric_predicates'] == 0 and result['symbolic_predicates'] == 2
    assert result['numeric_holds'] is None and result['max_signed_halfspace_violation'] is None


def test_known_numeric_clause_does_not_resolve_symbolic_clause():
    token = facet_token(np.array([[-1., 0, 0, 0]]), np.array([-.5]))
    result = literal_check([token, 'body:1:gt'], np.array([.8, .2, .3, .4]))
    assert result['numeric_holds'] and result['symbolic_predicates'] == 1


def test_inconsistent_report_is_not_an_ordinary_stimulus_disagreement():
    with pytest.raises(InconsistentReportError):
        literal_check(['body:0:gt', 'body:0:lt'], np.array([.8, .2, .3, .4]))


@pytest.mark.parametrize('point', [[.2, .3], [np.nan, .2, .3, .4], [-.1, .2, .3, .4]])
def test_invalid_physical_stimuli_rejected(point):
    with pytest.raises(ValueError):
        literal_check([], np.array(point))


def test_offsets_do_not_cross_session_boundaries():
    available = {(207, session, trial) for session in [1, 2] for trial in range(1, 6)}
    assert neighboring_keys(207, 1, 3, available) == [(207, 1, i) for i in range(1, 6)]
    assert neighboring_keys(207, 1, 5, available) is None
    assert neighboring_keys(207, 2, 1, available) is None


def test_offsets_preserve_original_trial_ids_when_rows_are_missing():
    available = {(207, 1, i) for i in [1, 2, 3, 5, 6, 7]}
    assert neighboring_keys(207, 1, 3, available) is None
