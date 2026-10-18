"""Protect figure input identity and complete comparison cells."""
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model0923_refit_joint_evidence import ROOT, audit_hashes, require_cells


@pytest.mark.parametrize('key', ['input_sha256', 'inputs_sha256'])
def test_audit_schema_normalizes_relative_inputs(key):
    assert audit_hashes({key: {'results/a.json': 'abc'}}) == {str(ROOT/'results/a.json'): 'abc'}


def test_audit_rejects_conflicting_absolute_and_relative_binding():
    with pytest.raises(ValueError, match='Conflicting'):
        audit_hashes({'inputs_sha256': {'results/a.json': 'old'},
                      'input_sha256': {str(ROOT/'results/a.json'): 'changed'}})


def test_audit_requires_binding():
    with pytest.raises(ValueError, match='no bound'):
        audit_hashes({'status': 'complete'})


@pytest.mark.parametrize('rows', [[('POOL', 'test')], [('POOL', 'test'), ('POOL', 'test')],
                                 [('POOL', 'test'), ('SEARCH', 'training')]])
def test_missing_duplicate_or_wrong_phase_fails(rows):
    with pytest.raises(ValueError, match='cells'):
        require_cells(pd.DataFrame(rows, columns=['arm', 'phase']), ['arm', 'phase'],
                      {('POOL', 'test'), ('SEARCH', 'test')})


def test_complete_cells_accept_any_row_order():
    require_cells(pd.DataFrame([('SEARCH', 'test'), ('POOL', 'test')], columns=['arm', 'phase']),
                  ['arm', 'phase'], {('POOL', 'test'), ('SEARCH', 'test')})
