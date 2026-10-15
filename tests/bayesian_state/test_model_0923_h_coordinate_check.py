import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_coordinate_check import coordinate_bank, reference_fit


def test_every_coordinate_gets_all_valid_one_step_neighbors():
    space={f'p{i}':[0,1,2] for i in range(14)}
    point={k:1 for k in space};bank=coordinate_bank(point,space)
    assert len(bank)==29
    assert {row['origin']['parameter'] for row in bank[1:]} == set(space)
    for row in bank[1:]:
        differences=[key for key in space if point[key]!=row['hyperparams'][key]]
        assert differences==[row['origin']['parameter']]


def test_boundary_coordinates_have_only_their_valid_neighbor():
    bank=coordinate_bank({'a':0,'b':2},{'a':[0,1,2],'b':[0,1,2]})
    assert [r['hyperparams'] for r in bank]==[{'a':0,'b':2},{'a':1,'b':2},{'a':0,'b':1}]


def test_incomplete_anchor_is_rejected():
    with pytest.raises(ValueError):
        coordinate_bank({'a':1},{'a':[0,1,2],'b':[0,1,2]})


def test_reference_lineage_distinguishes_original_audit_from_development(tmp_path):
    original = dict(status='frozen_before_test',parent_fit='older_search')
    assert reference_fit(tmp_path,original) == str(tmp_path.resolve())
    development = dict(status='frozen_for_development_evaluation',parent_fit='original_evaluated_fit')
    assert reference_fit(tmp_path,development) == 'original_evaluated_fit'
