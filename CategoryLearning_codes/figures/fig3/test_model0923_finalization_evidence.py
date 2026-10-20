"""Figure scope and oral missingness cannot imply evidence that is absent."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3 import model0923_finalization_evidence as figure


def manifest(subjects):
    return dict(status="complete", subjects=subjects, registered_subjects=[108, 205, 306],
        registered_cells=9, collected_cells=3*len(subjects), complete_primary_matrix=len(subjects)==3,
        missing_subjects=sorted({108, 205, 306}-set(subjects)))


def test_partial_requires_explicit_labelled_mode():
    with pytest.raises(ValueError, match="allow-partial"):
        figure.check_scope(manifest([108]), False)
    assert figure.check_scope(manifest([108]), True) == [108]
    assert "awaiting S205, S306" in figure.banner(manifest([108]))


def test_full_matrix_accepts_all_registered_participants():
    assert figure.check_scope(manifest([108, 205, 306]), False) == [108, 205, 306]


@pytest.mark.parametrize("key,value", [("complete_primary_matrix", True), ("collected_cells", 9),
    ("missing_subjects", []), ("registered_subjects", [108]), ("subjects", [108, 108])])
def test_partial_collection_cannot_be_relabelled_as_complete(key, value):
    data = manifest([108]); data[key] = value
    with pytest.raises(ValueError):
        figure.check_scope(data, True)


def oral():
    return pd.DataFrame([dict(subject=108, candidate="SEARCH_ADD10", reading="midpoint", window="test",
        domain="functional", subset="complete", reports=4, support=.2, conflict=.1, undetermined=.7),
        dict(subject=108, candidate="SEARCH_ADD10", reading="midpoint", window="test",
        domain="functional", subset="catalogue_supported", reports=0,
        support=np.nan, conflict=np.nan, undetermined=np.nan)])


def test_zero_report_score_stays_undefined():
    frame = oral(); original = deepcopy(frame)
    figure.check_oral(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("fault", ["zero_filled_missing", "missing_defined_score", "wrong_sum", "negative", "duplicate"])
def test_oral_missingness_or_mass_errors_are_rejected(fault):
    frame = oral()
    if fault == "zero_filled_missing": frame.loc[1, "support"] = 0.
    elif fault == "missing_defined_score": frame.loc[0, "support"] = np.nan
    elif fault == "wrong_sum": frame.loc[0, "support"] = .4
    elif fault == "negative": frame.loc[0, "conflict"] = -.1
    else: frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises((ValueError, AssertionError)):
        figure.check_oral(frame)
