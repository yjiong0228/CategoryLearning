"""Legacy readout imports resolve to the single module implementation."""

from importlib import import_module
import pickle

import numpy as np
import pytest


@pytest.mark.parametrize("legacy, canonical, symbol", [
    ("readout", "choice", "ChoicePrediction"),
    ("readout", "choice", "DecisionRuleSnapshot"),
    ("readout", "choice", "predict_choice_from_model"),
    ("readout", "shared", "normalize_probability_vector"),
    ("readout", "oral_kernel", "read_oral_report"),
    ("readout", "reaction_time", "read_reaction_time"),
    ("oral_readout", "oral", "OralReadout"),
    ("oral_readout", "oral", "OralReadoutParameters"),
    ("oral_observation", "observation", "predict_choice_oral"),
    ("oral_report_space", "report_space", "CatalogueReportSpace"),
    ("oral_report_space", "report_space", "_component_distribution"),
])
def test_legacy_and_canonical_imports_share_the_same_implementation(legacy, canonical, symbol):
    old = import_module(f"src.Bayesian_state.model.{legacy}")
    current = import_module(f"src.Bayesian_state.model.modules.readout.{canonical}")
    assert getattr(old, symbol) is getattr(current, symbol)


def test_saved_legacy_readout_prediction_still_unpickles():
    from src.Bayesian_state.model.modules.readout.choice import ChoicePrediction

    prediction = ChoicePrediction(
        cognitive_probabilities=np.array([.2, .8]),
        observed_probabilities=np.array([.2, .8]),
        readout_details={}, output_lapse=0., post_error_lapse_state=0.,
    )
    # Protocol 0 records the module path explicitly, so emulate an existing saved object.
    serialized = pickle.dumps(prediction, protocol=0).replace(
        b"src.Bayesian_state.model.modules.readout.choice\nChoicePrediction\n",
        b"src.Bayesian_state.model.readout\nChoicePrediction\n",
    )
    restored = pickle.loads(serialized)
    assert type(restored) is ChoicePrediction
    np.testing.assert_array_equal(restored.observed_probabilities, prediction.observed_probabilities)
