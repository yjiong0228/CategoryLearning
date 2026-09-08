"""Journal Model 0826 policy over the shared parameter-space implementation."""
from src.Bayesian_state.optimization import parameter_space as _shared
from src.Bayesian_state.optimization.parameter_space import *

MODEL_0826_SPIKE_PARAMETERS = _shared.MODEL_0818_SPIKE_PARAMETERS
MODEL_0826_SUBJECT_PARAMETERS = _shared.MODEL_0818_SUBJECT_PARAMETERS
SUPPORTED_MODEL_IDS = {"model_0826"}


def validate_model_parameter_space(config, *, expected_model_id=None):
    if expected_model_id not in (None, "model_0826"):
        raise ValueError("Journal workflow requires model_0826")
    return _shared.validate_model_parameter_space(config, expected_model_id="model_0826")


def load_model_parameter_space(path, expected_model_id=None):
    if expected_model_id not in (None, "model_0826"):
        raise ValueError("Journal workflow requires model_0826")
    return _shared.load_model_parameter_space(path, expected_model_id="model_0826")
