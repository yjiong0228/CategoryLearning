"""Compatibility imports; implementation lives in model.modules.readout."""

from .modules.readout.observation import (
    ChoiceOralPrediction,
    predict_choice_oral,
)

__all__ = [
    "ChoiceOralPrediction",
    "predict_choice_oral",
]
