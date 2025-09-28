"""
Module: Decision
"""

from abc import ABC
from collections.abc import Callable
from typing import List, Tuple, Dict, Set
import numpy as np
from .base_module import BasePartition, BaseModule
from .base_module import (cdist, softmax, BaseSet)


class BaseDecision(BaseModule):
    """
    Base Decision Module
    """

    def __init__(self, model, **kwargs):
        """
        """
        super().__init__(model, **kwargs)

    def decision(self, probabiilty: np.ndarray, **kwargs):
        """
        Make Decision
        """
        raise NotImplementedError("decision not implemented")


class Decision(BaseDecision):
    """
    Simple decision module
    """

    def __init__(self, model, **kwargs):
        """
        """
        super().__init__(model, **kwargs)
        self.method = kwargs.get("method", "sample")

    def decision(self, probability: np.ndarray, **kwargs) -> int:
        """
        Make decision
        """
        method = kwargs.get("method", self.method)
        assert len(probability.shape) == 1

        match method:
            case "top":
                choice = np.argmax(probability)

            case "sample":
                choice = np.random.choice(probability.shape[0], p=probability)

            case _:
                choice = np.random.choice(probability.shape[0], p=probability)

        return choice

    def process(self) -> int:
        pass
