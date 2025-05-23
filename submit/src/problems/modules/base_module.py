"""
Base Module
"""

from abc import ABC
from ..partitions import *
from ..base_problem import *


class BaseModule(ABC):

    def __init__(self, model, **kwargs):
        """
        Initialize
        """
        self.model = model
