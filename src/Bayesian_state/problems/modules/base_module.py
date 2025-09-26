"""
Base Module
"""

from abc import ABC
from ..partitions import *
from ..base_problem import *


class BaseModule(ABC):

    def __init__(self, engine, **kwargs):
        """
        Initialize
        """
        self.engine = engine
