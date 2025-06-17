"""
Module: Memory Mechanism
"""

from abc import ABC
from collections.abc import Callable
from typing import List, Tuple, Dict, Set
import numpy as np
from .base_module import BaseModule
from .base_module import (cdist, softmax, BaseSet, entropy)

class BaseMemory(BaseModule):
    """
    Base Memory
    """

    def __init__(self, model, **kwargs):
        """
        Initialize
        """
        super().__init__(model, **kwargs)
