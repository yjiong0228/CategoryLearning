"""
Some useful functions
"""
from typing import List
import numpy as np


def two_factor_decay(data: List, gamma: float, lower: float):
    """
    data: List

    Formula:
    For `(current - k)`-th element, the strength is
    $ gamma^k * (1 - lower) + lower $
    to guarantee the values in (lower, 1].
    """

    try:
        length = len(data[1])
    except TypeError:
        length = 1
    return (1 - lower) * gamma**np.arange(length - 1, -1, -1) + lower
