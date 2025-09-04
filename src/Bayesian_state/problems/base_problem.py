"""
Problem
"""
# type: ignore
from ..inference_engine import (BaseSet, BaseEngine, BaseDistribution,
                                BaseLikelihood, BasePrior)
from ..utils import softmax, cdist, euc_dist, two_factor_decay, entropy

ALL_K_SPACE = BaseSet([])
