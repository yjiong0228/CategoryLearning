"""
Bayesian Engine
"""

import numpy as np

from typing import Dict, Tuple, List, Any


class BaseSet:
    """Immutable"""

    def __init__(self, items: Dict | Tuple | List):
        """init"""
        self.items = dict((item, i) for i, item in enumerate(items))
        self._size = len(items)

    @property
    def length(self):
        return self._size

    def __getitem__(self, key):
        assert key in self.items, "Invalid key"
        return self.items.get(key, -1)


class BaseDistribution:

    def __init__(self, space: BaseSet):
        """\ndocstring"""
        self.value = np.ones(space.length, dtype=float) / space.length

    def update(self, value: np.ndarray):
        """
        Update value
        """
        self.value = value.copy()


class BasePrior(BaseDistribution):

    @property
    def get_prior(self):
        return self.value


class BaseLikelihood(BaseDistribution):
    """
    """

    def __init__(self, space: BaseSet, **kwargs):
        """
        Parameters
        ----------
        h_set
        d_set
        """
        super().__init__(space)
        self.h_set = space  # hypotheses set!
        # self.d_set = d_set
        self.cache = {None: self.value}

    def get_likelihood(self, observation):
        """
        Parameters
        ----------
        observation: any
        """
        if observation in self.cache:
            return self.cache[observation]

        raise NotImplementedError("get_likelihood not implemented")
        return None  # 1. Calculate 2. Save 3. Return.

    def set_all(self, value: Dict[Any, np.ndarray]):
        """
        Set the whole matrix
        """
        self.cache.update(value)

    def set_row(self, row, value):
        """
        Set Row
        """
        self.cache[row] = value


class BaseEngine:
    """
    Base Bayesian Engine
    """

    def __init__(self, hypotheses_set: BaseSet, data_set: BaseSet,
                 likelihood: BaseLikelihood):
        """

        """
        self.hypotheses_set = hypotheses_set
        self.data_set = data_set
        self.likelihood = likelihood
        self.h_state = BaseDistribution(self.hypotheses_set)

    def infer_single(self, observation):
        """
        Parameters
        ----------
        observation:
        """

        likelihood_row = self.likelihood.get_likelihood(observation)
        self.h_state.update(self.h_state.value * likelihood_row)
        self.h_state.update(self.h_state.value / self.h_state.value.sum())

        return self.h_state.value

    def infer(self, observations: list | tuple):
        """
        Parameters
        ----------
        observations: List | Tuple of observations
        """
        for o in observations:
            self.infer_single(o)

        return self.h_state.value
