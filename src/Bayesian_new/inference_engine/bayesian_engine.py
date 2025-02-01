"""
Bayesian Engine
"""

import numpy as np

from typing import Dict, Tuple, List


class BaseSpace:
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

    def __init__(self, space: BaseSpace):
        """\ndocstring"""
        self.value = np.ones(space.length, dtype=float)/space.length

    def update(self, value: np.ndarray):
        self.value = value.copy()


class BasePrior(BaseDistribution):

    @property
    def get_prior(self):
        return self.value


class BaseLikelihood:

    def __init__(self, h_set: BaseSpace, d_set: BaseSpace, **kwargs):
        """
        Parameters
        ----------
        h_set
        d_set
        """
        self.h_set = h_set
        self.d_set = d_set
        self.likelihood = None

    def get_likelihood(self, observation):
        """
        Parameters
        ----------
        observation: any
        """
        index = self.d_set[observation]
        if self.likelihood is not None:
            return self.likelihood[index]

        return None  # 1. Calculate 2. Save 3. Return.

    def set_all(self, value: np.ndarray):
        """
        Set the whole matrix
        """
        self.likelihood = value.copy()

    def set_row(self, row, value):
        """
        Set Row
        """
        self.likelihood[self.d_set[row]] = value


    def set_col(self, col, value):
        """
        Set column
        """
        self.likelihood[:, self.h_set[col]] = value


class BaseEngine:

    def __init__(self, hypotheses_set: BaseSpace, data_set: BaseSpace,
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

