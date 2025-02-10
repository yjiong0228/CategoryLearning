"""
Model
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood)
from .partitions import Partition, BasePartition


@dataclass(unsafe_hash=True)
class ModelParams:
    """
    Legacy
    """

    k: int  # index of partition method
    beta: float  # softness of partition


@dataclass
class ObservationType:
    """
    observation format
    """
    stimuli: tuple
    choices: tuple
    responses: tuple


class PartitionLikelihood(BaseLikelihood):
    """
    Likelihood in only partitions.
    """

    def __init__(self, space: BaseSet, partition: BasePartition):
        """Initialize

        space: the set of k's, must be included in the partition.
        """
        super().__init__(space)
        self.partition = partition
        # This may raise an exception if h_set is not a subset of
        # partition labels.
        self.h_indices = [v for k, v in self.h_set.elements.items()]

    def get_likelihood(self,
                       observation,
                       beta: list | tuple | float = 1.,
                       use_cached_dist: bool = False,
                       normalized: bool = True):
        """
        Get Likelihood, Base
        """

        ret = self.partition.calc_likelihood(self.h_indices, observation, beta,
                                             use_cached_dist, normalized)
        return ret


class SoftPartitionLikelihood(PartitionLikelihood):
    """
    Likelihood with (parition, beta) as hypotheses.
    """

    def __init__(self, space: BaseSet, partition: BasePartition,
                 beta_grid: list):
        """Initialize

        space: the set of k's, must be included in the partition.
        """
        super().__init__(space, partition)
        self.beta_grid = beta_grid

    def get_likelihood(self,
                       observation,
                       use_cached_dist: bool = False,
                       normalized: bool = True):
        """
        Get Likelihood, Base
        """

        ret = []
        for beta in self.beta_grid:
            ret += [
                self.partition.calc_likelihood(self.h_indices, observation,
                                               beta, use_cached_dist,
                                               normalized)
            ]
        return np.concatenate(ret, axis=1)


class BaseModel:
    """
    Base Model
    """

    def __init__(self, config: Dict, **kwargs):
        self.config = config
        self.all_centers = None
        self.hypotheses_set = BaseSet([])
        self.data_set = BaseSet([])
        self.partition_model = kwargs.get(
            "partition", Partition(config["n_dims"], config["n_cats"]))
        space = config["space"]
        self.engine = BaseEngine(
            self.hypotheses_set, self.data_set,
            PartitionLikelihood(space, self.partition_model))

    def set_hypotheses(self, h_set: Dict | Tuple | List):
        """
        Set hypotheses set
        """
        self.hypotheses_set = BaseSet(h_set)

    def refresh_engine(self, h_set, likelihood):
        """
        Refresh engine with new set
        """

        self.hypotheses_set = h_set
        self.engine = BaseEngine(h_set, self.data_set, likelihood)

    def fit(self, data) -> Tuple[ModelParams, float, float, Dict]:
        """
        Parameters
        ----------
        data :


        Returns
        -------
        out :

        """
        raise NotImplementedError
        # return (data, 0., 0., {})


class SingleRationalModel(BaseModel):
    """
    Pure Rational
    """

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self.hypotheses_set = BaseSet([])

    def fit(self, data) -> Tuple[ModelParams, float, float, Dict]:
        """
        Fit
        """
