"""
Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
from .base_problem import *
from .partitions import Partition, BasePartition
from ..inference_engine import (BaseEngine, BaseSet, BaseDistribution,
                                BaseLikelihood)


@dataclass
class ModelParams:
    k: int  # index of partition method
    beta: float  # softness of partition


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

    def get_likelihood(self, observation):
        """
        """
        try:
            return super().get_likelihood(observation)
        except NotImplementedError:
            pass

        ret = None


        return ret



class SoftPartitionLikelihood(PartitionLikelihood):
    """
    Likelihood with (parition, beta) as hypotheses.
    """



class BaseModel:

    def __init__(self, config: Dict, **kwargs):
        self.config = config
        self.all_centers = None
        self.hypotheses_set = []
        self.data_set = []
        self.likelihood_model = kwargs.get(
            "likelihood", Partition(config["n_dims"], config["n_cats"]))
        self.engine = BaseEngine(self.hypotheses_set, self.data_set,
                                 PartitionLikelihood(self.likelihood_model))

    def set_hypotheses(self, h_set: Dict | Tuple | List):
        self.hypotheses_set = BaseSet(h_set)

    def refresh_engine(self, h_set, likelihood):
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
        pass


class SingleRationalModel(BaseModel):

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self.hypotheses_set = []
