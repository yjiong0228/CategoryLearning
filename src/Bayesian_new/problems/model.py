"""
Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
from .base_problem import *
from .partitions import Partition, BasePartition
from ..inference_engine import (BaseEngine, BaseSpace, BaseDistribution,
                                BaseLikelihood)


@dataclass
class ModelParams:
    k: int  # index of partition method
    beta: float  # softness of partition


class PartitionLiklihood(BaseLikelihood):

    def __init__(self, partition: BasePartition):
        """Initialize"""
        self.partition = partition


class BaseModel:

    def __init__(self, config: Dict, **kwargs):
        self.config = config
        self.all_centers = None
        self.hypotheses_set = []
        self.data_set = []
        self.engine = BaseEngine()
        self.likelihood_model = kwargs.get(
            "likelihood", Partition(config["n_dims"], config["n_cats"]))

    def set_hypotheses(self, h_set: Dict | Tuple | List):
        self.hypotheses_set = BaseSpace(h_set)

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
