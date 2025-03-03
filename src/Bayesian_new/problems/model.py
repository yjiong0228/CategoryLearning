"""
Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
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
    stimulus: tuple
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
        self.h_indices = list(self.h_set)

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
    Likelihood with (partition, beta) as hypotheses.
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
                       beta=None,
                       use_cached_dist: bool = False,
                       normalized: bool = True,
                       **kwargs) -> np.ndarray:
        """
        Get Likelihood, Base
        """

        ret = []
        for beta_ in self.beta_grid:
            ret += [
                self.partition.calc_likelihood(self.h_indices, observation,
                                               beta_, use_cached_dist,
                                               normalized, **kwargs)
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
        self.observation_set = BaseSet([])

        condition = kwargs.get("condition", 1)
        ndims = 4
        ncats = 2 if condition == 1 else 4

        self.partition_model = kwargs.get("partition", Partition(ndims, ncats))
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        self.engine = BaseEngine(
            self.hypotheses_set, self.observation_set,
            BasePrior(self.hypotheses_set),
            PartitionLikelihood(self.hypotheses_set, self.partition_model))

    def set_hypotheses(self, h_set: Dict | Tuple | List):
        """
        Set hypotheses set
        """
        self.hypotheses_set = BaseSet(h_set)

    def refresh_engine(self, h_set, prior, likelihood):
        """
        Refresh engine with new set
        """

        self.hypotheses_set = h_set
        self.engine = BaseEngine(h_set, self.observation_set, prior,
                                 likelihood)

    def fit(self, data, **kwargs) -> Tuple[ModelParams, float, Dict, Dict]:
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

    def fit(self, data, **kwargs) -> Tuple[ModelParams, float, Dict, Dict]:
        """
        Fit
        """
        all_hypo_params = {}
        all_hypo_ll = {}

        def _ll_per_hypo(beta, hypo=None):
            likelihood = self.partition_model.calc_likelihood_entry(
                hypo, data, beta[0], **kwargs)
            return np.sum(np.log(np.maximum(likelihood, 0)), axis=0)

        for hypo in self.hypotheses_set:
            result = minimize(lambda beta: -_ll_per_hypo(beta, hypo),
                              x0=[self.config["param_inits"]["beta"]],
                              bounds=[self.config["param_bounds"]["beta"]])
            beta_opt, ll_max = result.x[0], -result.fun

            all_hypo_params[hypo] = ModelParams(hypo, beta_opt)
            all_hypo_ll[hypo] = ll_max

        best_hypo = max(all_hypo_ll, key=all_hypo_ll.get)
        return (all_hypo_params[best_hypo], all_hypo_ll[best_hypo],
                all_hypo_params, all_hypo_ll)

    def fit_trial_by_trial(self, data: Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]):
        """
        Fit the model trial-by-trial to observe parameter evolution.

        Args:
            data (DataFrame): Data containing features, choices, and feedback.

        Returns:
            List[Dict]: List of results for each trial step.
        """
        step_results = []
        nTrial = len(data[2])

        for step in tqdm(range(nTrial, 0, -1)):

            trial_data = [x[:step] for x in data]
            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit(
                trial_data, use_cached_dist=(step != nTrial))

            all_hypo_post = self.engine.infer_log(trial_data,
                                                  use_cached_dist=(step
                                                                   != nTrial),
                                                  normalized=True)

            hypo_details = {}
            for i, hypo in enumerate(self.hypotheses_set.elements):
                hypo_details[hypo] = {
                    'beta_opt': all_hypo_params[hypo].beta,
                    'll_max': all_hypo_ll[hypo],
                    'post_max': all_hypo_post[i],
                    'is_best': hypo == best_params.k
                }

            step_results.append({
                'best_k': best_params.k,
                'best_beta': best_params.beta,
                'best_params': best_params,
                'best_log_likelihood': best_ll,
                'best_norm_posterior': np.max(all_hypo_post),
                'hypo_details': hypo_details
            })

        return step_results[::-1]

    def predict_choice(self, params: ModelParams, x: np.ndarray,
                       condition: int):
        """
        predict choice
        """
