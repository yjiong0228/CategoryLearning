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
                       beta=None,
                       use_cached_dist: bool = False,
                       normalized: bool = True):
        """
        Get Likelihood, Base
        """

        ret = []
        for beta_ in self.beta_grid:
            ret += [
                self.partition.calc_likelihood(self.h_indices, observation,
                                               beta_, use_cached_dist,
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
        opt_params = {}
        opt_log_likelihood = {}

        def inner(beta, hypo_=None):
            likelihood = self.partition_model.calc_likelihood_entry(
                hypo_, data, beta[0], **kwargs)
            return np.sum(np.log(np.maximum(likelihood, 0)), axis=0)

        for hypo in self.hypotheses_set:

            result = minimize(lambda beta: -inner(beta, hypo),
                              x0=[self.config["param_inits"]["beta"]],
                              bounds=[self.config["param_bounds"]["beta"]])
            beta_opt, posterior_opt = result.x[0], -result.fun
            opt_log_likelihood[hypo] = posterior_opt
            # log_likelihood = inner(beta_opt, hypo)
            opt_params[hypo] = ModelParams(hypo, beta_opt)

        opt_hypo = max(opt_log_likelihood, key=opt_log_likelihood.__getitem__)
        # print(opt_log_likelihood)
        return (opt_params[opt_hypo], opt_log_likelihood[opt_hypo], opt_params,
                opt_log_likelihood)

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
        results = data[2]
        for step in tqdm(range(len(results), 0, -1)):
            # for step in range(len(results), 0, -1):
            # print(len(results), step)
            trial_data = [x[:step] for x in data]
            fitted_params, best_ll, _, _ = self.fit(
                trial_data, use_cached_dist=(step != len(results)))

            best_post = self.engine.infer_log(
                trial_data,
                use_cached_dist=(step != len(results)),
                normalized=True)

            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'best_log_likelihood': best_ll,
                'best_posterior': np.max(best_post),
                'k_posteriors': best_post,
                'params': fitted_params
            })

        return step_results[::-1]
