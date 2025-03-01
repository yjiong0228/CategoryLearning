"""
Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
from .base_problem import softmax, cdist, euc_dist
from .partitions import Partition, BasePartition
from copy import deepcopy

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
        
        for step in tqdm(range(nTrial), 0, -1):
            trial_data = [x[:step] for x in data]
            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit(
                trial_data, use_cached_dist=(step != nTrial))

            all_hypo_post = self.engine.infer_log(
                trial_data,
                use_cached_dist=(step != nTrial),
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


class ForgetModel(BaseModel):
    """
    Add forgetting mechanism
    """
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)
        self.base_weight = 0.1  # 可配置参数

    def _apply_memory_weights(self, base_likelihood, gamma):
        """应用记忆权重"""
        n_trials = len(base_likelihood)
        decay_weights = gamma ** np.arange(n_trials-1, -1, -1)
        memory_weights = self.base_weight + (1-self.base_weight)*decay_weights
        return np.exp(memory_weights * np.log(np.maximum(base_likelihood, 1e-10)))

    def fit_with_given_gamma(self, data, gamma=1.0, **kwargs) -> Tuple[ModelParams, float, Dict, Dict]:
        """
        Fit
        """
        all_hypo_params = {}
        all_hypo_ll = {}

        def _ll_per_hypo(beta, hypo=None):
            # 获取基础似然
            base_likelihood = self.partition_model.calc_likelihood_entry(
                hypo, data, beta[0], **kwargs)
            # 应用记忆权重
            weighted_likelihood = self._apply_memory_weights(base_likelihood, gamma)

            return np.sum(np.log(np.maximum(weighted_likelihood, 0)), axis=0)

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
                                             np.ndarray], gamma=1.0):
        """
        Fit the model trial-by-trial to observe parameter evolution.

        Args:
            data (DataFrame): Data containing features, choices, and feedback.

        Returns:
            List[Dict]: List of results for each trial step.
        """
        step_results = []
        nTrial = len(data[2])
        
        for step in tqdm(range(nTrial), 0, -1):
            trial_data = [x[:step] for x in data]
            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit_with_given_gamma(
                trial_data, gamma, use_cached_dist=(step != nTrial))

            all_hypo_post = self.engine.infer_log(
                trial_data,
                use_cached_dist=(step != nTrial),
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



    def error_function(self, data, gamma, window_size=16, **kwargs):
        """
        误差目标函数，用于最小化每个试次段（如每16个试次）的预测准确率与真实准确率之间的差异。
        
        Args:
            gamma (float): 当前的gamma值
            block_data (DataFrame): 数据集
            window_size (int): 每个段的大小，默认为16
            
        Returns:
            float: 误差（目标函数值）
        """
        # 拟合k和beta
        step_results = self.fit_trial_by_trial(data, gamma)

        # 计算每16个试次的误差
        n_windows = len(data) // window_size
        errors = []

        for i in range(n_windows):
            # 计算真实准确率（基于feedback）
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            true_accuracy = np.mean(data['feedback'].iloc[start_idx:end_idx] == 1)

            # 计算预测准确率（基于模型参数）
            predicted = []

            for j, trial in data.iterrows():
                fitted_params = step_results[j-data.index[0]]['params']

                p_true = self.partition_model.calc_trueprob_entry(
                    fitted_params.k, trial, fitted_params.beta, **kwargs)
                
                predicted.append(p_true)

            pred_accuracy = np.mean(predicted)
            error = abs(pred_accuracy - true_accuracy)
            errors.append(error)

        return np.mean(errors)
        
        
    def optimize_gamma(self, data):
        """改进的gamma优化方法"""
        from scipy.optimize import differential_evolution

        # 使用全局优化算法
        result = differential_evolution(
            lambda gamma: self.error_function(gamma, data),
            bounds=[self.config['param_bounds']['gamma']],
            init='latinhypercube',
            maxiter=100
        )
        return result.x