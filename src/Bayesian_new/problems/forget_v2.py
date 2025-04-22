"""
Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
from itertools import product
import numpy as np
from tqdm import tqdm
from itertools import product
from scipy.optimize import minimize

from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
from .partitions import Partition, BasePartition
from .model import BaseModelParams, BaseModel, SingleRationalModel, PartitionLikelihood
from .base_problem import softmax, cdist, euc_dist, two_factor_decay

HYPO_CLUSTER_PROTOTYPE_AMOUNT = 1

@dataclass(unsafe_hash=True)
class ForgetModelParams(BaseModelParams):
    """
    Extended parameter class that includes forgetting parameters.

    Attributes
    ----------
    k : int
        Index of the partition method (inherited from ModelParams).
    beta : float
        Softness of the partition (inherited from ModelParams).
    gamma : float
        Memory decay rate.
    w0 : float
        Base memory weight.
    """
    gamma: float
    w0: float


class ForgetModel(SingleRationalModel):
    """
    A model that adds a forgetting mechanism on top of SingleRationalModel.
    """

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)

        personal_memory_range = kwargs.pop("personal_memory_range", {"gamma": (0.05, 1.0), "w0": (0.00375, 0.075)})
        param_resolution = 20
        # personal_memory_range = kwargs.pop("personal_memory_range", {"gamma": (0.1, 1.0), "w0": (0.01, 0.1)})
        # param_resolution = 10
        
        # 初始化参数搜索空间
        self.gamma_values = np.linspace(*personal_memory_range["gamma"], param_resolution, endpoint=True)
        self.w0_values = np.linspace(*personal_memory_range["w0"], param_resolution, endpoint=True)
        # self.w0_values = [personal_memory_range["w0"][1] / (i + 1) for i in range(param_resolution)]

    def fit_single_step(self,
                        data: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                        gamma: float, 
                        w0: float, 
                        **kwargs) -> Tuple[ForgetModelParams, float, Dict, Dict]:
        """
        Optimize beta for each hypothesis while using fixed gamma and w0.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing (stimuli, choices, responses).
        gamma : float
            Memory decay rate.
        w0 : float
            Base memory weight.
        **kwargs : dict
            Additional arguments passed to the likelihood function.

        Returns
        -------
        best_params : ForgetModelParams
            Best parameters (including k, beta, gamma, w0).
        best_ll : float
            The maximum log-likelihood found.
        all_hypo_params : Dict
            A mapping from hypothesis index to fitted ForgetModelParams.
        all_hypo_ll : Dict
            A mapping from hypothesis index to its best log-likelihood.
        """
        all_hypo_params = {}
        all_hypo_ll = {}

        def _ll_per_hypo(beta, gamma, w0, hypo=None):
            likelihood = self.partition_model.calc_likelihood_entry(
                hypo, data, beta[0], gamma=gamma, w0=w0, **kwargs)
            return np.sum(np.log(np.maximum(likelihood, 0)), axis=0)

        for hypo in self.hypotheses_set:
            result = minimize(lambda beta: -_ll_per_hypo(beta, gamma, w0, hypo),
                              x0=[self.config["param_inits"]["beta"]],
                              bounds=[self.config["param_bounds"]["beta"]])
            beta_opt, ll_max = result.x[0], -result.fun

            all_hypo_params[hypo] = ForgetModelParams(hypo,
                                                      beta_opt,
                                                      gamma=gamma,
                                                      w0=w0)
            all_hypo_ll[hypo] = ll_max

        best_hypo = max(all_hypo_ll, key=all_hypo_ll.get)
        return (all_hypo_params[best_hypo], all_hypo_ll[best_hypo],
                all_hypo_params, all_hypo_ll)

    def fit_step_by_step(self, 
                         data: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                         gamma: float, 
                         w0: float,
                         limited_hypos_list: List[List[int]] = None, 
                         **kwargs) -> List[Dict]:
        
        """
        Fit the model step-by-step with fixed gamma and w0.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple (stimuli, choices, responses).
        gamma : float
            Memory decay rate.
        w0 : float
            Base memory weight.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the fitting results for each step.
        """

        stimulus, _, responses = data
        n_trials = len(responses)

        # Precompute all distances
        self.partition_model.precompute_all_distances(stimulus)

        step_results = []        

        for step_idx in range(1, n_trials + 1):
            selected_data = [x[:step_idx] for x in data]

            if limited_hypos_list is not None:
                # Use a limited subset of hypotheses for this trial
                current_hypos = limited_hypos_list[step_idx - 1]
                new_hypotheses_set = BaseSet(current_hypos)

                new_prior = BasePrior(new_hypotheses_set)
                new_likelihood = PartitionLikelihood(new_hypotheses_set, self.partition_model)

                # Refresh the engine
                self.refresh_engine(new_hypotheses_set, new_prior, new_likelihood)
            else:
                # If limited_hypos_list is None, keep using self.hypotheses_set as is
                pass

            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit_single_step(
                selected_data, gamma, w0, use_cached_dist=True,
                                                 **kwargs)

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            all_hypo_post = self.engine.infer_log(
                selected_data,
                use_cached_dist=True,
                beta=hypo_betas,
                gamma=gamma,
                w0=w0,
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

        return step_results


    def compute_error_for_params(self, 
                                 data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                 gamma: float, 
                                 w0: float,
                                 limited_hypos_list: bool = False,
                                 window_size=16,
                                 **kwargs) -> Tuple[List[Dict], float]:
        """
        Perform a single pass of model fitting and prediction, then compute an error metric.

        Parameters
        ----------
        data : tuple
            A tuple containing (stimuli, choices, responses, categories).
        gamma : float
            Memory decay rate.
        w0 : float
            Base memory weight.
        window_size : int
            The window size used for sliding accuracy measurements.

        Returns
        -------
        step_results : List[Dict]
            Output of fit_step_by_step showing model fits for each trial.
        mean_error : float
            The average windowed error between predicted and true accuracy.
        """
        # Fit the model with fixed gamma and w0
        selected_data = data[:3]
        step_results = self.fit_step_by_step(selected_data, gamma, w0, limited_hypos_list, **kwargs)
        
        # Get the predicted accuracy
        predict_results = self.predict_choice(
            data, 
            step_results,
            use_cached_dist=True, 
            window_size=window_size
        )
        
        # Calculate the mean absolute error between predicted and true accuracy
        mean_error = np.mean(
            np.abs(
                np.array(predict_results['sliding_true_acc']) 
                - np.array(predict_results['sliding_pred_acc'])
            )
        )
        
        return step_results, mean_error

    def optimize_params(self, 
                        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                        cluster: bool = False,
                        **kwargs) -> Tuple[ForgetModelParams, list]:
        """
        Perform a 2D grid search over gamma and w0 to find the parameters
        that minimize the prediction error.

        Parameters
        ----------
        data : tuple
            A tuple containing (stimuli, choices, responses, categories).

        Returns
        -------
        Dict
            A dictionary containing:
            - 'best_params': The (gamma, w0) pair that yields the minimal error.
            - 'best_error': The best error value found.
            - 'best_step_results': The step-by-step fit results for the best params.
            - 'grid_errors': A dictionary mapping parameter pairs to error values.
        """
        grid_errors = {}
        grid_step_results = {}

        total_combinations = len(self.gamma_values) * len(self.w0_values)
        for gamma, w0 in tqdm(
            product(self.gamma_values, self.w0_values),
            desc="Gamma-W0 Grid Search",
            total=total_combinations):

            step_results, mean_error = self.compute_error_for_params(data, gamma, w0, cluster, window_size=16, **kwargs)
            
            key = (round(gamma, 2), round(w0, 5))
            grid_errors[key] = mean_error
            grid_step_results[key] = step_results

        # Identify the best (gamma, w0) by minimum error
        best_key = min(grid_errors, key=lambda k: grid_errors[k])

        optimize_results = {
            'best_params': best_key,
            'best_error': grid_errors[best_key],
            'best_step_results': grid_step_results[best_key],
            'grid_errors': grid_errors
        }

        return optimize_results


    def oral_generate_hypos(self,
                            data: Tuple[np.ndarray, np.ndarray],
                            top_k: int = 10,
                            dist_tol: float = 1e-9) -> List[List[int]]:
        """
        Generate a limited hypothesis set for each trial based on the
        participant's verbally reported center and the chosen category.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray]
            data[0] : (n_trials, n_dims) containing the verbally reported category center per trial.
            data[1] : (n_trials,) containing the chosen category index (1-indexed).
        top_k : int
            The number of closest hypotheses to choose if no exact match is found.
        dist_tol : float
            Tolerance within which distances are considered zero (floating-point errors).

        Returns
        -------
        List[List[int]]
            A list of length n_trials, where each element is a list of hypothesis indices.
        """

        oral_centers, choices = data
        n_trials = len(choices)

        n_hypos = self.partition_model.prototypes_np.shape[0]
        all_hypos = range(n_hypos)

        limited_hypos_list = []

        for trial_idx in range(n_trials):
            cat_idx = choices[trial_idx] - 1
            reported_center = oral_centers[trial_idx]

            distance_map = []
            for hypo_idx in all_hypos:
                # Compare with the (cat_idx)-th category prototype of hypothesis h
                true_center = self.partition_model.prototypes_np[hypo_idx, 0, cat_idx, :]
                distance_val = np.linalg.norm(reported_center - true_center)
                distance_map.append((distance_val, hypo_idx))

            # Check for exact matches within tolerance
            exact_matches = [hypo_idx for (dist, hypo_idx) in distance_map if dist <= dist_tol]

            if len(exact_matches) > 0:
                chosen_hypos = exact_matches
            else:
                distance_map.sort(key=lambda x: x[0])
                chosen_hypos = [hypo_idx for (_, hypo_idx) in distance_map[:top_k]]

            limited_hypos_list.append(chosen_hypos)

        return limited_hypos_list



@dataclass(unsafe_hash=True)
class AdaptiveAmnesiaParams(BaseModelParams):
    alpha_gamma: float  # gamma更新速率
    alpha_w0: float  # w0更新速率

class AdaptiveAmnesiaModel(ForgetModel):
    """
    实现“动态” trial-specific 遗忘机制。
    """
    def __init__(self, config: Dict, 
                 base_gamma: float,
                 base_w0: float,
                 alpha_gamma_values: np.ndarray = None,
                 alpha_w0_values: np.ndarray = None,
                 **kwargs):
        super().__init__(config, **kwargs)

        self.base_gamma = base_gamma
        self.base_w0 = base_w0

        self.alpha_gamma_values = alpha_gamma_values if alpha_gamma_values is not None else np.linspace(0., 0.1, 11, endpoint=True)
        self.alpha_w0_values = alpha_w0_values if alpha_w0_values is not None else np.linspace(0., 0.1, 11, endpoint=True)

    def precompute_distances(self, stimuli: np.ndarray):
        """
        一次性预计算好所有 trial 的距离
        """
        if hasattr(self.partition_model, "precompute_all_distances"):
            self.partition_model.precompute_all_distances(stimuli)

    def _build_amnesia_func(self, gamma_list: List[float], w0_list: List[float]):
        """
        step is NOT in use.
        给定 (gamma_list, w0_list) 和当前 step,
        返回一个 callable: trial_specific_amnesia
        其中:
           coeff[i] = w0_list[i] + (1 - w0_list[i]) * (gamma_list[i])^(step-1 - i)
        """

        def trial_specific_amnesia(data, **_):
            n = len(data[0])
            coeff = np.ones(n, dtype=float)

            for iTrial in range(n):
                g_i = gamma_list[iTrial]
                w_i = w0_list[iTrial]
                exponent = max(0, (n - 1 - iTrial))
                coeff[iTrial] = (w_i) + (1 - w_i)*(g_i**exponent)
            return coeff
        
        return trial_specific_amnesia

    def fit_single_step(
            self, data: tuple, 
            alpha_gamma: float,
            alpha_w0: float,
            gamma_list: List[float],
            w0_list: List[float],
            **kwargs) -> Tuple[AdaptiveAmnesiaParams, float, Dict, Dict]:
        """
        在给定 gamma_list, w0_list(长度=step)下拟合数据

        Returns:
          best_params, best_ll, all_hypo_params, all_hypo_ll
        """
        # 1) 构造 adaptive_amnesia 函数
        amnesia_func = self._build_amnesia_func(gamma_list, w0_list)

        # 2) 定义对数似然(对每个hypo)
        def _ll_per_hypo(beta, hypo=None):
            likelihood = self.partition_model.calc_likelihood_entry(
                hypo, data, beta[0], adaptive_amnesia=amnesia_func, **kwargs)
            return np.sum(np.log(np.maximum(likelihood, 0)), axis=0)

        # 3) 遍历所有 hypo, 优化 beta
        all_hypo_params = {}
        all_hypo_ll = {}

        for hypo in self.hypotheses_set:
            result = minimize(lambda beta: -_ll_per_hypo(beta, hypo),
                              x0=[self.config["param_inits"]["beta"]],
                              bounds=[self.config["param_bounds"]["beta"]])
            beta_opt, ll_max = result.x[0], -result.fun

            all_hypo_params[hypo] = AdaptiveAmnesiaParams(hypo, beta_opt, alpha_gamma, alpha_w0)
            all_hypo_ll[hypo] = ll_max

        best_hypo = max(all_hypo_ll, key=all_hypo_ll.get)
        return (all_hypo_params[best_hypo], all_hypo_ll[best_hypo],
                all_hypo_params, all_hypo_ll)

    def fit_step_by_step(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        alpha_gamma: float,
        alpha_w0: float,
        **kwargs
    ):
        """
        在每个step都:
          - 根据当前 step-1 之前的 posterior, 计算 gamma_i, w0_i
          - 插入到 gamma_list, w0_list
          - 调用 fit_single_step(sub_data, gamma_list, w0_list)
        """
        stimuli, choices, responses = data
        n_trials = len(responses)
        
        # 先对 partition 做一次 precompute_all_distances
        self.partition_model.precompute_all_distances(stimuli)

        gamma_list = [self.base_gamma]*n_trials
        w0_list    = [self.base_w0]*n_trials

        step_results = []

        # 初始 posterior
        prior = np.ones(self.hypotheses_set.length) / self.hypotheses_set.length
        prev_posterior = prior

        for step in range(1, n_trials+1):
            trial_data = [x[:step] for x in data]

            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit_single_step(
                trial_data, alpha_gamma, alpha_w0, gamma_list[:step], w0_list[:step], 
                use_cached_dist=False, **kwargs
            )

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            amnesia_func = self._build_amnesia_func(gamma_list[:step], w0_list[:step])
            all_hypo_post = self.engine.infer_log(
                trial_data,
                use_cached_dist=False,
                beta=hypo_betas,
                adaptive_amnesia=amnesia_func,
                normalized=True
            )

            hypo_details = {}
            for i, hypo in enumerate(self.hypotheses_set.elements):
                hypo_details[hypo] = {
                    'beta_opt': all_hypo_params[hypo].beta,
                    'll_max': all_hypo_ll[hypo],
                    'post_max': all_hypo_post[i],
                    'is_best': hypo == best_params.k
                }

            # 更新 gamma_list[i], w0_list[i]
            delta_post_i = np.sum(np.abs(all_hypo_post - prev_posterior))/2
            new_gi = max(0., min(1., (1- alpha_gamma) * self.base_gamma + alpha_gamma * delta_post_i))
            new_wi = max(0., min(1., (1- alpha_w0) * self.base_w0 + alpha_w0 * delta_post_i))
            gamma_list[step-1] = new_gi
            w0_list[step-1] = new_wi

            # 更新 prev_posterior 为当前 step 的 posterior
            prev_posterior = all_hypo_post.copy()

            step_results.append({
                'best_k': best_params.k,
                'best_beta': best_params.beta,
                'best_params': best_params,
                'best_log_likelihood': best_ll,
                'best_norm_posterior': np.max(all_hypo_post),
                'hypo_details': hypo_details,
                'gamma_list': gamma_list[:step].copy(),
                'w0_list': w0_list[:step].copy()
            })

        return step_results

    def optimize_params(
            self, data_with_cat: tuple) -> Tuple[AdaptiveAmnesiaParams, list]:
        """二维网格搜索优化gamma和w0"""
        grid_errors = {}
        grid_step_results = {}

        total_combinations = len(self.alpha_gamma_values) * len(self.alpha_w0_values)
        for alpha_gamma, alpha_w0 in tqdm(
            product(self.alpha_gamma_values, self.alpha_w0_values),
            desc="Alpha_gamma-Alpha_w0 Grid Search",
            total=total_combinations):

            step_results, mean_error = self.compute_error_for_params(data_with_cat, alpha_gamma, alpha_w0, window_size=16)

            key = (round(alpha_gamma, 2), round(alpha_w0, 2))
            grid_errors[key] = mean_error
            grid_step_results[key] = step_results

        # 查找最优参数
        best_key = min(grid_errors, key=lambda k: grid_errors[k])

        optimize_results = {
            'best_params': best_key,
            'best_error': grid_errors[best_key],
            'best_step_results': grid_step_results[best_key],
            'grid_errors': grid_errors
        }

        return optimize_results