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
from .model import ModelParams, BaseModel
from .base_problem import softmax, cdist, euc_dist, two_factor_decay


@dataclass(unsafe_hash=True)
class ForgetModelParams(ModelParams):
    """扩展参数类，添加遗忘参数"""
    gamma: float  # 遗忘率参数
    w0: float  # 基础记忆权重


class ForgetModel(BaseModel):
    """
    Add forgetting mechanism
    """

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)

        # 初始化参数搜索空间
        self.gamma_values = np.arange(0.1, 1.1, 0.1)
        self.w0_values = np.array([0.1 / i for i in range(1, 11)])

        # 初始化记忆权重缓存
        self.memory_weights_cache: Dict = {}

    def fit_with_given_params(
            self, data: tuple, gamma: float, w0: float,
            **kwargs) -> Tuple[ForgetModelParams, float, Dict, Dict]:
        """
        Fit
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

    def fit_trial_by_trial(self, data: Tuple[np.ndarray, np.ndarray,
                                             np.ndarray], gamma, w0):
        step_results = []
        nTrial = len(data[2])

        for step in range(nTrial, 0, -1):
            trial_data = [x[:step] for x in data]
            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit_with_given_params(
                trial_data, gamma, w0, use_cached_dist=(step != nTrial))

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            all_hypo_post = self.engine.infer_log(trial_data,
                                                  use_cached_dist=(step
                                                                   != nTrial),
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

        return step_results[::-1]

    def error_function(self,
                       data_with_cat: tuple,
                       step_results: list,
                       use_cached_dist,
                       window_size=16) -> float:
        """
        计算滑动窗口预测误差
        输入数据需包含category信息：(stimuli, choices, responses, categories)
        """
        # 解包带类别信息的数据
        stimuli, choices, responses, categories = data_with_cat
        n_trials = len(responses)

        # 计算每个试次的预测准确率
        predicted_acc = []
        for i in range(n_trials):

            # 构造单个试次数据
            trial_data = ([stimuli[i]], [choices[i]], [responses[i]],
                  [categories[i]])  
            
            hypo_details = step_results[i]['hypo_details']
            post_max = [hypo_details[k]['post_max']
                for k in hypo_details.keys()]

            weighted_p_true = 0
            for k, post in zip(hypo_details.keys(), post_max):
                p_true = self.partition_model.calc_trueprob_entry(
                    k, trial_data, hypo_details[k]['beta_opt'], use_cached_dist=use_cached_dist, indices=[i])
                weighted_p_true += post * p_true
            predicted_acc.append(weighted_p_true)

        # 转换为numpy数组
        pred_acc = np.array(predicted_acc)
        true_acc = (np.array(responses) == 1).astype(float)

        # 滑动窗口误差计算
        pred_acc_avg = []
        true_acc_avg = []
        errors = []
        for start in range(len(pred_acc) - window_size + 1):
            window_pred = pred_acc[start:start + window_size]
            window_true = true_acc[start:start + window_size]
            pred_acc_avg.append(np.mean(window_pred))
            true_acc_avg.append(np.mean(window_true))
            errors.append(np.abs(np.mean(window_pred) - np.mean(window_true)))

        error_info = {
            'pred_acc': pred_acc,
            'true_acc': true_acc,
            'pred_acc_avg': pred_acc_avg,
            'true_acc_avg': true_acc_avg,
            'errors': errors
        }

        return error_info

    def optimize_params(
            self, data_with_cat: tuple) -> Tuple[ForgetModelParams, list]:
        """二维网格搜索优化gamma和w0"""
        grid_errors = {}
        grid_step_results = {}

        # 解包数据
        s_data = data_with_cat[:3]  # (stimuli, choices, responses)

        # 网格搜索
        for gamma, w0 in tqdm(product(self.gamma_values, self.w0_values),
                              desc="Gamma-W0", total=100):
            # 逐试次拟合
            step_results = self.fit_trial_by_trial(s_data, gamma, w0)
            # 计算误差
            error_info = self.error_function(data_with_cat, step_results, use_cached_dist=True)
            error = np.mean(error_info['errors'])
            # 记录结果
            key = (round(gamma, 2), round(w0, 4))
            grid_errors[key] = error
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



@dataclass(unsafe_hash=True)
class AdaptiveAmnesiaParams(ModelParams):
    alpha_gamma: float  # gamma更新速率
    alpha_w0: float  # w0更新速率

class AdaptiveAmnesiaModel(BaseModel):
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

        self.alpha_gamma_values = alpha_gamma_values if alpha_gamma_values is not None else np.arange(0.0, 1.05, 0.1)
        self.alpha_w0_values = alpha_w0_values if alpha_w0_values is not None else np.arange(0.0, 1.05, 0.1)

    def precompute_distances(self, stimuli: np.ndarray):
        """
        一次性预计算好所有 trial 的距离
        """
        if hasattr(self.partition_model, "precompute_all_distances"):
            self.partition_model.precompute_all_distances(stimuli)

    def _build_amnesia_func(self, gamma_list: List[float], w0_list: List[float], step: int):
        """
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
                exponent = max(0, (step-1 - iTrial))
                coeff[iTrial] = (w_i) + (1 - w_i)*(g_i**exponent)
            return coeff
        
        return trial_specific_amnesia

    def fit_with_given_params(
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
        step = len(data[2])
        amnesia_func = self._build_amnesia_func(gamma_list, w0_list, step)

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

    def fit_trial_by_trial(
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
          - 调用 fit_with_given_params(sub_data, gamma_list, w0_list)
        """
        stimuli, choices, responses = data
        nTrial = len(responses)
        
        # 先对 partition 做一次 precompute_all_distances
        self.partition_model.precompute_all_distances(stimuli)

        gamma_list = [self.base_gamma]*nTrial
        w0_list    = [self.base_w0]*nTrial

        step_results = []

        # 初始 posterior
        prior = np.ones(self.hypotheses_set.length) / self.hypotheses_set.length
        prev_posterior = prior

        for step in range(1, nTrial+1):
            trial_data = [x[:step] for x in data]

            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit_with_given_params(
                trial_data, alpha_gamma, alpha_w0, gamma_list[:step], w0_list[:step], step=step,
                use_cached_dist=False, **kwargs
            )

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            amnesia_func = self._build_amnesia_func(gamma_list[:step], w0_list[:step], step=step)
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
            delta_post_i = np.sum(np.abs(all_hypo_post - prev_posterior))
            new_gi = max(0., min(1., self.base_gamma + alpha_gamma * delta_post_i))
            new_wi = max(0., min(1., self.base_w0 + alpha_w0 * delta_post_i))
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

    def error_function(self,
                       data_with_cat: tuple,
                       step_results: list,
                       use_cached_dist, 
                       window_size=16) -> float:
        """
        计算滑动窗口预测误差
        输入数据需包含category信息：(stimuli, choices, responses, categories)
        """
        # 解包带类别信息的数据
        stimuli, choices, responses, categories = data_with_cat
        n_trials = len(responses)

        # 计算每个试次的预测准确率
        predicted_acc = []
        for i in range(n_trials):

            # 构造单个试次数据
            trial_data = ([stimuli[i]], [choices[i]], [responses[i]],
                  [categories[i]])  
            
            amnesia_func = self._build_amnesia_func(step_results[i]['gamma_list'], step_results[i]['w0_list'], step=i+1)

            hypo_details = step_results[i]['hypo_details']
            post_max = [hypo_details[k]['post_max']
                for k in hypo_details.keys()]

            weighted_p_true = 0
            for k, post in zip(hypo_details.keys(), post_max):
                p_true = self.partition_model.calc_trueprob_entry(
                    k, trial_data, hypo_details[k]['beta_opt'], use_cached_dist=use_cached_dist, indices=[i],
                    adaptive_amnesia=amnesia_func)
                weighted_p_true += post * p_true
            predicted_acc.append(weighted_p_true)

        # 转换为numpy数组
        pred_acc = np.array(predicted_acc)
        true_acc = (np.array(responses) == 1).astype(float)

        # 滑动窗口误差计算
        pred_acc_avg = []
        true_acc_avg = []
        errors = []
        for start in range(len(pred_acc) - window_size + 1):
            window_pred = pred_acc[start:start + window_size]
            window_true = true_acc[start:start + window_size]
            pred_acc_avg.append(np.mean(window_pred))
            true_acc_avg.append(np.mean(window_true))
            errors.append(np.abs(np.mean(window_pred) - np.mean(window_true)))

        error_info = {
            'pred_acc': pred_acc,
            'true_acc': true_acc,
            'pred_acc_avg': pred_acc_avg,
            'true_acc_avg': true_acc_avg,
            'errors': errors
        }

        return error_info

    def optimize_params(
            self, data_with_cat: tuple) -> Tuple[AdaptiveAmnesiaParams, list]:
        """二维网格搜索优化gamma和w0"""
        grid_errors = {}
        grid_step_results = {}

        # 解包数据
        s_data = data_with_cat[:3]  # (stimuli, choices, responses)

        # 网格搜索
        for alpha_gamma, alpha_w0 in tqdm(product(self.alpha_gamma_values, self.alpha_w0_values),
                              desc="Alpha_g-Alpha_w", total=100):
            # 逐试次拟合
            step_results = self.fit_trial_by_trial(s_data, alpha_gamma, alpha_w0)
            # 计算误差
            error_info = self.error_function(data_with_cat, step_results, use_cached_dist=True)
            error = np.mean(error_info['errors'])
            # 记录结果
            key = (round(alpha_gamma, 2), round(alpha_w0, 2))
            grid_errors[key] = error
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