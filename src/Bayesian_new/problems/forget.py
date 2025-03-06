"""
Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
from itertools import product
import numpy as np
from tqdm import tqdm
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
        self.w0_values = np.arange(0.1, 1.1, 0.1)

        # 初始化记忆权重缓存
        self.memory_weights_cache: Dict = {}

    def _get_memory_weights(self, n_trials: int, gamma: float,
                            w0: float) -> np.ndarray:
        """预计算记忆衰减权重"""
        cache_key = (n_trials, gamma, w0)
        if cache_key not in self.memory_weights_cache:
            decay = gamma**np.arange(n_trials - 1, -1, -1)
            weights = w0 + (1 - w0) * decay
            self.memory_weights_cache[cache_key] = weights / np.sum(weights)
        return self.memory_weights_cache[cache_key]

    def get_weighted_log_likelihood(self, hypo: int, data: tuple, beta: float,
                                    gamma: float, w0: float,
                                    **kwargs) -> float:
        """计算加权后的似然值"""
        # 获取基础似然
        base_likelihood = self.partition_model.calc_likelihood_entry(
            hypo, data, beta, **kwargs)

        # 应用记忆衰减权重
        n_trials = len(data[2])
        memory_weights = self._get_memory_weights(n_trials, gamma, w0)
        weighted_log_likelihood = memory_weights * np.log(base_likelihood)
        return np.sum(weighted_log_likelihood)

    def fit_with_given_params(
            self, data: tuple, gamma: float, w0: float,
            **kwargs) -> Tuple[ForgetModelParams, float, Dict, Dict]:
        """
        Fit
        """
        all_hypo_params = {}
        all_hypo_ll = {}

        for hypo in self.hypotheses_set.elements:
            result = minimize(lambda beta: -self.get_weighted_log_likelihood(
                hypo, data, beta, gamma, w0, **kwargs),
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
            params = step_results[i]['best_params']
            k, beta = params.k, params.beta

            # 构造单个试次数据
            trial_data = ([stimuli[i]], [choices[i]], [responses[i]],
                          [categories[i]])

            p_true = self.partition_model.calc_trueprob_entry(
                k, trial_data, beta, use_cached_dist=True, indices=[i])

            predicted_acc.append(p_true)

        # 转换为numpy数组
        pred_acc = np.array(predicted_acc)
        true_acc = (np.array(responses) == 1).astype(float)

        # 滑动窗口误差计算
        errors = []
        for start in range(len(pred_acc) - window_size + 1):
            window_pred = pred_acc[start:start + window_size]
            window_true = true_acc[start:start + window_size]
            errors.append(np.abs(np.mean(window_pred) - np.mean(window_true)))

        return np.mean(errors)

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
            error = self.error_function(data_with_cat, step_results)
            # 记录结果
            key = (round(gamma, 2), round(w0, 2))
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
