"""
加入遗忘
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple, List
from .M_base import M_Base

@dataclass
class ModelParams:
    k: int
    beta: float
    gamma: float

class M_Fgt(M_Base):
    def __init__(self, config):
        self.config = config
        self.all_centers = None

    def prior(self, params: ModelParams, condition: int) -> float:
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        max_k = len(centers)
        k_prior = 1/max_k if 1 <= params.k <= max_k else 0
        beta_prior = 1 if params.beta > 0 else 0
        gamma_prior = 1 if 0 <= params.gamma <= 1 else 0
        return k_prior * beta_prior * gamma_prior
    
    def likelihood(self, params: ModelParams, data, condition: int) -> np.ndarray:
        k, beta, gamma = params.k, params.beta, params.gamma
        x = data[['feature1', 'feature2', 'feature3', 'feature4']].values
        c = data['choice'].values
        r = data['feedback'].values

        centers = self.get_centers(k, condition)
        distances = distances = np.linalg.norm(x[:, np.newaxis, :] - np.array(centers), axis=2)
        
        probs = np.exp(-beta * distances)
        probs /= np.sum(probs, axis=1, keepdims=True)
        p_c = probs[np.arange(len(c)), c - 1]
        base_likelihood = np.where(r == 1, p_c, 1 - p_c)

        # 记忆衰减
        base_weight = 0.1
        decay_weights = gamma ** np.arange(len(data)-1, -1, -1)
        memory_weights = base_weight + (1-base_weight) * decay_weights

        weighted_log_likelihood = memory_weights * np.log(base_likelihood)

        return np.exp(weighted_log_likelihood)

    def fit_with_given_gamma(self, data, gamma: float) -> Tuple[ModelParams, float, float, Dict]:
        """在给定gamma时优化k和beta"""
        condition = data['condition'].iloc[0]
        max_k = self.get_max_k(condition)
        
        k_results = {}  # 存储每个k的优化结果
        
        for k in range(1, max_k + 1):
            result = minimize(
                lambda beta: self.posterior(ModelParams(k, beta[0], gamma), data, condition),
                x0=[self.config['param_inits']['beta']],
                bounds=[self.config['param_bounds']['beta']]
            )
            
            beta_opt, log_posterior = result.x[0], -result.fun
            
            log_likelihood = np.sum(np.log(
                self.likelihood(ModelParams(k, beta_opt, gamma), data, condition)
            ))

            # 保存结果
            k_results[k] = {
                'beta': beta_opt,
                'log_likelihood': log_likelihood,
                'log_posterior': log_posterior
            }
        
        # 找到具有最大对数后验的k
        best_k = max(k_results, key=lambda x: k_results[x]['log_posterior'])
        best_entry = k_results[best_k]
        best_params = ModelParams(k=best_k, beta=best_entry['beta'])
        best_log_likelihood = best_entry['log_likelihood']
        best_log_posterior = best_entry['log_posterior']

        # Normalize posteriors
        log_posteriors = [entry['log_posterior'] for entry in k_results.values()]
        max_log = max(log_posteriors)
        k_posteriors = {k: np.exp(lp - max_log) for k, lp in zip(k_results.keys(), log_posteriors)}
        total = sum(k_posteriors.values())
        details = {
            k: {
                'beta': k_results[k]['beta'],
                'posterior_prob': prob / total,  # 归一化后的后验概率
                'log_likelihood': k_results[k]['log_likelihood'],
                'log_posterior': k_results[k]['log_posterior']
            }
            for k, prob in k_posteriors.items()
        }
        
        return best_params, best_log_likelihood, best_log_posterior, details

    def fit_trial_by_trial(self, data, gamma):
        step_results = []
        for step in range(1, len(data)+1):
            trial_data = data.iloc[:step]
            fitted_params, best_ll, best_post, details = self.fit_with_given_gamma(trial_data, gamma)
            
            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'best_log_likelihood': best_ll,
                'best_posterior': best_post,
                'details': details,
                'params': fitted_params
            })
        
        return step_results

    # 定义目标函数
    def error_function(self, gamma, data, window_size=16):
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
                condition = trial['condition'] 
                x = trial[['feature1', 'feature2', 'feature3', 'feature4']].values
                true_category = int(trial['category'])

                true_category = np.where(condition == 1, 
                                        np.where(np.isin(true_category, [1, 2]), 1, 2), 
                                        true_category)

                centers = self.get_centers(fitted_params.k, condition)
                    
                distances = np.linalg.norm(x - np.array(centers), axis=1)
                probs = np.exp(-fitted_params.beta * distances)
                probs /= np.sum(probs)

                p_true = probs[true_category - 1]

                predicted.append(p_true)
            
            pred_accuracy = np.mean(predicted)
            error = abs(pred_accuracy - true_accuracy)

            errors.append(error)

        return np.mean(errors)

    def optimize_gamma(self, data):
        """优化gamma"""

        # 使用minimize来最小化目标函数
        result = minimize(
            lambda gamma: self.error_function(gamma, data),
            x0=[self.config['param_inits']['gamma']],
            bounds=[self.config['param_bounds']['gamma']],
            method='L-BFGS-B'
        )
        
        # 获取最优gamma
        best_gamma = result.x[0]
        return best_gamma

    def fit(self, data):
        step_results = []
        best_gammas = []

        best_gamma = self.optimize_gamma(data)
        best_gammas.append(best_gamma)

        # 逐试次拟合k和beta
        step_results_for_block = self.fit_trial_by_trial(data, best_gamma)
        for result in step_results_for_block:
            step_results.append(result)
        
        return step_results, best_gammas
