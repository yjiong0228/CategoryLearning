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
    w0: float

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
        w0_prior = 1 if 0 <= params.w0 <= 1 else 0  
        return k_prior * beta_prior * gamma_prior * w0_prior
    
    def likelihood(self, params: ModelParams, data, condition: int) -> np.ndarray:
        k, beta, gamma, w0 = params.k, params.beta, params.gamma, params.w0
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
        base_weight = w0
        decay_weights = gamma ** np.arange(len(data)-1, -1, -1)
        memory_weights = base_weight + (1-base_weight) * decay_weights

        weighted_log_likelihood = memory_weights * np.log(base_likelihood)

        return np.exp(weighted_log_likelihood)

    def fit_with_given_params(self, data, gamma: float, w0: float) -> Tuple[ModelParams, float, float, Dict]:
        """在给定gamma和w0时优化k和beta"""
        condition = data['condition'].iloc[0]
        max_k = self.get_max_k(condition)
        
        best_params = None
        best_log_likelihood = -np.inf
        best_posterior = -np.inf
        k_posteriors = {}
        
        for k in range(1, max_k + 1):
            result = minimize(
                lambda beta: self.posterior(ModelParams(k, beta, gamma, w0), data, condition),
                x0=[self.config['param_inits']['beta']],
                bounds=[self.config['param_bounds']['beta']]
            )
            
            beta_opt, posterior_opt = result.x[0], -result.fun
            k_posteriors[k] = posterior_opt
            
            log_likelihood = np.sum(np.log(
                self.likelihood(ModelParams(k, beta_opt, gamma, w0), data, condition)
            ))

            if posterior_opt > best_posterior:
                best_params = ModelParams(k=k, beta=beta_opt, gamma=gamma, w0=w0)
                best_log_likelihood, best_posterior = log_likelihood, posterior_opt
        
        # Normalize posteriors
        max_log_posterior = max(k_posteriors.values())
        k_posteriors = {k: np.exp(log_p - max_log_posterior) for k, log_p in k_posteriors.items()}
        total = sum(k_posteriors.values())
        k_posteriors = {k: p / total for k, p in k_posteriors.items()}
        
        return best_params, best_log_likelihood, best_posterior, k_posteriors

    def fit_trial_by_trial(self, data, gamma, w0):
        step_results = []
        for step in range(1, len(data)+1):
            trial_data = data.iloc[:step]
            fitted_params, best_ll, best_post, k_post = self.fit_with_given_params(trial_data, gamma, w0)
            
            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'best_log_likelihood': best_ll,
                'best_posterior': best_post,
                'k_posteriors': k_post,
                'params': fitted_params
            })
        
        return step_results

    # 定义目标函数
    def error_function(self, params, data, window_size=16):
        """
        误差目标函数，使用滑动窗口计算每个窗口的预测准确率与真实准确率之间的差异。
        
        Args:
            gamma (float): 当前的gamma值
            block_data (DataFrame): 数据集
            window_size (int): 滑动窗口的大小，默认为16
            
        Returns:
            float: 误差（目标函数值）
        """
        # 拟合k和beta
        gamma, w0 = params
        step_results = self.fit_trial_by_trial(data, gamma, w0)

        # 预计算所有试次的预测概率
        predicted = []
        for j, trial in data.iterrows():
            fitted_params = step_results[j-data.index[0]]['params']
            condition = trial['condition'] 
            x = trial[['feature1', 'feature2', 'feature3', 'feature4']].values

            centers = self.get_centers(fitted_params.k, condition)
            distances = np.linalg.norm(x - np.array(centers), axis=1)
            probs = np.exp(-fitted_params.beta * distances)
            probs /= np.sum(probs)

            true_category = int(trial['category'])
            true_category = np.where(condition == 1, 
                                    np.where(np.isin(true_category, [1, 2]), 1, 2), 
                                    true_category)
            p_true = probs[true_category - 1]
            predicted.append(p_true)

        predicted = np.array(predicted)
        true_accuracy = (data['feedback'] == 1).values.astype(float)  # 转换为0/1数组

        # 计算滑动窗口误差
        n_trials = len(data)
        errors = []
        for start_idx in range(n_trials - window_size + 1):
            end_idx = start_idx + window_size
            
            # 获取当前窗口数据
            window_true = true_accuracy[start_idx:end_idx]
            window_pred = predicted[start_idx:end_idx]
            
            # 计算误差
            true_acc = np.mean(window_true)
            pred_acc = np.mean(window_pred)
            errors.append(abs(pred_acc - true_acc))

        return np.mean(errors), step_results

    def optimize_params(self, data):
        """二维网格搜索优化gamma和w0"""
        param_values = {
            'gamma': np.arange(0.1, 1.1, 0.1),
            'w0': np.arange(0.1, 1.1, 0.1)
        }
        
        best_params = (None, None)
        best_error = float('inf')
        best_step_results = None

        for gamma in param_values['gamma']:
            for w0 in param_values['w0']:
                error, step_results = self.error_function((gamma, w0), data)
                if error < best_error:
                    best_error = error
                    best_params = (gamma, w0)
                    best_step_results = step_results
        
        return best_params, best_step_results

    def fit(self, data):
        step_results = []
        best_params = []

        # 使用网格搜索优化gamma
        (best_gamma, best_w0), step_results_for_block = self.optimize_params(data)
        best_params.append((best_gamma, best_w0))
        for result in step_results_for_block:
            step_results.append(result)
        
        return step_results, best_params
