"""
加入遗忘
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple
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
        beta_prior = np.exp(-params.beta) if params.beta > 0 else 0
        #gamma_prior = 1 if (0 <= params.gamma <= 1) else 0
        return k_prior * beta_prior
    
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

        # 添加记忆衰减
        memory_weights = gamma ** np.arange(len(data)-1, -1, -1)

        return np.where(r == 1, p_c, 1 - p_c) * memory_weights

    def fit_with_gamma(self, data, gamma: float) -> Tuple[ModelParams, float, float, Dict]:
        """在给定gamma时优化k和beta"""
        condition = data['condition'].iloc[0]
        max_k = self.get_max_k(condition)
        
        best_params = None
        best_log_likelihood = -np.inf
        best_posterior = -np.inf
        k_posteriors = {}
        
        for k in range(1, max_k + 1):
            result = minimize(
                lambda beta: self.posterior(ModelParams(k, beta[0], gamma), data, condition),
                x0=[self.config['param_inits']['beta']],
                bounds=[self.config['param_bounds']['beta']]
            )
            
            beta_opt, posterior_opt = result.x[0], -result.fun
            k_posteriors[k] = posterior_opt
            
            log_likelihood = np.sum(np.log(
                self.likelihood(ModelParams(k, beta_opt, gamma), data, condition)
            ))

            if posterior_opt > best_posterior:
                best_params = ModelParams(k=k, beta=beta_opt, gamma=gamma)
                best_log_likelihood, best_posterior = log_likelihood, posterior_opt
        
        # Normalize posteriors
        max_log_posterior = max(k_posteriors.values())
        k_posteriors = {k: np.exp(log_p - max_log_posterior) for k, log_p in k_posteriors.items()}
        total = sum(k_posteriors.values())
        k_posteriors = {k: p / total for k, p in k_posteriors.items()}
        
        return best_params, best_log_likelihood, best_posterior, k_posteriors

    def fit_trial_by_trial(self, data, gamma):
        step_results = []
        for step in range(1, len(data)+1):
            trial_data = data.iloc[:step]
            fitted_params, best_ll, best_post, k_post = self.fit_with_gamma(trial_data, gamma)
            
            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'best_log_likelihood': best_ll,
                'best_posterior': best_post,
                'k_posteriors': k_post,
                'params': fitted_params
            })
        
        return step_results

    def optimize_gamma(self, block_data, n_steps=50, gamma_step_factor = 0.05, min_step_size = 0.001):
        """优化gamma并返回最优gamma对应的step_results"""

        current_gamma = self.config['param_inits']['gamma']
        best_error = float('inf')
        best_gamma = current_gamma
        best_step_results = []

        prev_error = best_error  # 之前的误差，用于计算梯度

        for _ in range(n_steps):
            # 逐试次拟合k和beta
            step_results = self.fit_trial_by_trial(block_data, current_gamma)
            
            # 计算预测准确率
            predicted = []
            true_category = block_data['category'].values

            for i, trial in block_data.iterrows():
                fitted_params = step_results[i-block_data.index[0]]['params']
                x = trial[['feature1', 'feature2', 'feature3', 'feature4']].values
                condition = trial['condition']
                pred = self.predict_choice(fitted_params, x, condition)
                predicted.append(pred)
            
            predicted = np.array(predicted)
            pred_accuracy = np.mean(predicted == true_category)
            
            # 计算真实准确率（基于feedback）和误差
            true_accuracy = np.mean(block_data['feedback'] == 1)
            error = abs(pred_accuracy - true_accuracy)
        
            # 计算误差梯度（误差的变化）
            error_gradient = error - prev_error  # 误差变化率

            # 根据误差梯度调整gamma的更新幅度
            if error_gradient > 0:
                # 如果误差增大，减小gamma更新幅度
                gamma_step_factor = max(gamma_step_factor * 0.9, min_step_size)
            elif error_gradient < 0:
                # 如果误差减小，增大gamma更新幅度
                gamma_step_factor = min(gamma_step_factor * 1.1, 1.0)

            # 更新gamma
            if pred_accuracy < true_accuracy:
                current_gamma = min(current_gamma + gamma_step_factor, self.config['param_bounds']['gamma'][1])
            else:
                current_gamma = max(current_gamma - gamma_step_factor, self.config['param_bounds']['gamma'][0])

            # 保留最优gamma
            if error < best_error:
                best_error = error
                best_gamma = current_gamma
                best_step_results = step_results

            # 更新上一步的误差
            prev_error = error
            
        return best_gamma, best_step_results

    def fit_block_by_block(self, data, block_size=64):
        step_results = []
        best_gammas = []

        for start in range(0, len(data), block_size):
            end = start + block_size
            block_data = data.iloc[start:end]

            # 优化gamma
            best_gamma, step_results_for_block = self.optimize_gamma(block_data)
            best_gammas.append(best_gamma)
            step_results.extend(step_results_for_block)
        
        return step_results, best_gammas

