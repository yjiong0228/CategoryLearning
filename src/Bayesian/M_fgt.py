"""
加入遗忘
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class ModelParams:
    k: int
    beta: float
    gamma: float

class M_Fgt:
    def __init__(self, config):
        self.config = config
        self.all_centers = None
        
    def set_centers(self, centers):
        """Set the centers dictionary for the model"""
        self.all_centers = centers
        
    def get_centers(self, k: int, condition: int) -> tuple:
        """Get centers based on k and condition"""
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        if 1 <= k <= len(centers):
            return tuple(list(centers[k-1][1][i]) for i in centers[k-1][1])
        else:
            raise ValueError(f"Invalid k for condition {condition}")

    def likelihood(self, params: ModelParams, data, condition: int) -> np.ndarray:
        k, beta, gamma = params.k, params.beta, params.gamma
        x = data[['feature1', 'feature2', 'feature3', 'feature4']].values
        c = data['choice'].values
        r = data['feedback'].values

        centers = self.get_centers(k, condition)
        distances = np.array([np.linalg.norm(x - np.array(center), axis=1) for center in centers])
        
        probs = np.exp(-beta * distances)
        probs /= np.sum(probs, axis=0, keepdims=True)
        p_c = probs[c - 1, np.arange(len(c))]
        
        # 添加记忆衰减
        memory_weights = gamma ** np.arange(len(data)-1, -1, -1)

        if condition == 3:
            family_probs = np.sum(probs.reshape(-1, 2, probs.shape[1]), axis=1)
            family_choice = (c - 1) // 2
            p_family = family_probs[family_choice, np.arange(len(c))]
            p_combined = p_family * (1 - p_c)
            return np.where(r == 1, p_c, np.where(r == 0.5, p_combined, 1 - p_c)) * memory_weights
        else:
            return np.where(r == 1, p_c, 1 - p_c) * memory_weights

    def prior(self, params: ModelParams, condition: int) -> float:
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        max_k = len(centers)
        k_prior = 1/max_k if 1 <= params.k <= max_k else 0
        beta_prior = np.exp(-params.beta) if params.beta > 0 else 0
        gamma_prior = 1 if 0 <= params.gamma <= 1 else 0
        return k_prior * beta_prior * gamma_prior

    def posterior(self, params: ModelParams, data, condition: int) -> float:
        prior_value = self.prior(params, condition)
        log_prior = np.log(prior_value) if prior_value > 0 else -np.inf
        log_likelihood = np.sum(np.log(self.likelihood(params, data, condition)))
        return -(log_prior + log_likelihood)

    def fit_with_memory_decay(self, data) -> Tuple[float, Dict]:
        """在最外层优化 memory_decay 参数"""
        condition = data['condition'].iloc[0]
        
        def objective(gamma):
            # 对固定的 memory_decay，找到最优的 k 和 beta
            best_posterior = -np.inf
            centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
            max_k = len(centers)
            
            for k in range(1, max_k + 1):
                result = minimize(
                    lambda beta: self.posterior(
                        ModelParams(k, beta[0], gamma), 
                        data, 
                        condition
                    ),
                    x0=[self.config['param_inits']['beta']],
                    bounds=[self.config['param_bounds']['beta']]
                )
                
                posterior_opt = -result.fun
                if posterior_opt > best_posterior:
                    best_posterior = posterior_opt
            
            return -best_posterior  # 最小化负后验概率
        
        # 优化 memory_decay
        result = minimize(
            objective,
            x0=[self.config['param_inits']['gamma']],
            bounds=[self.config['param_bounds']['gamma']],
            method='L-BFGS-B'
        )
        
        gamma_opt = result.x[0]
        return gamma_opt

    def fit_trial_by_trial(self, data):
        # 首先估计整体的 memory_decay
        gamma = self.fit_with_memory_decay(data)
        print(f"Estimated memory_decay: {gamma}")
        
        num_trials = len(data)
        step_results = []
        
        # 使用固定的 memory_decay 进行分步拟合
        for step in range(1, num_trials):
            trial_data = data.iloc[:step]
            
            # 使用已经估计的 memory_decay 进行拟合
            condition = trial_data['condition'].iloc[0]
            centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
            max_k = len(centers)
            
            best_params = None
            best_log_likelihood = -np.inf
            best_posterior = -np.inf
            k_posteriors = {}
            
            for k in range(1, max_k + 1):
                result = minimize(
                    lambda beta: self.posterior(
                        ModelParams(k, beta[0], gamma), 
                        trial_data, 
                        condition
                    ),
                    x0=[self.config['param_inits']['beta']],
                    bounds=[self.config['param_bounds']['beta']]
                )
                
                beta_opt = result.x[0]
                posterior_opt = -result.fun
                k_posteriors[k] = posterior_opt

                current_log_likelihood = np.sum(np.log(
                    self.likelihood(ModelParams(k, beta_opt, gamma), trial_data, condition)
                ))

                if posterior_opt > best_posterior:
                    best_posterior = posterior_opt
                    best_params = ModelParams(k=k, beta=beta_opt, gamma=gamma)
                    best_log_likelihood = current_log_likelihood
            
            # Normalize posteriors
            max_log_posterior = max(k_posteriors.values())
            k_posteriors = {k: np.exp(log_p - max_log_posterior) for k, log_p in k_posteriors.items()}
            total = sum(k_posteriors.values())
            k_posteriors = {k: p / total for k, p in k_posteriors.items()}
            
            step_results.append({
                'k': best_params.k,
                'beta': best_params.beta,
                'gamma': gamma,  # 记录固定的 memory_decay
                'best_log_likelihood': best_log_likelihood,
                'best_posterior': best_posterior,
                'k_posteriors': k_posteriors,
                'params': best_params
            })
        
        return step_results

    def predict_choice(self, params: ModelParams, x: np.ndarray, condition: int) -> int:
        k, beta = params.k, params.beta
        centers = self.get_centers(params.k, condition)
        distances = np.array([np.linalg.norm(x - np.array(center)) for center in centers])
        probs = np.exp(-params.beta * distances)
        probs /= np.sum(probs)
        return np.argmax(probs) + 1
