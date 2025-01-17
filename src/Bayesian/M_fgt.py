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

        return np.where(r == 1, p_c, 1 - p_c) * memory_weights

    def prior(self, params: ModelParams, condition: int) -> float:
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        max_k = len(centers)
        k_prior = 1/max_k if 1 <= params.k <= max_k else 0
        beta_prior = np.exp(-params.beta) if params.beta > 0 else 0
        gamma_prior = 1 if 0 <= params.gamma <= 1 else 0
        return k_prior * beta_prior * gamma_prior

    def fit(self, data) -> Tuple[ModelParams, float, float, Dict]:
        condition = data['condition'].iloc[0]
        max_k = self.get_max_k(condition)
        
        best_params, best_log_likelihood, best_posterior = None, -np.inf, -np.inf
        k_posteriors = {}
        
        for k in range(1, max_k + 1):
            result = minimize(
                lambda beta: self.posterior(ModelParams(k, beta[0], gamma=self.config['param_inits']['gamma']), data, condition),
                x0=[self.config['param_inits']['beta']],
                bounds=[self.config['param_bounds']['beta']]
            )
            
            beta_opt, posterior_opt = result.x[0], -result.fun
            k_posteriors[k] = posterior_opt

            log_likelihood = np.sum(np.log(
                self.likelihood(ModelParams(k, beta_opt, gamma=self.config['param_inits']['gamma']), data, condition)
            ))

            if posterior_opt > best_posterior:
                best_params = ModelParams(k=k, beta=beta_opt, gamma=self.config['param_inits']['gamma'])
                best_log_likelihood, best_posterior = log_likelihood, posterior_opt
        
        # Normalize posteriors
        max_log_posterior = max(k_posteriors.values())
        k_posteriors = {k: np.exp(log_p - max_log_posterior) for k, log_p in k_posteriors.items()}
        total = sum(k_posteriors.values())
        k_posteriors = {k: p / total for k, p in k_posteriors.items()}
        
        return best_params, best_log_likelihood, best_posterior, k_posteriors

    def fit_trial_by_trial(self, data):
        step_results = []
        for step in range(1, len(data)):
            trial_data = data.iloc[:step]
            fitted_params, best_ll, best_post, k_post = self.fit(trial_data)
            
            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'gamma': fitted_params.gamma,
                'best_log_likelihood': best_ll,
                'best_posterior': best_post,
                'k_posteriors': k_post,
                'params': fitted_params
            })
        
        return step_results

    def predict_choice(self, params: ModelParams, x: np.ndarray, condition: int) -> int:
        k, beta = params.k, params.beta
        centers = self.get_centers(k, condition)
        distances = np.array([np.linalg.norm(x - np.array(center)) for center in centers])
        probs = np.exp(-beta * distances)
        probs /= np.sum(probs)
        return np.argmax(probs) + 1
