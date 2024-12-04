"""
加入决策噪音
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple, List
from .M_base import M_Base

@dataclass
class ModelParams:
    k: int
    beta: float
    phi: float

class M_Dec(M_Base):
    def __init__(self, config):
        self.config = config
        self.all_centers = None
    
    def likelihood(self, params: ModelParams, data, condition: int) -> np.ndarray:
        """Override likelihood calculation with modified formula"""
        k, beta, phi = params.k, params.beta, params.phi
        
        x = data[['feature1', 'feature2', 'feature3', 'feature4']].values
        c = data['choice'].values
        r = data['feedback'].values

        centers = self.get_centers(k, condition)
        distances = np.array([np.linalg.norm(x - np.array(center), axis=1) for center in centers])
        
        probs = np.exp(-beta * distances)
        probs /= np.sum(probs, axis=0, keepdims=True)
        softmax_probs = np.exp(phi * probs) / np.sum(np.exp(phi * probs))
        p_c = softmax_probs[c - 1, np.arange(len(c))]

        if condition == 3:
            family_probs = np.sum(softmax_probs.reshape(-1, 2, softmax_probs.shape[1]), axis=1)
            family_choice = (c - 1) // 2
            p_family = family_probs[family_choice, np.arange(len(c))]
            p_combined = p_family * (1 - p_c)
            return np.where(r == 1, p_c, np.where(r == 0.5, p_combined, 1 - p_c))
        else:
            return np.where(r == 1, p_c, 1 - p_c)
    
    def prior(self, params, condition):
        """Override prior to include phi parameter"""
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        max_k = len(centers)
        k_prior = 1/max_k if 1 <= params.k <= max_k else 0
        beta_prior = np.exp(-params.beta) if params.beta > 0 else 0
        phi_prior = np.exp(-params.phi) if params.phi > 0 else 0
        return k_prior * beta_prior * phi_prior
    
    def fit(self, data) -> Tuple[ModelParams, float, float, Dict]:
        condition = data['condition'].iloc[0]
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        max_k = len(centers)
        
        best_params = None
        best_log_likelihood = -np.inf
        best_posterior = -np.inf
        k_posteriors = {}
        
        for k in range(1, max_k + 1):
            result = minimize(
                lambda params: self.posterior(ModelParams(k, params[0], params[1]), data, condition),
                x0=[self.config['param_inits']['beta'],
                    self.config['param_inits']['phi']],
                bounds=[self.config['param_bounds']['beta'],
                    self.config['param_bounds']['phi']]
            )
            
            beta_opt, phi_opt = result.x
            posterior_opt = -result.fun
            k_posteriors[k] = posterior_opt

            current_log_likelihood = np.sum(np.log(
                self.likelihood(ModelParams(k, beta_opt, phi_opt), data, condition)
            ))

            if posterior_opt > best_posterior:
                best_posterior = posterior_opt
                best_params = ModelParams(k=k, beta=beta_opt, phi=phi_opt)
                best_log_likelihood = current_log_likelihood
        
        # Normalize posteriors
        max_log_posterior = max(k_posteriors.values())
        k_posteriors = {k: np.exp(log_p - max_log_posterior) for k, log_p in k_posteriors.items()}
        total = sum(k_posteriors.values())
        k_posteriors = {k: p / total for k, p in k_posteriors.items()}
        
        return best_params, best_log_likelihood, best_posterior, k_posteriors
    
    def fit_trial_by_trial(self, data):
        num_trials = len(data)
        step_results = []
        
        for step in range(1, num_trials):
            trial_data = data.iloc[:step]
            fitted_params, best_ll, best_post, k_post = self.fit(trial_data)
            
            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'phi': fitted_params.phi,
                'best_log_likelihood': best_ll,
                'best_posterior': best_post,
                'k_posteriors': k_post,
                'params': fitted_params
            })
        
        return step_results
    
    def predict_choice(self, params: ModelParams, x: np.ndarray, condition: int) -> int:
        k, beta, phi = params.k, params.beta, params.phi
        centers = self.get_centers(k, condition)
        distances = np.array([np.linalg.norm(x - np.array(center)) for center in centers])
        probs = np.exp(-beta * distances)
        probs /= np.sum(probs)
        softmax_probs = np.exp(phi * probs) / np.sum(np.exp(phi * probs))
        return np.argmax(probs) + 1