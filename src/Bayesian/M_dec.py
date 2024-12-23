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
    k: int  # index of partition method 
    beta: float  # softness of partition
    phi: float  # randomness of decision

class M_Dec(M_Base):
    def __init__(self, config):
        self.config = config
        self.all_centers = None
    
    def prior(self, params, condition):
        """Override prior to include phi parameter"""
        max_k = self.get_max_k(condition)
        k_prior = 1/max_k if 1 <= params.k <= max_k else 0
        beta_prior = np.exp(-params.beta) if params.beta > 0 else 0
        phi_prior = np.exp(-params.phi) if params.phi > 0 else 0
        return k_prior * beta_prior * phi_prior

    def fit(self, data) -> Tuple[ModelParams, float, float, Dict]:
        condition = data['condition'].iloc[0]
        max_k = self.get_max_k(condition)
        
        best_params, best_log_likelihood, best_posterior = None, -np.inf, -np.inf
        k_posteriors = {}
        
        for k in range(1, max_k + 1):
            result = minimize(
                lambda beta: self.posterior(ModelParams(k, beta[0], phi=self.config['param_inits']['phi']), data, condition),
                x0=[self.config['param_inits']['beta']],
                bounds=[self.config['param_bounds']['beta']]
            )
            
            beta_opt, posterior_opt = result.x[0], -result.fun
            k_posteriors[k] = posterior_opt

            log_likelihood = np.sum(np.log(
                self.likelihood(ModelParams(k, beta_opt, phi=self.config['param_inits']['phi']), data, condition)
            ))

            if posterior_opt > best_posterior:
                best_params = ModelParams(k=k, beta=beta_opt, phi=self.config['param_inits']['phi'])
                best_log_likelihood, best_posterior = log_likelihood, posterior_opt
        
        # Normalize posteriors
        max_log_posterior = max(k_posteriors.values())
        k_posteriors = {k: np.exp(log_p - max_log_posterior) for k, log_p in k_posteriors.items()}
        total = sum(k_posteriors.values())
        k_posteriors = {k: p / total for k, p in k_posteriors.items()}
        
        return best_params, best_log_likelihood, best_posterior, k_posteriors

    def fit_trial_by_trial(self, data):
        """Override fit process to include phi parameter"""
        step_results = []

        for step in range(1, len(data)):
            trial_data = data.iloc[:step]
            fitted_params, best_ll, best_post, k_post = self.fit(trial_data)
            
            # Predict next trial's choice
            next_trial = data.iloc[step]
            x_next = next_trial[['feature1', 'feature2', 'feature3', 'feature4']].values
            c_actual = int(next_trial['choice'])

            condition = next_trial['condition']
            if condition == 1:
                n_categories = 2
            else:
                n_categories = 4
            # Define a function to compute the negative log-likelihood for phi
            def nll_phi(phi):
                # Compute choice probabilities based on current posterior
                choice_probs = np.zeros(n_categories)
                for k, posterior in k_post.items():
                    centers = self.get_centers(k, condition)
                    distances = np.linalg.norm(x_next - np.array(centers), axis=1)
                    logits = -fitted_params.beta * distances  # 计算logits
                    logits /= phi  # 调整温度
                    exp_logits = np.exp(logits - np.max(logits))  # 稳定计算
                    probs = exp_logits / np.sum(exp_logits)
                    choice_probs += posterior * probs
                
                # Negative log-likelihood for the actual choice
                return -np.log(choice_probs[c_actual - 1] + 1e-9)

            # Optimize phi
            result = minimize(
                nll_phi,
                x0=[self.config['param_inits']['phi']],
                bounds=[self.config['param_bounds']['phi']]
            )
            phi = result.x[0]  # Update phi with the optimized value

            params_with_phi = ModelParams(k=fitted_params.k, beta=fitted_params.beta, phi=phi)

            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'phi': phi,
                'best_log_likelihood': best_ll,
                'best_posterior': best_post,
                'k_posteriors': k_post,
                'params': params_with_phi
            })
        
        return step_results
    
    def predict_choice(self, params: ModelParams, x: np.ndarray, condition: int) -> int:
        k, beta, phi = params.k, params.beta, params.phi
        centers = self.get_centers(k, condition)
        
        # Ensure x is 2D (shape: [1, 4]) for consistency
        if x.ndim == 1:
            x = x[np.newaxis, :]  # Reshape to [1, 4]
            
        distances = np.linalg.norm(x[:, np.newaxis, :] - np.array(centers), axis=2)
        logits = -beta * distances / phi
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.argmax(probs) + 1