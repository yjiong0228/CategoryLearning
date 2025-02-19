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

    def fit_trial_by_trial(self, data, base_step_results):
        """Override fit process to include phi parameter"""

        fitted_params = [item['params'] for item in base_step_results]
        k_post = [item['k_posteriors'] for item in base_step_results]
        
        step_results = []

        condition = data['condition'].iloc[0]
        if condition == 1:
            n_categories = 2
        else:
            n_categories = 4

        for step in range(1, len(data)+1):
            fitted_k = fitted_params[step-1].k
            fitted_beta = fitted_params[step-1].beta
            step_k_post = k_post[step-1]

            trial_data = data.iloc[:step]
            x = trial_data[['feature1', 'feature2', 'feature3', 'feature4']].values
            c = trial_data['choice'].values

            # Define a function to compute the negative log-likelihood for phi
            def nll_phi(phi):
                dec_prior = np.tile(np.full(n_categories, 1/n_categories), (step, 1))
                dec_lik = np.zeros_like(dec_prior)

                # Compute the likelihood of each choice for each k and posterior
                for k, posterior in step_k_post.items():
                    centers = self.get_centers(k, condition)
                    distances = np.linalg.norm(x[:, np.newaxis, :] - np.array(centers), axis=2)
                    logits = -fitted_beta * distances # logits based on k and beta
                    exp_logits = np.exp(logits)
                    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    dec_lik += posterior * probs # Summing weighted by k_post (posterior for each k)
                
                # Apply phi to mix decision likelihood with decision prior
                dec_prob = (1 - phi) * dec_lik + phi * dec_prior  # phi is used to weight between data-driven and prior
                
                # Negative log-likelihood for the actual choice
                nll = -np.log(dec_prob[np.arange(len(c)), c - 1] + 1e-9)
            
                return np.mean(nll)
        
            # Optimize phi
            result = minimize(
                nll_phi,
                x0=[self.config['param_inits']['phi']],
                bounds=[self.config['param_bounds']['phi']]
            )
            phi = result.x[0]  # Update phi with the optimized value

            params_with_phi = ModelParams(k=fitted_k, beta=fitted_beta, phi=phi)

            step_results.append({
                'k': fitted_k,
                'beta': fitted_beta,
                'phi': phi,
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
        logits = -beta * distances
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        
        if condition == 1:
            n_categories = 2
        else:
            n_categories = 4
        dec_prior = np.full(n_categories, 1/n_categories)

        probs = (1 - phi) * probs + phi * dec_prior
        return np.argmax(probs) + 1