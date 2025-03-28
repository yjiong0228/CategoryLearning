"""
基线模型：理性贝叶斯
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class ModelParams:
    k: int  # index of partition method 
    beta: float  # softness of partition

class M_Base:
    def __init__(self, config: Dict):
        self.config = config
        self.all_centers = None
        
    def set_centers(self, centers: Dict):
        self.all_centers = centers

    def get_centers(self, k: int, condition: int) -> List[np.ndarray]:
        """ Get the specific category centers for a given k and condition."""
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        if 1 <= k <= len(centers):
            return list(centers[k - 1][1].values())
        else:
            raise ValueError(f"Invalid k for condition {condition}")
        
    def get_max_k(self, condition: int) -> int:
        """Return max k based on the condition."""
        centers = self.all_centers['2_cats'] if condition == 1 else self.all_centers['4_cats']
        return len(centers)

    def likelihood(self, params: ModelParams, data, condition: int) -> np.ndarray:
        """
        Compute the likelihood of the data given the model parameters.

        Args:
            params (ModelParams): Model parameters (k and beta).
            data (DataFrame): Data containing features, choices, and feedback.
                - 'feature1', 'feature2', 'feature3', 'feature4': Features (shape: [nTrials, 4])
                - 'choice': Category subjects chose (1-based, shape: [nTrials])
                - 'feedback': Feedback subjects got (1, 0.5, or 0, shape: [nTrials])
            condition (int): Experimental condition (1, 2 or 3).

        Returns:
            np.ndarray: Likelihood values for each data point (shape: [nTrials]).
        """
        k, beta = params.k, params.beta
        x = data[['feature1', 'feature2', 'feature3', 'feature4']].values  # Shape: [nTrials, 4]
        c = data['choice'].values  # Shape: [nTrials]
        r = data['feedback'].values  # Shape: [nTrials]

        centers = self.get_centers(k, condition)  # Tuple of category centers
        distances = np.linalg.norm(x[:, np.newaxis, :] - np.array(centers), axis=2)  # Shape: [nTrials, nCategoris]
        
        probs = np.exp(-beta * distances)  # Shape: [nTrials, nCategoris]
        probs /= np.sum(probs, axis=1, keepdims=True)  # Normalize probabilities
        p_c = probs[np.arange(len(c)), c - 1]  # Probability of chosen category (shape: [nTrials])

        return np.where(r == 1, p_c, 1 - p_c)

    def prior(self, params: ModelParams, condition: int) -> float:
        """Compute the prior probability of the model parameters."""
        max_k = self.get_max_k(condition)
        k_prior = 1/max_k if 1 <= params.k <= max_k else 0
        beta_prior = 1 if params.beta > 0 else 0
        return k_prior * beta_prior

    def posterior(self, params: ModelParams, data, condition: int) -> float:
        """Compute the negative log-posterior of the model."""
        prior_value = self.prior(params, condition)
        log_prior = np.log(prior_value) if prior_value > 0 else -np.inf
        log_likelihood = np.sum(np.log(self.likelihood(params, data, condition)))
        return -(log_prior + log_likelihood)

    def fit(self, data) -> Tuple[ModelParams, float, float, Dict]:
        """
        Fit the model to the data by maximizing the posterior.

        Args:
            data (DataFrame): Data containing features, choices, and feedback.

        Returns:
            Tuple[ModelParams, float, float, Dict]: Best parameters, log-likelihood, posterior, and posterior distribution over k.
        """
        condition = data['condition'].iloc[0]
        max_k = self.get_max_k(condition)
        
        k_results = {}  # 存储每个k的优化结果
        
        for k in range(1, max_k + 1):
            result = minimize(
                lambda beta: self.posterior(ModelParams(k, beta[0]), data, condition),
                x0=[self.config['param_inits']['beta']],
                bounds=[self.config['param_bounds']['beta']]
            )
            
            beta_opt, log_posterior = result.x[0], -result.fun

            log_likelihood = np.sum(np.log(
                self.likelihood(ModelParams(k, beta_opt), data, condition)
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

    def fit_trial_by_trial(self, data):
        """
        Fit the model trial-by-trial to observe parameter evolution.

        Args:
            data (DataFrame): Data containing features, choices, and feedback.

        Returns:
            List[Dict]: List of results for each trial step.
        """
        step_results = []
        for step in range(1, len(data)+1):
            trial_data = data.iloc[:step]
            fitted_params, best_ll, best_post, details = self.fit(trial_data)
            
            step_results.append({
                'k': fitted_params.k,
                'beta': fitted_params.beta,
                'best_log_likelihood': best_ll,
                'best_posterior': best_post,
                'details': details,
                'params': fitted_params
            })
        
        return step_results

    def predict_choice(self, params: ModelParams, x: np.ndarray, condition: int) -> int:
        """
        Predict the choice for a given input.

        Args:
            params (ModelParams): Model parameters (k and beta).
            x (np.ndarray): Input features (shape: [4]).
            condition (int): Experimental condition (1, 2 or 3).

        Returns:
            int: Predicted cluster index (1-based).
        """
        k, beta = params.k, params.beta
        centers = self.get_centers(k, condition)

        # Ensure x is 2D (shape: [1, 4]) for consistency
        if x.ndim == 1:
            x = x[np.newaxis, :]  # Reshape to [1, 4]
            
        distances = np.linalg.norm(x[:, np.newaxis, :] - np.array(centers), axis=2)
        probs = np.exp(-beta * distances)
        probs /= np.sum(probs, axis=1, keepdims=True) 
        return np.argmax(probs) + 1
