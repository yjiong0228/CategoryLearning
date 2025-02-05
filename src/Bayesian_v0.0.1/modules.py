import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union

from .utils.data import SubjectDataset, TrialNumpyData



class Module(object):
    def __init__(self):
        self.params = None
    

class Base(Module):
    def __init__(self):
        super().__init__()
        self.params = dict(k = int, beta = float)

    def check_params(self, params: dataclass) -> None:
        for key in self.params.keys():
            if key not in params.__annotations__:
                raise ValueError(f"Missing parameter {key}")
            
        if not isinstance(params.k, int) or not isinstance(params.beta, float):
            print(params.k, params.beta)
            raise ValueError("Invalid parameter type")

    def prior(self, params: dataclass, *,max_k: int = -1) -> float:
        if max_k == -1:
            raise ValueError("max_k must be specified")
        
        k_prior = 1/max_k if 0 <= params.k < max_k else 0
        beta_prior = np.exp(-np.maximum(0, params.beta))
        
        return k_prior * beta_prior
    
    def likelihood(self, params: dataclass, data: TrialNumpyData, centers: np.ndarray) -> np.ndarray:
        """
        Compute the likelihood of the data given the model parameters.

        Args:
            params (ModelParams): Model parameters (k and beta).
            data (TrialData): Data containing features, choices, and feedback.
            centers (torch.Tensor): Centers of categories. Shape: (nCategories, nFeatures)

        Returns:
            torch.Tensor: Likelihood values for each data point. Shape: (1,)
        """
        beta = params.beta
        x = data.features  # Shape: (4,)
        c = data.choice  # Shape: (1,)
        r = data.feedback  # Shape: (1,)

        distances = np.linalg.norm(x - centers, axis=1)  # Shape: (nCategories,)
        logits = -beta * distances  # Shape: (nCategories,)
        probs = np.exp(logits) / np.sum(np.exp(logits))  # Shape: (nCategories,)
        p_c = probs[c]  # Probability of chosen category (shape: (1,))
        p_r = r * p_c + (1 - r) * (1 - p_c)
        return p_r

    def loss_fn(self, params: dataclass, data: List[TrialNumpyData], centers: np.ndarray, *,max_k: int = -1) -> np.ndarray:
        """
        Compute the negative log likelihood of the data given the model parameters.

        Args:
            params (ModelParams): Model parameters (k and beta).
            data (List[TrialData]): Data containing features, choices, and feedback.
            centers (torch.Tensor): Centers of categories. Shape: (nCategories, nFeatures)
        """
        loss = -np.log(self.prior(params, max_k=max_k))
        for trial_data in data:
            loss += -np.log(self.likelihood(params, trial_data, centers))
        return loss

    def posterior(self, params: dataclass, data: SubjectDataset, centers: np.ndarray, *,prod: bool = False) -> np.ndarray:    
        def calc_posterior(beta, trial_data, centers):        
            x = trial_data.features  # Shape: (4,)

            distances = np.linalg.norm(x - centers, axis=1)  # Shape: (nCategories,)
            logits = -beta * distances  # Shape: (nCategories,)
            probs = np.exp(logits) / np.sum(np.exp(logits))  # Shape: (nCategories,)
            return probs
        
        probs = [calc_posterior(params.beta, trial_data, centers) for trial_data in data]
        if prod:
            logits = np.prod(probs, axis=0)
            return logits / np.sum(logits) # Shape: (nCategories,)
        else:
            return probs

    