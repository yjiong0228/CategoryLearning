import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Callable
from scipy.optimize import minimize
from dataclasses import make_dataclass

from .utils.data import SubjectDataset, TrialNumpyData



class Module(object):
    def __init__(self):
        self.params = None

    def check_params(self, params: dataclass) -> None:
        """
        Check if the parameters are valid.
        """
        pass

    def prior_wrapper(self, *args, **kwargs) -> Callable:
        """
        Wrapper function for the prior method.

        Returns:
            Callable: Wrapper function for the prior method. func(params: dataclass)
        """
        pass

    def likelihood_wrapper(self, *args, **kwargs) -> Callable:
        """
        Wrapper function for the likelihood method.

        Returns:
            Callable: Wrapper function for the likelihood method. func(params: dataclass, data: TrialNumpyData)
        """
        pass

    def posterior_wrapper(self, *args, **kwargs) -> Callable:
        """
        Wrapper function for the posterior method.

        Returns:
            Callable: Wrapper function for the posterior method. func(params: dataclass, data: TrialNumpyData)
        """
        pass
    

class Base(Module):
    def __init__(self):
        """
        Base module for the Bayesian model. 
        This module has two parameters: k and beta.

        Args:
            Module (object): Base class for all modules.

        Returns:
            Base: Base module for the Bayesian model.
        """
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
        """
        Compute the prior probability of the model parameters.

        Args:
            params (ModelParams): Model parameters (k and beta).
            max_k (int): Maximum value of k.

        Returns:
            float: Prior probability of the model parameters.
        """
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
            data (TrialNumpyData): Data containing features, choices, and feedback.
            centers (np.ndarray): Centers of categories. Shape: (nCategories, nFeatures)

        Returns:
            np.ndarray: Likelihood values for each data point. Shape: (1,)
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
            data (List[TrialNumpyData]): Data containing features, choices, and feedback.
            centers (np.ndarray): Centers of categories. Shape: (nCategories, nFeatures)
        """
        loss = -np.log(self.prior(params, max_k=max_k))
        for trial_data in data:
            loss += -np.log(self.likelihood(params, trial_data, centers))
        return loss
    
    def prior_wrapper(self, max_k: int):
        def wrapper(params: dataclass):
            return self.prior(params, max_k=max_k)
        return wrapper
    
    def likelihood_wrapper(self, centers: np.ndarray):
        def wrapper(params: dataclass, data: TrialNumpyData):
            return self.likelihood(params, data, centers)
        return wrapper
    
    def predict(self, params: dataclass, trial_data: TrialNumpyData, centers: np.ndarray) -> np.ndarray:
        """
        Compute the posterior probabilities for the model parameters given the trial data and beta.
        
        Args:
            params (ModelParams): Model parameters (k and beta).
            trial_data (TrialNumpyData): Data containing features, choices, and feedback.
            centers (np.ndarray): Centers of categories. Shape: (nCategories, nFeatures)

        Returns:
            np.ndarray: Posterior probabilities for each category. Shape: (nCategories,)
        """
        x = trial_data.features
        distances = np.linalg.norm(x - centers, axis=1) # Shape: (nCategories,)
        logits = -params.beta * distances # Shape: (nCategories,)
        probs = np.exp(logits) / np.sum(np.exp(logits)) # Shape: (nCategories,)
        return probs
    
    def predict_wrapper(self, centers: np.ndarray):
        def wrapper(params: dataclass, trial_data: TrialNumpyData):
            return self.posterior(params, trial_data, centers)
        return wrapper

    def predicts(self, params: dataclass, data: SubjectDataset | List[TrialNumpyData], centers: np.ndarray, *, prod: bool = False) -> np.ndarray:       
        """
        Compute posterior probabilities across multiple trial data.

        Args:
            params (ModelParams): Model parameters (k and beta).
            data (SubjectDataset): Dataset containing trial data.
            centers (np.ndarray): Centers of categories. Shape: (nCategories, nFeatures)
            prod (bool): Whether to compute the product of the probabilities.

        Returns:
            np.ndarray: Posterior probabilities for each category. Shape: (nCategories,) or (nTrials, nCategories)
        """
        probs = [self.posterior(params, trial_data, centers) for trial_data in data]
        if prod:
            logits = np.prod(probs, axis=0)
            return logits / np.sum(logits) # Shape: (nCategories,)
        else:
            return probs # Shape: (nTrials, nCategories)

    def posterior(self, params: dataclass, data: TrialNumpyData, all_centers: List[Tuple[int, Dict[int, np.ndarray]]], weights: np.ndarray, max_k: int) -> np.ndarray:
        """
        Compute the posterior probabilities for the model parameters given the trial data and beta.

        Args:
            params (ModelParams): Model parameters (k and beta).
            data (TrialNumpyData): Data containing features, choices, and feedback.
            all_centers (List[Tuple[int, Dict[int, np.ndarray]]]): Centers of categories organized by condition.
            weights (np.ndarray): Weights for each k.

        Returns:
            np.ndarray: Posterior probabilities for each category. Shape: (nCategories,)
        """
        def get_centers(k: int) -> np.ndarray:
            """ Get the specific category centers for a given k."""
            if 0 <= k < len(all_centers):
                return np.array(list(all_centers[k][1].values()), dtype=np.float32)
            else:
                raise ValueError(f"Invalid k for condition {self.condition}")
        c = data.choice
        posteriors = []      
        for k in range(max_k):
            centers = get_centers(k)
            params.k = k   
            posteriors.append(self.predict(params, data, centers)[c] * weights[k])
        
        return np.sum(posteriors, axis=0) 

    def posterior_wrapper(self, all_centers: List[Tuple[int, Dict[int, np.ndarray]]], weights: np.ndarray, max_k: int):
        def wrapper(params: dataclass, data: TrialNumpyData):
            return self.posterior(params, data, all_centers, weights, max_k)
        return wrapper

    
    def fit(self,
            params: Dict[str, Union[int, float]],
            init_values: Dict[str, Union[int, float]],
            data: List[TrialNumpyData], 
            all_centers: List[Tuple[int, Dict[int, np.ndarray]]],
            max_k: int,
            beta_bounds: Tuple[float, float],
            ) -> Tuple[Dict[str, Union[int, float]], Callable, Callable]:
        """
        Fit the model to the data and return the best k and beta.

        Args:
            data (List[TrialNumpyData]): Data containing features, choices, and feedback.
            all_centers (List[Tuple[int, Dict[int, np.ndarray]]]): Centers of categories organized by condition.
            max_k (int): Maximum value of k.
            beta_init (float): Initial value for beta.
            beta_bounds (Tuple[float, float]): Bounds for beta.

        Returns:
            Tuple[Dict[str, Union[int, float]], Callable, Callable]: Best k and beta, prior wrapper, likelihood wrapper.
        """
        losses = []
        best_k = None
        best_beta = None     

        def get_centers(k: int) -> np.ndarray:
            """ Get the specific category centers for a given k."""
            if 0 <= k < len(all_centers):
                return np.array(list(all_centers[k][1].values()), dtype=np.float32)
            else:
                raise ValueError(f"Invalid k for condition {self.condition}")

        cls = make_dataclass('BaseParams', params)
        other_init_values = {key: value for key, value in init_values.items() if key != 'k' and key != 'beta'}
        for k in range(max_k):
            result = minimize(
                lambda beta: self.loss_fn(cls(k=k, beta=beta, **other_init_values), data, get_centers(k), max_k=max_k),
                x0=[init_values['beta']],
                bounds=[beta_bounds]
            )
            losses.append(result.fun)
            if best_k is None or result.fun < losses[best_k]:
                best_k = k
                best_beta = result.x[0]
        weights = np.exp(-np.array(losses) + losses[best_k]) # Shape: (max_k, )
        return {'best_params': {'k': best_k, 'beta': best_beta},
                'prior_fn': self.prior_wrapper(max_k),
                'likelihood_fn': self.likelihood_wrapper(get_centers(best_k)),
                'posterior_fn': self.posterior_wrapper(all_centers, weights, max_k)}
            


class Decision(Module):
    def __init__(self):
        super().__init__()
        self.params = dict(phi = float)

    def check_params(self, params: dataclass) -> None:
        for key in self.params.keys():
            if key not in params.__annotations__:
                raise ValueError(f"Missing parameter {key}")
            
        if not isinstance(params.phi, float):
            raise ValueError("Invalid parameter type")
        
    def posterior(self, params: dataclass, data: TrialNumpyData, other_posterior_fn: Callable, c_nums: int) -> np.ndarray:
        return (1 - params.phi) * other_posterior_fn(params, data) + params.phi / c_nums
    
    def posterior_wrapper(self, other_posterior_fn: Callable, c_nums: int):
        def wrapper(params: dataclass, data: TrialNumpyData):
            return self.posterior(params, data, other_posterior_fn, c_nums)
        return wrapper

    def loss_fn(self, params: dataclass, data: TrialNumpyData, other_posterior_fn: Callable, c_nums: int) -> np.ndarray:
        return -np.log(self.posterior(params, data, other_posterior_fn, c_nums))

        
    def fit(self,
            params: Dict[str, float],
            init_values: Dict[str, float],
            data: TrialNumpyData,
            other_posterior_fn: Callable,
            c_nums: int,
            phi_bounds: Tuple[float, float]
    ) -> Tuple[Dict[str, float], Callable, Callable]:
        best_phi = None       
        
        cls = make_dataclass('DecisionParams', params)
        other_init_values = {key: value for key, value in init_values.items() if key != 'phi'}

        result = minimize(
            lambda phi: self.loss_fn(cls(phi=phi, **other_init_values), data, other_posterior_fn, c_nums),
            x0=[init_values['phi']],
            bounds=[phi_bounds]
        )
        best_phi = result.x[0]
        return {'best_params': {'phi': best_phi},
                'prior_fn': None,
                'likelihood_fn': None,
                'posterior_fn': self.posterior_wrapper(other_posterior_fn, c_nums)}

        