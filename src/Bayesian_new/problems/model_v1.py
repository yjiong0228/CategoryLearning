"""
Base Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
from .partitions import Partition, BasePartition


@dataclass(unsafe_hash=True)
class BaseModelParams:
    """
    A data class that holds the base model parameters.

    Attributes
    ----------
    k : int
        The index of the partition method.
    beta : float
        The softness of the partition.
    """
    k: int
    beta: float


@dataclass
class ObservationType:
    """
    A data class describing the format of an observation.

    Attributes
    ----------
    stimulus : tuple
        Stimulus data (e.g., visual inputs).
    choices : tuple
        Choices made in response to the stimulus.
    responses : tuple
        Response correctness or other feedback metrics.
    categories : tuple
        The true category of the stimulus.
    """
    stimulus: tuple
    choices: tuple
    responses: tuple
    categories: tuple


class PartitionLikelihood(BaseLikelihood):
    """
    Likelihood for partitions only.
    """

    def __init__(self, space: BaseSet, partition: BasePartition):
        """
        Initialize PartitionLikelihood.

        Parameters
        ----------
        space : BaseSet
            The set of hypotheses indices (k's).
        partition : BasePartition
            The partitioning object that calculates likelihoods.
        """
        super().__init__(space)
        self.partition = partition
        # This may raise an exception if h_set is not a subset of partition labels.
        self.h_indices = list(self.h_set)

    def get_likelihood(self,
                       observation,
                       beta: list | tuple | float = 1.,
                       use_cached_dist: bool = False,
                       normalized: bool = True,
                       **kwargs) -> np.ndarray:
        """
        Compute the likelihood of an observation given the current partition.

        Parameters
        ----------
        observation : any
            Observation data.
        beta : float or list/tuple
            Softness parameter.
        use_cached_dist : bool
            Whether to use cached distances for speed-up.
        normalized : bool
            Whether to normalize the result.

        Returns
        -------
        np.ndarray
            An array of likelihood values.
        """
        likelihood_values = self.partition.calc_likelihood(
            self.h_indices,
            observation,
            beta,
            use_cached_dist,
            normalized,
            **kwargs
        )
        return likelihood_values


class SoftPartitionLikelihood(PartitionLikelihood):
    """
    Likelihood using (partition, beta) as hypotheses.
    """

    def __init__(self, space: BaseSet, partition: BasePartition, beta_grid: list):
        """
        Initialize SoftPartitionLikelihood.

        Parameters
        ----------
        space : BaseSet
            The set of hypotheses indices (k's).
        partition : BasePartition
            The partitioning object that calculates likelihoods.
        beta_grid : list
            A list of beta values to be considered.
        """
        super().__init__(space, partition)
        self.beta_grid = beta_grid

    def get_likelihood(self,
                       observation,
                       beta: list | tuple | float = 1.,
                       use_cached_dist: bool = False,
                       normalized: bool = True,
                       **kwargs) -> np.ndarray:
        """
        Compute the likelihood of an observation over a grid of beta values.

        Parameters
        ----------
        observation : any
            Observation data.
        beta : float or list/tuple
            Softness parameter (not used directly, since we use beta_grid).
        use_cached_dist : bool
            Whether to use cached distances for speed-up.
        normalized : bool
            Whether to normalize the result.

        Returns
        -------
        np.ndarray
            A concatenated array of likelihood values for each beta in beta_grid.
        """
        likelihood_collection = []
        for beta_value in self.beta_grid:
            likelihood_val = self.partition.calc_likelihood(
                self.h_indices,
                observation,
                beta_value,
                use_cached_dist,
                normalized,
                **kwargs
            )
            likelihood_collection.append(likelihood_val)
        return np.concatenate(likelihood_collection, axis=1)


class BaseModel:
    """
    Base Model class that initializes a default partition model,
    hypotheses set, and inference engine.
    """

    def __init__(self, config: Dict, **kwargs):
        """
        Initialize BaseModel with a given configuration.

        Parameters
        ----------
        config : Dict
            A dictionary containing initial settings (e.g., parameter bounds).
        **kwargs : dict
            Additional keyword arguments.
        """
        self.config = config
        self.all_centers = None
        self.hypotheses_set = BaseSet([])
        self.observation_set = BaseSet([])

        condition = kwargs.get("condition", 1)
        n_dims = 4
        n_cats = 2 if condition == 1 else 4

        self.partition_model = kwargs.get("partition", Partition(n_dims, n_cats))
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        self.engine = BaseEngine(
            self.hypotheses_set, self.observation_set,
            BasePrior(self.hypotheses_set),
            PartitionLikelihood(self.hypotheses_set, self.partition_model))

    def set_hypotheses(self, hypothesis_collection: Dict | Tuple | List):
        """
        Set the hypotheses set manually.

        Parameters
        ----------
        hypothesis_collection : Dict or Tuple or List
            A collection of hypotheses indices.
        """
        self.hypotheses_set = BaseSet(hypothesis_collection)

    def refresh_engine(self, new_hypotheses_set, new_prior, new_likelihood):
        """
        Re-initialize the engine with a new set of hypotheses, prior, and likelihood.

        Parameters
        ----------
        new_hypotheses_set : BaseSet
            New set of hypotheses.
        new_prior : BasePrior
            New prior object.
        new_likelihood : BaseLikelihood
            New likelihood object.
        """
        self.hypotheses_set = new_hypotheses_set
        self.engine = BaseEngine(
            new_hypotheses_set,
            self.observation_set,
            new_prior,
            new_likelihood
        )

    def fit(self, data, **kwargs) -> Tuple[BaseModelParams, float, Dict, Dict]:
        """
        Fit the model to data. NotImplementedError by default.

        Parameters
        ----------
        data : any
            Training data.

        Returns
        -------
        (BaseModelParams, float, Dict, Dict)
            Stub return to be overridden in subclasses.
        """
        raise NotImplementedError


class SingleRationalModel(BaseModel):
    """
    A model that fits each hypothesis with a rational approach
    and returns the best-fitting one.
    """

    def precompute_distances(self, stimulus: np.ndarray):
        """
        Precompute all distance.

        Parameters
        ----------
        stimulus : np.ndarray
        """
        if hasattr(self.partition_model, "precompute_all_distances"):
            self.partition_model.precompute_all_distances(stimulus)

    def fit_single_step(self, 
                        data: Tuple[np.ndarray, np.ndarray,np.ndarray], 
                        **kwargs) -> Tuple[BaseModelParams, float, Dict, Dict]:
        """
        Fit the rational model by optimizing beta for each hypothesis.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing (stimulus, choices, responses).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        best_params : BaseModelParams
            Parameters (hypothesis, beta) that achieve the highest likelihood.
        best_ll : float
            The maximum log-likelihood value.
        all_hypo_params : Dict
            Mapping from hypothesis to BaseModelParams.
        all_hypo_ll : Dict
            Mapping from hypothesis to its best log-likelihood.
        """
        all_hypo_params = {}
        all_hypo_ll = {}

        def _ll_per_hypo(beta, hypo=None):
            likelihood = self.partition_model.calc_likelihood_entry(
                hypo, data, beta[0], **kwargs)
            return np.sum(np.log(np.maximum(likelihood, 0)), axis=0)

        for hypo in self.hypotheses_set:
            result = minimize(lambda beta: -_ll_per_hypo(beta, hypo),
                              x0=[self.config["param_inits"]["beta"]],
                              bounds=[self.config["param_bounds"]["beta"]])
            beta_opt, ll_max = result.x[0], -result.fun

            all_hypo_params[hypo] = BaseModelParams(hypo, beta_opt)
            all_hypo_ll[hypo] = ll_max

        best_hypo_idx = max(all_hypo_ll, key=all_hypo_ll.get)

        return (all_hypo_params[best_hypo_idx], 
                all_hypo_ll[best_hypo_idx],
                all_hypo_params, 
                all_hypo_ll)

    def fit_step_by_step(self, 
                         data: Tuple[np.ndarray, np.ndarray,np.ndarray],
                         limited_hypos_list: List[List[int]] = None, 
                         **kwargs) -> List[Dict]:
        """
        Fit the model step-by-step to observe how parameters evolve.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing (stimulus, choices, responses).
        limited_hypos_list : List[List[int]], optional
            If not None, it is a list of length n_trials. Each element is a
            subset of hypotheses to be considered at each trial. If None, the
            original full set (self.hypotheses_set) is used.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the fitting results for each
            step, such as the best hypothesis, beta, log-likelihood, and posterior.
        """
        
        stimulus, choices, responses = data
        n_trials = len(responses)

        # Precompute all distances
        self.partition_model.precompute_all_distances(stimulus)

        # Backup the original hypotheses set
        original_hypotheses_set = self.hypotheses_set
        original_prior = self.engine.prior
        original_likelihood = self.engine.likelihood

        step_results = []

        for step_idx in tqdm(range(1, n_trials+1)):
            selected_data = [x[:step_idx] for x in data]

            if limited_hypos_list is not None:
                # Use a limited subset of hypotheses for this trial
                current_hypos = limited_hypos_list[step_idx - 1]
                new_hypotheses_set = BaseSet(current_hypos)

                new_prior = BasePrior(new_hypotheses_set)
                new_likelihood = PartitionLikelihood(new_hypotheses_set, self.partition_model)

                # Refresh the engine
                self.refresh_engine(new_hypotheses_set, new_prior, new_likelihood)
            else:
                # If limited_hypos_list is None, keep using self.hypotheses_set as is
                pass

            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit_single_step(
                selected_data, 
                use_cached_dist=True, 
                **kwargs)

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            all_hypo_post = self.engine.infer_log(
                selected_data,
                use_cached_dist=True,
                beta=hypo_betas,
                normalized=True)

            hypo_details = {}
            for i, hypo in enumerate(self.hypotheses_set.elements):
                hypo_details[hypo] = {
                    'beta_opt': all_hypo_params[hypo].beta,
                    'll_max': all_hypo_ll[hypo],
                    'post_max': all_hypo_post[i],
                    'is_best': hypo == best_params.k
                }

            step_results.append({
                'best_k': best_params.k,
                'best_beta': best_params.beta,
                'best_params': best_params,
                'best_log_likelihood': best_ll,
                'best_norm_posterior': np.max(all_hypo_post),
                'hypo_details': hypo_details
            })

        # Restore the original hypotheses set if limited_hypos_list was used
        if limited_hypos_list is not None:
            self.refresh_engine(
                original_hypotheses_set,
                original_prior,
                original_likelihood
            )

        return step_results

    def oral_generate_hypos(self,
                            data: Tuple[np.ndarray, np.ndarray],
                            top_k: int = 10,
                            dist_tol: float = 1e-9) -> List[List[int]]:
        """
        Generate a limited hypothesis set for each trial based on the
        participant's verbally reported center and the chosen category.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray]
            data[0] : (n_trials, n_dims) containing the verbally reported category center per trial.
            data[1] : (n_trials,) containing the chosen category index (1-indexed).
        top_k : int
            The number of closest hypotheses to choose if no exact match is found.
        dist_tol : float
            Tolerance within which distances are considered zero (floating-point errors).

        Returns
        -------
        List[List[int]]
            A list of length n_trials, where each element is a list of hypothesis indices.
        """

        oral_centers, choices = data
        n_trials = len(choices)

        n_hypos = self.partition_model.prototypes_np.shape[0]
        all_hypos = range(n_hypos)

        limited_hypos_list = []

        for trial_idx in range(n_trials):
            cat_idx = choices[trial_idx] - 1
            reported_center = oral_centers[trial_idx]

            distance_map = []
            for hypo_idx in all_hypos:
                # Compare with the (cat_idx)-th category prototype of hypothesis h
                true_center = self.partition_model.prototypes_np[hypo_idx, 0, cat_idx, :]
                distance_val = np.linalg.norm(reported_center - true_center)
                distance_map.append((distance_val, hypo_idx))

            # Check for exact matches within tolerance
            exact_matches = [hypo_idx for (dist, hypo_idx) in distance_map if dist <= dist_tol]

            if len(exact_matches) > 0:
                chosen_hypos = exact_matches
            else:
                distance_map.sort(key=lambda x: x[0])
                chosen_hypos = [hypo_idx for (_, hypo_idx) in distance_map[:top_k]]

            limited_hypos_list.append(chosen_hypos)

        return limited_hypos_list

    def predict_choice(self,
                       data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                       step_results: list,
                       use_cached_dist, 
                       window_size) -> Dict[str, np.ndarray]:
        """
        Predict choice trial by trial using fitted parameters and hypotheses.

        Parameters
        ----------
        data : tuple
            A tuple containing (stimulus, choices, responses, categories).
        step_results : list
            Output of fit_trial_by_trial, containing fitted results for each trial.
        use_cached_dist : bool
            Whether to use cached distances to speed up calculations.
        window_size : int
            Size of the sliding window for computing average accuracy.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing arrays of true accuracy, predicted accuracy, 
            and their sliding averages.
        """
        stimulus, choices, responses, categories = data
        n_trials = len(responses)

        true_acc = (np.array(responses) == 1).astype(float)
        pred_acc = np.full(n_trials, np.nan, dtype=float)
        
        for trial_idx in range(1, n_trials):
            trial_data = ([stimulus[trial_idx]], [choices[trial_idx]], 
                          [responses[trial_idx]], [categories[trial_idx]])

            # Extract the posterior probabilities for each hypothesis at last trial
            hypo_details = step_results[trial_idx-1]['hypo_details']
            post_max = [hypo_details[k]['post_max']
                for k in hypo_details.keys()]

            # Compute the weighted probability of being correct
            weighted_p_true = 0
            for k, post in zip(hypo_details.keys(), post_max):
                p_true = self.partition_model.calc_trueprob_entry(
                    k, trial_data, hypo_details[k]['beta_opt'], 
                    use_cached_dist=use_cached_dist, indices=[trial_idx])
                weighted_p_true += post * p_true

            pred_acc[trial_idx] = weighted_p_true

        # Compute sliding averages using a sliding window
        sliding_true_acc = []
        sliding_pred_acc = []
        sliding_pred_acc_std = []
        
        for start_idx in range(1, n_trials - window_size + 2):  # Start from index 1
            end_idx = start_idx + window_size
            sliding_true_acc.append(np.mean(true_acc[start_idx:end_idx]))
            pred_window = pred_acc[start_idx:end_idx]
            sliding_pred_acc.append(np.mean(pred_window))
            sliding_pred_acc_std.append(np.sqrt(np.sum(pred_window*(1 - pred_window)))/window_size)

        predict_results = {
            'true_acc': true_acc,
            'pred_acc': pred_acc,
            'sliding_true_acc': sliding_true_acc,
            'sliding_pred_acc': sliding_pred_acc,
            'sliding_pred_acc_std': sliding_pred_acc_std
        }

        return predict_results