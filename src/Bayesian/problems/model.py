"""
Base Model
"""
from abc import ABC
from dataclasses import dataclass, make_dataclass, asdict
from typing import Dict, Tuple, List, Callable, Optional
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize
from collections import Counter
from itertools import product
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
from .partitions import Partition, BasePartition

# [TODO] MOVE to CONFIG
HYPO_CLUSTER_PROTOTYPE_AMOUNT = 1


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
            self.h_indices, observation, beta, use_cached_dist, normalized,
            **kwargs)
        return likelihood_values


class SoftPartitionLikelihood(PartitionLikelihood):
    """
    Likelihood using (partition, beta) as hypotheses.
    """

    def __init__(self, space: BaseSet, partition: BasePartition,
                 beta_grid: list):
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
                self.h_indices, observation, beta_value, use_cached_dist,
                normalized, **kwargs)
            likelihood_collection.append(likelihood_val)
        return np.concatenate(likelihood_collection, axis=1)


class BaseModel:
    """
    Base Model class that initializes a default partition model,
    hypotheses set, and inference engine.
    """

    def __init__(self, config: Dict, module_config: Dict = {}, **kwargs):
        """
        Initialize BaseModel with a given configuration.

        Parameters
        ----------
        config : Dict
            A dictionary containing initial settings (e.g., parameter bounds).
        module_config: Dict
            Modules, in terms of (module_name: (module_class, module_kwargs)).
        **kwargs : dict
            Additional keyword arguments.
        """

        self.module_config = module_config
        self.config = config
        self.all_centers = None
        self.hypotheses_set = BaseSet([])
        self.observation_set = BaseSet([])

        self.condition = kwargs.get("condition", 1)
        n_dims = 4
        self.n_cats = 2 if self.condition == 1 else 4

        self.partition_model = kwargs.get("partition",
                                          Partition(n_dims, self.n_cats))
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        self.full_likelihood = PartitionLikelihood(
            BaseSet(list(range(self.partition_model.length))),
            self.partition_model)
        self.engine = BaseEngine(
            self.hypotheses_set, self.observation_set,
            BasePrior(self.hypotheses_set),
            PartitionLikelihood(self.hypotheses_set, self.partition_model))

        if "initial_states" in kwargs:
            self.initial_states = kwargs["initial_states"]
        self.initialize_modules()

    def initialize_modules(self):
        """
        Initialize Modules
        """
        self.modules = {}

        for key, (mod_cls, mod_kwargs) in self.module_config.items():
            self.modules[key] = mod_cls(self, **mod_kwargs)

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
        self.engine = BaseEngine(new_hypotheses_set, self.observation_set,
                                 new_prior, new_likelihood)

    def fit_single_step(self, data,
                        **kwargs) -> Tuple[BaseModelParams, float, Dict, Dict]:
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


class StandardModel(BaseModel):
    """
    Standard Model
    """

    def __init__(self, config: Dict, module_config: Dict = {}, **kwargs):
        """
        """
        super().__init__(config, module_config, **kwargs)

    def precompute_distances(self, stimulus: np.ndarray):
        """
        Precompute all distance.

        Parameters
        ----------
        stimulus : np.ndarray
        """
        if hasattr(self.partition_model, "precompute_all_distances"):
            self.partition_model.precompute_all_distances(stimulus)

    def initialize_modules(self):
        super().initialize_modules()

        # Initialize model parameters
        self.params_dict = {'k': int, 'beta': float}
        for key, mod in self.modules.items():
            if hasattr(mod, 'params_dict'):
                self.params_dict.update(mod.params_dict)

        self.optimize_params_dict = {}
        for key, mod in self.modules.items():
            if hasattr(mod, 'optimize_params_dict'):
                self.optimize_params_dict.update(mod.optimize_params_dict)

    def fit_single_step(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray],
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

            ModelParams = make_dataclass('ModelParams',
                                         self.params_dict.keys())
            params_values = {}
            for key in self.params_dict.keys():
                if key == "k":
                    params_values[key] = hypo
                elif key == "beta":
                    params_values[key] = beta_opt
                else:
                    params_values[key] = kwargs.get(key)
            all_hypo_params[hypo] = ModelParams(**params_values)
            all_hypo_ll[hypo] = ll_max

        best_hypo_idx = max(all_hypo_ll, key=all_hypo_ll.get)

        return (all_hypo_params[best_hypo_idx], all_hypo_ll[best_hypo_idx],
                all_hypo_params, all_hypo_ll)

    def fit_step_by_step(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         **kwargs) -> List[Dict]:
        """
        Fit the model step-by-step to observe how parameters evolve.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing (stimulus, choices, responses).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the fitting results for each
            step, such as the best hypothesis, beta, log-likelihood, and posterior.
        """

        if "perception" in self.modules:
            iSub = kwargs.get("iSub")
            new_stimulus = self.modules["perception"].sample(iSub, data[0])
            data = (new_stimulus, data[1], data[2])
        stimulus, _, responses = data
        n_trials = len(responses)

        # Precompute all distances
        self.partition_model.precompute_all_distances(stimulus)

        step_results = []

        if "cluster" in self.modules:
            next_hypos, init_strategy_amounts = self.modules[
                "cluster"].cluster_init(**kwargs.get("cluster_kwargs", {}))
            new_hypotheses_set = BaseSet(next_hypos)
            new_prior = BasePrior(new_hypotheses_set)
            new_likelihood = PartitionLikelihood(new_hypotheses_set,
                                                 self.partition_model)
            self.refresh_engine(new_hypotheses_set, new_prior, new_likelihood)

        for step_idx in range(1, n_trials + 1):
            selected_data = [x[:step_idx] for x in data]

            (best_params, best_ll, all_hypo_params,
             all_hypo_ll) = self.fit_single_step(selected_data,
                                                 use_cached_dist=True,
                                                 **kwargs)

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            infer_log_kwargs = {
                "use_cached_dist": True,
                "normalized": True,
            }
            for key in self.params_dict.keys():
                if key == "k":
                    continue
                elif key == "beta":
                    infer_log_kwargs[key] = hypo_betas
                else:
                    infer_log_kwargs[key] = kwargs.get(key)
            all_hypo_post = self.engine.infer_log(selected_data,
                                                  **infer_log_kwargs)

            hypo_details = {}

            for i, hypo in enumerate(self.hypotheses_set.elements):
                hypo_details[hypo] = {
                    'beta_opt': all_hypo_params[hypo].beta,
                    'll_max': all_hypo_ll[hypo],
                    'post_max': all_hypo_post[i],
                    'is_best': hypo == best_params.k
                }

            step_results.append({
                'best_k':
                best_params.k,
                'best_beta':
                best_params.beta,
                'best_params':
                asdict(best_params),
                'best_log_likelihood':
                best_ll,
                'best_norm_posterior':
                np.max(all_hypo_post),
                'hypo_details':
                hypo_details,
                'perception_stimuli':
                data[0][step_idx -
                        1] if "perception" in self.modules else None,
            })

            if "cluster" in self.modules:
                if step_idx < n_trials:
                    cur_post_dict = {
                        h: (det["post_max"], det["beta_opt"])
                        for h, det in hypo_details.items()
                    }
                    next_hypos, strategy_amounts = self.modules[
                        "cluster"].cluster_transition(
                            stimulus=data[0][step_idx],
                            feedbacks=data[2][max(0, step_idx - 16):step_idx],
                            posterior=cur_post_dict,
                            proto_hypo_amount=kwargs.get(
                                "cluster_prototype_amount",
                                HYPO_CLUSTER_PROTOTYPE_AMOUNT),
                            **kwargs.get("cluster_kwargs", {}))
                    step_results[-1]['best_step_amount'] = strategy_amounts
                    new_hypotheses_set = BaseSet(next_hypos)
                    new_prior = BasePrior(new_hypotheses_set)
                    new_likelihood = PartitionLikelihood(
                        new_hypotheses_set, self.partition_model)
                    self.refresh_engine(new_hypotheses_set, new_prior,
                                        new_likelihood)
                elif step_idx == n_trials:
                    step_results[-1]['init_amount'] = init_strategy_amounts

        return step_results

    def predict_choice(self,
                       data: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray],
                       step_results: list,
                       use_cached_dist,
                       window_size,
                       start_idx=0) -> Dict[str, np.ndarray]:
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

        for trial_idx in range(start_idx + 1, n_trials):
            trial_data = ([stimulus[trial_idx]], [choices[trial_idx]],
                          [responses[trial_idx]], [categories[trial_idx]])

            if (step_results[trial_idx - start_idx]['perception_stimuli']
                    is not None):

                trial_data = list(trial_data)
                trial_data[0] = step_results[trial_idx -
                                             start_idx]['perception_stimuli']
                trial_data = tuple(trial_data)

            # Extract the posterior probabilities for each hypothesis at last trial
            hypo_details = step_results[trial_idx - 1 -
                                        start_idx]['hypo_details']

            post_max = [
                hypo_details[k]['post_max'] for k in hypo_details.keys()
            ]

            # Compute the weighted probability of being correct
            weighted_p_true = 0
            for k, post in zip(hypo_details.keys(), post_max):
                p_true = self.partition_model.calc_trueprob_entry(
                    k,
                    trial_data,
                    hypo_details[k]['beta_opt'],
                    use_cached_dist=use_cached_dist,
                    indices=[trial_idx])
                weighted_p_true += post * p_true

            pred_acc[trial_idx] = weighted_p_true

        # Compute sliding averages using a sliding window
        sliding_true_acc = []
        sliding_pred_acc = []
        sliding_pred_acc_std = []

        for start_idx in range(1, n_trials - window_size +
                               2):  # Start from index 1
            end_idx = start_idx + window_size
            sliding_true_acc.append(np.mean(true_acc[start_idx:end_idx]))
            pred_window = pred_acc[start_idx:end_idx]
            sliding_pred_acc.append(np.mean(pred_window))
            sliding_pred_acc_std.append(
                np.sqrt(np.sum(pred_window * (1 - pred_window))) / window_size)

        predict_results = {
            'true_acc': true_acc,
            'pred_acc': pred_acc,
            'sliding_true_acc': sliding_true_acc,
            'sliding_pred_acc': sliding_pred_acc,
            'sliding_pred_acc_std': sliding_pred_acc_std
        }

        return predict_results

    def compute_error_for_params(self,
                                 data: Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray],
                                 window_size=16,
                                 repeat=1,
                                 multiprocess=False,
                                 **kwargs) -> Tuple[List[Dict], float]:
        """
        Perform a single pass of model fitting and prediction, then compute an error metric.

        Parameters
        ----------
        data : tuple
            A tuple containing (stimuli, choices, responses, categories).
        window_size : int
            The window size used for sliding accuracy measurements.
        repeat : int
            The number of times to repeat the fitting process.
        multiprocess : bool
            Whether to use multiprocessing for parallel computation.
        n_jobs : int
            The number of jobs to run in parallel (if multiprocess is True).

        Returns
        -------
        all_step_results : List[List[Dict]]
            A list of output of fit_step_by_step showing model fits for each trial.
        all_mean_error : List[float]
            A list of average windowed errors between predicted and true accuracy.
        """

        # Fit the model with fixed params
        if multiprocess:

            def compute_single_fit(data, **kwargs):
                step_results = self.fit_step_by_step(data[:3],
                                                     ground_truth=data[3],
                                                     **kwargs)

                predict_results = self.predict_choice(data,
                                                      step_results,
                                                      use_cached_dist=True,
                                                      window_size=window_size)
                mean_error = np.mean(
                    np.abs(
                        np.array(predict_results['sliding_true_acc']) -
                        np.array(predict_results['sliding_pred_acc'])))
                return step_results, mean_error

            results = Parallel(n_jobs=kwargs.get("n_jobs", 2))(
                delayed(compute_single_fit)(data, **kwargs) for i in tqdm(
                    range(repeat), desc="Computing error for params"))
            all_step_results = [result[0] for result in results]
            all_mean_error = [result[1] for result in results]

        else:
            selected_data = data[:3]
            all_step_results = []
            all_mean_error = []

            for _ in range(repeat):
                step_results = self.fit_step_by_step(selected_data, **kwargs)
                all_step_results.append(step_results)

                # Get the predicted accuracy
                predict_results = self.predict_choice(data,
                                                      step_results,
                                                      use_cached_dist=True,
                                                      window_size=window_size)

                # Calculate the mean absolute error between predicted and true accuracy
                mean_error = np.mean(
                    np.abs(
                        np.array(predict_results['sliding_true_acc']) -
                        np.array(predict_results['sliding_pred_acc'])))
                all_mean_error.append(mean_error)

        return all_step_results, all_mean_error

    def optimize_params(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray],
                        **kwargs) -> Tuple[BaseModelParams, list]:

        grid_errors = {}
        grid_step_results = {}

        total_combinations = 1
        for key, values in self.optimize_params_dict.items():
            total_combinations *= len(values)

        def evaluate_params(grid_values):
            grid_params = dict(
                zip(self.optimize_params_dict.keys(), grid_values))
            all_step_results, all_mean_error = self.compute_error_for_params(
                data,
                window_size=kwargs.get("window_size", 16),
                repeat=kwargs.get("grid_repeat", 5),
                multiprocess=False**grid_params)
            return tuple(
                grid_params.values()), all_step_results, all_mean_error

        eval_list = Parallel(n_jobs=kwargs.get("n_jobs", 2))(
            delayed(evaluate_params)(grid_values) for grid_values in tqdm(
                product(*self.optimize_params_dict.values()),
                desc="Evaluating parameter combinations",
                total=total_combinations,
            ))

        for key, all_step_results, all_mean_error in eval_list:
            grid_errors[key] = all_mean_error
            grid_step_results[key] = all_step_results

        best_key = min(grid_errors, key=lambda x: np.mean(grid_errors[x]))

        mc_samples = kwargs.get("mc_samples", 100)

        def refit_model(specific_params):
            step_results, mean_error = self.compute_error_for_params(
                data,
                window_size=kwargs.get("window_size", 16),
                repeat=mc_samples,
                multiprocess=True,
                n_jobs=kwargs.get("n_jobs", 2),
                **specific_params)
            idx = np.argmin(mean_error)
            return step_results[idx], mean_error[idx]

        best_step_results, best_mean_error = refit_model(
            dict(zip(self.optimize_params_dict.keys(), best_key)))

        optimized_params_results = {
            "optim_params": self.optimize_params_dict.keys(),
            "best_params": best_key,
            "best_error": best_mean_error,
            "best_step_results": best_step_results,
            "grid_errors": grid_errors,
        }

        return optimized_params_results

    def on_policy_compute_error_for_params(
            self,
            data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            window_size=16,
            repeat=1,
            multiprocess=False,
            initial_states: Dict | None = None,  # [TODO]
            **kwargs) -> Tuple[List[Dict], float]:
        """
        Perform a single pass of model fitting and prediction, then compute an error metric.

        Parameters
        ----------
        data : tuple
            A tuple containing (stimuli, choices, responses, categories).
        window_size : int
            The window size used for sliding accuracy measurements.
        repeat : int
            The number of times to repeat the fitting process.
        multiprocess : bool
            Whether to use multiprocessing for parallel computation.
        n_jobs : int
            The number of jobs to run in parallel (if multiprocess is True).

        Returns
        -------
        all_step_results : List[List[Dict]]
            A list of output of fit_step_by_step showing model fits for each trial.
        all_mean_error : List[float]
            A list of average windowed errors between predicted and true accuracy.
        """
        initial_states = initial_states or self.initial_states
        on_policy_start_trial = len(initial_states["best_step_results"])

        def compute_single_fit(data, **kwargs):
            step_results = self.on_policy_fit_step_by_step(
                deepcopy(data[:3]), ground_truth=data[3], **kwargs)

            predict_results = self.predict_choice(
                data,
                step_results,
                use_cached_dist=False,
                window_size=window_size,
                start_idx=on_policy_start_trial)

            mean_error = np.mean(
                np.abs(
                    np.array(predict_results['sliding_true_acc']) -
                    np.array(predict_results['sliding_pred_acc'])))
            return step_results, mean_error

        # Fit the model with fixed params
        if multiprocess:
            results = Parallel(n_jobs=kwargs.get("n_jobs", 2))(
                delayed(compute_single_fit)(data, **kwargs) for i in tqdm(
                    range(repeat), desc="Computing error for params"))
            all_step_results = [result[0] for result in results]
            all_mean_error = [result[1] for result in results]

        else:
            all_step_results = []
            all_mean_error = []

            for _ in range(repeat):
                step_results, mean_error = compute_single_fit(data, **kwargs)
                all_step_results.append(step_results)
                all_mean_error.append(mean_error)

        return all_step_results, all_mean_error

    def on_policy_judge(self, a, b):
        if a == b:
            return 1.
        if self.condition == 3 and a // 2 == b // 2:
            return 0.5
        return 0.

    def on_policy_decision_making(self, x, beta, hypo_set, prior, **kwargs):
        """
        Make a decision based on the given stimulus.
        Parameters
        ----------
        x : np.ndarray
            The stimulus.
        Returns
        -------
        int
            The decision.
        """
        prob = np.zeros(self.n_cats)

        for k, i in hypo_set.inv.items():
            likelihood = self.partition_model.calc_likelihood_base(
                k, [np.array([x]), np.array([1]),
                    np.array([0])], beta[i], **kwargs)
            prob += likelihood.reshape(-1) * prior[i]

        # [TODO] 这里把Decision模块加进来实现吧？
        return np.random.choice(self.n_cats, p=prob) + 1

    def on_policy_fit_step_by_step(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        ground_truth: np.ndarray,
        **kwargs,
    ) -> List[Dict]:
        """
        Fit the model step-by-step to observe how parameters evolve.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing (stimulus, choices, responses).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the fitting results for each
            step, such as the best hypothesis, beta, log-likelihood, and posterior.
        """
        if "perception" in self.modules:
            iSub = kwargs.get("iSub")
            new_stimulus = self.modules["perception"].sample(iSub, data[0])
            data = (new_stimulus, data[1], data[2])
        stimulus, _, responses = data
        n_trials = len(responses)

        # Precompute all distances
        self.partition_model.precompute_all_distances(stimulus)

        step_results = []

        # ==============  recover from initial states  =======================
        initial_states = kwargs.get("initial_states", self.initial_states)

        on_policy_start_trial = len(initial_states["best_step_results"])
        initial_step_state = initial_states["best_step_results"][-1]
        # kwargs["cluster_kwargs"] = initial_step_state["hypo_details"]
        hypo_betas = [
            x["beta_opt"] for x in initial_step_state["hypo_details"].values()
        ]

        # ==============  recover from initial states  =======================
        if "cluster" in self.modules:

            next_hypos, init_strategy_amounts = self.modules[
                "cluster"].cluster_init(**kwargs.get("cluster_kwargs", {}))
            next_hypos = initial_step_state["hypo_details"].keys()
            new_hypotheses_set = BaseSet(next_hypos)
            new_prior = BasePrior(new_hypotheses_set)
            new_prior.update(
                np.array([
                    x["post_max"]
                    for x in initial_step_state["hypo_details"].values()
                ]))

            new_likelihood = PartitionLikelihood(new_hypotheses_set,
                                                 self.partition_model)
            self.refresh_engine(new_hypotheses_set, new_prior, new_likelihood)
        hypo_set = self.engine.hypotheses_set
        prior = self.engine.prior.value
        # print("\n" * 5, initial_step_state)

        for step_idx in range(on_policy_start_trial + 1, n_trials + 1):

            # generate selection (choice) and response
            data[1][step_idx - 1] = self.on_policy_decision_making(
                data[0][step_idx - 1], hypo_betas, hypo_set, prior)

            data[2][step_idx - 1] = self.on_policy_judge(
                data[1][step_idx - 1], ground_truth[step_idx - 1])

            selected_data = [x[:step_idx] for x in data]

            (best_params, best_ll, all_hypo_params,
             all_hypo_ll) = self.fit_single_step(selected_data,
                                                 use_cached_dist=False,
                                                 **kwargs)

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            infer_log_kwargs = {
                "use_cached_dist": True,
                "normalized": True,
            }
            for key in self.params_dict.keys():
                if key == "k":
                    continue
                elif key == "beta":
                    infer_log_kwargs[key] = hypo_betas
                else:
                    infer_log_kwargs[key] = kwargs.get(key)
            all_hypo_post = self.engine.infer_log(selected_data,
                                                  **infer_log_kwargs)

            hypo_details = {}

            for i, hypo in enumerate(self.hypotheses_set.elements):
                hypo_details[hypo] = {
                    'beta_opt': all_hypo_params[hypo].beta,
                    'll_max': all_hypo_ll[hypo],
                    'post_max': all_hypo_post[i],
                    'is_best': hypo == best_params.k
                }

            hypo_set = deepcopy(self.hypotheses_set)
            prior = [
                hypo_details[hypo]["post_max"] for hypo in hypo_set.elements
            ]
            # Update the step result
            step_results.append({
                'best_k':
                best_params.k,
                'best_beta':
                best_params.beta,
                'best_params':
                asdict(best_params),
                'best_log_likelihood':
                best_ll,
                'best_norm_posterior':
                np.max(all_hypo_post),
                'hypo_details':
                hypo_details,
                'perception_stimuli':
                data[0][step_idx -
                        1] if "perception" in self.modules else None,
                'on_policy_choice':
                data[1],
                'on_policy_response':
                data[2],
            })

            if "cluster" in self.modules:
                if step_idx < n_trials:
                    cur_post_dict = {
                        h: (det["post_max"], det["beta_opt"])
                        for h, det in hypo_details.items()
                    }
                    next_hypos, strategy_amounts = self.modules[
                        "cluster"].cluster_transition(
                            stimulus=data[0][step_idx],
                            feedbacks=data[2][max(0, step_idx - 16):step_idx],
                            posterior=cur_post_dict,
                            proto_hypo_amount=kwargs.get(
                                "cluster_prototype_amount",
                                HYPO_CLUSTER_PROTOTYPE_AMOUNT),
                            **kwargs.get("cluster_kwargs", {}))
                    step_results[-1]['best_step_amount'] = strategy_amounts
                    new_hypotheses_set = BaseSet(next_hypos)
                    new_prior = BasePrior(new_hypotheses_set)
                    new_likelihood = PartitionLikelihood(
                        new_hypotheses_set, self.partition_model)
                    self.refresh_engine(new_hypotheses_set, new_prior,
                                        new_likelihood)
                elif step_idx == n_trials:
                    step_results[-1]['init_amount'] = init_strategy_amounts

        return step_results
