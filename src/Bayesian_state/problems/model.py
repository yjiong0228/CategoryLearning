"""
Base Model
"""
from abc import ABC
from dataclasses import dataclass, make_dataclass, asdict
from typing import Dict, Tuple, List, Callable, Optional
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize
from collections import Counter
from itertools import product
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
from .partitions import Partition, BasePartition
from ..utils import softmax, PATHS, BASE_CONFIG, MODEL_STRUCT, print
from ..utils.perception_stats import (
    get_perception_noise_stats,
    PerceptionStatsError,
)
from ..utils.base import LOGGER

print(MODEL_STRUCT, s=2)

EPS = 1e-15

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


class SoftGridFlatLikelihood(BaseLikelihood):
    """
    Wrap SoftPartitionLikelihood to present a flattened hypothesis space
    consisting of (k, beta_idx) pairs.

    The inner SoftPartitionLikelihood computes an (n_h, n_beta) matrix of
    likelihoods; we reshape it into a vector of length n_h * n_beta so the
    BaseEngine can treat each (k, beta) as one hypothesis.
    """

    def __init__(self, space: BaseSet, partition: BasePartition,
                 base_k_set: BaseSet, beta_grid: list):
        super().__init__(space)
        self.partition = partition
        self.base_k_set = base_k_set
        self.beta_grid = list(beta_grid)
        # Reuse the provided SoftPartitionLikelihood for efficient batch calc
        self.inner = SoftPartitionLikelihood(base_k_set, partition,
                                             self.beta_grid)
        self.n_h = len(base_k_set.elements)
        self.n_b = len(self.beta_grid)

    def get_likelihood(self,
                       observation,
                       beta: float | list | tuple = 1.0,
                       use_cached_dist: bool = False,
                       normalized: bool = True,
                       **kwargs) -> np.ndarray:
        # inner returns an array shaped (n_h, n_b) by concatenating along axis=1
        mat = self.inner.get_likelihood(observation,
                                        use_cached_dist=use_cached_dist,
                                        normalized=normalized,
                                        **kwargs)
        mat = np.atleast_2d(mat)
        # Flatten row-major: index = k_idx * n_b + b_idx
        return mat.reshape(self.n_h * self.n_b)

    # Optional helpers for mapping indices
    def index_to_pair(self, idx: int) -> tuple[int, int]:
        k_idx, b_idx = divmod(idx, self.n_b)
        return k_idx, b_idx

    def pair_to_index(self, k_idx: int, b_idx: int) -> int:
        return k_idx * self.n_b + b_idx




#################################### 新的 Model ####################################

class StateModel:
    """
    State Model, initialize an engine
    Refreshes the engine step by step:
        StateModel ----[data]----> engine
        engine ----[posterior]----> StateModel
    """

    def __init__(self, engine_config, **kwargs):
        """
        """

        # Initialize attributes
        self.engine_config = deepcopy(engine_config)
        self.all_centers = None
        self.data = None
        self.hypotheses_set = BaseSet([])
        self.observation_set = BaseSet([])

        self.condition = kwargs.get("condition", 1)
        self.subject_id = kwargs.pop("subject_id", None)
        processed_data_dir = kwargs.pop("processed_data_dir", None)
        if processed_data_dir is None:
            self.processed_data_dir = (
                PATHS["root"].parent / "data" / "processed"
            ).resolve()
        else:
            self.processed_data_dir = Path(processed_data_dir).resolve()

        n_dims = 4
        self.n_cats = 2 if self.condition == 1 else 4

        # Initialize partition
        self.partition_model = kwargs.get(
            "partition", Partition(n_dims, self.n_cats))
        # Initialize hypotheses set (length = partition_model.length)
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        # Merge module overrides provided via kwargs
        for key, value in kwargs.items():
            if key in self.engine_config.get("modules", {}):
                self.engine_config["modules"][key].update(value)

        self.perception_mean = None
        self.perception_std = None
        self._inject_perception_parameters()

        # initialize engine
        self.engine = BaseEngine(
            self.engine_config["agenda"],
            hypotheses_set=self.hypotheses_set,
            partition=self.partition_model,
        )
        # build modules for the engine
        self.engine.build_modules(self.engine_config["modules"])

    def _inject_perception_parameters(self) -> None:
        modules_cfg = self.engine_config.get("modules", {})
        if not modules_cfg:
            return

        perception_modules = [
            name for name, cfg in modules_cfg.items()
            if self._is_perception_module(cfg.get("class"))
        ]
        if not perception_modules:
            return

        if self.subject_id is None:
            raise ValueError(
                "StateModel requires 'subject_id' when a perception module is configured."
            )

        try:
            mean_map, std_map = get_perception_noise_stats(self.processed_data_dir)
        except PerceptionStatsError as exc:
            raise ValueError(
                f"Failed to compute perception statistics from {self.processed_data_dir}"
            ) from exc

        if self.subject_id not in mean_map:
            raise ValueError(
                f"Subject {self.subject_id} does not exist in perception statistics"
            )

        mean_vector = mean_map[self.subject_id]
        std_vector = std_map[self.subject_id]

        self.perception_mean = mean_vector
        self.perception_std = std_vector

        mean_list = np.asarray(mean_vector, dtype=float).tolist()
        std_list = np.asarray(std_vector, dtype=float).tolist()

        for mod_name in perception_modules:
            mod_cfg = modules_cfg[mod_name]
            mod_kwargs = mod_cfg.setdefault("kwargs", {})

            if not self._parameter_is_vector(mod_kwargs.get("mean")):
                mod_kwargs["mean"] = mean_list
            else:
                mod_kwargs["mean"] = np.asarray(mod_kwargs["mean"], dtype=float).tolist()

            if not self._parameter_is_vector(mod_kwargs.get("std")):
                mod_kwargs["std"] = std_list
            else:
                mod_kwargs["std"] = np.asarray(mod_kwargs["std"], dtype=float).tolist()

            mod_kwargs.setdefault("subject_id", self.subject_id)

    @staticmethod
    def _parameter_is_vector(value, size: int = 4) -> bool:
        if value is None:
            return False
        if isinstance(value, (float, int)):
            return False
        if isinstance(value, dict):
            if all(isinstance(k, int) for k in value.keys()):
                return all(k in value for k in range(size))
            return all(k in value for k in ["neck", "head", "leg", "tail"])
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return False
        return arr.ndim == 1 and arr.shape[0] == size

    @staticmethod
    def _is_perception_module(class_spec) -> bool:
        if class_spec is None:
            return False
        if isinstance(class_spec, str):
            return class_spec.split(".")[-1] == "PerceptionModule"
        try:
            from .modules.perception import PerceptionModule as _PerceptionModule  # pylint: disable=import-outside-toplevel
        except ImportError:
            return False
        try:
            return issubclass(class_spec, _PerceptionModule)
        except TypeError:
            return False


    def precompute_distances(self, stimulus: np.ndarray):
        """
        Precompute all distances.
        """
        if hasattr(self.partition_model, "precompute_all_distances"):
            self.partition_model.precompute_all_distances(stimulus)



    def save(self, posterior_log, step_log):
        """
        保存结果
        """
        self.posterior_log = posterior_log
        self.step_log = step_log



    def fit_step_by_step(self, data: List | np.ndarray, **kwargs):
        """
        """
        # TODO: optimize w0, gamma in memory module
        
        # load module kwargs
        mod_kwargs = kwargs.get("module_kwargs", {})
        # fit step by step
        data = data or self.data
        step_log = []
        posterior_log = []
        prior_log = []
        for datum in data:
            posterior, log = self.engine.infer_single(datum, mod_kwargs)
            # DEBUG
            #print("Current observation:", self.engine.observation, s=2)
            step_log += [log]
            posterior_log += [posterior]
            prior_log += [log.get('prior')]

        self.save(posterior_log, step_log)
        return posterior_log, prior_log



#################################### 新的 Model END 
