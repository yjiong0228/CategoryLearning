"""
Module: Perception Mechanism
"""

from typing import Dict, List
import numpy as np
import os
import pandas as pd
from .base_module import BaseModule

DEFAULT_PROCESSED_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../../../data/processed")

class BasePerception(BaseModule):
    """
    Base Perception
    """

    def __init__(self, model, **kwargs):
        """
        Initialize

        Args:
            model (BaseModelParams): Model parameters
            **kwargs: Additional keyword arguments
        """
        super().__init__(model, **kwargs)
        self.processed_data_dir = kwargs.pop("processed_data_path", DEFAULT_PROCESSED_DATA_DIR)
        self.mean : Dict[str, Dict[str, float]] = {}
        self.std : Dict[str, Dict[str, float]] = {}
        processed_data = pd.read_csv(os.path.join(self.processed_data_dir, "Task1b_processed.csv"))
        error = self.error_calculation(processed_data)
        self.mean, self.std = self.calculate_mean_std(error)
        self.structures : Dict[str, List]= self.get_structures()

    def error_calculation(self, processed_data):
        """
        Calculate error between target and adjust_after

        Args:
            processed_data (pd.DataFrame): DataFrame containing processed data
        Returns:
            pd.DataFrame: DataFrame containing error data
        """
        columns = ['neck_length', 'head_length', 'leg_length', 'tail_length']
        
        results = []
        for iSub, group in processed_data.groupby('iSub'):
            target = group[group['type'] == 'target'].reset_index(drop=True)
            adjust_after = group[group['type'] == 'adjust_after'].reset_index(drop=True)

            result = target[['iSub','iTrial'] + columns].reset_index(drop=True).copy()
            for col in columns:
                result[f'{col}_diff'] = adjust_after[col] - target[col]
            results.append(result)
            
        error = pd.concat(results, ignore_index=True)

        return error
    
    def calculate_mean_std(self, error):
        """
        Calculate mean and standard deviation for each subject

        Args:
            error (pd.DataFrame): DataFrame containing error data
        Returns:
            dict: Mean and standard deviation for each subject
        """
        mean = error.groupby('iSub').apply(
            lambda group: group.filter(like='_diff').mean()
        )
        std = error.groupby('iSub').apply(
            lambda group: group.filter(like='_diff').std()
        )

        mean = mean.rename(columns=lambda x: x.replace('_length_diff', '')).to_dict(orient='index')
        std = std.rename(columns=lambda x: x.replace('_length_diff', '')).to_dict(orient='index')
        return mean, std

    def get_structures(self):
        """
        Get structures for each subject

        Returns:
            dict: Structures for each subject
        """
        structures = {}
        processed_data = pd.read_csv(os.path.join(self.processed_data_dir, "Task2_processed.csv"))

        for iSub, group in processed_data.groupby('iSub'):
            structures[iSub] = group[['structure1', 'structure2']].values
            assert np.all(structures[iSub] == structures[iSub][0]), f"iSub {iSub} has inconsistent structures."
            structures[iSub] = structures[iSub][0].tolist()

        return structures

    def sample(self, iSub, stimulus):
        """
        Sample from the model

        Args:
            iSub (int): Subject index
            stimulus (np.ndarray): Stimulus to sample from, shape (trials, features)
        Returns:
            np.ndarray: Sampled stimulus with noise added, shape (trials, features)
        """
        if iSub not in self.mean or iSub not in self.std:
            raise ValueError(f"Subject {iSub} not found in mean or std data.")
        
        def convert(structure):
            # feature selection
            if structure[0] == 1:
                features = ["neck", "head", "leg", "tail"]
            elif structure[0] == 2:
                features = ["neck", "head", "tail", "leg"]
            elif structure[0] == 3:
                features = ["neck", "leg", "tail", "head"]
            elif structure[0] == 4:
                features = ["head", "leg", "tail", "neck"]

            # feature space segmentation
            if structure[1] == 1:
                features = features[:]
            elif structure[1] == 2:
                features = [features[0], features[2], features[1], features[3]]
            elif structure[1] == 3:
                features = [features[1], features[0], features[2], features[3]]
            elif structure[1] == 4:
                features = [features[1], features[2], features[0], features[3]]
            elif structure[1] == 5:
                features = [features[2], features[0], features[1], features[3]]
            elif structure[1] == 6:
                features = [features[2], features[1], features[0], features[3]]

            # Final rearrangement
            features = [features[0], features[2], features[1], features[3]]

            # Add suffix to feature names
            return features
        
        feat = convert(self.structures[iSub])
        n_trials = len(stimulus)
        for i in range(stimulus.shape[1]):
            stimulus[:, i] = stimulus[:, i] + np.random.normal(
                loc=self.mean[iSub][feat[i]], scale=self.std[iSub][feat[i]], size=n_trials
            )
        # Ensure the values are in the range of [0, 1]
        stimulus = np.clip(stimulus, 0, 1)
        return stimulus
    


################### NEW Perception Module ###################

class PerceptionModule(BaseModule):
    """Perception noise module with subject-specific parameters."""

    def __init__(self, engine, **kwargs):
        """Initialize the perception module.

        Parameters
        ----------
        engine: BaseEngine
            Hosting inference engine instance.
        features: int, optional
            Number of stimulus features (defaults to 4).
        mean: Sequence[float] | dict | float, optional
            Mean offsets for each feature. Scalars are broadcast, iterables are
            coerced to numpy arrays, and dictionaries keyed by feature names or
            indices are supported.
        std: Sequence[float] | dict | float, optional
            Standard deviations for each feature. Scalars are broadcast and
            dictionaries mirror the behaviour described for ``mean``.
        subject_id: int, optional
            Stored for debugging/logging; it does not affect computations.
        """

        super().__init__(engine, **kwargs)
        self.features = kwargs.pop("features", 4)
        self.subject_id = kwargs.pop("subject_id", None)

        self.mean = self._coerce_vector(kwargs.pop("mean", 0.0), "mean")
        self.std = np.abs(
            self._coerce_vector(kwargs.pop("std", 0.1), "std")
        )

    def _coerce_vector(self, value, name: str) -> np.ndarray:
        """Convert incoming parameter to a feature-sized numpy array."""

        if isinstance(value, (float, int)):
            return np.full(self.features, float(value), dtype=float)

        if isinstance(value, dict):
            if all(k in value for k in range(self.features)):
                ordered = [float(value[i]) for i in range(self.features)]
                return np.asarray(ordered, dtype=float)
            if all(k in value for k in ["neck", "head", "leg", "tail"]):
                ordered = [
                    float(value[key]) for key in ["neck", "head", "leg", "tail"]
                ]
                if len(ordered) != self.features:
                    raise ValueError(
                        f"Expected {self.features} feature entries for {name}"
                    )
                return np.asarray(ordered, dtype=float)
            raise ValueError(
                f"Unsupported dictionary format for parameter '{name}'"
            )

        array = np.asarray(value, dtype=float)
        if array.ndim != 1 or array.shape[0] != self.features:
            raise ValueError(
                f"Parameter '{name}' must be a 1-D array of length {self.features}"
            )
        return np.nan_to_num(array, nan=0.0)

    def sample(self, stimu):
        """
        stimu for single trial, shape: (features, )
        """
        # stimu 的每个维度加上各个特征各自的mean和std采样的噪声
        noise = np.random.normal(loc=self.mean, scale=self.std, size=stimu.shape)
        return stimu + noise
    
    def process(self, **kwargs):
        """
        Process the stimulus with perception noise

        Args:
            **kwargs: Additional keyword arguments
                - stimu (np.ndarray): Stimulus to process, shape: (features, )
        """
        stimu = kwargs.get("stimu", self.engine.observation[0])
        sampled = self.sample(stimu)
        # Ensure the values are in the range of [0, 1]
        sampled = np.clip(sampled, 0, 1)
        self.engine.observation = (sampled, self.engine.observation[1], self.engine.observation[2])
        return sampled
