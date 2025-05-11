"""
Module: Perception Mechanism
"""

from typing import Dict
import numpy as np
import os
import pandas as pd
from .base_module import BaseModule

DEFAULT_PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "../../../../data/processed/Task1b_processed.csv")

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
        processed_data_path = kwargs.pop("processed_data_path", DEFAULT_PROCESSED_DATA_PATH)
        self.variances : Dict[str, Dict[str, float]] = {}
        processed_data = pd.read_csv(processed_data_path)
        error = self.error_calculation(processed_data)
        self.variances = self.variances_calculation(error)

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
    
    def variances_calculation(self, error):
        """
        Calculate variances for each subject

        Args:
            error (pd.DataFrame): DataFrame containing error data
        Returns:
            dict: Variances for each subject
        """
        variances = error.groupby('iSub').apply(
            lambda group: group.filter(like='_diff').pow(2).mean()
        )
        variances = variances.to_dict(orient='index')
        variances = {k: {key.split('_length')[0]: value for key, value in v.items()} for k, v in variances.items()}
        return variances
    
    def sample(self, iSub, stimulus):
        """
        Sample from the model

        Args:
            iSub (int): Subject index
            stimulus (np.ndarray): Stimulus to sample from, shape (trials, features)
        Returns:
            np.ndarray: Sampled stimulus with noise added, shape (trials, features)
        """
        if iSub not in self.variances:
            raise ValueError(f"iSub {iSub} not found in variances.")

        feat = ['leg', 'head', 'tail', 'neck']
        n_trials = len(stimulus)
        for i in range(stimulus.shape[1]):
            stimulus[:, i] = stimulus[:, i] + np.random.normal(
                loc=0, scale=np.sqrt(self.variances[iSub][feat[i]]), size=n_trials
            )
        return stimulus