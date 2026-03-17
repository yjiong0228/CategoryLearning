"""
Module: Perception Mechanism
"""

from typing import Dict, List
import numpy as np
import os
import pandas as pd
from .base_module import BaseModule

DEFAULT_processed_data_DIR = os.path.join(os.path.dirname(__file__),
                                          "../../../../data/processed")


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
        self.processed_data_dir = kwargs.pop("processed_data_path",
                                             DEFAULT_processed_data_DIR)

        perception_error_data = pd.read_csv(
            os.path.join(self.processed_data_dir, "Task1b_errorsummary_24.csv"))

        self.mean: Dict[str, Dict[str, float]] = {}
        self.std: Dict[str, Dict[str, float]] = {}
        self.mean, self.std = self.calculate_mean_std(perception_error_data)
        
        learning_data = pd.read_csv(
            os.path.join(self.processed_data_dir, "Task2_processed.csv"))    
        self.feature_names : Dict[int, Dict[str, str]]
        self.feature_names = self.get_feature_names(learning_data)
        

    def calculate_mean_std(self, error):
        """
        Calculate mean and standard deviation for each subject

        Args:
            error (pd.DataFrame): DataFrame containing error data
        Returns:
            dict: Mean and standard deviation for each subject
        """
        mean = error.groupby('iSub').apply(
            lambda group: group.filter(like='_error_mean').mean())
        std = error.groupby('iSub').apply(
            lambda group: group.filter(like='_error_sd').std())

        mean = mean.rename(
            columns=lambda x: x.replace('_length_error_mean', '')).to_dict(
                orient='index')
        std = std.rename(
            columns=lambda x: x.replace('_length_error_sd', '')).to_dict(
                orient='index')
        return mean, std


    def get_feature_names(self, learning_data):
        feature_cols = ["feature1_name", "feature2_name", "feature3_name", "feature4_name"]

        sub_feature_df = learning_data[["iSub"] + feature_cols].drop_duplicates(subset=["iSub"])

        for sub_id, group in learning_data.groupby("iSub"):
            unique_rows = group[feature_cols].drop_duplicates()
            if len(unique_rows) > 1:
                raise ValueError(f"iSub={sub_id} has inconsistent feature names across rows.")

            feature_names = {
                int(row["iSub"]): [
                    row["feature1_name"],
                    row["feature2_name"],
                    row["feature3_name"],
                    row["feature4_name"],
                ]
            for _, row in sub_feature_df.iterrows()
        }
        return feature_names


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

        feature_names_sub = self.feature_names[iSub]
        n_trials = len(stimulus)
        for i in range(stimulus.shape[1]):
            stimulus[:, i] = stimulus[:, i] + np.random.normal(
                loc=self.mean[iSub][feature_names_sub[i]],
                scale=self.std[iSub][feature_names_sub[i]],
                size=n_trials)
        # Ensure the values are in the range of [0, 1]
        stimulus = np.clip(stimulus, 0, 1)
        return stimulus
