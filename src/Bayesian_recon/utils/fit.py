import os
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, List, Dict
from itertools import product
from collections import defaultdict

from ..problems import (
    StandardModel
)

PROJECT_ROOT_PATH = Path(os.getcwd()).parent.parent.parent.parent.parent
DEFAULT_DATA_PATH = Path(PROJECT_ROOT_PATH, "data", "processed", "Task2_processed.csv")

class FitOptimizer(object):
    def __init__(self, module_config: dict, n_jobs: int):
        """
        Initialize the FitOptimizer class.

        Args:
            module_config (dict): Configuration dictionary containing model parameters and settings.
            n_jobs (int): Number of parallel jobs to run.
        """
        self.module_config = module_config
        self.n_jobs = n_jobs
        self.optimize_params_dict = {}

        modules = {}
        for key, (mod_cls, mod_kwargs) in self.module_config.items():
            try:
                modules[key] = mod_cls(self, **mod_kwargs)
            except Exception as e:
                print(f"Error initializing module {key}: {e}")
        for key, mod in modules.items():
            if hasattr(mod, 'optimize_params_dict'):
                self.optimize_params_dict.update(mod.optimize_params_dict)

    def prepare_data(self, data_path: str = DEFAULT_DATA_PATH):
        """
        Prepare data for analysis.
        
        Args:
            data_path (str): The path to the data file.
        """
        self.learning_data = pd.read_csv(data_path)

    def optimize_params_with_subs_parallel(self, model_config: Dict, subjects: Optional[List[str]] = None):
        if subjects is None:
            subjects = self.learning_data["subject"].unique()
        subject_data_map = {iSub: self.learning_data[self.learning_data["iSub"] == iSub] for iSub in subjects}

        def process_single_task(iSub, subject_data, **kwargs):
            condition = subject_data["condition"].iloc[0]

            s_data = (
                subject_data[["feature1", "feature2", "feature3", "feature4"]].values,
                subject_data["choice"].values,
                subject_data["feedback"].values, 
                subject_data["category"].values
            )

            model = StandardModel(model_config, module_config=self.module_config, condition=condition)
            grad_params = {}
            for key in self.optimize_params_dict.keys():
                grad_params[key] = kwargs[key]
            step_results, mean_error = model.compute_error_for_params(
                s_data, window_size=kwargs.get("window_size", 16), **grad_params
            )
            return iSub, grad_params, mean_error, step_results
        
        all_kwargs = []
        for task_values in product(subjects, *self.optimize_params_dict.values()):
            iSub = task_values[0]
            grid_values = task_values[1:]
            kwargs = dict(zip(self.optimize_params_dict.keys(), grid_values))
            kwargs['iSub'] = iSub
            kwargs['subject_data'] = subject_data_map[iSub]
            all_kwargs.append(kwargs)

        results = Parallel(n_jobs=self.n_jobs, batch_size=1)(
            delayed(process_single_task)(**kwargs) for kwargs in tqdm(all_kwargs, desc="Processing tasks", total=len(all_kwargs), ncols=100, leave=True, position=0)
        )

        subject_grid_errors = defaultdict(dict)
        subject_best_combo = {}

        for iSub, grad_params, mean_error, step_results in results:
            subject_grid_errors[iSub][tuple(grad_params.values())] = mean_error

            if iSub not in subject_best_combo:
                subject_best_combo[iSub] = {
                    "params": grad_params,
                    "error": mean_error,
                    "step_results": step_results
                }
            else:
                best_error = subject_best_combo[iSub]["error"]
                if mean_error < best_error:
                    subject_best_combo[iSub] = {
                        "params": grad_params,
                        "error": mean_error,
                        "step_results": step_results
                    }

        fitting_results = {}
        for iSub in subject_grid_errors.keys():
            fitting_results[iSub] = {
                "condition": subject_data_map[iSub]["condition"].iloc[0],
                "best_params": subject_best_combo[iSub]["params"],
                "best_error": subject_best_combo[iSub]["error"],
                "best_step_results": subject_best_combo[iSub]["step_results"],
                "grid_errors": subject_grid_errors[iSub]
            }
        return fitting_results

            

        

