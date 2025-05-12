import os
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, List, Dict, Union
from itertools import product
from collections import defaultdict, UserDict
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..problems import (
    StandardModel
)

from .stream import StreamList

PROJECT_ROOT_PATH = Path(os.getcwd()).parent.parent.parent.parent.parent
DEFAULT_DATA_PATH = Path(PROJECT_ROOT_PATH, "data", "processed", "Task2_processed.csv")


class ConstantDict(UserDict):
    def __init__(self, constant_value):
        super().__init__()
        self._constant_value = constant_value

    def __getitem__(self, key):
        return self._constant_value
    
    def get(self, key, default=None):
        return self._constant_value

class Optimizer(object):

    def __init__(self, module_config: dict, n_jobs: int):
        """
        Initialize the Optimizer class.

        Args:
            module_config (dict): Configuration dictionary containing model parameters and settings.
            n_jobs (int): Number of parallel jobs to run.
        """
        self.module_config = module_config
        self.n_jobs = n_jobs
        

        if self.module_config and all(isinstance(key, int) for key in self.module_config.keys()):
            print("Using parallel optimization for multiple subjects.")
            self.optimize_params_dict = defaultdict(dict)
            modules = defaultdict(dict)
            for iSub, config in self.module_config.items():
                for key, (mod_cls, mod_kwargs) in config.items():
                    try:
                        modules[iSub][key] = mod_cls(self, **mod_kwargs)
                    except Exception as e:
                        print(f"Error initializing module {key}: {e}")
                for key, mod in modules[iSub].items():
                    if hasattr(mod, 'optimize_params_dict'):
                        self.optimize_params_dict[iSub].update(mod.optimize_params_dict)
        else:
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
            self.module_config = ConstantDict(self.module_config)
            self.optimize_params_dict = ConstantDict(self.optimize_params_dict)

    def prepare_data(self, data_path: str = DEFAULT_DATA_PATH):
        """
        Prepare data for analysis.
        
        Args:
            data_path (str): The path to the data file.
        """
        self.learning_data = pd.read_csv(data_path)

    def optimize_params_with_subs_parallel(
        self,
        model_config: Dict,
        subjects: Optional[List[str]] = None,
        window_size: Union[int, Dict[int, int]] = 16,
        grid_repeat: Union[int, Dict[int, int]] = 5,
        mc_samples: Union[int, Dict[int, int]] = 100,
    ) -> Dict:
        """
        Optimize parameters using grid search across subjects in parallel.
        Args:
            model_config (Dict): The model configuration.
            subjects (Optional[List[str]]): List of subjects to optimize for.
            window_size (Union[int, Dict[int, int]]): Size of the sliding window.
            grid_repeat (Union[int, Dict[int, int]]): Number of repetitions for grid search.
            mc_samples (Union[int, Dict[int, int]]): Number of Monte Carlo samples.
        Returns:
            Dict: A dictionary containing the optimization results for each subject.
        """
        if subjects is None:
            subjects = self.learning_data["subject"].unique()
        subject_data_map = {
            iSub: self.learning_data[self.learning_data["iSub"] == iSub]
            for iSub in subjects
        }
        if isinstance(window_size, int):
            window_size = ConstantDict(window_size)
        if isinstance(grid_repeat, int):
            grid_repeat = ConstantDict(grid_repeat)
        if isinstance(mc_samples, int):
            mc_samples = ConstantDict(mc_samples)

        def process_single_task(iSub, subject_data, **kwargs):
            condition = subject_data["condition"].iloc[0]

            s_data = (subject_data[[
                "feature1", "feature2", "feature3", "feature4"
            ]].values, subject_data["choice"].values,
                      subject_data["feedback"].values,
                      subject_data["category"].values)

            model = StandardModel(model_config,
                                  module_config=self.module_config[iSub],
                                  condition=condition)

            grid_params = {}
            for key in self.optimize_params_dict[iSub].keys():
                grid_params[key] = kwargs[key]

            grid_step_results, grid_error = model.compute_error_for_params(
                s_data,
                window_size=window_size[iSub],
                repeat=grid_repeat[iSub],
                iSub=iSub,
                **grid_params)

            return iSub, grid_params, grid_error, grid_step_results

        all_kwargs = []
        for iSub in subjects:
            for values in product(*self.optimize_params_dict[iSub].values()):
                kwargs = dict(zip(self.optimize_params_dict[iSub].keys(), values))
                kwargs['iSub'] = iSub
                kwargs['subject_data'] = subject_data_map[iSub]
                all_kwargs.append(kwargs)

        results = Parallel(n_jobs=self.n_jobs, batch_size=1)(
            delayed(process_single_task)(**kwargs)
            for kwargs in tqdm(all_kwargs,
                               desc="Processing tasks",
                               total=len(all_kwargs),
                               ncols=100,
                               leave=True,
                               position=0))

        subject_grid_errors = defaultdict(dict)
        subject_best_combo = {}

        for iSub, grid_params, grid_error, grid_step_results in results:
            subject_grid_errors[iSub][tuple(grid_params.values())] = grid_error

            if iSub not in subject_best_combo:
                subject_best_combo[iSub] = {
                    "params": grid_params,
                    "error": grid_error,
                    "step_results": grid_step_results
                }
            else:
                best_error = np.mean(subject_best_combo[iSub]["error"])
                if np.mean(grid_error) < best_error:
                    subject_best_combo[iSub] = {
                        "params": grid_params,
                        "error": grid_error,
                        "step_results": grid_step_results
                    }

        def refit_model(iSub, subject_data, specific_params):
            condition = subject_data["condition"].iloc[0]
            s_data = (subject_data[[
                "feature1", "feature2", "feature3", "feature4"
            ]].values, subject_data["choice"].values,
                      subject_data["feedback"].values,
                      subject_data["category"].values)

            model = StandardModel(model_config,
                                  module_config=self.module_config[iSub],
                                  condition=condition)
            all_step_results, all_mean_error = model.compute_error_for_params(
                s_data,
                window_size=window_size[iSub],
                repeat=mc_samples[iSub],
                multiprocess=True,
                n_jobs=self.n_jobs,
                iSub=iSub,
                **specific_params)
            return iSub, all_step_results, all_mean_error

        fitting_results = {}
        for iSub in subject_grid_errors.keys():
            _, all_step_results, all_mean_error = refit_model(
                iSub, subject_data_map[iSub],
                subject_best_combo[iSub]["params"])
            idx = np.argmin(all_mean_error)

            fitting_results[iSub] = {
                "condition": subject_data_map[iSub]["condition"].iloc[0],
                "best_params": subject_best_combo[iSub]["params"],
                "best_error": all_mean_error[idx],
                "best_step_results": all_step_results[idx],
                # "raw_step_results": all_step_results,
                "grid_errors": subject_grid_errors[iSub],
                "sample_errors": all_mean_error
            }
        return fitting_results

    def save_results(self,
                     results: Dict,
                     name: str,
                     output_dir: str = os.getcwd()) -> None:
        """
        Save the optimization results to a file.

        Args:
            results (Dict): The optimization results to save.
            output_name (str): The name to save the results.
        """
        for iSub, subject_info in results.items():
            cache_path = os.path.join(output_dir, "cache", name, f"{iSub}.gz")
            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            raw_step_results = StreamList(cache_path, 0)
            raw_step_results.extend(subject_info['raw_step_results'])
            subject_info['raw_step_results'] = (cache_path,
                                                len(raw_step_results))
        output_path = os.path.join(output_dir, f'{name}.joblib')
        with open(output_path, 'wb') as f:
            joblib.dump(results, f)
        logger.info(f"Results saved to {output_path}")

    def load_results(self, input_path: str) -> Dict:
        """
        Load the optimization results from a file.

        Args:
            input_path (str): The path to the file containing the results.
        """
        with open(input_path, 'rb') as f:
            results = joblib.load(f)
        for iSub, subject_info in results.items():
            cache_path = subject_info['raw_step_results'][0]
            raw_step_results = StreamList(cache_path,
                                          subject_info['raw_step_results'][1])
            subject_info['raw_step_results'] = raw_step_results
        logger.info(f"Results loaded from {input_path}")
        return results

    def set_results(self, results: Dict) -> None:
        """
        Set the optimization results directly.

        Args:
            results (Dict): The optimization results to set.
        """
        self.fitting_results = results

    def predict_with_subs_parallel(self,
                                   model_config: Dict,
                                   subjects: Optional[List[str]] = None,
                                   **kwargs) -> Dict:
        """
        Predict the choice probabilities for each subject using the fitted model parameters.
        """

        window_size_arg = kwargs.get("window_size", 16)
        if subjects is None:
            subjects = self.learning_data["subject"].unique()

        predict_results = {}

        def process_single_task(iSub):
            subject_data = self.learning_data[self.learning_data["iSub"] ==
                                              iSub]
            condition = subject_data["condition"].iloc[0]
            s_data = (subject_data[[
                "feature1", "feature2", "feature3", "feature4"
            ]].values, subject_data["choice"].values,
                      subject_data["feedback"].values,
                      subject_data["category"].values)
            model = StandardModel(model_config,
                                  module_config=self.module_config[iSub],
                                  condition=condition)

            sub_results = self.fitting_results[iSub]
            step_results = sub_results.get(
                'step_results', sub_results.get('best_step_results'))
            
            if isinstance(window_size_arg, dict):
                ws = window_size_arg.get(iSub, 16)
            elif isinstance(window_size_arg, (list, tuple)):
                # 假设 subjects 顺序和 list 对应
                idx = subjects.index(iSub)
                ws = window_size_arg[idx]
            else:
                ws = window_size_arg 

            results = model.predict_choice(s_data,
                                           step_results,
                                           use_cached_dist=False,
                                           window_size=ws)
            predict_result = {
                'condition': condition,
                'true_acc': results['true_acc'],
                'pred_acc': results['pred_acc'],
                'sliding_true_acc': results['sliding_true_acc'],
                'sliding_pred_acc': results['sliding_pred_acc'],
                'sliding_pred_acc_std': results['sliding_pred_acc_std']
            }
            return iSub, predict_result

        results = Parallel(n_jobs=self.n_jobs, batch_size=1)(
            delayed(process_single_task)(iSub)
            for iSub in tqdm(subjects,
                             desc="Predicting tasks",
                             total=len(subjects),
                             ncols=100,
                             leave=True,
                             position=0))
        for iSub, predict_result in results:
            predict_results[iSub] = predict_result

        return predict_results
