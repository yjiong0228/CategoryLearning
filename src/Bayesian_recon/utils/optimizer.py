import os
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, List, Dict
from itertools import product
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..problems import (
    StandardModel
)

PROJECT_ROOT_PATH = Path(os.getcwd()).parent.parent.parent.parent.parent
DEFAULT_DATA_PATH = Path(PROJECT_ROOT_PATH, "data", "processed", "Task2_processed.csv")


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

    def optimize_params_with_subs_parallel(self,
                                           model_config: Dict,
                                           subjects: Optional[List[str]] = None,
                                           window_size: int = 16,
                                           grid_repeat: int = 5,
                                           mc_samples: int = 100,                                            
                                           ) -> Dict:
        """
        Optimize parameters using grid search across subjects in parallel.
        Args:
            model_config (Dict): The model configuration.
            subjects (Optional[List[str]]): List of subjects to optimize for.
            window_size (int): Size of the sliding window.
            grid_repeat (int): Number of repetitions for grid search.
            mc_samples (int): Number of Monte Carlo samples.
        Returns:
            Dict: A dictionary containing the optimization results for each subject.
        """
        if subjects is None:
            subjects = self.learning_data["subject"].unique()
        subject_data_map = {
            iSub: self.learning_data[self.learning_data["iSub"] == iSub]
            for iSub in subjects
        }

        def process_single_task(iSub, subject_data, **kwargs):
            condition = subject_data["condition"].iloc[0]

            s_data = (subject_data[[
                "feature1", "feature2", "feature3", "feature4"
            ]].values, subject_data["choice"].values,
                      subject_data["feedback"].values,
                      subject_data["category"].values)

            model = StandardModel(model_config,
                                  module_config=self.module_config,
                                  condition=condition)
            grid_params = {}
            for key in self.optimize_params_dict.keys():
                grid_params[key] = kwargs[key]
            all_step_results, all_mean_error = model.compute_error_for_params(
                s_data,
                window_size=window_size, 
                repeat=grid_repeat,
                **grid_params)
            return iSub, grid_params, all_mean_error, all_step_results

        all_kwargs = []
        for task_values in product(subjects,
                                   *self.optimize_params_dict.values()):
            iSub = task_values[0]
            grid_values = task_values[1:]
            kwargs = dict(zip(self.optimize_params_dict.keys(), grid_values))
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

        for iSub, grid_params, all_mean_error, all_step_results in results:
            subject_grid_errors[iSub][tuple(grid_params.values())] = all_mean_error

            if iSub not in subject_best_combo:
                subject_best_combo[iSub] = {
                    "params": grid_params,
                    "error": all_mean_error,
                    "step_results": all_step_results
                }
            else:
                best_error = np.mean(subject_best_combo[iSub]["error"])
                if np.mean(all_mean_error) < best_error:
                    subject_best_combo[iSub] = {
                        "params": grid_params,
                        "error": all_mean_error,
                        "step_results": all_step_results
                    }

        def refit_model(iSub, subject_data, specific_params):
            condition = subject_data["condition"].iloc[0]
            s_data = (subject_data[[
                "feature1", "feature2", "feature3", "feature4"
            ]].values, subject_data["choice"].values,
                      subject_data["feedback"].values,
                      subject_data["category"].values)

            model = StandardModel(model_config,
                                  module_config=self.module_config,
                                  condition=condition)
            all_step_results, all_mean_error = model.compute_error_for_params(
                s_data,
                window_size=window_size,
                repeat=mc_samples,
                multiprocess=True,
                n_jobs=self.n_jobs,
                **specific_params)
            return iSub, all_step_results, all_mean_error

        fitting_results = {}
        for iSub in subject_grid_errors.keys():
            _, all_step_results, all_mean_error = refit_model(iSub, subject_data_map[iSub],
                                        subject_best_combo[iSub]["params"])
            idx = np.argmin(all_mean_error)
            fitting_results[iSub] = {
                "condition": subject_data_map[iSub]["condition"].iloc[0],
                "best_params": subject_best_combo[iSub]["params"],
                "best_error": all_mean_error[idx],
                "best_step_results": all_step_results[idx],
                "raw_step_results": all_step_results,
                "grid_errors": subject_grid_errors[iSub]
            }
        return fitting_results

    def optimize_params_with_mcmc(self,
                                  model_config: Dict,
                                  gamma_values,
                                  w0_values,
                                  subjects: Optional[List[str]] = None,
                                  mc_samples: int = 1000,
                                  **kwargs):
        """
        Optimize parameters using MCMC across subjects and samples in parallel.
        - raw_step_results: list of all sample step_results (length mc_samples)
        - step_results: aggregated results per step with averaged fields
        Returns:
            Dict of {iSub: {
                condition, gamma, w0,
                raw_step_results: List[List[step_dict]],
                step_results: List[step_dict]
            }}
        """
        # 1. Prepare subjects
        if subjects is None:
            subjects = list(self.learning_data["iSub"].unique())

        # 2. Normalize gamma and w0
        if not isinstance(gamma_values, dict):
            gamma_map = {
                subj: gamma_values[idx]
                for idx, subj in enumerate(subjects)
            }
        else:
            gamma_map = gamma_values
        if not isinstance(w0_values, dict):
            w0_map = {
                subj: w0_values[idx]
                for idx, subj in enumerate(subjects)
            }
        else:
            w0_map = w0_values

        # 3. Data per subject
        subject_data_map = {
            iSub: self.learning_data[self.learning_data["iSub"] == iSub]
            for iSub in subjects
        }

        # 4. Flatten tasks
        tasks = [(iSub, idx) for iSub in subjects for idx in range(mc_samples)]

        # 5. Worker
        def run_task(iSub, _):
            data = subject_data_map[iSub]
            cond = data["condition"].iloc[0]
            stim = data[["feature1", "feature2", "feature3",
                         "feature4"]].values
            choices = data["choice"].values
            feedback = data["feedback"].values
            s_data = (stim, choices, feedback)

            model = StandardModel(model_config,
                                  module_config=self.module_config,
                                  condition=cond)

            task_kwargs = dict(kwargs)  # shallow copy
            task_kwargs["gamma"] = gamma_map[iSub]
            task_kwargs["w0"] = w0_map[iSub]

            # fit_step_by_step 会从 **task_kwargs 里 pick up gamma/w0
            step_res = model.fit_step_by_step(s_data, **task_kwargs)
            return iSub, step_res

        # 6. Parallel execution
        results = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(run_task)(iSub, idx)
            for iSub, idx in tqdm(tasks, desc="MCMC tasks", ncols=80))

        # 7. Group raw samples
        samples_by_sub = defaultdict(list)
        for iSub, step_res in results:
            samples_by_sub[iSub].append(step_res)

        # 8. Aggregate and assemble output
        fitting_results = {}
        for iSub, sample_list in samples_by_sub.items():
            data = subject_data_map[iSub]
            cond = data["condition"].iloc[0]
            gamma = gamma_map[iSub]
            w0 = w0_map[iSub]

            raw_step_results = sample_list
            n_steps = len(sample_list[0])
            step_results = []
            for step_idx in range(n_steps):
                # Collect post_max values per hypothesis
                post_vals = defaultdict(list)
                for sr in sample_list:
                    for h, det in sr[step_idx]["hypo_details"].items():
                        post_vals[h].append(det["post_max"])

                # Build hypo_details with averaged post_max
                hypo_details = {}
                for h, vals in post_vals.items():
                    # find a representative det0
                    det0 = None
                    for sr in sample_list:
                        det0 = sr[step_idx]["hypo_details"].get(h)
                        if det0 is not None:
                            break
                    hypo_details[h] = {
                        "post_max": float(np.sum(vals)/mc_samples),
                        "beta_opt": det0["beta_opt"],
                        "ll_max": det0["ll_max"]
                    }

                # Compute best_step_amount safely
                amounts = []
                for sr in sample_list:
                    bsa = sr[step_idx].get("best_step_amount")
                    if isinstance(bsa, (int, float)):
                        amounts.append(bsa)

                # Determine best hypothesis by averaged post_max
                best_h = max(hypo_details,
                             key=lambda x: hypo_details[x]["post_max"])
                entry = {
                    "best_k": best_h,
                    "best_beta": hypo_details[best_h]["beta_opt"],
                    "best_log_likelihood": hypo_details[best_h]["ll_max"],
                    "best_norm_posterior": hypo_details[best_h]["post_max"],
                    "hypo_details": hypo_details
                }
                if amounts:
                    entry["best_step_amount"] = sum(amounts) / len(amounts)

                step_results.append(entry)

            fitting_results[iSub] = {
                "condition": cond,
                "gamma": gamma,
                "w0": w0,
                # "raw_step_results": raw_step_results,
                "step_results": step_results
            }

        return fitting_results

    def save_results(self, results: Dict, output_path: str):
        """
        Save the optimization results to a file.

        Args:
            results (Dict): The optimization results to save.
            output_path (str): The path to save the results.
        """
        with open(output_path, 'wb') as f:
            joblib.dump(results, f)
        logger.info(f"Results saved to {output_path}")

    def load_results(self, input_path: str) -> None:
        """
        Load the optimization results from a file.

        Args:
            input_path (str): The path to the file containing the results.
        """
        with open(input_path, 'rb') as f:
            results = joblib.load(f)
        logger.info(f"Results loaded from {input_path}")
        self.set_results(results)

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

        window_size = kwargs.get("window_size", 16)
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
                                  module_config=self.module_config,
                                  condition=condition)

            sub_results = self.fitting_results[iSub]
            step_results = sub_results.get(
                'step_results', sub_results.get('best_step_results'))
            results = model.predict_choice(s_data,
                                           step_results,
                                           use_cached_dist=False,
                                           window_size=window_size)
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

    def plot_posterior_probabilities(self,
                                     results: Dict,
                                     subjects: Optional[List[str]] = None,
                                     save_path: str = None,
                                     **kwargs) -> None:
        if subjects is not None:
            results = {
                iSub: results[iSub]
                for iSub in subjects if iSub in results
            }

        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        # 确定行列数
        n_conditions = len(grouped_results)
        max_subjects_per_condition = max(
            len(subjects) for subjects in grouped_results.values())
        n_cols = kwargs.get("n_cols", max_subjects_per_condition)
        n_rows = n_conditions

        g = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        g.suptitle('Posterior Probabilities for k by Subject',
                   fontsize=kwargs.get("fontsize", 16),
                   y=kwargs.get("y", 0.99))
        limit = kwargs.get("limit", True)

        # 按 condition 绘制子图
        for row_idx, (condition,
                      subjects) in enumerate(sorted(grouped_results.items())):
            for col_idx, (iSub, subject_info) in enumerate(subjects):
                step_results = subject_info.get(
                    'step_results', subject_info.get('best_step_results'))

                plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)

                if limit:
                    max_k = 19 if condition == 1 else 116
                    data = []
                    for step, result in enumerate(step_results):
                        for k in range(max_k):
                            if k in result['hypo_details']:
                                data.append({
                                    'Step':
                                    step + 1,
                                    'k':
                                    k,
                                    'Posterior':
                                    result['hypo_details'][k]['post_max']
                                })
                    df = pd.DataFrame(data)

                else:
                    max_k = max(k for result in step_results
                                for k in result['hypo_details'].keys())
                    data = []
                    for step, result in enumerate(step_results):
                        for k in range(max_k):
                            if k in result['hypo_details']:
                                data.append({
                                    'Step':
                                    step + 1,
                                    'k':
                                    k,
                                    'Posterior':
                                    result['hypo_details'][k]['post_max']
                                })
                    df = pd.DataFrame(data)

                sns.scatterplot(data=df,
                                x='Step',
                                y='Posterior',
                                hue='k',
                                palette='tab10',
                                alpha=0.5,
                                legend=False)

                highlight_k = 0 if condition == 1 else 42
                highlight_data = df[df['k'] == highlight_k]
                sns.scatterplot(
                    data=highlight_data,
                    x='Step',
                    y='Posterior',
                    color='red',
                    s=50,
                )
                plt.title(f'Subject {iSub} (Condition {condition})')
                plt.xlabel('Trail')
                plt.ylabel('Posterior Probability')

        plt.tight_layout()
        if save_path:
            g.savefig(save_path)
            logger.info(f"Posterior probabilities saved to {save_path}")

    def plot_accuracy_comparison(self,
                                 results: Dict,
                                 subjects: Optional[List[str]] = None,
                                 save_path: str = None,
                                 **kwargs) -> None:
        if subjects is not None:
            results = {
                iSub: results[iSub]
                for iSub in subjects if iSub in results
            }

        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        # 确定行列数
        n_conditions = len(grouped_results)
        max_subjects_per_condition = max(
            len(subjects) for subjects in grouped_results.values())
        n_cols = kwargs.get("n_cols", max_subjects_per_condition)
        n_rows = n_conditions

        g = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        g.suptitle('Predicted vs True Accuracy by Subject',
                   fontsize=kwargs.get("fontsize", 16),
                   y=kwargs.get("y", 0.99))

        # 按 condition 绘制子图
        for row_idx, (condition,
                      subjects) in enumerate(sorted(grouped_results.items())):
            for col_idx, (iSub, subject_info) in enumerate(subjects):
                sliding_true_acc = results[iSub]['sliding_true_acc']
                sliding_pred_acc = results[iSub]['sliding_pred_acc']
                sliding_pred_acc_std = results[iSub]['sliding_pred_acc_std']
                condition = results[iSub]['condition']

                data = pd.DataFrame({
                    'Trial':
                    range(len(sliding_pred_acc)),
                    'Predicted Accuracy':
                    sliding_pred_acc,
                    'True Accuracy':
                    sliding_true_acc,
                    'Lower Bound':
                    np.array(sliding_pred_acc) -
                    np.array(sliding_pred_acc_std),
                    'Upper Bound':
                    np.array(sliding_pred_acc) + np.array(sliding_pred_acc_std)
                })

                plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)
                sns.lineplot(data=data,
                             x='Trial',
                             y='Predicted Accuracy',
                             label='Predicted Accuracy',
                             color='blue')
                sns.lineplot(data=data,
                             x='Trial',
                             y='True Accuracy',
                             label='True Accuracy',
                             color='orange')
                plt.fill_between(data['Trial'],
                                 data['Lower Bound'],
                                 data['Upper Bound'],
                                 color='blue',
                                 alpha=0.2,
                                 label='Predicted Accuracy ± Std')
                plt.ylim(0, 1)
                plt.title(f'Subject {iSub} (Condition {condition})')
                plt.xlabel('Trial')
                plt.ylabel('Accuracy')
                plt.legend()

        plt.tight_layout()
        if save_path:
            g.savefig(save_path)
            logger.info(f"Accuracy comparison saved to {save_path}")

    def plot_error_grids(self,
                         results: Dict,
                         subjects: Optional[List[str]] = None,
                         field_names: Optional[List[str]] = None,
                         save_path: str = None,
                         **kwargs) -> None:
        if subjects is not None:
            results = {
                iSub: results[iSub]
                for iSub in subjects if iSub in results
            }

        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        # 确定行列数
        n_conditions = len(grouped_results)
        max_subjects_per_condition = max(
            len(subjects) for subjects in grouped_results.values())
        n_cols = kwargs.get("n_cols", max_subjects_per_condition)
        n_rows = n_conditions

        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle('Grid Search Error by Subject',
                     fontsize=kwargs.get("fontsize", 16),
                     y=kwargs.get("y", 0.99))

        # 按 condition 绘制子图
        for row_idx, (condition,
                      subjects) in enumerate(sorted(grouped_results.items())):
            for col_idx, (iSub, subject_info) in enumerate(subjects):

                grid_errors = results[iSub]['grid_errors']
                condition = results[iSub]['condition']
                if field_names is None:
                    field_names = list(self.optimize_params_dict.keys())[:2]

                data = []
                for params, error in grid_errors.items():
                    param_dict = dict(zip(field_names, params))
                    param_dict['Error'] = error
                    data.append(param_dict)
                df = pd.DataFrame(data)

                error_matrix = df.pivot_table(index=field_names[0],
                                              columns=field_names[1],
                                              values='Error')
                ax = fig.add_subplot(n_rows, n_cols,
                                     row_idx * n_cols + col_idx + 1)
                sns.heatmap(
                    error_matrix,
                    #annot=True,
                    #fmt=".2f",
                    cmap='viridis',
                    cbar_kws={'label': 'Error'})
                ax.set_title(f'Subject {iSub} (Condition {condition})')
                ax.set_xlabel(field_names[1])
                ax.set_ylabel(field_names[0])
                ax.xaxis.set_major_formatter(
                    mticker.FormatStrFormatter('%.4f'))
                ax.yaxis.set_major_formatter(
                    mticker.FormatStrFormatter('%.2f'))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Error grids saved to {save_path}")

    def plot_cluster_amount(self,
                            results: Dict,
                            subjects: Optional[List[str]] = None,
                            save_path: str = None,
                            **kwargs) -> None:
        if subjects is not None:
            results = {
                iSub: results[iSub]
                for iSub in subjects if iSub in results
            }

        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        # 确定行列数
        n_conditions = len(grouped_results)
        max_subjects_per_condition = max(
            len(subjects) for subjects in grouped_results.values())
        n_cols = kwargs.get("n_cols", max_subjects_per_condition)
        n_rows = n_conditions

        g = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        g.suptitle('Posterior Probabilities for k by Subject',
                   fontsize=kwargs.get("fontsize", 16),
                   y=kwargs.get("y", 0.99))
        limit = kwargs.get("limit", True)

        # 按 condition 绘制子图
        for row_idx, (condition,
                      subjects) in enumerate(sorted(grouped_results.items())):
            for col_idx, (iSub, subject_info) in enumerate(subjects):
                step_results = subject_info['best_step_results']

                plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)

                if limit:
                    max_k = 19 if condition == 1 else 116
                    data = []
                    for step, result in enumerate(step_results):
                        for k in range(max_k):
                            if k in result['hypo_details']:
                                data.append({
                                    'Step':
                                    step + 1,
                                    'k':
                                    k,
                                    'Posterior':
                                    result['hypo_details'][k]['post_max']
                                })
                    df = pd.DataFrame(data)

                else:
                    max_k = max(k for result in step_results
                                for k in result['hypo_details'].keys())
                    data = []
                    for step, result in enumerate(step_results):
                        for k in range(max_k):
                            if k in result['hypo_details']:
                                data.append({
                                    'Step':
                                    step + 1,
                                    'k':
                                    k,
                                    'Posterior':
                                    result['hypo_details'][k]['post_max']
                                })
                    df = pd.DataFrame(data)

                sns.scatterplot(data=df,
                                x='Step',
                                y='Posterior',
                                hue='k',
                                palette='tab10',
                                alpha=0.5,
                                legend=False)

                highlight_k = 0 if condition == 1 else 42
                highlight_data = df[df['k'] == highlight_k]
                sns.scatterplot(
                    data=highlight_data,
                    x='Step',
                    y='Posterior',
                    color='red',
                    s=50,
                )
                plt.title(f'Subject {iSub} (Condition {condition})')
                plt.xlabel('Trail')
                plt.ylabel('Posterior Probability')

        plt.tight_layout()
        if save_path:
            g.savefig(save_path)
            logger.info(f"Posterior probabilities saved to {save_path}")
