"""
评估相关函数：
- 评估指标计算
- 模型验证
- 结果可视化
例如：
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

class ModelEval:
    # plot parameters over trials
    def plot_params_over_trials(self, results: Dict, params: str, save_path: str = None):
        """
        Plots specified parameter over trials for a given subject.

        Args:
            step_results (list of dict): List of dictionaries containing parameter values for each trial.
            params (str): The parameter to plot (e.g., 'k', 'beta').
        """

        n_subjects = len(results)
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows

        fig = plt.figure(figsize=(8*n_cols, 5*n_rows))
        fig.suptitle(f'{params}_over_trials by Subject', fontsize=16, y=0.99)

        sorted_subjects = sorted(results.keys())

        for idx, iSub in enumerate(sorted_subjects):
            subject_info = results[iSub]
            step_results = subject_info['step_results']
            condition = subject_info['condition']
            
            row = idx % n_rows
            col = idx // n_rows
            ax = fig.add_subplot(n_rows, n_cols, row*n_cols + col + 1)
            
            num_steps = len(step_results)
            param_values = [result[params] for result in step_results]

            ax.plot(range(1, num_steps + 1), param_values, marker='o')

            ax.set_title(f'Subject {iSub} (Condition {condition})')
            ax.set_xlabel('Trial')
            ax.set_ylabel(f'{params} value')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_posterior_probabilities(self, results: Dict, limit: bool, save_path: str = None):
        n_subjects = len(results)
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows
        
        fig = plt.figure(figsize=(8*n_cols, 5*n_rows))
        fig.suptitle('Posterior Probabilities for k by Subject', fontsize=16, y=0.99)
        
        sorted_subjects = sorted(results.keys())
        
        for idx, iSub in enumerate(sorted_subjects):
            subject_info = results[iSub]
            step_results = subject_info['step_results']
            condition = subject_info['condition']
            
            row = idx % n_rows
            col = idx // n_rows
            ax = fig.add_subplot(n_rows, n_cols, row*n_cols + col + 1)
            
            num_steps = len(step_results)

            if limit:
                max_k = 19 if condition == 1 else 115
                k_posteriors = {k: [] for k in range(max_k)}

                for step, result in enumerate(step_results):
                    for k in range(max_k):
                        if k in result['hypo_details']:
                            k_posteriors[k].append((step + 1, result['hypo_details'][k]['post_max']))

                for k in range(max_k):
                    if k_posteriors[k]:
                        steps, values = zip(*k_posteriors[k])
                        if (condition == 1 and k == 0) or (condition != 1 and k == 42):
                            ax.scatter(steps, values, color='red', s=50, label=f'k={k}')
                        else:
                            ax.scatter(steps, values, label=f'k={k}', alpha=0.5)                                
            else:
                max_k = max(k for result in step_results for k in result['hypo_details'].keys())
            
                k_posteriors = {k: np.zeros(num_steps) for k in range(0, max_k)}
                for step, result in enumerate(step_results):
                    for k in range(0, max_k):
                        k_posteriors[k][step] = result['hypo_details'][k]['post_max']
            
                for k in range(0, max_k):
                    if (condition == 1 and k == 0) or (condition != 1 and k == 42):
                        ax.plot(range(1, num_steps + 1), k_posteriors[k], 
                            linewidth=3, color='red', label=f'k={k}')
                    else:
                        ax.plot(range(1, num_steps + 1), k_posteriors[k], 
                            label=f'k={k}', alpha=0.5)

            ax.set_title(f'Subject {iSub} (Condition {condition})')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Posterior Probability')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_accuracy_comparison(self, results: Dict, save_path: str = None):
        """
        Plots predicted and true accuracy for each subject.

        Args:
            results (Dict): Dictionary containing accuracy results for each subject.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """
        n_subjects = len(results)
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows))
        fig.suptitle('Predicted vs True Accuracy by Subject', fontsize=16, y=0.99)

        sorted_subjects = sorted(results.keys())

        for idx, iSub in enumerate(sorted_subjects):
            row = idx % n_rows
            col = idx // n_rows
            ax = axes[row, col]
            
            sliding_true_acc = results[iSub]['sliding_true_acc']
            sliding_pred_acc = results[iSub]['sliding_pred_acc']
            sliding_pred_acc_std = results[iSub]['sliding_pred_acc_std']

            condition = results[iSub].get('condition', 'Unknown')

            ax.plot(sliding_pred_acc, label='Predicted Accuracy', color='blue')
            lower_bound = np.array(sliding_pred_acc) - np.array(sliding_pred_acc_std)
            upper_bound = np.array(sliding_pred_acc) + np.array(sliding_pred_acc_std)
            ax.fill_between(range(len(sliding_pred_acc)), lower_bound, upper_bound, color='blue', alpha=0.2, label='Predicted Accuracy ± Std')
            ax.plot(sliding_true_acc, label='True Accuracy', color='orange')

            ax.set_ylim(0, 1)
            ax.set_title(f'Subject {iSub} (Condition {condition})')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Accuracy')
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()