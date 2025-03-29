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
            # optimize_results = subject_info['optimize_results']
            step_results = subject_info['best_step_results']
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

    def plot_posterior_probabilities(self, results: Dict, save_path: str = None):
        n_subjects = len(results)
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows
        
        fig = plt.figure(figsize=(8*n_cols, 5*n_rows))
        fig.suptitle('Posterior Probabilities for k by Subject', fontsize=16, y=0.99)
        
        sorted_subjects = sorted(results.keys())
        
        for idx, iSub in enumerate(sorted_subjects):
            subject_info = results[iSub]
            # optimize_results = subject_info['optimize_results']
            step_results = subject_info['best_step_results']
            condition = subject_info['condition']
            
            row = idx % n_rows
            col = idx // n_rows
            ax = fig.add_subplot(n_rows, n_cols, row*n_cols + col + 1)
            
            num_steps = len(step_results)
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


    def plot_error_grids(self, results: Dict, save_path: str = None):
        """
        Plots error grids for each subject based on optimize_results.

        Args:
            results (Dict): Dictionary containing optimization results for each subject.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """
        n_subjects = len(results)
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows

        fig = plt.figure(figsize=(8*n_cols, 5*n_rows))
        fig.suptitle('Error Grids by Subject', fontsize=16, y=0.99)

        sorted_subjects = sorted(results.keys())

        for idx, iSub in enumerate(sorted_subjects):
            subject_info = results[iSub]
            # optimize_results = subject_info['optimize_results']
            grid_errors = subject_info['grid_errors']
            condition = subject_info['condition']

            row = idx % n_rows
            col = idx // n_rows
            ax = fig.add_subplot(n_rows, n_cols, row*n_cols + col + 1)

            # 提取 gamma 和 w0 的取值
            gamma_values = sorted(set(key[0] for key in grid_errors.keys()))
            w0_values = sorted(set(key[1] for key in grid_errors.keys()))

            # 创建 error 矩阵
            error_matrix = np.zeros((len(gamma_values), len(w0_values)))
            for (gamma, w0), error in grid_errors.items():
                gamma_idx = gamma_values.index(gamma)
                w0_idx = w0_values.index(w0)
                error_matrix[gamma_idx, w0_idx] = error

            # 创建 grid plot
            cax = ax.matshow(error_matrix, cmap='viridis_r')
            fig.colorbar(cax, ax=ax)

            ax.set_xticks(range(len(w0_values)))
            ax.set_yticks(range(len(gamma_values)))
            ax.set_xticklabels([f'{w0:.2f}' for w0 in w0_values], rotation=90)
            ax.set_yticklabels([f'{gamma:.1f}' for gamma in gamma_values])

            ax.set_title(f'Subject {iSub} (Condition {condition})')
            ax.set_xlabel('w0')
            ax.set_ylabel('Gamma')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()



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