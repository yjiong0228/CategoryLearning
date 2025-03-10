"""
评估相关函数：
- 评估指标计算
- 模型验证
- 结果可视化
例如：
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

class ModelEval:
    # plot parameters over trials
    def plot_params_over_trials(self, results: List, params: str, save_path: str = None):
        """
        Plots specified parameter over trials for each subject.

        Args:
            results (list): List of subject data, each containing 'condition' and 'best_step_results'.
                            We will add 'iSub' (subject ID) based on list index.
            params (str): The parameter to plot (e.g., 'k', 'beta').
        """
        # Dynamically add 'iSub' to each subject based on their index in the list
        for idx, subject_info in enumerate(results):
            subject_info["iSub"] = idx  # Add iSub as the list index
        
        # Sort by the dynamically added 'iSub'
        sorted_results = sorted(results, key=lambda x: x['iSub'])
        n_subjects = len(sorted_results)
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows

        fig = plt.figure(figsize=(8*n_cols, 5*n_rows))
        fig.suptitle(f'{params}_over_trials by Subject', fontsize=16, y=0.99)

        for idx, subject_info in enumerate(sorted_results):
            iSub = subject_info['iSub']
            condition = subject_info['condition']
            step_results = subject_info['best_step_results']
            
            row = idx % n_rows
            col = idx // n_rows
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            
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

    def plot_posterior_probabilities(self, results: List, save_path: str = None):
        # Dynamically add 'iSub' to each subject based on their index in the list
        for idx, subject_info in enumerate(results):
            subject_info["iSub"] = idx  # Add iSub as the list index
        
        # Sort by the dynamically added 'iSub'
        sorted_results = sorted(results, key=lambda x: x['iSub'])
        n_subjects = len(sorted_results)
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows

        fig = plt.figure(figsize=(8*n_cols, 5*n_rows))
        fig.suptitle('Posterior Probabilities for k by Subject', fontsize=16, y=0.99)

        for idx, subject_info in enumerate(sorted_results):
            iSub = subject_info['iSub']
            condition = subject_info['condition']
            step_results = subject_info['best_step_results']
            
            row = idx % n_rows
            col = idx // n_rows
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            
            num_steps = len(step_results)
            max_k = max(k for result in step_results for k in result['hypo_details'].keys())
            
            k_posteriors = {k: np.zeros(num_steps) for k in range(max_k + 1)}
            for step, result in enumerate(step_results):
                for k in range(max_k + 1):
                    k_posteriors[k][step] = result['hypo_details'].get(k, {'post_max': 0})['post_max']
            
            for k in range(max_k + 1):
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

    def calculate_predictions(self, model, data, step_results):
        """Calculate predictions based on fitted model results"""
        predictions = []
        
        for step in range(len(step_results)-1):
            next_trial = data.iloc[step + 1]
            x = next_trial[['feature1', 'feature2', 'feature3', 'feature4']].values
            fitted_params = step_results[step]['params']
            
            predicted_choice = model.predict_choice(fitted_params, x, data['condition'].iloc[0])
            actual_choice = next_trial['choice']
            
            predictions.append({
                'trial': step + 1,
                'predicted_choice': predicted_choice,
                'actual_choice': actual_choice,
                'correct': predicted_choice == actual_choice
            })
        
        return predictions

    def calculate_sliding_accuracy(self, predictions, window_size=32):
        """Calculate sliding window accuracy from predictions"""
        sliding_accuracy = []
        num_predictions = len(predictions)
        
        for step in range(window_size - 1, num_predictions):
            window_start = step - window_size + 1
            window_predictions = predictions[window_start:step+1]
            correct_in_window = sum(p['correct'] for p in window_predictions)
            sliding_accuracy.append(correct_in_window / window_size)
        
        return sliding_accuracy
    
    def plot_predictive_accuracy(self, results: Dict, save_path: str = None):
        # Plot cumulative accuracy for each subject, grouped by condition
        plt.figure(figsize=(12, 8))

        # Define colors and labels for each condition
        condition_colors = {1: 'blue', 2: 'red', 3: 'green'}
        condition_labels = {1: 'Family', 2: 'Species', 3: 'Both'}

        # Plot separately for each condition
        for condition in [1, 2, 3]:
            for iSub, result in results.items():
                if result['condition'] == condition:
                    plt.plot(range(2, len(result['sliding_accuracy']) + 2), 
                            result['sliding_accuracy'], 
                            color=condition_colors[condition], 
                            alpha=0.3)

        # Add one line per condition for the legend
        for condition in [1, 2, 3]:
            plt.plot([], [], 
                    color=condition_colors[condition], 
                    label=condition_labels[condition])

        plt.xlabel('Mini-block')
        plt.ylabel('Sliding Prediction Accuracy')
        plt.title('Prediction Accuracy Over Trials')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()