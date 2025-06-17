"""
评估相关函数：
- 评估指标计算
- 模型验证
- 结果可视化
例如：
"""
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from tqdm import tqdm
from .plot_utils import init_figure, style_subject_ax, add_global_labels, COLOR_PALETTE
import pandas as pd


class ModelEval:

    # plot parameters over trials
    def plot_params_over_trials(self,
                                results: Dict,
                                params: str,
                                save_path: str = None):
        """
        Plots specified parameter over trials for each condition separately.

        Args:
            results (Dict): Dictionary containing results for each subject.
            params (str): The parameter to plot (e.g., 'k', 'beta').
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """
        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            n_rows = 2
            n_cols = 4

            # 初始化 fig, axs
            fig, axs = init_figure(n_rows, n_cols)

            global_max = max(
                len(subject_info['best_step_results'])
                for _, subject_info in subjects)

            for idx, (iSub, subject_info) in enumerate(subjects):
                step_results = subject_info['best_step_results']

                row, col = divmod(idx, n_cols)
                ax = axs[row, col]

                num_steps = len(step_results)
                param_values = [result[params] for result in step_results]

                ax.plot(range(1, num_steps + 1),
                        param_values,
                        marker='o',
                        linewidth=3)

                for x in range(64, num_steps + 1, 64):
                    ax.axvline(x=x,
                               color='grey',
                               alpha=0.3,
                               linestyle='dashed',
                               linewidth=1)

                # 调用风格化
                style_subject_ax(ax,
                                 row=row,
                                 col=col,
                                 n_rows=n_rows,
                                 n_cols=n_cols,
                                 num_steps=global_max,
                                 condition=condition,
                                 subject_idx=idx + 1,
                                 y_range=(0, 30),
                                 n_yticks=6,
                                 y_tick_format='int')

            plt.tight_layout(rect=[0.02, 0.05, 1, 1])  # 先排版并预留底边
            # 全局标签
            add_global_labels(fig, ylabel=f"Optimal beta value")

            if save_path:
                fig.savefig(f"{save_path}_{params}_condition_{condition}.png")
            plt.close()

    def plot_posterior_probabilities(self,
                                     results: Dict,
                                     limit: bool,
                                     save_path: str = None):
        """
        Plots posterior probabilities for each condition separately.

        Args:
            results (Dict): Dictionary containing results for each subject.
            limit (bool): Whether to limit the hypotheses.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """
        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            n_rows = 2
            n_cols = 4

            # 初始化 fig, axs
            fig, axs = init_figure(n_rows, n_cols)

            global_max = max(
                len(subject_info['best_step_results'])
                for _, subject_info in subjects)

            for idx, (iSub, subject_info) in enumerate(subjects):
                step_results = subject_info['best_step_results']

                row, col = divmod(idx, n_cols)
                ax = axs[row, col]

                num_steps = len(step_results)

                if limit:
                    max_k = 19 if condition == 1 else 116
                    k_posteriors = {k: [] for k in range(max_k)}

                    for step, result in enumerate(step_results):
                        for k in range(max_k):
                            if k in result['hypo_details']:
                                k_posteriors[k].append(
                                    (step + 1,
                                     result['hypo_details'][k]['post_max']))

                    for k in range(max_k):
                        if k_posteriors[k]:
                            steps, values = zip(*k_posteriors[k])
                            ax.scatter(steps,
                                       values,
                                       label=f'k={k}',
                                       alpha=0.5)
                else:
                    max_k = max(k for result in step_results
                                for k in result['hypo_details'].keys())
                    k_posteriors = {
                        k: np.zeros(num_steps)
                        for k in range(0, max_k)
                    }

                    for step, result in enumerate(step_results):
                        for k in range(0, max_k):
                            k_posteriors[k][step] = result['hypo_details'][k][
                                'post_max']

                for k in range(0, max_k):
                    if (condition == 1 and k == 0) or (condition != 1
                                                       and k == 42):
                        ax.plot(range(1, num_steps + 1),
                                k_posteriors[k],
                                linewidth=3,
                                color='red',
                                label=f'k={k}')
                    else:
                        ax.plot(range(1, num_steps + 1),
                                k_posteriors[k],
                                linewidth=2,
                                label=f'k={k}',
                                alpha=0.5)

                # 调用风格化
                style_subject_ax(ax,
                                 row=row,
                                 col=col,
                                 n_rows=n_rows,
                                 n_cols=n_cols,
                                 num_steps=global_max,
                                 condition=condition,
                                 subject_idx=idx + 1,
                                 y_range=(0, 1),
                                 n_yticks=6,
                                 y_tick_format='decimal')

            plt.tight_layout(rect=[0.02, 0.05, 1, 1])  # 先排版并预留底边
            # 全局标签
            add_global_labels(fig, ylabel=f"Posterior probability")

            if save_path:
                fig.savefig(f"{save_path}_post_condition_{condition}.png")
            plt.close()

    def plot_hypo_posterior_sums(self,
                                 base_results: Dict,
                                 forget_results: Dict,
                                 limited_hypos_list: Dict,
                                 save_path: str = None):
        """
        Plots the sum of posterior probabilities for hypotheses in limited_hypos_list over trials.

        Args:
            results (Dict): Dictionary containing results for each subject.
            limited_hypos_list (Dict): Dictionary containing hypothesis lists for each subject.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """
        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            n_rows = 2
            n_cols = 4

            # 初始化 fig, axs
            fig, axs = init_figure(n_rows, n_cols)

            global_max = max(
                len(subject_info['best_step_results'])
                for _, subject_info in subjects)

            for idx, (iSub, subject_info) in enumerate(subjects):
                step_results = subject_info['best_step_results']
                condition = subject_info['condition']
                subject_hypos = limited_hypos_list[iSub]

                row, col = divmod(idx, n_cols)
                ax = axs[row, col]

                num_steps = len(step_results)
                posterior_sums = []

                for step, result in enumerate(step_results):
                    # 获取当前试次的假设列表
                    current_hypos = subject_hypos[step]
                    # 计算当前试次中所有假设的后验概率之和
                    posterior_sum = sum(result['hypo_details'][k]['post_max']
                                        for k in current_hypos
                                        if k in result['hypo_details'])
                    posterior_sums.append(posterior_sum)

                # 计算滑动平均值
                window_size = 16
                series = pd.Series(np.log(posterior_sums))
                sliding_avg = series.rolling(window=window_size,
                                             min_periods=window_size).mean()

                # 计算 cumulative average
                cumulative_avg = [np.mean(posterior_sums[:i + 1]) for i in range(len(posterior_sums))]

                # 绘制 cumulative average 曲线
                ax.plot(range(1, num_steps + 1),
                        cumulative_avg,
                        label='Cumulative Avg',
                        linewidth=3)

                # ax.plot(range(1, num_steps + 1),
                #         sliding_avg,
                #         label='Posterior Sum',
                #         linewidth=3)

                # 调用风格化
                style_subject_ax(ax,
                                 row=row,
                                 col=col,
                                 n_rows=n_rows,
                                 n_cols=n_cols,
                                 num_steps=global_max,
                                 condition=condition,
                                 subject_idx=idx + 1,
                                 y_range=(0, 1),
                                 n_yticks=6,
                                 y_tick_format='decimal')

            plt.tight_layout(rect=[0.02, 0.05, 1, 1])  # 先排版并预留底边
            # 全局标签
            add_global_labels(fig, ylabel=f"Accumulated posterior probability")

            if save_path:
                fig.savefig(f"{save_path}_postsum_condition_{condition}.png")
            plt.close()

    def plot_accuracy_comparison(self, results: Dict, save_path: str = None):
        """
        Plots predicted and true accuracy for each subject.

        Args:
            results (Dict): Dictionary containing accuracy results for each subject.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """

        # 按 condition 分组
        grouped_results = defaultdict(list)
        for iSub, subject_info in results.items():
            condition = subject_info['condition']
            grouped_results[condition].append((iSub, subject_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            n_rows = 2
            n_cols = 4

            # 初始化 fig, axs
            fig, axs = init_figure(n_rows, n_cols)

            global_max = max(
                len(subject_info['sliding_true_acc'])
                for _, subject_info in subjects)

            for idx, (iSub, subject_info) in enumerate(subjects):
                row, col = divmod(idx, n_cols)
                ax = axs[row, col]

                sliding_true_acc = results[iSub]['sliding_true_acc']
                sliding_pred_acc = results[iSub]['sliding_pred_acc']
                sliding_pred_acc_std = results[iSub]['sliding_pred_acc_std']

                # 在每列数据的最前面添加15个空值
                sliding_true_acc = [np.nan] * 15 + sliding_true_acc
                sliding_pred_acc = [np.nan] * 15 + sliding_pred_acc
                sliding_pred_acc_std = [np.nan] * 15 + sliding_pred_acc_std

                num_steps = len(sliding_pred_acc)

                condition = results[iSub].get('condition', 'Unknown')

                ax.plot(range(1, num_steps + 1),
                        sliding_true_acc,
                        label='True',
                        color='orange',
                        linewidth=3)
                
                ax.plot(range(1, num_steps + 1),
                        sliding_pred_acc,
                        label='Predicted',
                        color='blue',
                        linewidth=3)

                lower_bound = np.array(sliding_pred_acc) - np.array(
                    sliding_pred_acc_std)
                upper_bound = np.array(sliding_pred_acc) + np.array(
                    sliding_pred_acc_std)
                ax.fill_between(range(1, num_steps + 1),
                                lower_bound,
                                upper_bound,
                                color='blue',
                                alpha=0.2,
                                label='Predicted Std')

                # 调用风格化
                style_subject_ax(ax,
                                 row=row,
                                 col=col,
                                 n_rows=n_rows,
                                 n_cols=n_cols,
                                 num_steps=global_max+15,
                                 condition=condition,
                                 subject_idx=idx + 1,
                                 y_range=(0, 1),
                                 n_yticks=6,
                                 y_tick_format='decimal')

            plt.tight_layout(rect=[0.02, 0.05, 1, 1])  # 先排版并预留底边
            # 全局标签
            add_global_labels(
                fig,
                ylabel=f"Accuracy",
            )
            # 添加图例
            handles, labels = ax.get_legend_handles_labels()
            if condition == 3:
                fig.legend(handles, labels, loc=(0.87,0.2), fontsize=20)
            else:
                fig.legend(handles, labels, loc='upper right', fontsize=20)

            if save_path:
                fig.savefig(f"{save_path}_acc_condition_{condition}.png")
            plt.close()




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

        fig = plt.figure(figsize=(8 * n_cols, 5 * n_rows))
        fig.suptitle('Error Grids by Subject', fontsize=16, y=0.99)

        sorted_subjects = sorted(results.keys())

        for idx, iSub in enumerate(sorted_subjects):
            subject_info = results[iSub]
            # optimize_results = subject_info['optimize_results']
            grid_errors = subject_info['grid_errors']
            condition = subject_info['condition']

            row = idx % n_rows
            col = idx // n_rows
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)

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
            ax.set_xticklabels([f'{w0:.4f}' for w0 in w0_values], rotation=90)
            ax.set_yticklabels([f'{gamma:.2f}' for gamma in gamma_values])

            ax.set_title(f'Subject {iSub} (Condition {condition})')
            ax.set_xlabel('w0')
            ax.set_ylabel('Gamma')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
