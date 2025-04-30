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
from typing import Dict, Optional
from tqdm import tqdm
from .plot_utils import init_figure, style_subject_ax, add_global_labels, COLOR_PALETTE
import pandas as pd


class ModelEval:

    def plot_oral_hypo_list(self,
                            results,
                            save_path: str = None,
                            iSub: Optional[int] = None):

        if iSub is not None:
            if iSub not in results:
                raise ValueError(f"iSub {iSub} not found in results.")
            # 只保留这一个人，仍用 defaultdict 便于后续按 condition 分组
            subject_info = results[iSub]
            grouped_results = defaultdict(list)
            grouped_results[subject_info["condition"]].append(
                (iSub, subject_info))
        else:
            # 原逻辑：按 condition 分组
            grouped_results = defaultdict(list)
            for _iSub, subj_info in results.items():
                grouped_results[subj_info["condition"]].append(
                    (_iSub, subj_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            if n_subjects == 1:
                n_rows, n_cols = 1, 1
                fig, axs = plt.subplots(n_rows,
                                        n_cols,
                                        figsize=(8, 5),
                                        facecolor='none',
                                        squeeze=False)
            else:
                n_rows, n_cols = 2, 4
                fig, axs = init_figure(n_rows, n_cols)

            # --- 统一成 2‑D ndarray，避免多重嵌套 / 单轴对象 ---
            axs = np.asarray(axs)  # 单轴 -> 0‑D / 1‑D ndarray
            if axs.ndim == 0:  # 只有一个 Axes 的情况
                axs = axs.reshape(1, 1)
            elif axs.ndim == 1:  # 1‑D 列表式 (n,) -> (1, n)
                axs = axs.reshape(1, -1)
            # 现在 axs 一定是 shape == (n_rows, n_cols)

            global_max = max(
                len(subject_info['hit']) for _, subject_info in subjects)

            for idx, (iSub, subject_info) in enumerate(subjects):
                step_hit = subject_info['hit']

                row, col = divmod(idx, n_cols)
                ax = axs[row, col]

                num_steps = len(step_hit)

                ax.plot(range(1, num_steps + 1), step_hit, linewidth=3)

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
                                 y_range=(0, 1),
                                 n_yticks=6,
                                 y_tick_format='decimal')
            if iSub is not None:
                plt.tight_layout(rect=[0.04, 0.06, 1, 1])  # 先排版并预留底边
            else:
                plt.tight_layout(rect=[0.02, 0.05, 1, 1])

            # 全局标签
            add_global_labels(fig, ylabel=f"Probability")

            if save_path:
                if iSub is not None:
                    fig.savefig(f"{save_path}_oral_iSub_{iSub}.png")
                else:
                    fig.savefig(f"{save_path}_oral_condition_{condition}.png")
            plt.close()

    # plot parameters over trials
    def plot_params_over_trials(self,
                                results: Dict,
                                params: str,
                                save_path: str = None,
                                iSub: Optional[int] = None):
        """
        Plots specified parameter over trials for each condition separately.

        Args:
            results (Dict): Dictionary containing results for each subject.
            params (str): The parameter to plot (e.g., 'k', 'beta').
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """
        if iSub is not None:
            if iSub not in results:
                raise ValueError(f"iSub {iSub} not found in results.")
            # 只保留这一个人，仍用 defaultdict 便于后续按 condition 分组
            subject_info = results[iSub]
            grouped_results = defaultdict(list)
            grouped_results[subject_info["condition"]].append(
                (iSub, subject_info))
        else:
            # 原逻辑：按 condition 分组
            grouped_results = defaultdict(list)
            for _iSub, subj_info in results.items():
                grouped_results[subj_info["condition"]].append(
                    (_iSub, subj_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            if n_subjects == 1:
                n_rows, n_cols = 1, 1
                fig, axs = plt.subplots(n_rows,
                                        n_cols,
                                        figsize=(8, 5),
                                        facecolor='none',
                                        squeeze=False)
            else:
                n_rows, n_cols = 2, 4
                fig, axs = init_figure(n_rows, n_cols)

            # --- 统一成 2‑D ndarray，避免多重嵌套 / 单轴对象 ---
            axs = np.asarray(axs)  # 单轴 -> 0‑D / 1‑D ndarray
            if axs.ndim == 0:  # 只有一个 Axes 的情况
                axs = axs.reshape(1, 1)
            elif axs.ndim == 1:  # 1‑D 列表式 (n,) -> (1, n)
                axs = axs.reshape(1, -1)
            # 现在 axs 一定是 shape == (n_rows, n_cols)

            global_max = max(
                len(subject_info['step_results'])
                for _, subject_info in subjects)

            for idx, (iSub, subject_info) in enumerate(subjects):
                step_results = subject_info['step_results']

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
            if iSub is not None:
                plt.tight_layout(rect=[0.04, 0.06, 1, 1])  # 先排版并预留底边
            else:
                plt.tight_layout(rect=[0.02, 0.05, 1, 1])

            # 全局标签
            add_global_labels(fig, ylabel=f"Optimal beta value")

            if save_path:
                if iSub is not None:
                    fig.savefig(f"{save_path}_{params}_iSub_{iSub}.png")
                else:
                    fig.savefig(
                        f"{save_path}_{params}_condition_{condition}.png")
            plt.close()

    def plot_posterior_probabilities(self,
                                     results: Dict,
                                     limit: bool,
                                     save_path: str = None,
                                     iSub: Optional[int] = None):
        """
        Plots posterior probabilities for each condition separately.

        Args:
            results (Dict): Dictionary containing results for each subject.
            limit (bool): Whether to limit the hypotheses.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """
        if iSub is not None:
            if iSub not in results:
                raise ValueError(f"iSub {iSub} not found in results.")
            # 只保留这一个人，仍用 defaultdict 便于后续按 condition 分组
            subject_info = results[iSub]
            grouped_results = defaultdict(list)
            grouped_results[subject_info["condition"]].append(
                (iSub, subject_info))
        else:
            # 原逻辑：按 condition 分组
            grouped_results = defaultdict(list)
            for _iSub, subj_info in results.items():
                grouped_results[subj_info["condition"]].append(
                    (_iSub, subj_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            if n_subjects == 1:
                n_rows, n_cols = 1, 1
                fig, axs = plt.subplots(n_rows,
                                        n_cols,
                                        figsize=(8, 5),
                                        facecolor='none',
                                        squeeze=False)
            else:
                n_rows, n_cols = 2, 4
                fig, axs = init_figure(n_rows, n_cols)

            # --- 统一成 2‑D ndarray，避免多重嵌套 / 单轴对象 ---
            axs = np.asarray(axs)  # 单轴 -> 0‑D / 1‑D ndarray
            if axs.ndim == 0:  # 只有一个 Axes 的情况
                axs = axs.reshape(1, 1)
            elif axs.ndim == 1:  # 1‑D 列表式 (n,) -> (1, n)
                axs = axs.reshape(1, -1)
            # 现在 axs 一定是 shape == (n_rows, n_cols)

            global_max = max(
                len(subject_info['step_results'])
                for _, subject_info in subjects)

            for idx, (iSub, subject_info) in enumerate(subjects):
                step_results = subject_info['step_results']

                row, col = divmod(idx, n_cols)
                ax = axs[row, col]

                num_steps = len(step_results)

                if limit:
                    special_k = 0 if condition == 1 else 42
                    max_k     = 19 if condition == 1 else 116
                    win       = 16                   # 滚动窗口长度

                    # ---- 1. 收集原始 posterior ----
                    steps, red_vals, gray_vals = [], [], []

                    for step_idx, result in enumerate(step_results, start=1):
                        red_val  = result['hypo_details'].get(special_k, {}).get('post_max', 0.)
                        gray_val = sum(v['post_max'] for k, v in result['hypo_details'].items()
                                    if k != special_k and k < max_k)

                        steps.append(step_idx)
                        red_vals.append(red_val)
                        gray_vals.append(gray_val)

                    steps_arr     = np.asarray(steps)
                    red_series    = pd.Series(red_vals,  dtype=float)
                    gray_series   = pd.Series(gray_vals, dtype=float)

                    # ---- 2. rolling(mean)；前 15 个值为 NaN ----
                    red_ma   = red_series.rolling(window=win, min_periods=win).mean().to_numpy()
                    gray_ma  = gray_series.rolling(window=win, min_periods=win).mean().to_numpy()

                    # ---- 3. 绘图（NaN 自动被跳过）----
                    ax.plot(steps_arr, gray_ma, linewidth=2, linestyle='dashed',
                            color='grey', alpha=0.5,
                            label=f'others (MA{win})')
                    ax.plot(steps_arr, red_ma,  linewidth=3, color='red',
                            label=f'k={special_k} (MA{win})')

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

            if iSub is not None:
                plt.tight_layout(rect=[0.04, 0.06, 1, 1])  # 先排版并预留底边
            else:
                plt.tight_layout(rect=[0.02, 0.05, 1, 1])

            # 全局标签
            add_global_labels(fig, ylabel=f"Posterior probability")

            if save_path:
                if iSub is not None:
                    fig.savefig(f"{save_path}_post_iSub_{iSub}.png")
                else:
                    fig.savefig(f"{save_path}_post_condition_{condition}.png")
            plt.close()

    def plot_hypo_posterior_sums(self,
                                 base_results: Dict,
                                 forget_results: Dict,
                                 cluster_results: Dict,
                                 limited_hypos_list: Dict,
                                 save_path: str = None):
        """
        Plots the sum of posterior probabilities for hypotheses in limited_hypos_list over trials.

        Args:
            results (Dict): Dictionary containing results for each subject.
            limited_hypos_list (Dict): Dictionary containing hypothesis lists for each subject.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """

        def _collect_cumavg(subject_info, subject_hypos):
            """返回某个 subject 的 cumulative‐avg 序列"""
            step_results = subject_info.get(
                    'step_results', subject_info.get('best_step_results'))
            posterior_sums = [
                sum(r['hypo_details'][k]['post_max']  # 与原代码一致
                    for k in subject_hypos[step] if k in r['hypo_details'])
                for step, r in enumerate(step_results)
            ]
            log_posterior_sums = np.log(posterior_sums)  # 取 log
            cum_avg = np.cumsum(log_posterior_sums) / (
                np.arange(len(log_posterior_sums)) + 1)  # = cumulative average
            # 取 exp
            return np.exp(cum_avg)  # 返回 cumulative average 的 exp 值


        merged = defaultdict(
            list)  # {condition: [(iSub, base_info, forget_info), …]}

        for iSub, base_info in base_results.items():
            if iSub not in forget_results:  # 只画两份结果都齐全的 subject
                continue
            condition = base_info['condition']
            merged[condition].append(
                (iSub, base_info, forget_results[iSub], cluster_results[iSub]))

        for condition, triples in merged.items():
            n_rows = 2
            n_cols = 4

            # 初始化 fig, axs
            fig, axs = init_figure(n_rows, n_cols)

            global_max = max(
                len(t[1].get('step_results', t[1].get('best_step_results', [])))  # 取 base_info 的 trial 数
                for t in triples)

            for idx, (iSub, base_info, forget_info,
                      cluster_info) in enumerate(triples):
                row, col = divmod(idx, n_cols)
                ax = axs[row, col]

                cum_base = _collect_cumavg(base_info, limited_hypos_list[iSub])
                cum_forget = _collect_cumavg(forget_info,
                                             limited_hypos_list[iSub])
                cum_cluster = _collect_cumavg(cluster_info,
                                              limited_hypos_list[iSub])

                steps = np.arange(1, len(cum_base) + 1)

                ax.plot(steps, cum_base, lw=3, color='tab:blue', label='Base')
                ax.plot(steps,
                        cum_forget,
                        lw=3,
                        color='tab:green',
                        label='Forget')
                ax.plot(steps,
                        cum_cluster,
                        lw=3,
                        color='orange',
                        label='Limit')

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
            add_global_labels(fig, ylabel=f"Cumulative posterior probability")

            # 添加图例
            handles, labels = ax.get_legend_handles_labels()
            if condition == 3:
                fig.legend(handles, labels, loc=(0.87, 0.2), fontsize=20)
            else:
                fig.legend(handles, labels, loc='upper right', fontsize=20)

            if save_path:
                fig.savefig(f"{save_path}_postsum_condition_{condition}.png")
            plt.close()

    def plot_accuracy_comparison(self,
                                 results: Dict,
                                 save_path: str = None,
                                 iSub: Optional[int] = None):
        """
        Plots predicted and true accuracy for each subject.

        Args:
            results (Dict): Dictionary containing accuracy results for each subject.
            save_path (str): Path to save the plot. If None, the plot will be shown.
        """

        # 按 condition 分组
        if iSub is not None:
            if iSub not in results:
                raise ValueError(f"iSub {iSub} not found in results.")
            # 只保留这一个人，仍用 defaultdict 便于后续按 condition 分组
            subject_info = results[iSub]
            grouped_results = defaultdict(list)
            grouped_results[subject_info["condition"]].append(
                (iSub, subject_info))
        else:
            # 原逻辑：按 condition 分组
            grouped_results = defaultdict(list)
            for _iSub, subj_info in results.items():
                grouped_results[subj_info["condition"]].append(
                    (_iSub, subj_info))

        for condition, subjects in grouped_results.items():
            n_subjects = len(subjects)
            if n_subjects == 1:
                n_rows, n_cols = 1, 1
                fig, axs = plt.subplots(n_rows,
                                        n_cols,
                                        figsize=(8, 5),
                                        facecolor='none',
                                        squeeze=False)
            else:
                n_rows, n_cols = 2, 4
                fig, axs = init_figure(n_rows, n_cols)

            # --- 统一成 2‑D ndarray，避免多重嵌套 / 单轴对象 ---
            axs = np.asarray(axs)  # 单轴 -> 0‑D / 1‑D ndarray
            if axs.ndim == 0:  # 只有一个 Axes 的情况
                axs = axs.reshape(1, 1)
            elif axs.ndim == 1:  # 1‑D 列表式 (n,) -> (1, n)
                axs = axs.reshape(1, -1)
            # 现在 axs 一定是 shape == (n_rows, n_cols)

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
                                 num_steps=global_max + 15,
                                 condition=condition,
                                 subject_idx=idx + 1,
                                 y_range=(0, 1),
                                 n_yticks=6,
                                 y_tick_format='decimal')

            if iSub is not None:
                plt.tight_layout(rect=[0.04, 0.06, 1, 1])  # 先排版并预留底边
            else:
                plt.tight_layout(rect=[0.02, 0.05, 1, 1])

            # 全局标签
            add_global_labels(
                fig,
                ylabel=f"Accuracy",
            )

            # 添加图例
            if iSub is None:
                handles, labels = ax.get_legend_handles_labels()
                if condition == 3:
                    fig.legend(handles, labels, loc=(0.87, 0.2), fontsize=20)
                else:
                    fig.legend(handles, labels, loc='upper right', fontsize=20)

            if save_path:
                if iSub is not None:
                    fig.savefig(f"{save_path}_acc_iSub_{iSub}.png")
                else:
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
