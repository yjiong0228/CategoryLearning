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
import pandas as pd
from typing import Optional, List, Dict
from collections import defaultdict
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import logging
logger = logging.getLogger(__name__)


class ModelEval:
    @staticmethod
    def _filter_results(results, subjects):
        if subjects is not None:
            return {iSub: results[iSub] for iSub in subjects if iSub in results}
        return results

    def _plot_by_condition(self, results, subjects, save_path, title, plot_body, **kwargs):
        # Filter and group
        results = self._filter_results(results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info['condition']].append((iSub, info))

        # Layout
        n_rows = len(grouped)
        n_cols = kwargs.get('n_cols', max(len(lst) for lst in grouped.values()))
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(title, fontsize=kwargs.get('fontsize', 16), y=kwargs.get('y', 0.99))

        # Subplots
        for row, (condition, subs) in enumerate(sorted(grouped.items())):
            for col, (iSub, info) in enumerate(subs):
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                plot_body(ax, condition, iSub, info)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            logger.info(f"{title} saved to {save_path}")

    def plot_posterior_probabilities(self, results, subjects=None, save_path=None, limit=True, **kwargs):
        def body(ax, condition, iSub, info):
            step_results = info.get('step_results', info.get('best_step_results', []))
            max_k = (19 if condition == 1 else 116) if limit else \
                    max(k for sr in step_results for k in sr['hypo_details'])
            data = []
            for step, res in enumerate(step_results):
                for k in range(max_k):
                    if k in res['hypo_details']:
                        data.append({'Step': step + 1, 'k': k, 'Posterior': res['hypo_details'][k]['post_max']})
            df = pd.DataFrame(data)
            sns.scatterplot(data=df, x='Step', y='Posterior', hue='k', palette='tab10', alpha=0.5, legend=False, ax=ax)
            hk = 0 if condition == 1 else 42
            sns.scatterplot(data=df[df['k'] == hk], x='Step', y='Posterior', color='red', s=50, ax=ax)
            ax.set(title=f'Subject {iSub} (Condition {condition})', xlabel='Trial', ylabel='Posterior Probability')

        self._plot_by_condition(results, subjects, save_path,
                                'Posterior Probabilities for k by Subject', body, **kwargs)

    def plot_k_oral_comparison(self, model_results, oral_results, subjects=None, save_path=None, window_size=16, **kwargs):
        def extract_ma(step_results, k_special, win):
            posts = []
            for sr in step_results:
                p = sr['hypo_details'].get(k_special, {}).get('post_max', 0.0)
                try:
                    p = float(p)
                except:
                    p = 0.0
                posts.append(p)
            return pd.Series(posts).rolling(window=win, min_periods=win).mean().to_numpy()

        # Filter both dicts
        model_res = self._filter_results(model_results, subjects)
        oral_res = self._filter_results(oral_results, subjects)

        # Group by condition from true_res
        grouped = defaultdict(list)
        for iSub, info in model_res.items():
            grouped[info['condition']].append(iSub)

        n_rows = len(grouped)
        n_cols = kwargs.get('n_cols', max(len(lst) for lst in grouped.values()))
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle('True k vs Oral K by Subject', fontsize=kwargs.get('fontsize', 16), y=kwargs.get('y', 0.99))

        for row, (condition, subs) in enumerate(sorted(grouped.items())):
            for col, iSub in enumerate(subs):
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                # Model k
                info = model_res[iSub]
                sr = info.get('step_results', info.get('best_step_results', []))
                ks = 0 if condition == 1 else 42
                ma = extract_ma(sr, ks, window_size)
                x_ma = np.arange(1, len(ma) + 1)
                ax.plot(x_ma[:len(ma)], ma, lw=2, label='Model k')
                # Oral K
                oral_hits = oral_res[iSub]['rolling_hits']
                x_oral = np.arange(1, len(oral_hits) + 1)
                ax.plot(x_oral, oral_hits, lw=2, label='Oral K')
                ax.set_ylim(0, 1)
                ax.set(title=f'Subject {iSub} (Condition {condition})', xlabel='Trial', ylabel='Value')
                ax.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            logger.info(f"Model k vs Oral K comparison saved to {save_path}")
            
    def plot_accuracy_comparison(self, results, subjects=None, save_path=None, **kwargs):
        def body(ax, condition, iSub, info):
            t = info['sliding_true_acc']
            p = info['sliding_pred_acc']
            std = info['sliding_pred_acc_std']
            df = pd.DataFrame({'Trial': range(len(p)), 'Pred': p, 'True': t,
                               'Low': np.array(p) - std, 'High': np.array(p) + std})
            sns.lineplot(data=df, x='Trial', y='Pred', label='Predicted', ax=ax)
            sns.lineplot(data=df, x='Trial', y='True', label='True', ax=ax)
            ax.fill_between(df['Trial'], df['Low'], df['High'], alpha=0.2)
            ax.set_ylim(0, 1)
            ax.set(title=f'Subject {iSub} (Condition {condition})', xlabel='Trial', ylabel='Accuracy')
            ax.legend()

        self._plot_by_condition(results, subjects, save_path,
                                'Predicted vs True Accuracy by Subject', body, **kwargs)

    def plot_error_grids(self, results, subjects=None, field_names=None, save_path=None, **kwargs):
        def body(ax, condition, iSub, info):
            fname = field_names or list(self.optimize_params_dict.keys())[:2]
            data = [{'Param1': params[0], 'Param2': params[1], 'Error': err}
                    for params, err in info['grid_errors'].items()]
            df = pd.DataFrame(data)
            em = df.pivot_table(index='Param1', columns='Param2', values='Error')
            sns.heatmap(em, cbar_kws={'label': 'Error'}, ax=ax)
            ax.set(title=f'Subject {iSub} (Condition {condition})', xlabel=fname[1], ylabel=fname[0])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        self._plot_by_condition(results, subjects, save_path,
                                'Grid Search Error by Subject', body, **kwargs)

    def plot_cluster_amount(self, results, window_size=16, subjects=None, save_path=None, **kwargs):
        def body(ax, condition, iSub, info):
            steps = info.get('best_step_results', [])
            vals = [s['best_step_amount']['random_posterior'][0] for s in steps if 'best_step_amount' in s]
            r = [s['best_step_amount']['random'][0] for s in steps if 'best_step_amount' in s]
            re = pd.Series(vals).rolling(window=window_size, min_periods=window_size).mean()
            ex = pd.Series(r).rolling(window=window_size, min_periods=window_size).mean()
            x = np.arange(1, len(vals) + 1)
            ax.plot(x, re, label='Exploitation', lw=2)
            ax.plot(x, ex, label='Exploration', lw=2)
            ax.set(title=f'Subject {iSub} (Condition {condition})', xlabel='Trial', ylabel='Amount')
            ax.legend()

        self._plot_by_condition(results, subjects, save_path,
                                'Strategy Amount by Subject', body, **kwargs)
