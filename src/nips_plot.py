import os
import numpy as np
from scipy.interpolate import splprep, splev
from math import isfinite
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple, List, Any
import pandas as pd
from .plot_utils import (create_grid_figure,add_segmentation_lines,style_axis,annotate_label)


# 1. 注册本地字体（把路径换成你机器上 Arial.ttf 的实际路径）
font_path = '/home/yangjiong/CategoryLearning/src/Arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# 2. 在绘图前设置默认字体
mpl.rcParams['font.family'] = prop.get_name()


class Fig1_Oral:

    def read_data(self, df):
        """
        读取 DataFrame，只保留 feature1_oral 到 feature4_oral，并保留 iSub、iSession、iTrial、choice。
        """
        required = [
            'iSub', 'iSession', 'iTrial', 'choice', 'feature1_oral',
            'feature2_oral', 'feature3_oral', 'feature4_oral'
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")
        return df[required]

    def get_model_centers(self, step_results, model, ncats):

        all_centers = model.partition_model.get_centers()

        model_fitting = [[step['best_k'], all_centers[step['best_k']][1]]
                         for step in step_results]

        columns = [
            f'choice_{choice}_feature_{feature}'
            for choice in range(1, ncats + 1) for feature in range(1, 5)
        ]

        rows = []
        for entry in model_fitting:
            k, center_dict = entry
            row = []
            for choice_key in range(ncats):  # 键 0 到 3
                features = center_dict.get(choice_key, (None, ) * 4)
                row.extend(features)
            rows.append(row)

        df = pd.DataFrame(rows, columns=columns)
        return df

    def draw_cube(self, ax, edge_color, lw):
        verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1),
                 (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        edges = [(1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 4), (1, 5),
                 (2, 6), (3, 7)]
        for u, v in edges:
            x, y, z = zip(verts[u], verts[v])
            ax.plot(x, y, z, color=edge_color, linewidth=lw, alpha=0.9)

    def _beautify_axes(self, ax):
        ax.set_axis_off()

        # 三条主轴线
        lines = [([0, 1], [0, 0], [0, 0]), ([0, 0], [0, 1], [0, 0]),
                 ([0, 0], [0, 0], [0, 1])]
        for x, y, z in lines:
            ax.plot(x, y, z, color='#808080', lw=2.5)
        # 轴标题和刻度
        self._label_axes(ax)
        self._add_ticks(ax)

    def _label_axes(self, ax):
        labels = [((1.1, -0.05, 0), 'F$_2$'), ((0, 1.12, 0.0), 'F$_1$'),
                  ((0, 0, 1.12), 'F$_3$')]
        for pos, txt in labels:
            ax.text(*pos,
                    txt,
                    fontsize=18,
                    ha='center',
                    va='center',
                    fontfamily='Arial')

    def _add_ticks(self, ax):
        ticks = [((0, 0, -0.05), '0', 'right', 'top'),
                 ((0.45, 0, -0.05), '0.5', 'center', 'top'),
                 ((0.9, 0, -0.05), '1', 'center', 'top'),
                 ((0, 0.48, -0.05), '0.5', 'right', 'center'),
                 ((0, 0.96, -0.05), '1', 'right', 'center'),
                 ((0, -0.03, 0.45), '0.5', 'right', 'bottom'),
                 ((0, -0.03, 0.9), '1', 'right', 'bottom')]
        for pos, txt, ha, va in ticks:
            ax.text(*pos, txt, fontsize=15, ha=ha, va=va, fontfamily='Arial')

    def _plot_smooth_grad_line_and_points(self,
                                          ax,
                                          xs,
                                          ys,
                                          zs,
                                          cmap_name="Blues",
                                          cmap_low=0.25,
                                          cmap_high=1.0,
                                          lw=1.5,
                                          s=30,
                                          n_interp=300,
                                          smooth=0.0):
        xs, ys, zs = map(np.asarray, (xs, ys, zs))

        # ---------- 1. 清洗无效点 ----------
        mask_valid = np.array([all(map(isfinite, p)) for p in zip(xs, ys, zs)])
        xs, ys, zs = xs[mask_valid], ys[mask_valid], zs[mask_valid]
        if len(xs) == 0:
            return

        # ---------- 2. 去掉连续重复 ----------
        # 保留第一个点 & 所有与前一个不同的点
        keep = [0] + [
            i for i in range(1, len(xs))
            if (xs[i], ys[i], zs[i]) != (xs[i - 1], ys[i - 1], zs[i - 1])
        ]
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        n_pts = len(xs)

        # ---------- 3. 小于 3 点：用直线/单点 ----------
        cmap = cm.get_cmap(cmap_name)
        if n_pts == 1:
            ax.scatter(xs, ys, zs, color=cmap(cmap_high), s=s, edgecolors='w')
            return
        if n_pts == 2:
            c1, c2 = cmap(cmap_low), cmap(cmap_high)
            ax.plot(xs, ys, zs, color=c1, linewidth=lw)
            ax.scatter(xs, ys, zs, color=[c1, c2], s=s, edgecolors='w')
            return
        else:
            self._plot_spline_or_fallback(ax, xs, ys, zs, cmap, cmap_low,
                                          cmap_high, lw, s)

    def _plot_spline_or_fallback(self, ax, xs, ys, zs, cmap, cmap_low,
                                 cmap_high, lw, s):
        try:
            k = min(3, len(xs) - 1)
            tck, u = splprep([xs, ys, zs], s=0.0, k=k)
            u_fine = np.linspace(0, 1, 300)
            x_f, y_f, z_f = splev(u_fine, tck)
            colors = cmap(np.linspace(cmap_low, cmap_high, len(x_f) - 1))
            for i in range(len(x_f) - 1):
                ax.plot(x_f[i:i + 2],
                        y_f[i:i + 2],
                        z_f[i:i + 2],
                        color=colors[i],
                        linewidth=lw)
            pt_colors = cmap(cmap_low + (cmap_high - cmap_low) * u)
            ax.scatter(xs, ys, zs, color=pt_colors, s=s, edgecolors='w')
        except Exception:
            self._plot_fallback_lines(ax, xs, ys, zs, cmap, cmap_low,
                                      cmap_high, lw, s)

    def _plot_fallback_lines(self, ax, xs, ys, zs, cmap, cmap_low, cmap_high,
                             lw, s):
        n = len(xs)
        colors = cmap(np.linspace(cmap_low, cmap_high, n))
        for i in range(n - 1):
            ax.plot(xs[i:i + 2],
                    ys[i:i + 2],
                    zs[i:i + 2],
                    color=colors[i],
                    linewidth=lw)
        ax.scatter(xs, ys, zs, color=colors, s=s, edgecolors='w')

    def plot_feature123(self, ncats, iSub, iSession, iTrial, choice,
                        features_list, plots_dir):
        folder = os.path.join(plots_dir)
        os.makedirs(folder, exist_ok=True)
        fig = plt.figure(figsize=(6, 6), facecolor='none')
        ax = fig.add_subplot(111, projection='3d')

        # 黄色点背景
        self._draw_radial_highlight(ax, ncats, choice)

        # 主线
        xs = [f['feature2_oral'] for f in features_list]
        ys = [f['feature1_oral'] for f in features_list]
        zs = [f['feature3_oral'] for f in features_list]

        if len(xs) > 1:
            self._plot_smooth_grad_line_and_points(ax, xs, ys, zs)

        ax.view_init(elev=15, azim=30)
        self._beautify_axes(ax)
        self.draw_cube(ax, '#808080', 1.5)
        ax.patch.set_alpha(0)

        fname = f"{iSub}_{iSession}_{iTrial}_c{choice}.png"
        plt.savefig(os.path.join(folder, fname), transparent=True, dpi=300)
        plt.close()

    def _draw_radial_highlight(self, ax, ncats, choice):
        coords = {
            2: {
                1: (0.5, 0.25, 0.5),
                2: (0.5, 0.75, 0.5)
            },
            4: {
                1: (0.25, 0.25, 0.5),
                2: (0.25, 0.75, 0.5),
                3: (0.75, 0.5, 0.25),
                4: (0.75, 0.5, 0.75)
            }
        }
        point = coords.get(ncats, {}).get(choice)
        if not point: return
        n_layers, outer_size, inner_size = 20, 6000, 80
        outer_alpha, inner_alpha = 0.05, 0.1
        sizes = np.linspace(outer_size, inner_size, n_layers)
        alphas = np.linspace(outer_alpha, inner_alpha, n_layers)
        for s, a in zip(sizes, alphas):
            ax.scatter(*point,
                       color='yellow',
                       s=s,
                       alpha=a,
                       edgecolors='none',
                       depthshade=False)

    def plot_feature4_time_series(self,
                                  iSub,
                                  iSession,
                                  iTrial,
                                  choice,
                                  features_list,
                                  plots_dir,
                                  cmap_name="Blues",
                                  cmap_low=0.35,
                                  cmap_high=1.0,
                                  lw=2,
                                  s=30,
                                  axis_lw=2,
                                  tick_lw=1.5):
        """
        Plot the trajectory of feature4_oral over trials: x-axis is trial number, y-axis is feature4_oral, with optional gradient coloring.
        """
        # Ensure output folder exists for this choice
        folder = os.path.join(plots_dir)
        os.makedirs(folder, exist_ok=True)

        # Prepare figure and axis
        fig, ax = plt.subplots(figsize=(3.5, 4), facecolor='none')

        # Extract trial numbers and feature4 values
        trials = np.array(
            [f.get('iTrial', idx + 1) for idx, f in enumerate(features_list)],
            dtype=float)
        vals = np.array([f['feature4_oral'] for f in features_list],
                        dtype=float)

        # Mask invalid points
        mask = np.isfinite(trials) & np.isfinite(vals)
        trials, vals = trials[mask], vals[mask]
        n_pts = len(trials)

        # Plot data
        if n_pts == 0:
            return
        elif n_pts == 1:
            ax.scatter(trials, vals, color='tab:blue', s=s, edgecolors='w')
        else:
            # Gradient line segments
            points = np.stack((trials, vals), axis=1).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = cm.get_cmap(cmap_name)
            norm = plt.Normalize(cmap_low, cmap_high)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw)
            lc.set_array(np.linspace(cmap_low, cmap_high, len(segments)))
            ax.add_collection(lc)
            ax.scatter(trials,
                       vals,
                       c=np.linspace(cmap_low, cmap_high, n_pts),
                       cmap=cmap,
                       s=s,
                       edgecolors='w')

        # Set labels
        ax.set_xlabel('Trial', fontfamily='Arial', fontsize=21)
        # ax.set_ylabel('Feature 4', fontfamily='Arial', fontsize=18)

        # Add text to the top-right corner
        ax.text(0.95,
                0.95,
                'F$_4$',
                transform=ax.transAxes,
                fontsize=21,
                ha='right',
                va='top',
                fontfamily='Arial')

        # Set ticks: y at [0,0.5,1], x only last at 190
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'], fontsize=18)
        ax.set_xticks([30])
        ax.set_xticklabels(['30'], fontsize=18)

        # Set limits
        ax.set_xlim(trials.min() - 0.5, trials.max() + 0.5)
        ax.set_ylim(0, 1)

        for spine in ax.spines.values():
            spine.set_linewidth(axis_lw)
        ax.tick_params(width=tick_lw, length=6)

        # Grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Adjust layout to prevent clipping
        fig.tight_layout()
        plt.subplots_adjust(left=0.2)

        # Save and close
        fname = f"{iSub}_{iSession}_{iTrial}_c{choice}_f4_ts.png"
        plt.savefig(os.path.join(folder, fname), transparent=True, dpi=300)
        plt.close()

    def plot_human_trajactory(self,
                              ncats,
                              subject_data,
                              type,
                              plots_dir,
                              row_indices=None):
        """处理 DataFrame 并仅对指定行绘图，展示行之前的累积轨迹"""
        df = self.read_data(subject_data)
        # 确定要绘制的行
        if row_indices is None:
            target = set(df.index)
        elif isinstance(row_indices, int):
            target = {row_indices}
        else:
            target = set(row_indices)
        # 创建输出文件夹
        os.makedirs(plots_dir, exist_ok=True)
        for c in range(1, ncats + 1):
            os.makedirs(os.path.join(plots_dir, f"choice{c}"), exist_ok=True)

        # 累积 human 特征，并在目标行时绘图
        last_feats = {c: [] for c in range(1, ncats + 1)}
        for idx, row in df.iterrows():
            choice = int(row['choice'])
            feat = {
                f'feature{i}_oral': row[f'feature{i}_oral']
                for i in range(1, 5)
            }
            last_feats[choice].append(feat)
            if idx in target:
                folder = os.path.join(plots_dir, f"choice{c}")
                if type == 1:
                    self.plot_feature123(ncats, row.get('iSub', 'Unknown'),
                                         row['iSession'], row['iTrial'],
                                         choice, last_feats[choice], folder)
                else:
                    self.plot_feature4_time_series(row.get('iSub', 'Unknown'),
                                                   row['iSession'],
                                                   row['iTrial'], choice,
                                                   last_feats[choice], folder)

        print(f"完成：仅绘制行 {sorted(target)}，图表已保存至 {plots_dir}/choice*/ 文件夹。")

    def plot_model_trajactory(self,
                              ncats,
                              df,
                              type,
                              plots_dir,
                              row_indices=None):
        # 确定要绘制的行
        if row_indices is None:
            target = set(df.index)
        elif isinstance(row_indices, int):
            target = {row_indices}
        else:
            target = set(row_indices)
        # 创建输出文件夹
        os.makedirs(plots_dir, exist_ok=True)
        for c in range(1, ncats + 1):
            os.makedirs(os.path.join(plots_dir, f"Model_choice{c}"),
                        exist_ok=True)

        # 累积 human 特征，并在目标行时绘图
        last_feats = {c: [] for c in range(1, ncats + 1)}
        for idx, row in df.iterrows():
            for c in range(1, ncats + 1):
                feat = {
                    f'feature{i}_oral': row[f'choice_{c}_feature_{i}']
                    for i in range(1, 5)
                }
                last_feats[c].append(feat)

            # 在目标行输出图
            if idx in target:
                iSub = row['iSub']
                iTrial = row['iTrial']

                for c in range(1, ncats + 1):
                    model_folder = os.path.join(plots_dir, f"Model_choice{c}")
                    if type == 1:
                        self.plot_feature123(
                            ncats,
                            iSub,
                            None,  # 没有 session
                            iTrial,
                            c,
                            last_feats[c],
                            model_folder)
                    else:
                        self.plot_feature4_time_series(iSub, None, iTrial, c,
                                                       last_feats[c],
                                                       model_folder)

        print(f"完成模型轨迹绘制：行 {sorted(target)}，图表已保存至 {plots_dir}/choice*/ 文件夹。")


class Fig1_Ntrial:

    def plot_trial_number(self,
                          learning_data: pd.DataFrame,
                          figsize=(15, 5),
                          color_1='#45B53F',
                          color_2='#DDAA33',
                          color_3='#A6A6A6',
                          label_1='Cond 1',
                          label_2='Cond 2',
                          label_3='Cond 3',
                          save_path: str = None):

        counts = (learning_data.groupby(
            ['condition', 'iSub']).size().reset_index(name='total_trials'))

        cond_order = sorted(counts['condition'].unique())

        pivot = counts.pivot(index='iSub',
                             columns='condition',
                             values='total_trials')

        summary = pd.DataFrame({
            'condition': cond_order,
            'mean': pivot.mean(axis=0).values,
            'sem': pivot.sem(axis=0).values
        })

        fig, ax = plt.subplots(figsize=figsize)
        x_base = np.arange(len(cond_order))
        colors = [color_1, color_2, color_3]

        ax.bar(x_base,
               summary['mean'],
               yerr=summary['sem'],
               color=colors,
               capsize=5,
               width=0.7,
               error_kw={'elinewidth': 2})

        subs = pivot.index.tolist()
        jitter = np.random.uniform(-0.1,
                                   0.1,
                                   size=(len(subs), len(cond_order)))

        for i, sid in enumerate(subs):
            ys = pivot.loc[sid].values
            xs = x_base + jitter[i]
            ax.scatter(xs, ys, color='black', alpha=0.6, s=30)

        ax.tick_params(axis='y', labelsize=15)
        ax.set_ylabel('Total trial number', fontsize=18)
        ax.set_xticks(x_base)
        ax.set_xticklabels([label_1, label_2, label_3], fontsize=18)

        ax.grid(False)
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Figure saved to {save_path}")
        plt.close(fig)


class Fig1_Acc:

    def plot_accuracy(self,
                      learning_data: pd.DataFrame,
                      block_size: int = 64,
                      figsize=(15, 5),
                      widths=(2, 1, 1),
                      save_path: str = None):

        # 1) 给每个 trial 添加序号和 block 编号
        learning_data = learning_data.copy()
        learning_data['trial_in_sub'] = learning_data.groupby(
            'iSub').cumcount() + 1
        learning_data['block'] = (
            (learning_data['trial_in_sub'] - 1) // block_size + 1)

        learning_data = learning_data[learning_data['ambigous'] != 1]
        # 2) 按 condition、iSub、block 计算 block-wise accuracy
        blk_acc = (learning_data.groupby(
            ['condition', 'iSub',
             'block'])['feedback'].mean().reset_index(name='block_accuracy'))

        max_blocks = (blk_acc.groupby('condition')['block'].max())

        # 3) 确定 condition 列表及其子图宽度
        conds = sorted(blk_acc['condition'].unique())
        if len(conds) != 3:
            raise ValueError(
                f"Expected 3 conditions, but found {len(conds)}: {conds}")

        # 4) 创建一行三列子图，按 widths 比例分配宽度
        fig, axes = plt.subplots(1,
                                 3,
                                 figsize=figsize,
                                 gridspec_kw={'width_ratios': widths},
                                 sharey=True)

        # 5) 对每个 condition 画所有被试的 block accuracy 曲线
        for ax, cond in zip(axes, conds):
            sub_df = blk_acc[blk_acc['condition'] == cond]
            # 1) count trials per subject (after filtering ambiguous trials)
            cond_data = learning_data[learning_data['condition'] == cond]
            trial_counts = cond_data.groupby('iSub').size()
            # 2) sort subjects by count (ascending)
            sorted_sids = trial_counts.sort_values().index.tolist()

            # 3) build an 8-entry gradient from magenta to cyan
            mag_cy = LinearSegmentedColormap.from_list('mag_cy', ['magenta', 'cyan'])
            color_gradient = [mag_cy(t) for t in np.linspace(0, 1, len(sorted_sids))]

            # 4) plot in order of trial-count
            for idx, sid in enumerate(sorted_sids):
                df_sub = sub_df[sub_df['iSub'] == sid]
                col = color_gradient[idx]
                ax.plot(df_sub['block'],
                        df_sub['block_accuracy'],
                        alpha=0.6,
                        lw=1,
                        color=col,
                        label='Subject lines' if idx == 0 else None,
                        clip_on=False)
                ax.scatter(df_sub['block'],
                        df_sub['block_accuracy'],
                        s=20,
                        alpha=0.8,
                        color=col,
                        clip_on=False)

            ax.axhline(y=0.9,
                       color='grey',
                       linestyle='--',
                       linewidth=1,
                       alpha=0.8)

            max_blk = max_blocks[cond]
            ax.set_xlim(1, max_blk)
            xticks = np.arange(1, max_blocks[cond] + 1, 2)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=15)

            ax.set_title(f'Exp {cond}', fontsize=18, pad=12)
            ax.set_xlabel('Block', fontsize=18)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)

            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2)

        # 6) 全局美化：左侧子图显示 y 轴标签
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='y', labelsize=15)
        axes[0].set_ylabel('Block-wise Accuracy', fontsize=18)

        # axes.margins(x=0.02, y=0.02)

        plt.tight_layout()

        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)
        print(f"Figure saved to {save_path}")
        plt.close(fig)




class Fig3_Group:

    def plot_group_acc_error(
            self,
            results_1,
            results_2,
            results_3,
            results_4,
            mc_method='bonferroni',  # 'bonferroni', 'holm', 'fdr_bh', etc.
            figsize=(6, 5),
            color_1='#45B53F',
            color_2='#478ECC',
            color_3='#DDAA33',
            color_4='#A6A6A6',
            label_1='Base',
            label_2='Jump',
            label_3='Forget',
            label_4='Fgt+Jump',
            save_path=None):

        # 计算每个被试的 error
        def compute_errors(results):
            errors = {}
            for subject_id, data in results.items():
                err = np.mean(
                    np.abs(
                        np.array(data['sliding_true_acc']) -
                        np.array(data['sliding_pred_acc'])))
                errors[subject_id] = err
            return errors

        errors = [
            compute_errors(results_1),
            compute_errors(results_2),
            compute_errors(results_3),
            compute_errors(results_4),
        ]

        df = pd.DataFrame({
            'Model':
            sum([[f'Model {i+1}'] * len(errors[i]) for i in range(4)], []),
            'Error':
            sum([list(e.values()) for e in errors], [])
        })
        summary = df.groupby('Model')['Error'].agg(['mean', 'sem']).reindex(
            ['Model 1', 'Model 2', 'Model 3', 'Model 4']).reset_index()

        subs = sorted(
            set(errors[0]) & set(errors[1]) & set(errors[2]) & set(errors[3]))

        arrs = [np.array([errors[i][sid] for sid in subs]) for i in range(4)]

        raw_ps = []
        for i in range(3):
            _, p = ttest_rel(arrs[i], arrs[i + 1])
            raw_ps.append(p)

        # 5) 多重比较校正
        reject, p_corrected, _, _ = multipletests(raw_ps,
                                                  alpha=0.05,
                                                  method=mc_method)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(summary['Model'],
                      summary['mean'],
                      yerr=summary['sem'],
                      color=[color_1, color_2, color_3, color_4],
                      capsize=5,
                      width=0.6,
                      edgecolor='black')

        # 画连线+散点
        x_base = np.arange(4)
        jitter = np.random.uniform(-0.1, 0.1, size=(len(subs), 4))
        for i, sid in enumerate(subs):
            ys = [errors[j][sid] for j in range(4)]
            xs = x_base + jitter[i]
            ax.plot(xs, ys, color='gray', alpha=0.5, linewidth=1)
            ax.scatter(xs,
                       ys,
                       color='black',
                       alpha=0.6,
                       s=20,
                       label='Individual Data' if i == 0 else None)

        # 7) 在柱子上方添加显著性标注
        def get_star(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return 'n.s.'

        y_max = (summary['mean'] + summary['sem']).max()
        h = y_max * 0.05  # 标注高度增量
        for i, (p_corr, rej) in enumerate(zip(p_corrected, reject)):
            x1, x2 = x_base[i], x_base[i + 1]
            y = max(summary.loc[i, 'mean'] + summary.loc[i, 'sem'],
                    summary.loc[i + 1, 'mean'] + summary.loc[i + 1, 'sem']) + h
            # 横线
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                    lw=1.5,
                    color='black')
            # 星号或 n.s.
            ax.text((x1 + x2) / 2,
                    y + h * 1.1,
                    get_star(p_corr),
                    ha='center',
                    va='bottom',
                    fontsize=14)

        ax.set_ylabel('Accuracy deviation', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        # ax.set_title('Accuracy deviation Comparison Across Models', fontsize=16)
        ax.set_xticks(x_base)
        ax.set_xticklabels([label_1, label_2, label_3, label_4], fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()

        if save_path:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close()

        print("Raw p-values:        ", np.round(raw_ps, 4))
        print(f"Corrected p-values ({mc_method}):", np.round(p_corrected, 4))
        print("Reject null hypotheses:", reject)

    def plot_group_k_corr(self,
                          oral_hypo_hits,
                          results_1,
                          results_2,
                          results_3,
                          results_4,
                          mc_method='bonferroni',
                          figsize=(6, 5),
                          color_1='#45B53F',
                          color_2='#478ECC',
                          color_3='#DDAA33',
                          color_4='#A6A6A6',
                          label_1='Base',
                          label_2='Jump',
                          label_3='Forget',
                          label_4='Fgt+Jump',
                          save_path=None):
        """
        绘制四个模型在rolling_hits与模型移动平均之间相关度的组水平对比。

        参数：
        - oral_hypo_hits: dict, 每个被试的真实rolling_hits和condition信息
        - results_1..4: dict, 四个模型的step_results或best_step_results
        - mc_method: 多重比较校正方法
        - color_* / label_*: 各模型颜色与标签
        - save_path: 保存路径
        """

        # ---------- 移动平均提取函数 ----------
        def _extract_ma(step_dict, k_special, win=16):
            step_res = step_dict.get('step_results',
                                     step_dict.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special,
                                                {}).get('post_max', 0.0)
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            return pd.Series(post_vals, dtype=float).rolling(
                window=win, min_periods=win).mean().to_numpy()

        # ---------- 计算每个被试的相关度 ----------
        def compute_k_corrs(results):
            corrs = {}
            for subject_id, odict in oral_hypo_hits.items():
                if subject_id not in results:
                    continue
                rolling_hits = np.array(odict['rolling_hits'], dtype=float)
                condition = odict['condition']
                k_special = 0 if condition == 1 else 42
                ma = _extract_ma(results[subject_id], k_special)
                n = min(len(rolling_hits), len(ma))
                x = rolling_hits[:n]
                y = ma[:n]
                valid = ~np.isnan(y)
                if valid.sum() < 2 or np.nanstd(x[valid]) == 0 or np.nanstd(
                        y[valid]) == 0:
                    corrs[subject_id] = np.nan
                else:
                    corrs[subject_id] = np.corrcoef(x[valid], y[valid])[0, 1]
            return corrs

        # 四个模型相关度字典
        corrs_list = [
            compute_k_corrs(results_1),
            compute_k_corrs(results_2),
            compute_k_corrs(results_3),
            compute_k_corrs(results_4),
        ]

        # 构建DataFrame并计算mean与sem
        df = pd.DataFrame({
            'Model':
            sum([[f'Model {i+1}'] * len(corrs_list[i]) for i in range(4)], []),
            'Correlation':
            sum([list(c.values()) for c in corrs_list], [])
        })
        summary = df.groupby('Model')['Correlation'].agg(
            ['mean',
             'sem']).reindex(['Model 1', 'Model 2', 'Model 3',
                              'Model 4']).reset_index()

        # 取共有被试
        shared_subs = sorted(
            set(corrs_list[0]) & set(corrs_list[1]) & set(corrs_list[2])
            & set(corrs_list[3]))
        arrs = [
            np.array([corrs_list[i][sid] for sid in shared_subs])
            for i in range(4)
        ]

        # 统计检验（配对 t 检验）
        raw_ps = []
        for i in range(3):
            _, p = ttest_rel(arrs[i], arrs[i + 1])
            raw_ps.append(p)
        reject, p_corrected, _, _ = multipletests(raw_ps,
                                                  alpha=0.05,
                                                  method=mc_method)

        # ---------- 绘图 ----------
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(summary['Model'],
                      summary['mean'],
                      yerr=summary['sem'],
                      color=[color_1, color_2, color_3, color_4],
                      capsize=5,
                      width=0.6,
                      edgecolor='black')
        # 个体连线与散点
        x_base = np.arange(4)
        jitter = np.random.uniform(-0.1, 0.1, size=(len(shared_subs), 4))
        for i, sid in enumerate(shared_subs):
            ys = [corrs_list[j][sid] for j in range(4)]
            xs = x_base + jitter[i]
            ax.plot(xs, ys, color='gray', alpha=0.5, linewidth=1)
            ax.scatter(xs,
                       ys,
                       color='black',
                       alpha=0.6,
                       s=20,
                       label='Individual Data' if i == 0 else None)

        # 显著性标注函数
        def get_star(p):
            if p < 0.001:
                return '***'
            if p < 0.01:
                return '**'
            if p < 0.05:
                return '*'
            return 'n.s.'

        # 添加显著性标注
        y_max = (summary['mean'] + summary['sem']).max()
        h = y_max * 0.05
        for i, (p_corr, rej) in enumerate(zip(p_corrected, reject)):
            x1, x2 = x_base[i], x_base[i + 1]
            y = max(summary.loc[i, 'mean'] + summary.loc[i, 'sem'],
                    summary.loc[i + 1, 'mean'] + summary.loc[i + 1, 'sem']) + h
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                    lw=1.5,
                    color='black')
            ax.text((x1 + x2) / 2,
                    y + h * 1.1,
                    get_star(p_corr),
                    ha='center',
                    va='bottom',
                    fontsize=14)

        # 坐标与标签
        ax.set_ylabel('Correlation coefficient', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_xticks(x_base)
        ax.set_xticklabels([label_1, label_2, label_3, label_4], fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()

        # 保存与输出
        if save_path:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            print(f"Figure saved to {save_path}")
            plt.close()
        print("Raw p-values:        ", np.round(raw_ps, 4))
        print(f"Corrected p-values ({mc_method}):", np.round(p_corrected, 4))
        print("Reject null hypotheses:", reject)

    def plot_group_aic(self,
                       results_1,
                       results_2,
                       results_3,
                       results_4,
                       mc_method='bonferroni',
                       figsize=(6, 5),
                       color_1='#45B53F',
                       color_2='#478ECC',
                       color_3='#DDAA33',
                       color_4='#A6A6A6',
                       label_1='Base',
                       label_2='Jump',
                       label_3='Forget',
                       label_4='Fgt+Jump',
                       save_path=None):
        """
        计算并绘制四个模型的AIC对比。
        """

        def compute_aic(results, extra_params=0):
            aics = {}
            for subject_id, res in results.items():
                step_res = res.get('step_results',
                                   res.get('best_step_results', []))
                ll_vals = []
                for step in step_res:
                    ll = step.get('best_log_likelihood', 0.0)
                    try:
                        ll = float(ll)
                    except:
                        ll = 0.0
                    ll_vals.append(ll)
                total_ll = np.mean(ll_vals)
                k = 2 + extra_params
                aics[subject_id] = 2 * k - 2 * total_ll
            return aics

        extras = [0, 0, 2, 2]
        aic_list = [
            compute_aic(results_1, extras[0]),
            compute_aic(results_2, extras[1]),
            compute_aic(results_3, extras[2]),
            compute_aic(results_4, extras[3]),
        ]
        df = pd.DataFrame({
            'Model':
            sum([[f'Model {i+1}'] * len(aic_list[i]) for i in range(4)], []),
            'AIC':
            sum([list(a.values()) for a in aic_list], [])
        })
        summary = df.groupby('Model')['AIC'].agg(['mean', 'sem']).reindex(
            ['Model 1', 'Model 2', 'Model 3', 'Model 4']).reset_index()

        shared = sorted(
            set(aic_list[0]) & set(aic_list[1]) & set(aic_list[2])
            & set(aic_list[3]))
        arrs = [
            np.array([aic_list[i][sid] for sid in shared]) for i in range(4)
        ]
        raw_ps = []
        for i in range(3):
            _, p = ttest_rel(arrs[i], arrs[i + 1])
            raw_ps.append(p)
        reject, p_corrected, _, _ = multipletests(raw_ps,
                                                  alpha=0.05,
                                                  method=mc_method)

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(summary['Model'],
               summary['mean'],
               yerr=summary['sem'],
               color=[color_1, color_2, color_3, color_4],
               capsize=5,
               width=0.6,
               edgecolor='black')
        x_base = np.arange(4)
        jitter = np.random.uniform(-0.1, 0.1, size=(len(shared), 4))
        for i, sid in enumerate(shared):
            ys = [aic_list[j][sid] for j in range(4)]
            xs = x_base + jitter[i]
            ax.plot(xs, ys, color='gray', alpha=0.5, linewidth=1)
            ax.scatter(xs,
                       ys,
                       color='black',
                       alpha=0.6,
                       s=20,
                       label='Individual Data' if i == 0 else None)

        def get_star(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return 'n.s.'

        y_max = (summary['mean'] + summary['sem']).max()
        h = y_max * 0.05
        for i, (p_corr, rej) in enumerate(zip(p_corrected, reject)):
            x1, x2 = x_base[i], x_base[i + 1]
            y = max(summary.loc[i, 'mean'] + summary.loc[i, 'sem'],
                    summary.loc[i + 1, 'mean'] + summary.loc[i + 1, 'sem']) + h
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                    lw=1.5,
                    color='black')
            ax.text((x1 + x2) / 2,
                    y + h * 1.1,
                    get_star(p_corr),
                    ha='center',
                    va='bottom',
                    fontsize=14)

        ax.set_ylabel('AIC', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_xticks(x_base)
        ax.set_xticklabels([label_1, label_2, label_3, label_4], fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()
        if save_path:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            plt.close()
        print("Raw p-values:", np.round(raw_ps, 4))
        print(f"Corrected p-values ({mc_method}):", np.round(p_corrected, 4))
        print("Reject:", reject)
        # print(aic_list)



class Fig3_Individual:

    def plot_acc_comparison(self,
                            results,
                            subject_id,
                            label,
                            color,
                            figsize=(15, 5),
                            color_true='#A6A6A6',
                            save_path=None):
        # 基础设置
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')
        padding = [np.nan] * 15
        x_vals = None

        # 真实准确率
        true_acc = padding + list(results[subject_id]['sliding_true_acc'])
        x_vals = np.arange(1, len(true_acc) + 1)
        ax.plot(x_vals, true_acc, label='Human', color=color_true, linewidth=2)

        # 绘制预测结果
        pred = padding + list(results[subject_id]['sliding_pred_acc'])
        std = padding + list(results[subject_id]['sliding_pred_acc_std'])
        ax.plot(x_vals, pred, label=label, color=color, linewidth=2)
        low = np.array(pred) - np.array(std)
        high = np.array(pred) + np.array(std)
        ax.fill_between(x_vals, low, high, color=color, alpha=0.4)

        # 辅助元素
        add_segmentation_lines(ax,
                               len(x_vals),
                               interval=128,
                               color='grey',
                               alpha=0.3,
                               linestyle='--',
                               linewidth=1)
        style_axis(ax, show_ylabel=True, xtick_interval=128)
        ax.legend()
        fig.tight_layout()

        # 保存或展示
        if save_path:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)


    def get_oral_hypos_list(self,
                            condition: int,
                            data: Tuple[np.ndarray, np.ndarray],
                            model,
                            dist_tol: float = 1e-9) -> Dict[str, Any]:

        oral_centers, choices = data
        n_trials = len(choices)

        n_hypos = model.partition_model.prototypes_np.shape[0]
        all_hypos = range(n_hypos)

        oral_hypos_list = []

        for trial_idx in range(n_trials):
            reported_center = oral_centers[trial_idx]

            # If reported_center is missing or all NaNs, return empty list
            if reported_center is None \
            or (isinstance(reported_center, np.ndarray) and reported_center.size == 0) \
            or (isinstance(reported_center, np.ndarray) and np.all(np.isnan(reported_center))):
                oral_hypos_list.append([])
                continue

            cat_idx = choices[trial_idx] - 1

            # Compute distances to each hypothesis prototype
            distance_map = []
            for hypo_idx in all_hypos:
                true_center = model.partition_model.prototypes_np[hypo_idx, 0, cat_idx, :]
                distance_val = np.linalg.norm(reported_center - true_center)
                distance_map.append((distance_val, hypo_idx))

            # Exact matches within tolerance
            exact_matches = [h for (d, h) in distance_map if d <= dist_tol]

            if condition == 1:
                top_k = 5
            else:
                top_k = 10

            if exact_matches:
                chosen_hypos = exact_matches
            else:
                distance_map.sort(key=lambda x: x[0])
                chosen_hypos = [h for (_, h) in distance_map[:top_k]]

            oral_hypos_list.append(chosen_hypos)

        return oral_hypos_list


    def plot_k_comparison(self,
                          oral_hypo_hits: Dict[int, Dict[str, Any]],
                          results: Dict[int, Any],
                          subject_id: int,
                          figsize: Tuple[int, int] = (6, 5),
                          color: str = '#45B53F',
                          color_true: str = '#4C7AD1',
                          label: str = 'Model 1',
                          save_path: str | None = None):
        """
        Overlay (i) the subject’s distance-to-prototype trajectory and
        (ii) the posterior probability of the *same* target hypothesis
        produced by one or two competing models.

        Parameters
        ----------
        oral_hypo_hits : dict
            Subject-wise output from ``get_oral_distance`` –
            ``oral_hypo_hits[subject_id]['rolling_hits']`` is the smoothed curve.
        results_1 / results_2 : dict
            Step-wise posterior dumps from your fitting routine.  Each must have
            the structure  
            ``results_[subject_id]['step_results'][step]['hypo_details'][k]['post_max']``.
        subject_id : int
            ID whose data will be drawn.
        """
        # ---------- figure -------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')

        # ---------- oral-distance curve -----------------------------------------
        odict = oral_hypo_hits[subject_id]
        hits = odict['hits']
        n_steps = len(hits)
        x_vals = np.arange(1, n_steps + 1)
        win = 16

        # compute rolling_hits, ignoring empty-list entries in each window
        rolling_hits = []
        for i in range(n_steps):
            if i + 1 < win:
                rolling_hits.append(np.nan)
            else:
                window = hits[i-win+1:i+1]
                # filter out empty entries
                vals = [h for h in window if isinstance(h, (int, float))]
                if not vals:
                    rolling_hits.append(np.nan)
                else:
                    rolling_hits.append(float(np.mean(vals)))

        ax.plot(x_vals,
                hits,
                lw=3,
                label='Human',
                color=color_true)
        
        valid_idx = [i for i, h in enumerate(hits) if isinstance(h, (int, float))]

        # ---------- posterior curve(s) ------------------------------------------
        def _extract_ma_filtered(results_sub, k_special, valid_idx, win):
            # get raw posteriors
            step_res = results_sub.get('step_results', results_sub.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special, {}).get('post_max', 0.0)
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            # filter to only valid trials
            filtered = [post_vals[i] for i in valid_idx]
            # compute rolling mean
            return pd.Series(filtered, dtype=float).rolling(window=win, min_periods=win).mean().to_numpy()

        condition = odict['condition']
        k_special = 0 if condition == 1 else 42
        rolling_k = _extract_ma_filtered(results[subject_id], k_special, valid_idx, win)
        # plot posterior aligned to valid x positions
        x_k = np.array(valid_idx)[win-1:] + 1  # only plot where rolling defined
        ax.plot(x_k,
                rolling_k[win-1:],
                lw=3,
                color=color,
                label=f'{label}: k={k_special}')

        # ---------- cosmetics ----------------------------------------------------
        add_segmentation_lines(ax,
                               n_steps,
                               interval=128,
                               color='grey',
                               alpha=0.3,
                               linestyle='--',
                               linewidth=1)
        style_axis(ax,
                   show_ylabel=True,
                   ylabel='Probability',
                   xtick_interval=128)

        ax.legend()
        fig.tight_layout()

        # ---------- save / show --------------------------------------------------
        if save_path:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            print(f'[Fig3D] saved → {save_path}')
        plt.close(fig)


class Fig4:

    def plot_amount(self,
                    results: Dict[int, Any],
                    subject_id: int,
                    window_size: int = 16,
                    figsize: Tuple[int, int] = (6, 5),
                    color_1: str = '#DDAA33',
                    color_2: str = '#DDAA33',
                    label_1: str = 'Exploitation',
                    label_2: str = 'Exploration',
                    save_path: str | None = None):

        # Extract data for the subject
        step_results = results[subject_id].get('best_step_results', [])
        random_posterior = [
            step['best_step_amount']['random_posterior'][0]
            for step in step_results if 'best_step_amount' in step
        ]
        random = [
            step['best_step_amount']['random'][0]
            for step in step_results if 'best_step_amount' in step
        ]

        # Compute rolling averages with a window of 16 trials
        rolling_exploitation = pd.Series(random_posterior).rolling(window=window_size, min_periods=window_size).mean().to_list()
        rolling_exploration = pd.Series(random).rolling(window=window_size, min_periods=window_size).mean().to_list()
        x_vals = np.arange(1, len(random_posterior) + 1)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')

        # Plot the data
        ax.plot(x_vals, rolling_exploitation, label=label_1, color=color_1, lw=2)
        ax.plot(x_vals,
                rolling_exploration,
                label=label_2,
                color=color_2,
                lw=2)

        # Add labels and legend
        ax.set_xlabel('Trial', fontsize=14)
        ax.set_ylabel('Amount', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Adjust layout
        fig.tight_layout()

        # Save or show the plot
        if save_path:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)
