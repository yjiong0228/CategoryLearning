import os
import numpy as np
from scipy.interpolate import splprep, splev
from math import isfinite
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager
import matplotlib as mpl
from matplotlib.collections import LineCollection
from typing import Dict, Tuple, List, Any
import pandas as pd
from .plot_utils import (create_grid_figure,add_segmentation_lines,style_axis,annotate_label)


# 1. 注册本地字体（把路径换成你机器上 Arial.ttf 的实际路径）
font_path = '/home/yangjiong/CategoryLearning/src/Arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# 2. 在绘图前设置默认字体
mpl.rcParams['font.family'] = prop.get_name()


class Fig1D:

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
        # 直接返回所需列，无需重命名
        return df[required]

    def draw_cube(self, ax, edge_color='#808080', lw=1.5):
        verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1),
                 (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        edges = [(1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 4), (1, 5),
                 (2, 6), (3, 7)]
        for u, v in edges:
            x, y, z = zip(verts[u], verts[v])
            ax.plot(x, y, z, color=edge_color, linewidth=lw, alpha=0.9)

    def _beautify_axes(self, ax):
        ax.set_axis_off()

        colors = {
            'x': '#B56DFF',  # 浅紫
            'y': '#70AD47',  # 浅绿
            'z': '#ED7D31',  # 浅橙
        }

        # 手动绘制三条主轴线
        ax.plot([0, 1], [0, 0], [0, 0], color='#808080', lw=2.5)
        ax.plot([0, 0], [0, 1], [0, 0], color='#808080', lw=2.5)
        ax.plot([0, 0], [0, 0], [0, 1], color='#808080', lw=2.5)

        # 三个轴的手动标题
        ax.text(1.2,
                -0.4,
                0.0,
                'Feature 2',
                fontsize=18,
                horizontalalignment='left',
                verticalalignment='center',
                fontfamily='Arial')
        ax.text(0.0,
                1.26,
                0.0,
                'Feature 1',
                fontsize=18,
                horizontalalignment='center',
                verticalalignment='bottom',
                fontfamily='Arial')
        ax.text(0.2,
                0.0,
                1.08,
                'Feature 3',
                fontsize=18,
                horizontalalignment='center',
                verticalalignment='bottom',
                fontfamily='Arial')

        # 4) 手动添加刻度标签
        # 原点
        ax.text(0,
                0,
                -0.05,
                '0',
                fontsize=15,
                ha='right',
                va='top',
                fontfamily='Arial')
        # X 轴刻度
        ax.text(0.45,
                0,
                -0.05,
                '0.5',
                fontsize=15,
                ha='center',
                va='top',
                fontfamily='Arial')
        ax.text(0.9,
                0,
                -0.05,
                '1',
                fontsize=15,
                ha='center',
                va='top',
                fontfamily='Arial')
        # Y 轴刻度
        ax.text(0,
                0.48,
                -0.05,
                '0.5',
                fontsize=15,
                ha='right',
                va='center',
                fontfamily='Arial')
        ax.text(0,
                0.96,
                -0.05,
                '1',
                fontsize=15,
                ha='right',
                va='center',
                fontfamily='Arial')
        # Z 轴刻度
        ax.text(0,
                -0.03,
                0.45,
                '0.5',
                fontsize=15,
                ha='right',
                va='bottom',
                fontfamily='Arial')
        ax.text(0,
                -0.03,
                0.9,
                '1',
                fontsize=15,
                ha='right',
                va='bottom',
                fontfamily='Arial')

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

        # 若全是 NaN，就直接退出
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

        # ---------- 4. ≥3 点：尝试样条 ----------
        k = min(3, n_pts - 1)
        try:
            tck, u = splprep([xs, ys, zs], s=smooth, k=k)
            u_fine = np.linspace(0, 1, n_interp)
            x_f, y_f, z_f = splev(u_fine, tck)
        except Exception as e:  # 任意插值失败都回退到折线
            # 折线 & 渐变散点
            colors = cmap(np.linspace(cmap_low, cmap_high, n_pts))
            for i in range(n_pts - 1):
                ax.plot(xs[i:i + 2],
                        ys[i:i + 2],
                        zs[i:i + 2],
                        color=colors[i],
                        linewidth=lw)
            ax.scatter(xs, ys, zs, color=colors, s=s, edgecolors='w')
            return

        # ---------- 5. 成功则画平滑曲线 ----------
        line_colors = cmap(np.linspace(cmap_low, cmap_high, n_interp - 1))
        for i in range(n_interp - 1):
            ax.plot(x_f[i:i + 2],
                    y_f[i:i + 2],
                    z_f[i:i + 2],
                    color=line_colors[i],
                    linewidth=lw)

        pt_colors = cmap(cmap_low + (cmap_high - cmap_low) * u)
        ax.scatter(xs, ys, zs, color=pt_colors, s=s, edgecolors='w')

    def plot_choice_graph(self, ncats, iSub, iSession, iTrial, choice,
                          features_list, plots_dir):
        folder = os.path.join(plots_dir, f"choice{choice}")
        os.makedirs(folder, exist_ok=True)
        fig = plt.figure(figsize=(6, 6), facecolor='none')
        ax = fig.add_subplot(111, projection='3d')

        if ncats == 2:
            yellow_point_coords = {
                1: (0.5, 0.25, 0.5),
                2: (0.5, 0.75, 0.5),
            }
        elif ncats == 4:
            yellow_point_coords = {
                1: (0.25, 0.25, 0.5),
                2: (0.25, 0.75, 0.5),
                3: (0.75, 0.5, 0.25),
                4: (0.75, 0.5, 0.75)
            }

        if choice in yellow_point_coords:
            y_point = yellow_point_coords[choice]
            ax.scatter(*y_point,
                       color='yellow',
                       s=200,
                       alpha=0.7,
                       edgecolors='k')

        xs = [f['feature2_oral'] for f in features_list]
        ys = [f['feature1_oral'] for f in features_list]
        zs = [f['feature3_oral'] for f in features_list]

        if len(xs) > 1:
            self._plot_smooth_grad_line_and_points(ax, xs, ys, zs)

        ax.view_init(elev=15, azim=30)
        self._beautify_axes(ax)
        self.draw_cube(ax)
        ax.patch.set_alpha(0)

        fname = f"{iSub}_{iSession}_{iTrial}_c{choice}.png"
        # fig.tight_layout()
        plt.savefig(os.path.join(folder, fname), transparent=True, dpi=300)
        plt.close()

    def process_and_plot(self,
                         ncats,
                         subject_data,
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
                self.plot_choice_graph(ncats, row.get('iSub', 'Unknown'),
                                       row['iSession'], row['iTrial'], choice,
                                       last_feats[choice], plots_dir)
                # self.plot_feature4_time_series(
                #     row.get('iSub', 'Unknown'),
                #     row['iSession'],
                #     row['iTrial'],
                #     choice,
                #     last_feats[choice],
                #     plots_dir
                # )

        print(f"完成：仅绘制行 {sorted(target)}，图表已保存至 {plots_dir}/choice*/ 文件夹。")


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
        folder = os.path.join(plots_dir, f"choice{choice}")
        os.makedirs(folder, exist_ok=True)

        # Prepare figure and axis
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')

        # Extract trial numbers and feature4 values
        trials = np.array([f.get('iTrial', idx+1) for idx, f in enumerate(features_list)], dtype=float)
        vals = np.array([f['feature4_oral'] for f in features_list], dtype=float)

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
        ax.set_xlabel('Trial', fontfamily='Arial', fontsize=18)
        ax.set_ylabel('Feature 4', fontfamily='Arial', fontsize=18)

        # Set ticks: y at [0,0.5,1], x only last at 190
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'], fontsize=15)
        ax.set_xticks([83])
        ax.set_xticklabels(['190'], fontsize=15)

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
        fname = f"{iSub}_{iSession}_{iTrial}_c{choice}_feature4_ts.png"
        plt.savefig(os.path.join(folder, fname), transparent=True, dpi=300)
        plt.close()



class Fig1C:
    def plot_accuracy(self,
                    learning_data,
                    subject_ids,
                    widths=(2, 1, 1),
                    figsize=(15, 5),
                    color_acc='#DDAA33',
                    save_path=None):
        """
        为三个被试绘制滑动窗口正确率曲线，三个子图之间间距更小，并可保存整张图。

        参数
        ----
        learning_data : pandas.DataFrame
            已包含 'iSub', 'trial_in_sub' 和 'rolling_accuracy' 等字段。
        subject_ids : list of int
            长度为 3 的被试编号列表。
        widths : tuple of 3 ints
            三个子图的相对宽度，例如 (2,1,1)。
        figsize : tuple
            整个画布的尺寸 (宽, 高)。
        color_acc : str
            正确率曲线的颜色。
        save_path : str or pathlib.Path, optional
            如果提供，保存整张图到该路径（支持 png, pdf 等格式）。
        """
        if len(subject_ids) != 3:
            raise ValueError("请提供三个被试编号，例如 [1, 5, 7]")

        # 创建画布和 GridSpec，wspace 调小以减小子图间距
        fig, gs = create_grid_figure(widths, figsize)

        for idx, sub in enumerate(subject_ids):
            ax = fig.add_subplot(gs[idx])
            subj_data = learning_data[learning_data['iSub'] == sub].reset_index(
                drop=True)
            trials = subj_data['trial_in_sub']
            acc = subj_data['rolling_accuracy']

            # 画曲线
            ax.plot(trials, acc, color=color_acc, linewidth=2)

            add_segmentation_lines(ax, len(trials), interval=64, color='grey', alpha=0.3, linestyle='--', linewidth=1)
            style_axis(ax, show_ylabel=(idx == 0))
            annotate_label(ax, f"S{sub}")

            # if idx == 2:
            #     ax.legend(loc='upper right', fontsize=12)

        # 调整子图间距
        # plt.subplots_adjust(left=0.05,
        #                     right=0.98,
        #                     top=0.95,
        #                     bottom=0.10,
        #                     wspace=0.10)

        # 保存或展示
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close()

    def plot_feature(self,
                     learning_data,
                     subject_ids,
                     widths=(2, 1, 1),
                     figsize=(15, 5),
                     feature_cols=('rolling_feature1_use',
                                   'rolling_feature2_use',
                                   'rolling_feature3_use',
                                   'rolling_feature4_use'),
                     colors=('#70AD47', '#B56DFF', '#ED7D31', '#808080'),
                     save_path=None):
        """
        为三个被试绘制四条特征使用率轨迹，画在三张子图中，并可保存透明背景图。

        参数
        ----
        learning_data : pandas.DataFrame
            已包含 'iSub', 'trial_in_sub' 及各 rolling_featureX_use 字段。
        subject_ids : list of int
            长度为 3 的被试编号列表。
        widths : tuple of 3 ints
            三个子图的相对宽度。
        figsize : tuple
            整个画布尺寸 (宽, 高)。
        feature_cols : tuple of str
            要绘制的四个列名。
        colors : tuple of str
            对应四条曲线的颜色。
        save_path : str or pathlib.Path, optional
            如果提供，保存整张图到该路径（支持 png, pdf 等格式）。
        """
        if len(subject_ids) != 3:
            raise ValueError("请提供三个被试编号，例如 [1, 5, 7]")

        fig, gs = create_grid_figure(widths, figsize)

        for idx, sub in enumerate(subject_ids):
            ax = fig.add_subplot(gs[idx])
            subj_data = learning_data[learning_data['iSub'] == sub].reset_index(drop=True)
            trials = subj_data['trial_in_sub']

            # 绘制四条特征曲线
            for col, col_color in zip(feature_cols, colors):
                ax.plot(trials,
                        subj_data[col],
                        label=col.replace('rolling_', ''),
                        color=col_color,
                        alpha=0.8,
                        linewidth=2)

            add_segmentation_lines(ax, len(trials), interval=64, color='grey', alpha=0.3, linestyle='--', linewidth=1)
            style_axis(ax, show_ylabel=(idx == 0))
            annotate_label(ax, f"S{sub}")

            # if idx == 2:
            #     ax.legend(loc='upper right', fontsize=12)

        # plt.subplots_adjust(left=0.05,
        #                     right=0.98,
        #                     top=0.95,
        #                     bottom=0.10,
        #                     wspace=0.10)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close()


class Fig3:

    def plot_error_comparison(self,
                              results_1,
                              results_2,
                              results_3,
                              figsize=(6, 5),
                              color_1='#45B53F',
                              color_2='#DDAA33',
                              color_3='#A6A6A6',
                              label_1='Base',
                              label_2='Forget',
                              label_3='Fgt+Jump',
                              save_path=None):

        # 计算每个被试的 error
        def compute_errors(results):
            errors = {}
            for subject_id, data in results.items():
                err = np.mean(
                    np.abs(
                        np.array(data['sliding_true_acc']) -
                        np.array(data['sliding_pred_acc'])
                    )
                )
                errors[subject_id] = err
            return errors

        errors_1 = compute_errors(results_1)
        errors_2 = compute_errors(results_2)
        errors_3 = compute_errors(results_3)

        df = pd.DataFrame({
            'Model': ['Model 1'] * len(errors_1) +
                    ['Model 2'] * len(errors_2) +
                    ['Model 3'] * len(errors_3),
            'Error': list(errors_1.values()) +
                    list(errors_2.values()) +
                    list(errors_3.values())
        })
        summary = df.groupby('Model')['Error'].agg(['mean','sem']).reindex(
            ['Model 1','Model 2','Model 3']
        ).reset_index()

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(summary['Model'],
            summary['mean'],
            yerr=summary['sem'],
            color=[color_1, color_2, color_3],
            capsize=5,
            width=0.6,
            edgecolor='black')

        subs = sorted(set(errors_1) & set(errors_2) & set(errors_3))
        x_base = np.arange(3)
        jitter = np.random.uniform(-0.1, 0.1, size=(len(subs), 3))

        for i, sid in enumerate(subs):
            ys = [errors_1[sid], errors_2[sid], errors_3[sid]]
            xs = x_base + jitter[i]
            # 连线
            ax.plot(xs, ys, color='gray', alpha=0.5, linewidth=1)
            # 散点（只给第一次画点加 legend）
            ax.scatter(xs, ys,
                    color='black', alpha=0.6, s=20,
                    label='Individual Data' if i == 0 else None)

        ax.set_ylabel('Error', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_title('Error Comparison Across Models', fontsize=16)
        ax.set_xticks(x_base)
        ax.set_xticklabels([label_1,label_2,label_3], fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close()

    def plot_acc_comparison(self,
                            results_1,
                            results_2,
                            subject_id,
                            figsize=(15, 5),
                            color_1='#45B53F',
                            color_2='#DDAA33',
                            color_true='#A6A6A6',
                            label_1='Model 1',
                            label_2='Model 2',
                            save_path=None):
        # 基础设置
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')
        padding = [np.nan] * 15
        x_vals = None

        # 真实准确率
        true_acc = padding + list(results_1[subject_id]['sliding_true_acc'])
        x_vals = np.arange(1, len(true_acc) + 1)
        ax.plot(x_vals, true_acc, label='Human', color=color_true, linewidth=2)

        # 循环绘制预测结果
        for results, label, color in ((results_1, label_1, color_1),
                                      (results_2, label_2, color_2)):
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
        # annotate_label(ax, f"S{subject_id}")
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


class Fig3D:

    def get_oral_probability(self,
                             data: Tuple[np.ndarray, np.ndarray],
                             model,
                             condition: int,
                             window_size: int = 16) -> Dict[str, Any]:
        """
        For each trial, compute an (unnormalised) Euclidean distance from the
        reported centre to every hypothesis' prototype of the chosen category,
        **normalise the distances across hypotheses** so they sum to 1, convert
        them to probabilities via `p = 1 - d_norm`, and finally return the moving
        average trajectory for the target hypothesis (h=0 for condition-1, h=42
        otherwise).

        Returns
        -------
        dict
            {
                'step_hit' : List[float]   # smoothed P(h=target)
                'raw'      : np.ndarray    # per-trial P(h=target)
                'condition': int
            }
        """
        centres, choices = data
        n_trials = len(choices)
        protos = model.partition_model.prototypes_np
        n_hypos = protos.shape[0]

        # ---------- distance matrix --------------------------------------------
        dis = np.empty((n_hypos, n_trials), dtype=float)
        for t in range(n_trials):
            cat = int(choices[t]) - 1
            dis[:, t] = np.linalg.norm(protos[:, 0, cat, :] - centres[t],
                                       axis=1)

        eps = 1e-12
        sims = 1.0 / (dis + eps)
        probs = sims / sims.sum(axis=0, keepdims=True)

        target_h = 0 if condition == 1 else 42
        prob_true = probs[target_h, :]

        rolling_prob_true = (pd.Series(prob_true).rolling(
            window=window_size, min_periods=window_size).mean().tolist())

        return {
            'condition': condition,
            'dis_true': dis[target_h, :],
            'prob_true': prob_true,
            'rolling_prob_true': rolling_prob_true,
        }

    def plot_k_comparison(self,
                          oral_probabilitis: Dict[int, Dict[str, Any]],
                          results_1: Dict[int, Any],
                          results_2: Dict[int, Any],
                          subject_id: int,
                          figsize: Tuple[int, int] = (6, 5),
                          color_1: str = '#45B53F',
                          color_2: str = '#DDAA33',
                          color_true: str = '#4C7AD1',
                          label_1: str = 'Model 1',
                          label_2: str = 'Model 2',
                          save_path: str | None = None):
        """
        Overlay (i) the subject’s distance-to-prototype trajectory and
        (ii) the posterior probability of the *same* target hypothesis
        produced by one or two competing models.

        Parameters
        ----------
        oral_distances : dict
            Subject-wise output from ``get_oral_distance`` –
            ``oral_distances[subject_id]['step_hit']`` is the smoothed curve.
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
        odict = oral_probabilitis[subject_id]
        rolling_prob_true = odict['rolling_prob_true']
        n_steps = len(rolling_prob_true)
        x_vals = np.arange(1, n_steps + 1)

        ax.plot(x_vals, rolling_prob_true, lw=3, label='Human', color=color_true)

        # ---------- posterior curve(s) ------------------------------------------
        def _extract_ma(results, k_special, win=16):
            step_res = results.get('step_results',
                                   results.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special,
                                                {}).get('post_max', 0.0)
                # 保底：非数字或 None 一律视为 0
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            return (pd.Series(post_vals, dtype=float).rolling(
                window=win, min_periods=win).mean().to_numpy())

        condition = odict['condition']
        k_special = 0 if condition == 1 else 42
        ma1 = _extract_ma(results_1[subject_id], k_special)
        ax.plot(x_vals[:len(ma1)],
                ma1,
                lw=3,
                color=color_1,
                label=f'{label_1}: k={k_special}')
        ma2 = _extract_ma(results_2[subject_id], k_special)
        ax.plot(x_vals[:len(ma2)],
                ma2,
                lw=3,
                color=color_2,
                label=f'{label_2}: k={k_special}')

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
