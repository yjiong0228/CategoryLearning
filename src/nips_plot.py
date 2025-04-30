import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from math import isfinite
from matplotlib import cm
from matplotlib import font_manager
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional
from .plot_utils import (create_grid_figure,add_segmentation_lines,style_axis,annotate_label)



# 1. 注册本地字体（把路径换成你机器上 Arial.ttf 的实际路径）
font_path = '/home/yangjiong/CategoryLearning/src/Arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# 2. 在绘图前设置默认字体
mpl.rcParams['font.family'] = prop.get_name()


class Fig1B:

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
    def plot_acc_comparison(self,
                            results,
                            subject_ids,
                            widths=(2, 1, 1),
                            figsize=(15, 5),
                            color_acc='#45B53F',
                            save_path=None):

        # 创建画布和 GridSpec，wspace 调小以减小子图间距
        fig, gs = create_grid_figure(widths, figsize)

        for idx, sub in enumerate(subject_ids):
            ax = fig.add_subplot(gs[idx])

            sliding_true_acc = results[sub]['sliding_true_acc']
            sliding_pred_acc = results[sub]['sliding_pred_acc']
            sliding_pred_acc_std = results[sub]['sliding_pred_acc_std']

            # 在每列数据的最前面添加15个空值
            sliding_true_acc = [np.nan] * 15 + sliding_true_acc
            sliding_pred_acc = [np.nan] * 15 + sliding_pred_acc
            sliding_pred_acc_std = [np.nan] * 15 + sliding_pred_acc_std

            num_steps = len(sliding_pred_acc)

            ax.plot(range(1, num_steps + 1),
                    sliding_true_acc,
                    label='True',
                    color='#A6A6A6',
                    linewidth=2)

            ax.plot(range(1, num_steps + 1),
                    sliding_pred_acc,
                    label='Predicted',
                    color=color_acc,
                    linewidth=2)

            lower_bound = np.array(sliding_pred_acc) - np.array(
                sliding_pred_acc_std)
            upper_bound = np.array(sliding_pred_acc) + np.array(
                sliding_pred_acc_std)
            ax.fill_between(range(1, num_steps + 1),
                            lower_bound,
                            upper_bound,
                            color=color_acc,
                            alpha=0.4,
                            label='Predicted Std')

            add_segmentation_lines(ax, num_steps, interval=64, color='grey', alpha=0.3, linestyle='--', linewidth=1)
            style_axis(ax, show_ylabel=(idx == 0))
            annotate_label(ax, f"S{sub}")

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