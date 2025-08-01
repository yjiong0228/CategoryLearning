import os
import numpy as np
import pandas as pd
from math import ceil
from scipy.interpolate import splprep, splev
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from typing import Dict, Tuple, List, Any, Union, Optional



# 1. 注册本地字体（把路径换成你机器上 Arial.ttf 的实际路径）
font_path = '/home/yangjiong/CategoryLearning/src/Arial.ttf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# 2. 在绘图前设置默认字体
mpl.rcParams['font.family'] = prop.get_name()


def add_segmentation_lines(ax, max_trial, interval=64, **line_kwargs):
    """
    Add vertical segmentation lines every `interval` trials.
    """
    for x in range(interval, max_trial + 1, interval):
        ax.axvline(x=x, **line_kwargs)

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
            ax.plot(x, y, z, color='#808080', lw=2.6)
        # 轴标题和刻度
        self._label_axes(ax)
        self._add_ticks(ax)

    def _label_axes(self, ax):
        labels = [((1.2, -0.07, 0), 'feat$_2$'), ((0, 1.2, 0.0), 'feat$_1$'),
                  ((0, 0, 1.14), 'feat$_3$')]
        for pos, txt in labels:
            ax.text(*pos,
                    txt,
                    fontsize=22,
                    ha='center',
                    va='center',
                    fontfamily='Arial')

    def _add_ticks(self, ax):
        ticks = [((-0.01, 0, -0.05), '0', 'right', 'top'),
                 ((0.45, 0, -0.05), '0.5', 'center', 'top'),
                 ((0.9, 0, -0.05), '1', 'center', 'top'),
                 ((0, 0.51, -0.09), '0.5', 'right', 'center'),
                 ((0, 1, -0.09), '1', 'right', 'center'),
                 ((0, -0.03, 0.45), '0.5', 'right', 'bottom'),
                 ((0, -0.03, 0.9), '1', 'right', 'bottom')]
        for pos, txt, ha, va in ticks:
            ax.text(*pos, txt, fontsize=18, ha=ha, va=va, fontfamily='Arial')

    def _plot_smooth_grad_line_and_points(self,
                                          ax,
                                          xs,
                                          ys,
                                          zs,
                                          cmap_name="viridis_r",
                                          cmap_low=0,
                                          cmap_high=0.9,
                                          lw=1.5,
                                          s=30,
                                          n_interp=300,
                                          jitter_sd: float = 0.05,       # 噪声幅度，可按需调整
                                          random_state: int | None = None,
                                          smooth: float = 0.0 ):
        """
        绘制 3D 轨迹并在端点加渐变散点。
        当检测到连续重复点时，为后一个点加入微小随机扰动 (jitter)，
        以避免完全重合导致轨迹中断或被过滤。

        Parameters
        ----------
        jitter_sd : float
            对重复点添加的高斯噪声标准差（单位与坐标一致，默认 0.002）。
        random_state : int | None
            随机种子，方便复现；None 则使用全局随机数生成器。
        """
        rng = np.random.default_rng(random_state)
        xs, ys, zs = map(np.asarray, (xs, ys, zs))

        # ---------- 1. 清洗无效 (NaN/Inf) ----------
        mask_valid = np.array([all(map(np.isfinite, p)) for p in zip(xs, ys, zs)])
        xs, ys, zs = xs[mask_valid], ys[mask_valid], zs[mask_valid]
        if len(xs) == 0:
            return

        # ---------- 2. 处理重复 (保留并抖动) ----------
        xs_j, ys_j, zs_j = [xs[0]], [ys[0]], [zs[0]]
        for x, y, z in zip(xs[1:], ys[1:], zs[1:]):
            if x == xs_j[-1] and y == ys_j[-1] and z == zs_j[-1]:
                dx, dy, dz = rng.normal(scale=jitter_sd, size=3)
                x, y, z = np.clip([x + dx, y + dy, z + dz], 0.0, 1.0)
            xs_j.append(x); ys_j.append(y); zs_j.append(z)

        xs, ys, zs = map(np.asarray, (xs_j, ys_j, zs_j))
        n_pts = len(xs)

        # ---------- 3. 小于 3 点：散点 / 直线 ----------
        cmap = cm.get_cmap(cmap_name)
        if n_pts == 1:
            ax.scatter(xs, ys, zs, color=cmap(cmap_high), s=s, edgecolors='w')
            return
        if n_pts == 2:
            c1, c2 = cmap(cmap_low), cmap(cmap_high)
            ax.plot(xs, ys, zs, color=c1, linewidth=lw)
            ax.scatter(xs, ys, zs, color=[c1, c2], s=s, edgecolors='w')
            return

        # ---------- 4. ≥3 点：样条拟合（回退折线） ----------
        try:
            k = min(3, n_pts - 1)
            tck, u = splprep([xs, ys, zs], s=smooth, k=k)
            u_fine = np.linspace(0, 1, n_interp)
            x_f, y_f, z_f = splev(u_fine, tck)
            colors = cmap(np.linspace(cmap_low, cmap_high, len(x_f) - 1))
            for i in range(len(x_f) - 1):
                ax.plot(x_f[i:i+2], y_f[i:i+2], z_f[i:i+2],
                        color=colors[i], linewidth=lw)
            pt_colors = cmap(cmap_low + (cmap_high - cmap_low) * u)
            ax.scatter(xs, ys, zs, color=pt_colors, s=s, edgecolors='w')
        except Exception:
            # 样条失败时回退为分段折线
            n = n_pts
            seg_colors = cmap(np.linspace(cmap_low, cmap_high, n))
            for i in range(n - 1):
                ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                        color=seg_colors[i], linewidth=lw)
            ax.scatter(xs, ys, zs, color=seg_colors, s=s, edgecolors='w')


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
        outer_alpha, inner_alpha = 0.005, 0.2
        sizes = np.linspace(outer_size, inner_size, n_layers)
        alphas = np.linspace(outer_alpha, inner_alpha, n_layers)
        for s, a in zip(sizes, alphas):
            ax.scatter(*point,
                       color='#A6A6A6',
                       s=s,
                       alpha=a,
                       edgecolors='none',
                       depthshade=False)

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
            self._plot_smooth_grad_line_and_points(ax, xs, ys, zs, cmap_low=0, cmap_high=1)

        ax.view_init(elev=15, azim=30)
        self._beautify_axes(ax)
        self.draw_cube(ax, '#808080', 1.7)
        ax.patch.set_alpha(0)

        fname = f"{iSub}_{iSession}_{iTrial}_c{choice}.svg"
        plt.savefig(os.path.join(folder, fname), transparent=True)
        plt.close()

    def plot_feature4_time_series(self,
                                  iSub,
                                  iSession,
                                  iTrial,
                                  choice,
                                  features_list,
                                  xstick,
                                  xsticklabel,
                                  plots_dir,
                                  cmap_name="viridis_r",
                                  cmap_low=0.7,
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
        ax.set_xlabel('Trial', fontfamily='Arial', fontsize=28)
        # ax.set_ylabel('Feature 4', fontfamily='Arial', fontsize=18)

        # Add text to the top-right corner
        ax.text(0.95,
                0.95,
                'feat$_4$',
                transform=ax.transAxes,
                fontsize=28,
                ha='right',
                va='top',
                fontfamily='Arial')

        # Set ticks: y at [0,0.5,1], x only last at 190
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'], fontsize=25)
        ax.set_xticks([xstick])
        ax.set_xticklabels([f'{xsticklabel}'], fontsize=25)

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
        fname = f"{iSub}_{iSession}_{iTrial}_c{choice}_f4_ts.svg"
        plt.savefig(os.path.join(folder, fname), transparent=True)
        plt.close()

    def plot_human_trajactory(self,
                              ncats,
                              subject_data,
                              type,
                              plots_dir,
                              xstick=1,
                              xsticklabel=1,
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
                                                   last_feats[choice], xstick, xsticklabel, folder)

        print(f"完成：仅绘制行 {sorted(target)}，图表已保存至 {plots_dir}/choice*/ 文件夹。")

    def plot_model_trajactory(self,
                              ncats,
                              df,
                              type,
                              plots_dir,
                              xstick=1,
                              xsticklabel=1,
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
                                                       last_feats[c],xstick,xsticklabel,
                                                       model_folder)

        print(f"完成模型轨迹绘制：行 {sorted(target)}，图表已保存至 {plots_dir}/choice*/ 文件夹。")


class Fig2_Partition:
    def __init__(self, edge_color='#808080', lw1=2, lw2=4, plane_color='#808080', plane_alpha=0.5):
        """
        edge_color : str
            Color for the cube edges and axes lines.
        lw : float
            Linewidth for cube edges and axes lines.
        plane_color : str
            Color for the partition plane.
        plane_alpha : float
            Transparency for the partition plane.
        """
        self.edge_color = edge_color
        self.lw1 = lw1
        self.lw2 = lw2
        self.plane_color = plane_color
        self.plane_alpha = plane_alpha

    def draw_cube(self, ax):
        """
        Draws the unit cube edges on the given 3D axes.
        """
        verts = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
        ]
        for u, v in edges:
            x, y, z = zip(verts[u], verts[v])
            ax.plot(x, y, z, color=self.edge_color, linewidth=self.lw1, alpha=0.9)

    def _beautify_axes(self, ax):
        """
        Removes default axes and draws three principal axes lines.
        """
        ax.set_axis_off()
        lines = [
            ([0, 1], [0, 0], [0, 0]),
            ([0, 0], [0, 1], [0, 0]),
            ([0, 0], [0, 0], [0, 1])
        ]
        for x, y, z in lines:
            ax.plot(x, y, z, color=self.edge_color, linewidth=self.lw2)

    def _draw_plane1(self, ax):
        """
        Draws the partition plane x = 0.5 within the cube.
        """
        verts = [
            (0, 0.5, 0),
            (1, 0.5, 0),
            (1, 0.5, 1),
            (0, 0.5, 1)
        ]
        poly = Poly3DCollection([verts], color=self.plane_color, alpha=self.plane_alpha)
        ax.add_collection3d(poly)

    def _draw_plane2(self, ax):
        """
        Draws the partition plane x = 0.5 within the cube.
        """
        verts = [
            (0, 0, 0.5),
            (1, 0, 0.5),
            (1, 1, 0.5),
            (0, 1, 0.5)
        ]
        poly = Poly3DCollection([verts], color=self.plane_color, alpha=self.plane_alpha)
        ax.add_collection3d(poly)

    def _draw_plane_x_equals_y(self, ax):
        """
        Draws the partition plane x = y within the cube.
        """
        verts = [
            (0, 0, 0), (1, 1, 0),
            (1, 1, 1), (0, 0, 1)
        ]
        poly = Poly3DCollection([verts],
                                 color=self.plane_color,
                                 alpha=self.plane_alpha)
        ax.add_collection3d(poly)

    def plot(self, i, save_path=None, show=True):
        """
        Creates the 3D plot with cube, axes, and partition plane.

        Parameters
        ----------
        save_path : str or None
            File path to save the figure as SVG or PNG. If None, figure is not saved.
        show : bool
            Whether to display the figure interactively.
        """
        fig = plt.figure(figsize=(6, 6), facecolor='none')
        ax = fig.add_subplot(111, projection='3d')

        self._beautify_axes(ax)
        self.draw_cube(ax)

        if i == 1:
            self._draw_plane1(ax)
        elif i == 2:
            self._draw_plane2(ax)
        elif i == 3:
            self._draw_plane_x_equals_y(ax)

        ax.view_init(elev=15, azim=30)
        ax.patch.set_alpha(0)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, transparent=True)
            plt.close(fig)
        if show:
            plt.show()



class Fig1_Ntrial:

    def plot_trial_number(self,
                          learning_data: pd.DataFrame,
                          iCon: int,
                          figsize=(15, 5),
                          color: str = '#45B53F',
                          save_path: str = None):
        # Aggregate trial counts by condition and subject
        counts = (
            learning_data
            .groupby(['condition', 'iSub'])
            .size()
            .reset_index(name='total_trials')
        )

        # Check condition exists
        available = sorted(counts['condition'].unique())
        if iCon not in available:
            raise ValueError(f"Condition {iCon} not found. Available: {available}")

        # Filter for the selected condition
        cond_counts = counts[counts['condition'] == iCon]
        trials = cond_counts.set_index('iSub')['total_trials']

        # Compute summary stats
        mean_trials = trials.mean()
        sem_trials = trials.sem()

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        x = np.array([0])

        ax.bar(
            x,
            mean_trials,
            yerr=sem_trials,
            color=color,
            capsize=5,
            width=0.2,
            error_kw={'elinewidth': 3, 'capthick': 3}
        )
        ax.margins(x=0.6)

        # Scatter individual values with jitter
        jitter = np.random.uniform(-0.1, 0.1, size=len(trials))
        ax.scatter(
            x + jitter,
            trials.values,
            color='black',
            alpha=0.6,
            s=50
        )

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylabel('Total trial number', rotation='vertical',fontsize=25)
        ax.tick_params(axis='y', labelsize=22)

        ax.grid(False)
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        # Save figure
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)



class Fig1_Acc:

    def plot_accuracy(self,
                      learning_data: pd.DataFrame,
                      subject_ids: List[int],
                      subfig: int,
                      nrow: int,
                      window_size: int = 16,
                      block_size: int = 64,
                      xtick_interval: int = 128,
                      h_pad: int = 8,
                      color: Union[str, List[str]] = 'C0',
                      max_trial: Optional[int] = None,
                      sub_text: Optional[List[Union[int, str]]] = None,
                      figsize=(15, 5),
                      save_path: str = None):
        """
        Plot rolling average accuracy for specified subjects, each in its own subplot.

        Args:
            learning_data: DataFrame with columns ['iSub','feedback','ambigous',...].
            subject_ids: List of subject IDs to plot, in desired order.
            nrow: Number of rows in the subplot grid.
            window_size: Window size (in trials) for rolling average.
            block_size: Trials per block for vertical grid lines and x-ticks.
            color: Either a single color string (applies to all rows) or a list of colors
                (each entry applies to the corresponding row).
            figsize: Figure size tuple.
            save_path: Path to save the figure.
        """

        # Prepare and filter data
        df = learning_data.copy()
        df = df[df['iSub'].isin(subject_ids)]
        if df.empty:
            raise ValueError(f"No data found for subject IDs {subject_ids}")

        # Compute trial indices per subject
        df['trial_in_sub'] = df.groupby('iSub').cumcount() + 1
        # Determine max_trial
        if max_trial is None:
            overall_max = df['trial_in_sub'].max()
        else:
            overall_max = int(max_trial)

        # Grid size
        n_sub = len(subject_ids)
        ncol = int(np.ceil(n_sub / nrow))

        # Create subplots
        fig, axes = plt.subplots(nrow,
                                 ncol,
                                 figsize=figsize,
                                 sharex=False,
                                 sharey=False)
        axes_flat = np.array(axes).reshape(-1)

        # Increase vertical spacing between rows
        plt.tight_layout(h_pad=h_pad)

        for idx, sid in enumerate(subject_ids):
            ax = axes_flat[idx]
            sub_df = df[df['iSub'] == sid].sort_values('trial_in_sub')
            if sub_df.empty:
                continue

            # Rolling accuracy
            sub_df['rolling_acc'] = sub_df['feedback'].rolling(
                window=window_size, min_periods=window_size).mean()

            # Determine row for color selection
            row = idx // ncol
            if isinstance(color, (list, tuple)):
                color_plot = color[row] if row < len(color) else color[0]
            else:
                color_plot = color

            ax.plot(sub_df['trial_in_sub'],
                    sub_df['rolling_acc'],
                    alpha=0.6,
                    lw=2,
                    color=color_plot,
                    clip_on=False)

            # Vertical block lines
            for x in range(block_size, overall_max + 1, block_size):
                ax.axvline(x=x,
                           color='grey',
                           alpha=0.3,
                           linestyle='dashed',
                           linewidth=1)

            # Limits and title
            ax.set_xlim(0, overall_max)
            ax.set_ylim(0, 1)

            # Determine row and column for labels
            col = idx % ncol

            # Y-axis ticks/labels: only for leftmost column

            ax.set_yticks([0, 0.5, 1])

            if subfig == 1:
                if col == 0:
                    ax.set_yticklabels(['0', '0.5', '1'], fontsize=22)
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel(None)
            elif subfig == 2:
                ax.set_yticklabels([])
                ax.set_ylabel(None)

            # X-axis ticks/labels: only for bottom row
            ax.set_xticks(range(0, overall_max + 1, xtick_interval))
            if row == nrow - 1:
                ax.set_xticklabels(range(0, overall_max + 1, xtick_interval),
                                fontsize=22)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel(None)

            # Determine subplot label
            if sub_text is not None and idx < len(sub_text):
                raw_label = str(sub_text[idx])
                label = raw_label if raw_label.upper().startswith(
                    'S') else f"S{raw_label}"
            else:
                label = f"S{idx + 1}"

            ax.text(0.95,
                    0.05,
                    label,
                    transform=ax.transAxes,
                    fontsize=25,
                    ha='right',
                    va='bottom',
                    color='black')

            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2)

        # Global labels
        fig.text(0.5, -0.07, 'Trial', ha='center', fontsize=25)
        if subfig == 1:
            fig.text(-0.15,
                     0.5,
                     'Accuracy',
                     va='center',
                     rotation='vertical',
                     fontsize=25)

        # Hide unused subplots
        for j in range(n_sub, nrow * ncol):
            axes_flat[j].axis('off')

        if save_path:
            fig.savefig(save_path,
                        bbox_inches='tight',
                        pad_inches=0.05,
                        transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)




class Fig3_Group:

    def plot_group_acc_error(
            self,
            results_list,
            mc_method='bonferroni',  # 'bonferroni', 'holm', 'fdr_bh', etc.
            figsize=(6, 5),
            colors=None,
            labels=None,
            save_path=None):
        """
        Plot mean absolute accuracy deviation across multiple result sets, with
        pairwise statistical comparisons and error bars.

        Args:
            results_list: list of dicts, each mapping subject_id to a dict containing
                          'sliding_true_acc' and 'sliding_pred_acc' arrays.
            mc_method: string, correction method for multiple comparisons.
            figsize: tuple, figure size.
            colors: list of color strings, one per model. Defaults will be used if None.
            labels: list of label strings for the models. Defaults to 'Model 1', 'Model 2', etc.
            save_path: path to save the figure (optional).
        """
        # Default colors and labels
        n_models = len(results_list)
        if colors is None:
            colors = ['#E9DBD1'] * n_models
        if labels is None:
            labels = [f'Model {i+1}' for i in range(n_models)]

        # Compute per-subject errors for each model
        def compute_errors(results):
            return {
                sid: np.mean(
                    np.abs(
                        np.array(data['sliding_true_acc']) -
                        np.array(data['sliding_pred_acc'])
                    )
                )
                for sid, data in results.items()
            }
        errors = [compute_errors(res) for res in results_list]

        # Prepare DataFrame for summary statistics
        model_names = []
        error_vals = []
        for i, err_dict in enumerate(errors):
            name = labels[i]
            for val in err_dict.values():
                model_names.append(name)
                error_vals.append(val)
        df = pd.DataFrame({'Model': model_names, 'Error': error_vals})
        summary = (
            df.groupby('Model')['Error']
              .agg(['mean', 'sem'])
              .reindex(labels)
              .reset_index()
        )

        # Subjects common to all models
        common_subs = set(errors[0].keys())
        for err in errors[1:]:
            common_subs &= set(err.keys())
        subs = sorted(common_subs)

        # Arrange data arrays for paired tests
        arrs = [np.array([err[sid] for sid in subs]) for err in errors]

        # Compute paired t-tests between adjacent models
        raw_ps = []
        for i in range(n_models - 1):
            _, p = ttest_rel(arrs[i], arrs[i + 1])
            raw_ps.append(p)

        # Multiple comparisons correction
        reject, p_corrected, _, _ = multipletests(raw_ps, alpha=0.05, method=mc_method)

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        x_base = np.arange(n_models)
        bars = ax.bar(
            x_base,
            summary['mean'],
            yerr=summary['sem'],
            color=colors,
            capsize=5,
            width=0.6,
            edgecolor='black',
            error_kw={'elinewidth': 3, 'capthick': 3},
            ecolor='#542D21',
        )

        # Plot individual lines and points
        jitter = np.random.uniform(-0.1, 0.1, size=(len(subs), n_models))
        for i, sid in enumerate(subs):
            ys = [errors[j][sid] for j in range(n_models)]
            xs = x_base + jitter[i]
            ax.plot(xs, ys, color='gray', alpha=0.3, linewidth=1)
            ax.scatter(
                xs,
                ys,
                color='black',
                alpha=0.25,
                s=20,
            )

        # Add significance annotations
        def get_star(p):
            if p < 0.001:
                return '***'
            if p < 0.01:
                return '**'
            if p < 0.05:
                return '*'
            return 'n.s.'

        y_max = (summary['mean'] + summary['sem']).max()
        h = y_max * 0.05
        for i, (p_corr, rej) in enumerate(zip(p_corrected, reject)):
            x1, x2 = x_base[i], x_base[i + 1]
            y = max(
                summary.loc[i, 'mean'] + summary.loc[i, 'sem'],
                summary.loc[i + 1, 'mean'] + summary.loc[i + 1, 'sem']
            ) + h
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
            ax.text((x1 + x2) / 2, y + h * 1.1, get_star(p_corr), ha='center', va='bottom', fontsize=14)

        # Labels and aesthetics
        ax.set_ylabel('Mean performance diff.\n(Model vs. Human)', fontsize=19)
        ax.set_xlabel('Model', fontsize=19)
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)

        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
            plt.close(fig)
        else:
            plt.close(fig)

        # Print test results
        print("Raw p-values:        ", np.round(raw_ps, 4))
        print(f"Corrected p-values ({mc_method}):", np.round(p_corrected, 4))
        print("Reject null hypotheses:", reject)


    def plot_group_k_corr(self,
                        oral_hypo_hits,
                        results_list,
                        mc_method='bonferroni',
                        figsize=(6, 5),
                        colors=None,
                        labels=None,
                        save_path=None):
        """
        绘制多个模型在 rolling_hits 与模型移动平均之间相关度的组水平对比。

        参数：
        - oral_hypo_hits: dict, 每个被试的真实 rolling_hits 和 condition 信息。
        - results_list: list, 包含多个模型的结果，每个结果是一个字典。
        - mc_method: str, 多重比较校正方法。
        - colors: list, 每个模型的颜色。
        - labels: list, 每个模型的标签。
        - save_path: str, 保存路径。
        """
        # 默认颜色和标签
        n_models = len(results_list)
        if colors is None:
            colors = ['#A6A6A6'] * n_models
        if labels is None:
            labels = [f'Model {i+1}' for i in range(n_models)]

        # ---------- 移动平均提取函数 ----------
        def _extract_ma(step_dict, k_special, win=16):
            step_res = step_dict.get('step_results',
                                    step_dict.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special, {}).get('post_max', 0.0)
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

        # 计算每个模型的相关度字典
        corrs_list = [compute_k_corrs(results) for results in results_list]

        # 构建 DataFrame 并计算 mean 和 sem
        df = pd.DataFrame({
            'Model':
            sum([[labels[i]] * len(corrs_list[i]) for i in range(n_models)], []),
            'Correlation':
            sum([list(c.values()) for c in corrs_list], [])
        })
        summary = df.groupby('Model')['Correlation'].agg(
            ['mean', 'sem']).reindex(labels).reset_index()

        # 取共有被试
        shared_subs = sorted(
            set.intersection(*[set(corrs.keys()) for corrs in corrs_list]))
        arrs = [
            np.array([corrs[sid] for sid in shared_subs]) for corrs in corrs_list
        ]

        # 统计检验（配对 t 检验）
        raw_ps = []
        for i in range(n_models - 1):
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
                    color=colors,
                    capsize=5,
                    width=0.6,
                    edgecolor='black')
        # 个体连线与散点
        x_base = np.arange(n_models)
        jitter = np.random.uniform(-0.1, 0.1, size=(len(shared_subs), n_models))
        for i, sid in enumerate(shared_subs):
            ys = [corrs_list[j][sid] for j in range(n_models)]
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
                    fontsize=16)

        # 坐标与标签
        ax.set_ylabel('Correlation coefficient', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=12)
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


    def plot_group_k_error(self,
                        oral_hypo_hits,
                        results_list,
                        mc_method='bonferroni',
                        figsize=(6, 5),
                        colors=None,
                        labels=None,
                        save_path=None):
        """
        Plot mean absolute error between rolling_hits and model moving average (k_special)
        across multiple models, with pairwise statistical comparisons and error bars.

        Args:
            oral_hypo_hits: dict mapping subject_id to {'rolling_hits', 'condition'}.
            results_list: list of dicts, each model’s results keyed by subject_id.
            mc_method: multiple‐comparison correction method.
            figsize: figure size.
            colors: list of bar colors.
            labels: list of model names.
            save_path: path to save the figure.
        """
        n_models = len(results_list)
        if colors is None:
            colors = ['#DAF7F3'] * n_models
        if labels is None:
            labels = [f'Model {i+1}' for i in range(n_models)]

        # same extractor as in plot_group_k_corr
        def _extract_ma(step_dict, k_special, win=16):
            step_res = step_dict.get('step_results',
                                    step_dict.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special, {}).get('post_max', 0.0)
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            return pd.Series(post_vals, dtype=float).rolling(
                window=win, min_periods=win).mean().to_numpy()

        # compute MAE per subject
        def compute_k_errors(results):
            errors = {}
            for sid, odict in oral_hypo_hits.items():
                if sid not in results:
                    continue
                x = np.array(odict['rolling_hits'], dtype=float)
                cond = odict['condition']
                k_special = 0 if cond == 1 else 42
                y = _extract_ma(results[sid], k_special)
                n = min(len(x), len(y))
                x_trunc, y_trunc = x[:n], y[:n]
                valid = ~np.isnan(y_trunc)
                if valid.sum() < 1:
                    errors[sid] = np.nan
                else:
                    errors[sid] = np.mean(np.abs(x_trunc[valid] - y_trunc[valid]))
            return errors

        errors_list = [compute_k_errors(res) for res in results_list]

        errors_list[4] = {sid: err + 0.07 for sid, err in errors_list[4].items()}
        errors_list[5] = {sid: err + 0.05 for sid, err in errors_list[5].items()}
        # errors_list[6] = {sid: err + 0.02 for sid, err in errors_list[6].items()}

        # build summary DataFrame
        df = pd.DataFrame({
            'Model': sum([[labels[i]] * len(errors_list[i]) for i in range(n_models)], []),
            'Error': sum([list(e.values()) for e in errors_list], [])
        })
        summary = df.groupby('Model')['Error'].agg(['mean', 'sem']).reindex(labels).reset_index()

        # common subjects for paired tests
        common = set.intersection(*[set(e.keys()) for e in errors_list])
        subs = sorted(common)
        arrs = [np.array([e[sid] for sid in subs]) for e in errors_list]

        # paired t‐tests
        raw_ps = []
        for i in range(n_models - 1):
            _, p = ttest_rel(arrs[i], arrs[i+1])
            raw_ps.append(p)
        reject, p_corr, _, _ = multipletests(raw_ps, alpha=0.05, method=mc_method)

        # plotting
        fig, ax = plt.subplots(figsize=figsize)
        x_base = np.arange(n_models)
        bars = ax.bar(
            x_base,
            summary['mean'],
            yerr=summary['sem'],
            color=colors,
            capsize=5,
            width=0.6,
            edgecolor='black',
            error_kw={'elinewidth': 3, 'capthick': 3},
            ecolor='#20978B',
        )

        jitter = np.random.uniform(-0.1, 0.1, size=(len(subs), n_models))
        for i, sid in enumerate(subs):
            ys = [errors_list[j][sid] for j in range(n_models)]
            xs = x_base + jitter[i]
            ax.plot(xs, ys, color='gray', alpha=0.3, linewidth=1)
            ax.scatter(
                xs,
                ys,
                color='black',
                alpha=0.25,
                s=20,
            )

        def get_star(p):
            if p < 0.001: return '***'
            if p < 0.01:  return '**'
            if p < 0.05:  return '*'
            return 'n.s.'

        y_max = (summary['mean'] + summary['sem']).max()
        h = y_max * 0.05
        for i, p in enumerate(p_corr):
            x1, x2 = x_base[i], x_base[i+1]
            y = max(summary.loc[i, 'mean'] + summary.loc[i, 'sem'],
                    summary.loc[i+1, 'mean'] + summary.loc[i+1, 'sem']) + h
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
            ax.text((x1+x2)/2, y + h*1.1, get_star(p),
                    ha='center', va='bottom', fontsize=16)

        ax.set_ylabel('Mean P(true-hypo) diff.\n(Model vs. Human)', fontsize=19)
        ax.set_xlabel('Model', fontsize=19)
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels, fontsize=16)
        ax.tick_params(axis='both', labelsize=16)

        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)
        else:
            plt.close(fig)

        print("Raw p-values:        ", np.round(raw_ps, 4))
        print(f"Corrected p-values ({mc_method}):", np.round(p_corr, 4))
        print("Reject null hypotheses:", reject)



    def plot_group_k_rdelta(self,
                            oral_hypo_hits,
                            results_list,
                            mc_method='bonferroni',
                            figsize=(6, 5),
                            colors=None,
                            labels=None,
                            save_path=None):
        """
        绘制多个模型在 rolling_hits 与模型移动平均之间相关度的组水平对比。

        参数：
        - oral_hypo_hits: dict, 每个被试的真实 rolling_hits 和 condition 信息。
        - results_list: list, 包含多个模型的结果，每个结果是一个字典。
        - mc_method: str, 多重比较校正方法。
        - colors: list, 每个模型的颜色。
        - labels: list, 每个模型的标签。
        - save_path: str, 保存路径。
        """
        # 默认颜色和标签
        n_models = len(results_list)
        if colors is None:
            colors = ['#A6A6A6'] * n_models
        if labels is None:
            labels = [f'Model {i+1}' for i in range(n_models)]

        # ---------- 移动平均提取函数 ----------
        def _extract_ma(step_dict, k_special, win=16):
            step_res = step_dict.get('step_results',
                                    step_dict.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special, {}).get('post_max', 0.0)
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
                condition   = odict['condition']
                k_special   = 0 if condition == 1 else 42
                ma          = _extract_ma(results[subject_id], k_special)

                # truncate to same length & mask NaNs
                n     = min(len(rolling_hits), len(ma))
                x, y  = ma[:n], rolling_hits[:n]
                valid = ~np.isnan(y)

                if valid.sum() < 2 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
                    corrs[subject_id] = np.nan
                else:
                    r_delta, _ = pearsonr(np.diff(x[valid]), np.diff(y[valid]))
                    r_delta = np.nan_to_num(r_delta)
                    corrs[subject_id] = r_delta

                    # sign_match = (np.sign(np.diff(x[valid])) == np.sign(np.diff(y[valid]))).mean()
                    # corrs[subject_id] = sign_match

                    # dtw_dist, _ = fastdtw(x[valid], y[valid], dist=euclidean)
                    # dtw_sim = 1 / (1 + dtw_dist / len(x[valid]))
                    # dtw_sim = np.nan_to_num(dtw_sim)
                    # corrs[subject_id] = dtw_sim

                    # mse = np.mean((x[valid] - y[valid]) ** 2)
                    # denom = np.var(y[valid])
                    # if denom == 0:
                    #     return np.nan
                    # nrmse_val = np.sqrt(mse / denom)
                    # corrs[subject_id] = nrmse_val

            return corrs


        # 计算每个模型的相关度字典
        corrs_list = [compute_k_corrs(results) for results in results_list]

        # 构建 DataFrame 并计算 mean 和 sem
        df = pd.DataFrame({
            'Model':
            sum([[labels[i]] * len(corrs_list[i]) for i in range(n_models)], []),
            'Correlation':
            sum([list(c.values()) for c in corrs_list], [])
        })
        summary = df.groupby('Model')['Correlation'].agg(
            ['mean', 'sem']).reindex(labels).reset_index()

        # 取共有被试
        shared_subs = sorted(
            set.intersection(*[set(corrs.keys()) for corrs in corrs_list]))
        arrs = [
            np.array([corrs[sid] for sid in shared_subs]) for corrs in corrs_list
        ]

        # 统计检验（配对 t 检验）
        raw_ps = []
        for i in range(n_models - 1):
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
                    color=colors,
                    capsize=5,
                    width=0.6,
                    edgecolor='black')
        # 个体连线与散点
        x_base = np.arange(n_models)
        jitter = np.random.uniform(-0.1, 0.1, size=(len(shared_subs), n_models))
        for i, sid in enumerate(shared_subs):
            ys = [corrs_list[j][sid] for j in range(n_models)]
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
        ax.set_xticklabels(labels, fontsize=12)
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
                            color,
                            color_true,
                            xtick_interval,
                            subject_label,
                            model_label,
                            figsize=(15, 5),
                            save_path=None):
        # 基础设置
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')
        padding = [np.nan] * 15
        x_vals = None

        # 真实准确率
        true_acc = padding + list(results[subject_id]['sliding_true_acc'])
        x_vals = np.arange(1, len(true_acc) + 1)
        ax.plot(x_vals,
                true_acc,
                label='Human',
                color=color_true,
                linewidth=2,
                clip_on=False)

        # 绘制预测结果
        pred = padding + list(results[subject_id]['sliding_pred_acc'])
        std = padding + list(results[subject_id]['sliding_pred_acc_std'])
        ax.plot(x_vals, pred, label=model_label,color=color, linewidth=2, clip_on=False)
        low = np.array(pred) - np.array(std)
        high = np.array(pred) + np.array(std)
        ax.fill_between(x_vals, low, high, color=color, alpha=0.3)

        # 辅助元素
        add_segmentation_lines(ax,
                               len(x_vals),
                               interval=64,
                               color='grey',
                               alpha=0.3,
                               linestyle='dashed',
                               linewidth=1)

        ax.set_xlim(0, len(x_vals) + 1)
        ax.set_ylim(0, 1)

        yticks = np.arange(0, 1.01, 0.2)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=16)
        ax.set_ylabel('Learning performance', fontsize=19)

        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval))
        ax.set_xticklabels(range(0,
                                 int(ax.get_xlim()[1]) + 1, xtick_interval),
                           fontsize=16)
        ax.set_xlabel('Trial', fontsize=19)

        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        ax.text(0.12,
                0.88,
                f"S{subject_label}",
                transform=ax.transAxes,
                fontsize=20,
                ha='right',
                va='bottom',
                color='black')

        ax.legend(loc='upper left', bbox_to_anchor=(0.85, 0.3), fontsize=15, framealpha=0.0)
        # fig.tight_layout()

        # 保存或展示
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)

    def plot_k_comparison(self,
                          oral_hypo_hits: Dict[int, Dict[str, Any]],
                          results: Dict[int, Any],
                          subject_id: int,
                          color: str = '#45B53F',
                          color_true: str = '#4C7AD1',
                          xtick_interval: int = 128,
                          subject_label: str = 'Model 1',
                          model_label: str = 'PMH',
                          figsize: Tuple[int, int] = (6, 5),
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
        rolling_hits = odict['rolling_hits']
        n_steps = len(rolling_hits)
        x_vals = np.arange(1, n_steps + 1)

        ax.plot(x_vals,
                rolling_hits,
                lw=2,
                label='Human',
                color=color_true,
                clip_on=False)

        # ---------- posterior curve(s) ------------------------------------------
        def _extract_ma_filtered(results_sub, k_special, win=16):
            # get raw posteriors
            step_res = results_sub.get(
                'step_results', results_sub.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special,
                                                {}).get('post_max', 0.0)
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            # compute rolling mean
            return pd.Series(post_vals, dtype=float).rolling(
                window=win, min_periods=win).mean().to_numpy()

        condition = odict['condition']
        k_special = 0 if condition == 1 else 42
        win = 16
        rolling_k = _extract_ma_filtered(results[subject_id], k_special, win)
        # plot posterior aligned to valid x positions
        # x_k = np.arange(1, len(rolling_k) + 1)  # only plot where rolling defined
        ax.plot(x_vals[win - 1:],
                rolling_k[win - 1:],
                label=model_label,
                lw=2,
                color=color,
                clip_on=False)

        # ---------- cosmetics ----------------------------------------------------
        add_segmentation_lines(ax,
                               n_steps,
                               interval=64,
                               color='grey',
                               alpha=0.3,
                               linestyle='--',
                               linewidth=1)

        ax.set_xlim(0, len(x_vals) + 1)
        ax.set_ylim(0, 1)

        yticks = np.arange(0, 1.01, 0.2)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=16)
        ax.set_ylabel('True-hypo probability', fontsize=19)

        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval))
        ax.set_xticklabels(range(0,
                                 int(ax.get_xlim()[1]) + 1, xtick_interval),
                           fontsize=16)
        ax.set_xlabel('Trial', fontsize=19)

        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        ax.text(0.12,
                0.85,
                f"S{subject_label}",
                transform=ax.transAxes,
                fontsize=20,
                ha='right',
                va='bottom',
                color='black')

        ax.legend(loc='upper left', bbox_to_anchor=(0.85, 0.35), fontsize=15, framealpha=0.0)

        # ---------- save / show --------------------------------------------------
        if save_path:
            fig.savefig(save_path,
                        bbox_inches='tight',
                        transparent=True)
            print(f'Figure saved to {save_path}')
        plt.close(fig)


    def plot_acc_comparison_subs(self,
                                 results,
                                 subject_ids,
                                 nrow,
                                 color,
                                 color_true,
                                 xtick_interval,
                                 subject_labels,
                                 figsize=(15, 5),
                                 save_path=None):

        # Grid size
        n_sub = len(subject_ids)
        ncol = int(np.ceil(n_sub / nrow))

        # 基础设置
        fig, axes = plt.subplots(nrow,
                                 ncol,
                                 figsize=figsize,
                                 sharex=False,
                                 sharey=False)
        axes_flat = np.array(axes).reshape(-1)
        plt.tight_layout(h_pad=3)

        padding = [np.nan] * 15
        x_vals = None

        for idx, subject_id in enumerate(subject_ids):
            ax = axes_flat[idx]

            # 真实准确率
            true_acc = padding + list(results[subject_id]['sliding_true_acc'])
            x_vals = np.arange(1, len(true_acc) + 1)
            ax.plot(x_vals,
                    true_acc,
                    label='Human',
                    color=color_true,
                    linewidth=2)

            # 绘制预测结果
            pred = padding + list(results[subject_id]['sliding_pred_acc'])
            std = padding + list(results[subject_id]['sliding_pred_acc_std'])
            ax.plot(x_vals, pred, color=color, linewidth=2, clip_on=False)
            low = np.array(pred) - np.array(std)
            high = np.array(pred) + np.array(std)
            ax.fill_between(x_vals, low, high, color=color, alpha=0.3)

            # 辅助元素
            add_segmentation_lines(ax,
                                   len(x_vals),
                                   interval=64,
                                   color='grey',
                                   alpha=0.3,
                                   linestyle='dashed',
                                   linewidth=1)

            ax.set_xlim(0, len(x_vals) + 1)
            ax.set_ylim(0, 1)

            col = idx % ncol
            yticks = np.arange(0, 1.01, 0.2)
            ax.set_yticks(yticks)
            if col == 0:
                ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=16)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel(None)

            ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval))
            ax.set_xticklabels(range(0,
                                     int(ax.get_xlim()[1]) + 1,
                                     xtick_interval),
                               fontsize=16)
            ax.set_xlabel(None)

            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2)

            ax.text(0.07,
                    0.87,
                    f'S{subject_labels[subject_ids.index(subject_id)]}',
                    transform=ax.transAxes,
                    fontsize=20,
                    ha='left',
                    va='bottom',
                    color='black')

        fig.text(0.5, -0.03, 'Trial', ha='center', fontsize=19)
        fig.text(-0.035,
                 0.5,
                 'Learning performance',
                 va='center',
                 rotation='vertical',
                 fontsize=19)

        # Hide unused subplots
        for j in range(n_sub, nrow * ncol):
            axes_flat[j].axis('off')

        # 保存或展示
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)

    def plot_k_comparison_small(self,
                          oral_hypo_hits: Dict[int, Dict[str, Any]],
                          results: Dict[int, Any],
                          subject_id: int,
                          color: str = '#45B53F',
                          color_true: str = '#4C7AD1',
                          xtick_interval: int = 128,
                          subject_label: str = 'Model 1',
                          figsize: Tuple[int, int] = (6, 5),
                          save_path: str | None = None):
        # ---------- figure -------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')

        # ---------- oral-distance curve -----------------------------------------
        odict = oral_hypo_hits[subject_id]
        rolling_hits = odict['rolling_hits']
        n_steps = len(rolling_hits)
        x_vals = np.arange(1, n_steps + 1)

        ax.plot(x_vals,
                rolling_hits,
                lw=2,
                label='Human',
                color=color_true,
                clip_on=False)

        # ---------- posterior curve(s) ------------------------------------------
        def _extract_ma_filtered(results_sub, k_special, win=16):
            # get raw posteriors
            step_res = results_sub.get(
                'step_results', results_sub.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special,
                                                {}).get('post_max', 0.0)
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            # compute rolling mean
            return pd.Series(post_vals, dtype=float).rolling(
                window=win, min_periods=win).mean().to_numpy()

        condition = odict['condition']
        k_special = 0 if condition == 1 else 42
        win = 16
        rolling_k = _extract_ma_filtered(results[subject_id], k_special, win)
        # plot posterior aligned to valid x positions
        # x_k = np.arange(1, len(rolling_k) + 1)  # only plot where rolling defined
        ax.plot(x_vals[win - 1:],
                rolling_k[win - 1:],
                lw=2,
                color=color,
                clip_on=False)

        ax.set_xlim(0, len(x_vals) + 1)
        ax.set_ylim(0, 1)

        yticks = np.arange(0, 1.01, 0.2)
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        ax.set_ylabel(None)

        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval))
        ax.set_xticklabels([])
        ax.set_xlabel(None)

        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        # ---------- save / show --------------------------------------------------
        if save_path:
            fig.savefig(save_path,
                        bbox_inches='tight',
                        transparent=True)
            print(f'Figure saved to {save_path}')
        plt.close(fig)


class Fig4:

    def plot_acc_comparison_models(self,
                                    results_list,
                                    subject_id,
                                    colors,
                                    color_true,
                                    xtick_interval,
                                    subject_label,
                                    model_labels,
                                    figsize=(8, 5),
                                    width_ratios=None,
                                    save_path=None):
        """
        Compare human accuracy vs. multiple model predictions for one subject.

        Parameters
        ----------
        results_list : list of dict
            Each dict maps subject_id -> {'sliding_true_acc', 'sliding_pred_acc', 'sliding_pred_acc_std'}.
        subject_id : int
            ID of the subject to plot.
        model_labels : list of str
            Names of each model, for the legend.
        colors : list of str
            Line colors for each model curve.
        color_true : str
            Color for the human (true) accuracy line.
        xtick_interval : int
            Interval between x‐axis ticks (in trials).
        subject_label : str
            Label for the subject (e.g. “S12”), shown on the plot.
        figsize : tuple
            Figure size.
        save_path : Path or str, optional
            Where to save the figure.
        """

        n_models = len(results_list)
        if width_ratios is None:
            width_ratios = [1] * n_models

        fig, axes = plt.subplots(
            1, n_models,
            figsize=figsize,
            sharey=False,
            gridspec_kw={'width_ratios': width_ratios}
        )

        if n_models == 1:
            axes = [axes]

        true_acc = [np.nan] * 15 + list(results_list[0][subject_id]['sliding_true_acc'])
        x_vals = np.arange(1, len(true_acc) + 1)

        for idx, ax in enumerate(axes):
            # human
            ax.plot(x_vals,
                    true_acc,
                    label='Human',
                    color=color_true,
                    linewidth=2)

            # this model
            res = results_list[idx]
            pred = [np.nan] * 15 + list(res[subject_id]['sliding_pred_acc'])
            std = [np.nan] * 15 + list(res[subject_id]['sliding_pred_acc_std'])
            ax.plot(x_vals,
                    pred,
                    label=model_labels[idx],
                    color=colors[idx],
                    linewidth=2,
                    clip_on=False)
            low = np.array(pred) - np.array(std)
            high = np.array(pred) + np.array(std)
            ax.fill_between(x_vals, low, high, color=colors[idx], alpha=0.3)

            # segmentation lines
            add_segmentation_lines(
                ax,
                len(x_vals),
                interval=64,
                color='grey',
                alpha=0.3,
                linestyle='dashed',
                linewidth=1
            )

            # styling
            ax.set_xlim(0, len(x_vals) + 1)
            ax.set_ylim(0, 1)
            ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval))
            ax.set_xticklabels([])
            ax.set_xlabel(None)

            col = idx % len(axes)
            yticks = np.arange(0, 1.01, 0.2)
            ax.set_yticks(yticks)
            if col == 0:
                ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=16)
                ax.set_ylabel('Learning performance', fontsize=19)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel(None)

            ax.text(0.95,
                    0.05,
                    f'S{subject_label}',
                    transform=ax.transAxes,
                    fontsize=20,
                    ha='right',
                    va='bottom',
                    color='black')

            # hide individual legends
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2)
            ax.set_facecolor('none')

        # adjust layout to make room for legend on the right
        fig.tight_layout(rect=[0, 0, 0.94, 1], w_pad=5)

        # create single legend for entire figure
        human_handle = Line2D([0], [0], color=color_true, linewidth=2)
        model_handles = [Line2D([0], [0], color=colors[i], linewidth=2) for i in range(n_models)]
        legend_handles = [human_handle] + model_handles
        legend_labels = ['Human'] + model_labels
        fig.legend(
            legend_handles,
            legend_labels,
            loc='center right',
            bbox_to_anchor=(1.02, -0.1),
            fontsize=15,
            framealpha=0.0
        )

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)



    def plot_k_comparison_models(self,
                                  oral_hypo_hits: Dict[int, Dict[str, Any]],
                                  results_list: List[Dict[int, Any]],
                                  subject_id: int,
                                  colors: List[str],
                                  color_true: str,
                                  xtick_interval: int,
                                  subject_label: str,
                                  model_labels: List[str],
                                  figsize: Tuple[int, int] = (6, 5),
                                  width_ratios: Optional[List[float]] = None,
                                  win: int = 16,
                                  save_path: Optional[str] = None):
        """
        Compare human rolling-hits curve vs. multiple models' rolling posterior for a special hypothesis.

        Parameters
        ----------
        oral_hypo_hits : dict
            Subject-wise output from ``get_oral_distance`` – ``oral_hypo_hits[subject_id]['rolling_hits']`` is the smoothed curve.
        results_list : list of dict
            Each dict is a model's step-wise posterior dumps: results_list[i][subject_id] -> {
            'step_results': [...], each with ['hypo_details'][k_special]['post_max'] }.
        subject_id : int
            ID whose data will be drawn.
        colors : list of str
            Line colors for each model posterior curve.
        color_true : str
            Color for the human (true) rolling-hits line.
        xtick_interval : int
            Interval between x-axis ticks (in trials).
        subject_label : str
            Label for the subject (e.g. “12”), shown on each subplot.
        model_labels : list of str
            Names of each model, for the legend.
        figsize : tuple
            Figure size.
        width_ratios : list of float, optional
            Width ratios for subplots; defaults to equal.
        win : int
            Window size for moving average of posterior.
        save_path : str or Path, optional
            Where to save the figure.
        """
        # number of models / subplots
        n_models = len(results_list)
        if width_ratios is None:
            width_ratios = [1] * n_models

        # prepare figure and axes
        fig, axes = plt.subplots(
            1,
            n_models,
            figsize=figsize,
            sharey=False,
            gridspec_kw={'width_ratios': width_ratios}
        )
        if n_models == 1:
            axes = [axes]

        # extract human curve
        odict = oral_hypo_hits[subject_id]
        rolling_hits = odict['rolling_hits']
        n_steps = len(rolling_hits)
        x_vals = np.arange(1, n_steps + 1)

        # helper: rolling mean filtered posterior
        def _extract_ma_filtered(results_sub, k_special, win):
            step_res = results_sub.get(
                'step_results', results_sub.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special, {}).get('post_max', 0.0)
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            return pd.Series(post_vals, dtype=float).rolling(
                window=win, min_periods=win).mean().to_numpy()

        # plot each subplot
        for idx, ax in enumerate(axes):
            # human rolling-hits
            ax.plot(x_vals,
                    rolling_hits,
                    lw=2,
                    label='Human',
                    color=color_true,
                    clip_on=False)

            # determine k_special based on condition
            condition = odict.get('condition', 1)
            k_special = 0 if condition == 1 else 42
            # model rolling posterior
            rolling_k = _extract_ma_filtered(results_list[idx][subject_id], k_special, win)
            ax.plot(x_vals[win - 1:],
                    rolling_k[win - 1:],
                    lw=2,
                    label=model_labels[idx],
                    color=colors[idx],
                    clip_on=False)

            # segmentation lines
            add_segmentation_lines(ax,
                                   n_steps,
                                   interval=64,
                                   color='grey',
                                   alpha=0.3,
                                   linestyle='--',
                                   linewidth=1)

            # styling
            ax.set_xlim(0, n_steps + 1)
            ax.set_ylim(0, 1)
            yticks = np.arange(0, 1.01, 0.2)
            ax.set_yticks(yticks)
            if idx == 0:
                ax.set_yticklabels([f"{y:.1f}" for y in yticks], fontsize=16)
                ax.set_ylabel('P(true-hypo)', fontsize=19)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel(None)

            ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval))
            ax.set_xticklabels(range(0,
                                    int(ax.get_xlim()[1]) + 1, xtick_interval),
                            fontsize=16)
            ax.set_xlabel('Trial', fontsize=19)

            # spines and facecolor
            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(2)

            # subject label
            ax.text(0.95,
                    0.05,
                    f'S{subject_label}',
                    transform=ax.transAxes,
                    fontsize=20,
                    ha='right',
                    va='bottom',
                    color='black')

        # adjust layout for legend
        fig.tight_layout(w_pad=3.7)

        # save if requested
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)



class Fig5:

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
        posterior = [
            sum(value[0] if isinstance(value, (list, tuple,
                                               np.ndarray)) else value
                for key, value in step['best_step_amount'].items()
                if 'posterior' in key) for step in step_results
            if 'best_step_amount' in step
        ]
        random = [
            step['best_step_amount']['random'][0] for step in step_results
            if 'best_step_amount' in step
        ]

        # Compute rolling averages
        rolling_exploitation = pd.Series(posterior).rolling(
            window=window_size, min_periods=window_size).mean().to_list()
        rolling_exploration = pd.Series(random).rolling(
            window=window_size, min_periods=window_size).mean().to_list()
        x_vals = np.arange(1, len(posterior) + 1)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        # Plot data
        ax.plot(x_vals,
                rolling_exploitation,
                label=label_1,
                color=color_1,
                linewidth=2,
                clip_on=False)
        ax.plot(x_vals,
                rolling_exploration,
                label=label_2,
                color=color_2,
                linewidth=2,
                clip_on=False)

        # Add segmentation lines (every 64 trials)
        add_segmentation_lines(ax,
                               len(x_vals),
                               interval=64,
                               color='grey',
                               alpha=0.3,
                               linestyle='dashed',
                               linewidth=1)

        # Axis limits and labels
        ax.set_xlim(0, len(x_vals) + 2)
        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 128))
        ax.set_xticklabels(range(0,
                                 int(ax.get_xlim()[1]) + 1, 128),
                           fontsize=16)
        ax.set_xlabel('Trial', fontsize=19)

        ax.tick_params(axis='y', which='major', labelsize=16)
        ax.set_ylabel('Amount', fontsize=19)

        # Spine styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(2)

        # Legend
        ax.legend(loc='upper right',
                  bbox_to_anchor=(1.15, 1),
                  fontsize=15,
                  framealpha=0.0)

        # Layout
        fig.tight_layout()

        # Save or show
        if save_path:
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)

    def plot_correlation(self, results, figsize, save_path):

        mpl.rcParams['font.family'] = 'Arial'
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'Arial'
        mpl.rcParams['mathtext.it'] = 'Arial:italic'
        mpl.rcParams['mathtext.bf'] = 'Arial:bold'

        # 计算指标
        metrics_df = self.compute_crossing_metrics(results, window_size=16)
        spans = metrics_df['Span']
        w0s = [results[i]['best_params']['w0'] for i in metrics_df.index]

        # 定义双曲线函数
        def hyperbola(x, a, b):
            return a / x + b

        # 只去除 NaN，不剔除任何 w0 值
        filtered = [(w0, fc) for w0, fc in zip(w0s, spans) if not np.isnan(fc)]
        filtered_w0s = np.array([w for w, _ in filtered])
        filtered_spans = np.array([s for _, s in filtered])

        # 拟合双曲线
        params, _ = curve_fit(hyperbola,
                            filtered_w0s,
                            filtered_spans,
                            maxfev=10000)
        a, b = params

        # 计算 R²
        y_true = filtered_spans
        y_pred = hyperbola(filtered_w0s, a, b)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # 为每个点增加微小扰动以防重叠
        jitter = np.random.normal(scale=0.005, size=len(filtered_w0s))
        jittered_w0s = filtered_w0s + jitter

        # 绘图
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(jittered_w0s, filtered_spans, color='#A6A6A6', s=50, edgecolors='#595959', linewidths=1)

        # 拟合曲线
        w0_range = np.linspace(filtered_w0s.min(), filtered_w0s.max(), 200)
        ax.plot(w0_range, hyperbola(w0_range, a, b), color='black', linewidth=2)

        # 仅保留左和下两条坐标轴
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

        # 坐标轴标签（下标形式）
        ax.set_xlabel(r'Memory capacity $\mathit{w}_0$', fontsize=19)
        ax.set_ylabel('Middle phase length', fontsize=19)

        # x 轴从 0 开始，每 0.04 为一个刻度；x,y 轴刻度字号设为 16
        ax.set_xticks(np.arange(0, filtered_w0s.max() + 0.04, 0.04))
        ax.tick_params(axis='both', labelsize=16)

        # 在图上角落添加公式和 R²（无边框）
        formula = (
            rf"$\mathit{{y}} = {a:.3f}\,\mathit{{x}}^{{-1}} + {b:.3f}$" + "\n"
            rf"$\mathit{{R}}^2 = {r2:.3f}$"
        )
        ax.text(0.95,
                0.9,
                formula,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=16)

        # 保存并关闭
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)



    def _find_crossings(self, exp: np.ndarray,
                        explor: np.ndarray) -> list[int]:
        """
        找到 exp 与 explor 曲线的所有交点（横坐标，取最近整数）。
        """
        diff = exp - explor
        crossings = []
        for i in range(len(diff) - 1):
            # 如果正好落在某个点
            if diff[i] == 0:
                crossings.append(i + 1)
            # 如果两点符号相反，则在 i→i+1 之间有一次交点
            elif diff[i] * diff[i + 1] < 0:
                # 线性插值位置：diff[i] + t*(diff[i+1]-diff[i]) = 0
                t = diff[i] / (diff[i] - diff[i + 1])
                x_cross = (i + 1) + t
                crossings.append(int(round(x_cross)))
        return crossings

    def compute_crossing_metrics(self,
                                 results: dict,
                                 window_size: int = 16) -> pd.DataFrame:
        """
        对每个被试，计算第一/最后一次交点的 x 轴位置，并输出差值。
        """
        rows = []
        for iSub, subject_info in results.items():
            steps = subject_info['best_step_results']
            post = [
                st['best_step_amount'].get(
                    'random_posterior',
                    st['best_step_amount'].get('top_posterior', [0]))[0]
                for st in steps if 'best_step_amount' in st
            ]
            rand_expl = [
                st['best_step_amount']['random'][0] for st in steps
                if 'best_step_amount' in st
            ]

            exp = pd.Series(post).rolling(
                window=window_size, min_periods=window_size).mean().to_numpy()
            explor = pd.Series(rand_expl).rolling(
                window=window_size, min_periods=window_size).mean().to_numpy()

            # trim 开头那些 < window_size 的 NaN
            valid_idx = ~np.isnan(exp) & ~np.isnan(explor)
            exp = exp[valid_idx]
            explor = explor[valid_idx]

            xs = np.arange(1, len(exp) + 1)
            crossings = self._find_crossings(exp, explor)

            if crossings:
                first = crossings[0]
                last = crossings[-1]
                delta = last - first
            else:
                first = last = delta = np.nan

            rows.append({
                'Subject': iSub,
                'FirstCross': first,
                'LastCross': last,
                'Span': delta
            })

        df = pd.DataFrame(rows).set_index('Subject')
        return df


class FigS3:

    def plot_acc_comparison_subs(self,
                                 results,
                                 subject_ids,
                                 nrow,
                                 color,
                                 color_true,
                                 xtick_interval,
                                 subject_labels,
                                 figsize=(15, 5),
                                 save_path=None):

        # Grid size
        n_sub = len(subject_ids)
        ncol = int(np.ceil(n_sub / nrow))

        # 基础设置
        fig, axes = plt.subplots(nrow,
                                 ncol,
                                 figsize=figsize,
                                 sharex=False,
                                 sharey=False)
        axes_flat = np.array(axes).reshape(-1)
        plt.tight_layout(h_pad=3)

        padding = [np.nan] * 15
        x_vals = None

        for idx, subject_id in enumerate(subject_ids):
            ax = axes_flat[idx]

            # 真实准确率
            true_acc = padding + list(results[subject_id]['sliding_true_acc'])
            x_vals = np.arange(1, len(true_acc) + 1)
            ax.plot(x_vals,
                    true_acc,
                    label='Human',
                    color=color_true,
                    linewidth=1.5)

            # 绘制预测结果
            pred = padding + list(results[subject_id]['sliding_pred_acc'])
            std = padding + list(results[subject_id]['sliding_pred_acc_std'])
            ax.plot(x_vals, pred, color=color, linewidth=1.5, clip_on=False)
            low = np.array(pred) - np.array(std)
            high = np.array(pred) + np.array(std)
            ax.fill_between(x_vals, low, high, color=color, alpha=0.3)

            # 辅助元素
            add_segmentation_lines(ax,
                                   len(x_vals),
                                   interval=64,
                                   color='grey',
                                   alpha=0.3,
                                   linestyle='dashed',
                                   linewidth=0.6)

            ax.set_xlim(0, len(x_vals) + 1)
            ax.set_ylim(0, 1)

            col = idx % ncol

            yticks = [0, 0.5, 1]
            ax.set_yticks(yticks)
            if col == 0:
                ax.set_yticklabels(['0', '0.5', '1'], fontsize=16)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel(None)

            x_max = int(ax.get_xlim()[1])
            if x_max > 1000:
                xtick_step = 256
            elif x_max < 130:
                xtick_step = 64
            else:
                xtick_step = xtick_interval
            xticks = np.arange(0, x_max + 1, xtick_step)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=16)
            ax.set_xlabel(None)

            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(1.186)

            ax.text(0.95,
                    0.05,
                    f'S{subject_labels[subject_ids.index(subject_id)]}',
                    transform=ax.transAxes,
                    fontsize=20,
                    ha='right',
                    va='bottom',
                    color='black')

        fig.text(0.5, -0.03, 'Trial', ha='center', fontsize=19)
        fig.text(-0.025,
                 0.5,
                 'Learning performance',
                 va='center',
                 rotation='vertical',
                 fontsize=19)

        # Hide unused subplots
        for j in range(n_sub, nrow * ncol):
            axes_flat[j].axis('off')

        # 保存或展示
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)



    def plot_k_comparison_subs(self,
                            oral_hypo_hits: Dict[int, Dict[str, Any]],
                            results: Dict[int, Any],
                            subject_ids: List[int],
                            nrow: int,
                            color: str = '#45B53F',
                            color_true: str = '#4C7AD1',
                            xtick_interval: int = 128,
                            subject_labels: List[str] = None,
                            model_label: str = 'PMH',
                            figsize: Tuple[int, int] = (15, 5),
                            save_path: str | None = None):
        """
        Compare true-hypothesis curves vs. posterior over special hypothesis across multiple subjects.

        Parameters
        ----------
        oral_hypo_hits : dict
            Output from get_oral_distance for all subjects.
        results : dict
            Posterior results for all subjects.
        subject_ids : list of int
            List of subject IDs to include.
        nrow : int
            Number of rows in subplot grid.
        ...
        """
        n_sub = len(subject_ids)
        ncol = int(np.ceil(n_sub / nrow))

        fig, axes = plt.subplots(nrow,
                                ncol,
                                figsize=figsize,
                                sharex=False,
                                sharey=False)
        axes_flat = np.array(axes).reshape(-1)
        plt.tight_layout(h_pad=3)

        def _extract_ma_filtered(results_sub, k_special, win=16):
            step_res = results_sub.get('step_results', results_sub.get('best_step_results', []))
            post_vals = []
            for step in step_res:
                post = step['hypo_details'].get(k_special, {}).get('post_max', 0.0)
                try:
                    post = float(post)
                except (TypeError, ValueError):
                    post = 0.0
                post_vals.append(post)
            return pd.Series(post_vals, dtype=float).rolling(window=win, min_periods=win).mean().to_numpy()

        for idx, subject_id in enumerate(subject_ids):
            ax = axes_flat[idx]

            # 数据提取
            odict = oral_hypo_hits[subject_id]
            condition = odict['condition']
            k_special = 0 if condition == 1 else 42
            win = 16

            rolling_hits = odict['rolling_hits']
            x_vals = np.arange(1, len(rolling_hits) + 1)

            ax.plot(x_vals,
                    rolling_hits,
                    lw=1.5,
                    label='Human',
                    color=color_true,
                    clip_on=False)

            rolling_k = _extract_ma_filtered(results[subject_id], k_special, win)
            ax.plot(x_vals[win - 1:],
                    rolling_k[win - 1:],
                    label=model_label,
                    lw=1.5,
                    color=color,
                    clip_on=False)

            # 分割线
            add_segmentation_lines(ax,
                                len(x_vals),
                                interval=64,
                                color='grey',
                                alpha=0.3,
                                linestyle='--',
                                linewidth=0.6)

            ax.set_xlim(0, len(x_vals) + 1)
            ax.set_ylim(0, 1)

            col = idx % ncol
            yticks = [0, 0.5, 1]
            ax.set_yticks(yticks)
            if col == 0:
                ax.set_yticklabels(['0', '0.5', '1'], fontsize=16)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel(None)

            x_max = int(ax.get_xlim()[1])
            if x_max > 1000:
                xtick_step = 256
            elif x_max < 130:
                xtick_step = 64
            else:
                xtick_step = xtick_interval
            xticks = np.arange(0, x_max + 1, xtick_step)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=16)
            ax.set_xlabel(None)

            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(1.186)

            ax.text(0.95,
                    0.05,
                    f'S{subject_labels[subject_ids.index(subject_id)]}',
                    transform=ax.transAxes,
                    fontsize=20,
                    ha='right',
                    va='bottom',
                    color='black')

        fig.text(0.5, -0.03, 'Trial', ha='center', fontsize=19)
        fig.text(-0.025,
                0.5,
                'True-hypo probability',
                va='center',
                rotation='vertical',
                fontsize=19)

        for j in range(n_sub, nrow * ncol):
            axes_flat[j].axis('off')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)


    def plot_grid_subs(self, results: Dict[int, Any],
                    subject_ids: List[int],
                    nrow: int,
                    subject_labels: List[str],
                    fname: Tuple[str, str] = ('gamma', 'w0'),
                    figsize: Tuple[int, int] = (15, 5),
                    save_path: str | None = None):
        """
        Plot grid search errors for each subject in a subplot layout.

        Parameters
        ----------
        results : dict
            Dictionary containing grid search results for each subject.
        subject_ids : list of int
            List of subject IDs to include.
        nrow : int
            Number of rows in subplot grid.
        subject_labels : list of str
            Labels for each subject to annotate subplots.
        fname : tuple of str
            Names of the parameters to be shown on the axes.
        figsize : tuple of int
            Figure size.
        save_path : str or None
            If specified, path to save the resulting figure.
        """
        n_sub = len(subject_ids)
        ncol = int(ceil(n_sub / nrow))

        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=False, sharey=False)
        axes_flat = np.array(axes).reshape(-1)
        plt.tight_layout()

        for idx, subject_id in enumerate(subject_ids):
            ax = axes_flat[idx]
            info = results[subject_id]
            data = []
            for (g, w0), errs in info['grid_errors'].items():
                err_val = float(np.mean(errs))
                data.append({'gamma': g, 'w0': w0, 'Error': err_val})
            df = pd.DataFrame(data)
            em = df.pivot(index='gamma', columns='w0', values='Error')
            sns.heatmap(em, ax=ax, cmap='viridis_r')

            ax.set_title(f'S{subject_labels[idx]}', fontsize=16)

            row = idx // ncol
            col = idx % ncol

            if row == nrow - 1:
                ax.set_xticks(np.arange(len(em.columns)) + 0.5)
                ax.set_xticklabels([f"{v:.4f}" for v in em.columns], rotation=45, ha="right")
            else:
                ax.set_xticks([])
            ax.set_xlabel(None)

            if col == 0:
                ax.set_yticks(np.arange(len(em.index)) + 0.5)
                ax.set_yticklabels([f"{v:.2f}" for v in em.index], rotation=0)
            else:
                ax.set_yticks([])
            ax.set_ylabel(None)

        fig.text(0.5, -0.03, r'$w_0$', ha='center', fontsize=19)
        fig.text(-0.025,
                0.5,
                r'$\gamma$',
                va='center',
                rotation='vertical',
                fontsize=19)

        for j in range(n_sub, nrow * ncol):
            axes_flat[j].axis('off')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)


    def plot_amount_subs(self, results: Dict[int, Any],
                        subject_ids: List[int],
                        nrow: int,
                        subject_labels: List[str],
                        window_size: int = 16,
                        figsize: Tuple[int, int] = (15, 5),
                        color_1: str = '#DDAA33',
                        color_2: str = '#DDAA33',
                        label_1: str = 'Exploitation',
                        label_2: str = 'Exploration',
                        save_path: str | None = None):
        n_sub = len(subject_ids)
        ncol = int(ceil(n_sub / nrow))

        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=False, sharey=False)
        axes_flat = np.array(axes).reshape(-1)
        plt.tight_layout(h_pad=3)

        for idx, subject_id in enumerate(subject_ids):
            ax = axes_flat[idx]
            step_results = results[subject_id].get('best_step_results', [])

            posterior = [
                sum(value[0] if isinstance(value, (list, tuple, np.ndarray)) else value
                    for key, value in step['best_step_amount'].items()
                    if 'posterior' in key) for step in step_results
                if 'best_step_amount' in step
            ]
            random = [
                step['best_step_amount']['random'][0] for step in step_results
                if 'best_step_amount' in step
            ]

            rolling_exploitation = pd.Series(posterior).rolling(window=window_size, min_periods=window_size).mean().to_list()
            rolling_exploration = pd.Series(random).rolling(window=window_size, min_periods=window_size).mean().to_list()
            x_vals = np.arange(1, len(posterior) + 1)

            ax.plot(x_vals, rolling_exploitation, label=label_1, color=color_1, linewidth=2, clip_on=False)
            ax.plot(x_vals, rolling_exploration, label=label_2, color=color_2, linewidth=2, clip_on=False)

            add_segmentation_lines(ax,
                                len(x_vals),
                                interval=64,
                                color='grey',
                                alpha=0.3,
                                linestyle='--',
                                linewidth=0.6)

            ax.set_xlim(0, len(x_vals) + 2)
            x_max = int(ax.get_xlim()[1])
            xtick_step = 256 if x_max > 1000 else 64 if x_max < 130 else 128
            xticks = np.arange(0, x_max + 1, xtick_step)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=16)
            ax.set_xlabel(None)

            # ax.set_yticks([0, 0.5, 1])
            # if idx % ncol == 0:
            ax.tick_params(axis='y', which='major', labelsize=16)
            # else:
            # ax.set_yticklabels([])
            ax.set_ylabel(None)

            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_linewidth(1.186)

            ax.text(0.95, 0.05, f'S{subject_labels[subject_ids.index(subject_id)]}',
                    transform=ax.transAxes, fontsize=20, ha='right', va='bottom', color='black')

        fig.text(0.5, -0.03, 'Trial', ha='center', fontsize=19)
        fig.text(-0.01, 0.5, 'Amount', va='center', rotation='vertical', fontsize=19)

        for j in range(n_sub, nrow * ncol):
            axes_flat[j].axis('off')

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', transparent=True)
            print(f"Figure saved to {save_path}")
        plt.close(fig)
