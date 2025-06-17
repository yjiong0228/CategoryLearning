# -*- coding: utf-8 -*-
"""
A PyVista‑based rewrite of the original `Processor` class.
All Matplotlib drawing calls have been replaced by PyVista so that the three
coordinate axes intersect at the world origin (0, 0, 0) without the need to
manually fake axes.  The public API is unchanged – you can still call
``process_and_plot`` and it will create one PNG per trial inside the
``plots_dir/choiceX`` folders.  Each PNG now comes from an off‑screen PyVista
renderer (so it works on a headless server).

Dependencies
------------
- pandas
- numpy
- scipy
- pyvista >= 0.43

Install with::

    pip install pandas numpy scipy pyvista

Notes
-----
* PyVista does **not** support a transparent background in screenshots; images
  are saved with a white background.
* ``Plotter.add_axes_at_origin()`` is called for every subplot so the coloured
  axes always originate from (0, 0, 0).
* The spline/gradient logic is identical to the Matplotlib version, but the
  spline and points are rendered with per‑point scalars so that the colormap is
  continuous along the curve **and** the vertices.
"""
from __future__ import annotations

import os
import re
import colorsys
from math import isfinite
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.colors as mc
from scipy.interpolate import splprep, splev

import pyvista as pv


class Processor:
    # ──────────────────────────────── data munging ────────────────────────────
    def read_data(self, df: pd.DataFrame, input_modelfitting: List, ncats: int) -> pd.DataFrame:
        """Merge behavioural CSV with model‑fitting centres (unchanged)."""
        oral_columns = ["neck_oral", "head_oral", "leg_oral", "tail_oral"]
        for col in oral_columns:
            if col not in df.columns:
                raise ValueError(f"column '{col}' not found in dataframe")

        rename_mapping = {
            "feature1_oral": "human_feature_1",
            "feature2_oral": "human_feature_2",
            "feature3_oral": "human_feature_3",
            "feature4_oral": "human_feature_4",
        }
        df1 = df.rename(columns=rename_mapping)

        columns = [f"choice_{c}_feature_{f}" for c in range(1, ncats + 1) for f in range(1, 5)]
        rows = []
        for k, center_dict in input_modelfitting:
            row = []
            for choice_key in range(ncats):
                features = center_dict.get(choice_key, (None,) * 4)
                row.extend(features)
            rows.append(row)
        df2 = pd.DataFrame(rows, columns=columns)
        return pd.concat([df1, df2], axis=1)

    # ──────────────────────────────── helpers ─────────────────────────────────
   
    def _add_cube(self, plotter, colour="#A0A0A0", radius=0.015):
        cube = pv.Cube(bounds=(0, 1, 0, 1, 0, 1))
        tube = cube.extract_all_edges().tube(radius=radius)   # 半径=世界坐标
        plotter.add_mesh(tube, color=colour, lighting=False)

    def _add_axes(self, plotter: pv.Plotter,
                axis_len=0.8,          # 整体长度
                cyl_r=0.012,           # 轴杆粗细
                cone_r=0.03,           # 箭头半径
                tip_len=0.08,          # 箭头长度  (0-1 之间)
                shaft_len=0.92,        # 箭杆长度  (0-1 之间)
                font_sz=12):
        actor = plotter.add_axes_at_origin(
            xlabel="Feature 1", ylabel="Feature 2", zlabel="Feature 3",
            x_color="#d62728", y_color="#2ca02c", z_color="#1f77b4",
            line_width=1.0,
        )

        # ① 整体缩放
        actor.SetTotalLength(axis_len, axis_len, axis_len)

        # ② 细调尺寸（VTK 9.x）
        actor.SetCylinderRadius(cyl_r)                    # 轴杆半径
        actor.SetConeRadius(cone_r)                       # 箭头半径
        actor.SetNormalizedShaftLength(shaft_len,         # 轴杆占总长
                                    shaft_len,
                                    shaft_len)
        actor.SetNormalizedTipLength(tip_len,             # 箭头占总长
                                    tip_len,
                                    tip_len)

        # ③ 字号
        for cap in (actor.GetXAxisCaptionActor2D(),
                    actor.GetYAxisCaptionActor2D(),
                    actor.GetZAxisCaptionActor2D()):
            tp = cap.GetTextActor().GetTextProperty()
            tp.SetFontSize(font_sz)


    # ─────────────────────────── gradient spline drawing ─────────────────────
    def _add_smooth_grad_line(self, plotter: pv.Plotter, xs, ys, zs,
                              cmap_name="Blues", cmap_low=0.25, cmap_high=1.0,
                              line_width=2, point_size=8, n_interp=300,
                              smooth=0.0):
        xs, ys, zs = map(np.asarray, (xs, ys, zs))
        # remove NaN/inf
        mask = np.array([all(map(isfinite, p)) for p in zip(xs, ys, zs)])
        xs, ys, zs = xs[mask], ys[mask], zs[mask]
        if len(xs) == 0:
            return

        # merge consecutive duplicates
        keep = [0] + [i for i in range(1, len(xs)) if (xs[i], ys[i], zs[i]) != (xs[i-1], ys[i-1], zs[i-1])]
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        n_pts = len(xs)

        cmap = cm.get_cmap(cmap_name)
        # handle 1‑2 points quickly ------------------------------------------------
        if n_pts == 1:
            plotter.add_points(np.c_[xs, ys, zs], color=cmap(cmap_high), point_size=point_size,
                               render_points_as_spheres=True)
            return
        if n_pts == 2:
            colors = [cmap(cmap_low), cmap(cmap_high)]
            plotter.add_lines(np.c_[xs, ys, zs], color=colors[0], width=line_width)
            plotter.add_points(np.c_[xs, ys, zs], scalars=[0, 1], cmap=cmap_name,
                               point_size=point_size, render_points_as_spheres=True,
                               clim=[0, 1])
            return

        # ≥ 3 points: spline -------------------------------------------------------
        k = min(3, n_pts - 1)
        try:
            tck, u = splprep([xs, ys, zs], s=smooth, k=k)
            u_fine = np.linspace(0, 1, n_interp)
            x_f, y_f, z_f = splev(u_fine, tck)
        except Exception:
            # fallback to polyline
            scalars = np.linspace(0, 1, n_pts)
            poly = pv.lines_from_points(np.c_[xs, ys, zs])
            poly["scalars"] = scalars
            plotter.add_mesh(poly, scalars="scalars", cmap=cmap_name, line_width=line_width,
                             clim=[0, 1], show_scalar_bar=False)
            plotter.add_points(np.c_[xs, ys, zs], scalars=scalars, cmap=cmap_name,
                               point_size=point_size, render_points_as_spheres=True,
                               clim=[0, 1], show_scalar_bar=False)
            return

        # create smooth polydata
        spline_pts = np.c_[x_f, y_f, z_f]
        poly = pv.lines_from_points(spline_pts)
        poly["scalars"] = np.linspace(cmap_low, cmap_high, poly.n_points)
        plotter.add_mesh(poly, scalars="scalars", cmap=cmap_name, line_width=line_width,
                         clim=[cmap_low, cmap_high], show_scalar_bar=False, lighting=False)

        # original vertices as spheres -------------------------------------------
        pt_colors = np.linspace(cmap_low, cmap_high, n_pts)
        plotter.add_points(np.c_[xs, ys, zs], scalars=pt_colors, cmap=cmap_name,
                           point_size=point_size, render_points_as_spheres=True,
                           clim=[cmap_low, cmap_high], show_scalar_bar=False)

    # ────────────────────────────── main figure ───────────────────────────────
    def plot_choice_graph(self, ncats: int, iSub, iSession, iTrial, choice: int,
                          features_list: List[Dict], plots_dir: str,
                          plot_side: str = "both"):
        choice_folder = os.path.join(plots_dir, f"choice{choice}")
        os.makedirs(choice_folder, exist_ok=True)

        # yellow target locations -------------------------------------------------
        if ncats == 2:
            yellow_point_coords = {1: (0.5, 0.25, 0.5), 2: (0.5, 0.75, 0.5)}
        else:
            yellow_point_coords = {1: (0.25, 0.25, 0.5), 2: (0.25, 0.75, 0.5),
                                   3: (0.75, 0.5, 0.25), 4: (0.75, 0.5, 0.75)}

        # extract trajectories ----------------------------------------------------
        human_x = [d["human_feature_2"] for d in features_list]
        human_y = [d["human_feature_1"] for d in features_list]
        human_z = [d["human_feature_3"] for d in features_list]

        bayes_x = [d[f"choice_{choice}_feature_2"] for d in features_list]
        bayes_y = [d[f"choice_{choice}_feature_1"] for d in features_list]
        bayes_z = [d[f"choice_{choice}_feature_3"] for d in features_list]

        # plotter setup -----------------------------------------------------------
        if plot_side == "both":
            plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1000, 600))
            subplots = [0, 1]
        else:
            plotter = pv.Plotter(off_screen=True, window_size=(700, 700))
            subplots = [0]

        # helper to add a subplot -------------------------------------------------
        def _populate_subplot(idx: int, xs, ys, zs, camera=None):
            plotter.subplot(0, idx) if len(subplots) == 2 else None
            self._add_cube(plotter)
            self._add_axes(plotter,
                        axis_len=1,
                        cyl_r=10,
                        cone_r=0.5,
                        tip_len=0.07,
                        shaft_len=0.93,
                        font_sz=5)

            if choice in yellow_point_coords:
                plotter.add_points(np.array(yellow_point_coords[choice]).reshape(1, 3),
                                   color="yellow", point_size=15, render_points_as_spheres=True)
            if len(xs) > 1:
                self._add_smooth_grad_line(plotter, xs, ys, zs, cmap_name="Blues", cmap_low=0.25)
            if camera:
                plotter.camera_position = camera
            else:
                plotter.view_isometric()
                plotter.camera.Azimuth(-30)
                plotter.camera.Elevation(-15)
            
            plotter.set_background(None)

        # populate views ----------------------------------------------------------
        if plot_side in ("left", "both"):
            _populate_subplot(0, human_x, human_y, human_z)
        if plot_side in ("right", "both"):
            idx = 1 if plot_side == "both" else 0
            _populate_subplot(idx, bayes_x, bayes_y, bayes_z)

        # save --------------------------------------------------------------------
        filename = f"{iSub}_{iSession}_{iTrial}_c{choice}.png"
        filepath = os.path.join(choice_folder, filename)
        plotter.screenshot(str(filepath), transparent_background=True)
        plotter.close()

    # ──────────────────────────── batch processing ────────────────────────────
    def process_and_plot(self, ncats: int, subject_data: pd.DataFrame,
                         input_modelfitting: List, output_csv: str,
                         plots_dir: str, plot_side: str = "both"):
        df = self.read_data(subject_data, input_modelfitting, ncats)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        for c in range(1, 5):
            Path(plots_dir, f"choice{c}").mkdir(parents=True, exist_ok=True)

        last_known: Dict[int, List[Dict]] = {1: [], 2: [], 3: [], 4: []}
        for _, row in df.iterrows():
            iSub = row.get("iSub", "Unknown")
            iSession = row["iSession"]
            iTrial = row["iTrial"]
            current_choice = int(row["choice"])

            # build current feature entry ---------------------------------------
            human = {f"human_feature_{k}": row[f"human_feature_{k}"] for k in range(1, 5)}
            choice_feats = {f"choice_{c}_feature_{k}": row[f"choice_{c}_feature_{k}"]
                            for c in range(1, ncats + 1) for k in range(1, 5)}
            entry = {**human, **{k: v for k, v in choice_feats.items()
                                  if k.startswith(f"choice_{current_choice}")}}
            last_known[current_choice].append(entry)

            # draw current choice ------------------------------------------------
            self.plot_choice_graph(ncats, iSub, iSession, iTrial, current_choice,
                                   last_known[current_choice], plots_dir, plot_side)

            # update the rest ----------------------------------------------------
            for c in range(1, 5):
                if c == current_choice or not last_known[c]:
                    continue
                self.plot_choice_graph(ncats, iSub, iSession, iTrial, c,
                                       last_known[c], plots_dir, plot_side)

        print("PyVista plots saved under:", plots_dir)

    # ──────────────────────────── utility -------------------------------------
    @staticmethod
    def extract_session_trial(filename: str, pattern: str):
        match = re.match(pattern, filename)
        return (int(match.group(1)), int(match.group(2))) if match else (None, None)





    def create_sorted_gif(self, plots_dir, output_gif, pattern, duration=0.5):
        """
        将指定文件夹中的所有PNG图像按照iSession和iTrial的顺序合成为一个GIF文件。

        参数:
        - plots_dir (Path): 存放PNG图像的子文件夹路径。
        - output_gif (Path): 输出GIF文件的路径。
        - pattern (str): 用于匹配文件名的正则表达式模式。
        - duration (float): 每帧之间的时间间隔（秒）。
        """
        # 获取所有PNG文件
        all_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]

        # 提取iSession和iTrial，并过滤无效文件
        valid_files = []
        for f in all_files:
            iSession, iTrial = self.extract_session_trial(f, pattern)
            if iSession is not None and iTrial is not None:
                valid_files.append((iSession, iTrial, f))
            else:
                print(f"文件名 '{f}' 不符合预期格式，已跳过。")

        if not valid_files:
            print(f"在文件夹 '{plots_dir}' 中未找到符合格式的PNG图像。")
            return

        # 按iSession和iTrial排序
        sorted_files = sorted(valid_files, key=lambda x: (x[0], x[1]))

        # 读取图像
        images = []
        for iSession, iTrial, filename in sorted_files:
            filepath = plots_dir / filename
            try:
                images.append(imageio.imread(filepath))
            except Exception as e:
                print(f"读取文件 '{filepath}' 时出错: {e}")

        if not images:
            print(f"没有成功读取任何图像用于 '{output_gif}'。")
            return

        # 创建GIF
        try:
            imageio.mimsave(output_gif, images, duration=duration)
            print(f"GIF已成功创建并保存为 '{output_gif}'。")
        except Exception as e:
            print(f"保存GIF '{output_gif}' 时出错: {e}")
