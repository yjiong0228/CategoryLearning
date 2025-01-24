"""
绘制Task2的口头汇报和模型参数的动图对比
"""

import numpy as np
import pandas as pd
import os
import matplotlib.colors as mc
import colorsys
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import os
import re
import imageio

class Processor:
    def read_data(self, input_rec_csv, input_bhv_csv, input_modelfitting):
        """
        读取并合并三个数据文件。
        
        Parameters:
            input_rec_csv (str): Task2_15_rec.csv 的路径。
            input_bhv_csv (str): Task2_15_bhv.csv 的路径。
            input_modelfitting (list): 包含模型拟合数据的列表，每个元素为 (k, center_dict)。
        
        Returns:
            pd.DataFrame: 合并后的数据框，包含human_feature和choice特征。
        """
        # 1. 读取CSV文件
        try:
            df_rec = pd.read_csv(input_rec_csv)
            df_bhv = pd.read_csv(input_bhv_csv)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"无法找到文件: {e.filename}")
        
        # 2. 合并两个DataFrame，基于'iSession'和'iTrial'
        if not {'iSession', 'iTrial'}.issubset(df_rec.columns):
            raise ValueError("Task2_15_rec.csv中必须包含'iSession'和'iTrial'列。")
        if not {'iSession', 'iTrial', 'choice'}.issubset(df_bhv.columns):
            raise ValueError("Task2_15_bhv.csv中必须包含'iSession', 'iTrial'和'choice'列。")
        
        df = pd.merge(df_rec, df_bhv[['iSession', 'iTrial', 'choice']], on=['iSession', 'iTrial'], how='left')

        # 3. 处理四个value列
        value_columns = ['neck_value', 'head_value', 'leg_value', 'tail_value']
        for col in value_columns:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 在CSV文件中不存在。")

        # 4. 填充空值为2
        df[value_columns] = df[value_columns].fillna(2)

        # 5. 映射[1, 2, 3]到[0.25, 0.5, 0.75]
        mapping = {1: 0.25, 2: 0.5, 3: 0.75}
        df[value_columns] = df[value_columns].replace(mapping)

        # 6. 重命名列
        rename_mapping = {
            'head_value': 'human_feature_1',
            'leg_value': 'human_feature_2',
            'tail_value': 'human_feature_3',
            'neck_value': 'human_feature_4'
        }
        df1 = df.rename(columns=rename_mapping)

        # 定义列名
        columns = [f'choice_{choice}_feature_{feature}' 
                for choice in range(1, 5) 
                for feature in range(1, 5)]

        # 提取数据行
        rows = []
        for entry in input_modelfitting:
            k, center_dict = entry
            row = []
            for choice_key in range(4):  # 键 0 到 3
                features = center_dict.get(choice_key, (None,)*4)
                row.extend(features)
            rows.append(row)

        # 创建 DataFrame
        df2 = pd.DataFrame(rows, columns=columns)

        result = pd.concat([df1, df2], axis=1)

        return result


    def draw_cube(self, ax):
        """
        在给定的轴上绘制一个立方体。
        
        Parameters:
            ax (Axes3D): 三维坐标轴对象。
        """
        # 定义立方体的8个顶点
        vertices = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1)
        ]
        
        # 定义立方体的12条边，连接顶点索引
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 连接底面和顶面
        ]
        
        # 绘制边线
        for edge in edges:
            start, end = edge
            x_vals = [vertices[start][0], vertices[end][0]]
            y_vals = [vertices[start][1], vertices[end][1]]
            z_vals = [vertices[start][2], vertices[end][2]]
            ax.plot(x_vals, y_vals, z_vals, color='grey')

    def draw_intersection_lines(self, ax):
        """
        在给定的轴上绘制灰色的三个平面的交线，共六条。
        
        Parameters:
            ax (Axes3D): 三维坐标轴对象。
        """
        # 定义平面位置
        plane_position = 0.5
        
        # # 绘制平面 feature1=0.5
        # y, z = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        # x = np.full_like(y, plane_position)
        # ax.plot_surface(x, y, z, color='grey', alpha=0.07)
        
        # # 绘制平面 feature2=0.5
        # x, z = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        # y = np.full_like(x, plane_position)
        # ax.plot_surface(x, y, z, color='grey', alpha=0.07)
        
        # # 绘制平面 feature3=0.5
        # x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        # z = np.full_like(x, plane_position)
        # ax.plot_surface(x, y, z, color='grey', alpha=0.07)
        
        # 绘制交线（六条）
        # 1. feature1=0.5 与 feature2=0.5 的交线 (x=0.5, y=0.5, z从0到1)
        ax.plot([plane_position, plane_position], [plane_position, plane_position], [0, 1], color='grey', linestyle='--', linewidth=1)
        
        # 2. feature1=0.5 与 feature3=0.5 的交线 (x=0.5, y从0到1, z=0.5)
        ax.plot([plane_position, plane_position], [0, 1], [plane_position, plane_position], color='grey', linestyle='--', linewidth=1)
        
        # 3. feature2=0.5 与 feature3=0.5 的交线 (x从0到1, y=0.5, z=0.5)
        ax.plot([0, 1], [plane_position, plane_position], [plane_position, plane_position], color='grey', linestyle='--', linewidth=1)

    def lighten_color(self, color, amount=0.5):
        """
        淡化颜色，使其更浅。

        Parameters:
            color (str): 原始颜色名称或RGB值。
            amount (float): 淡化程度，0表示不变，1表示白色。

        Returns:
            tuple: 淡化后的RGB颜色。
        """
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        # 淡化颜色
        new_color = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        return new_color

    def plot_choice_graph(self, iSub, iSession, iTrial, choice, features_list, color_mapping, plots_dir, plot_side='both'):
        """
        绘制特定 choice 值的图像，并保存。
        
        Parameters:
            iSub (int/str): 受试者编号。
            iSession (int/str): 会话编号。
            iTrial (int/str): 试验编号。
            choice (int): 当前的 choice 值（1, 2, 3, 4）。
            features (list or tuple): 当前行的特征值 [feature1, feature2, feature3, feature4]。
            color_mapping (dict): 每个 choice 值对应的颜色映射。
            plots_dir (str): 图像保存的文件夹路径。
            plot_side (str): 绘制的子图类型，可选 'left', 'right', 'both'。
        """
        # 创建对应 choice 的子文件夹路径
        choice_folder = os.path.join(plots_dir, f"choice{choice}")
        if not os.path.exists(choice_folder):
            os.makedirs(choice_folder)
        
        # 创建图表
        fig = plt.figure(figsize=(12, 6) if plot_side == 'both' else (6, 6))
        
        # 添加主标题
        fig.suptitle(f"iSub={iSub}, iSession={iSession}, iTrial={iTrial}, Category={choice}", fontsize=16)

        # Prepare yellow point coordinates
        yellow_point_coords = {
            1: (0.25, 0.25, 0.5),
            2: (0.25, 0.75, 0.5),
            3: (0.75, 0.5, 0.25),
            4: (0.75, 0.5, 0.75)
        }

        # Extract all human and Bayesian learner features for trajectory
        human_x = [feat['human_feature_1'] for feat in features_list]
        human_y = [feat['human_feature_2'] for feat in features_list]
        human_z = [feat['human_feature_3'] for feat in features_list]

        bayesian_x = [feat[f'choice_{choice}_feature_1'] for feat in features_list]
        bayesian_y = [feat[f'choice_{choice}_feature_2'] for feat in features_list]
        bayesian_z = [feat[f'choice_{choice}_feature_3'] for feat in features_list]

        # 绘制左图（human_feature_1, human_feature_2, human_feature_3）
        if plot_side in ['left', 'both']:
            ax_left = fig.add_subplot(1, 2, 1, projection='3d') if plot_side == 'both' else fig.add_subplot(1, 1, 1, projection='3d')
            self.draw_cube(ax_left)

            # 淡化后的颜色
            lighter_color = self.lighten_color(color_mapping[choice], amount=0.7)

            # 目标点
            if choice in yellow_point_coords:
                y_point = yellow_point_coords[choice]
                ax_left.scatter(*y_point, color='yellow', s=200, alpha=0.7, edgecolors='k')

            # Plot trajectory line
            if len(features_list) > 1:
                ax_left.plot(human_x, human_y, human_z, color=lighter_color, linewidth=1)
            # Plot current point
            ax_left.scatter(human_x[-1], human_y[-1], human_z[-1], color=color_mapping[choice], s=100, alpha=0.8, edgecolors='w')
            # 设置坐标轴刻度
            ax_left.set_xticks([0, 0.5, 1])
            ax_left.set_yticks([0, 0.5, 1])
            ax_left.set_zticks([0, 0.5, 1])
            ax_left.set_xlim(0, 1)
            ax_left.set_ylim(0, 1)
            ax_left.set_zlim(0, 1)
            ax_left.set_xlabel('Feature 1')
            ax_left.set_ylabel('Feature 2')
            ax_left.set_zlabel('Feature 3')
            ax_left.view_init(elev=15., azim=30)  # 调整视角
            # 绘制平面和交线
            # draw_intersection_lines(ax_left)
            # 添加子图标题
            ax_left.set_title("Human")

        # 绘制右图（feature2, 3, 4）
        if plot_side in ['right', 'both']:
            if plot_side == 'both':
                ax_right = fig.add_subplot(1, 2, 2, projection='3d')
            else:
                ax_right = fig.add_subplot(1, 1, 1, projection='3d')
            self.draw_cube(ax_right)

            # 淡化后的颜色
            lighter_color = self.lighten_color(color_mapping[choice], amount=0.7)

            # 目标点
            if choice in yellow_point_coords:
                y_point = yellow_point_coords[choice]
                ax_right.scatter(*y_point, color='yellow', s=200, alpha=0.7, edgecolors='k')
            
            # Plot trajectory line
            if len(features_list) > 1:
                ax_right.plot(bayesian_x, bayesian_y, bayesian_z, color=lighter_color, linewidth=1)
            # Plot current point
            ax_right.scatter(bayesian_x[-1], bayesian_y[-1], bayesian_z[-1], color=color_mapping[choice], s=100, alpha=0.8, edgecolors='w')
            # 设置坐标轴刻度
            ax_right.set_xticks([0, 0.5, 1])
            ax_right.set_yticks([0, 0.5, 1])
            ax_right.set_zticks([0, 0.5, 1])
            ax_right.set_xlim(0, 1)
            ax_right.set_ylim(0, 1)
            ax_right.set_zlim(0, 1)
            ax_right.set_xlabel('Feature 1')
            ax_right.set_ylabel('Feature 2')
            ax_right.set_zlabel('Feature 3')
            ax_right.view_init(elev=15., azim=30)  # 调整视角
            # 绘制平面和交线
            # draw_intersection_lines(ax_right)
            # 添加子图标题
            ax_right.set_title("Bayesian learner")
        
        # 保存图表到对应的 choice 文件夹
        filename = f"{iSub}_{iSession}_{iTrial}_c{choice}.png"
        filepath = os.path.join(choice_folder, filename)
        plt.savefig(filepath)
        plt.close()

    def process_and_plot(self, input_rec_csv, input_bhv_csv, input_modelfitting, output_csv, plots_dir, plot_side='both'):
        """
        处理数据并生成按 choice 分别绘制的图像。
        
        Parameters:
            input_rec_csv (str): Task2_15_rec.csv 的路径。
            input_bhv_csv (str): Task2_15_bhv.csv 的路径。
            input_modelfitting (list): 包含模型拟合数据的列表，每个元素为 (k, center_dict)。
            output_csv (str): 处理后的CSV文件保存路径。
            plots_dir (str): 图像保存的文件夹路径。
            plot_side (str): 绘制的子图类型，可选 'left', 'right', 'both'。
        """
        # 1. 读取并合并数据
        df = self.read_data(input_rec_csv, input_bhv_csv, input_modelfitting)
        
        # 3. 保存处理后的CSV
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        # 4. 创建各个 choice 的子文件夹
        for choice in range(1, 5):
            folder_name = f"choice{choice}"
            folder_path = os.path.join(plots_dir, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        # 5. 定义颜色映射
        color_mapping = {
            1: 'darkgreen',
            2: 'darkgreen',
            3: 'darkred',
            4: 'darkred'
        }
        
        # 6. 初始化 last_known_features
        # last_known_features = {choice: [feature_dict1, feature_dict2, ...]}
        last_known_features = {1: [], 2: [], 3: [], 4: []}
        
        # 7. 迭代每一行数据，生成图表
        for index, row in df.iterrows():
            iSub = row.get('iSub', 'Unknown')  # 假设有 'iSub' 列
            iSession = row['iSession']
            iTrial = row['iTrial']
            current_choice = row['choice']
            
            if pd.isna(current_choice):
                print(f"第 {index} 行缺少 'choice' 数据，跳过绘图。")
                continue
            
            current_choice = int(current_choice)
            # 提取human_feature
            human_features = {
                'human_feature_1': row['human_feature_1'],
                'human_feature_2': row['human_feature_2'],
                'human_feature_3': row['human_feature_3'],
                'human_feature_4': row['human_feature_4']
            }
            # 提取choice特征
            choice_features = {}
            for choice in range(1, 5):
                choice_features[f'choice_{choice}_feature_1'] = row[f'choice_{choice}_feature_1']
                choice_features[f'choice_{choice}_feature_2'] = row[f'choice_{choice}_feature_2']
                choice_features[f'choice_{choice}_feature_3'] = row[f'choice_{choice}_feature_3']
                choice_features[f'choice_{choice}_feature_4'] = row[f'choice_{choice}_feature_4']
            
            # 更新 last_known_features for the current_choice
            feature_entry = {
                'human_feature_1': human_features['human_feature_1'],
                'human_feature_2': human_features['human_feature_2'],
                'human_feature_3': human_features['human_feature_3'],
                'human_feature_4': human_features['human_feature_4'],
                f'choice_{current_choice}_feature_1': choice_features[f'choice_{current_choice}_feature_1'],
                f'choice_{current_choice}_feature_2': choice_features[f'choice_{current_choice}_feature_2'],
                f'choice_{current_choice}_feature_3': choice_features[f'choice_{current_choice}_feature_3'],
                f'choice_{current_choice}_feature_4': choice_features[f'choice_{current_choice}_feature_4']
            }
            last_known_features[current_choice].append(feature_entry)
            
            # 绘制当前 choice 的图像
            self.plot_choice_graph(
                iSub=iSub,
                iSession=iSession,
                iTrial=iTrial,
                choice=current_choice,
                features_list=last_known_features[current_choice],
                color_mapping=color_mapping,
                plots_dir=plots_dir,
                plot_side=plot_side
            )
        
            # 绘制其他 choices 的图像，使用 last_known_features
            for choice in range(1, 5):
                if choice == current_choice:
                    continue  # 已经绘制当前选择的 choice
                if last_known_features[choice]:
                    # 绘制该 choice 的图像，使用上一次已知的特征值
                    self.plot_choice_graph(
                        iSub=iSub,
                        iSession=iSession,
                        iTrial=iTrial,
                        choice=choice,
                        features_list=last_known_features[choice],
                        color_mapping=color_mapping,
                        plots_dir=plots_dir,
                        plot_side=plot_side
                    )
                else:
                    # 如果该 choice 之前没有数据，则跳过或使用默认图像
                    print(f"Choice {choice} 在第 {index} 行之前没有数据，跳过生成图像。")

        print(f"处理完成，图表已分别保存到 '{plots_dir}/choice1', '{plots_dir}/choice2', '{plots_dir}/choice3', 和 '{plots_dir}/choice4' 文件夹中。")

    
    def extract_session_trial(self, filename, pattern):
        """
        从文件名中提取iSession和iTrial。

        参数:
        - filename (str): 文件名。
        - pattern (str): 用于匹配文件名的正则表达式模式。

        返回:
        - tuple: (iSession, iTrial) 如果匹配成功，否则 (None, None)。
        """
        match = re.match(pattern, filename)
        if match:
            iSession = int(match.group(1))
            iTrial = int(match.group(2))
            return iSession, iTrial
        else:
            return None, None
        
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