"""
对Task1a, Task1b, Task3c的数据进行预处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import math


BODY_LENGTH = 130
FEATURES_RANGE = {
    'neck_length': {
        'min': BODY_LENGTH / 4,
        'max': BODY_LENGTH * 5 / 4
    },
    'head_length': {
        'min': BODY_LENGTH / 4,
        'max': BODY_LENGTH * 5 / 4
    },

    'leg_length': {
        'min': BODY_LENGTH / 4,
        'max': BODY_LENGTH * 5 / 4
    },
    'tail_length': {
        'min': BODY_LENGTH / 4,
        'max': BODY_LENGTH * 5 / 4
    },    
    'neck_angle': {
        'min': -math.pi / 3,
        'max': 0
    },
    'head_angle': {
        'min': 0,
        'max': math.pi / 3
    },
    'leg_angle': {
        'min': math.pi * 11 / 36,
        'max': math.pi * 17 / 36
    },
    'tail_angle': {
        'min': -math.pi / 3,
        'max': 0
    }
}

CANVAS_SETTINGS = {
    'Task1a' : {
        'canvas_height': 600,
        'canvas_width' : 1200
    },
    'Task1b' : {
        'canvas_height_max': 700,
        'canvas_height_min': 400,
        'canvas_width' : 780
    },
    'Task3c' : {
        'canvas_height': 600,
        'canvas_width' : 1200
    },
}


class Preprocessor_A:
    def process(self, taskID, feature_init, mouse_trajactory):
        # 移动前的初始特征值
        feature_init_real = self.transform_to_real_feature(FEATURES_RANGE, feature_init)

        # 转换为坐标
        if taskID == 'Task1a':
            feature_init_real = self.extract_body_ori(feature_init_real, mouse_trajactory, CANVAS_SETTINGS)
        coor_init = self.transform_feature_to_coor(taskID, feature_init_real, CANVAS_SETTINGS, BODY_LENGTH)

        # 得到移动后的坐标
        coor_trajactory = self.complete_coor(taskID, coor_init, mouse_trajactory, CANVAS_SETTINGS)
        feature_trajactory = self.transform_coor_to_feature(taskID, coor_trajactory, FEATURES_RANGE)
        
        return feature_trajactory

    # transform normalized feature values into real values
    def transform_to_real_feature(self, features_range, data):
        features = [
            'neck_length', 'neck_angle', 
            'head_length', 'head_angle', 
            'leg_length', 'leg_angle', 
            'tail_length', 'tail_angle'
        ]
        results = data.copy()
        for feature in features:
            if feature in data:
                feature_range = features_range[feature]
                min_val = feature_range['min']
                max_val = feature_range['max']
                results[feature] = min_val + data[feature] * (max_val - min_val)
        
        return results

    def extract_body_ori(self, feature_init_real, mouse_trajactory, canvas_settings):
        # 定义前部和后部节点列表
        front_nodes = ['head_end', 'head_neck', 'neck_body_leg', 'leg_end_0', 'leg_end_1']
        back_nodes = ['body_leg_tail', 'leg_end_2', 'leg_end_3', 'tail_end']
        all_nodes = front_nodes + back_nodes
        
        # 获取第一次出现目标节点的行
        first_target_row = mouse_trajactory[mouse_trajactory['node'].isin(all_nodes)].iloc[0]
        node_type = first_target_row['node']
        x_value = first_target_row['x']
        
        # 判断body_ori
        midpoint_x = canvas_settings['Task1a']['canvas_width']/2
        if (node_type in front_nodes and x_value < midpoint_x) or \
            (node_type in back_nodes and x_value > midpoint_x):
            body_ori = -1
        elif (node_type in front_nodes and x_value > midpoint_x) or \
            (node_type in back_nodes and x_value < midpoint_x):
            body_ori = 1
        else:
            body_ori = None  # 处理可能的边界情况
            
        # 将body_ori添加到adjust_init_real
        feature_init_real['body_ori'] = body_ori
        
        return feature_init_real

    # transform feature values into coordinates 
    def transform_feature_to_coor(self, taskID, data, canvas_settings, body_length):
        if taskID in ['Task1a', 'Task3c']:
            canvas_height = canvas_settings[taskID]['canvas_height']
            canvas_width = canvas_settings[taskID]['canvas_width']
        elif taskID == 'Task1b':
            display_height = data['display_height']
            canvas_height_min = canvas_settings[taskID]['canvas_height_min']
            canvas_height_max = canvas_settings[taskID]['canvas_height_max']
            canvas_height = canvas_height_min + display_height * (canvas_height_max - canvas_height_min)
            canvas_width = canvas_settings[taskID]['canvas_width']       

        body_ori = data['body_ori']
        neck_length = data['neck_length']
        head_length = data['head_length']
        leg_length = data['leg_length']
        tail_length = data['tail_length']
        neck_angle = data['neck_angle']
        head_angle = data['head_angle']
        leg_angle = data['leg_angle']
        tail_angle = data['tail_angle']

        x2 = canvas_width / 2 + body_ori * body_length / 2
        y2 = canvas_height / 2

        x3 = canvas_width / 2 - body_ori * body_length / 2
        y3 = y2

        x1 = x2 + body_ori * neck_length * np.cos(neck_angle)
        y1 = y2 + neck_length * np.sin(neck_angle)

        x0 = x1 + body_ori * head_length * np.cos(head_angle)
        y0 = y1 + head_length * np.sin(head_angle)

        x8 = x3 - body_ori * tail_length * np.cos(tail_angle)
        y8 = y3 + tail_length * np.sin(tail_angle)

        x4 = x2 + body_ori * leg_length * np.cos(leg_angle)
        y4 = y2 + leg_length * np.sin(leg_angle)
        x5 = x2 - body_ori * leg_length * np.cos(leg_angle)
        y5 = y2 + leg_length * np.sin(leg_angle)

        x6 = x3 + body_ori * leg_length * np.cos(leg_angle)
        y6 = y3 + leg_length * np.sin(leg_angle)
        x7 = x3 - body_ori * leg_length * np.cos(leg_angle)
        y7 = y3 + leg_length * np.sin(leg_angle)

        base_columns = {
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3,
            'x4': x4, 'y4': y4, 'x5': x5, 'y5': y5,
            'x6': x6, 'y6': y6, 'x7': x7, 'y7': y7,
            'x8': x8, 'y8': y8
        }

        if taskID == 'Task1a':
            results = pd.DataFrame(base_columns).reset_index(drop=True)
        else:
            results = pd.DataFrame({'iTrial': data['iTrial'], **base_columns}).reset_index(drop=True)

        return results

    # generate the complete coordinates trajactory
    def complete_coor(self, taskID, coor_init, mouse_trajactory, canvas_settings):
        node_mapping = {
            'head_end': ('x0', 'y0'),
            'head_neck': ('x1', 'y1'),
            'neck_body_leg': ('x2', 'y2'),
            'body_leg_tail': ('x3', 'y3'),
            'leg_end_0': ('x4', 'y4'),
            'leg_end_1': ('x5', 'y5'),
            'leg_end_2': ('x6', 'y6'),
            'leg_end_3': ('x7', 'y7'),
            'tail_end': ('x8', 'y8')
        }

        results = []

        if taskID == 'Task1a':
            data_groups = [(coor_init.iloc[0], mouse_trajactory)]
        else:
            data_groups = [(coor_init[coor_init['iTrial'] == iTrial].iloc[0],
                        mouse_trajactory[mouse_trajactory['iTrial'] == iTrial])
                        for iTrial in mouse_trajactory['iTrial'].unique()]
            
        canvas_width = canvas_settings[taskID]['canvas_width']

        for init_coor, trial_behavior in data_groups:
            current_coor = init_coor.copy()
            
            for _, row in trial_behavior.iterrows():
                node = row['node']
                new_coor = current_coor.copy()

                if 'flip' in node:
                    for i in range(9):  # x0 到 x8
                        x_col = f'x{i}'
                        new_coor[x_col] = canvas_width - current_coor[x_col]

                elif node in node_mapping:
                    x_col, y_col = node_mapping[node]

                    # 计算 delta_x 和 delta_y
                    delta_x = row['x'] - current_coor[x_col]
                    delta_y = row['y'] - current_coor[y_col]

                    # 更新当前节点的坐标
                    new_coor[x_col] = row['x']
                    new_coor[y_col] = row['y']
                    
                    # 处理腿的逻辑
                    if node == 'leg_end_0':
                        new_coor['x6'] += delta_x
                        new_coor['y6'] += delta_y
                        new_coor['x5'] -= delta_x
                        new_coor['y5'] += delta_y
                        new_coor['x7'] -= delta_x
                        new_coor['y7'] += delta_y
                    elif node == 'leg_end_1':
                        new_coor['x7'] += delta_x
                        new_coor['y7'] += delta_y
                        new_coor['x4'] -= delta_x
                        new_coor['y4'] += delta_y
                        new_coor['x6'] -= delta_x
                        new_coor['y6'] += delta_y
                    elif node == 'leg_end_2':
                        new_coor['x4'] += delta_x
                        new_coor['y4'] += delta_y
                        new_coor['x5'] -= delta_x
                        new_coor['y5'] += delta_y
                        new_coor['x7'] -= delta_x
                        new_coor['y7'] += delta_y
                    elif node == 'leg_end_3':
                        new_coor['x5'] += delta_x
                        new_coor['y5'] += delta_y
                        new_coor['x4'] -= delta_x
                        new_coor['y4'] += delta_y
                        new_coor['x6'] -= delta_x
                        new_coor['y6'] += delta_y

                    # 处理 head_neck 的逻辑
                    if node == 'head_neck':
                        new_coor['x0'] = current_coor['x0'] + delta_x
                        new_coor['y0'] = current_coor['y0'] + delta_y

                new_coor['timestamp'] = int(row['timestamp'])
                
                if taskID != 'Task1a':
                    new_coor['iTrial'] = init_coor['iTrial']
                
                current_coor = new_coor
                results.append(new_coor)

        return pd.DataFrame(results)

    def distance(self, left_x, left_y, right_x, right_y):
        dx = left_x - right_x
        dy = left_y - right_y
        return np.sqrt(dx * dx + dy * dy)

    def get_angle(self, p1_x, p1_y, p2_x, p2_y):
        return np.arcsin((p1_y - p2_y) / self.distance(p1_x, p1_y, p2_x, p2_y))

    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def transform_coor_to_feature(self, taskID, data, features_range):
        results = []
        
        for _, row in data.iterrows():
            raw_features = {
                'neck_length': self.distance(row.x2, row.y2, row.x1, row.y1),
                'neck_angle': self.get_angle(row.x1, row.y1, row.x2, row.y2),
                'head_length': self.distance(row.x1, row.y1, row.x0, row.y0),
                'head_angle': self.get_angle(row.x0, row.y0, row.x1, row.y1),
                'leg_length': self.distance(row.x4, row.y4, row.x2, row.y2),
                'leg_angle': self.get_angle(row.x4, row.y4, row.x2, row.y2),
                'tail_length': self.distance(row.x3, row.y3, row.x8, row.y8),
                'tail_angle': self.get_angle(row.x8, row.y8, row.x3, row.y3)
            }
            
            normalized_features = {}
            for feature in raw_features:
                value = self.normalize(raw_features[feature],
                                features_range[feature]['min'],
                                features_range[feature]['max'])
                normalized_features[feature] = float(f"{value:.14f}")
            if taskID in ['Task1b', 'Task3c']:
                normalized_features['iTrial'] = int(row['iTrial'])
            normalized_features['timestamp'] = int(row['timestamp'])
            results.append(normalized_features)

        base_columns = ['neck_length', 'head_length', 'leg_length', 'tail_length',
                    'neck_angle', 'head_angle', 'leg_angle', 'tail_angle', 'timestamp']
        columns = ['iTrial'] + base_columns if taskID in ['Task1b', 'Task3c'] else base_columns
        
        return pd.DataFrame(results)[columns]


    ###======================= error analyses ===============================
    def error_calculation(self, processed_data):
        columns = ['neck_length', 'head_length', 'leg_length', 'tail_length']
        
        results = []
        for iSub, group in processed_data.groupby('iSub'):
            target = group[group['type'] == 'target'].reset_index(drop=True)
            adjust_after = group[group['type'] == 'adjust_after'].reset_index(drop=True)

            result = target[['iSub','iTrial'] + columns].reset_index(drop=True).copy()
            for col in columns:
                result[f'{col}_diff'] = adjust_after[col] - target[col]
            results.append(result)
            
        error = pd.concat(results, ignore_index=True)

        return error


    def error_summary(self, error):
        parts = ['neck', 'head', 'leg', 'tail']
        rows = []

        for part in parts:
            length_col = f'{part}_length'
            diff_col = f'{part}_length_diff'

            grouped = (
                error.groupby(['iSub', length_col], as_index=False)[diff_col]
                .agg(['mean', 'std'])
                .reset_index()
                .rename(
                    columns={
                        length_col: 'feature_value',
                        'mean': 'error_mean',
                        'std': 'error_std',
                    }
                )
            )
            grouped.insert(1, 'feature_name', part)
            rows.append(grouped[['iSub', 'feature_name', 'feature_value', 'error_mean', 'error_std']])

        result = pd.concat(rows, ignore_index=True)
        return result.sort_values(['iSub', 'feature_name', 'feature_value']).reset_index(drop=True)


    def plot_error(self, error, attribute, save_path):
        # 根据attribute设置要绘制的列和输出文件名
        if attribute == "length":
            columns_to_plot = ['neck_length_diff', 'head_length_diff', 
                            'leg_length_diff', 'tail_length_diff']
            output_filename = save_path / f'error_length.png'
        elif attribute == "angle":
            columns_to_plot = ['neck_angle_diff', 'head_angle_diff', 
                            'leg_angle_diff', 'tail_angle_diff']
            output_filename = save_path / f'error_angle.png'
        else:
            raise ValueError("attribute must be either 'length' or 'angle'")

        subjects = error['iSub'].unique()
        n_subjects = len(subjects)

        n_rows = 3  # 每列3个图
        n_cols = (n_subjects + n_rows - 1) // n_rows  # 向上取整
        
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))

        y_ticks = [-1, -0.5, 0, 0.5, 1]

        for idx, subject in enumerate(subjects, 1):
            
            subject_data = error[error['iSub'] == subject]
            
            df_long = subject_data[columns_to_plot].melt(
                var_name='Feature', 
                value_name='Error'
            )
            
            ax = plt.subplot(n_rows, n_cols, idx)
            sns.violinplot(x='Feature', y='Error', data=df_long)
            sns.stripplot(x='Feature', y='Error', data=df_long,
                    color='red', alpha=0.3, jitter=0.2, size=4)

            plt.ylim(-1, 1)
            plt.yticks(y_ticks)
            plt.title(f'Subject {subject}', fontsize=14)
            if idx % n_cols == 1:  # 只在每行第一个图上显示y轴标签
                plt.ylabel('Error', fontsize=12)
            else:
                plt.ylabel('')

            x_ticks = range(len(columns_to_plot))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([col.replace('_diff', '') for col in columns_to_plot], 
                            rotation=45, ha='right')
            
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # 根据attribute修改标题
        fig.suptitle(f'{attribute.capitalize()} Error Distribution (adjust_after - target) by Subject', 
                    fontsize=16, y=1.02)
        
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_error_by_feature(self, error, save_path):
        subjects = error['iSub'].unique()
        n_subjects = len(subjects)
        
        n_rows = 3 
        n_cols = (n_subjects + n_rows - 1) // n_rows
        
        feature_pairs = [
            ('neck_length', 'neck_length_diff'),
            ('head_length', 'head_length_diff'),
            ('leg_length', 'leg_length_diff'),
            ('tail_length', 'tail_length_diff')
        ]
        
        y_ticks = [-1, -0.5, 0, 0.5, 1]
        
        for target_feature, diff_feature in feature_pairs:
            fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
            
            for idx, subject in enumerate(subjects, 1):

                subject_data = error[error['iSub'] == subject]
                
                ax = plt.subplot(n_rows, n_cols, idx)
                
                # 创建小提琴图
                sns.violinplot(x=target_feature, y=diff_feature, data=subject_data)
                
                # 添加散点图，使用抖动效果避免重叠
                sns.stripplot(x=target_feature, y=diff_feature, data=subject_data,
                            color='red', alpha=0.3, jitter=0.2, size=4)
                
                plt.ylim(-1, 1)
                plt.yticks(y_ticks)
                
                plt.title(f'Subject {subject}', fontsize=14)
                if idx % n_cols == 1:
                    plt.ylabel('Error', fontsize=12)
                else:
                    plt.ylabel('')
                
                # 设置x轴标签为1-7
                x_ticks = range(7)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([str(i+1) for i in x_ticks])
                
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            feature_name = target_feature.replace('_', ' ').title()
            fig.suptitle(f'{feature_name} : Error Distribution by Subject', 
                        fontsize=16, y=1.02)
            
            output_filename = save_path / f'error_{target_feature}.png'
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            plt.close()


    def analyze_length_error_relationship(self, error, n_points=100):
        """
        为每个被试分别分析长度和误差之间的关系
        
        Parameters:
        error: DataFrame containing the original error data
        n_points: number of points to predict for continuous length values
        
        Returns:
        DataFrame with predicted means and confidence intervals for continuous length values
        """
        results = []
        
        # 获取所有被试ID
        subject_ids = error['iSub'].unique()
        
        for subject_id in subject_ids:
            # 获取当前被试的数据
            subject_data = error[error['iSub'] == subject_id]
            
            for part in ['neck', 'head', 'leg', 'tail']:
                # 提取当前被试当前身体部位的数据并删除缺失值
                part_data = pd.DataFrame({
                    'length': subject_data[f'{part}_length'],
                    'error': subject_data[f'{part}_length_diff']
                }).dropna()
                
                # 如果没有足够的有效数据，跳过这个部位
                if len(part_data) < 3:
                    print(f"警告: 被试 {subject_id} 的 {part} 有效数据不足，已跳过")
                    continue
                
                part_data = part_data.sort_values('length')
                length = part_data['length']
                error_vals = part_data['error']
                
                # 生成连续的长度值
                length_continuous = np.linspace(length.min(), length.max(), n_points)
                
                # 1. 线性回归
                linear_reg = LinearRegression()
                linear_reg.fit(length.values.reshape(-1, 1), error_vals)
                linear_pred = linear_reg.predict(length_continuous.reshape(-1, 1))
                
                # 2. 多项式回归 (3次)
                poly = PolynomialFeatures(degree=3)
                X_poly = poly.fit_transform(length.values.reshape(-1, 1))
                poly_reg = LinearRegression()
                poly_reg.fit(X_poly, error_vals)
                X_continuous_poly = poly.transform(length_continuous.reshape(-1, 1))
                poly_pred = poly_reg.predict(X_continuous_poly)
                
                # 3. 平滑样条
                spline = UnivariateSpline(length, error_vals, k=3, s=len(error_vals))
                spline_pred = spline(length_continuous)
                
                # 计算预测误差的标准差
                # 使用局部窗口来估计不同长度值处的标准差
                window_size = max(len(part_data) // 10, 2)  # 确保窗口至少包含2个点
                
                std_estimates = []
                for x in length_continuous:
                    # 找到距离当前点最近的实际数据点
                    distances = np.abs(length - x)
                    nearest_indices = np.argsort(distances)[:window_size]
                    local_std = error_vals.iloc[nearest_indices].std()
                    std_estimates.append(local_std)
                
                # 存储结果
                for i, x in enumerate(length_continuous):
                    results.append({
                        'iSub': subject_id,
                        'body_part': part,
                        'length': x,
                        'predicted_error_linear': linear_pred[i],
                        'predicted_error_poly': poly_pred[i],
                        'predicted_error_spline': spline_pred[i],
                        'estimated_std': std_estimates[i]
                    })
        
        return pd.DataFrame(results)

    def plot_error_interpolate(self, error, continuous_predictions, save_path):
        # 获取所有被试ID
        subject_ids = error['iSub'].unique()
        n_subjects = len(subject_ids)
        
        # 设置每列显示的图数
        n_rows = 3
        n_cols = (n_subjects + n_rows - 1) // n_rows
        
        # 为每个身体部位创建一个大图
        for body_part in ['neck', 'head', 'leg', 'tail']:
            # 创建大图
            fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
            fig.suptitle(f'{body_part.capitalize()} Length Error Distribution', fontsize=16, y=1)
            
            # 为每个被试创建子图
            for idx, subject_id in enumerate(subject_ids,1):

                # 获取当前被试的数据
                subject_data = error[error['iSub'] == subject_id]
                
                # 获取当前被试的预测数据
                subject_predictions = continuous_predictions[
                    (continuous_predictions['iSub'] == subject_id) & 
                    (continuous_predictions['body_part'] == body_part)
                ]
                
                # 如果没有预测数据，跳过这个被试
                if len(subject_predictions) == 0:
                    continue
                
                # 创建子图
                ax = plt.subplot(n_rows, n_cols, idx)
                
                # 绘制散点图（实际数据点）
                ax.scatter(
                    subject_data[f'{body_part}_length'],
                    subject_data[f'{body_part}_length_diff'],
                    alpha=0.5,
                    label='Actual Data'
                )
                
                # 绘制预测线
                ax.plot(
                    subject_predictions['length'],
                    subject_predictions['predicted_error_linear'],
                    'r-',
                    label='Linear',
                    alpha=0.7
                )
                ax.plot(
                    subject_predictions['length'],
                    subject_predictions['predicted_error_poly'],
                    'g-',
                    label='Polynomial',
                    alpha=0.7
                )
                ax.plot(
                    subject_predictions['length'],
                    subject_predictions['predicted_error_spline'],
                    'b-',
                    label='Spline',
                    alpha=0.7
                )
                
                # 添加误差范围
                ax.fill_between(
                    subject_predictions['length'],
                    subject_predictions['predicted_error_spline'] - subject_predictions['estimated_std'],
                    subject_predictions['predicted_error_spline'] + subject_predictions['estimated_std'],
                    color='blue',
                    alpha=0.1
                )
                
                # 设置标题和标签
                ax.set_title(f'Subject {subject_id}')
                ax.set_xlabel('Length')
                ax.set_ylabel('Error')
                
                # 只在第一个子图显示图例
                if idx == 0:
                    ax.legend()
            
            # 调整子图之间的间距
            plt.tight_layout()
            
            # 保存图片
            output_filename = save_path / f'error_{body_part}_distribution.png'
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()
