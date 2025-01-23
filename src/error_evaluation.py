"""
对Task1b进行误差分析
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

class Processor:

    def error_calculation(self, processed_data):
        columns = ['neck_length', 'head_length', 'leg_length', 'tail_length', 
                'neck_angle', 'head_angle', 'leg_angle', 'tail_angle']
        
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
        
        lengths = error['neck_length'].unique()
        subs = error['iSub'].unique()
        base_df = pd.DataFrame([(sub, length) for sub in subs for length in lengths],
                            columns=['iSub', 'length'])
        
        stats = []
        for part in parts:
            grouped = error.groupby(['iSub', f'{part}_length'])[f'{part}_length_diff'].agg(['mean', 'std']).reset_index()
            
            grouped.columns = ['iSub', f'{part}_length', 
                            f'{part}_length_error_mean', f'{part}_length_error_sd']
            stats.append(grouped)
        
        result = base_df.copy()
        for part in parts:
            result[f'{part}_length'] = result['length']
        result = result.drop('length', axis=1)
        
        for stat_df in stats:
            merge_cols = ['iSub', f'{stat_df.columns[1].split("_")[0]}_length']
            result = result.merge(stat_df, on=merge_cols, how='left')
        
        return result.sort_values(['iSub', 'neck_length'])


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
            # 计算行列位置（按列优先排列）
            row = (idx - 1) % n_rows
            col = (idx - 1) // n_rows
            plot_position = row * n_cols + col + 1

            subject_data = error[error['iSub'] == subject]
            
            # 将数据重塑为长格式
            df_long = subject_data[columns_to_plot].melt(
                var_name='Feature', 
                value_name='Error'
            )
            
            ax = plt.subplot(n_rows, n_cols, plot_position)
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
                # 计算行列位置（按列优先排列）
                row = (idx - 1) % n_rows
                col = (idx - 1) // n_rows
                plot_position = row * n_cols + col + 1

                subject_data = error[error['iSub'] == subject]
                
                ax = plt.subplot(n_rows, n_cols, plot_position)
                
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
        n_cols = ceil(n_subjects / n_rows)
        
        # 为每个身体部位创建一个大图
        for body_part in ['neck', 'head', 'leg', 'tail']:
            # 创建大图
            fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
            fig.suptitle(f'{body_part.capitalize()} Length Error Distribution', fontsize=16, y=1)
            
            # 为每个被试创建子图
            for idx, subject_id in enumerate(subject_ids):
                # 计算行列位置（按列优先排列）
                row = idx % n_rows
                col = idx // n_rows
                plot_position = row * n_cols + col + 1

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
                ax = plt.subplot(n_rows, n_cols, plot_position)
                
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


    def process(self, stimulus_data, continuous_predictions):
          
        # 处理每个身体部位
        for body_part in ['neck', 'head', 'leg', 'tail']:
            length_col = f'{body_part}_length'
            
            # 对每一行应用get_error_for_length函数
            errors = []
            for length in stimulus_data[length_col]:
                subject_predictions = continuous_predictions[
                    continuous_predictions['body_part'] == body_part
                ]
                
                spline_interpolator = interp1d(
                    subject_predictions['length'],
                    subject_predictions['predicted_error_spline'],
                    bounds_error=False,  # 如果超出范围，返回nan
                    fill_value="extrapolate"  # 或者可以外推
                )

                predicted_error = float(spline_interpolator(length))
                errors.append(length + predicted_error)
            
            # 将处理后的长度添加到结果数据框
            stimulus_data[length_col] = errors
            
        return pd.DataFrame(stimulus_data)