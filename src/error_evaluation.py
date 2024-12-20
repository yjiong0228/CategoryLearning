"""
对Task1b进行误差分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline

class Processor:
    # def process(self, processed_data):

                

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


    def error_summary(error):
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


    def plot_error(self, error, attribute):
        # 根据attribute设置要绘制的列和输出文件名
        if attribute == "length":
            columns_to_plot = ['neck_length_diff', 'head_length_diff', 
                            'leg_length_diff', 'tail_length_diff']
            output_filename = 'error_length.png'
        elif attribute == "angle":
            columns_to_plot = ['neck_angle_diff', 'head_angle_diff', 
                            'leg_angle_diff', 'tail_angle_diff']
            output_filename = 'error_angle.png'
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

    def plot_error_by_feature(self, difference_df, output_prefix='error'):
        subjects = difference_df['iSub'].unique()
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

                subject_data = difference_df[difference_df['iSub'] == subject]
                
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
            
            output_filename = f'{output_prefix}_{target_feature}.png'
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            plt.close()


