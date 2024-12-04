"""
对Task1b进行误差分析
"""

import pandas as pd
import numpy as np
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
            
        final_results = pd.concat(results, ignore_index=True)

        return final_results


    def error_summary(error):
        parts = ['neck', 'head', 'leg', 'tail']
        
        # 创建基础数据框架
        lengths = error['neck_length'].unique()
        subs = error['iSub'].unique()
        base_df = pd.DataFrame([(sub, length) for sub in subs for length in lengths],
                            columns=['iSub', 'length'])
        
        # 为每个部位计算统计量
        stats = []
        for part in parts:
            # 分组计算均值和标准差
            grouped = error.groupby(['iSub', f'{part}_length'])[f'{part}_length_diff'].agg(['mean', 'std']).reset_index()
            
            # 重命名列
            grouped.columns = ['iSub', f'{part}_length', 
                            f'{part}_length_error_mean', f'{part}_length_error_sd']
            stats.append(grouped)
        
        # 合并所有统计结果
        result = base_df.copy()
        for part in parts:
            result[f'{part}_length'] = result['length']
        result = result.drop('length', axis=1)
        
        # 合并统计数据
        for stat_df in stats:
            merge_cols = ['iSub', f'{stat_df.columns[1].split("_")[0]}_length']
            result = result.merge(stat_df, on=merge_cols, how='left')
        
        # 排序并返回结果
        return result.sort_values(['iSub', 'neck_length'])
