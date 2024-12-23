"""
基于原型的类别中心点生成
"""

import pandas as pd
import numpy as np
import itertools

# define partition rules
class Partition:
    """
    Partition类:
    1. 使用 get_all_splits(n_dims, n_cats) 生成各种分割方式对应的超平面组合。
    2. 使用 get_centers(n_dims, n_cats) 计算在这些分割方式下, 各个区域(类别)的代表中心点(重心)。
    """

    def get_all_splits(self, n_dims : int, n_cats : int):
        """
        根据维度数 n_dims 和分类数 n_cats, 生成所有可用的分割方式 (split_type) 及对应的超平面列表 hyperplanes.
        
        返回值: 
          List[ (split_type: str, hyperplanes: List[(coef_list, threshold)]) ]
        
        其中:
          - split_type: 字符串, 标识某种分割方式.
          - hyperplanes: 若干个超平面, 每个超平面用 (coefs, threshold) 表示:
                coefs: [float]*n_dims, 各维度对应的系数
                threshold: float, 表示 coefs·x = threshold
          
        比如 "x_i = 0.5" 可表示为 ([0, ..., 1, ..., 0], 0.5) 其中第 i 个位置系数=1, 其余=0.
        """
        splits = []

        # --------------- 两分类情况 ---------------
        if n_cats == 2: 
            # 1. 单维轴对齐超平面 (Axis-aligned): x_i = 0.5
            for i in range(n_dims):
                coeff = [0] * n_dims
                coeff[i] = 1
                splits.append(('axis', [(tuple(coeff), 0.5)]))

            # 2a. 二维相等超平面 (Equality): x_i = x_j
            for i in range(n_dims):
                for j in range(i+1, n_dims):
                    coeff = [0] * n_dims
                    coeff[i] = 1
                    coeff[j] = -1
                    splits.append(('equality', [(tuple(coeff), 0)]))

            # 2b. 二维和式超平面 (Sum): x_i + x_j = 1
            for i in range(n_dims):
                for j in range(i+1, n_dims):
                    coeff = [0] * n_dims
                    coeff[i] = 1
                    coeff[j] = 1
                    splits.append(('sum', [(tuple(coeff), 1)]))
            
            # 3. 四维混合超平面 (Mixed): x_i + x_j = x_k + x_l
            if n_dims >= 4:
                dim_pairs = [
                    ((0,1,2,3)), # x1 + x2 = x3 + x4
                    ((0,2,1,3)), # x1 + x3 = x2 + x4
                    ((0,3,1,2))  # x1 + x4 = x2 + x3
                ]
                for i, j, k, l in dim_pairs:
                    coeff = [0] * n_dims
                    coeff[i] = coeff[j] = 1
                    coeff[k] = coeff[l] = -1
                    splits.append(('mixed', [(tuple(coeff), 0)]))
        
        # --------------- 四分类情况 ---------------
        elif n_cats == 4:
            # 1. 两个超平面
            # 1.1 涉及两个维度
            # 1.1a 两个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5)
            for i in range(n_dims):
                for j in range(i+1, n_dims):
                    plane1 = ([0]*n_dims, 0.5)
                    plane2 = ([0]*n_dims, 0.5)
                    plane1[0][i] = 1
                    plane2[0][j] = 1
                    splits.append(('2d_axis_pair', [plane1, plane2]))

            # 1.1b 二维相等 + 二维和式: (x_i = x_j, x_i + x_j = 1)
            for i, j in itertools.combinations(range(n_dims), 2):
                plane1 = ([0]*n_dims, 0)
                plane2 = ([0]*n_dims, 1)
                plane1[0][i], plane1[0][j] = 1, -1
                plane2[0][i] = plane2[0][j] = 1
                splits.append(('2d_equality_sum', [plane1, plane2]))
            
            # 1.2 涉及三个维度
            # 1.2a 单维轴对齐 + 二维相等: (x_i = 0.5, x_j = x_k)
            if n_dims >= 3:
                for i in range(n_dims):
                    remaining = [j for j in range(n_dims) if j != i]
                    for j, k in itertools.combinations(remaining, 2):
                        plane1 = ([0]*n_dims, 0.5)
                        plane2 = ([0]*n_dims, 0)
                        plane1[0][i] = 1
                        plane2[0][j], plane2[0][k] = 1, -1
                        splits.append(('3d_axis_equality', [plane1, plane2]))

            # 1.2b 单维轴对齐 + 二维和式: (x_i = 0.5, x_j + x_k = 1)
            if n_dims >= 3:
                for i in range(n_dims):
                    remaining = [j for j in range(n_dims) if j != i]
                    for j, k in itertools.combinations(remaining, 2):
                        plane1 = ([0]*n_dims, 0.5)
                        plane2 = ([0]*n_dims, 1)
                        plane1[0][i] = 1
                        plane2[0][j] = plane2[0][k] = 1
                        splits.append(('3d_axis_sum', [plane1, plane2]))

            # 1.3 涉及四个维度
            # 1.3a 两个二维相等超平面: (x_i = x_j, x_k = x_l)
            if n_dims >= 4:
                dim_pairs = [
                    ((0,1,2,3)), # x1 = x2, x3 = x4
                    ((0,2,1,3)), # x1 = x3, x2 = x4
                    ((0,3,1,2))  # x1 = x4, x2 = x3
                ]
                for i, j, k, l in dim_pairs:
                    plane1 = ([0]*n_dims, 0)
                    plane2 = ([0]*n_dims, 0)
                    plane1[0][i], plane1[0][j] = 1, -1
                    plane2[0][k], plane2[0][l] = 1, -1
                    splits.append(('4d_equality_pair', [plane1, plane2]))

            # 1.3b 两个二维和式超平面: (x_i + x_j = 1, x_k + x_l = 1)
            if n_dims >= 4:
                dim_pairs = [
                    ((0,1,2,3)), # x1 + x2 = 1, x3 + x4 = 1
                    ((0,2,1,3)), # x1 + x3 = 1, x2 + x4 = 1
                    ((0,3,1,2))  # x1 + x4 = 1, x2 + x3 = 1
                ]
                for i, j, k, l in dim_pairs:
                    plane1 = ([0]*n_dims, 1)
                    plane2 = ([0]*n_dims, 1)
                    plane1[0][i] = plane1[0][j] = 1
                    plane2[0][k] = plane2[0][l] = 1
                    splits.append(('4d_sum_pair', [plane1, plane2]))
            
            # 2. 三个超平面
            # 2.1 涉及三个维度
            # 2.1a 三个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5, x_k = 0.5)
            if n_dims >= 3:
                for i, j, k in itertools.combinations(range(n_dims), 3):
                    # 考虑所有可能的排列顺序
                    for m, n1, n2 in itertools.permutations([i, j, k]):
                        plane_m = ([0]*n_dims, 0.5)
                        plane_n1 = ([0]*n_dims, 0.5)
                        plane_n2 = ([0]*n_dims, 0.5)
                        plane_m[0][m] = 1
                        plane_n1[0][n1] = 1
                        plane_n2[0][n2] = 1
                        splits.append(('3d_axis_triple', [plane_m, plane_n1, plane_n2]))

            # 2.1b 单维轴对齐 + 二维相等 + 二维和式: (x_i = 0.5, x_j = x_k, x_j + x_k = 1)
            if n_dims >= 3:
                for m in range(n_dims):
                    remaining = [i for i in range(n_dims) if i != m]
                    for i, j in itertools.combinations(remaining, 2):
                        plane_m = ([0]*n_dims, 0.5)  # 轴对齐
                        plane_eq = ([0]*n_dims, 0)   # 相等
                        plane_sum = ([0]*n_dims, 1)  # 和式
                        
                        plane_m[0][m] = 1
                        plane_eq[0][i], plane_eq[0][j] = 1, -1
                        plane_sum[0][i] = plane_sum[0][j] = 1
                    
                        splits.append(('3d_axis_equality_sum', [plane_m, plane_eq, plane_sum]))
                        splits.append(('3d_axis_equality_sum', [plane_m, plane_sum, plane_eq]))

            # 2.2 涉及四个维度
            # 2.2a 二维相等 + 两个单维轴对齐: (x_i = x_j, x_k = 0.5, x_l = 0.5)
            if n_dims >= 4:
                for i, j in itertools.combinations(range(n_dims), 2):
                    remaining = [k for k in range(n_dims) if k not in (i, j)]
                    for k, l in itertools.combinations(remaining, 2):
                        plane_eq = ([0]*n_dims, 0)  # 二维相等
                        plane_axis1 = ([0]*n_dims, 0.5)  # 单维轴对齐
                        plane_axis2 = ([0]*n_dims, 0.5)  # 单维轴对齐
                        
                        plane_eq[0][i], plane_eq[0][j] = 1, -1
                        plane_axis1[0][k] = 1
                        plane_axis2[0][l] = 1
                        
                        # 固定二维相等在第一个位置，后两个单维轴对齐排列组合
                        splits.append(('4d_equality_axis_pair', [plane_eq, plane_axis1, plane_axis2]))
                        splits.append(('4d_equality_axis_pair', [plane_eq, plane_axis2, plane_axis1]))
            
            # 2.2b 二维和式 + 两个单维轴对齐: (x_i + x_j = 1, x_k = 0.5, x_l = 0.5)
            if n_dims >= 4:
                for i, j in itertools.combinations(range(n_dims), 2):
                    remaining = [k for k in range(n_dims) if k not in (i, j)]
                    for k, l in itertools.combinations(remaining, 2):
                        plane_sum = ([0]*n_dims, 1)  # 二维和式
                        plane_axis1 = ([0]*n_dims, 0.5)  # 单维轴对齐
                        plane_axis2 = ([0]*n_dims, 0.5)  # 单维轴对齐
                        
                        plane_sum[0][i], plane_sum[0][j] = 1, 1
                        plane_axis1[0][k] = 1
                        plane_axis2[0][l] = 1
                        
                        # 固定二维和式在第一个位置，后两个单维轴对齐排列组合
                        splits.append(('4d_sum_axis_pair', [plane_sum, plane_axis1, plane_axis2]))
                        splits.append(('4d_sum_axis_pair', [plane_sum, plane_axis2, plane_axis1]))

            # 3. C(n_dims,2)个超平面（所有二维相等超平面：x_i = x_j）
            # 当 n_dims == n_cats = 4 时, 这些超平面可组合成"哪一维最大"的划分(dimension_max)
            if n_dims == n_cats:
                eq_planes = []
                for i, j in itertools.combinations(range(n_dims), 2):
                    plane = ([0]*n_dims, 0)
                    plane[0][i], plane[0][j] = 1, -1
                    eq_planes.append(plane)

                splits.append(('dimension_max', eq_planes))

        return splits


    # generate centers
    def get_centers(self, n_dims : int, n_cats : int):
        """
        计算每种分割方法下各个区域的中心点(重心).

        返回: List[ (split_type, {cat_idx: (center_x1, center_x2, ...)}) ]
        """
        splits = self.get_all_splits(n_dims, n_cats)
        results = []
        
        for split_type, hyperplanes in splits:
            centers = {cat_idx: [] for cat_idx in range(n_cats)}

            # --------------- 两分类情况 ---------------
            if n_cats == 2:
                # 1. 单维轴对齐超平面: x_i = 0.5
                if split_type == 'axis':
                    split_dim = next(dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef != 0)

                    for dim in range(n_dims):
                        if dim == split_dim:
                            centers[0].append(0.25)  # x < 0.5 的区域
                            centers[1].append(0.75)  # x > 0.5 的区域
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                
                # 2a. 二维相等超平面: x_i = x_j
                elif split_type == 'equality':
                    split_dims = [dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef != 0]
                    dim1, dim2 = split_dims[0], split_dims[1]
                    
                    # 对于涉及的两个维度，一个区域是(1/3, 2/3)，另一个是(2/3, 1/3)
                    for dim in range(n_dims):
                        if dim == dim1:
                            centers[0].append(1/3)
                            centers[1].append(2/3)
                        elif dim == dim2:
                            centers[0].append(2/3)
                            centers[1].append(1/3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                
                # 2b. 二维和式超平面: x_i + x_j = 1
                elif split_type == 'sum':
                    split_dims = [dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef != 0]
                    dim1, dim2 = split_dims[0], split_dims[1]
                    
                    # 对于涉及的两个维度，一个区域是(1/3, 1/3)，另一个是(2/3, 2/3)
                    for dim in range(n_dims):
                        if dim == dim1 or dim == dim2:
                            centers[0].append(1/3)
                            centers[1].append(2/3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                
                # 3. 四维混合超平面: x_i + x_j = x_k + x_l
                elif split_type == 'mixed':
                    pos_dims = [dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef == 1]
                    neg_dims = [dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef == -1]
                    
                    for dim in range(n_dims):
                        if dim in pos_dims:
                            centers[0].append(1/3)
                            centers[1].append(2/3)
                        elif dim in neg_dims:
                            centers[0].append(2/3)
                            centers[1].append(1/3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

            # --------------- 四分类情况 ---------------
            elif n_cats == 4:
                # 1.1a 两个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5)
                if split_type == '2d_axis_pair':
                    split_dims = []
                    for hyperplane in hyperplanes:
                        dim_idx = next(dim_idx for dim_idx, coef in enumerate(hyperplane[0]) if coef != 0)
                        split_dims.append(dim_idx)
                    
                    for dim in range(n_dims):
                        if dim == split_dims[0]:
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        elif dim == split_dims[1]:
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.1b 二维相等 + 二维和式: (x_i = x_j, x_i + x_j = 1)
                elif split_type == '2d_equality_sum':
                    split_dims = [dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef != 0]
                    
                    for dim in range(n_dims):
                        if dim in split_dims:
                            if dim == split_dims[0]:  # i
                                centers[0].append(1/2)
                                centers[1].append(1/2)
                                centers[2].append(5/6)
                                centers[3].append(1/6)
                            else:  # j
                                centers[0].append(1/6)
                                centers[1].append(5/6)
                                centers[2].append(1/2)
                                centers[3].append(1/2)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.2a 单维轴对齐 + 二维相等: (x_i = 0.5, x_j = x_k)
                # 1.2b 单维轴对齐 + 二维和式: (x_i = 0.5, x_j + x_k = 1)
                elif split_type in ['3d_axis_equality', '3d_axis_sum']:
                    axis_hyperplane = next(plane for plane in hyperplanes if sum(1 for c in plane[0] if c != 0) == 1)
                    other_hyperplane = next(plane for plane in hyperplanes if plane != axis_hyperplane)
                    
                    axis_dim = next(dim_idx for dim_idx, coef in enumerate(axis_hyperplane[0]) if coef != 0)
                    other_dims = [dim_idx for dim_idx, coef in enumerate(other_hyperplane[0]) if coef != 0]
                    
                    for dim in range(n_dims):
                        if dim == axis_dim:
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim in other_dims:
                            if split_type == '3d_axis_equality':
                                if dim == other_dims[0]:
                                    centers[0].append(1/3)
                                    centers[1].append(2/3)
                                    centers[2].append(1/3)
                                    centers[3].append(2/3)
                                else:
                                    centers[0].append(2/3)
                                    centers[1].append(1/3)
                                    centers[2].append(2/3)
                                    centers[3].append(1/3)
                            else:  # 3d_axis_sum
                                centers[0].append(1/3)
                                centers[1].append(2/3)
                                centers[2].append(1/3)
                                centers[3].append(2/3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.3a 两个二维相等超平面: (x_i = x_j, x_k = x_l)
                elif split_type == '4d_equality_pair':
                    split_dim_pairs = []
                    for hyperplane in hyperplanes:
                        dim_pair = [dim_idx for dim_idx, coef in enumerate(hyperplane[0]) if coef != 0]
                        split_dim_pairs.append(dim_pair)
                    
                    for dim in range(n_dims):
                        if dim in split_dim_pairs[0]:
                            if dim == split_dim_pairs[0][0]:  # i
                                centers[0].append(1/3)
                                centers[1].append(2/3)
                                centers[2].append(1/3)
                                centers[3].append(2/3)
                            else:  # j
                                centers[0].append(2/3)
                                centers[1].append(1/3)
                                centers[2].append(2/3)
                                centers[3].append(1/3)
                        elif dim in split_dim_pairs[1]:
                            if dim == split_dim_pairs[1][0]:  # k
                                centers[0].append(1/3)
                                centers[1].append(1/3)
                                centers[2].append(2/3)
                                centers[3].append(2/3)
                            else:  # l
                                centers[0].append(2/3)
                                centers[1].append(2/3)
                                centers[2].append(1/3)
                                centers[3].append(1/3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.3b 两个二维和式超平面: (x_i + x_j = 1, x_k + x_l = 1)
                elif split_type == '4d_sum_pair':
                    split_dim_pairs = []
                    for hyperplane in hyperplanes:
                        dim_pair = [dim_idx for dim_idx, coef in enumerate(hyperplane[0]) if coef != 0]
                        split_dim_pairs.append(dim_pair)
                    
                    for dim in range(n_dims):
                        if dim in split_dim_pairs[0]:
                            centers[0].append(1/3)
                            centers[1].append(2/3)
                            centers[2].append(1/3)
                            centers[3].append(2/3)
                        elif dim in split_dim_pairs[1]:
                            centers[0].append(1/3)
                            centers[1].append(1/3)
                            centers[2].append(2/3)
                            centers[3].append(2/3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 2.1a 三个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5, x_k = 0.5)
                elif split_type == '3d_axis_triple':
                    split_dims = []
                    for hyperplane in hyperplanes:
                        dim_idx = next(dim_idx for dim_idx, coef in enumerate(hyperplane[0]) if coef != 0)
                        split_dims.append(dim_idx)
                    
                    for dim in range(n_dims):
                        if dim == split_dims[0]:  # 第一个切割
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim == split_dims[1]:  # 第二个切割 (x1 < 0.5 部分)
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.5)
                            centers[3].append(0.5)
                        elif dim == split_dims[2]:  # 第三个切割 (x1 > 0.5 部分)
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 2.1b 单维轴对齐 + 二维相等 + 二维和式: (x_i = 0.5, x_j = x_k, x_j + x_k = 1)
                elif split_type == '3d_axis_equality_sum':
                    axis_dim = next(dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef != 0)
                    other_dims = set()
                    for hyperplane in hyperplanes[1:]:
                        other_dims.update(dim_idx for dim_idx, coef in enumerate(hyperplane[0]) if coef != 0)
                    other_dims = list(other_dims)
                    
                    # 检查第二个超平面是否为二维相等
                    is_second_equality = sum(1 for c in hyperplanes[1][0] if c != 0) == 2 and hyperplanes[1][1] == 0
                    
                    for dim in range(n_dims):
                        if dim == axis_dim:  # x1
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim in other_dims:
                            if is_second_equality: # 第二个超平面是二维相等
                                # x1 < 0.5 部分用 x2 = x3, x1 > 0.5 部分用 x2 + x3 = 1
                                if dim == other_dims[0]:  # x2
                                    centers[0].append(1/3)
                                    centers[1].append(2/3)
                                    centers[2].append(1/3)
                                    centers[3].append(2/3)
                                else:  # x3
                                    centers[0].append(2/3)
                                    centers[1].append(1/3)
                                    centers[2].append(1/3)
                                    centers[3].append(2/3)
                            else: # 第二个超平面是二维和式
                                # x1 < 0.5 部分用 x2 + x3 = 1, x1 > 0.5 部分用 x2 = x3
                                if dim == other_dims[0]:  # x2
                                    centers[0].append(1/3)
                                    centers[1].append(2/3)
                                    centers[2].append(1/3)
                                    centers[3].append(2/3)
                                else:  # x3
                                    centers[0].append(1/3)
                                    centers[1].append(2/3)
                                    centers[2].append(2/3)
                                    centers[3].append(1/3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 2.2a 二维相等 + 两个单维轴对齐: (x_i = x_j, x_k = 0.5, x_l = 0.5)
                # 2.2b 二维和式 + 两个单维轴对齐: (x_i + x_j = 1, x_k = 0.5, x_l = 0.5)
                elif split_type in ['4d_equality_axis_pair', '4d_sum_axis_pair']:
                    first_dims = [dim_idx for dim_idx, coef in enumerate(hyperplanes[0][0]) if coef != 0]
                    axis_dims = []
                    for hyperplane in hyperplanes[1:]:
                        dim_idx = next(dim_idx for dim_idx, coef in enumerate(hyperplane[0]) if coef != 0)
                        axis_dims.append(dim_idx)
                    
                    for dim in range(n_dims):
                        if dim in first_dims:
                            if split_type == '4d_equality_axis_pair':
                                if dim == first_dims[0]:  # i
                                    centers[0].append(1/3)
                                    centers[1].append(1/3)
                                    centers[2].append(2/3)
                                    centers[3].append(2/3)
                                else:  # j
                                    centers[0].append(2/3)
                                    centers[1].append(2/3)
                                    centers[2].append(1/3)
                                    centers[3].append(1/3)
                            else:  # 4d_sum_axis_pair
                                centers[0].append(1/3)
                                centers[1].append(1/3)
                                centers[2].append(2/3)
                                centers[3].append(2/3)
                        elif dim == axis_dims[0]:  # k 
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.5)
                            centers[3].append(0.5)
                        elif dim == axis_dims[1]:  # l 
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 3. C(n_dims,2)个超平面（所有二维相等超平面：x_i = x_j）
                elif split_type == 'dimension_max':
                    for cat_idx in range(n_cats):
                        center_coords = [0.4] * n_dims  # 先令所有维度都为 0.4
                        center_coords[cat_idx] = 0.8    # 第 cat_idx 维取 0.8
                        centers[cat_idx] = tuple(center_coords)

            # 将列表转换为元组
            centers = {k: tuple(v) for k, v in centers.items()}
            results.append((split_type, centers))
        
        return results

    
