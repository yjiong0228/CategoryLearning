"""
对所有任务的行为数据进行预处理
"""

import pandas as pd
import numpy as np
 
## Preprocessor of task1a & task1b & task3c ##
class Preprocessor_A:
    def process(self, taskID, feature_init, mouse_trajactory, features_range, canvas_settings, body_length):
        # 移动前的初始特征值
        feature_init_real = self.transform_to_real_feature(features_range, feature_init)

        # 转换为坐标
        if taskID == 'Task1a':
            feature_init_real = self.extract_body_ori(feature_init_real, mouse_trajactory, canvas_settings)
        coor_init = self.transform_feature_to_coor(taskID, feature_init_real, canvas_settings, body_length)

        # 得到移动后的坐标
        coor_trajactory = self.complete_coor(taskID, coor_init, mouse_trajactory, canvas_settings)
        feature_trajactory = self.transform_coor_to_feature(taskID, coor_trajactory, features_range)
        
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


## Preprocessor of task2 ##
class Preprocessor_B:
    def process(self, taskID, stimulus_data, behavior_data):
        if taskID in ['Task2', 'Task3a']:
            joint_data = pd.merge(stimulus_data, behavior_data, on=['iSession', 'stiID'], suffixes=('', '_y'))
            joint_data = joint_data.drop('category_y', axis=1)

            version = behavior_data['version'][0]
            structure1 = behavior_data['structure1'][0]
            structure2 = behavior_data['structure2'][0]

            # Convert exact features into feature1-4
            if version == 1:
                feature_names = self.convert("_length", structure1, structure2)
            else:
                feature_names = self.convert("_angle", structure1, structure2)

            rename_map = {
                feature_names[i]: f'feature{i+1}' for i in range(4)
            }

            base_columns = ['version', 'condition', 'iSession', 'iBlock', 'iTrial',
                            'neck_length', 'head_length', 'leg_length', 'tail_length',
                            'neck_angle', 'head_angle', 'leg_angle', 'tail_angle']
            remaining_columns = ['category', 'choice', 'feedback', 'choRT']
   
            combined_data = joint_data[base_columns].copy()

            for i, feature in enumerate(feature_names):
                new_name = f'feature{i+1}'
                combined_data[new_name] = joint_data[feature]

            combined_data[remaining_columns] = joint_data[remaining_columns]

            combined_data = combined_data.sort_values(by=['iSession', 'iBlock', 'iTrial'])

        return combined_data

    def convert(self, suffix, structure1, structure2):
        # feature selection
        if structure1 == 1:
            features = ["neck", "head", "leg", "tail"]
        elif structure1 == 2:
            features = ["neck", "head", "tail", "leg"]
        elif structure1 == 3:
            features = ["neck", "leg", "tail", "head"]
        elif structure1 == 4:
            features = ["head", "leg", "tail", "neck"]
        
        # feature space segmentation
        if structure2 == 1:
            features = features[:]
        elif structure2 == 2:
            features = [features[0], features[2], features[1], features[3]]
        elif structure2 == 3:
            features = [features[1], features[0], features[2], features[3]]
        elif structure2 == 4:
            features = [features[1], features[2], features[0], features[3]]
        elif structure2 == 5:
            features = [features[2], features[0], features[1], features[3]]
        elif structure2 == 6:
            features = [features[2], features[1], features[0], features[3]]
        
        # Final rearrangement
        features = [features[0], features[2], features[1], features[3]]
        
        # Add suffix to feature names
        return [f + suffix for f in features]