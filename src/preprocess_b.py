"""
对Task2, Task3a, Task3b的数据进行预处理
"""

import re
import pandas as pd
import numpy as np
 
class Preprocessor_B:
    def process(self, taskID, stimulus_data, behavior_data, recording_data = None):
        if taskID in ['Task2', 'Task3a']:
            joint_data = pd.merge(stimulus_data, behavior_data, on=['iSession', 'stiID'], suffixes=('', '_y'))
            joint_data = joint_data.drop('category_y', axis=1)

            version = behavior_data['version'][0]
            structure1 = behavior_data['structure1'][0]
            structure2 = behavior_data['structure2'][0]

            # Convert exact features into feature1-4
            if version == 1:
                feature_names = self.convert("_length", [structure1, structure2])
            else:
                feature_names = self.convert("_angle", [structure1, structure2])

            base_columns = ['version', 'condition', 'iSession', 'iBlock', 'iTrial',
                            'neck_length', 'head_length', 'leg_length', 'tail_length',
                            'neck_angle', 'head_angle', 'leg_angle', 'tail_angle']
            remaining_columns = ['category', 'choice', 'feedback', 'ambigous', 'choRT']
   
            combined_data = joint_data[base_columns].copy()

            for i, feature in enumerate(feature_names):
                new_name = f'feature{i+1}'
                combined_data[new_name] = joint_data[feature]

            combined_data[remaining_columns] = joint_data[remaining_columns]

            combined_data = combined_data.sort_values(by=['iSession', 'iBlock', 'iTrial'])

        if taskID in ['Task2']:
            rec_processor = Recording_Processor()
            recording_coded = rec_processor.process(recording_data, [structure1, structure2])
            combined_data = pd.merge(combined_data, recording_coded, on=['iSession', 'iTrial'])

        return combined_data

    def convert(self, suffix, structure):
        # feature selection
        if structure[0] == 1:
            features = ["neck", "head", "leg", "tail"]
        elif structure[0] == 2:
            features = ["neck", "head", "tail", "leg"]
        elif structure[0] == 3:
            features = ["neck", "leg", "tail", "head"]
        elif structure[0] == 4:
            features = ["head", "leg", "tail", "neck"]
        
        # feature space segmentation
        if structure[1] == 1:
            features = features[:]
        elif structure[1] == 2:
            features = [features[0], features[2], features[1], features[3]]
        elif structure[1] == 3:
            features = [features[1], features[0], features[2], features[3]]
        elif structure[1] == 4:
            features = [features[1], features[2], features[0], features[3]]
        elif structure[1] == 5:
            features = [features[2], features[0], features[1], features[3]]
        elif structure[1] == 6:
            features = [features[2], features[1], features[0], features[3]]
        
        # Final rearrangement
        features = [features[0], features[2], features[1], features[3]]
        
        # Add suffix to feature names
        return [f + suffix for f in features]
    

class Recording_Processor:
    def __init__(self):
        """初始化处理器，定义各类关键词映射"""
        self.body_parts = {
            '脖子': 0,
            '头': 1,
            '腿': 2,
            '尾巴': 3,
        }

        self.direct_descriptions = {
            '长': 0.75,
            '短': 0.25,
            '中等': 0.5,
            '适中': 0.5
        }

        self.comp_descriptions = {
            '长': 2/3,
            '短': 1/3
        }

        self.max_descriptions = {
            '长': 0.8,
            '短': 0.4
        }

        self.quantifiers = {'四个', '所有', '每一个', '每个', '全部', '各'}
        self.exclude_pattern = re.compile(r'除(?:了)?([^，。]+?)外')
        self.punctuation = '。.？?！!、'

    def process(self, file_path, new_file_path):

        df = pd.read_csv(file_path)
        texts = df['text']
        res = {
            'val': [],
            'un_pro': [],
            'origin': [],
        }
        for text in texts:
            vals, un_pro = self.extract_values(text)
            if len(vals) == 0:
                vals = [[-1]*4]
            vals = np.array(vals).mean(axis=0).tolist()
            res['val'].append(vals)
            res['un_pro'].append(un_pro)
            res['origin'].append(text)
        

        result_df = pd.DataFrame(res)
        result_df.to_csv(new_file_path, index=False)
    
    def extract_values(self, text):
        """主处理函数"""
        results = []
        un_pro = []

        if pd.isna(text) or not str(text).strip():
            results.append([-1]*4)

        # 分割并清理items
        raw_items = re.split(r'[，,]', str(text))
        items = [self._clean_item(i) for i in raw_items]

        items = [i for i in items if i != '']  # 去除空值        

        i = 0
        while i < len(items):
            # 处理排除逻辑
            if '除' in items[i]:
                res, is_pro = self._handle_exclusion(items[i], items[i+1])
                if is_pro:
                    results.append(res)
                    i += 2
                    continue


            # 处理最高级
            res, is_pro = self._handle_superlative(items[i])
            if is_pro:
                results.append(res)
                i += 1
                continue


            # 处理全称量词
            res, is_pro = self._handle_universal_quantifier(items[i])
            if is_pro:
                results.append(res)
                i += 1
                continue

            
            # 处理'只有'逻辑
            res, is_pro = self._handle_exclusive_case(items[i])
            if is_pro:
                results.append(res)
                i += 1
                continue


            # 处理比较逻辑
            res, is_pro = self._handle_comparison(items[i])
            if is_pro:
                results.append(res)
                i += 1
                continue

            # 处理普通描述逻辑
            res, is_pro = self._handle_general_case(items[i])
            if is_pro:
                results.append(res)
                i += 1
                continue

            un_pro.append(items[i])  
            i += 1       

        return results, un_pro

    def _clean_item(self, item):
        """清理文本中的标点符号"""
        return item.strip(self.punctuation)

    def _get_description_value(self, item, desc_dict):
        """从文本中提取描述词对应的数值"""
        if '长' in item and '长度' not in item:
            return desc_dict['长']
        
        for desc in desc_dict:
            if desc == '长': 
                continue
            if desc in item:
                return desc_dict[desc]
        
        return None

    def _handle_general_case(self, item):
        """处理普通描述逻辑"""
        res = [0.5]*4
        if '最' in item:
            return res, False

        mentioned_parts = [part for part in self.body_parts if part in item]
        if not mentioned_parts:
            return res, False
        
        # 尝试直接描述词
        desc_value = self._get_description_value(item, self.direct_descriptions)
        if desc_value is None:
            return res, False
        
        # 仅更新未赋值的部位
        for part in mentioned_parts:
            col = self.body_parts[part]
            res[col] = desc_value
        return res, True

    def _handle_superlative(self, item):
        """处理最高级逻辑"""
        res = [0.5] * 4
        if '最' not in item:
            # 处理句式是“比其他”或“比其余”的情况
            if '比其他' in item or '比其余' in item:
                item = item.replace('比其他', '最').replace('比其余', '最')
            else:
                return res, False
        
        superlative_match = re.search(r'最(长|短)', item)
        if not superlative_match:
            return res, False
        
        # 只能包含一个身体部位
        mentioned_parts = [part for part in self.body_parts if part in item]
        if len(mentioned_parts) != 1:
            return res, False

        # 获取描述词和对应数值
        current_desc = superlative_match.group(1)
        superlative_value = self.max_descriptions.get(current_desc)
        if not superlative_value:
            return res, False
        
        # 获取相反描述的值
        opposite_desc = '短' if current_desc == '长' else '长'
        other_value = self.max_descriptions.get(opposite_desc)
        
        the_part = mentioned_parts[0]
        res[self.body_parts[the_part]] = superlative_value
        
        # 设置其他部位为相反值
        for p, col in self.body_parts.items():
            if p != the_part:
                res[col] = other_value

        return res, True

    def _handle_exclusion(self, current_item, next_item):
        """处理排除逻辑"""
        res = [0.5] * 4
        match = self.exclude_pattern.search(current_item)
        if not match:
            return res, False
        
        excluded_parts = [p.strip() for p in re.split('[和、及与]', match.group(1))]
        desc_value = self._get_description_value(next_item, self.direct_descriptions)
        if not desc_value:
            return res, False

        # 处理被排除的部位
        valid_excluded = []
        for part in excluded_parts:
            if part in self.body_parts:
                valid_excluded.append(part)
                res[self.body_parts[part]] = 1 - desc_value

        # 处理未被排除的部位
        for part, col in self.body_parts.items():
            if part not in valid_excluded:
                res[col] = desc_value
        return res, True

    def _handle_universal_quantifier(self, item):
        """处理全称量词"""
        res = [0.5] * 4
        if '都' not in item or not any(q in item for q in self.quantifiers):
            return res, False

        desc_value = self._get_description_value(item, self.direct_descriptions)
        if not desc_value:
            return res, False


        res = [desc_value] * 4

        return res, True

    def _handle_exclusive_case(self, item):
        """处理'只有'逻辑"""
        res = [0.5] * 4
        if '只有' not in item:
            return res, False

        mentioned_parts = [part for part in self.body_parts if part in item]
        if not mentioned_parts:
            return res, False

        desc_value = self._get_description_value(item, self.direct_descriptions)
        if not desc_value:
            return res, False

        opposite = 1 - desc_value
        for part in mentioned_parts:
            res[self.body_parts[part]] = desc_value
        for part, col in self.body_parts.items():
            if part not in mentioned_parts:
                res[col] = opposite

        return res, True

    def _handle_comparison(self, item):
        """处理比较逻辑"""
        res = [0.5] * 4
        if "比" not in item or "比较" in item:
            return res, False

        desc_value = self._get_description_value(item, self.comp_descriptions)
        if not desc_value:
            return res, False

        current_desc = next(
            (desc for desc, val in self.comp_descriptions.items() if val == desc_value),
            None
        )
        if not current_desc:
            return res, False

        opposite_desc = '短' if current_desc == '长' else '长'
        opposite_value = self.comp_descriptions.get(opposite_desc, 1 - desc_value)

        mentioned_parts = [part for part in self.body_parts if part in item]
        split_index = item.find('比')

        parts_before = [p for p in mentioned_parts if item.find(p) < split_index]
        parts_after = [p for p in mentioned_parts if item.find(p) > split_index]
        if len(parts_after) == 0 and ('比其他' in item or '比其余' in item):
            parts_after = list(set(self.body_parts.keys()) - set(parts_before))


        for part in parts_before:
            res[self.body_parts[part]] = desc_value
        for part in parts_after:
            res[self.body_parts[part]] = opposite_value

        return res, True