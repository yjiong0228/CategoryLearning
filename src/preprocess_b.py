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
            '脖子': 'neck_oral',
            '头': 'head_oral',
            '腿': 'leg_oral',
            '尾巴': 'tail_oral'
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

    def process(self, recording_raw, structure):
        # Initialize new columns
        recording_raw['invalid'] = 0
        recording_raw['noinfo'] = 0
        recording_raw['neck_oral'] = None
        recording_raw['head_oral'] = None
        recording_raw['leg_oral'] = None
        recording_raw['tail_oral'] = None
        
        # Apply the extraction function to each row
        extracted_data = recording_raw['text'].apply(self.extract_values)
        
        # Populate the new columns based on the extracted data
        recording_raw['invalid'] = extracted_data.apply(lambda x: x['invalid'])
        recording_raw['noinfo'] = extracted_data.apply(lambda x: x['noinfo'])
        recording_raw['neck_oral'] = extracted_data.apply(lambda x: x['neck_oral'])
        recording_raw['head_oral'] = extracted_data.apply(lambda x: x['head_oral'])
        recording_raw['leg_oral'] = extracted_data.apply(lambda x: x['leg_oral'])
        recording_raw['tail_oral'] = extracted_data.apply(lambda x: x['tail_oral'])
        
        preprocessor_b = Preprocessor_B()
        feature_names = preprocessor_b.convert("_oral", structure)
        for i, feature in enumerate(feature_names):
            new_name = f'feature{i+1}_oral'
            recording_raw[new_name] = recording_raw[feature]
        
        return recording_raw
    
    def extract_values(self, text):
        """主处理函数"""
        result = {
            'invalid': 0,
            'noinfo': 0,
            'neck_oral': None,
            'head_oral': None,
            'leg_oral': None,
            'tail_oral': None
        }

        if pd.isna(text) or not str(text).strip():
            result['invalid'] = 1
            return result

        # 分割并清理items
        raw_items = re.split(r'[，,]', str(text))
        items = [self._clean_item(i) for i in raw_items]

        # 处理排除逻辑
        i = 0
        while i < len(items):
            current_item = self._clean_item(items[i])
            if not current_item:
                i += 1
                continue

            i = self._handle_exclusion(current_item, items, i, result)
            i += 1

        # 处理最高级
        if self._handle_superlative(items, result):
            result['noinfo'] = 0
            return result

        # 处理其他逻辑
        for item in items:
            if not item:
                continue
            if self._handle_superlative(item, result):
                continue
            if self._handle_universal_quantifier(item, result):
                continue
            if self._handle_exclusive_case(item, result):
                continue
            self._handle_comparison(item, result)
            if self._handle_remaining_cases(item, result):
                break
            self._handle_general_case(item, result)

        # 最终校验
        body_values = [result[col] for col in self.body_parts.values()]
        result['noinfo'] = int(all(v is None for v in body_values))

        if result['noinfo'] == 0:
            for part in self.body_parts.values():
                if result[part] is None:
                    result[part] = 0.5

        return result

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

    def _handle_general_case(self, item, result):
        """处理普通描述逻辑"""
        if '最' in item:
            return False

        mentioned_parts = [part for part in self.body_parts if part in item]
        if not mentioned_parts:
            return False
        
        # 尝试直接描述词
        desc_value = self._get_description_value(item, self.direct_descriptions)
        if desc_value is None:
            return False
        
        # 仅更新未赋值的部位
        for part in mentioned_parts:
            col = self.body_parts[part]
            if result[col] is None:
                result[col] = desc_value
        return True

    def _handle_superlative(self, item, result):
        """处理最高级逻辑"""
        if '最' not in item:
            # 处理句式是“比其他”或“比其余”的情况
            if '比其他' in item or '比其余' in item:
                item = item.replace('比其他', '最').replace('比其余', '最')
            else:
                return False
        
        superlative_match = re.search(r'最(长|短)', item)
        if not superlative_match:
            return False
        
        # 只能包含一个身体部位
        mentioned_parts = [part for part in self.body_parts if part in item]
        if len(mentioned_parts) != 1:
            return False

        # 获取描述词和对应数值
        current_desc = superlative_match.group(1)
        superlative_value = self.max_descriptions.get(current_desc)
        if not superlative_value:
            return False
        
        # 获取相反描述的值
        opposite_desc = '短' if current_desc == '长' else '长'
        other_value = self.max_descriptions.get(opposite_desc)
        
        the_part = mentioned_parts[0]
        result[self.body_parts[the_part]] = superlative_value
        
        # 设置其他部位为相反值
        for p, col in self.body_parts.items():
            if p != the_part:
                result[col] = other_value
        
        return True

    def _handle_exclusion(self, current_item, items, i, result):
        """处理排除逻辑"""
        match = self.exclude_pattern.search(current_item)
        if not match:
            return i
        
        excluded_parts = [p.strip() for p in re.split('[和、及与]', match.group(1))]
        if i+1 >= len(items):
            return i

        next_item = self._clean_item(items[i+1])
        desc_value = self._get_description_value(next_item, self.direct_descriptions)
        if not desc_value:
            return i

        # 处理被排除的部位
        valid_excluded = []
        for part in excluded_parts:
            if part in self.body_parts:
                valid_excluded.append(part)
                result[self.body_parts[part]] = 1 - desc_value

        # 处理未被排除的部位
        for part, col in self.body_parts.items():
            if result[col] is None and part not in valid_excluded:
                result[col] = desc_value

        return i + 1  # 跳过已处理的下一条

    def _handle_universal_quantifier(self, item, result):
        """处理全称量词"""
        if '都' not in item or not any(q in item for q in self.quantifiers):
            return False

        desc_value = self._get_description_value(item, self.direct_descriptions)
        if not desc_value:
            return False

        for col in result:
            if col.endswith('_oral'):
                result[col] = desc_value
        return True

    def _handle_exclusive_case(self, item, result):
        """处理'只有'逻辑"""
        if '只有' not in item:
            return False

        mentioned_parts = [part for part in self.body_parts if part in item]
        if not mentioned_parts:
            return False

        desc_value = self._get_description_value(item, self.direct_descriptions)
        if not desc_value:
            return False

        opposite = 1 - desc_value
        for part in mentioned_parts:
            result[self.body_parts[part]] = desc_value
        for part, col in self.body_parts.items():
            if part not in mentioned_parts:
                result[col] = opposite
        return True

    def _handle_comparison(self, item, result):
        """处理比较逻辑"""
        if "比" not in item or "比较" in item:
            return

        desc_value = self._get_description_value(item, self.comp_descriptions)
        if not desc_value:
            return

        current_desc = next(
            (desc for desc, val in self.comp_descriptions.items() if val == desc_value),
            None
        )
        if not current_desc:
            return

        opposite_desc = '短' if current_desc == '长' else '长'
        opposite_value = self.comp_descriptions.get(opposite_desc, 1 - desc_value)

        mentioned_parts = [part for part in self.body_parts if part in item]
        split_index = item.find('比')

        parts_before = [p for p in mentioned_parts if item.find(p) < split_index]
        parts_after = [p for p in mentioned_parts if item.find(p) > split_index]

        for part in parts_before:
            result[self.body_parts[part]] = desc_value
        for part in parts_after:
            result[self.body_parts[part]] = opposite_value

    def _handle_remaining_cases(self, item, result):
        """处理'其他'逻辑"""
        if '比其他' in item or '比其余' in item:
            return False
        
        if not any(kw in item for kw in ['其他', '其余']):
            return False

        desc_value = self._get_description_value(item, self.direct_descriptions)
        if not desc_value:
            return False

        for col in result:
            if col.endswith('_oral') and result[col] is None:
                result[col] = desc_value
        return True