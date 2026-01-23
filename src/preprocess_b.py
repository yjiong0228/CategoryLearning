"""
对Task2, Task3a, Task3b的数据进行预处理
"""

import re
import pandas as pd
import numpy as np

class Preprocessor_B:
    def process(self, stimulus_data, behavior_data, recording_data = None):

        joint_data = pd.merge(behavior_data, stimulus_data, on=['iSession', 'stiID'], suffixes=('', '_y'))
        joint_data = joint_data.drop('category_y', axis=1)

        feature1_name = joint_data['feature1_name'][0]
        feature2_name = joint_data['feature2_name'][0]
        feature3_name = joint_data['feature3_name'][0]
        feature4_name = joint_data['feature4_name'][0]

        base_columns = ['condition', 'feature1_name', 'feature2_name', 'feature3_name', 'feature4_name', 
                        'iSession', 'iBlock', 'iTrial', 'feature1', 'feature2', 'feature3', 'feature4', 
                        'category', 'choice', 'feedback', 'ambiguous', 'choRT']

        combined_data = joint_data[base_columns].copy()
        
        combined_data = combined_data.sort_values(by=['iSession', 'iBlock', 'iTrial'])

        if recording_data is not None:
            rec_processor = Recording_Processor()
            
            # process
            result_df = rec_processor.process(recording_data)
            result_df[['neck_oral', 'head_oral', 'leg_oral', 'tail_oral']] = pd.DataFrame(result_df['all'].tolist(), index=result_df.index)

            recording_coded = result_df[['iSession', 'iTrial', 'text', 'neck_oral', 'head_oral', 'leg_oral', 'tail_oral']].copy()
            
            feature1_oral_col = f"{feature1_name}_oral"
            feature2_oral_col = f"{feature2_name}_oral"
            feature3_oral_col = f"{feature3_name}_oral"
            feature4_oral_col = f"{feature4_name}_oral"

            recording_coded['feature1_oral'] = recording_coded[feature1_oral_col]
            recording_coded['feature2_oral'] = recording_coded[feature2_oral_col]
            recording_coded['feature3_oral'] = recording_coded[feature3_oral_col]
            recording_coded['feature4_oral'] = recording_coded[feature4_oral_col]

            combined_data = pd.merge(combined_data, recording_coded, on=['iSession', 'iTrial'])
            
            # process_use
            result_use = rec_processor.process_use(recording_data).copy()
            
            feature1_oraluse_col = f"{feature1_name}_oraluse"
            feature2_oraluse_col = f"{feature2_name}_oraluse"
            feature3_oraluse_col = f"{feature3_name}_oraluse"
            feature4_oraluse_col = f"{feature4_name}_oraluse"

            result_use['feature1_oraluse'] = result_use[feature1_oraluse_col]
            result_use['feature2_oraluse'] = result_use[feature2_oraluse_col]
            result_use['feature3_oraluse'] = result_use[feature3_oraluse_col]
            result_use['feature4_oraluse'] = result_use[feature4_oraluse_col]

            combined_data = pd.merge(combined_data, result_use, on=['iSession', 'iTrial'])

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

        self.direct_descriptions = {'长': 0.75, '短': 0.25, '中等': 0.5, '适中': 0.5}

        self.comp_descriptions = {'长': 2 / 3, '短': 1 / 3}

        self.max_descriptions = {'长': 0.8, '短': 0.4}
        self.addition_descriptions = {
            '长': 2 / 3,
            '短': 1 / 3,
            '大于': 2 / 3,
            '小于': 1 / 3
        }

        self.quantifiers = {
            '四个', '所有', '每一个', '每个', '全部', '各', '各个', '总体', '整体'
        }
        self.exclude_pattern = re.compile(r'除(?:了)?([^，。]+?)外')
        self.punctuation = '。.？?！!、'

    def process_use(self, df):
        """
        根据 text 列生成一个新的 DataFrame，包含 iSession、iTrial 和四个新的列：
        neck_use、head_use、leg_use、tail_use。
        """
        # 初始化结果字典
        results = {
            'iSession': df['iSession'],
            'iTrial': df['iTrial'],
            'neck_use': [0] * len(df),
            'head_use': [0] * len(df),
            'leg_use': [0] * len(df),
            'tail_use': [0] * len(df),
        }

        # 遍历每一行的文本
        for idx, text in enumerate(df['text']):
            if pd.isna(text) or not str(text).strip():
                continue  # 跳过空文本

            # 检查每个身体部位是否出现在文本中
            if '脖子' in text:
                results['neck_use'][idx] = 1
            if '头' in text:
                results['head_use'][idx] = 1
            if '腿' in text:
                results['leg_use'][idx] = 1
            if '尾巴' in text:
                results['tail_use'][idx] = 1

        # 转换为 DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def process(self, df):
        texts = df['text']
        results = {
            'exclusion': [],
            'superlative': [],
            'universal_quantifier': [],
            'exclusive_case': [],
            'comparison': [],
            'general_case': [],
            'addition': [],
            'all': [],
            'un_pro': [],
            'text': [],
        }
        for text in texts:
            res, un_pro = self.extract_values(text)
            for key in res:
                results[key].append(res[key])
            results['un_pro'].append(un_pro)
            results['text'].append(text)

        result_df = pd.DataFrame(results)
        # Retain iSession and iTrial columns
        result_df['iSession'] = df['iSession']
        result_df['iTrial'] = df['iTrial']

        # Move iSession and iTrial to the front
        columns = ['iSession', 'iTrial'] + [
            col
            for col in result_df.columns if col not in ['iSession', 'iTrial']
        ]
        result_df = result_df[columns]

        return result_df

    def extract_values(self, text):
        """主处理函数"""
        results = {
            'exclusion': [None] * 4,
            'superlative': [None] * 4,
            'universal_quantifier': [None] * 4,
            'exclusive_case': [None] * 4,
            'comparison': [None] * 4,
            'general_case': [None] * 4,
            'addition': [None] * 4,
            'all': [None] * 4,
        }
        un_pro = []

        def merge(*vals):
            """合并多个值"""
            new_val = [None] * 4
            for i in range(4):
                merged_val = [val[i] for val in vals if val[i] is not None]
                if merged_val:
                    new_val[i] = np.mean(merged_val)
            return new_val

        if pd.isna(text) or not str(text).strip():
            return results, ''

        # 分割并清理items
        raw_items = re.split(r'[，,]', str(text))
        items = [self._clean_item(i) for i in raw_items]

        items = [i for i in items if i != '']  # 去除空值

        i = 0
        while i < len(items):
            # 处理排除逻辑
            if '除' in items[i] or (i + 1 < len(items) and
                                   ('其他' in items[i + 1] or '其余'
                                    in items[i + 1] or '另外' in items[i + 1])):
                res, is_pro = self._handle_exclusion(items[i], items[i + 1])
                if is_pro:
                    results['exclusion'] = merge(results['exclusion'], res)
                    i += 2
                    continue

            # 处理最高级
            res, is_pro = self._handle_superlative(items[i])
            if is_pro:
                results['superlative'] = merge(results['superlative'], res)
                i += 1
                continue

            # 处理'只有'逻辑
            res, is_pro = self._handle_exclusive_case(items[i])
            if is_pro:
                results['exclusive_case'] = merge(results['exclusive_case'],
                                                  res)
                i += 1
                continue

            # 处理相加逻辑
            res, is_pro = self._handle_addition(items[i])
            if is_pro:
                results['addition'] = merge(results['addition'], res)
                i += 1
                continue

            # 处理比较逻辑
            res, is_pro = self._handle_comparison(items[i])
            if is_pro:
                results['comparison'] = merge(results['comparison'], res)
                i += 1
                continue

            # 处理全称量词
            res, is_pro = self._handle_universal_quantifier(items[i])
            if is_pro:
                results['universal_quantifier'] = merge(
                    results['universal_quantifier'], res)
                i += 1
                continue

            # 处理普通描述逻辑
            res, is_pro = self._handle_general_case(items[i])
            if is_pro:
                results['general_case'] = merge(results['general_case'], res)
                i += 1
                continue

            un_pro.append(items[i])
            i += 1

        results['all'] = merge(
            results['exclusion'],
            results['superlative'],
            results['universal_quantifier'],
            results['exclusive_case'],
            results['comparison'],
            results['general_case'],
            results['addition'],
        )

        if all(v is None for v in results['all']):
            pass
        else:
            for i in range(4):
                if results['all'][i] is None:
                    results['all'][i] = 0.5

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
        res = [None] * 4
        if '一样' in item or '差不多' in item or '不是' in item:
            return res, False

        mentioned_parts = [part for part in self.body_parts if part in item]
        if not mentioned_parts:
            return res, False

        # 尝试直接描述词
        desc_value = self._get_description_value(item,
                                                 self.direct_descriptions)
        if desc_value is None:
            return res, False

        # 仅更新未赋值的部位
        for part in mentioned_parts:
            col = self.body_parts[part]
            res[col] = desc_value
        return res, True

    def _handle_superlative(self, item):
        """处理最高级逻辑"""
        res = [None] * 4
        if '不是' in item and '最' in item:
            return res, False

        if '最' not in item:
            # 处理句式是“比其他”或“比其余”的情况
            if '比其他' in item or '比其余' in item:
                item = item.replace('比其他', '最').replace('比其余', '最')
            else:
                return res, False

        superlative_match = re.search(r'最.{0,2}(长|短)', item)
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
        res = [None] * 4

        excluded_parts = [
            part for part in self.body_parts if part in current_item
        ]
        desc_value = self._get_description_value(next_item,
                                                 self.direct_descriptions)
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
        res = [None] * 4
        if '都' not in item or not any(q in item for q in self.quantifiers):
            return res, False

        desc_value = self._get_description_value(item,
                                                 self.direct_descriptions)
        if not desc_value:
            return res, False

        res = [desc_value] * 4

        return res, True

    def _handle_exclusive_case(self, item):
        """处理'只有'逻辑"""
        res = [None] * 4
        if '只有' not in item:
            return res, False

        mentioned_parts = [part for part in self.body_parts if part in item]
        if not mentioned_parts:
            return res, False

        desc_value = self._get_description_value(item,
                                                 self.direct_descriptions)
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
        res = [None] * 4
        if "比" not in item or "比较" in item:
            return res, False
        if '比躯干' in item:
            return res, False

        desc_value = self._get_description_value(item, self.comp_descriptions)
        if not desc_value:
            return res, False

        current_desc = next((desc
                             for desc, val in self.comp_descriptions.items()
                             if val == desc_value), None)
        if not current_desc:
            return res, False

        opposite_desc = '短' if current_desc == '长' else '长'
        opposite_value = self.comp_descriptions.get(opposite_desc,
                                                    1 - desc_value)

        mentioned_parts = [part for part in self.body_parts if part in item]
        split_index = item.find('比')

        parts_before = [
            p for p in mentioned_parts if item.find(p) < split_index
        ]
        parts_after = [
            p for p in mentioned_parts if item.find(p) > split_index
        ]
        if len(parts_after) == 0 and ('比其他' in item or '比其余' in item):
            parts_after = list(set(self.body_parts.keys()) - set(parts_before))

        for part in parts_before:
            res[self.body_parts[part]] = desc_value
        for part in parts_after:
            res[self.body_parts[part]] = opposite_value

        return res, True

    def _handle_addition(self, item):
        """处理相加逻辑"""
        res = [None] * 4
        if '加' not in item:
            return res, False

        if '比' in item:
            items = item.split('比')
        elif '大于' in item:
            items = item.split('大于')
        elif '小于' in item:
            items = item.split('小于')
        else:
            return res, False

        front_item, back_item = items[0], items[1]
        front_parts = [part for part in self.body_parts if part in front_item]
        back_parts = [part for part in self.body_parts if part in back_item]

        desc_value = self._get_description_value(item,
                                                 self.addition_descriptions)

        for part in front_parts:
            res[self.body_parts[part]] = desc_value
        for part in back_parts:
            res[self.body_parts[part]] = 1 - desc_value
        return res, True
