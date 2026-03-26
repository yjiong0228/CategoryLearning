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

        # 这些是 recording_data 存在时会额外产生的列
        extra_columns = [
            'text', 'feature1_oraluse', 'feature2_oraluse', 'feature3_oraluse', 'feature4_oraluse',
            'feature1_oralvalue', 'feature2_oralvalue', 'feature3_oralvalue', 'feature4_oralvalue'
        ]

        if recording_data is not None:
            rec_processor = Recording_Processor()
            
            # process value
            result_df = rec_processor.process(recording_data)
            result_df[['neck_oralvalue', 'head_oralvalue', 'leg_oralvalue', 'tail_oralvalue']] = pd.DataFrame(result_df['all'].tolist(), index=result_df.index)

            recording_coded = result_df[['iSession', 'iTrial', 'text', 'neck_oralvalue', 'head_oralvalue', 'leg_oralvalue', 'tail_oralvalue']].copy()
            
            feature1_oralvalue_col = f"{feature1_name}_oralvalue"
            feature2_oralvalue_col = f"{feature2_name}_oralvalue"
            feature3_oralvalue_col = f"{feature3_name}_oralvalue"
            feature4_oralvalue_col = f"{feature4_name}_oralvalue"

            recording_coded['feature1_oralvalue'] = recording_coded[feature1_oralvalue_col]
            recording_coded['feature2_oralvalue'] = recording_coded[feature2_oralvalue_col]
            recording_coded['feature3_oralvalue'] = recording_coded[feature3_oralvalue_col]
            recording_coded['feature4_oralvalue'] = recording_coded[feature4_oralvalue_col]

            keep_cols = [
                'iSession', 'iTrial', 'text', 
                'feature1_oralvalue', 'feature2_oralvalue',
                'feature3_oralvalue', 'feature4_oralvalue'
            ]

            combined_data = pd.merge(
                combined_data, recording_coded[keep_cols], 
                on=['iSession', 'iTrial'], 
                how='left')
            
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

            keep_cols = [
                'iSession', 'iTrial',
                'feature1_oraluse', 'feature2_oraluse',
                'feature3_oraluse', 'feature4_oraluse'
            ]
            combined_data = pd.merge(
                combined_data, result_use[keep_cols],
                on=['iSession', 'iTrial'],
                how='left'
            )
        else:
            # recording_data 缺失时，补齐这些列，值为空
            for col in extra_columns:
                combined_data[col] = pd.NA
                
        final_columns = base_columns + [
            'text','feature1_oraluse', 'feature2_oraluse', 'feature3_oraluse', 'feature4_oraluse',
            'feature1_oralvalue', 'feature2_oralvalue', 'feature3_oralvalue', 'feature4_oralvalue'
        ]
        combined_data = combined_data[final_columns]                

        return combined_data


    def process_new(self, stimulus_data, behavior_data, recording_data = None):

        joint_data = pd.merge(behavior_data, stimulus_data, on=['iSession', 'stiID'], suffixes=('', '_y'))
        joint_data = joint_data.drop('category_y', axis=1)

        feature1_name = joint_data['feature1_name'][0]
        feature2_name = joint_data['feature2_name'][0]
        feature3_name = joint_data['feature3_name'][0]
        feature4_name = joint_data['feature4_name'][0]
        
        # 将[feature1, feature2, feature3, feature4]转换为对应的数字顺序
        feature_map = {'neck': 0, 'head': 1, 'leg': 2, 'tail': 3}
        feature_order = [
            feature_map[feature1_name],
            feature_map[feature2_name],
            feature_map[feature3_name],
            feature_map[feature4_name],
        ]

        base_columns = ['condition', 'feature1_name', 'feature2_name', 'feature3_name', 'feature4_name', 
                        'iSession', 'iBlock', 'iTrial', 'feature1', 'feature2', 'feature3', 'feature4', 
                        'category', 'choice', 'feedback', 'ambiguous', 'choRT']

        combined_data = joint_data[base_columns].copy()
        
        combined_data = combined_data.sort_values(by=['iSession', 'iBlock', 'iTrial'])

        # 这些是 recording_data 存在时会额外产生的列
        extra_columns = [
            'text', 'A', 'b', 'n_constraints', 'matched_rules', 'un_pro',
        ]

        if recording_data is not None:
            rec_processor = Recording_Processor_new()
            
            # process value
            recording_coded = rec_processor.process(recording_data)
            
            # 对A的每一行，按照feature_order重新排列每个向量的顺序
            def reorder_A(A, feature_order):
                if not isinstance(A, list):
                    return A
                return [[row[i] for i in feature_order] for row in A]

            recording_coded['A'] = recording_coded['A'].apply(lambda A: reorder_A(A, feature_order))

            combined_data = pd.merge(
                combined_data, recording_coded, 
                on=['iSession', 'iTrial'], 
                how='left')
        else:
            # recording_data 缺失时，补齐这些列，值为空
            for col in extra_columns:
                combined_data[col] = pd.NA
                
        final_columns = base_columns + [
            'text','A', 'b', 'n_constraints', 'matched_rules', 'un_pro'
        ]
        combined_data = combined_data[final_columns]                

        return combined_data


class Recording_Processor_new:
    """
    将口头汇报编码为四维空间中的一个区域 Ax > b
    维度顺序固定为:
        0: 脖子
        1: 头
        2: 腿
        3: 尾巴
    """

    def __init__(
        self,
        long_threshold=0.5,
        short_threshold=0.5,
        middle_lower=0.25,
        middle_upper=0.75,
        comparison_margin=0.0,
        use_average_in_addition=False,
    ):
        self.body_parts = {
            '脖子': 0,
            '头': 1,
            '腿': 2,
            '尾巴': 3,
        }

        self.quantifiers = {
            '四个', '所有', '每一个', '每个', '全部', '各', '各个', '总体', '整体'
        }

        self.punctuation = '。.？?！!、'
        self.exclude_pattern = re.compile(r'除(?:了)?([^，。]+?)外')

        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.middle_lower = float(middle_lower)
        self.middle_upper = float(middle_upper)
        self.comparison_margin = float(comparison_margin)
        self.use_average_in_addition = use_average_in_addition

    # =========================
    # 对外主函数
    # =========================
    def process(self, df):
        """
        输入:
            df: 至少包含 ['iSession', 'iTrial', 'text']
        输出:
            DataFrame，包含:
            - iSession
            - iTrial
            - text
            - A: list of list
            - b: list
            - n_constraints
            - matched_rules
            - un_pro
        """
        results = {
            'iSession': [],
            'iTrial': [],
            'text': [],
            'A': [],
            'b': [],
            'n_constraints': [],
            'matched_rules': [],
            'un_pro': [],
        }

        for _, row in df.iterrows():
            text = row['text'] if 'text' in row else None
            A, b, matched_rules, un_pro = self.extract_region(text)

            results['iSession'].append(row['iSession'])
            results['iTrial'].append(row['iTrial'])
            results['text'].append(text)
            results['A'].append(A)
            results['b'].append(b)
            results['n_constraints'].append(len(b))
            results['matched_rules'].append(matched_rules)
            results['un_pro'].append(un_pro)

        return pd.DataFrame(results)

    def extract_region(self, text):
        """
        将单条文本转为区域约束 Ax > b
        返回:
            A: list[list[float]]
            b: list[float]
            matched_rules: list[str]
            un_pro: list[str]
        """
        all_constraints = []
        matched_rules = []
        un_pro = []

        if pd.isna(text) or not str(text).strip():
            return [], [], [], []

        raw_items = re.split(r'[，,]', str(text))
        items = [self._clean_item(i) for i in raw_items]
        items = [i for i in items if i != '']

        i = 0
        while i < len(items):
            handled = False

            # 1) exclusion
            if ('除' in items[i]) or (
                i + 1 < len(items) and (
                    '其他' in items[i + 1] or
                    '其余' in items[i + 1] or
                    '另外' in items[i + 1]
                )
            ):
                if i + 1 < len(items):
                    cons = self._handle_exclusion(items[i], items[i + 1])
                    if cons is not None:
                        all_constraints.extend(cons)
                        matched_rules.append('exclusion')
                        i += 2
                        handled = True

            if handled:
                continue

            # 2) superlative
            cons = self._handle_superlative(items[i])
            if cons is not None:
                all_constraints.extend(cons)
                matched_rules.append('superlative')
                i += 1
                continue

            # 3) exclusive_case
            cons = self._handle_exclusive_case(items[i])
            if cons is not None:
                all_constraints.extend(cons)
                matched_rules.append('exclusive_case')
                i += 1
                continue

            # 4) addition
            cons = self._handle_addition(items[i])
            if cons is not None:
                all_constraints.extend(cons)
                matched_rules.append('addition')
                i += 1
                continue

            # 5) comparison
            cons = self._handle_comparison(items[i])
            if cons is not None:
                all_constraints.extend(cons)
                matched_rules.append('comparison')
                i += 1
                continue

            # 6) universal_quantifier
            cons = self._handle_universal_quantifier(items[i])
            if cons is not None:
                all_constraints.extend(cons)
                matched_rules.append('universal_quantifier')
                i += 1
                continue

            # 7) general_case
            cons = self._handle_general_case(items[i])
            if cons is not None:
                all_constraints.extend(cons)
                matched_rules.append('general_case')
                i += 1
                continue

            # 未识别
            un_pro.append(items[i])
            i += 1

        A, b = self._merge_constraints(all_constraints)
        return A, b, matched_rules, un_pro

    # =========================
    # 基础工具函数
    # =========================
    def _clean_item(self, item):
        return str(item).strip(self.punctuation).strip()

    def _parts_in_item(self, item):
        return [part for part in self.body_parts if part in item]

    def _get_desc_label(self, item, allow_middle=True):
        """
        返回:
            'long' / 'short' / 'middle' / None
        """
        if '长' in item and '长度' not in item:
            return 'long'
        if '短' in item:
            return 'short'
        if allow_middle and ('中等' in item or '适中' in item):
            return 'middle'
        return None

    def _constraint_gt(self, dim, thr):
        row = np.zeros(4, dtype=float)
        row[dim] = 1.0
        return row, float(thr)

    def _constraint_lt(self, dim, thr):
        row = np.zeros(4, dtype=float)
        row[dim] = -1.0
        return row, float(-thr)

    def _constraint_diff_gt(self, dim1, dim2, margin=0.0):
        row = np.zeros(4, dtype=float)
        row[dim1] = 1.0
        row[dim2] = -1.0
        return row, float(margin)

    def _constraints_for_desc(self, parts, desc_label):
        """
        将 [part1, part2, ...] + {long/short/middle}
        转成若干单维阈值约束
        """
        constraints = []

        for part in parts:
            dim = self.body_parts[part]

            if desc_label == 'long':
                constraints.append(self._constraint_gt(dim, self.long_threshold))

            elif desc_label == 'short':
                constraints.append(self._constraint_lt(dim, self.short_threshold))

            elif desc_label == 'middle':
                # middle_lower < x_i < middle_upper
                constraints.append(self._constraint_gt(dim, self.middle_lower))
                constraints.append(self._constraint_lt(dim, self.middle_upper))

        return constraints if len(constraints) > 0 else None

    def _merge_constraints(self, constraints):
        """
        把所有约束合并，并做简单去重
        """
        if not constraints:
            return [], []

        dedup = []
        seen = set()

        for row, rhs in constraints:
            key = (tuple(np.round(row.astype(float), 8)), round(float(rhs), 8))
            if key not in seen:
                seen.add(key)
                dedup.append((row.astype(float), float(rhs)))

        A = [row.tolist() for row, _ in dedup]
        b = [rhs for _, rhs in dedup]
        return A, b

    # =========================
    # 各类规则
    # =========================
    def _handle_general_case(self, item):
        """
        普通描述：
        - 头长 -> x_head > 0.5
        - 腿短 -> x_leg < 0.5
        - 脖子适中 -> 0.25 < x_neck < 0.75
        """
        if '一样' in item or '差不多' in item or '不是' in item:
            return None

        mentioned_parts = self._parts_in_item(item)
        if not mentioned_parts:
            return None

        desc_label = self._get_desc_label(item, allow_middle=True)
        if desc_label is None:
            return None

        return self._constraints_for_desc(mentioned_parts, desc_label)

    def _handle_universal_quantifier(self, item):
        """
        全称量词：
        - 四个都长
        - 所有都短
        """
        if '都' not in item or not any(q in item for q in self.quantifiers):
            return None

        desc_label = self._get_desc_label(item, allow_middle=True)
        if desc_label is None:
            return None

        parts = list(self.body_parts.keys())
        return self._constraints_for_desc(parts, desc_label)

    def _handle_exclusive_case(self, item):
        """
        只有逻辑：
        - 只有头长  -> 头>0.5, 其他<0.5
        - 只有腿短  -> 腿<0.5, 其他>0.5
        """
        if '只有' not in item:
            return None

        mentioned_parts = self._parts_in_item(item)
        if not mentioned_parts:
            return None

        desc_label = self._get_desc_label(item, allow_middle=False)
        if desc_label is None:
            return None

        constraints = []
        constraints.extend(self._constraints_for_desc(mentioned_parts, desc_label))

        if desc_label == 'long':
            opposite_label = 'short'
        elif desc_label == 'short':
            opposite_label = 'long'
        else:
            return None

        other_parts = [p for p in self.body_parts if p not in mentioned_parts]
        constraints.extend(self._constraints_for_desc(other_parts, opposite_label))
        return constraints

    def _handle_exclusion(self, current_item, next_item):
        """
        排除逻辑：
        - 除了尾巴, 其他都长
          -> 尾巴<0.5, 其余>0.5
        """
        excluded_parts = self._parts_in_item(current_item)
        if not excluded_parts:
            return None

        desc_label = self._get_desc_label(next_item, allow_middle=False)
        if desc_label is None:
            return None

        constraints = []

        # 未被排除者
        remain_parts = [p for p in self.body_parts if p not in excluded_parts]
        constraints.extend(self._constraints_for_desc(remain_parts, desc_label))

        # 被排除者取相反方向
        if desc_label == 'long':
            opposite_label = 'short'
        elif desc_label == 'short':
            opposite_label = 'long'
        else:
            return None

        constraints.extend(self._constraints_for_desc(excluded_parts, opposite_label))
        return constraints

    def _handle_comparison(self, item):
        """
        比较逻辑：
        - 头比腿长 -> x_head - x_leg > 0
        - 头比腿短 -> x_leg - x_head > 0
        - 头比其他长 -> x_head - x_j > 0 for all j != head
        """
        if '比' not in item or '比较' in item:
            return None
        if '比躯干' in item:
            return None

        desc_label = self._get_desc_label(item, allow_middle=False)
        if desc_label is None:
            return None

        mentioned_parts = self._parts_in_item(item)
        split_index = item.find('比')

        parts_before = [p for p in mentioned_parts if item.find(p) < split_index]
        parts_after = [p for p in mentioned_parts if item.find(p) > split_index]

        if len(parts_before) == 0:
            return None

        if len(parts_after) == 0 and ('比其他' in item or '比其余' in item):
            parts_after = [p for p in self.body_parts if p not in parts_before]

        if len(parts_after) == 0:
            return None

        constraints = []

        for p1 in parts_before:
            d1 = self.body_parts[p1]
            for p2 in parts_after:
                d2 = self.body_parts[p2]
                if desc_label == 'long':
                    constraints.append(self._constraint_diff_gt(d1, d2, self.comparison_margin))
                elif desc_label == 'short':
                    constraints.append(self._constraint_diff_gt(d2, d1, self.comparison_margin))

        return constraints if len(constraints) > 0 else None

    def _handle_superlative(self, item):
        """
        最高级：
        - 头最长 -> 头比其他每一维都大
        - 腿最短 -> 其他每一维都比腿大
        """
        if '不是' in item and '最' in item:
            return None

        normalized_item = item
        if '最' not in normalized_item:
            if '比其他' in normalized_item or '比其余' in normalized_item:
                normalized_item = normalized_item.replace('比其他', '最').replace('比其余', '最')
            else:
                return None

        superlative_match = re.search(r'最.{0,2}(长|短)', normalized_item)
        if not superlative_match:
            return None

        mentioned_parts = self._parts_in_item(normalized_item)
        if len(mentioned_parts) != 1:
            return None

        target_part = mentioned_parts[0]
        target_dim = self.body_parts[target_part]
        current_desc = superlative_match.group(1)

        constraints = []
        for p, dim in self.body_parts.items():
            if p == target_part:
                continue

            if current_desc == '长':
                constraints.append(self._constraint_diff_gt(target_dim, dim, self.comparison_margin))
            elif current_desc == '短':
                constraints.append(self._constraint_diff_gt(dim, target_dim, self.comparison_margin))

        return constraints if len(constraints) > 0 else None

    def _handle_addition(self, item):
        """
        相加逻辑：
        - 头加尾巴比腿长
          默认编码为:
              x_head + x_tail - x_leg > 0
        - 若 use_average_in_addition=True，则编码为均值比较:
              (x_head + x_tail)/2 - x_leg > 0
        """
        if '加' not in item:
            return None

        operator = None
        if '比' in item:
            operator = '比'
        elif '大于' in item:
            operator = '大于'
        elif '小于' in item:
            operator = '小于'
        else:
            return None

        items = item.split(operator)
        if len(items) != 2:
            return None

        front_item, back_item = items[0], items[1]
        front_parts = self._parts_in_item(front_item)
        back_parts = self._parts_in_item(back_item)

        if len(front_parts) == 0 or len(back_parts) == 0:
            return None

        row = np.zeros(4, dtype=float)

        if self.use_average_in_addition:
            front_coef = 1.0 / len(front_parts)
            back_coef = 1.0 / len(back_parts)
        else:
            front_coef = 1.0
            back_coef = 1.0

        for part in front_parts:
            row[self.body_parts[part]] += front_coef
        for part in back_parts:
            row[self.body_parts[part]] -= back_coef

        # “小于”时翻转方向
        if operator == '小于':
            row = -row

        return [(row, 0.0)]




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
            'neck_oraluse': [0] * len(df),
            'head_oraluse': [0] * len(df),
            'leg_oraluse': [0] * len(df),
            'tail_oraluse': [0] * len(df),
        }

        # 遍历每一行的文本
        for idx, text in enumerate(df['text']):
            if pd.isna(text) or not str(text).strip():
                continue  # 跳过空文本

            # 检查每个身体部位是否出现在文本中
            if '脖子' in text:
                results['neck_oraluse'][idx] = 1
            if '头' in text:
                results['head_oraluse'][idx] = 1
            if '腿' in text:
                results['leg_oraluse'][idx] = 1
            if '尾巴' in text:
                results['tail_oraluse'][idx] = 1

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
