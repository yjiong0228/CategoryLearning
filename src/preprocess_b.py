"""
对 Task2, Task3a, Task3b 的数据进行预处理。
"""

from __future__ import annotations

import pandas as pd

from src.oral_coding import CenterEncoder, PARTS, PART_TO_DIM, RegionEncoder, SemanticParser


class Preprocessor_B:
    def process(self, task_type, feature_map, stimulus_data, behavior_data, recording_data=None):
        if "iSession" in stimulus_data.columns:
            joint_data = pd.merge(behavior_data, stimulus_data, on=["iSession", "stiID"], suffixes=("", "_y"))
        else:
            joint_data = pd.merge(behavior_data, stimulus_data, on=["stiID"], suffixes=("", "_y"))
        if "category_y" in joint_data.columns:
            joint_data = joint_data.drop("category_y", axis=1)

        feature1_name = joint_data["feature1_name"][0]
        feature2_name = joint_data["feature2_name"][0]
        feature3_name = joint_data["feature3_name"][0]
        feature4_name = joint_data["feature4_name"][0]

        feature_order = [
            feature_map[feature1_name],
            feature_map[feature2_name],
            feature_map[feature3_name],
            feature_map[feature4_name],
        ]

        if task_type == 1:
            category_columns = ["category"]
        elif task_type == 2:
            category_columns = ["probCat1", "probCat2"]
        else:
            category_columns = []

        block_column = "iBlock" if "iBlock" in joint_data.columns else "iRun"
        base_columns = [
            "condition",
            "feature1_name",
            "feature2_name",
            "feature3_name",
            "feature4_name",
            "iSession",
            block_column,
            "iTrial",
            "feature1",
            "feature2",
            "feature3",
            "feature4",
            *category_columns,
            "choice",
            "feedback",
            "ambiguous",
            "choRT",
        ]
        if "rating" in joint_data.columns:
            base_columns.insert(base_columns.index("choice") + 1, "rating")

        combined_data = joint_data[base_columns].copy()
        combined_data = combined_data.sort_values(by=["iSession", "iTrial"])

        extra_columns = ["text", "oral_center", "oral_A", "oral_b"]
        if recording_data is not None:
            center_processor = Recording_Processor_Center()
            center_df = center_processor.process(recording_data)
            center_coded = center_df[["iSession", "iTrial", "text", "all"]].copy()

            def reorder_center(values, order):
                if not isinstance(values, list):
                    return values
                return [None if pd.isna(values[i]) else float(values[i]) for i in order]

            center_coded["oral_center"] = center_coded["all"].apply(lambda values: reorder_center(values, feature_order))

            region_processor = Recording_Processor_Region()
            region_coded = region_processor.process(recording_data)

            def reorder_A(A, order):
                if not isinstance(A, list):
                    return A
                return [[None if pd.isna(row[i]) else float(row[i]) for i in order] for row in A]

            def clean_vector(values):
                if not isinstance(values, list):
                    return values
                return [None if pd.isna(value) else float(value) for value in values]

            region_coded["oral_A"] = region_coded["A"].apply(lambda A: reorder_A(A, feature_order))
            region_coded["oral_b"] = region_coded["b"].apply(clean_vector)

            recording_coded = pd.merge(
                center_coded[["iSession", "iTrial", "text", "oral_center"]],
                region_coded[["iSession", "iTrial", "oral_A", "oral_b"]],
                on=["iSession", "iTrial"],
                how="left",
            )
            combined_data = pd.merge(combined_data, recording_coded, on=["iSession", "iTrial"], how="left")
        else:
            for col in extra_columns:
                combined_data[col] = pd.NA

        final_columns = base_columns + extra_columns
        return combined_data[final_columns]


class Recording_Processor_Region:
    """
    将口头汇报编码为四维空间中的区域约束 Ax > b。
    维度顺序固定为: 脖子, 头, 腿, 尾巴。
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
        self.body_parts = dict(PART_TO_DIM)
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.middle_lower = float(middle_lower)
        self.middle_upper = float(middle_upper)
        self.comparison_margin = float(comparison_margin)
        self.use_average_in_addition = use_average_in_addition
        self.semantic_parser = SemanticParser(body_value=0.5)
        self.region_encoder = RegionEncoder(
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
            middle_lower=self.middle_lower,
            middle_upper=self.middle_upper,
            comparison_margin=self.comparison_margin,
            equality_epsilon=0.10,
        )

    def process(self, df):
        results = {
            "iSession": [],
            "iTrial": [],
            "text": [],
            "A": [],
            "b": [],
            "n_constraints": [],
            "matched_rules": [],
            "un_pro": [],
        }
        for _, row in df.iterrows():
            text = row["text"] if "text" in row else None
            A, b, matched_rules, un_pro = self.extract_region(text)
            results["iSession"].append(row["iSession"])
            results["iTrial"].append(row["iTrial"])
            results["text"].append(text)
            results["A"].append(A)
            results["b"].append(b)
            results["n_constraints"].append(len(b))
            results["matched_rules"].append(matched_rules)
            results["un_pro"].append(un_pro)
        return pd.DataFrame(results)

    def extract_region(self, text):
        parsed = self.semantic_parser.parse(text)
        return self.region_encoder.encode(parsed)


class Recording_Processor_Center:
    """将口头汇报编码为四维 center。维度顺序固定为: 脖子, 头, 腿, 尾巴。"""

    def __init__(self):
        self.body_parts = dict(PART_TO_DIM)
        self.semantic_parser = SemanticParser(body_value=0.5)
        self.center_encoder = CenterEncoder(body_value=0.5)

    def process_use(self, df):
        results = {
            "iSession": df["iSession"],
            "iTrial": df["iTrial"],
            "neck_oraluse": [0] * len(df),
            "head_oraluse": [0] * len(df),
            "leg_oraluse": [0] * len(df),
            "tail_oraluse": [0] * len(df),
        }
        use_columns = {
            "脖子": "neck_oraluse",
            "头": "head_oraluse",
            "腿": "leg_oraluse",
            "尾巴": "tail_oraluse",
        }
        for idx, text in enumerate(df["text"]):
            if pd.isna(text) or not str(text).strip():
                continue
            text = str(text)
            for part, col in use_columns.items():
                if part in text:
                    results[col][idx] = 1
        return pd.DataFrame(results)

    def process(self, df):
        rule_columns = list(self.center_encoder.rule_names) + ["all"]
        results = {col: [] for col in rule_columns}
        results.update({"un_pro": [], "text": []})

        for text in df["text"]:
            encoded, un_pro = self.extract_values(text)
            for col in rule_columns:
                results[col].append(encoded.get(col, [None] * len(PARTS)))
            results["un_pro"].append(un_pro)
            results["text"].append(text)

        result_df = pd.DataFrame(results)
        result_df["iSession"] = df["iSession"].to_numpy()
        result_df["iTrial"] = df["iTrial"].to_numpy()
        columns = ["iSession", "iTrial"] + [col for col in result_df.columns if col not in ["iSession", "iTrial"]]
        return result_df[columns]

    def extract_values(self, text):
        parsed = self.semantic_parser.parse(text)
        encoded, matched_rules, un_pro = self.center_encoder.encode(parsed)
        return encoded, un_pro


# Backward-compatible names; both now use the shared oral_coding implementation.
Recording_Processor_new = Recording_Processor_Region
Recording_Processor = Recording_Processor_Center
