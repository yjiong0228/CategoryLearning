"""
对 Task2 行为和口头汇报数据进行预处理。
"""

from __future__ import annotations

import pandas as pd

from src.oral_coding import Recording_Processor_Center, Recording_Processor_Region, parts_from_feature_map


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
            oral_parts = parts_from_feature_map(feature_map)
            center_processor = Recording_Processor_Center(parts=oral_parts)
            center_df = center_processor.process(recording_data)
            center_coded = center_df[["iSession", "iTrial", "text", "all"]].copy()

            def reorder_center(values, order):
                if not isinstance(values, list):
                    return values
                return [None if pd.isna(values[i]) else float(values[i]) for i in order]

            center_coded["oral_center"] = center_coded["all"].apply(lambda values: reorder_center(values, feature_order))

            region_processor = Recording_Processor_Region(parts=oral_parts)
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
