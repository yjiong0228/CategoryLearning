from typing import Optional, List, Dict, Union, Tuple, Any
import numpy as np
import pandas as pd
from copy import deepcopy
import json

from ..utils.load_config import MODEL_STRUCT 
from ..problems.model import StateModel

# default model choice for oral hypothesis overlap analysis
# prefer a model without perception module for portability in preprocessing-only flows
def _select_default_model_without_perception():
    preferred_order = ["m_model", "pm_model", "base_model", "default_model", "pmh_model"]
    for name in preferred_order:
        if name in MODEL_STRUCT:
            modules = MODEL_STRUCT[name].get("modules", {})
            if "perception_mod" not in modules:
                return name
    for name, cfg in MODEL_STRUCT.items():
        if "perception_mod" not in cfg.get("modules", {}):
            return name
    return None

model_choice = _select_default_model_without_perception()
if model_choice is None:
    model_choice = "pmh_model" if "pmh_model" in MODEL_STRUCT else next(iter(MODEL_STRUCT.keys()), None)

engine_config = deepcopy(MODEL_STRUCT.get(model_choice))
if engine_config is None:
    raise ValueError("MODEL_STRUCT is empty; no engine_config available for StateModel.")

from ..problems.partitions import Partition


class Oral_hit_analysis:

    def _parse_region(self, region):
        """
        将 region 统一解析为 (A, b)
        支持:
        1. tuple/list: (A, b)
        2. dict: {'A': A, 'b': b}
        3. object array / length-2 structure
        """
        if region is None:
            return None, None

        if isinstance(region, dict):
            A = np.asarray(region['A'], dtype=float)
            b = np.asarray(region['b'], dtype=float)
            return A, b

        if isinstance(region, str):
            # 处理字符串格式的区域（例如 CSV/JSON 中的存储）
            try:
                region = json.loads(region)
            except json.JSONDecodeError:
                raise ValueError(f"Unsupported region string format: {region}")

        if isinstance(region, (tuple, list)) and len(region) == 2:
            A = np.asarray(region[0], dtype=float)
            b = np.asarray(region[1], dtype=float)
        else:
            # 兼容 numpy object / 其他 length=2 结构
            try:
                if len(region) == 2:
                    A = np.asarray(region[0], dtype=float)
                    b = np.asarray(region[1], dtype=float)
                else:
                    raise ValueError
            except Exception:
                raise ValueError(f"Unsupported region format: {type(region)}")

        if np.isnan(A).any() or np.isnan(b).any():
            return None, None

        if A.ndim == 0 or b.ndim == 0:
            # A/b should represent inequality constraints (A: [m,d], b: [m]); scalar invalid
            return None, None

        if A.ndim == 1:
            # single inequality vector (d,) -> reshape to (1,d)
            A = np.atleast_2d(A)

        if b.ndim == 1 and A.shape[0] == 1 and b.size == 1:
            b = np.atleast_1d(b)

        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
            # fallback for malformed region
            return None, None

        return A, b


    def _points_in_region(
        self,
        points: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        dist_tol: float = 1e-9,
    ) -> np.ndarray:
        """
        判断点集是否落在 Ax > b 中
        points: (N, d)
        A: (m, d)
        b: (m,)
        return: mask (N,)
        """
        if A is None or b is None:
            return np.zeros(points.shape[0], dtype=bool)

        if A.size == 0:
            # 空约束视为整个空间都满足
            return np.ones(points.shape[0], dtype=bool)

        lhs = points @ A.T  # (N, m)
        return np.all(lhs > (b - dist_tol), axis=1)

    def _estimate_region_overlap(
        self,
        region1,
        region2,
        n_samples: int = 50000,
        bounds: Tuple[float, float] = (0.0, 1.0),
        random_state: Optional[int] = 42,
        dist_tol: float = 1e-9,
    ) -> Dict[str, float]:
        """
        Monte Carlo 估计两个区域在 [low, high]^4 中的体积与重叠
        返回:
            {
                'vol1': ...,
                'vol2': ...,
                'intersection': ...,
                'union': ...,
                'iou': ...,
                'precision_like': ...,
                'recall_like': ...
            }
        """
        A1, b1 = self._parse_region(region1)
        A2, b2 = self._parse_region(region2)

        if A1 is None or A2 is None:
            return {
                'vol1': np.nan,
                'vol2': np.nan,
                'intersection': np.nan,
                'union': np.nan,
                'iou': np.nan,
                'precision_like': np.nan,
                'recall_like': np.nan,
            }

        rng = np.random.default_rng(random_state)

        d = A1.shape[1] if A1.size > 0 else A2.shape[1]
        low, high = bounds
        points = rng.uniform(low, high, size=(n_samples, d))

        in_r1 = self._points_in_region(points, A1, b1, dist_tol=dist_tol)
        in_r2 = self._points_in_region(points, A2, b2, dist_tol=dist_tol)

        box_volume = (high - low) ** d

        p1 = np.mean(in_r1)
        p2 = np.mean(in_r2)
        pint = np.mean(in_r1 & in_r2)
        punion = np.mean(in_r1 | in_r2)

        vol1 = p1 * box_volume
        vol2 = p2 * box_volume
        intersection = pint * box_volume
        union = punion * box_volume

        iou = intersection / union if union > 0 else 0.0
        precision_like = intersection / vol1 if vol1 > 0 else 0.0
        recall_like = intersection / vol2 if vol2 > 0 else 0.0

        return {
            'vol1': vol1,
            'vol2': vol2,
            'intersection': intersection,
            'union': union,
            'iou': iou,
            'precision_like': precision_like,
            'recall_like': recall_like,
        }

    def get_oral_hypos_list(
        self,
        condition: int,
        oral_region,
        choices: np.ndarray,
        model,
        dist_tol: float = 1e-9,
        top_k: Optional[int] = None,
        n_samples: int = 50000,
        bounds: Tuple[float, float] = (0.0, 1.0),
        random_state: Optional[int] = 42,
        overlap_metric: str = 'iou',
    ) -> List[Dict[str, Any]]:
        """
        对每个 trial，计算 reported_region 与每个 hypothesis 对应 true_region 的重叠程度，
        并返回按重叠度排序的 hypo 列表。

        参数
        ----
        oral_region:
            可以是 list/array，每个元素是一个 region，格式为 (A,b) 或 {'A':A,'b':b}
        choices:
            被试在每个 trial 的 choice，1-based category index
        overlap_metric:
            'iou' / 'intersection' / 'precision_like' / 'recall_like'

        返回
        ----
        oral_hypos_list: list of dict
            每个 trial 一个 dict，包含该 trial 下所有 hypothesis 的重叠排序结果
        """
        n_trials = len(choices)
        n_hypos = len(model.partition_model.regions)
        all_hypos = range(n_hypos)

        oral_hypos_list = []

        for trial_idx in range(n_trials):
            reported_region = oral_region[trial_idx]
            cat_idx = int(choices[trial_idx]) - 1

            overlap_map = []

            for hypo_idx in all_hypos:
                regions = model.partition_model.regions
                if hasattr(regions, 'shape') and isinstance(regions, np.ndarray):
                    true_region = regions[hypo_idx, 1, cat_idx, :]
                elif isinstance(regions, (list, tuple)):
                    if hypo_idx >= len(regions):
                        raise IndexError(f"Hypothesis index {hypo_idx} out of range for regions length {len(regions)}")
                    hypo_regions = regions[hypo_idx]
                    if not (isinstance(hypo_regions, (list, tuple)) and len(hypo_regions) > cat_idx):
                        raise IndexError(f"Category index {cat_idx} out of range for hypo_regions length {len(hypo_regions)}")
                    true_region = hypo_regions[cat_idx]
                else:
                    raise TypeError(f"Unsupported partition_model.regions type: {type(regions)}")

                overlap_stats = self._estimate_region_overlap(
                    reported_region,
                    true_region,
                    n_samples=n_samples,
                    bounds=bounds,
                    random_state=None if random_state is None else random_state + trial_idx * 100000 + hypo_idx,
                    dist_tol=dist_tol,
                )

                if overlap_metric not in overlap_stats:
                    raise ValueError(
                        f"Unsupported overlap_metric={overlap_metric}. "
                        f"Choose from {list(overlap_stats.keys())}."
                    )

                overlap_score = overlap_stats[overlap_metric]

                overlap_map.append({
                    'hypo_idx': hypo_idx,
                    'overlap_score': overlap_score,
                    'overlap_stats': overlap_stats,
                })

            # 按重叠分数从大到小排序
            overlap_map = sorted(
                overlap_map,
                key=lambda x: x['overlap_score'],
                reverse=True
            )

            if top_k is not None:
                top_results = overlap_map[:top_k]
            else:
                top_results = overlap_map

            oral_hypos_list.append({
                'trial_idx': trial_idx,
                'choice': int(choices[trial_idx]),
                'reported_region': reported_region,
                'top_hypos': [item['hypo_idx'] for item in top_results],
                'top_scores': [item['overlap_score'] for item in top_results],
                'all_results': overlap_map,
            })

        return oral_hypos_list


    def get_oral_hypo_hits(
        self,
        data: pd.DataFrame,
        top_k: Optional[int] = None,
        window_size: int = 16,
        n_samples: int = 50000,
        bounds: Tuple[float, float] = (0.0, 1.0),
        random_state: Optional[int] = 42,
        overlap_metric: str = 'iou',
    ):
        """
        基于 oral region -> hypothesis overlap 的方法，计算每个被试每个 trial 的 oral_hypo_hits

        参数
        ----
        data : pd.DataFrame
            至少包含列:
            - iSub
            - condition
            - choice
            - A
            - b

        top_k : Optional[int]
            只看每个 trial overlap 排名前 k 的 hypotheses。
            若为 None，则使用全部 hypotheses。

        window_size : int
            rolling hit rate 的窗口大小

        n_samples : int
            Monte Carlo 估计区域重叠时的采样数

        bounds : tuple
            oral / true region 所定义的搜索空间边界，默认 [0,1]^4

        random_state : Optional[int]
            随机种子

        overlap_metric : str
            用哪个重叠指标排序，可选 'iou', 'intersection', 'precision_like', 'recall_like'
        """
        learning_data = data.copy()
        oral_hypos_list = {}

        # ========= 先为每个被试计算 oral_hypos_list =========
        for _, subj_df in learning_data.groupby('iSub'):
            subj_df = subj_df.sort_values(['iSession', 'iTrial']).reset_index(drop=True)

            iSub = int(subj_df['iSub'].iloc[0])
            cond = int(subj_df['condition'].iloc[0])
            model = StateModel(engine_config, condition=cond, subject_id=iSub)

            # 组装成每个 trial 一个 (A, b)
            oral_region = [
                (row['A'], row['b'])
                for _, row in subj_df.iterrows()
            ]

            choices = subj_df['choice'].values

            oral_hypos_list[iSub] = self.get_oral_hypos_list(
                condition=cond,
                oral_region=oral_region,
                choices=choices,
                model=model,
                top_k=top_k,
                n_samples=n_samples,
                bounds=bounds,
                random_state=random_state,
                overlap_metric=overlap_metric,
            )

        # ========= 再把 hypos 转成 hits =========
        oral_hypo_hits = {}

        for iSub, trial_results in oral_hypos_list.items():
            condition = int(
                learning_data.loc[learning_data['iSub'] == iSub, 'condition'].iloc[0]
            )
            target_value = 0 if condition == 1 else 42

            hits = []
            top_hypos_per_trial = []
            top_scores_per_trial = []

            for trial_res in trial_results:
                if trial_res is None:
                    hits.append(np.nan)
                    top_hypos_per_trial.append([])
                    top_scores_per_trial.append([])
                    continue

                current_top_hypos = trial_res.get('top_hypos', [])
                current_top_scores = trial_res.get('top_scores', [])

                top_hypos_per_trial.append(current_top_hypos)
                top_scores_per_trial.append(current_top_scores)

                if len(current_top_hypos) == 0:
                    hits.append(np.nan)
                else:
                    hits.append(1 if target_value in current_top_hypos else 0)

            rolling_hits = pd.Series(hits).rolling(
                window=window_size,
                min_periods=window_size
            ).mean().tolist()

            oral_hypo_hits[iSub] = {
                'iSub': iSub,
                'condition': condition,
                'target_hypo': target_value,
                'hits': hits,
                'rolling_hits': rolling_hits,
                'top_hypos_per_trial': top_hypos_per_trial,
                'top_scores_per_trial': top_scores_per_trial,
                'trial_results': trial_results,
            }

        return oral_hypo_hits




class Oral_to_coordinate:

    def get_oral_hypos_list(self,
                            condition: int,
                            data: Tuple[np.ndarray, np.ndarray],
                            partition,
                            dist_tol: float = 1e-9,
                            top_k: Optional[int] = None,
                            ) -> Dict[str, Any]:

        oral_centers, choices = data
        n_trials = len(choices)

        n_hypos = partition.prototypes_np.shape[0]
        all_hypos = range(n_hypos)

        oral_hypos_list = []

        for trial_idx in range(n_trials):
            reported_center = oral_centers[trial_idx]

            # If reported_center is missing or all NaNs, return empty list
            if reported_center is None \
            or (isinstance(reported_center, np.ndarray) and reported_center.size == 0) \
            or (isinstance(reported_center, np.ndarray) and np.all(np.isnan(reported_center))):
                oral_hypos_list.append([])
                continue

            cat_idx = choices[trial_idx] - 1

            # Compute distances to each hypothesis prototype
            distance_map = []
            for hypo_idx in all_hypos:
                true_center = partition.prototypes_np[hypo_idx, 0, cat_idx, :]
                distance_val = np.linalg.norm(reported_center - true_center)
                distance_map.append((distance_val, hypo_idx))

            # Exact matches within tolerance
            exact_matches = [h for (d, h) in distance_map if d <= dist_tol]

            if top_k is None or top_k == 0:
                if condition == 1:
                    top_k = 4
                else:
                    top_k = 10

            if exact_matches:
                chosen_hypos = exact_matches
            else:
                distance_map.sort(key=lambda x: x[0])
                chosen_hypos = [h for (_, h) in distance_map[:top_k]]

            oral_hypos_list.append(chosen_hypos)

        return oral_hypos_list

    def get_oral_hypo_hits(self,
                            data: pd.DataFrame,
                            ):
        
        learning_data = data
        oral_hypos_list = {}

        for _, subj_df in learning_data.groupby('iSub'):
            iSub   = int(subj_df['iSub'].iloc[0])
            cond   = int(subj_df['condition'].iloc[0])
            # Use Partition directly instead of StandardModel
            n_cats = 2 if cond == 1 else 4
            partition = Partition(n_dims=4, n_cats=n_cats)

            oral_cols = ['feature1_oral','feature2_oral','feature3_oral','feature4_oral']
            oral_value_cols = ['feature1_oralvalue','feature2_oralvalue','feature3_oralvalue','feature4_oralvalue']

            if all(col in subj_df.columns for col in oral_cols):
                centres = subj_df[oral_cols].to_numpy(dtype=float)
            elif all(col in subj_df.columns for col in oral_value_cols):
                centres = subj_df[oral_value_cols].to_numpy(dtype=float)
            else:
                missing = set(oral_cols) - set(subj_df.columns)
                raise KeyError(f"Oral coordinate columns missing. Expected {oral_cols} or {oral_value_cols}; missing {sorted(missing)}")
            choices = subj_df['choice'].values

            oral_hypos_list[iSub] = self.get_oral_hypos_list(cond,
                (centres, choices), partition)
            
        oral_hypo_hits = {}

        for iSub, hypos in oral_hypos_list.items():
            condition = learning_data[learning_data['iSub'] ==
                                        iSub]['condition'].iloc[0]
            target_value = 0 if condition == 1 else 42

            hits = []  # 用于存储每个 trial 的 hit 值
            for trial_hypos in hypos:
                if not trial_hypos:
                    hits.append([])
                else:
                    hits.append(1 if target_value in trial_hypos else 0)

            # 计算hits的16试次滑动平均（统一在循环结束后计算，避免未定义）
            numeric_hits = [h if isinstance(h, (int, float)) else 0 for h in hits]
            rolling_hits = pd.Series(numeric_hits).rolling(window=16, min_periods=16).mean().tolist()
                    
            oral_hypo_hits[iSub] = {
                'iSub': iSub,
                'condition': condition,
                'hits': hits,
                'rolling_hits': rolling_hits
            }
        
        return oral_hypo_hits
