from typing import Optional, List, Dict, Union, Tuple, Any
import numpy as np
import pandas as pd


class Oral_to_coordinate:

    def get_oral_hypos_list(self,
                            condition: int,
                            data: Tuple[np.ndarray, np.ndarray],
                            model,
                            dist_tol: float = 1e-9,
                            top_k: Optional[int] = None,
                            ) -> Dict[str, Any]:

        oral_centers, choices = data
        n_trials = len(choices)

        n_hypos = model.partition_model.prototypes_np.shape[0]
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
                true_center = model.partition_model.prototypes_np[hypo_idx, 0, cat_idx, :]
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
