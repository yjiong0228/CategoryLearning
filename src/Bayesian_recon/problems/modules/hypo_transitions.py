"""
Module: Modeling Hypothesis-cluster transition dynamics
"""
from abc import ABC
from collections.abc import Callable
from typing import List, Tuple, Dict, Set
import numpy as np
from .base_module import BasePartition, BaseModule
from .base_module import (cdist, softmax, BaseSet, entropy)


class BaseCluster(BaseModule):

    amount_evaluators = {}

    def __init__(self, model, cluster_config: Dict = {}, **kwargs):
        super().__init__(model, **kwargs)
        self.model = model
        self.config = cluster_config
        self.partition = self.model.partition_model
        self.n_dims = self.partition.n_dims
        self.n_cats = self.partition.n_cats
        self.current_cluster = BaseSet([])
        self.length = self.partition.length

    @classmethod
    def adaptive_amount_evalutator(cls, amount: float | str | Callable,
                                   **kwargs) -> int:
        """
        Adaptively deal with evaluator / number format of amount
        """
        match amount:
            case int():
                return amount
            case Callable():
                return amount(**kwargs)
            case str() if amount in cls.amount_evaluators:
                return cls.amount_evaluators[amount](**kwargs)
            case _:
                raise Exception("Unexpected amount type.")

    @classmethod
    def _amount_entropy_gen(cls, max_amount=3):

        def _amount_entropy_based(posterior: Dict,
                                  max_amount=max_amount,
                                  **kwargs) -> int:
            """
            Use whether a model is decisive in its posterior to tune the amount.
            """
            posterior_ = np.array(list(posterior.values()))
            p_entropy = entropy(posterior_)
            return max(
                0,
                int(max_amount - min(np.exp(p_entropy), max_amount + 30)) + 2)

        return _amount_entropy_based

    @classmethod
    def _amount_max_gen(cls, max_amount=3):

        def _amount_max_based(posterior=Dict, max_amount=max_amount, **kwargs):
            posterior = np.array(list(posterior.values()))
            max_post = np.max(posterior)
            return 0 if 3. / max_post > max_amount else int(3. / max_post)

        return _amount_max_based

    @classmethod
    def _amount_random_gen(cls, max_amount=3):

        def _amount_random_based(posterior=Dict,
                                 max_amount=max_amount,
                                 **kwargs):
            posterior = np.array(list(posterior.values()))
            max_post = np.max(posterior)
            return np.random.choice(max_amount + 1,
                                    p=[1 - max_post] +
                                    [max_post / max_amount] * max_amount)

        return _amount_random_based


class PartitionCluster(BaseCluster):
    """
    Cluster 模块

    示例 cluster_config
    -------------------
    {
        "amount_range": [(1, 5), (1, 5), (1, 5)],
        "transition_spec": ["posterior_entropy", "ksimilar_centers", "random"]
    }
    """

    # 静态注册 amount_evaluators（继承即可使用）
    amount_evaluators = {
        # entropy-系列
        **{f"entropy_{n}": BaseCluster._amount_entropy_gen(n) for n in range(1, 6)},
        # max-系列
        **{f"max_{n}": BaseCluster._amount_max_gen(n) for n in range(1, 6)},
        # random-系列
        **{f"random_{n}": BaseCluster._amount_random_gen(n) for n in range(1, 6)},
    }

    # ---------------------- 初始化 & 网格声明 -------------------------------

    def __init__(self, model, cluster_config: Dict = {}, **kwargs):
        super().__init__(model, **kwargs)

        # 1. 解析配置
        self.amount_range: List[Tuple[int, int]] = cluster_config.get("amount_range", [(1, 5)])
        self.transition_spec: List[str] = cluster_config.get("transition_spec", ["random"])
        if len(self.amount_range) != len(self.transition_spec):
            raise ValueError("amount_range 与 transition_spec 长度必须一致")

        # 2. 声明供模型收集的参数
        self._params_dict: Dict[str, type] = {
            f"cluster_amount_{i}": int for i in range(len(self.amount_range))
        }
        self._optimize_params_dict: Dict[str, List[int]] = {
            f"cluster_amount_{i}": list(range(lo, hi + 1))
            for i, (lo, hi) in enumerate(self.amount_range)
        }

        # 3. 预计算中心距离
        self.cached_dist: Dict[Tuple, float] = {}
        self._calc_cached_dist()

    # ----- properties 让 StandardModel 自动收集 -----------------------------

    @property
    def params_dict(self):
        return self._params_dict

    @property
    def optimize_params_dict(self):
        return self._optimize_params_dict

    # ---------------------- 距离缓存 ----------------------------------------

    def _calc_cached_dist(self):
        """
        计算并缓存所有 center 间欧氏距离
        """
        self.cached_dist.clear()
        for i_l, left in self.partition.centers:
            for i_r, right in self.partition.centers:
                try:
                    for _, c_l in left.items():
                        for _, c_r in right.items():
                            key = (*c_l, *c_r)
                            inv = (*c_r, *c_l)
                            if key in self.cached_dist:
                                continue
                            d = np.linalg.norm(np.array(c_l) - np.array(c_r))
                            self.cached_dist[key] = self.cached_dist[inv] = d
                except Exception as e:
                    print("Error in _calc_cached_dist:", e)
                    raise

    def center_dist(self, this, other) -> float:
        return self.cached_dist.get((*this, *other), np.inf)

    # ----------------------- 基础 strategy ----------------------------------

    @classmethod
    def _cluster_strategy_stable(cls, amount: int, available: Set[int], **_):
        return list(available)[:amount]

    @classmethod
    def _cluster_strategy_random(cls, amount: int, available: Set[int], **_):
        return np.random.choice(list(available), size=amount, replace=False).tolist()

    @classmethod
    def _cluster_strategy_top_post(
        cls,
        amount: int,
        available: Set[int],
        posterior: Dict | None = None,
        **kwargs,
    ):
        posterior = posterior or {}
        sorted_h = sorted(
            [(i, p) for i, p in posterior.items() if i in available],
            key=lambda x: x[1],
            reverse=True,
        )
        return [i for i, _ in sorted_h][:amount]

    # -- k-similar-centers (与原实现保持一致，略有精简) -----------------------

    def _cluster_strategy_ksimilar_centers(
        self,
        amount: int,
        available_hypos: Set[int],
        posterior: Dict | None = None,
        stimulus: np.ndarray = np.zeros(4),
        **kwargs,
    ):
        if posterior is None:
            raise ValueError("posterior 必须提供给 ksimilar_centers")

        # proto hypo
        proto_amount = kwargs.get("proto_hypo_amount", 1)
        proto_method = kwargs.get("proto_hypo_method", "top")
        ref_hypos = sorted(posterior.items(), key=lambda x: x[1], reverse=True)

        if proto_method == "random":
            idx = np.random.choice(len(ref_hypos), size=proto_amount, replace=False, p=None)
            ref_hypos = [ref_hypos[i] for i in idx]
        else:  # "top"
            ref_hypos = ref_hypos[:proto_amount]

        ref_idx = np.array([i for i, _ in ref_hypos])
        ref_post = np.array([p for _, p in ref_hypos])
        ref_beta = ref_post  # 原实现把 posterior 当 beta，这里保持一致

        ref_centers = np.array(
            [list(self.partition.centers[i][1].values()) for i in ref_idx]
        )  # shape: (P, n_cats, n_dims)

        ref_dist = cdist(
            stimulus.reshape(1, -1),
            ref_centers.reshape(-1, self.n_dims),
        ).reshape(len(ref_idx), self.n_cats)
        ref_choices = [
            np.random.choice(self.n_cats, p=softmax(dist, beta=-b))
            for dist, b in zip(ref_dist, ref_beta)
        ]
        ref_center_pts = ref_centers[np.arange(len(ref_idx)), ref_choices]

        # candidate
        cand_idx = [k for k in available_hypos if k not in ref_idx]
        cand_centers = np.array(
            [list(self.partition.centers[k][1].values()) for k in cand_idx]
        )

        exp_dist = np.exp(
            [
                [
                    -self.center_dist(ref_center_pts[i], cand_centers[j, ref_choices[i]])
                    for i in range(len(ref_idx))
                ]
                for j in range(len(cand_idx))
            ]
        )  # shape: (C, P)

        score = exp_dist @ ref_post  # (C,)
        if kwargs.get("cluster_hypo_method", "top") == "random":
            chosen = np.random.choice(cand_idx, size=amount, replace=False, p=softmax(score))
        else:
            chosen = np.array(cand_idx)[np.argsort(score)[-amount:]]
        return chosen.tolist()

    # --------------------- 核心：cluster_transition -------------------------

    def cluster_transition(self, full_hypo_set: BaseSet | None = None, **kwargs):
        """
        动态读取 cluster_amount_i，组装并执行策略
        """
        available = set(range(self.length)) if full_hypo_set is None else set(full_hypo_set)
        new_hypos: Set[int] = set()

        for idx, spec in enumerate(self.transition_spec):
            amt_key = f"cluster_amount_{idx}"
            amount_val = kwargs.get(amt_key, self.amount_range[idx][0])  # 默认区间下界

            # spec → callable & amount 处理
            if spec.startswith("posterior_"):
                method = self._cluster_strategy_top_post
                amount_val = f"{spec.split('_')[1]}_{amount_val}"
            elif spec == "ksimilar_centers":
                method = self._cluster_strategy_ksimilar_centers
            elif spec == "random":
                method = self._cluster_strategy_random
            elif spec == "stable":
                method = self._cluster_strategy_stable
            else:
                raise ValueError(f"未知 transition_spec: {spec}")

            n_amount = self.adaptive_amount_evalutator(amount_val, **kwargs)
            part = method(n_amount, available, **kwargs)
            new_hypos.update(part)
            available.difference_update(part)

        if not new_hypos:
            new_hypos.add(np.random.choice(list(available)))

        return list(new_hypos)

    def cluster_init(self, **kwargs):
        """
        fit_step_by_step() 在第 0 trial 调用，用于给模型一个初始 hypo pool
        """
        init_size = min(10, self.length)
        return self._cluster_strategy_random(init_size, set(range(self.length)))
