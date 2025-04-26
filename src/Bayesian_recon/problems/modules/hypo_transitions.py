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
        self.previous_accuracy = None  # 用于存储上一轮的准确率
        self.accuracy_history = []  # 用于记录每次计算的准确率

    @classmethod
    def adaptive_amount_evaluator(cls, amount: float | str | Callable,
                                  **kwargs) -> int:
        """
        Adaptively deal with evaluator / number format of amount
        """
        print(f"Evaluating amount: {amount}")  # 调试输出   
        match amount:
            case int():
                return amount
            case Callable():
                return amount(**kwargs)
            case str() if amount in cls.amount_evaluators:
                return cls.amount_evaluators[amount](**kwargs)
            case _:
                raise Exception("Unexpected amount type: {type(amount)}.")

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
            # ---------- 加一行归一化 ----------
            posterior_vals = np.array(list(posterior.values()), dtype=float)
            if posterior_vals.size == 0 or posterior_vals.sum() == 0:
                max_post_norm = 0.0  # 退化情况
            else:
                max_post_norm = posterior_vals.max() / posterior_vals.sum()
            # ----------------------------------

            probs = [1 - max_post_norm
                     ] + [max_post_norm / max_amount] * max_amount
            return np.random.choice(max_amount + 1, p=probs)

        return _amount_random_based

    @classmethod
    def _amount_accuracy_gen(cls, max_amount=3):
        """
        基于准确率动态调整 amount
        """

        def _amount_accuracy_based(current_accuracy: float, **kwargs) -> int:
            """
            根据反馈准确率调整策略的amount。
            如果当前准确率比上一轮高，增加top_posterior的amount，减少random的amount。
            如果当前准确率比上一轮低，减少top_posterior的amount，增加random的amount。
            """

            # 保存当前准确率到history中
            cls.accuracy_history.append(current_accuracy)

            # 如果有上一轮的准确率，则计算差异
            if cls.previous_accuracy is not None:
                # 如果当前准确率比上一轮高，增加top_posterior的amount，减少random的amount
                # 如果当前准确率比上一轮低，减少top_posterior的amount，增加random的amount
                top_posterior_amount = kwargs.get('top_posterior_amount',
                                                  5)  # 默认值为5
                random_amount = kwargs.get('random_amount', 5)  # 默认值为5

                if current_accuracy > cls.previous_accuracy:
                    top_posterior_amount = min(top_posterior_amount + 1,
                                               max_amount)
                    random_amount = max(random_amount - 1, 0)
                elif current_accuracy < cls.previous_accuracy:
                    top_posterior_amount = max(top_posterior_amount - 1, 0)
                    random_amount = min(random_amount + 1, max_amount)

                # 更新上一轮的准确率
                cls.previous_accuracy = current_accuracy

                # 返回调整后的amount
                return top_posterior_amount, random_amount
            else:
                # 如果没有上一轮准确率，则初始化
                cls.previous_accuracy = current_accuracy
                return kwargs.get('top_posterior_amount',
                                  5), kwargs.get('random_amount', 5)

        return _amount_accuracy_based



class PartitionCluster(BaseCluster):
    """
    Partition with hypothesis cluster structure
    """

    amount_evaluators = {
        "entropy": BaseCluster._amount_entropy_gen(3),
        "entropy_1": BaseCluster._amount_entropy_gen(1),
        "entropy_2": BaseCluster._amount_entropy_gen(2),
        "entropy_3": BaseCluster._amount_entropy_gen(3),
        "entropy_4": BaseCluster._amount_entropy_gen(4),
        "entropy_5": BaseCluster._amount_entropy_gen(5),
        "max_1": BaseCluster._amount_max_gen(1),
        "max_2": BaseCluster._amount_max_gen(2),
        "max_3": BaseCluster._amount_max_gen(3),
        "max_4": BaseCluster._amount_max_gen(4),
        "max_5": BaseCluster._amount_max_gen(5),
        "random_1": BaseCluster._amount_random_gen(1),
        "random_2": BaseCluster._amount_random_gen(2),
        "random_3": BaseCluster._amount_random_gen(3),
        "random_4": BaseCluster._amount_random_gen(4),
        "random_5": BaseCluster._amount_random_gen(5),
        "accuracy": BaseCluster._amount_accuracy_gen(8)
    }

    def __init__(self, model, cluster_config: Dict = {}, **kwargs):
        """
        Initialize

        Model:
        current_cluster: a subset of all partition prototypes
                         (named by their index in self.partition)
        strategy: a spectrum, in terms of a list
                  [(amount: int, method: str|Callable)]
        """
        super().__init__(model, **kwargs)
        self.set_cluster_transition_strategy(
            kwargs.get("transition_spec", [(10, "stable")]))
        self.cached_dist: Dict[Tuple, float] = {}

        self._calc_cached_dist()

    def _calc_cached_dist(self):
        """
        Calculate Cached diatances
        """
        self.cached_dist = {}
        for i_l, left in self.partition.centers:
            for i_r, right in self.partition.centers:
                try:
                    for _, c_l in left.items():
                        for _, c_r in right.items():
                            key = (*c_l, *c_r)
                            inv = (*c_r, *c_l)
                            if key in self.cached_dist or inv in self.cached_dist:
                                continue
                            self.cached_dist[key] = np.sum(
                                (np.array(c_l) - np.array(c_r))**2)**0.5
                            self.cached_dist[inv] = self.cached_dist[key]
                except Exception as e:
                    print(e)
                    print(i_l, left, i_r, right)
                    raise e

    def center_dist(self, this, other) -> float:
        """
        Read out center distances
        """
        return self.cached_dist.get((*this, *other), np.inf)

    @classmethod
    def _cluster_strategy_stable(cls, amount: int, available_hypos: Set,
                                 **kwargs) -> List:
        """
        Cluster strategy: stable
        """
        return list(available_hypos)[:amount]

    @classmethod
    def _cluster_strategy_random(cls, amount: int, available_hypos: Set,
                                 **kwargs) -> List:
        """
        Cluster strategy: stable
        """
        return np.random.choice(list(available_hypos),
                                size=amount,
                                replace=False).tolist()

    @classmethod
    def _cluster_strategy_top_post(cls,
                                   amount: int,
                                   available_hypos: Set,
                                   posterior: Dict | None = None,
                                   **kwargs) -> List:
        """
        Cluster strategy: stable


        - functional args in kwargs:
        top_p: float between 0 to 1. Drives this method filter via top-p
               mechanism, default value results in top-n with amount setup.
        """
        # posterior as a dict
        posterior = posterior or {}
        # sort the posterior in their posterior probabilities
        sorted_by_post = sorted(
            [(i, p) for i, p in posterior.items() if i in available_hypos],
            key=lambda x: x[1],
            reverse=True)

        # process the "Top-p" case
        if (prob := kwargs.get("top_p", 0.)) > 0:
            new_hypos = [sorted_by_post[0]]
            for i in range(1, len(sorted_by_post)):
                if new_hypos[-1][1] > prob:
                    break
                new_hypos.append((i, new_hypos[-1][1] + sorted_by_post[i][1]))

            return [x for x, _ in new_hypos]

        # original top-n case
        return [x for x, _ in sorted_by_post][:amount]

    def _cluster_strategy_ksimilar_centers(self,
                                           amount: int,
                                           available_hypos: Set,
                                           posterior: Dict | None = None,
                                           stimulus: np.ndarray = np.zeros(4),
                                           **kwargs):
        """
        Cluster strategy: ksimilar distance version

        - functional args in kwargs:
        proto_hypo_amounts: use this number of hypotheses as prototype for
                            recalling other hypos from full k-space, default 1.

        top_p: float between 0 to 1. Drives this method filter via top-p
               mechanism, default value results in top-n with amount setup.
        """

        if posterior is None:
            raise Exception("ArgumentError: posterior is absent or not a Dict")
        proto_hypo_amount = kwargs.get("proto_hypo_amount", 1)
        ref_hypos = sorted([(i, *p) for i, p in posterior.items()],
                           key=lambda x: x[1],
                           reverse=True)

        match kwargs.get("proto_hypo_method", "top"):
            case "top":
                ref_hypos = ref_hypos[:proto_hypo_amount]
            case "random":
                ref_hypos = [
                    ref_hypos[i]
                    for i in np.random.choice(np.arange(len(ref_hypos)),
                                              size=proto_hypo_amount,
                                              p=[x for _, x, _ in ref_hypos],
                                              replace=False)
                ]
            case _:
                ref_hypos = ref_hypos[:proto_hypo_amount]

        # in case that proto_hypo_amount is greater than len(ref_hypos)
        proto_hypo_amount = len(ref_hypos)
        # prepare the reference hypos: index, chioce, posterior
        ref_hypos_index = np.array([k for k, _, _ in ref_hypos])
        ref_hypos_post = np.array([x for _, x, _ in ref_hypos])
        ref_hypos_beta = np.array([x for _, _, x in ref_hypos])
        # ref_full_centers is of shape (proto_hypo_amount, n_cats, n_dims)
        ref_full_centers = np.array([
            list(self.partition.centers[k][1].values())
            for k in ref_hypos_index
        ])
        ref_dist = cdist(
            np.array(stimulus).reshape(1, -1),
            ref_full_centers.reshape(-1, self.n_dims))
        # given stimulus, the argmin choices on each reference hypo
        ref_choices = [
            np.random.choice(self.n_cats, p=prob)
            for prob in softmax(ref_dist.reshape(-1, self.n_cats),
                                beta=-ref_hypos_beta.reshape(-1, 1),
                                axis=1)
        ]
        # prepare the reference centers(shape=[proto_hypo_amount, n_dims])
        ref_hypos_center = ref_full_centers[range(proto_hypo_amount),
                                            ref_choices]

        # prepare the candidate hypos: index and center(shape=[K,d])
        candidate_hypos_index = [
            k for k in available_hypos if k not in ref_hypos_index
        ]
        candidate_full_center = np.array([
            list(self.partition.centers[k][1].values())
            for k in candidate_hypos_index
        ])

        exp_dist = np.exp([[
            -1 * self.center_dist(ref_hypos_center[i],
                                  candidate_full_center[j, ref_choices[i]])
            for i, _ in enumerate(ref_hypos_index)
        ] for j, _ in enumerate(candidate_hypos_index)])
        score = np.einsum("ij,j->i", exp_dist, ref_hypos_post)

        match kwargs.get("cluster_hypo_method", "top"):
            case "top":
                argscore = np.argsort(score)[-amount:]
                ret_val = np.array(candidate_hypos_index)[argscore]
            case "random":
                ret_val = np.random.choice(candidate_hypos_index,
                                           p=softmax(score),
                                           size=amount,
                                           replace=False)
            case _:
                argscore = np.argsort(score)[-amount:]
                ret_val = np.array(candidate_hypos_index)[argscore]

        return list(ret_val)

    def cluster_transition(self,
                           full_hypo_set: BaseSet | None = None,
                           **kwargs) -> List:
        """
        Make the transition
        """
        new_hypos: Set[int] = set([])
        if full_hypo_set is None:
            available_hypos = set(range(self.length))
        else:
            available_hypos = set(full_hypo_set)

        # Get feedback from the data (past 15 trials)
        feedback = kwargs.get("feedback", [])
        current_accuracy = np.mean(feedback)  # Calculate accuracy

        for amount, method in self.cluster_transition_strategy:

            numerical_amount = self.adaptive_amount_evaluator(
                amount, current_accuracy=current_accuracy, **kwargs)
            new_part = method(numerical_amount, available_hypos, **kwargs)
            new_hypos = new_hypos.union(set(new_part))
            available_hypos = available_hypos.difference(new_part)

        return list(new_hypos)

    def cluster_init(self, **kwargs):
        return self._cluster_strategy_random(10, set(range(self.length)))

    def set_cluster_transition_strategy(
        self,
        strategy: List[Tuple[int, str | Callable]],
    ):
        """
        Set cluster transition strategy

        strategy: a list in terms of [(amount, method)]
        """

        self.cluster_transition_strategy = []
        for k_amount, k_strategy in strategy:
            match k_strategy:
                case "stable":
                    self.cluster_transition_strategy.append(
                        (k_amount, self._cluster_strategy_stable))

                case "top_posterior":
                    self.cluster_transition_strategy.append(
                        (k_amount, self._cluster_strategy_top_post))

                case "random":
                    self.cluster_transition_strategy.append(
                        (k_amount, self._cluster_strategy_random))

                case "ksimilar_centers":
                    self.cluster_transition_strategy.append(
                        (k_amount, self._cluster_strategy_ksimilar_centers))

                case Callable() as method:
                    self.cluster_transition_strategy.append((k_amount, method))

                case _:
                    raise Exception(
                        f"Filtering method {method} is not a valid choice!")
