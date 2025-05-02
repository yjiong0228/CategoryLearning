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


    def adaptive_amount_evalutator(self, amount: float | str | Callable,
                                   **kwargs) -> int:
        """
        Adaptively deal with evaluator / number format of amount
        """
        match amount:
            case int():
                return amount
            case Callable():
                return amount(**kwargs)
            case str() if amount in self.amount_evaluators:
                return self.amount_evaluators[amount](**kwargs)
            case _:
                raise Exception(f"Unexpected amount type. {amount}")

    @classmethod
    def _amount_entropy_gen(cls, max_amount=3):

        def _amount_entropy_based(posterior: Dict,
                                  max_amount=max_amount,
                                  **kwargs) -> int:
            """
            Use whether a model is decisive in its posterior to tune the amount.
            """
            posterior_ = np.array(list(posterior.values()))[:, 0]
            p_entropy = entropy(posterior_)
            return max(
                0,
                int(max_amount - min(np.exp(p_entropy), max_amount + 30)) + 2)

        return _amount_entropy_based

    @classmethod
    def _amount_max_gen(cls, max_amount=3):

        def _amount_max_based(posterior=Dict, max_amount=max_amount, **kwargs):
            posterior_ = np.array(list(posterior.values()))[:, 0]
            max_post = np.max(posterior_)
            return 0 if 3. / max_post > max_amount else int(3. / max_post)

        return _amount_max_based

    @classmethod
    def _amount_random_gen(cls, max_amount=3):

        def _amount_random_based(posterior=Dict,
                                 max_amount=max_amount,
                                 **kwargs):
            posterior_ = np.array(list(posterior.values()))[:, 0]
            max_post = np.max(posterior_)
            return np.random.choice(max_amount + 1,
                                    p=[1 - max_post] +
                                    [max_post / max_amount] * max_amount)

        return _amount_random_based

    @classmethod
    def _amount_accuracy_gen(cls, amount_function: Callable, max_amount=3):

        def _amount_accuracy_static(
                feedbacks: List[float],
                amount_function: Callable = amount_function,
                **kwargs) -> int:
            feedbacks = [int(f) for f in feedbacks]
            accuracy = np.sum(feedbacks) / len(feedbacks)
            return amount_function(accuracy) if amount_function(
                accuracy) < max_amount else max_amount

        return _amount_accuracy_static



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
        "random_6": BaseCluster._amount_random_gen(6),
        "random_7": BaseCluster._amount_random_gen(7),
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
        amount = int(amount)
        N = len(available_hypos)
        if amount <= 0 or N == 0:
            return []
        # clamp to the size of the pool
        amount = min(amount, N)

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
        Cluster strategy: top posterior

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

    @classmethod
    def _cluster_strategy_random_post(cls,
                                      amount: int,
                                      available_hypos: Set,
                                      posterior: Dict | None = None,
                                      **kwargs) -> List:
        """
        Cluster strategy: random n from posterior                                        
        """
        # posterior as a dict
        posterior = posterior or {}

        # 只保留同时出现在 posterior 与 available_hypos 里的下标
        cand_idx = [i for i in available_hypos if i in posterior]
        if not cand_idx:
            return []

        # 取概率质量（支持 float 或 (prob, β) 形式）
        raw_w = np.asarray([
            posterior[i][0] if isinstance(posterior[i], (list, tuple, np.ndarray))
            else posterior[i]
            for i in cand_idx
        ], dtype=float)

        n_pos = int((raw_w > 0).sum())
        amount = min(amount, len(cand_idx))            # 安全上限
        if n_pos < amount:                             # 权重为 0 的太多
            # → 退化为均匀随机（或也可以给零权重加一个极小值）
            chosen = np.random.choice(cand_idx, size=amount, replace=False)
            return chosen.tolist()

        prob = raw_w / raw_w.sum()
        chosen = np.random.choice(cand_idx, size=amount,
                                replace=False, p=prob)
        return chosen.tolist()        

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
        if amount == 0:
            return []
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

        return ret_val.tolist()

    def cluster_transition(self,
                           full_hypo_set: BaseSet | None = None,
                           **kwargs) -> List:
        """
        Make the transition
        """
        new_hypos: Set[int] = set([])
        numerical_amounts = []
        hypo_choices = []

        if full_hypo_set is None:
            available_hypos = set(range(self.length))
        else:
            available_hypos = set(full_hypo_set)

        for amount, method in self.cluster_transition_strategy:
            # 1. compute how many to draw
            numerical_amount = self.adaptive_amount_evalutator(
                amount, **kwargs)
            # 2. clamp so you never ask for more than you have
            numerical_amount = max(0,
                                   min(numerical_amount, len(available_hypos)))

            numerical_amounts.append(numerical_amount)

            # 3. sample
            new_part = method(numerical_amount, available_hypos, **kwargs)
            hypo_choices.append(new_part)
            new_hypos |= set(new_part)
            available_hypos -= set(new_part)

        if len(new_hypos) == 0:
            new_hypos = set(
                np.random.choice(list(available_hypos),
                                 size=1,
                                 replace=False).tolist())
            return list(new_hypos), {
                'random from available': (1, list(new_hypos))
            }
        else:
            return list(new_hypos), {
                k: (amt, ch)
                for k, amt, ch in zip(self.strategy_name, numerical_amounts,
                                    hypo_choices)
            }

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
        self.strategy_name = []
        for k_amount, k_strategy in strategy:
            if isinstance(k_strategy, str):
                self.strategy_name.append(k_strategy)
            else:
                self.strategy_name.append(k_strategy.__name__)
            match k_strategy:
                case "stable":
                    self.cluster_transition_strategy.append(
                        (k_amount, self._cluster_strategy_stable))

                case "top_posterior":
                    self.cluster_transition_strategy.append(
                        (k_amount, self._cluster_strategy_top_post))

                case "random_posterior":
                    self.cluster_transition_strategy.append(
                        (k_amount, self._cluster_strategy_random_post))

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
