"""
Modeling Hypothesis-cluster transition dynamics
"""
from collections.abc import Callable
from typing import List, Tuple, Dict, Set
import numpy as np
from .partitions import Partition
from .base_problem import (cdist, BaseSet)


class PartitionCluster(Partition):
    """
    Partition with hypothesis cluster structure
    """

    def __init__(self, n_dims: int, n_cats: int, n_protos: int = 1, **kwargs):
        """
        Initialize

        Model:
        current_cluster: a subset of all partition prototypes
                         (named by their index in self.partition)
        strategy: a spectrum, in terms of a list
                  [(amount: int, method: str|Callable)]
        """
        super().__init__(n_dims, n_cats, n_protos, **kwargs)
        self.set_cluster_transition_strategy(
            kwargs.get("transition_spec", [(10, "stable")]))
        self.current_cluster = BaseSet([])
        self.cached_dist: Dict[Tuple, float] = {}
        for _, left in self.centers:
            for _, right in self.centers:
                for _, c_l in left.items():
                    for _, c_r in right.items():
                        key = (*c_l, *c_r)
                        inv = (*c_r, *c_l)
                        if key in self.cached_dist or inv in self.cached_dist:
                            continue
                        self.cached_dist[key] = np.sum(
                            (np.array(c_l) - np.array(c_r))**2)**0.5
                        self.cached_dist[inv] = self.cached_dist[key]

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

            return [x for x, _ in new_hypos]\

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
        ref_hypos = sorted([(i, p) for i, p in posterior.items()],
                           key=lambda x: x[1],
                           reverse=True)[:proto_hypo_amount]
        # in case that proto_hypo_amount is greater than len(ref_hypos)
        proto_hypo_amount = len(ref_hypos)
        # prepare the reference hypos: index, chioce, posterior
        ref_hypos_index = np.array([k for k, _ in ref_hypos])
        ref_hypos_post = np.array([x for _, x in ref_hypos])
        # ref_full_centers is of shape (proto_hypo_amount, n_cats, n_dims)
        ref_full_centers = np.array(
            [list(self.centers[k][1].values()) for k in ref_hypos_index])
        ref_dist = cdist(
            np.array(stimulus).reshape(1, -1),
            ref_full_centers.reshape(-1, self.n_dims))
        # given stimulus, the argmin choices on each reference hypo
        ref_choices = np.argmin(ref_dist.reshape(-1, self.n_cats), axis=1)
        # prepare the reference centers(shape=[proto_hypo_amount, n_dims])
        ref_hypos_center = ref_full_centers[range(proto_hypo_amount),
                                            ref_choices]

        # prepare the candidate hypos: index and center(shape=[K,d])
        candidate_hypos_index = [
            k for k in available_hypos if k not in ref_hypos_index
        ]
        candidate_full_center = np.array(
            [list(self.centers[k][1].values()) for k in candidate_hypos_index])

        # print("test",
        #       set(range(self.length)).difference(set(candidate_hypos_index)))
        # print("candidate_full_center.shape", candidate_full_center.shape)
        # print("ref_hypos_center.shape", ref_hypos_center.shape)
        exp_dist = np.exp([[
            -10 * self.center_dist(ref_hypos_center[i],
                                   candidate_full_center[j, ref_choices[i]])
            for i, _ in enumerate(ref_hypos_index)
        ] for j, _ in enumerate(candidate_hypos_index)])
        score = np.einsum("ij,j->i", exp_dist, ref_hypos_post)

        return list(
            np.array(candidate_hypos_index)[np.argsort(score)[-amount:]])

    def cluster_transition(self,
                           full_hypo_set: BaseSet | None = None,
                           **kwargs) -> List:
        """
        Make the transition
        """
        new_hypos: Set[int] = set([])
        full_hypos = set(range(self.length))
        if full_hypo_set is None:
            available_hypos = set(range(self.length))
        else:
            available_hypos = set(full_hypo_set)

        for amount, method in self.cluster_transition_strategy:

            new_part = method(amount, available_hypos, **kwargs)
            new_hypos = new_hypos.union(set(new_part))
            available_hypos = available_hypos.difference(new_part)

        return list(new_hypos)

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
