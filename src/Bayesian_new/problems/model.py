"""
Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
from .partitions import Partition, BasePartition


@dataclass(unsafe_hash=True)
class ModelParams:
    """
    Legacy
    """

    k: int  # index of partition method
    beta: float  # softness of partition


@dataclass
class ObservationType:
    """
    observation format
    """
    stimulus: tuple
    choices: tuple
    responses: tuple


class PartitionLikelihood(BaseLikelihood):
    """
    Likelihood in only partitions.
    """

    def __init__(self, space: BaseSet, partition: BasePartition):
        """Initialize

        space: the set of k's, must be included in the partition.
        """
        super().__init__(space)
        self.partition = partition
        # This may raise an exception if h_set is not a subset of
        # partition labels.
        self.h_indices = list(self.h_set)

    def get_likelihood(self,
                       observation,
                       beta: list | tuple | float = 1.,
                       use_cached_dist: bool = False,
                       normalized: bool = True,
                       **kwargs):
        """
        Get Likelihood, Base
        """

        ret = self.partition.calc_likelihood(self.h_indices, observation, beta,
                                             use_cached_dist, normalized,
                                             **kwargs)
        return ret


class SoftPartitionLikelihood(PartitionLikelihood):
    """
    Likelihood with (partition, beta) as hypotheses.
    """

    def __init__(self, space: BaseSet, partition: BasePartition,
                 beta_grid: list):
        """Initialize

        space: the set of k's, must be included in the partition.
        """
        super().__init__(space, partition)
        self.beta_grid = beta_grid

    def get_likelihood(self,
                       observation,
                       beta: list | tuple | float = 1.,
                       use_cached_dist: bool = False,
                       normalized: bool = True,
                       **kwargs) -> np.ndarray:
        """
        Get Likelihood, Base
        """

        ret = []
        for beta_ in self.beta_grid:
            ret += [
                self.partition.calc_likelihood(self.h_indices, observation,
                                               beta_, use_cached_dist,
                                               normalized, **kwargs)
            ]
        return np.concatenate(ret, axis=1)


class BaseModel:
    """
    Base Model
    """

    def __init__(self, config: Dict, **kwargs):
        self.config = config
        self.all_centers = None
        self.hypotheses_set = BaseSet([])
        self.observation_set = BaseSet([])

        condition = kwargs.get("condition", 1)
        ndims = 4
        ncats = 2 if condition == 1 else 4

        self.partition_model = kwargs.get("partition", Partition(ndims, ncats))
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        self.engine = BaseEngine(
            self.hypotheses_set, self.observation_set,
            BasePrior(self.hypotheses_set),
            PartitionLikelihood(self.hypotheses_set, self.partition_model))

    def set_hypotheses(self, h_set: Dict | Tuple | List):
        """
        Set hypotheses set
        """
        self.hypotheses_set = BaseSet(h_set)

    def refresh_engine(self, h_set, prior, likelihood):
        """
        Refresh engine with new set
        """

        self.hypotheses_set = h_set
        self.engine = BaseEngine(h_set, self.observation_set, prior,
                                 likelihood)

    def fit(self, data, **kwargs) -> Tuple[ModelParams, float, Dict, Dict]:
        """
        Parameters
        ----------
        data :


        Returns
        -------
        out :

        """
        raise NotImplementedError
        # return (data, 0., 0., {})


class SingleRationalModel(BaseModel):
    """
    Pure Rational
    """

    def fit(self, data, **kwargs) -> Tuple[ModelParams, float, Dict, Dict]:
        """
        Fit
        """
        all_hypo_params = {}
        all_hypo_ll = {}

        def _ll_per_hypo(beta, hypo=None):
            likelihood = self.partition_model.calc_likelihood_entry(
                hypo, data, beta[0], **kwargs)
            return np.sum(np.log(np.maximum(likelihood, 0)), axis=0)

        for hypo in self.hypotheses_set:
            result = minimize(lambda beta: -_ll_per_hypo(beta, hypo),
                              x0=[self.config["param_inits"]["beta"]],
                              bounds=[self.config["param_bounds"]["beta"]])
            beta_opt, ll_max = result.x[0], -result.fun

            all_hypo_params[hypo] = ModelParams(hypo, beta_opt)
            all_hypo_ll[hypo] = ll_max

        best_hypo = max(all_hypo_ll, key=all_hypo_ll.get)
        return (all_hypo_params[best_hypo], all_hypo_ll[best_hypo],
                all_hypo_params, all_hypo_ll)

    def fit_trial_by_trial(self, 
                           data: Tuple[np.ndarray, np.ndarray,np.ndarray],
                           limited_hypos_list: List[List[int]] = None, 
                           **kwargs):
        """
        Fit the model trial-by-trial to observe parameter evolution.

        Args:
            data (DataFrame): Data containing features, choices, and feedback.
            limited_hypos_list: 如果不为 None，则是一个长度为 n_trial 的列表，
                                每个元素都是一个假设子集（List[int]），
                                用于限制当前 trial 的候选假设。
                                若为 None，则使用原有的 '全集' self.hypotheses_set。

        Returns:
            List[Dict]: List of results for each trial step.
        """
        step_results = []
        n_trial = len(data[2])

        # 备份一下原假设集
        original_hypotheses_set = self.hypotheses_set
        original_prior = self.engine.prior
        original_likelihood = self.engine.likelihood

        for step in tqdm(range(n_trial, 0, -1)):

            trial_data = [x[:step] for x in data]

            # 若给定了 limited_hypos_list，就用子集，否则用原有全集
            if limited_hypos_list is not None:
                current_hypos = limited_hypos_list[step - 1]
                h_set = BaseSet(current_hypos)

                # 重新构造 prior & likelihood
                new_prior = BasePrior(h_set)
                new_likelihood = PartitionLikelihood(h_set, self.partition_model)

                # 刷新引擎
                self.refresh_engine(h_set, new_prior, new_likelihood)
            else:
                # 不动；直接使用 self.hypotheses_set
                pass

            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit(
                trial_data, use_cached_dist=(step != n_trial), **kwargs)

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            all_hypo_post = self.engine.infer_log(trial_data,
                                                  use_cached_dist=(step
                                                                   != n_trial),
                                                  beta=hypo_betas,
                                                  normalized=True)

            hypo_details = {}
            for i, hypo in enumerate(self.hypotheses_set.elements):
                hypo_details[hypo] = {
                    'beta_opt': all_hypo_params[hypo].beta,
                    'll_max': all_hypo_ll[hypo],
                    'post_max': all_hypo_post[i],
                    'is_best': hypo == best_params.k
                }

            step_results.append({
                'best_k': best_params.k,
                'best_beta': best_params.beta,
                'best_params': best_params,
                'best_log_likelihood': best_ll,
                'best_norm_posterior': np.max(all_hypo_post),
                'hypo_details': hypo_details
            })

        # 恢复原假设集
        if limited_hypos_list is not None:
            self.refresh_engine(
                original_hypotheses_set,
                original_prior,
                original_likelihood
            )

        return step_results[::-1]

    def oral_generate_hypos(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        top_k: int = 10,
        dist_tol: float = 1e-9
    ) -> List[List[int]]:
        """
        基于被试每个试次的口头汇报坐标 (oral_center) 与选择的类别 (choice)，
        为每个试次生成一个可能的假设子集（limited_hypos）。

        Args:
            data:
                - data[0]: oral_centers, shape = (n_trial, n_dims)
                  每个试次被试“口头汇报”的类别中心坐标
                - data[1]: choices, shape = (n_trial,)
                  每个试次被试选择的类别(1-index)
            top_k: 如果没有任何假设“完全符合” (距离=0) 时，
                   就选取最小距离的前 top_k 个假设索引。
            dist_tol: 判断距离是否等于 0 的容忍度（浮点运算误差范围）。

        Returns:
            limited_hypos_list: List[List[int]]，长度 = n_trial，
                                每个元素是该试次对应的一批假设索引。
        """

        oral_centers, choices = data
        n_trial = len(choices)
        n_dims = oral_centers.shape[1]  # 假设 oral_centers.shape = (n_trial, n_dims)

        # 假设全集指的是 partition 中所有可能的 hypothesis 索引
        # self.partition_model.prototypes_np.shape = [num_hypos, n_cats, n_dims]
        # 也可以用 self.hypotheses_set.elements，但大多数情况下
        # 这俩应该保持一致（整型序号从 0 到 num_hypos-1）。
        num_hypos = self.partition_model.prototypes_np.shape[0]
        all_hypos = range(num_hypos)

        limited_hypos_list = []

        for i in range(n_trial):
            # 当前试次选择了哪一类？(1-index) => (0-index)
            cat_idx = choices[i] - 1

            # 被试口头汇报的中心
            oral_center_i = oral_centers[i]  # shape=(n_dims,)

            # 计算该 oral_center_i 与每个假设 (h) 的对应类别 cat_idx 的“真中心”距离
            # partition_model.prototypes_np[h, cat_idx, :] 就是 hypo=h 对应的某一类别中心
            dist_list = []
            for h in all_hypos:
                true_center = self.partition_model.prototypes_np[h, 0, cat_idx, :]
                dist = np.linalg.norm(oral_center_i - true_center)
                dist_list.append((dist, h))

            # 1) 判断有没有 dist=0 (在 float 范围内接近 0)
            #    如果有，则只取这些“完全符合”的假设
            #    如果没有，则取最小距离的前 top_k 个。
            exact_matches = [h for (dist, h) in dist_list if dist <= dist_tol]

            if len(exact_matches) > 0:
                # 直接拿到全部“距离为0”的假设即可
                chosen_hypos = exact_matches
            else:
                # 没有完全匹配，则按距离排序，取前 top_k
                dist_list.sort(key=lambda x: x[0])  # 升序
                chosen_hypos = [h for (_, h) in dist_list[:top_k]]

            limited_hypos_list.append(chosen_hypos)

        return limited_hypos_list


    def predict_choice(self, params: ModelParams, x: np.ndarray,
                       condition: int):
        """
        predict choice
        """
