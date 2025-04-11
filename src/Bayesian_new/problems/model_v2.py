"""
Base Model
"""
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from .base_problem import (BaseSet, BaseEngine, BaseLikelihood, BasePrior)
from .partitions import Partition, BasePartition

# [TODO] MOVE to CONFIG
HYPO_CLUSTER_PROTOTYPE_AMOUNT = 1


@dataclass(unsafe_hash=True)
class BaseModelParams:
    """
    A data class that holds the base model parameters.

    Attributes
    ----------
    k : int
        The index of the partition method.
    beta : float
        The softness of the partition.
    """
    k: int
    beta: float


@dataclass
class ObservationType:
    """
    A data class describing the format of an observation.

    Attributes
    ----------
    stimulus : tuple
        Stimulus data (e.g., visual inputs).
    choices : tuple
        Choices made in response to the stimulus.
    responses : tuple
        Response correctness or other feedback metrics.
    categories : tuple
        The true category of the stimulus.
    """
    stimulus: tuple
    choices: tuple
    responses: tuple
    categories: tuple


class PartitionLikelihood(BaseLikelihood):
    """
    Likelihood for partitions only.
    """

    def __init__(self, space: BaseSet, partition: BasePartition):
        """
        Initialize PartitionLikelihood.

        Parameters
        ----------
        space : BaseSet
            The set of hypotheses indices (k's).
        partition : BasePartition
            The partitioning object that calculates likelihoods.
        """
        super().__init__(space)
        self.partition = partition
        # This may raise an exception if h_set is not a subset of partition labels.
        self.h_indices = list(self.h_set)

    def get_likelihood(self,
                       observation,
                       beta: list | tuple | float = 1.,
                       use_cached_dist: bool = False,
                       normalized: bool = True,
                       **kwargs) -> np.ndarray:
        """
        Compute the likelihood of an observation given the current partition.

        Parameters
        ----------
        observation : any
            Observation data.
        beta : float or list/tuple
            Softness parameter.
        use_cached_dist : bool
            Whether to use cached distances for speed-up.
        normalized : bool
            Whether to normalize the result.

        Returns
        -------
        np.ndarray
            An array of likelihood values.
        """
        likelihood_values = self.partition.calc_likelihood(
            self.h_indices, observation, beta, use_cached_dist, normalized,
            **kwargs)
        return likelihood_values


class SoftPartitionLikelihood(PartitionLikelihood):
    """
    Likelihood using (partition, beta) as hypotheses.
    """

    def __init__(self, space: BaseSet, partition: BasePartition,
                 beta_grid: list):
        """
        Initialize SoftPartitionLikelihood.

        Parameters
        ----------
        space : BaseSet
            The set of hypotheses indices (k's).
        partition : BasePartition
            The partitioning object that calculates likelihoods.
        beta_grid : list
            A list of beta values to be considered.
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
        Compute the likelihood of an observation over a grid of beta values.

        Parameters
        ----------
        observation : any
            Observation data.
        beta : float or list/tuple
            Softness parameter (not used directly, since we use beta_grid).
        use_cached_dist : bool
            Whether to use cached distances for speed-up.
        normalized : bool
            Whether to normalize the result.

        Returns
        -------
        np.ndarray
            A concatenated array of likelihood values for each beta in beta_grid.
        """
        likelihood_collection = []
        for beta_value in self.beta_grid:
            likelihood_val = self.partition.calc_likelihood(
                self.h_indices, observation, beta_value, use_cached_dist,
                normalized, **kwargs)
            likelihood_collection.append(likelihood_val)
        return np.concatenate(likelihood_collection, axis=1)


class BaseModel:
    """
    Base Model class that initializes a default partition model,
    hypotheses set, and inference engine.
    """

    def __init__(self, config: Dict, **kwargs):
        """
        Initialize BaseModel with a given configuration.

        Parameters
        ----------
        config : Dict
            A dictionary containing initial settings (e.g., parameter bounds).
        **kwargs : dict
            Additional keyword arguments.
        """
        self.config = config
        self.all_centers = None
        self.hypotheses_set = BaseSet([])
        self.observation_set = BaseSet([])

        condition = kwargs.get("condition", 1)
        n_dims = 4
        n_cats = 2 if condition == 1 else 4

        self.partition_model = kwargs.get("partition",
                                          Partition(n_dims, n_cats))
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        self.full_likelihood = PartitionLikelihood(
            BaseSet(list(range(self.partition_model.length))),
            self.partition_model)
        self.engine = BaseEngine(
            self.hypotheses_set, self.observation_set,
            BasePrior(self.hypotheses_set),
            PartitionLikelihood(self.hypotheses_set, self.partition_model))

    def set_hypotheses(self, hypothesis_collection: Dict | Tuple | List):
        """
        Set the hypotheses set manually.

        Parameters
        ----------
        hypothesis_collection : Dict or Tuple or List
            A collection of hypotheses indices.
        """
        self.hypotheses_set = BaseSet(hypothesis_collection)

    def refresh_engine(self, new_hypotheses_set, new_prior, new_likelihood):
        """
        Re-initialize the engine with a new set of hypotheses, prior, and likelihood.

        Parameters
        ----------
        new_hypotheses_set : BaseSet
            New set of hypotheses.
        new_prior : BasePrior
            New prior object.
        new_likelihood : BaseLikelihood
            New likelihood object.
        """
        self.hypotheses_set = new_hypotheses_set
        self.engine = BaseEngine(new_hypotheses_set, self.observation_set,
                                 new_prior, new_likelihood)

    def fit(self, data, **kwargs) -> Tuple[BaseModelParams, float, Dict, Dict]:
        """
        Fit the model to data. NotImplementedError by default.

        Parameters
        ----------
        data : any
            Training data.

        Returns
        -------
        (BaseModelParams, float, Dict, Dict)
            Stub return to be overridden in subclasses.
        """
        raise NotImplementedError


class SingleRationalModel(BaseModel):
    """
    A model that fits each hypothesis with a rational approach
    and returns the best-fitting one.
    """

    def precompute_distances(self, stimulus: np.ndarray):
        """
        Precompute all distance.

        Parameters
        ----------
        stimulus : np.ndarray
        """
        if hasattr(self.partition_model, "precompute_all_distances"):
            self.partition_model.precompute_all_distances(stimulus)

    def fit_single_step(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        **kwargs) -> Tuple[BaseModelParams, float, Dict, Dict]:
        """
        Fit the rational model by optimizing beta for each hypothesis.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing (stimulus, choices, responses).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        best_params : BaseModelParams
            Parameters (hypothesis, beta) that achieve the highest likelihood.
        best_ll : float
            The maximum log-likelihood value.
        all_hypo_params : Dict
            Mapping from hypothesis to BaseModelParams.
        all_hypo_ll : Dict
            Mapping from hypothesis to its best log-likelihood.
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

            all_hypo_params[hypo] = BaseModelParams(hypo, beta_opt)
            all_hypo_ll[hypo] = ll_max

        best_hypo_idx = max(all_hypo_ll, key=all_hypo_ll.get)

        return (all_hypo_params[best_hypo_idx], all_hypo_ll[best_hypo_idx],
                all_hypo_params, all_hypo_ll)

    def fit_step_by_step(self,
                         data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         dynamic_limit: bool = False,
                         **kwargs) -> List[Dict]:
        """
        Fit the model step-by-step to observe how parameters evolve.

        Parameters
        ----------
        data : Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing (stimulus, choices, responses).
        dynamic_limit : bool
            是否启用动态限缩假设集的机制 (True / False)。
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        List[Dict]
            A list of dictionaries containing the fitting results for each
            step, such as the best hypothesis, beta, log-likelihood, and posterior.
        """

        stimulus, _, responses = data
        n_trials = len(responses)

        # Precompute all distances
        self.partition_model.precompute_all_distances(stimulus)

        # Backup the original hypotheses set
        original_hypotheses_set = self.hypotheses_set
        original_prior = self.engine.prior
        original_likelihood = self.engine.likelihood

        # 全集 (所有可能的 hypo) - 假设 BaseSet 里 .elements 就是这些
        full_hypo_list = list(self.hypotheses_set.elements)

        step_results = []

        for step_idx in tqdm(range(1, n_trials + 1)):
            selected_data = [x[:step_idx] for x in data]

            (best_params, best_ll, all_hypo_params,
             all_hypo_ll) = self.fit_single_step(selected_data,
                                                 use_cached_dist=True,
                                                 **kwargs)

            hypo_betas = [
                all_hypo_params[hypo].beta
                for hypo in self.hypotheses_set.elements
            ]

            all_hypo_post = self.engine.infer_log(selected_data,
                                                  use_cached_dist=True,
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

            if kwargs.get("cluster", False):
                # [TODO] WIP: test it in real environment
                if step_idx < n_trials:
                    cur_post_dict = {
                        h: det["post_max"]
                        for h, det in hypo_details.items()
                    }
                    next_hypos = self.partition_model.cluster_transition(
                        stimulus=data[0][step_idx],
                        posterior=cur_post_dict,
                        proto_hypo_amount=kwargs.get(
                            "cluster_prototype_amount",
                            HYPO_CLUSTER_PROTOTYPE_AMOUNT),
                        **kwargs.get("cluster_kwargs", {}))
                    new_hypotheses_set = BaseSet(next_hypos)
                    new_prior = BasePrior(new_hypotheses_set)
                    new_likelihood = PartitionLikelihood(
                        new_hypotheses_set, self.partition_model)
                    self.refresh_engine(new_hypotheses_set, new_prior,
                                        new_likelihood)

            # 如果启用了 dynamic_limit，就基于后验分布来动态筛选下一步的假设子集
            if dynamic_limit:

                # new_hypo_subset = self.model_generate_hypos(all_hypo_post,
                #                                             top_k=10)

                # new_hypo_subset = self.model_hypos_transition(all_hypo_post, selected_data,
                #                                               # customizable.
                #                                               rule="top-likelihood"
                #                                               )

                # 如果还没到最后一试次，就生成下一步要使用的 10 个假设，然后 refresh_engine
                if step_idx < n_trials:
                    # 把当前所有 hypo 的后验拼成一个 dict {hypo: posterior}, 以及 {hypo: BaseModelParams}
                    cur_post_dict = {}
                    cur_param_dict = {}
                    for h, det in hypo_details.items():
                        cur_post_dict[h] = det['post_max']
                        # 这个 beta_opt 就是上一步对单一 hypo 拟合得到的最优 beta
                        cur_param_dict[h] = BaseModelParams(
                            k=h, beta=det['beta_opt'])

                    next_hypos = self.select_next_hypotheses(
                        step_idx=step_idx,
                        data=data,
                        all_hypos=full_hypo_list,
                        prev_post=cur_post_dict,
                        prev_params=cur_param_dict,
                        use_cached_dist=True,
                        **kwargs)

                # 重新刷新引擎
                new_hypotheses_set = BaseSet(next_hypos)
                new_prior = BasePrior(new_hypotheses_set)
                new_likelihood = PartitionLikelihood(new_hypotheses_set,
                                                     self.partition_model)
                self.refresh_engine(new_hypotheses_set, new_prior,
                                    new_likelihood)
            else:
                # 如果不做限制，就保持 hypotheses_set 不变，不需刷新
                pass

        self.refresh_engine(original_hypotheses_set, original_prior,
                            original_likelihood)

        return step_results

    def select_next_hypotheses(self,
                               step_idx: int,
                               data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                               all_hypos: List[int],
                               prev_post: Dict[int, float],
                               prev_params: Dict[int, BaseModelParams],
                               top_posterior: int = 5,
                               top_likelihood: int = 3,
                               random_pick: int = 2,
                               use_cached_dist: bool = False,
                               **kwargs) -> List[int]:
        """
        依据上一个 step 的结果，挑选下一步要使用的 10 个假设：
          1) posterior 最高的 5 个
          2) sum-likelihood (在最近 5 个 trial) 最高的 3 个
          3) 在剩余中随机选 2 个

        Parameters
        ----------
        step_idx : int
            当前 step 的索引（1-based）。
        data : (stimulus, choices, responses)
        all_hypos : List[int]
            当前可供挑选的“全集”假设列表。
        prev_post : Dict[int, float]
            上一 step 每个假设对应的后验概率。键是 hypo，值是后验。
        prev_params : Dict[int, BaseModelParams]
            上一 step 每个假设的最优参数 (k, beta)。键是 hypo，值是 BaseModelParams。
        top_posterior : int
            选取后验最高的多少个（默认为 5）。
        top_likelihood : int
            选取 sum-likelihood 最高的多少个（默认为 3）。
        random_pick : int
            随机再挑选多少个（默认为 2）。
        use_cached_dist : bool
            是否使用缓存距离以加速计算。
        **kwargs : 
            其余会传给 `calc_likelihood` (比如一些自定义 amnesia_mechanism 等)。

        Returns
        -------
        List[int]
            下一个 step 要使用的假设列表（共 10 个，如数量不足则取实际数量）。
        """

        # ============ (1) 从 prev_post 中选 posterior 最高的 top_posterior ============ #
        # prev_post 是一个 {hypo: posterior_value} dict
        sorted_by_post = sorted(prev_post.items(),
                                key=lambda x: x[1],
                                reverse=True)
        top_posterior_hypos = [h for (h, _) in sorted_by_post[:top_posterior]]

        # ============ (2) 在剩余假设里，根据最近 5 个 trial 的 sum-likelihood 选 top_likelihood ============ #
        # 剩余假设 = “全集” - “posterior最高5个”
        remain_after_5 = list(set(all_hypos) - set(top_posterior_hypos))

        # 最近 5 个 trial 的数据索引
        start_idx = max(0, step_idx - 5)  # 若 step_idx<5 则自动取 0
        # slice 数据
        partial_data = [arr[start_idx:step_idx] for arr in data]
        # partial_data 形如 [stimulus_slice, choices_slice, responses_slice]

        # 计算每个剩余假设在这段 data 上的 sum-likelihood
        # 注意：为了计算 likelihood，需要用到上一轮该 hypo 的 best beta
        beta_list = []
        for h in remain_after_5:
            if h in prev_params:
                beta_list.append(prev_params[h].beta)
            else:
                # 如果某些 hypo 不在上一步出现过，可能不会有 prev_params[h]
                # 这里给一个默认值，比如 1.0，或你项目中的默认 init
                beta_list.append(1.0)

        # 调用 partition_model.calc_likelihood 或 engine.likelihood.get_likelihood
        # normalized=False 以便我们直接拿到未归一化的 likelihood，然后对 trial 做 sum
        # 这里假设你在 partition_model 上有 calc_likelihood 方法
        # (或者用 self.engine.likelihood.get_likelihood 也可以，但要小心参数匹配)
        likelihood_mat = self.partition_model.calc_likelihood(
            hypos=remain_after_5,
            data=partial_data,
            beta=beta_list,
            use_cached_dist=use_cached_dist,
            normalized=False,
            **kwargs)
        # likelihood_mat shape = [num_trials_in_window, len(remain_after_5)]
        # 对试次维度做 sum，得到每个 hypo 的 sum-likelihood
        sum_lik = likelihood_mat.sum(axis=0)  # shape = [len(remain_after_5)]

        # 取 sum-likelihood 最高的 top_likelihood
        idx_sorted = np.argsort(-sum_lik)  # 从大到小排序
        top_lik_indices = idx_sorted[:top_likelihood]
        top_lik_hypos = [remain_after_5[i] for i in top_lik_indices]

        # ============ (3) 在剩余假设里随机挑选 random_pick 个 ============ #
        # 剩余 = “全集” - “(posterior前5) ∪ (likelihood前3)”
        remain_after_8 = list(
            set(all_hypos) - set(top_posterior_hypos) - set(top_lik_hypos))
        # 随机取 random_pick 个，注意万一剩余数量 < random_pick，需要取 min(...)
        num_to_pick = min(random_pick, len(remain_after_8))
        rnd_hypos = np.random.choice(remain_after_8,
                                     size=num_to_pick,
                                     replace=False)

        # ============ (4) 整合得到下一个 step 的 10 个假设 ============ #
        next_hypos = top_posterior_hypos + top_lik_hypos + list(rnd_hypos)
        # 如果担心去重（比如万一 top_lik_hypos 里出现了和 top_posterior_hypos 重叠的情况），可以再去一下重
        next_hypos = list(set(next_hypos))

        # 最终最多 10 个
        # 若去重后恰好小于 10 也没关系，这里就保留实际数量
        # 如果你想严格限制成 10 个，可能需要再补一些随机，但一般来说不会有那么巧重叠
        return next_hypos

    # def model_hypos_transition(self,
    #                            all_hypo_post: np.ndarray,
    #                            data: np.ndarray,
    #                            hypo_spec: Dict = {
    #                                "remain": 7,
    #                                "rule": 2,
    #                                "random": 1
    #                            },
    #                            rule: Callable | str = "top-likelihood",
    #                            **kwargs) -> List[int]:
    #     """
    #     Make a k-cluster transition.

    #     Parameters
    #     ----------
    #     hypo_spec: Dict
    #         hypothesis-spectrum.
    #             "remain": the remaining hypos
    #             "rule": allocate rule-selected likelihood amounts
    #             "random": pure random.
    #     """
    #     remain_amt = hypo_spec["remain"]
    #     rule_amt = hypo_spec["rule"]
    #     random_amt = hypo_spec["random"]

    #     # Retain top posterior hypotheses
    #     all_hypos = deepcopy(self.hypotheses_set.elements)
    #     sorted_pairs = sorted(zip(all_hypo_post, all_hypos),
    #                           key=lambda x: x[0],
    #                           reverse=True)
    #     new_hypo_list = [sorted_pairs[i][0] for i in range(remain_amt)]
    #     all_hypos = set(all_hypos) - set(new_hypo_list)

    #     # Select hypotheses based on the rule
    #     match rule:
    #         case "top-likelihood":
    #             obs = [d[-1:] for d in data]
    #             likelihood = self.full_likelihood.get_likelihood(obs, beta=2)
    #             # Ensure valid indices
    #             valid_hypos = [h for h in all_hypos if h < likelihood.shape[1]]
    #             labeled = sorted(list(
    #                 zip(valid_hypos, likelihood[:, valid_hypos].mean(axis=0))),
    #                             key=lambda x: x[1],
    #                             reverse=True)
    #             rule_hypos = [x[0] for x in labeled[:rule_amt]]
    #             all_hypos = all_hypos - set(rule_hypos)
    #             new_hypo_list += rule_hypos

    #         case "top-5-likelihood":
    #             obs = [d[-1:] for d in data]
    #             likelihood = np.sum(-np.log(self.full_likelihood.get_likelihood(obs, beta=2)),
    #                                 axis=-1)
    #             # Ensure valid indices
    #             valid_hypos = [h for h in all_hypos if h < likelihood.shape[0]]
    #             labeled = sorted(list(
    #                 zip(valid_hypos, likelihood[valid_hypos])),
    #                             key=lambda x: x[1],
    #                             reverse=True)
    #             rule_hypos = [x[0] for x in labeled[:rule_amt]]
    #             all_hypos = all_hypos - set(rule_hypos)
    #             new_hypo_list += rule_hypos

    #         case callable():
    #             ref = rule(data)
    #             labeled = sorted(list(zip(all_hypos, ref)),
    #                              key=lambda x:x[1],
    #                              reverse=True)
    #             rule_hypos = [x[0] for x in labeled[:rule_amt]]
    #             all_hypos = all_hypos - set(rule_hypos)
    #             new_hypo_list += rule_hypos

    #         case _:
    #             pass

    #     # Randomly select hypotheses
    #     if len(all_hypos) < random_amt:
    #         random_amt = len(all_hypos)
    #     new_hypo_list += list(
    #         np.random.choice(list(all_hypos), size=random_amt, replace=False))

    #     return new_hypo_list

    def model_generate_hypos(self,
                             all_hypo_post: np.ndarray,
                             top_k: int = 10) -> List[int]:
        """
        基于上一试次的后验分布，选取前 top_k 个假设，返回其索引列表。

        Parameters
        ----------
        all_hypo_post : np.ndarray
            当前假设集合下，每个假设的后验概率（或已归一化过的 posterior）。
        top_k : int
            要保留的假设数量。

        Returns
        -------
        List[int]
            根据后验排序后取前 top_k 个假设对应的索引。
        """
        # 获取当前引擎里的假设元素
        current_hypos = self.hypotheses_set.elements

        # (posterior, hypothesis) 组合后按 posterior 从大到小排序
        sorted_pairs = sorted(zip(all_hypo_post, current_hypos),
                              key=lambda x: x[0],
                              reverse=True)

        # 取前 top_k 个假设
        new_hypo_subset = [h for _, h in sorted_pairs[:top_k]]

        return new_hypo_subset

    def predict_choice(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray], step_results: list,
                       use_cached_dist, window_size) -> Dict[str, np.ndarray]:
        """
        Predict choice trial by trial using fitted parameters and hypotheses.

        Parameters
        ----------
        data : tuple
            A tuple containing (stimulus, choices, responses, categories).
        step_results : list
            Output of fit_trial_by_trial, containing fitted results for each trial.
        use_cached_dist : bool
            Whether to use cached distances to speed up calculations.
        window_size : int
            Size of the sliding window for computing average accuracy.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing arrays of true accuracy, predicted accuracy, 
            and their sliding averages.
        """
        stimulus, choices, responses, categories = data
        n_trials = len(responses)

        true_acc = (np.array(responses) == 1).astype(float)
        pred_acc = np.full(n_trials, np.nan, dtype=float)

        for trial_idx in range(1, n_trials):
            trial_data = ([stimulus[trial_idx]], [choices[trial_idx]],
                          [responses[trial_idx]], [categories[trial_idx]])

            # Extract the posterior probabilities for each hypothesis at last trial
            hypo_details = step_results[trial_idx - 1]['hypo_details']
            post_max = [
                hypo_details[k]['post_max'] for k in hypo_details.keys()
            ]

            # Compute the weighted probability of being correct
            weighted_p_true = 0
            for k, post in zip(hypo_details.keys(), post_max):
                p_true = self.partition_model.calc_trueprob_entry(
                    k,
                    trial_data,
                    hypo_details[k]['beta_opt'],
                    use_cached_dist=use_cached_dist,
                    indices=[trial_idx])
                weighted_p_true += post * p_true

            pred_acc[trial_idx] = weighted_p_true

        # Compute sliding averages using a sliding window
        sliding_true_acc = []
        sliding_pred_acc = []
        sliding_pred_acc_std = []

        for start_idx in range(1, n_trials - window_size +
                               2):  # Start from index 1
            end_idx = start_idx + window_size
            sliding_true_acc.append(np.mean(true_acc[start_idx:end_idx]))
            pred_window = pred_acc[start_idx:end_idx]
            sliding_pred_acc.append(np.mean(pred_window))
            sliding_pred_acc_std.append(
                np.sqrt(np.sum(pred_window * (1 - pred_window))) / window_size)

        predict_results = {
            'true_acc': true_acc,
            'pred_acc': pred_acc,
            'sliding_true_acc': sliding_true_acc,
            'sliding_pred_acc': sliding_pred_acc,
            'sliding_pred_acc_std': sliding_pred_acc_std
        }

        return predict_results
