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
from .model import ModelParams, BaseModel
from .base_problem import softmax, cdist, euc_dist


@dataclass(unsafe_hash=True)
class ForgetModelParams(ModelParams):
    """扩展参数类，添加遗忘参数"""
    gamma: float = 1.0  # 遗忘率参数
    w0: float = 0.1     # 基础记忆权重


class ForgetModel(BaseModel):
    """
    Add forgetting mechanism
    """
    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, **kwargs)

        # 初始化参数搜索空间
        self.gamma_values = np.arange(0.1, 1.1, 0.1)
        self.w0_values = np.arange(0.1, 1.1, 0.1)
        
        # 初始化记忆权重缓存
        self.memory_weights_cache = {}

    def _get_memory_weights(self, n_trials: int, gamma: float, w0: float) -> np.ndarray:
        """预计算记忆衰减权重"""
        cache_key = (n_trials, gamma, w0)
        if cache_key not in self.memory_weights_cache:
            decay = gamma ** np.arange(n_trials-1, -1, -1)
            weights = w0 + (1 - w0) * decay
            self.memory_weights_cache[cache_key] = weights / np.sum(weights)
        return self.memory_weights_cache[cache_key]

    def get_weighted_log_likelihood(self, hypo: int, data: tuple, beta: float, 
                              gamma: float, w0: float, **kwargs) -> float:
        """计算加权后的似然值"""
        # 获取基础似然
        base_likelihood = self.partition_model.calc_likelihood_entry(
            hypo, data, beta, **kwargs
        )
        
        # 应用记忆衰减权重
        n_trials = len(data[2])
        memory_weights = self._get_memory_weights(n_trials, gamma, w0)
        weighted_log_likelihood = memory_weights * np.log(base_likelihood)
        return np.sum(weighted_log_likelihood)

    def fit_with_given_params(self, data: tuple, gamma: float, w0: float, **kwargs) -> Tuple[ForgetModelParams, float, Dict, Dict]:
        """
        Fit
        """
        all_hypo_params = {}
        all_hypo_ll = {}

        for hypo in self.hypotheses_set.elements:
            result = minimize(lambda beta: -self.get_weighted_log_likelihood(hypo, data, beta, gamma, w0, **kwargs),
                              x0=[self.config["param_inits"]["beta"]],
                              bounds=[self.config["param_bounds"]["beta"]])
            beta_opt, ll_max = result.x[0], -result.fun
            
            all_hypo_params[hypo] = ForgetModelParams(hypo, beta_opt, gamma=gamma, w0=w0)
            all_hypo_ll[hypo] = ll_max

        best_hypo = max(all_hypo_ll, key=all_hypo_ll.get)
        return (all_hypo_params[best_hypo], all_hypo_ll[best_hypo], 
                all_hypo_params, all_hypo_ll)

    def fit_trial_by_trial(self, data: Tuple[np.ndarray, np.ndarray,
                                             np.ndarray], gamma, w0):
        step_results = []
        nTrial = len(data[2])
        
        for step in tqdm(range(nTrial), 0, -1):
            trial_data = [x[:step] for x in data]
            best_params, best_ll, all_hypo_params, all_hypo_ll = self.fit_with_given_params(
                trial_data, gamma, w0, use_cached_dist=(step != nTrial))

            all_hypo_post = self.engine.infer_log(
                trial_data,
                use_cached_dist=(step != nTrial),
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

        return step_results[::-1]


    def error_function(self, data_with_cat: tuple, step_results: list, window_size=16) -> float:
        """
        计算滑动窗口预测误差
        输入数据需包含category信息：(stimuli, choices, responses, categories)
        """
        # 解包带类别信息的数据
        stimuli, choices, responses, categories = data_with_cat
        
        # 计算每个试次的预测准确率
        predicted_acc = []
        for i, res in enumerate(step_results):
            params = res['best_params']
            
            # 获取当前试次的条件信息
            condition = 1 if len(np.unique(categories[:i+1])) <= 2 else 2
            
            # 计算类别中心
            centers = self.partition_model.prototypes_np[params.k]
            
            # 计算当前刺激的预测概率
            distances = euc_dist(centers, np.array(stimuli[:i+1]))
            probs = softmax(np.min(distances, axis=0), -params.beta)
            
            # 处理类别映射
            true_cats = categories[:i+1].copy()
            if condition == 1:
                true_cats = np.where(np.isin(true_cats, [1,2]), 1, 2)
            
            # 记录正确类别的预测概率
            predicted_acc.append(probs[true_cats[-1]-1])
        
        # 转换为numpy数组
        pred_acc = np.array(predicted_acc)
        true_acc = (np.array(responses) == 1).astype(float)
        
        # 滑动窗口误差计算
        errors = []
        for start in range(len(pred_acc) - window_size + 1):
            window_pred = pred_acc[start:start+window_size]
            window_true = true_acc[start:start+window_size]
            errors.append(np.abs(np.mean(window_pred) - np.mean(window_true)))
        
        return np.mean(errors)
        
        
    def optimize_params(self, data_with_cat: tuple) -> Tuple[ForgetModelParams, list]:
        """二维网格搜索优化gamma和w0"""
        best_error = float('inf')
        best_params = None
        best_step_results = None
        
        # 解包数据（包含category）
        s_data = data_with_cat[:3]  # (stimuli, choices, responses)
        
        # 网格搜索
        for gamma in tqdm(self.gamma_values, desc="Gamma"):
            for w0 in self.w0_values:
                # 逐试次拟合
                step_results = self.fit_trial_by_trial(s_data, gamma, w0)
                
                # 计算误差（传入完整数据含category）
                error = self.error_function(data_with_cat, step_results)
                
                # 更新最优结果
                if error < best_error:
                    best_error = error
                    best_params = (gamma, w0)
                    best_step_results = step_results
        
        return best_params, best_step_results

    def fit(self, data):
        step_results = []
        best_params = []

        # 使用网格搜索优化gamma
        (best_gamma, best_w0), step_results_for_block = self.optimize_params(data)
        best_params.append((best_gamma, best_w0))
        for result in step_results_for_block:
            step_results.append(result)
        
        return step_results, best_params