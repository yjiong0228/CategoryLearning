"""
基于原型的类别中心点生成
"""
from abc import ABC
from typing import List, Tuple, Dict
from copy import deepcopy
import itertools
from itertools import product
# import pandas as pd
import numpy as np
from .base_problem import softmax, cdist, euc_dist, two_factor_decay
from ..inference_engine import BaseDistribution, BaseLikelihood


def amnesia_mechanism(func):
    """
        Wrapper of the amnesia_mechanism
        if there is "gamma" or "amnesia" in kwargs,
        it transforms likelihood(*) into amnesia_likelihood(*)
        """

    def wrapped(self,
                hypo: int,
                data: list | tuple,
                beta: float,
                use_cached_dist: bool = False,
                **kwargs) -> np.ndarray:

        nonlocal func
        prob = func(self, hypo, data, beta, use_cached_dist, **kwargs)

        # 1) 如果 kwargs 里有"gamma" (以及"w0") => 用 two_factor_decay
        if "gamma" in kwargs:
            coeff = two_factor_decay(list(data), kwargs["gamma"], kwargs["w0"])
            log_prob = np.log(prob)
            log_prob *= coeff.reshape(list(prob.shape)[:-1] + [-1])
            return np.exp(log_prob)
        # 2) 如果 kwargs 中有 'amnesia' => 调用用户自定义函数
        if (amnesia := kwargs.get("amnesia", False)):
            coeff = amnesia(data, **kwargs.get("amnesia_kwargs", {}))
            log_prob = np.log(prob)
            log_prob *= coeff.reshape(list(prob.shape)[:-1] + [-1])
            return np.exp(log_prob)
        # 3) 如果 kwargs 中有 'adaptive_amnesia' => 实现“试次个性化”逻辑
        if "adaptive_amnesia" in kwargs:
            adapt_info = kwargs["adaptive_amnesia"]

            # 如果 adapt_info 是可调用:
            if callable(adapt_info):
                # adapt_info(data) => shape=[n_trials]
                coeff = adapt_info(data, **kwargs.get("amnesia_kwargs", {}))
            elif isinstance(adapt_info, np.ndarray):
                coeff = adapt_info
            else:
                raise ValueError(
                    "adaptive_amnesia must be callable or an array")

            log_prob = np.log(prob)
            log_prob *= coeff.reshape(list(prob.shape)[:-1] + [-1])
            return np.exp(log_prob)

        return prob

    return wrapped


class BasePartition(ABC):
    EPS = 1e-12
    """
    Base Partition
    """

    def __init__(self, n_dims: int, n_cats: int, n_protos: int = 1, **kwargs):
        """Initialize"""
        self.n_dims = n_dims
        self.n_cats = n_cats
        self.n_protos = n_protos
        coeffs = [tuple(x.tolist()) for x in np.eye(self.n_dims, dtype=int)]
        # the lower and upper boundaries of the cube, such as `x_0=0`, `x_2=1`.
        self.base_spaces = [[(x, 0) for x in coeffs], [(x, 1) for x in coeffs]]
        self.splits = self.get_all_splits()
        self.centers = self.get_centers()
        self.prototypes = self.get_prototypes()

        self.prototypes_np = np.array([[[t for i, t in sorted(x.items())]
                                        for x in p[1:]]
                                       for p in self.prototypes])

        self.cached_dist: Dict[int, np.ndarray] = {}
        # self.labels = [p[0] for p in self.prototypes]
        # self.inv_labels = dict((l, i) for i, l in enumerate(self.labels))

    @property
    def length(self):
        """
        Property: length
        """

        return len(self.prototypes)

    def get_all_splits(self):
        """
        Abstract
        """
        raise NotImplementedError

    def get_centers(self):
        """
        Abstract
        """
        raise NotImplementedError

    def get_prototypes(self):
        """
        Parameters
        ----------
        n_protos: int, number of prototypes per category. Could be calculated
                  via Wasserstein Discretization. When `n_protos == 1`, it is
                  just the barycenter of each category.
        """
        if self.n_protos == 1:
            return self.get_centers()
        raise NotImplementedError(
            "get_prototypes: Method case not implemented.")

    def precompute_all_distances(self, stimuli: np.ndarray):
        """
        提前把所有 trial 的距离缓存到 self.cached_dist[hypo] 中，
        这样后面做正序或倒序都不再重复算距离。

        stimuli: shape = [n_trials, n_dims]
        """
        n_trials = stimuli.shape[0]
        for hypo in range(self.length):
            partition = self.prototypes_np[
                hypo]  # shape = [n_protos, n_cats, n_dims]
            distances = euc_dist(
                partition, stimuli)  # shape = [n_protos, n_cats, n_trials]
            typical_distances = np.min(distances,
                                       axis=0)  # shape = [n_cats, n_trials]
            self.cached_dist[hypo] = typical_distances

    def calc_likelihood(self,
                        hypos: List[int] | Tuple[int],
                        data: list | tuple,
                        beta: list | tuple | float = 1.,
                        use_cached_dist: bool = False,
                        normalized: bool = True,
                        **kwargs) -> np.ndarray:  # BaseLikelihood:
        """
        Calculate likelihood.

        use_cached_dist: bool. This object caches the most recent distance
                         to closest prototype. If True, use the cached dist.
                         MAKE SURE you know the dist you calculate last time
                         is suitable to use before setting
                         `use_cached_dist = True`.
        """
        match beta:
            case float() as float_b:
                beta = [float_b] * len(hypos)
            case _:
                pass
        ret = np.zeros([len(data[2]), len(hypos)], dtype=float)

        # 如果 beta 是标量，将其转换为与 hypos 长度一致的数组
        if isinstance(beta, (int, float)):
            beta = [beta] * len(hypos)

        for j, h in enumerate(hypos):
            ret[:, j] = self.calc_likelihood_entry(h, data, beta[j],
                                                   use_cached_dist, **kwargs)
        if normalized:
            return ret / np.sum(ret, axis=1, keepdims=True)
        return ret

    def calc_likelihood_base(self,
                             hypo: int,
                             data: list | tuple,
                             beta: float,
                             use_cached_dist: bool = False,
                             **kwargs) -> np.ndarray:
        """
        计算给定 hypo 的 "raw" prob(选到该类别) (shape=[n_cats, n_trials])。

        Parameters
        ----------
        data: stimulus, choices, responses

        read partition (hypo) first, then calculate class probabilities
        over each classes.

        USE minimal distances between `data.stimulus` and `prototypes`
        (if there are more than one prototypes else just barycenter)

        "indices": None | list | np.ndarray.
                   retrieving distances cache on indices
        """
        stimulus, _ = data[:2]  # shape = [n_trials, n_dims]
        n_trials = len(stimulus)
        indices = kwargs.get("indices", None)

        if use_cached_dist and (hypo in self.cached_dist):
            typical_distances = (self.cached_dist[hypo][:, :n_trials]
                                 if indices is None else
                                 self.cached_dist[hypo][:, indices])
        else:
            partition = self.prototypes_np[
                hypo]  # shape = [n_protos, n_cats, n_dims]
            distances = euc_dist(
                partition,
                np.array(stimulus))  # shape = [n_protos, n_cats, n_trials]
            typical_distances = np.min(distances,
                                       axis=0)  # shape = [n_cats, n_trials]
            self.cached_dist[hypo] = typical_distances

        prob = softmax(typical_distances, -beta,
                       axis=0)  # shape = [n_cats, n_trials]

        return prob

    def calc_likelihood_entry(self,
                              hypo: int,
                              data: list | tuple,
                              beta: float,
                              use_cached_dist: bool = False,
                              **kwargs) -> np.ndarray:
        """
        核心函数, 被 amnesia_mechanism 包装，
        内部先用 calc_likelihood_base 得到 prob (shape=[n_trials]),
        然后在 wrapper 中根据kwargs进行遗忘衰减/试次个性化处理.

        Parameters
        ----------
        data: stimulus, choices, responses

        kwargs: dict
        "gamma" and "w0" activates a two-factor-decay on memory
        "amnesia":callable(data, **kwargs) and "amnesia_kwargs":dict
            enables a more flexible way to implement the decay-forgetting
            mechanism.
        """
        prob = self.calc_likelihood_base(hypo, data, beta, use_cached_dist,
                                         **kwargs)  # shape = [n_cats, n_trials]
        # Convert to numpy array to ensure correct operations
        choices = np.array(data[1])
        responses = np.array(data[2]) # shape = [n_trials]
        choices -= 1
        n_trials = len(choices)

        # (a) species-level 正确
        p_species = prob[choices, np.arange(n_trials)]

        # (b) family-level 正确（species 错）
        fam_sum = np.zeros(n_trials)
        conn_map = getattr(self, "connectivity_map", {})
        if conn_map:
            mask = np.zeros_like(prob, dtype=bool)  # shape = [n_cats, n_trials]
            for t in range(n_trials):
                alt_cats = conn_map[hypo][choices[t]]   # 不含自身
                mask[alt_cats, t] = True
            fam_sum = (prob * mask).sum(axis=0)
        else:
            fam_sum = np.zeros(n_trials)

        # (c) 完全错误
        p_wrong = 1.0 - p_species

        # (d) 根据 feedback 取对应概率
        likelihood = np.where(responses == 1,      p_species,
                      np.where(responses == 0.5,   fam_sum,
                               p_wrong))

        likelihood = np.clip(likelihood, self.EPS, 1. - self.EPS)

        return likelihood      # shape = (n_trials,)

    @amnesia_mechanism
    def calc_trueprob_entry(self,
                            hypo: int,
                            data: list | tuple,
                            beta: float | list | tuple | np.ndarray,
                            use_cached_dist: bool = False,
                            **kwargs) -> np.ndarray:
        """
        计算true category被选中的概率, 同样交由amnesia_mechanism 装饰器来做加权处理。
        """

        prob = self.calc_likelihood_base(hypo, data, beta, use_cached_dist,
                                         **kwargs) # shape: (n_cats, nTrial)

        category = np.asarray(data[3], dtype=int) - 1
        true_cat = category // 2 if self.n_cats == 2 else category # shape: (nTrial,)
        if prob.ndim == 1:
            prob = prob.reshape(-1, 1)
        return prob[true_cat.flatten(), np.arange(prob.shape[1])] # shape: (nTrial,)
        # return prob[true_cat]

    def MBase_likelihood(self, params: tuple, data) -> np.ndarray:
        """
        Similar to old version M_Base.likelihood. NO `condition` argument,
        `params` is just a tuple
        """

        k, beta = params
        x = data[['feature1', 'feature2', 'feature3',
                  'feature4']].values  # Shape: [n_trialss, 4]
        c = data['choice'].values  # Shape: [n_trialss]
        r = data['feedback'].values  # Shape: [n_trialss]
        return self.calc_likelihood_entry(k, (x, c, r), beta)



def signed_distance_to_category(x, cat_ineqs):
    """
    计算点x到类别区域的带符号距离:
      >0 在区域内部
       0 在边界上
      <0 在区域外
    """
    dists = []
    for (a, b, sign) in cat_ineqs:
        a = np.asarray(a)
        norm = np.linalg.norm(a) + 1e-9
        signed = (np.dot(a, x) - b) / norm
        inward = -sign * signed
        dists.append(inward)
    return np.min(dists)



# define partition rules
class Partition(BasePartition):
    """
    Partition类:
    1. 使用 get_all_splits(n_dims, n_cats) 生成各种分割方式对应的超平面组合。
    2. 使用 get_centers(n_dims, n_cats) 计算在这些分割方式下, 各个区域(类别)的代表中心点(重心)。
    """
    EPS = 1e-7

    def __init__(self, n_dims: int, n_cats: int, n_protos: int = 1, **kwargs):
        """Initialize"""
        super().__init__(n_dims, n_cats, n_protos, **kwargs)
        self.vertices: List[Tuple[float, float, float, float]] = []

        self.connectivity_map = self._compute_connectivity_map()

    def _compute_connectivity_map(self) -> dict[int, dict[int, list[int]]]:
        conn = {}
        n_cats, n_dims = self.n_cats, self.n_dims
        # 原型坐标：shape = [n_hypos, n_cats, n_dims]
        centers_all = self.prototypes_np.squeeze(
            axis=1)  # n_protos=1 → squeeze

        for h in range(self.length):
            centers = centers_all[h]  # shape (n_cats, n_dims)
            conn[h] = {c: [] for c in range(n_cats)}

            # 枚举所有类别对 (a,b)
            for a in range(n_cats):
                for b in range(a + 1, n_cats):
                    # 统计有多少维“明显不同”
                    diff_cnt = np.sum(
                        np.abs(centers[a] - centers[b]) > self.EPS)
                    if diff_cnt == 1:  # 仅 1 维不同 → 认为 family 相连
                        conn[h][a].append(b)
                        conn[h][b].append(a)
        return conn

    # ======================================================================
    # ========== 一、辅助几何函数 ===========================================
    # ======================================================================
    @staticmethod
    def _project_to_halfspace(y, a, b):
        a = np.asarray(a, dtype=float)
        diff = np.dot(a, y) - b
        if diff <= 0:
            return y
        return y - diff / (np.dot(a, a) + 1e-9) * a

    @staticmethod
    def _project_to_box01(y):
        return np.clip(y, 0.0, 1.0)

    @classmethod
    def _project_to_polytope(cls, y, A, b, n_iter=100):
        """
        Dykstra投影到多面体 {x | A x <= b, 0<=x<=1}
        """
        yk = y.copy()
        p = [np.zeros_like(y) for _ in range(len(A))]
        for _ in range(n_iter):
            for i in range(len(A)):
                y_hat = yk + p[i]
                y_new = cls._project_to_halfspace(y_hat, A[i], b[i])
                p[i] = y_hat - y_new
                yk = y_new
            yk = cls._project_to_box01(yk)
        return yk

    @classmethod
    def _distance_to_region(cls, x, A, b):
        """
        点到区域的距离：区域内=0，区域外=到多面体最近点距离
        """
        vals = A @ x - b
        if np.all(vals <= 0 + 1e-9):
            return 0.0
        proj = cls._project_to_polytope(x, A, b)
        return np.linalg.norm(x - proj)

    # ======================================================================
    # ========== 二、生成类别区域的不等式定义 ===============================
    # ======================================================================
    def generate_category_inequalities(self, split_type, hyperplanes):
        """
        根据分割方式生成每个类别区域对应的线性约束 (A,b)
        """
        three_plane_types = {
            "3d_axis_triple", "3d_axis_equality_sum", "4d_equality_axis_pair",
            "4d_sum_axis_pair"
        }

        # ---------- 一般情况 ----------
        if split_type not in three_plane_types and split_type not in (
                "dimension_max", "dimension_min"):
            categories = []
            for signs in product([-1, 1], repeat=len(hyperplanes)):
                A, b = [], []
                for (a, bi), s in zip(hyperplanes, signs):
                    A.append(s * np.array(a))
                    b.append(s * bi)
                categories.append({'A': np.vstack(A), 'b': np.array(b)})
            return categories

        # ---------- 三平面类型 ----------
        if split_type in three_plane_types:
            (a1, b1), (a2, b2), (a3, b3) = hyperplanes
            a1, a2, a3 = map(np.asarray, (a1, a2, a3))
            cats = [
                {
                    'A': np.vstack([a1, a2]),
                    'b': np.array([b1, b2])
                },  # 左下
                {
                    'A': np.vstack([a1, -a2]),
                    'b': np.array([b1, -b2])
                },  # 左上
                {
                    'A': np.vstack([-a1, a3]),
                    'b': np.array([-b1, b3])
                },  # 右下
                {
                    'A': np.vstack([-a1, -a3]),
                    'b': np.array([-b1, -b3])
                }  # 右上
            ]
            return cats

        # ---------- 维度极值类型 ----------
        if split_type in ("dimension_max", "dimension_min"):
            n_dims = self.n_dims
            cats = []
            for i in range(n_dims):
                As, bs = [], []
                for j in range(n_dims):
                    if i == j:
                        continue
                    a = np.zeros(n_dims, dtype=float)
                    if split_type == "dimension_max":
                        # x_i >= x_j   <=>   -x_i + x_j <= 0
                        a[i], a[j] = -1., 1.
                        bi = 0.0
                    else:
                        # x_i <= x_j   <=>    x_i - x_j <= 0
                        a[i], a[j] = 1., -1.
                        bi = 0.0
                    As.append(a)
                    bs.append(bi)
                A = np.vstack(As) if As else np.zeros((0, n_dims))
                b = np.array(bs, dtype=float)
                cats.append({'A': A, 'b': b})
            return cats

    # ======================================================================
    # ========== 三、基于边界距离的Likelihood计算 ==========================
    # ======================================================================
    def calc_likelihood_boundary(self,
                                 hypo: int,
                                 data: list | tuple,
                                 beta: float = 5.0,
                                 **kwargs):
        """
        基于边界距离的likelihood计算：
          - 区域内: 距离=0
          - 区域外: 距离=点到该多面体的最短欧式距离
        """
        stimuli, _ = data[:2]
        split_type, hyperplanes = self.splits[hypo]
        categories = self.generate_category_inequalities(
            split_type, hyperplanes)

        n_cats = len(categories)
        n_trials = len(stimuli)
        dmat = np.zeros((n_trials, n_cats))

        for c, cat in enumerate(categories):
            A, b = cat['A'], cat['b']
            for t, x in enumerate(stimuli):
                dmat[t, c] = self._distance_to_region(np.array(x), A, b)

        # softmax(-β * 距离)
        scores = np.exp(-beta * dmat)
        prob = scores / np.sum(scores, axis=1, keepdims=True)

        return prob.T

    # ======================================================================
    # ========== 四、对接主接口 calc_likelihood_entry ======================
    # ======================================================================
    def calc_likelihood_entry(self,
                              hypo: int,
                              data: list | tuple,
                              beta: float,
                              use_cached_dist: bool = False,
                              **kwargs) -> np.ndarray:
        """
        当 use_boundary=True 时，使用基于边界距离的概率计算。
        其余部分（反馈映射）与原逻辑一致。
        """
        if kwargs.get("use_boundary", False):
            prob = self.calc_likelihood_boundary(hypo, data, beta, **kwargs)
        else:
            prob = self.calc_likelihood_base(hypo, data, beta, use_cached_dist,
                                             **kwargs)

        choices, responses = data[1].copy(), data[2]
        choices = np.asarray(data[1])
        choices -= 1
        n_trials = len(choices)
        p_species = prob[choices, np.arange(n_trials)]

        fam_sum = np.zeros(n_trials)
        conn_map = getattr(self, "connectivity_map", {})
        if conn_map:
            mask = np.zeros_like(prob, dtype=bool)
            for t in range(n_trials):
                alt = conn_map[hypo][choices[t]]
                mask[alt, t] = True
            fam_sum = (prob * mask).sum(axis=0)
        p_wrong = 1. - p_species

        likelihood = np.where(responses == 1, p_species,
                              np.where(responses == 0.5, fam_sum, p_wrong))
        return np.clip(likelihood, self.EPS, 1. - self.EPS)


    # ======================================================================
    # ========== 原版分割 ======================
    # ======================================================================
    def get_all_splits(self):
        """
        根据维度数 n_dims 和分类数 n_cats, 生成所有可用的分割方式 (split_type)
        及对应的超平面列表 hyperplanes.

        返回值:
          List[ (split_type: str, hyperplanes: List[(coef_list, threshold)]) ]

        其中:
          - split_type: 字符串, 标识某种分割方式.
          - hyperplanes: 若干个超平面, 每个超平面用 (coefs, threshold) 表示:
                coefs: [float]*n_dims, 各维度对应的系数
                threshold: float, 表示 coefs·x = threshold

        比如 "x_i = 0.5" 可表示为 ([0, ..., 1, ..., 0], 0.5) 其中第 i 个位置系数=1, 其余=0.
        """
        splits = []
        n_dims = self.n_dims
        n_cats = self.n_cats

        # --------------- 两分类情况 ---------------
        if n_cats == 2:
            # 1. 单维轴对齐超平面 (Axis-aligned): x_i = 0.5
            for i in range(n_dims):
                coeff = [0] * n_dims
                coeff[i] = 1
                splits.append(('axis', [(tuple(coeff), 0.5)]))

            # 2a. 二维相等超平面 (Equality): x_i = x_j
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    coeff = [0] * n_dims
                    coeff[i] = 1
                    coeff[j] = -1
                    splits.append(('equality', [(tuple(coeff), 0)]))

            # 2b. 二维和式超平面 (Sum): x_i + x_j = 1
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    coeff = [0] * n_dims
                    coeff[i] = 1
                    coeff[j] = 1
                    splits.append(('sum', [(tuple(coeff), 1)]))

            # 3. 四维混合超平面 (Mixed): x_i + x_j = x_k + x_l
            if n_dims >= 4:
                dim_pairs = [
                    ((0, 1, 2, 3)),  # x1 + x2 = x3 + x4
                    ((0, 2, 1, 3)),  # x1 + x3 = x2 + x4
                    ((0, 3, 1, 2))  # x1 + x4 = x2 + x3
                ]
                for i, j, k, l in dim_pairs:
                    coeff = [0] * n_dims
                    coeff[i] = coeff[j] = 1
                    coeff[k] = coeff[l] = -1
                    splits.append(('mixed', [(tuple(coeff), 0)]))

        # --------------- 四分类情况 ---------------
        elif n_cats == 4:
            # 1. 两个超平面
            # 1.1 涉及两个维度
            # 1.1a 两个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5)
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    plane1 = ([0] * n_dims, 0.5)
                    plane2 = ([0] * n_dims, 0.5)
                    plane1[0][i] = 1
                    plane2[0][j] = 1
                    splits.append(('2d_axis_pair', [plane1, plane2]))

            # 1.1b 二维相等 + 二维和式: (x_i = x_j, x_i + x_j = 1)
            for i, j in itertools.combinations(range(n_dims), 2):
                plane1 = ([0] * n_dims, 0)
                plane2 = ([0] * n_dims, 1)
                plane1[0][i], plane1[0][j] = 1, -1
                plane2[0][i] = plane2[0][j] = 1
                splits.append(('2d_equality_sum', [plane1, plane2]))

            # 1.2 涉及三个维度
            # 1.2a 单维轴对齐 + 二维相等: (x_i = 0.5, x_j = x_k)
            if n_dims >= 3:
                for i in range(n_dims):
                    remaining = [j for j in range(n_dims) if j != i]
                    for j, k in itertools.combinations(remaining, 2):
                        plane1 = ([0] * n_dims, 0.5)
                        plane2 = ([0] * n_dims, 0)
                        plane1[0][i] = 1
                        plane2[0][j], plane2[0][k] = 1, -1
                        splits.append(('3d_axis_equality', [plane1, plane2]))

            # 1.2b 单维轴对齐 + 二维和式: (x_i = 0.5, x_j + x_k = 1)
            if n_dims >= 3:
                for i in range(n_dims):
                    remaining = [j for j in range(n_dims) if j != i]
                    for j, k in itertools.combinations(remaining, 2):
                        plane1 = ([0] * n_dims, 0.5)
                        plane2 = ([0] * n_dims, 1)
                        plane1[0][i] = 1
                        plane2[0][j] = plane2[0][k] = 1
                        splits.append(('3d_axis_sum', [plane1, plane2]))

            # 1.3 涉及四个维度
            # 1.3a 两个二维相等超平面: (x_i = x_j, x_k = x_l)
            if n_dims >= 4:
                dim_pairs = [
                    ((0, 1, 2, 3)),  # x1 = x2, x3 = x4
                    ((0, 2, 1, 3)),  # x1 = x3, x2 = x4
                    ((0, 3, 1, 2))  # x1 = x4, x2 = x3
                ]
                for i, j, k, l in dim_pairs:
                    plane1 = ([0] * n_dims, 0)
                    plane2 = ([0] * n_dims, 0)
                    plane1[0][i], plane1[0][j] = 1, -1
                    plane2[0][k], plane2[0][l] = 1, -1
                    splits.append(('4d_equality_pair', [plane1, plane2]))

            # 1.3b 两个二维和式超平面: (x_i + x_j = 1, x_k + x_l = 1)
            if n_dims >= 4:
                dim_pairs = [
                    ((0, 1, 2, 3)),  # x1 + x2 = 1, x3 + x4 = 1
                    ((0, 2, 1, 3)),  # x1 + x3 = 1, x2 + x4 = 1
                    ((0, 3, 1, 2))  # x1 + x4 = 1, x2 + x3 = 1
                ]
                for i, j, k, l in dim_pairs:
                    plane1 = ([0] * n_dims, 1)
                    plane2 = ([0] * n_dims, 1)
                    plane1[0][i] = plane1[0][j] = 1
                    plane2[0][k] = plane2[0][l] = 1
                    splits.append(('4d_sum_pair', [plane1, plane2]))

            # 2. 三个超平面
            # 2.1 涉及三个维度
            # 2.1a 三个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5, x_k = 0.5)
            if n_dims >= 3:
                for i, j, k in itertools.combinations(range(n_dims), 3):
                    # 考虑所有可能的排列顺序
                    for m, n1, n2 in itertools.permutations([i, j, k]):
                        plane_m = ([0] * n_dims, 0.5)
                        plane_n1 = ([0] * n_dims, 0.5)
                        plane_n2 = ([0] * n_dims, 0.5)
                        plane_m[0][m] = 1
                        plane_n1[0][n1] = 1
                        plane_n2[0][n2] = 1
                        splits.append(
                            ('3d_axis_triple', [plane_m, plane_n1, plane_n2]))

            # 2.1b 单维轴对齐 + 二维相等 + 二维和式: (x_i = 0.5, x_j = x_k, x_j + x_k = 1)
            if n_dims >= 3:
                for m in range(n_dims):
                    remaining = [i for i in range(n_dims) if i != m]
                    for i, j in itertools.combinations(remaining, 2):
                        plane_m = ([0] * n_dims, 0.5)  # 轴对齐
                        plane_eq = ([0] * n_dims, 0)  # 相等
                        plane_sum = ([0] * n_dims, 1)  # 和式

                        plane_m[0][m] = 1
                        plane_eq[0][i], plane_eq[0][j] = 1, -1
                        plane_sum[0][i] = plane_sum[0][j] = 1

                        splits.append(('3d_axis_equality_sum',
                                       [plane_m, plane_eq, plane_sum]))
                        splits.append(('3d_axis_equality_sum',
                                       [plane_m, plane_sum, plane_eq]))

            # 2.2 涉及四个维度
            # 2.2a 二维相等 + 两个单维轴对齐: (x_i = x_j, x_k = 0.5, x_l = 0.5)
            if n_dims >= 4:
                for i, j in itertools.combinations(range(n_dims), 2):
                    remaining = [k for k in range(n_dims) if k not in (i, j)]
                    for k, l in itertools.combinations(remaining, 2):
                        plane_eq = ([0] * n_dims, 0)  # 二维相等
                        plane_axis1 = ([0] * n_dims, 0.5)  # 单维轴对齐
                        plane_axis2 = ([0] * n_dims, 0.5)  # 单维轴对齐

                        plane_eq[0][i], plane_eq[0][j] = 1, -1
                        plane_axis1[0][k] = 1
                        plane_axis2[0][l] = 1

                        # 固定二维相等在第一个位置，后两个单维轴对齐排列组合
                        splits.append(('4d_equality_axis_pair',
                                       [plane_eq, plane_axis1, plane_axis2]))
                        splits.append(('4d_equality_axis_pair',
                                       [plane_eq, plane_axis2, plane_axis1]))

            # 2.2b 二维和式 + 两个单维轴对齐: (x_i + x_j = 1, x_k = 0.5, x_l = 0.5)
            if n_dims >= 4:
                for i, j in itertools.combinations(range(n_dims), 2):
                    remaining = [k for k in range(n_dims) if k not in (i, j)]
                    for k, l in itertools.combinations(remaining, 2):
                        plane_sum = ([0] * n_dims, 1)  # 二维和式
                        plane_axis1 = ([0] * n_dims, 0.5)  # 单维轴对齐
                        plane_axis2 = ([0] * n_dims, 0.5)  # 单维轴对齐

                        plane_sum[0][i], plane_sum[0][j] = 1, 1
                        plane_axis1[0][k] = 1
                        plane_axis2[0][l] = 1

                        # 固定二维和式在第一个位置，后两个单维轴对齐排列组合
                        splits.append(('4d_sum_axis_pair',
                                       [plane_sum, plane_axis1, plane_axis2]))
                        splits.append(('4d_sum_axis_pair',
                                       [plane_sum, plane_axis2, plane_axis1]))

            # 3. C(n_dims,2)个超平面（所有二维相等超平面：x_i = x_j）
            # 当 n_dims == n_cats = 4 时, 这些超平面可组合成"哪一维最..."的划分(dimension_max)
            if n_dims == n_cats:
                eq_planes = []
                for i, j in itertools.combinations(range(n_dims), 2):
                    plane = ([0] * n_dims, 0)
                    plane[0][i], plane[0][j] = 1, -1  # x_i - x_j = 0
                    eq_planes.append(plane)

                splits.append(('dimension_max', eq_planes))
                splits.append(('dimension_min', eq_planes))

        return splits

    # generate centers
    def get_centers(self):
        """
        计算每种分割方法下各个区域的中心点(重心).

        返回: List[ (split_type, {cat_idx: (center_x1, center_x2, ...)}) ]
        """
        n_cats = self.n_cats
        n_dims = self.n_dims
        splits = self.get_all_splits()
        results = []

        for split_type, hyperplanes in splits:
            centers = {cat_idx: [] for cat_idx in range(n_cats)}

            # --------------- 两分类情况 ---------------
            if n_cats == 2:
                # 1. 单维轴对齐超平面: x_i = 0.5
                if split_type == 'axis':
                    split_dim = next(
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0)

                    for dim in range(n_dims):
                        if dim == split_dim:
                            centers[0].append(0.25)  # x < 0.5 的区域
                            centers[1].append(0.75)  # x > 0.5 的区域
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

                # 2a. 二维相等超平面: x_i = x_j
                elif split_type == 'equality':
                    split_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]
                    dim1, dim2 = split_dims[0], split_dims[1]

                    # 对于涉及的两个维度，一个区域是(1/3, 2/3)，另一个是(2/3, 1/3)
                    for dim in range(n_dims):
                        if dim == dim1:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                        elif dim == dim2:
                            centers[0].append(2 / 3)
                            centers[1].append(1 / 3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

                # 2b. 二维和式超平面: x_i + x_j = 1
                elif split_type == 'sum':
                    split_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]
                    dim1, dim2 = split_dims[0], split_dims[1]

                    # 对于涉及的两个维度，一个区域是(1/3, 1/3)，另一个是(2/3, 2/3)
                    for dim in range(n_dims):
                        if dim in [dim1, dim2]:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

                # 3. 四维混合超平面: x_i + x_j = x_k + x_l
                elif split_type == 'mixed':
                    pos_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef == 1
                    ]
                    neg_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef == -1
                    ]

                    for dim in range(n_dims):
                        if dim in pos_dims:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                        elif dim in neg_dims:
                            centers[0].append(2 / 3)
                            centers[1].append(1 / 3)
                        else:
                            centers[0].append(0.5)
                            centers[1].append(0.5)

            # --------------- 四分类情况 ---------------
            elif n_cats == 4:
                # 1. 两个超平面
                # 1.1a 两个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5)
                if split_type == '2d_axis_pair':
                    split_dims = []
                    for hyperplane in hyperplanes:
                        dim_idx = next(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                        split_dims.append(dim_idx)

                    for dim in range(n_dims):
                        if dim == split_dims[0]:
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        elif dim == split_dims[1]:
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.1b 二维相等 + 二维和式: (x_i = x_j, x_i + x_j = 1)
                elif split_type == '2d_equality_sum':
                    split_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]

                    for dim in range(n_dims):
                        if dim in split_dims:
                            if dim == split_dims[0]:  # i
                                centers[0].append(1 / 2)
                                centers[1].append(1 / 2)
                                centers[2].append(5 / 6)
                                centers[3].append(1 / 6)
                            else:  # j
                                centers[0].append(1 / 6)
                                centers[1].append(5 / 6)
                                centers[2].append(1 / 2)
                                centers[3].append(1 / 2)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.2a 单维轴对齐 + 二维相等: (x_i = 0.5, x_j = x_k)
                # 1.2b 单维轴对齐 + 二维和式: (x_i = 0.5, x_j + x_k = 1)
                elif split_type in ['3d_axis_equality', '3d_axis_sum']:
                    axis_hyperplane = next(plane for plane in hyperplanes
                                           if sum(1 for c in plane[0]
                                                  if c != 0) == 1)
                    other_hyperplane = next(plane for plane in hyperplanes
                                            if plane != axis_hyperplane)

                    axis_dim = next(
                        dim_idx
                        for dim_idx, coef in enumerate(axis_hyperplane[0])
                        if coef != 0)
                    other_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(other_hyperplane[0])
                        if coef != 0
                    ]

                    for dim in range(n_dims):
                        if dim == axis_dim:
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim in other_dims:
                            if split_type == '3d_axis_equality':
                                if dim == other_dims[0]:
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                                else:
                                    centers[0].append(2 / 3)
                                    centers[1].append(1 / 3)
                                    centers[2].append(2 / 3)
                                    centers[3].append(1 / 3)
                            else:  # 3d_axis_sum
                                centers[0].append(1 / 3)
                                centers[1].append(2 / 3)
                                centers[2].append(1 / 3)
                                centers[3].append(2 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.3a 两个二维相等超平面: (x_i = x_j, x_k = x_l)
                elif split_type == '4d_equality_pair':
                    split_dim_pairs = []
                    for hyperplane in hyperplanes:
                        dim_pair = [
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0
                        ]
                        split_dim_pairs.append(dim_pair)

                    for dim in range(n_dims):
                        if dim in split_dim_pairs[0]:
                            if dim == split_dim_pairs[0][0]:  # i
                                centers[0].append(1 / 3)
                                centers[1].append(2 / 3)
                                centers[2].append(1 / 3)
                                centers[3].append(2 / 3)
                            else:  # j
                                centers[0].append(2 / 3)
                                centers[1].append(1 / 3)
                                centers[2].append(2 / 3)
                                centers[3].append(1 / 3)
                        elif dim in split_dim_pairs[1]:
                            if dim == split_dim_pairs[1][0]:  # k
                                centers[0].append(1 / 3)
                                centers[1].append(1 / 3)
                                centers[2].append(2 / 3)
                                centers[3].append(2 / 3)
                            else:  # l
                                centers[0].append(2 / 3)
                                centers[1].append(2 / 3)
                                centers[2].append(1 / 3)
                                centers[3].append(1 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 1.3b 两个二维和式超平面: (x_i + x_j = 1, x_k + x_l = 1)
                elif split_type == '4d_sum_pair':
                    split_dim_pairs = []
                    for hyperplane in hyperplanes:
                        dim_pair = [
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0
                        ]
                        split_dim_pairs.append(dim_pair)

                    for dim in range(n_dims):
                        if dim in split_dim_pairs[0]:
                            centers[0].append(1 / 3)
                            centers[1].append(2 / 3)
                            centers[2].append(1 / 3)
                            centers[3].append(2 / 3)
                        elif dim in split_dim_pairs[1]:
                            centers[0].append(1 / 3)
                            centers[1].append(1 / 3)
                            centers[2].append(2 / 3)
                            centers[3].append(2 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 2. 三个超平面
                # 2.1a 三个单维轴对齐超平面: (x_i = 0.5, x_j = 0.5, x_k = 0.5)
                elif split_type == '3d_axis_triple':
                    split_dims = []
                    for hyperplane in hyperplanes:
                        dim_idx = next(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                        split_dims.append(dim_idx)

                    for dim in range(n_dims):
                        if dim == split_dims[0]:  # 第一个切割
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim == split_dims[1]:  # 第二个切割 (x1 < 0.5 部分)
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.5)
                            centers[3].append(0.5)
                        elif dim == split_dims[2]:  # 第三个切割 (x1 > 0.5 部分)
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 2.1b 单维轴对齐 + 二维相等 + 二维和式:
                #      (x_i = 0.5, x_j = x_k, x_j + x_k = 1)
                elif split_type == '3d_axis_equality_sum':
                    axis_dim = next(
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0)
                    other_dims = set()
                    for hyperplane in hyperplanes[1:]:
                        other_dims.update(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                    other_dims = list(other_dims)

                    # 检查第二个超平面是否为二维相等
                    is_second_equality = sum(
                        1 for c in hyperplanes[1][0]
                        if c != 0) == 2 and hyperplanes[1][1] == 0

                    for dim in range(n_dims):
                        if dim == axis_dim:  # x1
                            centers[0].append(0.25)
                            centers[1].append(0.25)
                            centers[2].append(0.75)
                            centers[3].append(0.75)
                        elif dim in other_dims:
                            if is_second_equality:  # 第二个超平面是二维相等
                                # x1 < 0.5 部分用 x2 = x3,
                                # x1 > 0.5 部分用 x2 + x3 = 1
                                if dim == other_dims[0]:  # x2
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                                else:  # x3
                                    centers[0].append(2 / 3)
                                    centers[1].append(1 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                            else:  # 第二个超平面是二维和式
                                # x1 < 0.5 部分用 x2 + x3 = 1,
                                # x1 > 0.5 部分用 x2 = x3
                                if dim == other_dims[0]:  # x2
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(2 / 3)
                                else:  # x3
                                    centers[0].append(1 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(2 / 3)
                                    centers[3].append(1 / 3)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 2.2a 二维相等 + 两个单维轴对齐: (x_i = x_j, x_k = 0.5, x_l = 0.5)
                # 2.2b 二维和式 + 两个单维轴对齐: (x_i + x_j = 1, x_k = 0.5, x_l = 0.5)
                elif split_type in [
                        '4d_equality_axis_pair', '4d_sum_axis_pair'
                ]:
                    first_dims = [
                        dim_idx
                        for dim_idx, coef in enumerate(hyperplanes[0][0])
                        if coef != 0
                    ]
                    axis_dims = []
                    for hyperplane in hyperplanes[1:]:
                        dim_idx = next(
                            dim_idx
                            for dim_idx, coef in enumerate(hyperplane[0])
                            if coef != 0)
                        axis_dims.append(dim_idx)

                    for dim in range(n_dims):
                        if dim in first_dims:
                            if split_type == '4d_equality_axis_pair':
                                if dim == first_dims[0]:  # i
                                    centers[0].append(1 / 3)
                                    centers[1].append(1 / 3)
                                    centers[2].append(2 / 3)
                                    centers[3].append(2 / 3)
                                else:  # j
                                    centers[0].append(2 / 3)
                                    centers[1].append(2 / 3)
                                    centers[2].append(1 / 3)
                                    centers[3].append(1 / 3)
                            else:  # 4d_sum_axis_pair
                                centers[0].append(1 / 3)
                                centers[1].append(1 / 3)
                                centers[2].append(2 / 3)
                                centers[3].append(2 / 3)
                        elif dim == axis_dims[0]:  # k
                            centers[0].append(0.25)
                            centers[1].append(0.75)
                            centers[2].append(0.5)
                            centers[3].append(0.5)
                        elif dim == axis_dims[1]:  # l
                            centers[0].append(0.5)
                            centers[1].append(0.5)
                            centers[2].append(0.25)
                            centers[3].append(0.75)
                        else:
                            for cat_idx in range(4):
                                centers[cat_idx].append(0.5)

                # 3. C(n_dims,2)个超平面（所有二维相等超平面：x_i = x_j）
                # 当 n_dims == n_cats = 4 时, 这些超平面可组合成"哪一维最..."的划分(dimension_max)
                # 最高级情况: 第 cat_idx 维取 0.8，其余维度取 0.4
                elif split_type == 'dimension_max':
                    centers_high = {}
                    for cat_idx in range(n_cats):
                        center_coords_high = [0.4] * n_dims
                        center_coords_high[cat_idx] = 0.8
                        centers_high[cat_idx] = tuple(center_coords_high)
                    results.append((split_type, centers_high))
                    continue

                # 最低级情况: 第 cat_idx 维取 0.4，其余维度取 0.8
                elif split_type == 'dimension_min':
                    centers_low = {}
                    for cat_idx in range(n_cats):
                        center_coords_low = [0.8] * n_dims
                        center_coords_low[cat_idx] = 0.4
                        centers_low[cat_idx] = tuple(center_coords_low)
                    results.append((split_type, centers_low))
                    continue

            # 将列表转换为元组
            centers = {k: tuple(v) for k, v in centers.items()}
            results.append((split_type, centers))

        return results
