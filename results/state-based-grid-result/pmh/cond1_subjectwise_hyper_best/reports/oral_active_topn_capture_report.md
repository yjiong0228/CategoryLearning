# Oral Active-Set Top-N Capture Report: Condition 1

生成时间：2026-05-26  
结果目录：`results/state-based-grid-result/pmh/cond1_subjectwise_hyper_best`

## 1. 分析目的

此前的 full-space oral/model alignment 直接比较两个完整 hypothesis 分布：

- 模型分布：choice-conditioned prior `q_t(h)`
- 口头报告分布：oral distribution `o_t(h)`

这个比较会受到 hypothesis transition 模块的强烈影响。模型每个 trial 只保留一个 active hypothesis set，因此 active set 之外的 hypothesis 在 `prior_t` 和 `q_t` 中质量为 0。这样 full-space JS similarity 或 dot product 会同时反映两件事：

1. 模型当前 active set 是否覆盖 oral 支持的 hypothesis 区域。
2. 在已经覆盖的区域内，模型权重是否和 oral 权重一致。

本报告采用一个更直观、更公平的 active-set/top-N 诊断：在每个 trial 中，先取模型 active set 的大小 `N_t`，再问：

> 如果 oral distribution 也只能选择 `N_t` 个 hypothesis，那么模型 active set 捕获了 oral top-N 最大质量中的多少？

这可以控制模型每个 trial 只考虑少数 hypothesis 的机制约束。

## 2. 数据和输出文件

输入数据：

- 模型结果：`subject_*.json`
- 口头报告数据：`data/processed/Task2_processed.csv`
- oral 编码模式：`center`

输出文件：

- 图：`plots/oral_active_topn_capture.png`
- trial-level 指标：`plots/oral_active_topn_capture_trial_metrics.csv`
- subject-level 均值：`plots/oral_active_topn_capture_subject_means.csv`
- time-bin 均值：`plots/oral_active_topn_capture_binned.csv`

有效样本：

- subjects = 32
- valid trials = 9501
- total hypothesis count = 19

## 3. 具体计算方式

### 3.1 模型 active set

每个 trial `t`，从模型保存的 `prior_log[t]` 中定义 active set：

```text
A_t = {h | prior_t(h) > 1e-12}
N_t = |A_t|
```

其中 `N_t` 是当前 trial 模型实际保留的 hypothesis 数量。

### 3.2 oral distribution

因为本分析使用 `oral_center`，先将每个 trial 的 oral center 映射到 hypothesis 空间。

令：

```text
c_t = oral center
k_t = participant choice
center(h, k_t) = hypothesis h 对 choice category k_t 的 prototype center
```

每个 hypothesis 的距离：

```text
d_t(h) = || c_t - center(h, k_t) ||_2
```

再用自适应 softmax 转成 oral distribution：

```text
spread_t = median(d_t) - min(d_t)
score_t(h) = exp(-(d_t(h) - min(d_t)) / spread_t)
o_t(h) = score_t(h) / sum_h score_t(h)
```

如果 `spread_t` 过小，则回退使用距离标准差；如果仍过小，则只给最小距离 hypothesis 分配质量。

### 3.3 oral top-N oracle

在同一个 trial，用模型 active set 的大小 `N_t` 作为预算，从 oral distribution 里选出 top-N：

```text
T_t = top N_t hypotheses under o_t(h)
```

这个集合是同等 N 预算下能捕获 oral mass 的最优集合。

### 3.4 指标

模型 active set 捕获的 oral mass：

```text
active_oral_mass_t = sum_{h in A_t} o_t(h)
```

同等 N 预算下 oracle top-N 能捕获的 oral mass：

```text
oracle_topN_oral_mass_t = sum_{h in T_t} o_t(h)
```

active set 捕获效率：

```text
active_capture_ratio_t =
    active_oral_mass_t / oracle_topN_oral_mass_t
```

这个值越接近 1，说明模型 active set 越接近 oral 最支持的那 `N_t` 个 hypothesis。

active set 和 oral top-N 的身份重叠：

```text
active_topN_overlap_t =
    |A_t ∩ T_t| / N_t
```

随机 baseline：

```text
random_expected_mass_t = N_t / total_hypotheses
```

它表示如果随机选择 `N_t` 个 hypothesis，期望捕获的 oral mass。

## 4. 图的读法

主图：`plots/oral_active_topn_capture.png`

左上：Coverage efficiency under same N

- 蓝线：`active_capture_ratio`
- 橙线：`active_topN_overlap`
- 越高表示模型 active set 越接近 oral top-N。
- 蓝线高但橙线低，说明模型捕获了不少 oral mass，但具体 hypothesis identity 未必完全相同。

右上：How much oral mass is captured

- 绿线：模型 active set 捕获的 oral mass。
- 红线：oral top-N oracle 能捕获的最大 oral mass。
- 灰线：随机选择 `N_t` 个 hypothesis 的期望 baseline。
- 绿线越接近红线越好；绿线高于灰线表示模型 active set 不是随机覆盖。

左下：Between-subject summary

- 每个点代表一个被试的 trial 平均。
- 箱线图展示个体差异。

右下：Model hypothesis-set size

- 紫线：平均 active hypothesis 数量 `N_t`。
- 棕线：`N_t / total_hypotheses`。
- 这个面板用于判断 active set 预算是否随训练变化。

## 5. 结果

### 5.1 整体结果

| 指标 | mean | median | p25 | p75 |
|---|---:|---:|---:|---:|
| `n_active` | 3.018 | 3.000 | 3.000 | 3.000 |
| `active_fraction` | 0.159 | 0.158 | 0.158 | 0.158 |
| `active_oral_mass` | 0.185 | 0.181 | 0.143 | 0.227 |
| `oracle_topN_oral_mass` | 0.294 | 0.290 | 0.254 | 0.333 |
| `random_expected_mass` | 0.159 | 0.158 | 0.158 | 0.158 |
| `active_capture_ratio` | 0.648 | 0.630 | 0.499 | 0.826 |
| `active_topN_overlap` | 0.254 | 0.333 | 0.000 | 0.333 |

Subject-level 均值：

| 指标 | mean | median | min | max |
|---|---:|---:|---:|---:|
| `active_capture_ratio` | 0.659 | 0.656 | 0.509 | 0.905 |
| `active_topN_overlap` | 0.265 | 0.256 | 0.135 | 0.485 |

### 5.2 时间趋势

| 指标 | early bin | late bin |
|---|---:|---:|
| `active_capture_ratio` | 0.564 | 0.770 |
| `active_topN_overlap` | 0.211 | 0.346 |
| `active_oral_mass` | 0.178 | 0.201 |
| `oracle_topN_oral_mass` | 0.323 | 0.266 |
| `random_expected_mass` | 0.165 | 0.153 |
| `n_active` | 3.129 | 2.903 |

## 6. 解读

Condition 1 中模型每个 trial 平均只保留约 3 个 active hypotheses，占全空间约 15.9%。在这个很小的 hypothesis 预算下，模型 active set 捕获了约 18.5% 的 oral mass，高于随机 baseline 的 15.9%，说明模型 active set 对 oral 支持区域有一定选择性覆盖。

同等 N 预算下，oral top-N oracle 可捕获约 29.4% 的 oral mass，而模型 active set 平均达到 oracle 的约 64.8%。这说明模型 active set 并非随机，但也尚未完全覆盖 oral 最支持的 hypothesis 集合。

更有意思的是时间趋势：`active_capture_ratio` 从早期约 0.56 上升到晚期约 0.77，`active_topN_overlap` 从约 0.21 上升到约 0.35。这表示随着学习推进，模型 active set 越来越接近 oral top-N 区域。

因此，Condition 1 的结论可以概括为：

> full-space oral/model similarity 偏低不能简单解释为 oral 和模型完全不对齐。控制 active-set 大小后，模型 active set 对 oral-supported hypotheses 有中等程度覆盖，而且这种覆盖随试次推进明显增强。

## 7. 展示建议

如果向其他人展示，建议重点讲三点：

1. 模型受 hypothesis transition 限制，每个 trial 只考虑少数 hypotheses，因此 full-space alignment 会天然偏低。
2. 本分析用 oral top-N oracle 作为同等 N 预算下的公平 baseline。
3. Condition 1 中 active set 捕获效率约为 65%，并从早期到晚期持续上升，说明 active set 与 oral 支持区域的对齐在学习中增强。
