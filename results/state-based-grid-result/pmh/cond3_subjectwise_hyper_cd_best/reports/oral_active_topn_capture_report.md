# Oral Active-Set Top-N Capture Report: Condition 3

生成时间：2026-05-26  
结果目录：`results/state-based-grid-result/pmh/cond3_subjectwise_hyper_cd_best`

## 1. 分析目的

此前的 full-space oral/model alignment 直接比较两个完整 hypothesis 分布：

- 模型分布：choice-conditioned prior `q_t(h)`
- 口头报告分布：oral distribution `o_t(h)`

这个比较会受到 hypothesis transition 模块的强烈影响。模型每个 trial 只保留一个 active hypothesis set，因此 active set 之外的 hypothesis 在 `prior_t` 和 `q_t` 中质量为 0。这样 full-space JS similarity 或 dot product 会同时反映两件事：

1. 模型当前 active set 是否覆盖 oral 支持的 hypothesis 区域。
2. 在已经覆盖的区域内，模型权重是否和 oral 权重一致。

本报告采用一个更直观、更公平的 active-set/top-N 诊断：在每个 trial 中，先取模型 active set 的大小 `N_t`，再问：

> 如果 oral distribution 也只能选择 `N_t` 个 hypothesis，那么模型 active set 捕获了 oral top-N 最大质量中的多少？

这可以控制模型每个 trial 只考虑少数 hypothesis 的机制约束。Condition 3 的 hypothesis 空间更大，因此该诊断尤其重要。

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
- valid trials = 20349
- total hypothesis count = 116

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
| `n_active` | 7.470 | 8.000 | 6.000 | 9.000 |
| `active_fraction` | 0.064 | 0.069 | 0.052 | 0.078 |
| `active_oral_mass` | 0.078 | 0.073 | 0.054 | 0.101 |
| `oracle_topN_oral_mass` | 0.132 | 0.138 | 0.109 | 0.160 |
| `random_expected_mass` | 0.064 | 0.069 | 0.052 | 0.078 |
| `active_capture_ratio` | 0.589 | 0.571 | 0.467 | 0.705 |
| `active_topN_overlap` | 0.165 | 0.111 | 0.000 | 0.300 |

Subject-level 均值：

| 指标 | mean | median | min | max |
|---|---:|---:|---:|---:|
| `active_capture_ratio` | 0.607 | 0.608 | 0.480 | 0.731 |
| `active_topN_overlap` | 0.187 | 0.195 | 0.031 | 0.341 |

### 5.2 时间趋势

| 指标 | early bin | late bin |
|---|---:|---:|
| `active_capture_ratio` | 0.491 | 0.740 |
| `active_topN_overlap` | 0.061 | 0.366 |
| `active_oral_mass` | 0.062 | 0.096 |
| `oracle_topN_oral_mass` | 0.128 | 0.128 |
| `random_expected_mass` | 0.061 | 0.064 |
| `n_active` | 7.029 | 7.465 |

## 6. 解读

Condition 3 的 hypothesis 空间有 116 个 hypotheses，而模型每个 trial 平均只保留约 7.5 个 active hypotheses，占全空间约 6.4%。在这样强烈受限的 active set 下，直接做 full-space JS 或 dot product 会明显惩罚模型，因为大部分 hypothesis 在模型分布中结构性为 0。

控制同等 `N_t` 预算后，模型 active set 捕获了约 7.8% 的 oral mass，高于随机 baseline 的 6.4%。同等 N 下，oral top-N oracle 可捕获约 13.2% 的 oral mass，因此模型平均达到 oracle 的约 58.9%。这说明模型 active set 并非随机，但距离 oral 最支持的 top-N 集合仍有明显空间。

时间趋势很清楚：`active_capture_ratio` 从早期约 0.49 上升到晚期约 0.74，`active_topN_overlap` 从约 0.06 上升到约 0.37。也就是说，虽然整体 full-space similarity 很低，但随着学习推进，模型 active set 越来越覆盖 oral top-N 区域。

Condition 3 的 overlap 低于 Condition 1 是可以预期的：hypothesis 空间从 19 扩展到 116，active set 占比更小，oral center 映射到 hypothesis 空间后也更 diffuse。因此，低 full-space alignment 不应被直接解释为 oral 与模型完全不一致，而应理解为 active-set 截断和大 hypothesis 空间共同造成的支持集不匹配。

Condition 3 的结论可以概括为：

> 模型 active set 对 oral-supported hypotheses 有高于随机的覆盖，并且这种覆盖随学习显著增强；但在 116 个 hypothesis 的大空间中，active set 与 oral top-N 的身份重合仍偏低，提示模型搜索/transition 机制只部分覆盖 oral 表征区域。

## 7. 展示建议

如果向其他人展示，建议重点讲三点：

1. Condition 3 的 hypothesis 空间更大，模型 active set 只覆盖约 6.4% 的全空间，因此 full-space alignment 天然严苛。
2. 用 oral top-N oracle 控制同等 N 预算后，模型 active set 的捕获效率约为 59%，明显高于随机，但低于 oracle。
3. 学习过程中 active set 与 oral top-N 的关系明显改善，说明模型 transition 逐渐把 oral-relevant hypotheses 纳入 active set。
