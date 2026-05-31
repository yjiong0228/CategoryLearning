# Subject-wise Hyperparameter Profile Analysis

本报告合并外层 subject-wise hyper-opt、内层 memory grid (`gamma`, `w0`) 与 `oral_center_mode` 对齐指标。
排除被试: 102, 108, 123, 313, 317, 319, 321, 328。

## 样本

| condition   |   n_subjects |
|:------------|-------------:|
| cond1       |           29 |
| cond3       |           27 |

注意: cond1 与 cond3 的内层 memory 网格不同，跨 condition 的数值差异应先作为描述性结果，更稳妥的解释应结合近似最优区域与外层策略差异。

## cond1 画像概览

### Memory 参数

`gamma` 取值分布:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   0.400 |      16 |        0.552 |
|   0.200 |      10 |        0.345 |
|   0.600 |       1 |        0.034 |
|   0.800 |       1 |        0.034 |
|   0.950 |       1 |        0.034 |

`w0` 取值分布:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   0.010 |      13 |        0.448 |
|   0.030 |      11 |        0.379 |
|   0.080 |       3 |        0.103 |
|   0.150 |       1 |        0.034 |
|   0.300 |       1 |        0.034 |

组合画像:

| value                       |   count |   proportion |
|:----------------------------|--------:|-------------:|
| moderate+low_floor          |       8 |        0.276 |
| moderate+moderate_floor     |       7 |        0.241 |
| fast_recent+moderate_floor  |       6 |        0.207 |
| fast_recent+low_floor       |       4 |        0.138 |
| moderate+high_floor         |       2 |        0.069 |
| long_history+low_floor      |       1 |        0.034 |
| long_history+moderate_floor |       1 |        0.034 |

近似最优区域稳定性:

| value    |   count |   proportion |
|:---------|--------:|-------------:|
| moderate |      12 |        0.414 |
| sharp    |      11 |        0.379 |
| broad    |       6 |        0.207 |

### 外层策略与 beta 动态

`strategy_family`:

| value                    |   count |   proportion |
|:-------------------------|--------:|-------------:|
| top_random               |      27 |        0.931 |
| entropy_random_posterior |       1 |        0.034 |
| top_ksim_random          |       1 |        0.034 |

`strategy_signature`:

| value                           |   count |   proportion |
|:--------------------------------|--------:|-------------:|
| top2+random1                    |      11 |        0.379 |
| top3+random1                    |       6 |        0.207 |
| top1+random2                    |       5 |        0.172 |
| top2_p0.7+random1               |       5 |        0.172 |
| randpost_entropy_4+opp_random_2 |       1 |        0.034 |
| top2+ksim1_proto1               |       1 |        0.034 |

`max_active_hypotheses`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|       3 |      16 |        0.552 |
|       4 |      13 |        0.448 |

`init_num`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|       2 |      15 |        0.517 |
|       3 |      14 |        0.483 |

`beta_init`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   3.000 |      12 |        0.414 |
|   0.800 |      10 |        0.345 |
|   1.500 |       7 |        0.241 |

`decrease_rate`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   0.100 |      14 |        0.483 |
|   0.200 |       8 |        0.276 |
|   0.300 |       7 |        0.241 |

`prior_beta_scale`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|       5 |      16 |        0.552 |
|      10 |       7 |        0.241 |
|      15 |       6 |        0.207 |

## cond3 画像概览

### Memory 参数

`gamma` 取值分布:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   0.700 |       8 |        0.296 |
|   0.600 |       6 |        0.222 |
|   0.500 |       4 |        0.148 |
|   0.800 |       3 |        0.111 |
|   0.100 |       2 |        0.074 |
|   0.300 |       1 |        0.037 |
|   0.850 |       1 |        0.037 |
|   0.900 |       1 |        0.037 |

`w0` 取值分布:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   0.010 |       7 |        0.259 |
|   0.015 |       4 |        0.148 |
|   0.050 |       4 |        0.148 |
|   0.075 |       3 |        0.111 |
|   0.200 |       3 |        0.111 |
|   0.020 |       2 |        0.074 |
|   0.100 |       2 |        0.074 |
|   0.005 |       1 |        0.037 |

组合画像:

| value                       |   count |   proportion |
|:----------------------------|--------:|-------------:|
| long_history+low_floor      |       9 |        0.333 |
| moderate+low_floor          |       4 |        0.148 |
| moderate+moderate_floor     |       4 |        0.148 |
| long_history+moderate_floor |       3 |        0.111 |
| long_history+high_floor     |       2 |        0.074 |
| moderate+high_floor         |       2 |        0.074 |
| fast_recent+high_floor      |       1 |        0.037 |
| fast_recent+low_floor       |       1 |        0.037 |

近似最优区域稳定性:

| value    |   count |   proportion |
|:---------|--------:|-------------:|
| broad    |      13 |        0.481 |
| sharp    |       8 |        0.296 |
| moderate |       6 |        0.222 |

### 外层策略与 beta 动态

`strategy_family`:

| value                       |   count |   proportion |
|:----------------------------|--------:|-------------:|
| entropy_random_posterior    |      20 |        0.741 |
| top_ksim_random             |       5 |        0.185 |
| confidence_random_posterior |       1 |        0.037 |
| random_random_posterior     |       1 |        0.037 |

`strategy_signature`:

| value                                               |   count |   proportion |
|:----------------------------------------------------|--------:|-------------:|
| randpost_entropy_7+ksim2_proto2+opp_entropy_4       |      12 |        0.444 |
| randpost_entropy_7+ksim2_proto1+opp_entropy_4       |       6 |        0.222 |
| top3_p0.7+ksim1_proto1+random2                      |       5 |        0.185 |
| randpost_entropy_7+ksim1_proto1+opp_entropy_7       |       2 |        0.074 |
| randpost_confidence_7+ksim1_proto1+opp_confidence_7 |       1 |        0.037 |
| randpost_random_7+ksim1_proto1+opp_random_7         |       1 |        0.037 |

`max_active_hypotheses`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|      10 |      14 |        0.519 |
|      12 |      12 |        0.444 |
|       7 |       1 |        0.037 |

`init_num`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|       7 |      10 |        0.370 |
|      15 |       9 |        0.333 |
|      10 |       8 |        0.296 |

`beta_init`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   1.500 |      11 |        0.407 |
|   0.800 |       8 |        0.296 |
|   3.000 |       8 |        0.296 |

`decrease_rate`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|   0.300 |      17 |        0.630 |
|   0.100 |       6 |        0.222 |
|   0.200 |       4 |        0.148 |

`prior_beta_scale`:

|   value |   count |   proportion |
|--------:|--------:|-------------:|
|      10 |      10 |        0.370 |
|       5 |       9 |        0.333 |
|      15 |       8 |        0.296 |

## 群体层面关联

Cramér's V 使用 condition 内 permutation p 值，主要用于探索性排序。

### cond1 strongest pairwise associations

| feature_x             | feature_y              |   n |   levels_x |   levels_y |   cramers_v |   p_perm |
|:----------------------|:-----------------------|----:|-----------:|-----------:|------------:|---------:|
| max_active_hypotheses | memory_identifiability |  29 |          2 |          3 |       0.538 |    0.014 |
| max_active_hypotheses | prior_beta_scale       |  29 |          2 |          3 |       0.515 |    0.023 |
| init_num              | w0_exact               |  29 |          2 |          5 |       0.562 |    0.026 |
| gamma_exact           | memory_identifiability |  29 |          5 |          3 |       0.495 |    0.027 |
| decrease_rate         | memory_identifiability |  29 |          3 |          3 |       0.409 |    0.053 |
| beta_init             | gamma_exact            |  29 |          3 |          5 |       0.449 |    0.084 |
| strategy_signature    | gamma_exact            |  29 |          6 |          5 |       0.591 |    0.093 |
| w0_exact              | memory_identifiability |  29 |          5 |          3 |       0.436 |    0.152 |
| prior_beta_scale      | memory_identifiability |  29 |          3 |          3 |       0.320 |    0.208 |
| init_num              | memory_identifiability |  29 |          2 |          3 |       0.345 |    0.232 |
| beta_init             | w0_exact               |  29 |          3 |          5 |       0.411 |    0.264 |
| max_active_hypotheses | init_num               |  29 |          2 |          2 |       0.239 |    0.281 |

### cond3 strongest pairwise associations

| feature_x             | feature_y              |   n |   levels_x |   levels_y |   cramers_v |   p_perm |
|:----------------------|:-----------------------|----:|-----------:|-----------:|------------:|---------:|
| decrease_rate         | memory_identifiability |  27 |          3 |          3 |       0.450 |    0.025 |
| init_num              | prior_beta_scale       |  27 |          3 |          3 |       0.419 |    0.049 |
| strategy_signature    | gamma_exact            |  27 |          6 |          9 |       0.654 |    0.109 |
| init_num              | memory_identifiability |  27 |          3 |          3 |       0.360 |    0.137 |
| gamma_exact           | w0_exact               |  27 |          9 |          9 |       0.601 |    0.155 |
| max_active_hypotheses | decrease_rate          |  27 |          3 |          3 |       0.357 |    0.156 |
| decrease_rate         | w0_exact               |  27 |          3 |          9 |       0.626 |    0.157 |
| beta_init             | gamma_exact            |  27 |          3 |          9 |       0.610 |    0.157 |
| prior_beta_scale      | memory_identifiability |  27 |          3 |          3 |       0.335 |    0.202 |
| init_num              | gamma_exact            |  27 |          3 |          9 |       0.602 |    0.206 |
| decrease_rate         | prior_beta_scale       |  27 |          3 |          3 |       0.317 |    0.271 |
| w0_exact              | memory_identifiability |  27 |          9 |          3 |       0.586 |    0.308 |

## Association Rules

规则格式为 `A -> B`，报告 support、confidence、lift；仅保留 support_count >= 3。

### cond1

| antecedent                           | consequent                      |   support_count |   support |   confidence |   lift |
|:-------------------------------------|:--------------------------------|----------------:|----------:|-------------:|-------:|
| strategy_signature=top1+random2      | decrease_rate=0.3               |               3 |     0.103 |        0.600 |  2.486 |
| strategy_signature=top2_p0.7+random1 | decrease_rate=0.2               |               3 |     0.103 |        0.600 |  2.175 |
| w0_exact=0.08                        | init_num=2                      |               3 |     0.103 |        1.000 |  1.933 |
| prior_beta_scale=10                  | max_active_hypotheses=3         |               7 |     0.241 |        1.000 |  1.812 |
| beta_init=1.5                        | gamma_exact=0.4                 |               7 |     0.241 |        1.000 |  1.812 |
| memory_identifiability=broad         | max_active_hypotheses=3         |               6 |     0.207 |        1.000 |  1.812 |
| strategy_signature=top1+random2      | w0_exact=0.01                   |               4 |     0.138 |        0.800 |  1.785 |
| decrease_rate=0.3                    | memory_identifiability=moderate |               5 |     0.172 |        0.714 |  1.726 |
| strategy_signature=top1+random2      | init_num=3                      |               4 |     0.138 |        0.800 |  1.657 |
| max_active_hypotheses=4              | memory_identifiability=sharp    |               8 |     0.276 |        0.615 |  1.622 |
| w0_exact=0.01                        | memory_identifiability=sharp    |               8 |     0.276 |        0.615 |  1.622 |
| memory_identifiability=sharp         | max_active_hypotheses=4         |               8 |     0.276 |        0.727 |  1.622 |
| memory_identifiability=sharp         | w0_exact=0.01                   |               8 |     0.276 |        0.727 |  1.622 |
| memory_identifiability=broad         | init_num=2                      |               5 |     0.172 |        0.833 |  1.611 |
| prior_beta_scale=15                  | memory_identifiability=moderate |               4 |     0.138 |        0.667 |  1.611 |

### cond3

| antecedent                                                       | consequent                                                       |   support_count |   support |   confidence |   lift |
|:-----------------------------------------------------------------|:-----------------------------------------------------------------|----------------:|----------:|-------------:|-------:|
| decrease_rate=0.2                                                | memory_identifiability=moderate                                  |               3 |     0.111 |        0.750 |  3.375 |
| gamma_exact=0.7                                                  | w0_exact=0.01                                                    |               6 |     0.222 |        0.750 |  2.893 |
| w0_exact=0.01                                                    | gamma_exact=0.7                                                  |               6 |     0.222 |        0.857 |  2.893 |
| strategy_signature=top3_p0.7+ksim1_proto1+random2                | memory_identifiability=moderate                                  |               3 |     0.111 |        0.600 |  2.700 |
| decrease_rate=0.2                                                | prior_beta_scale=15                                              |               3 |     0.111 |        0.750 |  2.531 |
| gamma_exact=0.8                                                  | beta_init=1.5                                                    |               3 |     0.111 |        1.000 |  2.455 |
| gamma_exact=0.8                                                  | strategy_signature=randpost_entropy_7+ksim2_proto2+opp_entropy_4 |               3 |     0.111 |        1.000 |  2.250 |
| memory_identifiability=moderate                                  | init_num=10                                                      |               4 |     0.148 |        0.667 |  2.250 |
| memory_identifiability=moderate                                  | prior_beta_scale=15                                              |               4 |     0.148 |        0.667 |  2.250 |
| w0_exact=0.2                                                     | memory_identifiability=broad                                     |               3 |     0.111 |        1.000 |  2.077 |
| gamma_exact=0.5                                                  | init_num=7                                                       |               3 |     0.111 |        0.750 |  2.025 |
| strategy_signature=top3_p0.7+ksim1_proto1+random2                | beta_init=0.8                                                    |               3 |     0.111 |        0.600 |  2.025 |
| strategy_signature=randpost_entropy_7+ksim2_proto1+opp_entropy_4 | prior_beta_scale=5                                               |               4 |     0.148 |        0.667 |  2.000 |
| gamma_exact=0.6                                                  | init_num=15                                                      |               4 |     0.148 |        0.667 |  2.000 |
| gamma_exact=0.5                                                  | max_active_hypotheses=10                                         |               4 |     0.148 |        1.000 |  1.929 |

## Oral Alignment 探索

下面列出超参与 oral/model alignment 指标的 Spearman 相关中 p 值最小的条目。

### cond1

| predictor        | metric                                   |   n |   spearman_rho |   p_value |
|:-----------------|:-----------------------------------------|----:|---------------:|----------:|
| prior_beta_scale | center_target_pearson_r_union_topn       |  26 |         -0.662 |     0.000 |
| prior_beta_scale | center_target_pearson_r_active           |  26 |         -0.623 |     0.001 |
| prior_beta_scale | center_target_spearman_rho_union_topn    |  26 |         -0.603 |     0.001 |
| prior_beta_scale | center_target_spearman_rho_active        |  26 |         -0.562 |     0.003 |
| prior_beta_scale | center_target_spearman_rho_full          |  26 |         -0.465 |     0.017 |
| prior_beta_scale | center_hit_cohen_kappa                   |  29 |         -0.427 |     0.021 |
| init_num         | center_hit_phi_correlation               |  24 |         -0.464 |     0.022 |
| prior_beta_scale | center_hit_phi_correlation               |  24 |         -0.453 |     0.026 |
| w0               | center_distribution_js_active            |  29 |         -0.401 |     0.031 |
| init_num         | center_coverage_active_capture_ratio     |  29 |         -0.396 |     0.034 |
| gamma            | center_distribution_js_active            |  29 |         -0.390 |     0.036 |
| prior_beta_scale | center_target_pearson_r_full             |  26 |         -0.404 |     0.041 |
| prior_beta_scale | center_target_cosine_similarity_active   |  26 |         -0.392 |     0.048 |
| init_num         | center_target_oral_target_mass_mean_full |  29 |         -0.363 |     0.053 |
| init_num         | center_oral_oral_based_similarity        |  29 |         -0.355 |     0.059 |

### cond3

| predictor     | metric                                     |   n |   spearman_rho |   p_value |
|:--------------|:-------------------------------------------|----:|---------------:|----------:|
| w0            | center_distribution_js_full                |  27 |         -0.515 |     0.006 |
| gamma         | center_hit_oral_topn_mass_mean             |  27 |          0.515 |     0.006 |
| gamma         | center_coverage_oracle_topn_oral_mass      |  27 |          0.515 |     0.006 |
| gamma         | center_coverage_random_expected_mass       |  27 |          0.500 |     0.008 |
| gamma         | center_coverage_n_active                   |  27 |          0.500 |     0.008 |
| gamma         | center_coverage_active_fraction            |  27 |          0.500 |     0.008 |
| gamma         | center_hit_active_set_size_mean            |  27 |          0.496 |     0.009 |
| w0            | center_hit_cohen_kappa                     |  27 |          0.495 |     0.009 |
| beta_init     | center_hit_active_set_size_mean            |  27 |          0.494 |     0.009 |
| beta_init     | center_coverage_random_expected_mass       |  27 |          0.494 |     0.009 |
| beta_init     | center_coverage_n_active                   |  27 |          0.494 |     0.009 |
| beta_init     | center_coverage_active_fraction            |  27 |          0.494 |     0.009 |
| w0            | center_target_pearson_r_active             |  27 |          0.491 |     0.009 |
| w0            | center_hit_phi_correlation                 |  27 |          0.483 |     0.011 |
| decrease_rate | center_target_cosine_similarity_union_topn |  27 |         -0.479 |     0.011 |

## 综合认知解释

下面的解释把模型参数当成认知加工的不同层次: `gamma/w0` 描述经验证据在时间上的保留方式，`strategy_signature/max_active_hypotheses/init_num` 描述被试可能在假设空间里如何搜索规则，`beta_init/decrease_rate/prior_beta_scale` 描述规则信心如何初始化和被反馈更新，oral alignment 指标则作为外显报告对这些隐变量解释的约束。

### 1. 条件差异: cond1 更像近因驱动的规则利用，cond3 更像不确定性驱动的结构搜索

`cond1` 的主导画像是低到中等 `gamma` 加低 `w0`: `gamma=0.2/0.4` 覆盖 26/29 个被试，`w0=0.01/0.03` 覆盖 24/29 个被试。这意味着模型通常只需要较强地依赖近期反馈，远期试次保留很弱，就能解释被试行为。外层策略也高度一致，27/29 个被试属于 `top_random`: 保留当前 posterior 较高的少数候选规则，再加一点随机探索。认知上，这更像一个相对紧凑的规则空间: 被试主要在少数高可信规则附近更新，而不是持续广泛搜索。
`cond3` 的画像明显不同。`gamma=0.6/0.7/0.8` 占 17/27，最大组合是 `long_history+low_floor`。外层策略中 20/27 是 `entropy_random_posterior`，并且多数还组合 `ksimilar_centers`。这说明 cond3 中行为更像是在较大的、结构更复杂的假设空间里工作: 被试需要保留更长历史，同时在不确定性较高时补充候选假设，并利用原型或相似中心来组织规则。
因此，两个 condition 的差异不应只解释为记忆强弱差异。更合理的图景是: cond1 主要负荷在快速利用和局部更新，cond3 同时负荷在历史整合、候选假设维护和结构化搜索。

### 2. Memory 参数: `gamma` 是历史整合，`w0` 是远期经验的底线影响

`gamma` 越高，越说明较早试次仍能影响当前似然或规则信念；`w0` 越高，越说明即使很久以前的经验也不会衰减到接近零。这两个参数的认知含义不同: 高 `gamma` 是连续的历史整合，高 `w0` 更像远期经验的背景偏置或稳定先验。
在 cond1 中，低 `w0` 与 `sharp` identifiability 共现较明显: association rule 显示 `memory_identifiability=sharp -> w0=0.01` 的 confidence 为 0.727，lift 为 1.622。这说明对一部分 cond1 被试，模型能比较明确地识别出一个近因主导、远期记忆底线很低的加工方式。
在 cond3 中，`gamma=0.7` 与 `w0=0.01` 是最清楚的 memory 共现规则: `gamma=0.7 -> w0=0.01` 的 confidence 为 0.750，反向 `w0=0.01 -> gamma=0.7` 的 confidence 为 0.857，lift 均约 2.893。这个组合很有认知意义: 被试整合较长历史，但远期经验的最低权重仍很低。换句话说，他们不是简单地把所有旧经验都平均保留，而是保留一条较长的证据轨迹，同时允许很旧的信息逐渐退出主导地位。

### 3. Identifiability 本身也是认知信号: broad 不是噪声，而可能是多策略等价

`memory_identifiability` 衡量的是误差面是否尖锐。`sharp` 表示只有少数组合能解释行为，`broad` 表示很多 memory 组合都差不多好。这个指标不只是技术诊断，也能反映行为是否足够约束某种单一机制。
cond1 中 `broad` 只有 6/29，但它和 `max_active_hypotheses=3` 强关联: Cramer's V=0.538, permutation p=0.014；列联表中 active=3 包含全部 6 个 broad，被试 active=4 时没有 broad。一个可能解释是，较小 active set 会把行为压缩到少数候选规则上，导致不同记忆衰减曲线都能产生类似预测；而 active=4 时模型保留更多竞争规则，行为中的细微差异反而更能揭示具体 memory profile。
cond3 中 `broad` 高达 13/27，应更加谨慎。这里 broad 不一定是坏拟合，而可能说明被试在复杂任务里采用了混合加工: 有时依赖长期结构，有时依赖近期反馈，有时受显性策略或局部原型吸引。单个 `gamma/w0` 点只是这个混合过程的一个等价投影。

### 4. 假设空间搜索: active set、init_num 与 strategy 反映探索-利用权衡

cond1 的 `top_random` 策略说明多数被试像是在做窄范围的 exploitation: posterior 高的规则被持续保留，随机候选只提供少量探索。association rules 中 `top1+random2 -> decrease_rate=0.3` 和 `top2_p0.7+random1 -> decrease_rate=0.2` 提示搜索策略和反馈敏感性是耦合的: 当模型更依赖极少数 top 规则时，错误反馈后需要更强地惩罚不一致规则，才能跟上被试的转向。
cond1 里 `init_num` 与 `w0` 的关联也值得注意: Cramer's V=0.562, p=0.026。`init_num=2` 更多对应 `w0=0.03/0.08`，而 `init_num=3` 更多对应 `w0=0.01`。这可以理解为一种替代关系: 初始假设更少时，模型需要给旧经验保留一点底线影响来稳定行为；初始探索稍多时，模型可以通过候选假设本身吸收不确定性，不必让远期记忆维持较高权重。
cond3 的搜索机制更开放。主要策略 `randpost_entropy_7+ksim...` 表示候选补充由 entropy 驱动，且通过 `ksimilar_centers` 组织相似规则。这和人类在复杂分类任务中的一种常见加工方式相符: 不是枚举所有规则，而是在不确定时围绕当前可解释的原型或相似中心扩展候选空间。虽然 `strategy_signature ~ gamma` 的 permutation p=0.109 未达到常规阈值，但 Cramer's V=0.654，模式上显示不同搜索策略倾向对应不同历史整合程度，值得作为后续假设。

### 5. Beta 和反馈动态: 信心不是单独参数，而是和搜索/记忆共同塑形

`beta_init` 可以理解为新候选规则一开始的决策锐度，`prior_beta_scale` 是 prior 对初始锐度的放大，`decrease_rate` 是错误反馈后对不一致规则的惩罚强度。
cond1 中 `max_active_hypotheses ~ prior_beta_scale` 关联较强，V=0.515, p=0.023；association rule 显示 `prior_beta_scale=10 -> max_active_hypotheses=3` 的 confidence 为 1.000。这提示当 active set 较小的时候，模型更依赖 prior 来快速区分候选规则；当 active set 较大时，则可以通过保留更多候选来表达不确定性。也就是说，人类可能有两种等价方式处理不确定性: 一种是少候选但强信心筛选，另一种是多候选但让后续反馈慢慢筛。
cond3 中 `decrease_rate ~ memory_identifiability` 是最强 pairwise 结果之一，V=0.450, p=0.025。`decrease_rate=0.2 -> memory_identifiability=moderate` 的 confidence 为 0.750，lift 为 3.375；`decrease_rate=0.1` 则更容易落在 broad。认知上可以理解为: 过弱的错误惩罚让多种 memory 轨迹都能解释行为，中等惩罚反而让模型更容易识别出较稳定的更新模式；而强惩罚 0.3 在 cond3 中同时出现在 broad 和 sharp，说明有些被试对反馈非常敏感，但这种敏感性可以服务于不同策略。

### 6. Oral alignment: 哪些模型机制更像被试自己说出来的策略

oral alignment 结果很关键，因为它帮助区分“能拟合选择”的机制和“接近被试显性报告”的机制。如果一个超参只改善选择拟合，但和 oral report 不一致，它可能是模型补偿项；如果它同时提高 hit agreement、active capture 或 target correlation，则更可能反映被试真实使用的策略。
cond1 中最稳定的 oral 关联是 `prior_beta_scale` 的负相关: 它与 active/union_topn 空间中的 target Pearson/Spearman 均为负相关，其中 `center_target_pearson_r_union_topn` rho=-0.662, p<0.001，`center_target_pearson_r_active` rho=-0.623, p=0.001；它也与 `cohen_kappa` 和 `phi_correlation` 负相关。这说明 prior 放大越强，模型的 target belief 越不贴近被试口述中心所指向的假设。认知解释上，cond1 被试的显性策略可能更依赖近期可观察反馈，而不是模型内部的强 prior 初始化；过强 prior 虽能帮助选择拟合，却可能把模型推向被试没有口头报告的隐性规则偏置。
cond1 中 `init_num` 与 oral alignment 也多为负相关: `init_num` 与 `center_hit_phi_correlation` rho=-0.464, p=0.022，与 active capture rho=-0.396, p=0.034。这个结果支持一个简单解释: cond1 的显性策略比较集中，初始候选过多会让模型 active set 包含更多被试没有报告的规则，从而降低和口述策略的重合。
cond1 的 `gamma` 和 `w0` 都与 active-space JS similarity 负相关。也就是说，在这个条件下，历史整合越强或远期底线越高，模型在 active set 内的分布形状越不像 oral distribution。这进一步支持 cond1 是近因主导、显性策略较局部的任务状态。
cond3 的 oral 结果呈现另一种机制。`w0` 与 hit/kappa/phi/target correlation 多为正相关: `w0 ~ cohen_kappa` rho=0.495, p=0.009，`w0 ~ phi_correlation` rho=0.483, p=0.011，`w0 ~ target_pearson_r_active` rho=0.491, p=0.009。这说明在复杂任务里，保留一定远期经验底线反而更接近被试口述策略。一种解释是，被试在 cond3 中会把较早形成的结构性假设维持为背景框架，即使近期反馈有波动，也不会完全丢掉。
同时，cond3 的 `w0` 与 full-space JS similarity 为负相关 rho=-0.515, p=0.006。这看似矛盾，但其实很有信息量: 高 `w0` 让模型更容易在 target hit 层面和口述报告一致，但它不一定让整个假设空间分布形状一致。换言之，被试口述可能抓住了目标或局部结构，而模型的全局 posterior 仍包含许多未被口述的竞争规则。
cond3 中 `gamma` 与 oral_topn_mass、active_oral_mass、active_set_size 等指标为正相关，例如 `gamma ~ oral_topn_mass_mean` rho=0.515, p=0.006，`gamma ~ active_set_size_mean` rho=0.496, p=0.009。这提示长期历史整合会让模型保留更大的候选集合，并使 oral top-N 所覆盖的质量更高。但 `gamma ~ center_distribution_js_active` 为负相关 rho=-0.446, p=0.020，说明高 gamma 捕获了更多口述相关假设，却未必在 active set 内按同样比例分配概率。认知上，这像是“保持多个可解释结构”而不是“锁定一个口述规则”。

### 7. 可以形成的被试加工类型

基于这些结果，可以先把被试粗分为几类，而不是逐个解释孤立参数。
第一类是 cond1 中的近因利用型: `top_random`、低 `w0`、`sharp/moderate` memory identifiability。这类被试可能主要维护少数当前有效规则，近期反馈快速改变 posterior，口述策略也更集中。
第二类是 cond1 中的压缩/等价型: `max_active_hypotheses=3` 且 `broad`。这些被试的行为可由多种 memory 曲线解释，可能说明他们的选择主要由少数规则切换或局部启发式驱动，而不是由稳定的时间衰减机制唯一决定。
第三类是 cond3 中的不确定性结构搜索型: `entropy_random_posterior + ksimilar_centers`，较大 active set，中高 `gamma`。这类被试可能持续维护多个候选分类结构，并在不确定时围绕相似中心扩展规则。
第四类是 cond3 中的长期轨迹低底线型: 典型组合为 `gamma=0.7,w0=0.01`。他们整合较长历史，但不会让很旧的经验保持高权重，适合解释一种“有历史感但仍能更新”的策略。
第五类是 cond3 中的远期框架保留型: 较高 `w0`，oral hit/kappa/phi 更好，但 full-space JS 更差。这类被试可能有一个稳定的显性结构框架，口述时能命中目标，但模型内部仍需要许多竞争假设来解释完整选择轨迹。

### 8. 解释限制和下一步分析

这些结果是探索性的。pairwise association 使用 permutation p 值，但没有作为确认性假设做多重比较校正；oral alignment 的 Spearman p 值也应主要作为排序线索，而不是最终显著性结论。
此外，cond1 和 cond3 的 memory grid 不同，所以跨条件比较应看整体画像和分箱，不应过度比较某一个精确取值。尤其是 `broad` 被试，不适合写成“这个人就是 gamma=某值”，更适合写成“这个人的行为不能唯一约束记忆衰减机制”。
下一步最有价值的分析是: 对上述五类被试分别画 trial-level posterior、active-set size、oral hit trace 和 feedback 后 beta 更新，看这些机制是否真的在时间序列上表现为不同的学习阶段。若这些类别在 trial-level 动态中也分离，就可以更有把握地把它们解释为人类任务加工机制，而不仅是参数共现。

## 输出图

- `figures/memory_pair_heatmap_cond1.png`: cond1 的 `gamma x w0` 组合频数。
- `figures/memory_pair_heatmap_cond3.png`: cond3 的 `gamma x w0` 组合频数。
- `figures/gamma_w0_scatter.png`: 两个 condition 的 memory 组合散点，点大小表示人数。
- `figures/memory_identifiability_by_condition.png`: 近似最优区域稳定性分布。
- `figures/strategy_family_by_condition.png`: 外层策略家族分布。
- `subject_profile_table.md`: 便于人工浏览的被试级精简画像表。
