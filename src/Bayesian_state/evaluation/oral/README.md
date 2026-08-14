# 口述报告评价

本目录将中心点和区域口述报告映射到共享 hypothesis space，并从分布、口述表征、目标、命中
和覆盖五个角度评价模型对齐程度。所有代码都属于后处理，不改变模型状态。

| 文件 | 职责 |
|---|---|
| `mapping.py` | 解析中心/区域报告，计算区域与各 hypothesis 的几何重合 |
| `scoring.py` | 构造口述与模型分布，计算五类对齐指标；不写盘、不作图 |
| `reporting.py` | 汇总统计、CSV 序列化和组/被试级绘图 |
| `alignment.py` | 组合 scoring 与 reporting，并提供稳定的 `OralModelAlignmentMixin` |
| `__init__.py` | 重导出公开接口 |

## Oral hypothesis distribution

五类 alignment 名称保持不变。依赖概率分布的 distribution/target/hit/coverage 默认共享
category-state oral belief；`oral-based` 仍直接比较当前 trial 的口述几何表征。首次有效报告之前
的状态保持全 `NaN`，不会被替换成 uniform distribution。

### Latest-by-category state（默认）

每名被试维护每个 category 最近一次有效口述报告。trial `t` 只替换当前报告 category `k_t`
的 likelihood；其他 category 保留上一状态，缺失或无法解析的当前报告不清空已有状态。设
`O_t` 为截至当前已有有效状态的 category 集合，则主分布为：

```text
log L_state,t(h) = sum_(k in O_t) log L(y_latest,k | h, k)
q_state,t(h) = pi(h) L_state,t(h) / sum_j pi(j) L_state,t(j)
```

`pi(h)` 是只应用一次的 uniform hypothesis prior。同一 category 的新报告替换旧报告，而不是
与所有历史报告重复相乘；因此连续重复同一句话不会被误当成多份独立证据。只有一个 category
可用时，state distribution 与该报告的 instantaneous distribution 完全相同；观察到更多
category 后，它们共同约束完整 hypothesis。

`oral_mass` 保存上述 category-state 分布，`instantaneous_oral_mass` 保存当前 trial 的单-category
分布。`valid_oral` 表示 state 是否可用，`valid_oral_report` 表示当前 trial 是否产生有效新报告。
CLI 默认 `--oral-state-mode latest_by_category`；使用
`--oral-state-mode instantaneous` 可复现旧的 current-report-only 行为。

### Center mode

对单个 trial 的口述中心 `y_t`、实际选择类别 `k_t` 和 hypothesis `h`，每个类别连通分量的
自动几何中心 `c_(h,k,m)` 定义一个各向同性 Gaussian 报告似然。一个 hypothesis 内的分量似然
等权平均，避免多分量 hypothesis 因中心数量更多而自动获得更多质量：

```text
L_t(h) = mean_m Normal(y_t; c_(h,k,m), sigma_oral^2 I)
q_t(h) = L_t(h) / sum_j L_t(j)
```

单-category `q_t` 使用独立于待评价模型的 uniform hypothesis prior，在当前 fixed-label
hypothesis space 上严格归一化为 1。category-state 联合时先恢复每个 category 的 likelihood，
求积后只应用一次 prior。默认 `sigma_oral=0.10`，对所有 trial 和 subject 固定；CLI 可用
`--oral-center-sigma` 显式覆盖。禁止根据每个 trial 的 hypothesis 距离重新估计 temperature，
因为那会消除绝对距离尺度并强制每个 trial 产生近似宽度的分布。

### Region mode

Region mode 先计算口述 region 与每个 hypothesis category region 的 IoU，再把
`1 - IoU` 当作 mismatch，通过固定的 `--oral-region-temperature` 转换和归一化。默认尺度为
`0.10`，同样不做逐 trial 自适应。

### Provenance and diagnostics

`oral_mass_probabilities.npz` 保存完整分布以及：

- encoder version、distribution method、state aggregation method、uniform prior；
- center sigma / region temperature；
- hypothesis-space version 与 signature；
- state 的已观察 category 数和 bit mask，以及本 trial 更新 category/更新是否有效；
- state 与 instantaneous report 各自的距离、log evidence、entropy、effective hypothesis count、
  最大概率和绝对 fit。

同样的信息以长表写入 `oral_mass_diagnostics.csv`，并传播到依赖 oral mass 的
distribution/target/hit/coverage trial CSV。`oral-based` 直接在口述几何空间评价，不依赖上述
概率编码，因此切换 state aggregation 不改变其科学定义。

### Target-based PF interval

Particle-filter target alignment 不再只读取 representative repeat。评价器从每个被试的 raw-run
stream 读取所有可用 PF repeats 的 `marginal_prior`，先在相同 comparison space 内计算并平均
trialwise target mass，以降低有限粒子的 Monte-Carlo 误差。随后按该 target mass 抽取 Bernoulli
latent target/non-target 序列，并使用与 `basic/accuracy_band.png` 相同的 rolling-quantile 协议生成
50% 与 90% pointwise interval。该色带表示固定参数、条件于真实观测历史的 latent target
occupancy，而不是 target 概率估计值的置信区间，也不是 PF repeats 之间的误差带。

`target_based_alignment_trial_metrics.csv` 保留未平滑的 expected target mass、PF repeat 数及
repeat SD，并增加 rolling expected、q05/q25/q50/q75/q95、抽样数、固定 seed、window 和 band
semantics。抽样数与 seed 复用统一 CLI 的 `--accuracy-band-draws` 和
`--accuracy-band-seed`，使 behavioral 与 latent-state 两类区间采用相同可复现设置。
Subject-wise target 图同时复用 `basic/accuracy_band.png` 的8列布局、面板尺寸、字体、线宽、
蓝色区间带、橙色模型期望、黑色观测对照、首面板 legend、水平网格与两行标题层级；仅指标名称
由 behavioral accuracy 替换为 latent target occupancy / Oral target probability。

Trajectory backend 使用同一套 raw-run 语义但不伪装成 PF：评价器读取所有可用 repeats 的
`state_log.prior`，逐条计算 rolling target mass，再跨 realized trajectories 取 50%/90% pointwise
quantile band。这与 `basic/accuracy_band.png` 的 trajectory ensemble 来源一致；图和 trial CSV 会明确
记录 `model_inference_backend=trajectory`、trajectory run 数及
`observed_history_conditional_trajectory_repeat_target_mass`，不会写成 PF draws 或 PF runs。
