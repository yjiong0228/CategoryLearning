# 模型评价

本目录负责读取已经完成的 simulation 输出，计算或复核指标，并生成状态轨迹、行为 PPC 和
oral/model alignment 图表。它不参与模型拟合，也不改变 hyperparameter selection。
它可以用冻结参数运行明确的评价协议，但不得根据评价结果重新搜索参数或覆盖冻结配置。

## 文件

| 文件 | 职责 |
|---|---|
| `evaluator.py` | `ModelEvaluator` 通用评价门面：accuracy、choice Brier、posterior/prior、beta、行为 PPC 和 trajectory-rank 图表 |
| `transition.py` | 仅在相应日志存在时使用的 dynamic-discrete、dynamic-continuous 与 active-set 诊断 |
| `fft_clustering.py` | 保存后的 run-level 轨迹 FFT 聚类 |
| `oral/mapping.py` | oral center/region 到共享 hypothesis space 的映射 |
| `oral/scoring.py` | latest-by-category oral state、oral/model 分布构造及五类对齐计算 |
| `oral/reporting.py` | 对齐结果汇总、CSV 保存与绘图 |
| `oral/alignment.py` | 组合上述能力并保持 `OralModelAlignmentMixin` 公共接口 |
| `particle_filter/summary.py` | PF 条件行为预测区间、ESS 与粒子边际 active-state 诊断 |
| `particle_filter/strategy.py` | PF controller strategy 反事实审计 |
| `particle_filter/choice_transmission.py` | 相同粒子状态下的 hypothesis/readout 反事实 |
| `particle_filter/residuals.py` | 真实选择的持续性、局部阶段和因果 residual-state 诊断 |
| `__init__.py` | 轻量公共入口；首次访问 `ModelEvaluator` 时才导入绘图栈 |

`ModelEvaluator` 组合 `TransitionEvaluationMixin` 与 `OralModelAlignmentMixin`，所以公共行为评价、
transition 特异诊断与口述规则对齐仍共享同一个结果读取上下文。调用方不需要按模型实例化不同的
evaluator。

## 正式入口

```bash
python -m src.Bayesian_state.run_model_evaluation \
  --input-dir results/state-based-simulation/pmh/cond1_0806
```

输入目录通常包含：

```text
subjects/subject_<id>.json
cache/subject_<id>_raw_runs.gz   # optional
```

`run_model_evaluation.py` 将 subject JSON 规范化为 `ModelEvaluator` 使用的 result mapping，并在
`<input-dir>/evaluation/` 下写图、CSV 和 `evaluation_manifest.json`。

## 日志依赖

- 基础 accuracy/Brier 图只需要 representative-run metrics。
- posterior/prior/beta 图需要相应 state log。
- dynamic-discrete profile 需要 `state_probabilities` 或 `policy_probabilities`；active-set 图需要
  `active_total`、`strategies` 或 `profile_policy`。CLI 按这些日志字段判断能力，不按模型类名判断。
- dynamic-continuous 的 `predictive_m`/`transition_rate`、`predictive_g`/`search_range`、
  feedback surprise/uncertainty 会被识别为独立能力。trajectory backend 保留原始控制轨迹；
  particle backend 改用下述 continuous strategy profile，并继续生成反馈信号图。
- particle state log 额外生成归一化 pre/post-choice ESS、重采样事件和
  `marginal_active_probability` heatmap；顺序留出的切分点会画在这些 trial-level 图上。
- trajectory rank、posterior rank 和 behavior PPC 的完整 run 分布需要 `raw_runs_ref`。
- oral alignment 还需要 Task2/oral 数据与相同的 partition 定义。

Center oral report 默认通过固定 `sigma=0.10` 的 Gaussian component-mixture likelihood 映射为
完整 hypothesis distribution；所有 hypothesis 使用 uniform encoder prior，再归一化为 1。
`--oral-center-sigma` 可覆盖这个跨 trial 固定的测量尺度。Region mode 对 `1-IoU` 使用固定
`--oral-region-temperature`。两种模式均不再从每个 trial 的候选距离自适应 temperature；encoder
版本、尺度、hypothesis-space signature 及绝对 fit 诊断写入 NPZ/CSV。

particle backend 保存的是 `marginal_prior`、`marginal_active_probability` 和 ESS/transition
诊断，不是每个粒子的 posterior 轨迹。结果 adapter 将 `marginal_prior` 显式映射为通用
`prior_log`，并以 `state_distribution_kind: particle_marginal` 标记其统计含义；它不会把该对象
冒充 posterior。因而粒子结果可以复用 prior 图，并使用专门的 marginal active/ESS 图；不存在
真实 posterior 或 beta 日志时，manifest 将对应步骤记为 `not_applicable`。

## 准确率区间带的推理后端语义

统一 CLI 只生成一套 PNG 主图 `basic/accuracy_band.png` 和
`basic/accuracy_band_summary.csv`，不再重复写逐被试 `predictive_accuracy_band/` 目录。

- `particle_filter`：先对相同观测历史下的 PF repeats 求 trialwise 正确概率均值，再按这些概率
  抽取 Bernoulli 行为序列并应用相同 rolling window。图中的 50%/90% 色带是固定拟合参数、
  条件于真实观测历史的 pointwise behavioral predictive interval；不是 4 次 PF 重跑之间的
  Monte-Carlo 数值误差，也不是 autonomous rollout。
- `trajectory`：保留跨单条 latent-trajectory runs 的 ensemble band，并在标题和图例中显式标为
  trajectory band，避免与 PF behavioral interval 混淆。

PF 行为抽样数量和固定随机种子可通过 `--accuracy-band-draws` 与
`--accuracy-band-seed` 设置，并记录在 summary CSV 中。

`oral_alignment_*/target_based_alignment` 对 PF 使用同构但不同观测量的抽样：先跨 raw-run
PF repeats 平均 trialwise target-hypothesis marginal mass，再抽取 latent target/non-target
Bernoulli 序列并计算 rolling 50%/90% interval。其中心线是 expected target mass，色带是
observed-history-conditional latent target occupancy；它不是行为区间，也不是粒子数值误差。
target interval 复用上述 draws/seed CLI 设置，并将完整分位数与 provenance 写入 trial CSV。

## 连续策略概况

PF dynamic-continuous 结果的 canonical 策略图是 `basic/dynamic_strategy_profile.png`，配套
`basic/dynamic_strategy_profile_summary.csv`。它不画 hypothesis 概率，也不把 continuous
controller 强行分箱为离散状态，而是以理论“至少替换一条规则”的概率 `E_t` 分解：

```text
exploit = 1 - E_t
local exploration = E_t * (1 - g_t)
global exploration = E_t * g_t
```

新结果优先使用 current-choice 之前的粒子权重直接聚合上述三项；它们逐 trial 加和为 1。旧结果
若尚未保存 pre-choice 字段，则用 filtered `m_t`、`g_t` 和被试容量推导兼容估计，并在图注与
summary 的 `source_semantics` 中明确标记。主图叠加被试与 PF 滚动准确率、实际替换诊断；低表现
与掌握期只按之前一个窗口的观察准确率定义，不使用当前 trial 信息。PF continuous 不再重复生成
`dynamic_continuous_controls.png`；原始 controls 数值仍保存在 state log，反馈驱动信号继续由
`dynamic_continuous_signals.png` 诊断。

当结果来自 `failure_accumulator_v2` 时，strategy summary 还报告低表现/掌握期的
`predictive_failure_pressure` 与 `predictive_mastery_evidence`。这些是 pre-choice PF 边缘状态，
用于检验“连续失败触发探索、稳定正确恢复利用”，不改变上述 phase 的因果定义。若启用 v2b，
同一 summary 还报告 `predictive_prior_reset_strength` 与
`predictive_prior_reset_mass_shift` 的低表现/掌握期均值；前者是实际发生 replacement 时采用的
混合系数，后者是 prior 被移动的总变差质量。

若启用 misconception capture，同一 `basic/dynamic_strategy_profile.png` 会叠加
`Wrong-rule lock-in` 虚线，表示 current-choice 之前处于 capture dwell 的加权粒子比例；
summary CSV 同时报告它在低表现期、掌握期和全程的均值。该线是 latent strategy state，
不是 hypothesis 编号的平均，也不会另外生成重复图。

### 策略贡献审计

只有显式传入 `--strategy-audit-config` 时，统一 evaluation CLI 才会运行策略贡献审计，并把
结果写入 `evaluation/strategy_audit/`。审计用相同 PF seeds 比较三种条件：原 fitted
dynamic controller、保持平均探索事件率与 local/global 比例不变的
`mean_matched_static`、以及保留 fitted baseline `m/g` 但关闭 controller 的
`controller_off`。它不重新拟合参数，也不覆盖 simulation/Hyper-CD 结果。

```bash
python -m src.Bayesian_state.run_model_evaluation \
  --input-dir results/model_dynamic_continuous/0809_v1/simulation \
  --output-dir results/model_dynamic_continuous/0809_v1/model_evaluation \
  --eval-prediction-mode prior_t \
  --strategy-audit-config configs/simulation_cfg/generated_from_hyper/model0809_selected8_best.yaml \
  --strategy-audit-particles 32 \
  --strategy-audit-seeds 20260821 20260822 20260823 20260824 \
  --skip-basic --skip-trajectory --skip-behavior-ppc --skip-oral
```

输出包括逐 trial/source-data CSV、被试反事实 accuracy、汇总贡献图和错误串/低表现入口的事件
对齐图。审计中的 behavioral variance 是条件 Bernoulli 方差；across-seed variance 是有限粒子
数值误差。策略贡献单独以 dynamic 与冻结反事实的 expected-accuracy 差异报告，三者不被错误地
包装成可加的方差分量。默认 32 particles 是筛查设置，并在 summary CSV 中记录；正式拟合结果
仍使用原 simulation 中的粒子数。

### 选择传递审计

`--choice-transmission-audit-config` 在同一统一 CLI 内运行另一项只读诊断。每个 subject/common
seed 只重放一次 fitted dynamic PF；原 fitted readout 继续更新粒子权重，同时从完全相同的
pre-choice 状态旁路计算：当前粒子边际、每粒子的 MAP hypothesis、探索期降低 sharpening，以及
探索概率直接耦合 choice uncertainty。它还保存当前 readout 下粒子正确概率的 10%/50%/90%
分位数，并利用系统重采样的父索引回溯完整祖先路径，用于区分：

- 粒子内 hypothesis 平均造成的平滑；
- 粒子之间求边际造成的平滑；
- 缺少 strategy-to-choice coupling；
- 所有粒子自身都相似，因而瓶颈更早地位于 hypothesis proposal/category prediction。

输出写入 `evaluation/choice_transmission_audit/`，包含逐 trial、event-aligned 和 subject
summary CSV，以及同名 PNG 图。`ancestral_strategy_trajectories.png` 将所有 common-seed 终点
粒子组成等权 seed mixture；代表路径是策略成分与 choice probability 上的
posterior-weighted medoid，范围是完整祖先路径的 pointwise 10%–90% 分位数。它不是 PF 边际
均值的置信区间，也不是逐 trial 独立挑选的粒子。配套 `ancestral_trajectory_paths.csv` 保留每条
路径的 seed、终点粒子、逐 trial 父系粒子与权重。所有图遵循仓库规则只输出 PNG。审计不修改
simulation、controller 或 Hyper-CD 结果。

同一次回放还输出 `error_transmission_layers.png`、`error_transmission_trial_data.csv` 和
`error_transmission_phase_summary.csv`，按“当前 trial 有正确预测的 active hypothesis → prior belief
分配 → 未 sharpen 的 category prediction → hypothesis sharpening → strategy confidence → persistent
execution → 最终输出”
定位误差进入模型的层级，
并并列呈现 exploration、failure pressure、mastery evidence 和执行规则切换概率。只有配置启用
`continuous_controller.execution` 时才显示 persistent-execution 层；旧配置仍保持原图层。
逐 trial/phase CSV 同时记录执行层对正确率的增量、切换概率、粒子加权切换事件概率和执行规则
停留 trial 数。阶段标签只使用当前 trial 之前的
rolling accuracy：窗口未满为 warm-up，`<=0.60` 为低表现，`(0.60, 0.85)` 为学习/恢复，`>=0.85`
为 mastery，避免用当前选择反过来定义当前策略。“正确预测的 hypothesis”只表示它在当前刺激上
预测了 task-correct category，不等于认定它是全局真实规则。

可选的 `--choice-transmission-gain-screen 0 1 2 3` 会在同一审计内冻结其余参数，
让每个 gain 共用相同 PF seeds，并额外写出
`strategy_confidence_gain_screen_trial_data.csv`、
`strategy_confidence_gain_screen_summary.csv` 和唯一 PNG
`strategy_confidence_gain_screen.png`。筛查以 observed-choice NLL 为主指标，`gain=0`
是关闭机制的消融基线；同时按上述因果阶段报告结果。`deep_valley` 是低表现的严格子集，
默认定义为当前 choice 之前的一个 rolling window 准确率 `<=0.40`，可用
`--choice-transmission-deep-valley-threshold` 调整。这个分层专门检验 gain 的改善是否只发生
在 mastery；它不会把使用当前 trial 结果定义的谷底标签泄漏回当前预测。

### 序列残差诊断

binary particle-filter 的常规 behavior PPC 会自动在同一 `behavior_ppc/` 目录生成：

```text
sequential_residual_diagnostics.png
sequential_residual_subject_summary.csv
sequential_residual_lag_tests.csv
sequential_residual_trial_data.csv
```

该诊断先对相同冻结参数的 PF repeats 求 trialwise category probability 均值，避免把有限粒子随机
误差误判为行为结构。它同时检查三件事：

1. 在只去除全局 logit intercept 偏差后，accuracy 与 choice-label 的 one-step-ahead 残差是否仍
   被过去 1--8 个 trial 的残差预测；lag family 内使用 Bonferroni 校正，被试之间再报告 BH q 值。
2. 一个 rolling window 中是否出现超出 Bernoulli 条件方差的局部连续偏差；扫描的所有窗口使用
   Bonferroni 校正。
3. 一个只读取过去残差的单状态 EWMA correction，能否在 chronological expanding folds 中比
   intercept-only correction 获得更低的 held-out NLL。负的 `state_minus_intercept_nll` 才表示
   residual state 带来额外预测信息；同一 trial 的 choice 从不进入自己的预测。

这些结果是“是否值得增加潜在动态状态”的筛查证据，不把显著残差直接命名为探索/利用。周期性
刺激难度、遗漏的 stimulus feature 或 readout bias 也可能产生残差结构，仍需通过后续 ablation
区分。

因此，若后续计划画 run-level 图，最终 simulation 应设置：

```yaml
keep_logs: true
```

## 指标边界

共享数值定义以 `metrics/` 为准；结构化 statistics schema 由 `simulation/runner.py` 维护。
本目录负责评价 protocol、结果适配、比较、图表和解释，不应重新
实现 Brier、NLL、CRPS、学习曲线或行为统计。

## 添加新评价

1. 先确认输入来自 subject JSON 还是 raw-run stream。
2. 将纯数值指标放到公共 `metrics/` 层。
3. 将结果读取、汇总表和作图放在本目录。
4. 在 `run_model_evaluation.py` 增加可跳过的执行步骤。
5. 把所有产物登记进 manifest，避免静默漏图。
