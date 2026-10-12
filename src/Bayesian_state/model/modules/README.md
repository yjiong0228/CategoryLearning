# Bayesian_state 模块完整建模手册

本文档面向论文写作与模型维护，系统说明 `src/Bayesian_state/model/modules/` 下各模块的构造、输入输出、状态变量、关键公式和模块间耦合关系。目标是把“模型如何从 trial 数据走到 posterior，再走到下一 trial prior”的全过程讲清楚。

Condition 3 的 PMH 使用 `pairing_memory.HierarchicalPairingMemoryModule`：保存
`joint[h,a]`，配对顺序为 `12|34, 13|24, 14|23`，各规则初始配对均匀。
`engine.compute_likelihood()` 只计算活跃规则的四类概率，由无状态的
`ObservationLikelihood.pairing_kernel()` 产生三配对证据；记忆先对联合先验作
`gamma` 次幂，再乘一次反馈核。`feedback_evidence` 保留反馈前的绝对事件概率，
供 `beta_update_mode: hierarchical_feedback` 使用；不能跨规则归一化或用更新后的配对重算。
新模块第一版要求 `w0=0, feedback_gain=1`。规则替换时联合质量沿用原 prior assignment，
局部投影为零时回退到携带旧配对信念的全局投影，并记录 fallback。
联合状态与待消费证据均通过模块快照复制。搜索显式设
`feedback_interpretation: full_success`，但 observation 中仍保留原始三值反馈。
完整原理见 [Model 0826 第 13 节](../../docs/model_architecture/model_0826.tex)。

Model 0923 full-v3 显式设置 `forgetting_mode: factorized`。记忆模块先将规则信念
取 `gamma` 次幂并在活跃规则间归一化，再将每条规则的条件配对信念取同一次幂、
在该规则的三种配对间归一化。两者相乘后接受一次反馈更新。这样，规则可信度的
遗忘由规则旧信念决定，配对知识继续按规则分别保存。`feedback_pairing_weights`
缓存遗忘后、加入当前反馈前的条件配对分布，供精度梯度共用。
`gamma=0` 显式使用均匀分布；零规则边际对应均匀条件配对。默认
`forgetting_mode: joint` 保留既有联合幂更新。快照记录模式和反馈缓存，恢复时检查
模式一致；旧快照按 `joint` 读取。`initialize_prior()` 和 `reset()` 清空待更新证据。

Full-v3 的二值反馈使用 `DualMemoryModule` 的
`forgetting_mode: exact_power`，固定 `w0=0`、`feedback_gain=1`。每题直接读取候选管理
给出的规则先验，在对数域计算归一化的 `prior**gamma * likelihood`，保留极小的新规则
信念；`gamma=0` 将所有活跃规则作为等权起点。新精度模式的二值反馈似然由
`ObservationLikelihood.binary_feedback_likelihood()` 返回绝对事件概率，错误反馈直接
汇总其余类别的概率。这条路径取消原记忆的 `1e-15` 下限和连续 partition 的 `1e-7`
反馈概率下限，使记忆与精度更新使用同一反馈证据。默认 `dual_trace` 及旧观测路径
保留已有数值运算；这两处概率下限调整仅接入 full-v3。

`readout/` 实现输出层，包含 choice 与 oral 两个分支，读取本题内部状态并返回观测分布。
由 `StateModel` 在反馈前调用；学习步骤仍按 engine agenda 执行。
代码组织和兼容接口见 [readout/README.md](readout/README.md)。

## 1. 整体推理链条（模块管线）

在 `StateModel` 初始化时，`model/assembly.py` 按配置创建模块实例并注册到 engine。
`StateModel` 将同一 `agenda` 按 choice 是否已经产生拆成前后两段；不存在另一条绕过该
lifecycle 的快捷执行路径。

PMH 常用顺序：

1. `perception_mod`
2. `hypo_transitions_mod`
3. engine 固定 likelihood 阶段（不属于 module/agenda）
4. 可选 `mapping_mod`（仅二分类 M1 sensitivity）
5. `memory_mod`
6. `beta_mod`

对应的信息流：

- choice 前输入：`observation = (stimulus, None, None)`
- `perception_mod`：把原始 stimulus 映射为感知 stimulus
- `hypo_transitions_mod`：给出当前活跃假设集合（`hypotheses_mask`）并迁移 `prior`
- choice/feedback 产生后：`observation = (perceived_stimulus, choice, feedback)`
- `engine.compute_likelihood()`：调用固定 observation model 计算 `p(data_t | h)`
- 可选 `mapping_mod`：对固定与反向类别标签解析边缘化，并更新 orientation belief
- `memory_mod`：融合历史记忆得到 `posterior_t`
- `beta_mod`：依据 trial 结果更新每个假设的 `beta`

跨 trial 连接规则：

- 下一步开始时 `prior_{t+1} <- posterior_t`
- 但若该步 `hypo_transitions_mod` 改变 active set，会进一步把“旧 posterior”映射为“新 prior”（详见第 4 节）

## 2. 基类层：`base_module.py`

### 2.1 结构作用

`BaseModule` 定义所有 engine module 的最小协议：

- 保存 `self.engine`
- 用 `ModulePhase` 声明 pre-choice 或 post-choice 阶段
- 用 `ModuleRole` 声明唯一语义职责，供运行时代码稳定查找
- 在 engine 调用 `process()` 前可通过 `prepare_for_process()` 同步内部状态
- trial 更新接口 `process(**kwargs)`
- 完成 choice/feedback 后的统一钩子 `record_outcome(observation)`
- 粒子状态接口 `state_dict()` / `load_state_dict()`
- 日志清理接口 `clear_logs()`
- 随机模块的 future-stream 接口 `reseed_future()`

无状态 module 可以继承空快照默认实现；有状态 module 必须覆盖保存与恢复。所有具体模块共享
一个 engine 状态容器，模块间通过 engine 字段读写完成协作。`process()` 是抽象方法；具体模块
必须实现，但名称继续统一使用 `process`。

`record_outcome()` 在基类中是空实现。H controller 等需要跨 trial 因果历史的模块可以覆盖它；
engine 只在 `StateModel.complete_trial()` 收到 outcome 后统一广播一次，不允许 pre-choice
`process()` 顺带消费当前 trial feedback。

### 2.2 建模意义

这是“黑板架构（blackboard architecture）”：

- 模块本身只实现局部变换
- 全局状态集中在 engine
- `agenda` 决定因果顺序

## 3. `perception.py`：感知噪声模块（PerceptionModule）

### 3.1 模块目的

把物理刺激 `x_t` 转成内部感知刺激 `\tilde{x}_t`，从而建模被试的感知误差来源。后续 likelihood 不是用真实刺激，而是用感知后的刺激。

### 3.2 构造参数与数据依赖

关键参数：

- `features`：特征维度，默认 4
- `mean`, `std`：正态噪声参数（可标量或长度=features 向量）
- `subject_id`：用于读取被试特异统计
- `noise_mode`：`auto` 按被试分组，`normal` 或 `uniform` 显式指定噪声机制
- `normal_subject_ids`, `uniform_subject_ids`：两类被试对应不同噪声机制
- `processed_data_dir`：统计文件目录

`auto` 的默认正态组为125–132、225–232、325–332；默认均匀组为101–124、201–224、301–324，以及编号333及之后的所有被试。显式指定 `noise_mode` 或被试分组可覆盖自动选择，用于保留明确配置的实验设定。

外部数据表：

- `Task1b_errorsummary_24.csv`：提供 `error_mean`, `error_std`
- `Task1b_errorsummary_72.csv`：提供 `threshold_mean_mean`
- `Task2_processed.csv`：提供 feature 顺序映射（`feature1_name...feature4_name`）

MEG 与 exp4 的阈值表名为 `Task1b_errorsummary.csv`。运行配置通过 `dataset.perception_summary_72` 指向该表，通过 `dataset.feature_order_data` 指向相应学习数据；字段名中的 `72` 沿用已有接口，所指阈值表可属于其他实验。

### 3.3 两种噪声机制

#### 机制 A：正态噪声

若被试属于 normal 组：

\[
\epsilon_t \sim \mathcal{N}(\mu_s,\,\sigma_s^2),\quad
\tilde{x}_t = x_t + \epsilon_t
\]

其中 `\mu_s, \sigma_s` 是被试 `s` 的 4 维向量，并已按 Task2 的 feature 呈现顺序重排。

#### 机制 B：均匀噪声

若被试属于 uniform 组，使用 `threshold_mean_mean` 作为半宽 `a_s`：

\[
\epsilon_{t,j} \sim \mathcal{U}(-a_{s,j},\,a_{s,j}),\quad
\tilde{x}_{t,j} = x_{t,j} + \epsilon_{t,j}
\]

同一被试、同一特征的多个测量水平先取平均，得到该特征的半宽，再按学习任务的特征顺序排列。半宽为 $a_{s,j}$ 时，误差范围为 $[-a_{s,j},a_{s,j}]$，总宽为 $2a_{s,j}$。这一设置改变内部刺激，并进一步影响规则预测和学习轨迹。

### 3.4 边界处理

最终感知刺激执行截断：

\[
\tilde{x}_{t,j} \leftarrow \min(1,\max(0,\tilde{x}_{t,j}))
\]

保证输入 partition/likelihood 的数值在 `[0,1]`。

### 3.5 模块输出

更新：

- `engine.observation = (sampled_stimulus, choice, feedback)`

## 4. 假设转移

Hypothesis transition 只有一个公共认知过程，但有三种不同的策略模式：

| 文件 | 公开类 | 每个 trial 发生什么变化 |
|---|---|---|
| `hypothesis_transition/contracts.py` | 公共数据契约与 two-step lifecycle | 不定义具体策略 |
| `hypothesis_transition/fixed_strategy.py` | `FixedStrategyHypothesisTransitionModule` | 策略固定；只改变输入和选择结果 |
| `hypothesis_transition/dynamic_discrete_strategy.py` | `DynamicDiscreteStrategyHypothesisTransitionModule` | 离散 strategy state `z_t` 变化 |
| `hypothesis_transition/dynamic_adaptive_control.py` | `DynamicAdaptiveControlHypothesisTransitionModule` | 连续 control state（如 `m_t/g_t`）变化 |
| `hypothesis_transition/feedback_reactive.py` | `FeedbackReactiveHypothesisTransitionModule` | 上一 trial feedback 决定一步探索率（H3 参照） |
| `hypothesis_transition/nested_feedback_accumulator.py` | `NestedFeedbackAccumulatorHypothesisTransitionModule` | 一步反应加可归零的累计失败效应（H4） |

H 模块已经收拢到 `hypothesis_transition/` 子包，不再提供重构前的 H 类或兼容 class path。

### 4.1 公共的两步认知过程

`hypothesis_transition/contracts.py` 把一次 transition 固定为：

1. `select_hypotheses(context)`：选择下一试次的 active hypotheses；
2. `assign_prior(context, selection)`：在新 active set 上分配 prior。

第一步返回 `HypothesisSelection`，显式记录：

- `active_before` / `active_after`
- `survivors`
- `dropped`
- `newcomers`
- 可选 `replacement_pairs`

第二步返回完整 prior 向量。公共 lifecycle 检查 active indices、集合分解、prior 有限非负、
归一化以及 inactive hypotheses 上零质量，并把最终结果保存为 `last_transition_result`。

抽象地，

\[
D_t=\mathcal S(x_t;\theta_t^S),
\qquad
p_t^-=\mathcal P(p_{t-1},D_t;\theta_t^P).
\]

其中 `D_t` 是 selection result，`S` 和 `P` 分别是 hypothesis selection 与 prior assignment。

### 4.2 固定策略

Static 表示同一被试在全部 trials 中使用固定的 `selection strategy + prior assignment` 及固定
参数。策略仍可读取 trial-specific posterior、entropy、confidence 或 feedback history；变化的
是策略输出，不是策略身份或 controller state。

`hypothesis_transition/fixed_strategy.py` 将当前策略空间显式分为：

- amount/count：fixed、entropy、confidence、accuracy-history、binomial replacement 等；
- selector/proposal：top-posterior、posterior-weighted、random、similarity/local-global 等；
- prior assignment：相似度新颖性、保守继承、错误增强、随机重置和成对质量转移。

`FixedStrategyHypothesisTransitionModule` 执行固定 strategy chain；
`FixedWorkspaceHypothesisTransitionModule` 表示 `m/g` 均固定的 bounded-workspace policy。bounded
workspace 支持历史兼容的 `pairwise_mass_transfer`，以及 Model 0815 H5 使用的无额外参数
`similarity_transport`。Model 0826 的单因素 sensitivity 另提供同样无额外参数的
`mass_preserving_similarity_transport`；其他 prior-assignment 方法会在初始化时失败。

`FeedbackReactiveHypothesisTransitionModule` 复用同一 bounded-workspace selection、local/global
proposal 与所配置的 bounded-workspace prior assignment，但让下一 trial 的 replacement-event probability 只取决于
上一 trial 已完成的 feedback：正确后使用 `event_after_correct`，错误后使用
`event_after_error`，并要求后者不小于前者。`global_search` 在被试内保持固定。该模式不维护
failure/mastery accumulator，也不允许 persistent execution；它用于检验一步反馈反应能否替代
更复杂的连续 controller。

`NestedFeedbackAccumulatorHypothesisTransitionModule` 是 Model 0815 后续使用的嵌套 controller。它保留
上述一步反馈反应，并维护唯一的 exponentially weighted failure state；累计增益精确等于 0 时
直接执行 `FeedbackReactiveHypothesisTransitionModule` 的更新路径。若此时
`event_after_correct == event_after_error`，又进一步退化为 constant-event bounded workspace。
因此 constant、reactive 和 accumulator 可以由同一实现的被试级参数表达，而不需要为被试硬选
三个互不嵌套的 H 类。H4 历史筛选固定使用 pairwise transfer；H5 保持同一 controller，但改用
`similarity_transport`，并允许同一 failure state 通过可归零 gain 扩大 global-search 比例。
Model 0826 固定设置 `event_history_excludes_latest_error: true`：搜索事件的累计增益
读取更新前的 $F_{t-1}$，使 `event_after_correct/error` 独占最近一次反馈；global-search range
仍读取包含最近反馈的 $F_t$。未设置该键的历史 H4/H5 配置保留原行为。
H4/H5 还可把 `persistent_execution.enabled` 作为被试级 false/true 坐标：false 时按 workspace
posterior 边际作答，true 时由一条受保护且可持续的 executed rule 作答。当前 switch scale 固定，
不随被试另行优化；H3 仍明确限制为 execution-off。

### 4.3 动态离散策略状态

`DynamicDiscreteStrategyHypothesisTransitionModule` 为每个被试拟合一个固定 controller，但逐 trial 产生
离散认知状态：

\[
P(z_t=k\mid x_t)=\operatorname{softmax}(b_k+w_k^\top x_t).
\]

controller features 可包括 previous error、recent accuracy、accuracy delta、posterior
entropy/confidence、trial progress 和 latent-volatility feature。选中的 strategy state 定义该 trial
使用的 selection 与 prior-assignment 策略，统一配置名为 `state_controller`。

### 4.4 连续控制动力学

`DynamicAdaptiveControlHypothesisTransitionModule` 不切换策略类型。当前实现固定使用 bounded-workspace
selection，并在 pairwise compatibility 或 similarity transport 两种 prior assignment 中固定选择一种；
显式 control state 随 trial 演化：

\[
\operatorname{logit}(m_t)=\operatorname{logit}(m_0)
+\phi_m\left[\operatorname{logit}(m_{t-1})-\operatorname{logit}(m_0)\right]
+\beta_{m,S}z(S_{t-1})+\beta_{m,U}z(U_{t-1}),
\]

`g_t` 使用相同形式及独立系数。随后：

1. 抽取 `K_t ~ Binomial(C, m_t)`；
2. 按 `1-posterior` 权重移除 `K_t` 个 active hypotheses；
3. 从 `(1-g_t) local + g_t global` proposal 无放回抽取同数 newcomer；
4. 按固定 prior-assignment policy 产生新 prior：pairwise compatibility path 搬运 dropped mass；
   H5 similarity transport 则按替换比例混合 survivor carry-over 与 local/global semantic projection；
   mass-preserving sensitivity 保留 survivors 的绝对质量，只把 dropped mass 在 newcomers 间语义分配。

新配置可将两组 controller 写成：

```yaml
continuous_controller:
  rate: {m: 0.15, m_phi: 0.5, m_beta_surprise: 0.8}
  range: {g: 0.35, g_phi: 0.5, g_beta_surprise: 0.4}
```

历史 `rate_controller` / `range_controller` 以及顶层 `m_*` / `g_*` 参数继续兼容。该公开类要求
至少一个 `m_beta_*` 或 `g_beta_*` 非零；固定 `m/g` 应使用 static bounded-workspace 类。

### 4.5 三种模式的边界

```text
static:
context_t -> fixed strategy -> transition result

dynamic discrete:
context_t -> discrete strategy state z_t -> state-specific policy -> transition result

continuous dynamic:
context_t + controls_(t-1) -> controls_t -> fixed strategy -> transition result
```

所有模式都只管理 hypothesis transition。感知、memory、beta 由各自 module 负责；
likelihood 由 engine 的固定观测模型阶段负责。新加入 hypotheses 会继续调用
`beta_mod.initialize_beta_for_hypotheses()`；配置了可选 mapping 时，还会调用
`mapping_mod.initialize_orientation_for_hypotheses()`。

## 5. 固定 observation-likelihood 阶段

### 5.1 作用

将当前 observation 与每个 hypothesis 的 partition 结构进行匹配，计算 trial 似然向量。
实现位于 `../../hypothesis_space/observation_model/likelihood.py`，其中
`ObservationLikelihood` 不持有 engine 状态，也不继承 `BaseModule`。

### 5.2 输入与内部调用

- `BayesianStateEngine.compute_likelihood()` 从 `engine.observation` 与 `engine.beta` 收集运行状态
- `ObservationLikelihood.process()` 调用 `partition.calc_likelihood(...)`
- continuous partition 支持 `prototype` / `boundary`，discrete partition 支持 `rule`

### 5.3 Beta 的使用规则

若 `engine.beta` 存在，则按 hypothesis 使用 `\beta_h`；否则使用
`likelihood.default_beta`。

### 5.4 数学形式（抽象）

记感知刺激为 `\tilde{x}_t`，反应信息为 `y_t`，则模块输出：

\[
L_t(h)=p(y_t\mid \tilde{x}_t,h,\beta_h)
\]

其中 `\beta_h` 可为 per-hypothesis，也可退化为全局常数。代码层面最终写入 `engine.likelihood` 的是向量 `\mathbf{L}_t`。

### 5.5 可选二元 orientation 边缘化

`mapping.py` 的 `BinaryOrientationMappingModule` 只用于 M0/M1 mapping sensitivity。
它不扩大 geometry hypothesis space，也不抽样额外粒子。对每个 active geometry (h)，维护

\[
m_t(h)=P(o_h=\mathrm{fixed}\mid y_{1:t},f_{1:t},h),
\]

另一个方向的概率自动为 (1-m_t(h))。在 choice 前，固定标签 emission
\(q_t(y\mid h)\) 被解析边缘化为

\[
p_t(y\mid h)=m_{t-1}(h)q_t(y\mid h)
 +[1-m_{t-1}(h)]q_t(\operatorname{reverse}(y)\mid h).
\]

feedback 到来后，用同一个 fixed-label likelihood 进行两状态 Bayes 更新；随后 memory 接收已经
边缘化的 geometry likelihood。新进入 workspace 的 geometry 从 (m=.5) 重新开始。该更新固定、
Rao--Blackwellized，新增 latent state 但不新增自由参数。`predictive_orientation_probability`
保留产生当前 choice 的 pre-feedback belief，确保后执行的 beta 更新不会使用已经看到当前反馈的
orientation posterior。

M0 没有该 module，行为保持 fixed mapping。当前实现刻意只支持二分类；四分类 permutation、
完整 mapping learner 和 label-invariant similarity 均不在此模块范围内。

## 6. `memory.py`：双通道记忆后验模块（DualMemoryModule）

### 6.1 设计目标

在 Bayesian 乘法累积之外显式建模“长期累积记忆 + 近期衰减记忆”的混合证据整合。

### 6.2 核心状态

- `state["static"]`：长期累积轨（log-space）
- `state["fade"]`：衰减轨（log-space）
- `baseline_state`：用于 active set 变化时对齐 newcomer 的参考轨
- `mask`：当前有效假设掩码

### 6.3 状态更新方程

对活跃假设：

\[
\text{fade}_t(h)=\gamma\,\text{fade}_{t-1}(h)+\log L_t(h)
\]
\[
\text{static}_t(h)=\text{static}_{t-1}(h)+\log L_t(h)
\]

其中 `\gamma\in[0,1]`。

### 6.4 通道融合方程

\[
\log q_t(h)=w_0\,\text{static}_t(h)+(1-w_0)\,\text{fade}_t(h)
\]

再做掩码归一化得到 posterior：

\[
p_t(h)=\frac{\exp(\log q_t(h))\,m_t(h)}{\sum_u \exp(\log q_t(u))\,m_t(u)}
\]

其中 `m_t(h)\in\{0,1\}` 是 `hypotheses_mask`。

### 6.5 活跃集合变化时的状态迁移

`_state_transition()` 会在 mask 改变前执行：

- 移除假设：状态置 `-\infty`
- 新增假设：根据目标 `prior` 与 `baseline_state` 计算一致的初始 `static/fade`

这样保证集合扩缩时数值连续，不会因为 newcomer 无历史而导致不合理极值。

### 6.6 参数含义

- `w0` 大：更依赖长期累积（惯性强）
- `w0` 小：更依赖近期证据（适应快）
- `\gamma` 大：fade 记忆衰减慢
- `\gamma` 小：fade 更关注近几步

## 7. `beta.py`：假设特异逆温度模块（BetaModule）

### 7.1 建模动机

允许不同 hypothesis 拥有不同决策锐度（inverse temperature），并随 trial 成败动态演化。

### 7.2 状态与边界

- `beta[h]`：每个 hypothesis 的逆温度
- 边界：`beta_min <= beta[h] <= beta_max`
- 非活跃假设在更新后会置为 `0`

### 7.3 新加入假设的初始化

若启用 prior 缩放（`use_prior_scaling=True`）：

\[
\beta_h^{init}=\beta_{init}+\lambda\cdot \frac{p(h)}{\max_{j\in N}p(j)}
\]

其中 `\lambda=prior_beta_scale`，随后截断到 `[beta_min,beta_max]`。

### 7.4 历史模式的 trial 内更新（基于反馈一致性）

先根据 `(choice, feedback)` 推断正确类别（2 类情形下错误时取另一类）。

- 若假设预测正确类别：获取剩余 headroom 的一部分
\[
\beta_h \leftarrow \beta_h + \alpha_+(\beta_{max}-\beta_h)
\]

- 若假设预测错误类别：按当前 beta 比例惩罚
\[
\beta_h \leftarrow \beta_h - \eta_-\beta_h
\]

再做边界截断。

`increase_rate` 是当前推荐的无量纲参数 \(\alpha_+\in[0,1]\)。历史
`correct_additive` 仍可读取，并严格换算为

\[
\alpha_+=\frac{\texttt{correct\_additive}}{\beta_{max}}.
\]

两个键不能同时配置。Model 0815 H5 固定 \(\beta_{min}=0.1\)、
\(\beta_{max}=25\)，并用 `increase_rate=0.04` 精确替代原来的
`correct_additive=1`；因此这次迁移只改变参数含义和优化坐标，不改变既有 beta 轨迹。

`beta_update_mode: probabilistic_feedback` 不先硬判“哪条规则正确”，而是用该规则对实际
choice 的预测 \(p\) 与二值反馈 \(r\) 形成证据 \(\ell=rp+(1-r)(1-p)\)。
2026-10-05 修正多分类的中性点：对 \(C\) 类任务，反馈机会概率为
\(b=r/C+(1-r)(1-1/C)\)。取

\[
\kappa=
\begin{cases}
(\ell-b)/(1-b),&\ell\ge b,\\
(\ell-b)/b,&\ell<b.
\end{cases}
\]

支持时增加 \(\alpha_+\kappa(\beta_{max}-\beta_h)\)，反驳时减少
\(\eta_-\beta_h\min(1,-\kappa)\)，上下界不变。
四分类正确／错误的中性点分别为 0.25／0.75，均匀规则在两种反馈下均无精度漂移。
二分类保留原 \(\kappa=2(\ell-0.5)\) 运算和数值结果。
`hierarchical_feedback` 的三值反馈机会校准保持原样，不走这一二值分支。
这会改变旧的多分类二值反馈轨迹；历史结果按其记录的源码版本复现，不覆盖或改写。

`update_scope` 决定哪些规则消费当前反馈：

- `active_hypotheses`：更新所有 active rules，是向后兼容的默认行为；
- `executed_hypothesis`：只更新本 trial 实际执行的 rule，未执行候选的 beta 保持不变；
  该模式要求开启 persistent execution。

beta 在 outcome 后更新，因此 trial `t` 的反馈只影响 trial `t+1` 的预测。

在 failure-accumulator controller 的 persistent execution 下，还可实验性配置
`rule_commitment`。其 entry 同时要求近期失败、足够的历史选择证据、一个排他性较高的
choice-compatible rule，并可要求历史 `peak_mastery_evidence` 达标；release 可由累计负反馈或
该 rule 的 choice compatibility 跌破阈值触发。默认未启用，不能与
`misconception_capture` 同时使用。

### 7.5 与 likelihood 耦合

更新后的 `engine.beta` 在下一 trial 同时进入 choice readout 与配置为
`beta_source: action` 的 `ObservationLikelihood`，直接影响 softmax/距离映射的陡峭程度。
H5 因而使用一个统一的动态 beta，而不是相互独立的 choice/evidence beta。

### 7.6 Model 0923 的单学习率精度校准

`beta_update_mode: feedback_gradient` 要求显式配置有限的
`learning_rate`，范围为 `[0,1]`。该模式更新全部活跃候选；新规则和重新进入的规则
统一使用 `beta_init`，配置中的 `use_prior_scaling` 为 `false`。
共享模块继续读取 `beta_min` 与 `beta_max`，0923 配置固定为 `0.1` 与 `25`。

每条规则的更新方向是其反馈对数似然对精度的梯度。梯度等于作答前的平均类别距离，
减去按实际反馈重新加权后的平均类别距离。将梯度乘以 `learning_rate` 加到当前
精度，再限制到上下界。计算固定本题内部刺激与类别距离；条件3使用记忆模块缓存的
`feedback_pairing_weights`，更新后的精度从下一题起生效。

`ObservationLikelihood.feedback_precision_gradient()` 计算这一步，使用
`boundary` 距离、`action` 精度与零反馈失误率，并复用几何的距离缓存。
新模式的参数校验拒绝 `increase_rate`、`decrease_rate`、`correct_additive` 和
`executed_hypothesis` 更新范围。旧精度模式继续采用第7.4节的运算。
精度快照记录更新模式，恢复时校验模式一致；缺少模式标记的旧快照可恢复至旧精度模式。

## 8. `readout.py`：可观测读出

### 8.1 现状

该文件统一保存从认知状态到观测分布的映射：

- choice：hypothesis expectation、sharpened expectation、MAP、sample 和 sticky readout；
- commitment confidence：仅在可选 rule commitment active 时对 category preference 做对称
  precision 变换，`rule_commitment_confidence_gain: 0` 为严格恒等映射；
- output noise：base/history-dependent lapse 的统一混合；
- RT：从 choice uncertainty、替换比例、newcomer distance 和 practice 得到 log-RT
  Student-t 参数；
- oral report：用归一化 hypothesis-to-report mapping 和 reliability 得到报告分布。

`Decision.decision(probability)` 仍保留两种兼容采样策略：

- `top`：`argmax`
- `sample`：按分布采样

`process()` 保持 no-op 仅用于旧 agenda 兼容。新代码直接调用 readout 函数；读出只读取
latent state，不更新 posterior、memory 或 transition controller。RT/oral 的函数接口已经
存在，但只有相应观测数据和评分协议接入 backend 后才参与拟合。

自主执行使用 `predict_choice_from_model()` 生成 `ChoicePrediction`，同时保留认知 choice
distribution、加入 output noise 后的可观测 distribution、readout 诊断和 lapse 状态。模型随后
才采样 choice 并接收任务 feedback，category schedule 不参与这个 pre-outcome readout。

## 9. 参数层面的整体可辨识性建议

当拟合不稳定时，建议按以下顺序调参，避免模块间混淆归因：

1. 先固定 `hypo_transitions` 策略结构，只调 `memory (w0, gamma)`
2. 再放开 `beta` 动态（`decrease_rate`, `increase_rate`）
3. 最后微调 perception 噪声来源与被试分组

原因：

- `hypo_transitions` 改变的是“候选集合结构”
- `memory` 改变的是“证据时间整合形状”
- `beta` 改变的是“似然曲率/判别尖锐度”
- `perception` 改变的是“输入噪声分布”

四者同时自由会导致参数可辨识性显著下降。

## 10. 代码索引

- 模块目录：`src/Bayesian_state/model/modules/`
- 动态调度：`src/Bayesian_state/model/engine.py`
- 模型装配：`src/Bayesian_state/model/assembly.py`
- 常用配置：`configs/exp123/model_struct/pmh_*.yaml`
- H 模块说明：`hypothesis_transition/README.md`
- strategy/controller candidate 资源：`configs/shared/candidates/hypothesis_transition/README.md`
- likelihood/partition 执行机制：`../../hypothesis_space/observation_model/README.md`
