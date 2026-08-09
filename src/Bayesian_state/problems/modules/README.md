# Bayesian_state Modules 完整建模手册

本文档面向论文写作与模型维护，系统说明 `src/Bayesian_state/problems/modules/` 下各模块的构造、输入输出、状态变量、关键公式和模块间耦合关系。目标是把“模型如何从 trial 数据走到 posterior，再走到下一 trial prior”的全过程讲清楚。

## 1. 整体推理链条（Module Pipeline）

在 `StateModel` 初始化后，`BaseEngine.build_modules()` 按配置创建模块实例。兼容入口可由
`BaseEngine.infer_single()` 一次调度完整 `agenda`；正式 `StateModel` 路径将同一 agenda 按
choice 是否已经产生拆成前后两段。

PMH 常用顺序：

1. `perception_mod`
2. `hypo_transitions_mod`
3. `likelihood_mod`
4. `memory_mod`
5. `beta_mod`

对应的信息流：

- choice 前输入：`observation = (stimulus, None, None)`
- `perception_mod`：把原始 stimulus 映射为感知 stimulus
- `hypo_transitions_mod`：给出当前活跃假设集合（`hypotheses_mask`）并迁移 `prior`
- choice/feedback 产生后：`observation = (perceived_stimulus, choice, feedback)`
- `likelihood_mod`：计算 `p(data_t | h)`
- `memory_mod`：融合历史记忆得到 `posterior_t`
- `beta_mod`：依据 trial 结果更新每个假设的 `beta`

跨 trial 连接规则：

- 下一步开始时 `prior_{t+1} <- posterior_t`
- 但若该步 `hypo_transitions_mod` 改变 active set，会进一步把“旧 posterior”映射为“新 prior”（详见第 4 节）

## 2. 基类层：`base_module.py`

### 2.1 结构作用

`BaseModule` 定义所有 engine module 的最小协议：

- 保存 `self.engine`
- trial 更新接口 `process(**kwargs)`
- 粒子状态接口 `state_dict()` / `load_state_dict()`
- 日志清理接口 `clear_logs()`
- 随机模块的 future-stream 接口 `reseed_future()`

无状态 module 可以继承空快照默认实现；有状态 module 必须覆盖保存与恢复。所有具体模块共享
一个 engine 状态容器，模块间通过 engine 字段读写完成协作。

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
- `normal_subject_ids`, `uniform_subject_ids`：两类被试对应不同噪声机制
- `processed_data_dir`：统计文件目录

外部数据表：

- `Task1b_errorsummary_24.csv`：提供 `error_mean`, `error_std`
- `Task1b_errorsummary_72.csv`：提供 `threshold_mean_mean`
- `Task2_processed.csv`：提供 feature 顺序映射（`feature1_name...feature4_name`）

### 3.3 两种噪声机制

#### 机制 A：Normal 噪声（默认）

若被试属于 normal 组：

\[
\epsilon_t \sim \mathcal{N}(\mu_s,\,\sigma_s^2),\quad
\tilde{x}_t = x_t + \epsilon_t
\]

其中 `\mu_s, \sigma_s` 是被试 `s` 的 4 维向量，并已按 Task2 的 feature 呈现顺序重排。

#### 机制 B：Uniform 噪声（72-subject 分支）

若被试属于 uniform 组，使用 `threshold_mean_mean` 作为半宽 `a_s`：

\[
\epsilon_{t,j} \sim \mathcal{U}(-a_{s,j},\,a_{s,j}),\quad
\tilde{x}_{t,j} = x_{t,j} + \epsilon_{t,j}
\]

### 3.4 边界处理

最终感知刺激执行截断：

\[
\tilde{x}_{t,j} \leftarrow \min(1,\max(0,\tilde{x}_{t,j}))
\]

保证输入 partition/likelihood 的数值在 `[0,1]`。

### 3.5 模块输出

更新：

- `engine.observation = (sampled_stimulus, choice, feedback)`

## 4. Hypothesis transition

Hypothesis transition 只有一个公共认知过程，但有三种不同的策略模式：

| 文件 | 公开类 | 每个 trial 发生什么变化 |
|---|---|---|
| `hypo_transition/process.py` | 公共数据契约与 two-step lifecycle | 不定义具体策略 |
| `hypo_transition/static.py` | `StaticHypothesisTransitionModule` | 策略固定；只改变输入和选择结果 |
| `hypo_transition/dynamic_discrete.py` | `DynamicDiscreteHypothesisTransitionModule` | 离散 strategy state `z_t` 变化 |
| `hypo_transition/dynamic_continuous.py` | `DynamicContinuousHypothesisTransitionModule` | 连续 control state（如 `m_t/g_t`）变化 |

H 模块已经收拢到 `hypo_transition/` 子包，不再提供重构前的 H 类或兼容 class path。

### 4.1 公共的两步认知过程

`hypo_transition/process.py` 把一次 transition 固定为：

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

### 4.2 Static strategy

Static 表示同一被试在全部 trials 中使用固定的 `selection strategy + prior assignment` 及固定
参数。策略仍可读取 trial-specific posterior、entropy、confidence 或 feedback history；变化的
是策略输出，不是策略身份或 controller state。

`hypo_transition/static.py` 将当前策略空间显式分为：

- amount/count：fixed、entropy、confidence、accuracy-history、binomial replacement 等；
- selector/proposal：top-posterior、posterior-weighted、random、similarity/local-global 等；
- prior assignment：similarity-novelty、conservative carryover、error-boost、stochastic reset、
  pairwise mass transfer。

`StaticHypothesisTransitionModule` 执行固定 strategy chain；
`StaticWorkspaceHypothesisTransitionModule` 表示 `m/g` 均固定的 bounded-workspace policy。后者目前
只和 `pairwise_mass_transfer` 组合，其他不兼容组合会在初始化时失败。

### 4.3 Dynamic discrete strategy states

`DynamicDiscreteHypothesisTransitionModule` 为每个被试拟合一个固定 controller，但逐 trial 产生
离散认知状态：

\[
P(z_t=k\mid x_t)=\operatorname{softmax}(b_k+w_k^\top x_t).
\]

controller features 可包括 previous error、recent accuracy、accuracy delta、posterior
entropy/confidence、trial progress 和 latent-volatility feature。选中的 strategy state 定义该 trial
使用的 selection 与 prior-assignment 策略，统一配置名为 `state_controller`。

### 4.4 Continuous control dynamics

`DynamicContinuousHypothesisTransitionModule` 不切换策略类型。当前实现固定使用 bounded-workspace
selection 和 pairwise mass transfer，但让显式 control state 随 trial 演化：

\[
\operatorname{logit}(m_t)=\operatorname{logit}(m_0)
+\phi_m\left[\operatorname{logit}(m_{t-1})-\operatorname{logit}(m_0)\right]
+\beta_{m,S}z(S_{t-1})+\beta_{m,U}z(U_{t-1}),
\]

`g_t` 使用相同形式及独立系数。随后：

1. 抽取 `K_t ~ Binomial(C, m_t)`；
2. 按 `1-posterior` 权重移除 `K_t` 个 active hypotheses；
3. 从 `(1-g_t) local + g_t global` proposal 无放回抽取同数 newcomer；
4. 按 `replacement_pairs` 将 dropped posterior mass 转移给 newcomers。

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

所有模式都只管理 hypothesis transition。感知、likelihood、memory、beta、readout 和潜在路径
积分仍由各自 module 或 inference backend 负责。新加入 hypotheses 会继续调用
`beta_mod.initialize_beta_for_hypotheses()`。

## 5. `likelihood.py`：似然模块（LikelihoodModule）

### 5.1 作用

将当前 observation 与每个 hypothesis 的 partition 结构进行匹配，计算 trial 似然向量。

### 5.2 输入与内部调用

- 输入来自 `engine.observation`
- 调用 `partition.calc_likelihood(...)`
- 支持两种 `distance_mode`：`prototype` / `boundary`

### 5.3 Beta 使用规则

若 `engine.beta` 存在且是向量，则按 hypothesis 使用 `\beta_h`；否则使用全局标量 `beta`。

### 5.4 数学形式（抽象）

记感知刺激为 `\tilde{x}_t`，反应信息为 `y_t`，则模块输出：

\[
L_t(h)=p(y_t\mid \tilde{x}_t,h,\beta_h)
\]

其中 `\beta_h` 可为 per-hypothesis，也可退化为全局常数。代码层面最终写入 `engine.likelihood` 的是向量 `\mathbf{L}_t`。

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

### 6.5 active set 变化时的状态迁移

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

### 7.3 newcomer 初始化

若启用 prior 缩放（`use_prior_scaling=True`）：

\[
\beta_h^{init}=\beta_{init}+\lambda\cdot \frac{p(h)}{\max_{j\in N}p(j)}
\]

其中 `\lambda=prior_beta_scale`，随后截断到 `[beta_min,beta_max]`。

### 7.4 trial 内更新（基于反馈一致性）

先根据 `(choice, feedback)` 推断正确类别（2 类情形下错误时取另一类）。

- 若假设预测正确类别：加性上调
\[
\beta_h \leftarrow \beta_h + \eta_+\cdot\frac{\beta_{max}-\beta_h}{\beta_{max}}
\]

- 若假设预测错误类别：按当前 beta 比例惩罚
\[
\beta_h \leftarrow \beta_h - \eta_-\beta_h
\]

再做边界截断。

### 7.5 与 Likelihood 耦合

更新后的 `engine.beta` 在下一 trial 被 `LikelihoodModule` 读取，直接影响 softmax/距离映射的陡峭程度。

## 8. `readout.py`：可观测读出

### 8.1 现状

该文件统一保存从认知状态到观测分布的映射：

- choice：hypothesis expectation、sharpened expectation、MAP、sample 和 sticky readout；
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
2. 再放开 `beta` 动态（`decrease_rate`, `correct_additive`）
3. 最后微调 perception 噪声来源与被试分组

原因：

- `hypo_transitions` 改变的是“候选集合结构”
- `memory` 改变的是“证据时间整合形状”
- `beta` 改变的是“似然曲率/判别尖锐度”
- `perception` 改变的是“输入噪声分布”

四者同时自由会导致参数可辨识性显著下降。

## 10. 代码索引

- 模块目录：`src/Bayesian_state/problems/modules/`
- 动态调度：`src/Bayesian_state/inference_engine/bayesian_engine.py`
- 模型装配：`src/Bayesian_state/problems/model.py`
- 常用配置：`configs/model_struct/pmh_*.yaml`
- H 模块说明：`hypo_transition/README.md`
- strategy/controller candidate 资源：`hypo_transition/candidates/README.md`
- similarity cache：`../cache/README.md`
