# Bayesian_state Modules 完整建模手册

本文档面向论文写作与模型维护，系统说明 `src/Bayesian_state/problems/modules/` 下各模块的构造、输入输出、状态变量、关键公式和模块间耦合关系。目标是把“模型如何从 trial 数据走到 posterior，再走到下一 trial prior”的全过程讲清楚。

## 1. 整体推理链条（Module Pipeline）

在 `StateModel` 初始化后，`BaseEngine.build_modules()` 按配置创建模块实例；每个 trial 由 `BaseEngine.infer_single()` 调度 `agenda` 顺序依次调用 `process()`。

PMH 常用顺序：

1. `perception_mod`
2. `hypo_transitions_mod`
3. `likelihood_mod`
4. `memory_mod`
5. `beta_mod`

对应的信息流：

- 观测输入：`observation = (stimulus, choice, feedback)`
- `perception_mod`：把原始 stimulus 映射为感知 stimulus
- `hypo_transitions_mod`：给出当前活跃假设集合（`hypotheses_mask`）并迁移 `prior`
- `likelihood_mod`：计算 `p(data_t | h)`
- `memory_mod`：融合历史记忆得到 `posterior_t`
- `beta_mod`：依据 trial 结果更新每个假设的 `beta`

跨 trial 连接规则：

- 下一步开始时 `prior_{t+1} <- posterior_t`
- 但若该步 `hypo_transitions_mod` 改变 active set，会进一步把“旧 posterior”映射为“新 prior”（详见第 4 节）

## 2. 基类层：`base_module.py`

### 2.1 结构作用

`BaseModule` 只做两件事：

- 保存 `self.engine`
- 约定模块统一接口 `process(**kwargs)`

所以每个具体模块都共享一个 engine 状态容器，模块间通过 engine 字段读写完成协作。

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

## 4. `hypo_transitions.py`：动态假设迁移模块（DynamicHypothesisModule）

这是模型里“结构更新”最关键的一层：它决定当前有哪些 hypothesis 参与竞争，并把旧 posterior 迁移成新 prior。

### 4.1 核心状态

- `active`：当前 trial 活跃假设索引
- `old_active`：上一 trial 活跃假设索引
- `engine.hypotheses_mask`：0/1 掩码
- `strategy_counts_log`：每 trial 各策略选中数量

### 4.2 `process()` 三步

1. `_transition()`：根据策略得到新 `active`
2. `_apply_mask()`：写入 `engine.hypotheses_mask`
3. `_posterior_to_prior_transition()`：计算新 `engine.prior`

### 4.3 active 选择机制

每个策略由两部分组成：

- `amount`：选多少（可固定/熵驱动/随机驱动/置信驱动）
- `method`：怎么选（`top_posterior`, `random`, `random_posterior`, `ksimilar_centers`）

策略顺序执行并做集合并集，再受 `max_active_hypotheses` 预算约束。

### 4.4 posterior -> prior 的论文主公式（建议写法）

设：

- 第 `t` 步后验 `p_t(h)`
- 旧活跃集 `A_t`
- 新活跃集 `A_{t+1}`
- `S=A_t\cap A_{t+1}`（survivors）
- `N=A_{t+1}\setminus A_t`（newcomers）

第一式：置信度

\[
c_t=\max_h p_t(h)
\]

第二式：未归一化先验构造

\[
\tilde p_{t+1}(h)=
\begin{cases}
p_t(h), & h\in S\\[4pt]
\alpha_t\!\left[c_t\,s_h+(1-c_t)\,n_h\right], & h\in N\\[4pt]
0, & h\notin A_{t+1}
\end{cases}
\quad\text{其中 } \alpha_t=\max(1-c_t,\alpha_{\min})
\]

第三式：归一化

\[
p_{t+1}(h)=\frac{\tilde p_{t+1}(h)}{\sum_u \tilde p_{t+1}(u)}
\]

其中：

- `s_h`：newcomer 与旧活跃集的相似性项（由 `partition.similarity_matrix` 计算）
- `n_h`：新颖性项（与旧集最大相似度互补）
- `\alpha_{min}` 在代码里是 `0.05`

### 4.5 机制解释

- `c_t` 高：模型确信当前解释，偏利用（survivor + 相似扩展）
- `c_t` 低：模型不确信当前解释，偏探索（提升 newcomer 权重）

### 4.6 与 Beta 联动

新加入假设在迁移后会触发 `beta_mod.initialize_beta_for_hypotheses()`，给 newcomer 设置初始 beta。

### 4.7 `finite_workspace_transition.py`：0806 固定容量动态替换

`AdaptiveFiniteWorkspaceTransitionModule` 是 0806 在主框架中的 transition 实现。它保持
workspace 容量 `C` 不变；trial 0 只初始化集合，此后每个 trial 在 choice 前执行：

\[
\operatorname{logit}(m_t)=\operatorname{logit}(m_0)
+\phi\left[\operatorname{logit}(m_{t-1})-\operatorname{logit}(m_0)\right]
+\beta_s z(S_{t-1})+\beta_u z(U_{t-1})
\]

其中上一试次的反馈 surprise 为

\[
S_{t-1}=-\log\sum_h p^-_{t-1}(h)L_{t-1}(h),
\]

`U` 是 active posterior 的归一化熵。随后抽取
`K_t ~ Binomial(C, m_t)`，按 `1-posterior` 权重移除 `K_t` 个假设，再从
local/global proposal 混合中无放回抽取相同数量的 newcomer。被移除的 posterior mass
逐一转移给 newcomer，因此容量和总概率质量都保持不变。

可配置项包括：`capacity`, `m`, `m_phi`, `m_beta_surprise`,
`m_beta_uncertainty`, 两组 signal center/scale、`g`, `tau_local`。也可把动态控制参数
放入一个 `rate_controller` mapping，便于 Hyper-CD 把静态、单信号和联合控制器作为完整
候选进行比较。

该模块只管理 hypothesis transition。感知、likelihood、dual memory、beta、choice readout
和粒子积分仍由原有模块与 optimization 层负责。

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

## 8. `decision.py`：决策模块

### 8.1 现状

`Decision.decision(probability)` 已实现两种策略：

- `top`：`argmax`
- `sample`：按分布采样

但 `process()` 目前是空实现（`pass`），因此默认 pipeline 中通常不承担核心推理计算。它更像是可选输出层接口。

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
