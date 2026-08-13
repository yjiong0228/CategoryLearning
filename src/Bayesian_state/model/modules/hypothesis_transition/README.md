# 假设转移（H 模块）

H 模块描述被试怎样从上一试次的 hypothesis posterior 过渡到下一试次的 hypothesis prior。
它只包含两个认知步骤：

1. **Hypothesis selection**：决定下一试次考虑哪些 hypotheses；
2. **Prior assignment**：在新的 active set 上分配 prior probability。

配置值 `static`、`dynamic_discrete` 和 `dynamic_continuous` 的区别，不在于是否产生不同
active set，而在于
控制这两个步骤的 **trial-level policy state** 是否以及怎样随试次变化。

这里的 discrete/continuous 修饰的是“策略怎样跨试次变化”，与
`hypothesis_space/spaces/` 中 discrete/continuous 修饰“候选规则怎样表示”是两个独立维度。

## 1. 目录与公开边界

```text
hypothesis_transition/
├── contracts.py
├── fixed_strategy.py
├── dynamic_discrete_strategy.py
├── dynamic_adaptive_control.py
├── selection.py
├── prior_assignment.py
├── workspace.py
└── execution.py
```

公开模式与共享实现分开：

| 文件 | 职责 |
|---|---|
| `contracts.py` | 所有模式共同遵守的 select → assign-prior 过程与结果契约 |
| `fixed_strategy.py` | 被试内部不随 trial 改变的 selection/prior policy |
| `dynamic_discrete_strategy.py` | 随 trial 改变的离散 strategy state (z_t) |
| `dynamic_adaptive_control.py` | 随 trial 改变的连续 adaptive control state (c_t) |
| `selection.py` / `prior_assignment.py` | 选择和 prior 分配的共享机制 |
| `workspace.py` / `execution.py` | bounded workspace 状态与执行逻辑 |

只有前四个文件定义认知模式或公共契约；后四个是共享机制，不能在
model YAML 中当作独立 H 模式配置。被试级 candidate 资源位于
`configs/candidates/hypothesis_transition/`，不保存 trial-level state trajectory。

## 2. 公共的两步过程

设 trial (t) 开始 transition 时可用的因果信息为：

\[
x_t=(A_{t-1},p_{t-1},o_{<t},h_t),
\]

其中：

- (A_{t-1})：上一试次 active hypothesis set；
- (p_{t-1})：上一试次 hypothesis posterior；
- (o_{<t})：截至上一试次的 observation/feedback history；
- (h_t)：当前可用的 entropy、confidence、surprise、uncertainty 等派生信号。

H transition 写为：

\[
D_t=\mathcal S(x_t;\psi_t^S,\theta_s),
\]

\[
p_t^-=\mathcal P(p_{t-1},D_t;\psi_t^P,\theta_s).
\]

这里：

- (D_t) 是 selection result；
- (p_t^-) 是下一试次使用的 prior；
- \(\theta_s\) 是被试级参数，在被试内部固定；
- \(\psi_t^S\) 和 \(\psi_t^P\) 分别是 selection 与 prior-assignment 的 trial-level policy state。

`contracts.py` 将 `process()` 的执行顺序固定为：

```text
causal context x_t
        ↓
select_hypotheses(context)
        ↓
HypothesisSelection
        ↓
assign_prior(context, selection)
        ↓
prior_t
```

`HypothesisSelection` 显式记录：

- `active_before` / `active_after`
- `survivors`
- `dropped`
- `newcomers`
- 可选的 `replacement_pairs`

公共 lifecycle 负责：

- 检查 hypothesis indices 合法、非空且无重复；
- 检查 survivor/drop/newcomer 集合分解一致；
- 提交 `hypotheses_mask`；
- 检查 prior 有限、非负、归一化；
- 检查 inactive hypotheses 上没有 prior mass；
- 生成统一的 `HypothesisTransitionResult`。

具体策略不能绕过这个顺序另写一套完整 `process()`。

## 3. 三个容易混淆的层级

H 模型同时存在三个不同层级：

```text
subject-level optimization
选择或估计固定的 candidate / 参数 θ_s、φ_s
                    ↓
trial-level policy controller
无 controller、离散 z_t、或连续 c_t
                    ↓
H cognitive process
select hypotheses → assign prior
```

必须区分：

- **strategy/policy**：一套 selection rule 与 prior-assignment rule；
- **trial-level state**：当前 trial 实际使用哪套 policy，或 policy 取什么连续参数；
- **candidate/profile**：优化器在被试层面选择的一整套固定模型配置。

candidate 是被试级超参数；state 才是 trial-level 变量。一个 candidate 内部可以产生完整的
(z_{1:T}) 或 (c_{1:T}) trajectory。

## 4. 固定策略

static 的定义是：同一被试在所有 trials 使用相同的 selection/prior policy 及相同参数。

\[
\psi_t^S=\psi_s^S,
\qquad
\psi_t^P=\psi_s^P.
\]

static **不表示**：

- active set 每个 trial 相同；
- selection result 每个 trial 相同；
- prior 每个 trial 相同；
- 策略不能读取 posterior、entropy、confidence 或 feedback history。

例如，一条固定规则可以是：保留 posterior 最高的两个 hypotheses，再随机加入一个 inactive
hypothesis，并给 newcomer 固定的 prior mass。posterior 和随机结果逐 trial 变化，但规则本身
没有变化，因此仍然是 static。

### 4.1 固定策略候选

一个完整 static candidate 应同时定义：

```text
candidate k
├── hypothesis-selection rule
├── selection parameters
├── prior-assignment rule
└── prior-assignment parameters
```

当前 candidate JSON 允许优化器为每个被试离散选择：

\[
k_s^*=\arg\min_k L_s(k).
\]

但“static”和“离散优化”不是同义词。未来也可以固定策略结构，对其中参数做连续优化；只要这些
参数在该被试内部不随 trial 改变，仍属于 static。

### 4.2 固定策略与固定分支规则

一条 static policy 可以包含固定的条件分支，例如“上一试次错误时增加 exploration”。只要这仍
被建模为一个不可分的固定映射，并且没有显式 trial-level controller state，它仍归入 static。
如果这些分支被显式表示为可记录、可转移的 strategy states，则应归入 dynamic-discrete。

## 5. 动态离散策略

dynamic-discrete 引入显式离散 strategy state：

\[
z_t\in\{1,\ldots,K\}.
\]

每个 state (k) 定义一套完整 policy pair：

\[
\Pi_k=(\mathcal S_k,\mathcal P_k).
\]

trial (t) 使用：

\[
(\psi_t^S,\psi_t^P)=\Pi_{z_t}.
\]

例如可以定义 conservative、stable、exploratory、reset 等 states。它们可以使用不同的
hypothesis-selection 规则、不同的 prior-assignment 规则，或只改变其中一个步骤。

state controller 决定 state trajectory：

\[
P(z_t\mid z_{t-1},x_t;\phi_s).
\]

其中 \(\phi_s\) 是被试级 controller 参数，在被试内部固定；真正随 trial 变化的是 (z_t)。

### 5.1 状态与 profile/candidate

代码中统一使用：

- `state_controller`：trial-level state controller 的完整配置；
- `states`：controller 可以选择的离散 strategy states；
- `selected_state`：当前 trial 选中的 state；
- `state_probabilities`：当前 trial 的 state probability distribution。

上层 candidate/profile 则是优化单位，它可以定义：

```text
controller candidate r
├── state 集合
├── 每个 state 的 selection/prior policy
├── state activation/transition equation
└── controller 参数 φ_s
```

之前的做法是预先给出若干离散 controller profiles，再让 Grid/CD 为每个被试选择一个。这个
做法不是 dynamic-discrete 的必要条件。未来可以固定 state 集合和 controller 形式，直接连续
优化 bias、weights、temperature 等 \(\phi_s\)；只要 trial-level (z_t) 仍然离散，模型仍是
dynamic-discrete。

### 5.2 确定性状态与随机状态

dynamic-discrete 的 state update 可以是：

- 随机潜在状态转移；
- 受历史信息控制的 softmax 采样；
- 确定性的 argmax 控制；
- 由预先规定的 trial schedule 决定。

离散动态的定义来自显式 (z_t)，不要求 state transition 一定随机。

## 6. 动态连续控制

dynamic-continuous 引入连续 control state：

\[
c_t\in\mathbb R^d.
\]

control 按固定的被试级机制更新：

\[
c_t=G(c_{t-1},x_t;\phi_s),
\]

然后控制 selection 和/或 prior assignment：

\[
D_t=\mathcal S(x_t;c_t,\theta_s),
\]

\[
p_t^-=\mathcal P(p_{t-1},D_t;c_t,\theta_s).
\]

与 dynamic-discrete 不同，dynamic-continuous 通常不在若干算法之间切换，而是在固定 policy
family 内连续改变控制参数。当前实现中的例子是：

- (m_t)：控制 active hypotheses 的 replacement rate；
- (g_t)：控制 newcomer search 的 local/global mixture。

更新方程可以写成：

\[
\operatorname{logit}(m_t)=\operatorname{logit}(m_0)
+\phi_m[\operatorname{logit}(m_{t-1})-\operatorname{logit}(m_0)]
+\beta_{m,S}z(S_{t-1})+\beta_{m,U}z(U_{t-1}),
\]

并为 (g_t) 使用对应的独立参数。这里 (m_t/g_t) 随 trial 改变，而
\(m_0,\phi_m,\beta_{m,S},\beta_{m,U}\) 等 controller 参数在被试内部固定，可作为被试级
超参数进行 Grid/CD 或未来的连续优化。

dynamic-continuous 同样可以是 deterministic recursion，也可以扩展为 stochastic continuous
latent process；是否需要粒子积分取决于具体状态方程，而不是文件名。

## 7. 假设选择与先验分配可以采用不同模式

selection 和 prior assignment 是两个独立认知步骤，不要求它们同时变化。

| Selection policy | Prior-assignment policy | 整体解释 |
|---|---|---|
| static | static | 完全 static |
| dynamic-discrete | static | 离散切换 selection，prior rule 固定 |
| static | dynamic-discrete | selection 固定，离散切换 prior rule |
| dynamic-continuous | static | 连续控制 selection，prior rule 固定 |
| static | dynamic-continuous | selection 固定，连续控制 prior 参数 |
| dynamic-continuous | dynamic-continuous | 两个步骤都由连续 state 控制 |

只要 policy pair

\[
\Pi_t=(\psi_t^S,\psi_t^P)
\]

中至少一个分量随 trial 变化，整体 H transition 就是 dynamic。

当前 0806 的 (m_t/g_t) 主要改变 hypothesis selection；dropped hypothesis 到 newcomer 的
pairwise prior-transfer rule 本身保持固定。因此更精确的描述是：

```text
dynamic-continuous hypothesis selection
+
static prior assignment
```

整体仍归入 `dynamic_adaptive_control.py`，因为完整 H policy pair 已经随 trial 变化。

### `failure_accumulator_v2` 控制器

`continuous_controller.mode: failure_accumulator_v2` 是保持上述 selection/prior 边界不变的
Controller v2a。它只使用 trial `t-1` 已完成的 feedback，维护快速 failure pressure 与慢速
mastery evidence；当前 trial 的 choice/feedback 不会进入 pre-choice controller。

控制器直接产生“至少替换一个 hypothesis”的探索事件概率 `E_t`，再按 workspace capacity
换算为 slot replacement rate：

```text
m_t = 1 - (1 - E_t) ** (1 / capacity)
```

因此相同 `E_t` 在不同 capacity 下具有相同的总体探索含义。`exploration.failure_threshold`
控制局部探索开始上升的位置；更高的 `range.failure_threshold` 让 persistent failure 才启动
global search。`rise_rate` 与 `recovery_rate` 分开，允许快速进入探索、较慢恢复利用。

```yaml
continuous_controller:
  mode: failure_accumulator_v2
  state:
    failure_decay: 0.60
    mastery_decay: 0.90
  exploration:
    event_min: 0.05
    event_max: 0.65
    failure_threshold: 0.55
    failure_gain: 10.0
    rise_rate: 0.80
    recovery_rate: 0.20
  range:
    global_min: 0.05
    global_max: 0.80
    failure_threshold: 0.75
    failure_gain: 12.0
    rise_rate: 0.80
    recovery_rate: 0.20
```

不配置 `prior_reset`（或将 `max_strength` 设为 0）就是 v2a，prior assignment 仍为
`pairwise_mass_transfer`。v2b 可在同一 controller 下增加：

```yaml
  prior_reset:
    max_strength: 0.35
```

只有当前 trial 确实替换了 hypothesis 时，v2b 才生效。实际混合强度由归一化后的 global-search
状态从 0 连续增加到 `max_strength`；它把 pairwise-transfer prior 与当前 active set 上的基础先验
混合，使 persistent failure 引入的远端新 hypothesis 不再只能继承被淘汰 hypothesis 的很小质量。
没有 replacement、global search 位于基线或未配置该项时，prior assignment 与 v2a 完全相同。

v2e 可在同一 controller 下加入 persistent overt execution：

```yaml
  execution:
    enabled: true
    switch_scale: 0.20
```

它把“工作空间里正在考虑的 hypotheses”和“当前真正用于作答的 hypothesis”分开。每个
trajectory/particle 保存一个 `executed_hypothesis`，choice 只从该 rule 读出；一个 workspace
slot 在本 trial 的内部搜索中保护该 rule，其余 `capacity - 1` 个 slot 继续替换。为维持
`E_t` 的总体探索含义，搜索 slot rate 改写为：

```text
m_search,t = 1 - (1 - E_t) ** (1 / (capacity - 1))
```

只有确实发生内部搜索时才有机会切换 overt rule；条件切换率为 `switch_scale`，所以
pre-choice 边际 hazard 为 `E_t * switch_scale`。切换后的 rule 从当前 active alternatives
按 transition 后 prior 抽取，不读取当前正确答案。当前 choice 只在预测完成后用于粒子权重
更新，因此一段一致的选择会逐步支持执行同一 rule 的粒子，不会造成 current-trial leakage。

该机制表示 strategy commitment、task-set inertia、perseveration 与 switching cost；它既可能
维持错误 rule 形成深谷，也可能维持正确 rule 形成 mastery。关闭 `execution.enabled` 时仍使用
原来的 active-hypothesis average readout 和 capacity-slot replacement。

v2g 可在 v2e 的 `execution` 下选择性加入 history-supported misconception capture：

```yaml
  execution:
    enabled: true
    switch_scale: 0.20
    misconception_capture:
      enabled: true
      choice_decay: 0.85
      failure_threshold: 0.55
      min_evidence_trials: 6
      min_advantage: 0.05
      min_choice_compatibility: 0.70
      min_dwell_trials: 8
```

每个 particle 为全部 hypotheses 保存“近期被试选择与该 rule 一致”的指数衰减比例。trial `t`
结束后才用 choice 更新，因此 trial `t` 的 trace 最早在 trial `t+1` 参与决策。当 failure
pressure 超过 `failure_threshold` 且历史长度足够时，这个 trace 会：

1. 提高更能解释近期选择的 inactive rule 被搜索进 workspace 的概率；
2. overt switch 发生时，若最佳 alternative 比当前 executed rule 至少高
   `min_advantage`，且自身 choice compatibility 不低于
   `min_choice_compatibility`，将切换定向到该 alternative；
3. capture 后用 `min_dwell_trials` 抑制 overt switch，但内部 workspace search 仍继续。

`min_choice_compatibility: 0` 是保持 v2g 行为的向后兼容默认值；提高它只收紧 capture，
不改变 choice-trace 更新、failure controller 或普通 overt switch。

dwell 到期后，后续选择改变 trace，或普通 overt switch 再次发生，模型即可恢复。该机制表示
被试在连续失败后仍把近期选择组织成一条自洽但可能错误的规则，并短暂坚持它；它不读取目标
类别、不使用未来 choice，也不硬编码任何被试或 trial 区间。关闭
`misconception_capture.enabled` 时 v2e 行为不变。

更保守的 v2j/v2k 结构探针使用 full-space guided rule commitment：

```yaml
  execution:
    enabled: true
    switch_scale: 0.20
    rule_commitment:
      enabled: true
      choice_decay: 0.875
      failure_threshold: 0.70
      min_evidence_trials: 16
      min_prior_mastery: 0.70
      min_choice_compatibility: 0.80
      min_runner_up_margin: 0.10
      entry_probability: 1.0
      min_dwell_trials: 8
      min_hold_choice_compatibility: 0.60
      disconfirmation_decay: 0.90
      recovery_threshold: 4.0
      reentry_cooldown_trials: 16
```

它与 v2g 的区别是：候选从完整 hypothesis space 中直接解析，并在满足门控时保证进入 workspace
和成为 overt executed rule；active commitment 暂停普通 overt switch。`min_prior_mastery`
使用截至 trial `t-1` 的历史 mastery 最大值，避免把最初学习阶段的一串错误误判为“学会后的
错误规则回退”。达到 `min_dwell_trials` 后，累计负反馈超过 `recovery_threshold`，或候选的
历史 choice compatibility 低于 `min_hold_choice_compatibility`，都会解除固着并进入 cooldown。
将两个新阈值保留为默认值 0 时，不改变最初 v2j 的 entry/release 逻辑。

PF 另存 commitment active/eligible/entry/exit 概率、age、disconfirmation、margin、confidence
signal/precision，以及 `predictive_peak_mastery_evidence`。这些字段全部是 pre-choice 边际量；
当前 trial 的 choice 只会影响下一 trial。`rule_commitment` 与 `misconception_capture` 互斥，且
默认关闭。当前 selected-eight 守门结果不支持全局默认启用，因此应作为预先声明的模型比较
分支使用，而不是无条件应用到所有被试。

不得把 v2 controller 与 legacy
`rate_controller`/`range_controller` 或顶层 `m_beta_*`/`g_beta_*` 同时配置。PF 日志保存
`predictive_failure_pressure`、`predictive_mastery_evidence`、
`predictive_exploration_target`、`predictive_global_target`、
`predictive_prior_reset_strength` 和 `predictive_prior_reset_mass_shift`，用于验证触发信号、策略输出
及先验重置的实际作用量。

v2e 另外保存 `executed_hypothesis`、`execution_switch_probability`、
`execution_switch_event` 和 `execution_dwell_trials`。
v2g 还保存 capture eligibility/hold/switch 的 PF 边际概率，以及当前 executed rule 和最佳
alternative 的 choice compatibility；这些都是 current-choice 之前的预测状态。

## 8. 优化与推理的分工

optimization 和 trial-level inference 位于不同层级。

### 8.1 优化

优化器为每个被试选择或估计固定参数：

- static：选择 strategy candidate 或估计固定 strategy parameters；
- dynamic-discrete：选择 controller candidate，或估计固定 controller parameters
  \(\phi_s\)；
- dynamic-continuous：估计 control-update parameters \(\phi_s\)。

这些参数可以通过 Grid、CD 或未来的连续优化得到。优化方法是外层搜索方式，不决定模型属于
static、dynamic-discrete 还是 dynamic-continuous。

### 8.2 推理后端

给定一组被试级参数后，inference backend 负责运行或积分 trial trajectory：

- static 没有额外 policy controller state，但 active-set replacement 仍可能随机；
- dynamic-discrete 需要产生或积分 (z_{1:T})；
- dynamic-continuous 需要递推或积分 (c_{1:T})。

trajectory backend 运行一条路径；particle-filter backend 可以对不可见的随机路径做近似积分。
因此 inference backend 与 Grid/CD 可以自由组合，只要对应模块实现了所需的状态快照、恢复和
future reseeding 接口。

## 9. 配置示例

### 9.1 固定策略

```yaml
class: src.Bayesian_state.model.modules.hypothesis_transition.fixed_strategy.FixedStrategyHypothesisTransitionModule
kwargs:
  strategies:
    - {amount: fixed, value: 2, method: top_posterior, pool: active}
    - {amount: fixed, value: 1, method: random, pool: inactive}
  post_to_prior:
    method: conservative_carryover
    newcomer_mass: 0.10
```

### 9.2 动态离散策略

```yaml
class: src.Bayesian_state.model.modules.hypothesis_transition.dynamic_discrete_strategy.DynamicDiscreteStrategyHypothesisTransitionModule
kwargs:
  state_controller:
    method: feedback_gated_softmax
    activation: {temperature: 0.85}
    features:
      recent_accuracy_window: 8
      accuracy_delta_window: 8
      feedback_mode: exact
      padding: chance
    states:
      - id: stable
        strategies: [...]
        post_to_prior: {method: conservative_carryover, newcomer_mass: 0.05}
      - id: explore
        strategies: [...]
        post_to_prior: {method: similarity_novelty}
```

### 9.3 动态连续控制

```yaml
class: src.Bayesian_state.model.modules.hypothesis_transition.dynamic_adaptive_control.DynamicAdaptiveControlHypothesisTransitionModule
kwargs:
  # 可由 subject_overrides block 提供，并在该被试的全部 trial 中保持固定
  # trials for that subject; this module does not implement a dynamic M_t.
  capacity: 3
  m: 0.15
  g: 0.35
  continuous_controller:
    rate:
      m_phi: 0.50
      m_beta_surprise: 0.80
      m_beta_uncertainty: 0.00
    range:
      g_phi: 0.20
      g_beta_surprise: 0.40
      g_beta_uncertainty: 0.00
```

## 10. 因果性、日志与扩展约束

- trial (t) 的 controller 只能使用 transition 前已经可用的信息；当前 trial feedback 不能反向
  决定同一 trial 的 state。
- 所有公开模式必须通过 `contracts.py` 的公共 lifecycle。
- 所有有状态模块必须实现快照、恢复和日志清理；随机模块还必须支持 future reseeding。
- dynamic-discrete 日志应记录 `selected_state` 和 `state_probabilities`。
- dynamic-continuous 日志应记录 control trajectory，例如 `predictive_m`、`predictive_g`。
- candidate/profile 是被试级配置资源，不应保存某次运行产生的 (z_{1:T}) 或 (c_{1:T})。
- `capacity` 可以作为被试级候选由 Hyper-CD 选择；选定后必须在该被试内保持固定。若要实现
  trial-varying capacity，应新增显式状态转移，而不是在运行中直接修改 `capacity`。
- `selection.py`、`prior_assignment.py`、`workspace.py` 与 `execution.py`
  是共享实现而非稳定配置入口，配置文件不得直接引用。

本子包不提供重构前的 H class path 或兼容入口。新增 H 设计时，应先明确它改变的是 selection、
prior assignment，还是二者；再明确 trial-level policy state 是不存在、离散还是连续，最后才决定
使用哪种 optimization 和 inference backend。
