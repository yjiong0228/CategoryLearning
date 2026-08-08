# Hypothesis transition（H 模块）

H 模块描述被试怎样从上一试次的 hypothesis posterior 过渡到下一试次的 hypothesis prior。
它只包含两个认知步骤：

1. **Hypothesis selection**：决定下一试次考虑哪些 hypotheses；
2. **Prior assignment**：在新的 active set 上分配 prior probability。

static、dynamic-discrete 和 dynamic-continuous 的区别，不在于是否产生不同 active set，而在于
控制这两个步骤的 **trial-level policy state** 是否以及怎样随试次变化。

## 1. 目录与公开边界

```text
hypo_transition/
├── process.py
├── static.py
├── dynamic_discrete.py
├── dynamic_continuous.py
├── candidates/
└── _internal/
```

四个公开文件是：

| 文件 | 职责 |
|---|---|
| `process.py` | 定义所有模式共同遵守的 select → assign-prior 过程 |
| `static.py` | 实现被试内部不随 trial 改变的 selection/prior policy |
| `dynamic_discrete.py` | 实现随 trial 改变的离散 strategy state (z_t) |
| `dynamic_continuous.py` | 实现随 trial 改变的连续 control state (c_t) |

`_internal/` 只是代码复用层，不是第四种认知模式，也不能在 model YAML 中直接配置。
`candidates/` 保存版本化的被试级 candidate 资源，不保存 trial-level state trajectory。

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

`process.py` 将执行顺序固定为：

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

## 4. Static strategy

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

### 4.1 Static strategy candidate

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

### 4.2 Static 与固定分支规则

一条 static policy 可以包含固定的条件分支，例如“上一试次错误时增加 exploration”。只要这仍
被建模为一个不可分的固定映射，并且没有显式 trial-level controller state，它仍归入 static。
如果这些分支被显式表示为可记录、可转移的 strategy states，则应归入 dynamic-discrete。

## 5. Dynamic discrete

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

### 5.1 State 与 profile/candidate

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

### 5.2 Deterministic 与 stochastic state

dynamic-discrete 的 state update 可以是：

- stochastic latent transition；
- history-gated softmax sampling；
- deterministic argmax gating；
- 由预先规定的 trial schedule 决定。

离散动态的定义来自显式 (z_t)，不要求 state transition 一定随机。

## 6. Dynamic continuous

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

## 7. Selection 与 prior assignment 可以采用不同模式

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

整体仍归入 `dynamic_continuous.py`，因为完整 H policy pair 已经随 trial 变化。

## 8. Optimization 与 inference 的分工

optimization 和 trial-level inference 位于不同层级。

### 8.1 Optimization

优化器为每个被试选择或估计固定参数：

- static：选择 strategy candidate 或估计固定 strategy parameters；
- dynamic-discrete：选择 controller candidate，或估计固定 controller parameters
  \(\phi_s\)；
- dynamic-continuous：估计 control-update parameters \(\phi_s\)。

这些参数可以通过 Grid、CD 或未来的连续优化得到。优化方法是外层搜索方式，不决定模型属于
static、dynamic-discrete 还是 dynamic-continuous。

### 8.2 Inference backend

给定一组被试级参数后，inference backend 负责运行或积分 trial trajectory：

- static 没有额外 policy controller state，但 active-set replacement 仍可能随机；
- dynamic-discrete 需要产生或积分 (z_{1:T})；
- dynamic-continuous 需要递推或积分 (c_{1:T})。

trajectory backend 运行一条路径；particle-filter backend 可以对不可见的随机路径做近似积分。
因此 inference backend 与 Grid/CD 可以自由组合，只要对应模块实现了所需的状态快照、恢复和
future reseeding 接口。

## 9. 配置示例

### 9.1 Static

```yaml
class: src.Bayesian_state.problems.modules.hypo_transition.static.StaticHypothesisTransitionModule
kwargs:
  strategies:
    - {amount: fixed, value: 2, method: top_posterior, pool: active}
    - {amount: fixed, value: 1, method: random, pool: inactive}
  post_to_prior:
    method: conservative_carryover
    newcomer_mass: 0.10
```

### 9.2 Dynamic discrete

```yaml
class: src.Bayesian_state.problems.modules.hypo_transition.dynamic_discrete.DynamicDiscreteHypothesisTransitionModule
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

### 9.3 Dynamic continuous

```yaml
class: src.Bayesian_state.problems.modules.hypo_transition.dynamic_continuous.DynamicContinuousHypothesisTransitionModule
kwargs:
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

## 10. Causality、日志与扩展约束

- trial (t) 的 controller 只能使用 transition 前已经可用的信息；当前 trial feedback 不能反向
  决定同一 trial 的 state。
- 所有公开模式必须通过 `process.py` 的公共 lifecycle。
- 所有有状态模块必须实现快照、恢复和日志清理；随机模块还必须支持 future reseeding。
- dynamic-discrete 日志应记录 `selected_state` 和 `state_probabilities`。
- dynamic-continuous 日志应记录 control trajectory，例如 `predictive_m`、`predictive_g`。
- candidate/profile 是被试级配置资源，不应保存某次运行产生的 (z_{1:T}) 或 (c_{1:T})。
- `_internal/` 中的类不是稳定 API，配置文件不得直接引用。

本子包不提供重构前的 H class path 或兼容入口。新增 H 设计时，应先明确它改变的是 selection、
prior assignment，还是二者；再明确 trial-level policy state 是不存在、离散还是连续，最后才决定
使用哪种 optimization 和 inference backend。
