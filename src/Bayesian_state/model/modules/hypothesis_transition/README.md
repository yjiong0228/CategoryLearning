# 假设转移（H 模块）

Model 0923 将其称为“假设更新模块（H）”，负责搜索与候选管理。

**2026-10-14 full-v3实验干预：** `mechanistic_rule_search.py`接受可选的`intervention`映射。
`B`保留原流程；`S_control`乘搜索赔率，`S1`累计失败；`R_control`乘退出率，`R1`按近期规则反馈证据控制退出；
`C1`让满容量时的新旧候选比较临时权重。省略配置时使用B。
S1和R1的新增状态随快照保存，R1在下一题读取上一题已完成的反馈似然。
该入口用于固定参数的机制对照，详细定义、配置和结果见
[第一轮机制干预](../../../docs/model_architecture/model_0923_mechanism_interventions.md)。

**2026-10-15 连续证据与联合保留实验：** 同一入口增加`E`与`J`两类配置，
用于连续证据筛查、训练前缀拟合和参数诊断。研究设计、数据分界及结果见
[H模块连续研究](../../../docs/model_architecture/model_0923_h_design_study.md)。

E为每条在场规则记录折扣反馈证据`retention_evidence`及折扣观察量`retention_exposure`。
下一题开始时，模型用上一题学习更新前的反馈似然更新这两个量。
证据增量是规则反馈似然与无类别知识时反馈概率之比的自然对数；正值表示该条反馈支持规则。
`decay`可取0至1的保持率，或`memory`以共享记忆模块的保持率。
累计证据为负时，规则按原full-v3概率自主退出；证据为非负值时，本步骤的退出概率为0。
若抽样后没有获得旧信念支持的规则，恢复原信念最高的一条，再进入搜索步骤。
容量不足时，`capacity`指定旧规则的淘汰分数：`posterior`使用相对信念，
`evidence`使用累计证据，`evidence_mean`使用累计证据除以折扣观察量。
满容量的搜索事件移除分数最低的一条旧规则，并纳入新规则。
尚未获得反馈的规则分数为0；旧规则离开或规则重新进入时，相应历史重置。

J先按原局部／全局核提出新规则，再从旧规则和该候选中选择符合容量的非空集合。
集合权重是成员分数之和的指数。每条规则的分数为反馈证据减去`cost`，
旧规则另加非负的`stay_bias`；新规则从证据0开始。
`score: sum`使用累计证据，`score: mean`使用平均证据。
`refresh: every_trial`每题重新选择集合，`refresh: search`只在发生搜索时选择。
`candidate_retention.py`用动态规划精确汇总允许的集合，并按该分布抽样。
J的移除记录在`joint_dropped`，原自主退出与容量替换字段分别保留各自的事件含义。
日志中的条件准入概率在搜索抽样前计算，表示假定本题发生搜索时，新规则进入集合的概率。

E与J均可设置`search: accumulate`以使用S1的累积失败搜索，默认值`original`使用原搜索控制器。
集合确定后，按full-v3的规则归一化幸存者信念，并通过affinity（规则功能相似性）向新规则分配支持。
J仅留下新规则时，其信念为1；若幸存者的旧信念总和为0，按基础先验重新分配支持。
连续证据、折扣观察量及适用的累积失败状态都参与快照保存和恢复。
当前B仍为默认配置；E、J及其组合的机制去留按连续研究中的证据判断。

以下full-v1说明保留旧版流程；当前full-v3使用连续作答贡献、无执行保护的自主退出和最终集合的一次信念分配。

Model 0923 B0 新增 `unified_rule_search.py` 公共模式：一条失败痕迹、
每题最多一个候选修订、固定局部/全局比例，以及保留规则质量不变的迁移。
它复用下述 select → assign-prior 生命周期；旧模式保留原二项计数与随机流。
参见[0923 规范](../../../docs/model_architecture/model_0923.md)。

Model 0923 完整模型已接入公共模式
MechanisticRuleSearchHypothesisTransitionModule（mechanistic_rule_search.py），
配置版本为 0923-full-v1。它沿用本包的两步生命周期、反馈学习、BetaModule、粒子过滤
及选择／口述读出，不改变 B0 或 0826 配置的含义。

完整模型初始只抽取一个候选，max_capacity 是数量上限。每题非执行候选按
exit_rate × (1 − posterior) 退出；退出与新搜索分别发生，不自动补满。
搜索读取最新已完成反馈，failure_decay 是可按被试覆盖的参数；
基础搜索、失败增益、失败保持、基础全局比例和失败范围增益共用一条失败状态。

局部搜索以 local_structure_weight 混合结构编辑与功能提出。结构语法位于
../../../hypothesis_space/rule_programs.py，统一表示判断特征、比较原语、参考和分支，
保留类别标签。edit_costs 必须明确列出 feature、relation、reference、branch 四项正成本。
示例分别为 1、1.5、1、2，是共同的初版假设，不是心理测量值；
关系编辑替换完整比较原语及其标准参照，分支编辑只改条件树的一条下级判断。
没有可用结构邻居时回退到功能提出，保留局部来源并记录回退。

功能关系在固定种子和共同均匀样本上计算完整类别对应张量；
功能一致率是其对角线之和。独立的只读内存缓存不读写旧相似度资源。
tau_proposal 固定为 0.10；transfer_scale 是独立的被试支持泛化参数。
新人按功能关系从旧支持中获得质量；满员竞争可拒绝新人或移除非执行旧候选。
移除旧候选后重新计算最终集合的守恒迁移，不重复从临时权重扣质量。

每题按 prior × exp(execution_inertia × previous_execution_indicator) 抽取执行规则，
不要求搜索。choice 和反馈前 oral readout 共用这条执行规则。
幸存者保留 beta，新人恢复 beta_init；三值任务保留幸存者的条件配对，
新人继承迁移前的整体配对边际。配对边际本身不承诺在规则迁移后保持不变。

transition_log 区分 search_event（尝试）、admission_event（入场）、
autonomous_exits（自主退出）、capacity_dropped（满员淘汰）与 execution_switch_event。
为兼容现有 PF 汇总，swap_event 表示新人入场，swap_probability 表示
搜索概率乘当前剩余集合下的条件准入概率；replacement_count 仅计满员替换，
replacement_fraction 用旧实际候选数量作分母。PF 的 predictive_exploration_target
保存搜索概率 E，predictive_search_range 保存全局比例 g。
不能将搜索尝试、候选入场与执行切换解释为同一个事件。

三条件示例配置为 configs/exp123/model_struct/model_0923_cond{1,2,3}_full.yaml。
数值仅用于运行和机制审查。初始化、退出函数、单候选提议、执行保护及准入竞争
仍是需要审视的模型假设；完整拟合、参数恢复和 0826 表现保持尚未验证。

2026-10-10 验证：297 项相关测试通过，包含期刊已保存的独立数值参考；
三个条件各一名被试、32 题、8 粒子、一个 worker 的真实数据 CLI 检查通过。
另核对输出概率归一化、候选数量上限、反馈使用时序及条件 3 配对分布。
运行环境为 Python 3.11.10、NumPy 1.26.4、SciPy 1.13.1、PyYAML 6.0.2。
这些检查验证实现与流程，未验证拟合表现或参数可识别性。

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
├── feedback_reactive.py
├── nested_feedback_accumulator.py
├── unified_rule_search.py
├── mechanistic_rule_search.py
├── candidate_retention.py
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
| `feedback_reactive.py` | 只读取上一 feedback 的一步反应 H3 参照模式 |
| `nested_feedback_accumulator.py` | 严格嵌套 constant/reactive/accumulator 的 H4 模式 |
| `unified_rule_search.py` | 0923 B0：统一失败控制与单候选修订 |
| `mechanistic_rule_search.py` | 0923 完整模型：可变候选数量、结构与功能提出、关系驱动支持迁移 |
| `candidate_retention.py` | 非空、容量受限候选集合的精确分布与采样 |
| `selection.py` / `prior_assignment.py` | 选择和 prior 分配的共享机制 |
| `workspace.py` / `execution.py` | bounded workspace 状态与执行逻辑 |

表中前八个文件定义认知模式或公共契约；后五个是共享机制，不能在
model YAML 中当作独立 H 模式配置。被试级 candidate 资源位于
`configs/shared/candidates/hypothesis_transition/`，不保存 trial-level state trajectory。

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

### 9.4 一步反馈反应控制

```yaml
class: src.Bayesian_state.model.modules.hypothesis_transition.feedback_reactive.FeedbackReactiveHypothesisTransitionModule
kwargs:
  capacity: 3
  feedback_reactive_controller:
    event_after_correct: 0.20
    event_after_error: 0.50
    initial_event_probability: 0.20
    global_search: 0.30
  prior_assignment: {method: pairwise_mass_transfer}
```

该模式在 trial `t` 开始前消费 trial `t-1` 已完成的 feedback；当前 trial outcome 不会进入
同一 trial 的 transition。`event_after_*` 表示“至少替换一个 workspace slot”的概率，模块会
按 capacity 转换为 binomial slot rate。selection、newcomer proposal 与 prior transfer 和其他
bounded-workspace 模式完全相同。

### 9.5 嵌套式一步反应 + failure accumulator

```yaml
class: src.Bayesian_state.model.modules.hypothesis_transition.nested_feedback_accumulator.NestedFeedbackAccumulatorHypothesisTransitionModule
kwargs:
  capacity: 3
  nested_feedback_accumulator_controller:
    event_after_correct: 0.20
    event_after_error: 0.50
    initial_event_probability: 0.20
    global_search: 0.30
    accumulator_decay: 0.60
    accumulator_logit_gain: 0.00
    global_search_failure_gain: 0.00
    initial_failure: 0.00
    event_history_excludes_latest_error: true
  prior_assignment: {method: pairwise_mass_transfer}
```

令 `e_(t-1) = 1 - feedback_(t-1)`，该模式只维护一个历史状态：

```text
F_t = accumulator_decay * F_(t-1)
      + (1 - accumulator_decay) * e_(t-1)
H_t = F_(t-1)  # Model 0826 event-history input
logit(E_t) = logit(E_reactive,t) + accumulator_logit_gain * H_t
g_t = global_search
      + (1 - global_search) * global_search_failure_gain * F_t
```

`E_reactive,t` 在正确后等于 `event_after_correct`，错误后等于
`event_after_error`。因此最近一次反馈只进入 reactive baseline，正的 accumulator gain 只读取
更早的错误史。`event_history_excludes_latest_error` 默认 false 仅用于复现历史 H4/H5 配置；
Model 0826 将其固定为 true。边界关系是：

```text
accumulator_logit_gain = 0
and global_search_failure_gain = 0
    -> 与 feedback-reactive 逐 trial 严格相同

accumulator_logit_gain = 0
and global_search_failure_gain = 0
and event_after_correct = event_after_error
    -> constant-event bounded workspace
```

两个 gain 都为零时 `accumulator_decay` 不可辨识，因此优化器必须把精确的零作为显式候选，而不能期待
连续优化自动落在该边界。`global_search_failure_gain=0` 保持固定 global mixture；正值只复用同一
`F_t` 增大全局 newcomer proposal 的比例，不增加第二套 range accumulator。独立 mastery state、
阈值和额外 rise/recovery smoothing 仍不属于这个精简控制器。完整模板见
`configs/exp123/model_struct/pmh_model_cond1_0815_h4_nested_feedback_accumulator.yaml`。

H3 继续作为纯 execution-off 参照。H4/H5 则允许在同一个 nested controller 下配置：

```yaml
persistent_execution:
  enabled: false
  switch_scale: 0.20
```

`enabled` 是被试级二元结构坐标；关闭时不保存 executed rule，选择继续对 workspace posterior
求边际，并与未写该配置的路径严格相同。开启时每个 trajectory/particle 保存一条
`executed_hypothesis`，保护它不被普通 workspace replacement 淘汰，并由该规则单独产生当前选择；
只有发生搜索事件时才可能按固定 `switch_scale` 转向另一条 active rule。`switch_scale` 在当前架构中
固定，不作为第二个被试级连续自由度。execution 状态随 module snapshot 和 PF resampling 一起传播。
这一扩展不改变 accumulator 的零边界：execution-off 且两个 accumulator gain 为零时仍与 H3
逐 trial 严格相同。

历史低维筛选入口为 `src/Bayesian_state/workflows/runs/run_model_0815_h4_nested_subject_screen.py`。它先在训练 trials 上
顺序校准 `event_after_correct`、即时错误 logit gain 与固定 `global_search`，再在三种共同 decay
下 profile 包含精确 0 的 subject gain。正 gain 必须同时越过训练实用阈值和配对 PF seed-noise
门槛，否则保留 0。最终留出评估使用不同的 seed role，不能反向参与参数选择。对应预声明配置为
`configs/exp123/specific_models/model_0815_h4_nested_subject_screen.yaml`。该入口不包含新加入的动态 global
gain，且其 32/32 切分结果只保留为历史技术产物，不再用于当前架构去留决定。

### 9.6 H5 similarity-transport prior assignment

H4 的历史配置把每条 dropped hypothesis 的 posterior mass 原样交给与之随机配对的 newcomer。
H5 保留这一方法作为兼容路径，但当前架构改用：

```yaml
prior_assignment: {method: similarity_transport}
```

令 `K_t` 为本 trial 实际替换数、`C` 为固定 capacity、`r_t=K_t/C`。`c_t` 是 survivors 上按旧
posterior 归一化的 carry-over distribution；`z_t` 则把完整旧 posterior 通过当前
`(1-g_t) local + g_t global` kernel 投影到新 active set。最终

```text
prior_t = (1 - r_t) * c_t + r_t * z_t
```

`K_t=0` 时严格返回旧 posterior；全部槽位替换时完全使用 semantic projection。该方法没有
prior-specific 拟合参数：`E_t` 通过 realized `K_t` 决定更新比例，`g_t` 决定概率转移的局部/全局
范围。H5 将 `tau_local=0.10` 固定为公共架构尺度，不允许它与 `g_t` 一起成为被试级的两个
搜索宽度补偿参数。该核表示固定标签规则的功能一致性，而不是另行假定的口头/心理结构距离。
它不与 failure-accumulator `prior_reset` 叠加。新模板为
`configs/exp123/model_struct/pmh_model_cond1_0815_h5_similarity_transport.yaml`；H4 配置和既有结果保持不变。
该 H5 模板同时暴露默认关闭的 `persistent_execution.enabled`；被试级拟合只枚举 false/true，
不改变 similarity transport、workspace capacity 或固定的 execution switch scale。

Model 0826 还提供一个一次性、无新参数的质量守恒反事实：

```yaml
prior_assignment: {method: mass_preserving_similarity_transport}
```

令 $D_t$ 为 dropped rules、$N_t$ 为 newcomers，$m_t^{\rm drop}$ 为被删除 posterior mass，
$\widetilde z_t$ 为只在 newcomers 上归一化的同一 local/global semantic projection，则

```text
m_drop = sum(posterior_(t-1)[D_t])
prior_t[survivors] = posterior_(t-1)[survivors]
prior_t[newcomers] = m_drop * z_tilde[newcomers]
```

这个反事实只检验 workspace turnover 是否需要槽位比例的 belief reallocation；配置
`configs/exp123/specific_models/model_0826_belief_transport_counterfactual.yaml` 保证 A/B 之间仅改变
`prior_assignment.method`。

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
- `candidate_retention.py`、`selection.py`、`prior_assignment.py`、`workspace.py` 与 `execution.py`
  是共享实现而非稳定配置入口，配置文件不得直接引用。

本子包不提供重构前的 H class path 或兼容入口。新增 H 设计时，应先明确它改变的是 selection、
prior assignment，还是二者；再明确 trial-level policy state 是不存在、离散还是连续，最后才决定
使用哪种 optimization 和 inference backend。

## 11. Model 0923 历史证据实验

`episodic_admission.py`通过已完成反馈评价新规则，并在POOL方案中用同一段折扣历史比较新旧规则。
`episodic_proposal.py`在POOL之上提供两个独立的实验开关：历史证据引导候选身份，以及新规则入集时
依据共同历史重建规则信念。其入口为`workflows.analysis.model_0923_recalled_evidence_study`，
配置为`configs/shared/model0923_recalled_evidence_study.yaml`。正式默认模型的装配保持原配置。

历史证据是各次反馈的对数似然比之折扣和；似然表示规则对已经观察到的反馈赋予的概率，
比较基准为相应条件下的随机选择反馈概率。折扣率沿用保留证据的保持参数，所有规则均在初始精度上回顾。
条件3使用每条反馈产生前记录的配对分布。SEARCH以该证据调整原提议概率，保留原局部／全局路线、
父规则及语法允许的候选范围。PRIOR在候选被接纳后，从基础规则权重与共同历史证据重新形成最终集合的信念。
两次集合变更之间沿用原在线信念更新。

历史重建假设被试在纳入新解释时重新评估既往经验，其计算精度与当时在线精度可以不同。
这项假设需要行为重拟合和表征检验。历史证据缓存保存已完成事件链所确定的数值和，随粒子快照共享，
搜索时只计算缓存之后新增的事件；缓存复用不增加独立学习参数。
试验采用POOL、PRIOR、SEARCH和BOTH四种组合，全部保留自主轨迹。
设计依据、检验边界及计算状态见[持续研究记录](../../../docs/model_architecture/model_0923_research_progress_20261018.md)。

在SEARCH中，活跃集合限制当前可用于作答的规则；历史记录还可为未活跃规则提供检索依据。
因此，容量参数表示在线活跃规则数的上限。将其解释为人的工作记忆容量，需要容量操纵或相应恢复证据。
历史证据保持率控制过去反馈对检索和准入的影响，在线信念保持率控制活跃规则信念跨反馈的延续。
两者具有不同的计算步骤，其行为可区分性由定向参数恢复检验。
当前实现对完整目录计算候选检索概率；所需心理计算成本尚未检验。

`compact_evidence.py`提供显式启用的 H 简化对照。活跃规则的一份在线证据同时用于作答、父规则
选择和容量竞争；搜索提出的新规则从零证据开始，集合满时替换支持最低的旧规则。搜索沿用失败
累积控制，局部提出使用结构邻域。设计、参数关闭方式和比较协议见
[Model 1021 简化研究](../../../docs/model_architecture/model_1021_simplification.md)。
执行入口为`python -m src.Bayesian_state.workflows.analysis.model_1021_simplification`，
配置为`configs/shared/model1021_simplification.yaml`；已有默认装配保持原行为。

固定参数对照后，`workflows.analysis.model_0923_recalled_search_refit`以相同预算比较POOL和SEARCH，
仅用训练选择估计参数。`workflows.analysis.model_0923_recalled_memory_recovery`进一步检查两种记忆保持率，
该恢复阶段固定其余参数为生成值。各入口的协议、科学结果和限制保存在持续研究记录中。

`compact_belief.py`提供保留 Model 1022 各模块的规则信念对照。记忆模块的规则后验用于作答、
搜索父规则、容量替换及精度继承；新规则获得最终集合内的一份均匀先验质量，旧规则按原后验
比例分配剩余质量。条件3分别保留规则与配对知识，再更新同一联合信念。参数数量保持八项／九项。
配置、数据边界与评价解释见[信念对照说明](../../../docs/model_architecture/model_1022_belief_comparison.md)。
运行入口为`python -m src.Bayesian_state.workflows.analysis.model_1022_belief_comparison`；
使用独立工作进程启用，输出目录必须为新目录。
