# V13 模型改动简要总结

这份总结面向阶段性汇报，重点解释 V13 的模型变化、Subject 118 深低谷能力测试，以及当前仍需要注意的问题。

## 1. 模型主要改动

### 1.1 Hypothesis Space

Cond1 的 hypothesis space 从原来的 19 个 hypotheses 扩展为 38 个 hypotheses：保留原 19 个连续 prototype / partition 规则，再追加每个规则的 category-label reversed 版本。Cond1 是二分类，所以 label reversal 只需要交换 category 1 和 category 2。

原先模型很难稳定产生低于 chance-level 的预测正确率，核心原因是 hypothesis space 不完整：原 19 个 hypotheses 中缺少“同一个 partition 但类别标签完全相反”的规则。因此即使模型选错 hypo，很多错误 hypo 对 true category 的概率也不会低到 0，整体 accuracy 很难持续低于 0.5。

加入 label-reversed hypotheses 后，模型可以表达“规则结构学对了，但类别标签对应关系学反了”的人类错误，从而自然产生 below-chance 行为，而不是依赖随机 lapse。

## 1.2 Strategies

### 四种基础策略

V13 用 4 种基础 profile policy 替代大量固定 strategy 组合。每个 trial 先动态选择一个 profile，再执行该 profile 对 active hypotheses 的更新。

记号：

- `A_{t-1}`：上一 trial 的 active hypothesis set。
- `p_{t-1}(h)`：上一 trial 后 hypothesis `h` 的 posterior。
- `A_t`：当前 trial 的 active set。
- `K`：active set 上限，Cond1 v13 默认 `K = 5`。
- `I_t = H \ A_{t-1}`：inactive hypotheses。

| 基础策略 | 机制 | Amount / Sampling / P2P |
|---|---|---|
| `conservative` | 保守，坚持当前 active set | `n_new = 0`；`A_t = A_{t-1}` 截断到 `K`；若 active 为空，用 posterior top1 兜底；p2p 为 `conservative_carryover` |
| `stable` | 稳定学习，轻度探索 | 通常 `n_new = explore_count = 1`；active 已满时至少丢 1 个旧 hypo；旧 hypo 按 survivor score 抽样保留，新 hypo 从 inactive 抽样；p2p 为 `similarity_novelty` |
| `aggressive` | 低信心或错误后快速刷新 | 只保留 top1；`n_new = round((1 - p_top) * max_newcomers)`；newcomers 从 inactive 抽样；top1 prior 约为 `p_top`，newcomers 均分 `1 - p_top` |
| `stubborn` | 顽固，坚持少数核心 hypo | 保留 `retain_count` 个 core hypo；以 `q_explore` 概率探索 0 或 1 个 newcomer；旧 hypo 保持主要 mass，newcomer mass 较低 |

`stubborn` 的探索概率：

```text
q_explore
  = base_explore_prob
    + post_correct_explore_prob * (1 - last_error)
    + post_error_explore_prob * last_error
```

stable / stubborn 的 survivor score 可以是普通 posterior，也可以是 choice-informed：

```text
score_survivor(h)
  = p_{t-1}(h)^a * P(choice_{t-1} | stimulus_{t-1}, h)^b
```

普通 posterior survivor 等价于 `a = 1, b = 0`。choice-informed survivor 会更倾向保留能解释上一 trial 选择的 hypothesis。

### 动态策略配置

旧版策略是固定的：一次 run 从头到尾使用同一组 strategy。V13 改成动态 profile controller：

```text
s_k(t) = bias_k + w_k · x_t
P(profile = k | history_t)
  = exp(s_k(t) / T) / sum_j exp(s_j(t) / T)
```

其中 `x_t` 是历史特征，包括：

- `last_error`
- `recent_accuracy`
- `accuracy_delta`
- `posterior_entropy`
- `posterior_confidence`
- `trial_progress`

这些特征只使用当前 trial 之前的信息，不使用当前 trial feedback。

### Profile 与 Readout 分离

本次修改后，profile candidate 只控制 transition 和 p2p，不再绑定 readout。Readout 作为独立 coordinate，与 profile 做直积：

```yaml
engine.modules.hypo_transitions_mod.kwargs:
  values_from_json:
    path: ../../src/Bayesian_state/problems/modules/hypo_transition_strategies/hypo_transition_profile_v13_candidates.json
    key: cond1_v13
    value_key: hypo_transitions_kwargs

engine.choice_readout.kwargs:
  values:
    - method: expectation
    - method: map_hypothesis
```

当前 Cond1 v13 有 16 个 transition profile，因此会形成 `16 * 2 = 32` 个 profile-readout 组合。

### 候选 Profile

| Candidate | 类型 | 主要逻辑 |
|---|---|---|
| `c1_v13_stable_dominant` | 稳定学习 | stable 占主导，保守和轻度探索并存 |
| `c1_v13_error_aggressive` | 错误后刷新 | 错误和高 entropy 增强 aggressive refresh |
| `c1_v13_error_stubborn` | 错误后坚持 | 错误后更容易激活 stubborn，模拟持续错误 |
| `c1_v13_low_capacity_stubborn` | 低容量 | active core 更小，更容易丢失正确 hypo |
| `c1_v13_early_explore_late_stable` | 前探后稳 | 早期 aggressive/stable，后期 conservative |
| `c1_v13_conservative_heavy` | 保守学习 | conservative 权重高，适合平滑上升型被试 |
| `c1_v13_volatile_switch` | 经常切换 | softmax temperature 更高，profile 激活更随机 |
| `c1_v13_low_accuracy_refresh` | 低正确率后刷新 | recent_error 和 negative accuracy_delta 触发 aggressive |
| `c1_v13_low_accuracy_stubborn` | 低正确率后坚持 | recent_error 触发 stubborn，适合 extended below-chance |
| `c1_v13_confidence_locked` | 高信心锁定 | posterior confidence 触发 conservative / stubborn |
| `c1_v13_error_choice_newcomer` | 错误选择引导 | V17 启发；recent error choices guide newcomer sampling |
| `c1_v13_balanced` | 基线 | 四类 profile 更均衡 |
| `c1_v13_choice_stubborn_valley` | 大波动低谷 | 错误选择推动 stubborn wrong-rule lock-in，但保留恢复机会 |
| `c1_v13_choice_volatile_refresh` | 大波动刷新 | choice-informed + 高温度，错误后可能广泛刷新 |
| `c1_v13_choice_error_recovery_switch` | 深谷后恢复 | 错误时可进入 stubborn/aggressive，改善时转向 stable/conservative |
| `c1_v13_choice_low_capacity_wave` | 低容量波动 | 小 active set + choice-informed newcomer，使规则替换更突然 |

### Choice-informed transition

`survivor_score: posterior_choice`：

```text
score_survivor(h)
  = p_{t-1}(h)^a * P(c_{t-1} | x_{t-1}, h)^b
```

`newcomer_score: recent_error_choice`：

```text
L(h)
  = (1 / m) * sum_{i in recent error trials} log P(c_i | x_i, h)
score_newcomer(h) = exp(L(h))^w
```

这让模型更容易保留或采样能解释被试近期错误选择的 hypothesis，从而进入“稳定但错误的规则”状态。当前使用该机制的 profile 包括：

- `c1_v13_error_choice_newcomer`
- `c1_v13_choice_stubborn_valley`
- `c1_v13_choice_volatile_refresh`
- `c1_v13_choice_error_recovery_switch`
- `c1_v13_choice_low_capacity_wave`

## 1.3 能力测试：Subject 118 深低谷

测试目录：

`results/state-based-simulation/pmh/cond1_v13_subject118_probe_v17_error_choice_fast_memory`

测试目标是验证模型能否产生 Subject 118 的“中高 accuracy -> 深低谷 -> 恢复 -> 后期高 accuracy”动态范围。

关键设置：

- Cond1 label-reversal hypothesis space：38 hypos。
- Active set 上限：5。
- Readout：`map_hypothesis`。
- Profile：接近 `c1_v13_error_choice_newcomer`。
- Choice-informed transition：`posterior_choice` survivor + `recent_error_choice` newcomer。
- Memory：V17 中较好的区域约为 `gamma = 0.85`, `w0 = 0.02`。
- Beta：当前 v13 使用 `beta_init = 5.0`。

图中两类代表曲线：

1. 平均最优曲线：所有 repeats 中整体 trajectory loss / curve MSE 最小的一条 run。V17 记录中整体 best MSE 约为 `0.0096`。
2. 深谷最优曲线：特别检查深谷前、深谷、恢复、后期四段，选择最符合“先高、再低、再恢复、后期高”的 run。代表性 run 大致为：深谷前约 `0.70`，深谷约 `0.40`，恢复约 `0.68`，后期约 `0.91`。

这说明：label reversal 和 choice-informed transition 使模型具备产生 below-chance 深低谷的能力。

## 1.4 Prediction

当前默认使用 `prior_t` 做 prediction，因为当前 trial 的 choice 应由 trial 开始前的 belief 决定，而不是由当前 feedback 后的 posterior 决定。

Readout 有两种主要方法：

| Readout | 公式 | 特点 |
|---|---|---|
| `expectation` | `P(c|x) = sum_h q(h) P(c|x,h)` | 平滑、稳定，但动态范围较窄 |
| `map_hypothesis` | `h* = argmax_h q(h)`，`P(c|x)=P(c|x,h*)` | 更像“当前相信一个规则”，动态范围更大，更容易 below-chance |

这里 `q(h)` 当前默认来自 `prior_t`。

## 2. 潜在问题

### 2.1 深低谷频率仍偏低

模型已经可以产生低于 0.5 的深低谷，但不是多数 repeats 的典型行为。后续 CD 优化需要检验是否能为这类被试稳定选择合适的 profile / memory / readout 参数。

### 2.2 Choice-informed transition 的过拟合风险

Choice-informed transition 依赖被试过去的 choice 序列来影响 active set。当前实现是因果的，只使用过去 trial，不使用当前 trial feedback；但如果权重过强，模型会过度贴合具体 choice 序列。因此它应只是候选机制之一，需要限制 window 和 weight。

### 2.3 Hypothesis space 目前只解决 Cond1

Cond1 二分类 label reversal 只让 hypothesis 数量翻倍：`19 -> 38`。但 Cond2/3 是四分类，完整 label permutation 会产生 `4! = 24` 倍扩展，不可接受。后续需要更受限的 label swap 或把 label mapping 放到 readout / response layer。

### 2.4 Beta 模块仍需重新审视

Beta 控制每个 hypothesis 对 category probability 的确定性：

- beta 高：判断更尖锐，更接近确定规则。
- beta 低：判断更平滑，更不确定。

当前 v13 固定配置：

```yaml
beta_init: 5.0
beta_min: 0.1
beta_max: 25.0
decrease_rate: 0.15
correct_additive: 0.5
use_prior_scaling: true
prior_beta_scale: 3.0
beta_update_mode: inferred_correct_category
```

目前没有把 Beta 放入重点优化空间，以减少自由度。但 Beta 仍会影响动态范围：太低不够 below-chance，太高可能过早锁死，active mean beta 过高也可能不利于渐进学习。

## 3. 当前结论

V13 的核心进展是：模型不再只能表达“随机或渐进学对”，而开始能表达“学到稳定但错误的规则，然后再恢复”的学习轨迹。

关键贡献：

1. Cond1 加入 label-reversed hypotheses，使 below-chance 在结构上成为可能。
2. Profile controller 允许同一 run 内策略动态切换。
3. Choice-informed transition 让模型能把被试历史选择解释为 active hypothesis set 的一部分，从而产生更有解释力的持续错误。

当前仍是阶段性结果：深低谷可以出现但频率不够高；choice-informed transition 有过拟合风险；Cond2/3 label permutation 还没解决；Beta 模块也需要后续系统评估。
