# Hypothesis-transition candidates

本目录保存供被试级 optimization 使用的 H-transition candidate 资源。JSON 是配置数据，不是
独立模型，也不表示已经拟合出的 trial trajectory。

## 两类 candidate

- `hypo_transition_strategy_*_candidates.json`：static strategy candidates。优化器为一个被试选择
  一套固定的 selection/prior-assignment policy；policy 可以响应 posterior 或历史，但其结构和
  参数不随 trial 改变。
- `hypo_transition_profile_v*_candidates.json`：dynamic-discrete controller candidates。每个外层
  candidate 定义固定的 controller 参数和一组 `states`，controller 再逐 trial 产生离散 strategy
  state `z_t`。

文件名中的 `profile` 是历史沿用的“被试级 controller candidate/profile”，不是 trial-level
state。为避免继续混淆，代码和 JSON 内部统一使用：

- `state_controller`：离散状态控制器；
- `states`：可选的 trial-level strategy states；
- `selected_state`：某 trial 实际选中的 state；
- `state_probabilities`：该 trial 的 state 概率。

因此，一个 profile candidate 是被试级优化单位；它运行后可以产生完整的
`z_1, ..., z_T` state trajectory。不要把外层 candidate/profile 和内层 strategy state 当成同一层。

`strategy_reference.md` 记录当前 strategy-chain、prior assignment、discrete state controller
以及历史版本 candidate 的详细字段。

## 配置加载

Hyper config 一般按以下方式把某个候选集合载入 module kwargs：

```yaml
engine.modules.hypo_transitions_mod.kwargs:
  values_from_json:
    path: ../../src/Bayesian_state/problems/modules/hypo_transition/candidates/<file>.json
    key: <candidate-set-key>
    value_key: hypo_transitions_kwargs
```

路径相对于声明它的 hyper YAML 文件解析。

## 与 dynamic-continuous 的关系

`../dynamic_continuous.py` 使用连续 control state，例如 `m_t/g_t`，不从这里的离散
controller-profile 文件产生 `z_t`。0806 的连续 controller 当前直接写在 model/hyper YAML 中；
其被试级 controller 参数仍然可以由 Grid/CD 优化。

## 维护约定

- static candidate 必须能直接构造 `StaticHypothesisTransitionModule` 或相应 static 公开类。
- dynamic-discrete candidate 必须能直接构造 `DynamicDiscreteHypothesisTransitionModule`。
- 每个 candidate 应有稳定、唯一的 id，便于从 hyper result 反查。
- 修改已用于正式结果的 candidate 会破坏复现；优先新增版本文件或 candidate-set key。
- JSON 不应保存被试拟合结果、输出路径、随机运行状态或 trial-level trajectory。
