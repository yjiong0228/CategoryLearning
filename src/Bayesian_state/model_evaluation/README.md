# model_evaluation

本目录负责读取已经完成的 simulation 输出，计算或复核指标，并生成状态轨迹、行为 PPC 和
oral/model alignment 图表。它不参与模型拟合，也不改变 hyperparameter selection。
它可以用冻结参数运行明确的评价协议，但不得根据评价结果重新搜索参数或覆盖冻结配置。

## 文件

| 文件 | 职责 |
|---|---|
| `model_evaluation.py` | `ModelEval` 通用评价门面：accuracy、choice Brier、posterior/prior、beta、行为 PPC 和 trajectory-rank 图表 |
| `transition_evaluation.py` | 仅在相应日志存在时使用的 dynamic-discrete、dynamic-continuous、active-set 与 particle-marginal 诊断 |
| `oral_model_alignment.py` | oral center/region 到 hypothesis space 的映射、分布相似性、coverage/target/hit alignment |
| `__init__.py` | 包说明；正式 CLI 位于上一级 `run_model_evaluation.py` |

`ModelEval` 组合 `TransitionEvaluationMixin` 与 `OralModelAlignmentMixin`，所以公共行为评价、
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

`run_model_evaluation.py` 将 subject JSON 规范化为 `ModelEval` 使用的 result mapping，并在
`<input-dir>/model_evaluation/` 下写图、CSV 和 `evaluation_manifest.json`。

## 日志依赖

- 基础 accuracy/Brier 图只需要 representative-run metrics。
- posterior/prior/beta 图需要相应 state log。
- dynamic-discrete profile 需要 `state_probabilities` 或 `policy_probabilities`；active-set 图需要
  `active_total`、`strategies` 或 `profile_policy`。CLI 按这些日志字段判断能力，不按模型类名判断。
- dynamic-continuous 的 `predictive_m`/`transition_rate`、`predictive_g`/`search_range`、
  feedback surprise/uncertainty 会被识别为独立能力，并生成控制轨迹和反馈信号图。
- particle state log 额外生成归一化 pre/post-choice ESS、重采样事件和
  `marginal_active_probability` heatmap；顺序留出的切分点会画在这些 trial-level 图上。
- trajectory rank、posterior rank 和 behavior PPC 的完整 run 分布需要 `raw_runs_ref`。
- oral alignment 还需要 Task2/oral 数据与相同的 partition 定义。

particle backend 保存的是 `marginal_prior`、`marginal_active_probability` 和 ESS/transition
诊断，不是每个粒子的 posterior 轨迹。结果 adapter 将 `marginal_prior` 显式映射为通用
`prior_log`，并以 `state_distribution_kind: particle_marginal` 标记其统计含义；它不会把该对象
冒充 posterior。因而粒子结果可以复用 prior 图，并使用专门的 marginal active/ESS 图；不存在
真实 posterior 或 beta 日志时，manifest 将对应步骤记为 `not_applicable`。

因此，若后续计划画 run-level 图，最终 simulation 应设置：

```yaml
keep_logs: true
```

## 指标边界

共享数值定义以 `metrics/` 为准；`utils/simulation_statistics.py` 暂时保留旧 import path 和
结构化 statistics schema。本目录负责评价 protocol、结果适配、比较、图表和解释，不应重新
实现 Brier、NLL、CRPS、学习曲线或行为统计。

## 添加新评价

1. 先确认输入来自 subject JSON 还是 raw-run stream。
2. 将纯数值指标放到公共 `metrics/` 层。
3. 将结果读取、汇总表和作图放在本目录。
4. 在 `run_model_evaluation.py` 增加可跳过的执行步骤。
5. 把所有产物登记进 manifest，避免静默漏图。
