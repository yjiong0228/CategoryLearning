# model_evaluation

本目录负责读取已经完成的 simulation 输出，计算或复核指标，并生成状态轨迹、行为 PPC 和
oral/model alignment 图表。它不参与模型拟合，也不改变 hyperparameter selection。
它可以用冻结参数运行明确的评价协议，但不得根据评价结果重新搜索参数或覆盖冻结配置。

## 文件

| 文件 | 职责 |
|---|---|
| `model_evaluation.py` | `ModelEval`：accuracy、choice Brier、posterior/prior、beta、strategy、active-set 和 run-rank 图表 |
| `oral_model_alignment.py` | oral center/region 到 hypothesis space 的映射、分布相似性、coverage/target/hit alignment |
| `__init__.py` | 包说明；正式 CLI 位于上一级 `run_model_evaluation.py` |

`ModelEval` 继承 `OralModelAlignmentMixin`，所以公共行为评价与口述规则对齐使用同一个结果
读取上下文。

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
- trajectory rank、posterior rank 和 behavior PPC 的完整 run 分布需要 `raw_runs_ref`。
- oral alignment 还需要 Task2/oral 数据与相同的 partition 定义。

particle backend 当前保存的是 `marginal_prior`、`marginal_active_probability` 和 ESS/transition
诊断，不是每个粒子的 posterior 轨迹。旧的 `posterior`/`prior` 绘图函数不会自动把这些字段
解释成同一对象；粒子结果应使用专门的 marginal-state adapter 或图表。

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
