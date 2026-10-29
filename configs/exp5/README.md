# Exp5 Rule1：Model 1024 第一阶段接入

本目录提供完整行为输入准备与单人短序列验证。规则信念、搜索、容量、精度学习和持续执行
使用共享 Model 1024 运行机制；任务适配采用五个有序黑白方格及现有62条奇偶规则。
规则定义、精度尺度、局部搜索和研究边界见[模型接入说明](../../src/Bayesian_state/docs/model_architecture/model_1024_exp5.md)。

## 准备行为输入

从仓库根目录运行，输出目录必须尚不存在：

```bash
python -m src.Bayesian_state.workflows.runs.prepare_model_1024_exp5 \
  --config configs/exp5/model1024_inputs.yaml \
  --output data/exp5/processed/model1024_rule1_NEW
```

`model1024_inputs.yaml`登记501–519及会话处理依据。准备过程读取原始行为与刺激表，核对
类别、反馈、按键映射、试次顺序和源文件哈希。原始数据、口述文件和旧合并表保留原状。

| 产物 | 内容 |
|---|---|
| `behavior.csv` | 全部3296条记录，附刺激、来源文件、数据行号、连续编号、会话类型和`analysis_included`标记 |
| `<subject>/observations.npz` | 本阶段使用的`stimulus`、`choices`、`feedback` |
| `<subject>/task_metadata.npz` | 正确类别、按键、原会话/试次、块和来源行号，供环境评分与核对 |
| `<subject>/metadata.json` | 会话安排、暂缓分析的试次、按键映射、每块刺激频次和输入哈希 |
| `manifest.json` | 输入协议、人数与题数、代码和输出哈希、Python及依赖版本 |

515、517、518的两会话组成连续历史。508第一天160题进入本阶段分析，第二天64题保留在
完整表中，`analysis_included=false`；本阶段分析合计3232题。`source_row`从1计数，表示
源文件中的数据行，CSV物理行号为该值加1。`condition=1`表示共享接口的二分类二值反馈，
任务身份另存为`exp5_rule1`。模型观测使用`choice`类别编号，按键映射单独保存。

## 小规模闭环检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m src.Bayesian_state.workflows.runs.run_model_1024_exp5 \
  --config configs/exp5/model1024_smoke.yaml \
  --output results/model_1024/exp5_smoke_NEW
```

默认使用503前32题、8粒子、2条自主路径和单进程。入口限制为最多64题、16粒子、8条路径。
模型参数为固定示例；`execution_inertia: 0`可运行零持续性边界，默认示例为2。
直接使用模型时，应在`model_1024_runtime()`上下文内构造和执行`StateModel`。

检查包括实际行为过滤的固定种子重复、生成反馈与模拟选择的一致性，以及用生成选择和反馈
重放同一随机历史后的信念、精度、活跃集合和执行身份一致性。另用独立推断种子重建第一条
模拟路径的选择前在场概率，保存真值与概率坐标；这项小样本检查用于验证恢复接口。

结果包含`observed.npz`、`task_metadata.npz`、每条`autonomous_*.npz`、
`recovery_smoke.npz`及`manifest.json`。实际行为和恢复的状态摘要时点均为本题选择前；
自主文件保存完整潜在轨迹及反馈后的信念。报告明确记录固定参数、条件信息和计算规模。

## 后续范围

### 被试501正式拟合

`model1024_fit_501.yaml`登记完整128题的九参数搜索与数值确认。
评分、预算、源码隔离和证据范围见[正式拟合协议](../../src/Bayesian_state/docs/model_architecture/model_1024_exp5_fit.md)。
从仓库根目录依次执行，输出目录须为新目录：

```bash
python -m src.Bayesian_state.workflows.analysis.model_1024_exp5_fit \
  --phase prepare --output results/model_1024/exp5_fit_501_NEW
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m src.Bayesian_state.workflows.analysis.model_1024_exp5_fit \
  --phase run --output results/model_1024/exp5_fit_501_NEW
```

`--phase smoke --output NEW_SMOKE`走完单进程短序列搜索流程；`--phase report`在新的报告子目录
汇总已完成运行。默认采用2小时、128 CPU小时上限。正式参数依据完整选择似然估计，
自主分布在冻结后评价。当前运行状态见对应结果目录的`progress.json`与`completion.json`。

任务的时间预留为保守CPU估计的1.1倍，实际CPU与运行时间仍由硬上限约束。
若旧配置的双倍时间预留阻止了4096粒子的部分任务，可在新目录补齐；入口会检查原队列
已启动任务的归属，沿用原截止时间，并为原运行的剩余CPU消耗预留上界：

```bash
python -m src.Bayesian_state.workflows.analysis.model_1024_exp5_complete \
  --phase prepare --source results/model_1024/ORIGINAL_RUN \
  --output results/model_1024/NEW_COMPLETION
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m src.Bayesian_state.workflows.analysis.model_1024_exp5_complete \
  --phase run --output results/model_1024/NEW_COMPLETION
```

补算只适用于501已经完整完成2048粒子检查、且原队列无法再启动余下4096粒子任务的情况。
补算保留相同候选、随机种子、科学源码和验收标准，结果报告合计两边的计算消耗。

参数可辨识性、逐题状态恢复、跨日遗忘、口述参与停止决策的完整模拟和口述对照待后续研究。
运行按实际刺激和长度推进；停止时点保留实验安排的含义。
62条规则作为接口基线，数量、整体对称和连续同色分组等策略的完整覆盖待单独检查。

旧`pmh_rule1.yaml`及其模拟/搜索配置保持原有用途。新的准备入口拒绝覆盖已有目录。
