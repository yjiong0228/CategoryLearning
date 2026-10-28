# exp4：Model 1024 观测数据接入

`model1024_inputs.yaml` 登记401、402、403三名被试的输入文件和字段约定。
本阶段准备观测序列，模型的信念、搜索、精度和持续执行机制沿用共享实现。
输入检查使用实际选择和实际二值反馈。被试在任务开始前已被告知反馈具有概率性；
该事实作为实验元数据保存，具体概率知识及其学习方式留待机制比较。

## 准备输入

从仓库根目录运行，`--output` 必须是尚不存在的目录：

```bash
python -m src.Bayesian_state.workflows.runs.prepare_model_1024_exp4 \
  --config configs/exp4/model1024_inputs.yaml \
  --output results/model_1024/exp4_inputs_20261028_v1
```

默认核对配置中的全部被试；`--subjects 401` 可限定为一人。命令只读取数据并保存输入包，
正式拟合、自主模拟和参数／状态恢复分别安排。

每人的三个文件如下：

| 文件 | 内容与用途 |
|---|---|
| `observations.npz` | `stimulus`、`choices`、实际 `feedback` 和原始 `presskeys`；模型观测使用前三项 |
| `task_metadata.npz` | 原始刺激表的完整精度 `target_probs`、行为表的三位小数 `recorded_target_probs`、原试次标识、原数据行号、刺激编号及两套模糊标记 |
| `metadata.json` | 特征顺序、知觉阈值、按键映射、session边界、来源哈希和标记差异 |

总目录的 `manifest.json` 保存完整输入配置、文件哈希、实际Python及依赖版本，
状态为 `inputs_validated`。它表示输入核对通过，模型执行状态单独记为 `not_run`。
来源文件的SHA-256哈希用于检查之后读取的文件是否与本次相同。

## 时序、反馈和特征坐标

每名被试形成一条按 `iSession, iTrial` 递增的完整历史。加载器检查源记录顺序和唯一性；
出现乱序或重复标识时报告错误。`iBlock` 原样保留。session边界由零起点、左闭右开区间表示，
例如401的第一场为 `[0, 320)`，第二场为 `[320, 640)`。
同一模型实例跨session继续接受观测，活跃规则、信念、精度、执行身份和随机流持续演化。

`choice` 使用类别编号，`presskey` 来自原始行为表。加载器按原试次逐项核对选择和反馈，
并检查整段按键映射的一致性。概率较高的选择仍可能收到负反馈；这类试次完整进入学习序列。
实验概率存入任务元数据，供后续环境抽样和评价使用。学习者每题接收刺激、实际选择和实际反馈。

四个特征保持各被试数据中的颜色顺序：401为绿、黄、粉、蓝；402为绿、粉、黄、蓝；
403为黄、绿、粉、蓝。知觉模块显式使用 `noise_mode: uniform`，从exp4的
`Task1b_errorsummary.csv` 读取阈值。共享知觉实现对同一颜色的各level取平均，
再按该被试的特征顺序排列，得到均匀知觉误差的半宽。输入接入会核查阈值完整性和排列结果。

库接口位于 `src.Bayesian_state.simulation.exp4_data`：

```python
from src.Bayesian_state.simulation.exp4_data import load_exp4_subject

case = load_exp4_subject("configs/exp4/model1024_inputs.yaml", 401)
observations = case.observations()
context = case.model_context()
# 在已安装Model 1024共享机制的运行环境中，用context创建一个模型实例，
# 将observations整段传入fit_step_by_step；知觉模块显式配置noise_mode="uniform"。
```

`arrays.categories` 为 `None`，`arrays.target_probs` 保留原始刺激表中的概率。
现有部分1024研究入口按确定性类别核查反馈；概率反馈运行入口尚待接入。
当前配置用于观测准备，已有 `pmh_prob_proto` 配置保留其原模型与用途。

## 本批数据核对与验证范围

当前三人的有效记录数为960、320、512，分别有3、1、2场，共1792题。
实际刺激、选择、反馈和颜色坐标均与原始记录对应。
行为表概率与原始刺激概率的差异处于三位小数舍入范围内。

402有54题的行为表 `ambiguous` 与对应刺激表标记不一致。
输入包分别保存 `ambiguous` 和 `stimulus_ambiguous`，差异试次列在个人元数据中。
观测学习使用全部有效试次，这两套标记均不参与筛选；任务停止规则的复现另需核查标记来源。

定向测试覆盖原始记录对齐、随机负反馈保留、特征与知觉阈值映射、无效输入拒绝、
环境概率与模型观测分离，以及旧输出保护。共享1024机制的接入测试使用401的前324题，
比较逐题连续执行与第一场分段输入后的延续，检查跨场状态、预测概率及固定种子一致性。
该测试使用小型相似度采样和示例参数，用于接口验证；参数拟合与科学恢复尚待开展。
