# MEG 配置

此目录提供 MEG Task3b 的 Model 0923 full-v3 拟合配置和单被试试跑入口。
探索与预处理 notebook 使用 `data/meg/` 中的数据。

## 模型与拟合配置

`simulation_cfg/model0923_task3b.yaml` 引用共享的 condition-3 full-v3 引擎。
模型使用152条四分类规则，学习未知的类别反应配对。反馈1表示所选类别正确，0.5表示同科但类别错误，0表示另一科。
配对信念按规则初始化为均匀分布；本题选择在反馈前预测，本题反馈影响后续试次。
每名被试的有效试次按 `iSession, iRun, iTrial` 保持顺序，跨 session 延续学习状态。
第一条有效试次用于初始化，后续有效试次参与选择负对数似然（NLL）评分。
NLL 越小，表示模型对实际选择赋予的预测概率越高。

333–348的知觉噪声明确设为均匀分布。各特征的噪声半宽取自 Task1b 的独立测量，
沿用当前知觉模块对五个 level 的平均与特征顺序匹配。模型内部刺激限制在 `[0, 1]`。
知觉参数固定；rating 和口述仅用于独立检验，拟合目标为选择 NLL。

`hyper_cd_cfg/model0923_task3b.yaml` 为16名被试各估计13项行为参数。
候选是有限离散网格；下表给出端点，全部候选值见 YAML。端点是初始搜索设定。

| 参数 | 搜索范围 |
|---|---|
| 基础搜索概率 | 0.005–0.95 |
| 失败推动搜索的增益 | 0–8 |
| 失败状态保持率 | 0–0.99 |
| 基础全局比例 | 0–1 |
| 失败对范围的增益 | 0–1 |
| 候选容量 | 2、3、4、6、8 |
| 退出率 | 0–1 |
| 局部结构来源倾向 | 0–1 |
| 信念迁移尺度 | 0.01–1 |
| 主导规则的作答惯性 | 0–8 |
| 证据保持率 | 0–1 |
| 初始精度 | 0.1–25 |
| 精度学习率 | 0–1 |

固定设置包括精度上下限0.1和25、共享结构编辑成本、100,000个均匀参考刺激及其种子0、
候选提名温度、零静态记忆权重和当前共同选择读出。输出噪声关闭。
搜索使用各被试的全部有效试次；当前结果属于拟合内描述，留出预测需另行设置时间切分。

搜索流程为：

1. coarse：两个随机起点，每点最多三轮坐标搜索，32粒子、4个随机重复。
2. fine：从 coarse 的两个入围点出发，在同一网格上重新搜索，64粒子、8个随机重复。
3. final rescore：最多四个入围点使用独立种子族、128粒子和16个重复重新评分，并据此选参数。

每个阶段内候选共享随机数。各试次的预测概率先跨重复平均，再计算 NLL。
一整轮没有达到平均 NLL 改善0.0001时停止；每个起点最多三轮。
按全部候选都计算的保守上限，每人两个搜索阶段加复评分最多980次参数点评价；缓存可减少实际次数。
三个搜索与复评分阶段均设置 `engine_config.oral_readout: null`，省去每个候选的口述分布计算。
固定参数仿真基配置保留共同口述设置，选定参数后可生成口述预测。选择概率和NLL使用同一行为模型。
达到轮数上限的状态保留在 `restart_summary.json`，需结合 `stopped_by` 判断。
搜索上限、参数范围和粒子滤波（PF）预算尚需由完整拟合试点检查触边、局部最优，
以及 Monte Carlo 采样造成的评分波动。

并行预算为128，实际进程数受可用 CPU 和待计算任务数限制。每个进程使用一个数值线程，
候选与重复复用现有单层并行执行。运行前为 YAML 的 `output_dir` 设置新目录。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m src.Bayesian_state.optimization.cli --backend cd \
  --config configs/meg/hyper_cd_cfg/model0923_task3b.yaml
```

配置、输入和 stage 不变时，可添加 `--resume` 继续同一次搜索。已有产物会阻止重新启动并覆盖。

## 单被试试跑

`model0923_task3b_smoke.yaml` 选定341号被试，检查前32条有效记录。
原第8题为漏答，能在短序列内验证跳过后的试次对齐。
试跑使用完整规则目录和100,000个参考刺激，将每项参数的候选缩为示例值及相邻值，
coarse/fine各一个起点、一轮、4粒子、一次重复和一个进程；两个入围点以8粒子和两个独立重复复评分。
之后用入选参数检查短序列的固定种子复现，并以4粒子、一次重复评分该被试全部有效试次。
完整序列检查保持参数固定。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m src.Bayesian_state.workflows.runs.prepare_model_0923_meg \
  --freeze-source \
  --output-dir results/model_0923/meg/preparation_NEW
```

输出包含16人输入检查、实际配置快照、代码和输入内容哈希、包版本、搜索与复评分记录、
原始试次对应、完整序列预测、运行耗时和 `validation_report.json`。
验收独立重算 NLL，核对评分掩码、类别及规则/配对概率的归一化，并核对固定种子复现。
`status: passed` 表示数据与小规模运行流程通过；参数恢复、搜索范围及高预算 PF 精度仍需后续评价。
该入口只执行上述单被试试跑。

`--freeze-source` 将模型源码、配置和两份 MEG 处理后数据保存为只读快照，随后在独立进程中运行。
快照位于输出目录的 `source/`，验收产物位于 `run/`；`snapshot_manifest.json` 保存文件哈希，
`live_source_comparison.json` 记录运行期间原仓库是否继续更新。
该模式允许其他工作继续修改原仓库。验收结论对应快照版本；后续速度优化版本应复核同一流程。

2026年10月12日的[验收报告](../../results/model_0923/meg/preparation_20261012_frozen_v1/run/validation_report.json)
为 `status: passed`。16人共14,646个有效试次通过输入检查。
341号被试的32个有效试次完成28次参数点评价和2次独立复评分；
同种子短序列的预测逐项一致，完整序列860个有效试次中的859个参与评分。
类别概率、规则权重及配对信念均通过有限值、非负与归一化检查，原试次标识保持一致。
搜索约22.1秒，4粒子、一次重复的完整序列评分约8.1秒，总验收约45.5秒；
这些耗时对应本次小规模预算。相同快照的85项定向测试通过，两个 CLI 的 `--help` 检查通过。
此前因原仓库源码更新而失效的三次运行保留在 `preparation_20261012_v{1,2,3}`。
本次验收采用只读快照，全体被试的完整参数搜索尚未运行。

## 数据预处理

`notebooks/preprocess.ipynb` 的 MEG 部分生成两份数据：

- `data/meg/processed/Task1b_errorsummary.csv`：读取 Task1b 原始文件，按 `feature` 和 `level` 平均 `threshold_mean`，生成 `threshold_mean_mean`。特征编号1–4依次对应 green、yellow、pink、blue；被试编号取自文件名。
- `data/meg/processed/Task3b_processed.csv`：自动发现 `Task3b_<被试编号>_bhv.csv`，根据各被试的 `seqID` 连接对应的 `Seq1.csv` 或 `Seq2.csv`，检查刺激覆盖、类别对应和试次数。

同一被试的分段行为 CSV 按 session 和原始 trial 顺序合并。`iSession`、`iRun`、`iTrial` 和 `stiID` 保留原编号，分段中独有的 `success` 列也保留。Task1b 的特征列统一命名为 `feature`。

处理后的行为表保留 `presskey`、`rating`、`miss` 及原始事件持续时间。漏答试次保留在原位置，缺失的选择、评分和反馈保持为空。模型读取时跳过 `miss=1` 的整条记录，下一条有效试次继续已有学习状态；结果记录原始试次标识，便于与评分及 MEG 时序对齐。rating 仅用于独立检验模型预测。

2026年10月12日更新覆盖333–348共16名被试。新版 `Seq2.csv` 含1440个刺激，覆盖第4个 session。复现处理时，执行 notebook 的导入单元、MEG 知觉汇总单元以及 MEG 行为预处理定义和调用单元。
