# 当前模型工作流

新观察数据的 Model 0826 PMH 拟合默认走
`python -m src.Bayesian_state.run_model_0826_fit`，见 [配置与用法](../optimization/ADAPTIVE_FIT.md)。
下列 dated pilot 仍是试验记录与复核入口，不是正式默认入口。它们共用的提案和数值诊断
已迁入优化层，原导入接口保持兼容；历史结果没有改写，但旧源码指纹的续跑需检出旧提交。

本目录按当前 0826 工作需要精简。保留清单：

| 位置 | 用途与保留原因 |
| --- | --- |
| `recovery/design.py` | 读取、验证已注册的恢复实验设计 |
| `recovery/generation.py` | 配置真值模型，调用 simulation 自主生成并保存实验数据 |
| `recovery/run.py` | 恢复阶段调度、恢复运行及来源指纹 |
| `runs/run_model_0826_recovery.py` | 原工作流模块路径的兼容入口 |
| `runs/run_model_0826_belief_transport_counterfactual.py` | 当前 0826 信念迁移方法比较 |
| `runs/run_model_0818_boundary_recovery.py` | 0826 比较仍直接使用其数据读取、引擎设置和种子工具；保留的历史依赖 |
| `analysis/behavior_diagnostics.py` | 通用行为诊断 |
| `analysis/audit_model_0826_finalists.py` | S129 原始四个 finalist 的定点复算和配对 PF 重采样；不搜索新参数 |
| `analysis/audit_model_0826_system.py` | 小规模机制、输入契约与 exp4/exp5 适配检查；隔离进程内验证加速原型 |
| `analysis/probe_model_0826_search.py` | S129 分散起点与跨块联合移动的有限补充搜索，复用共享评分器 |
| `analysis/pilot_model_0826_simplified_fit.py` | 简化拟合的隔离试验：单阶段低预算搜索、独立筛选和确认，与历史入围参数在当前代码下比较 |
| `analysis/pilot_model_0826_adaptive_effort.py` | 后续试点：固定候选上限的分散搜索与局部/联合调整、固定参数的粒子/种子校准及按被试追加计算的诊断 |
| `analysis/pilot_model_0826_search_stopping.py` | 搜索与停止检查：逐参数方向提案、停止后的额外挑战、S129 历史起点续搜、S229 评分波动定位，以及 condition 3 完整序列试点 |
| `analysis/validate_model_0826_joint_square.py` | 独立复核已预选的 S129 四点联合移动例子，保留两个单块对照 |
| `analysis/diagnose_model_0826_numerics.py` | 六人验收后的缓存波动诊断与预先固定的数值/边界探测；不重新搜索或修改默认配置 |
| `benchmarks/benchmark_boundary_geometry.py` | 几何计算性能检查 |

从仓库根目录运行：

```bash
python -m src.Bayesian_state.run_recovery --help
python -m src.Bayesian_state.workflows.runs.run_model_0826_belief_transport_counterfactual --help
python -m src.Bayesian_state.workflows.analysis.behavior_diagnostics --help
```

恢复拟合实现位于 `optimization/recovery.py`；冻结候选评分、预算评价和恢复统计/绘图位于
`evaluation/recovery.py`。工作流只组织这些步骤，不维护另一套 PF 或认知机制。

76 个已无当前依赖的历史 Python 工作流和 3 个历史 shell 批处理入口已删除。
其对应旧命令不再可用，历史结果、配置和文稿保留；需要复现已删除工具时检出原 Git 版本。
删除路径及哈希见 [精简记录](../docs/maintenance/recovery_cleanup_20260908.json)。

模型文稿与配套编译脚本已移到 [docs/model_architecture/](../docs/model_architecture/)。

S129 参数估计审计使用原 final-rescore 种子、128 粒子和 16 次重复，保留逐试次
观测选择概率。此专用脚本核对 256 个试次、255 个计分试次和四个候选；不适用于
直接复算其他被试。输出目录必须不存在。配对 bootstrap 只评价这些 PF 重复的
数值波动，不提供参数置信区间，也不检验全局最优。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m src.Bayesian_state.workflows.analysis.audit_model_0826_finalists \
  --hyper-config results/model_0826/cond1/subject_129/pipeline_20260908_v1/configs/PMH_hyper.yaml \
  --finalists results/model_0826/cond1/subject_129/pipeline_20260908_v1/models/PMH/optimization/subject_129/final_rescore.jsonl \
  --output results/model_0826/cond1/subject_129/estimation_audit_new \
  --jobs 16
```

该命令包含 64 次完整 PF 运行，需要按任务授权与其他正在运行的拟合协调并行预算。

## 系统审查的轻量复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/model0826_audit_mpl NUMBA_CACHE_DIR=/tmp/model0826_audit_numba \
python -m src.Bayesian_state.workflows.analysis.audit_model_0826_system \
  --output-dir results/model_0826/system_audit_new
```

输出目录必须不存在。检查使用 S129 的前 32 试次与 8–16 粒子，另对 exp4/exp5 各做
16 试次、4 粒子的一次集成检查；不搜索参数。`evidence.json` 保存环境、输入与源码哈希、
问题复现及计时，`baseline_arrays.npz` 保存基线输出。零精度快速路径和几何缓存仅通过
进程内临时替换进行对比，不修改正式模型；逐数组相等检查覆盖公开预测、状态、控制量和
部分 PF 诊断，不等于所有配置均已验证。exp4/exp5 的通过仅说明短序列接口可运行。

## S129 跨块搜索验证

`probe_model_0826_search` 在原细搜取值范围内，以 C4、χ=1、低记忆、低容量、
低初始精度的代表点为起点。每个两块联合改变都保留两个单块对照；另检查八个固定种子
分散网格点，再对新点中的前两名各做两组联合移动。它是有限的诊断实验，不能认证
全局最优，也不是完整的多起点坐标下降重拟合。

先精确复核这些锚点的旧细搜评分，才复用已有 fine 缓存；新点采用同一 64 粒子 ×
8 重复预算和种子。最后固定最多四个候选（始终含 C4），用独立共同种子、128 粒子 ×
16 重复复评。数据、计分掩码、模型范围不变。输出记录提案来源、每个单块/联合对照、
实际评分与逐试次概率；配对 bootstrap 只度量 PF 数值波动。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/fig2_mpl NUMBA_CACHE_DIR=/tmp/model0826_probe_numba \
python -m src.Bayesian_state.workflows.analysis.probe_model_0826_search \
  --pipeline results/model_0826/cond1/subject_129/pipeline_20260908_v1 \
  --output results/model_0826/cond1/subject_129/joint_search_probe_new --jobs 16
```

`--prepare-only` 只生成提案；随后在同一目录加 `--resume` 执行。正常运行也可用
`--resume` 恢复，逐批完成的评分保留。断点检查配置、被引用数据、原搜索记录与源码
内容哈希；上下文变化时拒绝恢复。只能恢复本工具自己的新目录，不能指向原拟合目录。

`validate_model_0826_joint_square` 读取补充验证目录中的 `trapped_start4_square.json`，
在主 probe 完成后，用相同独立种子族评价四个固定点。若某点已经在主验证中按完全相同
设置评价，则复用其概率文件；其余点调用共享 PF。输出到新建的
`coordinate_trap_validation/`，用于区别“细搜数值目标中的局部停滞”和“更高精度下
仍成立的联合改善”。入口参数见 `python -m ...validate_model_0826_joint_square --help`。

## 简化拟合试点

`pilot_model_0826_simplified_fit` 是试验入口，尚未替换正式拟合。默认配置位于
`configs/exp123/specific_models/model_0826_simplified_fit_pilot.yaml`：S129/S229 全序列、
32 粒子 × 4 seeds、既有 coarse 网格、两个原始起点、最多三轮、patience=1。
使用共享 Hyper-CD，不运行 dense fine 阶段。候选保留全局前四名及每种已评估 M/χ 的最佳解。
粒子数、停止规则与网格密度变化可能改变估计结果，不属于逐元素等价加速。
两个原始起点只在工作空间设置上不同；首个工作空间块遍历后可能合并为同一条搜索路径。
因此应检查 restart 的新增评价数，不能把配置中的两个起点视为两次独立探索。

历史完整拟合的参数仅在新搜索完成后读入，既不作起点，也不作搜索候选。
入围参数用当前共享引擎在独立共同 seeds 上以 64×8 比较；每方前两名和低预算原始赢家
再以另一独立 seed family 的 128×16 确认。主要比较对象在 64×8 阶段预先选定。
本轮固定做高精度确认，是为了评价未来是否能省去统一高精度计算，不是新增一轮密集搜索。

输出包括各阶段配置、搜索记录、入围解逐 seed 预测/状态数组、实际计时、输入/源码指纹及
比较报告。比较使用首试次递推但不评分的原契约，先平均概率再计算 NLL。状态为 pre-choice
过滤边际，不宣称恢复了真实心理路径。配对 seed bootstrap 仅表示固定候选的数值不确定性；
YAML 中的差异阈值是试点筛查标准，不是通用科学等价界限。参数相同但 seeds 分半产生的
状态差异也保存，供区分数值波动和参数解释变化。输出不含新的被试泛化检验。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/tmp/model0826_pilot_numba MPLCONFIGDIR=/tmp/model0826_pilot_mpl \
  python -m src.Bayesian_state.workflows.analysis.pilot_model_0826_simplified_fit \
  --config configs/exp123/specific_models/model_0826_simplified_fit_pilot.yaml \
  --output-dir results/model_0826/simplified_fit_pilot_new
```

先添加 `--smoke` 在另一个新目录运行：仅 S129、32 trials、2 粒子 × 2 seeds 和单进程，
检查参数确实作用于引擎及全流程输出；不能用 smoke 的筛查标志判定方法有效。
默认入口则使用 128 的完整进程预算，每 worker 单线程。`--resume` 只接受完全相同的
输入/代码/环境；搜索可恢复，完整筛选/确认批次可复用。筛选/确认的半成品批次不会覆盖，
若该批中断应保留旧目录并在新的输出目录重做。

## 按被试分配搜索与数值预算的试点

`pilot_model_0826_adaptive_effort` 延续简化拟合诊断，配置为
`configs/exp123/specific_models/model_0826_adaptive_effort_pilot.yaml`，不修改正式配置或模型核心。

- S129 从现有参数空间生成 36 个初始点：所有工作空间格的标准起点，以及各参数块分层抽样的
  分散起点。随后最多两轮、每轮四个不同精英点、每点最多 24 个局部/整块/双块候选，总上限 228。
  候选来自现有 fine 支持集，没有穷举 dense fine。搜索始终使用同一组 32×4 seeds。
  历史参数只在搜索结束后加入独立 64×8 筛选与 128×16 确认；本轮是新提案策略的探索试验，
  同时改变了候选分布和搜索种子，不能当作只改变一个因素的优化器比较。
- S229 固定上一轮独立筛选的两名简化候选和两名历史参照，比较 R=32/64/128 与 B=4/8/16/32。
  每个 R 只算到 B=32，再分析相同 seeds 的前缀。核对旧配置、输入、源码和候选后，R=128
  只补齐 16 个新 seeds，旧前缀只读复用。完整 32-seed 结果包含已见过的前缀，不能称为全新独立验证。
- 阈值控制平均 NLL 数值区间、bootstrap regret、逐试次预测与状态差异。Regret 允许近似等价的
  候选换位；不要求每次随机运行都有唯一赢家。分半指标使用 B/2 对 B/2，不等于两个独立 B-seed
  重复；有限候选、有限粒子下的 bootstrap 也不是全局最优或积分收敛证明。
  追加参数搜索使用评分/排序精度检查；最终输出另查预测和状态精度。已明确的评分差距不会
  仅因输出精度稍低而阻止继续搜索，停止交付前仍须检查目标输出精度。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/tmp/model0826_adaptive_numba MPLCONFIGDIR=/tmp/model0826_adaptive_mpl \
  python -m src.Bayesian_state.workflows.analysis.pilot_model_0826_adaptive_effort \
  --config configs/exp123/specific_models/model_0826_adaptive_effort_pilot.yaml \
  --output-dir results/model_0826/adaptive_effort_pilot_new
```

先以 `--smoke` 在另一个新目录检查两个被试的 32-trial 单进程流程；默认运行完整序列、128 进程
预算。`--resume` 检查完整输入/源码/环境并复用完成批次；半成品批次不覆盖。输出包括提案来源、
逐轮收益、逐 seed 数组、各预算诊断及追加计算建议。

按被试调整的是计算量，所有被试仍使用相同模型、目标、试次契约和预先确定的精度要求。
优先区分评分噪声与搜索覆盖不足：前者校准 R/B，后者增加真正不同的起点或有限局部搜索。
低 NLL 本身不能证明搜索充分，高 NLL 也不自动意味着需要更多搜索。状态分析还需单独达到状态
稳定性要求。达到预设计算上限但未过检查时，应标记“尚未稳定”，不能把耗尽预算写成“已收敛”。
当前建议器没有自动选择正式的被试级预算，推广到全体被试与模型比较仍需独立验证。

### 搜索覆盖、评分波动与停止检查（2026-09-18）

`pilot_model_0826_search_stopping` 使用独立配置
`configs/exp123/specific_models/model_0826_search_stopping_pilot.yaml`，继续复用共享引擎和已有评分、
配对种子比较函数。没有修改认知方程、正式拟合默认值、评分掩码或预测时点。

- S129、S103、S203、S301 使用完整序列。后三人按各 condition 试次数距中位数最近、
  同距离取较小编号选定，排除前期试点被试，不按拟合好坏选择。
- 45 个起点加最多四轮、每轮两名不同精英候选、每名最多 48 个提案。提案按原始参数方向轮流
  安排远近调整，联合块中的派生量保持约束，不穷举稠密联合网格。
- 低预算搜索为 R32×B4；两轮改善不超过 0.0001 时冻结拟停止候选，但本试验继续至上限，
  并增加最多 60 个局部方向和 36 个全局起点。所有停后候选都属于挑战组；去重可减少实际数量。
  未出现平台时以预算上限处为参照，不能声称已收敛。
- 基础组和挑战组各保留四名候选，经独立 R64×B8 筛选，固定各组主候选并以 R128×B16 确认。
  NLL 差定义为基础组减挑战组，正值表示挑战组更好。只有评分精度通过、有平台、且改善区间
  上界不超过容差，才给出有限范围内的暂时停止建议。该小试点不证明全局最优。
- S129 在通用搜索和挑战结束后，另读取历史候选，从两名历史筛选优胜者附近续搜；
  `reference`、`warm`、`base` 分开记录，不能把历史起点的效果归入通用搜索覆盖。
- S229 对上一轮四名固定候选定位种子波动及 64-trial 时间段贡献；以同一旧种子前缀重跑
  R128×B16，增加 ESS/重采样记录，并要求所有概率与状态数组与归档逐元素完全一致。
  两名历史候选另用 R256×B16 检查粒子敏感性。它是旧种子配对诊断，不是新种子独立确认。
  时间段方差使用保留 trial 间协方差的种子影响量分解；不能将 trial 当独立观测来降低误差。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/model0826_stopping_numba \
  MPLCONFIGDIR=/tmp/model0826_stopping_mpl \
  python -m src.Bayesian_state.workflows.analysis.pilot_model_0826_search_stopping \
  --config configs/exp123/specific_models/model_0826_search_stopping_pilot.yaml \
  --output-dir results/model_0826/search_stopping_pilot_new
```

先另选新目录加 `--smoke` 检查 32-trial、小粒子、单进程的普通与 condition 3 分支。
`--resume` 只复用相同输入/源码/环境的完整批次，未完成的数值产物保留并拒绝覆盖。
`context.json` 记录依赖内容和实际库版本；`workflow_source_at_run.py.txt` 保留执行源文件。
输出分为各被试的逐轮、挑战、筛选、确认目录和 `noise_229` 诊断目录。

通俗说明与科学解释边界见
[`Model 0826 Plus`](../docs/model_architecture/model_0826_plus.tex)。本轮只执行小规模诊断，
全体正式拟合、恢复实验及 exp5 不在本入口的执行范围。

本轮完整结果见 [搜索与停止检查报告](../../../results/model_0826/search_stopping_pilot_20260918/README.md)。
S129 的追加挑战已找到接近历史参照得分的候选；S103 的评分检查通过但未到搜索平台，
S129/S203/S301 的整体评分精度检查未通过。四名搜索被试均未触发两轮平台，因此本轮
不能验证实际早停的误停率；当前入口仍是诊断试点。

### 围绕选参决策的精度与平台挑战（2026-09-18）

`pilot_model_0826_decision_precision` 使用
`configs/exp123/specific_models/model_0826_decision_precision_pilot.yaml`，回答两个有边界的问题：

- 固定 S103/S129/S203/S229/S301 先前入围的候选，并在新计算之前以旧分数选定主候选。
  新规则检查主候选相对整库最好的平均 NLL 损失是否不超过 0.005；在每次完整 seed bootstrap
  重采样内取最大损失，不要求两个明显差的候选之间也有精确的排名。计算使用概率均值再取 log，
  不把 trial 当独立重复。依次尝试 64×8、128×16、256×32，明确通过或明确选错后停止升级。
  三次预定检查各使用 alpha/3 的单侧分位数；这是有限 seeds 下的近似数值诊断，不是严格覆盖保证。
  所有原始候选均用另一个独立 seed family 的 256×32 复验，复验不能反过来更换已选参数。
  旧的最宽配对区间门槛仅作并列比较，不重写旧报告。跨预算与独立种子间的差异单独报告，
  不能把同一个有限粒子预算上的种子稳定称为粒子积分已经收敛。
- S103 复用经全部输入、旧源码和数值库版本核验的历史搜索记录；全部行为序列继续使用。
  新候选仍以 32×4 发现，每轮取前 8 名并补最多 4 名不同候选，用固定另一组 128×16 种子
  引导下一轮。低预算与引导分数不混排。最多追加六轮；每轮最多两名精英各 48 个方向提案，
  再补 18 个通用起点中尚未见过的点。连续两轮引导分数改善不超过 0.0001 时冻结拟停止点。
  再用两轮局部/整块/联合参数提案及最多 90 个新起点挑战它，最后独立 256×32 检查主候选
  对基础与挑战入围库的损失上界。没有平台、预算耗尽或区间不明确都不会写成收敛。

这里改变的是试验性的搜索和数值决策规则，认知机制和生产默认配置保持不变。搜索引导种子
会被自适应反复使用，因此引导分数只用于发现候选，最终判断依靠新种子；低预算初筛仍可能
遗漏好参数。单个被试通过挑战也不能估计全体被试误停率。输出只保留选择概率及评分元数据，
不作为状态/潜在心理轨迹交付；各 worker 仍调用同一个已验证的共享评分入口。

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/model0826_precision_numba \
  MPLCONFIGDIR=/tmp/model0826_precision_mpl \
  python -m src.Bayesian_state.workflows.analysis.pilot_model_0826_decision_precision \
  --config configs/exp123/specific_models/model_0826_decision_precision_pilot.yaml \
  --output-dir results/model_0826/decision_precision_pilot_new
```

先以另一个新目录加 `--smoke` 执行 32-trial、2 粒子×2 seeds、单进程的 condition 1/3 检查。
完整试点按被试×候选×seed 在同一个进程层并行，最多使用当前可用的 128 核，不人为保留核。
`--resume` 核对完整输入、源码、库版本和批次计划，仅复用完整批次；未完成批次不覆盖。
生成报告与数值数组保存在新结果目录，旧结果保持只读。
此入口针对预置的 exp123 归档、数据、引擎和参数支持；复用旧候选/搜索记录要求这些对象保持一致，
不能仅修改数据路径就将其当作新实验的拟合入口。

本轮完整结果见 [选参精度与平台挑战报告](../../../results/model_0826/decision_precision_pilot_20260918/README.md)。
在固定候选库内，S103/S203 的 128×16 判断和 S129/S229 的 256×32 判断得到独立种子复验支持；
S301 到 256×32 仍未明确，复验揭晓前补充提名的替代点也未通过。S229 的选参虽通过，
两组独立计算的逐试次概率 RMSE 约 0.0265，仍超出之前预测输出的 0.02 试点标准。
S103 追加三轮后出现两轮平台，经 237 个新挑战候选和独立审查，拟停止点对最终入围库的
平均 NLL 损失上界为 0.00464，小于探索性容差 0.005；余量较小，只支持该案例的有限停止。
追加搜索相对初始提名的改善未获独立确认，不能把引导分数下降当成必然的拟合收益。
这些是计算分配规则的个案证据，尚未把生产默认配置切换为该试点，也不验证最终状态精度。

## 六人验收后的定向复核

`diagnose_model_0826_numerics` 的冻结配置为
`configs/exp123/specific_models/model_0826_numerical_boundary_followup.yaml`。
先读取上一轮带校验和的概率缓存，区分更换种子、粒子预算和参数候选造成的差异；逐试次诊断不删除或重新加权试次。
S307 固定三个历史候选，比较 R128/256/512 各 B32；S221/S314 固定各两点，以 R256×B64 检查随机重复。
S102/S206 用 R256×B32 做 E_C、gamma、M 的少量外层探测。E_C 降低时，分别检查保持 delta_E 和保持 E_E 两种路径，初始事件概率仍按原约束与 E_C 绑定。
S206 的 M 探测从近优 M5 备选出发，其原代表是 M3；不把备选的触边误写为代表触边。

总计上限 896 次 PF，候选、提名和预算在计算前冻结，五个主问题各用 alpha=.01、平均 NLL 容差 .005。
每例全部预算无条件执行，只有最后一档作主判断；前缀和较低档只诊断，不选择最有利的一档报告。
这只能复核有限点，不能替代整库验收、共同范围的全面校准、参数恢复或状态精度检查。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/model0826_followup_numba \
  python -m src.Bayesian_state.workflows.analysis.diagnose_model_0826_numerics \
  --output-dir results/model_0826/numerical_boundary_followup_new
```

先用 `--dry-run` 核对范围，在另一个新目录用 `--smoke` 跑 S307 的32试次、R2×B2、单进程。
`--cache-only` 在新目录只生成缓存诊断；不能把该目录当追加计算目录。
`--resume` 只复用相同协议、源码、输入和环境，半完成批次保留逐 seed 原子缓存；不改写既有产物。
完整定向复核需任务授权，默认使用128进程预算、单线程数值库，实际进程数受就绪任务数约束。

2026-09-20的[完整定向复核](../../../results/model_0826/numerical_boundary_diagnosis_20260920/README.md)
共896次PF、59.6分钟。S221的两点检查通过，S314在R256×B64下边缘通过；S307到R512×B32仍不确定。
S102/S206的固定外层点没有显示超容差收益。这些结果不替代整库验收，不改变默认预算或原报告状态。

### S307：固定参数后增加随机重复

`configs/exp123/specific_models/model_0826_s307_repeat_followup.yaml` 复用上述入口，
固定相同三个历史点及提名 `800a7612ad099c8d`，只运行 R256×B64 的192次完整PF。
这次增加的是新随机种子的重复次数，不搜索新参数。主判断仍为平均NLL损失上界≤.005、alpha=.01；
bootstrap预先设为60000以减小分位数计算的抽样波动，正式拟合默认值不变。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/model0826_followup_numba \
  python -m src.Bayesian_state.workflows.analysis.diagnose_model_0826_numerics \
  --config configs/exp123/specific_models/model_0826_s307_repeat_followup.yaml \
  --output-dir results/model_0826/s307_repeat_followup_new
```

前16/32组种子、两个32组半样本和历史R256/R512×B32只作诊断；只有预定的完整R256×B64决定本次结论。
历史R512×B32与新R256×B64的粒子×重复数相同，但使用不同种子，不能当作严格配对的效率实验。
本次协议不继续追加PF，也不以三个固定点通过代替整个拟合流程验收。
执行记录、与两轮历史种子的去重检查及对照缓存指纹见
[S307重复次数检查](../../../results/model_0826/s307_repeat_followup_20260920/README.md)。

该检查已完成：192次完整PF耗时10.20分钟，损失上界.006001仍高于.005，结论保留为不确定。
新种子前16/32/64组的描述性上界依次为.012141/.009465/.006001；增加重复有助于缩小波动，
但本例仍未达到门槛，不据此统一增加默认预算。6项测试、完整序列校验和完成后的resume均通过。
本轮没有重搜，S307的原整库审查及M=5触边未在这三点检查中解决。

六人原本用于覆盖三个condition的完整流程和成本试点，已各完成一次完整拟合；随着诊断结果用于调整规则，
他们属于开发/校准样本。后续检查只针对有具体疑点的被试与环节，不必重跑全部六人。
拟合规则冻结后，应另选少量未参与调整的被试检查；保留未解决状态也是有效输出，不应无限算到全部通过。

### 冻结规则后的三人完整流程核查

S122/S222/S315各代表一个condition，按长度接近各组中位数选择，并排除已参与近期校准的被试。
直接调用正式 `run_model_0826_fit`，配置为 `model_0826_frozen_validation.yaml`；与默认v2仅analysis_id不同。
每人最多9252次完整PF；不因本轮结果追加预算、替换被试、扩范围或修改门槛。
详细规则与命令见[冻结核查协议](../optimization/FROZEN_VALIDATION.md)，
执行记录及自动收尾产物位于[本轮目录](../../../results/model_0826/frozen_validation_20260920/README.md)。
