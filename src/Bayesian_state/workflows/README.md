# 当前模型工作流

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
| `analysis/validate_model_0826_joint_square.py` | 独立复核已预选的 S129 四点联合移动例子，保留两个单块对照 |
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
