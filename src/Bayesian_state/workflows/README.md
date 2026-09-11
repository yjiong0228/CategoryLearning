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
| `analysis/probe_model_0826_search.py` | S129 分散起点与跨块联合移动的有限补充搜索，复用共享评分器 |
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
