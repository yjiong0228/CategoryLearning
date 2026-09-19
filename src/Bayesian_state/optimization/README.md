# 优化

Model 0826 观察数据 PMH 拟合现在默认使用
`python -m src.Bayesian_state.run_model_0826_fit`，统一 CLI 也会按新配置的 backend 分派。
新流程包括分散/联合搜索、分级选参、独立审查、边界标记和续跑；旧 Grid/CD 与恢复配置
继续使用原方法。软件接入不等于范围、恢复或全体拟合已经验证。
具体命令、默认预算、边界扩展和产物见 [自适应拟合说明](ADAPTIVE_FIT.md)。

`recovery.py` 拥有 fit_recovery_dataset 和冻结搜索预算解析；`recovery_parameters.py` 提供真值参数映射。阶段编排位于 workflows/recovery/run.py，评价层不再承担拟合实现。本文历史专用脚本可能已移除，以 workflows/README.md 保留清单为准。

本目录是 `Bayesian_state` 的参数搜索与模型选择层。它调用 `simulation/`，把候选
hyperparameters 对应的 `SingleRunResult`/`SimulationResult` 组成可比较的 objective，但不拥有
固定参数运行时、结果契约、指标公式或具体认知机制。
凡是会根据数据表现改变下一轮参数、候选权重或冻结配置的操作属于本目录；冻结模型的 PPC、
留出验证和外部通道解释属于 `evaluation/`。

## 文件说明

| 文件 | 职责 |
|---|---|
| `candidates.py` | condition-1 单机制候选空间与 engine-config 参数注入 |
| `parameter_space.py` | 冻结模型参数空间的边界、零点和派生参数校验 |
| `objectives.py` | 有容差和 anchor guard 的有序多目标比较 |
| `artifacts.py` | 候选展开、结果 schema、provenance、compact/full artifact 构建 |
| `cli.py` | grid/CD 统一 CLI |
| `search/common.py` | Grid/CD 共用运行时、配置解析、候选注入和 JSONL I/O |
| `search/grid.py` | 显式 joint grid 搜索 |
| `search/coordinate_descent.py` | coarse/fine coordinate descent、多 restart 与 trace |
| `diagnostics/search.py` | 只读取既有搜索产物的收敛、plateau 和 selection diagnostics |
| `diagnostics/predictive.py` | 需要重新运行候选模型的 accuracy sampling 与 volatility diagnostics |

## 标准数据对象

这些对象的唯一实现分别位于 `simulation/data.py` 与 `simulation/results.py`。`TrialArrays`：

```python
TrialArrays(
    stimulus=...,
    choices=...,
    feedback=...,
    categories=...,     # optional diagnostic truth
    target_probs=...,   # optional probabilistic target
)
```

`SingleRunResult` 保存一条 trajectory 或一次 particle marginal run 的：

- `metrics_by_mode`
- `mean_error`
- seeds 与 params
- 可选的 state/trial/transition 日志

`SimulationResult` 聚合多个 `SingleRunResult`，并保存 sample errors、representative run 和
`compute_simulation_statistics()` 的结构化结果。

## 预测模式

- `posterior_t_minus_1`：以上一 trial feedback 后 posterior 预测当前 trial。
- `prior_t`：使用当前 trial transition 后、choice 前 prior。
- `both`：同时计算两者；必须指定用于模型选择的 mode。

动态 transition 和粒子滤波的因果预测应使用 `prior_t`。

## 损失函数

公共 loss 包括：

- 准确率曲线 MAE/MSE/BerHu
- accuracy/family Brier 与 NLL
- 选择 Brier/NLL
- wrong-choice 与 conditional-wrong-choice NLL
- 目标概率 Brier

loss strategy 与全部 loss 数值定义位于 `metrics/losses.py`，并只读取标准 metrics mapping。
Brier、NLL、CRPS、曲线和行为统计等其他纯数值定义也位于 `metrics/`；本目录只负责构造模型
预测、调用共享指标，以及规定如何把这些数值组成 selection objective、容差和 anchor guard。
增加新后端时，应把后端输出转换成公共 prediction/metrics contract，而不是在 optimizer 中另写
一套评分。

## 随机种子层级

种子由 `utils/seeding.py` 通过稳定 hash 分层派生：

```text
hyper_base_seed
  → hyper_candidate_seed
      → simulation_point_seed
          → trajectory/filter seed
              → module-specific seed
```

同一配置、被试、参数点和 repeat index 应产生相同 seed；不要依赖 Python 进程内置的随机 hash。

## Hyper-CD 与 Grid

Hyperparameter key 必须以以下前缀开头：

```text
engine.modules.memory_mod.kwargs.gamma
engine.choice_readout.kwargs
simulation.window_size
```

mapping-valued coordinate 会整体替换目标 mapping，适合把一个策略 profile 或 controller 当成
不可拆分候选。不要同时声明父路径和其子路径；`validate_no_nested_hyperparam_paths()` 会拒绝
这种歧义。

当一次搜索需要多个彼此独立的 mapping-valued coordinate 时，可使用命名打包键
`__profile_candidate__:<name>`。展开器会把所有打包坐标还原为 engine 路径，并拒绝打包坐标之间
或打包坐标与普通坐标之间的重复路径。Hyper-CD 还支持 `cd.initial_points` 指定与 restart 一一
对应的完整起点；设置 `common_random_numbers_within_candidate_comparisons: true` 后，同一 stage
和被试的候选共享 PF 随机数，只消除候选比较噪声，不改变最终独立重评分的种子族。

Hyper-CD 的 `objective_order` 按顺序比较目标；只有前一目标落入容差集合时，后一目标才用于
区分。coarse/fine stage 可分别覆盖 simulation repeats、particle count 或日志预算。

Grid/CD 都继承 `HyperSearchBase`。外部 workflow 运行单被试时调用
`optimizer.run_subject(subject_id, stage=...)`，批量运行调用 `optimizer.run(subjects, stage=...)`；
`_run_subject_pipeline()` 是后端内部实现，不是 orchestration API。

Hyper-CD 默认 `hyperparam_selection_mode: per_subject`，即每位被试各选一套参数。若某次分析的
目的只是为新数值尺度校准一套共享参数，可显式设置
`hyperparam_selection_mode: shared`。此时每个候选在全部指定被试上运行，先分别计算被试级
objective，再对被试等权平均后选择同一候选；不会因不同被试的 trial 数不同而让长序列被试
自动获得更高权重。共享模式的 `best_hyperparams.json` 用 `subject_id: -1` 表示群组结果，并在
`search_context.subjects` 保存实际开发集。

Model 0818 的恢复前候选支持保存在
`configs/exp123/specific_models/model_0818_cond1_parameter_space.yaml`。其中 `delta_E`、`c_A` 和
`c_G` 使用 `spike_and_positive_grid`：精确零边界写在 `zero_value`，严格正值写在
`positive_values`，不得把两者合并成一个普通连续区间。`delta_E` 在 logit 尺度上导出
`E_E = sigmoid(logit(E_C) + delta_E)`，从结构上保证 `E_E >= E_C`。该配置在参数和模型恢复
通过前保持 provisional，不得直接用于冻结真实被试估计。

三条零边界的最小生成--恢复入口是
`src/Bayesian_state/workflows/runs/run_model_0818_boundary_recovery.py --smoke`。它只读取 condition-1 的刺激和正确类别作为
固定任务日程，自主生成 choice/feedback，再用共同 PF 随机数和“先跨种子平均逐试次概率、后计算
NLL”的规则盲评四个候选剖面。smoke 输出只用于验证管线和候选边界是否可达，不作为参数可恢复
的科学证据。

`--pilot` 增加任务日程、独立生成轨迹、序列长度和 PF 预算，并对预声明数据集执行多组 `R/B`
审计。数值审计使用 A/B 两个互不重叠的 PF 种子集合；每个集合内的 R32 与 R64 共享同四个逻辑
种子，从而分别估计配对粒子数敏感性和独立种子复现性。高预算校准同时要求候选 NLL 排序和赢家
一致性达到配置中预先声明的门槛，输出 `pf_budget_calibration.json` 与逐数据比较表。pilot 仍不是
正式恢复；只有该校准冻结了 provisional PF 预算后才能扩大合成恢复设计，且真实被试拟合仍需等待
正式恢复成功。

当候选接近、赢家会随 PF 种子改变时，使用
`src/Bayesian_state/workflows/runs/run_model_0818_seed_convergence.py` 运行逐种子数值诊断。每个
dataset × candidate × filter-seed 是一个独立并行任务，并单独保存逐试次选择概率、ESS 和重采样
记录；同一缓存可按配置中的 `nested_checkpoints` 无损汇总不同 B 前缀，不允许把已经聚合的 NLL
再作平均。汇总使用配对种子 bootstrap 估计候选间 delta-NLL 的 Monte Carlo 区间，并把近似并列
与评分器不稳定分开。默认配置检查 R32/B16；
`configs/exp123/specific_models/model_0818_high_budget_convergence.yaml` 是 R128/B128 的暴力扩容终止测试。
该步骤即使通过，也只允许进入稳定 B 下的配对粒子数比较；它本身不授权正式恢复或真实被试拟合。

在使用者明确授权跳过恢复、只查看当前模型拟合状况时，可运行
`src/Bayesian_state/workflows/runs/run_model_0818_exploratory_observed_fit.py`。当前正式配置
`model_0818_cond1_full_observed_fit.yaml` 使用 condition 1 的 32 名被试各自全部可用试次（共
10,048 个；每人 64--768 个）、完整 PMH 架构和固定的被试知觉参数。搜索阶段使用 R32/B4 的
两起点单轮块坐标筛选；入选点用独立种子族执行 R128/B128，并先平均逐试次概率再计算 NLL。
输出同时报告随机选择和只使用历史选择的因果偏置基线、种子间 NLL 波动以及逐试次概率 MCSE。
目录和 manifest 明确标记 `exploratory_only`；它仍是拟合内描述性结果，不能替代参数恢复、模块
恢复或 held-out 泛化，也不能直接进入论文的参数或模块结论。旧的
`exploratory_observed_fit_pre_recovery_v2` 只取每人前 64 个试次，已标记为无效审计记录。

上述真实数据拟合的模型评价不另设平行实现。全部被试搜索完成后，
`--phase evaluation-config` 会把每名被试选中的固定参数物化为公共 simulation YAML；评价用
R128、4 个不参与 Hyper-CD 选择的新 PF seeds 和 `keep_logs: true` 生成标准
`simulation/subjects`、`simulation/cache` 输入。随后运行
`src.Bayesian_state.run_model_evaluation --oral-mode center`，得到与
`results/model_dynamic_adaptive_control/0813_pf/model_evaluation` 同构的 basic、trajectory、
behavior PPC、sequential residual 和 oral/model alignment 产物。快捷编排入口是
`src/Bayesian_state/workflows/runs/run_model_0818_full_standard_evaluation.sh`。这里的标准评价覆盖每名被试的全部拟合内
试次；R128/B128 的最终重评分单独保留，不能把 4-seed 图形评价误称为最终 likelihood 精度。

若基础 simulation config 声明 `evaluation_protocol.mode: sequential_holdout`，Grid 和 Hyper-CD
统一以 `optimization` 角色解析评分掩码。主 loss、边际预测、accuracy shape、history kernel、
switch behavior 与 distribution objectives 都从带该掩码的公共 metrics mapping 计算；完整序列
只用于保持在线状态递推，不允许留出后缀进入候选比较。每个候选的结构化结果保存 `scoring`
上下文，便于审计实际切分。

## 与推理后端的边界

model structure 中设置：

```yaml
inference:
  backend: particle_filter
  particle_count: 512
  resample_threshold_fraction: 0.5
```

`evaluate_state_model_run()` 调用 `inference.dispatcher`，再将 trajectory 或 particle
输出转换成公共 metrics/loss。粒子实现位于
`inference/backends/particle_filter.py`；optimization 不维护另一份推理算法。

choice/output-noise 的实现也不再位于 optimizer；统一从
`model/readout.py` 调用。RT/oral readout 已有状态到测量分布的接口，但尚未加入
当前 choice-only backend 的观测输入和 loss。

当前限制：condition 1、expectation 类 readout、uniform base lapse；RT emission 尚未接入。

## 入口与输出

```bash
python -m src.Bayesian_state.optimization.cli --backend cd --config <yaml>
python -m src.Bayesian_state.optimization.cli --backend grid --config <yaml>
```

`search_schema_version: 2` 的 Hyper-CD 运行采用显式续跑策略。每完成一个坐标会原子更新
`search_checkpoint.json`；同一输出目录若已存在搜索产物，新运行会立即报错。确认配置、基础
simulation 配置、被试顺序和 stage 均未改变后，可用：

```bash
python -m src.Bayesian_state.optimization.cli --backend cd --config <yaml> --resume
```

续跑会从 `all_combinations.jsonl` 重建同一 stage 的候选缓存，并确定性回放搜索控制；已计算
参数点不会重复运行 PF。`--resume` 与旧的 `--resume-from-coarse` 含义不同：前者继续同一次
schema-v2 搜索，后者只用于从既有 coarse 结果启动单独的 fine stage。

Schema v2 可再配置独立最终复评分：

```yaml
final_rescore:
  enabled: true
  shortlist_size: 4
  seed_family: model0826_recovery_final_rescore_v1
  simulation_overrides:
    simulation_repeats: 32
    repeat_aggregation: mean_probability
```

复评分从最终 search stage 的唯一候选中按目标顺序取 shortlist，使用独立 seed family 和
候选间共同随机数重新运行 PF。它必须先平均每个试次的 choice probability，再计算 NLL；最终
参数只能由该复评分决定。`best_hyperparams.json` 同时保留低预算 `search_best` 与实际采用的
`final_rescore_best`，完整复评分记录写入 `final_rescore.jsonl`。若基础配置使用 sequential
holdout，复评分沿用同一 optimization trial mask；完整序列仍参与因果状态递推，但只有训练前缀
参与参数选择。全试次分析应明确保持 `max_trials: null`，不能继承旧的 64-trial 探索上限。

Model0826 的正式恢复由 `src/Bayesian_state/workflows/runs/run_model_0826_recovery.py` 编排。模块恢复的 Hyper-CD 与
final-rescore 都只能读取 70% 时间前缀；冻结赢家后才由 evaluation 层用新的候选配对 PF seeds
计算 30% 后缀 NLL。参数恢复使用全部试次拟合，但恢复判定不会只比较一个估计点：final-rescore
shortlist 的全部候选和生成真值会在同一组独立 seeds 下重新评分，真值相对该集合最小 total NLL
的差值决定 `delta NLL <= 2` near-best coverage。搜索产生的粒子权重、坐标移动和 shortlist
选择都属于统计估计程序，不是认知机制。

计算优化后的预注册配置是
`configs/exp123/specific_models/model_0826_recovery_v2.yaml`。它不改模型结构或最终评分精度：coarse
搜索用 R16×B4，fine 搜索用 R64×B8，shortlist 仍用校准冻结的高预算独立 seeds 最终复评分。
低预算只负责保留候选；校准必须逐数据集检查高预算赢家是否分别位于 coarse top-4 和 fine
top-2，未通过便禁止进入正式恢复。初始 R32×B4 fine 预算只在 3/6 个校准数据中保留了
R128×B16 赢家，补测 R32×B8 也只有 5/6；R64×B8 达到 6/6，因此在不放宽 top-2
门槛的前提下采用最小通过预算。PF 最终预算的相邻稳定性也以“高预算赢家是否位于低预算
top-k”判定，同时保留 exact-winner agreement 作为诊断，避免把近乎并列的 NLL 抖动误判成
数值失败。

受墙钟限制时，runner 的 `priority-all --priority-subject <id>` 会先完成指定 subject template
的模块和参数恢复，再继续其余被试。它不缩减任何被试的试次数、生成重复、候选结构或最终 PF
预算；subject-level 图和 JSON 明确是运行检查点，只有所有 36+40 数据集齐全时才计算正式总门槛。

顶层 workflow 与结果序列化由：

- `src.Bayesian_state.run_hyper_then_simulation`
- `src.Bayesian_state.run_simulation`
- `src.Bayesian_state.run_hyper_evaluation`

负责。optimizer 本身应保持可由测试和 notebook 直接调用。

Condition 3 的 `model_0826.build_model_0826_cell_engine()` 第一版只接受 PMH，保留
`hierarchical_pairing` 似然与 `HierarchicalPairingMemoryModule` 的联合更新。二者必须同时
显式配置；构建器拒绝 P、PM、PH，不会把联合记忆替换为普通 `DualMemoryModule`。拟合仍使用
已有 `gamma` 参数路径，配对权重由每条候选轨迹重新学习。该接口检查不代表完成了 condition 3
的搜索预算校准或恢复验证；已有 condition 1/2 架构单元的构建方式不变。

`diagnostics/search.py` 不启动模型，适合快速检查搜索产物；
`diagnostics/predictive.py` 会重新采样或运行模型，应显式控制 repeats、subjects 和 `n_jobs`。
