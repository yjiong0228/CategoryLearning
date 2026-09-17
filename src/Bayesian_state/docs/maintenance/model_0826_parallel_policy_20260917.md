# Model 0826：128 核并行策略（2026-09-17）

用户要求正式计算尽量用满当前 128 核 CPU，不主动留余量。本次仅调整计算资源分配；
不改变认知更新、似然、预测时序、粒子数、重复次数、随机种子、候选顺序和评分定义。

## 正式配置

| 入口 | 设置 |
| --- | --- |
| `configs/exp123/specific_models/model_0826_recovery_v1.yaml`、`model_0826_recovery_v2.yaml` | `search.cd.parallel_budget: 128` |
| `configs/exp123/simulation_cfg/model0826_cond1_recovery_base.yaml` | `n_jobs: 128` |
| `CategoryLearning_codes/Bayesian_model/configs/recovery_v1.yaml`、`recovery_v2.yaml` | `search.cd.parallel_budget: 128` |
| `CategoryLearning_codes/Bayesian_model/configs/recovery_simulation.yaml` | `n_jobs: 128` |
| `optimization.model_0826.build_model_0826_hyper_config()` | 未显式指定时生成 `cd.parallel_budget: 128` |
| `run_autonomous_trajectory_evaluation --n-jobs` | 默认 128 |
| `run_internal_cognitive_trajectory_evaluation --jobs` | 默认 128 |
| `run_hyper_evaluation --volatility-n-jobs` | 默认 128 |

以上六份 recovery YAML 原预算为 64，现统一为 128。
`model0826_cond1_exploratory_observed_fit.yaml` 原本就是 128，无需改动。
显式的小预算覆盖仍有效；单进程 smoke/test 配置保留原值。旧模型、历史运行快照、
一次性审计入口和基准脚本不批量更改默认预算。

## 执行约束

共享实现位于 `src/Bayesian_state/utils/parallel.py`：

1. 进程数为 `min(请求预算, 当前可用 CPU 数, 当前就绪独立任务数)`。可用 CPU 由
   joblib 按 affinity / 系统配额识别；本机验证为 128。没有减去预留核。
2. 每个 worker 的 BLAS/OpenMP/NumExpr/Numba 线程预算为 1。通过
   `parallel_config(backend="loky", inner_max_num_threads=1)` 显式覆盖继承环境，
   避免 128 个进程再各开几十个计算线程。`threadpool_limits(1)` 同时约束父进程中
   已加载的数值库，退出计算上下文后恢复调用方设置。
3. 外层进程池内的模型调用串行执行，不再启动内部进程池。当前任务使用一层并行预算。
4. 任务 seed 仍由既有稳定派生方法确定，结果按输入任务顺序收集。完成先后不会重排
   候选评分或改变 seed 分配。

该上下文接入固定参数仿真、Hyper-CD 评分、recovery 候选/冻结评分、预测诊断、
自主轨迹与内部认知路径生成。新直接依赖 `threadpoolctl>=3.1.0` 已写入 requirements；
本机原环境已有该包，没有重装环境。

## 利用率的实际边界

128 是允许使用的完整并行预算，不代表每一时刻都有 128 个独立任务。
例如当前坐标只有 3 个候选、每个候选 8 个 PF seed 时，这一批最多使用 24 个 worker；
固定参数仿真的单个被试只有 4 次 repeat 时，最多使用 4 个 worker。

现有跨被试循环和 CD 坐标依赖关系保留。要让多个小批次共同持续占满全部 CPU，还需单独实现
跨被试/独立搜索的统一调度；这不是把 `n_jobs` 再调高就能解决的。本次不为填满 CPU 增加
科学重复数，也不改变 CD 的逐坐标更新方式。多个各自申请 128 的独立脚本同时运行会竞争同一
机器资源，当前预算不是跨脚本的全局调度器。

## 验证与启用

验证结果保存在 `results/model_0826/parallel_128_20260917/`：

- 单元测试检查预算无预留、任务/CPU 上限、父线程设置恢复、继承的 64 线程环境被覆盖，
  以及嵌套调用不创建额外进程。
- `check_parallel_runtime --workers 128` 用屏障确认实际同时进入 128 个不同 worker；
  每个任务只计算 S129 的 8 个 trial、16 粒子，6 项输出对照独立保存的旧版数值参考。
- 共享核心及期刊回归测试包含独立保存的合并前数值参考；CLI 用 `--help` 检查入口。

这些是有界验证，没有启动完整拟合或 recovery，也不把短任务启动时间解释为长期吞吐提升。
已在运行的进程不会自动采用新配置，`results/` 内的旧冻结 YAML 不修改。
共享源码变更会改变恢复指纹，新正式运行使用新的输出目录。
