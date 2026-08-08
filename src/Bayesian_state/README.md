# Bayesian_state

`Bayesian_state` 是本项目的试次级 Bayesian 状态模型包。它把“模型结构”“逐试次推理”
“潜在路径积分”“超参数搜索”“重复仿真”和“结果评价”分成相互独立的层，而不是把一个模型
写成一份从数据读取到画图的独立脚本。

当前正式建模路径以 `StateModel + BaseEngine + modules` 为核心。`model_0806` 的动态连续
hypothesis-transition 模型也已经接入这条路径；`manuscript_models/` 中的同名实现保留作
恢复实验和数值参考，不是正式拟合入口。

## 1. 总体结构

```text
configs/*.yaml
    │
    ▼
run_* entrypoints
    │
    ├── optimization/        配置解析、损失、重复评价、Hyper-CD/Grid
    ├── metrics/             共享 proper scores、曲线、行为和跨重复统计
    │
    ▼
inference_engine/dispatcher
    │
    ├── trajectory backend
    └── particle-filter backend
    │
    ▼
problems/StateModel → inference_engine/BaseEngine
    │
    ▼
problems/modules/            perception → transition → likelihood → memory → beta
    │
    ▼
results/subjects/*.json + optional compressed run streams
    │
    ├── model_evaluation/    指标、PPC、状态轨迹、口述规则对齐
    └── clustering/          run-level 轨迹的事后聚类
```

依赖方向原则是：执行与分析层可以调用模型层，模型层不应反向调用具体 CLI、结果目录或绘图
脚本。共享路径、数据解析和小型统计函数放在 `utils/`。

## 2. 目录职责

| 目录 | 职责 | 是否属于正式运行主路径 |
|---|---|---|
| `inference_engine/` | backend dispatch、单轨迹/粒子推理、概率容器、trial scheduler 与状态快照 | 是 |
| `problems/` | `StateModel`、hypothesis space、partition 几何 | 是 |
| `problems/modules/` | 感知、transition、likelihood、memory、beta 等认知机制 | 是 |
| `metrics/` | optimization、simulation 与 evaluation 共享的纯数值指标和数据契约 | 是，底层支持 |
| `optimization/` | 推理结果标准化、损失、重复评价、机制候选与 Hyper-CD/Grid | 是 |
| `simulation/` | 自主 choice/feedback trajectory 和 synthetic recovery 数据生成 | 按需 |
| `model_evaluation/` | 已完成仿真的统计、作图和 oral/model alignment | 是，属于后处理 |
| `clustering/` | 对保存的 run-level trajectory 做 FFT 聚类 | 可选后处理 |
| `manuscript_models/` | 论文阶段的独立数值实现、恢复和 oracle | 否；参考与复现用途 |
| `utils/` | 路径、数据集、subject override、stream、公共统计 | 是，底层支持 |

根目录的可执行文件只负责 orchestration：

| 文件 | 作用 |
|---|---|
| `run_simulation.py` | 固定参数、逐被试重复 simulation 并序列化 |
| `run_hyper_then_simulation.py` | Hyper 搜索、生成 subjectwise simulation YAML、再运行 simulation |
| `run_hyper_evaluation.py` | 已完成 Hyper-CD 输出的收敛和选择诊断 |
| `run_model_evaluation.py` | 已完成 simulation 输出的统一后处理 |
| `__init__.py` | package marker；公共对象应从职责明确的子包显式导入 |

更详细的说明见各目录 README：

- [`inference_engine/README.md`](inference_engine/README.md)
- [`metrics/README.md`](metrics/README.md)
- [`problems/README.md`](problems/README.md)
- [`problems/modules/README.md`](problems/modules/README.md)
- [`optimization/README.md`](optimization/README.md)
- [`simulation/README.md`](simulation/README.md)
- [`model_evaluation/README.md`](model_evaluation/README.md)
- [`manuscript_models/README.md`](manuscript_models/README.md)
- [`clustering/README.md`](clustering/README.md)
- [`utils/README.md`](utils/README.md)

根目录的 `PMH modules.svg` 是较早期的 module-loop 示意图，可用于理解黑板式调度，但它
早于当前的 inference backend 分层；目录边界和正式入口以本 README 与 YAML
配置为准。

## 3. 建议阅读顺序

第一次阅读代码时建议按以下顺序：

1. `problems/model.py`：模型如何从 YAML 装配。
2. `inference_engine/bayesian_engine.py`：一个 trial 怎样按 `agenda` 调度。
3. `problems/modules/README.md`：各认知模块的输入、状态与公式。
   `problems/modules/readout.py` 集中 choice、output-noise、RT 和 oral-report 读出。
4. `inference_engine/dispatcher.py` 与 `inference_engine/backends/`：怎样选择和运行推理后端。
5. `metrics/README.md`：共享指标的数据契约与依赖边界。
6. `optimization/optimizer_common.py`：推理输出怎样转换为公共指标和损失。
7. `optimization/optimizer_simulation.py`：独立随机重复怎样聚合。
8. `optimization/hyper_cd_optimizer.py` 或 `optimization/hyper_grid_optimizer.py`：超参数搜索。
9. `run_simulation.py`、`run_hyper_then_simulation.py`：顶层执行和序列化。

若只关心 0806，再阅读：

1. `problems/modules/hypo_transition/README.md`
2. `problems/modules/hypo_transition/process.py`
3. `problems/modules/hypo_transition/dynamic_continuous.py`
4. `inference_engine/backends/particle_filter.py`
5. `configs/model_struct/pmh_model_cond1_0806.yaml`
6. `docs/model_0806_workflow.md`

## 4. 一个 trial 的数据流

标准 observation 为：

```python
(stimulus, choice, feedback)
```

典型 `agenda`：

```text
perception_mod
  → hypo_transitions_mod
  → likelihood_mod
  → memory_mod
  → beta_mod
```

各步职责：

1. `perception_mod` 产生内部感知刺激并写回 `engine.observation`。
2. `hypo_transitions_mod` 决定当前 active hypotheses，并把上一 posterior 映射为当前 prior。
3. `likelihood_mod` 通过 partition 计算当前反馈对各 hypothesis 的 likelihood。
4. `memory_mod` 融合短期/长期证据，写入 `engine.posterior`。
5. `beta_mod` 更新下一 trial 使用的 hypothesis-specific inverse temperature。

在下一个 trial 开始时，engine 先执行 `prior <- posterior`，transition module 可再对其进行
集合迁移。`prior_t` 因而表示看到 trial `t` 的 choice/feedback 之前的预测状态。

## 5. 两种推理后端

### 5.1 单轨迹后端

未配置 `inference`，或设置：

```yaml
inference:
  backend: trajectory
```

时，`inference_engine/backends/trajectory.py` 调用 `StateModel.fit_step_by_step()`，运行一条
随机认知轨迹。重复仿真由
`StateModelSimulationRunner` 生成独立 trajectory seeds。

### 5.2 粒子后端

```yaml
inference:
  backend: particle_filter
  particle_count: 512
  resample_threshold_fraction: 0.5
```

`inference_engine/dispatcher.py` 读取这段配置，标准 `evaluate_state_model_run()` 随后获得粒子
边际输出，对不可见的 perception/transition
路径求边际预测。每个粒子仍然是正常的 `StateModel`，重采样依靠 engine/module 的
`state_dict()` 与 `load_state_dict()`，没有另一套认知模型状态。

两个 backend 均返回 `inference_engine/results.py` 定义的 `InferenceResult`。优化与评估层
优先读取其中的公共 probability/state/latent/diagnostic mappings；旧 backend-specific 属性
继续作为兼容入口。

当前通用粒子入口支持 condition 1、`expectation`/`sharpened_expectation` readout，以及
uniform `base_lapse`。历史依赖 lapse、RT emission 和其他 condition 尚未进入这一入口。

## 6. 配置分层

配置位于仓库根目录 `configs/`，分成三层：

```text
hyper_cd_cfg or hyper_grid_cfg
        │ base_sim_config_path
        ▼
simulation_cfg
        │ engine_config_path / engine_config
        ▼
model_struct
```

- `model_struct/*.yaml`：partition、module class/kwargs、agenda、readout、output noise、inference backend。
- `simulation_cfg/*.yaml`：subjects、dataset、repeats、prediction mode、loss、输出目录。
- `hyper_*_cfg/*.yaml`：搜索空间、目标排序、coarse/fine 预算。

相对路径永远相对于“声明该路径的 YAML 所在目录”解析。subject-specific 设置通过
`subject_overrides` 合并；超参数路径必须以 `engine.` 或 `simulation.` 开头。

## 7. 正式入口

### 超参数搜索

```bash
python -m src.Bayesian_state.optimization.hyper_cli \
  --backend cd \
  --config configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml
```

将 `cd` 替换为 `grid` 可运行显式网格搜索。

### 搜索后生成 subjectwise 配置并仿真

```bash
python -m src.Bayesian_state.run_hyper_then_simulation \
  --backend hyper_cd \
  --hyper-config configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml
```

### 固定参数仿真

```bash
python -m src.Bayesian_state.run_simulation \
  --config configs/simulation_cfg/pmh_cond1_simulation_0806.yaml
```

### Hyper 搜索诊断

```bash
python -m src.Bayesian_state.run_hyper_evaluation \
  --input-dir results/state-based-hyper-cd/pmh/cond1_0806_selected8
```

### 仿真结果评价

```bash
python -m src.Bayesian_state.run_model_evaluation \
  --input-dir results/state-based-simulation/pmh/cond1_0806
```

所有命令建议从仓库根目录执行。

## 8. 输出约定

Hyper 搜索通常写入：

```text
<hyper-output>/
  best_hyperparams.json
  subject_<id>/
    best_hyperparams.json
    all_combinations.jsonl
    stage_summary.json
    restart_summary.json          # Hyper-CD
    coordinate_trace.jsonl        # Hyper-CD
```

固定仿真通常写入：

```text
<simulation-output>/
  subjects/subject_<id>.json
  cache/subject_<id>_raw_runs.gz  # keep_logs=True 时可能存在
```

subject JSON 保存轻量 summary、representative run 和 stream reference；大规模逐 run 状态通过
`utils.stream.StreamList` 压缩存储，避免 JSON 无限膨胀。需要 trajectory-rank、posterior-rank
或 FFT clustering 时，最终仿真必须保留相应日志。

## 9. 0806 在当前架构中的位置

0806 的 choice 主路径已经主框架化：

- dynamic-continuous hypothesis transition 是普通 `hypo_transitions_mod`；
- surprise、uncertainty 及联合 controller 都通过 module kwargs 表达；
- 粒子滤波由 `inference.backend` 选择；
- choice Brier、NLL、accuracy curves、Hyper-CD 和结果序列化复用公共实现。

尚保留在 `manuscript_models/model_0806.py` 的 RT emission、autonomous recovery 与旧 rolling
实验属于参考工作流。把这些功能迁入正式框架时，应增加独立 module/adapter，而不是继续扩展
一套平行的完整模型。

## 10. 扩展原则

新增模型机制时：

1. 如果改变 trial 内认知状态，放入 `problems/modules/`。
2. 如果只是改变 hypothesis geometry，扩展 `problems/partitions.py` 或新增 partition class。
3. 如果改变潜在状态积分方法，放入 `inference_engine/backends/`；不要复制认知更新方程。
4. 如果新增评价指标，优先接入 `optimizer_common.py` 和 `utils/simulation_statistics.py`。
5. 如果只是复现论文某阶段的冻结算法，可放在 `manuscript_models/`，但必须明确不是正式入口。
6. 为所有有状态 module 实现快照、恢复、日志清理；随机 module 还应实现 future reseeding。
7. 配置 class path、README、回归测试必须与代码一起更新。
