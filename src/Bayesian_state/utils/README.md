# 通用工具

本目录只放跨模型、跨执行层复用的底层工具。认知机制、optimizer、评价图和 manuscript model
都已移到各自 package；不要再把新的完整工作流堆回 `utils/`。

## 文件说明

| 文件 | 职责 |
|---|---|
| `paths.py` | 从源码位置推导仓库根目录、configs、data、logs 和 results 路径 |
| `datasets.py` | 解析 simulation YAML 的 dataset block 和数据文件绝对路径 |
| `subjects.py` | deep merge、移除/读取 subject overrides、生成 subject-specific config |
| `seeding.py` | 跨 inference、simulation、optimization 共享的稳定 seed 派生 |
| `parallel.py` | 完整 CPU 预算、独立任务进程并行、每个 worker 单线程及嵌套池约束 |
| `streaming.py` | `StreamList`：大体积 run log 的 gzip 流式存取 |
| `numeric.py` | `softmax`、Euclidean distance、entropy 等小型数值函数 |
| `decay.py` | 保留的经典衰减工具 |
| `config.py` | 轻量 YAML loader 与延迟加载的 `MODEL_STRUCT` 映射 |
| `logging.py` | import-safe logger、路径容器与显式 `configure_logging()` |
| `console.py` | 彩色控制台输出 |
| `__init__.py` | 明确列出公共 utility API |

## 路径约定

`paths.py` 以文件位置而非当前工作目录确定：

```text
ROOT_DIR
CONFIGS_DIR
PROCESSED_DATA_DIR
SIMULATION_RESULTS_DIR
```

YAML 内的相对路径不要直接拼到 `ROOT_DIR`。它们由 `datasets.py` 或
`simulation/config.py` 相对于声明该路径的 YAML 目录解析。

## 导入行为

导入 `src.Bayesian_state.utils` 不创建目录、日志文件，不配置 root logger，也不扫描 model YAML。
CLI 在 `main()` 中显式调用 `configure_logging()`；库调用方可自行管理 logging。
`MODEL_STRUCT` API 在首次迭代、取值或查询长度时加载配置。

`parallel.single_threaded_processes()` 在调用期间将数值线程限制为 1，并给 loky worker
显式设置 `inner_max_num_threads=1`；退出时恢复调用方的线程设置。`parallel_job_count()`
取请求预算、可用 CPU 和当前独立任务数的最小值，不预留核心；进程池内再次调用时串行执行，
由外层统一占用预算。Model 0826 的正式默认预算是 128，明确的小规模配置仍被尊重。
导入该模块不设置全局环境变量。完整说明见
[并行策略](../docs/maintenance/model_0826_parallel_policy_20260917.md)。

## 被试级配置覆盖

以下 key 被视为同义 subject override block：

```text
subject_overrides
subject_configs
per_subject
```

合并顺序是 base config 后叠加对应 subject override。mapping 递归合并；需要整体替换 mapping
的 hyperparameter candidate 由 optimization 层显式处理。

## 重复仿真统计

共享指标的规范实现位于 `metrics/`，重复运行的聚合由 `simulation/runner.py` 维护，并输出：

```text
statistics.loss.*
statistics.marginal_prediction.*
statistics.diagnostics.*
statistics.scores.*
```

Hyper-CD 的 `objective_order` 会直接引用这些 dotted paths。因此修改字段名或定义属于结果
schema 变更，必须同步 configs、hyper evaluation 和 tests。

## 流式存储

`StreamList` 用于存放 `keep_logs=True` 时的大型 run records。subject JSON 通常只保存类似：

```json
{"path": "cache/subject_103_raw_runs.gz", "count": 4}
```

`StreamList` 的整数索引和切片保持 Python 列表的顺序语义，包括负索引、负步长和
缓存命中后的切片。切片顺序扫描压缩记录，仅保留所请求条目；切片越大，返回列表占用越大。

移动结果目录时必须保持 JSON reference 与 cache 的相对结构，或通过公共 rebase helper 更新。

## 不应放在这里的内容

- 新 transition/memory/perception：放 `model/modules/`
- 新 proper score、曲线或行为数值指标：放 `metrics/`
- 新 optimizer 或 loss：放 `optimization/`
- 新 plot/alignment：放 `evaluation/`
- 论文专用冻结算法：放 `reference_models/`
- 新 inference backend 或 posterior predictive：放 `inference/`
- autonomous behavior execution：放 `simulation/`
