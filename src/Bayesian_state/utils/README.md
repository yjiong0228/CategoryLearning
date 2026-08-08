# utils

本目录只放跨模型、跨执行层复用的底层工具。认知机制、optimizer、评价图和 manuscript model
都已移到各自 package；不要再把新的完整工作流堆回 `utils/`。

## 文件说明

| 文件 | 职责 |
|---|---|
| `paths.py` | 从源码位置推导仓库根目录、configs、data、logs 和 results 路径 |
| `datasets.py` | 解析 simulation YAML 的 dataset block 和数据文件绝对路径 |
| `config_subjects.py` | deep merge、移除/读取 subject overrides、生成 subject-specific config |
| `simulation_statistics.py` | 旧 statistics schema、selection aliases 与 `metrics/` 的兼容入口 |
| `seeding.py` | 跨 inference、simulation、optimization 共享的稳定 seed 派生 |
| `stream.py` | `StreamList`：大体积 run log 的 gzip 流式存取 |
| `basic_stat.py` | `softmax`、Euclidean distance、entropy 等小型数值函数 |
| `classical_tools.py` | 保留的经典衰减工具 |
| `load_config.py` | 轻量 YAML loader 与旧 `MODEL_STRUCT` 全局容器 |
| `base.py` | logger 和旧路径容器 |
| `console_styles.py` | 兼容旧代码的彩色 `print` |
| `__init__.py` | 重导出部分 legacy utility API |

## 路径约定

`paths.py` 以文件位置而非当前工作目录确定：

```text
ROOT_DIR
CONFIGS_DIR
PROCESSED_DATA_DIR
SIMULATION_RESULTS_DIR
```

YAML 内的相对路径不要直接拼到 `ROOT_DIR`。它们由 `datasets.py` 或
`optimization/optimization_config.py` 相对于声明该路径的 YAML 目录解析。

## Subject overrides

以下 key 被视为同义 subject override block：

```text
subject_overrides
subject_configs
per_subject
```

合并顺序是 base config 后叠加对应 subject override。mapping 递归合并；需要整体替换 mapping
的 hyperparameter candidate 由 optimization 层显式处理。

## Repeated-simulation statistics

共享指标的规范实现位于 `metrics/`。`simulation_statistics.py` 暂时接受标准
`SingleRunResult` 序列并维持原有结构化 mapping：

```text
statistics.loss.*
statistics.marginal_prediction.*
statistics.diagnostics.*
statistics.scores.*
```

Hyper-CD 的 `objective_order` 会直接引用这些 dotted paths。因此修改字段名或定义属于结果
schema 变更，必须同步 configs、hyper evaluation 和 tests。

## Stream storage

`StreamList` 用于存放 `keep_logs=True` 时的大型 run records。subject JSON 通常只保存类似：

```json
{"path": "cache/subject_103_raw_runs.gz", "count": 4}
```

移动结果目录时必须保持 JSON reference 与 cache 的相对结构，或通过公共 rebase helper 更新。

## 不应放在这里的内容

- 新 transition/memory/perception：放 `problems/modules/`
- 新 proper score、曲线或行为数值指标：放 `metrics/`
- 新 optimizer 或 loss：放 `optimization/`
- 新 plot/alignment：放 `model_evaluation/`
- 论文专用冻结算法：放 `manuscript_models/`
- 新 inference backend 或 posterior predictive：放 `inference_engine/`
- synthetic trajectory generation：放 `simulation/`
