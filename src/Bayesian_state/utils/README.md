# `src/Bayesian_state/utils` 说明

这个目录主要放三类内容：

- 运行优化所需的核心工具（`state_grid_optimizer.py`, `state_amr_optimizer.py`）
- 运行优化所需的核心工具（`state_grid_optimizer.py`, `state_amr_optimizer.py`, `optimization_common.py`）
- 结果评估与口头报告对齐工具（`model_evaluation.py`, `oral_process.py`）
- 基础通用工具（路径/日志、统计函数、配置加载、流式缓存）

下面按“是否是当前主链路”来梳理。

## 1) 当前主链路核心文件（建议优先看）

### `state_grid_optimizer.py`
- 作用：对 `StateModel` 做网格搜索优化（按被试）。
- 主要类：`GridPointResult`, `StateModelGridOptimizer`。
- 功能：网格搜索策略层（共享评估逻辑已抽到 `optimization_common.py`）。
- 被谁调用：`src/Bayesian_state/run_grid_optimization.py`。

### `state_amr_optimizer.py`
- 作用：对 `StateModel` 做 AMR（自适应网格细化）优化。
- 主要类：`StateModelAMROptimizer`, `AMRGridSearch`, `AMRResult`。
- 功能：AMR 搜索策略层（共享评估逻辑已抽到 `optimization_common.py`）。
- 被谁调用：`src/Bayesian_state/run_amr_optimization.py`。

### `optimization_common.py`
- 作用：`grid` / `amr` 共用逻辑层。
- 主要内容：`BaseStateOptimizer`, `GridPointResult`, `evaluate_state_model_run` 等。
- 价值：避免双份维护数据切片、参数注入、单次评估、指标计算。

## 2) 评估与分析文件

### `model_evaluation.py`
- 作用：画图与结果可视化。
- 主要类：`ModelEval`。
- 典型图：accuracy、posterior、cluster 动态、oral 对齐等。
- 被谁调用：`eval_grid_results.py`, `eval_amr_results.py`。

### `oral_process.py`
- 作用：把口头报告信息映射到 hypothesis 命中/对齐指标。
- 主要类：`Oral_region_analysis`, `Oral_center_analysis`。
- 说明：区域重叠法与坐标映射法分别封装为两个分析类。
- 被谁调用：`eval_grid_results.py`, `eval_amr_results.py`（主要用 `Oral_center_analysis`）。

### `perception_stats.py`
- 作用：计算被试级 perception 噪声统计（mean/std），供模型注入。
- 主要接口：`get_perception_noise_stats`。
- 被谁调用：`src/Bayesian_state/problems/model.py`。

## 3) 基础设施与通用函数

### `base.py`
- 作用：定义项目路径 `PATHS` 和 logger `LOGGER`。
- 特点：导入时会创建目录并初始化日志配置（有导入副作用）。
- 被谁调用：`model.py`, `state_*_optimizer.py`, `load_config.py` 等。

### `load_config.py`
- 作用：加载 `configs/` 下 YAML，构建 `MODEL_STRUCT`。
- 特点：导入时会扫描并加载配置（有导入副作用）。
- 被谁调用：`oral_process.py`，并通过 `utils/__init__.py` 间接暴露给其它模块。

### `basic_stat.py`
- 作用：基础数学工具：`softmax`, `euc_dist`, `entropy`。
- 被谁调用：`problems/base_problem.py`，进一步被模型模块使用。

### `classical_tools.py`
- 作用：传统衰减函数 `two_factor_decay`。
- 被谁调用：`problems/base_problem.py`。

### `console_styles.py`
- 作用：终端彩色打印封装（重载 `print`）。
- 被谁调用：`base.py`，并通过 `utils/__init__.py` 暴露。

### `stream.py`
- 作用：`StreamList`，基于 `gzip+pickle` 的流式列表存储。
- 状态：在当前 `Bayesian_state` 主链路中使用很少，主要历史兼容用途。

## 4) 兼容/历史文件（主链路里不推荐新增依赖）

`optimizer.py` 已移除。当前主链路统一使用 `state_grid_optimizer.py` 与 `state_amr_optimizer.py`。

### `__init__.py`
- 作用：聚合导出常用工具（`base/basic_stat/classical_tools/load_config/console_styles` + `StateModelGridOptimizer`）。
- 注意：由于导入了 `base.py` 和 `load_config.py`，`import src.Bayesian_state.utils` 会触发路径创建/配置加载。

## 5) 快速导航（按你的任务）

- 跑参数搜索：看 `state_grid_optimizer.py` / `state_amr_optimizer.py`
- 改搜索算法：看 `state_amr_optimizer.py` 里的 `AMRGridSearch`
- 改模型评估图：看 `model_evaluation.py`
- 改口头报告对齐：看 `oral_process.py`
- 改 perception 噪声注入：看 `perception_stats.py` + `problems/model.py`
- 改路径/日志/配置加载：看 `base.py`, `load_config.py`

## 6) 当前重构建议（仅目录层面）

- 保留主链路：`state_grid_optimizer.py`, `state_amr_optimizer.py`
- 保留主链路：`state_grid_optimizer.py`, `state_amr_optimizer.py`, `optimization_common.py`
- 逐步降级旧链路：`optimizer.py`, `stream.py`（先标注 legacy，再决定是否迁移）
- 拆分 `oral_process.py` 为两个文件会更清晰：
  - `oral_region_analysis.py`（区域重叠）
  - `oral_coordinate_mapping.py`（坐标映射）
