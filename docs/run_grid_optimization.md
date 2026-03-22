# `run_grid_optimization.py` 使用说明


## 概览

`src/Bayesian_state/run_grid_optimization.py`

1. 读取单个 YAML 配置文件。
2. 创建 `StateModelGridOptimizer`。
3. 按 `subjects`（或 `subject_range`）逐个被试执行网格搜索。
4. 对每个被试保存一个 JSON 结果文件。

> 运行策略是“被试外层串行 + 被试内部并行”（内部并行由 `n_jobs` 控制）。

---

## 运行方法

在项目根目录执行：

```bash
python -m src.Bayesian_state.run_grid_optimization --config configs/grid_opt_cfg/pmh_cond1.yaml
```

替换为其他配置：

```bash
python -m src.Bayesian_state.run_grid_optimization --config configs/grid_opt_cfg/pmh_cond2.yaml
python -m src.Bayesian_state.run_grid_optimization --config configs/grid_opt_cfg/pmh_cond3.yaml
```

---

## YAML 配置项说明

### 必填项

- `param_grid`：网格参数字典，格式为 `参数名 -> 列表`
  - 例如：
    - `gamma: [0.05, 0.1, 0.2]`
    - `w0: [0.01, 0.05, 0.1]`

- 被试范围（二选一）：
  - `subjects: [325, 326, 327]`
  - 或 `subject_range: [325, 332]`

- 引擎配置（二选一）：
  - `engine_config: {...}`（直接内嵌）
  - 或 `engine_config_path: ...`（外部 YAML 路径）

### 常用可选项

- `data_path`
  - 含义：行为数据 CSV 路径（默认：`data/processed/Task2_processed.csv`）
  - 注意：优化器内部 `processed_data_dir` 会自动取这个路径的父目录。

- `output_dir`
  - 含义：结果输出目录
  - 默认：`results/state-based-grid-result`

- `n_jobs`
  - 含义：被试内并行 worker 数
  - 默认：`4`

- `n_repeats`
  - 含义：每个参数组合重复次数
  - 默认：`4`

- `refit_repeats`
  - 含义：最佳参数重拟合次数
  - 默认：`64`

- `window_size`
  - 含义：滑窗大小
  - 支持两种形式：
    1. 标量（所有被试同一窗口）
    2. 列表（长度必须与 `subjects` 一致）

- `window_size_overrides`
  - 含义：按被试覆盖 `window_size`
  - 示例：`{325: 12, 326: 16}`

- `stop_at`
  - 含义：使用 trial 比例（默认 `1.0`）

- `max_trials`
  - 含义：最多 trial 数（可选，不填表示不截断）

- `keep_logs`
  - 含义：是否保留较详细日志信息（默认 `false`）

---

## 结果保存

输出目录由 `output_dir` 决定，脚本会自动创建目录。

每个被试输出一个 JSON：

- `subject_325.json`
- `subject_326.json`
- ...

保存路径示例：

```text
results/state-based-grid-result/subject_325.json
```

---

## 单个 JSON 结果字段

每个 `subject_*.json` 主要包含：

- `subject_id`：被试编号
- `condition`：条件编号
- `best_params`：最优参数（例如 `gamma`, `w0`）
- `mean_error`：最优参数平均误差
- `std_error`：最优参数误差标准差
- `n_repeats`：最优点重复次数
- `sample_errors`：最优点的样本误差列表（若可用）
- `metrics`：最优结果指标（如滑窗拟合指标）
- `param_grid`：本次网格定义
- `best_step_results`：最优重拟合 step 级结果（若可用）
- `strategy_counts_log`：策略计数日志（若可用）
- `posterior_log` / `prior_log`：后验/先验日志（若可用）
- `grid_summary`：所有网格点的紧凑摘要（`params`, `mean_error`, `std_error`）

> 所有内容会被转换成 JSON 可序列化类型（例如 numpy 数组转 list）。

---

## 推荐工作流

1. 先用小网格 + 小重复次数做 smoke test（如 `n_repeats=2`, `refit_repeats=8`）。
2. 确认 JSON 输出字段正常后，再切换到正式参数（如 `32/256`）。
3. 用不同 YAML（cond1/cond2/cond3）分批运行，便于管理输出目录。

---

## 画图（从 JSON 一步出图）

网格优化完成后，可直接对 `subject_*.json` 批量聚合并出图：

```bash
python -m src.Bayesian_state.eval_grid_results \
  --input-dir results/state-based-grid-result/pmh/cond3
```

默认会输出：

- 聚合 JSON：`<input-dir>/all_subjects.json`
- 图目录：`<input-dir>/plots/`
  - `accuracy.png`（best fit 曲线：预测 vs 真实）
  - `error_grid.png`（gamma × w0 误差热图）
  - `posterior.png`（posterior 轨迹）
  - `cluster_amount.png`（hypo 集合变化/策略动态）
  - `oral_vs_model.png`（oral 对齐）

可选参数：

- `--aggregate-output <path>`：指定聚合 JSON 路径
- `--plots-dir <dir>`：指定图输出目录
- `--plot-accuracy/--plot-grid/--plot-posterior/--plot-cluster/--plot-oral <path>`：分别指定单张图输出路径
- `--oral-data <csv>`：口头报告数据路径（默认 `data/processed/Task2_processed.csv`）

建议：

- 若希望稳定输出 5 张图，请在优化阶段设置 `keep_logs: true`。
- `keep_logs: false` 时，仍可输出 `accuracy` 与 `error_grid`，但 `posterior/cluster/oral` 可能因缺少 step-level 日志而自动跳过。
