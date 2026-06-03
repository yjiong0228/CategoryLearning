# Bayesian_state 读者手册

`Bayesian_state` 是本仓库当前主用的 state-based 建模管线。  
这份文档把 `state-based_model.md` 和 `StateBasedOpt.md` 的核心内容合并在一起，目标是让你“一份文档看懂模型如何运作 + 如何跑优化评估”。

## 1. 你先要知道的三件事

1. 这个模型是“模块化贝叶斯引擎”：每一步推断按 `agenda` 顺序执行模块。  
2. 配置分两层：  
   - `configs/model_struct/*.yaml` 定义模型结构（有哪些模块、执行顺序、模块默认参数）。  
   - `configs/grid_opt_cfg/*.yaml` / `configs/amr_opt_cfg/*.yaml` 定义优化任务（被试、参数搜索范围、输出路径等）。  
3. 每个被试优化后会在输出目录的 `subjects/` 下生成 `subject_<id>.json`，评估脚本再聚合并出图。

## 2. 模块结构总览

目录里最关键的文件：

- `inference_engine/bayesian_engine.py`：核心引擎 `BaseEngine`
- `problems/model.py`：`StateModel` 封装，负责创建 partition 和 engine
- `problems/modules/`：各功能模块（perception / likelihood / memory / beta / hypo transition）
- `utils/optimizer_grid.py`：网格搜索优化器
- `utils/optimizer_amr.py`：AMR 优化器
- `utils/oral_model_alignment.py`：口头报告分析与 oral/model 对齐（region/center 两种）

## 3. 模型如何在一步 trial 中运行

引擎持有共享状态：`prior`、`likelihood`、`posterior`、当前 observation、活跃假设集合等。  
每个 trial 调用 `infer_single` 时，按 `agenda` 依次执行模块（例如 perception -> hypothesis transition -> likelihood -> memory/beta）。

常见模块作用：

- `PerceptionModule`：给刺激加入被试特异的知觉噪声（可显式给 `mean/std`，也可自动读取数据）。
- `LikelihoodModule`：计算当前观察在每个假设下的似然。
- `DualMemoryModule`：融合历史信息（静态 + 衰减记忆）更新后验。
- `HypothesisTransitions` 系列：控制活跃假设集合如何扩展/替换。
- `BetaModule`：更新 softmax 温度/决策锐度相关参数。

## 4. 配置文件关系（最容易混淆）

建议记住：

- `model_struct` 文件是“模型骨架”：模块类路径、agenda、默认 kwargs。
- `*_opt_cfg` 文件是“实验计划”：要优化哪些参数、哪些被试、并行设置、输出目录。

优化配置里你通常会看到：

- `engine_config_path`: 指向某个 `configs/model_struct/*.yaml`（推荐）
- `engine_config`: 可选的局部覆盖（会与 `engine_config_path` 深合并）
- `param_grid`: 要搜索的参数空间
- `subjects` 或 `subject_range`: 默认被试；`subject_range: [125, 132]` 是闭区间
- `dataset`: 数据文件映射；推荐用它切换 `data` / `data_meg` / `data_exp4`
- `keep_logs`: 是否保存 step 级日志（画 cluster/oral 图通常需要）

运行时也可以用 terminal 参数覆盖 YAML 默认被试：`--subjects 125 126` 或 `--subject-range 125 132`。如果同时提供，`--subjects` 优先。

`dataset` 示例：

```yaml
dataset:
  processed_dir: ../../data_meg/processed
  learning_data: Task3b_processed.csv
  perception_summary: Task1b_errorsummary.csv
  perception_summary_72: Task1b_errorsummary_72.csv
  feature_order_data: Task3b_processed.csv
```

其中 `learning_data` 是拟合用行为/刺激主表；`perception_summary` 和 `perception_summary_72` 给 `PerceptionModule` 自动加载噪声参数；`feature_order_data` 用来读取 `feature1_name` 到 `feature4_name` 的顺序。

旧的 `data_path` 仍然兼容，但新配置建议使用 `dataset`。

## 5. 运行流程

### 5.1 Grid 优化

```bash
python -m src.Bayesian_state.run_grid_optimization \
  --config configs/grid_opt_cfg/pmh_cond1.yaml
```

覆盖默认被试：

```bash
python -m src.Bayesian_state.run_grid_optimization \
  --config configs/grid_opt_cfg/pmh_cond1.yaml \
  --subject-range 125 132

python -m src.Bayesian_state.run_grid_optimization \
  --config configs/grid_opt_cfg/pmh_cond1.yaml \
  --subjects 125 126 127
```

### 5.2 AMR 优化

```bash
python -m src.Bayesian_state.run_amr_optimization \
  --config configs/amr_opt_cfg/pmh_cond1.yaml
```

AMR 同样支持 `--subjects` 和 `--subject-range` 覆盖 YAML 默认被试。

`run_amr_optimization` 兼容旧参数名 `--opt-config`，但建议统一用 `--config`。

## 6. 评估与作图

### 6.1 AMR 结果评估

```bash
python -m src.Bayesian_state.eval_amr_results \
	  --input-dir results/state-based-AMR-result/pmh/cond1 \
	  --aggregate-output results/state-based-AMR-result/pmh/cond1/all_subjects.json \
	  --plot-accuracy results/state-based-AMR-result/pmh/cond1/accuracy.png \
	  --plot-cluster results/state-based-AMR-result/pmh/cond1/cluster_amount.png
```

### 6.2 Grid 结果评估

`eval_grid_results.py` 的参数风格与 AMR 评估基本一致（同样支持 `--input-dir`、`--plot-accuracy`、`--plot-cluster` 等）。

## 7. 结果文件约定

优化输出目录下通常有：

- `subjects/subject_<id>.json`：单被试结果摘要（schema v4，含 best run 与 refit 统计）
- `cache/subject_<id>_raw_step_results.gz`：可选的 refit 全部 step 轨迹流式缓存（当 `keep_logs: true`）
- `all_subjects.json`：聚合结果
- `accuracy.png`：准确率对比图
- `cluster_amount.png`：策略/簇变化图（需要 step 日志）
- `plots/oral_<mode>_mode/*_based_alignment_*.png`：五类 oral/model alignment 图

### 7.1 `subjects/subject_<id>.json`（schema v4）核心字段

- `best_error`：最佳参数下 refit 多次中的最小误差（best run）
- `mean_error`：与 `best_error` 对齐的兼容字段（用于旧脚本兼容）
- `refit_mean_error` / `refit_std_error`：最佳参数下 refit 误差分布统计
- `sample_errors`：最佳参数下每次 refit 的误差样本
- `best_metrics` / `metrics`：best run 对应的预测曲线（`metrics` 保留为兼容别名）
- `grid_errors`：每个参数组合的误差样本列表
- `grid_summary`：每个参数组合的均值/方差摘要
- `selection_meta`：参数选择与 run 选择规则
- `raw_runs_ref` / `raw_step_results_ref`：若存在，指向 `cache/*.gz` 的流式轨迹引用；在 `subjects/` 布局中通常保存为 `../cache/*.gz`

说明：

- 参数选择口径：按参数组合平均误差最小（`min_mean_error`）
- run 保存口径：在最佳参数下保存误差最小的那次 run（`min_error`）

评估脚本（`eval_grid_results.py` / `eval_amr_results.py`）已兼容旧 schema 与新 schema。

## 8. 口头报告分析（oral_model_alignment）

`utils/oral_model_alignment.py` 提供两个基础 oral -> hypothesis 映射类：

- `Oral_region_mapping`：基于口头报告区域 `(A, b)` 与假设区域的重叠分数比较
- `Oral_center_mapping`：基于口头报告中心点与假设原型中心的距离比较

这两个映射类供 `oral_mass` 计算和五类 oral/model alignment 方法复用。

## 9. 常见问题

1. 为什么 cluster/oral 图没有生成？  
通常是因为优化阶段没有保存足够日志（例如 `keep_logs` 关闭），或评估时未提供可用 oral 数据文件。

2. `model_struct` 和 `*_opt_cfg` 谁覆盖谁？  
先加载 `engine_config_path`，再叠加优化配置里的 `engine_config` 覆盖项。

3. 每个被试能否用不同 window size？  
可以，在优化配置里使用 `window_size` 列表或 `window_size_overrides`。

## 10. 推荐阅读顺序（代码）

1. `problems/model.py`
2. `inference_engine/bayesian_engine.py`
3. `problems/modules/`（尤其 perception / memory / hypo_transitions）
4. `utils/optimizer_grid.py` 与 `utils/optimizer_amr.py`
5. `eval_amr_results.py` / `eval_grid_results.py`

## 11. Oral mode configuration and error policy

Both `eval_grid_results.py` and `eval_amr_results.py` support dual oral encodings from the learning-data CSV by default:

- `center`: uses `Oral_center_mapping` and center columns (`oral_center`)
- `region`: uses `Oral_region_mapping` and region columns (`oral_A`, `oral_b`)

CLI arguments:

- `--config`: optimization yaml (reads `oral.mode`; oral data defaults to `dataset.learning_data`)
- `--oral-mode {center,region}`: overrides yaml `oral.mode`
- `--oral-data`: optionally overrides the default oral csv path (highest priority)
- `--oral-region-n-samples`: region mode overlap sampling count (overrides `oral.region_n_samples`)

Priority order:

1. CLI (`--oral-mode`, `--oral-data`)
2. YAML (`oral.mode`) plus `dataset.learning_data`
3. Missing required values -> raise error

Region speed tip:

- default `region_n_samples` is reduced for evaluation speed.
- for smoke tests, keep it small (for example 200-1000).

Examples:

```bash
python -m src.Bayesian_state.eval_grid_results \
  --input-dir results/state-based-grid-result/pmh/cond1 \
  --config configs/grid_opt_cfg/pmh_cond1.yaml
```

```bash
python -m src.Bayesian_state.eval_grid_results \
  --input-dir results/state-based-grid-result/pmh/cond1 \
  --config configs/grid_opt_cfg/pmh_cond1.yaml \
  --oral-mode region
```

```bash
python -m src.Bayesian_state.eval_amr_results \
  --input-dir results/state-based-AMR-result/pmh/cond1 \
  --config configs/amr_opt_cfg/pmh_cond1.yaml \
  --oral-data data/processed/Task2_processed.csv
```

Error policy:

- Oral mode/data mismatch raises explicit error (with missing columns).
- Missing oral file raises `FileNotFoundError`.
- Empty oral hit results raise `RuntimeError`.
- Grid/AMR evaluation no longer silently skips oral plotting when oral is requested.

## Prediction Mode (Dual Path)

`Bayesian_state` now supports configurable prediction paths for accuracy metrics:

- `prediction_mode: posterior_t_minus_1`
- `prediction_mode: prior_t`
- `prediction_mode: both`

Optional selector for optimization target:

- `selection_prediction_mode: posterior_t_minus_1 | prior_t`

Rules:

- If `prediction_mode` is `posterior_t_minus_1` or `prior_t`, `selection_prediction_mode` must be the same mode.
- If `prediction_mode` is `both`, both metric paths are computed in one run and stored side-by-side.

Evaluation scripts can choose which path to read:

- `python -m src.Bayesian_state.eval_grid_results --input-dir <dir> --eval-prediction-mode prior_t`
- `python -m src.Bayesian_state.eval_amr_results --input-dir <dir> --eval-prediction-mode posterior_t_minus_1`

### Result Schema Notes (`subjects/subject_<id>.json`)

Important fields:

- `prediction_mode`
- `selection_prediction_mode`
- `available_prediction_modes`
- `metrics_by_mode`

Example shape:

```json
{
  "prediction_mode": "both",
  "selection_prediction_mode": "posterior_t_minus_1",
  "available_prediction_modes": ["posterior_t_minus_1", "prior_t"],
  "metrics_by_mode": {
    "posterior_t_minus_1": {
      "mean_error": 0.123,
      "sliding_true_acc": [...],
      "sliding_pred_acc": [...],
      "sliding_pred_acc_std": [...],
      "pred_acc": [...],
      "true_acc": [...]
    },
    "prior_t": {
      "mean_error": 0.118,
      "sliding_true_acc": [...],
      "sliding_pred_acc": [...],
      "sliding_pred_acc_std": [...],
      "pred_acc": [...],
      "true_acc": [...]
    }
  }
}
```

`metrics` / `best_metrics` are no longer emitted in the new schema.

## Likelihood Distance Mode

`LikelihoodModule` supports a single explicit config key:

- `modules.likelihood_mod.kwargs.distance_mode: prototype | boundary`

This mode is used consistently across:

- trial likelihood updates in inference
- prediction metric computation in optimization/evaluation
- category assignment logic inside `BetaModule`

Note: `oral_model_alignment` center/region analyses are standalone analysis paths and are not controlled by `distance_mode`.

## Optimization Loss Metric (Required)

`run_grid_optimization.py` and `run_amr_optimization.py` now require a config key:

- `loss_metric: accuracy_curve_mae | accuracy_curve_mse | accuracy_curve_family_mse | accuracy_curve_berhu | accuracy_brier | accuracy_family_brier | accuracy_nll | choice_brier | choice_nll | wrong_choice_nll | conditional_wrong_choice_nll`

No backward-compatible fallback is applied. Missing or unsupported `loss_metric`
raises an explicit error.

Definitions:

Accuracy-based losses:

- `accuracy_curve_mae`: sliding-window mean absolute error on accuracy curves.
- `accuracy_curve_mse`: sliding-window mean squared error on accuracy curves.
- `accuracy_curve_family_mse`: sliding-window mean squared error on
  family-level accuracy curves. For 4-category tasks, categories `1/2` and
  `3/4` are grouped into two families.
- `accuracy_curve_berhu`: reverse Huber on sliding-window residuals \(r = \hat{a} - a\):
  - `|r|` when `|r| <= loss_delta`
  - `(r^2 + loss_delta^2) / (2 * loss_delta)` when `|r| > loss_delta`
- `accuracy_brier`: trial-level correctness Brier score, i.e.
  `mean_t (p_t(category_t) - 1[feedback_t = 1])^2`.
- `accuracy_family_brier`: trial-level family-correctness Brier score, i.e.
  `mean_t (sum_{k in family(category_t)} p_t(k) - 1[choice_t in family(category_t)])^2`.
- `accuracy_nll`: trial-level binary log loss for correctness, i.e.
  `mean_t -y_t log p_t(category_t) - (1-y_t) log (1-p_t(category_t))`,
  where `y_t = 1[feedback_t = 1]`.

Choice-based losses:

- `choice_brier`: trial-level Brier score against the observed subject choice,
  i.e. `mean_t sum_k (p_t(k) - 1[k = choice_t])^2`.
- `choice_nll`: trial-level negative log-likelihood against the observed subject
  choice, i.e. `mean_t -log p_t(choice_t)`.
- `wrong_choice_nll`: `choice_nll` restricted to incorrect trials
  (`choice_t != category_t`).
- `conditional_wrong_choice_nll`: on incorrect trials, the negative log
  likelihood of the observed wrong choice conditioned on the model not choosing
  the true category, i.e.
  `mean_t -log (p_t(choice_t) / (1 - p_t(category_t)))`.

When `loss_metric: accuracy_curve_berhu`, config must also include:

- `loss_delta` (float, `> 0`)

Example:

```yaml
loss_metric: accuracy_curve_berhu
loss_delta: 0.05
window_size: 16
```

`selection_prediction_mode` is still respected and controls whether hypothesis
weights come from `posterior_t_minus_1` or `prior_t`.
