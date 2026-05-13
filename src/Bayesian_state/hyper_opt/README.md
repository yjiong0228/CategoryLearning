# Hyper Opt（二层超参数优化）

`src/Bayesian_state/hyper_opt`
- 外层：搜索超参数组合（例如 beta 模块参数、window size、hypothesis transition 配置）
- 内层：复用已有 `grid` / `amr` 优化器搜索模型参数（如 `gamma`、`w0`）

优化目标：

`argmin_hyperparams min_inner_params(mean_error)`

每个外层超参数组合先运行一次内层优化，再用该组合下内层最优的 `mean_error` 作为外层评分。

## 目录结构

- `optimizer.py`：外层优化主逻辑
- `cli.py`：命令行入口
- `../run_hyper_optimization.py`：兼容入口（转发到 `hyper_opt.cli`）

## 快速运行

运行示例：

```bash
python -m src.Bayesian_state.run_hyper_optimization \
  --config configs/hyper_opt_cfg/pmh_cond1_hyper_grid_example.yaml
```

可选覆盖参数：

```bash
--subjects 101 102
--subject-range 101 110
--stage coarse|fine|all
```

## 配置说明

- `inner_optimizer`：`grid` 或 `amr`
- `inner_base_config_path`：内层优化基础 YAML（现有 grid/amr 配置）
- `hyperparam_space`：外层搜索空间（必须用户显式定义）
- `stages.coarse.inner_overrides`：coarse 阶段内层预算（如 `n_repeats`、`refit_repeats`、`param_grid`）
- `stages.fine.inner_overrides`：fine 阶段内层预算
- `refine_policy.top_k`：coarse 阶段选前 `k` 个组合进入 fine
- `save_level`：`compact` 或 `full`（默认 `compact`）

## 超参数路径写法

`hyperparam_space` 里的键必须以以下前缀开头：

- `engine.`：注入模型结构配置（engine config）
- `inner.`：注入内层优化配置（inner config）

示例：

```yaml
hyperparam_space:
  engine.modules.beta_mod.kwargs.beta_init:
    values: [1.0, 2.0]
  inner.window_size:
    values: [8, 16]
```

## refine 机制（当前版本）

当前 refine 采用“top-k 直传”：

1. coarse 阶段跑完后按 `aggregated_error` 排序
2. 取 `refine_policy.top_k` 个最优超参数组合
3. fine 阶段直接评估这些组合（不做邻域扩展）

这种方式对“无邻域”的离散策略超参数（例如某些 hypothesis transition 模板）更适用。

## 输出结果

每次运行会在 `output_dir` 生成：

- `all_trials.jsonl`：
  - `compact`：仅保存阶段、超参数、聚合误差、seed（推荐日常运行）
  - `full`：额外保存 `subject_metrics` 明细
- `stage_summary.json`：每阶段摘要与 top trials
- `best_hyperparams.json`：
  - `compact`：保存最终最佳超参数与聚合指标
  - `full`：额外保存最佳 trial 的 `subject_metrics`

## 建议

- 先用小被试集 + 小预算做 smoke test，再放大到完整优化。
- `selection_metric` 在 v1 固定为 `min_inner_mean_error`。
