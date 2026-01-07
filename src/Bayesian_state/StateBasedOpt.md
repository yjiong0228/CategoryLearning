# State-Based Model Optimization & Evaluation Pipeline
## 流程概览

1. **配置 (Configuration)**: 修改 YAML 配置文件，设置模型参数、优化范围和输出路径。
2. **优化 (Optimization)**: 运行 `run_amr_optimization.py`，根据配置并行搜索最优参数。
3. **评估 (Evaluation)**: 运行 `eval_amr_results.py`，聚合优化结果并绘制各类分析图表。

---

## 1. 运行优化 (Optimization)

使用 `src.Bayesian_state.run_amr_optimization` 模块进行参数搜索。

### 命令行使用

```bash
python -m src.Bayesian_state.run_amr_optimization --opt-config <path_to_config.yaml>
```

### 关键配置 (YAML)

为了支持后续的高级画图功能（如 Cluster Amount, Oral comparison），**必须**在 YAML 配置文件中设置 `keep_logs: true`。这会保存每一步的后验概率和策略，虽然会增加结果文件大小，但是生成 Grid 图和口头报告对比图所必需的。

**示例配置文件结构 (`configs/amr_opt_cfg/example.yaml`):**

```yaml
# 1. 基础输出设置
output_dir: "results/state-based-AMR-result/experiment_name"
keep_logs: true   # <--- 必须开启此项才能画 Cluster/Oral 图

# 2. 并行设置
n_jobs: 8         # 同时并行的被试数

# 3. AMR 优化参数
amr_kwargs:
  initial_points: 20
  # ... 其他 AMR 设置 ...

# 4. 被试与参数网格
subjects: ["1", "2", "3"]
param_grid:
  alpha: [0.1, 5.0]
  # ...

# 5. 模型配置 (Window size 等)
engine_config:
  model_name: "Bayesian_state"
  window_size: 15  # 或列表 [10, 15, ...]
```

---

## 2. 运行评估与画图 (Evaluate & Plot)

使用 `src.Bayesian_state.eval_amr_results` 模块聚合结果并生成图像。

### 命令行参数

- `--input-dir`: **(必须)** 包含 `subject_*.json` 结果文件的目录 (即优化步骤中的 output_dir)。
- `--aggregate-output`: (可选) 聚合后的 JSON 保存路径。
- `--plot-accuracy`: (可选) Accuracy 对比图保存路径。
- `--plot-cluster`: (可选) Cluster Amount 对比图保存路径 (需 `keep_logs: true`)。
- `--plot-oral`: (可选) Oral Reporting vs Model K 对比图保存路径 (需 `keep_logs: true`)。
- `--oral-data`: (可选) 包含真实口头报告数据的 CSV 路径 (默认为 `data/processed/Task2_processed.csv`)。

e.g.
```bash
python -m src.Bayesian_state.eval_amr_results \
    --input-dir results/state-based-AMR-result/pmh/cond1 \
    --plot-accuracy results/state-based-AMR-result/pmh/cond1/accuracy.png \
    --plot-cluster results/state-based-AMR-result/pmh/cond1/cluster.png \
    #--plot-oral results/state-based-AMR-result/pmh/cond1/oral.png \
    #--oral-data data/processed/Task2_processed.csv
```

### 完整运行示例

假设优化结果保存在 `results/state-based-AMR-result/pmh/cond1`，且需要生成所有图表：

```bash
python -m src.Bayesian_state.eval_amr_results \
    --input-dir results/state-based-AMR-result/pmh/cond1 \
    --aggregate-output results/state-based-AMR-result/pmh/cond1/all_subjects.json \
    --plot-accuracy results/state-based-AMR-result/pmh/cond1/accuracy.png \
    --plot-cluster results/state-based-AMR-result/pmh/cond1/cluster_amount.png \
    --plot-oral results/state-based-AMR-result/pmh/cond1/oral_comparison.png \
    --oral-data data/processed/Task2_processed.csv
```

---

## 常见问题 (FAQ)

**Q: 为什么运行 evaluate 时提示 "Cluster plot skipped" 或 "Oral plot skipped"?**
A: 检查你的优化结果 JSON 文件。如果其中没有 `best_step_results` 字段，说明在通过 `run_amr_optimization` 运行时，YAML 配置里的 `keep_logs` 被设为了 `false` (或未设置)。请修改 YAML 为 `keep_logs: true` 并重新运行优化。

**Q: 如何让每个被试使用不同的 Window Size?**
A: 在 YAML 的 `engine_config` -> `window_size` 中提供一个列表，长度需与 `subjects` 列表长度一致。例如 `window_size: [10, 15, 20]` 分别对应前三个被试。

**Q: Oral Data 的 CSV 格式有什么要求?**
A: 默认通过 `Oral_to_coordinate` 工具处理 `Task2_processed.csv`，该文件应包含被试的真实分类行为和口头报告字段。
