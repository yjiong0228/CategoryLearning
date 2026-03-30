# Bayesian_state 读者手册

`Bayesian_state` 是本仓库当前主用的 state-based 建模管线。  
这份文档把 `state-based_model.md` 和 `StateBasedOpt.md` 的核心内容合并在一起，目标是让你“一份文档看懂模型如何运作 + 如何跑优化评估”。

## 1. 你先要知道的三件事

1. 这个模型是“模块化贝叶斯引擎”：每一步推断按 `agenda` 顺序执行模块。  
2. 配置分两层：  
   - `configs/model_struct/*.yaml` 定义模型结构（有哪些模块、执行顺序、模块默认参数）。  
   - `configs/grid_opt_cfg/*.yaml` / `configs/amr_opt_cfg/*.yaml` 定义优化任务（被试、参数搜索范围、输出路径等）。  
3. 每个被试优化后会生成 `subject_<id>.json`，评估脚本再聚合并出图。

## 2. 模块结构总览

目录里最关键的文件：

- `inference_engine/bayesian_engine.py`：核心引擎 `BaseEngine`
- `problems/model.py`：`StateModel` 封装，负责创建 partition 和 engine
- `problems/modules/`：各功能模块（perception / likelihood / memory / beta / hypo transition）
- `utils/state_grid_optimizer.py`：网格搜索优化器
- `utils/state_amr_optimizer.py`：AMR 优化器
- `utils/oral_process.py`：口头报告分析（region/center 两种）

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
- `subjects` 或 `subject_range`
- `keep_logs`: 是否保存 step 级日志（画 cluster/oral 图通常需要）

## 5. 运行流程

### 5.1 Grid 优化

```bash
python -m src.Bayesian_state.run_grid_optimization \
  --config configs/grid_opt_cfg/pmh_cond1.yaml
```

### 5.2 AMR 优化

```bash
python -m src.Bayesian_state.run_amr_optimization \
  --config configs/amr_opt_cfg/pmh_cond1.yaml
```

`run_amr_optimization` 兼容旧参数名 `--opt-config`，但建议统一用 `--config`。

## 6. 评估与作图

### 6.1 AMR 结果评估

```bash
python -m src.Bayesian_state.eval_amr_results \
  --input-dir results/state-based-AMR-result/pmh/cond1 \
  --aggregate-output results/state-based-AMR-result/pmh/cond1/all_subjects.json \
  --plot-accuracy results/state-based-AMR-result/pmh/cond1/accuracy.png \
  --plot-cluster results/state-based-AMR-result/pmh/cond1/cluster_amount.png \
  --plot-oral results/state-based-AMR-result/pmh/cond1/oral_vs_model.png \
  --oral-data data/processed/Task2_processed.csv
```

### 6.2 Grid 结果评估

`eval_grid_results.py` 的参数风格与 AMR 评估基本一致（同样支持 `--input-dir`、`--plot-accuracy`、`--plot-cluster`、`--plot-oral` 等）。

## 7. 结果文件约定

优化输出目录下通常有：

- `subject_<id>.json`：单被试最优结果和（可选）step日志
- `all_subjects.json`：聚合结果
- `accuracy.png`：准确率对比图
- `cluster_amount.png`：策略/簇变化图（需要 step 日志）
- `oral_vs_model.png`：口头报告与模型假设对比图

## 8. 口头报告分析（oral_process）

`utils/oral_process.py` 提供两条路径：

- `Oral_region_analysis`：基于口头报告区域 `(A, b)` 与假设区域的重叠分数比较
- `Oral_center_analysis`：基于口头报告中心点与假设原型中心的距离比较

两条路径都输出被试级 `hits` 和 `rolling_hits`，用于和模型过程对比。

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
4. `utils/state_grid_optimizer.py` 与 `utils/state_amr_optimizer.py`
5. `eval_amr_results.py` / `eval_grid_results.py`
