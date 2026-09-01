# Hyper-CD 2.0 设计规范

## 1. 目标与边界

Hyper-CD 2.0 是现有 `HyperCDOptimizer` 的向后兼容升级，用于在粒子滤波产生 Monte Carlo 噪声的条件下，对离散、连续网格、精确零点和联合参数块进行可恢复、可中断、可审计的坐标搜索。它不改变 Model0826 的认知方程，不把不同超参数的粒子混在同一个粒子群中，也不把优化过程解释为被试的认知过程。

本子项目只改变搜索控制与搜索产物。粒子滤波仍由 `inference/backends/particle_filter.py` 提供，候选评分仍使用公共 simulation/metrics 契约。

## 2. 当前实现中必须修复的问题

1. `cd.min_delta` 已被解析但没有参与候选接受判定。
2. 当前只能从 coarse 结果进入 fine，不能在同一个 stage 的中断位置继续。
3. fine stage 缺少明确的“coarse shortlist 作为 fine 起点”契约。
4. 搜索阶段的赢家没有统一的独立随机种子高精度重评分入口。
5. compact JSONL 恢复后缺少完整 subject metrics，不能把缓存候选直接冒充最终高精度结果。
6. 当前会在非 resume 运行时删除既有搜索 JSONL；恢复实验必须改为显式的新运行或显式 resume，不能静默覆盖。

## 3. 配置契约

现有配置在没有新增字段时保持原行为。Hyper-CD 2.0 配置显式设置：

```yaml
search_schema_version: 2
cd:
  n_restarts: 4
  max_outer_iters: 6
  patience: 2
  min_delta: 0.0001
  coordinate_order: fixed
  parallel_budget: 64
  resume_mode: explicit
  checkpoint_every_coordinate: true
refine_policy:
  top_k: 4
  fine_initialization: coarse_shortlist
final_rescore:
  enabled: true
  shortlist_size: 4
  seed_family: model0826_recovery_final_rescore_v1
  simulation_overrides: {}
```

规则如下：

- `resume_mode: explicit` 要求调用者传入 resume；若输出目录已有搜索产物而没有 resume，立即报错，不删除文件。
- `min_delta` 作用于第一目标，单位与第一目标相同。候选必须先按 `objective_order` 优于当前点，且 `current_primary - candidate_primary >= min_delta` 才能移动。`min_delta=0` 保持现有严格改善语义。
- `fine_initialization: coarse_shortlist` 取 coarse 阶段互不重复的前 `top_k` 个点；标量数值坐标投影到 fine 显式支持中的相同值或最近值，并以较小数值打破等距并列；mapping-valued 坐标必须在 fine 支持中存在完全相同的候选，否则报错。fine restart 数等于投影去重后的起点数，并覆盖全局 `cd.n_restarts`。
- fine 候选支持必须由配置明确给出；优化器不在运行时发明边界或把精确零点插值成正数。
- `final_rescore` 使用独立于 coarse/fine 的 stage seed family，以候选间共同随机数对 shortlist 重新评分。最终赢家只能来自这次重评分，而不能直接沿用低预算搜索赢家。

## 4. 搜索状态与续跑

每完成一个坐标，原子写入 `search_checkpoint.json`。checkpoint 至少包含：

- 配置、基础 simulation 配置和参数空间的内容哈希；
- stage、restart、outer iteration、当前坐标位置；
- 当前点、restart 局部最优点和 objective；
- 已完成 restart 摘要、无改善轮数和停止原因；
- 下一个 combination index；
- Python `random.Random` 状态的 JSON 安全表示；
- `all_combinations.jsonl` 与 `coordinate_trace.jsonl` 的已确认记录数。

resume 前逐项核对哈希和 subject 列表。JSONL 只允许最后一行因进程中断而不完整；这种情况下截去不完整尾行并记录 repair，任何中间损坏均报错。缓存键使用规范化、排序后的完整 hyperparameter JSON。恢复时从 JSONL 重建缓存，但最终入选点必须重新执行 `final_rescore`，因此不依赖 compact 记录中缺失的 subject metrics。

正常完成后把 checkpoint 状态标为 `complete`，不删除 checkpoint。这样运行结束和意外中断可以被区分。

## 5. 坐标、起点与停止规则

- mapping-valued coordinate 继续作为不可拆开的联合块，例如 `(M, chi)`、`(E_C, delta_E)` 或 `(g_0, c_G)`。
- 精确零点继续是候选集合中的独立元素。
- 每个 restart 先评分其起点，再完整扫描全部坐标。
- 一轮中只接受达到 `min_delta` 的移动；接受后后续坐标从新点继续。
- 连续 `patience` 个完整 outer sweep 没有移动才停止。
- 所有 restart 都运行到停止条件，不因某个 restart 的好结果提前取消其他 restart。
- 相同参数点在同一 stage 只计算一次；不同 stage 的粒子预算或 seed family 不共用评分缓存。

## 6. 输出

每个搜索目录保留：

```text
all_combinations.jsonl
coordinate_trace.jsonl
search_checkpoint.json
restart_summary.json
stage_summary.json
final_rescore.jsonl
best_hyperparams.json
```

trace 对每个坐标记录候选数、新计算数、缓存命中、被 `min_delta` 拒绝的改善数、起止 objective、移动结果和并行预算。`best_hyperparams.json` 明确区分 `search_best` 与 `final_rescore_best`，并保存所用 R、B、seed family、trial mask 和配置哈希。

## 7. 验证要求

实现采用测试驱动方式，至少证明：

1. `min_delta` 会拒绝小改善、接受达到阈值的改善，零阈值保持兼容。
2. mapping-valued 离散块和精确零点不会被拆分或改变。
3. 多 sweep 能走出一轮扫描无法到达的组合。
4. 中断后 resume 与不中断运行得到相同的候选序列、最终点和 trace。
5. 配置或 subject 哈希不一致时拒绝 resume。
6. fine 起点确实来自 coarse shortlist，并稳定投影到 fine 支持。
7. final rescore 使用独立 seed family，且最终选择来自 seed-averaged trial probabilities 后计算的 NLL。
8. 既有 Hyper-CD 测试继续通过。

## 8. 完成标准

上述契约均由自动化测试覆盖；一个全试次、低粒子预算的 Model0826 smoke run 能够中断、续跑并生成完整产物；README 与 CLI help 已同步。满足这些条件后，Hyper-CD 2.0 才可作为 Model0826 恢复管线的拟合器。
