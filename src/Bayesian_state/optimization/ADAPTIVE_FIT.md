# Model 0826 默认自适应拟合

参数边界的定向完整序列检查见 [范围校准协议](BOUNDARY_CALIBRATION.md)。
该试点单独声明扩展取值，不自动改写本入口的默认支持范围。

入口支持 exp123 三个 condition 的观察数据 PMH 选择拟合。认知模型、作答前预测和
先平均概率再取 log 的定义不变；搜索策略、种子、数值预算会改变参数估计，不能称为
逐结果等价的工程提速。旧 Grid/CD、恢复设计和历史配置仍走原入口。exp4/exp5、条件3
消融、最终状态精度和参数恢复不因本次软件接入而获验证。

## 运行

```bash
# 验证输入与计划，不运行 PF。
python -m src.Bayesian_state.run_model_0826_fit --dry-run
# 一名被试、32 试次、单进程小检查；使用新的输出目录。
python -m src.Bayesian_state.run_model_0826_fit --subjects 103 --smoke --output-dir results/model_0826/adaptive_smoke_new
# 正式预算调用示例；本次接入没有执行这项全体拟合。
python -m src.Bayesian_state.run_model_0826_fit --conditions 1 2 3 --output-dir results/model_0826/adaptive_fit_new
```

默认配置是 `configs/exp123/specific_models/model_0826_adaptive_fit.yaml`。统一 CLI
读取其 `backend: model0826_adaptive`，无需另指定 backend：

```bash
python -m src.Bayesian_state.optimization.cli --config configs/exp123/specific_models/model_0826_adaptive_fit.yaml --subjects 103 --dry-run
```

不指定 subjects/conditions 时选择数据内全部三个条件的被试。指定二者时必须相符；
指定了不在条件内的被试会报错。入口保留完整序列递推和默认首试次不计分，不接受隐式
试次过滤、留出后缀或数据重编码选项。需要留出拟合时另定义对应协议，不能混用全序列参数。

## 默认过程和预算

- 45 个分散起点；R32×B4 发现候选；前 8 个、最多 4 个分散候选和最多 2 个随机抽查
  候选，用 R128×B16 引导后续搜索。引导分数与发现分数不混排。
- 每轮从最多两个较好候选各提议 48 个局部/整块/跨块点（配额 25%/25%/50%），加18个
  分散点。去重后不补凑工作量。最多六轮，连续两轮改善不超过平均 NLL 0.0001 记为平台。
- 随后两轮联合挑战，第一轮另提议90个分散点。人为边界附近还有方向和联合提案，直接
  进入引导评分，不被低预算筛掉。挑战继续改善时再搜索，最多三个完整循环。
- 代表和候选库在数值复核前冻结。独立决策从 R128×B16 开始，未明确时升至 R256×B32；
  再以独立种子 R256×B32 审查。相对候选库最佳者的平均 NLL 损失上界≤0.005才通过。
- alpha=0.05，决策按预算档×最大循环数分配，审查按最大循环数分配。这是有限粒子下、
  完整 PF 种子 bootstrap 的近似数值诊断，不是参数区间或严格全局错误率保证。
- 审查发现代表较差时，用已见结果引导下一轮，下一轮审查换新种子；不会在同一批结果上
  换赢家后又宣称独立通过。无法区分的结果明确保留 unresolved。

默认值是有上限的工作设置。联合搜索、随机抽查和有限候选库的检查都不能认证全局最优；
新的组合策略尚无全体被试的速度/质量基准，不能用 smoke 耗时推算正式拟合耗时。

## 边界与范围

`boundary.extensions` 默认空，保留原暂定范围。可明确声明额外取值，例如：

```yaml
boundary:
  extensions:
    gamma: [0.985]
  near_tolerance: 0.005
  max_candidates: 4
  proposals_per_elite: 16
```

扩展值作为所有被试共享的允许支持，进入分散、方向和联合搜索；不回写冻结的恢复 YAML。
这会改变估计范围，应在检查结果前记录科学理由。接口只允许原参数、原理论定义域及合法
workspace组合，不允许修改固定 beta 上下限或偷偷加入零更新率消融。扩容后各全局提案
数量必须至少覆盖所有 workspace组合。原始恢复配置仍经原冻结支持校验；扩展另行校验。

人工上限即使扩大后仍是人工上限；碰到新上限仍需复核。自然零边界和二元chi不作为扩大
范围的理由。边界提案检查配置限定数量的近优点；最终标记检查代表及独立审查中接近的
备选。即使有限候选库的评分通过，人为边缘仍会产生 `boundary_review_required`，不会
自动被判为已完成。长期贴边也可能涉及不可辨识性或模型限制，需要恢复及科学诊断。

## 状态、产物和续跑

`provisional_stop_within_tested_scope` 要求平台、挑战、独立评分及边界检查全部通过。
否则为 `unresolved`，列出 `search_budget_without_plateau`、`challenge_improved`、
`selection_precision_unresolved`、`independent_audit_unresolved`、`boundary_review_required`
中相应原因。没有通过的参数可以检查，但不能被称为最终稳定估计。所有结果都明确标注
`state_precision: not_checked` 和 `parameter_recovery: not_checked`。

| 路径 | 内容 |
|---|---|
| `manifest.json`、`plan.json` | 数据、代码、引擎、资源指纹、环境、被试与预算 |
| `effective_parameter_support.json` | 本次完整支持，含声明的扩展 |
| `batches/`、`progress/` | 冻结提案来源、引导分数、平台、挑战、边界及选参记录 |
| `cache/` | 每个被试×参数×粒子数×种子的概率、实际选择、掩码；完整文件原子发布 |
| `subjects/<id>/fit_result.json` | 代表参数、候选库、近优点、独立审查、边界和未解决项 |
| `fit_results.json` | 全体汇总；不是旧 Hyper-CD 的 `best_hyperparams.json` |

这一步不自动运行状态输出。正式并行上限128，按可用 CPU 和未完成任务数取小，每个
worker一个数值线程，仅一层进程；批次跨被试×候选×种子。smoke始终单被试单进程，
结果含 `smoke_only: true`。程序不为占满CPU而增加无必要候选或PF重复。

中断后以完全相同参数加 `--resume`。已完整保存的seed文件复用，尚未保存的任务重算；
完整批次的缓存校验和也会核对。配置、被试、代码、数据或数值环境变化会拒绝续跑。同一
目录的并发写入受文件锁保护。研究输出不覆盖，损坏的缓存不会被静默“修复”。

旧试点公共提案/诊断函数已迁入 `search/adaptive_proposals.py` 和
`diagnostics/decision_precision.py`，旧导入保持兼容。源码变化意味着历史试点的严格
指纹续跑检查会拒绝当前版本；旧结果只读保留，可检出当时提交复现，不能改旧哈希强行续跑。

软件验证见 `results/model_0826/adaptive_default_integration_20260919/README.md`；
通俗说明与试点证据见 `src/Bayesian_state/docs/model_architecture/model_0826_plus.pdf`。
