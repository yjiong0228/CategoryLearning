# Model 0923 full-v3：实现与运行

`0923-full-v3` 实现2026年10月12日确认的行为模型，共13项被试参数。算法位于共享核心 `src/Bayesian_state/`；完整公式见 [模型规范](model_0923.pdf)。正式拟合、参数恢复及与0826的预测比较尚未完成。

## 运行版本

三条件引擎配置为 `configs/exp123/model_struct/model_0923_cond{1,2,3}_full_v3.yaml`。
每份配置显式声明 `provenance.specification_version: 0923-full-v3`。
原 `model_0923_cond{1,2,3}_full.yaml` 保留 `0923-full-v2`，用于复现原来的精度更新、联合遗忘和单规则作答。
版本校验会拒绝把新版精度、候选管理或记忆模式与旧版标识混用。

新版包含以下行为变化：

| 环节 | 实现行为 |
|---|---|
| 候选管理 | 全部旧候选按信念独立提出退出；全退时恢复最高信念规则。搜索从本题未活跃的规则中引入一条；容量已满时先删除最低信念旧规则，再进行一次信念分配。 |
| 选择读出 | 各规则按信念及上一主导规则的惯性加成形成连续权重，再汇总类别预测。权重最大者成为下一题的惯性对象。 |
| 精度更新 | 全部活跃规则使用反馈对数似然梯度和同一个学习率，更新后限制在 `[0.1, 25]`；结果从下一题起影响预测。 |
| 条件1/2记忆 | `forgetting_mode: exact_power` 按规则信念的保持率次幂乘以反馈似然，再归一化。计算保留浮点数可表示的极小信念和反馈概率。 |
| 条件3记忆 | `forgetting_mode: factorized` 分别遗忘并归一化规则信念、各规则下的配对信念，再联合吸收本题反馈。精度梯度使用同一份反馈前配对分布。 |

二值反馈的错误概率通过其余类别概率求和计算。记忆与精度校准使用同一个反馈事件概率。条件3的新规则继承退出和容量处理之前的整体配对信念，保留规则延续自身的条件配对信念。

## 参数配置

以下路径均位于引擎 YAML 的 `modules` 下。通过运行配置的 `fixed_hyperparams` 或通用参数搜索覆盖时，完整键名加前缀 `engine.modules.`。

| 参数 | 路径 | 数学范围 |
|---|---|---|
| 基础搜索概率 | `hypo_transitions_mod.kwargs.search_controller.baseline_probability` | `(0, 1)` |
| 失败推动搜索的增益 | `hypo_transitions_mod.kwargs.search_controller.error_gain` | `[0, ∞)` |
| 失败状态保持率 | `hypo_transitions_mod.kwargs.search_controller.failure_decay` | `[0, 1)` |
| 基础全局比例 | `hypo_transitions_mod.kwargs.search_controller.global_search` | `[0, 1]` |
| 失败对范围的增益 | `hypo_transitions_mod.kwargs.search_controller.global_search_failure_gain` | `[0, 1]` |
| 候选容量 | `hypo_transitions_mod.kwargs.max_capacity` | 整数，至少2，小于目录大小 |
| 退出率 | `hypo_transitions_mod.kwargs.exit_rate` | `[0, 1]` |
| 局部结构来源倾向 | `hypo_transitions_mod.kwargs.local_structure_weight` | `[0, 1]` |
| 信念迁移尺度 | `hypo_transitions_mod.kwargs.transfer_scale` | `(0, ∞)` |
| 主导规则的作答惯性 | `hypo_transitions_mod.kwargs.execution_inertia` | `[0, ∞)` |
| 证据保持率 | `memory_mod.kwargs.gamma` | `[0, 1]` |
| 初始精度 | `beta_mod.kwargs.beta_init` | `[0.1, 25]` |
| 精度学习率 | `beta_mod.kwargs.learning_rate` | `[0, 1]` |

例值用于实现检查。新增学习率 `0.1` 尚待校准。拟合前还需确定自由与固定参数、有限搜索范围、训练与评价边界，以及粒子数和重复次数。知觉参数由独立测量提供。口述的现有 `rule_content_v1` 读出可使用连续规则权重；显著度选择补充部位的新设计及批量口述比较尚待接入。

## 小规模运行

从仓库根目录运行：

```bash
python -m src.Bayesian_state.run_simulation \
  --config configs/exp123/simulation_cfg/model0923_cond1_full_v3_smoke.yaml
```

条件2、3的文件名分别为 `model0923_cond2_full_v3_smoke.yaml`、`model0923_cond3_full_v3_smoke.yaml`。这些配置各运行一名被试、最多32试次、8粒子、一次重复和一个进程。再次运行前为 `output_dir` 选择新的目录；入口会拒绝覆盖已有被试结果。

## 状态输出与验证边界

逐试次记录包含反馈前的 `choice_weights`、`dominant_hypothesis` 和 `effective_participation`。保存轨迹后的预测回放使用当时记录的作答权重。

粒子滤波输出 `choice_weights`、`dominant_probability` 及各自的 `filtered_` 版本。前者按本题选择进入滤波之前的粒子权重汇总；后者用本题选择更新后的粒子权重汇总同一份反馈前状态。`effective_participation` 汇总各粒子的有效参与数量 `1 / sum(w²)`。运行结果保存这些数组和计算时序说明。

定向检查覆盖梯度的有限差分、精度方向与边界、记忆更新与极小概率、候选退出与容量、单次信念迁移、连续读出、快照恢复、反馈时序、固定种子复现和旧版数值基准。三条件短序列用于检查真实数据入口和输出格式。这些检查验证实现及数据通路；学习速度、参数可恢复性和行为预测效果需要后续拟合与模拟评价。

2026年10月12日验证了469项不同测试用例，均通过；其中期刊回归逐项比较232个冻结数值数组。实际环境为 Python 3.11.10、NumPy 1.26.4、SciPy 1.13.1、PyYAML 6.0.2、pytest 7.4.4。可按以下命令复核相同测试范围：

```bash
python -m pytest -q tests/bayesian_state/test_model_0923*.py \
  tests/bayesian_state/test_beta_feedback_chance.py \
  tests/bayesian_state/test_condition3_pairing_memory.py \
  tests/bayesian_state/test_model_0826_condition3_controls.py \
  tests/bayesian_state/test_model_0806_framework.py \
  tests/bayesian_state/test_model_0818_observed_fit.py \
  tests/bayesian_state/test_condition2_particle_filter.py \
  tests/bayesian_state/test_condition3_particle_filter.py \
  CategoryLearning_codes/Bayesian_model/tests/test_migration.py
```

真实数据入口检查使用S122、S206、S314各8试次、4粒子、一次重复和一个进程，输出至临时目录。条件1/2保留100,000个关系参考样本，条件3为短检查使用512个样本，并在临时配置与来源记录中同步这一设置。三条件的类别概率、连续权重、主导概率及条件3配对概率均有限、非负且归一化。8试次不足以估计默认8阶历史统计量，旧统计汇总会对不可估计值发出分位数警告；本次验收使用选择概率、NLL及状态不变量。
