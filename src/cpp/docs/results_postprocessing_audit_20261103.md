> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# Model 1031 C++ 结果与画图后处理适配审计

审计日期：2026-11-03。审计对象为 `results/local_cond123_20261103_121458_843`（下文称"交付批次"）
与当时的 C++、Python 两份 Model 1031 实现。本轮只读、只核验，未修改任何代码，
未在交付批次内写入任何文件。

## 1 结论

合作者提出的两条问题成立，用户关于"画图后处理适配较差"的判断也成立，且缺陷范围大于这两条。
对合作者原话的五项具体主张逐条判定如下。

| 合作者的主张 | 判定 | 依据 |
| --- | --- | --- |
| 结果保存了 Example、Local summary、Rule 三类条件状态 | 基本成立，条目数不对 | 保存的是四类语义轴 `uncommitted / example / generalization / catalogue_rule`，见 `conditional_prior` 等 17 个 `[T,4]` 状态数组 |
| 没有保存 H0/H4/H6 各规则的条件概率 | 成立（对该批次） | 96 份 `evaluation_inputs.npz` 全部不含 `catalogue_*`；96 份 `evaluation_manifest.json` 全部不含 `catalogue_state_keys` |
| 需要真实历史下、每题反馈前的粒子内容状态（模板位置/保留维度、类别信念、规则 ID、精度、混合权重） | 成立，且属结构性缺失 | C++ 中逐内容导出函数 `export_content` 只在自主模拟分支被调用（`src/inference.cpp:98`、`:118`），滤波分支只累加类型均值 |
| 当前包只保存了这些状态的类型均值 | 需要修正 | 另有 `content_path.json`，逐条保存 `anchor`、`mask`、`category_belief`、`rule_id`、`beta`、`belief`、`uid`、`parent_uid`；但它对应自主模拟路径 0，不对应真实历史 |
| 以及一条自主模拟路径 | 成立 | `src/cli.cpp:241` 只对第一个自主重复保留状态（`keep_states && r==0`） |

一句话总结：交付批次的 C++ 输出停在"四类语义均值 + 一条自主路径"的粒度，
而合作者需要的"逐规则条件概率 + 真实历史逐粒子内容状态"在当时的二进制里根本没有生成路径。
当前源码已经补上了逐规则部分，逐粒子部分仍然没有。

## 2 核对方法

1. 直接读取交付批次的全部 `evaluation_inputs.npz`、`evaluation_manifest.json`、
   `png/arrays.npz`、`png/summary.json`、`content_path.json`，统计键名、形状、取值与一致性。
2. 逐行阅读 C++ 的实现：`src/inference.cpp`、`src/cli.cpp`、`src/model.cpp`、
   `src/io.cpp`、`include/gm/*.hpp`。
3. 逐段阅读旧版绘图后处理：`evaluation/plot_results.py`、`evaluation/figures.py`。
4. 用当前 Release 二进制对 Cond1/S101 做一次 16 trial 的 `evaluate --smoke` 到系统临时目录，
   确认当前源码实际能产出哪些数组；再用当前 `plot_results.py` 处理该输出，确认画图层实际消费哪些数组。
5. 对照 Python 实现的保存路径：`src/Bayesian_state/workflows/analysis/model_1031_subject_pilot.py`、
   `src/Bayesian_state/inference/backends/particle_filter.py`、
   `src/Bayesian_state/model/state_model.py`、
   `src/Bayesian_state/workflows/reports/model_1031_subject_pilot_report.py`、
   `src/Bayesian_state/workflows/reports/model_1031_rule_diagnostics.py`。

### 2.1 时间线：批次早于逐规则扩展约 9 小时

| 证据 | 值 |
| --- | --- |
| 批次目录名时间戳 | 2026-11-03 12:14:58 |
| 交付批次 `summary.json` 的 `provenance.cpp.implementation_sha256` | `28c03d9b99…` |
| 交付批次 `png/summary.json` 记录的 `plot_results.py` 摘要 | `f7a57889…`（当前为 `463700830…`） |
| 交付批次记录的 `figures.py` 摘要 | `ea584efd…`（与当前相同，说明绘图模块未变） |
| `src/inference.cpp`、`src/cli.cpp`、`include/gm/inference.hpp` 修改时间 | 2026-11-03 21:40—21:50 |
| `README.md` 逐规则段落的修改时间 | 2026-11-03 22:02 |
| 当前 Release 二进制构建时间 | 2026-11-03 21:50 |

因此交付批次由逐规则扩展之前的源码生成，`png/` 也由同一时期的 `plot_results.py` 生成。
两者都比扩展早约九小时。这解释了为什么合作者拿到的包里连
`summary.json.catalogue_probabilities` 这个字段都不存在：旧绘图脚本还没有这个分支。

## 3 问题一：H0/H4/H6 逐规则条件概率

### 3.1 交付批次的事实

96 名被试（cond1 32 名、cond2 32 名、cond3 32 名）的 `evaluate/subject_*/evaluation_inputs.npz`
只含 57 个数组，Cond3 因配对另加 `conditional_pairing` 与
`autonomous_diagnostic_pairing`、共 59 个；其中状态数组为 17 个 `conditional_*` 与 17 个
`autonomous_diagnostic_*`（Cond3 各多 1 个），形状为 `[T,4]`、`[T]`
或 Cond3 的 `[T,4,3]`。`evaluation_manifest.json` 的 `state_keys` 列出同名 17 项，
不含任何 `catalogue_*` 键。`png/arrays.npz` 为 89 个键（Cond3 为 91），同样没有逐规则数组。

`[T,4]` 的第四列（`catalogue_rule`）是全部目录规则之和：
Cond1 汇总 29 条规则，Cond2/3 各汇总 152 条。以 S101 为例，
`conditional_active_probability` 四列均值分别为 0.0031、0.8443、0.6882、1.4393，
第四列 1.44 表示平均有 1.44 条规则在场，无法分辨是哪一条。
`png/conditional_states.png` 中 "catalogue rule" 一行整段高值，
正是这一折叠的直接后果：规则维度在图上完全不可见。

### 3.2 当前源码已经补上逐规则记录

`include/gm/inference.hpp:5-8` 定义六个稳定数组名；
`src/inference.cpp:36-54` 在逐粒子循环中同时累加语义轴与规则轴；
`src/cli.cpp:150-171` 在 PF 重复间取均值，`src/cli.cpp:306-313` 写出
`conditional_catalogue_*` 与 `autonomous_diagnostic_catalogue_*`；
`src/cli.cpp:320-333` 写出 `catalogue_metadata`（Cond1 为 `H0…H28`，
Cond2/3 为 `H0…H151`，各带 `family` 与 `program`）和 `catalogue_axis`（时点与条件说明）。

本轮 smoke 复现结果（Cond1/S101，16 trial，8 粒子，1 次重复）：

```
conditional_catalogue_prior                          (16, 29) float64
conditional_catalogue_filtered_prior                 (16, 29) float64
conditional_catalogue_posterior                      (16, 29) float64
conditional_catalogue_predictive_execution_probability (16, 29) float64
conditional_catalogue_execution_probability          (16, 29) float64
conditional_catalogue_active_probability             (16, 29) float64
catalogue_rule_ids                                   (29,)    int64
```

`catalogue_metadata[i].label == "H"+str(i)`，`catalogue_rule_ids == arange(29)`，
所以 H0/H4/H6 对应列 0/4/6，与合作者的检索需求一致。
质量守恒已核对：`sum_h conditional_catalogue_prior ≈ conditional_prior[:,3]`（最大偏差 2.8e-17），
`sum_h conditional_catalogue_predictive_execution_probability` 与
`conditional_dominant_probability[:,3]` 完全相同（偏差 0.0），无效试次整行为 NaN。
用当前 `plot_results.py` 处理后，`png/arrays.npz` 从 89 键升到 102 键，
六个逐规则数组与 `catalogue_rule_ids` 被原样保留，`summary.json.catalogue_probabilities.status == "available"`，
`rule_count == 29`。

### 3.3 仍然存在的四个缺口

1. **旧批次不会自动获得新数组。** 逐规则概率是模型运行时的产物，
   无法从语义均值和 `content_path.json` 反推。要得到它必须用新二进制重跑 `evaluate`
   到新目录。`tools/export_evaluation_configs.py` 提供了复用旧拟合参数、只重做 evaluation 的路径，
   避免重新拟合。
2. **画图层不画逐规则曲线。** `evaluation/figures.py:197-224` 只接受 `[T,4]` 矩阵与 `[T]` 轨迹；
   `evaluation/plot_results.py:582-584` 只把 `STATE_KEYS` 里的数组收进 `states`，
   `CATALOGUE_KEYS` 仅用于校验（`:242-324`）和写元数据（`:611`），
   `:648-653` 只生成 `behavior.png`、`conditional_states.png`、`autonomous_states.png` 三张图。
   `README.md:111` 也明确写了"离线画图仍保存这些数组，但不自动增加新图"。
   合作者即使拿到新数组，仍需要自己写绘图代码。
3. **没有数值表。** Python 侧会写出 `rule_windows.csv`、`rule_trials.csv`、
   `rule_catalogue_windows.csv`（`workflows/reports/model_1031_rule_diagnostics.py:111-114`），
   C++ 侧只有 npz 与 json，没有可直接进论文的窗口汇总表。
4. **没有独立数值对照。** `docs/validation_report.md` 的独立参考只覆盖
   emission、梯度、标签更新与 29/152 条规则的类别概率、提议权重；
   逐规则条件概率的 C++ 测试（`tests/test_workspace.cpp:185-256`）只验证稳定 ID、
   与语义第三列的质量守恒、以及开/关日志时预测不变。
   没有与 Python `catalogue_execution_probability` 做同数据同种子的对照。
   这是新增量的验证空白。

## 4 问题二：真实历史下的逐粒子内容状态

### 4.1 C++ 只有聚合路径

`src/inference.cpp:25-67` 的 `record()` 在每题的粒子循环里把
`w*belief`、`w*beta`、`w*choice_weights` 等累加进 `[T,4]` 与 `[T]` 数组后退化为均值，
逐粒子的 `Slot`（`include/gm/model.hpp:25-33`：`kind`、`id`、`mask`、`anchor`、`labels`、
`pairing`、`belief`、`beta`、`uid`）与粒子权重 `weights` 没有写出路径。
`export_content`（`src/model.cpp:244-254`）会把 `anchor`、`mask`、`category_belief`、
`rule_id`、`belief`、`beta`、`pairing` 序列化，但只在两处被调用：
`src/inference.cpp:100` 与 `:120`，条件都是 `keep && simulation`。
滤波分支（真实历史）走的是 `record()`，不产生任何内容级输出。
粒子权重在任何输出里都没有出现。

### 4.2 `content_path.json` 的实际内容（修正合作者的表述）

交付批次每个被试都有一份 `content_path.json`，是 320 条记录的列表。每条记录含
`trial_index`、`choice`、`feedback`、`perceived_stimulus`、`executed_uid`、
`contents_pre`、`contents_post`。每个内容含 `kind`、`uid`、`parent_uid`、`born_trial`、
`belief`、`beta`；模板另含 `anchor`（四维位置）、`mask`（四维布尔保留维度）、
`category_belief`；规则另含 `rule_id`。

时点上也符合"反馈前"的要求：`contents_pre` 在 `prepare_trial` 与 `decide` 之后、
抽取选择之前拍摄，`contents_post` 在反馈更新之后。

所以合作者说"只保存了类型均值"并不准确：内容级字段是有的。
问题在另一处——这份路径来自自主模拟，不是真实历史。
以 S101 为例，路径选择的 320 题中有 73 题与真实选择不同（一致率 77.19%），
反馈也随之不同。用它去和真实被试的口述做逐题比对会在 23% 的题上对错题。
另外它是单粒子（粒子权重恒为 1），没有跨粒子分布；`belief` 字段是单个内容的内容信念，
在单粒子路径下无法还原为分布。

### 4.3 Python 侧的对照：C++ 对齐的是 finalization 级别，不是 pilot 级别

| 能力 | Python 单被试 pilot | Python 12 小时 finalization | C++ 交付批次 | C++ 当前源码 |
| --- | --- | --- | --- | --- |
| 逐规则条件概率 `[T,H]` | 有（`catalogue_execution_probability`，T×29） | 无 | 无 | 有（六个阵列） |
| 16 位特征集合分布 `[T,16]` | 有（`feature_set_probability`） | 无 | 无 | 无 |
| 每部位边缘 `[T,4]` | 有（由 16 位分布与码位点积得到） | 无 | 无 | 无 |
| 口述集合与模型集合的精确一致概率 `[T]` | 有（`feature_exact`） | 无 | 无 | 无 |
| 真实历史逐粒子内容 `[T,P,K,…]` | 有（11 个 `particle_*` 数组） | 无 | 无 | 无 |
| 语义四类曲线 `[T,4]` | 有 | 有 | 有 | 有 |
| 条件类别概率 `[T,C]` | 有 | 有 | 有（含 12 次重复） | 有 |
| 自主多路径 `[S,T]` | 有 | 有 | 有（128 条） | 有 |
| 逐内容自主路径 | 有（`.paths.json` 全量 `representations_pre`） | 部分 | 有（路径 0，json） | 有（路径 0，json） |

Python 的 `particle_*` 数组就是合作者要的东西，见
`workflows/analysis/model_1031_subject_pilot.py:219-223` 与 `:241-243`：

```python
keys = ('weights', 'choice_weights', 'representation_kind', 'representation_uid',
        'representation_parent', 'catalogue_rule', 'template_mask', 'template_anchor',
        'template_category_belief', 'beta', 'perceived_stimulus')
...
saved['particle_' + key] = np.stack([item[key] for item in contents_log])
```

观察者在 `filtered` 相位触发（`inference/backends/particle_filter.py:2043-2044`），
早于 `complete_trial` 的学习更新，因此这批数组正是"本题选择已知、本题反馈未知"的内容状态，
形状为 `[T,P,K]` 或 `[T,P,K,4]`（P 粒子、K 工作区槽）。
Python 的 `particle_filter` 观察载荷还包含 `belief (P,K)`（每槽先验）、`active`、`dominant`、
`pairing`、`search`、`failure`、`rule_choice_probabilities (P,K,C)`，
见 `particle_filter.py:2001-2012`。pilot 只取其中 11 项，`belief` 未取。
smoke 脚本（`workflows/runs/run_model_1031_smoke.py:136-138`）会写全两相位所有字段。

由此可以定位性质：C++ 移植对齐的是 Python **finalization 研究**的输出粒度
（只有四类曲线与配对曲线），而合作者需要的是 Python **pilot** 的粒度。
逐粒子导出在 Python pilot 中已有实现，移植时被省略。

另有两处 Python 侧的事实需要一并说明，避免把移植缺口误当成 Python 的做法：

- Python pilot 的报告层在读取结果时主动丢弃 `particle_*`
  （`workflows/reports/model_1031_subject_pilot_report.py:25-28`，
  注释写明"Large particle contents remain in the original job archive"）。
  Python 自己的图也不用逐粒子数组，而是用 16 位特征集合分布 `feature_set_probability`
  与逐规则 `catalogue_execution_probability` 这两个**聚合**量。
  因此复现 Python 图的语义只需要这两个聚合量，逐粒子立方体是给新分析（例如按粒子分层的区间、
  不同 IoU 定义）用的。
- Python 的口述比较使用**特征集合精确一致**这一指标：
  `exact = np.where(eligible, feature_mean[np.arange(n), codes], np.nan)`
  （`model_1031_subject_pilot_report.py:102`），`summary.json` 里同时标注
  `oral_scope='original_feature_use_only; not_complete_content_or_report_probability'`。
  IoU 机制存在于共享评价器（`evaluation/oral/mapping.py:121-163` 的 `_estimate_overlap_score(metric="iou")`、
  `evaluation/oral/scoring.py:1536-1563` 的 `fuzzy_iou_similarity`），
  但需要逐题假设空间后验与 `oral_A`/`oral_b` 列，Model 1031 没有接入
  （`src/Bayesian_state/model/model_1031.py:34-35` 明确拒绝 `oral_readout`）。

### 4.4 "固定四维点上的类别概率"

C++ 没有任何网格或探针导出命令：`Geometry::distances` 只在模型内部被
`conditional_emission`、`decide_perceived`、`rule_proposal_weights` 调用，CLI 没有对应入口
（`src/cli.cpp:398-482` 只有 `fit`、`evaluate`、`simulate`、`filter`）。

Python 也没有落盘的网格。唯一的四维探针在
`workflows/analysis/model_1031_catalogue_audit.py:33-46`：
用 `qmc.Sobol(4, scramble=True, seed=10312026).random_base2(8)`（256 点）在 β=3 下
计算每条规则的类别概率，仅用于检测"两条规则在所有探针上预测相同"，
逐点概率被丢弃，只保留差异统计。

结论：这一项属于新增需求，Python 侧同样没有落盘。若只要求"每题在**观测到的**刺激上的反馈前类别概率"，
现有输出已足够：`conditional_category_probabilities [T,C]` 与
`conditional_category_probability_repeats [R,T,C]`（Cond1 为 12×320×2）。

## 5 更深一层的后处理缺陷

以下按对论文作图的影响程度分级。它们共同解释了"适配较差"的整体观感。

### 5.1 P0：直接阻塞论文图

**（1）语义折叠是全局设计，不只是缺一张图。**
`evaluation/figures.py` 的 `plot_states` 只有 7 个 `[T,4]` 热图与 9 条 `[T]` 轨迹；
`evaluation/plot_results.py:218-223` 的 `STATE_KEYS` 固定为四类语义。
凡涉及"哪一条规则""哪一个内容"的图，当前管道都没有输入维度。

**（2）没有跨被试、跨条件的聚合层。**
`evaluation/plot_results.py:664-687` 的 `main` 支持传入上层目录，
但只是对每个被试各生成一份 `png/`（`:675`、`:684`）。交付批次有 96 份被试级输出，
没有合并 CSV、没有条件均值曲线、没有组水平统计图、没有参数分布图。
论文需要的"三条件准确率曲线""参数散点""逐规则组均值"都要另写脚本。
Python 侧对应的组水平函数在 `src/nips_plot.py`（`Fig3_Group`、`Fig4`、`Fig5`），
C++ 侧没有等价物。

**（3）口述数据完全未接入。**
`evaluation/plot_results.py:586-588` 把 `oral` 固定为 `status="not_implemented_1031"`、
`used_for_fit=False`，`raw_reports` 原样透传。
数据其实齐全：`content_path` 之外，`manifest.oral_reports` 逐题保存
`text`、`oral_center`、`oral_A`、`oral_b`、`feature1_use`…`feature4_use`。
也就是说"把口述当作未评分元数据"是当前实现的选择，缺的是编码、对齐、评分与绘图四段代码。
合作者要做的 IoU/集合一致性图，正是这四段中的最后两段。

**（4）没有 16 位特征集合分布。**
这是 Python pilot 图与口述比较的核心量（`model_1031_subject_pilot.py:96-111`）：
把每个粒子被执行内容的保留维度编码成 0—15，按 `权重 × 执行权重` 加权求和，
得到 `[T,16]` 的 "本题执行内容用了哪几个维度" 分布。规则内容的码位由该规则
hyperplane 的非零列并集给出。C++ 端没有这一步，也没有等价输出，
因此无法算"口述部位集合与模型部位集合的一致概率"，也无法算基于该分布的 IoU。

**（5）没有逐题复核表与可读报告。**
Python 侧输出 `report.md`、`review.html`、`trial_review.csv`、`phase_summary.csv`、
`final_candidates.csv`、`numerical_sensitivity.csv`（`model_1031_subject_pilot_report.py:158-192`）。
C++ 侧只有 `summary.json` 与 `arrays.npz`。合作者要在论文里放"逐题口述 vs 模型"的附录表，
需要自己从 npz 里拼。

### 5.2 P1：容易导致误读

**（6）`dominant_probability` 这个名字与它的内容不符。**
`src/inference.cpp:55` 写入的是"被执行的**内容类别**的预测权重之和"，
时点为选择之前，用的是 `w^-`。它统计的是被执行内容的语义类别，
不含任何主导内容或最大后验的判断。
`conditional_choice_weights` 同样用 `w^-`（`:42`，`tapes[i].choice_weights[j]` 只有被执行槽为 1）。
而 `conditional_catalogue_execution_probability` 用的是滤波后的 `w^c`（`:51`）。
`docs/catalogue_probabilities.md:61` 已写明"Current-choice-filtered execution has no matching saved
semantic array"，但数组名本身不提示这一点。把两列直接相减或画在同一张图上会得到错误结论。

**（7）H 编号在不同被试上指向不同的物理比较。**
Task2 CSV 每个被试有自己的特征列顺序：96 名被试共有 24 种顺序，
`head` 在四个位置上分别出现 24、22、26、24 次。规则程序用 0—3 的列索引书写，
所以 Cond1 的 H4 在某被试上是"脖子对腿"，在另一被试上可能是"脖子对头"。
Python 与 C++ 都采用"列索引即被试列顺序"的约定，两边一致，模型结果可比；
但**跨被试汇总逐规则曲线时必须先按 `protocol.feature_order` 重映射到部位名**。
`manifest.protocol.feature_order` 已经记录了该顺序，画图层完全没有使用它，
列标签只写 `H0…H28`。这是当前最容易做出错误论文图的地方。

**（8）`content_path` 的图题写成 "Autonomous path 1"。**
`evaluation/plot_results.py:652` 的标题字符串是
`"Autonomous path 1 (fixed seed; not selected for fit)"`，实际数据是 `simulation_seeds[0]`，
即索引 0。`manifest.content_path.path_index` 写的是 0。图题与元数据相差 1，易被读成第二条路径。

**（9）`beta` 面板的色标自动缩放，看起来像常数。**
`evaluation/figures.py:290-302` 对 `beta` 用 `vmin=None, vmax=None`，
而矩阵里包含未激活槽的 0。S101 的 `conditional_beta[:,3]`（规则类）实际在 0—12.29 之间
有 318 个不同取值，图上却呈现为整段均匀高值。
读者会误以为规则精度从未更新。

**（10）自主带的 100% 区间是 128 条路径的 min/max。**
`summary.autonomous.empirical_100_band` 已写明
"finite-repeat min/max, not full population support"，但图上只有 "100% empirical" 图例，
没有标 n=128。论文引用时需要人工补注。

### 5.3 P2：工程与一致性

**（11）`evaluate` 忽略配置里的 `keep_states`。**
`src/cli.cpp:174` 的默认值是 `false`，`src/cli.cpp:179` 校验类型，
但 `write_evaluation` 在 `:282` 把 `keep_states` 硬编码为 `true` 传给 `run_tasks`，
随后在 `:314` 把 `effective_config["keep_states"]` 改写成 `true` 再写清单。
行为本身合理（评估总要状态），但配置项实际上只在 `filter` 命令生效（`:463`），
读配置文件的人会以为可以关掉状态。

**（12）`simulate` 命令永不保存状态。**
`:469` 传入 `keep_states=false`，因此独立的 `simulate` 输出没有任何 `content_path`。
需要逐内容路径时必须走 `evaluate`。

**（13）`filter` 是开发用命令，不能替代 `evaluate`。**
`:462-466` 注释写明 "Development-only raw filter: the configured root seed is used literally"，
它只跑一次 PF、不做重复平均、不做自主模拟、不做逐规则导出。

**（14）文档与交付产物不一致。**
`README.md:100-111` 描述"新 `evaluate` 自动保存全部目录规则的 `[trial, rule]` 诊断，
Cond1 为 29 列，Cond2/3 为 152 列"，但交付批次是 0 列。
README 的演进说明与批次内的能力声明之间没有交叉引用，
合作者按 README 找数组会直接失败。

**（15）`png/arrays.npz` 内嵌约 91 KB 的 `provenance_json` 字符串。**
`evaluation/plot_results.py:655` 把整份 provenance 序列化为单个 Unicode 字符串数组写入 npz。
压缩后体积占比不大，但它在数组列表里以标量字符串出现，
会让按数值 dtype 批量读取的脚本遇到一个非数值项，需要在读取时单独跳过。

**（16）逐规则数组没有 CSV/窗口汇总，也没有"正确规则/错误规则"这类语义标签。**
Python 的 `rule_diagnostics` 会按冻结窗口输出 `wrong`、`correct`、重复范围与
"头与脖子范围"，并把规则 ID 与部位名对应起来
（`model_1031_rule_diagnostics.py:82-99`、`:129-161`）。
C++ 端只有原始 `[T,H]` 与 `family`/`program` 元数据，语义解读留给读者。

## 6 与论文图需求的对应关系

把当前 C++ 输出逐项对到 Python 的论文/报告图需求上，缺口一目了然。

| 图或量 | Python 输入 | C++ 交付批次 | C++ 当前源码 |
| --- | --- | --- | --- |
| 观察历史后正确类别概率 | `choice_probabilities` | 有 | 有 |
| 自主路径带 | `feedback (paths,T)` | 有（128 条） | 有 |
| 四类表示执行概率 | `filtered_representation_execution (T,4)` | 有 | 有 |
| 逐规则执行/后验 | `catalogue_execution_probability (T,29)` | 无 | 有 |
| 部位使用概率 `[T,4]` | `feature_set_probability (T,16)` 的点积 | 无 | 无 |
| 部位集合一致概率 | 同上按口述码取值 | 无 | 无 |
| 口述编码 `[T,4]` | `feature{1..4}_use` | 有（在 `manifest.oral_reports`） | 有 |
| 口述与模型的区域 IoU | 逐粒子 mask 或 `[T,16]` 分布 | 无 | 无 |
| 逐粒子内容（anchor/mask/θ/rule/beta/weight） | 11 个 `particle_*` | 无 | 无 |
| 固定四维点的类别概率 | 无（Python 也未落盘） | 无 | 无 |
| 组水平图与统计 | `nips_plot.Fig3_Group/Fig4/Fig5` | 无 | 无 |
| 逐题复核表/HTML | `trial_review.csv`、`review.html` | 无 | 无 |

另有一个正面发现值得记录：论文的 `k_comparison` 系列图
（`src/nips_plot.py:1051-1056`、`:1782-1802`）画的是"某一指定规则的逐题后验"，
规则索引在 Cond1 取 0、Cond2/3 取 42。Cond2 与 Cond3 共用
`resources/catalogue_c2.json`（`src/resources.cpp:10`），所以 42 在两者上同义；
Cond2 的规则 42 是 `3d_axis_triple`。当前 C++ 的
`conditional_catalogue_posterior[:,0]`（Cond1）与 `[:,42]`（Cond2/3）
是最接近该曲线的量化量。两者的生成含义并不完全相同：
Python 那是假设空间后验的 `post_max`，C++ 这里是内容信念质量。
把这条曲线接入论文前需要先固定含义并做一次对照，
不能仅因数值都落在 0—1 就当成同一个量。

## 7 建议的修复顺序（本轮未实施）

1. **先重跑 evaluation，不动拟合。** 用当前二进制对 96 名被试各做一次 `evaluate`
   到全新目录，复用 `fit/subject_*/best_parameters.json`；
   `tools/export_evaluation_configs.py` 已提供参数导出与来源校验。
   成本约为重算 PF 与自主模拟，不含参数搜索。完成后逐规则概率即到位。
2. **补画图层的四张图与三张表。** 输入都已在 npz 里：
   逐规则曲线（`conditional_catalogue_execution_probability`、`..._posterior`、`..._prior`）、
   部位使用与集合一致（需要新增 `[T,16]` 分布，见第 3 步）、
   组水平汇总（需新增聚合脚本）、
   以及 `rule_trials.csv`、`rule_windows.csv`、`catalogue_windows.csv` 三个表。
3. **新增 16 位特征集合分布。** 在 `record()` 里按 `(weight, semantic kind, mask 或规则的
   hyperplane 支撑集)` 累加 16 个桶；规则码位由 `Catalogue.rules[h]` 的 polytope 非零列
   并集给出，与 Python `active_mask_distribution` 同义。这是最省成本、收益最大的一项：
   它同时支撑"部位使用概率""集合一致概率"和"区域 IoU 的离散热图"。
4. **新增可选的逐粒子导出。** 加一个 `--export-particles` 开关，
   在 `record()` 旁并行写 `[T,P,K]` 的 `kind/id/mask/anchor/beta/belief` 与 `[T,P]` 权重。
   默认关闭，避免把 320×256×4 的立方体写进每个被试的常规输出。
5. **新增区域概率探针。** 一个 `probe` 子命令，读入四维点列表，
   输出每条规则、每个语义内容在该点的类别概率，供 IoU 与决策面图使用。
   Python 侧同样没有这一能力，属新增功能。
6. **补一次跨语言对照。** 用同一被试、同一 32 trial、同一种子集合，
   对照 C++ 与 Python 的 `catalogue_execution_probability` 与（补齐后的）
   `feature_set_probability`。现有 `tests/fixtures/reference_1031.json` 的机制只覆盖类别概率，
   没有覆盖这两个聚合量。

## 8 证据附录

### 8.1 关键源码位置

| 主题 | 位置 |
| --- | --- |
| 逐粒子聚合为类型均值 | `src/inference.cpp:25-67` |
| 逐规则累加 | `src/inference.cpp:36-54` |
| 逐内容导出仅限自主模拟 | `src/inference.cpp:98-101`、`:118-122`、`:140` |
| 只有路径 0 保留状态 | `src/cli.cpp:241` |
| 逐规则均值与写出 | `src/cli.cpp:150-171`、`:306-313` |
| 逐规则元数据与轴说明 | `src/cli.cpp:320-333` |
| `evaluate` 硬编码 `keep_states=true` | `src/cli.cpp:282`、`:314` |
| `simulate` 不保存状态 | `src/cli.cpp:469` |
| `filter` 为开发用 | `src/cli.cpp:461-466` |
| 内容序列化字段 | `src/model.cpp:244-254` |
| `Slot` 与 `ParticleState` 定义 | `include/gm/model.hpp:25-42` |
| 六个逐规则数组名 | `include/gm/inference.hpp:5-8` |
| 画图层只接受 `[T,4]` | `evaluation/figures.py:197-224`、`:290-302` |
| 逐规则只校验不绘图 | `evaluation/plot_results.py:226-239`、`:242-324`、`:582-584`、`:648-653` |
| 口述固定为未实现 | `evaluation/plot_results.py:586-588` |
| 无跨被试聚合 | `evaluation/plot_results.py:664-687` |
| Python 逐粒子观察载荷 | `src/Bayesian_state/inference/backends/particle_filter.py:2001-2012` |
| Python pilot 逐粒子保存 | `src/Bayesian_state/workflows/analysis/model_1031_subject_pilot.py:219-223`、`:241-243` |
| Python 16 位特征集合分布 | 同上 `:96-111` |
| Python 逐规则执行分布 | 同上 `:114-133` |
| Python 报告读取时丢弃逐粒子 | `src/Bayesian_state/workflows/reports/model_1031_subject_pilot_report.py:25-28` |
| Python 口述集合精确一致 | 同上 `:102` |
| Python 论文图的规则索引 | `src/nips_plot.py:1051-1056`、`:1782-1802` |
| Python IoU 机制 | `src/Bayesian_state/evaluation/oral/mapping.py:121-163`、`evaluation/oral/scoring.py:1536-1563` |
| Python 禁止 1031 口述读出 | `src/Bayesian_state/model/model_1031.py:34-35` |

### 8.2 本轮执行的只读命令

```powershell
# 1. 统计交付批次是否含逐规则数组（结果：96/96 不含）
python -c "…glob evaluation_manifest.json，检查 catalogue 键…"

# 2. 核对 content_path 与真实历史的一致性（结果：S101 一致率 77.19%）
python -c "…比较 content_path.json 的 choice 与 evaluation_inputs.npz 的 observed_choices…"

# 3. 用当前二进制生成对照样本（写入系统临时目录，未写入仓库）
.\build\Release\genmodel_cpp.exe evaluate --config configs\cond1.yaml `
  --subjects 101 --jobs 2 --smoke --output $env:TEMP\gm1031_probe_<id>

# 4. 用当前绘图脚本处理该样本（写入系统临时目录）
python -B evaluation\plot_results.py --input $env:TEMP\gm1031_probe_<id>\subject_101 `
  --output $env:TEMP\gm1031_probe_png

# 5. 质量守恒核对（结果：偏差 0.0 与 2.8e-17）
python -c "…比较 conditional_catalogue_prior 行和与 conditional_prior[:,3]…"
```

绘图脚本需要 `matplotlib`，本轮使用批次日志中记录的
所选 Python 环境的解释器；当时系统默认 `python` 未安装 `matplotlib`，
直接运行 `python -B evaluation/plot_results.py` 会在导入绘图模块时抛
`ModuleNotFoundError: No module named 'matplotlib'`。这属于环境缺包。

### 8.3 交付批次的规模

| 项目 | 值 |
| --- | --- |
| 被试数 | 96（Cond1 101–132，Cond2 201–232，Cond3 301–332） |
| 每名被试输出 | `fit/`、`evaluate/subject_*/`（`evaluation_inputs.npz`、`evaluation_manifest.json`、`content_path.json`）、`png/`（3 张 PNG、`arrays.npz`、`summary.json`） |
| 试次数 | Cond1 320；Cond2 640；Cond3 640 |
| `evaluation_inputs.npz` 数组数 | 57（Cond3 为 59；均无逐规则） |
| `png/arrays.npz` 数组数 | 89（Cond3 为 91；均无逐规则） |
| 目录规则数 | Cond1 29；Cond2 152；Cond3 152（与 Cond2 共用 `catalogue_c2.json`） |
| 条件类别概率形状 | Cond1 `[12,320,2]`；Cond2/3 `[12,640,4]` |
| 自主模拟重复数 | 128 |
| 被试特征列顺序种类 | 24 |
