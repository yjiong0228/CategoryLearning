# Model 0826：S129 pipeline 结果目录说明

检查对象为 `results/model_0826/cond1/subject_129/pipeline_20260908_v1/`。
这是一次包含性能调查、接口验证和中断恢复记录的单被试 pipeline 审计，因此比普通正式运行目录复杂。
下面的“按需”表示将来无需每次生成，不表示可以删除本次已存在的研究记录。

## 顶层目录

| 目录或文件 | 内容和用途 | 后续普通运行是否需要 |
|---|---|---|
| `models/` | 按 P、PM、PH、PMH 保存拟合和评价 | 核心；本次后续范围已收窄为 PMH |
| `configs/` | engine、搜索、基础 simulation 和冻结参数 YAML | 必须保留，解释与复现结果依赖它们 |
| `source_data/` | S129 输入子集与代表被试选择依据 | 保留已有快照；不是可以改写的原始数据 |
| `logs/` | 阶段命令、返回码、耗时和运行日志 | 应保留，识别实际完成与失败状态 |
| `provenance.json` | 代码、数据、依赖与计算环境来源 | 应保留 |
| `provenance/interruption_20260909_1237/` | 中断时的搜索与状态备份 | 本次恢复审计依据；普通无中断运行无需生成 |
| `pipeline_smoke/` | P、PMH、PMH24 等小预算接口检查 | 按需，不能当作最终拟合 |
| `performance/` | 性能基线、profile 和优化前后数值对照 | 按需，仅做性能调查时生成 |
| `validation/` | 数值回归、数据/反馈审计、图形检查与问题记录 | 按需保留验证依据，无需复制整套调试记录 |
| `status*.json`、`scope_update_*.json`、`README.md` | 阶段状态、范围变化及导航 | 保留；旧“运行中”文字不能单独证明仍在运行 |

2026-09-09 的范围调整文件明确只继续 PMH；P、PM、PH 是之前四模型计划的已有结果或检查点，
不能因为仍有目录就把它们都视为已完成的消融对照。

## models/PMH 内部

| 子目录 | 回答的问题 / 内容 | 必要性和再生成成本 |
|---|---|---|
| `optimization/` | coarse/fine 参数搜索、候选评分、checkpoint、独立 seed 复评分 | 核心拟合证据，重跑昂贵，保留 |
| `simulation/subjects/` | 冻结参数后的评分、代表 PF run 状态和模型 provenance | 后处理的直接输入，保留 |
| `simulation/cache/` | 由 JSON 的 `raw_runs_ref` 引用的压缩重复运行记录 | PPC、跨 seed 汇总等依赖它；不是可随手清除的临时缓存 |
| `search_diagnostics/coarse/`、`fine/` | 搜索是否收敛、候选如何选择 | 建议保留；可由搜索记录重新生成 |
| `evaluation/basic/` | accuracy、在线 prior、策略、active set、ESS 与条件预测区间 | 常规 PF 评价，保留 |
| `evaluation/behavior_ppc/` | 条件于真实历史的行为预测检查与序列残差 | 常规评价，保留 |
| `evaluation/oral_alignment_*/` | 口述规则与模型状态的外部对齐 | 使用口述验证时需要；模式必须与模型表示匹配 |
| `evaluation/trajectory_accuracy/`、`trajectory_posterior/` | 将重复 run 按误差排序后的旧 trajectory 图 | PF 默认不再生成；已有内容只作为历史产物保留 |
| `internal_trajectories/` | 真实完整历史条件下的 PF 祖先路径、信念与 genealogy 充分性 | 解释内部过程时按需运行；需要额外 PF 推理 |
| `autonomous_trajectories/` | 用模型自己的 choice/feedback 演化的完整学习曲线、形态和掌握时间 | 检验自主行为生成时按需运行；需要额外模拟 |

`internal_trajectories` 的 terminal genealogy 是完整历史条件下的近似，需结合祖先退化诊断；
`autonomous_trajectories` 使用模型自己的历史。二者都与旧的 PF seed 排名图不同，不能因
取消 `trajectory_accuracy` / `trajectory_posterior` 而一并视为冗余。

## evaluation 的来源与 oral mode 修正

`logs/PMH_evaluation_status.json` 记录的评价入口是
`python -m src.Bayesian_state.run_model_evaluation`，输入为 `models/PMH/simulation/`。
原命令只设置了 `--oral-center-sigma 0.05`，没有设置 `--oral-mode`，所以采用当时默认的
`center`。PMH engine YAML 与 subject JSON 的 resolved provenance 均明确为
`inference.backend: particle_filter` 和 `likelihood.distance_mode: boundary`。
因此原 `oral_alignment_center_mode/` 确实不符合这次要求的表示一致性。
manifest 中的 `ok` 表示步骤执行成功，不代表当时验证了这种一致性。

新的共享入口默认 `--oral-mode auto`：boundary 对应 region，prototype 对应 center；
不依赖文件夹中的模型名或当前可变 YAML。没有足够 provenance 的旧 JSON 需显式提供 oral mode。
PF 默认跳过两项旧 trajectory 图，但保持基础评价、PPC 与条件预测 accuracy band。
这些改动属于评价流程与口述测量模式的变化，不改变拟合、PF 算法、模型参数或选择概率。

oral 目录内各项是同一口述证据的不同对齐角度：

| 内容 | 用途 |
|---|---|
| `oral_mass_probabilities.npz`、diagnostics CSV/PNG | 将口述映射到假设空间的分布，以及编码有效性检查；供后续对齐共享 |
| `distribution_based_alignment/` | 在假设空间比较口述分布与模型分布 |
| `oral_based_alignment/` | 将模型信念投影回口述 center/region 表示再比较 |
| `target_based_alignment/` | 比较目标假设的模型概率与口述概率 |
| `hit_based_alignment/` | 比较目标是否进入模型/口述候选集合 |
| `coverage_based_alignment/` | 检查模型 active set 对口述候选集合的覆盖 |

这些对齐角度并非每篇论文都必须全部展示；保留 NPZ、诊断和实际使用的指标即可支持解释。
本次未删除任何历史产物，也未把 region 的 temperature 替换为 center 的 Gaussian sigma，
两者是不同的测量参数。

## 2026-09-13 重新评价

S129 已保存 simulation 的重新评价写入新的
`models/PMH/evaluation_pf_region_20260913_v1/`，默认自动选中
`oral_alignment_region_mode/`；没有生成两项旧 trajectory 目录。
`evaluation_manifest.json` 共记录 24 项成功、2 项不适用、2 项跳过，没有失败项。

本次只读使用原仿真输出，没有重新拟合或重新运行 PF。旧 evaluation、simulation 与 configs
共 72 个文件的 SHA-256 校验均保持不变。新目录的 `evaluation_run.json` 保存命令、输入与
入口代码哈希、依赖版本；`validation.json` 保存产物与概率归一化检查、55 项相关测试结果。
所有生成图均为 PNG。旧 `evaluation/` 中的 center 报告保留为历史结果，应使用新 region
报告来解释本次 boundary 模型的口述对齐。
