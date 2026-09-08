# Model 0826 — manuscript/implementation audit

比较基准：manuscript/model_0826.tex 与 configs/model_struct/pmh_model_cond1_0826.yaml。
新 configs/model_0826.yaml 只改 Python 类路径；参数值与冻结配置一致。
此核对建立实现对应关系，不证明模型已获得行为或外部验证支持。

| 文稿机制 | 有效实现 | 核对结果 |
|---|---|---|
| 29 条固定二分类规则、固定标签、均匀基础先验 | hypothesis_space/spaces/continuous.py、observation_model/continuous_partition.py | 迁入固定目录及边界实现；未迁入标签方向模块 |
| 被试知觉误差、特征顺序、clip 到 [0,1] | model/modules/perception.py、utils/datasets.py | 保留原数据读取与随机流；数据路径仍指仓库 data/processed |
| 到类别区域的距离与 exp(-βd) 预测 | hypothesis_space/geometry/boundary.py、observation_model/likelihood.py | boundary 模式、β_source=action；不是 prototype 或平方距离替换 |
| 固定填满的 M 槽位，χ 两种执行结构 | model/modules/hypothesis_transition/workspace.py、execution.py | χ=1 保护执行槽；迁移不重新定义容量 |
| 当前反馈后更新记忆，w0=0 的衰减信念 | model/modules/memory.py、model/engine.py | 复用 DualMemoryModule 的 fade 分支，γ 与 α 不变；类名不表示本模型使用静态记忆 |
| 最近反馈决定基线，更早 F 决定事件增益 | nested_feedback_accumulator.py、feedback_reactive.py | event_history_excludes_latest_error=true；事件读 F_(t−1)，范围读 F_t；公式级测试覆盖 |
| P(K>0)=E、低信念淘汰、混合局部/全局提议 | workspace.py、execution.py | 槽位概率变换、不放回抽样、τ_local=.10 保持 |
| 相似度是固定标签功能一致率 | hypothesis_space/similarity.py、resources/similarity/*.npy | 仅迁入 0826 的 29×29 固定矩阵并核对 SHA256；原生成 seed 未记录的局限保留 |
| 主信念迁移 (1−K/M)Γ+(K/M)Z | workspace.py | similarity_transport 为主；不是旧的逐对质量迁移 |
| 质量守恒反事实 | workspace.py | 只作为文稿明确允许的敏感性，未增加自由参数 |
| 动态 β、支持/反驳两种更新、活跃规则反事实评价 | model/modules/beta.py | probabilistic_feedback、active_hypotheses、[.1,25]；新进入规则从 β0 开始、不按先验缩放 |
| χ=0 混合；χ=1 持续执行，搜索后独立 .20 切换 | model/readout.py、hypothesis_transition/execution.py | 两种结构均进行 PF 和自主生成逐值对照；无新增软最大化、信念锐化或 lapse |
| PF 作答前预测，见到选择后加权 | inference/backends/particle_filter.py | 保留状态/概率/诊断契约、重采样、随机数与时序；不是事后平滑 |
| P/PM/PH/PMH 及恢复 | optimization/model_0826.py、evaluation/model_recovery.py、run_recovery.py | 保留既有实验工具及源配置支持；不把 provisional 支持改成已通过恢复 |

## 已识别并处理的遗留项

- 旧公共 __init__ 导出会连带加载已移除的策略；新包导出收敛到实际机制。
- 旧 parameter_space 提供 0818 专用别名及旧常量名；新包移除旧别名，常量名改为 0826，
  数值支持不变。原 src 包继续保持旧接口。
- 旧 run_hyper_evaluation 默认指向不存在的旧离散候选 JSON；新包默认不使用该查找表。
- recovery runner 的代码 fingerprint 曾硬编码 src 路径；改为新包实际实现位置。
- metrics 的 group/residuals 经懒加载实际被 PF 诊断使用，静态依赖扫描遗漏后由导入检查发现，已补入。
- 控制器文档曾误把新时序称为 revised 0818；新副本文档改为 0826，不改公式。

## 保留而不扩大声称

共享 workspace/readout/beta 等文件仍含其他配置可用的分支。本轮迁移保持这些文件的
数值行为，使用冻结 0826 配置限定活跃路径，而不把它们当成论文模型的一部分。
尚未实现四分类及 0.5 反馈语义；不会因通用 partition 支持 n_cats=4 就宣称模型已扩展。
未更改 manuscript、旧 configs、src/Bayesian_state 或研究数据；也未改其他模型家族。
