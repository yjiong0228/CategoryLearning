# simulation

本目录负责从已配置的认知模型自主生成 synthetic choice/feedback trajectory。它与
`inference_engine/` 的区别是：inference 条件于被试观测，simulation 生成新的观测。

当前 `trajectory_generation.py` 提供 condition-1 recovery 使用的
`generate_condition1_trajectory()`。物理 stimulus/category schedule 固定；每个 trial 先从
choice readout 采样，再由真实类别产生 feedback，最后用同一 `StateModel` 更新状态。

重复运行、representative run 选择和统计聚合目前仍由
`optimization/optimizer_simulation.py` 编排，因为这些步骤直接服务 hyperparameter objective；
本目录不定义 loss 或搜索策略。
