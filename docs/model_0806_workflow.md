# Model 0806：唯一工作流

这份文件只回答三个问题：哪些旧方法必须继承，0806 为什么有少量不同，以及正式入口在哪里。

## 不推倒重来的部分

0806 必须继续使用仓库已经验证过的基础设施：

1. choice Brier 和 accuracy curve 统一调用 `src.Bayesian_state.metrics`；
   `ModelEval` 只消费这些共享结果并作图，不允许脚本各写一套定义。
2. 超参数搜索继续使用
   `src.Bayesian_state.optimization.hyper_cd_optimizer.HyperCDOptimizer` 的坐标下降框架。
3. 数据切分、被试级汇总、重复模拟、随机种子敏感性和模型评价，继续遵守
   `src/Bayesian_state` 的现有工作流。
4. 0805 已冻结的知觉输入、规则空间、局部/全局核、双通道记忆、选择读出和
   choice 历史条件化不重新设计。

## 坐标下降与模型平均怎样共存

二者用途不同，不应二选一：

- 坐标下降选择固定超参数，例如记忆、读出和动态系数的候选区域。
- 粒子滤波积分不可见的规则转移路径。
- 当前冻结预测使用 Hyper-CD 选出的参数候选，不挑一条“最佳潜在路径”；粒子滤波在每个候选内
  对潜在转移路径积分。跨离散参数候选的顺序模型平均属于后续独立工作，不在本轮实现中。

因此，0806 新动态模型先通过原有 Hyper-CD 选择超参数，再由粒子后端对该候选内的潜在路径
积分。跨候选模型平均尚未接入；当前直接枚举的网格只属于开发运行，不是对 Hyper-CD 的永久替代。

## 指标各自负责什么

- Hyper-CD 的首要目标沿用 choice Brier；accuracy-curve 指标用于并列约束轨迹。
- 当前顺序留出报告 choice Brier 及共享的轨迹统计；跨模型正式比较中的 NLL 属于后续工作。
- 同时报告公共实现产生的 choice Brier、accuracy-curve MAE/RMSE、正确率偏差和曲线相关。
- RT 联合预测和跨候选模型平均都尚未在本轮接入。

## 0806 当前正式入口

0806 已经嵌入通用 `StateModel`，正式运行不再调用一套专用模型代码：

- 模型结构：`configs/model_struct/pmh_model_cond1_0806.yaml`
- 固定参数仿真：`configs/simulation_cfg/pmh_cond1_simulation_0806.yaml`
- Hyper-CD：`configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml`
- 新机制模块：
  `src.Bayesian_state.problems.modules.hypo_transition.dynamic_continuous.DynamicContinuousHypothesisTransitionModule`
- 数值积分入口：`src.Bayesian_state.inference_engine.backends.particle_filter.run_state_model_particle_filter`
- 单次模型执行入口：`src.Bayesian_state.simulation.state_model_execution.evaluate_state_model_run`
- 已保存结果评价入口：`src.Bayesian_state.run_model_evaluation`

运行命令：

```bash
python -m src.Bayesian_state.optimization.hyper_cli \
  --backend cd \
  --config configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml

python -m src.Bayesian_state.run_simulation \
  --config configs/simulation_cfg/pmh_cond1_simulation_0806.yaml
```

配置中的 `inference.backend: particle_filter` 使标准 runner 自动积分潜在
active-set 路径。每个 simulation repeat 只是独立的 filter seed，用来检查有限粒子
近似的稳定性，不再代表一条被挑选的潜在轨迹。

`pmh_cond1_simulation_0806.yaml` 同时声明顺序留出协议：Hyper-CD 只用前 50% trial 选择参数，
冻结参数后的 simulation 只用后 50% trial 报告指标。两阶段都运行完整 trial 序列，所以后缀预测
仍可使用之前已经观察到的反馈更新状态，但后缀绝不反向参与参数选择。切分 provenance 保存为
`selection.selection_meta.score_context`。

评价入口会把粒子 `marginal_prior` 适配为通用 prior（不冒充 posterior），并输出动态
`m_t/g_t`、surprise/uncertainty、ESS/重采样以及 marginal active-set heatmap。

## 旧 0806 代码的保留边界

`src/Bayesian_state/manuscript_models/model_0806.py` 与原来的
`scripts/run_model_0806_*.py` 只保留为恢复实验、历史结果复现和数值 oracle。它们不再是
真实数据拟合的正式入口，也不应继续扩展新的 transition、评价或超参数搜索逻辑。

联合 surprise+uncertainty 和单信号 FA3-M 由 continuous transition 的
`rate_controller` 表达；可选 FA3-MG 用 `range_controller` 令 `g_t` 随上一试次信号变化。
Hyper-CD 把 transition class 与两组 controller 绑定成同一个机制坐标：全零 controller 的
静态 FA2 使用 `StaticWorkspaceHypothesisTransitionModule`，其余候选使用
`DynamicContinuousHypothesisTransitionModule`。因此静态/动态的认知解释不会依赖运行后再
猜测参数是否恰好为零。
