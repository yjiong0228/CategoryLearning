# Model 0806：唯一工作流

这份文件只回答三个问题：哪些旧方法必须继承，0806 为什么有少量不同，以及正式入口在哪里。

## 不推倒重来的部分

0806 必须继续使用仓库已经验证过的基础设施：

1. choice Brier 和 accuracy curve 统一调用
   `src.Bayesian_state.model_evaluation.model_evaluation.ModelEval`；不允许脚本各写一套定义。
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
- 正式预测对保留下来的离散参数候选作顺序模型平均，不挑一条“最佳潜在路径”。

因此，0806 新动态模型应先通过原有 Hyper-CD 选择较小的超参数区域，再在该区域内进行粒子积分和模型平均。当前直接枚举的网格只属于开发运行，不是对 Hyper-CD 的永久替代。

## 指标各自负责什么

- Hyper-CD 的首要目标沿用 choice Brier；accuracy-curve 指标用于并列约束轨迹。
- 正式留出模型比较报告 NLL，因为它评价完整预测概率并严惩自信的错误。
- 同时报告公共实现产生的 choice Brier、accuracy-curve MAE/RMSE、正确率偏差和曲线相关。
- RT 用于比较联合预测和缩小可接受候选集合；系数方向完整报告，但不预设必须为正。

## 0806 当前正式入口

0806 已经嵌入通用 `StateModel`，正式运行不再调用一套专用模型代码：

- 模型结构：`configs/model_struct/pmh_model_cond1_0806.yaml`
- 固定参数仿真：`configs/simulation_cfg/pmh_cond1_simulation_0806.yaml`
- Hyper-CD：`configs/hyper_cd_cfg/pmh_cond1_hyper_cd_0806.yaml`
- 新机制模块：
  `src/Bayesian_state.problems.modules.finite_workspace_transition.AdaptiveFiniteWorkspaceTransitionModule`
- 数值积分入口：`src.Bayesian_state.optimization.particle_filter.run_state_model_particle_filter`
- 统一评价入口：`src.Bayesian_state.optimization.optimizer_common.evaluate_state_model_run`

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

## 旧 0806 代码的保留边界

`src/Bayesian_state/manuscript_models/model_0806.py` 与原来的
`scripts/run_model_0806_*.py` 只保留为恢复实验、历史结果复现和数值 oracle。它们不再是
真实数据拟合的正式入口，也不应继续扩展新的 transition、评价或超参数搜索逻辑。

联合 surprise+uncertainty、静态 FA2 和单信号 FA3-M 的区别现在全部由同一个 transition
module 的 `rate_controller` 配置表达；无需新增模型脚本。
