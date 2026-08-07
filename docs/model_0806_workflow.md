# Model 0806：唯一工作流

这份文件只回答三个问题：哪些旧方法必须继承，0806 为什么有少量不同，以及正式入口在哪里。

## 不推倒重来的部分

0806 必须继续使用仓库已经验证过的基础设施：

1. choice Brier 和 accuracy curve 统一调用
   `src.Bayesian_state.utils.model_evaluation.ModelEval`；不允许脚本各写一套定义。
2. 超参数搜索继续使用
   `src.Bayesian_state.utils.hyper_cd_optimizer.HyperCDOptimizer` 的坐标下降框架。
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

- `scripts/run_model_0806_dynamic_m_recovery.py`：静态/动态模型恢复，支持 surprise 或 uncertainty。
- `scripts/run_model_0806_real_rolling.py`：真实 choice 滚动比较；内置组件完整性检查，并调用公共 Brier/accuracy 实现。
- `scripts/run_model_0806_targeted_diagnostics.py`：冻结 choice 状态后的 choice/RT 定向诊断。
- `scripts/run_model_0806_uncertainty_gate.py`：不确定性的低成本进入门槛。

对应的正式配置：

- `configs/model_0806_dynamic_m_recovery.yaml`
- `configs/model_0806_dynamic_m_real_rolling.yaml`
- `configs/model_0806_dynamic_u_recovery.yaml`
- `configs/model_0806_dynamic_u_real_rolling.yaml`
- `configs/model_0806_targeted_diagnostics.yaml`
- `configs/model_0806_dynamic_u_targeted_diagnostics.yaml`
- `configs/model_0806_uncertainty_gate.yaml`

下一次联合 surprise+uncertainty 动态运行应扩展这些入口，不再新建一套评价脚本。
