# 0826 维护范围与恢复职责整理

本次只执行用户指定的三项工作：删除 reference_models，精简工作流，明确恢复分析归属。
未修复文稿路径、未拆 oral/PF 核心、未移动 model_architecture，也未改模型参数、配置、数据或结果。

## 删除与保留

reference_models 已删除。删除前的静态检查显示其使用方仅属于旧历史工作流；
当前 StateModel、0826 恢复和保留工作流均无此依赖。对应结构测试现在检查该目录已移除。

删除 76 个历史 Python 工作流、3 个历史 shell 入口。保留现有 0826 恢复转发入口、信念迁移比较、
通用 behavior_diagnostics 和 benchmark。0818 boundary recovery 是唯一保留的历史运行脚本，
因为 0826 比较仍调用它的 _load_subject_frames、_subject_engine、_readout_args 等工具。
模型文稿及配套绘图/编译文件未参与精简。

删除 43 个只服务被删除工作流的测试函数。11 个不再包含测试的历史测试模块删除；
3 个混合测试模块仍保留 9 个核心/配置测试函数。没有删除基线失败的文稿检查来获取通过结果。
增加 3 个恢复职责/兼容性/来源指纹测试。

完整删除路径和原文件哈希见 [recovery_cleanup_20260908.json](recovery_cleanup_20260908.json)。
历史配置、文稿与研究结果保留；已经删除的历史运行工具需要从对应 Git 历史版本恢复。

## 恢复分层

| 职责 | 实现 |
| --- | --- |
| 恢复数据契约、任务序列哈希、合成 trial 表格 | simulation/recovery.py |
| 实际自主选择与反馈生成 | 原 simulation/autonomous.py，不改动 |
| 恢复设计验证和数据集规格 | workflows/recovery/design.py |
| 实验真值配置、调用自主仿真、保存数据包 | workflows/recovery/generation.py |
| 恢复参数映射 | optimization/recovery_parameters.py |
| 冻结预算下的数据集拟合 | optimization/recovery.py |
| 冻结候选评分、PF 预算判定、恢复统计和绘图 | evaluation/recovery.py |
| 原子写入与指纹工具 | utils/recovery_artifacts.py |
| 阶段编排与运行来源记录 | workflows/recovery/run.py |

simulation/recovery 不导入优化层。实验单元配置留在工作流，避免为了生成数据让仿真层承担
恢复设计和参数搜索职责。evaluation/recovery 不执行拟合搜索。

原 evaluation/model_recovery.py 变成兼容导出；旧导入得到同一实现对象。
根 run_recovery.py 为公共 CLI 转发，期刊入口继续可用。
运行指纹涵盖全部拆分文件，防止只检查空壳兼容文件而错误复用旧恢复缓存。

## 验证与已知限制

拆分前后 48 个函数/类的 AST 逐项一致（另有 5 个常量定义迁移），不改生成、拟合或统计公式。
恢复/兼容/指纹的定向测试通过；共享 recovery、期刊 recovery、0826 counterfactual 的 --help 通过。

完整调用：

```bash
python -m pytest -q tests CategoryLearning_codes/Bayesian_model/tests CategoryLearning_codes/figures CategoryLearning_codes/tests
```

- 改动前：465 passed，5 failed。
- 改动后：425 passed，5 failed。
- 数量变化：移除 43 个旧工作流测试，新增 3 个职责测试。
- 前后失败的 5 个 test ID 完全相同，均涉及旧 manuscript 路径不存在。
  这些路径问题不在本次授权范围，未修复、未跳过、未修改对应断言。
- 部分文稿绘图脚本的无效转义警告也保持原样。

日志保存在 results/repository_maintenance/recovery_cleanup_20260908。
未启动正式参数恢复、全体拟合或长计算任务；没有创建 commit。
