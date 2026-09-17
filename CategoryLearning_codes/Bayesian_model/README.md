# Bayesian_model — 期刊工作流

期刊和博士论文共同使用 **`src/Bayesian_state/` 中的一份模型实现**。
本目录保留期刊配置、入口、验证与历史兼容导入，不再维护另一份认知模块、PF、拟合或评价算法。

## 修改放在哪里

| 内容 | 维护位置 |
| --- | --- |
| 状态更新、memory、规则切换、选择读出 | `src/Bayesian_state/model/` |
| hypothesis space、几何和相似度资源 | `src/Bayesian_state/hypothesis_space/` |
| PF、仿真、搜索、oral alignment、恢复 | `src/Bayesian_state/{inference,simulation,optimization,evaluation}/` |
| 数据路径与被试覆盖解析 | `src/Bayesian_state/utils/{datasets,subjects}.py` |
| 期刊模型和运行参数 | 本目录 `configs/` |
| 期刊数据范围检查与入口默认值 | 本目录 `workflow.py`、`run_simulation.py`、`run_recovery.py`、`run_hyper_evaluation.py` |
| 期刊分析与图形 | `CategoryLearning_codes/figures/fig1/`、`figures/fig2/` |
| 博士论文的实验配置 | `configs/{exp123,exp4,exp5,meg}/`；共享定义在 `configs/shared/` |

本目录原来的层级文件是兼容入口。叶模块转发到同一个共享 Python 模块对象，
不是另行加载一份源码。旧 YAML 类路径和常用导入仍能使用；新代码直接导入
`src.Bayesian_state`，不再增加兼容文件。期刊参数空间加载器保留只接受 0826 的限制，
共享加载器继续兼容 0818。相似度矩阵仅保留共享目录中的一份。

## 日常工作流

从仓库根目录运行；无需复制模型或调整 PYTHONPATH：

```bash
python -m CategoryLearning_codes.Bayesian_model.run_simulation --config CategoryLearning_codes/Bayesian_model/configs/smoke_simulation.yaml
python -m CategoryLearning_codes.Bayesian_model.run_recovery --help
python -m src.Bayesian_state.run_recovery --help
python -m pytest -q CategoryLearning_codes/Bayesian_model/tests
```

示例 simulation 配置指向已有 `outputs/smoke32_v1`，再次执行前请复制 YAML 并将
`output_dir` 改成新的目录。模型结构、随机种子、粒子数和数据路径均由配置决定。
期刊 simulation/recovery 入口检查所选被试的解析后数据路径全部位于 `data/exp123/`。
其他保留的旧 CLI 是通用兼容入口，使用时仍需明确传入期刊配置。

正式 `recovery_v1.yaml`、`recovery_v2.yaml` 和 `recovery_simulation.yaml` 的并行预算为
128，不主动预留 CPU。共享执行层限制每个 worker 的数值线程为 1，并按当前独立任务数和
可用 CPU 收缩进程池；`smoke_simulation.yaml` 继续保持小规模。详见
[128 核并行策略](../../src/Bayesian_state/docs/maintenance/model_0826_parallel_policy_20260917.md)。

博士论文使用 `python -m src.Bayesian_state...` 和对应实验配置。现有 `dataset`
字段可以指定 processed_dir、learning_data、perception_summary、perception_summary_72、
feature_order_data，不必因为换数据目录复制核心算法。后续字段转换放在数据适配层，
类别结构和反馈含义由任务配置及对应任务实现处理，不在认知模块内根据文件夹名称分支。

**路径可配置不等于科学模型已支持所有任务。** 期刊默认配置仍以 condition 1 二分类为主；共享核心
已实现 condition 2 四分类二值反馈和 condition 3 层级部分反馈。condition 3 目前通过
定向测试和短序列验证，正式拟合、完整恢复及消融尚未完成。exp4/exp5 和 MEG 的任务
适配仍需独立验证。默认超参数、预测时序和 oral σ=0.05 保持不变。

## 发表版本与持续开发

开发阶段两篇文章共享核心修复；实验差异放到独立 YAML。改变科学行为时用新配置/显式选项，
避免为了博士论文实验直接改变期刊配置默认行为。运行结果使用新目录。

确定投稿版本时，提交相关代码与配置，给该 commit 建立明确的期刊版本 tag；博士论文继续
在后续提交开发。需要复现旧期刊结果时，在另一个 Git worktree 检出该 tag，使用已记录的
依赖环境、数据哈希、配置和随机种子。tag 冻结代码；数据和生成结果另行归档。
不通过拷贝一整份模型来冻结版本。当前未创建 tag、额外 worktree 或自动提交。

恢复流程现统一位于 `src/Bayesian_state/run_recovery.py`，原
`src/Bayesian_state/workflows/runs/run_model_0826_recovery.py` 仍可调用。来源指纹指向实际共享代码；
旧恢复产物的指纹可能不匹配，保留原产物并在其历史版本复现，不强行跨版本 resume。

## 验证与历史

`tests/fixtures/pre_shared_core.npz` 是合并前生成的固定数值参考：四种 PF 设置、
两种自主轨迹设置，共 232 个数组；元数据记录生成 commit 与数据/配置/参考文件哈希。
测试与该参考比较，不把两个同源导入互相比对当作数值回归证据。
详见 [SHARED_CORE_VALIDATION.md](SHARED_CORE_VALIDATION.md)。

`MIGRATION_MANIFEST.json`、`MODEL_0826_AUDIT.md`、`VALIDATION.md` 记录此前独立迁移的历史，
不代表当前包仍然独立。历史输出和它们的元数据保持原样；原 migration_tools 仅供追溯。
