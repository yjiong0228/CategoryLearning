# Bayesian_model — Model 0826

这是论文 Model 0826 的独立实现包，以 `manuscript/model_0826.tex` 与冻结配置为依据，
从 `src/Bayesian_state` 的有效依赖链迁入。原目录保留兼容旧模型、旧脚本与数值对照；
新工作使用 `CategoryLearning_codes.Bayesian_model`，不依赖旧包的 Python 实现。

## 保留的分层

- `model/`：StateModel 生命周期、engine、装配、认知模块与选择读出。
- `hypothesis_space/`：规则目录、边界几何、似然、固定相似度资源。
- `inference/`：PF、单轨迹分派与统一结果契约。
- `simulation/`：观察数据执行、自主生成、重复聚合与来源记录。
- `optimization/`：0826 参数空间、P/PM/PH/PMH 单元、搜索及收敛诊断。
- `evaluation/`：恢复、PF 检查、oral 对齐、自主/条件状态轨迹分析。
- `metrics/`、`utils/`：被多个执行层实际使用的指标和公共基础工具。
- `configs/`：迁移后的 0826 配置，均使用本包类路径与可移植的相对路径。
- `tests/`：新旧实现一致性、公式级机制测试及恢复契约测试。

规则：先 begin_trial(stimulus)，再 predict_choice()，最后 complete_trial(choice, feedback)。
当前结果不得进入当前预测。PF 是研究者的推断方法，不是新增认知机制。

## 使用

从仓库根目录运行：

```bash
python -m CategoryLearning_codes.Bayesian_model.run_simulation --config CategoryLearning_codes/Bayesian_model/configs/smoke_simulation.yaml
python -m CategoryLearning_codes.Bayesian_model.optimization.cli --help
python -m CategoryLearning_codes.Bayesian_model.run_recovery --help
python -m pytest -q CategoryLearning_codes/Bayesian_model/tests
```

smoke_simulation 是单被试、32 试次、1 次重复的流程检查，不是正式拟合；本次已写入
`outputs/smoke32_v1`。再次运行前将 output_dir 改为新的目录，避免覆盖。
8 试次初版不足以计算 16 试次滑窗指标，已保留失败目录并改用 32 试次检查。

正式结构配置 `configs/model_0826.yaml` 不改冻结参数或粒子数。参数支持仍为
pre-recovery provisional support；`recovery_v1.yaml`/`recovery_v2.yaml` 保留原设计，
仅修改入口路径及新的输出位置。迁移未启动完整恢复、参数搜索或全被试拟合。
已有恢复目录的代码/配置 fingerprint 不等于新包 fingerprint，不要强行跨包 resume。

## 范围与旧代码处理

不迁入旧 reference_models、旧动态/离散策略类、独立 label-mapping 模块、旧 0818
拟合封装及无关的 FFT/旧报告入口。详见 MIGRATION_MANIFEST.json 的逐文件清单。
`feedback_reactive.py` 是 0826 控制器的真实父类，必须保留。当前信念迁移位于
`workspace.py`，不是未迁入的旧 `prior_assignment.py`。

共享文件中仍有被通用结构复用的几何、读出或兼容分支；本次不为删除几行旧选项而
重写经过验证的数值实现。它们不属于 0826 的活跃模型：默认固定标签、w0=0、
expectation/power=1、lapse=0，无 orientation、commitment 或错误规则捕获。
本包不提供旧版本专用策略的公共导出，参数加载器仅接受 model_0826。

当前论文和 PF 路径仍限定 condition 1 的二分类；通用几何接口能表示其他空间，
不意味着 Model 0826 已支持四分类与部分反馈。Task2/3 扩展是下一项独立工作。

模型—文稿逐项核对见 MODEL_0826_AUDIT.md；验证记录见 VALIDATION.md。

## Oral 编码尺度

按论文分析决策，center oral encoder 默认 sigma=0.05（新包及旧入口统一）。这是口述后处理尺度的变更，不改变模型选择预测或拟合参数。历史产物保留其原尺度；比较或合并前检查 metadata 中 oral_center_sigma。Region temperature 不变。迁移清单仍记录迁移时的历史快照，后续此变更不重写清单。
