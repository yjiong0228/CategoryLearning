# 恢复实验的职责边界

- `design.py`：加载设计、生成数据集规格、检查参数支持与试次预算。
- `generation.py`：准备实验真值和 P/PM/PH/PMH 单元，调用已有自主仿真，保存可复现数据包。
- `run.py`：执行 smoke/generate/calibrate/module-fit/parameter-fit/summarize 等阶段。
- `simulation/recovery.py`：恢复数据契约、任务序列哈希、合成 trial 表格；不导入优化层。
- `simulation/autonomous.py`：实际自主选择与反馈轨迹生成。
- `optimization/recovery.py`：按冻结预算拟合一个合成数据集。
- `optimization/recovery_parameters.py`：恢复真值参数与可执行模型设置的对应。
- `evaluation/recovery.py`：冻结候选评分、PF 预算检查、恢复统计和绘图；不启动参数搜索。
- `utils/recovery_artifacts.py`：原子文件写入和来源指纹。

上述 simulation/optimization/evaluation/utils 路径均相对 `src/Bayesian_state/`。
`evaluation/model_recovery.py` 仅保留兼容导出，不再保存恢复实现。
`python -m src.Bayesian_state.run_recovery` 与期刊入口继续可用。

拆分后的运行来源指纹包含所有实际实现文件。旧缓存不会因兼容入口文件未变而误判为可继续运行；
不要强行跨源代码版本 resume。默认预算、种子、选择/反馈时间顺序和评分规则均保持原样。
