# MEG 配置

此目录用于 MEG 实验配置。探索与预处理 notebook 使用 `data/meg/` 中的数据。
正式模型配置尚待建立；配置需明确类别结构、反馈语义、知觉噪声与试次对应，并验证模型支持这些任务设定。

## 数据预处理

`notebooks/preprocess.ipynb` 的 MEG 部分生成两份数据：

- `data/meg/processed/Task1b_errorsummary.csv`：读取 Task1b 原始文件，按 `feature` 和 `level` 平均 `threshold_mean`，生成 `threshold_mean_mean`。特征编号1–4依次对应 green、yellow、pink、blue；被试编号取自文件名。
- `data/meg/processed/Task3b_processed.csv`：自动发现 `Task3b_<被试编号>_bhv.csv`，根据各被试的 `seqID` 连接对应的 `Seq1.csv` 或 `Seq2.csv`，检查刺激覆盖、类别对应和试次数。

同一被试的分段行为 CSV 按 session 和原始 trial 顺序合并。`iSession`、`iRun`、`iTrial` 和 `stiID` 保留原编号，分段中独有的 `success` 列也保留。Task1b 的特征列统一命名为 `feature`。

处理后的行为表保留 `presskey`、`rating`、`miss` 及原始事件持续时间。漏答试次保留在原位置，缺失的选择、评分和反馈保持为空。模型读取时跳过 `miss=1` 的整条记录，下一条有效试次继续已有学习状态；结果记录原始试次标识，便于与评分及 MEG 时序对齐。rating 仅用于独立检验模型预测。

2026年10月12日更新覆盖333–348共16名被试。新版 `Seq2.csv` 含1440个刺激，覆盖第4个 session。复现处理时，执行 notebook 的导入单元、MEG 知觉汇总单元以及 MEG 行为预处理定义和调用单元。
