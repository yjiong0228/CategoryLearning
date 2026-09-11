# 优化诊断

`search.py` 读取已有搜索产物，评价收敛性、plateau 和多目标选择。
`predictive.py` 可能重新运行选中的候选，以评价预测采样与波动性，因此调用方必须显式控制
subjects、repeats 和并行度。

`flatten_hyperparams` 同时读取旧版整块 kwargs、新版点分路径和
`__profile_candidate__:*` 打包坐标。0826 数值列包括 M、χ、γ、E_C、E_E、
g₀、c_A、c_G 和 β 更新参数；不存在的字段仍记为缺失，不能解释为零。
没有旧式策略列表的 0826 transition 使用稳定的 `workspace…:profile…` 标签区分
参数配置；该标签不表示额外的心理策略类别。原始参数签名保持不变。

2026-09-10 修复了新版 γ 和打包坐标在诊断表中被漏读的问题。修复只影响后处理，
不会修改搜索分数或已选参数；旧研究产物保留，重新生成诊断时必须选择新目录。
S129 更正后的参数表、数值复算与局限见
[`estimation_audit_20260910_v1`](../../../../results/model_0826/cond1/subject_129/estimation_audit_20260910_v1/README.md)。
