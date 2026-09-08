# Target-based oral alignment：0.8 平台核查

## 结论

没有发现将口述概率截断为 0.8 的绘图或归一化 bug。当前 center 编码在 sigma=0.10、29 条规则、latest_by_category、full space 下，对两个理想目标类别中心产生 H0 概率 0.8181183136。S101 的实际最大值完全相同。它是编码参考平台，不是对任意报告成立的数学上界。

计算链：`evaluation/oral/scoring.py::_center_oral_distribution` 对每个 hypothesis 的类别连通分量中心计算 Gaussian likelihood，分量内等权平均、规则间 uniform prior 归一化；`_category_state_distribution` 将每类最近有效报告的 likelihood 联合归一化；`compute_target_based_alignment` 提取目标质量；`reporting.py::plot_target_based_alignment_subjectwise` 仅 rolling mean，ylim=(0,1)。

在理想目标中心下，其他规则仍有非零 likelihood。latest_by_category 替换同类别旧报告，不累乘重复报告，因此反复报告同样的正确中心不会让概率继续向 1 累积。这与持续更新的模型信念不是相同的信息积累机制。

## 已运行的复现

`diagnostics/check_oral_ceiling.py` 调用原生产 encoder；无模型参数修改。
产物：`../outputs/fig2/oral_alignment_audit_v1`。

| 固定 sigma | 两类别理想中心的 H0 质量 |
|---|---:|
| .05 | .999993 |
| .075 | .985919 |
| .10 | .818118 |
| .15 | .348542 |
| .20 | .173717 |

以上为编码敏感性诊断，不是据模型对齐效果调参的建议。
现有 0818 full_observed_fit_v1/evaluation_after_8 的 full-space 源表最大值也是 .818118；另一次 exploratory 结果的 active-space 达 .831304，再次排除硬截断。active/union 会依模型支持集合重新归一化口述，不宜作为主要独立验证。

## Fig2b v3

新增 target-based alignment：完整 29-rule 空间，模型作答前 H0 信念对口述 category-state H0 概率，统一尾随 32 试次平均。虚线 .818118 为两个理想目标中心在当前编码下的参考值，非真值上限或置信区间。原 rule belief 和 reported features 保留。未为视觉对齐修改 sigma 或把口述除以 .818118。

口述只投影到目标 H0 的质量，是完整 hypothesis space 的一维摘要；两侧都不支持目标也可能使用完全不同的错误规则。因此该图不能单独证明全分布对齐。主文后续应补全分布相似度/距离及受约束对照。

时序：model prior 在当前选择之前；oral category-state 使用当前报告并依赖当前 choice 选择类别。它是同试次的描述性对照，不是两个同条件观测后验。缺失报告时 state 可能沿用既有状态，应与当前报告有效性分别记录。

## 后续更可比的验证

优先保持未经重标定的口述概率，并透明显示编码敏感性。若比较幅值，应建立报告观测模型：把模型预测的规则混合分布映射至相同报告表征/编码，比较模型预计的报告与实际报告；需明确口述发生时点和独立确定的噪声尺度。还可直接在报告空间评价预测 likelihood，避免将报告不确定性误判为规则推断失败。不能为了获得完美对齐而缩小 sigma。

验证：最小理想报告复现、真实 S101 重算、旧输出比对；生产模型/编码均未修改。v3 使用现有在线模型导出和新算的口述分布。诊断不是正式外部验证统计。
