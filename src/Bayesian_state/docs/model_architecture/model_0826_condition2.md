# Model 0826：condition 2 四分类二值反馈扩展

当前更新（2026-10-05）：已修正四分类二值反馈的 β 机会水平基准；这是改变学习行为的修复，尚未重新拟合被试。下面的 2026-09-09 运行记录及既有结果保留。

2026-09-09，按用户要求优先拟合 PMH；condition 3 与消融模型暂停。

入口为 `configs/exp123/model_struct/pmh_model_cond2_0826.yaml`。共享 PF 接收 `condition=2`；直接调用未传 condition 时仍默认为 1。simulation 和 dispatcher 都转交实际 condition。condition 2 必须使用四类 partition、choice 1–4、feedback 0/1。condition 3 已另行实现层级配对及部分反馈（见 `model_0826_condition3_design.md`），有定向测试与短序列验证，尚未完成正式拟合和完整恢复。

## 继承与变化

四维空间使用已有的 116 条四类固定标签假设、boundary 距离与对应 similarity 资源（具体 SHA256 记录在 YAML）。不新增标签学习，不将真实 category 作为作答前模型输入。二值反馈似然为正确时 P(chosen)，错误时 1−P(chosen)，即其余三类概率之和。预测概率数组按类别数分配；二元 orientation 状态轴保持二元。

工作空间、固定感知、记忆衰减、控制器、readout、lapse，以及 β 的增减率和上下界沿用 0826。
初版保留了二分类的支持量 \(\kappa=2(\ell-0.5)\)，导致均匀四类预测在错误反馈后提高精度、
正确反馈后降低精度。2026-10-05 经用户确认，修正共享 BetaModule 的
`probabilistic_feedback` 更新；规则信念的反馈似然不变。

令 \(p=P_h(y_t)\)、\(r\in\{0,1\}\)、类别数为 \(C\)，则

\[
\ell=rp+(1-r)(1-p),\qquad
b=r/C+(1-r)(1-1/C).
\]

以该反馈事件的机会概率 \(b\) 为零点，将两侧分别缩放到 \([-1,1]\)：

\[
\kappa=
\begin{cases}
(\ell-b)/(1-b), & \ell\ge b,\\
(\ell-b)/b, & \ell<b.
\end{cases}
\]

四分类的 \(b\) 在正确时为 0.25、错误时为 0.75；不能把两类反馈统一减去 0.25。
均匀规则在两种反馈下均不改变 β。相反证据仍可降低精度，支持证据仍可提高精度，
原有 `increase_rate`、`decrease_rate`、更新范围及预测时序保持不变。
两侧缩放保留“完全支持／反驳”处的原学习率含义。\(C=2\) 时公式精确退化为旧式；
代码保留原二分类算术分支，期刊冻结数值基准应逐元素一致。
Condition 3 的 `hierarchical_feedback` 使用独立的三值反馈支持公式，本次未改动。

该修复直接作用于当前共享核心中的多分类二值反馈路径，无需新增拟合参数或配置开关。
正式四分类配置的 provenance 标记为 `binary_feedback_chance_centered_v1`。
冻结的 `model_0826.tex` / PDF 及历史结果不重写；本节是当前四分类实现的修订说明。
旧结果属于修复前公式，不能改标签后当成新版拟合或沿用旧断点；重现它们应检出各次
运行记录的 Git 版本。新的拟合使用新输出目录，不在本次修复中启动。

搜索沿用 condition 1 代表性运行的参数支持与 coarse/fine/final 预算，标记为尚未通过四分类 recovery 的支持。观测 NLL 是样本内拟合分数，不是留出预测性能。

## 2026-10-05 修复验证

- 新增 20 项机会水平回归：修复前 16 项失败，修复后全部通过；覆盖均匀预测无漂移、
  不同选择标签、支持方向、非零 lapse、活跃范围及上下界。
- 相关测试共 353 项通过，包括三条件推理、0923、版本契约和期刊保存的独立数值参考；
  没有重生成参考数组。二分类冻结结果逐元素一致。
- 正式 simulation CLI 对 S206 前 32 题运行 8 粒子、1 次重复、1 worker，使用独立系统
  临时目录。输出记录修复版本，32 × 116 规则信念有限、非负且归一化。
- 环境：Python 3.11.10、NumPy 1.26.4、pytest 7.4.4、PyYAML 6.0.2。
  本次未重新拟合参数，也未评价新旧模型拟合优劣。

## 诊断边界

`choice_transmission_audit` 暂仅支持 condition 1：四分类错误反馈无法唯一反推正确 category。condition 2 请求该审计会显式报错，避免生成伪正确标签诊断；依赖该审计的内部 genealogy 导出尚不可用。常规 PF、拟合概率、参数搜索、模拟汇总和标准评估不依赖它。

## 本轮运行

被试 229 来自 Fig1 v10 Longer / larger gain 示例；1088 试次全部保留。配置、源码/数据哈希、试次导出、搜索检查点和日志放入 `results/model_0826/cond2/subject_229/pmh_20260909_v1/`。先以 24 试次、4 粒子、2 repeats、1 job 验证 CLI，再运行完整搜索。48 个进程用于 condition 2 的候选×重复任务，另外 16 个留给正在运行的 condition 1；BLAS/OMP/MKL/NUMEXPR 各限 1 线程。final simulation 的 16 repeats 最多使用 16 个进程。

评分约定核查：现有 `metrics/trial.py` 在未传显式 valid mask 时排除首试次。因此完整1088试次均进入PF状态更新并保留预测，正式choice NLL使用1087个试次；本轮保留该已有约定，不暗中改变目标。validation/probability_score_checks.json 同时记录正式掩码分数和全试次描述分数。
