# Model 0826：condition 2 四分类二值反馈扩展

2026-09-09，按用户要求优先拟合 PMH；condition 3 与消融模型暂停。

入口为 `configs/exp123/model_struct/pmh_model_cond2_0826.yaml`。共享 PF 接收 `condition=2`；直接调用未传 condition 时仍默认为 1。simulation 和 dispatcher 都转交实际 condition。condition 2 必须使用四类 partition、choice 1–4、feedback 0/1。condition 3 已另行实现层级配对及部分反馈（见 `model_0826_condition3_design.md`），有定向测试与短序列验证，尚未完成正式拟合和完整恢复。

## 继承与变化

四维空间使用已有的 116 条四类固定标签假设、boundary 距离与对应 similarity 资源（具体 SHA256 记录在 YAML）。不新增标签学习，不将真实 category 作为作答前模型输入。二值反馈似然为正确时 P(chosen)，错误时 1−P(chosen)，即其余三类概率之和。预测概率数组按类别数分配；二元 orientation 状态轴保持二元。

工作空间、固定感知、记忆衰减、控制器、动态 beta、readout、lapse 机制沿用 0826。特别是 beta 的 evidence=f*p+(1-f)*(1-p)、centered=2*(evidence−0.5) 完全保留，并未按四类机会水平 0.25 重新中心化。这是明确的直接任务扩展，不应宣称为已重新推导或恢复验证的四分类精度学习模型。condition 1 的数值基准必须继续通过。

搜索沿用 condition 1 代表性运行的参数支持与 coarse/fine/final 预算，标记为尚未通过四分类 recovery 的支持。观测 NLL 是样本内拟合分数，不是留出预测性能。

## 诊断边界

`choice_transmission_audit` 暂仅支持 condition 1：四分类错误反馈无法唯一反推正确 category。condition 2 请求该审计会显式报错，避免生成伪正确标签诊断；依赖该审计的内部 genealogy 导出尚不可用。常规 PF、拟合概率、参数搜索、模拟汇总和标准评估不依赖它。

## 本轮运行

被试 229 来自 Fig1 v10 Longer / larger gain 示例；1088 试次全部保留。配置、源码/数据哈希、试次导出、搜索检查点和日志放入 `results/model_0826/cond2/subject_229/pmh_20260909_v1/`。先以 24 试次、4 粒子、2 repeats、1 job 验证 CLI，再运行完整搜索。48 个进程用于 condition 2 的候选×重复任务，另外 16 个留给正在运行的 condition 1；BLAS/OMP/MKL/NUMEXPR 各限 1 线程。final simulation 的 16 repeats 最多使用 16 个进程。

评分约定核查：现有 `metrics/trial.py` 在未传显式 valid mask 时排除首试次。因此完整1088试次均进入PF状态更新并保留预测，正式choice NLL使用1087个试次；本轮保留该已有约定，不暗中改变目标。validation/probability_score_checks.json 同时记录正式掩码分数和全试次描述分数。
