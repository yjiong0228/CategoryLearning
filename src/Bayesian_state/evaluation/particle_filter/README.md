# 粒子滤波评价

本包包含粒子边际推理专用的汇总与诊断：状态和 ESS 汇总、策略审计、因果残差检查以及
选择传递反事实分析。这些模块只读取已经完成的结果，或执行用户明确要求的审计重跑；
它们不定义推理行为。

包级公开名称通过 `__getattr__` 按需加载。仅导入
`src.Bayesian_state.evaluation.particle_filter` 不会立即导入 matplotlib 或各审计实现；访问某个
具体函数时才加载对应模块。

Model 0923 的 `max_capacity` 表示候选数量的上限。策略概况图读取保存的反馈前策略概率，
标题标注容量上限；summary CSV 的 `workspace_capacity_kind` 为 `upper_bound`。
候选的增加和退出使实际数量随试次变化。`marginal_active_probability` 保存每条规则在反馈前
属于候选集的粒子边际概率；逐试次对这些概率求和，得到期望候选数。候选数量图展示该数量，
并单独画出容量上限。固定容量模型继续由替换比例计算保留与新增数量，容量类型记为 `fixed`。
