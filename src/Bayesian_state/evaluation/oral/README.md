# 口述报告评价

本目录将中心点和区域口述报告映射到共享 hypothesis space，并从分布、口述表征、目标、命中
和覆盖五个角度评价模型对齐程度。所有代码都属于后处理，不改变模型状态。

| 文件 | 职责 |
|---|---|
| `mapping.py` | 解析中心/区域报告，计算区域与各 hypothesis 的几何重合 |
| `scoring.py` | 构造口述与模型分布，计算五类对齐指标；不写盘、不作图 |
| `reporting.py` | 汇总统计、CSV 序列化和组/被试级绘图 |
| `alignment.py` | 组合 scoring 与 reporting，并提供稳定的 `OralModelAlignmentMixin` |
| `__init__.py` | 重导出公开接口 |

## Oral hypothesis distribution

五类 alignment 名称保持不变，并共享同一份 oral hypothesis distribution。缺失或无法解析的
口述报告保持为全 `NaN`，不会被替换成 uniform distribution。

### Center mode

对 trial `t` 的口述中心 `y_t`、实际选择类别 `k_t` 和 hypothesis `h`，每个类别连通分量的
自动几何中心 `c_(h,k,m)` 定义一个各向同性 Gaussian 报告似然。一个 hypothesis 内的分量似然
等权平均，避免多分量 hypothesis 因中心数量更多而自动获得更多质量：

```text
L_t(h) = mean_m Normal(y_t; c_(h,k,m), sigma_oral^2 I)
q_t(h) = L_t(h) / sum_j L_t(j)
```

`q_t` 使用独立于待评价模型的 uniform hypothesis prior，在当前 fixed-label hypothesis space 上
严格归一化为 1。默认 `sigma_oral=0.10`，对所有 trial 和 subject 固定；CLI 可用
`--oral-center-sigma` 显式覆盖。禁止根据每个 trial 的 hypothesis 距离重新估计 temperature，
因为那会消除绝对距离尺度并强制每个 trial 产生近似宽度的分布。

### Region mode

Region mode 先计算口述 region 与每个 hypothesis category region 的 IoU，再把
`1 - IoU` 当作 mismatch，通过固定的 `--oral-region-temperature` 转换和归一化。默认尺度为
`0.10`，同样不做逐 trial 自适应。

### Provenance and diagnostics

`oral_mass_probabilities.npz` 保存完整分布以及：

- encoder version、distribution method、uniform prior；
- center sigma / region temperature；
- hypothesis-space version 与 signature；
- 每 trial 的最小距离、log evidence、entropy、effective hypothesis count、最大概率和绝对 fit。

同样的信息以长表写入 `oral_mass_diagnostics.csv`，并传播到依赖 oral mass 的
distribution/target/hit/coverage trial CSV。`oral-based` 直接在口述几何空间评价，不依赖上述
概率编码。
