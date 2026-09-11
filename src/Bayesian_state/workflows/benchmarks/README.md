# 性能检查

benchmark_boundary_geometry.py 比较连续边界投影后端的冷启动与重复调用速度，
不属于拟合或模型准确性评价。原路径为 benchmarks/benchmark_boundary_geometry.py。

从仓库根目录查看参数：

```bash
python -m src.Bayesian_state.workflows.benchmarks.benchmark_boundary_geometry --help
```

默认检查多种类别结构和 512 个刺激，运行前按需要减小 --n-stimuli。

`profile_model_0826.py` 对 condition 1 的真实观察序列记录冷启动、独立于
profiler 的重复计时、热点调用与逐试次数值输出。默认仅 S129 前 32 试次、16 粒子；
不是完整拟合验证。输出目录必须不存在；优化前后使用相同参数和机器负载，
并通过 `--reference <旧目录>/arrays.npz` 检查数值完全一致。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.Bayesian_state.workflows.benchmarks.profile_model_0826 --output-dir /tmp/model0826-profile-new
```

2026-09-08 的热点检查发现 likelihood 重复进行单区域距离归并和无部分反馈时的
邻接类别汇总。当前实现跳过这两类无贡献计算，保留原边界投影、概率归一化、
随机数顺序与部分反馈公式。此优化不扩展 0826/PF 的 condition 支持范围。
