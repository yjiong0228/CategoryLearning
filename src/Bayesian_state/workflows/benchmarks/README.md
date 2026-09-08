# 性能检查

benchmark_boundary_geometry.py 比较连续边界投影后端的冷启动与重复调用速度，
不属于拟合或模型准确性评价。原路径为 benchmarks/benchmark_boundary_geometry.py。

从仓库根目录查看参数：

```bash
python -m src.Bayesian_state.workflows.benchmarks.benchmark_boundary_geometry --help
```

默认检查多种类别结构和 512 个刺激，运行前按需要减小 --n-stimuli。
