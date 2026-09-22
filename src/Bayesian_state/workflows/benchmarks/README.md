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


## 0826 零 beta 与有界距离缓存

`benchmark_model_0826_acceleration.py` 使用真实观察序列比较 condition 1/2/3、容量与执行结构、
condition 1 的 P/PM/PH、S129 完整 256 试次及 condition 1/3 自主生成。输出保存数值数组、实际
配置、数据／源码哈希、环境版本和计时；已有输出目录会报错。默认共 16 个小型验证场景，1 job、
每场景 1 次预热加 1 次计时，不是拟合或 recovery。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m src.Bayesian_state.workflows.benchmarks.benchmark_model_0826_acceleration \
  --output-dir /tmp/model0826-fast-check --reference results/model_0826/acceleration_20260917/before
```

修改前的独立数组保存在上述 `before/` 目录；它们没有在优化后重新生成。用 `--jobs 2` 或
`--jobs 4` 检查进程并行结果。用 `--cases c1_M3_chi0 c2_M3_chi0 c3_M3_chi0 --compare-modes
--repeats 3` 交错测量关闭加速、仅零 beta 和完整加速三种模式，并逐项校验输出相等。
这三个模式使用同一数值预算，计时包括重新构建 PF 状态。短任务的进程启动开销不代表正式任务的伸缩率。

缓存持续占用与超大批次绕过检查：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m src.Bayesian_state.workflows.benchmarks.benchmark_distance_cache \
  --output-dir /tmp/model0826-cache-memory
```

默认对两种预算各执行 20,000 个不同输入，并记录 tracemalloc 当前／峰值内存；这些量不是全进程
RSS。每条数值比较要求精确相等。`benchmark_boundary_geometry.py` 已显式关闭距离缓存，以继续
衡量 solver 后端；`profile_model_0826.py` 默认衡量实际启用加速的模型，并保存相关方法代码哈希。

## 后续候选：合并零 beta 的反馈似然列

`probe_zero_beta_likelihood_batch.py` 在独立基准进程内临时替换 likelihood 方法，比较当前正式
实现与探索原型；正式模型入口不会启用它。原型对二值类别反馈、单位立方体刺激的零 beta 列
只调用一次原反馈公式，其余列照常计算，保留完整矩阵及原归一化。部分反馈、非零 beta 等
路径仍使用原实现；这是可行性检查，尚未验证全部生产接口及异常输入契约。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/tmp/model0826_acceleration_numba \
  python -m src.Bayesian_state.workflows.benchmarks.probe_zero_beta_likelihood_batch \
  --output-dir /tmp/model0826-likelihood-batch-probe \
  --reference results/model_0826/acceleration_20260917/before --repeats 3
```

默认 5 个场景、单进程，每个模式各 3 次交错计时（每次另有一次预热）；比较独立保存的旧数组。
另检查 64 组原始／归一化 likelihood 矩阵，包括部分反馈回退和极小非零 beta。
输出 `report.json` 含实际配置、环境版本、数据／源码／参考数组哈希与每次计时，已有目录报错。

似然合并已正式化为 `ContinuousPartition.calc_likelihood`，以上原型目前比较的是**当前正式版**
与历史探索实现，不再代表上一轮的增量倍率。新的消融测量使用下列入口。

## 第二轮消融、完整状态和并行检查

`benchmark_model_0826_round2.py` 比较冻结源码中的上一轮实现、仅似然合并、当前正式版，以及
两个仅在进程内启用的原型（概率缓存、略去部分诊断汇总）。默认 4 个小场景、单进程、每模式
3 次交错计时；不运行拟合。概率缓存原型尚未验证可变 geometry、序列化和完整缓存契约。
轻量汇总原型故意不生成部分 latent summaries，**不能作为正式拟合或模型选择的输出**；仅用来
测量潜在收益，并比较其余概率、状态和重采样输出。正式搜索仍返回原来的完整字段。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/tmp/model0826_acceleration_numba \
  python -m src.Bayesian_state.workflows.benchmarks.benchmark_model_0826_round2 \
  --output-dir /tmp/model0826-round2-new \
  --baseline-source results/model_0826/acceleration_round2_20260917/baseline_source \
  --reference results/model_0826/acceleration_20260917/before --repeats 3
```

源码快照为本次改动前的 4 个文件的文本证据，加载前校验 SHA256；不是另一个维护的模型实现。
`--state-check` 对 condition 1/2/3、门控概率 0/0.35/1、持久执行、较频繁重采样及 condition 1
完整 ancestry 审计，比较每个粒子更新前后 state_dict 的全部数值项（含随机流状态）。
`--parallel --repeats 2` 对固定 8 个任务测量 1/2/4/8 worker；每任务 2 次 PF，另记录启动预热，
所有输出对照独立保存的旧数组。每个协议须指定新的输出目录。

## 正式并行预算的启动验证

`check_parallel_runtime.py` 用进程屏障检查请求的 worker 数量是否实际启动，再让每个
worker 执行 8 个 trial、16 粒子的短 PF。6 项模型数组逐元素对照旧数值参考；同时记录
线程库、线程环境、嵌套调用的进程预算，以及版本和文件哈希。它不测完整拟合的加速比。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/tmp/model0826_acceleration_numba \
  python -m src.Bayesian_state.workflows.benchmarks.check_parallel_runtime \
  --workers 128 --output-dir /tmp/model0826-workers-new \
  --reference results/model_0826/acceleration_20260917/before/c1_M3_chi0.npz
```

默认 `--workers 2`；显式检查 128 要求当前可用 CPU 至少为 128。输出目录必须不存在。
正式入口的预算和作用范围见
[128 核并行策略](../../docs/maintenance/model_0826_parallel_policy_20260917.md)。

## 三个 condition 的拟合工期计时输入

`estimate_model_0826_fit_time.py` 先运行一次 32-trial 小检查，再对 S129/S229/S301 的完整
序列分别测量 R16/R64、M3χ0/M5χ1，共 12 个单 seed 固定参数任务。它调用正式
`evaluate_state_model_run`，保留真实感知参数加载和评分开销，但不进行参数搜索。
线程约束与正式执行相同；进程预算 128，当前任务数为 12。输出 `timings.json` 保存
实际配置、试次数、计时、版本及数据/源码哈希；输出目录必须不存在。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/tmp/model0826_acceleration_numba \
  python -m src.Bayesian_state.workflows.benchmarks.estimate_model_0826_fit_time \
  --output-dir /tmp/model0826-fit-timing-new
```

这些单任务时间不是完整拟合时间，也不是 128-worker 吞吐实测。估计完整工期还需结合
实际搜索候选批次、seed 数、缓存命中、被试试次数和当前串行依赖。
