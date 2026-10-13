# 边界距离求解器：测量结果与精确 active-set 后端（2026-10-13）

本文件记录一次针对 `BoundaryGeometry` Dykstra 距离计算的加速评估与实现。结论与最初
假设不同：**距离计算不是当前模型的主要瓶颈，替换求解器内核不能显著加快端到端运行。**
因此新增的精确求解器作为显式可选后端保留，生产默认仍是历史 Numba 循环。

测量全部在 `/opt/anaconda3/bin/python`（Python 3.11.10、NumPy 1.26.4、Numba 0.60.0，
即 `validation_environment_20260912.md` 记录的环境）完成。数值库线程为 1。

## 1. 距离计算在模型中的位置

调用链是单一入口、逐层收窄的：

```text
ContinuousPartition.calc_likelihood()              每个 trial × 每个 particle 一次
  └─ ObservationLikelihood.process()
      └─ ContinuousPartition.get_category_probabilities()   每个候选规则一次
          └─ BoundaryGeometry.category_distances()          ExactDistanceCache 在此拦截
              └─ _category_distances_uncached()
                  └─ distances_to_category() / distances_to_polytope()
                      └─ _distances_dykstra()
                          └─ dykstra_distances_numba()      历史 100 轮投影循环
```

调用形状是关键事实：在 condition 1、16 粒子、32 试次的真实运行中，
`_distances_dykstra` 被调用 4,728 次，其中 2,107 次真正进入 Numba，
**平均每次调用 1.0 个刺激点、1–2 个 polytope**。即该函数几乎总是以最小批量被调用，
每次都要支付一遍 Python/NumPy 的固定开销。

## 2. 实测：距离层占 10–13%，Numba 内核占约 2%

用「返回逐位相同数组的无界精确缓存」替换距离计算，对比端到端 PF 耗时（严格上界，不是近似）：

| 场景 | 基线 | 距离耗时 | 占基线 |
|---|---:|---:|---:|
| condition 1，16 粒子 × 32 试次 | 0.555 s | 0.070 s | 12.6% |
| condition 1，8 粒子 × 64 试次 | 0.550 s | 0.065 s | 11.8% |
| condition 1，4 粒子 × 256 试次 | 1.091 s | 0.113 s | 10.3% |

cProfile 热点归属（1.05 s，157 万次函数调用）：

| 模块 | tottime | 占比 |
|---|---:|---:|
| Python 解释器与 C 层 | 0.233 s | 21.8% |
| `model/modules`（memory/beta/transition/readout） | 0.216 s | 20.2% |
| `numpy.fromnumeric`（`sum`/`any`/`all` 包装层） | 0.176 s | 16.5% |
| `hypothesis_space/geometry`（含 Dykstra 全部） | 0.112 s | 10.4% |
| `hypothesis_space/observation_model` | 0.094 s | 8.8% |

在 geometry 的 10.4% 之内再拆分：Numba 内核本体约 3.0 µs/次调用，Python/NumPy 包装
约 8.7 µs/次调用。**内核只占总运行时间约 2%（0.022 s / 1.05 s）**，因此即使把内核
优化到零耗时，端到端上限也只有 2%。

单点调用链的分项计时（4 维、2 类）：

| 步骤 | 耗时 |
|---|---:|
| Numba 求解（1 点） | 4.19 µs |
| **NumPy 外侧预筛 `any(... > tol)`** | **6.73 µs** |
| 违规矩阵乘法 `stimuli @ constraints.T` | 1.61 µs |
| `flatnonzero` | 9.17 µs |
| 结果数组分配 | 0.36 µs |
| `category_distances` 命中路径 | 2.82 µs |

外侧预筛的归约比数值求解本身更贵。这是距离层真正的开销来源，而不是投影循环。

## 3. 结构普查：现有求解器对问题规模过度设计

对真实假设空间逐 polytope 统计其半空间约束数：

| 空间 | polytope 数 | 单约束 | 2 约束 | 3 约束 | 最多约束 |
|---|---:|---:|---:|---:|---:|
| 4 维 / 2 类（29 规则） | 68 | **58（85%）** | 10 | 0 | 2 |
| 4 维 / 4 类（116 规则） | 464 | 0 | **456（98%）** | 8 | 3 |
| 2 维 / 2 类（7 规则） | 17 | 14 | 3 | 0 | 2 |

每个 polytope 只有 1–3 个半空间约束，却固定执行 100 轮 Dykstra。

## 4. 已实现的精确 active-set 求解器

新增 `src/Bayesian_state/hypothesis_space/geometry/active_set_acceleration.py`。
把距离写成对 `A` 的行的子集枚举：对每个子集 `K` 求解 `A_K z = b_K`，把结果裁剪进
单位立方体，保留所有满足 `A z <= b` 的候选中最近的一个。向立方体内裁剪只会减小 `A z`，
因此不会让已接受的候选变为不可行；立方体内的最优解就是最近的被接受候选。

- **精确**：与历史 Numba 循环在真实 polytope 上逐元素一致，最大偏差 2.2e-16（2 类）
  与 3.3e-16（4 类）。
- **内核更快**：批量 256 点时约 16×，单点时约 5×；融合版本（把外侧预筛并入内核）
  在单点调用上相对现有路径约 10–12×。
- **可选**：通过 `partition.kwargs.boundary_dykstra_backend: active_set_numba` 显式启用。
- **回退**：退化零宽 polytope（边界平面过原点）和立方体外刺激可能没有可行的半空间子集；
  这些行继续由历史 Numba 循环求解，保证对旧后端接受过的所有输入仍然有定义。

## 5. 为什么默认仍是历史求解器

融合内核把单点 polytope 距离从约 16 µs 降到约 1.3–1.9 µs（10–12×），但在真实 PF 上
反复交错的 A/B 只得到 0.97–1.02×：

| 场景（32 粒子 × 64 试次） | 历史 Numba | active-set | 倍率（min） |
|---|---:|---:|---:|
| condition 1 | 2.718 s | 2.713 s | 1.002 |
| condition 2 | 1.875 s | 1.904 s | 0.985 |
| condition 3 | 2.079 s | 2.121 s | 0.980 |

这与第 2 节的上界一致：距离层只占约 11%，而内核只占其中约 2%，所以 10× 的内核提速
在端到端只应带来约 1–2%，实测落在噪声范围内。默认切换会改变生产数值与已记录的
provenance，却没有可测收益，因此 `auto` 保持解析为 `numba`。

## 6. 距离层真正可压缩的部分

如果后续要继续优化这一层，可压缩的是第 2 节表里的非内核项，而非投影循环：

1. 外侧预筛的归约（6.7 µs/次）。融合内核已把单点情形并入 Numba。
2. `distances_to_polytope` 在调用点重复计算 `constraints`/`bounds` 广播与
   `flatnonzero`。可缓存或并入内核。
3. `ExactDistanceCache` 每次命中都要构造 `stimuli.tobytes()` 键；单点情形可改用坐标标量。

这些合计是距离层约 11% 中的大部分，量级仍在 5–8% 端到端。真正的数量级提升需要处理
占 58% 的「每 trial 每 particle 的 Python 对象操作」（`model/modules`、
`numpy.fromnumeric` 包装层、解释器开销），那是一次独立的架构改造。

## 7. 验证

- `tests/bayesian_state/hypothesis_space/test_active_set_acceleration.py`（8 项）：
  三个空间上逐 polytope 与历史求解器一致（atol 1e-12）、单点融合路径与批量路径
  逐位相等、`auto` 保持历史后端、显式可选、候选缓存、缺少 Numba 时报错。
- 端到端数值等价：condition 1/2/3 的 PF 输出在 52/52/54 个公开数组上逐项比较，
  最大绝对差 2.2e-16 / 2.2e-16 / 3.3e-16，无差异数组。
- 既有回归：`test_boundary_geometry.py`、`test_distance_acceleration.py` 及完整
  正式测试目录通过。

## 8. 复现命令

```bash
env NUMBA_CACHE_DIR=/tmp/dykstra_numba PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q \
  tests/bayesian_state/hypothesis_space/test_active_set_acceleration.py
```

显式启用新后端（不改变默认配置）：

```yaml
partition:
  kwargs:
    boundary_dykstra_backend: active_set_numba
```
