# 边界距离求解器：精确 active-set 实现与测量（2026-10-13）

本文件记录 `BoundaryGeometry` 距离求解器的实现变更与测量结果。原实现用 100 轮 Dykstra
投影迭代逼近；现在只有一条链路：**两阶段精确求解**。可选后端参数
`boundary_dykstra_backend` 已删除，旧的 Dykstra 内核模块已移除。

测量全部在 `/opt/anaconda3/bin/python`（Python 3.11.10、NumPy 1.26.4、Numba 0.60.0，
即 `validation_environment_20260912.md` 记录的环境）完成，数值库线程为 1。

## 1. 距离计算在模型中的位置

```text
ContinuousPartition.calc_likelihood()              每个 trial × 每个 particle 一次
  └─ ObservationLikelihood.process()
      └─ ContinuousPartition.get_category_probabilities()   每个候选规则一次
          └─ BoundaryGeometry.category_distances()          ExactDistanceCache 在此拦截
              └─ _category_distances_uncached()
                  └─ distances_to_category() / distances_to_polytope()
                      └─ _distances_exact() / fused_single_stimulus()
```

调用形状是关键事实：condition 1、16 粒子、32 试次的真实运行中，
`_distances_dykstra` 曾被调用 4,728 次，其中 2,107 次真正进入内核，
**平均每次 1.0 个刺激点、1–2 个 polytope**。即该函数几乎总是以最小批量被调用。

## 2. 算法

类别区域是凸多面体 `A x ≤ b` 与单位立方体的交。对模型里的 polytope 做结构普查：

| 空间 | polytope 数 | 1 个半空间 | 2 个 | 3 个 | 最多 |
|---|---:|---:|---:|---:|---:|
| 4 维 / 2 类（29 规则） | 68 | **58（85%）** | 10 | 0 | 2 |
| 4 维 / 4 类（116 规则） | 464 | 0 | **456（98%）** | 8 | 3 |
| 2 维 / 2 类（7 规则） | 17 | 14 | 3 | 0 | 2 |

约束极少，因此最优解的活跃集可以被完全枚举。求解分两阶段：

**第一阶段（廉价）**：枚举 polytope 自身半空间的子集，把刺激投影到该仿射子空间，
再裁剪进单位立方体，取最近且满足全部约束的候选。向立方体内裁剪只会减小 `A z`，
所以被接受的候选一定可行，其距离对其活跃集是精确的。

**第二阶段（精确，仅在必要时）**：枚举**全体约束**（polytope 半空间 + `2·n_dims` 个单位盒面）
中不超过 `n_dims` 个线性无关行的活跃子集，对每个子集解等式系统并检验可行性。

必须把单位盒面一并纳入枚举：只对半空间求解再裁剪，会在最近点落在盒面上时沿单一方向
移动而越过角点。例如 `x1 − x2 ≤ −0.1` 到 `(0.041, 0.017)` 的最近点是 `(0, 0.1)`，
此处 `x1 = 0`（盒面）与半空间**同时**活跃；只裁剪得到距离 0.983，而正确答案是 0.09299。

两阶段在最优点上是同一件事，所以结果与由哪一阶段回答无关。实测第一阶段即可解决绝大多数
刺激，只有极少数需要第二阶段：

| 空间 | 测试点数 | 回退到第二阶段 | 回退比例 |
|---|---:|---:|---:|
| 2 维 / 2 类 | 37,026 | 160 | **0.43%** |
| 4 维 / 2 类 | 148,104 | 1,735 | **1.17%** |
| 4 维 / 4 类 | 1,010,592 | 0 | **0.00%** |

可行性检验使用独立的较宽容差（`MIN_FEASIBILITY_TOLERANCE = 1e-7`），因为任务空间含退化
零宽 polytope（边界平面过原点），此处"在平面上"与"在内部"在几何容差下不可区分，
必须解析为距离零。

## 3. 正确性验收

| 对照 | 空间 | 最大绝对偏差 |
|---|---|---:|
| vs `kkt_active_set_projection`（独立精确方法） | 2D/2 类 | **0.0** |
| vs 同上 | 4D/2 类 | **1.7e-16** |
| vs 同上 | 4D/4 类 | **2.2e-16** |
| vs SLSQP oracle（通用约束优化器） | 2D/2 类 | **4.8e-15** |
| vs SLSQP oracle | 4D/2 类 | **2.2e-14** |
| vs SLSQP oracle | 4D/4 类 | **3.7e-15** |

覆盖每个 polytope 的全部 1,666 个测试刺激（均匀随机、近边界、近约束、盒顶点、整数角点），
**无一个返回非有限值**。

新增回归测试 `tests/bayesian_state/hypothesis_space/test_active_set_acceleration.py`
（11 项）固定这些性质：与 KKT 方法逐 polytope 一致、与 SLSQP oracle 一致、单点融合内核与
批量路径逐位相等、越界刺激被拒绝、两个候选族都被缓存、缺少 Numba 时报错、warmup 报告可用性。

## 4. 速度

相对被移除的 Dykstra 循环（同一 polytope、同一刺激、相同 Python 调用路径）：

| 空间 | 历史 Dykstra | 现在 | 倍率 |
|---|---:|---:|---:|
| 4 维 / 2 类（68 polytope，单点） | 15.93 µs/polytope | **14.47 µs** | **1.10×** |
| 4 维 / 4 类（464 polytope，单点） | 17.97 µs/polytope | **15.42 µs** | **1.17×** |

纯内核层面（只计投影本身）的倍率更高，但调用点的 Python/NumPy 开销占主导，因此上表是
调用点实测值。两阶段设计是取得这一倍率的前提：若每次都对全体约束做完整枚举，
调用点会比 Dykstra 慢 2–3 倍。

端到端仍需分开说：**整个模型的变化很小**，因为距离层只占一次 particle-filter 运行的
10–13%，而其中 Numba 内核只占约 2%：

| 模块 | 占端到端 |
|---|---:|
| Python 解释器与 C 层 | 21.8% |
| `model/modules`（memory/beta/transition/readout） | 20.2% |
| `numpy.fromnumeric` 包装层 | 16.5% |
| `hypothesis_space/geometry`（含距离求解全部） | 10.4% |
| `hypothesis_space/observation_model` | 8.8% |

单点调用链分项计时：NumPy 外侧预筛归约约 6.7 µs，Numba 求解约 4.2 µs。融合内核把这次
归约并入编译代码，是保留速度优势的必要部分。

本次变更的主要收益是**精确性与代码简化**（删除约 130 行迭代内核、一个模块与全部后端分支，
`boundary.py` 从 525 行降到 427 行），速度上相对旧实现有小幅改善。要获得数量级提升，
需要处理占 58% 的"每 trial 每 particle 的 Python 对象操作"，那是一次独立的架构改造。

## 5. 越界刺激的处理

类别区域只在单位立方体内定义。感知模块已把采样刺激截断到 `[0, 1]`
（`model/modules/perception.py`），因此生产路径永远给出立方体内的点。

`as_stimuli` 现在校验刺激有限且位于 `[0, 1]`，越界即报错。此前越界输入会走
"从原始点出发做 100 轮投影"的路径，其输出并非有定义的最优解（对同一几何，Dykstra 在
越界输入上与 SLSQP oracle 最大相差 0.33）。

## 6. 冻结数值参考的更新

本次求解器替换使期刊 `pre_shared_core.npz` 参考中 232 个数组里的 121 个发生**末位差异**：

- 最大绝对差 **1.78e-15**，最大相对差 **2.96e-16**；
- 没有数组超出 1e-12，即差异完全限于浮点舍入；
- `arrays changed: 121 / 232`，两个阶段（完整枚举版与两阶段版）产生**完全相同**的差异，
  互相印证两者的最优点一致。

参考通过一次性审计脚本重新生成：先逐数组比较新旧结果，仅在全部差异 ≤ 1e-12 时才写出新
文件并更新 `pre_shared_core.json` 中的 SHA256；否则中止。这是求解器的数值行为变更，
不是"为了让测试通过而重生成参考"。

## 7. 变更清单

| 文件 | 变更 |
|---|---|
| `geometry/active_set_acceleration.py` | 重写为唯一求解器（两阶段 + 融合单点内核） |
| `geometry/boundary.py` | 删除 Dykstra 后端分支、`_project_dykstra`、`_project_to_halfspace`；单一 `_distances_exact` 链路 |
| `geometry/dykstra_acceleration.py` | **删除** |
| `geometry/__init__.py` | 导出改为 `active_set_numba_available` / `warmup_active_set_numba` |
| `geometry/stimuli.py` | `as_stimuli` 校验有限性与单位立方体范围 |
| `observation_model/continuous_partition.py` | 删除 `boundary_dykstra_backend` 参数 |
| `simulation/provenance.py` | `encoding.boundary_solver` 固定记录求解器名 |
| `model/assembly.py` | 原型校验列表中移除该键 |
| `CategoryLearning_codes/.../geometry/` | shim 同步改名 |
| `CategoryLearning_codes/Bayesian_model/tests/fixtures/pre_shared_core.{npz,json}` | 审计后重生成 |
| 文档与测试 | `hypothesis_space/README.md`、本文件、`test_boundary_geometry.py`、`test_active_set_acceleration.py`、`test_encoding_recompute.py`、`test_model_0923_acceleration.py`、`test_feedback_mode_guards.py` |

## 8. 复现命令

```bash
env NUMBA_CACHE_DIR=/tmp/dykstra_numba PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/anaconda3/bin/python -m pytest -q \
  tests/bayesian_state/hypothesis_space/test_active_set_acceleration.py
```

`kkt_active_set_projection` 仍可通过 `boundary_distance_method` 选择，用于与主求解器做
独立的数值交叉验证。

## 9. 已知的既有失败（与本次变更无关）

`CategoryLearning_codes/Bayesian_model/tests/test_migration.py::test_shared_core_has_no_journal_dependency`
在本次改动前的 `HEAD` 上就已失败：`src/Bayesian_state/evaluation/model_0923_fit_source.py`
第 71 行引用了字符串 `'CategoryLearning_codes/figures/'`。已验证与本求解器变更无关。
