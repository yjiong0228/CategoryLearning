# Model0826 模块与参数恢复设计规范

## 1. 科学问题

本分析检验三个层次的问题：

1. 行为数据能否区分 P、PM、PH、PMH 四种架构；
2. 在 PMH 架构正确时，能否恢复 workspace-mixture 与 persistent single-rule readout 的被试级离散结构 `chi`；
3. 在每条自主生成的选择轨迹分别拟合时，能否恢复 Model0826 的连续和离散被试参数。

粒子滤波只用于对不可见的随机内部路径做数值积分。生成数据是一条条独立的自主行为轨迹，任何两条生成轨迹都不先平均；拟合候选内部的多个 PF seeds 先逐试次平均选择概率，再计算 NLL。

## 2. 冻结模型与禁止变更项

正式恢复使用 `configs/model_struct/pmh_model_cond1_0826.yaml` 和 `manuscript/model_0826.tex` 中一致的机制：P 固定、M 为现有 DualMemory、H 为现有 nested feedback accumulator 与 `similarity_transport`、beta 对 `active_hypotheses` 动态更新、readout power 为 1、strategy confidence gain 为 0、output lapse 为 0。beta 上下界固定为 `[0.1, 25]`。

不加入 misconception capture、rule commitment 或其他新机制；不把 mass-preserving transport 当作主模型；不在恢复过程中修改随机种子、trial 顺序或参数定义。

## 3. 数据模板与试次范围

只使用 condition 1 的三个真实任务日程：

| 被试模板 | 有效试次数 | 使用范围 |
|---|---:|---|
| 101 | 320 | 全部 320 试次 |
| 111 | 320 | 全部 320 试次 |
| 118 | 256 | 全部 256 试次 |

模板提供刺激顺序、正确类别和外部固定的被试知觉参数。真实 Task2 choice 不进入合成数据生成。模型自主采样 choice，任务环境据此产生 feedback。不同模板不截断为共同长度，也不补齐 118。

模块恢复的参数选择使用因果 sequential holdout：每个模板前 70% 为 optimization prefix，后 30% 为 evaluation suffix，即 101/111 为 224/96，118 为 179/77。模型仍按完整顺序运行；后缀每个 one-step-ahead 预测只使用此前已出现的合成历史。参数恢复使用完整序列评分，因为目标是已知真值与估计值的对应，而不是架构泛化。

## 4. 可执行架构单元

四个 cell 共用相同 P、动态 beta、readout 和固定参数：

| Cell | Memory | Hypothesis search | 自由参数 |
|---|---|---|---|
| P | `BayesianMemoryModule` | 无 H，29 条规则始终 active | `beta_0, eta_plus, eta_minus` |
| PM | `DualMemoryModule` | 无 H，29 条规则始终 active | `gamma, beta_0, eta_plus, eta_minus` |
| PH | `BayesianMemoryModule` | Model0826 H | `(M,chi), E_C, delta_E, g_0, c_A, c_G, beta_0, eta_plus, eta_minus` |
| PMH | `DualMemoryModule` | Model0826 H | 上述 H 参数加 `gamma` |

P/PM 没有 workspace，因此没有 `M` 或 `chi`。PH/PMH 的 `chi=0` 是 active-hypothesis belief mixture，`chi=1` 是 persistent executed-rule readout。

## 5. PF 数值预算校准

正式恢复前先生成六条冻结校准轨迹：三个模板各生成一条 `chi=0` 和一条 `chi=1` 的 PMH anchor 轨迹，全部使用完整试次。所有 R/B 设置只重评分这六条轨迹，不重新生成。

每条轨迹比较同一个八候选 bank：anchor 的 `chi=0/1` 两点；把 `gamma` 单独改为 0.50 的 `chi=0/1` 两点；把 `c_A,c_G` 单独改为 0 的 `chi=0/1` 两点；把 `beta_0,eta_plus,eta_minus` 单独改为 `1.00,0.01,0.03` 的 `chi=0/1` 两点。其他参数保持 anchor 值。这个 bank 固定用于所有 R/B 和 A/B ensemble，数值校准不运行超参数搜索。

按顺序检查：

1. `R=16,32,64`，seed ensemble A 的同一前 4 个逻辑 seeds；
2. `R=64` 下比较 A 的 `B=4,8`；
3. `R=64,B=8` 的独立 ensemble B；
4. 若任何 gate 失败，追加 `R=128,B=16` 的 A/B 检查。

候选内始终先平均 B 个逐试次概率再算 NLL。冻结满足全部门槛的最小 R/B：

- 相邻 R 的候选 NLL 排名 median Spearman `>=0.90` 且最差 `>=0.70`；
- 相邻 R 的赢家一致率 `>=5/6`；
- A/B 独立 ensemble 的赢家一致率 `>=5/6`；
- 相邻设置的逐试次选择概率 RMSE 中位数 `<=0.015`；
- seed-mean 选择概率的全 trial 第 95 百分位 MCSE `<=0.02`。

若 `R=128,B=16` 仍失败，停止正式恢复并报告数值不可稳定；不得用未通过的预算产生科学恢复结论。

## 6. 模块恢复

生成真值为每个 cell、每个模板、3 条独立轨迹，共 `4 × 3 × 3 = 36` 个数据集。生成参数固定为中等、可辨识值：

```text
gamma=0.80                    # 仅 PM/PMH
M=3, chi=0                   # 仅 PH/PMH
E_C=0.25, delta_E=0.80       # 仅 PH/PMH
g_0=0.20, c_A=2.00, c_G=0.50
beta_0=5.00, eta_plus=0.04, eta_minus=0.15
```

每个合成数据集都由 P、PM、PH、PMH 四个候选独立拟合。搜索和 Hyper-CD 2.0 shortlist 的独立高精度重评分都只使用 prefix objective；每个架构据此冻结一套参数。随后用另一组未参与参数选择的 PF seeds 运行固定参数模型，并且只在 suffix 计算 held-out score。以 held-out total choice NLL 最小者为 winner；`delta total NLL <= 2` 定义 near-best set。

主要输出为 4×4 confusion matrix、逐真值 cell 的 exact winner rate、true-cell near-best coverage、held-out delta NLL 和生成轨迹 accuracy 分布。预注册通过门槛：整体 exact cell recovery `>=0.70`，每个真值 cell `>=0.50`，true-cell near-best coverage `>=0.85`，任一固定错误 cell 不得吸收超过全部数据集的 30%。未通过时按混淆模式缩减模块结论，而不是改门槛或挑选轨迹。

## 7. 参数与 readout 结构恢复

只在 PMH 下运行，共 40 条独立合成轨迹。

### 7.1 局部中心

10 条轨迹使用当前 sub101 pattern-check 的冻结点：

```text
M=4, chi=1, gamma=0.97
E_C=0.02, delta_E=0.80
g_0=0.20, c_A=0.00, c_G=0.00
beta_0=5.00, eta_plus=0.04, eta_minus=0.07
```

模板分配为 101 四条、111 三条、118 三条；每条均使用对应模板全部试次。

### 7.2 全局对比剖面

六个支持内剖面各生成 5 条轨迹，共 30 条：

| Profile | M | chi | gamma | E_C | delta_E | g_0 | c_A | c_G | beta_0 | eta+ | eta- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 2 | 0 | 0.25 | 0.10 | 0.25 | 0.05 | 0.25 | 0.10 | 1.0 | 0.01 | 0.03 |
| C2 | 5 | 1 | 0.90 | 0.50 | 1.60 | 0.70 | 4.00 | 0.75 | 10.0 | 0.16 | 0.60 |
| C3 | 3 | 0 | 0.50 | 0.25 | 3.20 | 0.10 | 6.00 | 1.00 | 2.5 | 0.32 | 1.00 |
| C4 | 4 | 1 | 0.80 | 0.75 | 0.00 | 0.40 | 0.00 | 0.00 | 20.0 | 0.005 | 0.01 |
| C5 | 2 | 0 | 0.00 | 0.10 | 0.4795730802618863 | 0.00 | 2.00 | 1.00 | 0.5 | 0.08 | 0.30 |
| C6 | 5 | 0 | 0.80 | 0.02 | 0.80 | 0.20 | 0.50 | 0.25 | 10.0 | 0.02 | 0.15 |

每个 profile 的模板按循环分配：C1/C4 为 `[101,111,118,101,111]`，C2/C5 为 `[111,118,101,111,118]`，C3/C6 为 `[118,101,111,118,101]`。因此 30 条全局轨迹对三个模板各分配 10 条。连同局部中心，40 条数据中 `chi=0` 与 `chi=1` 各 20 条。

每条轨迹单独运行完整 Hyper-CD 2.0；不把 40 条 choice 合并，不在生成轨迹间平均。所有参数真值均来自 provisional fitted support，并且每个真值都是拟合时可精确到达的候选点。

### 7.3 指标与判定

- `M` 和 `chi`：confusion matrix、exact recovery、Wilson 95% interval；
- 连续参数：bias、MAE、RMSE、Spearman、按候选支持跨度归一化的 MAE；
- `delta_E/c_A/c_G`：零点与正值的 balanced accuracy，以及正值条件下的误差；
- 全参数向量：真值点是否进入 final-rescore `delta total NLL <= 2` near-best set；
- 参数间误差相关矩阵：定位补偿关系，不把相关误差解释为认知耦合。

readout 结构通过门槛为 `chi` exact recovery `>=0.70` 且 near-best coverage `>=0.85`。连续参数逐一判定：Spearman `>=0.60`、normalized MAE `<=0.20`、near-best coverage `>=0.80`；零点参数还要求 zero/positive balanced accuracy `>=0.70`。不满足门槛的参数应固定、合并或缩窄支持，不能作为可靠的个体差异参数报告。

## 8. 随机种子、缓存与失败处理

generation seed 由 analysis id、cell/profile、subject 和 replicate 稳定派生。候选内 PF seeds 对所有拟合候选配对；搜索 seed、final-rescore seed、数值 ensemble A/B 互不重叠。每条合成轨迹保存 choice、feedback、stimulus/category fingerprint、生成配置和 seed。

运行支持按 dataset 原子缓存和 resume。缓存命中前核对模型/config/data/code fingerprint。非有限概率、概率未归一化、trial 数不符或配置哈希不一致立即失败。已完成数据集不会因重启重新生成；`--force` 也只能写入新的显式 output directory，不能覆盖 `results/` 中的既有产物。

## 9. 输出目录

新结果统一写入：

```text
results/model_0826/recovery_v1/
  manifest.json
  numerical_calibration/
  module_recovery/
    synthetic/
    search/
    final_rescore/
    fit_scores.csv
    recovery_summary.csv
    module_recovery_overview.png
  parameter_recovery/
    synthetic/
    search/
    final_rescore/
    fit_scores.csv
    parameter_summary.csv
    readout_confusion.csv
    parameter_recovery_overview.png
  final_report.json
```

只生成 PNG 图。manifest 保存源 YAML 与代码哈希、三名模板的实际 trial 数、最终冻结 R/B、全部 seeds、运行阶段和完成状态。

## 10. 实施与科学完成标准

先完成 Hyper-CD 2.0 自动化测试，再实现 0826 通用参数空间 loader、四 cell builder 和恢复 runner。依次运行全试次 smoke、PF 校准、36 条模块恢复和 40 条参数恢复。只有当所有预声明数据集完成、数值 gate 通过、汇总文件与图可由缓存重建、且结果通过概率/形状/seed 审计后，才能给出最终模型结构和参数可识别性结论。
