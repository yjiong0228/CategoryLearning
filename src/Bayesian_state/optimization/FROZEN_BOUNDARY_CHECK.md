# 三人冻结拟合之后：一次有上限的范围检查

本轮仅检查S122、S222、S315已经发现的人工边界，并复用S222/S315缓存做一次重采样敏感性分析。
候选、预算和判断规则均在新评分前固定；不是重跑完整参数搜索，不改正式默认范围。

## 候选和计算上限

配置：`configs/exp123/specific_models/model_0826_frozen_boundary_check.yaml`。
每人保留原来的全部8个入围候选，新外侧候选分别为14、10、7个，总计55个点。
全部点使用一档R256×B32新种子评分，即最多1760次完整PF；所有原始试次递推，首试次不计分。
最多128进程，每进程一个数值线程，仅一层并行，不为占核额外加点。无第二档、无条件追加和自动重搜。

| 被试 | 完整试次 | 内侧/外侧点 | 主要检查 |
|---|---:|---:|---|
| S122 | 256 | 8/14 | M与gamma；E_C与delta_E；正向更新率；跨组联合变化 |
| S222 | 832 | 8/10 | E_C、正负更新率的单独及联合变化 |
| S315 | 704 | 8/7 | 从原代表及不同近优点扩大M；从记忆触边备选联合改变M/gamma |

扩展值为M=6/8、gamma=.985/.995、E_C=.01/.005、delta_E=4.8/6.4、
eta_plus=.0025/.001、eta_minus=.005/.0025。它们用于有限外侧探针；近、远两档检查对外推步长的依赖，
不声称这些取值就是最终推荐范围。所有新增点的chi仍为0；声明支持含chi=1不代表本轮测试了它。
更新率保持正值，零更新属于另外的消融。自然零边界和二元chi不作扩范围理由。

降低E_C分别检查固定delta_E与固定E_E两条路径；后者推导delta_E并记入本次支持。
初始事件概率仍与E_C绑定，未增加一个自由参数。联合探针也从不同近优解出发，减少只围绕单一代表检查的局限。
这不是重新优化其余所有参数的profile分析，也不能排除未检查的外侧组合。

## 如何避免把对照选差了当成扩范围收益

原8个候选全部保留，不能根据新分数缩小对照组。对每个原候选，检查它相对全部外侧候选的最大平均NLL损失。
每个比较复用共享的完整PF种子bootstrap和“先平均概率再取log”，不将trial当独立重复。
三人各分配alpha=.01，各自再平分给8个内侧比较，每项.00125；bootstrap在运行前设为60000。
各内侧比较使用同一套成组重采样权重。该分配防止从多个内侧对照里挑一次有利结果；仍是有限粒子下的近似数值诊断，
不是整个多轮研究的严格总体误判率保证。

- 至少一个原候选的损失上界≤.005：本次固定候选范围内，没有必须扩范围的实质收益证据。
- 所有原候选的损失下界均>.005：本次外侧候选相对全部8个原候选有实质收益，支持考虑共同范围调整。
- 其他情况：保留不确定。不能把“外侧没有显著更好”直接当作原范围合理。

报告中的最佳内侧损失界取8项对应上下界的最小值，始终保留逐项结果。
新数据中分数最佳的点只作描述，不成为当场换选后的“独立通过”参数。
上述结论只适用于这组有限候选，不能证明全局最优、原范围充分或心理参数唯一。
原三人拟合状态保留为原记录；本轮 `whole_fit_status=not_reassessed`。

## 缓存敏感性检查

S222/S315各仅做一次60000次bootstrap：原8个候选、原提名、R256×B32概率、原alpha=.00625、
容差.005及bootstrap种子全部保持原样。只提高重采样次数，新增PF为0。
原6000次结论与本次敏感性结果并列保存；即使数值跨过门槛，也不改写冻结核查的验收状态。
不重复尝试不同种子或重采样次数寻找通过。这两项是事后数值敏感性描述，不参与边界主判断。

## 运行与结束

```bash
python -m src.Bayesian_state.workflows.analysis.probe_model_0826_frozen_boundaries --dry-run
# 新目录，只读取缓存，不运行PF
python -m src.Bayesian_state.workflows.analysis.probe_model_0826_frozen_boundaries --cached-only --output-dir results/model_0826/boundary_cache_new
# 32试次、R2×B2、单进程，验证S122流程
python -m src.Bayesian_state.workflows.analysis.probe_model_0826_frozen_boundaries --smoke --output-dir results/model_0826/boundary_smoke_new
# 已授权完整检查：使用单线程数值库和独立新目录
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/model0826_outer_numba \
  python -m src.Bayesian_state.workflows.analysis.probe_model_0826_frozen_boundaries \
  --output-dir results/model_0826/boundary_full_new
```

`--resume`只能用于相同配置、输入、源码和环境；缓存检查目录不能当作完整计算目录继续追加。
既有源码内容没有更改；新增工作流文件仍会改变“当前所有源码”的清单，历史入口严格resume时应使用当时版本。
本轮只读核验并使用历史输入与缓存，不修改历史哈希或研究结果。

计算完成后自动核验全部PF的身份、形状、概率、试次顺序、掩码和收据校验和，实际检查完成后的resume不新增PF，生成报告。
有改善、不确定或未见实质收益都按本次上限结束，不自动改配置、扩被试、再拟合或进入exp4/exp5。
结果目录：[本轮记录](../../../results/model_0826/frozen_boundary_check_20260920/README.md)。
