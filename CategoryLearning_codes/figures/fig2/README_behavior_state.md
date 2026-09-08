# Fig2b/c 现有结果预览

当前图：`../outputs/fig2/fig2bc_v3/fig2bc_behavior_state_draft.png`。

## 图的任务与边界

本次为 Python/matplotlib quantitative grid，183 × 202 mm、450 dpi、PNG。
目标：展示一个现有案例的在线选择解释、规则信念与口述对照，以及预测偏差；不预设模型总体拟合良好。
正式版 b 保留代表性个体，c 改为三任务全体被试的评分分布与校准。当前 c 仅是 S101 的行为检查草稿，不能替代群体结果。

唯一选例依据是结果可用性，不按拟合最好或口述一致性最高选例。采用 primary 版本，未选择略低 NLL 的 mass-preserving 反事实版本。其他目录主要为合成数据恢复、数值预算检查，不能充当真实被试。
现有参数来自 0818 的已选参数，迁用至 0826；未重新优化、未划分留出集，故只解释为初步参数迁用检查。

## 数据、时序及变换

- b 使用 internal_cognitive_trajectories_v1 导出的 online_correct_probability 和 online_prior；由 16 次 PF、每次 128 粒子的在线边际量平均产生。
- online 指作答前、已条件化于过去实际选择/反馈。它不是自由生成，也不是参数在留出集上的泛化结果。
- 不用全序列条件化的 smoothed 路径，不挑 best_complete_path；此类轨迹可能解释历史，但不能当在线预测。
- 320 个试次全部进入统计和状态图。学习曲线统一使用尾随 32 试次平均，前 31 试次无完整窗口故不显示曲线，而非删除数据。
- 每行一条规则，全部 29 条；家族标签仅用于分组定位，没有合并规则概率。每试次信念总质量为 1。H0 是 F1 的 0.5 阈值规则。
- 口述使用现有 feature1_use–feature4_use，并保留逐试次位置。S101 共 320 条报告、无缺失。短线代表提及，并不确定规则类型、方向或实际执行规则；不计算未经定义的外部验证得分。
- c 左：8 个连续、不重叠的 40 试次窗口，编号 1–8，非 8 名被试；观察均值对模型期望均值。
- c 右：P(choice=2) 的预设等宽 5 箱，端点 0,.2,.4,.6,.8,1。二分类下由正确类别及 P(correct) 转换得到；纵轴为实际 choice=2 的比例。所有 320 试次进入相应箱。n 是试次数。
- NLL = -mean(log P(observed choice))，先平均已有导出的预测概率，再逐试次取 log；单位 nats/trial。均匀参照为 log(2)，仅是简单参照，不是充分的模型比较。
- 无置信区间、显著性检验或被试群体推断；重叠学习窗口和时间相关试次不当作独立重复。

## 当前数值与来源差异

观察准确率 0.809375，模型期望准确率 0.807519，NLL 0.391663。
早期总览 README 的 0.390710 来自另一份导出；本图全程采用内部认知轨迹这一次导出的概率，不拼接不同运行的得分或曲线。各输入 hash 见输出 manifest。

## 复现

从仓库根目录运行，必须使用新的输出目录：

```bash
python -m CategoryLearning_codes.figures.fig2.build_behavior_state --output CategoryLearning_codes/figures/outputs/fig2/fig2bc_v4
```

输出包括图、代码快照、逐试次源表、完整规则信念、分块和校准源表、输入 hash 与计算定义。未运行任何新拟合/模拟，也未更改数据或已有模型结果。

## Draft caption

**b**, Existing Task 1 case (S101, 320 trials). Observed accuracy and online model probability of a correct choice are shown as trailing 32-trial means. The heatmap shows pre-choice belief mass for all 29 candidate rules, grouped by family. Marks below indicate verbal mentions of standardized features F1–F4. **c**, Observed versus predicted accuracy in eight consecutive 40-trial windows (numbers indicate temporal order), and choice calibration in five prespecified equal-width probability bins (n indicates trials). Predictions use previously selected parameters and condition on observed history; these panels provide a descriptive parameter-transfer check, not held-out or population-level validation. Feature mentions are an observational comparison rather than proof of rule recovery.

## QA

Assertions passed: 320 unique session/block/trial keys; trial sequence 1–320; choices and feedback match processed data row-for-row; binary correctness agrees with category; finite bounded probabilities; normalized 29-rule mass; binary oral indicators; both c summaries retain all 320 trials. Render inspected with no clipping; small block labels separated in v2 and footer spacing increased.
Static validator: 9 pass, 4 warnings, 1 failure. Vector failure and TIFF warning are overridden by repository PNG-only policy; 450 dpi follows established draft exports. Width detector misparses 183/25.4 as inches (actual width 183 mm). Log input is explicitly clipped to [1e-12,1], so positivity warning is a parser limitation. This is a review draft, not submission certification.

新增 v3：target-based alignment 与 .818118 理想报告编码参考线；解释和复现见 ORAL_ALIGNMENT_AUDIT.md。

## sigma = 0.05 对照版

`../outputs/fig2/fig2bc_sigma005_v2` 使用 `--oral-sigma 0.05`，包含原 Fig2b/c 和 `full_space_alignment_draft.png`。CLI 默认和生产 encoder 默认仍为 0.10，试验值通过参数显式传入；理想报告参考值按该 sigma 实时计算。
全空间图共享 29 条规则顺序及 0–1 色标。第三行 overlap=sum_h min(model_h,oral_h)=1-TV，范围 0–1；1 表示分布完全相同，不是分类准确率或超过机会水平的证据。显示全部逐试次值及完整 32 试次尾随均值；缺失编码保留 NaN。两侧分布时序与信息来源不同的限制仍适用。原始分布与重叠源表随图保存；所有有效口述行归一化验证通过。
选择更小 sigma 会放大中心编码偏差、把近似报告解释为更确定的规则，因此此图是敏感性对照，不能因视觉一致性而视为已验证的最终尺度。
