# Model 0923 选择读出诊断

`model0923_readout_review.py` 使用已保存的逐被试评估，检查规则信念、作答权重和主导规则概率的关系。模型算法和拟合参数保持原值；运行过程只读取结果并生成诊断图。

本次审阅覆盖2026-10-13的13人：exp123初始12人中，307采用续跑评估；另加入MEG 335的续跑评估。各人的条件、数据来源与拟合阶段分别记录。脚本按命令行输入顺序选择来源：同一被试在后续目录中出现时，使用后续目录；被替代来源记录在清单中。

从仓库根目录运行，两个输出目录都须尚未存在：

```bash
MPLCONFIGDIR=/tmp/model0923_readout_mpl python -m CategoryLearning_codes.figures.fig3.model0923_readout_review \
  --evaluation-dir results/model_0923/exp123_selected12_postfit_20261013_v1 \
  --evaluation-dir results/model_0923/plus_335_then307_continuation_20261013_v1/subject_335/evaluation \
  --evaluation-dir results/model_0923/plus_335_then307_continuation_20261013_v1/subject_307/evaluation \
  --output-dir results/model_0923/readout_review_20261014_v1 \
  --figure-output-dir CategoryLearning_codes/figures/outputs/fig3/model0923_readout_review_20261014_v1
```

结果目录中的`index.html`连接18张PNG：逐人全序列图13张，335、307、104、122的局部图4张，总体对照图1张。`summary.csv`保存逐人指标，`trial_metrics.csv`保存逐题指标，`heatmap_source.csv`保存热图绘图数据，四份`subject_*_focus.csv`保存局部对照图的规则数值。完整目录和全部规则的原始数值继续保存在各人的来源目录中。`manifest.json`记录来源、SHA-256摘要、实际软件版本和绘图约定，`validation.json`记录数据检查。

## 读图定义

所有状态取自`predictive_*`：本题实际选择之前，按当时粒子权重汇总；多次粒子滤波等权平均。历史选择和历史反馈已经进入推断。本题实际选择后重新加权得到的`filtered_*`属于另一个推断时点。拟合参数来自完整序列，因此当前行为比较属于样本内诊断。

- **规则信念**表示历史证据对规则的相对支持。
- **作答权重**表示规则在本题类别概率混合中的权重。上一主导规则获得惯性倍率后，各权重归一化为总和1。
- **主导规则概率**表示某条规则为最大作答权重规则的路径概率。它刻画规则身份的推断不确定性。
- **总变差距离**为`0.5 * sum(abs(mean_weight - mean_belief))`，量化两份路径汇总分布之间的差异。图中乘100，以百分点呈现。各路径的差异可能抵消，路径汇总后的距离至多等于路径距离的加权平均。
- **有效参与数量**先在每条路径内计算`1 / sum(weight**2)`，再按路径权重汇总。图中阴影为固定参数下的路径10–90%范围；参数估计的不确定性尚未纳入。

全序列热图单列六条规则，其他规则按列求和，保留全部质量。规则选择先纳入局部焦点试次的最大作答权重规则与最大信念规则，再按全序列平均作答权重补齐。每张图的三幅热图使用同一顺序。局部焦点取总变差距离最大的试次；惯性为0时，取主导规则概率最集中的试次。规则编号沿用各条件目录的零基索引。

全序列状态指标包含初始化题。选择NLL沿用来源计分掩码，排除初始化题。行为曲线对过去32个有效试次求均值，包含当前题，前31题没有完整窗口。原始试次顺序、被试和条件保持不变。

## 解释范围与验证

当前结果可用于检查惯性是否产生明显的信念—作答分离、主导规则身份有多确定，以及作答权重集中程度怎样变化。相邻试次的主导概率最高规则可以来自不同路径；实际路径的切换率和驻留时间需要相邻状态的联合记录。逐规则对某个答案的具体贡献还需路径内的规则×类别联合概率。保存的汇总状态尚未提供这些量。

脚本检查概率有限性与归一化、被试与试次对齐、惯性为0时信念和权重一致、参与数量与容量范围、分位数顺序，以及保存的类别概率和NLL能否重现。相关验证入口：

```bash
python -m pytest -q CategoryLearning_codes/figures/fig3/test_model0923_readout_review.py
python -m pytest -q tests/bayesian_state/test_model_0923_v3_transition.py tests/bayesian_state/test_model_0923_full_v3_integration.py
python -m CategoryLearning_codes.figures.fig3.model0923_readout_review --help
```

输出用于屏幕机制审阅，字号和画布按该用途设置，PNG为唯一图像导出格式。总体图逐人呈现结果，统计检验和群体推断均未实施。初始搜索、续跑的边界审查状态以及自主模拟与真实行为的差距仍需结合各自原评估解释。

本次静态图源预检按投稿图默认规则报告了缺少SVG、TIFF以及画布和分辨率差异。仓库的PNG保留规则与屏幕审阅用途确定了本次导出约定：300 dpi、10–11英寸宽，最终以实际渲染检查标签可读性。预检还标记了`dropna`和对数运算：前者只清理搜索状态表中的空元数据字段，全部行为试次保留；后者在计算NLL前将概率限制到`[1e-12, 1]`，沿用来源评分规则。预检原始结果及逐项处置保存在报告目录的QA记录中。
