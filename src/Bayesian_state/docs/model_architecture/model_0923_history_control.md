# Model 0923：真实历史与等权路径对照

本对照固定已有拟合参数，检查 conditional prediction 的后期提高怎样依赖真实选择对潜在路径的约束。
它使用共享粒子滤波器已有的分析开关，认知模块及其参数保持原值。
2026-10-14 的两个案例为 MEG 335 和 exp123 307，均为条件3，分别保留748和704个有效试次。
原拟合均保留 `unresolved / boundary_review_required` 状态。

## 三种条件

| 条件 | 认知更新使用的回答与反馈 | 潜在路径的汇总方式 |
|---|---|---|
| A：conditional prediction | 被试真实记录 | 按真实选择的似然逐题更新粒子权重，必要时重采样 |
| B：真实历史、等权路径 | 被试真实记录 | 全部路径始终等权，关闭重采样 |
| C：自主生成 | 模型生成回答及其对应的任务反馈 | 独立生成多条完整轨迹，再等权汇总 |

粒子是对可能内部过程的一条数值样本路径。A的粒子权重表示观察者对这些路径的相对支持。
各条路径内部还分别保存规则信念和作答权重：规则信念随认知更新变化，作答权重结合信念与主导惯性，
决定各条规则对本题类别预测的贡献。B关闭观察者层面的粒子加权，保留路径内部的规则信念更新、
作答权重、候选搜索与退出、配对学习和精度学习。

具体设置为 `condition_on_observed_choice=False`、`resample_threshold_fraction=0`，
反馈学习概率沿用原值1。A和B均在读取本题实际选择、反馈之前保存预测及状态。
A在这个时点已经条件于过去的选择。全部拟合参数来自完整被试序列，本次是样本内机制诊断。
口述未进入本次计算。

B将同一真实回答和反馈提供给每条内部路径，回答的产生过程在此受到外部控制。
其选择NLL作为这种控制条件下的描述性分数保存。正式拟合目标继续使用A的条件选择概率。
A−B衡量按观察选择调整路径分布与重采样的共同影响；B−C衡量外部给定历史和自主行为历史的差异。
两组差异受模块间交互影响，不能直接换算成某个认知模块对准确率的独立贡献。

## 本次结果

下表使用每人最后64题；模型三列均为正确类别概率的算术平均。
真实正确率将反馈1计为完全正确，条件3的0.5反馈另有含义，未计为完全正确。

| 被试 | 惯性λ | 真实正确率 | A | B | C |
|---|---:|---:|---:|---:|---:|
| 335 | 0 | 82.8% | 84.1% | 31.5% | 32.0% |
| 307 | 2 | 95.3% | 80.1% | 36.5% | 42.4% |

两人的B都没有达到A的后期水平。B的16个独立种子中，后期均值范围分别为30.3–33.0%和35.2–37.8%。
335的B与C均值相近；307的B均值比C低约5.9个百分点。真实历史本身在这些固定参数下仍不足以使大多数路径学会。
每条B路径的后期平均正确概率达到0.8的比例分别为1.17%和0.98%。
因此应保留“少量路径能达到较高水平”和“大多数路径表现较低”这两个事实。

以A后期作答权重最高的规则作为参考，两人均选出R42（目录的零基索引，`3d_axis_triple`，维度顺序0、1、2）。
这是一项事后描述性选择。R42在B后期候选集合中的概率分别仅为3.09%和4.48%，作答权重分别为2.67%和4.19%；
在A中，对应候选概率为99.98%和98.30%，作答权重为98.98%和97.35%。
这说明A能将推断集中到与真实行为相容的内部路径，而B的大多数路径在候选集合层面就与A不同。
对于335，λ为0时每条路径的规则信念和作答权重相等，这个落差不能由较强作答惯性解释。
候选规则的生成与保留、反馈证据如何改变信念，以及精度变化值得继续逐路径检查。
当前结果尚未区分这些模块各自的作用，也尚未完成307的惯性消融。

A对潜在路径的条件推断属于模型的正式推断步骤。A预测较好说明模型能从过去行为推断出可解释下一题的状态；
自主学习还要求模型在自身回答产生的反馈下，经常到达相应状态。这两种表现需要分别检查。

## 运行与产物

配置位于 [`configs/shared/model0923_history_control.yaml`](../../../../configs/shared/model0923_history_control.yaml)。
每人使用16个种子、每种子128条路径，匹配原A预算；C复用原64条轨迹。
B重用A的种子以匹配初始分布。A开始加权或重采样后，随机数消费和路径祖先会分化；
相同种子不保证后续逐路径配对。

本次发现现行源码的几何求解器与拟合时版本不同，因此通过 `--runtime-source` 加载已有拟合源码快照。
新分析脚本在独立Python进程中调用该快照的共享模型实现，并验证科学输入哈希与数值依赖版本。
快照按只读方式使用，字节码写入关闭，Numba缓存指向临时目录。

从仓库根目录运行，输出目录须尚未存在。先做一个被试、前8题、一个种子的检查：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/model0923_history_numba \
MPLCONFIGDIR=/tmp/model0923_history_mpl \
python -m src.Bayesian_state.workflows.analysis.model_0923_history_control \
  --runtime-source results/model_0923/plus_335_then307_continuation_20261013_v1/source \
  --output-dir /tmp/model0923_history_control_smoke_new --smoke
```

完整两人对照使用同一入口，省略 `--smoke`，另选结果目录：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/model0923_history_numba \
MPLCONFIGDIR=/tmp/model0923_history_mpl \
python -m src.Bayesian_state.workflows.analysis.model_0923_history_control \
  --runtime-source results/model_0923/plus_335_then307_continuation_20261013_v1/source \
  --output-dir results/model_0923/history_control_335_307_new

MPLCONFIGDIR=/tmp/model0923_history_mpl \
python -m CategoryLearning_codes.figures.fig3.model0923_history_control \
  --analysis-dir results/model_0923/history_control_335_307_new \
  --figure-output-dir CategoryLearning_codes/figures/outputs/fig3/model0923_history_control_new
```

并行预算上限128，实际受可用CPU和独立任务数限制；本次32个任务使用32个进程，各进程一个数值线程。
运行没有新增参数拟合或自主模拟。新文件含：

- `comparison.csv`：全序列、最初64个计分试次、最后64题的A/B/C比较。
- `subject_*/trials.csv`：原始试次标识、逐题预测与参考规则状态。
- `subject_*/seeds/b_*.npz`：每个种子的B概率、路径均值状态、每条路径的正确概率。
- `subject_*/comparison.npz`：合并的B状态、逐种子正确概率、B/C逐路径正确概率。
- `manifest.json`、`validation.json`：配置、源码位置、来源及产物SHA-256、验证结果。
- `index.html`：三种条件的定义、数值表、两张PNG与解释边界。
- 图目录的 `window_plot_source.csv` 和 `late_path_plot_source.csv`：曲线及分布图的数据。

本次结果入口为 `results/model_0923/history_control_335_307_20261014_v1/index.html`。
PNG位于 `CategoryLearning_codes/figures/outputs/fig3/model0923_history_control_20261014_v1/`。
曲线按连续32个计分试次分组，保留最后一个不足32题的窗口；初始化题仅用于推进状态，排除计分。
分布图包含全部2048条B路径和64条C路径，比较每条路径最后64题的平均正确概率。
C在本图用算术平均，原行为图b另报告自主轨迹实际正确率中位数；两种统计量在报告中分别列出。

## 验证与解释范围

完整运行检查两名被试A的第一个种子各前8题，预测与保存值逐项一致；全部复用A/C文件与保存哈希一致。
B的每个试次检查粒子等权、加权前后状态一致、ESS等于128、无重采样、概率有限且归一化、路径均值等于汇总预测。
335进一步检查λ为0时信念与作答权重相等。真实试次、选择、类别和反馈与原评估逐项核对。
完整A重跑、重新拟合、参数不确定性分析及新的口述评价均未执行。

```bash
python -m pytest -q tests/bayesian_state/test_model_0923_history_control.py \
  CategoryLearning_codes/figures/fig3/test_model0923_history_control.py
python -m src.Bayesian_state.workflows.analysis.model_0923_history_control --help
python -m CategoryLearning_codes.figures.fig3.model0923_history_control --help
```

图像用于屏幕诊断，按仓库规则仅导出300 dpi PNG。静态图源预检中的矢量、TIFF、600 dpi和期刊栏宽要求
按这一用途记录为格式例外。实际渲染检查文字、图例和数值范围；每项统计量、路径数、分组和排除初始化题的规则均保留在图契约中。
两个被试是选定案例，种子及路径的差异表示固定参数下的模拟变异；本次没有群体推断或显著性检验。
