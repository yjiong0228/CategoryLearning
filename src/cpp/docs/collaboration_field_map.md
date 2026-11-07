> 迁移说明：本文记录独立 C++ 项目阶段的机制或验证。当前入口和共享数据路径见 [README](../README.md)，本次迁移验证见 [迁移记录](migration_validation.md)。

# 本地协作字段表

交付对象为 Cond1/2/3 共 96 名已完成被试的本地结果。打包程序负责写入清单、
索引、元数据与数组；本读取器只读取已保存数据。使用标准库与 NumPy，复制为
包根目录的 `python1031_io.py` 后可独立使用。

## 目录与读取

根目录包含 `README`、`manifest.json` 与索引。每名被试位于
`subjects/cond1/subject_101/`，条件和编号按实际对象替换，包含：

- `python1031_fit.npz`、`python1031_generation.npz`、`python1031_metadata.json`
- `subject_source.csv`
- `native/evaluation_inputs.npz`、`native/evaluation_manifest.json`
- `native/conditional_particle_contents.npz`

```python
from python1031_io import load_subject, load_particles, verify_bundle

data = load_subject(root, 1, 101)  # 也接受 "cond1", "subject_101"
fit, generation = data["fit"], data["generation"]
with load_particles(root, 1, 101) as particle:
    weights = particle["weights"]
    particle.validate_weights()  # 显式检查；只读取权重和 valid
report = verify_bundle(root)
```

`load_subject` 返回元数据顶层字段及两个数组字典，原样保留保存的字段。
元数据中的 `fit`、`generation` 为保留名称。粒子读取器支持 Mapping、`close()`
和上下文管理器；打开、枚举和查询键名均不解压数组。每次取值只读取该字段，
不缓存；重复计算时由调用者保存已读取的数组。缺失字段保持缺失，查询触发
`KeyError`。转换不生成逐 PF 状态副本或 ESS。

## 汇总数组

记 T 为试次数，C 为类别数，R 为条件滤波重复次数，S 为自主生成路径数，
H 为目录规则数。fit 汇总使用全部 16 个 PF 种子；逐粒子包固定使用第一个
PF 种子。元数据须分别记录这两个来源、种子、条件、被试及时间相位。
打包元数据使用 `pf_repeats`、`autonomous_repeats`、`feature_names`、
`conditional_particle_file` 与 `conditional_particle_contents`；读取器原样保留这些
字段，按固定文件名读取数组。元数据自身不要求包含 `files` 映射。

| fit 字段 | 形状与含义 |
| --- | --- |
| `choice_probabilities` | `[T,C]`，反馈前预测概率，跨 PF 重复均值 |
| `choice_probabilities_by_repeat` | `[R,T,C]`，各 PF 的预测概率 |
| `feature_set_probability` | `[T,16]`，当前选择过滤后的维度集合执行概率 |
| `catalogue_execution_probability` | `[T,H]`，当前选择过滤后的目录规则执行概率 |
| `representation_belief` / `representation_execution` | `[T,4]`，预测权重下的表征信念 / 实际执行质量 |
| `filtered_representation_belief` | `[T,4]`，选择过滤后的粒子权重加权反馈前槽位先验 |
| `filtered_representation_execution` | `[T,4]`，选择过滤后的实际执行质量，按全部 PF 精确线性汇总 |
| `choices` / `categories` | `[T]`，Python 编码；选择缺失值为 0 |
| `feedback` / `stimulus` | 实际观察反馈 `[T]` / 物理刺激 `[T,4]` |
| `valid` / `score_mask` / `trial_ids` / `session_ids` | 有效性、评分范围与原始试次及 session 标识；保留原序列 |

四类表征顺序为未形成内容、具体示例、局部概括、目录规则。
预先保存的其他别名（如 `original_ids`）随数组原样返回。读取器不推导缺失汇总量。

| generation 字段 | 形状与含义 |
| --- | --- |
| `feedback` | `[S,T]`，任务实际生成的等级反馈 |
| `choices` / `categories` | `[S,T]` / `[T]`，Python 从 1 开始编码 |
| `valid` / `category_probabilities` | `[S,T]` / `[S,T,C]`，路径有效性 / 类别概率 |
| `seeds` | 各路径的生成种子 |

若生产端保存 `category_probs` 别名，读取器同样原样保留；当前 smoke 使用
`category_probabilities`。

## 逐粒子字段

所有 `particle_` 前缀移除一次，其他观察输入键原样保留。保留原始字段全集，
命名碰撞会报错。记 P 为粒子数、K 为工作区容量、G 为配对数。

| 返回字段 | 处理与含义 |
| --- | --- |
| `weights` / `predictive_weights` | `[T,P]`，当前选择过滤后 / 当前选择前权重 |
| `active` / `template_mask` | 转为 bool；形状 `[T,P,K]` / `[T,P,K,4]` |
| `representation_kind` | 原始 0 未形成内容、1 模板、2 目录规则 |
| `semantic_kind` | 原生四类语义编码，独立保留 |
| `representation_uid` / `representation_parent` / `born_trial` | 粒子内部身份与谱系，保留原值 |
| `catalogue_rule` | 目录规则编号，保留原值 |
| `belief` / `beta` / `choice_weights` / `selection_probabilities` | 反馈前槽位信念、精度、实际执行、执行前抽样概率 |
| `template_anchor` / `feature_mask` | 模板位置 / 所用维度位掩码 |
| `conditional_template_category_belief` | 显式别名，始终保留原始 `[T,P,K,G,C]` |
| `template_category_belief` | 仅 G=1 时压缩倒数第二轴为 `[T,P,K,C]`；Cond3 的 G=3 保留 |
| `pairing` | 保留 `[T,P,K,G]`，包括 Cond1/2 的单元素 G 轴 |
| `rule_choice_probabilities` / `perceived_stimulus` | 内容回答分布 / 粒子感知刺激，原样保留 |

槽位与 uid 的身份范围局限于该被试的粒子历史；跨粒子或被试比较须使用内容
与语义信息。Cond3 的标签分布以配对为条件，联合量由 pairing 和对应条件分布
共同确定。权重校验仅检查 valid 试次；无 valid 字段时检查全部试次。

未激活槽的浮点内容字段为 NaN、ID 为 -1，计算时必须使用 `active`；
不要直接把这些 NaN 与零权重相乘后求和。模板位置和类别信念只对
`representation_kind == 1` 有定义，目录规则请使用 `catalogue_rule` 与规则几何。
例如汇总槽位先验可用 `np.where(particle['active'], particle['belief'], 0.0)`，
再乘相应的粒子权重；这与丢弃真实存在的零信念槽不同。

## 编码与证据范围

Python 汇总包选择和类别从 1 开始，选择缺失为 0；native 观察输入从 0 开始，
选择缺失为 -1。逐粒子读取器保留 native 观察输入编码。特征顺序取自元数据和
native manifest 的 `protocol.feature_order`；第 i 位对应刺激第 i 列，禁止固定
假设身体部位顺序。

粒子快照位于当前选择之后、当前反馈之前、重采样之前。predictive 权重还未
纳入当前选择；filtered 权重已纳入当前选择。反馈学习发生于快照之后。
无效试次可能含 NaN，分析时按 valid 与 score_mask 筛选。Cond3 反馈为任务等级
反馈，正确率须根据选择与类别另行定义。维度集合 Jaccard 衡量所用维度集合的
交并比；几何区域 IoU 衡量特征空间区域的交并比，两者含义不同。

交付支持沿用 Python1031 后处理所需的数组名称与明确记录的转换。原始 Cond1
pilot 的整套报告另依赖 registration、completion、模型选择及排名文件。
本包交付当前模型结果；原始 pilot 研究目录、多模型比较与排名均未构造。
口述分析保持暂定状态，等待用户完成最终人工复核。

## 完整性校验

```bash
python python1031_io.py --verify ROOT
```

根清单契约为 `{"files": [{"path": "相对/POSIX路径", "sha256": "64位十六进制", "bytes": 123}]}`。
读取器逐文件以 1 MiB 块计算 SHA256 并核对字节数，返回 `files_checked` 与
`bytes_checked`。拒绝绝对路径、`..`、重复路径、符号链接与目录 junction。
CLI 成功返回 0，失败返回非零，仅执行完整性检查。检查范围为清单列出的文件；
清单真实性、96 名被试是否齐全和科学有效性由打包与研究审核分别确认。
根清单的 `subjects` 为被试记录列表；完整性检查仅使用 `files` 列表。
