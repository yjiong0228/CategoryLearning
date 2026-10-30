# Category-learning poster

## 初稿与版式

2026-10-30 英文海报第二版：

- [PDF](draft_v2/CategoryLearning_poster_draft_v2.pdf)
- [可编辑 PowerPoint](draft_v2/CategoryLearning_poster_draft_v2.pptx)
- [整页 PNG 预览](draft_v2/poster_preview.png)
- [来源、字体、模板比对和检查记录](draft_v2/manifest.json)
- [第一版 PDF](draft_v1_final/CategoryLearning_poster_draft_v1.pdf)

沿用上传模板 `CategoryLearning_poster.pptx` 的 A0 竖版尺寸、会议页眉和主办方页脚。
第一部分为通栏的 Background and Motivation，参考示例用加粗引导词组织学习差异、
内部过程推断的困难和研究方法。随后采用双栏阅读顺序：左栏 02–04，右栏 05–06，底部结论。
字体层级和图文比例参考
`reference1.jpg` 与 `reference2.pdf`；上传目录中的 JPG 实际文件名为 `reference1.jpg`。
主标题压缩为 “Beyond behavioural fit”，副标题说明类别学习与信念动力学。
正文、标题、边框和结论条均为可编辑 PowerPoint 对象，旧图作为独立图片保留。

署名顺序为 Jiong Yang、Junqi Wang、Ran Zhu、Jiayu Zhan。第一、第三、第四作者单位为
School of Psychological and Cognitive Sciences, Peking University。
Junqi Wang 的单位保留显式占位；联系邮箱尚未提供。

## 内容结构与证据

故事依据为 [论文摘要](../CategoryLearning_paper/learning_paper_v1/learning_paper_v1.tex)：
从选择预测进入内部过程解释，再用口述内容和学习轨迹检查解释。面向认知科学会议读者，
先呈现实验问题，随后说明模型、证据和机制含义。

| 板块 | 读者需要理解的内容 | 初稿素材与证据范围 |
|---|---|---|
| 01 Background and Motivation | 为什么要从行为推断个体内部学习过程？ | 依据论文摘要，依次提出学习差异、推断困难和研究方法 |
| 02 Task and individual learning | 学习者在什么任务中学习，口述何时采集？ | 旧 Fig1a–b；96 人、三任务、四个刺激特征；选择后、反馈前口述 |
| 03 A model of learning processes | 搜索、反馈学习与选择如何形成连续过程？ | 旧 Fig2a；示意数值仅说明模型操作；区分资源限制与搜索、执行偏向 |
| 04 Choice fit and the next comparison | 模型对已有选择记录的拟合如何？ | 旧 Fig2e；12 人、7,924 个可评分选择；完整记录拟合后的校准；替代模型比较保留待补框 |
| 05 Reconstructing reported strategies | 从行为得到的信念是否与口述内容对应？ | 旧 Fig2b–d、f–g；7,600 条有效口述；11/12 人的动态信念相容度高于时间平均信念；口述待人工复核 |
| 06 Different paths to better performance | 改善形态与规则在场、信念及执行如何联系？ | 旧 Fig3a–c；早期、渐进和突变三个描述性个案；固定拟合参数下的回顾性重建 |
| Take-home message | 个体差异可以从哪些过程理解？ | 规则何时被发现、保留和用于选择；“aha”解释保留为待检验的机制假设 |

旧图来自 2026-09-21 的 Model 0826 开发结果。当前摘要与 Model 1024 研究提供叙事方向，
海报中的旧图保持各自版本与证据含义。第二版调用原 Python 绘图函数，只替换文字字体，
另存 PNG；裁切选择完整面板，保留坐标、色标与个案编号。口述编码与原始数据保持原样。
Fig2 的逐人、逐题、校准和规则分布表，以及 Fig3 的逐人、逐题、突变窗口和支持区间表，
与旧版精确一致。新增拟合、状态重放及口述编码均为零。

图中 11/12 是逐人比较旧表 `compatibility` 与 `static_compatibility` 后的描述性计数。
它表达当前口述内容与规则分布的相容程度；报告生成概率和精确规则识别准确率需要各自的
观测模型与评价。静态参照使用每名参与者的时间平均信念。原始图注同时记录了口述晚于选择、
展示状态位于选择前这一时序，以及完整行为记录参与参数估计这一条件。

旧 Fig3 的曲线为八次固定参数粒子重放的平均，阴影表示数值重放变化。
S122 的口述对应较弱；S215 的具体状态解释仍受数值敏感性限制。
这些曲线用于展示可能的过程解释。个体心理归因、状态恢复和独立预测分别需要进一步验证。

摘要中的替代神经网络比较、有限记忆的补偿机制、意识外积累及“aha”时刻解释，目前尚缺与
这些强结论相匹配的完整证据。初稿保留模型比较位置和机制假设，并使用与旧图相符的表述。
当前研究边界见 [research handoff](../src/Bayesian_state/docs/RESEARCH_HANDOFF.md)
与 [research guide](../src/Bayesian_state/docs/RESEARCH_GUIDE.md)。

| 固定术语 | 含义 |
|---|---|
| Rule availability / considered | 某规则进入当前候选集合的状态或概率 |
| Belief | 学习者对候选规则的支持权重 |
| Rule use / executed | 规则对本题选择的作用；具体量依原模型读出方式定义 |
| Report compatibility | 规则内容与口述内容的相容程度 |
| Calibration | 按预测概率分箱，比较预测正确率与实际正确率 |

## 图源与定义

- [Fig1 原图注](../CategoryLearning_codes/figures/fig1/JOURNAL_FIGURE.md)
- [Fig2 原图注](../CategoryLearning_codes/figures/fig2/JOURNAL_FIGURE.md)
- [Fig3 原图注](../CategoryLearning_codes/figures/outputs/fig3/journal_20260921_v3/legend.md)
- [Fig2 逐人统计](../CategoryLearning_codes/figures/outputs/fig2/journal_20260921_v2/subjects.csv)

生成目录的 `manifest.json` 保存输入文件 SHA-256、图像裁切范围、文字内容、字号、位置、
依赖版本和 PDF 检查结果。PowerPoint 备注页概括模型版本、资料用途和解释范围。

## 重建与检查

工具依赖为 `python-pptx==1.0.2`、`PyMuPDF==1.26.5`、Pillow、fontconfig 和 LibreOffice；
字体版图像沿用仓库 Python/matplotlib 绘图环境。
本次将新增 Python 依赖安装在系统临时目录，未修改模型环境或根目录依赖声明。
正文和图中常规、粗体文字分别使用 `src/Arial.ttf` 与 `src/Arial Bold.ttf`。
图像在保存前为每个文字对象绑定相应文件，数学标记也使用这两个字体。
PDF 导出采用独立的临时字体配置，通过文件哈希核对实际字体匹配结果，并检查正文 PDF 字体。

模板标题原字体为 Arial Black。第一版在缺少该字体的 Linux 环境中导出，发生了字体替换。
第二版在导出环境加入原 Arial Black，保留模板全部母版、版式、主题和图片文件的原始字节。
页眉、页脚与原模板在同一字体环境导出的 PDF 逐像素比较。模板字体作为原始设计保留。
字体本地文件为 `fonts/Arial Black.ttf`，来源为
[原 Microsoft 核心字体分发包](https://downloads.sourceforge.net/corefonts/arialb32.exe)。
字体二进制文件保留本地；PPTX 使用原生字体引用，PDF 包含实际导出的字体。

标题为 84/49 pt，板块标题为 33 pt，正文主要为 25–32 pt，图注为 21.5–24 pt。
模板内部字号保持原样，图中字号按原图比例显示。

字体版图像由以下入口生成，产物按图号保存在 `figures/outputs/figN/`：

```bash
python -m CategoryLearning_codes.figures.fig2.build_poster_font_variants \
  --version poster_arial_20261030_v1
```

该目录已经生成；重跑需指定新版本，并在海报脚本的 `FIGURES` 中选用对应图像。
字体审计保存在各图目录的 `poster_font_audit.json`。

在仓库根目录运行，并指定新的输出目录：

```bash
python -m CategoryLearning_poster.build_poster \
  --output-dir CategoryLearning_poster/draft_v3 \
  --draft-version 3 \
  --second-affiliation 'Confirmed institution'
```

`--draft-version` 设置文件名与页脚中的版本；`--email` 添加联系邮箱；
`--template-font` 指定模板的 Arial Black 字体文件；`--pptx-only` 仅构建 PowerPoint。
脚本拒绝覆盖已存在的输出目录。LibreOffice 通过临时配置目录转换真实 PPTX，随后检查
PDF 页数、文本完整性、文本框边界及图文重叠，并生成 PNG 预览。实际交付前还需目视检查
整页和局部细节。旧图的原脚本、数据、模型代码与已有研究结果均保持原样。

本次检查限于海报内容、来源和呈现；科研计算没有重跑。模板、参考海报和生成的二进制文件
保留在本地，自动 Git 提交包含本任务的构建脚本、说明和输出忽略规则。

第二版检查覆盖 A0 单页、39 个文字框的完整文字与边界、图文重叠、实际字体匹配、
35 个原模板文件的逐字节一致性和页眉页脚的逐像素一致性。
已有口述内容统计与摘要轨迹分析的两套相关回归测试共 17 项通过。
通用绘图静态检查器要求额外矢量导出，并且不能追踪委托给原绘图函数的 PNG 保存操作；
本任务按仓库 PNG 输出规则验收实际图像、字体审计和精确源表对照。
旧版位于 `draft_v1/` 与 `draft_v1_final/`，本版位于 `draft_v2/`。
