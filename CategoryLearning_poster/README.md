# Category-learning poster

## 初稿与版式

2026-10-31 英文海报第四版：

- [PDF](draft_v4_final/CategoryLearning_poster_draft_v4.pdf)
- [可编辑 PowerPoint](draft_v4_final/CategoryLearning_poster_draft_v4.pptx)
- [整页 PNG 预览](draft_v4_final/poster_preview.png)
- [来源、字体、模板比对和检查记录](draft_v4_final/manifest.json)
- [Model 1023 可编辑机制图 figure3](figures/figure3.pptx) · [图源说明](figures/README.md)
- [第三版 PDF](draft_v3/CategoryLearning_poster_draft_v3.pdf)
- [第二版 PDF](draft_v2/CategoryLearning_poster_draft_v2.pdf)
- [第一版 PDF](draft_v1_final/CategoryLearning_poster_draft_v1.pdf)

沿用上传模板 `CategoryLearning_poster.pptx` 的 A0 竖版尺寸、会议页眉和主办方页脚。
版式效仿 `reference1.jpg`：黑色粗体双行主标题、居中蓝色板块标题、白底和加粗引导词。
各板块从上到下通栏排列，图组在板块内部横向展开。第一部分为 Background and Motivation，
末节为 Summary。原来的左右分栏、编号、深色标题条和边框已取消。
主标题为 “Beyond behavioural fit: reconstructing belief dynamics / in human category learning”。
正文和标题均为可编辑 PowerPoint 对象，图片独立保留。
第四版将模型板块的左图替换为 Model 1023 机制图，图内的文字、形状和连线也全部保留为
原生可编辑对象。新图按用户指定保存在 `figures/`，独立图与海报共用绘图函数。

署名顺序为 Jiong Yang、Junqi Wang、Ran Zhu、Jiayu Zhan。第一、第三、第四作者单位为
School of Psychological and Cognitive Sciences, Peking University。
Junqi Wang 的单位保留显式占位；联系邮箱尚未提供。

## 内容结构与证据

故事依据为 [论文摘要](../CategoryLearning_paper/learning_paper_v1/learning_paper_v1.tex)：
从选择预测进入内部过程解释，再用口述内容和学习轨迹检查解释。面向认知科学会议读者，
先呈现实验问题，随后说明模型、证据和机制含义。

| 板块 | 读者需要理解的内容 | 初稿素材与证据范围 |
|---|---|---|
| Background and Motivation | 为什么要从行为推断个体内部学习过程？ | 依据论文摘要，依次提出学习差异、推断困难和研究方法 |
| Category learning across three tasks | 学习者在什么任务中学习，口述何时采集？ | 旧 Fig1a–b；96 人、三任务、四个刺激特征；选择后、反馈前口述 |
| A model of learning processes | 搜索、反馈学习与选择如何连接，选择拟合如何？ | 左侧为 Model 1023 机制示意，展示信念混合选择、反馈调节、搜索与入场继承；校准图仍为 Model 0826 旧 Fig2e，12 人、7,924 个可评分选择；模型比较待补 |
| Reconstructing reported strategies | 从行为得到的信念是否与口述内容对应？ | 三个任务的信念/口述热图和完整 12 人群体比较；7,600 条有效口述；11/12 人的动态信念相容度高于时间平均信念；口述待人工复核 |
| Different paths to better performance | 改善形态与规则在场、信念及执行如何联系？ | 旧 Fig3a–c 横向排列；早期、渐进和突变三个描述性个案；固定拟合参数下的回顾性重建 |
| Summary | 三项结果如何连接？ | 分开描述规则搜索、信念更新与规则使用；动态信念与口述的对应；“aha”解释为待检验的机制假设 |

旧图来自 2026-09-21 的 Model 0826 开发结果。当前摘要与 Model 1024 研究提供叙事方向，
海报中的旧图保持各自版本与证据含义。第二版调用原 Python 绘图函数替换文字字体；
第三版在这些函数创建的 Matplotlib 图形对象上调整面板位置、字号和线宽，另存横向 PNG。
Fig2 保留三个任务的信念/口述热图及两个完整群体比较，选择校准另行展示；个案正确率和
逐题相容度曲线不在本版显示。Fig3 保留原有三个个案的行为与状态曲线，群体汇总面板不在
本版显示。保留面板的曲线坐标、热图数组、缺失值掩码与阴影顶点在排版前后的哈希一致。
第二版已核对八份源表与旧版精确一致。新增拟合、状态重放及口述编码均为零。

图中 11/12 是逐人比较旧表 `compatibility` 与 `static_compatibility` 后的描述性计数。
它表达当前口述内容与规则分布的相容程度；报告生成概率和精确规则识别准确率需要各自的
观测模型与评价。静态参照使用每名参与者的时间平均信念。原始图注同时记录了口述晚于选择、
展示状态位于选择前这一时序，以及完整行为记录参与参数估计这一条件。

旧 Fig3 的曲线为八次固定参数粒子重放的平均，阴影表示数值重放变化。
S122 的口述对应较弱；S215 的具体状态解释仍受数值敏感性限制。
这些曲线用于展示可能的过程解释。个体心理归因、状态恢复和独立预测分别需要进一步验证。

摘要中的替代神经网络比较、有限记忆的补偿机制、意识外积累及“aha”时刻解释，目前尚缺与
这些强结论相匹配的完整证据。海报标注模型比较待补，并使用与旧图相符的表述。
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

两行主标题均为 72 pt，板块标题为 46 pt，正文主要为 30–32 pt，图注为 23.5–24.5 pt。
署名为 36 pt，单位为 28/25.5 pt。模板内部字号保持原样。
横向结果图按接近 1:1 的物理尺寸插入，坐标字号约为 20–23 pt，个案标题约为 25–27 pt。
任务图、模型示意与选择校准按保留面板的原比例缩放。

字体版图像由以下入口生成，产物按图号保存在 `figures/outputs/figN/`：

```bash
python -m CategoryLearning_codes.figures.fig2.build_poster_font_variants \
  --version poster_arial_20261030_v1
```

该目录已经生成；重跑需指定新版本，并在海报脚本的 `FIGURES` 中选用对应图像。
字体审计保存在各图目录的 `poster_font_audit.json`。

第三版的横向图组由以下入口生成：

```bash
python -m CategoryLearning_codes.figures.fig2.build_poster_row_figures \
  --version poster_rows_20261031_v3
```

该版本已生成，重跑需换用新目录名。两张图及其 `manifest.json` 分别位于
`figures/outputs/fig2/poster_rows_20261031_v3/` 和
`figures/outputs/fig3/poster_rows_20261031_v3/`。
清单保存原图来源、面板选择、绘图数据哈希和字体审计；海报清单也包含这两份记录。

在仓库根目录运行，并指定新的输出目录：

```bash
python -m CategoryLearning_poster.build_poster \
  --output-dir CategoryLearning_poster/draft_v5 \
  --draft-version 5 \
  --second-affiliation 'Confirmed institution'
```

`--draft-version` 设置文件名与页脚中的版本；`--email` 添加联系邮箱；
`--template-font` 指定模板的 Arial Black 字体文件；`--pptx-only` 仅构建 PowerPoint。
脚本拒绝覆盖已存在的输出目录。LibreOffice 通过临时配置目录转换真实 PPTX，随后检查
PDF 页数、文本完整性、文本框边界及图文重叠，并生成 PNG 预览。实际交付前还需目视检查
整页和局部细节。旧图的原脚本、数据、模型代码与已有研究结果均保持原样。

本次检查限于海报内容、来源和呈现；模型拟合与状态重放没有重跑。模板、参考海报和生成的
二进制文件保留在本地，自动 Git 提交包含本任务的构建脚本和说明。

第四版检查覆盖 A0 单页、全部原生文字框的完整文字与边界、图文重叠、实际字体匹配、
35 个原模板文件的逐字节一致性和页眉页脚的逐像素一致性。
第三版已运行的口述内容统计与摘要轨迹分析回归测试共 17 项通过。
第四版验证独立机制图的原生对象、零图片媒体、矢量渲染及其在海报中的对应关系。
机制图和版本页脚以外的海报像素与第三版一致。
结果图按仓库 PNG 输出规则验收实际图像、字体审计和精确源表对照；机制图按用户指定的
原生 PPTX 形式验收，具体检查见[图源说明](figures/README.md)。
旧版位于 `draft_v1/`、`draft_v1_final/`、`draft_v2/` 和 `draft_v3/`。
本版审阅输出保存在 `draft_v4/`，最终交付位于 `draft_v4_final/`。
