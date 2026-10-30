# Category-learning poster

## 初稿与版式

2026-10-30 英文海报初稿：

- [PDF](draft_v1_final/CategoryLearning_poster_draft_v1.pdf)
- [可编辑 PowerPoint](draft_v1_final/CategoryLearning_poster_draft_v1.pptx)
- [整页 PNG 预览](draft_v1_final/poster_preview.png)
- [来源、裁切坐标和检查记录](draft_v1_final/manifest.json)

沿用上传模板 `CategoryLearning_poster.pptx` 的 A0 竖版尺寸、会议页眉和主办方页脚。
采用双栏阅读顺序：左栏 01–03，右栏 04–05，底部结论。字体层级和图文比例参考
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
| 01 Task and individual learning | 学习者在什么任务中学习，口述何时采集？ | 旧 Fig1a–b；96 人、三任务、四个刺激特征；选择后、反馈前口述 |
| 02 A model of learning processes | 搜索、反馈学习与选择如何形成连续过程？ | 旧 Fig2a；示意数值仅说明模型操作；区分资源限制与搜索、执行偏向 |
| 03 Choice fit and the next comparison | 模型对已有选择记录的拟合如何？ | 旧 Fig2e；12 人、7,924 个可评分选择；完整记录拟合后的校准；替代模型比较保留待补框 |
| 04 Reconstructing reported strategies | 从行为得到的信念是否与口述内容对应？ | 旧 Fig2b–d、f–g；7,600 条有效口述；11/12 人的动态信念相容度高于时间平均信念；口述待人工复核 |
| 05 Different paths to better performance | 改善形态与规则在场、信念及执行如何联系？ | 旧 Fig3a–c；早期、渐进和突变三个描述性个案；固定拟合参数下的回顾性重建 |
| Take-home message | 个体差异可以从哪些过程理解？ | 规则何时被发现、保留和用于选择；“aha”解释保留为待检验的机制假设 |

旧图来自 2026-09-21 的 Model 0826 开发结果。当前摘要与 Model 1024 研究提供叙事方向，
海报中的旧图保持各自版本与证据含义。已有图未重新拟合或重绘；裁切仅选择完整面板，
保留坐标、色标、个案编号和原始像素。口述编码与原始数据保持原样。

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

工具依赖为 `python-pptx==1.0.2`、`PyMuPDF==1.26.5`、Pillow 和 LibreOffice。
本次将新增 Python 依赖安装在系统临时目录，未修改模型环境或根目录依赖声明。
构建时请求 Arial；当前 Linux 的字体映射使用 Liberation Sans。实际字号为标题 84/49 pt、
板块标题 33 pt、正文 25–32 pt、图注 22–24 pt。旧图内部字号沿用原图比例。

在仓库根目录运行，并指定新的输出目录：

```bash
python -m CategoryLearning_poster.build_poster \
  --output-dir CategoryLearning_poster/draft_v2 \
  --draft-version 2 \
  --second-affiliation 'Confirmed institution'
```

`--draft-version` 设置文件名与页脚中的版本；`--email` 添加联系邮箱；
`--pptx-only` 仅构建 PowerPoint。
脚本拒绝覆盖已存在的输出目录。LibreOffice 通过临时配置目录转换真实 PPTX，随后检查
PDF 页数、文本完整性、文本框边界及图文重叠，并生成 PNG 预览。实际交付前还需目视检查
整页和局部细节。旧图的脚本、数据、模型代码与已有研究结果均保持原样。

本次检查限于海报内容、来源和呈现；科研计算没有重跑。模板、参考海报和生成的二进制文件
保留在本地，自动 Git 提交包含本任务的构建脚本、说明和输出忽略规则。

交付版本已通过实际 PPTX→PDF 转换和重新打开检查：A0 单页，37 个文字框的完整文字与
边界检查通过，图文无重叠；来源哈希、逐人计数、CLI 帮助和本地文档链接均已核对。
整页与局部图像已目视检查。`draft_v1/` 保留本轮较早的检查稿，交付文件位于 `draft_v1_final/`。
