# 期刊文稿

两份工作稿按文章分别存放，每个目录包含主文件、可直接阅读的 PDF 和分节源文件。

| 稿件 | PDF | LaTeX 主文件 | 内容说明 |
| --- | --- | --- | --- |
| 旧工作稿 | [sn-article.pdf](sn_article/sn-article.pdf) | [sn-article.tex](sn_article/sn-article.tex) | 原有 Results 文稿；[目录说明](sn_article/README.md) |
| Model 0923 英文稿 | [model_0923_article.pdf](model_0923_article/model_0923_article.pdf) | [model_0923_article.tex](model_0923_article/model_0923_article.tex) | Results 中的模型主干与 Methods 中的机制细节，实证结果留空；[目录说明](model_0923_article/README.md) |

## 目录结构

```text
CategoryLearning_paper/
├── README.md
├── AGENTS.md
├── sn_article/
│   ├── sn-article.tex
│   ├── sn-article.pdf
│   ├── sn-article.bbl
│   ├── sn-bibliography.bib
│   ├── sections/
│   └── README.md
├── model_0923_article/
│   ├── model_0923_article.tex
│   ├── model_0923_article.pdf
│   ├── sections/
│   └── README.md
├── latex/
├── references/
└── figures/
```

`latex/` 集中存放两份稿件共享的 Springer Nature 类文件、参考文献样式和模板示例。
每份稿件的 `.latexmkrc` 加载共享配置，设置 LaTeX 和 BibTeX 的搜索路径。
`references/` 保存参考文章和既有参考材料；`figures/` 保存已确认的论文图。

两份稿件原有的本地编译状态移到各自的 `build/previous/`。
这些文件记录的路径对应整理前的目录，后续编译从新的输出目录开始。
目录整理保留了原 PDF、文献库、`.bbl`、模板和参考资料。
模型技术说明位于 [model_architecture](../src/Bayesian_state/docs/model_architecture/)。

## 阅读与编译

阅读时直接打开上表中的 PDF；编辑时打开对应的 LaTeX 主文件。
分节文件统一位于稿件目录内的 `sections/`。

编译需要 LaTeX、BibTeX 和 `latexmk`。具体命令见两份稿件的目录说明；
验证编译使用系统临时目录，生成的 PDF 经检查后可更新同目录下的阅读版本。
本地 `build/` 目录由 Git 忽略，稿件目录中的阅读版 PDF 保留在版本控制中。

## 路径变更

| 原位置 | 现位置 |
| --- | --- |
| `sn-article.tex/pdf/bbl`、`sn-bibliography.bib` | `sn_article/` |
| `sections/` | `sn_article/sections/` |
| `model_0923_article.tex/pdf` | `model_0923_article/` |
| `model_0923_article/results.tex`、`methods.tex` | `model_0923_article/sections/` |
| 根目录下的 `.cls`、`.bst` 和模板 `.eps` | `latex/` |
