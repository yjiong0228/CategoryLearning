# 旧工作稿

阅读入口为 [sn-article.pdf](sn-article.pdf)，主文件为
[sn-article.tex](sn-article.tex)。这份稿件与相邻的
[Model 0923 英文稿](../model_0923_article/README.md) 分别维护。

| 文件 | 用途 |
| --- | --- |
| `sn-article.tex` | 汇集前言、Results、Methods 和参考文献 |
| [sections/results.tex](sections/results.tex) | 原有 Results 文稿 |
| `sections/opening.tex`、`sections/methods.tex` | 当前为空，预留前言与 Methods |
| `sn-bibliography.bib` | 随稿保存的 BibTeX 文献库 |
| `sn-article.bbl` | 随稿保存的参考文献排版文件，当前引用列表为空 |
| `.latexmkrc` | 加载共享模板和文献搜索路径 |
| `build/previous/` | 整理前的本地编译状态，记录的路径对应旧目录 |

PDF、分节内容和参考文献文件在此次目录整理中原样保留。

## 编译

从仓库根目录进入本稿目录，以临时目录保存编译输出：

```bash
cd CategoryLearning_paper/sn_article
paper_build="$(mktemp -d /tmp/categorylearning-sn-article.XXXXXXXX)"
latexmk -pdf -bibtex- -interaction=nonstopmode -halt-on-error -outdir="$paper_build" sn-article.tex
```

当前编译使用随稿保存的 `.bbl`。补充文献引用后，使用 `latexmk` 的默认
BibTeX 流程更新参考文献。编译后的 PDF 和日志位于新建的临时目录。
