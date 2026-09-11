# Figure development

代码按图号分组，图产物按相同图号独立归档。

```text
figures/
├── fig1/       # Fig1 实现、配置、测试、修改记录
├── fig2/       # Fig2 实现、诊断代码、规划与修改记录
└── outputs/
    ├── fig1/   # Fig1 各版本和相关审计
    └── fig2/   # Fig2 各版本和相关审计
```

- [Fig1 代码与说明](fig1/README.md) · [已确认 v10](outputs/fig1/fig1_v10/fig1_behavior_draft.png)
- [Fig2 代码与说明](fig2/README.md) · [完整草稿 v15](outputs/fig2/fig2_complete_v15/Figure2_draft.png)
- [图产物索引](outputs/README.md)

从仓库根目录运行，使用新的输出版本目录：

```bash
python -m CategoryLearning_codes.figures.fig1.build_fig1 --output CategoryLearning_codes/figures/outputs/fig1/fig1_v11
python -m CategoryLearning_codes.figures.fig2.build_complete --output CategoryLearning_codes/figures/outputs/fig2/fig2_complete_v10
python -m pytest -q CategoryLearning_codes/figures CategoryLearning_codes/tests
```

Fig1 配置默认从 `fig1/config.json` 读取。Figures 根目录的 `conftest.py` 是共享的 pytest 配置，阻止收集 outputs 中的历史测试快照；其余实现代码均在对应 fig 目录中。

输出、快照和已有图不随代码整理重写。论文 `CategoryLearning_paper/figures/` 仅保留已确认图。
输出迁移与代码迁移分别记录在 `outputs_reorganization_20260908.json`、`code_reorganization_20260908.json`。
