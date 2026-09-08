# CategoryLearning_gitcode

类别学习的实验数据、共享计算模型、行为分析与论文工作区。
期刊文章和博士论文共用 `src/Bayesian_state/` 中的模型实现，通过各自配置组织实验。

## 从哪里开始

- [项目管理与清理规则](#仓库管理)：目录职责、保留策略和日常工作流。
- [共享模型说明](src/Bayesian_state/README.md)：模型模块、推断、拟合与评价接口。
- [期刊工作流](CategoryLearning_codes/Bayesian_model/README.md)：期刊配置、数据范围和版本冻结。
- [期刊配图](CategoryLearning_codes/figures/README.md)：Fig1/Fig2 代码与当前图版本。
- [自动化协作规则](AGENTS.md)：修改边界、科学正确性及验证要求。

## 目录职责

| 目录 | 用途 |
| --- | --- |
| `src/Bayesian_state/` | 期刊与博士论文共享的模型实现，修改机制的唯一维护位置 |
| `src/Bayesian/`、其他 `src/` 模型目录 | 基线、历史或其他模型家族，按各自研究用途保留 |
| `CategoryLearning_codes/` | 期刊配置、工作流、分析、配图代码和未确认图产物 |
| `CategoryLearning_paper/` | 期刊 LaTeX、参考资料；`figures/` 只放已确认图 |
| `src/Bayesian_state/docs/model_architecture/` | 模型技术文稿与历史模型定义；当前依据为 `model_0826.tex` |
| `data/exp123/` | 期刊数据，也是博士论文数据的一部分 |
| `data/exp4/`、`data/exp5/`、`data/meg/` | 博士论文其他实验数据 |
| `configs/{shared,exp123,exp4,exp5,meg}/` | 共享定义与按实验分组的配置；MEG 当前只预留位置 |
| `src/Bayesian_state/workflows/`、`notebooks/` | 模型运行/分析/报告/性能检查，以及探索与预处理 |
| `tests/` | 共享模型测试；期刊另有就近测试目录 |
| `results/`、`logs/` | 研究输出、报告、运行日志；不按文件年龄自动删除 |
| `src/Bayesian_state/docs/` | 工作流、管理文档和历史设计记录 |
| 系统临时目录 | 绘图库缓存与临时验证；不再维护仓库根 tmp/ |

## 可用入口

从仓库根目录运行。下列帮助命令不会启动拟合：

```bash
python -m src.Bayesian_state.run_simulation --help
python -m src.Bayesian_state.optimization.cli --help
python -m src.Bayesian_state.run_recovery --help
python -m CategoryLearning_codes.Bayesian_model.run_recovery --help
```

共享模型和期刊的相关检查：

```bash
python -m pytest -q CategoryLearning_codes/Bayesian_model/tests
python -m pytest -q CategoryLearning_codes/figures CategoryLearning_codes/tests
python -m pytest -q tests/bayesian_state/test_model_0826_versioning.py
```

根 `pytest.ini` 的默认收集范围是 `tests/`；仅运行 `python -m pytest` 不包含所有期刊测试。
依赖列表在 `requirements.txt`；目前未指定统一 Conda 环境，也未声明完整开发环境锁文件。
运行代码还需 PyYAML，运行测试需 pytest；正式复现时应记录实际 Python/依赖版本。

0826 当前冻结流程的二分类支持，不能直接等同于四分类、部分反馈或 MEG 任务已获验证。
既有初步结果也不等同于全体被试的正式拟合。具体边界见共享模型和期刊说明。

数据/配置迁移见 [第一阶段记录](src/Bayesian_state/docs/maintenance/LAYOUT_MIGRATION_20260908.md)；
文档/脚本/报告归属调整见 [第二阶段记录](src/Bayesian_state/docs/maintenance/OWNERSHIP_MIGRATION_20260908.md)。

## 仓库管理

### 文件保留规则

| 类型 | 处理方式 |
| --- | --- |
| `__pycache__`、`.pytest_cache` 等未跟踪运行缓存 | 仓库清理时可重建；排除数据、结果、图产物和版本控制目录后删除 |
| LaTeX `.aux/.fls/.fdb_latexmk/.out/.synctex.gz` | 构建中间文件；确认无编译任务后可清理。不要把 `.bbl`、`.pdf` 一概当缓存 |
| 系统临时目录 | 本地缓存与临时验证；仓库原 tmp/ 已检查并清理 |
| notebook checkpoint、备份文件 | 可能含未保存工作，先比较内容 |
| 图的早期版本 | 保留已确认版本、当前候选和支撑分析；其他版本逐批审查引用后删/归档 |
| 迁移清单、数据修正审计、数值回归 fixtures | 保留：用于证明数据/模型变更及复现 |
| `results/cache/`、PF 轨迹、checkpoint | 研究产物，重建成本可能很高，不能套用 Python 缓存规则 |
| 原始/处理后数据、正式结果与文稿版本 | 不自动删除；指定路径、依赖及备份后再决定 |

Git 忽略只决定是否跟踪，不表示文件可删。Git 无法恢复从未提交过的结果或图。
删除研究产物前，应记录具体路径、用途、引用检查、是否能重建、备份位置及预计释放空间。
归档也不要只把几十 GiB 从一个仓库目录搬到另一个目录；需要实际的外部归档位置。

### 输出与版本

- 新运行必须用新目录，沿用各工作流既有命名，不覆盖旧运行。
- 图代码继续按 `figures/fig1/`、`figures/fig2/` 分组，图产物按 `outputs/fig1/`、`outputs/fig2/` 分组。
- `CategoryLearning_paper/figures/` 仅保存确认的 FigureN；默认 PNG。
- 新研究运行记录代码 commit/工作区改动、配置、数据来源/哈希、随机种子、被试/试次范围、
  环境和完成状态。未完成或初步输出要明确标记，不能仅由目录名推断质量。
- 发表冻结使用 Git tag + 数据/输出归档；需要复现时使用对应 worktree。
  后续博士论文开发不维护另一份核心源码。

### 日常工作流

1. 查看 Git 状态，确认本次工作与已有未提交修改的边界。
2. 修改最近的实现/配置；增加新脚本时在相关 README 写清输入、输出和用途。
3. 做最小有效验证，避免为整理文档启动大规模搜索或重拟合。
4. 更新当前导航/状态文档。历史 QA、计划和 migration manifest 保留原始事实，必要时加历史标记。
5. 提交时按科学改动、重构、管理文档拆分；不要把当前工作区所有内容一并暂存。
6. 每个阶段结束后审核临时产物；投稿/章节冻结时再审核大体量研究输出。

## 录音转写

`src/audio_to_word.py` 替代原 notebook，将一个被试的一个会话录音目录转换为
UTF-8 BOM CSV，严格使用 `iSession,iTrial,text` 三列。文件名 `12.wav` 对应
`iTrial=12`，按数字排序；会话号通过 `--session` 指定，不假设每会话有 320 条。
默认从输入路径 `data/<实验>/` 自动选择实验 prompt：

- `exp123`：头、脖子、腿、尾巴、四肢、躯干等身体部位的长度与比较。
- `exp4`：绿色、黄色、粉色、蓝色器官的长度与比较。
- `exp5`：格子的黑色/白色描述。

录音在其他位置时可用 `--experiment exp123`、`exp4` 或 `exp5` 显式指定。
未识别的路径（包括 meg）使用通用 prompt，也可用 `--experiment generic` 主动选择。
`--prompt-file` 可追加 UTF-8 背景及词汇；仅需自定义背景时配合 `--experiment generic`。
所有 prompt 都要求忠实转写，不补全规则或未提及特征，不要在背景中提供目标规则或被试答案。

先预览（不访问 API、不写文件）：

```bash
python -m src.audio_to_word data/exp5/raw/Recording/502_1 \
  --session 1 --output results/transcription/502_rec.csv --dry-run
```

API 密钥、地址及模型名暂留空。申请完成后设置环境变量 `DASHSCOPE_API_KEY`、
`DASHSCOPE_BASE_URL`、`DASHSCOPE_MODEL`，安装可选依赖 `python -m pip install dashscope`，
去掉 `--dry-run` 即可调用当前 DashScope ASR 适配入口。新接口协议若不同，修改
`make_transcriber`；参考 [Qwen ASR 官方接口文档](https://www.alibabacloud.com/help/zh/model-studio/qwen-asr-api-reference)。
本次没有实际联网转写验证。

每成功一条便原子保存 CSV；失败或空结果重试后终止，不将错误信息写入口头报告。
已有文件默认拒绝覆盖。核对输入属于同一被试、prompt/API 配置一致后，可以添加
`--resume` 跳过已有 `(iSession,iTrial)`，也可追加同一被试的另一个会话。
不同被试必须使用不同输出文件；修改 prompt 或模型后应使用新输出文件。
建议先输出到 `results/transcription/` 审核，再人工纳入 raw 数据。
