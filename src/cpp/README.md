# Model 1031 C++

这里维护十参数 `1031-unified-v2` 的 C++20 实现。模型、粒子滤波、参数优化和自主模拟
在 C++ 中执行；Python 读取结果并生成评价表和 PNG。数学参考为主仓库 `249144de`。
本目录由独立的 CategoryLearning1031cpp 迁入，保留其学习公式、随机协议和结果字段。
后续 Python 模型的机制更新需要显式迁移并验证，两个实现分别维护。

## 目录与依赖

| 目录 | 职责 |
| --- | --- |
| `src/`、`include/gm/` | 模型、几何、推断、优化、数据与结果读写、CLI |
| `configs/` | 三条件配置和十参数有限候选空间 |
| `resources/` | Cond1 的 29 条规则、Cond2/3 的 152 条规则及结构/功能距离矩阵 |
| `evaluation/` | 离线评价、绘图、论文表格和交付目录读取 |
| `tools/` | 批量运行、显式参数转移、结果打包及参考对照 |
| `tests/` | C++、Python 测试和冻结数值夹具 |
| `third_party/` | 固定依赖版本、下载哈希、许可证 |
| `docs/` | 字段说明、历史验证记录和本次迁移记录 |

运行从主仓库 `data/` 读取输入，C++ 目录中不保存数据副本。exp123 的三份 CSV
与原独立项目字节一致，来源见 `resources/data_sources.json`。MEG 读取主仓库当前
数据，内容与独立项目旧副本不同；旧 MEG 对照结果不能代表当前输入的数值验证。
`resources/source_manifest.json` 保存早期资源导出时的历史来源，包含旧数据哈希。
核对本次 exp123 输入时使用 `resources/data_sources.json`。
该清单同时记录 Git 存储版本与 Windows 工作副本的确切哈希，两者仅有换行符差别。
规则资源和冻结夹具由本目录 `.gitattributes` 保留原始字节，确保克隆后的资源身份一致。

主仓库其他模型不导入、不链接本目录。CMake、绘图依赖、配置和测试单独维护；
现有 Python 模型的运行入口与默认测试范围保持不变。生产 C++ 不调用 Python 模型。
`tools/compare_1031.py` 是显式调用 Python 参考实现的验证工具。

## 构建

需要 CMake >= 3.24、C++20 编译器和 OpenMP。Windows 推荐 Visual Studio 2022，
Linux 推荐 GCC。以下命令均在 **CategoryLearning 主仓库根目录**执行。

Windows：

```powershell
cmake -S src/cpp -B src/cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build src/cpp/build --config Release --parallel 8
ctest --test-dir src/cpp/build -C Release --output-on-failure
```

Linux：

```bash
cmake -S src/cpp -B src/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/cpp/build --parallel 8
ctest --test-dir src/cpp/build --output-on-failure
```

CMake 下载并校验固定版本依赖，生成目录已被 Git 忽略。
断网构建可通过 `-DFETCHCONTENT_SOURCE_DIR_JSON=/path/to/json` 等变量指定本地源码；
八项名称为 `JSON/EIGEN/YAML/ZLIB/CNPY/CSV/SHA/DOCTEST`，并设置
`-DFETCHCONTENT_FULLY_DISCONNECTED=ON`。源码和配置不依赖原独立项目的位置。

## 三条件配置

| 用途 | Cond1 | Cond2 | Cond3 |
| --- | --- | --- | --- |
| 小预算、16 个有效试次 | `configs/cond1.yaml` | `configs/cond2.yaml` | `configs/cond3.yaml` |
| 全被试、全试次、默认 12 workers | `configs/cond1_full_local.yaml` | `configs/cond2_full_local.yaml` | `configs/cond3_full_local.yaml` |
| 完整粒子记录、默认 8 workers | `configs/local_records8/cond1_full_local.yaml` | `configs/local_records8/cond2_full_local.yaml` | `configs/local_records8/cond3_full_local.yaml` |
| Linux、默认 64 workers | `configs/cond1_full_remote.json` | `configs/cond2_full_remote.json` | `configs/cond3_full_remote.json` |

表中路径相对本 C++ 目录。所有输入路径相对 **配置文件自身**解析：

```yaml
dataset: exp123
data_root: ../../../data
resource_root: ../resources
parameter_space: parameter_space.yaml
subjects: [122]
jobs: 1
max_trials: 16
```

上述示例适用于 `src/cpp/configs/`；其子目录中的配置需要相应增加 `../` 层数。
`data_root` 必须显式配置。删除 `max_trials` 使用完整试次；`--subjects` 和 `--jobs`
可限制单次运行。`fit` 中设置搜索候选、轮数及三个阶段的粒子数/种子数；
`evaluation` 中设置 PF 重复、自主模拟和粒子记录预算。候选空间保持十参数版本，
位于 `configs/parameter_space.yaml`，将某项候选改为单值可固定该参数。

## 小预算全链路

安装的 Python 依赖只用于本 C++ 项目的离线处理：

```powershell
python -m pip install -r src/cpp/evaluation/requirements.txt
.\src\cpp\build\Release\genmodel_cpp.exe fit --config src/cpp/configs/cond1.yaml --subjects 122 --jobs 1 --output results/cpp/NEW_c1_fit
.\src\cpp\build\Release\genmodel_cpp.exe evaluate --config src/cpp/configs/cond1.yaml --subjects 122 --jobs 1 --fit-dir results/cpp/NEW_c1_fit --output results/cpp/NEW_c1_eval
python -B src/cpp/evaluation/plot_results.py --input results/cpp/NEW_c1_eval --output results/cpp/NEW_c1_png
```

Cond2/3 分别替换配置与被试编号，例如 `206`、`314`。
Linux 的可执行文件为 `./src/cpp/build/genmodel_cpp`。输出路径相对当前工作目录，
每次使用新目录。已有已完成结果拒绝覆盖。拟合恢复使用 `fit --resume`，必须匹配
数据、模型身份、候选空间、预算和随机协议。迁移修改了代码身份；旧参数可通过
`tools/export_evaluation_configs.py` 显式转移，保留旧结果身份后重新评价。

原生 CLI 的输出必须位于配置的 `data_root` 之外，恢复拟合也执行此检查。
检查会解析目录链接；Windows 批处理在生成日志和启动任务前同样保护主仓库 `data/`。
文件系统权限用于控制其他进程对目录的并发修改。

自主模拟可独立运行：

```powershell
.\src\cpp\build\Release\genmodel_cpp.exe simulate --config src/cpp/configs/cond1.yaml --subjects 122 --jobs 1 --fit-dir results/cpp/NEW_c1_fit --output results/cpp/NEW_c1_sim
```

`evaluate` 的条件推断基于真实历史；自主模拟根据自己的 choice 获得任务反馈。
PF seed 重复刻画推断的数值变动，自主路径 bands 刻画生成轨迹的分布。
最接近被试的单条路径属于描述性比较，参数解释还需要恢复与辨识检验。

## 批量运行

Windows 可从任何工作目录调用脚本；相对脚本参数按照 C++ 目录解析：

```powershell
& ./src/cpp/tools/run_cond123_local.ps1 -PlanOnly
& ./src/cpp/tools/run_cond123_local.ps1 -Jobs 8 -Conditions 1 -ConfigDirectory configs/local_records8 -ExportPaper -PythonExe python
```

移除 `-Conditions 1` 顺序处理三个条件。被试串行，每个原生任务内单层并行。
默认输出为 `src/cpp/results/local_cond123_时间戳`，可用 `-OutputRoot` 指定新目录。
`-PlanOnly` 只显示计划；正式全量运行需预留时间和磁盘。
Linux runner 的帮助为 `python -B src/cpp/tools/run_cond123_remote.py --help`；
它在部署机器上前台执行，检查 64 线程等效空闲及磁盘，保留四小时超时保护。
本次迁移不连接或同步服务器。

## 结果记录与交付

- `best_parameters.json` 保存最优参数，JSONL 与缓存保存优化经过。
- `evaluation_inputs.npz` 保存预测、条件状态、逐规则概率和自主模拟。
  H0/H4/H6 使用稳定目录规则 ID，字段见 [逐规则说明](docs/catalogue_probabilities.md)。
- `evaluation.export_particles: true` 保存首个 PF seed 的全部粒子，时点是当前 choice
  已知、当前 feedback 尚未学习。字段见 [条件粒子说明](docs/conditional_particle_contents.md)。
- `content_path.json` 记录固定第 0 条自主路径，不代表所有路径或条件 PF 状态。
- `evaluation/export_paper_tables.py` 导出逐规则和特征 CSV；`tools/package_results_1031.py`
  对已完成的 96 人完整记录批次做无损压缩与 Python 字段适配。旧结果可用显式
  `--input` 指向原独立项目中的批次；打包不运行模型。
- [协作字段映射](docs/collaboration_field_map.md) 说明可直接使用的 Python1031 字段及缺失项。

## 测试与维护

测试额外需要 pytest 和 PyYAML，可用 `python -m pip install -r src/cpp/requirements-dev.txt`
安装本子项目的开发依赖。

```powershell
ctest --test-dir src/cpp/build -C Release --output-on-failure
python -B -m pytest -c src/cpp/pytest.ini src/cpp/tests -q -p no:cacheprovider
python -B src/cpp/tools/compare_1031.py self-test
```

C++ 与 Python 测试都在本目录维护，主仓库默认 pytest 不包含它们。
已有数值夹具来自迁移前冻结实现，修改模型时须独立解释和验证数值差异。
临时输出放在本目录已忽略的 `.tmp/`，编译产物放在 `build/`。Git 提交仅包含源码、
配置、许可证、文档和小型冻结夹具。构建缓存、原始数据、研究结果和日志分别保留。
迁移范围与验证结果见 [迁移记录](docs/migration_validation.md)。
