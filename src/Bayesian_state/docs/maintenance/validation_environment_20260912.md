# 优先修订的验证环境（2026-09-12）

本文件记录此次修订所用的现有环境，不是完整环境锁，也不代表已经在 requirements.txt
声明的全部版本上通过验证。没有安装、升级或重装任何依赖。

Python：3.11.10。

| 包 | 本次实际版本 |
| --- | --- |
| joblib | 1.4.2 |
| matplotlib | 3.9.2 |
| numba | 0.60.0 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scipy | 1.13.1 |
| scikit-learn | 1.5.1 |
| seaborn | 0.13.2 |
| tqdm | 4.66.5 |
| PyYAML | 6.0.2 |
| pytest | 7.4.4 |

requirements.txt 保留此前的数值依赖版本，并补充 PyYAML==6.0.2；requirements-dev.txt
引用核心依赖并声明 pytest==7.4.4。实际环境中 NumPy、Numba 等版本与核心声明不同，
正式发表冻结前仍需选择目标环境，重新验证并归档完整依赖与 Python 版本。
可选的 PyTorch / DashScope 不属于这次核心与文件写入验证范围。

验证覆盖 StreamList 顺序读取、simulation 文件保护与短序列执行，以及正式套件收集和
相关数值回归。未运行正式拟合、完整参数恢复或原始录音联网转写。

## 本次验证结果

- 默认 `python -m pytest --collect-only -q` 正常收集 `tests/` 中的 410 项测试。
- 显式运行全部正式目录：483 项通过（31.09 秒），包含 33 项新增读取/文件保护测试及
  期刊保存的合并前数值参考；没有重新生成参考。
- 三个相关 CLI（共享 simulation、期刊 simulation、hyper-then-simulation）的 `--help` 均正常退出。
- 依赖声明语法与 `git diff --check` 通过，独立代码复核未发现阻断问题。

完整测试命令（绘图库和 Numba 缓存使用系统临时目录，BLAS/OMP 线程限制为 1）：

```bash
python -m pytest -q -p no:cacheprovider tests CategoryLearning_codes/Bayesian_model/tests CategoryLearning_codes/figures CategoryLearning_codes/tests
```

最终运行有 11 条警告：10 条来自已有源码字符串的无效转义序列，1 条来自四试次、
两粒子文件写入 smoke 中的 NumPy 分位数计算。后者在修复前同一 smoke 也出现；
此次未修改数值计算来消除警告。该 smoke 只验证文件流程，不能用于科研推断。
