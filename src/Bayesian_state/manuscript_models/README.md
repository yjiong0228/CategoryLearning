# manuscript_models

本目录保存论文模型系列的独立、冻结数值实现。它们用于方法推导、模型恢复、精确枚举、
particle-filter 数值审计和历史结果复现。

这些文件不是当前 `StateModel` 正式运行入口。真实数据的标准 simulation/Hyper-CD 应从
`configs/` 和 `optimization/` 启动；可复用的认知机制应迁入 `problems/modules/`。

## 模型系列

| 文件 | 主要内容 |
|---|---|
| `model_0803.py` | full-set/动态模型、transition kernels、参数解码与拟合、固定 `q` 输入 |
| `model_0804.py` | hard finite workspace、FA0/FA1/FA2/FA2R、粒子/精确/alive/resample-move 推断、RT likelihood |
| `model_0804_recovery.py` | 0804 autonomous choice/feedback recovery simulation |
| `model_0804_forgetting.py` | filtered anchor state 与 coupled forgetting history 分析 |
| `model_0804_pgas.py` | innovation path replay、exact smoothing 和 PGAS |
| `model_0806.py` | surprise/uncertainty-driven FA3-M autonomous simulation 与 RT emission |

`model_0804.py` 是该系列的数值基础；0806 复用其中的 finite-workspace state、transition 和
feedback update。

## 与主框架的关系

已经主框架化的 0806 choice 机制位于：

```text
problems/modules/hypo_transition/dynamic_continuous.py
inference_engine/backends/particle_filter.py
configs/model_struct/pmh_model_cond1_0806.yaml
```

目前仍只在本目录保留的能力主要包括：

- 基于 model_0803 预计算 `q` 的冻结输入路径
- autonomous recovery generation
- RT emission 与 choice–RT joint likelihood
- exact enumeration、alive filter、PGAS 等专用数值诊断
- 旧 rolling/model-averaging 实验需要的辅助表示

因此，“主框架结果”和“manuscript numerical result”不能仅凭同名参数假定逐 trial 完全相同；
两者的 perception/likelihood 输入、随机映射或 memory 初始化可能不同。

## 何时在这里改代码

适合：

- 验证论文公式；
- 构造小空间 exact oracle；
- 运行历史 recovery；
- 在迁移前验证新机制是否可识别。

不适合：

- 新增正式数据入口；
- 再实现一套公共 Brier/NLL；
- 复制 Hyper-CD、数据切分或结果写盘；
- 长期维护与 `StateModel` 重复的完整模型。

若某项机制将用于正式模型比较，应把最小机制迁入主框架，并用本目录实现作为 parity oracle。
