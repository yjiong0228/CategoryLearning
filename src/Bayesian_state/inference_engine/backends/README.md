# inference backends

本目录只实现“如何对已装配的 `StateModel` 做隐状态推理”，不定义认知机制、loss 或
hyperparameter search。

| Backend | 入口 | 含义 |
|---|---|---|
| trajectory | `run_state_model_trajectory()` | 条件于一个随机实现，返回单条 prior/posterior 状态轨迹 |
| particle filter | `run_state_model_particle_filter()` | 对多条潜在状态轨迹加权、重采样并返回边际预测 |

两者共享 `BaseEngine` 的状态协议。粒子后端额外要求所有有状态 module 正确实现
`state_dict()`、`load_state_dict()`、`clear_logs()`；随机 module 还应实现
`reseed_future(seed)`。

调用方通常不应自己判断 YAML 字符串，而应通过：

```python
from src.Bayesian_state.inference_engine.dispatcher import run_inference_backend
```

新增 backend 时，需要同步 dispatcher、backend result 类型、README 和定向测试，但不能在新
backend 中复制 transition、likelihood、memory 或 beta 更新公式。
