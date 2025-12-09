---
agent: agent
---
## 简介
Bayesian_state 是一个建模人类类别学习的模型代码库：
整个模型采用贝叶斯框架，并且模块化，由 bayesian_engine 的 BaseEngine 调度各个模块，输入一个 trial 的数据后，各个模块依次更新。主要模块包括 perception(向刺激的感知添加噪声)，likelihood(空间位置到hypo的似然计算)，memory(维护记忆 state，负责遗忘行为和贝叶斯更新)，hypo_transitions(维护 hypos 集合，每一步决定集合是否变化，并进行 posterior_t->prior_{t+1} 转换)。