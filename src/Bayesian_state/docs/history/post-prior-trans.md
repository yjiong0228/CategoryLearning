# Graph Diffusion for Hypothesis Prior Transition

## 1. 背景与问题

在当前模型中，我们需要将上一试次的 posterior 分布转换为下一试次的 prior，尤其是为 **newcomer hypotheses** 分配合理的初始概率。

现有方法本质为：

\[
p_{\text{new}}(h) \propto \sum_{h' \in \text{old}} \text{sim}(h, h') \cdot p_{\text{old}}(h')
\]

该方法存在局限：

- 仅使用 **一跳（1-hop）相似性**
- 未利用 hypothesis space 的**全局结构**
- 忽略高阶关系（如 A → B → D）

---

## 2. 核心思想：基于图的概率扩散（Graph Diffusion）

我们将整个 hypothesis space 表示为一个图：

- 每个 hypothesis = 一个节点
- 相似度 = 边权

定义相似矩阵：

\[
W_{ij} = \text{similarity}(h_i, h_j)
\]

---

## 3. 构造随机游走（Random Walk）

对相似矩阵做行归一化：

\[
\tilde W = D^{-1} W,\quad D_{ii} = \sum_j W_{ij}
\]

性质：

\[
\sum_j \tilde W_{ij} = 1
\]

👉 可解释为：

> 从节点 \(i\) 出发，转移到 \(j\) 的概率

---

## 4. 扩散模型（Random Walk with Restart）

定义：

- \(p_0 \in \mathbb{R}^N\)：上一 trial 的 posterior（全空间表示）
- \(q \in \mathbb{R}^N\)：扩散后的 prior

扩散方程：

\[
q = \alpha p_0 + (1-\alpha)\tilde W q
\]

其中：

- \(\alpha \in [0,1]\)：控制保守 vs 扩散
  - 大 \(\alpha\)：偏向保留旧 belief
  - 小 \(\alpha\)：偏向结构传播（探索）

---

## 5. 直觉解释

该过程等价于：

> 在图上做随机游走，但每一步都有概率 \(\alpha\) 回到初始分布 \(p_0\)

因此：

- 不仅考虑直接邻居（1-hop）
- 还考虑多跳路径（multi-hop）
- 利用整个 graph 的结构信息

---

## 6. 求解方法

该方程可通过迭代求解：

```python
q = p0.copy()
for _ in range(T):  # T ≈ 10~30
    q = alpha * p0 + (1 - alpha) * W_tilde @ q
```

通常快速收敛。

---

## 7. Active Set 投影

扩散是在 **全 hypothesis space** 上进行，但当前 trial 只使用 active hypotheses。

定义 mask：

* \[(m_{active} \in {0,1}^N)\]
投影：

\[
q_{\text{active}} \propto q \odot m_{active}
\]

并归一化：

```python
q_active = q[active_indices]
q_active /= q_active.sum()
```

---

## 8. 数值稳定性

### 8.1 防止概率塌缩

加入底噪：

```python
q_active = (1 - epsilon) * q_active + epsilon / len(q_active)
```

---

### 8.2 图连通性问题

若 graph 存在 disconnected components：

* 扩散可能无法覆盖某些节点

解决：

* 在 (W) 中加入小常数
* 或保证 similarity matrix 稠密

---

## 9. 与现有方法的对比

| 方法                | 特点               |
| ----------------- | ---------------- |
| 当前方法              | 一跳 similarity 加权 |
| 图扩散               | 利用全局结构，多跳传播      |
| Optimal Transport | 全局最优映射（更复杂）      |

👉 图扩散可视为：

> 在计算成本与表达能力之间的折中方案

---

## 10. 与当前模型的集成

当前 pipeline：

```
posterior(t-1) → prior(t) → likelihood → posterior(t)
```

修改点：

* 替换 `posterior → prior` 转换部分
* 保持其他模块（likelihood / beta / inference）不变

---

## 11. 与认知建模的对应关系

该方法可解释为：

* hypothesis space 是一个 **认知流形（cognitive manifold）**
* belief 更新是：

  * memory retention（(p_0)）
  * structure-based generalization（diffusion）

---

## 12. 可选扩展

### 12.1 confidence-based α

\[
\alpha = f(\text{confidence})
\]

* 高 confidence → 高 α（稳定）
* 低 confidence → 低 α（探索）

---

### 12.2 survivor 混合

\[
prior = \lambda \cdot q_{\text{diffusion}} + (1-\lambda)\cdot prior_{\text{survivor}}
\]

---

### 12.3 子图扩散（高效版本）

只在以下节点上做 diffusion：

* active hypotheses
* newcomer hypotheses
* top-K 相似邻居

---

## 13. 总结

该方法本质是：

> 将 hypothesis similarity 从局部启发式
> 提升为基于图结构的全局概率传播

优点：

* 利用高阶结构
* 提供平滑、稳定的 prior
* 与贝叶斯更新自然兼容

---

## 14. 一句话总结

> 用图扩散替代一跳相似性，实现基于 hypothesis manifold 的 belief propagation

