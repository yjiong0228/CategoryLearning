# State-Based Model

## Architecture

### Inference Engine (`BaseEngine`)
The `BaseEngine` is the core controller. It maintains shared informatoiin and the current state of the model, including:
- `hypotheses_set`: The set of all possible hypotheses.
- `hypotheses_mask`: A mask indicating currently active hypotheses.
- `prior`, `likelihood`, `posterior`: Probability distributions.
- `agenda`: A list of module names defining the execution order for each inference step.

The `infer_single` method executes the modules in the order defined by `agenda`.

### Modules

#### 1. Perception (`BasePerception`)
- **Function**: Processes raw stimuli and calculates perceptual errors/noise.
- **Implementation**: Loads processed data, calculates mean $\mu$ and standard deviation $\Sigma$ of reconstruction errors (e.g., `neck_length`, `head_length`). For data $x_t$, it samples a noise $\varepsilon\sim\mathcal{N}(\mu,\Sigma)$ and adds it to data as percepted stimuli: $\tilde{x_t}=x_t+\varepsilon$

#### 2. Hypothesis Transition (`FixedNumHypothesisModule`)

- **Function**: Manages the set of active hypotheses. It simulates the limited capacity of working memory.
- **Mechanism**:
    - Maintains a fixed number of active hypotheses (`fixed_hypo_num`).
    - **Sampling**: At each step, it may discard some hypotheses and sample new ones from the pool.
    - **Strategies**:
        - `random`: Randomly drops and samples.
        - `top_posterior`: Drops hypotheses with the lowest posterior.
    - **Posterior-to-prior transition**: when the active hypothesis set is updated, there's a transition $\mathbf{post}_t(h_t), h_t\in\mathcal{H}_t\to\mathbf{prior}_{t+1}(h_{t+1}), h_{t+1}\in\mathcal{H}_{t+1}$. 
      - For new hypos $h_+\in\mathcal{H}_{t+1}\setminus\mathcal{H}_t$, $\text{score}(h_+)=\frac{\sum_{h\in\mathcal{H_t}}s(h_+,h)\mathbf{post}_t(h_t)}{\sum_{h\in\mathcal{H_t}}s(h_+,h)}$;
      - For survivors $h_s\in\mathcal{H}_{t+1}\cap\mathcal{H}_t$, $\text{score}(h_s)=\mathbf{post}_t(h_s)$
      - Normalization factor: $Z_t = \sum_{h\in\mathcal{H}_{t+1}}\text{score}(h)$
      - $\mathbf{prior}_{t+1}(h) = \frac{\text{score}(h)}{Z_t},h\in\mathcal{H}_{t+1}$

#### 3. Likelihood (`PartitionLikelihood`)
- **Function**: Computes $P(D|h)$, the likelihood of the current observation $D=(x,c,r)$ given each hypothesis $h$.
- **Formulation**: Depends on the specific partition model (e.g., distance-based similarity).

#### 4. Memory (`DualMemoryModule`)
- **Function**: Updates the belief state based on the history of observations, incorporating forgetting and persistence.
- **Mathematical Formulation**:
    Raw formulation:
    $$\log p(h|D_{1:t})=\log p(h)+\sum_{\tau=1}^t w^\tau \log p(D_\tau|h) - \log Z_t$$
    where $w^\tau = w_0+\gamma^{t-\tau} $
    The model maintains two state components for each hypothesis $h\in\mathcal{H}_t$:
    1.  **Fading State**: $ S_{fade}^{(t)} = \gamma S_{fade}^{(t-1)} + \log P(D_t|h) $
    2.  **Static State**: $ S_{static}^{(t)} = S_{static}^{(t-1)} + \log P(D_t|h) $
    
    **Posterior Calculation**:
    The log-posterior is a weighted combination of these two states:
    $$ \log P(h|D_{1:t}) \propto w_0 S_{static}^{(t)} + (1-w_0) S_{fade}^{(t)} $$
    
    Where:
    - $\gamma$: Decay rate (0 < $\gamma$ < 1).
    - $w_0$: Weight of the static component (0 < $w_0$ < 1).

    **State Transition**:
    When the set of active hypotheses changes (handled by Hypothesis Transition module), the memory states are adjusted to align with the new prior:
    *目前实现有问题*：
    - 尝试1：staic=fade，导致新 hypos 很快取代旧的
    - 尝试2：使新 hypos staic-fade = mean(static - fade)[旧的上的]，结果看上去可以
    - 尝试3：用假的 likelihood（1/N_tau）维持 baseline states，依然新的会很快取代旧的
    - 尝试4：

## Execution Flow
For each observation $D_t$:
1.  **Perception**: Process $D_t$.
2.  **Hypothesis Transition**: Update the active hypothesis mask $\mathcal{H}_{t-1}\to\mathcal{H}_{t}$.
3.  **Likelihood**: Compute $P(D_t|h)$.
4.  **Memory**: Update $S_{fade}$ and $S_{static}$, then compute the new posterior $P(h|D_{1:t})$.
