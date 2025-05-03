
### 3.1 Rational backbone

 We posit that each learner entertains a library of geometric “partition” hypotheses that carve the four-dimensional feature space into regions associated with category labels. Bayesian belief updating over this hypothesis space provides a normative engine that converts stimuli into graded category probabilities and refines those probabilities from trial to trial.

 **Hypothesis space.**
 A parametric family of soft Voronoi partitions maps the *feature space* $\mathbf X=[0,1]^D$ (here $D=4$) to the *category probability space* $\Delta^{N-1}$. Throughout we denote a *hypothesis* by $\theta=(k,\beta,\mathbf C)$, where $k\in\{1,\dots,K\}$ indexes a discrete partition rule, $\beta>0$ governs category sharpness, and $\mathbf C=\{\mathbf c_i\}_{i=1}^{N}$ are region centroids.  
 
 **Bayesian updation.**
 Participants begin with an uninformative prior over the $K$ candidate partition rules, $p(k)=\frac1K$, and a uniform prior over $\beta\in[\beta_{\min},\beta_{\max}]$.

A schematic posterior update for a single trial is:
$$
p(\theta\mid\mathbf x_t,c_t,r_t)
\;\propto\;
\underbrace{p(r_t\mid c_t,\mathbf x_t,\theta)}_{\text{feedback}}\;
\underbrace{p(c_t\mid\mathbf x_t,\theta)}_{\text{choice}}\;
p(\theta\mid\mathcal D_{1:t-1}).
\tag{1}
$$
Whereas the choice-likelihood is calculated as
$$
p(c=C_i\mid\mathbf x,\theta)=
\frac{\exp\!\bigl[-\beta\,d_i(\mathbf x)\bigr]}
     {\sum_{j=1}^{N}\exp\!\bigl[-\beta\,d_j(\mathbf x)\bigr]},
\qquad
d_i(\mathbf x)=\lVert\mathbf x-\mathbf c_i\rVert_2.
\tag{2}
$$ 
and feedback-likelihood $p(r\mid c,\mathbf x,\theta)$ is deterministic under the 0/0.5/1 schedule described in Sec.2. 

After $T$ trials the log-posterior is
$$
\log p(\theta\mid\mathcal D_{1:T})=
\log p(\theta)
\;+\;
\sum_{t=1}^{T}w_t
        \log p(\mathbf x_t,c_t,r_t\mid\theta)
\;-\;\log Z_T,
\tag{3}
$$
where $w_t\equiv1$ for the rational backbone and $w_t\le1$ once leaky memory is enabled (Sec.3.2).
For a fixed $k$ we obtain $\hat\beta_k$ by maximising the weighted log-posterior with respect to $\beta$ over the interval $[\beta_{\min},\beta_{\max}]$.
The pair $(\hat k,\hat\beta_{\hat k})$ with highest posterior mass constitutes the online estimate at trial $T$.

### 3.2 Bounded-rational adaptations
Three auxiliary modules—noisy perception, leaky memory and hypothesis-set jumping—tune how closely the learner can approximate the Bayesian ideal, allowing the model to reconcile optimal computations with realistic human constraints. 

**Noisy perception.**  Noisy perception injects sensory uncertainty, blurring the effective stimulus the learner sees. Observed features are corrupted by an independent Gaussian channel,
$$
\tilde{\mathbf x}
=\mathbf x+\boldsymbol\varepsilon,\qquad
\boldsymbol\varepsilon\sim\mathcal N(\mathbf0,\Sigma),
\tag{4}
$$
with diagonal $\Sigma$ collected from the Perception Calibration task (see Sec.2.1 and App. A).

**Leaky memory.** Leaky memory down-weights older evidence via an exponential decay, mirroring finite mnemonic capacity. Earlier trials receive exponentially decaying weights and rescales the log-likelihood sum in (3):
$$
w_j = w_0+\gamma^{T-j},\qquad 0\le\gamma\le 1 .
\tag{5}
$$
Here $\gamma$ controls the decay rate (smaller $\gamma$ ⇒ faster forgetting) and $w_0$ provides a baseline weight that every past trial retains; both parameters are estimated separately for each participant.

**Hypothesis-set jumping.** Hypothesis-set jumping restricts on-line inference to a small, dynamically changing subset of hypotheses, capturing the exploratory “generate–and–test” behaviour observed in Section 2.  At each trial we maintain a working set $\mathcal H_t$ of size $M$ built from
$$
\mathcal H_{t+1}=S_1^{(t)}\cup S_2^{(t)}\cup S_3^{(t)},
\tag{6}
$$
where
 $S_1^{(t)}$:  top-$m_1$ hypotheses by posterior probability (exploitation*);
 $S_2^{(t)}$:  $m_2$ hypotheses most similar to the current stimulus (cue-driven associative recall*);
 $S_3^{(t)}$:  $m_3$ uniformly sampled novel candidates (exploration*).
Analytic forms for similarity scores and the swap schedule are summarised in App. B.3.

### 3.3 Model fitting and comparison
**Model space.**

**Optimisation protocol.**

**Predictive assessment.**
