
### 3.1 Rational backbone

We posit that each learner entertains a library of geometric “partition” hypotheses that carve the four-dimensional feature space into regions associated with category labels, following classic concept-learning works showing that humans preferentially generate and test explicit rules (Bruner, Goodnow & Austin 1956; Feldman 2000) rather than tuning infinitesimal weights. More importantly, Bayesian belief updating over this hypothesis space provides a normative engine that converts stimuli into graded category probabilities and refines those probabilities from trial to trial.

 **Hypothesis space.**
We formalise the hypothesis space $\mathcal{H}$ as a parametric family of soft Voronoi partitions maps the *feature space* $\mathbf X=[0,1]^D$ (here $D=4$) to the *category probability space* $\Delta^{N-1}$. Throughout we denote a *hypothesis* by $\theta=(k,\beta)$, where $k\in\{1,\dots,K\}$ indexes a specific partition rule drawn from a hand-authored set of $K$ geometric splits (axis-aligned, oblique, hierarchical, …, see App.B for more details), and $\beta>0$ governs category sharpness, modulating how deterministically that rule is applied. This keeps the hypothesis space expressive enough to approximate arbitrary decision surfaces, yet finite enough for efficient Bayesian updating on a trial-by-trial basis.
 
 **Bayesian updation.**
Participants begin with an uninformative prior over the $K$ candidate partition rules, $p(k)=1/K$, and a uniform prior over $\beta\in[\beta_{\min},\beta_{\max}]$. This prior is supported by work showing that naïve subjects start with flat priors unless given reason to do otherwise (Griffiths & Tenenbaum 2006). 

By Bayes’ rule, a schematic posterior update for a single trial $t$ is $p(\theta\mid\mathbf x_t,c_t,r_t)\;\propto\;p(\mathbf x_t,c_t,r_t\mid\theta)\;p(\theta)$, where $p(\theta)$ is the current prior belief over partition hypothesis $\theta$ and $p(\mathbf x_t,c_t,r_t\mid\theta)$ is the likelihood of this trial, which can be decomposed into: 
$$p(\mathbf{x}_t, c_t, r_t\mid\theta) = p(r \mid\mathbf{x}_t, c_t, \theta)\;p(c_t \mid\mathbf{x}_t, \theta)\;p(\mathbf{x}_t \mid\theta). \tag{1}$$
Here choice-likelihood $p(c_t\mid\mathbf x_t,\theta)$ is the probability that a learner holding hypothesis $\theta$ would emit the observed choice $c_t$ when presented with stimulus $\mathbf x_t$; in the model it is given by the soft-max distance rule in App.B. Feedback-likelihood $p(r_t\mid c_t,\mathbf{x}_t,\theta)$ is the probability that the environment would return feedback $r_t$ for that choice–stimulus pair under the mapping implied by $\theta$; in our task this term is deterministic (0, 0.5, or 1). Finally $p(\mathbf{x}_t\mid\theta)$ is a stimulus-generation constant actually independent of $\theta$. 

Aggregating evidence up to and including trial $T$, $\mathcal D_{1:T}\;=\;\bigl\{(\mathbf x_t,c_t,r_t)\bigr\}_{t=1}^{T}$, we could calculate the log-posterior as
$$
\log p(\theta\mid\mathcal D_{1:T})=
\log p(\theta)
\;+\;
\sum_{t=1}^{T}w_t
        \log p(\mathbf x_t,c_t,r_t\mid\theta)
\;-\;\log Z_T,
\tag{2}
$$
where ${{Z}_{T}}=p({{\mathcal D}_{1:T}})$ is a normalization constant which can be ignored in subsequent parameter estimation, and $w_t\equiv1$ is the likelihood weight for the rational backbone ($w_t\le1$ once leaky memory is enabled, see Sec.3.2).

### 3.2 Bounded-rational adaptations
Three auxiliary modules—noisy perception, leaky memory and hypothesis-set jumping—tune how closely the learner can approximate the Bayesian ideal, allowing the model to reconcile optimal computations with realistic human constraints. 

**Noisy perception.**  Noisy perception injects sensory uncertainty, blurring the effective stimulus the learner sees. Therefore, we assume that observed features are corrupted by an independent Gaussian channel,
$$
\tilde{\mathbf x}
=\mathbf x+\boldsymbol\varepsilon,\qquad
\boldsymbol\varepsilon\sim\mathcal N(\mathbf0,\Sigma),
\tag{3}
$$
with $\Sigma$ fixed per participant and collected from the Perception Calibration task (see Sec.2.1 and App. A).

**Leaky memory.** Forgetting curves in both working and episodic memory follow an exponential law (Wixted, 2004; Zhang & Luck, 2009). Therefore, we under-weight older evidence via an exponential decay, rescaling the log-likelihood sum in (1):
$$
w_j = w_0+\gamma^{T-j},\qquad 0\le\gamma\le 1 .
\tag{4}
$$
Here $w_j$ quantifies how strongly the observation from trial $j$ (with $1\!\le\! j\!\le\! t$) contributes to the current log-posterior update at trial $T$. $\gamma$ controls the decay rate (smaller $\gamma$ ⇒ faster forgetting) and $w_0$ reflects an irreducible baseline trace—conceptually similar to a “long-term store” that resists decay (Oberauer, 2009). By fitting $(\gamma,w_0)$ per participant, the model can express a spectrum of recency effects and capture systematic individual differences reported in human learning studies (Peterson & Anderson 1988; Collins & Frank 2012).

**Hypothesis-set jumping.** In order to capture the local drastic fluctuations in behaviour observed in Section 2, we assume that participants restrict on-line inference to a small, dynamically changing subset of hypotheses, rather than keeping a complete set of hypotheses from beginning to end.
This mechanism can be formalised as a **three-way strategy trade-off**: *posterior-based exploitation* ($S_1$), *cue-driven associative recall* ($S_2$), and stochastic *exploration* ($S_3$). The amount of each strategy varies dynamically across trials. 
At every trial $t$ the learner firstly decides the amount of hypotheses to draw from each strategy, $(m_1^{(t)}, m_2^{(t)}, m_3^{(t)})$, then generates the next working set $\mathcal H_{t+1}$ from
$$
   \begin{aligned}
   S_1^{(t)} &= \text{TOP}_{m_1^{(t)}}\bigl(q_t(k)\bigr) \quad&\text{(exploitation)}\\
   S_2^{(t)} &= \text{TOP}_{m_2^{(t)}}\bigl(s_t(k)\bigr) \quad&\text{(associative recall)}\\
   S_3^{(t)} &\sim \text{Uniform}\!\Bigl(\mathcal K\setminus(S_1^{(t)}\cup S_2^{(t)}),\;m_3^{(t)}\Bigr) \quad&\text{(exploration)}
   \end{aligned}
   \tag{5}
   $$
where $q_t(k)$ is the current posterior and $s_t(k)$ denotes the associative similarity between the current stimulus $\mathbf x_t$ and partition $k$ (see details in App.B.3).
Because $m_1^{(t)}$ grows when the posterior is *uncertain* (high entropy) and shrinks when a single hypothesis dominates, the mechanism realises an adaptive **exploration–exploitation schedule** akin to the *Upper Confidence Bound* principle in bandit learning (Auer et al., 2002).  The accuracy-based rule further tightens search when recent performance stagnates, reflecting findings that humans expand the hypothesis pool after repeated errors (Matsuka & Corter, 2008).
Empirically, this dynamic allocation aims to explain the early broad search followed by late convergence and the abrupt “jumps” in accuracy coinciding with spikes in $m_1^{(t)}$ and dips in $m_3^{(t)}$.


### 3.3 Model fitting and comparison
**Model space.** The full model space comprises the rational backbone plus one, two, or all three adaptations, yielding eight candidates. Ablations let us attribute predictive gains to specific cognitive constraints.

**Optimisation protocol.**
**Stage 1 (backbone fit).** For each participant we perform a coarse grid-search over the $K$ partition indices and, *for every trial horizon $T$*, run a bounded L-BFGS on $\beta$ using Eq.(2) with $w_t=1$. This yields a sequence of posteriors $p\!\bigl(k,\hat\beta_{k}^{(T)}\mid\mathcal D_{1:T}\bigr)$ , where $T=1,\dots,T_{\max}$, so that we can track how the learner’s belief over hypotheses evolves in real time.  At each horizon we record the maximiser $(\hat k^{(T)},\hat\beta_{\hat k}^{(T)})$ as the model’s online estimate.
**Stage 2 (adaptation fit).** Additional parameters arise *only when the leaky-memory module is active*.  In that case we run a grid search over $\gamma\in[0.05,1]$ (20 points) and $w_0\in[0,1]$ (6 points), selecting the pair that maximises the weighted log-posterior in Eq.(3). The other two adaptations require no extra free parameters fitting.

**Predictive assessment.**
