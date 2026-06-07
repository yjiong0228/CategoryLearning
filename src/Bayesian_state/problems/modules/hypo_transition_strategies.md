# Hypothesis Transition Strategies

`DynamicHypothesisModule` updates the active hypothesis set with an ordered list
of strategies. Each strategy must set:

```yaml
- label: optional_name
  amount: entropy_7
  method: random_posterior
  pool: active
```

`amount` decides how many hypotheses to request, `pool` decides which candidates
are eligible, and `method` decides how candidates are selected. Strategies run
in YAML order, and hypotheses selected by earlier strategies are removed from
later pools. `max_active_hypotheses` can still be set as an external hard cap,
but strategy candidates should normally control active-set size through their
amount choices.

Pools:

| Pool | Candidate set | Typical use |
|---|---|---|
| `active` | Previous active hypotheses not yet selected this step | Retention / exploitation |
| `inactive` | Previous inactive hypotheses not yet selected this step | Explicit exploration |

## Amount Strategies

Let \(p_i\) be posterior probability, \(p_{max}=\max_i p_i\),
\(H(p)=-\sum_i p_i\log p_i\), \(N\) the hypothesis count, and \(M\) the suffix
number in names such as `entropy_7`.

| Name | Idea | Parameters | Formula |
|---|---|---|---|
| `fixed` | Use a fixed request count | `value` | \(n=value\) |
| `entropy_M` | Legacy confidence-like retention; lower entropy keeps more | none | \(n=\max(0,\lfloor M-\min(e^H,M+30)\rfloor+2)\) |
| `opp_entropy_M` | More uncertainty requests more hypotheses | none | \(n=\min(1+\lfloor(H/\log N)(M-1)\rfloor,M)\) |
| `entropy_norm_M` | Clear normalized-entropy retention | `min_count=0` | \(n=\max(min\_count,round(M(1-H/\log N)))\) |
| `opp_entropy_norm_M` | Clear normalized-entropy exploration | `min_count=0` | \(n=\max(min\_count,round(MH/\log N))\) |
| `random_M` | Random count controlled by confidence | none | \(P(n=0)=1-p_{max}; P(n=k)=p_{max}/M\) |
| `opp_random_M` | Reverse of `random_M` | none | \(n=M-R,\ R\sim random_M\) |
| `confidence_M` | Step function of max posterior | `threshold_min=0.2`, `scale=10`, `min_count=1` | if \(p_{max}\le t\), \(n=min\_count\); else \(n=\min(M,\lfloor scale(p_{max}-t)\rfloor+min\_count)\) |
| `opp_confidence_M` | Reverse of `confidence_M` | same as `confidence_M` | \(n=\max(0,M-n_{confidence})\) |
| `max_M` | Hard threshold based on posterior peak | `reference_mass=3.0` | if \(reference\_mass/p_{max}>M\), \(n=0\); else \(n=\lfloor reference\_mass/p_{max}\rfloor\) |
| `recent_accuracy_inverse_M` | High recent accuracy requests fewer hypotheses | `window=10`, `padding=chance`, `feedback_mode=graded`, `min_count=1`, `gamma=1.0` | \(acc=(\sum recent+(window-m)pad)/window\); \(n=min\_count+round((1-acc)^\gamma(M-min\_count))\) |
| `acc_M`, `accuracy_static_M` | Legacy Bayesian-style accuracy step; higher recent accuracy requests more | `window=16`, `padding=chance`, `feedback_mode=exact`, `threshold_min=0.2`, `scale=10` | if \(acc\le threshold\_min\), \(n=0\); else \(n=\min(M,\lfloor scale(acc-threshold\_min)\rfloor+1)\) |
| `opp_acc_M`, `opp_accuracy_static_M` | Complement of the legacy accuracy step | same as `acc_M` | \(n=M-n_{acc}\) |
| `accuracy_delta_M` | Improvement-driven amount | `window=8`, `padding=chance`, `feedback_mode=exact`, `threshold=0`, `scale=0.5` | \(\Delta acc=acc_{new}-acc_{old}\); \(n=round(M\cdot clip((\Delta acc-threshold)/scale,0,1))\) |
| `opp_accuracy_delta_M` | Decline-driven amount, useful for exploration after performance drops | same as `accuracy_delta_M` | \(n=round(M\cdot clip((-\Delta acc-threshold)/scale,0,1))\) |

History-based amount strategies store feedback inside the transition module.
The current trial's feedback is appended after transition, so transition at
trial `t` uses feedback through trial `t-1`. `padding: chance` uses
`1 / n_cats`; numeric padding such as `0.5` or `0.25` is also allowed.
`feedback_mode: graded` clips feedback into `[0,1]`; `exact` records only
`feedback == 1.0` as correct. Delta strategies need `2 * window` history and
left-pad missing early trials before splitting into old/new windows.

## Selection Methods

| Name | Idea | Parameters | Formula / behavior |
|---|---|---|---|
| `top_posterior` | Deterministic exploitation | `top_p=0.0`, `top_p_scope=global|pool` | With no `top_p`, select top \(n\) by \(p_i\). With `top_p`, select the sorted prefix whose cumulative mass exceeds `top_p`. `pool` scope normalizes within candidates. |
| `random_posterior` | Posterior-weighted stochastic selection | none | Sample without replacement with \(P(i)\propto p_i\) inside the pool. |
| `random` | Uniform random selection | none | Sample without replacement uniformly from the pool. |
| `epsilon_posterior` | Posterior sampling mixed with uniform noise | `epsilon=0.25` | \(w_i=(1-\epsilon)p_i/\sum p+\epsilon/|pool|\). |
| `temperature_posterior` | Flatten or sharpen posterior weights | `temperature=1.0`, `weight_floor=1e-12` | \(w_i\propto(p_i+weight\_floor)^{1/T}\). Larger \(T\) is more random. |
| `ksimilar_centers` | Prototype-center association around active hypotheses | `proto_hypo_amount=1`, `proto_hypo_method=top|random`, `cluster_hypo_method=top|random` | Samples reference categories from active prototype centers, scores candidates by center similarity, then selects by top score or score-weighted random. Requires prototype-backed partitions. |

## Example Configurations

Retention plus explicit exploration:

```yaml
strategies:
  - label: retain_active
    amount: entropy_norm_7
    method: epsilon_posterior
    pool: active
    min_count: 1
    epsilon: 0.25
  - label: explore_inactive
    amount: opp_entropy_norm_7
    method: random
    pool: inactive
```

Recent-accuracy driven retention:

```yaml
strategies:
  - label: history_retention
    amount: recent_accuracy_inverse_7
    method: epsilon_posterior
    pool: active
    window: 10
    padding: chance
    feedback_mode: graded
    min_count: 1
    gamma: 1.0
    epsilon: 0.35
  - label: steady_exploration
    amount: fixed
    value: 1
    method: random
    pool: inactive
```

