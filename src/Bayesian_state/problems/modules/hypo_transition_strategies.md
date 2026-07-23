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
| `latent_volatility_M` | Requests more hypotheses when the module's latent volatility state is high | `min_count=0`, `threshold=0`, `power=1.0` | \(v=clip((state-threshold)/(v_{max}-threshold),0,1)\); \(n=min\_count+round(v^{power}(M-min\_count))\) |
| `opp_latent_volatility_M` | Complement of latent-volatility amount | same as `latent_volatility_M` | \(n=M-n_{latent}\) |
| `post_error_explore_M` | Requests more exploration after the immediately previous trial was wrong | `padding=chance`, `feedback_mode=exact`, `min_count=0`, `gamma=1.0` | \(e=1-acc_{t-1}\); \(n=min\_count+round(e^\gamma(M-min\_count))\) |

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
| `low_posterior` | Deliberately samples low-posterior candidates for jumpy exploration | none | Select the \(n\) candidates with the lowest posterior in the pool. |
| `ksimilar_centers` | Prototype-center association around active hypotheses | `proto_hypo_amount=1`, `proto_hypo_method=top|random`, `cluster_hypo_method=top|random` | Samples reference categories from active prototype centers, scores candidates by center similarity, then selects by top score or score-weighted random. Requires prototype-backed partitions. |

## Post-to-Prior Strategies

After the active set is selected, `post_to_prior` controls how
`posterior_{t-1}` initializes `prior_t` over the new active set. If omitted,
the default is `similarity_novelty`, which preserves the previous behavior.
Existing `prior_reset_*` options are still applied after this strategy as an
optional second mixing step.

| Method | Idea | Parameters | Formula / behavior |
|---|---|---|---|
| `similarity_novelty` | Carry survivor posterior and initialize newcomers by similarity/novelty | `confidence_source=max_posterior|entropy|recent_accuracy|latent_volatility`, `min_newcomer_scale=0.05` | For newcomer \(i\), \(score_i=c\cdot p_{sim,i}+(1-c)\cdot p_{nov,i}\), then scale by \(\max(1-c,min\_newcomer\_scale)\). |
| `conservative_carryover` | Strong survivor carryover, small fixed newcomer budget | `newcomer_mass=0.05` | Survivors get \(1-newcomer\_mass\) proportional to previous posterior; newcomers share `newcomer_mass`. |
| `error_boost_newcomers` | Boost newcomer prior mass after recent errors or volatility | `window=8`, `padding=chance`, `feedback_mode=exact`, `base_newcomer_mass=0.05`, `max_newcomer_mass=0.65`, `volatility_gain=0.0` | \(mass=base+(max-base)\cdot clip(1-acc+volatility\_gain\cdot state,0,1)\). |
| `stochastic_reset` | Occasional random redistribution over active hypotheses | `reset_probability=0.25`, `newcomer_mass=0.50`, `concentration=1.0` | With reset probability, draw random active weights; if both survivors and newcomers exist, rescale newcomers to `newcomer_mass`. Otherwise fall back to `similarity_novelty`. |

`stochastic_reset` is kept for v10 reproducibility, but it is deprecated for
v11 optimization candidates because it can disrupt learned trajectories too
aggressively. The v11 candidate files exclude it.

## Profile Controller

Static `strategies` remain supported. For v11 experiments, a
`strategy_controller` can choose one profile per trial:

```yaml
strategy_controller:
  method: feedback_gated_softmax
  features:
    recent_accuracy_window: 8
    accuracy_delta_window: 8
    padding: chance
    feedback_mode: exact
  activation:
    temperature: 0.7
  profiles:
    - id: exploit
      activation:
        recent_accuracy: 2.0
        posterior_confidence: 1.0
      strategies: [...]
      post_to_prior:
        method: conservative_carryover
```

Only causal history is used for activation: previous feedback, recent
accuracy, accuracy delta, posterior entropy/confidence, latent volatility, and
trial progress. The current trial feedback is appended after transition, so it
cannot select the profile for the same trial. The first v11 implementation uses
hard gating: a single profile is sampled from a softmax and then that profile's
`strategies` and `post_to_prior` are executed.

### Persistent belief-instability state (V14)

V14 distinguishes the fast error features (`last_error`, `recent_error`) from
a state that survives across trials. With `latent_volatility_signal` set to
`confidence_weighted_error`, the update is

\[
z_{t+1}=clip(\rho z_t+\alpha(1-feedback_t)confidence_t,0,z_{max}).
\]

`latent_volatility_decay` is \(\rho\) and
`latent_volatility_error_gain` is \(\alpha\). The confidence term is the
pre-feedback posterior confidence recorded at the preceding transition, so a
confident error contributes more evidence for a change point than an error
made while the model was already uncertain.

Controllers can use either the normalized raw state (`latent_volatility`) or
the thresholded smooth feature (`latent_volatility_pressure`). The latter is

\[
pressure_t=\sigma(s(z_t/z_{max}-threshold/z_{max})),
\]

where `latent_volatility_pressure_slope` is \(s\). Setting all volatility
gains to zero preserves the V13 state-off behavior. V14 candidates keep the
four inner policy profiles and use pressure primarily to increase the
aggressive profile's probability during sustained instability.

## Choice Readout

`engine.choice_readout.kwargs` controls how the model reads out category
probabilities from the current hypothesis distribution. It affects prediction
metrics and simulated choice probabilities only; it does not change posterior
updates.

| Method | Idea | Main parameters |
|---|---|---|
| `expectation` | Current default: average category probabilities under the hypothesis distribution | none |
| `sharpened_expectation` | Raise hypothesis weights to a power before averaging | `power`, `weight_floor` |
| `map_hypothesis` | Read out the single highest-weight active hypothesis | none |
| `sample_hypothesis` | Sample one active hypothesis per trial from the distribution | `weight_floor` |
| `sticky_sample` | Keep a sampled hypothesis until confidence/error/inactivity triggers switching | `switch_probability`, `post_error_switch_delta`, `low_confidence_switch_gain` |
| `stubborn_sticky` | Like sticky readout, but errors can reduce switching and create persistent wrong choices | `switch_probability`, `post_error_switch_delta` |

V11 profile candidates are stored as multi-path model kwargs, so a YAML file can
refer to a JSON candidate list with:

```yaml
hyperparam_space:
  __profile_candidate__:
    values_from_json:
      path: ../../src/Bayesian_state/problems/modules/hypo_transition_strategies/hypo_transition_profile_v11_candidates.json
      key: cond1_v11
      value_key: model_kwargs
```

V13 profile candidates separate transition/p2p profiles from readout. The JSON
candidate only provides `hypo_transitions_kwargs`; readout is a separate
coordinate, so every profile is evaluated with both expectation and MAP readout:

```yaml
hyperparam_space:
  engine.modules.hypo_transitions_mod.kwargs:
    values_from_json:
      path: ../../src/Bayesian_state/problems/modules/hypo_transition_strategies/hypo_transition_profile_v13_candidates.json
      key: cond1_v13
      value_key: hypo_transitions_kwargs

  engine.choice_readout.kwargs:
    values:
      - method: expectation
      - method: map_hypothesis
```

V14 preserves the V13 candidate file and adds a separate six-family candidate
set. Each family has a state-off ablation plus confidence-weighted persistent
state gains 0.20, 0.35, and 0.50. Readout remains a separate four-value
coordinate: expectation, sharpened expectation at powers 2 and 4, and MAP.

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

Jumpy post-error exploration with explicit post-to-prior behavior:

```yaml
post_to_prior:
  method: error_boost_newcomers
  window: 6
  padding: chance
  feedback_mode: exact
  base_newcomer_mass: 0.08
  max_newcomer_mass: 0.60
strategies:
  - label: retain_core
    amount: fixed
    value: 1
    method: top_posterior
    pool: active
  - label: post_error_refresh
    amount: post_error_explore_5
    method: low_posterior
    pool: inactive
    padding: chance
    feedback_mode: exact
    min_count: 1
```
