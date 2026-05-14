import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import logging

from src.Bayesian_state.problems.partitions import Partition
from src.Bayesian_state.utils.oral_process import Oral_center_analysis, Oral_region_analysis
logger = logging.getLogger(__name__)


class ModelEval:
    @staticmethod
    def _filter_results(results, subjects):
        if subjects is not None:
            return {iSub: results[iSub] for iSub in subjects if iSub in results}
        return results

    def _plot_by_condition(self, results, subjects, save_path, title, plot_body, **kwargs):
        # Filter and group
        results = self._filter_results(results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info['condition']].append((iSub, info))

        # Layout
        n_rows = len(grouped)
        n_cols = kwargs.get('n_cols', max(len(lst) for lst in grouped.values()))
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle(title, fontsize=kwargs.get('fontsize', 16), y=kwargs.get('y', 0.99))

        # Subplots
        for row, (condition, subs) in enumerate(sorted(grouped.items())):
            for col, (iSub, info) in enumerate(subs):
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                plot_body(ax, condition, iSub, info)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            logger.info(f"{title} saved to {save_path}")

    def plot_posterior_probabilities(self, results, subjects=None, save_path=None, limit=True, **kwargs):
        def _get_post_max(hypo_details, k):
            """Support both int keys and string keys after JSON round-trip."""
            if not isinstance(hypo_details, dict):
                return None

            entry = hypo_details.get(k)
            if entry is None:
                entry = hypo_details.get(str(k))

            if not isinstance(entry, dict):
                return None

            value = entry.get("post_max")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def body(ax, condition, iSub, info):
            step_results = info.get("step_results") or info.get("best_step_results") or []

            ax.set(
                title=f"Subject {iSub} (Condition {condition})",
                xlabel="Trial",
                ylabel="Posterior Probability",
            )

            if not step_results:
                ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
                return

            if limit:
                max_k = 19 if condition == 1 else 116
            else:
                all_keys = []
                for sr in step_results:
                    hypo_details = sr.get("hypo_details", {})
                    if isinstance(hypo_details, dict):
                        for key in hypo_details.keys():
                            try:
                                all_keys.append(int(key))
                            except (TypeError, ValueError):
                                pass
                max_k = max(all_keys) + 1 if all_keys else 0

            data = []
            for step, res in enumerate(step_results):
                hypo_details = res.get("hypo_details", {})
                for k in range(max_k):
                    post_max = _get_post_max(hypo_details, k)
                    if post_max is not None:
                        data.append({
                            "Step": step + 1,
                            "k": k,
                            "Posterior": post_max,
                        })

            df = pd.DataFrame(data)
            if df.empty or "Step" not in df.columns:
                ax.text(0.5, 0.5, "No posterior data", ha="center", va="center", transform=ax.transAxes)
                return

            sns.scatterplot(
                data=df,
                x="Step",
                y="Posterior",
                hue="k",
                palette="tab10",
                alpha=0.5,
                legend=False,
                ax=ax,
            )

            hk = 0 if condition == 1 else 42
            hk_df = df[df["k"] == hk]
            if not hk_df.empty:
                sns.scatterplot(
                    data=hk_df,
                    x="Step",
                    y="Posterior",
                    color="red",
                    s=50,
                    ax=ax,
                )

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            "Posterior Probabilities for k by Subject",
            body,
            **kwargs,
        )


    @staticmethod
    def _normalize_distribution(values):
        """Return a valid probability vector or an all-NaN vector."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0 or np.isnan(arr).all():
            return np.full(arr.shape, np.nan, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        total = float(arr.sum())
        if total <= 0:
            return np.full(arr.shape, np.nan, dtype=float)
        return arr / total

    @staticmethod
    def _adaptive_softmax_from_distances(distances):
        """Convert distances to a distribution without a fixed top-k cutoff.

        The scale is chosen from the current trial's distance spread, so clear
        oral reports become sharp and ambiguous reports remain broader.
        """
        d = np.asarray(distances, dtype=float).reshape(-1)
        if d.size == 0 or np.isnan(d).all():
            return np.full(d.shape, np.nan, dtype=float)
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            return np.full(d.shape, np.nan, dtype=float)
        d = np.where(np.isfinite(d), d, np.nanmax(finite))
        d_min = float(np.min(d))
        spread = float(np.median(d) - d_min)
        if spread <= 1e-12:
            spread = float(np.std(d))
        if spread <= 1e-12:
            exact = np.isclose(d, d_min)
            return exact.astype(float) / float(np.sum(exact))
        score = np.exp(-(d - d_min) / spread)
        return ModelEval._normalize_distribution(score)

    @staticmethod
    def _js_similarity(p, q):
        """Return 1 - normalized Jensen-Shannon divergence."""
        p = ModelEval._normalize_distribution(p)
        q = ModelEval._normalize_distribution(q)
        if np.isnan(p).any() or np.isnan(q).any() or p.shape != q.shape:
            return np.nan
        m = 0.5 * (p + q)

        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

        js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        return float(1.0 - min(js / np.log(2.0), 1.0))

    @staticmethod
    def _effective_sample_size(prob):
        """Return distribution effective sample size, or NaN if invalid."""
        p = ModelEval._normalize_distribution(prob)
        if np.isnan(p).any():
            return np.nan
        return float(1.0 / np.sum(p ** 2))

    @staticmethod
    def _extract_prior_log(info):
        """Use prior_t as the model state aligned with oral_t."""
        prior_log = info.get('prior_log') or []
        if prior_log:
            return [np.asarray(x, dtype=float) for x in prior_log]

        priors = []
        for step in info.get('best_step_results', []) or []:
            prior = step.get('prior')
            if prior is None:
                return []
            priors.append(np.asarray(prior, dtype=float))
        return priors

    @staticmethod
    def _center_oral_distribution(center, choice, partition):
        """Map one oral center report to a full hypothesis distribution."""
        center = np.asarray(center, dtype=float).reshape(-1)
        if center.size == 0 or np.isnan(center).any():
            return np.full(partition.length, np.nan, dtype=float)
        cat_idx = int(choice) - 1
        distances = np.linalg.norm(partition.prototypes[:, 0, cat_idx, :] - center, axis=1)
        return ModelEval._adaptive_softmax_from_distances(distances)

    @staticmethod
    def _region_oral_distribution(region, choice, partition, n_samples=1000, random_state=42):
        """Map one oral region report to a full hypothesis distribution."""
        cat_idx = int(choice) - 1
        scores = []
        for hypo_idx in range(len(partition.regions)):
            score = Oral_region_analysis._estimate_overlap_score(
                region,
                partition.regions[hypo_idx][cat_idx],
                metric='iou',
                n_samples=int(n_samples),
                bounds=(0.0, 1.0),
                random_state=None if random_state is None else int(random_state) + hypo_idx,
                dist_tol=1e-9,
            )
            scores.append(0.0 if np.isnan(score) else float(score))
        return ModelEval._normalize_distribution(scores)

    def compute_oral_model_alignment(self, model_results, oral_df, oral_mode='center', subjects=None, region_n_samples=1000):
        """Compute prior_t vs oral_t alignment metrics per subject.

        Metrics per trial:
        - target_model_prior: prior_t mass on target hypothesis.
        - target_oral_score: oral distribution mass on target hypothesis.
        - model_mass_on_oral: dot(prior_t, oral_dist_t).
        - model_oral_similarity: 1 - normalized JS divergence.
        - oral_ess: effective number of hypotheses supported by oral_dist_t.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        out = {}

        for iSub, info in model_res.items():
            subj_df = oral_df[oral_df['iSub'] == iSub].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get('condition', subj_df['condition'].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            target_hypo = 0 if condition == 1 else 42
            partition = Partition(n_dims=4, n_cats=n_cats)
            prior_log = self._extract_prior_log(info)
            n_trials = min(len(subj_df), len(prior_log))

            target_model_prior = []
            target_oral_score = []
            model_mass_on_oral = []
            model_oral_similarity = []
            oral_ess = []
            valid_oral = []

            for trial_idx in range(n_trials):
                prior = self._normalize_distribution(prior_log[trial_idx])
                choice = int(subj_df.loc[trial_idx, 'choice'])

                if oral_mode == 'center':
                    center = Oral_center_analysis._parse_center(subj_df.loc[trial_idx, 'oral_center'])
                    oral_dist = self._center_oral_distribution(center, choice, partition)
                elif oral_mode == 'region':
                    region = (subj_df.loc[trial_idx, 'oral_A'], subj_df.loc[trial_idx, 'oral_b'])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=42 + trial_idx * 100000,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                valid = not (np.isnan(prior).any() or np.isnan(oral_dist).any())
                valid_oral.append(bool(valid))
                if not valid:
                    target_model_prior.append(np.nan)
                    target_oral_score.append(np.nan)
                    model_mass_on_oral.append(np.nan)
                    model_oral_similarity.append(np.nan)
                    oral_ess.append(np.nan)
                    continue

                target_model_prior.append(float(prior[target_hypo]) if target_hypo < len(prior) else np.nan)
                target_oral_score.append(float(oral_dist[target_hypo]) if target_hypo < len(oral_dist) else np.nan)
                model_mass_on_oral.append(float(np.dot(prior, oral_dist)))
                model_oral_similarity.append(self._js_similarity(prior, oral_dist))
                oral_ess.append(self._effective_sample_size(oral_dist))

            out[iSub] = {
                'iSub': int(iSub),
                'condition': condition,
                'target_hypo': target_hypo,
                'alignment_mode': 'oral_t_vs_prior_t',
                'oral_mode': oral_mode,
                'target_model_prior': target_model_prior,
                'target_oral_score': target_oral_score,
                'model_mass_on_oral': model_mass_on_oral,
                'model_oral_similarity': model_oral_similarity,
                'oral_ess': oral_ess,
                'valid_oral': valid_oral,
            }
        return out

    def plot_oral_model_alignment(self, alignment_results, subjects=None, save_path=None, window_size=16, **kwargs):
        """Plot rolling model-oral alignment metrics by subject."""
        results = self._filter_results(alignment_results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info['condition']].append((iSub, info))

        if not grouped:
            raise RuntimeError('No oral-model alignment results to plot.')

        n_rows = len(grouped)
        n_cols = kwargs.get('n_cols', max(len(lst) for lst in grouped.values()))
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle('Oral-Model Alignment (oral_t vs prior_t)', fontsize=kwargs.get('fontsize', 16), y=kwargs.get('y', 0.99))

        def rolling(values):
            return pd.Series(values, dtype=float).rolling(window=window_size, min_periods=window_size).mean().to_numpy()

        for row, (condition, subs) in enumerate(sorted(grouped.items())):
            for col, (iSub, info) in enumerate(subs):
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                n = len(info.get('model_oral_similarity', []))
                x = np.arange(1, n + 1)
                ax.plot(x, rolling(info.get('model_oral_similarity', [])), lw=2, label='1 - JS(prior, oral)')
                ax.plot(x, rolling(info.get('model_mass_on_oral', [])), lw=2, label='Prior mass on oral')
                ax.plot(x, rolling(info.get('target_model_prior', [])), lw=1.5, alpha=0.8, label='Target prior')
                ax.plot(x, rolling(info.get('target_oral_score', [])), lw=1.5, alpha=0.8, label='Target oral score')
                ax.set_ylim(0, 1)
                ax.set(title=f'Subject {iSub} (Cond {condition})', xlabel='Trial', ylabel='Alignment')
                ax.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Oral-model alignment saved to {save_path}")

    @staticmethod
    def _choice_conditioned_prior(partition, prior, stimulus, choice, beta=10.0):
        """Condition prior_t on the category choice made before oral report."""
        prior = ModelEval._normalize_distribution(prior)
        if np.isnan(prior).any():
            return np.full_like(prior, np.nan, dtype=float)

        choice_idx = int(choice) - 1
        likelihood = np.zeros_like(prior, dtype=float)
        data = ([np.asarray(stimulus, dtype=float)], [int(choice)], [1.0], [choice_idx + 1])
        for hypo_idx, weight in enumerate(prior):
            if weight <= 0:
                continue
            prob = partition.get_category_probabilities(
                hypo=hypo_idx,
                data=data,
                beta=float(beta),
                distance_mode=partition.DISTANCE_MODE_PROTOTYPE,
            )[:, 0]
            if 0 <= choice_idx < len(prob):
                likelihood[hypo_idx] = float(prob[choice_idx])

        conditioned = prior * likelihood
        return ModelEval._normalize_distribution(conditioned)

    @staticmethod
    def _expected_center_similarity(partition, model_dist, oral_center, choice):
        """Compare oral center with the model's choice-conditioned expected center."""
        model_dist = ModelEval._normalize_distribution(model_dist)
        center = np.asarray(oral_center, dtype=float).reshape(-1)
        if np.isnan(model_dist).any() or center.size == 0 or np.isnan(center).any():
            return np.nan
        cat_idx = int(choice) - 1
        centers = partition.prototypes[:, 0, cat_idx, :]
        expected_center = np.sum(model_dist[:, None] * centers, axis=0)
        dist = float(np.linalg.norm(center - expected_center))
        max_dist = float(np.sqrt(partition.n_dims))
        if max_dist <= 0:
            return np.nan
        return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

    def compute_choice_conditioned_oral_alignment(self, model_results, oral_df, oral_mode='center', subjects=None, region_n_samples=1000, beta=10.0):
        """Compute oral_t alignment with prior_t conditioned on current choice.

        This matches the task timing: stimulus -> choice -> oral report -> feedback.
        The model state is therefore prior_t after observing stimulus and choice,
        but before feedback-driven posterior update.
        """
        model_res = self._filter_results(model_results, subjects)
        oral_df = oral_df.copy()
        out = {}

        for iSub, info in model_res.items():
            subj_df = oral_df[oral_df['iSub'] == iSub].reset_index(drop=True)
            if subj_df.empty:
                continue

            condition = int(info.get('condition', subj_df['condition'].iloc[0]))
            n_cats = 2 if condition == 1 else 4
            target_hypo = 0 if condition == 1 else 42
            partition = Partition(n_dims=4, n_cats=n_cats)
            prior_log = self._extract_prior_log(info)
            steps = info.get('best_step_results') or info.get('step_results') or []
            n_trials = min(len(subj_df), len(prior_log), len(steps) if steps else len(subj_df))

            choice_conditioned_similarity = []
            choice_conditioned_mass_on_oral = []
            choice_conditioned_target_prior = []
            target_oral_score = []
            expected_center_similarity = []
            valid_oral = []

            for trial_idx in range(n_trials):
                choice = int(subj_df.loc[trial_idx, 'choice'])
                step = steps[trial_idx] if trial_idx < len(steps) else {}
                stimulus = step.get('perceived_stimulus')
                if stimulus is None:
                    stimulus = subj_df.loc[trial_idx, ['feature1', 'feature2', 'feature3', 'feature4']].to_numpy(dtype=float)

                conditioned = self._choice_conditioned_prior(
                    partition=partition,
                    prior=prior_log[trial_idx],
                    stimulus=stimulus,
                    choice=choice,
                    beta=beta,
                )

                oral_center = None
                if oral_mode == 'center':
                    oral_center = Oral_center_analysis._parse_center(subj_df.loc[trial_idx, 'oral_center'])
                    oral_dist = self._center_oral_distribution(oral_center, choice, partition)
                elif oral_mode == 'region':
                    region = (subj_df.loc[trial_idx, 'oral_A'], subj_df.loc[trial_idx, 'oral_b'])
                    oral_dist = self._region_oral_distribution(
                        region,
                        choice,
                        partition,
                        n_samples=region_n_samples,
                        random_state=4242 + trial_idx * 100000,
                    )
                else:
                    raise ValueError(f"Unsupported oral_mode: {oral_mode}")

                valid = not (np.isnan(conditioned).any() or np.isnan(oral_dist).any())
                valid_oral.append(bool(valid))
                if not valid:
                    choice_conditioned_similarity.append(np.nan)
                    choice_conditioned_mass_on_oral.append(np.nan)
                    choice_conditioned_target_prior.append(np.nan)
                    target_oral_score.append(np.nan)
                    expected_center_similarity.append(np.nan)
                    continue

                choice_conditioned_similarity.append(self._js_similarity(conditioned, oral_dist))
                choice_conditioned_mass_on_oral.append(float(np.dot(conditioned, oral_dist)))
                choice_conditioned_target_prior.append(
                    float(conditioned[target_hypo]) if target_hypo < len(conditioned) else np.nan
                )
                target_oral_score.append(
                    float(oral_dist[target_hypo]) if target_hypo < len(oral_dist) else np.nan
                )
                if oral_mode == 'center':
                    expected_center_similarity.append(
                        self._expected_center_similarity(partition, conditioned, oral_center, choice)
                    )
                else:
                    expected_center_similarity.append(np.nan)

            out[iSub] = {
                'iSub': int(iSub),
                'condition': condition,
                'target_hypo': target_hypo,
                'alignment_mode': 'oral_t_vs_choice_conditioned_prior_t',
                'oral_mode': oral_mode,
                'choice_conditioned_similarity': choice_conditioned_similarity,
                'choice_conditioned_mass_on_oral': choice_conditioned_mass_on_oral,
                'choice_conditioned_target_prior': choice_conditioned_target_prior,
                'target_oral_score': target_oral_score,
                'expected_center_similarity': expected_center_similarity,
                'valid_oral': valid_oral,
            }
        return out

    def plot_choice_conditioned_oral_alignment(self, alignment_results, subjects=None, save_path=None, window_size=16, **kwargs):
        """Plot oral alignment with choice-conditioned prior_t."""
        results = self._filter_results(alignment_results, subjects)
        grouped = defaultdict(list)
        for iSub, info in results.items():
            grouped[info['condition']].append((iSub, info))

        if not grouped:
            raise RuntimeError('No choice-conditioned oral alignment results to plot.')

        n_rows = len(grouped)
        n_cols = kwargs.get('n_cols', max(len(lst) for lst in grouped.values()))
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle('Oral Alignment with Choice-Conditioned Prior', fontsize=kwargs.get('fontsize', 16), y=kwargs.get('y', 0.99))

        def rolling(values):
            return pd.Series(values, dtype=float).rolling(window=window_size, min_periods=window_size).mean().to_numpy()

        for row, (condition, subs) in enumerate(sorted(grouped.items())):
            for col, (iSub, info) in enumerate(subs):
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                n = len(info.get('choice_conditioned_similarity', []))
                x = np.arange(1, n + 1)
                ax.plot(x, rolling(info.get('choice_conditioned_similarity', [])), lw=2, label='1 - JS(choice-conditioned, oral)')
                ax.plot(x, rolling(info.get('choice_conditioned_mass_on_oral', [])), lw=2, label='Choice-cond. mass on oral')
                center_vals = info.get('expected_center_similarity', [])
                if center_vals and not np.all(np.isnan(np.asarray(center_vals, dtype=float))):
                    ax.plot(x, rolling(center_vals), lw=2, label='Expected center similarity')
                ax.plot(x, rolling(info.get('choice_conditioned_target_prior', [])), lw=1.5, alpha=0.8, label='Target choice-cond. prior')
                ax.plot(x, rolling(info.get('target_oral_score', [])), lw=1.5, alpha=0.8, label='Target oral score')
                ax.set_ylim(0, 1)
                ax.set(title=f'Subject {iSub} (Cond {condition})', xlabel='Trial', ylabel='Alignment')
                ax.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Choice-conditioned oral alignment saved to {save_path}")

    def plot_k_oral_comparison(self, model_results, oral_results, subjects=None, save_path=None, window_size=16, **kwargs):
        """
        Compare smoothed posterior of true k and smoothed oral hits, filtering out empty trials.
        """
        def _get_post_max(hypo_details, k_special):
            """Support both int and str hypothesis keys after JSON round-trip."""
            if not isinstance(hypo_details, dict):
                return 0.0
            entry = hypo_details.get(k_special)
            if entry is None:
                entry = hypo_details.get(str(k_special))
            if not isinstance(entry, dict):
                return 0.0
            return entry.get('post_max', 0.0)

        def extract_model_ma(step_results, k_special, win):
            posts = []
            for sr in step_results:
                p = _get_post_max(sr.get('hypo_details', {}), k_special)
                try:
                    p = float(p)
                except (TypeError, ValueError):
                    p = 0.0
                posts.append(p)
            return pd.Series(posts, dtype=float).rolling(window=win, min_periods=win).mean().to_numpy()

        def extract_oral_ma(hits, win):
            rolling = []
            n = len(hits)
            for i in range(n):
                if i + 1 < win:
                    rolling.append(np.nan)
                else:
                    window = np.asarray(hits[i-win+1 : i+1], dtype=float)
                    if np.all(np.isnan(window)):
                        rolling.append(np.nan)
                    else:
                        rolling.append(float(np.nanmean(window)))
            return np.array(rolling)

        # Filter both dicts
        model_res = self._filter_results(model_results, subjects)
        oral_res = self._filter_results(oral_results, subjects)

        # Group by condition
        grouped = defaultdict(list)
        for iSub, info in model_res.items():
            grouped[info['condition']].append(iSub)

        n_rows = len(grouped)
        n_cols = kwargs.get('n_cols', max(len(lst) for lst in grouped.values()))
        fig = plt.figure(figsize=(n_cols * 8, n_rows * 5))
        fig.suptitle('Model k vs Oral k (Filtered & Smoothed)', fontsize=kwargs.get('fontsize', 16), y=kwargs.get('y', 0.99))

        for row, (condition, subs) in enumerate(sorted(grouped.items())):
            for col, iSub in enumerate(subs):
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)

                # true model posterior
                info = model_res[iSub]
                sr = info.get('step_results', info.get('best_step_results', []))
                ks = 0 if condition == 1 else 42
                # prepare hits for this subject
                oral_hits = oral_res[iSub]['hits']

                rolling_model = extract_model_ma(sr, ks, window_size)
                valid_idx = np.arange(len(rolling_model))
                x_model = np.array(valid_idx)[window_size-1:] + 1
                ax.plot(x_model, rolling_model[window_size-1:], lw=2, label='Model k', **kwargs)

                # oral smoothed hits
                rolling_oral = extract_oral_ma(oral_hits, window_size)
                x_oral = np.array(valid_idx) + 1
                ax.plot(x_oral, rolling_oral, lw=2, label='Oral k', **kwargs)

                ax.set_ylim(0, 1)
                ax.set(title=f'Subject {iSub} (Cond {condition})', xlabel='Trial', ylabel='Probability')
                ax.legend()

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Filtered comparison saved to {save_path}")

            
    def plot_accuracy_comparison(self, results, subjects=None, save_path=None, window_size=None, **kwargs):
        def body(ax, condition, iSub, info):
            t = info['sliding_true_acc']
            p = info['sliding_pred_acc']
            std = info['sliding_pred_acc_std']
            win = info.get('window_size') or window_size
            try:
                win = int(win)
            except (TypeError, ValueError):
                win = 1
            trial = np.arange(win + 1, win + 1 + len(p))
            df = pd.DataFrame({'Trial': trial, 'Pred': p, 'True': t,
                               'Low': np.array(p) - std, 'High': np.array(p) + std})
            sns.lineplot(data=df, x='Trial', y='Pred', label='Predicted', ax=ax)
            sns.lineplot(data=df, x='Trial', y='True', label='True', ax=ax)
            ax.fill_between(df['Trial'], df['Low'], df['High'], alpha=0.2)
            n_trials = info.get('n_trials')
            if n_trials:
                ax.set_xlim(1, n_trials)
            ax.set_ylim(0, 1)
            ax.set(title=f'Subject {iSub} (Condition {condition})', xlabel='Trial', ylabel='Accuracy')
            ax.legend()

        self._plot_by_condition(results, subjects, save_path,
                                'Predicted vs True Accuracy by Subject', body, **kwargs)

    def plot_error_grids(self, results, subjects=None, fname=None, save_path=None, **kwargs):
        labels = fname if isinstance(fname, (list, tuple)) and len(fname) >= 2 else ("gamma", "w0")

        def body(ax, condition, iSub, info):
            data = []
            for (g, w0), errs in info['grid_errors'].items():
                # if errs is already a float, this does nothing
                err_val = float(np.mean(errs))  # or errs if it’s already scalar
                data.append({'gamma': g, 'w0': w0, 'Error': err_val})
            df = pd.DataFrame(data)
            em = df.pivot(index='gamma', columns='w0', values='Error')
            sns.heatmap(em, cbar_kws={'label': 'Error'}, ax=ax, cmap='viridis_r')
            ax.set(title=f'Subject {iSub} (Condition {condition})',
                xlabel=labels[1], ylabel=labels[0])
            ax.set_xticks(np.arange(len(em.columns)) + 0.5)
            ax.set_xticklabels([f"{v:.4f}" for v in em.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(em.index)) + 0.5)
            ax.set_yticklabels([f"{v:.2f}" for v in em.index], rotation=0)
        
        self._plot_by_condition(results, subjects, save_path,
                                'Grid Search Error by Subject', body, **kwargs)

    def plot_cluster_amount(self, results, window_size=16, subjects=None, save_path=None, **kwargs):
        def _first_numeric(value, default=0.0):
            if isinstance(value, (list, tuple)):
                if not value:
                    return float(default)
                return float(value[0])
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def body(ax, condition, iSub, info):
            steps = info.get('best_step_results', [])
            vals = []
            r    = []
            for s in steps:
                bsa = s.get('best_step_amount', {})

                # sum every posterior‐named list’s first entry
                posterior_vals = [
                    _first_numeric(v)
                    for k, v in bsa.items()
                    if 'posterior' in k
                ]
                vals.append(sum(posterior_vals))

                # always append something for 'random' (0 if missing)
                r.append(_first_numeric(bsa.get('random', 0.0)))

            # now both lists have the same length
            re = pd.Series(vals).rolling(window=window_size, min_periods=window_size).mean()
            ex = pd.Series(r).rolling(window=window_size, min_periods=window_size).mean()

            x = np.arange(1, len(vals) + 1)
            ax.plot(x, re, label='Exploitation', lw=2)
            ax.plot(x, ex, label='Exploration', lw=2)
            ax.set(
                title=f'Subject {iSub} (Condition {condition})',
                xlabel='Trial',
                ylabel='Amount'
            )
            ax.legend()

        self._plot_by_condition(
            results,
            subjects,
            save_path,
            'Strategy Amount by Subject',
            body,
            **kwargs
        )
