"""Summarize all frozen-parameter diagnostic paths and paired simulation error.

Use --source RUN --output NEW. Accuracy uses sampled choices; Energy uses
32-trial accuracy vectors after intervention. All errors are Monte Carlo errors
conditional on the recorded task schedule and fixed parameter point. Episode
endpoints include right censoring explicitly; absent targets carry zero choice
contribution. Human choices are opened here only for retrospective comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import load, digest, block_paths, distribution_scores, jackknife_se
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis.model_0923_h_validation import write


def episodes(a: dict, subject: int, model: str, arm: str) -> list[dict]:
    """Every uninterrupted target membership spell, including censored endings."""
    target = int(a['oracle']); rows = []
    present = np.any(a['rule'] == target, axis=2)
    for path, mask in enumerate(present):
        boundaries = np.diff(np.r_[False, mask, False].astype(int))
        for start, stop in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
            reason = 'right_censored' if stop == len(mask) else (
                'autonomous_exit' if target in a['exits'][path, stop] else
                'capacity' if a['capacity_dropped'][path, stop] == target else 'other')
            rows.append(dict(subject=subject, model=model, arm=arm, seed=int(a['seeds'][path]),
                start_trial=int(start+1), stop_trial_exclusive=int(stop+1), duration=int(stop-start),
                right_censored=stop == len(mask), removal=reason))
    return rows


def target_values(a: dict, key: str) -> np.ndarray:
    return np.where(a['rule'] == int(a['oracle']), np.nan_to_num(a[key]), 0.).sum(axis=2)


def run(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError('Use a new diagnostic summary directory')
    complete = load(source / 'completion.json'); protocol = load(source / 'protocol.json')
    if complete['status'] != 'complete' or complete['smoke']: raise ValueError('Need completed scientific paths')
    jobs = load(source / 'trace_plan.json')['jobs']; groups = {}
    artifacts = {}
    for job in jobs: groups.setdefault((job['subject'], job['model']), []).append(job)
    output.mkdir(parents=True)
    summary, pairs, paths, curves, spells = [], [], [], [], []
    for (subject, model), chunks in groups.items():
        rows = {arm: [] for arm in protocol['settings']['arms']}
        for job in chunks:
            folder = Path(job['target']); record = load(folder / 'completion.json')
            assert record['baseline_exact'] and record['replay_exact'] and record['preintervention_exact']
            for name, sha in record['artifact_sha256'].items():
                if digest(folder / name) != sha: raise ValueError('Diagnostic artifact changed')
                artifacts[str(folder / name)] = sha
            for arm in rows:
                with np.load(folder / (arm + '.npz')) as z: rows[arm].append({k: z[k] for k in z.files})
        prepared = Path(chunks[0]['folder']); meta = load(prepared / 'prepared.json')
        observed_parts = []
        for name in ('training.npz', 'sealed_test.npz'):
            if digest(prepared / name) != meta['array_sha256'][name]: raise ValueError('Observed arrays changed')
            with np.load(prepared / name) as z: observed_parts.append((z['choices'] == z['categories']).astype(float))
        observed = np.concatenate(observed_parts)
        references = {}
        for arm, pieces in rows.items():
            for piece in pieces[1:]:
                for key in ('categories', 'onset', 'oracle'): np.testing.assert_array_equal(piece[key], pieces[0][key])
            a = {k: pieces[0][k] if k in ('categories', 'onset', 'oracle') else
                 np.concatenate([p[k] for p in pieces], axis=0) for k in pieces[0]}
            n = len(a['seeds']); cut = int(a['onset']); trials = len(observed); late = max(cut, trials-128)
            assert n == protocol['settings']['paths'] and len(np.unique(a['seeds'])) == n
            correct = (a['choice'] == a['categories'][None, :]).astype(float)
            present = np.any(a['rule'] == int(a['oracle']), axis=2)
            contribution = target_values(a, 'contribution'); prior = target_values(a, 'prior')
            rule_probability = target_values(a, 'rule_correct')
            beta = target_values(a, 'beta_pre')
            samples = block_paths(correct, cut, trials); truth = block_paths(observed[None, :], cut, trials)[0]
            scores = distribution_scores(samples, truth)
            deleted = path_jackknife(samples, truth) if n > 1 else None
            late_accuracy = correct[:, late:].mean(axis=1)
            references[arm] = dict(seeds=a['seeds'], late=late_accuracy, scores=scores, deleted=deleted)
            summary.append(dict(subject=subject, model=model, arm=arm, paths=n, oracle=int(a['oracle']),
                onset=cut+1, observed_late=float(observed[late:].mean()), late_accuracy=float(late_accuracy.mean()),
                late_accuracy_mc_se=float(late_accuracy.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan,
                late_probability=float(np.take_along_axis(a['prediction'][:, late:],
                    a['categories'][None, late:, None]-1, axis=2).mean()),
                **scores, energy_mc_se=jackknife_se(deleted['energy']) if deleted else np.nan,
                target_ever=float(present.any(axis=1).mean()), target_present_before_onset=float(present[:, cut-1].mean()),
                target_late_presence=float(present[:, late:].mean()), target_late_contribution=float(contribution[:, late:].mean()),
                target_probability_when_present=float(rule_probability[:, late:].sum()/present[:, late:].sum()) if present[:, late:].any() else np.nan,
                onset_prior_median=float(np.median(prior[:, cut])), onset_prior_q10=float(np.quantile(prior[:, cut], .1)),
                onset_contribution_mean=float(contribution[:, cut].mean()),
                target_admitted_by_onset=float(present[:, :cut].any(axis=1).mean()),
                rejected_per_path=float(a['rejected'][:, cut:].sum(axis=1).mean()),
                target_proposed_per_path=float((a['proposed'][:, cut:] == int(a['oracle'])).sum(axis=1).mean()),
                target_rejected_per_path=float(((a['proposed'][:, cut:] == int(a['oracle'])) & a['rejected'][:, cut:].astype(bool)).sum(axis=1).mean()),
                already_available_per_path=float(a['already_available'][:, cut:].sum(axis=1).mean())))
            for p, seed in enumerate(a['seeds']):
                hits = np.flatnonzero(present[p]); after = present[p, cut:]
                first_gap = np.flatnonzero(~after)
                duration = int(first_gap[0]) if len(first_gap) else len(after)
                paths.append(dict(subject=subject, model=model, arm=arm, seed=int(seed), late_accuracy=float(late_accuracy[p]),
                    ever=bool(len(hits)), first_entry=int(hits[0]+1) if len(hits) else np.nan,
                    target_late_presence=float(present[p, late:].mean()), target_late_contribution=float(contribution[p, late:].mean()),
                    onset_prior=float(prior[p, cut]), onset_contribution=float(contribution[p, cut]),
                    onset_spell_observed_length=duration, onset_spell_censored=not len(first_gap)))
            for b, first in enumerate(range(cut, trials, 32)):
                last = min(first+32, trials); values = samples[:, b]
                curves.append(dict(subject=subject, model=model, arm=arm, first_trial=first+1, last_trial=last,
                    center_trial=(first+1+last)/2, observed=float(truth[b]), mean=float(values.mean()),
                    q10=float(np.quantile(values, .1)), q90=float(np.quantile(values, .9)),
                    target_presence=float(present[:, first:last].mean()),
                    target_contribution=float(contribution[:, first:last].mean()),
                    target_beta_when_present=float(beta[:, first:last].sum()/present[:, first:last].sum()) if present[:, first:last].any() else np.nan))
            spells.extend(episodes(a, subject, model, arm))
        base = references['BASE']
        for arm, values in references.items():
            np.testing.assert_array_equal(values['seeds'], base['seeds'])
            delta = values['late'] - base['late']
            pairs.append(dict(subject=subject, model=model, arm=arm, delta_late=float(delta.mean()),
                delta_late_mc_se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan,
                delta_energy=values['scores']['energy']-base['scores']['energy'],
                delta_energy_mc_se=jackknife_se(values['deleted']['energy']-base['deleted']['energy']) if values['deleted'] else np.nan))
    for name, records in [('summary', summary), ('paired_effects', pairs), ('paths', paths), ('curves', curves), ('episodes', spells)]:
        pd.DataFrame(records).to_csv(output / (name+'.csv'), index=False)
    audits = [load(p) for p in sorted((source / 'audit').glob('*/audit.json'))]
    pd.DataFrame(audits).to_csv(output / 'catalog_audit.csv', index=False)
    write(output / 'manifest.json', dict(status='complete', source=str(source.resolve()),
        source_protocol_sha256=digest(source / 'protocol.json'), source_artifact_sha256=artifacts,
        analysis_sha256=digest(Path(__file__)), subjects=protocol['settings']['subjects'], models=protocol['settings']['models'],
        arms=protocol['settings']['arms'], paths_per_cell=protocol['settings']['paths'],
        uncertainty='paired complete-path Monte Carlo SE; no population or parameter uncertainty',
        target_selection='training physical-stimulus geometry; privileged diagnostic information',
        summary_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.output)


if __name__ == '__main__': main()
