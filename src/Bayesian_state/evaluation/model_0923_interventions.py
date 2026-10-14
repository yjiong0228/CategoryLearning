"""Summarize fixed-parameter intervention paths; no fitting or model selection.

The independent unit for Monte Carlo comparisons is a complete seeded path.
Wrong-rule clearance starts at the first negative rolling evidence score in
one residence, and distinguishes right censoring from an observed exit.
Optional counterevidence_windows/residence_horizons add rule residence and
window-sensitivity tables. comparisons explicitly selects variant pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_cohort_paths import summarize_trace


def wrong_rule_clearance(a: dict, rules: list[int], horizon: int) -> list[dict]:
    """One record per wrong-rule residence with any net counterevidence."""
    n, paths, _ = a['rule'].shape
    rows = []
    for path in range(paths):
        opened = {}
        for trial in range(n+1):
            ids = a['rule'][trial,path] if trial < n else []
            ids = {int(h):slot for slot,h in enumerate(ids) if h >= 0 and h not in rules}
            for h in set(opened)-set(ids):
                first = opened.pop(h)
                if first is not None:
                    rows.append(dict(path=path, rule=h, first_negative_trial=first+1,
                        duration=trial-first, right_censored=trial==n,
                        clearance_eligible=first+horizon < n,
                        cleared_within_horizon=trial<n and trial-first<=horizon))
            for h, slot in ids.items():
                if h not in opened:
                    opened[h] = None
                if opened[h] is None and a['evidence_score'][trial,path,slot] < 0:
                    opened[h] = trial
    return rows


def summarize(source: Path) -> None:
    manifest = json.loads((source/'simulation_manifest.json').read_text())
    settings = manifest['settings']
    destinations = ['subjects.csv','paths.csv','set_residences.csv','exits.csv','wrong_rule_clearance.csv',
                    'curves.csv','comparisons.csv','batch_comparisons.csv','analysis_validation.json']
    if any((source/name).exists() for name in destinations):
        raise FileExistsError('Analysis already exists; retain previous research results')
    summaries, paths_all, episodes_all, exits_all, wrong_all, curves_all, checks = [],[],[],[],[],[],[]
    incompatible_all, sensitivity_all = [], []
    for record in manifest['records']:
        directory = source / record['id']
        prepared = json.loads((directory/'prepared.json').read_text())
        catalog = pd.read_csv(directory/'rule_catalog.csv')
        rules = catalog.loc[catalog.physical_argmax_accuracy>=settings['geometric_accuracy_threshold'],'rule'].astype(int).tolist()
        observed = pd.read_csv(directory/'observed_trials.csv')
        for variant in settings['variants']:
            tag = dict(id=record['id'], subject=record['subject'], condition=prepared['condition'], variant=variant)
            chunks = []
            for file in sorted((directory/variant/'traces').glob('C_*.npz')):
                check = json.loads(file.with_suffix('.json').read_text())
                if hashlib.sha256(file.read_bytes()).hexdigest()!=check['sha256']:
                    raise ValueError(f'Trace changed: {file}')
                checks.append(dict(tag, chunk=file.stem, **check))
                with np.load(file) as z:
                    chunks.append({k:z[k] for k in z.files})
            a = {key: np.concatenate([chunk[key] for chunk in chunks],axis=0 if key=='seeds' else 1)
                 for key in chunks[0] if key!='categories'}
            a['categories'] = chunks[0]['categories']
            np.testing.assert_array_equal(a['seeds'], prepared['seeds'])
            np.testing.assert_array_equal(a['categories'], observed.category)
            p, episodes, exits, risks, curve, slots, correct = summarize_trace(a,rules,
                settings['late_window'], settings['support_threshold'],settings['retention_trials'])
            wrong = wrong_rule_clearance(a,rules,settings['retention_trials'])
            extra = []
            if settings.get('counterevidence_windows'):
                from .model_0923_retention_diagnostics import retention_diagnostics
                extra, incompatible, sensitivity = retention_diagnostics(a,rules,prepared['condition'],
                    settings['late_window'],settings['counterevidence_windows'],settings['residence_horizons'])
                incompatible_all.extend(dict(item,**tag) for item in incompatible)
                sensitivity_all.extend(dict(item,**tag) for item in sensitivity)
            for path,row in enumerate(p):
                if extra:
                    row.update(extra[path])
                row.update(seed=int(a['seeds'][path]), batch=path//16,
                    admitted=int(a['admitted'][:,path].sum()), rejected=int(a['rejected'][:,path].sum()),
                    useful_admitted=int((a['admitted'][:,path]&np.isin(a['proposed'][:,path],rules)).sum()),
                    useful_rejected=int((a['rejected'][:,path]&np.isin(a['proposed'][:,path],rules)).sum()),
                    mean_search_probability=float(a['search_probability'][1:,path].mean()),
                    compatible_autonomous_exits=sum(x['path']==path and x['cause']=='autonomous_exit' for x in exits),
                    compatible_capacity_exits=sum(x['path']==path and x['cause']=='capacity' for x in exits),
                    compatible_joint_exits=sum(x['path']==path and x['cause']=='joint_selection' for x in exits),
                    high_support_exits=sum(x['path']==path and x['cause']=='autonomous_exit' and x['high_support'] for x in exits))
            frame = pd.DataFrame(p)
            row = dict(tag, paths=len(frame), trials=len(observed), rules=','.join(map(str,rules)), fit_status=prepared['fit_status'])
            measures = [key for key in p[0] if key not in ('path','seed','batch','first_entry')]
            for key in measures:
                row[key] = float(frame[key].mean())
            row['first_entry_median_given_found'] = float(frame.first_entry.median())
            eligible = [e for e in episodes if e['retention_eligible']]
            row['retention_episodes'] = len(eligible)
            row['retention_fraction'] = np.mean([e['retained'] for e in eligible]) if eligible else np.nan
            eligible_wrong = [e for e in wrong if e['clearance_eligible']]
            row['wrong_clearance_episodes'] = len(eligible_wrong)
            row['wrong_clearance_fraction'] = np.mean([e['cleared_within_horizon'] for e in eligible_wrong]) if eligible_wrong else np.nan
            for label, select in [('high_support',lambda r:r['high_support']), ('low_support',lambda r:not r['high_support'])]:
                group = [r for r in risks if select(r)]
                row[label+'_exit_hazard'] = np.mean([r['exited'] for r in group]) if group else np.nan
                row[label+'_exposure'] = len(group)
            row['late_observed'] = float((observed.choice==observed.category).iloc[-settings['late_window']:].mean())
            row['late_conditional'] = float(observed.conditional_correct.iloc[-settings['late_window']:].mean()) if 'conditional_correct' in observed else np.nan
            row['late_mean_absolute_error'] = abs(row['late_correct']-row['late_observed'])
            row['late_correct_q10'], row['late_correct_q90'] = frame.late_correct.quantile([.1,.9])
            block_rows = []
            realized = a['choice']==a['categories'][:,None]
            for start in range(0,len(observed),32):
                end = min(start+32,len(observed))
                per_path = correct[start:end].mean(axis=0)
                per_realized = realized[start:end].mean(axis=0)
                block_rows.append(dict(tag,first_trial=start+1,last_trial=end,
                    observed=float((observed.choice==observed.category).iloc[start:end].mean()),
                    mean=float(per_path.mean()),q10=float(np.quantile(per_path,.1)),q90=float(np.quantile(per_path,.9)),
                    realized_q10=float(np.quantile(per_realized,.1)),realized_q90=float(np.quantile(per_realized,.9))))
            blocks = pd.DataFrame(block_rows)
            row['block_mean_absolute_error'] = float((blocks['mean']-blocks.observed).abs().mean())
            row['block_realized_80_coverage'] = float(((blocks.observed>=blocks.realized_q10)&(blocks.observed<=blocks.realized_q90)).mean())
            row['block_realized_80_width'] = float((blocks.realized_q90-blocks.realized_q10).mean())
            summaries.append(row); curves_all.extend(block_rows)
            for collection,destination in ((p,paths_all),(episodes,episodes_all),(exits,exits_all),(wrong,wrong_all)):
                destination.extend([dict(item,**tag) for item in collection])
            with (directory/variant/'correct_paths.npz').open('xb') as stream:
                np.savez_compressed(stream,correct_probability=correct,realized=realized,seeds=a['seeds'])
    summary = pd.DataFrame(summaries)
    paths = pd.DataFrame(paths_all)
    comparisons, batches = [],[]
    metrics = ['late_correct','late_active','ever_active','useful_proposed','useful_admitted',
               'proposed','compatible_autonomous_exits','compatible_capacity_exits','high_support_exits']
    metrics.extend(key for key in ('mean_active_candidates','late_incompatible_contribution',
        'late_incompatible_candidates','total_autonomous_exits','total_capacity_replacements',
        'total_joint_removals','compatible_joint_exits') if key in paths)
    for record in manifest['records']:
        s = paths.loc[paths.id==record['id']]
        pairs = settings.get('comparisons', [(v,'B') for v in settings['variants'] if v!='B']+
            [(v,r) for v,r in [('S1','S_control'),('R1','R_control')] if v in settings['variants'] and r in settings['variants']])
        for target,reference in pairs:
            left,right = [s.loc[s.variant==v].sort_values('path') for v in (target,reference)]
            np.testing.assert_array_equal(left.seed,right.seed)
            for metric in metrics:
                delta = left[metric].to_numpy(dtype=float)-right[metric].to_numpy(dtype=float)
                tag = dict(id=record['id'],subject=record['subject'],target=target,reference=reference,metric=metric)
                comparisons.append(dict(tag,delta=float(delta.mean()),
                    paired_mc_se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan,
                    paths=len(delta)))
                for batch in sorted(left.batch.unique()):
                    values = delta[left.batch.to_numpy()==batch]
                    batches.append(dict(tag,batch=int(batch),delta=float(values.mean()),paths=len(values)))
    for name,rows in [('subjects',summary),('paths',paths),('set_residences',episodes_all),('exits',exits_all),
        ('wrong_rule_clearance',wrong_all),('curves',curves_all),('comparisons',comparisons),('batch_comparisons',batches)]:
        pd.DataFrame(rows).to_csv(source/(name+'.csv'),index=False)
    if settings.get('counterevidence_windows'):
        pd.DataFrame(incompatible_all).to_csv(source/'incompatible_residences.csv',index=False)
        pd.DataFrame(sensitivity_all).to_csv(source/'clearance_sensitivity.csv',index=False)
    validation = dict(status='passed',participants=len(manifest['records']),variants=len(settings['variants']),
        paths=len(paths),baseline_exact_paths=sum(c['baseline_exact_paths'] for c in checks),
        all_numeric_pools_single_thread=all(pool['num_threads']==1 for c in checks for pool in c['numeric_pools']),
        trace_cpu_seconds=sum(c['cpu_seconds'] for c in checks if not c.get('reused',False)),
        new_paths=sum(c['paths'] for c in checks if not c.get('reused',False)),
        reused_paths=sum(c['paths'] for c in checks if c.get('reused',False)),checks=checks)
    (source/'analysis_validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    summarize(parser.parse_args().source)
