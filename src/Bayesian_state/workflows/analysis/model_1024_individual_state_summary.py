"""Pool independent particle systems at their observation endpoints.

Each parameter candidate remains separate. Full-record weights use complete
sequence evidence; filtering weights use the evidence accumulated through that
trial. Numerical diagnostics are sensitivity checks, not calibrated coverage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_1024_individual_states import read, write, digest
from ...evaluation.independent_particle_systems import evidence_weights, bank_diagnostics


TARGETS = ('target_present', 'target_belief', 'target_high_belief', 'target_execution')
FIELDS = (*TARGETS, 'source', 'active', 'belief', 'belief_second', 'execution',
          'active_precision', 'active_precision_second', 'component')


def discrepancy(a: dict, b: dict, mode: str) -> dict:
    """Check target probabilities and whole-rule belief/execution distributions."""
    differences = [np.abs(a[mode+'_'+k]-b[mode+'_'+k]) for k in TARGETS]
    differences.extend(.5*np.abs(a[mode+'_'+k]-b[mode+'_'+k]).sum(axis=1)
                       for k in ('belief', 'execution'))
    d = np.stack(differences, axis=1)
    whole = [abs(a[mode+'_'+k].mean()-b[mode+'_'+k].mean()) for k in TARGETS]
    whole.extend(.5*np.abs(a[mode+'_'+k].mean(axis=0)-b[mode+'_'+k].mean(axis=0)).sum()
                 for k in ('belief', 'execution'))
    result = dict(mean=float(d.mean()), q95=float(np.quantile(d, .95)), maximum=float(d.max()),
                  whole_mean=float(np.mean(whole)), whole_maximum=float(np.max(whole)))
    # Preserve the registered aggregate gates. The separate quantities show
    # whether uncertainty concerns the target rule or alternative rule identities.
    for index, name in enumerate((*TARGETS, 'belief', 'execution')):
        result.update({name+'_mean': float(d[:, index].mean()),
                       name+'_q95': float(np.quantile(d[:, index], .95)),
                       name+'_whole': float(whole[index])})
    return result


def pool_arrays(arrays: list[dict], logz: np.ndarray) -> tuple[dict, dict]:
    """Pool all marginals with the same per-system weight at each endpoint."""
    result, diagnostic = {}, {}
    for mode in ('filtered', 'full'):
        weights = evidence_weights(logz if mode == 'filtered' else logz[:, -1])
        if mode == 'full':
            weights = np.broadcast_to(weights[:, None], logz.shape)
        for field in FIELDS:
            key = mode+'_'+field
            shape = arrays[0][key].shape
            total = np.zeros(shape)
            for r, a in enumerate(arrays):
                total += a[key]*weights[r].reshape((len(weights[r]),)+(1,)*(len(shape)-1))
            result[key] = total
        own = bank_diagnostics(weights)
        result[mode+'_system_ess'] = own['ess']
        result[mode+'_maximum_system_weight'] = own['maximum']
        diagnostic[mode+'_ess_min'] = float(own['ess'].min())
        diagnostic[mode+'_ess_median'] = float(np.median(own['ess']))
        diagnostic[mode+'_maximum_weight'] = float(own['maximum'].max())
    result['terminal_system_weights'] = evidence_weights(logz[:, -1])
    result['component_counts'] = arrays[0]['component_counts']
    result['full_unique_ancestors_per_system'] = np.stack([a['full_unique_ancestors'] for a in arrays])
    result['full_ancestral_ess_per_system'] = np.stack([a['full_ancestral_ess'] for a in arrays])
    return result, diagnostic


def load_bank(jobs: list[dict]) -> tuple[list[dict], np.ndarray]:
    arrays, logs = [], []
    for job in jobs:
        file = Path(job['individual_target']).with_suffix('.npz')
        meta = read(file.with_suffix('.json'))
        if digest(file) != meta['array_sha256']:
            raise ValueError('Captured states changed')
        with np.load(file) as z:
            a = {mode+'_'+field: z[mode+'_'+field] for mode in ('filtered', 'full') for field in FIELDS}
            for key in ('component_counts', 'full_unique_ancestors', 'full_ancestral_ess'):
                a[key] = z[key]
        with np.load(Path(job['target']).with_suffix('.npz')) as z:
            probability = z['observed_probability'][0]
            if not np.isfinite(probability).all() or np.any(probability <= 0):
                raise ValueError('Invalid evidence increments')
            logs.append(np.log(probability).cumsum())
        arrays.append(a)
    return arrays, np.stack(logs)


def summarize_bank(jobs: list[dict]) -> tuple[dict, dict]:
    arrays, logz = load_bank(jobs)
    pooled, diagnostics = pool_arrays(arrays, logz)
    diagnostics.update(systems=len(jobs), particles=jobs[0]['particles'],
        log_evidence_min=float(logz[:, -1].min()), log_evidence_max=float(logz[:, -1].max()))
    half = len(jobs)//2
    if half:
        a, _ = pool_arrays(arrays[:half], logz[:half])
        b, _ = pool_arrays(arrays[half:], logz[half:])
        for mode in ('filtered', 'full'):
            diagnostics.update({mode+'_half_'+k: v for k, v in discrepancy(a, b, mode).items()})
            for field in TARGETS:
                pooled[mode+'_half_min_'+field] = np.minimum(a[mode+'_'+field], b[mode+'_'+field])
                pooled[mode+'_half_max_'+field] = np.maximum(a[mode+'_'+field], b[mode+'_'+field])
    pooled['system_log_evidence'] = logz
    return pooled, diagnostics


def numerical_pass(row: dict, thresholds: dict, mode: str) -> bool:
    return bool(row.get('bank_complete', False)
        and row.get(mode+'_ess_min', 0) >= thresholds['minimum_system_ess']
        and row.get(mode+'_maximum_weight', 1) <= thresholds['maximum_system_weight']
        and row.get(mode+'_half_mean', 1) <= thresholds['half_mean_absolute_difference']
        and row.get(mode+'_half_q95', 1) <= thresholds['half_q95_absolute_difference'])


def run(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new summary directory')
    output.mkdir(parents=True)
    protocol = read(source/'protocol.json'); checks = protocol['policy']['numerical_checks']
    all_jobs = [read(f) for f in sorted((source/'jobs').glob('*.json'))]
    completed = [j for j in all_jobs if (source/'ledger'/(j['name']+'.json')).exists()
                 and read(source/'ledger'/(j['name']+'.json'))['status']=='complete']
    rows, extension, selected, recoveries = [], [], [], []
    truths = {t['case']: t for t in read(source/'truths.json')}
    for case in protocol['cases']:
        own = [j for j in completed if j['state_case']==case['id'] and j['state_stage']!='replay']
        banks = {}
        for label, stages in [('primary', ['primary']), ('resolution', ['resolution', 'extension']),
                              ('recovery', ['recovery'])]:
            jobs = sorted([j for j in own if j['state_stage'] in stages], key=lambda j: (j['state_stage'], j['name']))
            if not jobs:
                continue
            pooled, diag = summarize_bank(jobs)
            file = output/f'{case["id"]}_{label}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, **pooled)
            row = dict(case=case['id'], subject=case['subject'], kind=case['kind'], bank=label,
                       candidate=case['candidate']['id'], parameter_status=case['parameter_status'], **diag)
            expected = protocol['policy'][label]['systems']
            if label=='resolution' and any(j['state_case']==case['id'] and j['state_stage']=='extension' for j in all_jobs):
                expected += protocol['policy']['extension']['additional_systems']
            row.update(expected_systems=expected, bank_complete=len(jobs)==expected)
            for mode in ('filtered', 'full'):
                row[mode+'_bank_pass'] = numerical_pass(row, checks, mode)
            banks[label] = (pooled, row, jobs)
            rows.append(row)
        if case['kind']=='human':
            if 'primary' not in banks or 'resolution' not in banks:
                selected.append(dict(case=case['id'], status='missing_required_bank'))
                continue
            low, _, _ = banks['primary']; high, diag, jobs = banks['resolution']
            for mode in ('filtered', 'full'):
                delta = discrepancy(low, high, mode)
                diag.update({mode+'_resolution_'+k: v for k, v in delta.items()})
                diag[mode+'_resolution_pass'] = bool(delta['mean'] <= checks['resolution_mean_absolute_difference']
                    and delta['q95'] <= checks['resolution_q95_absolute_difference'])
                diag[mode+'_numerically_stable'] = diag[mode+'_bank_pass'] and diag[mode+'_resolution_pass']
            if not all(diag[m+'_numerically_stable'] for m in ('filtered', 'full')):
                extension.append(case['id'])
            selected.append(dict(case=case['id'], subject=case['subject'], candidate=case['candidate']['id'],
                file=str((output/f'{case["id"]}_resolution.npz').resolve()),
                status='numerical_check_pass' if case['id'] not in extension else 'numerically_limited',
                parameter_status=case['parameter_status'], jobs=[j['name'] for j in jobs]))
        elif 'recovery' in banks:
            pooled, diag, jobs = banks['recovery']; truth = truths[case['id']]
            if digest(Path(truth['file'])) != truth['sha256']:
                raise ValueError('Simulation truth changed')
            with np.load(truth['file']) as z:
                raw = {k: z[k][truth['index']] for k in ('rule', 'belief', 'precision', 'executed')}
            h = len(pooled['component_counts']); n = len(raw['rule'])
            ids = raw['rule']; valid = ids >= 0; t = np.broadcast_to(np.arange(n)[:, None], ids.shape)
            active = np.zeros((n, h)); belief = np.zeros_like(active); precision = np.zeros_like(active)
            active[t[valid], ids[valid]] = 1
            belief[t[valid], ids[valid]] = raw['belief'][valid]
            precision[t[valid], ids[valid]] = raw['precision'][valid]
            target = np.isin(ids, case['target_rules'])
            present = target.any(axis=1); target_belief = (target*raw['belief']).sum(axis=1)
            executed = np.isin(raw['executed'], case['target_rules'])
            known = dict(target_present=present, target_belief=target_belief,
                         target_high_belief=present & (target_belief >= .5), target_execution=executed)
            for mode in ('filtered', 'full'):
                row = dict(case=case['id'], subject=case['subject'], mode=mode, trials=n,
                           numerically_stable=diag[mode+'_bank_pass'],
                           rare_present_other_count=int(np.sum(present & ~executed)))
                for key, value in known.items():
                    row[key+'_mse'] = float(np.mean((pooled[mode+'_'+key]-value)**2))
                    row[key+'_truth_mean'] = float(np.mean(value))
                    row[key+'_inferred_mean'] = float(np.mean(pooled[mode+'_'+key]))
                row['rule_presence_brier'] = float(np.mean((pooled[mode+'_active']-active)**2))
                row['belief_total_variation'] = float(np.mean(.5*np.abs(pooled[mode+'_belief']-belief).sum(axis=1)))
                row['true_executed_rule_probability'] = float(np.mean(pooled[mode+'_execution'][np.arange(n), raw['executed']]))
                true_source = np.where(~present, 0, np.where(executed, 2, 1))
                for k, label in enumerate(('absent', 'available_other', 'target_executed')):
                    event = true_source == k; prediction = pooled[mode+'_source'][:, k]
                    row['source_'+label+'_count'] = int(event.sum())
                    row['source_'+label+'_brier'] = float(np.mean((prediction-event)**2))
                    row['source_'+label+'_probability_when_true'] = float(prediction[event].mean()) if event.any() else np.nan
                    row['source_'+label+'_probability_when_false'] = float(prediction[~event].mean()) if (~event).any() else np.nan
                # Active precision is a joint state quantity: inactive rules
                # carry zero precision here. No small-denominator conditioning.
                row['active_precision_rmse'] = float(np.sqrt(np.mean((pooled[mode+'_active_precision']-precision)**2)))
                row['active_precision_at_true_active_rmse'] = float(np.sqrt(np.mean(
                    (pooled[mode+'_active_precision'][active.astype(bool)]-precision[active.astype(bool)])**2)))
                recoveries.append(row)
            with (output/f'{case["id"]}_truth.npz').open('xb') as stream:
                np.savez_compressed(stream, **known, active=active, belief=belief, precision=precision,
                                    executed=raw['executed'])
        print(f'Summarized {case["id"]}', flush=True)
    pd.DataFrame(rows).to_csv(output/'numerical_checks.csv', index=False)
    pd.DataFrame(recoveries).to_csv(output/'recovery_scores.csv', index=False)
    write(output/'selected.json', selected)
    write(output/'extension_candidates.json', extension)
    write(output/'completion.json', dict(status='complete', source=str(source.resolve()),
        protocol_sha256=digest(source/'protocol.json'), banks=len(rows), cases=len(selected),
        completed_jobs=len(completed), extension_candidates=len(extension),
        source_sha256=digest(Path(__file__)), numerical_pass_is_calibration=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.output)
