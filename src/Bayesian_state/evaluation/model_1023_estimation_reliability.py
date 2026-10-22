"""Audit completed bounded searches and independent fixed-candidate scores.

Use --study COMPLETE_PHASE --output NEW [--allow-smoke]. Every executed filter,
candidate quota, particle budget, seed bank and data boundary is checked against
the frozen case plan. Independent audits retain the preselected point even if
another candidate scores better. Monte Carlo resolution describes numerical
precision at fixed parameters, not a confidence region for individual traits.
Recovery summaries reuse the post-freeze generating-point and state results.
No new fitting, state simulations, oral recoding or hypothesis tests occur here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import nll, jackknife_scores, jackknife_se
from ..workflows.analysis.model_1023_estimation_reliability import ROOT, read, write, digest
from ..workflows.analysis.model_1021_simplification_recovery import compact_zero_effect_coordinates
from ..optimization.search.compact_joint import behavior_key


def expected_stages(settings: dict, smoke: bool) -> dict[str, tuple[int, int, int]]:
    """Map each stage to (candidates, particles, filters per candidate)."""
    if smoke:
        return {name: (count, 2, 2) for name, count in
                [('screen', 4), ('refine_1', 6), ('selection', 2), ('confirmation', 2)]}
    plan = settings['proposals']
    new = plan['elites'] * (plan['rays'] + plan['pairs'] + plan['multi']) + plan['global_starts']
    counts = dict(screen=settings['initial_candidates'], selection=settings['selection_candidates'],
                  confirmation=settings['confirmation_candidates'])
    counts.update({f'refine_{i}': new + settings['carry_candidates'] for i in range(1, settings['rounds'] + 1)})
    return {name: (count, settings['budgets'][name]['particles'], settings['budgets'][name]['seeds'])
            for name, count in counts.items()}


def validate_stage(folder: Path, case: dict, stage: str, budget: tuple, smoke: bool,
                   bind, *, role: str) -> tuple[list[dict], list[dict]]:
    candidates, particles, repetitions = budget
    complete = read(bind(folder/(stage+'_completion.json')))
    if complete['status'] != 'complete' or len(complete['groups']) != 1:
        raise ValueError('Incomplete single-case stage')
    group, = complete['groups']
    rows = group['rows']
    if group['subject'] != case['subject'] or group['model'] != 'COMPACT' or len(rows) != candidates:
        raise ValueError('Stage identity or candidate quota differs')
    if len({behavior_key(r['hyperparams']) for r in rows}) != candidates:
        raise ValueError('Duplicate behavior-equivalent candidate in stage')
    seeds, table, jobs = None, [], []
    meta = read(bind(ROOT/case['settings']['data']/f'subject_{case["subject"]}'/'prepared.json'))
    length = min(8, meta['training_trials']) if smoke else meta['training_trials']
    for point in rows:
        file = Path(point['score_file'])
        record = read(bind(file))
        if (record['particles'] != particles or record['trials'] != length
                or record['candidate']['hyperparams'] != point['hyperparams']
                or record['subject'] != case['subject'] or record['condition'] != case['condition']):
            raise ValueError('Stage score has different parameters, precision or trial boundary')
        with np.load(bind(file.with_suffix('.npz'), record['predictions_sha256'])) as saved:
            probability = saved['observed_probability']
            own_seeds = saved['seeds'].copy()
            if (probability.shape != (repetitions, length) or not np.isfinite(probability).all()
                    or np.any((probability < 0) | (probability > 1))):
                raise ValueError('Invalid saved probability shape or range')
            np.testing.assert_allclose(nll(probability), [record['mean_nll'], point['mean_nll']], atol=1e-12, rtol=0)
        if len(set(own_seeds)) != repetitions:
            raise ValueError('Repeated stage filter seed')
        np.testing.assert_array_equal(own_seeds, record['seeds'])
        if seeds is not None:
            np.testing.assert_array_equal(own_seeds, seeds)
        seeds = own_seeds
        table.append(dict(case_id=case['case_id'], subject=case['subject'], stage=stage,
            phase_role=role, candidate=point['id'], nll=point['mean_nll'], particles=particles,
            filters=repetitions, trials=length))
    found = {}
    for file in sorted((folder/'jobs').glob(stage+'_S*.json')):
        job = read(bind(file))
        record = read(bind(Path(job['target']).with_suffix('.json')))
        if (job['compact_variant'] != 'TRANSFER' or job['entry_control'] != 'PRECISION'
                or job['phase'] != 'training' or len(job['seeds']) != 1
                or job['particles'] != particles or record['trials'] != length
                or record['seeds'] != job['seeds'] or record['candidate'] != job['candidate']
                or any(p['num_threads'] != 1 for p in record['numeric_pools'])):
            raise ValueError('Executed worker differs from the declared mechanism or single-thread budget')
        found.setdefault(job['candidate']['id'], []).extend(job['seeds'])
        jobs.append(dict(case_id=case['case_id'], phase_role=role, stage=stage,
            candidate=job['candidate']['id'], seed=job['seeds'][0], particles=particles, trials=length,
            cpu_seconds=record['cpu_seconds'], seconds=record['seconds']))
    if set(found) != {r['id'] for r in rows}:
        raise ValueError('Executed candidate set differs from stage completion')
    for bank in found.values():
        np.testing.assert_array_equal(sorted(bank), sorted(seeds))
    bank_name = stage if role == 'selection' else 'post_freeze_audit'
    np.testing.assert_array_equal(seeds, case['independent_seed_banks'][bank_name][:repetitions])
    return table, jobs


def summarize(study: Path, output: Path, allow_smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new estimation-summary directory')
    done, protocol = read(study/'completion.json'), read(study/'protocol.json')
    if done['status'] != 'complete' or (done['smoke'] and not allow_smoke):
        raise ValueError('Require a complete phase; smoke summaries need explicit --allow-smoke')
    smoke = done['smoke']
    cases = read(study/'cases.json')['cases']
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('Summary input changed: '+str(path))
        inputs[str(path)] = value
        return path

    for file in ('completion.json', 'protocol.json', 'cases.json', 'selected_nominees.json', 'resource_status.json'):
        bind(study/file)
    bind(Path(__file__))
    history, jobs, audited, parameters = [], [], [], []
    for case in cases:
        fit = Path(case['output'])
        frozen = read(bind(fit/'nominees.json'))
        frozen_done = read(bind(fit/'completion.json'))
        selected = frozen['nominee']['candidate']['id']
        reference = frozen['nominee']['reference_id']
        if (frozen_done['test_choices_used'] or frozen_done['nomination_sha256'] != digest(fit/'nominees.json')
                or frozen['smoke'] is not smoke):
            raise ValueError('Selection boundary or frozen fit changed')
        for stage, budget in expected_stages(case['settings'], smoke).items():
            rows, filters = validate_stage(fit, case, stage, budget, smoke, bind, role='selection')
            history.extend(rows); jobs.extend(filters)
        audit = study/'audit_fit'/case['case_id']
        own = read(bind(audit/'nominees.json'))
        if (own['selected_before_audit'] is not True or own['nominee']['candidate']['id'] != selected
                or own['selection_sha256'] != digest(fit/'nominees.json')):
            raise ValueError('Independent audit changed the selected estimate')
        count = 2 if smoke else case['settings']['audit_candidates']
        budget = (count, 2 if smoke else 1024, 2 if smoke else 8)
        rows, filters = validate_stage(audit, case, 'confirmation', budget, smoke, bind, role='independent_audit')
        history.extend(rows); jobs.extend(filters)
        estimates = {}
        for point in own['nominee']['finalists']:
            record = read(bind(Path(point['score_file'])))
            with np.load(bind(Path(point['score_file']).with_suffix('.npz'), record['predictions_sha256'])) as saved:
                estimates[point['id']] = nll(saved['observed_probability']), jackknife_scores(saved['observed_probability'])
                expected = case['independent_seed_banks']['post_freeze_audit'][:2 if smoke else 8]
                np.testing.assert_array_equal(saved['seeds'], expected)
        best_value, best_jk = estimates[selected]
        for point in own['nominee']['finalists']:
            value, jk = estimates[point['id']]
            difference, error = value-best_value, jackknife_se(jk-best_jk)
            tag = dict(case_id=case['case_id'], subject=case['subject'], condition=case['condition'],
                phase=done['phase'], cohort=case.get('cohort', 'synthetic_recovery'), candidate=point['id'],
                selected=point['id'] == selected, reference=point['id'] == reference,
                nll=value, nll_mc_se=jackknife_se(jk), delta_to_selected=difference,
                delta_mc_se=error, no_clear_loss_at_mc_resolution=difference <= 2*error)
            audited.append(tag)
            inactive = compact_zero_effect_coordinates(point['hyperparams'])
            for key, levels in case['space'].items():
                parameters.append(dict(tag, parameter=key, value=point['hyperparams'][key],
                    fitted=len(levels)>1, known_zero_effect=key in inactive,
                    normalized_grid_position=levels.index(point['hyperparams'][key])/max(1, len(levels)-1)))
        if case['kind'] == 'recovery':
            count, particles, repetitions = budget
            rows, filters = validate_stage(study/'recovery_scores'/case['case_id'], case, 'confirmation',
                                            (1, particles, repetitions), smoke, bind, role='generating_point')
            history.extend(rows); jobs.extend(filters)
    resources = read(study/'resource_status.json')
    if (resources['completed'] != len(jobs) or resources['failed'] or resources['active']
            or resources['peak'] > protocol['workers']):
        raise ValueError('Executed filters or shared worker accounting differs')
    output.mkdir(parents=True)
    for name, records in [('stage_candidates', history), ('executed_filters', jobs),
                          ('audit_candidates', audited), ('audit_parameters', parameters)]:
        pd.DataFrame(records).to_csv(output/(name+'.csv'), index=False)
    recovery_outputs = {}
    if done['phase'] != 'human':
        for name in ('likelihood_comparison', 'parameters'):
            path = bind(study/'recovery_scores'/(name+'.csv'))
            pd.read_csv(path).to_csv(output/('recovery_'+name+'.csv'), index=False)
            recovery_outputs[name] = str(path)
        for name in ('scores', 'trials'):
            path = bind(study/'recovery_states'/(name+'.csv'))
            pd.read_csv(path).to_csv(output/('state_'+name+'.csv'), index=False)
    write(output/'completion.json', dict(status='complete', smoke=smoke, phase=done['phase'],
        study=str(study.resolve()), cases=len(cases), checked_filters=len(jobs), peak_workers=resources['peak'],
        inputs_sha256=inputs, selected_before_independent_audit=True,
        uncertainty='Whole-filter Monte Carlo error at fixed points; candidate comparisons are not parameter confidence intervals',
        recovery=recovery_outputs, artifacts_sha256={p.name: digest(p) for p in output.glob('*.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.study, args.output, args.allow_smoke)
