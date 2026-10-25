"""Score fixed and enriched recovery cases and two oral content baselines.

Use --inference COMPLETE --oral COMPLETE --output NEW. Every table retains
case identity and conditioning. Monte Carlo errors delete complete filter
replicates; enriched cases are excluded from pooled calibration tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import model_1024_state_oral_validation as study
from . import model_1023_execution_recovery_score as recovery
from .model_1022_belief_summary import jackknife_se
from .model_1024_retention_capacity_summary import audit_jobs
from ...evaluation.oral.decision_basis_scoring import DOMAINS, LABELS, label_masses

read, write, digest, nll = study.read, study.write, study.digest, study.nll


def choice_catalogue_mass(choices: np.ndarray, counts: np.ndarray, components: int) -> np.ndarray:
    """Uniform rules and, within each rule, uniform components of the choice."""
    choices, counts = np.asarray(choices), np.asarray(counts)
    if (counts.ndim != 2 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0)
            or components < counts.max() or choices.ndim != 1
            or not np.isin(choices, np.arange(1, counts.shape[1]+1)).all()):
        raise ValueError('Require one-based choices and positive catalogue component counts')
    count = counts[:, choices.astype(int)-1].T
    mass = (np.arange(components)[None, None, :] < count[:, :, None]) / count[:, :, None] / counts.shape[0]
    np.testing.assert_allclose(mass.sum(axis=(1, 2)), 1., atol=1e-12)
    return mass


def window_states(state: dict, start: int, stop: int) -> dict:
    return {k: v if k in ('target_rules', 'seeds') else v[start:stop] for k, v in state.items()}


def window_truth(raw: dict, start: int, stop: int) -> dict:
    return {k: v[start:stop] for k, v in raw.items() if k != 'seeds'}


def error_scores(state: dict, raw: dict) -> tuple[list[dict], list[dict]]:
    n, rules = state['predictive_belief'].shape
    truth = recovery.truth_states(raw, n, rules, state['target_rules'])
    wrong = state['choices'] != state['categories']
    labels = np.where(~truth['present'], 0, np.where(truth['target_execution'] > .5, 2, 1))[wrong]
    p = state['conditional_decision_sources'][wrong]
    if not len(labels):
        return [], []
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12)
    rows = []
    for k, name in enumerate(('absent', 'available_other', 'target_execution')):
        positive = labels == k; selected = p.argmax(axis=1) == k
        rows.append(dict(timing='filtered', state='error_'+name,
            **recovery.binary_scores(p[:, k], positive),
            hard_true_positive=int((selected & positive).sum()),
            hard_false_positive=int((selected & ~positive).sum()),
            hard_recall=float(selected[positive].mean()) if positive.any() else np.nan,
            hard_precision=float(positive[selected].mean()) if selected.any() else np.nan,
            true_event_probability=float(p[positive, k].mean()) if positive.any() else np.nan))
    confusion = [dict(truth=int(a), inferred=int(b), count=int(((labels == a) & (p.argmax(axis=1) == b)).sum()))
                 for a in range(3) for b in range(3)]
    return rows, confusion


def recovery_metrics(parts: list[dict], raw: dict) -> tuple[dict, list, list, list]:
    state = recovery.pool_states(parts)
    metrics, onsets = recovery.state_metrics(state, raw, 'STICKY')
    extra, confusion = error_scores(state, raw); metrics.extend(extra)
    return state, metrics, onsets, confusion


def probability_arrays(jobs: list[dict]) -> np.ndarray:
    parts = [recovery.checked_arrays(Path(j['target'])) for j in jobs]
    for p, j in zip(parts, jobs):
        np.testing.assert_array_equal(p['seeds'], j['seeds'])
    return np.concatenate([p['observed_probability'] for p in parts])


def oral_baselines(oral: Path, output: Path) -> dict:
    basis = oral/'basis'; records = [read(f) for f in sorted((basis/'states').glob('*.json'))]
    groups = sorted({(r['subject'], r['arm']) for r in records})
    summaries = []; details = []; coverage = []
    for reading in ('midpoint', 'relative'):
        coding = oral/('coding_'+reading)
        coded = pd.read_csv(coding/'trials.csv'); matches = pd.read_csv(coding/'component_matches.csv')
        aligned = pd.read_csv(oral/('alignment_'+reading)/'trials.csv')
        for sid in sorted(coded.iSub.unique()):
            case = coded.loc[coded.iSub.eq(sid)]
            for scope, group in case.groupby('scope', dropna=False):
                coverage.append(dict(reading=reading, subject=int(sid), scope=scope, reports=len(group)))
        for sid, arm in groups:
            record = next(r for r in records if (r['subject'], r['arm']) == (sid, arm))
            # The exported record identifies the original state capture name.
            name = Path(record['source_capture']).name
            file = basis/'states'/name
            with np.load(file.with_suffix('.npz')) as z:
                arrays = {k: z[k].copy() for k in ('choices', 'component_counts', 'choice_component_joint_mass')}
            job = read(basis/'jobs'/name)
            frame, cut = study.alignment.align_identities(coded, job, len(arrays['choices']))
            h, m = arrays['choice_component_joint_mass'].shape[1:]
            labels = study.alignment.make_labels(frame, matches, h, m)
            mass = choice_catalogue_mass(arrays['choices'], arrays['component_counts'], m)
            baseline = label_masses(mass, labels)
            dynamic = aligned.loc[aligned.iSub.eq(sid) & aligned.arm.eq(arm)].sort_values('state_trial').reset_index(drop=True)
            for field in ('iSession', 'iTrial', 'scope'):
                np.testing.assert_array_equal(frame[field].fillna(''), dynamic[field].fillna(''))
            ordinal = np.arange(len(frame)); complete = frame.scope.eq('complete').to_numpy()
            windows = dict(whole=np.ones(len(frame), bool), training=ordinal < cut, test=ordinal >= cut,
                           first64=ordinal < 64, last64=ordinal >= max(0, len(frame)-64))
            for d, domain in enumerate(DOMAINS):
                dyn = dynamic[[domain+'_'+k for k in LABELS]].to_numpy()
                static = dynamic[['prefix_static_'+domain+'_'+k for k in LABELS]].to_numpy()
                covered = np.any(labels[..., d] == 0, axis=(1, 2))
                if complete.any():
                    np.testing.assert_allclose(baseline[complete, d].sum(axis=1), 1., atol=1e-12)
                for window, window_mask in windows.items():
                    for subset in ('complete', 'catalogue_supported'):
                        mask = complete & window_mask & (covered if subset == 'catalogue_supported' else True)
                        smask = mask & np.isfinite(static).all(axis=1)
                        row = dict(reading=reading, subject=sid, arm=arm, window=window, domain=domain,
                            subset=subset, reports=int(mask.sum()), full_reports=int((complete & window_mask).sum()),
                            static_reports=int(smask.sum()), covered_reports=int((covered & complete & window_mask).sum()))
                        for k, label in enumerate(LABELS):
                            for name_, values in (('dynamic', dyn), ('choice_only', baseline[:, d]), ('prefix_static', static)):
                                selected = smask if name_ == 'prefix_static' else mask
                                row[name_+'_'+label] = float(values[selected, k].mean()) if selected.any() else np.nan
                        row['dynamic_on_static'] = float(dyn[smask, 0].mean()) if smask.any() else np.nan
                        row['choice_only_on_static'] = float(baseline[smask, d, 0].mean()) if smask.any() else np.nan
                        row['dynamic_minus_choice'] = row['dynamic_support']-row['choice_only_support']
                        row['dynamic_minus_static'] = row['dynamic_on_static']-row['prefix_static_support']
                        summaries.append(row)
                for t in range(len(frame)):
                    details.append(dict(reading=reading, subject=sid, arm=arm, domain=domain, state_trial=t,
                        complete=bool(complete[t]), covered=bool(covered[t]),
                        choice_only_support=baseline[t, d, 0], dynamic_support=dyn[t, 0], static_support=static[t, 0]))
    pd.DataFrame(summaries).to_csv(output/'oral_baselines.csv', index=False)
    pd.DataFrame(details).to_csv(output/'oral_baseline_trials.csv', index=False)
    pd.DataFrame(coverage).to_csv(output/'oral_scope_counts.csv', index=False)
    return dict(baseline_rows=len(summaries), readings=2, full_report_denominator=True,
                baseline='uniform rules, uniform components within choice category; plus frozen prefix basis')


def run(inference: Path, oral: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new summary directory')
    done, protocol = read(inference/'completion.json'), read(inference/'protocol.json')
    if done['status'] != 'complete' or done['protocol_sha256'] != digest(inference/'protocol.json'):
        raise ValueError('Incomplete or modified state inference')
    if read(oral/'completion.json')['status'] != 'complete':
        raise ValueError('Incomplete oral validation')
    output.mkdir(parents=True)
    rows = []; onset_rows = []; confusion_rows = []; params = []; curves = []; resolution = []; nll_rows = []
    training_profile = pd.read_csv(Path(protocol['fitted'])/'screen_scores.csv')
    for case in protocol['cases']:
        c, truth, roles = case['definition'], case['truth'], case['roles']; identity = c['case_id']
        if digest(Path(truth['states']).with_suffix('.npz')) != truth['states_sha256']:
            raise ValueError('Truth array changed')
        with np.load(Path(truth['states']).with_suffix('.npz')) as z:
            raw = {k: z[k].copy() for k in z.files}
        points = {point_id: [recovery.checked_arrays(Path(stem), 'array_sha256') for stem in stems]
                  for point_id, stems in case['states'].items()}
        for point_id, stems in case['states'].items():
            for stem, part in zip(stems, points[point_id]):
                meta = read(Path(stem).with_suffix('.json'))
                if meta['candidate']['id'] != point_id or meta['subject'] != c['subject']:
                    raise ValueError('Captured state point or subject changed')
                np.testing.assert_array_equal(part['target_rules'], c['target_rules'])
                np.testing.assert_array_equal(part['choices'], raw['choices'][:len(part['choices'])])
                np.testing.assert_array_equal(part['categories'], raw['categories'][:len(part['choices'])])
        all_prob = {point_id: probability_arrays(jobs) for point_id, jobs in case['point_jobs'].items()}
        n = len(next(iter(points.values()))[0]['choices']); cut = min(c['training_trials'], n)
        base = dict(case_id=identity, subject=c['subject'], scope=c['scope'])
        for role, point in roles.items():
            probability = all_prob[point['id']]
            for window, a, b in [('whole', 0, n), ('training', 0, cut), ('test', cut, n)]:
                if a == b:
                    continue
                p = probability[:, a:b]; tp = all_prob[roles['true']['id']][:, a:b]
                gap = nll(p)-nll(tp)
                deleted = [nll(np.delete(p, i, 0))-nll(np.delete(tp, i, 0)) for i in range(len(p))]
                nll_rows.append(dict(**base, role=role, window=window, nll=nll(p), true_nll=nll(tp),
                                     difference=gap, difference_mcse=jackknife_se(np.array(deleted))))
            if role not in ('true', 'adopted'):
                continue
            parts = points[point['id']]
            for window, a, b in [('whole', 0, n), ('training', 0, cut), ('test', cut, n)]:
                if a == b:
                    continue
                own_parts = [window_states(x, a, b) for x in parts]; own_raw = window_truth(raw, a, b)
                state, metrics, onsets, confusion = recovery_metrics(own_parts, own_raw)
                deleted = [recovery_metrics(own_parts[:i]+own_parts[i+1:], own_raw)[1] for i in range(len(own_parts))]
                for metric in metrics:
                    key = metric['timing'], metric['state']
                    for field in ('brier', 'log_loss', 'accuracy', 'hard_recall', 'true_event_probability'):
                        if field in metric:
                            values = [next(x for x in d if (x['timing'], x['state']) == key)[field] for d in deleted]
                            metric[field+'_mcse'] = jackknife_se(np.array(values)) if np.isfinite(values).all() else np.nan
                    rows.append(dict(**base, role=role, window=window, **metric))
                confusion_rows.extend(dict(**base, role=role, window=window, **r) for r in confusion)
                if window == 'whole':
                    onset_rows.extend(dict(**base, role=role, **r) for r in onsets)
                    true_state = recovery.truth_states(raw, n, state['predictive_belief'].shape[1], state['target_rules'])
                    wrong = state['choices'] != state['categories']
                    error_truth = np.where(~true_state['present'], 0, np.where(true_state['target_execution'] > .5, 2, 1))
                    for t in range(n):
                        curves.append(dict(**base, role=role, trial=t+1, wrong=bool(wrong[t]),
                            true_present=bool(true_state['present'][t]), true_target_execution=float(true_state['target_execution'][t]),
                            error_truth=int(error_truth[t]),
                            predictive_present=float(state['predictive_target_present'][t]),
                            filtered_present=float(state['filtered_target_present'][t]),
                            predictive_target_execution=float(state['predictive_target_execution'][t]),
                            filtered_target_execution=float(state['filtered_target_execution'][t]),
                            p_error_absent=float(state['conditional_decision_sources'][t, 0]),
                            p_error_other=float(state['conditional_decision_sources'][t, 1]),
                            p_error_executed=float(state['conditional_decision_sources'][t, 2])))
            if point['id'] in case['resolution']:
                high_jobs = case['resolution'][point['id']]
                high_parts = [recovery.checked_arrays(Path(j['state_capture_target']), 'array_sha256') for j in high_jobs]
                low_state = recovery.pool_states(parts[:len(high_parts)]); high_state = recovery.pool_states(high_parts)
                for field in ('filtered_target_present', 'filtered_target_execution', 'conditional_decision_sources'):
                    drift = np.abs(high_state[field]-low_state[field])
                    row = dict(**base, role=role, field=field, mean_absolute_change=float(drift.mean()),
                               max_absolute_change=float(drift.max()), comparison='paired seeds; 512 to 1024 particles')
                    if field == 'conditional_decision_sources':
                        low_errors = error_scores(low_state, raw)[0]; high_errors = error_scores(high_state, raw)[0]
                        if low_errors:
                            lo = next(r for r in low_errors if r['state'] == 'error_available_other')
                            hi = next(r for r in high_errors if r['state'] == 'error_available_other')
                            row.update(rare_events=hi['positives'], low_rare_hits=lo['hard_true_positive'],
                                       high_rare_hits=hi['hard_true_positive'], low_true_event_probability=lo['true_event_probability'],
                                       high_true_event_probability=hi['true_event_probability'])
                    resolution.append(row)
                high_p = probability_arrays(high_jobs); low_p = probability[:len(high_p)]
                resolution.append(dict(**base, role=role, field='mean_nll', mean_absolute_change=abs(nll(high_p)-nll(low_p)),
                                       signed_change=nll(high_p)-nll(low_p), comparison='paired seeds; 512 to 1024 particles'))
        if case['fitted']:
            param = dict(**base, generating_point=roles['true']['id'])
            own_profile = training_profile.loc[training_profile.case_id.eq(identity)]
            generating_rows = own_profile.loc[own_profile.candidate.eq(roles['true']['id'])].to_dict('records')
            if len(generating_rows) != 1 and not done['smoke']:
                raise ValueError('Formal conditional support must contain the generating point')
            param['training_qualified_points'] = int(own_profile.eligible.sum())
            param['true_training_qualified'] = bool(generating_rows[0]['eligible']) if generating_rows else None
            param['true_screen_nll_gap'] = float(generating_rows[0]['training_nll']-own_profile.training_nll.min()) if generating_rows else np.nan
            for role, point in roles.items():
                param[role+'_delta'] = point['hyperparams'][study.DELTA]
                param[role+'_capacity'] = point['hyperparams'][study.CAPACITY]
                param[role+'_point'] = point['id']
            param['adopted_exact'] = param['true_point'] == param['adopted_point']
            param['nll_reference_exact'] = param['true_point'] == param['nll_reference_point']
            param['adopted_delta_error'] = param['adopted_delta']-param['true_delta']
            param['adopted_capacity_error'] = param['adopted_capacity']-param['true_capacity']
            param['generation_replaced_nll'] = case['fitted']['decision']['replaced']
            params.append(param)
    frames = dict(state_metrics=pd.DataFrame(rows), onsets=pd.DataFrame(onset_rows), error_confusion=pd.DataFrame(confusion_rows),
        parameters=pd.DataFrame(params), state_trials=pd.DataFrame(curves), resolution=pd.DataFrame(resolution), independent_nll=pd.DataFrame(nll_rows))
    for name, frame in frames.items():
        frame.to_csv(output/(name+'.csv'), index=False)
    calibration = []
    trial_frame = frames['state_trials'].loc[lambda x: x.scope.eq('fixed')]
    for role, group in trial_frame.groupby('role'):
        for field, target in [('filtered_present', 'true_present'), ('filtered_target_execution', 'true_target_execution'), ('p_error_other', 'error_truth')]:
            g = group.loc[group.wrong] if field == 'p_error_other' else group
            y = g[target].eq(1).to_numpy() if field == 'p_error_other' else g[target].to_numpy(float)
            probabilities = g[field].to_numpy(); bins = np.minimum((probabilities*10).astype(int), 9)
            for b in range(10):
                mask = bins == b
                calibration.append(dict(role=role, field=field, bin=b, trials=int(mask.sum()),
                    cases=int(g.loc[mask, 'case_id'].nunique()), mean_probability=float(probabilities[mask].mean()) if mask.any() else np.nan,
                    frequency=float(y[mask].mean()) if mask.any() else np.nan))
    pd.DataFrame(calibration).to_csv(output/'calibration.csv', index=False)
    oral_checks = oral_baselines(oral, output)
    fitted = Path(protocol['fitted']); audits, hashes = audit_jobs(fitted)
    pd.DataFrame(audits).to_csv(output/'training_boundary_audit.csv', index=False)
    # Observer jobs use the unchanged shared execution audit; numeric thread pools are checked individually.
    for file in sorted((inference/'jobs').glob('*.json')):
        job = read(file); stem = Path(job['target']); meta = read(stem.with_suffix('.json'))
        if any(p.get('num_threads', 1) != 1 for p in meta['numeric_pools']):
            raise ValueError('Multiple numeric threads in inference')
        if digest(stem.with_suffix('.npz')) != meta['predictions_sha256']:
            raise ValueError('Inference predictions changed')
        hashes[str(stem.with_suffix('.npz'))] = meta['predictions_sha256']
    write(output/'completion.json', dict(status='complete', smoke=done['smoke'], source_sha256=digest(Path(__file__)),
        inference_protocol_sha256=digest(inference/'protocol.json'), oral_completion_sha256=digest(oral/'completion.json'),
        cases=len(protocol['cases']), fixed_cases=len(params), enriched_cases=sum(c['definition']['scope'] != 'fixed' for c in protocol['cases']),
        fitting_jobs=len(audits), inference_jobs=len(protocol['jobs']), training_boundary_enforced=True, numeric_threads=1,
        scope='two-parameter conditional recovery; fixed other parameters; filtering before current feedback',
        oral=oral_checks, input_sha256=hashes,
        artifacts_sha256={f.name: digest(f) for f in sorted(output.glob('*.csv'))},
        evidence='development; oral content compatibility; complete latent truth only in simulations'))
    print('State recovery, numerical resolution, oral denominators and baselines checked', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('inference', 'oral', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(); run(args.inference, args.oral, args.output)
