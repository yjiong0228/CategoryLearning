"""Compare frozen estimates, archived references and parameter alternatives.

Use --evaluation COMPLETE_PAIRED_EVALUATION --config FROZEN_ORAL_CONFIG
--output NEW. Reuse original proper scores, whole-filter/path Monte Carlo
errors and both fixed oral readings. All three points were chosen from prefix
behavior before opening later behavior or oral reports. Test choices update
conditional states sequentially; autonomous paths start from the initial state.
Candidate sensitivity describes a finite set, not a parameter confidence set.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import (
    checked_arrays, block_paths, distribution_scores, jackknife_scores, jackknife_se, nll,
)
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis import model_0923_decision_basis_alignment as alignment
from ..workflows.analysis.model_1023_estimation_reliability import ROOT, read, write, digest
from ..optimization.search.compact_joint import behavior_key


def score_arrays(c: dict, a: dict, cut: int, width: int, *, target_rule: int | None = None) -> tuple:
    """Score chronological windows and retain every generated block trajectory."""
    length = len(c['choices'])
    if not 0 < cut < length:
        raise ValueError('Require nonempty prefix and later evaluation windows')
    np.testing.assert_array_equal(c['categories'], a['categories'])
    for bank in (c, a):
        if not np.isfinite(bank['probabilities']).all() or np.any(bank['probabilities'] < 0):
            raise ValueError('Invalid response probabilities')
        np.testing.assert_allclose(bank['probabilities'].sum(-1), 1, atol=1e-12, rtol=0)
        if len(np.unique(bank['seeds'])) != len(bank['seeds']):
            raise ValueError('Repeated evaluation seed')
    actual = c['choices'] == c['categories']
    generated = a['choices'] == a['categories'][None, :]
    scores, curves, paths, states, knives = [], [], [], [], {}
    for phase, first, last in [('training', 0, cut), ('test', cut, length), ('whole', 0, length)]:
        samples = block_paths(generated, first, last, width)
        observed = block_paths(actual[None, :], first, last, width)[0]
        probability = c['observed_probability'][:, first:last]
        jk = dict(choice_nll=jackknife_scores(probability), **path_jackknife(samples, observed))
        knives[phase] = jk
        late = max(first, last-128)
        late_paths = generated[:, late:last].mean(1)
        obs = actual[late:last].mean()
        qs = np.quantile(late_paths, [.1, .5, .9])
        scores.append(dict(phase=phase, trials=last-first, training_trials=cut,
            heldout_parameter_choices=phase == 'test', choice_nll=nll(probability),
            choice_nll_mc_se=jackknife_se(jk['choice_nll']), **distribution_scores(samples, observed),
            energy_mc_se=jackknife_se(jk['energy']), crps_mc_se=jackknife_se(jk['crps']),
            late_accuracy=late_paths.mean(), observed_late=obs,
            late_q10=qs[0], late_q50=qs[1], late_q90=qs[2],
            observed_late_midrank=np.mean(late_paths < obs)+.5*np.mean(late_paths == obs),
            target_late_present=a['present'][:, late:last].mean(),
            target_late_contribution=a['contribution'][:, late:last].mean(),
            target_ever=a['present'][:, :last].any(1).mean()))
        if phase != 'whole':
            continue
        predicted = block_paths(c['correct_probability'], 0, length, width).mean(0)
        lo, hi = np.quantile(samples, [.1, .9], axis=0)
        for i in range(len(observed)):
            t = min((i+1)*width, length)
            curves.append(dict(trial=t, training_trials=cut, observed=observed[i],
                autonomous=samples[:, i].mean(), lower=lo[i], upper=hi[i], conditional=predicted[i]))
            paths.extend(dict(trial=t, seed=int(seed), accuracy=value)
                         for seed, value in zip(a['seeds'], samples[:, i]))
    # Precision is meaningful only while the target rule is present.
    active = a['present'].astype(bool)
    if np.any(a['beta'][~active] != 0):
        raise ValueError('Inactive target precision must be zero in saved paths')
    for first in range(0, length, width):
        last = min(first+width, length)
        mask = active[:, first:last]
        row = dict(trial=last, training_trials=cut, target_present=mask.mean(),
            target_contribution=a['contribution'][:, first:last].mean(),
            target_precision_given_presence=a['beta'][:, first:last][mask].mean() if mask.any() else np.nan,
            target_precision_observations=int(mask.sum()))
        row.update({k: a[k][:, first:last].mean() for k in
                    ('search', 'rejected', 'target_capacity_dropped', 'active_count')})
        # Proposal records contain catalogue IDs; -1 denotes no proposal.
        proposals = a['proposed'][:, first:last]
        row['proposal_rate'] = (proposals >= 0).mean()
        row['target_proposal_rate'] = (proposals == target_rule).mean() if target_rule is not None else np.nan
        states.append(row)
    return scores, curves, paths, states, knives


def summarize(evaluation: Path, config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new frozen-evaluation summary directory')
    done = read(evaluation/'completion.json')
    protocol = read(evaluation/'protocol.json')
    if done['status'] != 'complete' or done['smoke']:
        raise ValueError('Require complete scientific paired evaluations')
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Frozen evaluation input changed: '+str(path))
        inputs[str(path)] = actual
        return path

    for file in (evaluation/'completion.json', evaluation/'protocol.json', config, Path(__file__),
                 Path(alignment.__file__), Path(inspect.getfile(block_paths)), Path(inspect.getfile(path_jackknife))):
        bind(file)
    plan = read(bind(evaluation/'cases.json', done['cases_sha256']))
    for file, sha in plan['inputs_sha256'].items():
        bind(file, sha)
    policy = yaml.safe_load(config.read_text())
    study = Path(protocol['study'])
    fitted_cases = {c['case_id']: c for c in read(bind(study/'cases.json'))['cases']}
    cases = plan['cases']
    if len(cases) != done['candidates'] or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Paired candidate count changed')
    scores, curves, paths, states, parameters = [], [], [], [], []
    knives, identities, oral_groups = {}, {}, {}
    for original, fitted in fitted_cases.items():
        own = [c for c in cases if c['original_case_id'] == original]
        audited = read(bind(study/'audit_fit'/original/'nominees.json'))['nominee']['finalists']
        if {c['candidate'] for c in own} != {c['id'] for c in audited}:
            raise ValueError('Evaluated alternatives differ from the fixed audit set')
        for role in ('selected', 'reference'):
            if sum(role in c['roles'] for c in own) != 1:
                raise ValueError('Each participant needs one selected estimate and one reference')
    for case in cases:
        fitted = fitted_cases[case['original_case_id']]
        own = Path(case['evaluation'])
        frozen_protocol = read(bind(own/'protocol.json'))
        frozen_done = read(bind(own/'completion.json'))
        fit = Path(frozen_protocol['fit'])
        frozen = read(bind(fit/'nominees.json', frozen_protocol['nomination_sha256']))
        fit_done = read(bind(fit/'completion.json'))
        if (frozen_done['status'] != 'complete' or frozen_done['smoke']
                or frozen['status'] != 'frozen_before_test' or fit_done['test_choices_used']
                or frozen_protocol['compact_variant'] != 'TRANSFER'):
            raise ValueError('Evaluation differs from the frozen prefix estimate')
        point = frozen['nominee']['candidate']
        if behavior_key(point['hyperparams']) != behavior_key(case['candidate_hyperparams']):
            raise ValueError('Evaluated parameters differ from the fixed candidate')
        arrays, records = {}, {}
        for role in ('conditional', 'autonomous'):
            record, = read(bind(own/role/'evaluation.json'))['rows']
            if (record['candidate'] != point or record['subject'] != case['subject']
                    or record['condition'] != case['condition']):
                raise ValueError('Saved predictions use different parameters or participant')
            file = own/role/'evaluate'/f'subject_{case["subject"]}'/'COMPACT'/(point['id']+'.npz')
            bind(file, record['predictions_sha256'])
            arrays[role] = checked_arrays(own/role, record)
            records[role] = record
            np.testing.assert_array_equal(arrays[role]['seeds'], case['reference_seeds'][role])
        c, a = arrays['conditional'], arrays['autonomous']
        length, cut = len(c['choices']), records['conditional']['training_trials']
        if (c['observed_probability'].shape != (8, length) or a['choices'].shape != (128, length)
                or records['conditional']['particles'] != 1024):
            raise ValueError('Evaluation particle or path budget changed')
        tag = dict(subject=case['subject'], condition=case['condition'], cohort=case['cohort'],
            candidate=case['candidate'], roles='|'.join(case['roles']),
            selected='selected' in case['roles'], reference='reference' in case['roles'])
        rows = score_arrays(c, a, cut, frozen['settings']['features']['block_width'],
                            target_rule=records['autonomous']['target_rule'])
        for target, source in zip((scores, curves, paths, states), rows[:4]):
            target.extend(dict(tag, **r) for r in source)
        knives[case['case_id']] = rows[4]
        identities[case['case_id']] = {role: {key: array[key] for key in ('seeds', 'categories')}
                                      for role, array in arrays.items()}
        identities[case['case_id']]['choices'] = c['choices']
        parameters.extend(dict(tag, parameter=k, value=v, fitted=len(fitted['space'][k]) > 1)
                          for k, v in case['candidate_hyperparams'].items())
        captured = []
        for file in sorted((own/'state_capture').glob('*.json')):
            record = read(bind(file))
            job = read(bind(record['original_job'], record['original_job_sha256']))
            if (record['candidate'] != point or job['compact_variant'] != 'TRANSFER'
                    or job['entry_control'] != 'PRECISION' or record['particles'] != 1024):
                raise ValueError('Original-filter oral capture uses different parameters or mechanism')
            bind(file.with_suffix('.npz'), record['array_sha256'])
            captured.append((file, dict(record, arm=case['candidate'])))
        if len(captured) != 8:
            raise ValueError('Require all eight original-filter oral captures')
        np.testing.assert_array_equal(sorted(r['seeds'][0] for _, r in captured), sorted(c['seeds']))
        oral_groups[case['case_id']] = own, captured, tag
    table = pd.DataFrame(scores)
    contrasts = []
    for case in cases:
        reference, = [r for r in cases if r['subject'] == case['subject'] and 'reference' in r['roles']]
        own, baseline = identities[case['case_id']], identities[reference['case_id']]
        np.testing.assert_array_equal(own['choices'], baseline['choices'])
        for role in ('conditional', 'autonomous'):
            for key in ('seeds', 'categories'):
                np.testing.assert_array_equal(own[role][key], baseline[role][key])
        for phase in ('training', 'test', 'whole'):
            rows = table[(table.subject == case['subject']) & (table.phase == phase)].set_index('candidate')
            row = dict(subject=case['subject'], cohort=case['cohort'], candidate=case['candidate'],
                selected='selected' in case['roles'], phase=phase, reference=reference['candidate'])
            for metric in ('choice_nll', 'energy', 'crps'):
                row['delta_'+metric] = rows.loc[case['candidate'], metric]-rows.loc[reference['candidate'], metric]
                row['delta_'+metric+'_mc_se'] = jackknife_se(knives[case['case_id']][phase][metric]
                                                        -knives[reference['case_id']][phase][metric])
            contrasts.append(row)
    output.mkdir(parents=True)
    for name, rows in [('scores', scores), ('contrasts', contrasts), ('curves', curves),
                       ('path_blocks', paths), ('autonomous_states', states), ('parameters', parameters)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'), index=False)
    for reading in ('midpoint', 'relative'):
        trial_frames, summary_rows, oral_jk = [], [], {}
        for cohort in sorted({c['cohort'] for c in cases}):
            audit = ROOT/policy['oral_audits'][cohort][reading]
            manifest = read(bind(audit/'manifest.json'))
            if manifest['status'] != 'complete':
                raise ValueError('Oral audit incomplete')
            for name, sha in manifest['input_sha256'].items():
                bind(ROOT/name, sha)
            for name in ('trials.csv', 'component_matches.csv'):
                bind(audit/name, manifest['artifacts'][name])
            oral, matches = pd.read_csv(audit/'trials.csv'), pd.read_csv(audit/'component_matches.csv')
            for case in [c for c in cases if c['cohort'] == cohort]:
                own, captured, tag = oral_groups[case['case_id']]
                trials, rows, jk = alignment.group_scores(own, captured, oral, matches)
                trial_frames.append(trials.assign(**tag))
                summary_rows.extend(dict(r, **{k: v for k, v in tag.items() if k not in r}) for r in rows)
                oral_jk[case['case_id']] = jk
        delta = []
        for case in cases:
            reference, = [r for r in cases if r['subject'] == case['subject'] and 'reference' in r['roles']]
            for key, a in oral_jk[case['case_id']].items():
                b = oral_jk[reference['case_id']][key]
                np.testing.assert_array_equal(a['seeds'], b['seeds'])
                row = dict(subject=case['subject'], cohort=case['cohort'], candidate=case['candidate'],
                    selected='selected' in case['roles'], reference=reference['candidate'],
                    window=key[0], domain=key[1], subset=key[2])
                for i, label in enumerate(('support', 'conflict', 'undetermined')):
                    row['delta_'+label] = a['point'][i]-b['point'][i]
                    row['delta_'+label+'_mc_se'] = jackknife_se(a['values'][:, i]-b['values'][:, i])
                delta.append(row)
        folder = output/('oral_'+reading)
        folder.mkdir()
        pd.concat(trial_frames).to_csv(folder/'trials.csv', index=False)
        pd.DataFrame(summary_rows).to_csv(folder/'summary.csv', index=False)
        pd.DataFrame(delta).to_csv(folder/'contrasts.csv', index=False)
    write(output/'completion.json', dict(status='complete', evaluation=str(evaluation.resolve()),
        inputs_sha256=inputs, participants=len(fitted_cases), candidates=len(cases),
        scope='Six development/extension participants; extension prior influence remains undetermined',
        oral_use=policy['oral_use'], conditional_state_use='Observed past choices, after-current-choice oral basis',
        sensitivity='Three prefix-selected points per subject; not a parameter confidence interval',
        numerical_uncertainty='Paired whole-filter or whole-path jackknife; not participant sampling uncertainty',
        artifacts_sha256={str(p.relative_to(output)): digest(p) for p in output.rglob('*.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.evaluation, args.config, args.output)
