"""Compare belief results with the originally frozen Model 1022 support model.

Use --study COMPLETED_BELIEF --reference-evaluation COMPACT_EVALUATION --output NEW.
The historical TRANSFER variant supplies the support model with inherited
precision. Saved choices and probabilities are rescored using the belief
comparison's NLL and autonomous Energy definitions. Every path, original trial
boundary and complete filter is retained. Independent Monte Carlo errors are
combined across studies; this command never fits parameters or simulates paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import pandas as pd

from .model_0923_h_validation import checked_arrays
from .model_1022_belief_fixed_references import independent_difference, scores, validate_comparison
from ..workflows.analysis.model_1022_belief_summary import digest, load_case, read

ROOT = Path(__file__).resolve().parents[3]


def validate_arrays(meta: dict, observed: dict, prediction: dict, autonomous: dict,
                    filters: int, paths: int) -> None:
    """Require complete draws, normalized probabilities and original trial labels."""
    count, length = 2 if meta['condition'] == 1 else 4, meta['trials']
    for bank, repeats in ((prediction, filters), (autonomous, paths)):
        p = bank['probabilities']
        if (p.shape != (repeats, length, count) or not np.isfinite(p).all()
                or np.any((p < 0) | (p > 1)) or len(bank['seeds']) != repeats
                or len(np.unique(bank['seeds'])) != repeats):
            raise ValueError('Invalid probability matrix or incomplete independent draws')
        np.testing.assert_allclose(p.sum(axis=-1), 1., rtol=0, atol=1e-12)
        np.testing.assert_array_equal(bank['categories'], observed['categories'])
    np.testing.assert_array_equal(prediction['choices'], observed['choices'])
    index = np.arange(length)
    for field, labels in (('observed_probability', observed['choices']),
                          ('correct_probability', observed['categories'])):
        if prediction[field].shape != (filters, length):
            raise ValueError('Missing full-sequence filter probabilities')
        np.testing.assert_allclose(prediction[field], prediction['probabilities'][:, index, labels-1],
                                   rtol=0, atol=1e-12)
    if autonomous['choices'].shape != (paths, length):
        raise ValueError('Autonomous path count or trial boundary differs')
    if not np.isin(autonomous['choices'], np.arange(1, count+1)).all():
        raise ValueError('Autonomous choices are outside the category support')
    np.testing.assert_array_equal(autonomous['feedback'] == 1,
        autonomous['choices'] == observed['categories'][None, :])


def load_support(evaluation: Path, subject: int, bind: Callable) -> dict:
    """Load the historical support variant and verify its frozen identities."""
    protocol, done = read(evaluation/'protocol.json'), read(evaluation/'completion.json')
    study = ROOT / protocol['study']
    bind(evaluation/'protocol.json'); bind(evaluation/'completion.json')
    bind(study/'cases.json', protocol['cases_sha256'])
    bind(study/'human_nominees.json', protocol['frozen_human_sha256'])
    if done['status'] != 'complete' or done['smoke'] or protocol['smoke']:
        raise ValueError('Require a completed scientific support evaluation')
    cases = [c for c in read(study/'cases.json')['cases'] if c['kind'] == 'human']
    if {c['case_id'] for c in cases} != {c['case_id'] for c in done['cases']}:
        raise ValueError('Historical evaluation differs from its registered case set')
    selected = [c for c in cases if c['subject'] == subject and c['variant'] == 'TRANSFER']
    if len(selected) != 1:
        raise ValueError('Historical support must contain exactly one matching participant')
    case, = selected
    fit, own = Path(case['output']), evaluation/case['case_id']
    frozen, fit_done = read(fit/'nominees.json'), read(fit/'completion.json')
    own_protocol, complete = read(own/'protocol.json'), read(own/'completion.json')
    if (frozen['smoke'] or frozen['status'] != 'frozen_before_test' or fit_done['status'] != 'complete'
            or fit_done['test_choices_used'] or complete['status'] != 'complete' or complete['smoke']
            or own_protocol['smoke'] or own_protocol['case_id'] != case['case_id']
            or own_protocol['cohort'] != case['cohort']
            or any(x['compact_variant'] != 'TRANSFER' for x in (own_protocol, complete))):
        raise ValueError('Historical support is not the declared frozen scientific variant')
    nominee, = frozen['nominees']
    if nominee['subject'] != subject or nominee['compact_variant'] != 'TRANSFER':
        raise ValueError('Historical nominee has a different participant or mechanism')
    for record in (fit_done, own_protocol, complete):
        bind(fit/'nominees.json', record['nomination_sha256'])
    for file in (fit/'protocol.json', fit/'completion.json', own/'protocol.json', own/'completion.json'):
        bind(file)
    for name, sha in frozen['source_sha256'].items():
        bind(study/'source'/name, sha)
    settings = frozen['settings']
    folder = Path(frozen['data'])/f'subject_{subject}'
    meta = read(folder/'prepared.json'); bind(folder/'prepared.json')
    if meta['subject'] != subject or meta['condition'] != case['condition']:
        raise ValueError('Historical prepared data identity differs')
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        bind(folder/name, meta['array_sha256'][name])
        with np.load(folder/name) as z:
            parts.append({k: z[k] for k in z.files})
    observed = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    banks, records = {}, {}
    for role in ('conditional', 'autonomous'):
        file = own/role/'evaluation.json'; bind(file)
        saved = read(file)
        record, = saved['rows']
        if (saved['status'] != 'complete' or record['subject'] != subject
                or record['condition'] != meta['condition'] or record['model'] != 'COMPACT'
                or record['training_trials'] != meta['training_trials'] or record['trials'] != meta['trials']
                or any(record['candidate'][k] != nominee['candidate'][k] for k in ('id', 'hyperparams'))):
            raise ValueError('Historical predictions differ from their frozen identity or boundary')
        array = own/role/'evaluate'/f'subject_{subject}'/'COMPACT'/(record['candidate']['id']+'.npz')
        bind(array, record['predictions_sha256'])
        banks[role], records[role] = checked_arrays(own/role, record), record
        phase = 'evaluate' if role == 'conditional' else 'autonomous'
        seeds = [seed for job in own_protocol['jobs'] if job['phase'] == phase for seed in job['seeds']]
        np.testing.assert_array_equal(banks[role]['seeds'], seeds)
    budget = settings['budgets']['evaluate']
    if records['conditional']['particles'] != budget['particles']:
        raise ValueError('Historical filter particle budget changed')
    validate_arrays(meta, observed, banks['conditional'], banks['autonomous'],
                    budget['seeds'], settings['autonomous_paths'])
    return dict(meta=meta, observed=observed, prediction=banks['conditional'], autonomous=banks['autonomous'],
                candidate=nominee['candidate']['id'], particles=budget['particles'], cohort=case['cohort'])


def run(study: Path, reference: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new frozen-support comparison directory')
    inputs = {}

    def bind(file: Path, expected: str | None = None):
        path = file.resolve(); sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Frozen comparison input changed: '+str(path))
        inputs[str(path)] = sha

    done, nomination = read(study/'completion.json'), read(study/'nominees.json')
    summary = read(study/'summary/completion.json')
    if (done['status'] != 'complete' or done['smoke'] or nomination['smoke']
            or nomination['status'] != 'frozen_before_test' or nomination['later_choices_used']
            or nomination['oral_used']):
        raise ValueError('Require completed scientific belief fits and frozen predictions')
    bind(study/'nominees.json', summary['nominee_sha256'])
    bind(study/'summary/summary.csv', done['summary_sha256'])
    for name in ('protocol.json', 'completion.json', 'autonomous_jobs.json', 'summary/completion.json'):
        bind(study/name)
    jobs = read(study/'autonomous_jobs.json')['jobs']
    originals, original_scores, rows, seen = {}, {}, [], set()
    for row in nomination['nominees']:
        sid, arm = row['subject'], row['arm']
        if (sid, arm) in seen:
            raise ValueError('Duplicate current participant/model case')
        seen.add((sid, arm))
        if sid not in originals:
            originals[sid] = load_support(reference, sid, bind)
            original_scores[sid] = scores(originals[sid])
        current = load_case(study, row, jobs)
        validate_comparison(originals[sid], current)
        a, b = original_scores[sid], scores(current)
        pred = study/'evaluate'/f'subject_{sid}'/arm/row['candidate']['id']
        for target in [pred] + [Path(j['target']) for j in jobs if (j['subject'], j['model']) == (sid, arm)]:
            bind(target.with_suffix('.json')); bind(target.with_suffix('.npz'))
        folder = Path(next(j['folder'] for j in jobs if (j['subject'], j['model']) == (sid, arm)))
        for name in ('prepared.json', 'training.npz', 'sealed_test.npz'):
            bind(folder/name)
        dn, sn = independent_difference(a['nll'], a['nll_mcse'], b['nll'], b['nll_mcse'])
        de, se = independent_difference(a['energy'], a['energy_mcse'], b['energy'], b['energy_mcse'])
        rows.append(dict(subject=sid, arm=arm, cohort=originals[sid]['cohort'],
            reference='frozen_model1022_support', candidate=row['candidate']['id'],
            reference_candidate=originals[sid]['candidate'], particles=read(pred.with_suffix('.json'))['particles'],
            reference_particles=originals[sid]['particles'], filters=b['filters'], reference_filters=a['filters'],
            paths=b['paths'], reference_paths=a['paths'], test_nll=b['nll'], reference_test_nll=a['nll'],
            test_nll_difference=dn, test_nll_difference_mcse=sn, late_energy=b['energy'],
            reference_late_energy=a['energy'], late_energy_difference=de, late_energy_difference_mcse=se,
            nll_difference_plus_two_mcse=dn+2*sn, energy_difference_plus_two_mcse=de+2*se))
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Comparison input changed during scoring')
    output.mkdir(parents=True)
    table = pd.DataFrame(rows); table.to_csv(output/'comparisons.csv', index=False)
    source_hashes = {}
    for file in (Path(__file__), Path(checked_arrays.__code__.co_filename), Path(scores.__code__.co_filename),
                 ROOT/'src/Bayesian_state/workflows/analysis/model_1022_belief_summary.py',
                 ROOT/'src/Bayesian_state/metrics/autonomous_fit.py'):
        relative = file.resolve().relative_to(ROOT); target = output/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file, target)
        source_hashes[str(relative)] = digest(file)
    from ..workflows.analysis.model_1022_belief_comparison import write
    write(output/'completion.json', dict(status='complete', comparisons=len(rows),
        study=str(study.resolve()), reference_evaluation=str(reference.resolve()), inputs_sha256=inputs,
        source_sha256=source_hashes, comparisons_sha256=digest(output/'comparisons.csv'),
        metrics='NLL and complete-path U-statistic Energy with the common belief-comparison feature definition',
        uncertainty='independent whole-filter or whole-path Monte Carlo SE; fixed parameters',
        parameter_selection_changed=False))
    print(table.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reference-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.reference_evaluation, args.output)
