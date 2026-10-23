"""Compare a completed belief study with frozen SUPPORT references.

Use --study COMPLETE --reference OLD [--reference ANOTHER] --output NEW.
Cross-study seed banks must be disjoint. Numerical errors combine independent
whole-filter/path estimates; model parameters and participant uncertainty are
not included. Original labels, trial boundaries and all paths are retained.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..workflows.analysis.model_1022_belief_summary import read, digest, load_case, nll, jackknife_se
from ..metrics.autonomous_fit import behavior_features, energy_u

ROOT = Path(__file__).resolve().parents[3]


def independent_difference(left: float, left_se: float, right: float, right_se: float) -> tuple[float, float]:
    """Right minus left, with independent numerical variance contributions."""
    return right-left, float(np.hypot(left_se, right_se))


def validate_comparison(reference: dict, current: dict) -> None:
    for key in ('subject', 'condition', 'trials', 'training_trials', 'test_trials'):
        if reference['meta'][key] != current['meta'][key]:
            raise ValueError('Reference participant or trial boundary differs')
    for key in reference['observed']:
        np.testing.assert_array_equal(reference['observed'][key], current['observed'][key])
    for kind in ('prediction', 'autonomous'):
        if set(reference[kind]['seeds']).intersection(current[kind]['seeds']):
            raise ValueError('Independent reference comparison requires disjoint seed banks')


def scores(case: dict) -> dict:
    cut = case['meta']['training_trials']
    p = case['prediction']['observed_probability'][:, cut:]
    categories = case['observed']['categories'][cut:]
    count = 2 if case['meta']['condition'] == 1 else 4
    observed = behavior_features(case['observed']['choices'][None, cut:], categories, n_categories=count)[0]
    paths = behavior_features(case['autonomous']['choices'][:, cut:], categories, n_categories=count)
    energy = energy_u(paths, observed)
    return dict(nll=nll(p), nll_mcse=jackknife_se(np.array([nll(np.delete(p, i, 0)) for i in range(len(p))])),
                energy=energy['value'], energy_mcse=energy['mc_se'], filters=len(p), paths=len(paths))


def run(study: Path, references: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new fixed-reference report directory')
    resolved = [p.resolve() for p in references]
    if not resolved or len(set(resolved)) != len(resolved) or study.resolve() in resolved:
        raise ValueError('Require distinct historical reference studies')
    inputs, cached = {}, {}

    def bind(file):
        file = Path(file).resolve()
        inputs[str(file)] = digest(file)

    def cases(folder, support_only):
        done, nomination = read(folder / 'completion.json'), read(folder / 'nominees.json')
        if (done['status'] != 'complete' or done['smoke'] or nomination['smoke']
                or nomination['status'] != 'frozen_before_test' or nomination['later_choices_used']):
            raise ValueError('Require completed studies with frozen parameters')
        summary = read(folder / 'summary' / 'completion.json')
        if (summary['nominee_sha256'] != digest(folder / 'nominees.json')
                or done['summary_sha256'] != digest(folder / 'summary' / 'summary.csv')):
            raise ValueError('Original frozen nomination or completed summary changed')
        jobs = read(folder / 'autonomous_jobs.json')['jobs']
        for name in ('protocol.json', 'completion.json', 'nominees.json', 'autonomous_jobs.json'):
            bind(folder / name)
        for name in ('completion.json', 'summary.csv'):
            bind(folder / 'summary' / name)
        rows = [r for r in nomination['nominees'] if not support_only or r['arm'] == 'SUPPORT']
        result = {}
        for row in rows:
            key = row['subject'], row['arm']
            if key in result:
                raise ValueError('Duplicate participant/model case')
            case = load_case(folder, row, jobs)
            pred = folder / 'evaluate' / f'subject_{key[0]}' / key[1] / row['candidate']['id']
            case['particles'] = read(pred.with_suffix('.json'))['particles']
            case['candidate'] = row['candidate']['id']
            targets = [pred] + [Path(j['target']) for j in jobs if (j['subject'], j['model']) == key]
            for target in targets:
                bind(target.with_suffix('.npz'))
                bind(target.with_suffix('.json'))
            data = Path(next(j['folder'] for j in jobs if (j['subject'], j['model']) == key))
            for name in ('prepared.json', 'training.npz', 'sealed_test.npz'):
                bind(data / name)
            result[key] = case
            cached[folder.resolve(), key] = scores(case)
        return result

    current = cases(study, False)
    rows = []
    for folder in references:
        baseline = cases(folder, True)
        for key, case in current.items():
            old_key = key[0], 'SUPPORT'
            if old_key not in baseline:
                raise ValueError('Historical SUPPORT reference is missing a participant')
            reference = baseline[old_key]
            validate_comparison(reference, case)
            a, b = cached[folder.resolve(), old_key], cached[study.resolve(), key]
            dn, sn = independent_difference(a['nll'], a['nll_mcse'], b['nll'], b['nll_mcse'])
            de, se = independent_difference(a['energy'], a['energy_mcse'], b['energy'], b['energy_mcse'])
            rows.append(dict(subject=key[0], arm=key[1], reference=folder.name,
                candidate=case['candidate'], reference_candidate=reference['candidate'],
                particles=case['particles'], reference_particles=reference['particles'],
                filters=b['filters'], reference_filters=a['filters'], paths=b['paths'], reference_paths=a['paths'],
                test_nll=b['nll'], reference_test_nll=a['nll'], test_nll_difference=dn, test_nll_difference_mcse=sn,
                late_energy=b['energy'], reference_late_energy=a['energy'],
                late_energy_difference=de, late_energy_difference_mcse=se,
                nll_difference_plus_two_mcse=dn+2*sn, energy_difference_plus_two_mcse=de+2*se))
    output.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'comparisons.csv', index=False)
    source_files = [Path(__file__).resolve(), ROOT / 'src/Bayesian_state/workflows/analysis/model_1022_belief_summary.py',
                    ROOT / 'src/Bayesian_state/metrics/autonomous_fit.py']
    hashes = {}
    for source in source_files:
        relative = source.relative_to(ROOT)
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = digest(source)
    for file, sha in inputs.items():
        if digest(Path(file)) != sha:
            raise ValueError('Frozen prediction changed during reference comparison')
    from ..workflows.analysis.model_1022_belief_comparison import write
    write(output / 'completion.json', dict(status='complete', study=str(study.resolve()),
        references=[str(p) for p in resolved], comparisons=len(rows), inputs_sha256=inputs, source_sha256=hashes,
        uncertainty='independent whole-filter or whole-path Monte Carlo errors; fixed parameters',
        parameter_selection_changed=False, comparisons_sha256=digest(output / 'comparisons.csv')))
    print(table.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reference', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.reference, args.output)
