"""Cross frozen SUPPORT/BELIEF parameters with both mechanisms.

Run --config YAML --output NEW [--smoke]. Every original training nominee is
evaluated under both mechanisms with matched filter and autonomous seeds.
The four cells separate parameter changes from mechanism changes. No fitting
or mechanism selection occurs here; all later observations are development
data. Cognitive code and audited prediction loading are shared with the pilot.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from . import model_1022_belief_comparison as pilot
from .model_1022_belief_summary import load_case, nll, jackknife_se
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...utils.seeding import stable_seed

ROOT, read, write, digest = pilot.ROOT, pilot.read, pilot.write, pilot.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
CELLS = {f'{mechanism}_AT_{origin}': (mechanism, origin)
         for mechanism in ('SUPPORT', 'BELIEF') for origin in ('SUPPORT', 'BELIEF')}


def sources(v) -> dict:
    return dict(pilot.sources(v), **{ENTRY: digest(ROOT / ENTRY)})


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    numerical = deepcopy(jobs)
    for job in numerical:
        job['diagnosis_cell'] = job['model']
        job['model'] = CELLS[job['model']][0]
    pilot.execute(numerical, output, budget)


def summarize(output: Path) -> None:
    """Use complete paired repeats for numerical uncertainty and factorial effects."""
    nomination = read(output / 'nominees.json')
    if nomination['smoke'] or nomination['status'] != 'frozen_before_test':
        raise ValueError('Scientific summary requires frozen full-sequence cells')
    jobs = read(output / 'autonomous_jobs.json')['jobs']
    rows, cases, features = [], {}, {}
    for nominee in nomination['nominees']:
        sid, arm = nominee['subject'], nominee['arm']
        case = cases[sid, arm] = load_case(output, nominee, jobs)
        meta, obs, pred, auto = [case[k] for k in ('meta', 'observed', 'prediction', 'autonomous')]
        cut = meta['training_trials']
        later = slice(cut, None)
        c = 2 if meta['condition'] == 1 else 4
        f = behavior_features(auto['choices'][:, later], obs['categories'][later], n_categories=c)
        empirical = behavior_features(obs['choices'][None, later], obs['categories'][later], n_categories=c)[0]
        features[sid, arm] = f, empirical
        score = energy_u(f, empirical)
        present = auto['present'][:, later].astype(bool)
        p = pred['observed_probability'][:, later]
        rows.append(dict(subject=sid, cell=arm, mechanism=CELLS[arm][0], parameter_source=CELLS[arm][1],
            candidate=nominee['candidate']['id'], condition=meta['condition'],
            training_nll=nll(pred['observed_probability'][:, :cut]), later_nll=nll(p),
            later_nll_mcse=jackknife_se(np.array([nll(np.delete(p, i, 0)) for i in range(len(p))])),
            observed_late_accuracy=float((obs['choices'][later] == obs['categories'][later]).mean()),
            generated_late_accuracy=float((auto['choices'][:, later] == obs['categories'][None, later]).mean()),
            energy=score['value'], energy_mcse=score['mc_se'], paths=len(f),
            target_present=float(present.mean()), target_contribution=float(auto['contribution'][:, later].mean()),
            target_beta_when_present=float(auto['beta'][:, later][present].mean()) if present.any() else None,
            search_rate=float(auto['search'][:, later].mean()),
            target_drops_per_path=float(auto['target_capacity_dropped'][:, later].sum(1).mean())))
    contrasts = []
    for sid in sorted({s for s, _ in cases}):
        pairs = [(f'mechanism_at_{origin}', f'SUPPORT_AT_{origin}', f'BELIEF_AT_{origin}')
                 for origin in ('SUPPORT', 'BELIEF')]
        pairs += [(f'parameters_under_{mechanism}', f'{mechanism}_AT_SUPPORT', f'{mechanism}_AT_BELIEF')
                  for mechanism in ('SUPPORT', 'BELIEF')]
        for name, left, right in pairs:
            a, b = cases[sid, left], cases[sid, right]
            np.testing.assert_array_equal(a['prediction']['seeds'], b['prediction']['seeds'])
            np.testing.assert_array_equal(a['autonomous']['seeds'], b['autonomous']['seeds'])
            cut = a['meta']['training_trials']
            p, q = [case['prediction']['observed_probability'][:, cut:] for case in (a, b)]
            df = np.array([nll(np.delete(q, i, 0)) - nll(np.delete(p, i, 0)) for i in range(len(p))])
            fa, empirical = features[sid, left]
            fb, _ = features[sid, right]
            de = np.array([energy_u(np.delete(fb, i, 0), empirical)['value']
                           - energy_u(np.delete(fa, i, 0), empirical)['value'] for i in range(len(fa))])
            contrasts.append(dict(subject=sid, contrast=name, left=left, right=right,
                later_nll_difference=nll(q)-nll(p), later_nll_difference_mcse=jackknife_se(df),
                energy_difference=energy_u(fb, empirical)['value']-energy_u(fa, empirical)['value'],
                energy_difference_mcse=jackknife_se(de)))
    folder = output / 'summary'
    folder.mkdir()
    table = pd.DataFrame(rows)
    table.to_csv(folder / 'summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(folder / 'contrasts.csv', index=False)
    write(output / 'completion.json', dict(status='complete', smoke=False, cases=len(rows),
        nomination_sha256=digest(output / 'nominees.json'), summary_sha256=digest(folder / 'summary.csv'),
        source_sha256=sources(pilot.driver.io), scope='fixed-parameter development diagnosis; no refit',
        uncertainty='Monte Carlo uncertainty across complete filter or autonomous repeats'))
    print(table.to_string(index=False), flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new cross-diagnosis directory')
    policy = yaml.safe_load(config.read_text())
    if policy['scope'] != 'fixed_parameter_development_diagnosis':
        raise ValueError('Require the fixed-parameter diagnosis policy')
    source = ROOT / policy['source_study']
    prior = read(source / 'nominees.json')
    done = read(source / 'completion.json')
    if prior['smoke'] or prior['status'] != 'frozen_before_test' or done['status'] != 'complete':
        raise ValueError('Require completed scientific pilot nominations')
    settings = deepcopy(read(source / 'protocol.json')['settings'])
    precision = deepcopy(settings['arms']['SUPPORT'])
    settings.update({k: deepcopy(policy[k]) for k in ('protocol', 'seed', 'subjects', 'parallel_budget',
                                                     'budgets', 'autonomous_paths', 'autonomous_chunk')})
    settings['arms'] = {cell: deepcopy(precision) for cell in CELLS}
    if smoke:
        settings['subjects'] = settings['subjects'][:1]
        settings['parallel_budget'] = 1
        settings['budgets']['screen'] = settings['budgets']['evaluate']
    nominees = []
    for subject in settings['subjects']:
        old = {r['arm']: r['candidate'] for r in prior['nominees'] if r['subject'] == subject}
        for cell, (mechanism, origin) in CELLS.items():
            nominees.append(dict(subject=subject, arm=cell, mechanism=mechanism,
                                 parameter_source=origin, candidate=deepcopy(old[origin])))
    output.mkdir(parents=True)
    pilot.driver.base.source_hashes, pilot.driver.base.execute = sources, execute
    pilot.driver.base.archive(output, config, sources(pilot.driver.io))
    write(output / 'protocol.json', dict(settings=settings, policy=policy, smoke=smoke,
        source_sha256=sources(pilot.driver.io), source_nomination_sha256=digest(source / 'nominees.json'),
        python=sys.version, versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        decision='All four crossed cells fixed before running; no parameter or mechanism selection'))
    write(output / 'nominees.json', dict(status='frozen_before_test', nominees=nominees, smoke=smoke,
                                       later_choices_used=False, oral_used=False))
    pilot.driver.base.stage(settings, ROOT / settings['data'], output,
        {(r['subject'], r['arm']): [r['candidate']] for r in nominees},
        'screen' if smoke else 'evaluate', smoke)
    preparation = read(ROOT / settings['data'] / 'preparation_manifest.json')
    runtime = str((ROOT / preparation['settings']['source_fit'] / 'source').resolve())
    jobs = []
    for row in nominees:
        subject, cell = row['subject'], row['arm']
        seeds = [stable_seed(dict(base=policy['seed'], subject=subject, stage='autonomous', repeat=i))
                 for i in range(2 if smoke else policy['autonomous_paths'])]
        audit = ROOT / settings['audit'] / f'audit/subject_{subject}/audit.json'
        width = policy['autonomous_chunk']
        for i in range(0, len(seeds), width):
            name = f'autonomous_S{subject}_{cell}_{i:03d}'
            jobs.append(dict(phase='autonomous', name=name, subject=subject, model=cell,
                candidate=row['candidate'], mechanism=dict(kind='E', decay=.95, capacity='evidence', search='accumulate'),
                folder=str((ROOT / settings['data'] / f'subject_{subject}').resolve()), runtime=runtime,
                arm_settings=precision, seeds=seeds[i:i+width], smoke=smoke, source_sha256=sources(pilot.driver.io),
                audit=str(audit), audit_sha256=digest(audit), target=str((output / 'autonomous' / name).resolve())))
    write(output / 'autonomous_jobs.json', dict(jobs=jobs))
    execute(jobs, output, settings['parallel_budget'])
    if smoke:
        write(output / 'completion.json', dict(status='complete', smoke=True,
              scope='eight-trial filters and sixteen-trial paths; scientific diagnosis pending'))
    else:
        summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.smoke)
