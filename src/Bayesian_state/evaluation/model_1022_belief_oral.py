"""Score frozen belief-model decision bases against existing oral audits.

Use --capture COMPLETE_CAPTURE --config AUDIT_YAML --output NEW. Whole-report
denominators, functional/structural readings and seed-level pooling use the
shared alignment implementation. Oral data enter only this reporting step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from ..workflows.analysis import model_0923_decision_basis_alignment as alignment
from ..workflows.analysis.model_1022_belief_capture import assert_replay
from ..workflows.analysis.model_1022_belief_summary import read, digest, jackknife_se

ROOT = Path(__file__).resolve().parents[3]


def run(capture: Path, config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new belief oral summary directory')
    done = read(capture / 'completion.json')
    if done['status'] != 'complete' or done['smoke'] or not done['exact_array_replay']:
        raise ValueError('Require complete, exact, full-sequence frozen-filter captures')
    inputs = {}

    def bind(file, expected=None):
        path = Path(file).resolve()
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Oral evaluation input changed: ' + str(path))
        inputs[str(path)] = sha

    for file in (config, capture / 'protocol.json', capture / 'completion.json'):
        bind(file)
    bind(capture / 'cases.json', done['cases_sha256'])
    declared = read(capture / 'cases.json')
    for name, sha in declared['inputs_sha256'].items():
        bind(name, sha)
    groups = {}
    for case in declared['cases']:
        key = case['subject'], case['arm']
        if key in groups:
            raise ValueError('Duplicate participant/model oral case')
        records = []
        for path in case['captures']:
            file = Path(path)
            record = read(file)
            job = read(Path(record['original_job']))
            proof_path = Path(job['target']).with_suffix('.replay.json')
            proof = read(proof_path)
            if (proof['status'] != 'complete' or not proof['exact_array_replay'] or proof['oral_inputs_used']
                    or record['subject'] != case['subject'] or job['comparison_arm'] != case['arm']
                    or any(record['candidate'][k] != case['candidate'][k] for k in ('id', 'hyperparams'))):
                raise ValueError('Oral capture differs from its frozen participant or model')
            bind(proof_path)
            bind(proof['reference'], proof['reference_sha256'])
            replay = Path(job['target']).with_suffix('.npz')
            bind(replay, proof['replay_sha256'])
            assert_replay(Path(proof['reference']), replay, proof['reference_sha256'])
            alignment.validate_basis_record(file.with_suffix('.npz'), record)
            bind(file)
            bind(file.with_suffix('.npz'), record['array_sha256'])
            bind(record['original_job'], record['original_job_sha256'])
            records.append((file, dict(record, arm=case['arm'])))
        np.testing.assert_array_equal(sorted(r['seeds'][0] for _, r in records), sorted(case['seeds']))
        groups[key] = records, case['cohort']
    if any((sid, 'SUPPORT') not in groups for sid, _ in groups):
        raise ValueError('Every participant requires the matched SUPPORT oral comparison')
    policy = yaml.safe_load(config.read_text())
    output.mkdir(parents=True)
    for reading in ('midpoint', 'relative'):
        frames, summaries, estimates = [], [], {}
        for cohort in sorted({cohort for _, cohort in groups.values()}):
            audit = ROOT / policy['oral_audits'][cohort][reading]
            manifest = read(audit / 'manifest.json')
            if manifest['status'] != 'complete':
                raise ValueError('Oral audit is incomplete')
            bind(audit / 'manifest.json')
            for name, sha in manifest['input_sha256'].items():
                bind(ROOT / name, sha)
            for name in ('trials.csv', 'component_matches.csv'):
                bind(audit / name, manifest['artifacts'][name])
            oral, matches = pd.read_csv(audit / 'trials.csv'), pd.read_csv(audit / 'component_matches.csv')
            for key, (records, group_cohort) in sorted(groups.items()):
                if group_cohort != cohort:
                    continue
                trials, rows, jackknife = alignment.group_scores(capture, records, oral, matches)
                frames.append(trials.assign(cohort=cohort))
                summaries.extend(dict(row, cohort=cohort) for row in rows)
                estimates[key] = jackknife
        contrasts = []
        for (sid, arm), estimates_by_window in sorted(estimates.items()):
            if arm == 'SUPPORT':
                continue
            for key, current in estimates_by_window.items():
                baseline = estimates[sid, 'SUPPORT'][key]
                np.testing.assert_array_equal(current['seeds'], baseline['seeds'])
                row = dict(subject=sid, arm=arm, cohort=groups[sid, arm][1],
                           window=key[0], domain=key[1], subset=key[2])
                for i, label in enumerate(alignment.LABELS):
                    row[label + '_difference'] = current['point'][i] - baseline['point'][i]
                    row[label + '_difference_mcse'] = jackknife_se(current['values'][:, i] - baseline['values'][:, i])
                contrasts.append(row)
        folder = output / ('oral_' + reading)
        folder.mkdir()
        pd.concat(frames, ignore_index=True).to_csv(folder / 'trials.csv', index=False)
        pd.DataFrame(summaries).to_csv(folder / 'summary.csv', index=False)
        pd.DataFrame(contrasts).to_csv(folder / 'contrasts.csv', index=False)
        print(f'{reading}: {len(groups)} frozen participant/model cases scored', flush=True)
    source_files = [Path(__file__), Path(alignment.__file__),
        ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis.py',
        ROOT / 'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py',
        ROOT / 'src/Bayesian_state/workflows/analysis/model_1022_belief_capture.py',
        ROOT / 'src/Bayesian_state/workflows/analysis/model_1022_belief_summary.py']
    source_hashes = {}
    for source in source_files:
        relative = source.resolve().relative_to(ROOT)
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(relative)] = digest(source)
    with (output / 'completion.json').open('x') as stream:
        json.dump(dict(status='complete', capture=str(capture.resolve()), inputs_sha256=inputs,
            source_sha256=source_hashes,
            oral_used_for_fitting=False, cases=len(groups), denominator='unchanged complete reports',
            comparison='each belief arm minus SUPPORT; whole-filter paired Monte Carlo SE',
            scope='external content compatibility within development and extension cohorts'), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.capture, args.config, args.output)
