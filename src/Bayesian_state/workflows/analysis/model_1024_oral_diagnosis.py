"""Audit catalogue ceilings, stimulus truth, and unchanged oral projections.

Run --config YAML --output NEW_DIRECTORY. This is a saved-state analysis, with
all original transcripts and state arrays hashed. It writes every trial,
preassigned oral blocks, geometric ceilings, and the prespecified review sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/model1024_oral_development_mpl')
import numpy as np
import pandas as pd
import yaml

from ...evaluation.oral.development import oral_split, review_indices, token_truth
from ...evaluation.oral.partial_reason_alignment import (
    component_geometry, predicate_coverage, stimulus_choice_reference, unresolved_reason_role)
from .model_0923_decision_basis_alignment import align_identities
from .model_0923_oral_constraint_audit import catalogue
from .model_1024_individual_oral_alignment import subject_features, constraint_text
from ...model.modules.readout.report_space import component_tokens


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new diagnosis directory')
    start, cpu = time.time(), time.process_time()
    policy = yaml.safe_load(config.read_text())
    inputs = {str(config): digest(config)}

    def bind(path: Path) -> Path:
        inputs[str(path)] = digest(path)
        return path

    source = Path(policy['state_source'])
    selected = json.loads(bind(Path(policy['state_analysis'])/'selected.json').read_text())
    cases = json.loads(bind(source/'protocol.json').read_text())['cases']
    old_scores = pd.read_csv(bind(Path(policy['previous_preliminary'])/'candidate_trial_scores.csv.gz'))
    old_scores = old_scores[old_scores['mode'].eq('filtered')]
    output.mkdir(parents=True)
    (output/'registration.json').write_text(json.dumps(dict(started_unix=start, policy=policy,
        data_use='Saved behavior-derived states; frozen oral coding; no new state inference'), indent=2))
    results, aggregates, samples, candidates = [], [], [], []
    numeric_checks = 0
    for sid in policy['subjects']:
        own = [s for s in selected if s['subject'] == sid]
        case = next(c for c in cases if c['id'] == own[0]['case'])
        oral = pd.read_csv(bind(Path(policy['previous_oral'])/f'S{sid}_review.csv'))
        frame, _ = align_identities(oral, case['template'], case['trials'])
        frame = frame.sort_values('trial').reset_index(drop=True)
        raw = pd.read_csv(bind(Path(policy['raw_directory'])/f'Task2_{sid}_rec.csv'))
        check = frame[['iSession', 'iTrial', 'text']].merge(raw, on=['iSession', 'iTrial'],
            how='outer', validate='one_to_one', suffixes=('_frozen', ''), indicator=True)
        if not check['_merge'].eq('both').all() or not check.text.fillna('').eq(check.text_frozen.fillna('')).all():
            raise ValueError('Raw oral identity or text changed')
        features = subject_features(frame)
        folder = Path(case['template']['folder'])
        bind(folder/'prepared.json')
        parts, feedback = [], []
        for name in ('training.npz', 'sealed_test.npz'):
            with np.load(bind(folder/name)) as z:
                parts.append(z['stimulus'])
                feedback.append(z['feedback'])
        x, feedback = np.concatenate(parts), np.concatenate(feedback)
        choices = frame.choice.to_numpy(int)
        tokens = [tuple(json.loads(value)) for value in frame.tokens]
        role = frame.text.fillna('').apply(unresolved_reason_role).to_numpy()
        valid = np.array([bool(t) and scope == 'complete' and not bad and
            all(a.startswith('facet:') for a in t) for t, scope, bad in zip(tokens, frame.scope, role)])
        n = len(frame)
        splits = oral_split(n, policy['split']['block_trials'], policy['split']['boundary_buffer_trials'])
        space = catalogue(2 if case['condition'] == 1 else 4)
        with np.load(bind(Path(own[0]['file']))) as z:
            h, m = z['filtered_component'].shape[1:]
            counts = z['component_counts']
        geometries = {c: component_geometry(space, c)[0] for c in np.unique(choices)}
        references = np.full((n, h, m), np.nan)
        uniform = np.zeros((n, h, m))
        values = np.zeros((n, h, m, 4))
        whole_ceiling = np.full(n, np.nan)
        atomic_ceiling = np.full(n, np.nan)
        individually_possible = np.full(n, np.nan)
        all_true = np.full(n, np.nan)
        mean_true = np.full(n, np.nan)
        basis = {}
        for c, entries in geometries.items():
            idx = np.flatnonzero(choices == c)
            references[idx], available = stimulus_choice_reference(x[idx], entries, h, m)
            for hh, mm, A, b in entries:
                uniform[idx, hh, mm] = 1/h/counts[hh, c-1]
                basis[c, hh, mm] = constraint_text(A, b, features)
            cache = {}
            for t in idx:
                if time.time() > policy['budget']['deadline_unix'] - policy['budget']['report_reserve_seconds']:
                    raise TimeoutError('Task reached the registered report reserve')
                if not valid[t]:
                    continue
                if tokens[t] not in cache:
                    v = np.zeros((h, m, 4))
                    atom_any = np.zeros(len(tokens[t]), bool)
                    for hh, mm, A, b in entries:
                        v[hh, mm] = predicate_coverage(tokens[t], A, b)
                        atom_any |= np.array([predicate_coverage((a,), A, b)[0] > .5 for a in tokens[t]])
                    cache[tokens[t]] = (v, float(atom_any.mean()))
                values[t], individually_possible[t] = cache[tokens[t]]
                atomic_ceiling[t] = max(values[t, hh, mm, 0] for hh, mm, *_ in entries)
                whole_ceiling[t] = float(atomic_ceiling[t] > 1-1e-10)
                truth = token_truth(tokens[t], x[t])
                all_true[t], mean_true[t] = float(truth.all()), truth.mean()
        current = frame[['trial', 'iSession', 'iTrial', 'text', 'choice', 'scope', 'tokens', 'unparsed']].copy()
        current.insert(0, 'subject', sid)
        current['split'] = splits
        current['block64'] = np.arange(n)//64+1
        current['early_middle'] = np.arange(n) < n*policy['early_middle']['primary_end_fraction']
        current['feedback'] = feedback
        current['eligible'] = valid
        current['atomic_ceiling'] = atomic_ceiling
        current['whole_ceiling'] = whole_ceiling
        current['individually_possible'] = individually_possible
        current['physical_all_true'] = all_true
        current['physical_mean_true'] = mean_true
        current['has_superlative'] = frame.text.fillna('').str.contains('最长|最短')
        current['has_body_reference'] = frame.text.fillna('').str.contains('躯干|身体')
        current['decoded_constraints'] = frame.decoded_constraints
        current['stimulus_description'] = ['；'.join(f'{f}={v:.4f}' for f, v in zip(features, row)) for row in x]
        ref = np.einsum('thm,thmk->tk', np.nan_to_num(references), values)
        ref[~np.isfinite(references).all((1, 2))] = np.nan
        cat = np.einsum('thm,thmk->tk', uniform, values)
        current['stimulus_reference'] = np.where(valid, ref[:, 0], np.nan)
        current['catalogue_reference'] = np.where(valid, cat[:, 0], np.nan)
        for s in own:
            path = bind(Path(s['file']))
            if inputs[str(path)] != s['sha256']:
                raise ValueError('Selected state archive changed')
            with np.load(path) as z:
                mass = z['filtered_component']
            if not np.isfinite(mass).all() or (mass < 0).any():
                raise ValueError('Invalid state probability')
            np.testing.assert_allclose(mass.sum((1, 2)), 1, atol=2e-10, rtol=0)
            score = np.einsum('thm,thmk->tk', mass, values)
            previous = old_scores[old_scores.subject.eq(sid) & old_scores.number.eq(s['number'])].sort_values('trial')
            np.testing.assert_allclose(score[valid, :3], previous.loc[valid, [
                'predicate_supported', 'predicate_conflicting', 'predicate_unexplained']].to_numpy(), atol=3e-12, rtol=0)
            numeric_checks += int(valid.sum())*3
            current[f'P{s["number"]}_support'] = np.where(valid, score[:, 0], np.nan)
            if s['number'] == 1:
                current['P1_conflict'] = np.where(valid, score[:, 1], np.nan)
                descriptions = []
                for t in range(n):
                    ids = np.argsort(mass[t].ravel())[-3:][::-1]
                    descriptions.append('；'.join(f'R{hh}：{basis[choices[t], hh, mm]}（{mass[t, hh, mm]:.1%}）'
                        for hh, mm in [np.unravel_index(i, (h, m)) for i in ids] if mass[t, hh, mm] > 0))
                current['P1_rule_content'] = descriptions
            candidates.append(dict(subject=sid, number=s['number'], case=s['case'], particles=s['particles'],
                systems=s['systems'], parameter_status=s['parameter_status']))
        results.append(current)
        if sid in policy['focus_subjects']:
            sample = current.iloc[review_indices(n, policy['review']['focus_rows_per_subject'])].copy()
            sample['manual_intended_meaning'] = ''
            sample['manual_notes'] = ''
            samples.append(sample)
        groups = [('all', np.ones(n, bool)), ('early_middle', current.early_middle.to_numpy())]
        groups += [(f'block64_{i+1}', np.arange(n)//64 == i) for i in range((n+63)//64)]
        for label, mask in groups:
            for split in ['all', 'calibration', 'assessment']:
                use = mask & (True if split == 'all' else splits == split)
                eligible = use & valid
                row = dict(subject=sid, window=label, split=split, trials=int(use.sum()), reports=int(eligible.sum()),
                    superlative_reports=int(current.loc[use, 'has_superlative'].sum()),
                    body_reference_reports=int(current.loc[use, 'has_body_reference'].sum()))
                for col in ['atomic_ceiling', 'whole_ceiling', 'individually_possible', 'physical_all_true',
                            'physical_mean_true', 'stimulus_reference', 'catalogue_reference', 'P1_support', 'P1_conflict']:
                    row[col] = current.loc[eligible, col].mean()
                aggregates.append(row)
        # The fixed catalogue-derived atom vocabulary is independent of text.
        atom_vocab = sorted(set(a for rule in space for c in rule.categories for reg in c.components
                                for a in component_tokens(reg.A, reg.b)))
        (output/f'S{sid}_metadata.json').write_text(json.dumps(dict(subject=sid, features=features,
            catalogue_signature=space.signature, atoms=atom_vocab, selected=own,
            source_behavior=case['template']['folder']), ensure_ascii=False, indent=2))
        np.savez_compressed(output/f'S{sid}_diagnostic_arrays.npz', stimulus=x, choice=choices,
            eligible=valid, values=values, stimulus_reference_mass=references, catalogue_mass=uniform)
        print(f'S{sid}: {n} trials, {valid.sum()} primary reports; unchanged projections verified', flush=True)
    pd.concat(results, ignore_index=True).to_csv(output/'trials.csv', index=False)
    pd.DataFrame(aggregates).to_csv(output/'summary.csv', index=False)
    pd.concat(samples, ignore_index=True).to_csv(output/'review_sample.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(candidates).to_csv(output/'candidates.csv', index=False)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Input changed during diagnosis: '+path)
    result = dict(status='complete', subjects=policy['subjects'], trials=sum(len(f) for f in results),
        review_rows=sum(len(f) for f in samples), projection_checks=numeric_checks,
        wall_seconds=time.time()-start, cpu_seconds=time.process_time()-cpu, inputs_sha256=inputs,
        source_sha256={str(Path(__file__)): digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print({k:v for k,v in result.items() if k not in ('inputs_sha256', 'source_sha256', 'artifacts')})
