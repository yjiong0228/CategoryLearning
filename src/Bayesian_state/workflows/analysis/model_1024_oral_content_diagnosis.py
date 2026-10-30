"""Separate shared task content from evidence about alternative rules.

Run --config ANCHOR_CONFIG --output NEW_DIRECTORY. Keep all six participants,
all eligible atomic judgments, correct-rule contrasts, and all rank probes.
Rank additions are catalogue diagnostics and are not fitted Model 1024 states.
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

from ...evaluation.oral.partial_reason_alignment import component_geometry, predicate_coverage
from ...evaluation.oral.rank_catalogue import rank_hypotheses
from ...evaluation.oral.decision_basis import nearest_component_weights
from ...evaluation.oral.development import token_truth
from ...hypothesis_space.geometry.boundary import BoundaryGeometry
from .model_0923_oral_constraint_audit import catalogue
from .model_1024_oral_readout import atom_vocabulary, observed_atoms, mixture_readout
from .model_1024_individual_oral_alignment import constraint_text


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new content diagnosis output')
    start, cpu = time.time(), time.process_time()
    p = yaml.safe_load(config.read_text())
    inputs = {str(config):digest(config)}

    def bind(path):
        path = Path(path); inputs[str(path)] = digest(path); return path

    folder = Path(p['diagnosis'])
    trials = pd.read_csv(bind(folder/'trials.csv'))
    source = json.loads(bind(Path(p['state_source'])/'protocol.json').read_text())
    selected = json.loads(bind(Path(p['state_analysis'])/'selected.json').read_text())
    settings = pd.read_csv(bind(Path(p['readout'])/'selected.csv'))
    output.mkdir(parents=True)
    atom_rows, trial_rows, summary_rows, rank_rows, choice_rows, contrast_rows = [], [], [], [], [], []
    ranks = rank_hypotheses()
    vocab = atom_vocabulary()
    for sid, frame in trials.groupby('subject', sort=True):
        frame = frame.reset_index(drop=True)
        own = next(s for s in selected if s['subject'] == sid and s['number'] == 1)
        case = next(c for c in source['cases'] if c['id'] == own['case'])
        metadata = json.loads(bind(folder/f'S{sid}_metadata.json').read_text())
        with np.load(bind(own['file'])) as z:
            mass = z['filtered_component']
        with np.load(bind(folder/f'S{sid}_diagnostic_arrays.npz')) as z:
            x, choices = z['stimulus'], z['choice']
        n, h, m = mass.shape
        target = case['target_rules']
        space = catalogue(2 if h == 29 else 4)
        geometry = BoundaryGeometry(space)
        dominant_wrong = int(np.argmax(np.where(np.isin(np.arange(h), target), -1,
            mass[:n//2].sum(-1).mean(0))))
        tokens = [tuple(json.loads(a)) for a in frame.tokens]
        cache = {}
        rank_cache = {}
        scores = {}
        correct_mass, fixed_wrong_mass = np.zeros_like(mass), np.zeros_like(mass)
        for choice in np.unique(choices):
            entries = component_geometry(space, int(choice))[0]
            indices = np.flatnonzero(choices == choice)
            semantic = np.zeros((len(entries), len(vocab)))
            for j, (hh, mm, A, b) in enumerate(entries):
                for k, atom in enumerate(vocab[:-1]):
                    verdict = predicate_coverage((atom,), A, b)
                    semantic[j, k] = verdict[0]-verdict[1]
            scores[choice] = (entries, semantic)
            for rule in target:
                allocation = nearest_component_weights(space, geometry, int(rule), int(choice)-1, x[indices])
                correct_mass[indices, rule, :allocation.shape[1]] = allocation/len(target)
            allocation = nearest_component_weights(space, geometry, dominant_wrong, int(choice)-1, x[indices])
            fixed_wrong_mass[indices, dominant_wrong, :allocation.shape[1]] = allocation
        for t in np.flatnonzero(frame.eligible.to_numpy()):
            entries = scores[choices[t]][0]
            different, incompatible, other_total, correct_total = [], [], [], []
            for atom in tokens[t]:
                key = (int(choices[t]), atom)
                if key not in cache:
                    support, conflict = np.zeros((h, m)), np.zeros((h, m))
                    for hh, mm, A, b in entries:
                        verdict = predicate_coverage((atom,), A, b)
                        support[hh, mm], conflict[hh, mm] = verdict[:2]
                    target_coords = [(hh, mm) for hh, mm, *_ in entries if hh in target]
                    cache[key] = (support, conflict, any(support[hh, mm] for hh, mm in target_coords),
                        all(conflict[hh, mm] for hh, mm in target_coords))
                support, conflict, correct_can_support, correct_all_conflict = cache[key]
                contribution = mass[t]*support
                on_correct = float(contribution[target].sum())
                on_other = float(contribution.sum()-on_correct)
                wrong_specific = on_other if not correct_can_support else 0.
                contradicts = on_other if correct_all_conflict else 0.
                atom_rows.append(dict(subject=int(sid), trial=t+1, text=frame.text.iloc[t], atom=atom,
                    split=frame.split.iloc[t], early_middle=bool(frame.early_middle.iloc[t]),
                    correct_region_can_support=correct_can_support, correct_region_all_conflict=correct_all_conflict,
                    support_from_correct=on_correct, support_from_other=on_other,
                    alternative_specific_support=wrong_specific, correct_conflicting_alternative_support=contradicts))
                different.append(wrong_specific); incompatible.append(contradicts)
                other_total.append(on_other); correct_total.append(on_correct)
            row = dict(subject=int(sid), trial=t+1, split=frame.split.iloc[t], early_middle=bool(frame.early_middle.iloc[t]),
                block64=int(frame.block64.iloc[t]), text=frame.text.iloc[t], feedback=float(frame.feedback.iloc[t]),
                correct_rule_mass=float(mass[t, target].sum()), dominant_wrong_rule=dominant_wrong,
                support_from_correct=np.mean(correct_total), support_from_other=np.mean(other_total),
                alternative_specific_support=np.mean(different), correct_conflicting_alternative_support=np.mean(incompatible))
            trial_rows.append(row)
            if int(sid) in p['rank_probe']['subjects']:
                key = (int(choices[t]), tokens[t])
                if key not in rank_cache:
                    values = [predicate_coverage(tokens[t], r.categories[choices[t]-1].components[0].A,
                        r.categories[choices[t]-1].components[0].b)[0] for r in ranks]
                    winner = int(np.argmax(values))
                    rank_cache[key] = (values[winner], ranks[winner].index, ranks[winner].family)
                maximum, rule, family = rank_cache[key]
                rank_rows.append(dict(subject=int(sid), trial=t+1, split=frame.split.iloc[t], early_middle=bool(frame.early_middle.iloc[t]),
                    text=frame.text.iloc[t], original_ceiling=float(frame.atomic_ceiling.iloc[t]),
                    added_rank_ceiling=float(maximum), combined_ceiling=max(float(frame.atomic_ceiling.iloc[t]), maximum),
                    oracle_rank_rule=rule, oracle_rank_family=family,
                    role='catalogue_ceiling_only_not_model_probability'))
        # Same frozen readout settings: compare the dynamic state to the known
        # task rule and one fixed alternative selected from behavior alone.
        setting = settings[settings.subject.eq(sid) & settings.selection_window.eq('early_middle') &
            settings.branch.eq('semantic_P1')].iloc[0]
        frequency = np.array(json.loads(bind(Path(p['readout'])/f'S{sid}_early_middle_frequency.json').read_text())['frequency'])
        truth = np.array([np.r_[token_truth(vocab[:-1], row), 0.] for row in x])
        background = frequency[None, :]*np.exp(float(setting.stimulus_temperature)*truth)
        background /= background.sum(1, keepdims=True)
        observed = observed_atoms(tokens, vocab)
        for name, distribution in [('correct_rule', correct_mass), ('fixed_alternative_rule', fixed_wrong_mass)]:
            prediction = np.zeros_like(background)
            for choice in np.unique(choices):
                idx = np.flatnonzero(choices == choice)
                entries, semantic = scores[choice]
                cmass = np.column_stack([distribution[idx, hh, mm] for hh, mm, *_ in entries])
                prediction[idx] = mixture_readout(cmass, semantic, background[idx], float(setting.component_temperature))
            fraction = float(setting.background_fraction)
            prediction = (1-fraction)*prediction + fraction*background
            score = np.sum(observed*np.log(prediction), axis=1)
            for t in np.flatnonzero(frame.eligible.to_numpy()):
                contrast_rows.append(dict(subject=int(sid), trial=t+1, split=frame.split.iloc[t],
                    early_middle=bool(frame.early_middle.iloc[t]), block64=int(frame.block64.iloc[t]),
                    comparator=name, rule=target if name == 'correct_rule' else dominant_wrong,
                    atom_log_score=float(score[t])))
        if int(sid) in p['rank_probe']['subjects']:
            calibration = frame.early_middle.to_numpy() & frame.split.eq('calibration').to_numpy()
            assessment = frame.early_middle.to_numpy() & frame.split.eq('assessment').to_numpy()
            prior_counts = np.bincount(choices[calibration]-1, minlength=4).astype(float)+1.
            prior = prior_counts/prior_counts.sum()
            for name, identity in [('longest', x.argmax(1)), ('shortest', x.argmin(1))]:
                counts = np.full((4, 4), float(p['rank_probe']['static_choice_dirichlet_per_category']))
                np.add.at(counts, (identity[calibration], choices[calibration]-1), 1.)
                probability = counts/counts.sum(1, keepdims=True)
                np.savez_compressed(output/f'S{sid}_{name}_choice_mapping.npz', probability=probability, counts=counts)
                for t in np.flatnonzero(assessment):
                    choice_rows.append(dict(subject=int(sid), trial=t+1, strategy=name,
                        longest_or_shortest_feature=int(identity[t]), observed_choice=int(choices[t]),
                        log_choice_probability=float(np.log(probability[identity[t], choices[t]-1])),
                        baseline_log_choice_probability=float(np.log(prior[choices[t]-1]))))
        print(f'S{sid}: alternative-rule evidence and fixed-rule comparisons completed', flush=True)
    trial_frame = pd.DataFrame(trial_rows)
    for sid, frame in trial_frame.groupby('subject'):
        for label, use in [('all', np.ones(len(frame), bool)), ('early_middle', frame.early_middle.to_numpy())]:
            for split in ['all', 'assessment']:
                subset = frame[use & (True if split == 'all' else frame.split.eq(split))]
                summary_rows.append(dict(subject=int(sid), window=label, split=split, reports=len(subset),
                    **{col:subset[col].mean() for col in ['correct_rule_mass', 'support_from_correct', 'support_from_other',
                        'alternative_specific_support', 'correct_conflicting_alternative_support']}))
    for name, rows in [('atoms', atom_rows), ('trials', trial_rows), ('summary', summary_rows),
                       ('rank_ceilings', rank_rows), ('rank_choice', choice_rows), ('fixed_rule_contrasts', contrast_rows)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv.gz' if name == 'atoms' else name+'.csv'), index=False)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Input changed during content diagnosis')
    result = dict(status='complete', atom_rows=len(atom_rows), scored_reports=len(trial_rows),
        rank_rules=len(ranks), rank_role='expression ceiling and static behavior diagnostic only',
        wall_seconds=time.time()-start, cpu_seconds=time.process_time()-cpu, inputs_sha256=inputs,
        source_sha256={str(Path(__file__)):digest(Path(__file__))},
        artifacts={f.name:digest(f) for f in output.iterdir() if f.is_file()})
    (output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print({k:v for k,v in result.items() if k not in ('inputs_sha256', 'source_sha256', 'artifacts')})
