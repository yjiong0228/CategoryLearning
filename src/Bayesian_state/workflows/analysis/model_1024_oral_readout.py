"""Compare normalized oral-atom readouts using frozen behavioral states.

Run --config YAML --output NEW_DIRECTORY. Model, catalogue, and stimulus-choice
references share the same expression layer and calibration-only grid. Every
report is kept; only prespecified complete numeric reports enter log scoring.
The score is the average categorical log probability of its coded atoms, not
the likelihood of a complete natural-language sentence.
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

from ...evaluation.oral.development import token_truth, token_feature_mask
from ...evaluation.oral.partial_reason_alignment import component_geometry, predicate_coverage
from ...model.modules.readout.report_space import component_tokens
from .model_0923_oral_constraint_audit import catalogue


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atom_vocabulary() -> tuple[str, ...]:
    """A fixed dictionary built from model definitions alone."""
    return tuple(sorted(set(t for cats in (2, 4) for rule in catalogue(cats)
        for region in rule.categories for part in region.components
        for t in component_tokens(part.A, part.b)))) + ('OTHER',)


def observed_atoms(tokens: list[tuple[str, ...]], vocabulary: tuple[str, ...]) -> np.ndarray:
    """Each row distributes one report's unit weight over all coded atoms."""
    lookup = {t:i for i, t in enumerate(vocabulary)}
    observed = np.zeros((len(tokens), len(vocabulary)))
    for row, words in enumerate(tokens):
        for token in words:
            observed[row, lookup.get(token, len(vocabulary)-1)] += 1/max(len(words), 1)
    return observed


def calibrated_frequency(observed: np.ndarray, calibration: np.ndarray, smoothing_total: float) -> np.ndarray:
    """Use calibration reports only, with positive uniform Dirichlet mass."""
    if not calibration.any() or smoothing_total <= 0:
        raise ValueError('Require calibration reports and positive smoothing')
    frequency = observed[calibration].sum(0) + smoothing_total / observed.shape[1]
    return frequency / frequency.sum()


def mixture_readout(mass: np.ndarray, scores: np.ndarray, background: np.ndarray,
                    temperature: float) -> np.ndarray:
    """Normalize within each component, then mix under its unchanged mass.

    mass: trials × represented components; scores: components × vocabulary;
    background: trials × vocabulary. All components remain in the mixture.
    """
    mass, scores, background = map(lambda a: np.asarray(a, float), (mass, scores, background))
    if mass.ndim != 2 or scores.shape[0] != mass.shape[1] or background.shape != (len(mass), scores.shape[1]):
        raise ValueError('Unaligned readout arrays')
    if not all(np.isfinite(a).all() for a in (mass, scores, background)) or (mass < 0).any() or (background <= 0).any():
        raise ValueError('Invalid readout probabilities')
    np.testing.assert_allclose(mass.sum(1), 1., atol=2e-10, rtol=0)
    exp_score = np.exp(temperature * scores)
    normalizer = background @ exp_score.T
    probability = background * ((mass / normalizer) @ exp_score)
    np.testing.assert_allclose(probability.sum(1), 1., atol=3e-10, rtol=0)
    return probability


def shuffle_state(mass: np.ndarray, choices: np.ndarray, seed: int) -> np.ndarray:
    """Preserve choice and broad time block; shuffle only the exact trial."""
    rng = np.random.default_rng(seed)
    result = mass.copy()
    blocks = np.arange(len(mass))//64
    for block in np.unique(blocks):
        for choice in np.unique(choices):
            idx = np.flatnonzero((blocks == block) & (choices == choice))
            result[idx] = mass[rng.permutation(idx)]
    return result


def run(config: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new readout directory')
    start, cpu = time.time(), time.process_time()
    p = yaml.safe_load(config.read_text())
    inputs = {str(config): digest(config)}

    def bind(path: Path) -> Path:
        inputs[str(path)] = digest(path)
        return path

    folder = Path(p['diagnosis'])
    trials = pd.read_csv(bind(folder/'trials.csv'))
    vocabulary = atom_vocabulary()
    v = len(vocabulary)
    output.mkdir(parents=True)
    (output/'registration.json').write_text(json.dumps(dict(started_unix=start, policy=p,
        vocabulary=vocabulary, normalized_outcome='one coded atomic judgment sampled uniformly within a report',
        state_weight_changes=False), ensure_ascii=False, indent=2))
    grid_rows, chosen_rows, trial_rows, summaries = [], [], [], []
    probability_checks = 0
    for sid, frame in trials.groupby('subject', sort=True):
        frame = frame.reset_index(drop=True)
        metadata = json.loads(bind(folder/f'S{sid}_metadata.json').read_text())
        with np.load(bind(folder/f'S{sid}_diagnostic_arrays.npz')) as z:
            x, choices, eligible = z['stimulus'], z['choice'], z['eligible']
            masses = dict(catalogue=z['catalogue_mass'], stimulus_choice=z['stimulus_reference_mass'])
        for candidate in metadata['selected']:
            file = bind(Path(candidate['file']))
            if inputs[str(file)] != candidate['sha256']:
                raise ValueError('State archive changed')
            with np.load(file) as z:
                masses[f'P{candidate["number"]}'] = z[p['primary_mode']+'_component']
        masses['P1_shuffled'] = shuffle_state(masses['P1'], choices, p['shuffle']['seed']+int(sid))
        tokens = [tuple(json.loads(value)) for value in frame.tokens]
        y = observed_atoms(tokens, vocabulary)
        # All branches use the same denominator, including control availability.
        comparable = eligible & np.isfinite(masses['stimulus_choice']).all((1, 2))
        truth = np.zeros((len(frame), v))
        for i in range(len(frame)):
            truth[i, :-1] = token_truth(vocabulary[:-1], x[i])
        h, m = masses['P1'].shape[1:]
        space = catalogue(2 if h == 29 else 4)
        components, scores = {}, {}
        for choice in np.unique(choices):
            entries, local, _ = component_geometry(space, int(choice))
            components[choice] = [(hh, mm) for hh, mm, *_ in entries]
            semantic, feature = np.zeros((len(entries), v)), np.zeros((len(entries), v))
            for j, (_, _, A, b) in enumerate(entries):
                for k, token in enumerate(vocabulary[:-1]):
                    verdict = predicate_coverage((token,), A, b)
                    semantic[j, k] = verdict[0] - verdict[1]
                    mask = token_feature_mask(token)
                    feature[j, k] = np.sum(mask & local[j])/mask.sum()
            scores[choice] = dict(semantic=semantic, feature=feature)
        for selection_window in p['selection_windows']:
            calibration = comparable & frame.split.eq('calibration').to_numpy()
            if selection_window == 'early_middle':
                calibration &= frame.early_middle.to_numpy()
            frequency = calibrated_frequency(y, calibration, p['background_smoothing_total'])
            (output/f'S{sid}_{selection_window}_frequency.json').write_text(json.dumps(dict(
                calibration_trials=(np.flatnonzero(calibration)+1).tolist(), frequency=frequency.tolist()), indent=2))
            best = {}

            def consider(branch: str, probability: np.ndarray, settings: dict) -> None:
                nonlocal probability_checks
                np.testing.assert_allclose(probability[comparable].sum(1), 1., atol=3e-10, rtol=0)
                if not np.isfinite(probability[comparable]).all() or (probability[comparable] <= 0).any():
                    raise ValueError('Invalid readout probability')
                probability_checks += int(comparable.sum())
                score = np.sum(y*np.log(np.maximum(probability, np.finfo(float).tiny)), axis=1)
                score[~comparable] = np.nan
                value = float(score[calibration].mean())
                row = dict(subject=int(sid), selection_window=selection_window, branch=branch,
                    calibration_reports=int(calibration.sum()), calibration_score=value, **settings)
                grid_rows.append(row)
                if branch not in best or value > best[branch]['calibration_score'] + 1e-12:
                    best[branch] = dict(row, score=score.copy(), probability=probability.copy())

            for stimulus_temperature in p['stimulus_temperature_grid']:
                background = frequency[None, :] * np.exp(stimulus_temperature*truth)
                background /= background.sum(1, keepdims=True)
                settings = dict(stimulus_temperature=stimulus_temperature, component_temperature=0.,
                    background_fraction=1., candidate='none')
                consider('expression_background', background, settings)
                if stimulus_temperature == 0:
                    consider('frequency_only', background, settings)
                for score_name in p['component_scores']:
                    for temperature in p['component_temperature_grid']:
                        if time.time() > p['budget']['deadline_unix'] or time.process_time()-cpu > 3600*p['budget']['max_cpu_hours']:
                            raise TimeoutError('Readout budget exceeded')
                        for name, mass in masses.items():
                            probability = np.zeros((len(frame), v))
                            for choice in np.unique(choices):
                                idx = np.flatnonzero((choices == choice) & comparable)
                                coords = components[choice]
                                cmass = np.column_stack([mass[idx, hh, mm] for hh, mm in coords])
                                probability[idx] = mixture_readout(cmass, scores[choice][score_name], background[idx], temperature)
                            probability[~comparable] = background[~comparable]
                            for fraction in p['background_mixture_grid']:
                                q = (1-fraction)*probability + fraction*background
                                settings = dict(stimulus_temperature=stimulus_temperature, component_temperature=temperature,
                                    background_fraction=fraction, candidate=name)
                                consider(score_name+'_'+name, q, settings)
            # Candidate selection sees only calibration scores. Assessment is
            # read below, after the entire chosen configuration has been frozen.
            for score_name in p['component_scores']:
                keys = [score_name+f'_P{s["number"]}' for s in metadata['selected']]
                winner = max(keys, key=lambda key: best[key]['calibration_score'])
                best[score_name+'_selected'] = dict(best[winner], branch=score_name+'_selected')
            save_probabilities = {}
            for branch, result in best.items():
                score, probability = result.pop('score'), result.pop('probability')
                chosen_rows.append(result)
                save_probabilities[branch] = probability
                for i, row in frame.iterrows():
                    trial_rows.append(dict(subject=int(sid), trial=int(row.trial), split=row.split,
                        early_middle=bool(row.early_middle), block64=int(row.block64), eligible=bool(comparable[i]),
                        selection_window=selection_window, branch=branch, candidate=result['candidate'],
                        atom_log_score=score[i], unseen_atom_fraction=float(y[i, -1]),
                        top_atom=vocabulary[int(np.argmax(probability[i]))],
                        top_atom_probability=float(probability[i].max())))
                windows = [('all', np.ones(len(frame), bool)), ('early_middle', frame.early_middle.to_numpy())]
                windows += [(f'block64_{i}', frame.block64.eq(i).to_numpy()) for i in frame.block64.unique()]
                for window, mask in windows:
                    for split in ('calibration', 'assessment'):
                        use = comparable & mask & frame.split.eq(split).to_numpy()
                        summaries.append(dict(subject=int(sid), selection_window=selection_window, branch=branch,
                            candidate=result['candidate'], window=window, split=split, reports=int(use.sum()),
                            mean_atom_log_score=float(score[use].mean()) if use.any() else np.nan,
                            mean_other_fraction=float(y[use, -1].mean()) if use.any() else np.nan))
            np.savez_compressed(output/f'S{sid}_{selection_window}_predictions.npz', **save_probabilities)
        print(f'S{sid}: calibrated readouts and block assessments completed', flush=True)
    for name, rows in [('grid', grid_rows), ('selected', chosen_rows), ('trial_scores', trial_rows), ('summary', summaries)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv.gz' if name in ('grid', 'trial_scores') else name+'.csv'), index=False)
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Input changed during readout: '+path)
    result = dict(status='complete', subjects=sorted(trials.subject.unique().tolist()), vocabulary_size=v,
        probability_checks=probability_checks, grid_rows=len(grid_rows), selected_rows=len(chosen_rows),
        wall_seconds=time.time()-start, cpu_seconds=time.process_time()-cpu, inputs_sha256=inputs,
        sources_sha256={str(Path(__file__)):digest(Path(__file__))},
        artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print({k:v for k,v in result.items() if k not in ('inputs_sha256', 'sources_sha256', 'artifacts')})
