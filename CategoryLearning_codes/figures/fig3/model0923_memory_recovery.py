"""Plot conditional recovery of belief and historical-evidence retention.

Run with --source completed_recovery --output NEW_DIRECTORY. Every synthetic
sequence and grid point is retained. Cells show NLL relative to the generating
point, with paired filter-seed Monte Carlo SE. Other parameters are known.
--allow-smoke produces an explicitly labelled implementation-check figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import (
    jackknife_scores, jackknife_se, nll,
)

MEMORY = 'engine.modules.memory_mod.kwargs.gamma'
DECAY = 'engine.modules.hypo_transitions_mod.kwargs.intervention.decay'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def score_contrasts(probabilities: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled NLL and delete-one-whole-seed scores for paired contrasts."""
    shapes = {p.shape for p in probabilities}
    if len(shapes) != 1 or not probabilities:
        raise ValueError('Need aligned nonempty candidate probability arrays')
    shape = probabilities[0].shape
    if len(shape) != 2 or shape[0] < 2 or shape[1] == 0:
        raise ValueError('Need multiple complete filter seeds and nonempty trials')
    if any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in probabilities):
        raise ValueError('Choice probabilities must be finite and within [0,1]')
    return (np.array([nll(p) for p in probabilities]),
            np.array([jackknife_scores(p) for p in probabilities]))


def load_table(source: Path, allow_smoke: bool = False) -> tuple[pd.DataFrame, dict]:
    completion = read(source / 'completion.json'); protocol = read(source / 'protocol.json')
    smoke = completion['smoke']
    if (completion['status'] != 'complete' or smoke != protocol['smoke']
            or (smoke and not allow_smoke) or not completion['nuisance_parameters_known']):
        raise ValueError('Need completed conditional recovery; smoke requires --allow-smoke')
    if digest(source / 'protocol.json') != completion['protocol_sha256']:
        raise ValueError('Recovery protocol changed')
    if protocol['scope'] != 'two_memory_grid_recovery_with_all_other_parameters_known':
        raise ValueError('Unexpected recovery interpretation')
    settings = protocol['recovery']
    grid = set(product(settings['memory_values'], settings['decay_values']))
    scores = pd.read_csv(source / 'scores.csv')
    recovered = pd.read_csv(source / 'recovered_memory.csv')
    truths = read(source / 'truths.json')['rows']
    cases = {f'{t["label"]}_R{i:02d}': t for t in truths for i in range(len(t['seeds']))}
    if (set(scores.case) != set(cases) or set(recovered.case) != set(cases)
            or recovered.case.duplicated().any() or len(cases) != completion['sequences']
            or len(scores) != completion['grid_points']):
        raise ValueError('Incomplete recovery cases or grid points')
    hashes = {name: digest(source / name) for name in (
        'protocol.json', 'completion.json', 'scores.csv', 'recovered_memory.csv', 'truths.json')}
    rows = []; seed_jobs = 0
    for case, group in scores.groupby('case', sort=True):
        truth = cases[case]; group = group.sort_values('candidate_id').reset_index(drop=True)
        if (group.duplicated(['memory', 'decay']).any()
                or set(zip(group.memory, group.decay)) != grid):
            raise ValueError('Require the complete declared grid for every sequence')
        probabilities = []; common_seeds = None
        for item in group.itertuples(index=False):
            path = source / 'scores' / case / f'{item.candidate_id}.json'
            if Path(item.score_file).resolve() != path.resolve():
                raise ValueError('Score table points outside its declared case')
            record = read(path); arrays = path.with_suffix('.npz')
            if digest(arrays) != record['predictions_sha256']:
                raise ValueError('Recovery predictions changed')
            expected = dict(truth['hyperparams'], **{MEMORY: item.memory, DECAY: item.decay})
            if (record['candidate']['hyperparams'] != expected or record['model'] != 'SEARCH'
                    or record['candidate']['id'] != item.candidate_id
                    or record['subject'] != item.subject or item.subject != truth['subject']
                    or item.true_memory != truth['true_memory'] or item.true_decay != truth['true_decay']):
                raise ValueError('Generating and scored parameter identities disagree')
            with np.load(arrays) as saved:
                probability = saved['observed_probability']; seeds = saved['seeds']
            expected_seeds = 2 if smoke else settings['seeds']
            if (len(seeds) != expected_seeds or len(set(seeds)) != len(seeds)
                    or probability.shape != (len(seeds), record['trials'])
                    or item.trials != record['trials']):
                raise ValueError('Incomplete seeds or trials')
            np.testing.assert_array_equal(seeds, record['seeds'])
            if common_seeds is not None:
                np.testing.assert_array_equal(seeds, common_seeds)
            common_seeds = seeds; probabilities.append(probability); seed_jobs += len(seeds)
            np.testing.assert_allclose(item.mean_nll, record['mean_nll'], rtol=0, atol=1e-12)
            for file in (path, arrays):
                hashes[str(file.relative_to(source))] = digest(file)
        values, jack = score_contrasts(probabilities)
        np.testing.assert_allclose(values, group.mean_nll, rtol=0, atol=1e-12)
        np.testing.assert_allclose([jackknife_se(j) for j in jack], group.mean_nll_mcse, rtol=0, atol=1e-12)
        true_indices = np.flatnonzero(group.memory.eq(truth['true_memory']) & group.decay.eq(truth['true_decay']))
        if len(true_indices) != 1:
            raise ValueError('Generating parameter point missing from grid')
        true_index = int(true_indices[0]); best = int(np.argmin(values))
        chosen = recovered.loc[recovered.case.eq(case)].iloc[0]
        if chosen.memory != group.iloc[best].memory or chosen.decay != group.iloc[best].decay:
            raise ValueError('Saved recovery selection disagrees with scored grid')
        # Exact ties retain the workflow's candidate-ID tie break. Numerical
        # uncertainty is shown separately using paired whole-seed differences.
        ties = int(np.sum(values == values[best]))
        for index, item in group.iterrows():
            rows.append(dict(item, mean_nll=values[index], filter_seeds=len(common_seeds),
                delta_to_truth=values[index]-values[true_index],
                delta_to_truth_mcse=jackknife_se(jack[index]-jack[true_index]),
                is_truth=index == true_index, selected=index == best, tied_minima=ties,
                truth_is_minimum=bool(values[true_index] == values[best]),
                unique_correct=bool(best == true_index and ties == 1)))
    if seed_jobs != completion['seed_jobs']:
        raise ValueError('Incomplete scoring-seed job count')
    return pd.DataFrame(rows), dict(smoke=smoke, input_sha256=hashes)


def plot(source: Path, output: Path, allow_smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery figure directory')
    table, provenance = load_table(source, allow_smoke)
    subjects = sorted(table.subject.unique())
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6,
        'ytick.labelsize': 6, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    max_rows = table.groupby('subject').case.nunique().max()
    height = max(100, 52 + 9 * max_rows)
    fig, axes = plt.subplots(1, len(subjects), figsize=(183/25.4, height/25.4), squeeze=False)
    fig.subplots_adjust(left=.16, right=.9, bottom=.32, top=.78, wspace=.75)
    limit = float(table.delta_to_truth.abs().max())
    # A completely flat smoke grid still needs a finite colour scale.
    norm = Normalize(-limit if limit else -1, limit if limit else 1)
    for column, subject in enumerate(subjects):
        ax = axes[0, column]; group = table.loc[table.subject.eq(subject)]
        cases = sorted(group.case.unique())
        points = sorted(set(zip(group.memory, group.decay)))
        values = np.array([[group.loc[group.case.eq(case) & group.memory.eq(gamma)
                    & group.decay.eq(delta), 'delta_to_truth'].iloc[0]
                    for gamma, delta in points] for case in cases])
        shown = ax.imshow(values, cmap='PuOr_r', norm=norm, aspect='auto')
        labels = []
        for y, case in enumerate(cases):
            case_rows = group.loc[group.case.eq(case)]; first = case_rows.iloc[0]
            label = f'γ {first.true_memory:g}, δ {first.true_decay:g} · R{int(first["repeat"])+1}'
            if first.tied_minima > 1:
                label += f'\n({int(first.tied_minima)} tied minima)'
            labels.append(label)
            for x, (gamma, delta) in enumerate(points):
                cell = case_rows.loc[case_rows.memory.eq(gamma) & case_rows.decay.eq(delta)].iloc[0]
                rgb = shown.cmap(norm(cell.delta_to_truth))[:3]
                color = 'black' if np.dot(rgb, [.2126, .7152, .0722]) > .52 else 'white'
                ax.text(x, y, f'{cell.delta_to_truth:+.3f}\n±{cell.delta_to_truth_mcse:.3f}',
                        ha='center', va='center', color=color, fontsize=6)
                if cell.is_truth:
                    ax.add_patch(Rectangle((x-.47, y-.46), .94, .92, fill=False, color='black', linewidth=1))
                if cell.selected:
                    ax.plot(x+.36, y-.32, marker='o', markersize=3, color='black',
                            markeredgecolor='white', markeredgewidth=.4)
        ax.set_xticks(range(len(points)), [f'γ {g:g}\nδ {d:g}' for g,d in points])
        ax.set_yticks(range(len(cases)), labels)
        ax.set_xlabel('Scored belief retention γ / evidence retention δ')
        trial_counts = ', '.join(str(int(n)) for n in sorted(group.trials.unique()))
        seed_counts = ', '.join(str(int(n)) for n in sorted(group.filter_seeds.unique()))
        ax.set_title(f'Participant {subject} · {len(cases)} sequences\n'
                     f'{trial_counts} trials each; {seed_counts} filter seeds', pad=9, fontsize=7)
        ax.text(-.08, 1.04, chr(97+column), transform=ax.transAxes, weight='bold', fontsize=8)
    cax = fig.add_axes([.93, .29, .015, .4]); fig.colorbar(shown, cax=cax)
    cax.set_title('ΔNLL\nper trial', fontsize=6)
    if limit == 0:
        cax.set_yticks([0])
    title = 'Conditional recovery of two memory parameters'
    if provenance['smoke']:
        title = 'IMPLEMENTATION CHECK: ' + title
    fig.suptitle(title, fontsize=9, y=.96)
    fig.text(.5, .045, 'Cells: NLL − generating-point NLL ± paired Monte Carlo SE\n'
        'Outline: generating point; dot: selected point (candidate-ID tie break)\n'
        'Other parameters known; simulation uncertainty only', ha='center', va='bottom', fontsize=7)
    output.mkdir(parents=True); table.to_csv(output / 'plot_source.csv', index=False)
    fig.savefig(output / 'memory_recovery.png', dpi=300); plt.close(fig)
    contract = dict(backend='python', archetype='quantitative grid', formats=['PNG'], dpi=300,
        width_mm=183, height_mm=int(height),
        claim='Show whether each independently generated sequence distinguishes the two generating memory levels.',
        source=str(source.resolve()), **provenance,
        uncertainty='Paired delete-one-whole-filter-seed jackknife MCSE of NLL difference; generating point is fixed.',
        observations='Every generated sequence and every declared grid point retained; no exclusions.',
        limitations=['Other parameters known', 'Coarse conditional recovery',
                     'No all-parameter recovery or empirical trait reliability claim', 'No hypothesis test'],
        export_exceptions=['PNG only by repository instruction; 300dpi discussion figure'])
    (output / 'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    (output / 'manifest.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)),
        artifacts={p.name: digest(p) for p in output.iterdir()}), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args(); plot(args.source, args.output, args.allow_smoke)
