"""Show every capacity/storage recovery case and candidate, including ties.

Run with --source completed_recovery --output NEW_DIRECTORY. Four panels show
the generating capacity/storage cells. Rows are independently generated choice
sequences; columns are all scored cells. Values are NLL relative to the known
generating cell, with paired whole-filter-seed Monte Carlo SE. Other parameters
are known. --allow-smoke labels a short implementation check explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3.model0923_memory_recovery import (
    digest, read, score_contrasts,
)
from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se

CAPACITY = 'engine.modules.hypo_transitions_mod.kwargs.max_capacity'
CELLS = ('K2_RESET', 'K2_RETAIN', 'K4_RESET', 'K4_RETAIN')


def load_table(source: Path, allow_smoke: bool = False) -> tuple[pd.DataFrame, dict]:
    """Bind the completed four-cell experiment and reconstruct plotted scores."""
    done, protocol = (read(source/name) for name in ('completion.json', 'protocol.json'))
    smoke = done['smoke']
    if (done['status'] != 'complete' or smoke != protocol['smoke']
            or (smoke and not allow_smoke) or done['nuisance_parameters_known'] is not True
            or protocol['settings']['protocol'] != 'model0923_capacity_precision_recovery_v1'):
        raise ValueError('Need completed conditional capacity/storage recovery; smoke requires --allow-smoke')
    if digest(source/'protocol.json') != done['protocol_sha256']:
        raise ValueError('Recovery protocol changed')
    for name in ('scores.csv', 'recovered.csv'):
        if digest(source/name) != done['artifacts'][name]:
            raise ValueError('Completed recovery table changed')
    table = pd.read_csv(source/'scores.csv', float_precision='round_trip')
    recovered = pd.read_csv(source/'recovered.csv', float_precision='round_trip')
    truth_rows = read(source/'truths.json')['rows']
    truths = {t['case']: t for t in truth_rows}
    if (len(truths) != len(truth_rows) or len(truths) != done['sequences']
            or len(truths) != (1 if smoke else 16) or set(table.case) != set(truths)
            or set(recovered.case) != set(truths) or recovered.case.duplicated().any()
            or len(table) != done['jobs'] or done['jobs'] != protocol['jobs']):
        raise ValueError('Incomplete recovery cases')
    if not smoke and pd.Series([t['true_cell'] for t in truths.values()]).value_counts().to_dict() != dict.fromkeys(CELLS, 4):
        raise ValueError('Need all four generating cells with four sequences each')
    hashes = {str((source/n).resolve()): digest(source/n) for n in
              ('protocol.json', 'completion.json', 'scores.csv', 'recovered.csv', 'truths.json')}
    rows, filters = [], 0
    for case, group in table.groupby('case', sort=True):
        if len(group) != 4 or group.cell.duplicated().any() or set(group.cell) != set(CELLS):
            raise ValueError('Need every scored cell exactly once per sequence')
        group = group.set_index('cell').loc[list(CELLS)].reset_index()
        truth = truths[case]
        arrays, common_seeds = [], None
        for row in group.itertuples(index=False):
            path = source/'scores'/case/(row.cell+'.json')
            record = read(path)
            job_file = source/'jobs'/('score_'+case+'_'+row.cell+'.json')
            job = read(job_file)
            precision_file = path.with_suffix('.precision.json')
            precision = read(precision_file)
            retain = row.cell.endswith('_RETAIN')
            expected = dict(truth['candidate']['hyperparams'], **{CAPACITY: row.capacity})
            if (Path(row.score_file).resolve() != path.resolve()
                    or Path(job['target']).with_suffix('.json').resolve() != path.resolve()
                    or record['candidate'] != job['candidate']
                    or record['candidate']['id'] != row.candidate_id
                    or record['candidate']['hyperparams'] != expected
                    or row.cell != f'K{row.capacity}_{row.variant}'
                    or row.true_cell != truth['true_cell'] or row.true_capacity != truth['capacity']
                    or row.true_variant != truth['variant']
                    or record['model'] != 'SEARCH' or record['subject'] != 208 or record['condition'] != 2
                    or job['retain_precision'] is not retain or precision['retain_precision'] is not retain):
                raise ValueError('Generating or scored identity mismatch')
            npz = path.with_suffix('.npz')
            if digest(npz) != record['predictions_sha256']:
                raise ValueError('Recovery probabilities changed')
            with np.load(npz) as z:
                probability, seeds = z['observed_probability'], z['seeds']
            expected_seeds = 2 if smoke else protocol['settings']['seeds']
            if (len(seeds) != expected_seeds or len(set(seeds)) != len(seeds)
                    or probability.shape != (len(seeds), truth['trials'])
                    or record['trials'] != truth['trials'] or row.trials != truth['trials']
                    or record['particles'] != (2 if smoke else protocol['settings']['particles'])):
                raise ValueError('Incomplete seeds, particles or trials')
            np.testing.assert_array_equal(seeds, record['seeds'])
            np.testing.assert_array_equal(seeds, job['seeds'])
            if common_seeds is not None:
                np.testing.assert_array_equal(seeds, common_seeds)
            common_seeds = seeds
            np.testing.assert_allclose(row.mean_nll, record['mean_nll'], rtol=0, atol=1e-12)
            arrays.append(probability)
            filters += len(seeds)
            for file in (path, npz, precision_file, job_file):
                hashes[str(file.resolve())] = digest(file)
        values, jack = score_contrasts(arrays)
        np.testing.assert_allclose(values, group.mean_nll, rtol=0, atol=1e-12)
        np.testing.assert_allclose([jackknife_se(j) for j in jack], group.mean_nll_mcse, rtol=0, atol=1e-12)
        true_index = CELLS.index(truth['true_cell'])
        minima = values == values.min()
        chosen = recovered.loc[recovered.case.eq(case)].iloc[0]
        selected = min(CELLS[i] for i in np.flatnonzero(minima))
        if (chosen.true_cell != truth['true_cell'] or chosen.selected_cell != selected
                or chosen.tied_minima != int(minima.sum())
                or bool(chosen.truth_is_minimum) != bool(minima[true_index])
                or bool(chosen.unique_joint_correct) != bool(minima[true_index] and minima.sum() == 1)):
            raise ValueError('Saved recovery selection or ties disagree with probabilities')
        for index, row in group.iterrows():
            rows.append(dict(row, filter_seeds=len(common_seeds),
                delta_to_truth=values[index]-values[true_index],
                delta_to_truth_mcse=jackknife_se(jack[index]-jack[true_index]),
                is_truth=index == true_index, is_minimum=bool(minima[index]),
                tied_minima=int(minima.sum()), unique_correct=bool(minima[true_index] and minima.sum() == 1)))
    if filters != done['filter_runs'] or filters != protocol['filter_runs']:
        raise ValueError('Incomplete filter-run count')
    return pd.DataFrame(rows), dict(smoke=smoke, input_sha256=hashes)


def plot(source: Path, output: Path, allow_smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new recovery figure directory')
    table, provenance = load_table(source, allow_smoke)
    generating = [c for c in CELLS if c in set(table.true_cell)]
    ncols = min(2, len(generating)); nrows = (len(generating)+ncols-1)//ncols
    height = 155 if nrows == 2 else 90
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6,
        'ytick.labelsize': 6, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(nrows, ncols, figsize=(183/25.4, height/25.4), squeeze=False)
    fig.subplots_adjust(left=.10, right=.88, bottom=.22 if nrows == 2 else .36,
                        top=.85 if nrows == 2 else .78, hspace=.70, wspace=.30)
    limit = float(table.delta_to_truth.abs().max())
    norm = Normalize(-limit if limit else -1, limit if limit else 1)
    for index, (ax, truth) in enumerate(zip(axes.flat, generating)):
        group = table.loc[table.true_cell.eq(truth)]
        cases = sorted(group.case.unique())
        values = group.pivot(index='case', columns='cell', values='delta_to_truth').loc[cases, list(CELLS)]
        shown = ax.imshow(values, cmap='PuOr_r', norm=norm, aspect='auto')
        for y, case in enumerate(cases):
            for x, cell in enumerate(CELLS):
                row = group.loc[group.case.eq(case) & group.cell.eq(cell)].iloc[0]
                rgb = shown.cmap(norm(row.delta_to_truth))[:3]
                color = 'black' if np.dot(rgb, [.2126, .7152, .0722]) > .52 else 'white'
                ax.text(x, y, f'{row.delta_to_truth:+.3f}\n±{row.delta_to_truth_mcse:.3f}',
                        ha='center', va='center', color=color, fontsize=6)
                if row.is_truth:
                    ax.add_patch(Rectangle((x-.47, y-.46), .94, .92, fill=False, color='black', linewidth=1))
                if row.is_minimum:
                    ax.plot(x+.36, y-.32, marker='o', markersize=3, color='black',
                            markeredgecolor='white', markeredgewidth=.4)
        ax.set_xticks(range(4), [c.replace('_', '\n') for c in CELLS])
        ax.set_yticks(range(len(cases)), [f'Sequence {i+1}' for i in range(len(cases))])
        count = group.groupby('case').unique_correct.first().sum()
        ax.set_title(f'Generated by {truth.replace("_", " / ")}\nUnique correct minimum: {count}/{len(cases)}', pad=8)
        ax.set_xlabel('Scored capacity / precision storage')
        ax.text(-.12, 1.10, chr(97+index), transform=ax.transAxes, weight='bold', fontsize=8)
    cax = fig.add_axes([.92, .31, .015, .39])
    fig.colorbar(shown, cax=cax)
    cax.set_title('ΔNLL\nper trial', fontsize=6)
    if limit == 0:
        cax.set_yticks([0])
    title = 'Can choices distinguish capacity and precision storage?'
    if provenance['smoke']:
        title = 'IMPLEMENTATION CHECK: capacity / storage recovery'
    fig.suptitle(title, fontsize=9, y=.97)
    trials = ', '.join(str(n) for n in sorted(table.trials.unique()))
    seeds = ', '.join(str(n) for n in sorted(table.filter_seeds.unique()))
    fig.text(.5, .025, f'{table.case.nunique()} synthetic sequences · {trials} training trials · {seeds} paired filter seeds\n'
        'Cells: NLL − generating-cell NLL ± Monte Carlo SE; negative favours the alternative\n'
        'Outline: generating cell; dots: all minimum-score cells, including exact ties\n'
        'Other parameters known; simulation uncertainty only', ha='center', va='bottom', fontsize=7)
    output.mkdir(parents=True)
    table.to_csv(output/'plot_source.csv', index=False)
    fig.savefig(output/'capacity_precision_recovery.png', dpi=300)
    plt.close(fig)
    contract = dict(backend='python', archetype='quantitative grid', formats=['PNG'], dpi=300,
        width_mm=183, height_mm=height,
        claim='Show whether each synthetic training sequence distinguishes the generating capacity and storage mechanism.',
        evidence='One panel per generating cell; every sequence and all four scored candidates retained.',
        source=str(source.resolve()), **provenance,
        field_mapping=dict(panel='true_cell', row='case', column='cell', value='mean_nll minus generating-cell mean_nll'),
        uncertainty='Paired delete-one-whole-filter-seed jackknife MCSE; fixed generating-cell reference.',
        observations='All 16 scientific sequences and 64 cells; smoke uses its one original short sequence.',
        ties='All exact minimum-score cells are marked; unique recovery requires one correct minimum.',
        limitations=['Other parameters known', 'Coarse two-capacity two-mechanism recovery',
                     'No all-parameter or empirical trait reliability claim', 'No hypothesis test'],
        adaptation='Structural adaptation of the existing memory-recovery grid; shared score computation reused.',
        export_exceptions=['PNG only by repository instruction; 300dpi discussion figure'])
    (output/'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)),
        shared_score_source_sha256=digest(Path(__file__).with_name('model0923_memory_recovery.py')),
        artifacts={p.name:digest(p) for p in output.iterdir()}), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    plot(args.source, args.output, args.allow_smoke)
