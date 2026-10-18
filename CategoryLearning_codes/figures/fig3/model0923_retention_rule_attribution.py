"""Plot frozen precision-retention changes in rule availability and contribution.

Use --source ATTRIBUTION --validation INDEPENDENT_CHECK --output NEW_DIRECTORY.
The left panel uses all 192 later trials. The right uses 119 complete later
reports and displays the union of five largest absolute rule-mass changes at
each capacity. All 152 rules remain in the copied source table. Error bars are
one paired, whole-filter Monte Carlo standard error at frozen parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(source: Path, validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new rule-attribution figure directory')
    inputs = {}

    def record(path: Path, expected: str | None = None) -> str:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError('Figure source changed: '+path.name)
        inputs[str(path.resolve())] = actual
        return actual

    manifest = json.loads((source/'manifest.json').read_text())
    receipt = json.loads((validation/'validation.json').read_text())
    record(source/'manifest.json'); record(validation/'validation.json')
    if (manifest['status'] != 'complete' or receipt['status'] != 'pass'
            or manifest['filter_seeds'] != 8 or manifest['reports'] != 119
            or manifest['target_rule'] != 42):
        raise ValueError('Need validated complete S208 precision attribution')
    for name in ('attribution.csv', 'target_states.csv', 'catalogue.csv'):
        path = source/name
        sha = record(path, manifest['artifacts'][name])
        if receipt['input_sha256'].get(str(path.resolve())) != sha:
            raise ValueError('Figure input differs from independent validation')
    target = pd.read_csv(source/'target_states.csv').query('window == "test"')
    attribution = pd.read_csv(source/'attribution.csv')
    probability_metrics = ['conditioned_active', 'conditioned_belief',
                           'prior_contribution', 'conditioned_contribution']
    selected = target[target.metric.isin(probability_metrics)]
    if (len(selected) != 8 or selected.duplicated(['capacity', 'metric']).any()
            or set(selected.capacity) != {2, 4} or not selected.trials.eq(192).all()):
        raise ValueError('Need both capacities and four later target-state metrics')
    all_rules = attribution.query('level == "rule" and metric == "unweighted" and domain == "structural"').copy()
    all_rules['rule'] = all_rules.group.astype(int)
    midpoint = all_rules[all_rules.reading == 'midpoint']
    relative = all_rules[all_rules.reading == 'relative']
    for capacity in (2, 4):
        a, b = (frame[frame.capacity == capacity].sort_values('rule') for frame in (midpoint, relative))
        np.testing.assert_array_equal(a.rule, np.arange(152))
        np.testing.assert_array_equal(a.rule, b.rule)
        np.testing.assert_allclose(a[['delta_mass', 'delta_mass_mcse']],
                                   b[['delta_mass', 'delta_mass_mcse']], atol=1e-12, rtol=0)
    largest = set()
    for _, frame in midpoint.groupby('capacity'):
        largest.update(frame.assign(magnitude=frame.delta_mass.abs())
                       .sort_values(['magnitude', 'rule'], ascending=[False, True]).iloc[:5].rule)
    order = sorted(largest, key=lambda rule: -midpoint[midpoint.rule == rule].delta_mass.abs().max())
    labels = {42:'42: task rule', 66:'66: axis / compare / sum', 24:'24: axis / sum',
              106:'106: sum / axis pair', 94:'94: compare / axis pair', 55:'55: axis triple'}
    if set(order) != set(labels):
        raise ValueError('Rule identities changed; revise labels before plotting new data')
    rows = [dict(panel='a', capacity=row.capacity, metric=row.metric,
                 value=100*row.delta, se=100*row.paired_mcse) for row in selected.itertuples()]
    rows += [dict(panel='b', capacity=row.capacity, metric=str(row.rule),
                  value=100*row.delta_mass, se=100*row.delta_mass_mcse)
             for row in midpoint[midpoint.rule.isin(order)].itertuples()]
    plot = pd.DataFrame(rows)
    if not np.isfinite(plot[['value', 'se']]).all().all() or (plot.se < 0).any():
        raise ValueError('Need finite contrasts and nonnegative simulation errors')
    output.mkdir(parents=True)
    plot.to_csv(output/'plot_source.csv', index=False)
    for name in ('attribution.csv', 'target_states.csv', 'catalogue.csv'):
        shutil.copy2(source/name, output/name)
    contract = dict(claim='Rule 42 remains active while its inferred belief and contribution decrease; alternative rule mass rises.',
        backend='python', archetype='quantitative grid', hero='b: largest rule-mass changes',
        width_mm=183, height_mm=110, dpi=300, formats=['PNG'], inputs_sha256=inputs,
        evidence={'a':'All 192 later trials: target presence, belief and contribution.',
                  'b':'119 complete later reports: rule contribution before current-choice likelihood weighting.'},
        selection={'a':'Four probability-scale target metrics; active precision has another unit and remains in source data.',
                   'b':'Union of five largest absolute mass changes per capacity; all 152 rules retained in attribution.csv.',
                   'shown_rules':order, 'other_rules':152-len(order)},
        statistics='RETAIN minus RESET; one paired whole-filter Monte Carlo SE, 8 seeds per mechanism; one development subject.',
        data_use='Two fixed behavior-selected points, capacities 2 and 4. Oral excluded from fitting; no population inference.',
        interpretation='Rule contribution uses past behavior conditioning and observed-category geometry; this is algebraic attribution.',
        adaptation='Style-only inheritance from the existing Python figure workflow; new field mapping and panel logic.',
        source_preflight_exceptions={'EXPORT-VECTOR':'Repository PNG-only rule.',
            'EXPORT-RASTER':'Repository PNG-only rule.', 'RASTER-DPI':'300 dpi discussion figure.'})
    (output/'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':7,
        'axes.titlesize':7, 'axes.labelsize':7, 'xtick.labelsize':6, 'ytick.labelsize':6,
        'legend.fontsize':7, 'axes.linewidth':.6, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig = plt.figure(figsize=(183/25.4, 110/25.4))
    axes = [fig.add_axes([.19,.28,.27,.49]), fig.add_axes([.70,.28,.27,.49])]
    colors = {2:'#405C77', 4:'#C28A55'}
    for panel, ax, names in [('a',axes[0],probability_metrics), ('b',axes[1],list(map(str,order)))]:
        for capacity, offset, marker in [(2,-.12,'o'), (4,.12,'s')]:
            own = plot[(plot.panel == panel) & (plot.capacity == capacity)].set_index('metric').loc[names]
            ax.errorbar(own.value, np.arange(len(names))+offset, xerr=own.se, fmt=marker,
                        color=colors[capacity], markersize=4, capsize=2.5, label=f'K = {capacity}')
        ax.axvline(0, color='#999999', linewidth=.8, linestyle=':')
        ax.set(yticks=np.arange(len(names)), ylim=(len(names)-.5,-.5), xlabel='RETAIN − RESET (percentage points)')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(length=2.5, width=.6)
        ax.text(-.08,1.13,panel,transform=ax.transAxes,weight='bold',fontsize=8)
    axes[0].set_yticklabels(['Active probability', 'Rule belief', 'Contribution before\nchoice weighting',
                             'Contribution after\nchoice weighting'])
    axes[1].set_yticklabels([labels[rule] for rule in order])
    axes[0].set_title('Task rule 42\nAll 192 later trials',pad=10)
    axes[1].set_title('Largest rule-mass changes\n119 complete later reports',pad=10)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.54,.92), ncol=2, frameon=False)
    fig.suptitle('Subject 208: rule 42 remains active while its contribution falls',y=.985,fontsize=9)
    fig.text(.07,.17,'Bars: ± 1 Monte Carlo SE across 8 paired filters. K is the active-rule capacity.',fontsize=6)
    fig.text(.07,.12,'Panel b uses contribution before current-choice likelihood weighting; both oral readings give identical rule masses.',fontsize=6)
    fig.text(.07,.07,'One development subject; two fixed parameter points. Largest-change selection is descriptive; all rules remain in source data.',fontsize=6)
    fig.savefig(output/'rule_attribution.png',dpi=300)
    plt.close(fig)
    shutil.copy2(__file__, output/'plot_source.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'validation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
