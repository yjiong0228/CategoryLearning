"""Draw audited conditional state recovery and oral compatibility (PNG only).

See the adjacent panel contract. All fixed cases and all three enriched paths
are retained; zero-event conditional scores are explicitly marked missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.Bayesian_state.workflows.analysis.model_1024_state_oral_validation import read, write, digest

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 6, 'legend.frameon': False,
    'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': .7,
    'svg.fonttype': 'none', 'pdf.fonttype': 42})
BLUE, ORANGE, GREY = '#39769C', '#B77E40', '#8D949C'
ARMS = ('MIX', 'MIX_ALT', 'STICKY', 'STICKY_ALT')


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output/(name+'.png'), dpi=300, facecolor='white')
    plt.close(fig)


def run(source: Path, oral: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new figure directory')
    completed = read(source/'completion.json')
    if completed['status'] != 'complete':
        raise ValueError('Require completed state and content summary')
    for name, sha in completed['artifacts_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError('Audited summary table changed')
    output.mkdir(parents=True)
    metrics = pd.read_csv(source/'state_metrics.csv'); paths = pd.read_csv(source/'state_trials.csv')
    baselines = pd.read_csv(source/'oral_baselines.csv')
    fixed = metrics.loc[metrics.scope.eq('fixed') & metrics.window.eq('whole') & metrics.timing.eq('filtered')]
    cases = fixed[['case_id', 'subject']].drop_duplicates().sort_values('case_id')
    case_ids = cases.case_id.tolist(); x = np.arange(len(cases))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    fig.subplots_adjust(left=.09, right=.98, top=.91, bottom=.22, hspace=.60, wspace=.30)
    panels = [('present', 'brier', 'Rule presence: Brier (lower is better)'),
              ('exact_execution_identity', 'accuracy', 'Executed rule: accuracy'),
              ('error_available_other', 'true_event_probability', 'Key error: true-class probability'),
              ('error_available_other', 'hard_recall', 'Key error: classification recall')]
    for i, (state, field, title) in enumerate(panels):
        ax = axes.flat[i]
        for role, shift, color, marker in [('true', -.13, BLUE, 'o'), ('adopted', .13, ORANGE, 's')]:
            own = fixed.loc[fixed.state.eq(state) & fixed.role.eq(role)].set_index('case_id').reindex(case_ids)
            value = own[field].to_numpy(); se = own.get(field+'_mcse', pd.Series(np.zeros(len(own)), index=own.index)).to_numpy()
            ax.errorbar(x+shift, value, yerr=2*se, fmt=marker, ms=3.3, color=color, capsize=1.5, lw=.65)
        if i >= 2:
            counts = fixed.loc[fixed.state.eq(state) & fixed.role.eq('true')].set_index('case_id').reindex(case_ids).positives.fillna(0).to_numpy(int)
            for position, count in zip(x, counts):
                ax.text(position, 1.035, str(count), ha='center', va='bottom', fontsize=6)
                if count == 0:
                    ax.text(position, .045, 'NA', ha='center', fontsize=5.5, color=GREY)
        ax.set_xticks(x, [f'{c}\n{s}' for c, s in cases.itertuples(index=False, name=None)], rotation=0)
        presence = fixed.loc[fixed.state.eq('present')]
        brier_upper = max(.15, float((presence.brier+2*presence.brier_mcse).max())*1.1)
        ax.set_xlim(-.7, len(x)-.3); ax.set_ylim(0, 1.15 if i >= 2 else (brier_upper if i == 0 else 1.05))
        ax.grid(axis='y', color='.92', lw=.5)
        ax.set_title(f'{chr(97+i)}  {title}', loc='left', fontweight='bold')
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='', label='Known generating parameters'),
                        Line2D([], [], color=ORANGE, marker='s', ls='', label='Training-estimated parameters')],
               loc='lower center', bbox_to_anchor=(.5, .105), ncol=2)
    fig.text(.09, .025, '12 fixed paths; case ID / source subject below each point. Counts = key errors in that path.\nKey error: correct rule present, another rule executed, wrong choice.\nClassification uses the largest class probability. Bars: +/-2 filter MCSE.', fontsize=7)
    save(fig, output, 'state_recovery')

    challenge = paths.loc[paths.scope.eq('enriched_challenge') & paths.role.eq('true')]
    identities = challenge[['case_id', 'subject']].drop_duplicates().sort_values('subject')
    fig, axes = plt.subplots(len(identities), 1, figsize=(7.2, 4.5), squeeze=False)
    fig.subplots_adjust(left=.09, right=.98, top=.93, bottom=.20, hspace=.60)
    for i, (identity, sid) in enumerate(identities.itertuples(index=False, name=None)):
        ax = axes[i, 0]; own = challenge.loc[challenge.case_id.eq(identity)].sort_values('trial')
        rare = own.wrong & own.error_truth.eq(1)
        probabilities = own[['p_error_absent', 'p_error_other', 'p_error_executed']].to_numpy()
        hits = int(((probabilities.argmax(axis=1) == 1) & rare.to_numpy()).sum())
        ax.plot(own.trial, own.p_error_other, color=BLUE, lw=.9)
        ax.scatter(own.loc[rare, 'trial'], np.full(rare.sum(), 1.03), s=7, marker='|', color='#222222', lw=.7)
        ax.set_ylim(0, 1.12); ax.set_xlim(1, own.trial.max()); ax.set_ylabel('P(present, other)')
        ax.set_title(f'{chr(97+i)}  Subject {sid}, {identity}: {rare.sum()} key errors; {hits} classified correctly', loc='left', fontweight='bold')
    axes[-1, 0].set_xlabel('Trial')
    fig.text(.09, .035, 'One truth-enriched path per subject; known parameters only. Marks indicate actual key errors.\nLine: P(correct rule present, other executed | current choice), before feedback.\nThese selected paths test event recognition; fixed paths estimate calibration.', fontsize=7)
    save(fig, output, 'rare_event_challenges')

    oral_summary = pd.read_csv(oral/'alignment_midpoint'/'summary.csv')
    selected = baselines.loc[baselines.reading.eq('midpoint') & baselines.window.eq('test') & baselines.subset.eq('complete')]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.2))
    fig.subplots_adjust(left=.10, right=.98, top=.93, bottom=.23, hspace=.68, wspace=.27)
    for r, sid in enumerate((129, 229, 317)):
        for d, domain in enumerate(('structural', 'functional')):
            ax = axes[r, d]; own = selected.loc[selected.subject.eq(sid) & selected.domain.eq(domain)].set_index('arm').loc[list(ARMS)]
            se = oral_summary.loc[oral_summary.subject.eq(sid) & oral_summary.window.eq('test') &
                oral_summary.subset.eq('complete') & oral_summary.domain.eq(domain)].set_index('arm').loc[list(ARMS)].support_mcse.to_numpy()
            xx = np.arange(4)
            ax.errorbar(xx-.07, own.dynamic_support, yerr=2*se, color=BLUE, fmt='o', ms=4, capsize=2, lw=.8)
            ax.plot(xx+.07, own.prefix_static_support, '^', color=ORANGE, ms=3.8)
            ax.plot(xx, own.choice_only_support, '_', color=GREY, ms=9)
            if not np.array_equal(own.reports.to_numpy(), own.static_reports.to_numpy()):
                raise ValueError('Main oral figure requires the same complete denominator for both baselines')
            ax.set_ylim(0, 1.05); ax.set_xticks(xx, ['MIX', 'MIX alt', 'STICKY', 'STICKY alt'])
            ax.set_title(f'{chr(97+r*2+d)}  Subject {sid}, {domain}; n={int(own.reports.iloc[0])}', loc='left', fontweight='bold')
            if d == 0:
                ax.set_ylabel('Content support mass')
            ax.grid(axis='y', color='.92', lw=.5)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, label='Dynamic; +/-2 filter MCSE'),
                        Line2D([], [], marker='^', ls='', color=ORANGE, label='Frozen prefix basis'),
                        Line2D([], [], marker='_', ls='', color=GREY, label='Choice-only catalogue')],
               loc='lower center', bbox_to_anchor=(.5, .105), ncol=3)
    fig.text(.10, .025, 'All complete suffix reports, including catalogue-unsupported content; midpoint interpretation.\nParameters selected with prefix behavior. Alternatives selected from training scores.\nContent compatibility is distinct from report probability and full latent-state accuracy.', fontsize=7)
    save(fig, output, 'oral_content_baselines')
    inputs = {str((source/name).resolve()): digest(source/name) for name in ('state_metrics.csv', 'state_trials.csv', 'oral_baselines.csv')}
    inputs[str((oral/'alignment_midpoint'/'summary.csv').resolve())] = digest(oral/'alignment_midpoint'/'summary.csv')
    write(output/'manifest.json', dict(status='rendered_pending_visual_review', source_sha256=digest(Path(__file__)),
        input_sha256=inputs, files=[f.name for f in sorted(output.glob('*.png'))], backend='python', dpi=300,
        export_policy='PNG only under standing repository instruction', exclusion='zero-event conditional scores labelled NA; enriched cases separate'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'oral', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.oral, args.output)
