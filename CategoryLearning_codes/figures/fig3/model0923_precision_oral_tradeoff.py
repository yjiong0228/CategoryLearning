"""Compare fixed-parameter precision retention across behavior and oral evidence.

Use completed choice scores, autonomous contrasts, midpoint/relative oral
comparisons and their independent availability validation. Export PNG and source
tables to a new directory. Error bars are one paired Monte Carlo SE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(choice: Path, autonomous: Path, midpoint: Path, relative: Path,
        validation: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new precision/oral figure directory')
    hashes = {}

    def record(path: Path, expected: str | None = None) -> str:
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and value != expected:
            raise ValueError(f'Changed figure input: {path.name}')
        hashes[str(path.resolve())] = value
        return value

    def read(path: Path) -> dict:
        record(path)
        return json.loads(path.read_text())

    receipt = read(validation / 'validation.json')
    if receipt['status'] != 'complete':
        raise ValueError('Need independent oral-score validation')
    capacity = receipt['capacity']
    tables, identity = {}, None
    for reading, folder in [('Midpoint', midpoint), ('Relative', relative)]:
        manifest = read(folder / 'manifest.json')
        if (manifest['status'] != 'complete' or manifest['contrast_kind'] != 'precision_retention'
                or manifest['seed_relation'] != 'paired' or manifest['subject'] != 208):
            raise ValueError('Need paired fixed-parameter S208 precision contrasts')
        candidate = manifest['revised_parameters']['candidate']
        point = candidate['id']
        if identity is not None and identity != candidate['hyperparams']:
            raise ValueError('Oral readings must use the same behavior parameter point')
        identity = candidate['hyperparams']
        if identity['engine.modules.hypo_transitions_mod.kwargs.max_capacity'] != capacity:
            raise ValueError('Validation and oral parameter capacities differ')
        file = folder / 'scores.csv'
        record(file, manifest['artifacts']['scores.csv'])
        if receipt['input_sha256'].get(str(file.resolve())) != hashes[str(file.resolve())]:
            raise ValueError('Oral comparison differs from the independently validated source')
        tables[reading] = pd.read_csv(file)
    for folder, filename in [(choice, 'scores.csv'), (autonomous, 'contrasts.csv')]:
        completed = read(folder / 'completion.json')
        if completed['status'] != 'complete' or completed.get('smoke', False):
            raise ValueError('Need completed scientific behavior comparisons')
        record(folder / filename, completed['artifacts'][filename])
    if (read(choice / 'completion.json')['filter_seeds'] != 8
            or read(autonomous / 'protocol.json')['settings']['paths'] != 64):
        raise ValueError('This figure contract requires eight filters and 64 autonomous paths')
    choices = pd.read_csv(choice / 'scores.csv')
    choices = choices[(choices.subject == 208) & (choices.point == point)].set_index('phase').loc[['prefix', 'later', 'full']]
    generation = pd.read_csv(autonomous / 'contrasts.csv')
    generation = generation[(generation.subject == 208) & (generation.point == point)
                            & (generation.window == 'whole') & (generation.metric == 'energy')]
    if len(choices) != 3 or len(generation) != 1:
        raise ValueError('Need all three choice windows and one whole-path Energy contrast')
    record(validation / 'availability_contrasts.csv')
    availability = pd.read_csv(validation / 'availability_contrasts.csv')
    rows = []
    for phase, row in choices.iterrows():
        rows.append(dict(panel='a',reading='Behavior',metric=phase,value=row.delta_choice_nll,se=row.delta_choice_nll_mc_se))
    row = generation.iloc[0]
    rows.append(dict(panel='b',reading='Behavior',metric='Whole',value=row.difference,se=row.mcse))
    reports = None
    for reading, table in tables.items():
        selected = table[(table.window == 'test') & (table.domain == 'structural') & (table.subset == 'complete')]
        if len(selected) != 1:
            raise ValueError('Need the complete-report later window')
        row = selected.iloc[0]
        if reports is not None and reports != row.reports:
            raise ValueError('Both readings must preserve the report denominator')
        reports = int(row.reports)
        for metric in ['support', 'conflict', 'undetermined']:
            rows.append(dict(panel='c',reading=reading,metric=metric,value=100*row[metric+'_difference'],se=100*row[metric+'_difference_mcse']))
        for bound in ['all_components', 'any_component']:
            subset = availability[(availability.reading == ('midpoint' if reading == 'Midpoint' else 'relative'))
                                  & (availability.window == 'test') & (availability.domain == 'structural')
                                  & (availability.subset == 'complete') & (availability.bound == bound)]
            if len(subset) != 1 or int(subset.iloc[0].reports) != reports:
                raise ValueError('Availability must use the same complete reports')
            av = subset.iloc[0]
            rows.append(dict(panel='d',reading=reading,metric=bound,value=100*av.difference,se=100*av.difference_mcse))
    source = pd.DataFrame(rows)
    if not np.isfinite(source[['value','se']]).all().all() or (source.se < 0).any():
        raise ValueError('Need finite contrasts and nonnegative errors')
    output.mkdir(parents=True)
    source.to_csv(output/'plot_source.csv', index=False)
    contract = dict(claim='Behavior and autonomous effects are assessed alongside oral support and rule availability.',
        backend='python', archetype='quantitative grid', subject=208, capacity=capacity, parameter_id=point,
        width_mm=183, height_mm=138, dpi=300, formats=['PNG'], complete_later_reports=reports,
        statistics='RETAIN minus RESET; one paired whole-filter/path Monte Carlo SE; eight filters and 64 paths per mechanism.',
        data_use='One development participant; fixed behavior-selected parameters; two fixed oral readings; oral excluded from fitting.',
        selection='All choice windows; complete reports in original later window; whole-sequence autonomous Energy. Other windows remain in source comparisons.',
        source_preflight_exceptions={'EXPORT-VECTOR':'Repository PNG-only rule.','EXPORT-RASTER':'Repository PNG-only rule.','RASTER-DPI':'300 dpi discussion figure.'},
        inputs_sha256=hashes)
    (output/'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'], 'font.size':7,
        'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 138/25.4))
    fig.subplots_adjust(left=.12,right=.97,bottom=.17,top=.84,hspace=.68,wspace=.5)
    titles = ['Choice sequence: lower is better','Autonomous Energy: lower is better',
              'Oral content attribution: later reports','Supporting-rule availability: later reports']
    colors = {'Midpoint':'#68778B','Relative':'#218C88'}
    for panel, ax, title in zip('abcd', axes.flat, titles):
        own = source[source.panel == panel]
        labels = own.metric.drop_duplicates().tolist()
        if panel in 'ab':
            ax.errorbar(np.arange(len(own)),own.value,yerr=own.se,fmt='o',color='#405C77',markersize=4,capsize=3)
        else:
            for offset, (reading, color) in zip([-.10,.10], colors.items()):
                part=own[own.reading == reading]
                ax.errorbar(np.arange(len(part))+offset,part.value,yerr=part.se,fmt='o' if reading=='Midpoint' else 's',
                            color=color,markersize=4,capsize=3,label=reading)
        labels=[{'prefix':'Prefix 384','later':'Later 192','full':'Whole 576','all_components':'All components',
                 'any_component':'Any component'}.get(label,label.capitalize()) for label in labels]
        ax.set(xticks=np.arange(len(labels)),xticklabels=labels,xlim=(-.45,len(labels)-.55),title=title,
               ylabel='Difference (percentage points)' if panel in 'cd' else 'NLL difference' if panel=='a' else 'Energy difference')
        ax.axhline(0,color='#999999',linestyle=':',linewidth=.8)
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.17,1.08,panel,transform=ax.transAxes,fontsize=8,weight='bold')
        ax.tick_params(length=2.5,width=.6)
    axes[1,0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.31),ncol=2)
    fig.suptitle(f'Subject 208, fixed K={capacity}: precision retention across three forms of evidence',fontsize=9,y=.97)
    fig.text(.12,.915,'All differences: RETAIN − RESET. Oral readings use the same frozen parameters.',fontsize=7)
    fig.text(.12,.075,f'Bars: ± 1 Monte Carlo SE; 8 filters / mechanism; 64 autonomous paths / mechanism; {reports} later complete reports.',fontsize=6)
    fig.text(.12,.043,'Development diagnostic; simulation errors exclude parameter and population uncertainty. Oral support is content mass.',fontsize=6)
    fig.savefig(output/'precision_oral_tradeoff.png',dpi=300)
    plt.close(fig)
    (output/'plot_source.py').write_text(Path(__file__).read_text())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('choice','autonomous','midpoint','relative','validation','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
