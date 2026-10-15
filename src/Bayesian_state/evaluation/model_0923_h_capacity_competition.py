"""Explain proposal admission in the fixed-cardinality H intervention.

The previous trial's post-feedback evidence reconstructs the scores available
at each proposal. All proposals, including rejected ones, remain in the event
table. Expected admission uses the declared sampler; realized and expected
counts are compared at the complete-path level. Geometric compatibility is a
retrospective diagnostic defined by the screen's prespecified threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import load, digest
from ..model.modules.hypothesis_transition.single_proposal_competition import drop_probabilities


def proposal_events(arrays: dict, mechanism: dict, compatible_rules: list[int]) -> list[dict]:
    """Reconstruct candidate competition before each recorded transition."""
    a = arrays; capacity = a['rule'].shape[-1]; events = []
    use_mean = mechanism.get('score', 'sum') == 'mean'; bias = mechanism.get('stay_bias', 0.)
    for trial, path in zip(*np.where(a['proposed'] >= 0)):
        if trial == 0:raise ValueError('Initial state cannot contain a search proposal')
        present = a['rule'][trial-1, path] >= 0
        old = a['rule'][trial-1, path, present]
        evidence = a['retention_score_post'][trial-1, path, present]
        mean = a['retention_mean_post'][trial-1, path, present]
        scores = (mean if use_mean else evidence) + bias
        dropped = drop_probabilities(np.r_[0., scores], capacity)
        admitted, rejected = bool(a['admitted'][trial, path]), bool(a['rejected'][trial, path])
        if admitted == rejected:raise ValueError('Every proposal must be admitted or rejected')
        newcomer = int(a['proposed'][trial, path])
        after = a['rule'][trial, path]; after = after[after >= 0]
        if len(after) != min(capacity, len(old)+1) or admitted != bool(newcomer in after):
            raise ValueError('Recorded transition differs from fixed-cardinality admission')
        if len(old) < capacity and not admitted:raise ValueError('Spare capacity must admit the proposal')
        useful = newcomer in compatible_rules
        removed = old[~np.isin(old, after)]
        events.append(dict(path=int(path), seed=int(a['seeds'][path]), trial=int(trial+1),
            proposed=newcomer, compatible_proposal=useful, admitted=admitted, rejected=rejected,
            old_count=len(old), capacity=capacity, capacity_full=len(old)==capacity,
            admission_probability=float(1-dropped[0]),
            expected_compatible_removal=float(dropped[1:][np.isin(old, compatible_rules)].sum()),
            compatible_removed=int(np.isin(removed, compatible_rules).sum()),
            old_evidence_min=float(evidence.min()), old_mean_evidence_min=float(mean.min()),
            old_score_min=float(scores.min()), old_score_max=float(scores.max())))
    return events


def summarize(source: Path, output: Path) -> None:
    import json
    if output.exists():raise FileExistsError('Use a new admission-diagnostic destination')
    manifest = load(source/'simulation_manifest.json'); competition = load(source/'competition_manifest.json')
    if manifest['status']!='complete' or manifest['smoke'] or competition['status']!='complete':
        raise ValueError('Need a complete scientific fixed-cardinality screen')
    repository = Path(__file__).resolve().parents[3]
    helper = 'src/Bayesian_state/model/modules/hypothesis_transition/single_proposal_competition.py'
    if digest(repository/helper) != competition['source_sha256'][helper]:
        raise ValueError('Admission sampler differs from the recorded experiment')
    recorded = pd.read_csv(source/'paths.csv').set_index(['id','variant','seed'])
    all_events, paths, hashes = [], [], {}
    for record in manifest['records']:
        directory = source/record['id'];catalog = pd.read_csv(directory/'rule_catalog.csv')
        useful = catalog.loc[catalog.physical_argmax_accuracy>=manifest['settings']['geometric_accuracy_threshold'],'rule'].astype(int).tolist()
        for model in competition['fixed_size_variants']:
            mechanism = manifest['settings']['variants'][model]
            for file in sorted((directory/model/'traces').glob('C_*.npz')):
                check = load(file.with_suffix('.json'))
                if digest(file)!=check['sha256']:raise ValueError('Recorded paths changed')
                hashes[str(file.relative_to(source))] = check['sha256']
                with np.load(file) as saved:a = {key:saved[key] for key in saved.files}
                events = proposal_events(a, mechanism, useful)
                tag = dict(id=record['id'],subject=record['subject'],model=model)
                all_events.extend(dict(event,**tag) for event in events)
                for path,seed in enumerate(a['seeds']):
                    selected = [event for event in events if event['path']==path]
                    compatible = [event for event in selected if event['compatible_proposal']]
                    full = [event for event in selected if event['capacity_full']]
                    row = dict(tag,seed=int(seed),proposals=len(selected),admitted=sum(e['admitted'] for e in selected),
                        compatible_proposed=len(compatible),compatible_admitted=sum(e['admitted'] for e in compatible),
                        expected_admitted=sum(e['admission_probability'] for e in selected),
                        expected_compatible_admitted=sum(e['admission_probability'] for e in compatible),
                        full_capacity_proposals=len(full),full_capacity_admitted=sum(e['admitted'] for e in full),
                        expected_full_capacity_admitted=sum(e['admission_probability'] for e in full),
                        compatible_removed=sum(e['compatible_removed'] for e in selected),
                        expected_compatible_removed=sum(e['expected_compatible_removal'] for e in selected))
                    reference = recorded.loc[(record['id'],model,int(seed))]
                    for left,right in [('proposals','proposed'),('admitted','admitted'),('compatible_proposed','useful_proposed'),('compatible_admitted','useful_admitted')]:
                        if row[left]!=reference[right]:raise ValueError('Admission counts disagree with independent path summary')
                    paths.append(row)
    output.mkdir(parents=True)
    pd.DataFrame(all_events).to_csv(output/'proposal_events.csv',index=False)
    frame = pd.DataFrame(paths); frame.to_csv(output/'paths.csv',index=False)
    summaries = []
    for (subject,model),group in frame.groupby(['subject','model'],sort=False):
        row = dict(subject=subject,model=model,paths=len(group))
        for name in ['proposals','admitted','compatible_proposed','compatible_admitted','full_capacity_proposals',
                     'full_capacity_admitted','expected_full_capacity_admitted','compatible_removed','expected_compatible_removed']:
            row[name] = float(group[name].sum())
        for numerator,denominator,label in [('admitted','proposals','admission_fraction'),
            ('compatible_admitted','compatible_proposed','compatible_admission_fraction'),
            ('full_capacity_admitted','full_capacity_proposals','full_capacity_admission_fraction'),
            ('expected_full_capacity_admitted','full_capacity_proposals','expected_full_capacity_admission_fraction')]:
            row[label] = row[numerator]/row[denominator] if row[denominator] else np.nan
        residual = (group.admitted-group.expected_admitted).to_numpy()
        row.update(admission_residual_per_path=float(residual.mean()),
            admission_residual_mc_se=float(residual.std(ddof=1)/np.sqrt(len(residual))))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(output/'participants.csv',index=False)
    with (output/'completion.json').open('x') as handle:
        json.dump(dict(status='complete',source=str(source.resolve()),source_sha256=digest(Path(__file__)),
            sampler_sha256=competition['source_sha256'][helper],trace_sha256=hashes,events=len(all_events),paths=len(paths),
            interpretation='fixed historical parameters; complete-path Monte Carlo residuals; event ratios reported with denominators'),handle,indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args();summarize(args.source,args.output)


if __name__=='__main__':main()
