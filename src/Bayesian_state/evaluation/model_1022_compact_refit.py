"""Summarize paired COMPACT/TRANSFER behavior and fixed oral constraints.

Use --study REFIT --evaluation EVALUATION --config YAML --output NEW. The
original six-person case plan and all frozen estimates must match. Shared
scoring functions retain chronological boundaries, full autonomous paths,
complete-report denominators and whole-seed Monte Carlo error definitions.
Development and extension groups are reported separately. These are comparison
scores at fitted parameters, not population-equivalence or parameter intervals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model_0923_h_validation import checked_arrays, block_paths, distribution_scores, jackknife_scores, jackknife_se, nll
from .model_0923_h_generation_uncertainty import path_jackknife
from ..workflows.analysis import model_0923_decision_basis_alignment as alignment
from ..workflows.analysis.model_1021_simplification import ROOT, io
from ..workflows.analysis.model_1021_simplification_recovery import compact_zero_effect_coordinates

read, write, digest = io.read, io.write, io.digest
ARMS = ('COMPACT', 'TRANSFER')


def training_budget(case: dict, bind, stages: tuple[str, ...] | None = None) -> list[dict]:
    """Verify actual parameter points, seed banks and filter precision per stage.

    An explicit stage subset supports an audit while later stages are running.
    The final comparison uses every registered stage and all its job records.
    """
    settings=case['settings'];fit=Path(case['output'])
    new=settings['elites']*(settings['rays_per_elite']+settings['joint_per_elite'])
    expected=dict(screen=settings['initial_candidates'],selection=settings['selection_candidates'],
                  confirmation=settings['confirmation_candidates'])
    expected.update({f'refine_{i}':new+settings['carry_candidates'] for i in range(1,settings['rounds']+1)})
    selected=set(expected) if stages is None else set(stages)
    if not selected or selected-set(expected):
        raise ValueError('Budget audit needs registered stages')
    data=ROOT/settings['data']/f'subject_{case["subject"]}'/'prepared.json';bind(data)
    cut=read(data)['training_trials']
    seed_banks=defaultdict(lambda:defaultdict(list));records=[]
    for file in sorted((fit/'jobs').glob('*.json')):
        job=read(file);stage=job['name'].split('_S',1)[0]
        if stage not in selected:
            if stages is None: raise ValueError('Unregistered training stage')
            continue
        score=Path(job['target']).with_suffix('.json');row=read(score)
        bind(file);bind(score)
        budget=settings['budgets'][stage]
        if (job['compact_variant']!=case['variant'] or job['phase']!='training' or len(job['seeds'])!=1
                or row['seeds']!=job['seeds'] or row['candidate']!=job['candidate']
                or row['subject']!=case['subject'] or row['condition']!=case['condition']
                or job['particles']!=budget['particles'] or row['particles']!=budget['particles']
                or row['trials']!=cut or any(p['num_threads']!=1 for p in row['numeric_pools'])):
            raise ValueError('Recorded training filter differs from its declared budget or identity')
        key=job['candidate']['id'];seed_banks[stage][key].extend(job['seeds'])
        records.append(dict(case_id=case['case_id'],subject=case['subject'],model=case['variant'],
            cohort=case.get('cohort','synthetic_recovery'),
            condition=case['condition'],stage=stage,candidate=key,seed=job['seeds'][0],filters=1,
            particles=row['particles'],trials=row['trials'],cpu_seconds=row['cpu_seconds'],seconds=row['seconds']))
    for stage in selected:
        file=fit/(stage+'_completion.json');bind(file);complete=read(file)
        group,=complete['groups'];points=group['rows']
        if (complete['status']!='complete' or group['subject']!=case['subject']
                or len(points)!=expected[stage] or len(seed_banks[stage])!=expected[stage]
                or {p['id'] for p in points}!=set(seed_banks[stage])):
            raise ValueError('Actual candidate count differs from the common search plan')
        seeds=[tuple(sorted(v)) for v in seed_banks[stage].values()]
        if (len(set(seeds))!=1 or len(seeds[0])!=settings['budgets'][stage]['seeds']
                or len(set(seeds[0]))!=len(seeds[0])):
            raise ValueError('Training seed counts or common seed banks differ')
    if stages is None:
        unique={k for bank in seed_banks.values() for k in bank}
        if len(unique)!=settings['initial_candidates']+settings['rounds']*new:
            raise ValueError('Distinct evaluated points differ from the declared search budget')
    return records


def zero_learning_equivalence(cases: list[dict], bind) -> pd.DataFrame:
    """Audit the exact zero-learning boundary on all matched initial banks.

    With learning rate zero every active precision stays at its initial value,
    so transferring that value reproduces neutral entry. The saved full-length
    filters test the observable implication without running another filter.
    """
    rows=[]
    humans=[c for c in cases if c['kind']=='human']
    for subject in sorted({c['subject'] for c in humans}):
        banks={}
        for variant in ARMS:
            case,=[c for c in humans if c['subject']==subject and c['variant']==variant]
            file=Path(case['output'])/'screen_completion.json';bind(file)
            stage=read(file);group,=stage['groups']
            if stage['status']!='complete' or group['subject']!=subject or group['model']!='COMPACT':
                raise ValueError('Require matching completed initial parameter banks')
            banks[variant]={r['id']:r for r in group['rows']}
        if set(banks['COMPACT'])!=set(banks['TRANSFER']):
            raise ValueError('Initial parameter points differ between variants')
        for key,a in banks['COMPACT'].items():
            if a['hyperparams']['engine.modules.beta_mod.kwargs.learning_rate']!=0:
                continue
            b=banks['TRANSFER'][key]
            if a['hyperparams']!=b['hyperparams']:
                raise ValueError('Paired point identifiers have different parameters')
            saved=[]
            for point in (a,b):
                file=Path(point['score_file']);bind(file);record=read(file)
                array=file.with_suffix('.npz');bind(array,record['predictions_sha256'])
                if (record['subject']!=subject or record['model']!='COMPACT'
                        or record['particles']!=case['settings']['budgets']['screen']['particles']
                        or record['candidate']['hyperparams']!=point['hyperparams']):
                    raise ValueError('Zero-learning comparison has a different parameter identity')
                with np.load(array) as z:
                    saved.append({k:z[k].copy() for k in ('seeds','observed_probability','correct_probability')})
            for name in saved[0]:
                np.testing.assert_array_equal(saved[0][name],saved[1][name])
            shape=saved[0]['observed_probability'].shape
            if shape[0]!=case['settings']['budgets']['screen']['seeds']:
                raise ValueError('Zero-learning comparison has a different filter budget')
            rows.append(dict(subject=subject,cohort=case['cohort'],condition=case['condition'],candidate=key,
                beta_init=a['hyperparams']['engine.modules.beta_mod.kwargs.beta_init'],
                particles=record['particles'],filters=shape[0],trials=shape[1],
                matched_filter_trials=int(np.prod(shape)),exact_arrays=True))
    if not rows:
        raise ValueError('Initial banks contain no zero-learning controls')
    return pd.DataFrame(rows)


def paired_contrast(points: dict, jackknives: dict, metrics: tuple[str, ...]) -> dict:
    """Difference of matched variants, with a whole-seed paired jackknife."""
    row = {}
    for metric in metrics:
        row['delta_'+metric] = float(points['TRANSFER'][metric]-points['COMPACT'][metric])
        row['delta_'+metric+'_mc_se'] = jackknife_se(jackknives['TRANSFER'][metric]-jackknives['COMPACT'][metric])
    return row


def confirmation_candidates(case: dict, bind, excluded_seeds: tuple[int, ...] | np.ndarray=()) -> list[dict]:
    """Read every confirmed candidate relative to the frozen nominee.

    Differences use matched whole-filter seeds within confirmation. They
    describe numerical uncertainty at evaluated points, not parameter intervals.
    """
    fit=Path(case['output']);file=fit/'nominees.json';bind(file)
    frozen=read(file);nominee,=frozen['nominees']
    expected_status='frozen_before_test' if case['kind']=='human' else 'frozen_before_truth'
    if frozen['status']!=expected_status or nominee['compact_variant']!=case['variant'] or frozen['smoke']:
        raise ValueError('Require the matching frozen scientific nominee')
    budget=case['settings']['budgets']['confirmation']
    prepared=ROOT/case['settings']['data']/f'subject_{case["subject"]}'/'prepared.json';bind(prepared)
    trials=read(prepared)['training_trials'];points=nominee['finalists']
    if len(points)!=case['settings']['confirmation_candidates'] or len({p['id'] for p in points})!=len(points):
        raise ValueError('Missing or duplicate confirmation candidates')
    estimates={};common_seeds=None
    for point in points:
        file=Path(point['score_file']);bind(file);record=read(file)
        bind(file.with_suffix('.npz'),record['predictions_sha256'])
        if (record['subject']!=case['subject'] or record['condition']!=case['condition']
                or record['model']!='COMPACT' or record['role']!='training_only_high_particle_audit'
                or record['particles']!=budget['particles']
                or record['candidate']['hyperparams']!=point['hyperparams']
                or record['candidate']['id']!=point['id'] or record['trials']!=trials):
            raise ValueError('Confirmation identity or numerical budget differs')
        with np.load(file.with_suffix('.npz')) as z:
            probability,seeds=z['observed_probability'],z['seeds']
        if (probability.shape!=(budget['seeds'],trials) or len(np.unique(seeds))!=budget['seeds']
                or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1))):
            raise ValueError('Invalid confirmation probabilities or seed count')
        np.testing.assert_array_equal(seeds,record['seeds'])
        if common_seeds is not None:np.testing.assert_array_equal(seeds,common_seeds)
        common_seeds=seeds
        if set(seeds)&set(excluded_seeds):raise ValueError('Confirmation and evaluation seeds overlap')
        value=nll(probability)
        np.testing.assert_allclose(value,[point['mean_nll'],record['mean_nll']],atol=1e-12,rtol=0)
        estimates[point['id']]=value,jackknife_scores(probability)
    chosen=nominee['candidate']['id'];reference,deleted=estimates[chosen]
    np.testing.assert_allclose(reference,min(v[0] for v in estimates.values()),atol=1e-12,rtol=0)
    rows=[]
    for rank,(key,(value,own_deleted)) in enumerate(sorted(estimates.items(),key=lambda p:(p[1][0],p[0])),1):
        rows.append(dict(candidate=key,selected=key==chosen,reference=key==nominee['reference_id'],
            confirmation_rank=rank,nll=value,mc_se=jackknife_se(own_deleted),delta_nll=value-reference,
            paired_mc_se=jackknife_se(own_deleted-deleted),particles=budget['particles'],filter_seeds=len(common_seeds)))
    return rows


def summarize(study: Path, evaluation: Path, config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new refit-summary directory')
    policy = yaml.safe_load(config.read_text())
    all_cases = read(study/'cases.json')['cases']
    cases = [c for c in all_cases if c['kind']=='human']
    done, protocol = read(evaluation/'completion.json'), read(evaluation/'protocol.json')
    if (done['status']!='complete' or done['smoke'] or len(cases)!=12
            or protocol['cases_sha256']!=digest(study/'cases.json')
            or protocol['frozen_human_sha256']!=digest(study/'human_nominees.json')):
        raise ValueError('Require completed evaluation of all frozen human fits')
    if {r['case_id'] for r in done['cases']}!={c['case_id'] for c in cases}:
        raise ValueError('Evaluation case set differs from the registered fits')
    inputs = {}

    def bind(file: Path, expected: str | None = None):
        value = digest(file)
        if expected is not None and value!=expected:
            raise ValueError('Summary input changed: '+str(file))
        inputs[str(file.resolve())] = value

    for file in (config,Path(__file__),Path(alignment.__file__),
                 Path(inspect.getfile(block_paths)),Path(inspect.getfile(path_jackknife)),
                 study/'cases.json',study/'human_nominees.json',
                 evaluation/'completion.json',evaluation/'protocol.json'):
        bind(file)
    scores, curves, parameters, finalists, budgets = [], [], [], [], []
    jacks, paired_seeds, oral_groups = {}, {}, {}
    for case in cases:
        subject, variant, cohort = case['subject'],case['variant'],case['cohort']
        fit, own = Path(case['output']),evaluation/case['case_id']
        frozen = read(fit/'nominees.json'); nominee, = frozen['nominees']
        own_protocol, complete = read(own/'protocol.json'),read(own/'completion.json')
        if (frozen['smoke'] or frozen['status']!='frozen_before_test' or complete['status']!='complete'
                or complete['smoke'] or own_protocol['nomination_sha256']!=digest(fit/'nominees.json')
                or nominee['compact_variant']!=variant or complete['compact_variant']!=variant
                or own_protocol['compact_variant']!=variant or own_protocol['case_id']!=case['case_id']):
            raise ValueError('Per-case evaluation differs from its frozen variant')
        for file in (fit/'nominees.json',fit/'protocol.json',own/'protocol.json',own/'completion.json'):
            bind(file)
        settings = frozen['settings']
        banks, records = {}, {}
        for role in ('conditional','autonomous'):
            file = own/role/'evaluation.json'; bind(file)
            record, = read(file)['rows']
            if (record['candidate']['hyperparams']!=nominee['candidate']['hyperparams']
                    or record['subject']!=subject or record['condition']!=case['condition']
                    or record['model']!='COMPACT'):
                raise ValueError('Evaluated parameters differ from selected parameters')
            if role=='conditional' and record['particles']!=settings['budgets']['evaluate']['particles']:
                raise ValueError('Conditional evaluation uses a different particle budget')
            records[role] = record
            banks[role] = checked_arrays(own/role,record)
            array_file = own/role/'evaluate'/f'subject_{subject}'/'COMPACT'/(record['candidate']['id']+'.npz')
            bind(array_file,record['predictions_sha256'])
        c,a = banks['conditional'],banks['autonomous']
        np.testing.assert_array_equal(c['categories'],a['categories'])
        np.testing.assert_allclose(c['probabilities'].sum(-1),1,atol=1e-12,rtol=0)
        np.testing.assert_allclose(a['probabilities'].sum(-1),1,atol=1e-12,rtol=0)
        cut,length = records['conditional']['training_trials'],len(c['choices'])
        if (a['choices'].shape!=(settings['autonomous_paths'],length)
                or c['observed_probability'].shape!=(settings['budgets']['evaluate']['seeds'],length)):
            raise ValueError('Missing filters or autonomous paths')
        paired_seeds[subject,variant] = c['seeds'],a['seeds']
        actual = c['choices']==c['categories']
        generated = a['choices']==a['categories'][None,:]
        tag = dict(subject=subject,model=variant,cohort=cohort,condition=case['condition'])
        for phase,first,last in [('training',0,cut),('test',cut,length),('whole',0,length)]:
            probability = c['observed_probability'][:,first:last]
            samples = block_paths(generated,first,last,settings['features']['block_width'])
            observed = block_paths(actual[None,:],first,last,settings['features']['block_width'])[0]
            jk = dict(choice_nll=jackknife_scores(probability),**path_jackknife(samples,observed))
            jacks[subject,variant,phase] = jk
            late = max(first,last-128)
            late_paths = generated[:,late:last].mean(1)
            observed_late = float(actual[late:last].mean())
            # This rank locates one observed sequence within the generated
            # sample. It is descriptive, with half weight for discrete ties.
            observed_rank = float(np.mean(late_paths < observed_late)
                                  + .5*np.mean(late_paths == observed_late))
            late_quantiles = np.quantile(late_paths,[.1,.5,.9])
            scores.append(dict(tag,phase=phase,trials=last-first,training_trials=cut,
                choice_nll=nll(probability),choice_nll_mc_se=jackknife_se(jk['choice_nll']),
                **distribution_scores(samples,observed),energy_mc_se=jackknife_se(jk['energy']),
                late_accuracy=float(late_paths.mean()),observed_late=observed_late,
                late_accuracy_q10=late_quantiles[0],late_accuracy_q50=late_quantiles[1],
                late_accuracy_q90=late_quantiles[2],observed_late_midrank=observed_rank,
                target_ever=float(a['present'][:,:last].any(1).mean()),
                target_late_present=float(a['present'][:,late:last].mean()),
                target_late_contribution=float(a['contribution'][:,late:last].mean()),
                mean_search=float(a['search'][:,first:last].mean()),
                target_capacity_drops=float(a['target_capacity_dropped'][:,first:last].sum(1).mean())))
            if phase=='whole':
                lo,hi = np.quantile(samples,[.1,.9],axis=0)
                conditional = block_paths(c['correct_probability'],0,length,settings['features']['block_width']).mean(0)
                curves.extend(dict(tag,trial=min((i+1)*settings['features']['block_width'],length),training_trials=cut,
                    observed=observed[i],autonomous=samples[:,i].mean(),lower=lo[i],upper=hi[i],conditional=conditional[i])
                    for i in range(len(observed)))
        zero_effect = compact_zero_effect_coordinates(nominee['candidate']['hyperparams'])
        parameters.extend(dict(tag,parameter=k,value=v,fitted=len(case['space'][k])>1,
                               known_zero_effect=k in zero_effect)
                          for k,v in nominee['candidate']['hyperparams'].items())
        finalists.extend(dict(tag,**row) for row in confirmation_candidates(case,bind,c['seeds']))
        budgets.extend(training_budget(case,bind))
        captured = []
        for file in sorted((own/'state_capture').glob('*.json')):
            record = read(file);job = read(Path(record['original_job']))
            if (job['compact_variant']!=variant or job['source_sha256']!=frozen['source_sha256']
                    or record['candidate']!=nominee['candidate'] or record['particles']!=settings['budgets']['evaluate']['particles']):
                raise ValueError('Oral state was captured under another fitted variant')
            alignment.validate_basis_record(file.with_suffix('.npz'),record)
            bind(file);bind(file.with_suffix('.npz'),record['array_sha256'])
            # A display label identifies the declared variant; source records remain unchanged.
            captured.append((file,dict(record,arm=variant)))
        if len(captured)!=settings['budgets']['evaluate']['seeds']:
            raise ValueError('Incomplete original-filter oral captures')
        np.testing.assert_array_equal(sorted(r['seeds'][0] for _,r in captured),sorted(c['seeds']))
        oral_groups[subject,variant] = own,captured,cohort
    for case in all_cases:
        if case['kind']=='recovery':
            budgets.extend(training_budget(case,bind))
    table = pd.DataFrame(scores)
    contrasts = []
    for subject,group in table.groupby('subject'):
        for i in (0,1):
            np.testing.assert_array_equal(paired_seeds[subject,'COMPACT'][i],paired_seeds[subject,'TRANSFER'][i])
        for phase,subset in group.groupby('phase'):
            contrasts.append(dict(subject=subject,cohort=subset.cohort.iloc[0],phase=phase,
                **paired_contrast(subset.set_index('model').to_dict('index'),
                    {a:jacks[subject,a,phase] for a in ARMS},('choice_nll','energy','crps'))))
    contrasts = pd.DataFrame(contrasts)
    aggregate = []
    for (cohort,phase),group in contrasts.groupby(['cohort','phase']):
        row = dict(cohort=cohort,phase=phase,subjects=len(group))
        for metric in ('choice_nll','energy','crps'):
            row['delta_'+metric] = group['delta_'+metric].mean()
            row['delta_'+metric+'_mc_se'] = np.linalg.norm(group['delta_'+metric+'_mc_se'])/len(group)
        aggregate.append(row)
    equivalence=zero_learning_equivalence(cases,bind)
    output.mkdir(parents=True)
    for name,frame in dict(scores=table,contrasts=contrasts,aggregate=pd.DataFrame(aggregate),curves=pd.DataFrame(curves),
            parameters=pd.DataFrame(parameters),finalists=pd.DataFrame(finalists),compute_budgets=pd.DataFrame(budgets),
            zero_learning_equivalence=equivalence).items():
        frame.to_csv(output/(name+'.csv'),index=False)
    for reading in ('midpoint','relative'):
        frames, summaries, estimates = [], [], {}
        for cohort,audits in policy['oral_audits'].items():
            audit = ROOT/audits[reading]; manifest = read(audit/'manifest.json')
            if manifest['status']!='complete':
                raise ValueError('Oral audit is incomplete')
            bind(audit/'manifest.json')
            for name,sha in manifest['input_sha256'].items():
                bind(ROOT/name,sha)
            for name in ('trials.csv','component_matches.csv'):
                bind(audit/name,manifest['artifacts'][name])
            oral,matches = pd.read_csv(audit/'trials.csv'),pd.read_csv(audit/'component_matches.csv')
            for (subject,variant),(own,records,group_cohort) in sorted(oral_groups.items()):
                if group_cohort!=cohort:
                    continue
                trial,rows,jk = alignment.group_scores(own,records,oral,matches)
                frames.append(trial.assign(cohort=cohort))
                summaries.extend(dict(r,cohort=cohort) for r in rows)
                estimates[subject,variant] = jk
        oral_contrasts = []
        for subject in sorted({s for s,a in estimates}):
            cohort = next(c['cohort'] for c in cases if c['subject']==subject)
            for key,a in estimates[subject,'TRANSFER'].items():
                b = estimates[subject,'COMPACT'][key]
                np.testing.assert_array_equal(a['seeds'],b['seeds'])
                row = dict(subject=subject,cohort=cohort,window=key[0],domain=key[1],subset=key[2])
                for i,label in enumerate(alignment.LABELS):
                    row[label+'_difference'] = a['point'][i]-b['point'][i]
                    row[label+'_difference_mcse'] = jackknife_se(a['values'][:,i]-b['values'][:,i])
                oral_contrasts.append(row)
        own = output/('oral_'+reading); own.mkdir()
        for name,frame in [('trials',pd.concat(frames,ignore_index=True)),('summary',pd.DataFrame(summaries)),
                           ('contrasts',pd.DataFrame(oral_contrasts))]:
            frame.to_csv(own/(name+'.csv'),index=False)
        print('Completed oral '+reading,flush=True)
    write(output/'completion.json',dict(status='complete',study=str(study.resolve()),evaluation=str(evaluation.resolve()),
        subjects=sorted(table.subject.unique().tolist()),inputs_sha256=inputs,working_tolerances=policy['working_tolerances'],
        comparison='TRANSFER minus COMPACT; both jointly refitted with equal budgets',
        cohorts='Development and extension reported separately; extension prior influence unknown',
        uncertainty='Paired whole-seed MCSE at fitted parameters, excluding parameter and participant uncertainty',
        oral='Noisy content compatibility, fixed dual readings, unchanged complete-report denominators',
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','evaluation','config','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    summarize(args.study,args.evaluation,args.config,args.output)
