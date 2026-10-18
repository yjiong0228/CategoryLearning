"""Compare completed RETAIN and ONLINE behavioral refits.

Use --retained-summary OLD --online-summary NEW --output NEW_DIRECTORY after
both episodic-refit summaries exist. The declared search policy is checked
against the RETAIN reference. Actual exchange work is reported separately.
Choice scores, autonomous distributions and feedback strata use identical
task trials and paired complete random replicates. This comparison includes
the belief mechanism and behavioral re-estimation together; oral is excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_0923_reentry_refit as retained
from .model_0923_estimation_scope import paired_difference, validate_generation
from .model_0923_h_validation import checked_arrays, digest, jackknife_scores, jackknife_se, load, nll
from .model_0923_refit_budget import audit_exchange, budget_contrasts
from ..metrics.task import category_learning_feedback
from ..workflows.analysis import model_0923_online_evidence_refit as refit
from ..workflows.analysis.model_0923_h_validation import write


def check_online_execution(job: dict, record: dict, execution: dict, fixed: dict) -> None:
    """Require the intervention's actual update count for filters or own-choice paths."""
    phase = job.get('phase')
    copies = len(job['seeds'])*(1 if phase == 'autonomous' else job['particles'])
    wrong_replicates = (record.get('paths') != len(job['seeds']) if phase == 'autonomous'
                        else record.get('seeds') != job['seeds'])
    if (phase not in ('evaluate', 'autonomous') or job.get('online_belief') is not True
            or job.get('belief_intervention') != refit.INTERVENTION
            or job.get('fixed_coordinates') != fixed
            or any(job['candidate']['hyperparams'].get(k) != v for k,v in fixed.items())
            or execution.get('enabled') is not True
            or execution.get('belief_intervention') != refit.INTERVENTION
            or execution.get('fixed_coordinates') != fixed
            or execution.get('exact_transition_timing') is not True
            or execution.get('evidence_matches_retained_state') is not True
            or wrong_replicates or record['candidate'] != job['candidate']
            or execution.get('reconstructions') != copies*(record['trials']-1)):
        raise ValueError('ONLINE execution differs from the nominated intervention or update count')


def bind_execution(manifest: dict, frozen: dict, online: bool) -> dict:
    """Bind executed jobs and both intervention sidecars, including autonomous runs."""
    settings = frozen['settings']
    retention = refit.helpers()
    references = (refit.reference_inputs(retention,retention.helpers(),settings) if online
                  else retained.refit.reference_inputs(retained.refit.helpers(),settings))
    inputs = dict(references)
    fit, evaluation = Path(manifest['fit']), Path(manifest['evaluation'])
    sidecar = fit/('online_refit_protocol.json' if online else 'retention_protocol.json')
    declaration = load(sidecar)
    if declaration['input_sha256'] != references:
        raise ValueError('Declared reference inputs changed')
    inputs[str(sidecar.resolve())] = digest(sidecar)
    if online and (declaration.get('belief_intervention') != refit.INTERVENTION
                   or declaration.get('fixed_coordinates') != settings['fixed_coordinates']):
        raise ValueError('ONLINE search declaration changed')
    for name, sha in frozen['source_sha256'].items():
        path = refit.ROOT/name
        if digest(path) != sha:
            raise ValueError('Executed fitting source changed')
        inputs[str(path.resolve())] = sha
    protocol = load(evaluation/'protocol.json')
    plans = {job['name']:job for job in protocol['jobs']}
    files = sorted((evaluation/'jobs').glob('*.json'))
    if (len(plans) != len(protocol['jobs']) or {file.stem for file in files} != set(plans)
            or len(plans) != load(evaluation/'completion.json')['jobs']):
        raise ValueError('Missing or duplicated evaluation jobs')
    for file in files:
        job = load(file)
        base = dict(job)
        if online:
            target = Path(job['target'])
            execution_path = target.with_suffix('.online.json')
            record = load(target.with_suffix('.json'))
            check_online_execution(job,record,load(execution_path),settings['fixed_coordinates'])
            array_file = target.with_suffix('.npz')
            if digest(array_file) != record['predictions_sha256']:
                raise ValueError('Executed prediction arrays changed')
            with np.load(array_file) as arrays:
                np.testing.assert_array_equal(arrays['seeds'],job['seeds'])
            inputs[str(array_file.resolve())] = digest(array_file)
            inputs[str(target.with_suffix('.json').resolve())] = digest(target.with_suffix('.json'))
            inputs[str(execution_path.resolve())] = digest(execution_path)
            for key in ('online_belief','belief_intervention','fixed_coordinates'):
                del base[key]
        inputs.update(retained.executed_retention_inputs(plans[file.stem],base,settings,references))
        inputs[str(file.resolve())] = digest(file)
    return inputs


def feedback_scores(old: dict, new: dict, feedback: np.ndarray, cut: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose pooled-filter log loss without reweighting trials or feedback levels."""
    np.testing.assert_array_equal(old['seeds'],new['seeds'])
    np.testing.assert_array_equal(old['choices'],new['choices'])
    p, q = new['observed_probability'], old['observed_probability']
    if p.shape != q.shape or feedback.shape != (p.shape[1],) or not 0 < cut < p.shape[1]:
        raise ValueError('Feedback and full-sequence filter arrays disagree')
    if not np.isin(feedback,[0.,.5,1.]).all():
        raise ValueError('Unsupported feedback level')
    length = p.shape[1]
    loss = [-np.log(np.maximum(x.mean(0),np.finfo(float).tiny)) for x in (q,p)]
    trials = pd.DataFrame(dict(trial=np.arange(1,length+1),feedback=feedback,
        phase=np.where(np.arange(length)<cut,'training','test'),
        retained_probability=q.mean(0),online_probability=p.mean(0),delta_choice_nll=loss[1]-loss[0]))
    rows = []
    for phase,start,stop in [('training',0,cut),('test',cut,length),('whole',0,length)]:
        window = (np.arange(length)>=start)&(np.arange(length)<stop)
        for label,level in [('all',None),('incorrect',0.),('partial',.5),('correct',1.)]:
            mask = window if level is None else window&(feedback==level)
            count = int(mask.sum())
            means = [nll(x[:,mask]) for x in (q,p)] if count else [np.nan,np.nan]
            delta = means[1]-means[0]
            error = jackknife_se(jackknife_scores(p[:,mask])-jackknife_scores(q[:,mask])) if count else np.nan
            rows.append(dict(phase=phase,feedback_group=label,trials=count,window_trials=stop-start,
                retained_choice_nll=means[0],online_choice_nll=means[1],delta_choice_nll=delta,
                delta_choice_nll_mc_se=error,delta_total_nll=delta*count if count else 0.,
                contribution_to_window_mean=delta*count/(stop-start) if count else 0.))
    return pd.DataFrame(rows),trials


def compare(retained_summary: Path, online_summary: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new ONLINE refit behavioral-comparison directory')
    old, old_frozen, old_tables, inputs = retained.load_summary(retained_summary)
    new, new_frozen, new_tables, more = retained.load_summary(online_summary)
    inputs.update(more)
    settings = new_frozen['settings']
    if (refit.ROOT/settings['retained_reference_fit']).resolve() != Path(old['fit']).resolve():
        raise ValueError('ONLINE fit names a different RETAIN reference')
    refit.validate_settings(settings,old_frozen['settings'])
    inputs.update(bind_execution(old,old_frozen,False))
    inputs.update(bind_execution(new,new_frozen,True))
    subjects, arms = settings['subjects'],list(settings['arms'])
    keys = {(subject,arm) for subject in subjects for arm in arms}
    audits = []
    for manifest in (old,new):
        audit,more = audit_exchange(Path(manifest['fit']),subjects)
        audits.append(audit);inputs.update(more)
    budgets = budget_contrasts(old_tables['budgets'],new_tables['budgets'],*audits)
    budgets = budgets.rename(columns={key:('retained_'+key[6:] if key.startswith('reset_')
        else 'online_'+key[7:]) for key in budgets if key.startswith(('reset_','retain_'))})
    banks, tables = {}, {}
    for label,manifest,frozen,saved in [('RETAIN',old,old_frozen,old_tables),('ONLINE',new,new_frozen,new_tables)]:
        evaluation = Path(manifest['evaluation'])
        nominees = {(r['subject'],r['model']):r['candidate'] for r in frozen['nominees']}
        if set(nominees) != keys or len(nominees) != len(frozen['nominees']):
            raise ValueError('Incomplete or duplicated nomination')
        roles = {}
        for role in ('conditional','autonomous'):
            rows = load(evaluation/role/'evaluation.json')['rows']
            roles[role] = {(r['subject'],r['model']):r for r in rows}
            if set(roles[role]) != keys or len(rows) != len(keys):
                raise ValueError('Incomplete or duplicated evaluation cells')
        for cell in sorted(keys):
            c_row,a_row = (roles[role][cell] for role in ('conditional','autonomous'))
            if any(row['candidate'] != nominees[cell] for row in (c_row,a_row)):
                raise ValueError('Evaluation differs from the behavioral nominee')
            c,a = (checked_arrays(evaluation/role,row) for role,row in [('conditional',c_row),('autonomous',a_row)])
            validate_generation(a,a_row)
            np.testing.assert_array_equal(c['categories'],a['categories'])
            if (not np.isfinite(c['probabilities']).all() or np.any(c['probabilities']<0)
                    or len(set(c['seeds'].tolist())) != len(c['seeds'])
                    or len(c['seeds']) != settings['budgets']['evaluate']['seeds']
                    or len(a['seeds']) != settings['autonomous_paths']
                    or any(c_row[k] != a_row[k] for k in ('condition','training_trials'))):
                raise ValueError('Invalid prediction bank or task boundary')
            np.testing.assert_allclose(c['probabilities'].sum(-1),1.,rtol=0,atol=1e-12)
            actual = np.take_along_axis(c['probabilities'],(c['choices']-1)[None,:,None],axis=2)[...,0]
            np.testing.assert_array_equal(actual,c['observed_probability'])
            banks[label,cell] = c,a,dict(c_row,target_rule=a_row['target_rule'])
        for name in ('scores','curves','parameters','paths'):
            tables.setdefault(name,[]).append(saved[name].assign(variant=label))
    contrasts,feedback_rows,trial_rows,path_rows = [],[],[],[]
    for subject,arm in sorted(keys):
        cell = subject,arm
        retained.match_task(banks['RETAIN',cell],banks['ONLINE',cell])
        c,_,row = banks['RETAIN',cell]
        cut,length = row['training_trials'],len(c['choices'])
        if not 0 < cut < length:
            raise ValueError('Need a nonempty original prefix and later window')
        for phase,start,stop in [('training',0,cut),('test',cut,length),('whole',0,length)]:
            estimates = []
            for label,saved in [('RETAIN',old_tables),('ONLINE',new_tables)]:
                c,a,_ = banks[label,cell]
                estimate = retained.window_estimate(c,a,start,stop,settings['features']['block_width'])
                table = saved['scores']
                selected = table[(table.subject==subject)&(table.model==arm)&(table.phase==phase)]
                if len(selected) != 1:
                    raise ValueError('Missing or duplicated saved score')
                for metric,value in estimate['metrics'].items():
                    np.testing.assert_allclose(value,selected.iloc[0][metric],rtol=0,atol=1e-12)
                estimates.append(estimate)
            contrasts.append(dict(subject=subject,model=arm,phase=phase,trials=stop-start,
                **paired_difference(estimates[1],estimates[0])))
        feedback = np.array([category_learning_feedback(int(choice),int(category),condition=row['condition'])
                             for choice,category in zip(c['choices'],c['categories'])])
        f,t = feedback_scores(banks['RETAIN',cell][0],banks['ONLINE',cell][0],feedback,cut)
        feedback_rows.append(f.assign(subject=subject,model=arm));trial_rows.append(t.assign(subject=subject,model=arm))
        paths = [table['paths'].query('subject == @subject and model == @arm').set_index('seed').sort_index()
                 for table in (old_tables,new_tables)]
        np.testing.assert_array_equal(paths[0].index,paths[1].index)
        for metric in ('late_accuracy','target_ever','target_late_present','target_late_contribution',
                       'target_exits','target_capacity_drops','search_events','target_proposals'):
            before,after = (p[metric].to_numpy(float) for p in paths)
            delta = after-before
            path_rows.append(dict(subject=subject,model=arm,metric=metric,retained=before.mean(),online=after.mean(),
                delta=delta.mean(),paired_mc_se=delta.std(ddof=1)/np.sqrt(len(delta)),paths=len(delta)))
    tables = {name:pd.concat(parts) for name,parts in tables.items()}
    tables.update(contrasts=pd.DataFrame(contrasts),feedback_scores=pd.concat(feedback_rows),trials=pd.concat(trial_rows),
        path_contrasts=pd.DataFrame(path_rows),budgets=budgets,
        exchange_candidates=pd.concat([a.assign(variant=v) for a,v in zip(audits,('RETAIN','ONLINE'))]))
    for path,sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError('Comparison input changed during scoring')
    output.mkdir(parents=True)
    for name,table in tables.items():
        table.to_csv(output/(name+'.csv'),index=False)
    sources = [Path(__file__),Path(retained.__file__),Path(refit.__file__),
               *[Path(__file__).with_name(name) for name in ('model_0923_episodic_refit.py',
                 'model_0923_estimation_scope.py','model_0923_h_generation_uncertainty.py',
                 'model_0923_h_validation.py','model_0923_refit_budget.py')]]
    (output/'source').mkdir()
    for path in sources:
        shutil.copy2(path,output/'source'/path.name)
    write(output/'manifest.json',dict(status='complete',retained_summary=str(retained_summary.resolve()),
        online_summary=str(online_summary.resolve()),subjects=subjects,arms=arms,inputs_sha256=inputs,
        source_sha256={str(p.relative_to(refit.ROOT)):digest(p) for p in sources},
        tables_sha256={name+'.csv':digest(output/(name+'.csv')) for name in tables},
        contrast='ONLINE minus RETAIN; belief mechanism and behavioral re-estimation jointly change',
        actual_search_compute_equal=bool(budgets.actual_compute_equal.all()),
        budget_matching='Same remaining supports and stage limits; verified exchange deduplication changes actual work',
        data_use='Behavior-prefix estimation; later windows are development diagnostics; oral excluded',
        uncertainty='Paired complete-filter/path simulation errors conditional on the frozen parameters'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('retained-summary','online-summary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args();compare(args.retained_summary,args.online_summary,args.output)
