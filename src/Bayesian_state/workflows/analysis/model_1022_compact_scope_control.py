"""Separate early-data extrapolation from full-behavior reconstruction.

Use --config YAML --output NEW [--prepare-only | --smoke]. Two ordinary
development participants receive TRANSFER fits to all choices. Four fresh
synthetic sequences, the two lowest saved generation seeds at each original
TRANSFER point, receive support-only joint recovery. The six fits use the
existing bounded optimizer and one shared numerical queue. Oral data never
select parameters. Full-behavior reconstructions have no held-out choices.

The existing optimizer writes training-only metadata. A narrowly scoped writer
adapter records the actual full-behavior scope on the two human outputs before
they are saved. Scientific equations and the optimizer are unchanged. Truth
scoring explicitly dispatches through the TRANSFER worker, after all estimates
freeze. Existing early-fit results remain separate and immutable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

from . import model_1022_compact_refit as refit
from . import model_1022_compact_evaluate as evaluate
from .model_0923_h_objective_recovery import save_arrays
from ...evaluation.model_0923_h_validation import checked_arrays
from ...utils.seeding import stable_seed

ROOT = refit.ROOT
read, write, digest = refit.read, refit.write, refit.digest
recovery, driver = refit.recovery, refit.driver


def joined_arrays(folder: Path) -> tuple[dict, dict]:
    """Read both chronological segments, verifying their original hashes."""
    meta = read(folder/'prepared.json')
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        if digest(folder/name) != meta['array_sha256'][name]:
            raise ValueError('Original behavioral arrays changed')
        with np.load(folder/name) as z:
            parts.append({k:z[k].copy() for k in z.files})
    if parts[0].keys() != parts[1].keys():
        raise ValueError('Chronological segments have different fields')
    arrays = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if any(len(a) != meta['trials'] for a in arrays.values()):
        raise ValueError('Full-behavior length differs from prepared trials')
    return arrays, meta


def save_fixture(original: Path, destination: Path, arrays: dict, runtime: str, role: str) -> None:
    """Save a new fixture with explicit data use, preserving original arrays."""
    destination.mkdir(parents=True)
    meta = deepcopy(read(original/'prepared.json'))
    write(destination/'engine.json',read(original/'engine.json'))
    meta.update(data_role=role, original_training_trials=meta['training_trials'],
        original_split=meta.get('split'),
        split=role,
        original_prepared_sha256=digest(original/'prepared.json'),
        engine_sha256=digest(destination/'engine.json'),training_trials=len(arrays['choices']),
        test_trials=0,trials=len(arrays['choices']),
        array_sha256={'training.npz':save_arrays(destination,'training.npz',arrays),
            'sealed_test.npz':save_arrays(destination,'sealed_test.npz',{k:v[:0] for k,v in arrays.items()})})
    write(destination/'prepared.json', meta)
    write(destination.parent/'preparation_manifest.json', dict(status='complete',
        subjects=[meta['subject']], settings=dict(source_fit=str(Path(runtime).parent)), data_role=role))


def prepare(policy: dict, budget: dict, output: Path, smoke: bool) -> tuple[list[dict],Path]:
    study, evaluation = ROOT/policy['study'], ROOT/policy['evaluation']
    if (read(study/'completion.json')['status'] != 'complete'
            or read(evaluation/'completion.json')['status'] != 'complete'
            or policy['subjects'] != [205,306] or policy['variant'] != 'TRANSFER'
            or policy['recovery_replicates'] != 2
            or policy['recovery_seed_selection'] != 'two_lowest_saved_seeds'):
        raise ValueError('Require the completed comparison and registered two-point control')
    old_cases = read(study/'cases.json')['cases']
    cases, declared, truths, bindings = [], [], [], {}
    prepared = output/'recovery_prepared'; prepared.mkdir()
    for subject in policy['subjects'][:1] if smoke else policy['subjects']:
        old, = [c for c in old_cases if c['kind']=='human' and c['subject']==subject and c['variant']=='TRANSFER']
        frozen_file = Path(old['output'])/'nominees.json'
        frozen = read(frozen_file); chosen = frozen['nominee']
        done = read(Path(old['output'])/'completion.json')
        if (done['nomination_sha256'] != digest(frozen_file) or frozen['smoke']
                or frozen['status'] != 'frozen_before_test' or done['test_choices_used'] is not False):
            raise ValueError('Require a frozen scientific early-behavior fit')
        for name,sha in frozen['source_sha256'].items():
            if digest(ROOT/name) != sha: raise ValueError('Generating scientific code changed')
        original = Path(frozen['data'])/f'subject_{subject}'
        physical,meta = joined_arrays(original)
        own_eval = evaluation/old['case_id']
        record, = read(own_eval/'autonomous/evaluation.json')['rows']
        if record['candidate'] != chosen['candidate']:
            raise ValueError('Generating candidate differs from the frozen TRANSFER point')
        bank = checked_arrays(own_eval/'autonomous',record)
        np.testing.assert_array_equal(bank['categories'],physical['categories'])
        if len(set(bank['seeds'])) != frozen['settings']['autonomous_paths']:
            raise ValueError('Require the complete original generation bank')
        source_jobs = sorted((own_eval/'jobs').glob('AUTO_*.json'))
        job = read(source_jobs[0])
        if job['compact_variant'] != 'TRANSFER' or job['entry_control'] != 'PRECISION':
            raise ValueError('Synthetic generation lacks the precision-transfer mechanism')
        full_id = f'S{subject}_FULL_TRANSFER'
        settings = refit.configure_settings(frozen['settings'], budget,
            seed=frozen['settings']['seed'], variant='TRANSFER', case_id=full_id)
        data = output/'full_behavior_data'/full_id/f'subject_{subject}'
        own_arrays = {k:v[:16] for k,v in physical.items()} if smoke else physical
        save_fixture(original,data,own_arrays,job['runtime'],'observed_full_behavior_reconstruction')
        settings.update(data=str(data.parent.resolve()),subjects=[subject],
            data_use='All original choices used; later segment is descriptive',
            original_data=frozen['data'], original_training_trials=meta['training_trials'])
        starts = refit.starting_bank(old['space'],[chosen['candidate']['hyperparams']],
                                    4 if smoke else budget['initial_candidates'],settings['seed'])
        cases.append(dict(case_id=full_id,subject=subject,condition=meta['condition'],cohort='development',
            variant='TRANSFER',kind='human',settings=settings,space=old['space'],starts=starts,required=starts[0],
            output=str((output/'human_fit'/full_id).resolve()),full_behavior_used=True,
            early_fit=str(Path(old['output']).resolve()),original_data=frozen['data']))
        # Generation truth is stored separately. Candidate starts use only the
        # parameter support, and neither truth fields nor outcomes affect seeds.
        for replicate,index in enumerate(np.argsort(bank['seeds'])[:1 if smoke else 2]):
            case_id = f'S{subject}_TRANSFER_R{replicate}'
            cut = min(16,meta['training_trials']) if smoke else meta['training_trials']
            folder = prepared/case_id/'data'/f'subject_{subject}'
            arrays = {k:physical[k][:cut] for k in ('stimulus','categories')}
            arrays.update(choices=bank['choices'][index,:cut],feedback=bank['feedback'][index,:cut])
            save_fixture(original,folder,arrays,job['runtime'],'synthetic_training_only')
            seed = stable_seed(dict(base=policy['seed'],subject=subject,replicate=replicate,role='fresh_transfer_recovery'))
            own = refit.configure_settings(frozen['settings'],budget,seed=seed,variant='TRANSFER',case_id=case_id)
            own.update(data=str(folder.parent.resolve()),subjects=[subject],recovery_target_rule=int(record['target_rule']))
            own.pop('warm_start_fits',None)
            starts = recovery.neutral_starts(old['space'],4 if smoke else budget['initial_candidates'],seed)
            case = dict(case_id=case_id,subject=subject,condition=meta['condition'],replicate=replicate,
                kind='recovery',variant='TRANSFER',settings=own,space=old['space'],starts=starts,required=starts[0],
                generating_parameters_used_for_starts=False,generation_seed=int(bank['seeds'][index]),trials=cut,
                output=str((output/'recovery_fit'/case_id).resolve()),
                arrays_sha256={n:digest(folder/n) for n in ('training.npz','sealed_test.npz','engine.json','prepared.json')})
            file = prepared/case_id/'case.json';write(file,case)
            declared.append(dict(case_id=case_id,file=str(file.resolve()),sha256=digest(file)))
            truth_file = prepared/case_id/'states_truth.npz'
            with truth_file.open('xb') as stream:
                np.savez_compressed(stream,**{k:bank[k][index,:cut] for k in ('present','contribution','beta')})
            truths.append(dict(case_id=case_id,subject=subject,replicate=replicate,model='COMPACT',
                compact_variant='TRANSFER',point=chosen['candidate']['hyperparams'],generation_seed=int(bank['seeds'][index]),
                states_truth=str(truth_file.resolve()),states_truth_sha256=digest(truth_file)))
            cases.append(case)
        for file in [frozen_file,own_eval/'autonomous/evaluation.json',*source_jobs,
                     original/'prepared.json',original/'training.npz',original/'sealed_test.npz']:
            bindings[str(file.resolve())] = digest(file)
        file=own_eval/'autonomous/evaluate'/f'subject_{subject}'/'COMPACT'/(record['candidate']['id']+'.npz')
        bindings[str(file.resolve())]=digest(file)
    write(prepared/'protocol.json',dict(cases=declared,inputs_sha256=bindings,
        generation_selection=policy['recovery_seed_selection'],variant='TRANSFER',oral_used=False))
    write(prepared/'truths.json',dict(rows=truths,use='After all fits freeze only'))
    write(prepared/'completion.json',dict(status='complete',cases=len(declared),
        protocol_sha256=digest(prepared/'protocol.json'),truths_sha256=digest(prepared/'truths.json')))
    write(output/'cases.json',dict(cases=cases,inputs_sha256=bindings,oral_used=False,
        human_later_choices_used=True,recovery_truth_used_for_proposals=False))
    recovery.checked_cases(prepared)
    return cases,prepared


@contextmanager
def full_behavior_metadata(cases: list[dict]):
    """Correct only full-behavior output labels before the common fitter saves."""
    folders = {Path(c['output']).resolve() for c in cases if c['kind']=='human'}
    original = refit.write

    def scoped_write(file, record):
        file = Path(file)
        if file.parent.resolve() in folders and file.name in ('nominees.json','completion.json'):
            record = deepcopy(record)
            record.update(original_later_choices_used=True, oral_used_for_fitting=False,
                          evaluation_scope='descriptive_reconstruction_no_heldout_choices')
            if file.name=='nominees.json': record['status']='frozen_before_external_oral_check'
            else: record['test_choices_used']=True
        return original(file,record)

    refit.write=scoped_write
    try: yield
    finally: refit.write=original


def evaluation_jobs(case: dict, output: Path, smoke: bool):
    """Use original chronology for descriptive scores at a full-behavior fit."""
    frozen = read(Path(case['output'])/'nominees.json')
    done = read(Path(case['output'])/'completion.json')
    if (frozen['status']!='frozen_before_external_oral_check' or done['test_choices_used'] is not True
            or done['nomination_sha256']!=digest(Path(case['output'])/'nominees.json')):
        raise ValueError('Full-behavior evaluation requires explicit reconstruction labels')
    settings=frozen['settings'];data=Path(case['original_data'])
    if smoke: data=evaluate.smoke_fixture(data,output/'software_fixture',case['subject'])
    groups,chunks={},[]
    for role in ('conditional','autonomous'):
        originals=driver.io.score_jobs(data,output/role,driver.base.settings_for_scoring(settings),
            {(case['subject'],'COMPACT'):[frozen['nominee']['candidate']]},'evaluate',smoke)
        groups[role]=[]
        for original in originals:
            job=deepcopy(original)
            job.update(arm_settings=settings['arms']['COMPACT'],source_sha256=frozen['source_sha256'],
                compact_variant='TRANSFER',entry_control='PRECISION',case_id=case['case_id'],
                evaluation_route_sources=evaluate.sources(),
                evaluation_scope='descriptive_reconstruction_no_heldout_choices')
            job['mechanism']['capacity']='evidence'
            if role=='autonomous':
                job.update(phase='autonomous',name='AUTO_'+job['name'])
                job['seeds']=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],role='autonomous',
                    subject=case['subject'],path=i)) for i in range(2 if smoke else settings['autonomous_paths'])]
                audit=ROOT/settings['audit']/f'audit/subject_{case["subject"]}/audit.json'
                job.update(audit=str(audit),audit_sha256=digest(audit))
            parts=[];width=settings['autonomous_chunk'] if role=='autonomous' else 1
            for i in range(0,len(job['seeds']),width):
                part=deepcopy(job);target=Path(job['target']);suffix=f'_C{i:03d}'
                part.update(name=job['name']+suffix,seeds=job['seeds'][i:i+width],
                    target=str(target.parent/'chunks'/(target.name+suffix)))
                if role=='conditional':
                    part.update(decision_basis_target=str((output/'state_capture'/part['name']).resolve()),
                                decision_basis_sources=evaluate.capture.sources())
                parts.append(part);chunks.append(part)
            groups[role].append((job,parts))
    return chunks,groups


def evaluate_case(case: dict, output: Path, smoke: bool, pool, gate):
    output.mkdir(parents=True)
    chunks,groups=evaluation_jobs(case,output,smoke)
    write(output/'protocol.json',dict(fit=case['output'],compact_variant='TRANSFER',smoke=smoke,
        nomination_sha256=digest(Path(case['output'])/'nominees.json'),jobs=chunks,
        data_use='All original choices used to fit; all behavioral scores descriptive'))
    write(output/'decision_basis_capture_protocol.json',dict(source_sha256=evaluate.capture.sources(),
        route_source_sha256=evaluate.sources(),additional_filters=0,timing='after_choice_before_current_feedback'))
    for future in as_completed([pool.submit(evaluate.launch,j,output,gate) for j in chunks]): future.result()
    for role,records in groups.items():
        rows=[driver.io.combine_evaluation_chunks(job,parts,role=='autonomous') for job,parts in records]
        write(output/role/'evaluation.json',dict(status='complete',rows=rows))
    write(output/'completion.json',dict(status='complete',smoke=smoke,compact_variant='TRANSFER',
        evaluation_scope='descriptive_reconstruction_no_heldout_choices',
        nomination_sha256=digest(Path(case['output'])/'nominees.json')))
    print(case['case_id']+': full-behavior reconstruction complete',flush=True)


def run(config: Path, output: Path, smoke: bool, prepare_only: bool):
    if output.exists(): raise FileExistsError('Use a new scope-control directory')
    policy=yaml.safe_load(config.read_text());budget=yaml.safe_load((ROOT/policy['budget_config']).read_text())
    if not 1<=policy['parallel_budget']<=128: raise ValueError('Invalid worker budget')
    output.mkdir(parents=True)
    shutil.copyfile(config,output/'config.yaml');shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    hashes=dict(refit.sources(driver.io),**{str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
        evaluate.ENTRY:digest(ROOT/evaluate.ENTRY)})
    driver.base.archive(output,config,hashes)
    cases,prepared=prepare(policy,budget,output,smoke)
    workers=1 if smoke else min(policy['parallel_budget'],len(os.sched_getaffinity(0)))
    write(output/'protocol.json',dict(settings=policy,smoke=smoke,workers=workers,source_sha256=hashes,
        python=sys.version,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','PyYAML')},
        oral_used_for_fitting=False,scientific_inference=not smoke))
    if prepare_only:
        write(output/'preparation_complete.json',dict(status='complete',cases=len(cases),numeric_jobs=0));return
    fit=output/'recovery_fit';fit.mkdir(exist_ok=True)
    write(fit/'protocol.json',dict(prepared=str(prepared.resolve()),smoke=smoke,workers=workers,source_sha256=hashes))
    print(f'{len(cases)} cases; {workers} shared single-thread numerical workers',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        refit.configure_executor(pool)
        with full_behavior_metadata(cases),ThreadPoolExecutor(max_workers=1 if smoke else len(cases)) as controls:
            nominees=[f.result() for f in [controls.submit(refit.fit_case,c,smoke) for c in cases]]
        recovered=[n for n in nominees if 'replicate' in n]
        write(fit/'nominees.json',dict(status='frozen_before_truth',smoke=smoke,nominees=recovered))
        write(fit/'completion.json',dict(status='complete',smoke=smoke,truths_opened=False,cases=len(recovered),
            nomination_sha256=digest(fit/'nominees.json')))
        write(output/'human_nominees.json',dict(status='frozen_before_external_oral_check',smoke=smoke,
            nominees=[n for n in nominees if 'replicate' not in n],original_later_choices_used=True))
        gate=evaluate.ResourceGate(output,workers,0)
        humans=[c for c in cases if c['kind']=='human']
        with ThreadPoolExecutor(max_workers=len(humans)) as controls:
            pending=[controls.submit(evaluate_case,c,output/'evaluation'/c['case_id'],smoke,pool,gate) for c in humans]
            original=recovery.configure_executor
            recovery.configure_executor=refit.configure_executor
            try: recovery.score_truth(prepared,fit,output/'recovery_scores',numerical_pool=pool)
            finally: recovery.configure_executor=original
            for f in pending:f.result()
    from ...evaluation.model_1021_simplification_recovery_states import summarize
    summarize(prepared,fit,output/'recovery_scores',output/'recovery_states')
    write(output/'completion.json',dict(status='complete',smoke=smoke,human_cases=len(humans),
        recovery_cases=len(recovered),oral_used_for_fitting=False,original_later_choices_used=True,
        human_behavior_scope='descriptive_reconstruction_no_heldout_choices'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    group=parser.add_mutually_exclusive_group();group.add_argument('--smoke',action='store_true');group.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();run(args.config,args.output,args.smoke,args.prepare_only)
