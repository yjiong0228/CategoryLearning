"""Audit the completed full-behavior and fresh TRANSFER recovery experiment.

Use --study COMPLETE_SCOPE_CONTROL --output NEW. This read-only audit checks
all six actual fitting budgets, neutral recovery starts, simulation identities,
post-freeze truth routing and the absence of future-trial effects on fixed-point
prefix predictions. It runs no new filter and never rewrites source results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import model_1022_compact_scope_control as control
from ...evaluation.model_1022_compact_refit import training_budget
from ...evaluation.model_0923_h_validation import checked_arrays, nll

ROOT,read,write,digest=control.ROOT,control.read,control.write,control.digest


def audit(study: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new completed-study audit directory')
    done=read(study/'completion.json')
    if done['status']!='complete' or done['smoke']:raise ValueError('Complete the scientific scope control first')
    inputs={}

    def bind(file,expected=None):
        file=Path(file).resolve();value=digest(file)
        if expected is not None and value!=expected:raise ValueError('Audit source changed: '+str(file))
        inputs[str(file)]=value
        return file

    protocol=read(bind(study/'protocol.json'));plan=read(bind(study/'cases.json'))
    bind(study/'completion.json');bind(Path(__file__))
    for name,sha in plan['inputs_sha256'].items():bind(name,sha)
    for name,sha in protocol['source_sha256'].items():
        # The archived source is authoritative even if later documentation or
        # workflow maintenance updates the live worktree after this run.
        bind(study/'source'/name,sha)
    cases=plan['cases']
    early_cases=read(bind(ROOT/protocol['settings']['study']/'cases.json'))['cases']
    if len(cases)!=6 or sum(c['kind']=='human' for c in cases)!=2:raise ValueError('Require all six registered cases')
    scored=read(bind(study/'recovery_scores/completion.json'))
    if scored['status']!='complete' or scored['truths_opened_after_freeze'] is not True:
        raise ValueError('Recovery truth scoring has not completed after freezing')
    truths={r['case_id']:r for r in read(bind(study/'recovery_prepared/truths.json'))['rows']}
    budgets,prefixes,synthetic,truth_scores=[],[],[],[]
    for case in cases:
        fit=Path(case['output']);frozen=read(bind(fit/'nominees.json'));finish=read(bind(fit/'completion.json'))
        if finish['status']!='complete' or finish['nomination_sha256']!=digest(fit/'nominees.json'):
            raise ValueError('Case is incomplete or the frozen estimate changed')
        budgets.extend(training_budget(case,bind))
        point=frozen['nominee']['candidate']
        if point['mean_nll']!=min(r['mean_nll'] for r in frozen['nominee']['finalists']):
            raise ValueError('Frozen point is not the best confirmed candidate')
        folder=Path(case['settings']['data'])/f'subject_{case["subject"]}'
        meta=read(bind(folder/'prepared.json'))
        bind(folder/'engine.json',meta['engine_sha256'])
        with np.load(bind(folder/'training.npz',meta['array_sha256']['training.npz'])) as z:
            arrays={k:z[k].copy() for k in z.files}
        with np.load(bind(folder/'sealed_test.npz',meta['array_sha256']['sealed_test.npz'])) as z:
            if any(len(z[k]) for k in z.files):raise ValueError('Full/synthetic fitting fixture contains held-out rows')
        if case['kind']=='human':
            if frozen['status']!='frozen_before_external_oral_check' or finish['test_choices_used'] is not True:
                raise ValueError('Full-behavior fit lost its data-use declaration')
            original=Path(case['original_data'])/f'subject_{case["subject"]}'
            all_arrays,original_meta=control.joined_arrays(original)
            for key in all_arrays:np.testing.assert_array_equal(arrays[key],all_arrays[key])
            early=read(bind(Path(case['early_fit'])/'nominees.json'))
            anchor=early['nominee']['candidate'];cut=original_meta['training_trials']
            matched,=[r for r in frozen['nominee']['finalists'] if r['id']==anchor['id']]
            old_file=Path(anchor['score_file']);new_file=Path(matched['score_file'])
            for file in (old_file,new_file):
                record=read(bind(file));bind(file.with_suffix('.npz'),record['predictions_sha256'])
            with np.load(old_file.with_suffix('.npz')) as old,np.load(new_file.with_suffix('.npz')) as new:
                np.testing.assert_array_equal(old['seeds'],new['seeds'])
                for key in ('observed_probability','correct_probability'):
                    np.testing.assert_array_equal(old[key],new[key][:,:cut])
                filters=len(old['seeds'])
            prefixes.append(dict(subject=case['subject'],candidate=anchor['id'],prefix_trials=cut,
                full_trials=meta['trials'],filters=filters,exact_fixed_point_prefix=True))
        else:
            if frozen['status']!='frozen_before_truth' or finish['test_choices_used'] is not False:
                raise ValueError('Synthetic fit lost its frozen-before-truth declaration')
            if case['starts']!=control.recovery.neutral_starts(case['space'],40,case['settings']['seed']):
                raise ValueError('Recovery starts depend on more than the declared support and seed')
            generated=ROOT/protocol['settings']['evaluation']/f'S{case["subject"]}_TRANSFER'/'autonomous'
            record,=read(bind(generated/'evaluation.json'))['rows'];bank=checked_arrays(generated,record)
            row=np.argsort(bank['seeds'])[case['replicate']]
            if int(bank['seeds'][row])!=case['generation_seed']:raise ValueError('Recovery sequence selection changed')
            cut=case['trials']
            original_case,=[c for c in early_cases if c['kind']=='human'
                and c['subject']==case['subject'] and c['variant']=='TRANSFER']
            original_folder=ROOT/original_case['settings']['data']/f'subject_{case["subject"]}'
            original_meta=read(bind(original_folder/'prepared.json'))
            with np.load(bind(original_folder/'training.npz',original_meta['array_sha256']['training.npz'])) as z:
                np.testing.assert_array_equal(arrays['stimulus'],z['stimulus'][:cut])
            for key in ('choices','feedback'):np.testing.assert_array_equal(arrays[key],bank[key][row,:cut])
            np.testing.assert_array_equal(arrays['categories'],bank['categories'][:cut])
            truth=truths[case['case_id']]
            if truth['compact_variant']!='TRANSFER' or truth['point']!=record['candidate']['hyperparams']:
                raise ValueError('Recovery truth differs from the TRANSFER generation point')
            with np.load(bind(truth['states_truth'],truth['states_truth_sha256'])) as z:
                for key in ('present','contribution','beta'):np.testing.assert_array_equal(z[key],bank[key][row,:cut])
            for file in sorted((study/'recovery_scores'/case['case_id']/'jobs').glob('*.json')):
                job=read(bind(file));score=read(bind(Path(job['target']).with_suffix('.json')))
                variant=read(bind(Path(job['target']).with_suffix('.variant.json')))
                if (job['compact_variant']!='TRANSFER' or job['entry_control']!='PRECISION'
                        or variant['compact_variant']!='TRANSFER' or job['candidate']['hyperparams']!=truth['point']
                        or job['particles']!=1024 or len(job['seeds'])!=1 or score['trials']!=cut
                        or any(p['num_threads']!=1 for p in score['numeric_pools'])):
                    raise ValueError('Generating-point scorer uses the wrong cognitive or numerical variant')
                bind(Path(job['target']).with_suffix('.npz'),score['predictions_sha256'])
                with np.load(Path(job['target']).with_suffix('.npz')) as z:
                    np.testing.assert_allclose(nll(z['observed_probability']),score['mean_nll'],atol=1e-12,rtol=0)
                truth_scores.append(dict(case_id=case['case_id'],seed=job['seeds'][0],particles=1024,trials=cut,variant='TRANSFER'))
            synthetic.append(dict(case_id=case['case_id'],generation_seed=case['generation_seed'],trials=cut,
                                  exact_generation_arrays=True,neutral_starts=40))
    if len(budgets)!=2544 or len(truth_scores)!=32:raise ValueError('Total number of fitted/truth filters differs from the declared budget')
    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/'source_snapshot.py')
    for name,rows in [('fit_filters',budgets),('fixed_point_prefixes',prefixes),('synthetic_cases',synthetic),('truth_filters',truth_scores)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    write(output/'completion.json',dict(status='passed',inputs_sha256=inputs,cases=6,
        fit_filters=len(budgets),truth_filters=len(truth_scores),additional_filters=0,
        full_behavior_prefix_parity='Exact at original frozen point, for every matched confirmation seed',
        recovery_truth_routing='TRANSFER for every post-freeze generating-point filter',
        metadata_note='Fixture scope note retains inherited split text and explicit effective data-use counts',
        artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.study,args.output)
