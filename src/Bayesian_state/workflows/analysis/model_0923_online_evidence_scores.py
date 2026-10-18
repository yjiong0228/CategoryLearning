"""Bounded choice-likelihood screen for online evidence belief at frozen nominees.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Both variants use the
declared common particles and original evaluation seed subset. Parameters are
unchanged. The smoke uses POOL, two original512-particle filters and eight
trials; disabled predictions must exactly match the saved reference prefixes.
The main screen compares baseline/online in POOL/SEARCH on the entire observed
sequence. All comparisons are development diagnostics; oral is excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
HELPER = 'src/Bayesian_state/workflows/analysis/model_0923_online_evidence_probe.py'
MECHANISM = 'src/Bayesian_state/model/modules/hypothesis_transition/online_evidence_belief.py'
SCORING = 'src/Bayesian_state/workflows/analysis/model_0923_capacity_memory_scores.py'


def helpers():
    spec = importlib.util.spec_from_file_location('online_choice_probe', ROOT / HELPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.helpers()


def worker(file: Path) -> None:
    retention = helpers()
    recalled = retention.helpers()
    retention.helpers = lambda: recalled
    original_validation = recalled.base.validation

    def validation():
        v = original_validation()
        original_score = v.score

        def score(job, workflow):
            if job['phase'] != 'evaluate':
                raise ValueError('Online choice screen requires complete evaluation trials')
            module = recalled.base.load(
                'src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief', MECHANISM)
            cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
            with module.online_evidence_belief(cls, enabled=job['online_belief']) as events:
                original_score(job, workflow)
            target = Path(job['target'])
            row = v.read(target.with_suffix('.json'))
            counts = Counter(event['trial'] for event in events)
            if job['online_belief'] and counts != {i: job['particles'] * len(job['seeds']) for i in range(1, row['trials'])}:
                raise ValueError('Online filter evidence did not update once per completed trial and particle')
            v.write(target.with_suffix('.online.json'), dict(enabled=job['online_belief'],
                reconstructions=len(events), nonentry_reconstructions=sum(e['newcomer_count'] == 0 for e in events),
                exact_transition_timing=True, evidence_matches_retained_state=True,
                unit='computational particle transitions; not autonomous path events'))

        v.score = score
        return v

    recalled.base.validation = validation
    retention.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    from .model_0923_estimation_scope import smoke_data
    from .model_0923_capacity_memory_scores import choice_contrasts
    if output.exists():
        raise FileExistsError('Use a new online evidence score output')
    settings = yaml.safe_load(config.read_text())
    if (not 2 <= settings['filter_seeds'] <= 8 or settings['particles'] < 2
            or not 1 <= settings['parallel_budget'] <= 128):
        raise ValueError('Invalid bounded particle/filter budget')
    retention = helpers()
    recalled = retention.helpers()
    v = recalled.base.validation()
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = v.digest(path)
        return path

    bind(config)
    source, automatic = ROOT / settings['reference_evaluation'], ROOT / settings['autonomous']
    evaluation = v.read(bind(source / 'protocol.json'))
    frozen = v.read(bind(Path(evaluation['fit']) / 'nominees.json'))
    auto = v.read(bind(automatic / 'protocol.json'))
    done = v.read(bind(automatic / 'completion.json'))
    if (done['status'] != 'complete' or done['smoke'] or done['nominees_changed']
            or done['protocol_sha256'] != v.digest(automatic / 'protocol.json')
            or frozen['status'] != 'frozen_before_test' or frozen['smoke']
            or evaluation['nomination_sha256'] != v.digest(Path(evaluation['fit']) / 'nominees.json')):
        raise ValueError('Need frozen behavioral nominees and completed autonomous screen')
    hashes = dict(auto['source_sha256'])
    if any(hashes.get(name) != sha for name, sha in evaluation['source_sha256'].items()):
        raise ValueError('Autonomous and conditional cognitive implementations differ')
    hashes.update({name: v.digest(ROOT / name) for name in (ENTRY, HELPER, MECHANISM, SCORING)})
    for name, sha in hashes.items():
        if v.digest(bind(ROOT / name)) != sha:
            raise ValueError('Executing source changed')
    templates = {}
    nominees = [r for r in frozen['nominees'] if r['subject'] == settings['subject']]
    if {r['model'] for r in nominees} != {'POOL', 'SEARCH'}:
        raise ValueError('Require both frozen proposal-arm nominees')
    for row in nominees:
        arm, point = row['model'], row['candidate']
        files = sorted((source / 'jobs').glob(f'evaluate_S{settings["subject"]}_{arm}_{point["id"]}_C*.json'))
        jobs = [v.read(bind(path)) for path in files]
        if (len(jobs) != 8 or any(job['candidate']['hyperparams'] != point['hyperparams']
                or not job['retain_precision'] or job['particles'] != 512 or job['smoke'] for job in jobs)):
            raise ValueError('Original frozen conditional specification differs')
        own = [bank for bank in auto['banks'] if bank['model'] == arm and bank['schedule'] == 'online'
               and bank['candidate']['hyperparams'] == point['hyperparams']]
        if len(own) != 1:
            raise ValueError('Frozen nominee absent from autonomous mechanism screen')
        templates[arm] = jobs
    references = retention.reference_inputs(recalled, frozen['settings'])
    inputs.update(references)
    retention.configure(recalled, references)
    recalled.ENTRY = ENTRY
    folder = Path(templates['POOL'][0]['folder'])
    for name in ('prepared.json', 'engine.json', 'training.npz', 'sealed_test.npz'):
        bind(folder / name)
    output.mkdir(parents=True)
    if smoke:
        folder = smoke_data(folder.parent, output, settings['subject'], v) / f'subject_{settings["subject"]}'
    cut = v.read(folder / 'prepared.json')['training_trials']
    particles = 512 if smoke else settings['particles']
    count = 2 if smoke else settings['filter_seeds']
    jobs, plans = [], []
    for arm in ('POOL',) if smoke else ('POOL', 'SEARCH'):
        seeds = [job['seeds'][0] for job in templates[arm][:count]]
        for enabled in (False, True):
            name = f"{'online' if enabled else 'baseline'}_S{settings['subject']}_{arm}"
            job = dict(deepcopy(templates[arm][0]), name=name, seeds=seeds, particles=particles,
                folder=str(folder.resolve()), online_belief=enabled, smoke=smoke, source_sha256=hashes,
                target=str((output / 'conditional' / name).resolve()))
            chunks = [dict(job, name=name+f'_C{i:03d}', seeds=[seed],
                target=str((output / 'conditional' / 'chunks' / (name+f'_C{i:03d}')).resolve())) for i,seed in enumerate(seeds)]
            jobs.extend(chunks)
            plans.append(dict(model=arm, variant='online' if enabled else 'baseline', job=job, chunks=chunks))
            if smoke and not enabled:
                for template in templates[arm][:count]:
                    bind(Path(template['target']).with_suffix('.json'))
                    bind(Path(template['target']).with_suffix('.npz'))
    v.write(output / 'protocol.json', dict(settings=frozen['settings'], score_settings=settings, smoke=smoke, plans=plans,
        particles=particles, filter_seeds=count, input_sha256=inputs, source_sha256=hashes,
        selection='No parameter selection; original nominees remain frozen',
        data_use='Whole sequence development screen; oral excluded; prefix/later are chronological segments'))
    recalled.base.execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    arrays, records = {}, []
    for plan in plans:
        record = v.combine_evaluation_chunks(plan['job'], plan['chunks'], False)
        target = Path(plan['job']['target']).with_suffix('.npz')
        with np.load(target) as z:
            a = {key:z[key].copy() for key in z.files}
        np.testing.assert_array_equal(a['seeds'], plan['job']['seeds'])
        if not np.isfinite(a['probabilities']).all() or np.any(a['probabilities'] < 0):
            raise ValueError('Invalid filter predictions')
        np.testing.assert_allclose(a['probabilities'].sum(-1), 1, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(a['observed_probability'], a['probabilities'][:,np.arange(record['trials']),a['choices']-1])
        arrays[plan['model'],plan['variant']] = a
        records.append(dict(model=plan['model'],variant=plan['variant'],record=record,target=str(target)))
    if smoke:
        for i, template in enumerate(templates['POOL'][:count]):
            with np.load(Path(template['target']).with_suffix('.npz')) as z:
                for key, value in arrays['POOL','baseline'].items():
                    actual = value if key in ('choices','categories') else value[i:i+1]
                    expected = z[key][:8] if key in ('choices','categories') else z[key] if key=='seeds' else z[key][:,:8]
                    np.testing.assert_array_equal(actual, expected)
    rows, trials = [], []
    for arm in ('POOL',) if smoke else ('POOL', 'SEARCH'):
        a,b = arrays[arm,'online'],arrays[arm,'baseline']
        for key in ('choices','categories','seeds'):
            np.testing.assert_array_equal(a[key],b[key])
        rows.extend(dict(model=arm,**r) for r in choice_contrasts(a['observed_probability'],b['observed_probability'],cut))
        p,q = a['observed_probability'].mean(0),b['observed_probability'].mean(0)
        loss = -np.log(np.maximum(p,np.finfo(float).tiny))+np.log(np.maximum(q,np.finfo(float).tiny))
        trials.extend(dict(model=arm,trial=i+1,phase='prefix' if i<cut else 'later',
            correct=int(a['choices'][i]==a['categories'][i]),online_probability=p[i],baseline_probability=q[i],
            delta_choice_nll=loss[i]) for i in range(len(p)))
    pd.DataFrame(rows).to_csv(output/'scores.csv',index=False)
    pd.DataFrame(trials).to_csv(output/'trial_scores.csv',index=False)
    for path, sha in inputs.items():
        if v.digest(Path(path)) != sha:
            raise ValueError('Bound score input changed')
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,records=records,
        jobs=len(jobs),particles=particles,filter_seeds=count,trials=len(p),training_trials=cut,
        exact_baseline_prefix=smoke,nominees_changed=False,protocol_sha256=v.digest(output/'protocol.json'),
        artifacts={p.name:v.digest(p) for p in output.glob('*.csv')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config,args.output,args.smoke)
    else: parser.error('Use --config and --output, or an internal --worker')
