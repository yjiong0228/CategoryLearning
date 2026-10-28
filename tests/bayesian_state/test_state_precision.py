"""Numerical acceptance is independent of bookkeeping; expansion needs a hard case."""
from copy import deepcopy
from pathlib import Path

import yaml

from src.Bayesian_state.workflows.analysis.model_1024_state_precision import (
    bank_pass, difference_pass, may_expand, estimate, child_cpu)


def policy():
    return yaml.safe_load(Path('configs/shared/model1024_state_precision.yaml').read_text())


def test_available_repetitions_do_not_inherit_old_completion_gate():
    p = policy()
    row = dict(systems=15, expected_systems=64, bank_complete=False, full_ess_min=8., full_maximum_weight=.2,
               full_half_mean=.02, full_half_q95=.08)
    assert bank_pass(row, p, 'full')
    row['systems'] = 7
    assert not bank_pass(row, p, 'full')
    row['systems'] = 16; row['full_half_q95'] = .11
    assert not bank_pass(row, p, 'full')


def test_cross_resolution_requires_mean_and_tail_agreement():
    assert difference_pass(dict(mean=.025, q95=.10), policy())
    assert not difference_pass(dict(mean=.03, q95=.09), policy())
    assert not difference_pass(dict(mean=.02, q95=.11), policy())


def test_expansion_requires_replicated_success_including_a_difficult_case():
    p = policy(); easy = p['pilot_cases'][:2]; hard = p['hard_cases']
    assert not may_expand(easy, p)
    assert not may_expand(hard[:1], p)
    assert may_expand([easy[0], hard[0]], p)
    assert not may_expand([easy[0], easy[0]], p)


def test_estimates_scale_resources_and_learn_observed_slowdown():
    protocol = dict(policy=policy(), estimates={'case':dict(cpu_seconds=100., wall_seconds=100.,
                                                           rss_bytes=1000., particles=1024)})
    job = dict(state_case='case', particles=2048)
    a = estimate(job, protocol, [])
    row = dict(case='case', status='complete', particles=1024, cpu_seconds=110.,
               wall_seconds=300., max_rss_kib=2.)
    b = estimate(job, protocol, [row])
    assert a['cpu']==290.
    assert b['wall'] > 2*a['wall']
    assert b['memory'] > a['memory']
    assert child_cpu(2147483647)==0.


def test_scheduler_accounts_real_children_and_resumes_completed_bank(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    from src.Bayesian_state.workflows.analysis import model_1024_state_precision as runner
    p = policy(); p['budget'].update(poll_seconds=.01, workers=4)
    protocol = dict(policy=p, sources={}, cases=[dict(id='case')],
                    estimates={'case':dict(cpu_seconds=.01, wall_seconds=.01,
                                           rss_bytes=1e6, particles=1024)})
    runner.write(tmp_path/'protocol.json', protocol)
    real_popen = subprocess.Popen

    def make_job(case, policy, output, stage, particles, index):
        name=f'{stage}_{index}'
        return dict(name=name, state_case='case', particles=particles, runtime='unused',
                    individual_target=str(output/'states'/name))

    def tiny_worker(args, **kwargs):
        job=json.loads(Path(args[-1]).read_text())
        code=('import json,pathlib,time; p=pathlib.Path('+repr(job['individual_target']+'.json')+'); '
              'p.parent.mkdir(parents=True,exist_ok=True); '
              't=time.process_time();\nwhile time.process_time()-t<.02: pass\n'
              'p.write_text(json.dumps({"status":"complete"}))')
        return real_popen([sys.executable,'-c',code], **kwargs)

    monkeypatch.setattr(runner.study, 'make_job', make_job)
    monkeypatch.setattr(runner.subprocess, 'Popen', tiny_worker)
    runner.run(tmp_path, ['case'], [2048], 4, 'test')
    rows = [json.loads(f.read_text()) for f in (tmp_path/'ledger').glob('*.json')]
    assert len(rows)==4 and all(r['status']=='complete' for r in rows)
    assert sum(r['cpu_seconds'] for r in rows) >= .08
    done=json.loads((tmp_path/'completions'/'test.json').read_text())
    assert done['completed_jobs']==4 and done['peak_workers']==4
    runner.run(tmp_path, ['case'], [2048], 4, 'resume')
    assert len(list((tmp_path/'ledger').glob('*.json')))==4
    assert json.loads((tmp_path/'completions'/'resume.json').read_text())['peak_workers']==0
    runner.run(tmp_path, ['case'], [2048], 4, 'mixed', plan=[
        dict(cases=['case'], particles=[2048], systems=4),
        dict(cases=['case'], particles=[4096], systems=2, confirmation=True)])
    rows=[json.loads(f.read_text()) for f in (tmp_path/'ledger').glob('*.json')]
    assert len(rows)==6 and sum(r['particles']==4096 for r in rows)==2
    assert json.loads((tmp_path/'completions'/'mixed.json').read_text())['completed_jobs']==6


def test_adaptive_actions_escalate_failures_and_confirm_successes_together(tmp_path):
    from src.Bayesian_state.workflows.analysis import model_1024_state_precision_control as control
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision import write
    p=policy();write(tmp_path/'protocol.json',dict(policy=p))
    cases=p['pilot_cases'][:2]
    rows=[dict(case=cases[0],particles=4096,systems=16,confirmation=False,full_numerically_consistent=True),
          dict(case=cases[1],particles=4096,systems=16,confirmation=False,full_numerically_consistent=False)]
    specs,stable=control.actions(tmp_path,rows,cases,[2048,4096])
    assert not stable
    assert specs[0]['confirmation'] and specs[0]['systems']==16
    assert specs[1]['particles']==[8192] and not specs[1]['confirmation']


def test_later_confirmation_uses_new_unseen_seed_block(tmp_path):
    from src.Bayesian_state.workflows.analysis import model_1024_state_precision_control as control
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision import write
    p=policy();cid=p['pilot_cases'][0];write(tmp_path/'protocol.json',dict(policy=p))
    write(tmp_path/'confirmations'/'prior.json',dict(case=cid,particles=16384,primary_systems=16,accepted=False))
    rows=[dict(case=cid,particles=16384,systems=32,confirmation=False,full_numerically_consistent=True),
          dict(case=cid,particles=16384,systems=16,confirmation=True)]
    specs,_=control.actions(tmp_path,rows,[cid],[2048,4096])
    assert specs[0]['confirmation'] and specs[0]['systems']==32
    assert specs[0]['primary_systems']==32


def test_admission_accounts_for_shared_hardware_threads_without_changing_hard_caps():
    protocol=dict(policy=policy(),resource_policy=dict(physical_cores=64),
                  estimates={'case':dict(cpu_seconds=100.,wall_seconds=100.,rss_bytes=1000.,particles=1024)})
    job=dict(state_case='case',particles=4096)
    row=dict(case='case',status='complete',particles=2048,cpu_seconds=440.,wall_seconds=450.,
             max_rss_kib=4.,peak_concurrent_workers=128)
    crowded=estimate(job,protocol,[row],128); physical=estimate(job,protocol,[row],64)
    assert physical['cpu'] < crowded['cpu']
    assert physical['cpu'] >= 400*1.3
    assert physical['memory']==crowded['memory']
    assert protocol['policy']['budget']['cpu_hours']==1536


def test_joint_rule_precision_preserves_rule_identity_and_closed_endpoint():
    import numpy as np
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision_distributions import executed_precision_histogram
    ids=np.array([[[0,1],[1,-1]]]);beta=np.array([[[0.,25.],[2.5,0.]]])
    execution=np.array([[[.25,.75],[1.,0.]]]);weights=np.array([[.4,.6]])
    hist=executed_precision_histogram(ids,beta,execution,weights,2).reshape(1,2,25)
    assert hist[0,0,0]==.1
    assert np.isclose(hist[0,1,24],.3)
    assert hist[0,1,2]==.6
    np.testing.assert_allclose(hist.sum(axis=2),[[.1,.9]])


def test_distribution_comparison_detects_precision_changes_with_identical_rule_mass():
    import numpy as np
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision_distributions import FIELDS,compare
    a={};b={}
    for mode in ('filtered','full'):
        for field in FIELDS:
            a[mode+'_'+field]=np.tile([1.,0.],(4,1))
            b[mode+'_'+field]=np.tile([0.,1.],(4,1))
    row,trace=compare(a,b)
    assert row['full_executed_rule_precision_mean']==1.
    np.testing.assert_array_equal(trace['filtered_precision_with_absence'],np.ones(4))


def test_replicated_unshared_timings_replace_stale_small_particle_extrapolation():
    protocol=dict(policy=policy(),resource_policy=dict(physical_cores=64,minimum_unshared_observations=8),
                  estimates={'case':dict(cpu_seconds=100.,wall_seconds=100.,rss_bytes=1000.,particles=1024)})
    job=dict(state_case='case',particles=16384)
    row=dict(case='case',status='complete',particles=8192,cpu_seconds=500.,wall_seconds=600.,
             max_rss_kib=4.,peak_concurrent_workers=64)
    sparse=estimate(job,protocol,[row]*7,64)
    calibrated=estimate(job,protocol,[row]*8,64)
    assert calibrated['cpu']==1330.
    assert calibrated['cpu'] < sparse['cpu']
    assert calibrated['memory']==sparse['memory']
    assert estimate(job,protocol,[row]*8,128)['cpu'] > calibrated['cpu']


def test_preallocated_history_preserves_appended_values_without_aliasing():
    import numpy as np
    import pytest
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision_storage import PreallocatedHistory
    h=PreallocatedHistory(2);a=np.array([[1.,2.]])
    h.append({'x':a});a[:]=3.;h.append({'x':a})
    np.testing.assert_array_equal(h.values['x'],[[[1.,2.]],[[3.,3.]]])
    with pytest.raises(ValueError): h.append({'x':a})
    partial=PreallocatedHistory(2,mapping=False);partial.append(np.array([.2,.8]))
    with pytest.raises(ValueError): partial.values


def test_blocked_summaries_match_original_reductions_exactly():
    import numpy as np
    from src.Bayesian_state.evaluation.individual_state_inference import state_means
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision_storage import blocked_means
    ids=np.tile(np.array([[[0,1],[1,-1],[0,-1]]]),(5,1,1))
    b=np.tile(np.array([[[.2,.8],[1.,0.],[1.,0.]]]),(5,1,1))
    values=dict(rule=ids,belief=b,execution=b,precision=np.where(ids>=0,7.,0.),component=b[...,None],
                component_counts=np.ones((2,2),int))
    weights=np.tile([.2,.3,.5],(5,1))
    expected=state_means(values,weights,2,[0])
    for block in (1,2,5):
        actual=blocked_means(state_means,values,weights,2,[0],block)
        for key in expected: np.testing.assert_array_equal(actual[key],expected[key])


def test_prefetch_preserves_primary_plan_and_allocates_fresh_confirmation_block(tmp_path):
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision import extend_confirmation_plan, write
    p=policy();cid=p['hard_cases'][0]
    plan=[dict(cases=[cid],particles=[8192,16384],systems=16,confirmation=False)]
    execution=dict(prefetch_confirmation_minimum_particles=16384)
    result=extend_confirmation_plan(tmp_path,plan,p,execution)
    assert len(plan)==1 and len(result)==2
    assert result[1]['particles']==[16384] and result[1]['systems']==16
    for index in range(16): write(tmp_path/'jobs'/f'confirm_n16384_{cid}_{index:03d}.json',{})
    write(tmp_path/'confirmations'/f'{cid}_n16384_r16.json',dict(accepted=False))
    assert extend_confirmation_plan(tmp_path,plan,p,execution)==plan
    plan[0]['systems']=32
    later=extend_confirmation_plan(tmp_path,plan,p,execution)
    assert later[1]['systems']==32 and later[1]['primary_systems']==32
    assert extend_confirmation_plan(tmp_path,later,p,execution)==later


def test_preallocated_memory_estimate_uses_new_layout_and_observed_adapter_peaks():
    from src.Bayesian_state.workflows.analysis.model_1024_state_precision import configure_storage
    protocol=dict(policy=policy(),estimates={'case':dict(cpu_seconds=100.,wall_seconds=100.,
                  rss_bytes=10*1024**3,particles=1024)})
    execution=dict(storage_minimum_particles=16384,worker_entry='adapter',worker_sources={'adapter':'hash'},
                   block_trials=16,case_dimensions={'case':dict(trials=1000,capacity=4,components=2,rules=100)})
    small=dict(state_case='case',particles=8192)
    assert 'state_worker_entry' not in configure_storage(small,execution)
    job=configure_storage(dict(state_case='case',particles=16384),execution)
    new=estimate(job,protocol,[])
    old=estimate(dict(state_case='case',particles=16384),protocol,[])
    assert new['memory'] < old['memory']/10
    row=dict(case='case',status='complete',particles=16384,cpu_seconds=1600.,wall_seconds=1600.,
             max_rss_kib=20*1024**2,worker_entry='adapter')
    assert estimate(job,protocol,[row])['memory'] >= row['max_rss_kib']*1024*1.3


def test_budget_continuation_selects_feasible_completed_resolution_without_acceptance(tmp_path,monkeypatch):
    from src.Bayesian_state.workflows.analysis import model_1024_state_precision_budget as continuation
    p=policy();p['budget']['wall_hours']=1
    continuation.write(tmp_path/'protocol.json',dict(policy=p,cases=[dict(id='case')]))
    continuation.write(tmp_path/'resource_policy_v2.json',{})
    continuation.write(tmp_path/'run_clock.json',dict(started_unix=0))
    monkeypatch.setattr(continuation.precision.study,'make_job',lambda c,p,o,s,n,i:dict(particles=n))
    monkeypatch.setattr(continuation.precision,'estimate',lambda j,p,l,w:dict(wall=j['particles']/2,cpu=100,memory=100))
    rows=[dict(case='case',particles=n,systems=16,confirmation=False) for n in (2048,4096,8192)]
    registration=dict(cases=['case'],particles=[8192,4096,2048],additional_systems=16)
    plan,evidence=continuation.choose_plan(tmp_path,registration,rows,now=1000)
    assert plan[0]['particles']==[4096] and plan[0]['systems']==32
    assert not plan[0]['confirmation'] and not evidence[0]['feasible']
    assert not list((tmp_path/'confirmations').glob('*'))
    plan,_=continuation.choose_plan(tmp_path,registration,rows,now=3599)
    assert not plan
