"""Bind precision interventions to their completed behavior and report data use."""
from copy import deepcopy
import json

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_decision_basis_replay as replay
from src.Bayesian_state.workflows.analysis.model_0923_decision_basis_alignment import estimation_context
from src.Bayesian_state.workflows.analysis import model_0923_oral_parameter_comparison as comparison


def save(path,value):
    path.write_text(json.dumps(value))


@pytest.fixture
def precision_source(tmp_path):
    (tmp_path/'jobs').mkdir();folder=tmp_path/'data';folder.mkdir()
    save(folder/'prepared.json',dict(subject=208))
    sources={replay.REENTRY:replay.digest(replay.ROOT/replay.REENTRY)}
    expected=dict(phase='evaluate',name='RETAIN_S208_point',folder=str(folder),runtime='archived',
        model='SEARCH',mechanism={'kind':'E'},arm_settings={'precision':'fast'},
        candidate=dict(id='point',hyperparams={'gamma':0}),retain_precision=True,seeds=[5,7],particles=512,
        source_sha256=sources,target=str(tmp_path/'combined'))
    chunks=[]
    for i,seed in enumerate(expected['seeds']):
        job=dict(deepcopy(expected),name=f'RETAIN_S208_point_C{i:03d}',seeds=[seed],target=str(tmp_path/f'chunk{i}'))
        save(tmp_path/'jobs'/(job['name']+'.json'),job);chunks.append(job)
    array=tmp_path/'combined.npz';array.write_bytes(b'fixture array hash; selector does not parse arrays')
    record={k:expected[k] for k in ('candidate','seeds','particles','model')}
    record['predictions_sha256']=replay.digest(array)
    plan=dict(point='point',variant='RETAIN',job=expected,chunks=chunks,reference='original')
    save(tmp_path/'protocol.json',dict(smoke=False,settings=dict(subject=208,points=['point']),
        plans=[plan],seeds=[5,7],particles=512,source_sha256=sources))
    save(tmp_path/'completion.json',dict(status='complete',smoke=False,filter_seeds=2,particles=512,
        records=[dict(point='point',variant='RETAIN',record=record,target=str(array))]))
    return dict(source_kind='reentry_precision_scores',subjects=[208],arms=['SEARCH'],
                parameter_point='point',seeds_per_arm=2),tmp_path


def test_precision_replay_preserves_declared_point_and_all_seeds(precision_source):
    settings,source=precision_source
    files,metadata=replay.source_jobs(settings,source,False)
    assert len(files)==2 and all('RETAIN' in p.name for p in files)
    assert metadata['selected_parameter_id']=='point'
    assert metadata['structural_intervention']=='retained_rule_precision'
    assert metadata['parameter_candidate_policy']=='declared_point_no_new_selection'
    assert len(replay.source_jobs(settings,source,True)[0])==1
    context,note=estimation_context(metadata)
    assert context['structural_intervention']=='retained_rule_precision'
    assert '保留已学精度' in note and '固定参数的开发诊断' in note
    assert '选择似然选定' not in note


@pytest.mark.parametrize('fault',['incomplete','smoke','point','duplicate_plan','retention_disabled',
    'seed_order','budget','chunk_identity','mechanism','reference','subject'])
def test_precision_replay_rejects_changed_mechanism_or_source(precision_source,fault):
    settings,source=precision_source
    protocol=replay.read(source/'protocol.json');done=replay.read(source/'completion.json')
    if fault=='incomplete':done['status']='running'
    elif fault=='smoke':done['smoke']=protocol['smoke']=True
    elif fault=='point':settings['parameter_point']='oral_favored_point'
    elif fault=='duplicate_plan':protocol['plans'].append(deepcopy(protocol['plans'][0]))
    elif fault=='retention_disabled':protocol['plans'][0]['job']['retain_precision']=False
    elif fault=='seed_order':protocol['seeds']=[7,5]
    elif fault=='budget':protocol['particles']=256
    elif fault=='chunk_identity':
        file=source/'jobs'/'RETAIN_S208_point_C000.json';job=replay.read(file)
        job['candidate']['hyperparams']['gamma']=.8;save(file,job)
    elif fault=='mechanism':protocol['source_sha256'][replay.REENTRY]='changed'
    elif fault=='reference':(source/'combined.npz').write_bytes(b'changed')
    elif fault=='subject':save(source/'data'/'prepared.json',dict(subject=207))
    save(source/'protocol.json',protocol);save(source/'completion.json',done)
    with pytest.raises(ValueError):replay.source_jobs(settings,source,False)


def test_parameter_comparison_rejects_a_structural_intervention(tmp_path,monkeypatch):
    original=dict(scientific={},mechanism=dict(retain_learned_precision=False),arm_settings={},
        archived_pf_sha256='same',task_sha256={},oral_manifest_sha256='same',model_sources={})
    revised=deepcopy(original);revised['mechanism']['retain_learned_precision']=True
    def load(directory,subject,arm):
        return None,original if directory.name=='original' else revised,{}
    monkeypatch.setattr(comparison,'load_group',load)
    output=tmp_path/'output'
    with pytest.raises(ValueError,match='same task, mechanism'):
        comparison.run(tmp_path/'original',tmp_path/'revised',output,208,'SEARCH')
    assert not output.exists()
