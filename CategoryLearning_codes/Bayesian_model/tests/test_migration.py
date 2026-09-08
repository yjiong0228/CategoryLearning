"""Migration parity against frozen source; not a model-fit quality claim."""
from copy import deepcopy
from pathlib import Path
import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT/'CategoryLearning_codes/Bayesian_model'


def test_new_import_and_resource_are_independent():
    from CategoryLearning_codes.Bayesian_model.model import StateModel
    from CategoryLearning_codes.Bayesian_model.hypothesis_space.similarity import ContinuousSimilarity
    assert StateModel.__module__.startswith('CategoryLearning_codes.Bayesian_model.')
    assert ContinuousSimilarity.RESOURCE_DIR.is_relative_to(PACKAGE)


@pytest.mark.parametrize('execution', [False,True])
@pytest.mark.parametrize('transport', ['similarity_transport','mass_preserving_similarity_transport'])
def test_fixed_seed_pf_matches_source(execution,transport):
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter as old_run
    from CategoryLearning_codes.Bayesian_model.inference.backends.particle_filter import run_state_model_particle_filter as new_run
    old=yaml.safe_load((ROOT/'configs/model_struct/pmh_model_cond1_0826.yaml').read_text())
    new=yaml.safe_load((PACKAGE/'configs/model_0826.yaml').read_text())
    for config in [old,new]:
        transition=config['modules']['hypo_transitions_mod']['kwargs']
        transition['persistent_execution']['enabled']=execution
        transition['prior_assignment']['method']=transport
        transition['nested_feedback_accumulator_controller']['accumulator_logit_gain']=.4
        transition['nested_feedback_accumulator_controller']['global_search_failure_gain']=.3
    data=pd.read_csv(ROOT/'data/processed/Task2_processed.csv')
    sub=data[data.iSub.eq(101)].sort_values(['iSession','iTrial']).iloc[:16]
    kwargs=dict(subject_id=101,stimulus=sub[[f'feature{i}' for i in range(1,5)]].to_numpy(),
                choices=sub.choice.to_numpy(),feedback=sub.feedback.to_numpy(),
                particle_count=4,choice_readout_power=1.,filter_seed=8326)
    a=old_run(engine_config=old,**kwargs);b=new_run(engine_config=new,**kwargs)
    for attr in ['observation_probabilities','state_probabilities','latent_summaries']:
        left,right=getattr(a,attr),getattr(b,attr)
        assert left.keys()==right.keys()
        for key in left:
            if left[key] is None:assert right[key] is None
            else: np.testing.assert_equal(left[key],right[key],err_msg=f'{attr}.{key}')
    np.testing.assert_equal(a.resampled,b.resampled)
    np.testing.assert_allclose(b.marginal_probabilities.sum(axis=1),1.,atol=1e-12)
    assert np.isfinite(b.marginal_probabilities).all()


@pytest.mark.parametrize('execution',[False,True])
def test_autonomous_choices_and_states_match_source(execution):
    from src.Bayesian_state.simulation.autonomous import run_autonomous_category_learning as old_run
    from CategoryLearning_codes.Bayesian_model.simulation.autonomous import run_autonomous_category_learning as new_run
    old=yaml.safe_load((ROOT/'configs/model_struct/pmh_model_cond1_0826.yaml').read_text())
    new=yaml.safe_load((PACKAGE/'configs/model_0826.yaml').read_text())
    for cfg in [old,new]:cfg['modules']['hypo_transitions_mod']['kwargs']['persistent_execution']['enabled']=execution
    data=pd.read_csv(ROOT/'data/processed/Task2_processed.csv')
    sub=data[data.iSub.eq(101)].sort_values(['iSession','iTrial']).iloc[:24]
    kw=dict(subject_id=101,condition=1,stimulus=sub[[f'feature{i}' for i in range(1,5)]].to_numpy(),
            categories=sub.category.to_numpy(),trajectory_seed=8261)
    a=old_run(engine_config=old,**kw).trajectory;b=new_run(engine_config=new,**kw).trajectory
    for key in ['choices','feedback','perceived_stimulus','prior','posterior','beta','cognitive_probabilities','observed_probabilities']:
        np.testing.assert_equal(getattr(a,key),getattr(b,key),err_msg=key)


def test_package_imports_with_legacy_namespace_blocked():
    import subprocess
    program='''
import importlib, importlib.abc, json, sys
from pathlib import Path
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith('src.Bayesian_state'):
            raise AssertionError('Legacy dependency: '+fullname)
sys.meta_path.insert(0,BlockLegacy())
p=Path('CategoryLearning_codes/Bayesian_model')
for file in p.rglob('*.py'):
    rel=file.relative_to(p)
    if rel.parts[0] in ('tests','outputs'):continue
    name='CategoryLearning_codes.Bayesian_model.'+str(rel).removesuffix('.py').replace('/','.').removesuffix('.__init__')
    if str(rel)=='__init__.py':name='CategoryLearning_codes.Bayesian_model'
    importlib.import_module(name)
'''
    subprocess.run([sys.executable,'-c',program],cwd=ROOT,check=True,capture_output=True,text=True)


def test_frozen_config_and_resource_match_manuscript():
    import hashlib
    old=(ROOT/'configs/model_struct/pmh_model_cond1_0826.yaml').read_text()
    config=yaml.safe_load((PACKAGE/'configs/model_0826.yaml').read_text())
    assert config==yaml.safe_load(old.replace('src.Bayesian_state','CategoryLearning_codes.Bayesian_model'))
    assert hashlib.sha256((ROOT/config['provenance']['manuscript_path']).read_bytes()).hexdigest()==config['provenance']['manuscript_sha256']
    similarity=config['provenance']['hypothesis_similarity']
    resource=PACKAGE/'hypothesis_space/resources/similarity'/similarity['resource_filename']
    assert hashlib.sha256(resource.read_bytes()).hexdigest()==similarity['resource_sha256']
    assert np.load(resource).shape==(29,29)
    from CategoryLearning_codes.Bayesian_model.evaluation.model_recovery import load_recovery_design
    for version in [1,2]:
        design=load_recovery_design(PACKAGE/f'configs/recovery_v{version}.yaml')
        assert design.model_engine_config.is_relative_to(PACKAGE)
        assert design.parameter_space_path.is_relative_to(PACKAGE)
        assert design.base_simulation_config.is_relative_to(PACKAGE)
