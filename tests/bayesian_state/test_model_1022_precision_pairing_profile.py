"""Protect the scope and incumbent of the retrospective two-axis profile."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis import model_1022_precision_pairing_profile as profile


def test_profile_crosses_complete_support_and_preserves_every_other_coordinate():
    space={profile.BETA:[.1,.5,1,2,5,10,15,25],profile.GAMMA:[0,.3,.6,.8,.95,.99,1],
           'search':[.05,.35], 'capacity':[3,4]}
    anchor={'hyperparams':{profile.BETA:5,profile.GAMMA:.8,'search':.05,'capacity':3}}
    untouched=deepcopy(anchor)
    bank,required=profile.profile_bank(space,anchor)
    assert anchor==untouched and len(bank)==len({r['id'] for r in bank})==56
    assert bank[0]==required and required['hyperparams']==anchor['hyperparams']
    assert {(r['hyperparams'][profile.BETA],r['hyperparams'][profile.GAMMA]) for r in bank}=={
        (b,g) for b in space[profile.BETA] for g in space[profile.GAMMA]}
    assert all(r['hyperparams']['search']==.05 and r['hyperparams']['capacity']==3 for r in bank)
    bad=deepcopy(anchor);bad['hyperparams']['capacity']=9
    with pytest.raises(ValueError,match='outside'):profile.profile_bank(space,bad)


def test_new_stage_seed_bank_is_disjoint_from_previous_protocol():
    old=dict(protocol='old',seed=12,budgets={s:dict(seeds=n) for s,n in [('screen',2),('selection',4),('confirmation',8)]})
    new=dict(old,protocol='new',seed=34)
    assert len(profile.seed_bank(new,306))==14
    assert not profile.seed_bank(old,306)&profile.seed_bank(new,306)
