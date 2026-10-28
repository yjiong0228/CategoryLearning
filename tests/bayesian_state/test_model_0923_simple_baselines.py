"""Validate single-rule mechanics and the common full-v3 feedback contract."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.model_0923 import validate_model_0923_config, is_model_0923_revised
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.simple_rule import SimpleRuleHypothesisTransitionModule
from src.Bayesian_state.workflows.analysis.model_0923_simple_baselines import make_engine, candidate_points


ROOT = Path(__file__).resolve().parents[2]


def configuration(variant="fixed_rule", rate=0.):
    original=yaml.safe_load((ROOT/"configs/exp123/model_struct/model_0923_cond3_full_v3.yaml").read_text())
    engine=make_engine(original,variant,dict(beta_init=2.,gamma=.6,learning_rate=.3,switch_probability=rate))
    engine["modules"]["perception_mod"]["kwargs"]={"features":4,"mean":[0.]*4,"std":[0.]*4,"module_seed":7}
    engine["modules"]["hypo_transitions_mod"]["kwargs"]["module_seed"]=11
    return original,engine


@pytest.mark.parametrize("variant,rate",[("fixed_rule",0.),("random_rule",1.)])
def test_rule_retention_switching_and_pairing_marginal(monkeypatch,variant,rate):
    # Distance is a logged diagnostic with global proposals; skip its costly
    # catalogue simulation while exercising real cognitive updates.
    monkeypatch.setattr(SimpleRuleHypothesisTransitionModule,"_newcomer_distance",lambda *args:0.)
    original,engine=configuration(variant,rate)
    model=StateModel(engine,context=ModelContext(condition=3))
    rules=[]
    for i,feedback in enumerate([0.,.5,1.,0.]):
        model.begin_trial(np.array([.8,.3,.2,.4]))
        active=np.flatnonzero(model.engine.hypotheses_mask)
        assert len(active)==1
        rules.append(int(active[0]))
        memory=model.engine.get_module(ModuleRole.MEMORY)
        np.testing.assert_allclose(memory.joint.sum(axis=1),model.engine.prior,atol=1e-12)
        model.complete_trial((i%4)+1,feedback)
        assert np.isfinite(model.engine.beta).all()
    assert all((a==b) is (rate==0) for a,b in zip(rules[:-1],rules[1:]))
    assert original["provenance"]["variant"]=="full"
    assert is_model_0923_revised(engine["provenance"])


@pytest.mark.parametrize("change",[{"capacity":2},{"capacity":True},{"g":.5},{"m":.1},
                                    {"prior_assignment":{"method":"similarity_transport"}}])
def test_fixed_rule_contract_rejects_unintended_mechanisms(change):
    _,engine=configuration()
    engine["modules"]["hypo_transitions_mod"]["kwargs"].update(change)
    with pytest.raises(ValueError):
        validate_model_0923_config(engine)


def test_simple_variants_preserve_beta_and_feedback_semantics():
    _,engine=configuration()
    altered=deepcopy(engine)
    altered["modules"]["beta_mod"]["kwargs"]["beta_update_mode"]="hierarchical_feedback"
    with pytest.raises(ValueError):
        validate_model_0923_config(altered)
    validate_model_0923_config(engine)


def test_candidate_screen_is_deterministic_distinct_and_bounded():
    original,_=configuration()
    settings=yaml.safe_load((ROOT/"configs/shared/model0923_simple_baselines.yaml").read_text())
    for variant in settings["variants"]:
        points=candidate_points(settings,original,variant,335)
        assert points==candidate_points(settings,original,variant,335)
        assert len(points)==24
        assert len({tuple(sorted(p.items())) for p in points})==24
        for point in points:
            validate_model_0923_config(make_engine(original,variant,point))
