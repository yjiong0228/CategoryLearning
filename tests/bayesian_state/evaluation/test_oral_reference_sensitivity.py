"""A fixed semantic sensitivity reading never chooses interpretations by fit."""
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_reference_sensitivity import (
    relative_tokens, change_reading, assemble,
)
from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import (
    numerical_constraints, audit_query,
)
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.evaluation.oral.constraint_alignment import evaluate_component
from src.Bayesian_state.workflows.analysis.model_0923_oral_scope_comparison import compare_reference_frames


def fixture():
    rows=[]
    for trial,text in enumerate(['头长，腿短','头长于一半，腿短于一半','头比腿长','头长'],1):
        r=encode_report(text,('head','neck','leg','tail'),half_reference='range_midpoint')
        rows.append(dict(iSub=208,iSession=1,iTrial=trial,condition=2,choice=1,text=text,
            tokens=json.dumps(r.tokens),flags=json.dumps(r.flags),n_predicates=len(r.tokens),
            scope='complete',query_id='original'+str(trial)))
    return pd.DataFrame(rows)


def test_all_cross_group_pairs_have_the_intended_direction():
    tokens=relative_tokens([0,1],[2,3]);A,b=numerical_constraints(tokens)
    assert len(tokens)==4
    for x in itertools.permutations([.1,.3,.7,.9]):
        assert bool((A@x<=b).all())==(min(x[:2])>=max(x[2:]))
    np.testing.assert_array_equal(b,np.zeros(4))


def test_relative_reading_equals_explicit_comparison_without_changing_half_reports():
    original=fixture();result,queries=change_reading(original)
    assert result.reference_sensitivity_changed.tolist()==[True,False,False,False]
    assert json.loads(result.tokens.iloc[0])==list(encode_report('头比腿长',('head','neck','leg','tail')).tokens)
    pd.testing.assert_frame_equal(result.loc[1:,original.columns],original.loc[1:])
    pd.testing.assert_series_equal(result.original_tokens,original.tokens,check_names=False)
    assert len(queries)==1


def test_geometry_can_imply_relative_content_without_explicit_predicate_support():
    tokens=relative_tokens([0],[2])
    r=encode_report('头长，腿短',('head','neck','leg','tail'))
    verdict=evaluate_component(tokens,*numerical_constraints(r.tokens))
    assert verdict.structural=='undetermined' and verdict.functional=='support'


def test_conversion_ignores_stimulus_and_model_scores_and_is_deterministic():
    original=fixture();a,_=change_reading(original)
    original['stimulus_match']=[False,True,False,True]
    original['model_support']=[0.,1.,.5,.2]
    b,_=change_reading(original)
    for name in ('tokens','query_id','reference_sensitivity_changed'):
        pd.testing.assert_series_equal(a[name],b[name])


def test_incomplete_bare_groups_keep_original_encoding():
    original=fixture();original.loc[0,'scope']='parsed_fragment'
    result,queries=change_reading(original)
    assert not result.reference_sensitivity_changed.any() and not queries
    pd.testing.assert_frame_equal(result[original.columns],original)


def test_same_original_query_can_remain_for_an_explicit_half_report():
    original=fixture().iloc[:2].copy();original['query_id']='shared'
    tokens=json.loads(original.tokens.iloc[0])
    summary,matches=audit_query(dict(query_id='shared',condition=2,choice=1,tokens=tokens,tolerance=1e-9))
    for name,value in summary.items():
        if name not in ('query_id','condition','choice','tokens'):original[name]=float(value) if isinstance(value,int) else value
    converted,queries=change_reading(original)
    new=audit_query(dict(queries[0],tolerance=1e-9))
    result,qs,cs=assemble(converted,pd.DataFrame([summary]),pd.DataFrame(matches),[new])
    assert set(qs.query_id)=={'shared',queries[0]['query_id']}
    assert result.query_id.iloc[1]=='shared'
    pd.testing.assert_frame_equal(result.loc[1:,original.columns],original.loc[1:])
    assert not cs.duplicated(['query_id','rule','component']).any()


@pytest.mark.parametrize('long,short',[([],[0]),([0],[0]),([0,0],[1]),([4],[0])])
def test_invalid_dimension_groups_are_rejected(long,short):
    with pytest.raises(ValueError):relative_tokens(long,short)


def alignment_frames():
    old=fixture();old['ordinal_trial']=old.iTrial;old['arm']='BASE';old['training_trials']=2
    old['status']='coded';old['mean_choice_probability']=.6
    for k,name in enumerate(('head','neck','leg','tail'),1):old[f'feature{k}_name']=name
    for prefix in ('','prior_'):
        for domain in ('structural','functional'):
            for label,value in [('support',.6),('conflict',.1),('undetermined',.3)]:old[prefix+domain+'_'+label]=value
    new,_=change_reading(old)
    new.loc[0,'structural_support']=.2;new.loc[0,'structural_undetermined']=.7
    return old,new


def test_reading_comparison_preserves_all_changed_and_unchanged_denominators():
    old,new=alignment_frames();summary,changed=compare_reference_frames(old,new)
    assert len(changed)==1
    rows=summary.query("window=='whole' and domain=='structural' and scope=='complete_all'").set_index('reading')
    assert rows.reports.to_dict()=={'absolute':4,'relative':4}
    assert rows.support.to_dict()==pytest.approx({'absolute':.6,'relative':.5})


@pytest.mark.parametrize('fault',['choice','scope','reference','untouched_score','probability'])
def test_reading_comparison_rejects_unrelated_changes(fault):
    old,new=alignment_frames()
    if fault=='choice':new.loc[1,'choice']=2
    elif fault=='scope':new.loc[1,'scope']='needs_review'
    elif fault=='reference':new.loc[0,'tokens']=old.tokens.iloc[0]
    elif fault=='untouched_score':new.loc[1,'structural_support']=.2
    else:new.loc[1,'mean_choice_probability']=.2
    with pytest.raises((ValueError,AssertionError)):compare_reference_frames(old,new)
