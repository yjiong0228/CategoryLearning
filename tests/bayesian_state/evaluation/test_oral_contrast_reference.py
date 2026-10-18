"""Reference diagnostics preserve explicit semantics, geometry and identities."""
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_contrast_reference import (
    contrast_dimensions, stimulus_diagnostics, join_stimuli,
)


def tokens():
    return json.dumps(['facet:[[-1,0,0,0,-0.5]]','facet:[[0,-1,0,0,-0.5]]',
                       'facet:[[0,0,1,0,0.5]]','facet:[[0,0,0,1,0.5]]'])


def test_bare_groups_use_existing_audited_dimension_order():
    reason,long,short=contrast_dimensions('头和脖子长，尾巴和腿短。',tokens(),'complete')
    assert (reason,long,short)==('eligible',[0,1],[2,3])


@pytest.mark.parametrize('text',['头长于一半，腿短','头比腿长','头最长，腿短','头长于身体，腿短','头超过0.5，腿短'])
def test_explicit_references_are_kept_out_of_bare_group_audit(text):
    assert contrast_dimensions(text,tokens(),'complete')[0]=='explicit_reference_or_ordering'


def test_incomplete_and_relational_or_single_polarity_reports_are_retained_as_ineligible():
    assert contrast_dimensions('头长，腿短',tokens(),'needs_review')[0]=='outside_complete_report_scope'
    assert contrast_dimensions('头长，腿短',json.dumps(['facet:[[-1,1,0,0,0]]']),'complete')[0]=='other_predicate'
    assert contrast_dimensions('头长',json.dumps(['facet:[[-1,0,0,0,-0.5]]']),'complete')[0]=='single_polarity_or_empty'


def test_relative_group_can_hold_when_all_parts_are_below_midpoint():
    row=stimulus_diagnostics(np.array([.49,.40,.10,.04]),[0,1],[2,3])
    assert row['relative_match'] and row['relative_only'] and row['all_reported_below_midpoint']
    assert not row['absolute_match'] and row['relative_margin']==pytest.approx(.30)


def test_relative_region_volume_matches_all_four_dimensional_orderings():
    for size in (1,2,3):
        long=list(range(size));short=list(range(size,4))
        passed=0
        for order in itertools.permutations([.1,.3,.7,.9]):
            r=stimulus_diagnostics(np.array(order),long,short)
            assert not r['absolute_match'] or r['relative_match']
            passed+=r['relative_match']
        assert passed/24==r['relative_unit_cube_volume']
        assert r['relative_unit_cube_volume']>r['absolute_unit_cube_volume']


def test_exact_group_ties_are_reported():
    row=stimulus_diagnostics(np.full(4,.5),[0,1],[2,3])
    assert row['absolute_match'] and row['relative_match'] and row['relative_tie']


@pytest.mark.parametrize('fault',['duplicate','names','choice','order','missing'])
def test_stimulus_join_rejects_changed_identities(fault):
    common=dict(iSub=[208,208],iSession=[1,1],iTrial=[1,2],condition=[2,2],choice=[1,2])
    for k,name in enumerate(['neck','head','tail','leg'],1):common[f'feature{k}_name']=[name,name]
    oral=pd.DataFrame(dict(common,ordinal_trial=[1,2]))
    behavior=pd.DataFrame(dict(common,category=[1,2],**{f'feature{k}':[.2,.4] for k in range(1,5)}))
    assert len(join_stimuli(oral,behavior))==2
    if fault=='duplicate':behavior=pd.concat([behavior,behavior.iloc[:1]],ignore_index=True)
    elif fault=='names':behavior.loc[0,'feature1_name']='tail'
    elif fault=='choice':behavior.loc[0,'choice']=3
    elif fault=='order':behavior=behavior.iloc[::-1]
    else:behavior=behavior.iloc[:1]
    with pytest.raises((ValueError,AssertionError)):join_stimuli(oral,behavior)
