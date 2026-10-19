"""Inspect active candidates behind frozen107 oral feature groups.

Use --annotations FROZEN_TEXT --capture ORIGINAL_PARTICLE_CAPTURE (repeatable)
--output NEW_DIRECTORY. For all four features and all 192 trials, distinguish
any boundary involving that feature from a single-feature boundary. Keep active
availability, belief mass, response contribution and candidate preference apart.
No oral likelihood is defined; text groups only select descriptive summaries.
Snapshots establish represented active rules, not conscious attention or the
historical process by which a participant accepted or rejected a candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..hypothesis_space.spaces import continuous
from ..workflows.analysis.model_0923_target_state_recovery import digest, read, verify, write
from .model_0923_rnn_discourse import verify_trial_identity
from .oral import candidate_cues

FEATURES = ('neck','tail','head','leg')


def run(annotations: Path, captures: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new active-candidate diagnostic directory')
    inputs={}

    def bind(path):
        path=Path(path).resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:
            raise ValueError('Original candidate-state input changed')
        inputs[str(path)]=sha;return path

    def record(path):
        return read(bind(path))

    import inspect
    for path in (__file__,candidate_cues.__file__,continuous.__file__,inspect.getfile(verify_trial_identity)):
        bind(path)
    audit=record(annotations/'completion.json');freeze=record(annotations/'text_annotation_freeze.json')
    if (audit['status']!='complete' or freeze['status']!='complete' or audit['subject']!=107
            or freeze['model_states_read'] or tuple(freeze['feature_order'])!=FEATURES):
        raise ValueError('Require original model-independent107 text groups')
    verify(freeze['inputs_sha256']);inputs.update(freeze['inputs_sha256'])
    file=bind(annotations/'text_annotations.csv')
    if digest(file)!=freeze['text_sha256'] or digest(file)!=audit['text_first_annotation_sha256']:
        raise ValueError('Original text annotations changed')
    text=pd.read_csv(file,keep_default_na=False).sort_values('state_trial').reset_index(drop=True)
    space=continuous.build_continuous_hypothesis_space(4,2)
    rule_features=np.array([np.any(np.asarray([plane[0] for plane in rule.hyperplanes])!=0,axis=0) for rule in space])
    for rule,used in zip(space,rule_features):
        rows=np.concatenate([part.A for category in rule.categories for part in category.components])
        np.testing.assert_array_equal(np.any(rows!=0,axis=0),used)
    scopes={'any_rule':rule_features,'single_feature_rule':rule_features&(rule_features.sum(1)==1)[:,None]}
    features=[];rules=[];groups=[];arms=[];max_reconstruction=0.
    for capture in captures:
        done=record(capture/'completion.json')
        if done['status']!='complete' or done['parameters_changed'] or done['oral_used']:
            raise ValueError('Require an unchanged report-independent particle capture')
        verify(done['inputs_sha256']);inputs.update(done['inputs_sha256'])
        jobs=list((capture/'jobs').glob('*.json'))
        if len(jobs)!=1:
            raise ValueError('Use one original preselected filter per capture')
        job=record(jobs[0]);target=Path(job['target']);state=record(target.with_suffix('.json'))
        proof=record(target.with_suffix('.particle_basis.json'))
        engine=record(Path(job['folder'])/'engine.json')
        if (proof!=done['worker'] or proof['schema_version']!=1 or state['subject']!=107 or state['condition']!=1
                or state['trials']!=192 or state['training_trials']!=128 or len(state['seeds'])!=1
                or state['arm'] in arms or not state['exact_behavior_replay']
                or state['catalogue_signature']!=json.loads(json.dumps(space.signature))
                or engine['provenance']['label_mapping']['mode']!='fixed_task_labels'):
            raise ValueError('Capture scientific identity or catalogue differs')
        file=bind(target.with_suffix('.npz'))
        if digest(file)!=state['array_sha256'] or digest(file)!=proof['array_sha256']:
            raise ValueError('Captured rule states changed')
        with np.load(file) as z:data={key:z[key] for key in z.files}
        verify_trial_identity(text,data['choices'],state['training_trials'])
        arms.append(state['arm'])
        for t,row in text.iterrows():
            ids=data['particle_basis_rule_ids'][t];active=ids>=0
            p=data['particle_basis_category_probabilities'][t]
            b=data['particle_basis_active_belief'][t];u=data['particle_basis_choice_weights'][t]
            beta=data['particle_basis_active_beta'][t]
            if beta.shape!=ids.shape or not np.isfinite(beta).all() or np.any(beta<0) or np.any(beta[~active]!=0):
                raise ValueError('Invalid original rule precision')
            predicted=np.sum(u[...,None]*p,axis=1)
            np.testing.assert_allclose(predicted,data['particle_basis_prediction'][t],atol=1e-12,rtol=0)
            max_reconstruction=max(max_reconstruction,float(abs(predicted-data['particle_basis_prediction'][t]).max()))
            prior=data['particle_basis_prior_weights'][t]
            conditional=prior*predicted[:,int(row.choice)-1];conditional/=conditional.sum()
            np.testing.assert_allclose(conditional,data['conditional_particle_weights'][t],atol=1e-12,rtol=0)
            identity=dict(subject=107,arm=state['arm'],state_trial=t,iSession=int(row.iSession),iTrial=int(row.iTrial),
                          phase=row.phase,choice=int(row.choice),text=row.text,nonempty=bool(row.nonempty))
            definitions=dict(all_named_or_collective=int(row.named_or_collective_mask),
                literal_contrast_before=int(row.before_mask) if row.has_contrast else 0,
                literal_contrast_after=int(row.after_mask) if row.has_contrast else 0,
                original_complete_encoded=int(row.encoded_mask) if row.original_scope=='complete' else 0)
            for weighting,w in [('prior',prior),('choice_conditioned',conditional)]:
                q=p[...,int(row.choice)-1];other=p[...,2-int(row.choice)]
                for scope,membership in scopes.items():
                    profile=candidate_cues.candidate_feature_profiles(ids,membership,p,b,u,w,int(row.choice))
                    for d,feature in enumerate(FEATURES):
                        features.append(dict(identity,weighting=weighting,scope=scope,feature=feature,
                            named=bool(row.named_or_collective_mask&(1<<d)),
                            contrast_before=bool(row.has_contrast and row.before_mask&(1<<d)),
                            contrast_after=bool(row.has_contrast and row.after_mask&(1<<d)),
                            complete_encoded=bool(row.original_scope=='complete' and row.encoded_mask&(1<<d)),
                            **{key:float(value[d]) for key,value in profile['means'].items()}))
                    for name,mask in definitions.items():
                        if mask and row.nonempty:
                            chosen=((mask>>np.arange(4))&1).astype(bool)
                            joint=float(w@np.all(profile['particle']['availability'][:,chosen],axis=1))
                            groups.append(dict(identity,weighting=weighting,scope=scope,group=name,
                                               feature_mask=mask,joint_availability=joint))
                for rule in space:
                    hit=ids==rule.index
                    presence=float(w@hit.any(1));belief=float(w@np.sum(hit*b,axis=1))
                    contribution=float(w@np.sum(hit*u,axis=1))
                    beta_mass=float(w@np.sum(hit*beta,axis=1));choice_mass=float(w@np.sum(hit*q,axis=1))
                    rules.append(dict(identity,weighting=weighting,rule=rule.index,family=rule.family,
                        presence=presence,belief_mass=belief,choice_weight=contribution,beta_presence_mass=beta_mass,
                        selected_probability_presence_mass=choice_mass,
                        beta_given_presence=beta_mass/presence if presence else np.nan,
                        selected_probability_given_presence=choice_mass/presence if presence else np.nan,
                        opposing_margin_mass=float(w@np.sum(hit*np.maximum(other-q,0),axis=1))))
                    if weighting=='choice_conditioned':
                        np.testing.assert_allclose(presence,data['filtered_active'][t,rule.index],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(belief,data['filtered_belief'][t,rule.index],atol=1e-12,rtol=0)
    if not arms:raise ValueError('Supply at least one original capture')
    ft,rt,gt=map(pd.DataFrame,(features,rules,groups));summary=[]
    measures=['availability','rule_count','belief_mass','choice_weight','max_opposing_margin',
              'max_supporting_margin','belief_opposing_margin','contribution_opposing_margin']
    keys=['arm','weighting','scope','phase','feature']
    for key,part in ft.groupby(keys):
        for subset,mask in [('all_trials',np.ones(len(part),bool)),('named',part.named),
                            ('contrast_before',part.contrast_before),('contrast_after',part.contrast_after),
                            ('complete_encoded',part.complete_encoded)]:
            chosen=part.loc[mask]
            if len(chosen):summary.append(dict(zip(keys,key),subset=subset,reports=len(chosen),
                **{name:float(chosen[name].mean()) for name in measures}))
    output.mkdir(parents=True)
    for name,table in [('feature_trials',ft),('rule_trials',rt),('group_trials',gt),('feature_summary',pd.DataFrame(summary))]:
        table.to_csv(output/(name+'.csv'),index=False)
    rs=rt.groupby(['arm','weighting','phase','rule','family'])[['presence','belief_mass','choice_weight',
        'beta_presence_mass','selected_probability_presence_mass','opposing_margin_mass']].mean().reset_index()
    for key,name in [('beta_presence_mass','beta_given_presence'),('selected_probability_presence_mass','selected_probability_given_presence')]:
        rs[name]=np.divide(rs[key],rs.presence,out=np.full(len(rs),np.nan),where=rs.presence>0)
    rs.to_csv(output/'rule_summary.csv',index=False)
    gt.groupby(['arm','weighting','scope','phase','group']).agg(reports=('state_trial','size'),
        joint_availability=('joint_availability','mean')).reset_index().to_csv(output/'group_summary.csv',index=False)
    write(output/'catalogue.json',dict(signature=space.signature,feature_order=FEATURES,rules=[dict(index=r.index,
        family=r.family,features=[FEATURES[d] for d in np.flatnonzero(rule_features[r.index])],
        hyperplanes=r.hyperplanes) for r in space]))
    verify(inputs)
    write(output/'completion.json',dict(status='complete',subject=107,trials=192,training_trials=128,arms=arms,
        inputs_sha256=inputs,artifacts_sha256={p.name:digest(p) for p in output.iterdir()},
        feature_rows=len(ft),rule_rows=len(rt),group_rows=len(gt),maximum_reconstruction_error=max_reconstruction,
        original_choice_reconstructed=True,original_active_and_belief_reconstructed=True,
        oral_used_for_fit=False,parameters_changed=False,original_coding_changed=False,
        scope='One original preselected filter per arm; active structural coverage and probability preference',
        interpretation='Structural membership and state availability do not establish report entailment, attention or a history of rejection'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('annotations','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--capture',required=True,action='append',type=Path)
    args=parser.parse_args();run(args.annotations,args.capture,args.output)
