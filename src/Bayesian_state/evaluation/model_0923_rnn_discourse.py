"""Read frozen107 text groups from RNN probes and compare original H diagnostics.

Use --rnn COMPLETED_RUN --annotations FROZEN_TEXT --h-profile H_DISCOURSE
(repeatable) --output NEW_DIRECTORY. All 192 trial identities and four group
coverage rows per trial are retained. The shared group-effect calculation uses
all-feature reflection contexts for every model. Text order supplies group
membership; it does not assign supporting or opposing roles. Parameters and
probes remain frozen, and oral data never change a model's state weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflows.analysis.model_0923_target_state_recovery import digest, read, verify, write
from .oral import discourse_effects, joint_relevance

FEATURES = ('neck', 'tail', 'head', 'leg')
MEASURES = ('actual_effect', 'positive_context_effect', 'negative_context_effect', 'absolute_context_effect')
GROUP_MEASURES = ('minimum_positive_effect', 'minimum_absolute_effect', 'minimum_actual_positive_effect',
                  'mean_positive_effect', 'mean_negative_effect', 'joint_reflection_effect')


def verify_trial_identity(frame: pd.DataFrame, choices: np.ndarray, cut: int) -> None:
    """Reject changed trial order, choice history, task boundary or text flags."""
    if (len(frame) != 192 or cut != 128 or frame.subject.nunique() != 1 or frame.subject.iloc[0] != 107
            or frame.state_trial.tolist() != list(range(192)) or frame.duplicated(['iSession','iTrial']).any()
            or frame.nonempty.dtype != bool or frame.has_contrast.dtype != bool):
        raise ValueError('Require the original frozen107 text identities and flags')
    np.testing.assert_array_equal(frame.choice, choices)
    np.testing.assert_array_equal(frame.phase, np.where(np.arange(192)<cut,'training','later'))
    for key in ('named_or_collective_mask','before_mask','after_mask','encoded_mask'):
        values = frame[key].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or np.any((values<0)|(values>15)):
            raise ValueError('Require four-coordinate integer feature masks')


def run(rnn: Path, annotations: Path, h_profiles: list[Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new cross-model discourse directory')
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest(path)
        return path

    def record(path):
        return read(bind(path))

    def completed(folder):
        done = record(folder/'completion.json')
        if done['status'] != 'complete':
            raise ValueError('Require completed artifacts')
        verify(done.get('inputs_sha256', {}))
        inputs.update(done.get('inputs_sha256', {}))
        for name, sha in done.get('artifacts_sha256', {}).items():
            if digest(bind(folder/name)) != sha:
                raise ValueError('Completed artifact changed')
        return done

    for file in (__file__, discourse_effects.__file__, joint_relevance.__file__):
        bind(file)
    done, audit = completed(rnn), completed(annotations)
    freeze = record(annotations/'text_annotation_freeze.json')
    if (done['smoke'] or done['oral_used'] or done['test_used_for_selection'] or audit['subject']!=107
            or freeze['status']!='complete' or freeze['model_states_read'] or tuple(freeze['feature_order'])!=FEATURES):
        raise ValueError('Require frozen behavior-only RNN and original text-only annotations')
    verify(freeze['inputs_sha256']); inputs.update(freeze['inputs_sha256'])
    file = bind(annotations/'text_annotations.csv')
    if digest(file)!=freeze['text_sha256'] or digest(file)!=audit['text_first_annotation_sha256']:
        raise ValueError('Text changed after its original model-independent grouping')
    frame = pd.read_csv(file, keep_default_na=False).sort_values('state_trial').reset_index(drop=True)
    original = next(r for r in done['evaluations'] if r['subject']==107)
    file = bind(rnn/'evaluation/evaluate_S107.npz')
    if digest(file)!=original['predictions_sha256'] or not original['exact_original_predictions']:
        raise ValueError('Original frozen RNN prediction changed')
    with np.load(file) as z:
        data = {key:z[key] for key in z.files}
    verify_trial_identity(frame,data['choices'],original['training_trials'])
    masks = ((np.arange(16)[:,None]>>np.arange(4))&1).astype(bool)
    np.testing.assert_array_equal(data['reflection_masks'],masks)
    if data['reflection_probabilities'].shape!=(192,16,2):
        raise ValueError('Require original107 two-response reflection probes')
    groups, features, coverage = [], [], []
    for t, row in frame.iterrows():
        definitions = dict(all_named_or_collective=int(row.named_or_collective_mask),
            literal_contrast_before=int(row.before_mask) if row.has_contrast else 0,
            literal_contrast_after=int(row.after_mask) if row.has_contrast else 0,
            original_complete_encoded=int(row.encoded_mask) if row.original_scope=='complete' else 0)
        active = {}
        for name, mask in definitions.items():
            available = bool(mask and row.nonempty)
            coverage.append(dict(subject=107,state_trial=t,iSession=int(row.iSession),iTrial=int(row.iTrial),
                phase=row.phase,text=row.text,original_scope=row.original_scope,has_contrast=bool(row.has_contrast),
                group=name,feature_mask=mask,available=available))
            if available:
                active[name] = ((mask>>np.arange(4))&1).astype(bool)
        if not active:
            continue
        q = data['reflection_probabilities'][t,:,int(row.choice)-1][None,:].astype(float)
        result = discourse_effects.group_effect_profiles(q,masks,active,np.ones(1))
        for name, value in result.items():
            identity = dict(subject=107,state_trial=t,iSession=int(row.iSession),iTrial=int(row.iTrial),
                phase=row.phase,choice=int(row.choice),text=row.text,original_scope=row.original_scope,
                group=name,feature_mask=definitions[name])
            groups.append(dict(identity,features=len(value['feature_indices']),**value['means']))
            for column,index in enumerate(value['feature_indices']):
                features.append(dict(identity,feature=FEATURES[index],
                    **{measure:float(value['feature_means'][measure][column]) for measure in MEASURES}))
    gt, ft, cv = pd.DataFrame(groups), pd.DataFrame(features), pd.DataFrame(coverage)
    identity = ['subject','state_trial','iSession','iTrial','phase','choice','text','original_scope','group','feature_mask']
    paired_groups, paired_features = [], []
    coverage_keys = ['subject','state_trial','iSession','iTrial','phase','text','original_scope',
                     'has_contrast','group','feature_mask','available']
    seen_arms = set()
    for source in h_profiles:
        h = completed(source)
        if (h['subject']!=107 or h['trials']!=192 or h['training_trials']!=128 or len(h['seeds'])!=1
                or h['oral_used_for_fit'] or h['original_coding_changed'] or h['arm'] in seen_arms):
            raise ValueError('Require distinct original one-filter107 H discourse results')
        seen_arms.add(h['arm'])
        if Path(h['annotation_source']).resolve()!=annotations.resolve():
            raise ValueError('Model families must use exactly the same frozen annotations')
        hc = pd.read_csv(source/'coverage.csv',keep_default_na=False)
        pd.testing.assert_frame_equal(cv[coverage_keys],hc[coverage_keys])
        for file,table,keys,measures,destination in [
            ('group_trials.csv',gt,identity,GROUP_MEASURES,paired_groups),
            ('feature_trials.csv',ft,identity+['feature'],MEASURES,paired_features)]:
            old = pd.read_csv(source/file,keep_default_na=False)
            for weighting, part in old.groupby('weighting'):
                joined = table[keys+list(measures)].merge(part[keys+list(measures)],on=keys,how='outer',
                    validate='one_to_one',indicator=True,suffixes=('_rnn','_h'))
                if len(joined)!=len(table) or not joined['_merge'].eq('both').all():
                    raise ValueError('Model families differ in report or feature coverage')
                joined=joined.drop(columns='_merge');joined['h_arm']=h['arm'];joined['h_weighting']=weighting
                destination.append(joined)
    output.mkdir(parents=True)
    pg, pf = pd.concat(paired_groups,ignore_index=True), pd.concat(paired_features,ignore_index=True)
    for name,table in [('coverage',cv),('group_trials',gt),('feature_trials',ft),('paired_group_trials',pg),('paired_feature_trials',pf)]:
        table.to_csv(output/(name+'.csv'),index=False)
    for name,table,keys,measures in [
        ('group_summary',pg,['h_arm','h_weighting','phase','group','feature_mask'],GROUP_MEASURES),
        ('feature_summary',pf,['h_arm','h_weighting','phase','group','feature'],MEASURES)]:
        aggregation = dict(reports=('state_trial','size'))
        aggregation.update({key+'_'+model:(key+'_'+model,'mean') for key in measures for model in ('rnn','h')})
        table.groupby(keys).agg(**aggregation).reset_index().to_csv(output/(name+'.csv'),index=False)
    verify(inputs)
    write(output/'completion.json',dict(status='complete',subject=107,trials=192,training_trials=128,
        inputs_sha256=inputs,artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')},
        groups=len(gt),feature_rows=len(ft),coverage_rows=len(cv),h_arms=sorted(seen_arms),
        parameters_changed=False,original_coding_changed=False,oral_used_for_fit=False,
        context_scope='Common all-feature reflections; frozen text groups identical across models',
        role_assignment='No supporting/opposing role inferred from before/after text position',
        scope='Development107; one preselected H filter per arm and one behavior-selected RNN; potential feature effects'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('rnn','annotations','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--h-profile',required=True,action='append',type=Path)
    args=parser.parse_args()
    run(args.rnn,args.annotations,args.h_profile,args.output)
