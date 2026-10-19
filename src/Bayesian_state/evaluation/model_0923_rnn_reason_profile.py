"""Evaluate frozen RNN reflection probes against the common oral feature sets.

Use --source COMPLETED_RNN --midpoint AUDIT --relative AUDIT --output NEW.
All original reports remain in coverage.csv. Complete coded reports receive the
same report-restricted reflection effects used for the H-model diagnostic.
Physical feature names follow each participant's feature columns. The RNN has
one deterministic state at its selected weights; no particle uncertainty or
oral-conditioned parameter weights are introduced. Results describe potential
feature effects and numerical statement compatibility, not report likelihood.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..workflows.analysis.model_0923_target_state_recovery import digest, read, verify, write
from ..workflows.analysis.model_0923_decision_basis_alignment import align_identities
from ..workflows.analysis.model_0923_joint_reason_profile import report_content
from .oral.joint_relevance import context_effects


def summarize(source: Path, midpoint: Path, relative: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new RNN oral diagnostic directory')
    inputs = {}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest(path)
        return path

    from ..workflows.analysis import model_0923_decision_basis_alignment, model_0923_joint_reason_profile
    from .oral import joint_relevance
    for path in (Path(__file__), Path(model_0923_decision_basis_alignment.__file__),
                 Path(model_0923_joint_reason_profile.__file__), Path(joint_relevance.__file__)):
        bind(path)
    done, protocol, frozen = [read(bind(source/name)) for name in ('completion.json','protocol.json','nominees.json')]
    if (done['status'] != 'complete' or done['smoke'] or done['oral_used'] or done['test_used_for_selection']
            or done['nomination_sha256'] != digest(source/'nominees.json')
            or done['protocol_sha256'] != digest(source/'protocol.json')
            or frozen['status'] != 'frozen_before_test'):
        raise ValueError('Require completed, frozen, choice-only RNN estimates')
    verify(done['inputs_sha256']); inputs.update(done['inputs_sha256'])
    rows, features, coverage = [], [], []
    for reading, oral in [('midpoint',midpoint),('relative',relative)]:
        manifest = read(bind(oral/'manifest.json'))
        if manifest['status'] != 'complete' or digest(bind(oral/'trials.csv')) != manifest['artifacts']['trials.csv']:
            raise ValueError('Common oral interpretation changed')
        raw = pd.read_csv(oral/'trials.csv', keep_default_na=False)
        for nominee in frozen['nominees']:
            subject = nominee['subject']
            job = read(bind(source/'jobs'/f'evaluate_S{subject}.json'))
            record = read(bind(Path(job['target']).with_suffix('.json')))
            file = bind(Path(job['target']).with_suffix('.npz'))
            if (record['predictions_sha256'] != digest(file) or record['seed'] != nominee['seed']
                    or not record['exact_original_predictions'] or not record['exact_training_replay']
                    or not record['exact_original_rho']):
                raise ValueError('Original RNN replay changed')
            with np.load(file) as z:
                data = {k:z[k] for k in z.files}
            frame, cut = align_identities(raw, job, len(data['choices']))
            meta = read(bind(Path(job['folder'])/'prepared.json'))
            bind(meta['paths']['learning_data'])
            np.testing.assert_array_equal(frame.choice, data['choices'])
            for key in ('stimulus', 'choices', 'feedback'):
                parts = []
                for name in ('training.npz', 'sealed_test.npz'):
                    with np.load(Path(job['folder'])/name) as z:
                        parts.append(z[key])
                np.testing.assert_array_equal(data[key], np.concatenate(parts))
            if cut != record['training_trials'] or len(frame) != record['training_trials']+record['test_trials']:
                raise ValueError('Training boundary differs from H-study inputs')
            p = data['reflection_probabilities']
            expected_masks = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
            np.testing.assert_array_equal(data['reflection_masks'], expected_masks)
            if p.shape != (len(frame), 16, job['responses']) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError('Invalid RNN reflection probabilities')
            np.testing.assert_allclose(p.sum(axis=-1), 1., atol=job['responses']*np.finfo(p.dtype).eps, rtol=0)
            np.testing.assert_allclose(p[:,0], data['probabilities'],
                                       atol=job['hidden_width']*np.finfo(p.dtype).eps, rtol=0)
            for trial, row in frame.iterrows():
                names = tuple(row[f'feature{i}_name'] for i in range(1,5))
                if set(names) != {'neck','tail','head','leg'}:
                    raise ValueError('Unknown physical feature order')
                base = dict(subject=subject,reading=reading,state_trial=int(trial),iSession=int(row.iSession),
                    iTrial=int(row.iTrial),choice=int(row.choice),phase='training' if trial<cut else 'later',
                    text=row.text,status=row.status,scope=row.scope)
                complete = row.status == 'coded' and row.scope == 'complete'
                mentioned, truth = report_content(json.loads(row.tokens), data['stimulus'][trial:trial+1]) if complete else (np.zeros(4,bool),None)
                usable = bool(complete and mentioned.any())
                coverage.append(dict(base,profile_available=usable,reported_features=int(mentioned.sum()),
                    numeric_report_truth_available=truth is not None,feature_order=json.dumps(names)))
                if not usable:
                    continue
                q = p[trial,:,int(row.choice)-1][None,:].astype(float)
                effects = context_effects(q,data['reflection_masks'],mentioned)
                report_mask = int(mentioned.astype(int) @ (1<<np.arange(4)))
                minimum = float(effects['minimum_positive_effect'][0])
                rows.append(dict(base,reported_features=int(mentioned.sum()),reported_mask=report_mask,
                    minimum_positive_effect_mean=minimum,
                    minimum_actual_positive_effect_mean=float(effects['minimum_actual_positive_effect'][0]),
                    minimum_absolute_effect_mean=float(effects['minimum_absolute_effect'][0]),
                    mean_positive_effect_mean=float(effects['mean_positive_effect'][0]),
                    numeric_report_truth_mass=float(truth[0]) if truth is not None else np.nan,
                    truth_and_minimum_positive_effect_mean=minimum*truth[0] if truth is not None else np.nan,
                    report_reflection_effect=float(q[0,0]-q[0,report_mask])))
                for index, dimension in enumerate(np.flatnonzero(mentioned)):
                    features.append(dict(base,feature_index=int(dimension),feature=names[dimension],
                        **{key:float(effects[key][0,index]) for key in
                           ('actual_effect','positive_context_effect','negative_context_effect','absolute_context_effect')}))
    output.mkdir(parents=True)
    reports, feature_table = pd.DataFrame(rows), pd.DataFrame(features)
    for name, table in [('report_trials',reports),('reported_feature_effects',feature_table),('coverage',pd.DataFrame(coverage))]:
        table.to_csv(output/(name+'.csv'),index=False)
    keys = ['subject','reading','phase']
    metrics = [k for k in reports if k.endswith('_mean') or k in ('numeric_report_truth_mass','report_reflection_effect')]
    summary = reports.groupby(keys)[metrics].mean().reset_index()
    counts = reports.groupby(keys).agg(reports=('state_trial','size'),numeric_reports=('numeric_report_truth_mass','count')).reset_index()
    summary.merge(counts,on=keys,validate='one_to_one').to_csv(output/'summary.csv',index=False)
    feature_table.groupby(keys+['feature'])[['actual_effect','positive_context_effect','negative_context_effect','absolute_context_effect']].mean().reset_index().to_csv(output/'feature_summary.csv',index=False)
    verify(inputs)
    write(output/'completion.json',dict(status='complete',subjects=sorted(reports.subject.unique().tolist()),
        inputs_sha256=inputs,report_rows=len(reports),coverage_rows=len(coverage),
        artifacts_sha256={p.name:digest(p) for p in output.iterdir()},oral_used_for_fitting=False,
        scope='Development feature-effect diagnostic at one behavior-selected RNN point per subject',
        timing='Current stimulus; previous choices and feedback; network weights frozen before later-segment evaluation',
        uncertainty='Deterministic RNN states; no parameter posterior or particle intervals',
        interpretation='Potential signed feature effects and statement truth retained separately; no report-probability claim'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','midpoint','relative','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    summarize(args.source,args.midpoint,args.relative,args.output)
