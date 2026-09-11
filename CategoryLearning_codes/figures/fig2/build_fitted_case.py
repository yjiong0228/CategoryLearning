"""Export a fitted condition-1 or condition-2 case without rerunning inference.

Use the same saved PF repeats for behavior and pre-choice belief. Oral encoding
uses the shared evaluation implementation. Outputs are new directories only.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import pickle
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def build(model_dir: Path, output: Path, subject: int = 129) -> None:
    from src.Bayesian_state.evaluation.oral.scoring import OralAlignmentScoringMixin
    payload_path = model_dir / f'simulation/subjects/subject_{subject}.json'
    payload = json.loads(payload_path.read_text())
    condition = int(payload['condition'])
    assert payload['subject_id'] == subject and condition in (1, 2)
    n_categories = 2 if condition == 1 else 4
    scorer = OralAlignmentScoringMixin()
    partition, _ = scorer._partition_for_model_result(payload, n_categories)
    n_rules = partition.length
    target_hypothesis = 0 if condition == 1 else 42
    stream = (payload_path.parent / payload['raw_runs_ref']['path']).resolve()
    runs = []
    with gzip.open(stream, 'rb') as handle:
        while True:
            try:
                runs.append(pickle.load(handle))
            except EOFError:
                break
    assert len(runs) == payload['raw_runs_ref']['count'] == 16
    raw_path = ROOT / 'data/exp123/processed/Task2_processed.csv'
    raw = pd.read_csv(raw_path).query('iSub == @subject').copy()
    n = len(raw)
    assert raw.condition.eq(condition).all() and not raw.duplicated(['iSession','iBlock','iTrial']).any()
    predictions, beliefs = [], []
    for run in runs:
        assert run['subject_id'] == subject and run['selection_prediction_mode'] == 'prior_t'
        m = run['metrics_by_mode']['prior_t']
        np.testing.assert_array_equal(np.asarray(m['observed_choice_index']), raw.choice.to_numpy()-1)
        predictions.append(np.asarray(m['pred_category_probs'], dtype=float))
        beliefs.append(np.asarray(run['state_log']['marginal_prior'], dtype=float))
    probs = np.mean(predictions, axis=0)
    prior = np.mean(beliefs, axis=0)
    assert probs.shape == (n,n_categories) and prior.shape == (n,n_rules)
    oral = scorer.compute_oral_mass_probabilities(
        raw, subjects=[subject], oral_center_sigma=.05,
        partitions_by_subject={subject: partition})[subject]['oral_mass']
    oral = np.asarray(oral, dtype=float)
    for matrix in [probs, prior]:
        assert np.isfinite(matrix).all() and (matrix >= 0).all()
        np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-8)
    valid = np.isfinite(oral).all(axis=1)
    assert oral.shape == prior.shape and (oral[valid] >= 0).all()
    np.testing.assert_allclose(oral[valid].sum(axis=1), 1, atol=1e-8)
    correct = raw.choice.eq(raw.category).to_numpy(dtype=float)
    np.testing.assert_array_equal(correct,raw.feedback.eq(1).to_numpy())
    pcorrect = probs[np.arange(n),raw.category.to_numpy(dtype=int)-1]
    d = pd.DataFrame({'trial':np.arange(1,n+1),'observed_accuracy':correct,
                      'model_correct_probability':pcorrect,
                      'observed_choice':raw.choice.to_numpy(),
                      'observed_feedback':raw.feedback.to_numpy(),
                      'report_available':raw.text.fillna('').str.strip().ne('').to_numpy()})
    for col in ['observed_accuracy','model_correct_probability']:
        d[col+'_rolling32'] = d[col].rolling(32,min_periods=32).mean()
    overlap = np.minimum(prior,oral).sum(axis=1)
    np.testing.assert_allclose(overlap[valid],1-.5*np.abs(prior[valid]-oral[valid]).sum(axis=1),atol=1e-8)
    output.mkdir(parents=True,exist_ok=False)
    d.to_csv(output/'trial_source.csv',index=False)
    pd.DataFrame({'trial':np.repeat(d.trial,n_rules),'hypothesis':np.tile(np.arange(n_rules),n),
                  'online_prior':prior.ravel()}).to_csv(output/'belief_source.csv',index=False)
    pd.DataFrame(oral,columns=[f'H{i}' for i in range(n_rules)]).to_csv(output/'oral_distribution.csv',index=False)
    pd.DataFrame({'trial':d.trial,'overlap':overlap}).to_csv(output/'distribution_overlap.csv',index=False)
    selected = json.loads((model_dir/f'optimization/subject_{subject}/best_hyperparams.json').read_text())
    manifest = {'subject':subject,'condition':condition,'n_trials':n,'n_subjects':1,
        'task':1 if condition == 1 else 3, 'target_hypothesis':target_hypothesis,
        'case_label':'Individually fitted PMH','oral_center_sigma':.05,'rolling_window':32,
        'oral_state_mode':'latest_by_category','oral_alignment_space':'full',
        'overlap_definition':'sum(min(model, oral)) = 1 - total variation; not JS similarity',
        'prediction_timing':'pre-current-choice; conditioned on past observed history',
        'fit_scope':'Full-sequence individual fit; no held-out evaluation',
        'pf_repeats':len(runs),'filter_seeds':[r['trajectory_seed'] for r in runs],
        'persistent_execution':bool(payload['best_params']['engine.modules.hypo_transitions_mod.kwargs.persistent_execution.enabled']),
        'n_rules':n_rules,
        'final_rescore_choice_nll':selected['selection']['value'],
        'mean_overlap':float(np.nanmean(overlap)),
        'model_provenance':payload['model_provenance'],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [payload_path.resolve(),stream,raw_path,Path(__file__).resolve()]}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(__file__,output/Path(__file__).name)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subject',type=int,default=129)
    args = parser.parse_args()
    build(args.model_dir.resolve(),args.output,args.subject)
