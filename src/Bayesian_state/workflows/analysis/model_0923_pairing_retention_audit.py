"""Diagnose which graded-feedback channels survive zero pairing retention.

Use --score-results COMPLETED_SUBJECT_DIRECTORY --output NEW_DIRECTORY.
The reference must use condition-3 memory with gamma=0. This deterministic
module-level calculation uses constructed distances, three precision vectors,
four choices and uniform/informed pairing states. It does not fit participants
or change any mechanism. Shared memory, likelihood and H methods are executed;
algebraic feedback identities and finite differences provide independent checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.Bayesian_state.hypothesis_space.observation_model.likelihood import ObservationLikelihood
from src.Bayesian_state.hypothesis_space.observation_model.pairing import pairing_feedback_kernel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)

ROOT = Path(__file__).resolve().parents[4]
DISTANCES = np.array([[0., .1, .4, np.sqrt(.17)], [.8, 0., .3, .9], [.2, .7, .05, .4]])
PRECISIONS = (np.array([.1, .5, 1.]), np.array([1., 5., 15.]), np.array([25., 2., 10.]))
RULE_PRIOR = np.array([.5, .3, .2])
INFORMED_PAIRING = np.array([[.8, .15, .05], [.1, .7, .2], [.2, .15, .65]])
STIMULUS = np.array([.4, .1, .3, .7])


def probabilities(beta: np.ndarray) -> np.ndarray:
    values = -DISTANCES * np.asarray(beta)[:, None]
    values = np.exp(values - values.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


class FixedDistances:
    """Supply a controlled geometry to the production likelihood gradient."""
    n_cats, n_dims = 4, 4
    VALID_DISTANCE_MODES = ('boundary',)

    def __init__(self) -> None:
        self.boundary_geometry = self

    def category_distances(self, hypo, stimuli):
        return np.tile(DISTANCES[hypo], (len(stimuli), 1))

    def calc_likelihood(self, **kwargs):
        raise AssertionError('Controlled gradient uses absolute category probabilities')

    def get_category_probabilities(self, hypo, data, beta, distance_mode):
        if distance_mode != 'boundary':
            raise ValueError('Controlled geometry requires boundary distances')
        return probabilities(np.full(3, beta))[hypo, :, None]


def feedback_state(gamma: float, informed: bool, beta: np.ndarray, choice: int, feedback: float) -> dict:
    """Run shared feedback methods with a fixed pre-feedback state."""
    pairing = INFORMED_PAIRING if informed else np.full((3, 3), 1/3)
    engine = SimpleNamespace(prior=RULE_PRIOR.copy(), set_size=3,
        hypotheses_mask=np.ones(3), partition=FixedDistances())
    memory = HierarchicalPairingMemoryModule(engine, gamma=gamma, forgetting_mode='factorized')
    memory.joint = RULE_PRIOR[:, None] * pairing
    engine.get_module = lambda role: memory if role == ModuleRole.MEMORY else None
    likelihood = ObservationLikelihood(engine.partition, distance_mode='boundary', feedback_likelihood_mode='hierarchical_pairing')
    kernel = pairing_feedback_kernel(probabilities(beta), choice, feedback)
    engine.likelihood = memory.prepare_feedback(kernel)
    effective_pairing = memory.feedback_pairing_weights.copy()
    gradient = likelihood.feedback_precision_gradient((STIMULUS, choice, feedback), [0, 1, 2], beta,
                                                       pairing_weights=effective_pairing)
    h = SimpleNamespace(engine=engine, active=np.arange(3), previous_feedback=feedback,
        outcome_pending=True, intervention_kind='E', intervention={'search':'accumulate'},
        retention_decay=.95, retention_evidence=np.array([.2, -.1, .3]), retention_exposure=np.ones(3),
        failure_decay=.6, failure_pressure=.2, accumulated_failure=.4,
        baseline_probability=.005, error_gain=2., global_search=.05, global_search_failure_gain=.25)
    h._previous_log_evidence = lambda: Transition._previous_log_evidence(h)
    increment = h._previous_log_evidence()
    Transition._prepare_hypothesis_transition(h)
    memory.process()
    return dict(evidence=engine.likelihood, effective_pairing=effective_pairing,
        rule_posterior=engine.posterior, pairing_posterior=memory.conditional_pairing(),
        h_increment=increment, h_evidence=h.retention_evidence, gradient=gradient,
        failure_pressure=h.failure_pressure, accumulated_failure=h.accumulated_failure,
        search_probability=h.current_event_probability, global_fraction=h.current_g)


def audit(score_results: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Use a new output directory')
    protocol = json.loads((score_results/'protocol.json').read_text())
    done = json.loads((score_results/'completion.json').read_text())
    if done['status'] != 'complete' or done['smoke'] or protocol['smoke']:
        raise ValueError('Require a complete full choice screen')
    jobs = [p['job'] for p in protocol['plans']]
    gamma_key = 'engine.modules.memory_mod.kwargs.gamma'
    if len(jobs)!=2 or any(j['candidate']['hyperparams'][gamma_key]!=0 for j in jobs):
        raise ValueError('Reference must use gamma=0 in both variants')
    if jobs[0]['candidate']!=jobs[1]['candidate'] or jobs[0]['runtime']!=jobs[1]['runtime']:
        raise ValueError('Reference variants must share parameters and runtime')
    inputs = {}

    def bind(path: Path) -> str:
        sha=hashlib.sha256(path.read_bytes()).hexdigest();inputs[str(path.resolve())]=sha;return sha

    for name in ('protocol.json','completion.json'):
        bind(score_results/name)
    prepared_folder=Path(jobs[0]['folder'])
    prepared=json.loads((prepared_folder/'prepared.json').read_text())
    engine_config=json.loads((prepared_folder/'engine.json').read_text())
    bind(prepared_folder/'prepared.json')
    memory_config=engine_config['modules']['memory_mod']
    if (prepared['condition']!=3 or jobs[1]['folder']!=jobs[0]['folder']
            or memory_config['class']!='src.Bayesian_state.model.modules.pairing_memory.HierarchicalPairingMemoryModule'
            or memory_config['kwargs']['forgetting_mode']!='factorized'
            or bind(prepared_folder/'engine.json')!=prepared['engine_sha256']):
        raise ValueError('Reference must use the verified condition-3 factorized memory')
    runtime=Path(jobs[0]['runtime'])
    # The calculation imports the maintained core only when it is byte-identical
    # to the source actually used by the reference behavioral worker.
    for suffix in ('model/modules/pairing_memory.py','model/modules/beta.py','model/modules/base_module.py',
                   'hypothesis_space/observation_model/likelihood.py','hypothesis_space/observation_model/pairing.py'):
        relative=Path('src/Bayesian_state')/suffix
        actual=bind(ROOT/relative)
        if actual!=bind(runtime/relative) or actual!=prepared['frozen_cognitive_source_sha256'][str(relative)]:
            raise ValueError('Feedback implementation differs from archived reference')
    hpath='src/Bayesian_state/model/modules/hypothesis_transition/mechanistic_rule_search.py'
    if bind(ROOT/hpath)!=protocol['source_sha256'][hpath]:
        raise ValueError('H implementation differs from evaluated intervention')
    rows=[];checks=0;collapse_cases=0;positive_controls=0
    for gamma in (0., .8, 1.):
        for informed in (False,True):
            for bank,beta in enumerate(PRECISIONS):
                for choice in (1,2,3,4):
                    partial=feedback_state(gamma,informed,beta,choice,.5)
                    wrong=feedback_state(gamma,informed,beta,choice,0.)
                    collapsed=gamma==0. or not informed
                    if collapsed:
                        other=probabilities(beta)[:,np.arange(4)!=choice-1].sum(axis=1)
                        np.testing.assert_allclose(partial['evidence'],other/3,rtol=1e-13,atol=1e-15)
                        np.testing.assert_allclose(wrong['evidence'],other*2/3,rtol=1e-13,atol=1e-15)
                        for field in ('rule_posterior','h_increment','h_evidence','gradient'):
                            np.testing.assert_allclose(partial[field],wrong[field],atol=2e-14,rtol=1e-12)
                        collapse_cases+=1;checks+=18
                    else:
                        for field in ('rule_posterior','h_increment','gradient'):
                            if np.max(np.abs(partial[field]-wrong[field]))<=1e-6:
                                raise AssertionError('Informed pairing control did not distinguish feedback')
                        positive_controls+=1;checks+=3
                    if np.allclose(partial['pairing_posterior'],wrong['pairing_posterior']):
                        raise AssertionError('Pairing posterior must still distinguish partial and wrong')
                    for field in ('failure_pressure','accumulated_failure','search_probability','global_fraction'):
                        if not wrong[field]>partial[field]:
                            raise AssertionError('Failure/search channel should preserve graded feedback')
                    checks+=5
                    for feedback,state in ((.5,partial),(0.,wrong)):
                        epsilon=1e-5
                        # Independent derivative of the absolute feedback probability.
                        plus=pairing_feedback_kernel(probabilities(beta+epsilon),choice,feedback)
                        minus=pairing_feedback_kernel(probabilities(beta-epsilon),choice,feedback)
                        finite=(np.log((plus*state['effective_pairing']).sum(axis=1))-
                                np.log((minus*state['effective_pairing']).sum(axis=1)))/(2*epsilon)
                        np.testing.assert_allclose(state['gradient'],finite,atol=2e-9,rtol=1e-7);checks+=3
                    rows.append(dict(gamma=gamma,informed_pairing=informed,precision_bank=bank,choice=choice,
                        collapsed_rule_channels=collapsed,
                        **{f'max_{f}_difference':float(np.max(np.abs(partial[f]-wrong[f])))
                           for f in ('rule_posterior','h_increment','gradient','pairing_posterior')},
                        partial_failure_pressure=partial['failure_pressure'],wrong_failure_pressure=wrong['failure_pressure'],
                        partial_search_probability=partial['search_probability'],wrong_search_probability=wrong['search_probability']))
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'comparisons.csv',index=False)
    config=dict(distances=DISTANCES.tolist(),precisions=[x.tolist() for x in PRECISIONS],
        rule_prior=RULE_PRIOR.tolist(),informed_pairing=INFORMED_PAIRING.tolist(),gammas=[0.,.8,1.],
        choices=[1,2,3,4],reference_candidate=jobs[0]['candidate'],scope='Deterministic module controls, not participant refits')
    (output/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    for name,sha in inputs.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Reference input changed during audit')
    shutil.copy2(__file__,output/'audit_source.py')
    result=dict(status='complete',comparisons=len(rows),numeric_checks=checks,collapsed_cases=collapse_cases,
        informed_retention_positive_controls=positive_controls,reference_gamma=0.,
        conclusion='Uniform effective pairing makes partial and wrong feedback identical in relative H evidence, rule posterior and precision gradient; pairing posterior and failure/search remain distinct.',
        limitation='Constructed fixed pre-feedback states. Does not establish the cause of any participant loss or equivalence of complete trajectories.',
        input_sha256=inputs,source_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.score_results,args.output)
    print({k:result[k] for k in ('status','comparisons','numeric_checks','collapsed_cases','informed_retention_positive_controls')})
