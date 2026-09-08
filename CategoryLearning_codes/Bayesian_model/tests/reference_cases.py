"""Small deterministic cases; the saved reference predates shared-core consolidation."""
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'CategoryLearning_codes/Bayesian_model'


def run_cases(namespace):
    pf = importlib.import_module(namespace + '.inference.backends.particle_filter')
    autonomous = importlib.import_module(namespace + '.simulation.autonomous')
    data = pd.read_csv(ROOT / 'data/exp123/processed/Task2_processed.csv')
    sub = data[data.iSub.eq(101)].sort_values(['iSession', 'iTrial'])
    result = {}
    for execution in (False, True):
        for transport in ('similarity_transport', 'mass_preserving_similarity_transport'):
            cfg = yaml.safe_load((PACKAGE / 'configs/model_0826.yaml').read_text())
            t = cfg['modules']['hypo_transitions_mod']['kwargs']
            t['persistent_execution']['enabled'] = execution
            t['prior_assignment']['method'] = transport
            t['nested_feedback_accumulator_controller']['accumulator_logit_gain'] = .4
            t['nested_feedback_accumulator_controller']['global_search_failure_gain'] = .3
            x = sub.iloc[:16]
            r = pf.run_state_model_particle_filter(engine_config=cfg, subject_id=101,
                stimulus=x[[f'feature{i}' for i in range(1, 5)]].to_numpy(),
                choices=x.choice.to_numpy(), feedback=x.feedback.to_numpy(),
                particle_count=4, choice_readout_power=1., filter_seed=8326)
            prefix = f'pf/{execution}/{transport}/'
            for attr in ('observation_probabilities', 'state_probabilities', 'latent_summaries'):
                for key, value in getattr(r, attr).items():
                    if value is not None:
                        result[prefix + attr + '/' + key] = np.asarray(value)
            result[prefix + 'resampled'] = np.asarray(r.resampled)
        cfg = yaml.safe_load((PACKAGE / 'configs/model_0826.yaml').read_text())
        cfg['modules']['hypo_transitions_mod']['kwargs']['persistent_execution']['enabled'] = execution
        x = sub.iloc[:24]
        r = autonomous.run_autonomous_category_learning(engine_config=cfg, subject_id=101,
            condition=1, stimulus=x[[f'feature{i}' for i in range(1, 5)]].to_numpy(),
            categories=x.category.to_numpy(), trajectory_seed=8261).trajectory
        for key in ('choices', 'feedback', 'perceived_stimulus', 'prior', 'posterior', 'beta',
                    'cognitive_probabilities', 'observed_probabilities'):
            result[f'autonomous/{execution}/{key}'] = np.asarray(getattr(r, key))
    return result
