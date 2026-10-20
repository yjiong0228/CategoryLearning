"""Experimental finite-H association learning with frozen-state POOL recall.

This first integration uses condition 2, global proposals, POOL admission,
SEPARATE belief and RETAIN precision. Only active rules learn associations.
Recall evaluates completed episodes using each rule's current associations
and its active/last-active precision; unseen rules use beta_init and uniform
associations. Recall performs no training. This is an explicit minimal-entry
comparison, whose ability to admit untrained rules must be evaluated.
The explicit rehearse_candidates switch initializes a new candidate from
discounted completed feedback before this same frozen-state POOL evaluation.

Install in a single worker process for the entire lifetime of its models.
The adapter reuses maintained H selection, belief, geometry and learning
kernels. Production defaults and historical experiment sources stay intact.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256

import numpy as np

from ...active_rule_associations import ActiveRuleAssociations
from ...state_model import StateModel
from ..base_module import ModuleRole
from ..readout import choice as choice_module
from ....hypothesis_space.observation_model.likelihood import ObservationLikelihood
from ....inference.backends import particle_filter as filter_module
from . import episodic_admission as admission_module
from . import separate_belief_evidence as belief_module
from .reentry_precision import reentry_precision
from .region_candidate_rehearsal import rehearse_new_candidates


class ActiveAssociationLikelihood(ObservationLikelihood):
    """Read the engine-owned association state at the supplied predictive beta."""

    def __init__(self, base: ObservationLikelihood, transition) -> None:
        if (base.distance_mode != 'boundary' or base.beta_source != 'action'
                or base.feedback_likelihood_mode != 'category_feedback' or base.feedback_lapse != 0.
                or base.partition.n_cats != 4):
            raise ValueError('Finite region learning requires boundary/action binary four-response feedback')
        super().__init__(base.partition, distance_mode=base.distance_mode,
                         default_beta=base.default_beta, beta_source=base.beta_source,
                         feedback_likelihood_mode=base.feedback_likelihood_mode,
                         feedback_lapse=base.feedback_lapse)
        self.transition = transition

    @property
    def associations(self) -> ActiveRuleAssociations:
        return self.transition.region_associations

    def evaluate(self, observation, hypotheses, beta, *, precision=False):
        if observation is None or len(observation) != 3:
            raise ValueError('Require a completed stimulus, choice and feedback')
        stimulus, response, feedback = observation
        values = np.asarray(stimulus, float)
        if (values.shape != (4,) or not np.isfinite(values).all()
                or np.any(values < 0.) or np.any(values > 1.)):
            raise ValueError('Require a stimulus in [0,1]^4')
        indices = self.associations._rules(hypotheses)
        betas = np.broadcast_to(np.asarray(beta, float), (len(indices),))
        if not np.isfinite(betas).all() or np.any(betas <= 0):
            raise ValueError('Require positive finite rule precision')
        data = ([values], [response], [feedback])
        q = np.asarray([self.partition.get_category_probabilities(int(h), data, float(b),
                        distance_mode=self.distance_mode)[:, 0] for h, b in zip(indices, betas)])
        distances = None if not precision else np.asarray([
            self.partition.boundary_geometry.category_distances(int(h), [values])[0] for h in indices])
        return self.associations.evaluate_feedback(indices, q, response, feedback, distances=distances)

    def binary_feedback_likelihood(self, observation, hypotheses, beta=None):
        if beta is None:
            raise ValueError('Active association feedback needs predictive precision')
        return np.exp(self.evaluate(observation, hypotheses, beta).log_event)

    def feedback_precision_gradient(self, observation, hypotheses, beta, *, pairing_weights=None):
        if pairing_weights is not None:
            raise ValueError('Finite region learning currently supports binary feedback')
        return self.evaluate(observation, hypotheses, beta, precision=True).precision_gradient.copy()

    def process(self, *args, **kwargs):
        raise ValueError('Use exact binary feedback with feedback-gradient precision learning')

    def recall_precision(self, rules: np.ndarray) -> np.ndarray:
        """Match the beta that a rule would use if available on the next trial."""
        h = self.transition
        module = h.engine.get_module(ModuleRole.BETA)
        values = np.full(len(rules), module.beta_init, dtype=float)
        seen = h.reentry_precision_seen[rules]
        values[seen] = h.reentry_precision_values[rules[seen]]
        # Selection commits membership before assigning the new prior. A new
        # member's beta is still zero until the later precision initialization.
        active = np.isin(rules, h.active) & (h.engine.beta[rules] > 0.)
        values[active] = h.engine.beta[rules[active]]
        return values


def frozen_association_recall(likelihood, episodes, rules, beta):
    """Evaluate history at the current association state without updating it.

    The scalar beta argument belongs to the shared recall interface. This
    experimental profile instead uses each rule's current or stored beta.
    Scores are log feedback probabilities relative to uniform response chance.
    """
    del beta
    if not isinstance(likelihood, ActiveAssociationLikelihood) or not episodes:
        raise ValueError('Frozen association recall needs this profile and completed episodes')
    indices = likelihood.associations._rules(rules)
    betas = likelihood.recall_precision(indices)
    data = (np.asarray([e.stimulus for e in episodes]),
            np.asarray([e.choice for e in episodes]), np.asarray([e.feedback for e in episodes]))
    if np.any(~np.isin(data[2], [0., 1.])) or any(e.pairing is not None for e in episodes):
        raise ValueError('Frozen association recall requires binary feedback without pairings')
    q = np.stack([likelihood.partition.get_category_probabilities(int(h), data, float(b),
                  distance_mode=likelihood.distance_mode).T for h, b in zip(indices, betas)], axis=1)
    # Use the same response transformation as the online observation/readout.
    response = np.einsum('thg,hgc->thc', q, likelihood.associations.associations(indices))
    if (not np.isfinite(response).all() or np.any(response < 0)
            or not np.allclose(response.sum(-1), 1., atol=1e-12, rtol=1e-10)):
        raise ValueError('Invalid recalled response probabilities')
    selected = np.arange(4)[None, :] == data[1][:, None]-1
    allowed = np.where(data[2][:, None] == 1., selected, ~selected)
    evidence = np.sum(response * allowed[:, None, :], axis=-1)
    chance = np.where(data[2] == 1., .25, .75)
    scores = np.log(np.maximum(evidence, np.finfo(float).tiny))-np.log(chance[:, None])
    # Uniform association rows give exact chance for every geometry. Preserve
    # this analytic tie rather than letting geometry roundoff decide admission.
    a = likelihood.associations.associations(indices)
    uniform = np.all(a == .25, axis=(1, 2))
    scores[:, uniform] = 0.
    return scores, float(evidence.max())


@contextmanager
def finite_region_learning(cls, *, learning_rate: float, rehearse_candidates: bool = False):
    """Install finite associations, optionally learning a new candidate by recall.

    Rehearsal uses existing delta episode weights and the association learning
    rate. It precedes POOL comparison; the default retains uniform newcomers.
    """
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError('Require a finite nonnegative association learning rate')
    with (rehearse_new_candidates(cls, enabled=rehearse_candidates),
          admission_module.episodic_admission(cls, mode='POOL'), reentry_precision(cls),
          belief_module.separate_belief_evidence(cls, enabled=True)):
        methods = {name: getattr(cls, name) for name in ('__init__', 'state_dict', 'load_state_dict')}
        complete, mapping, oral = StateModel.complete_trial, choice_module.apply_hypothesis_mapping, StateModel.predict_oral
        filter_mapping = filter_module.apply_hypothesis_mapping
        recall_functions = [(module, module.recalled_log_evidence) for module in (admission_module, belief_module)]

        def initialize(self, *args, **kwargs):
            methods['__init__'](self, *args, **kwargs)
            if (self.engine.context.condition != 2 or self.global_search != 1. or self.exit_rate != 0.
                    or self.engine.get_module(ModuleRole.MAPPING) is not None
                    or hasattr(self, 'region_associations')):
                raise ValueError('Finite region profile requires condition 2, global search, exit_rate=0 and no mapping module')
            self.region_associations = ActiveRuleAssociations(self.total_hypo, 4, 4, learning_rate=learning_rate)
            self.engine.observation_likelihood = ActiveAssociationLikelihood(self.engine.observation_likelihood, self)
            self.configuration_signature = sha256(
                f'{self.configuration_signature}:finite-region-frozen-recall-v1:{float(learning_rate)}'.encode()).hexdigest()

        def snapshot(self):
            return dict(methods['state_dict'](self), region_associations=self.region_associations.state_dict())

        def restore(self, state):
            candidate = ActiveRuleAssociations(self.total_hypo, 4, 4, learning_rate=learning_rate)
            candidate.load_state_dict(state.get('region_associations', {}))
            methods['load_state_dict'](self, {k: v for k, v in state.items() if k != 'region_associations'})
            self.region_associations = candidate

        def finish_trial(self, choice, feedback, *, update_state=True, module_kwargs=None):
            evaluator = self.engine.observation_likelihood
            if not isinstance(evaluator, ActiveAssociationLikelihood):
                return complete(self, choice, feedback, update_state=update_state, module_kwargs=module_kwargs)
            beta_module = self.engine.get_module(ModuleRole.BETA)
            if beta_module is None or beta_module.beta_update_mode != 'feedback_gradient':
                raise ValueError('Finite region profile requires feedback-gradient precision')
            prepared = self._pending_trial
            if prepared is None:
                raise RuntimeError('begin_trial() must precede association feedback')
            active = evaluator.transition.active.copy()
            evaluation = None if not update_state else evaluator.evaluate(
                (prepared.perceived_stimulus, choice, feedback), active, self.engine.beta[active], precision=True)
            result = complete(self, choice, feedback, update_state=update_state, module_kwargs=module_kwargs)
            if evaluation is not None:
                evaluator.associations.commit_feedback(evaluation)
                result[2]['region_association_feedback_rules'] = active.tolist()
            return result

        def transform(engine, hypothesis, probabilities, *, use_predictive_snapshot=False):
            evaluator = engine.observation_likelihood
            if isinstance(evaluator, ActiveAssociationLikelihood):
                if engine.get_module(ModuleRole.MAPPING) is not None or use_predictive_snapshot:
                    raise ValueError('Independent associations use their current pre-feedback state')
                return evaluator.associations.predict([hypothesis], np.asarray(probabilities)[None])[0]
            return mapping(engine, hypothesis, probabilities, use_predictive_snapshot=use_predictive_snapshot)

        def predict_oral(self, choice):
            if isinstance(self.engine.observation_likelihood, ActiveAssociationLikelihood):
                raise ValueError('Use the region_basis projection for independent-association oral validation')
            return oral(self, choice)

        def recalled(likelihood, episodes, rules, beta):
            if isinstance(likelihood, ActiveAssociationLikelihood):
                return frozen_association_recall(likelihood, episodes, rules, beta)
            return recall_functions[0][1](likelihood, episodes, rules, beta)

        try:
            cls.__init__, cls.state_dict, cls.load_state_dict = initialize, snapshot, restore
            StateModel.complete_trial, StateModel.predict_oral = finish_trial, predict_oral
            choice_module.apply_hypothesis_mapping = transform
            filter_module.apply_hypothesis_mapping = transform
            for module, _ in recall_functions: module.recalled_log_evidence = recalled
            yield
        finally:
            for name, method in methods.items(): setattr(cls, name, method)
            StateModel.complete_trial, StateModel.predict_oral = complete, oral
            choice_module.apply_hypothesis_mapping = mapping
            filter_module.apply_hypothesis_mapping = filter_mapping
            for module, function in recall_functions: module.recalled_log_evidence = function
