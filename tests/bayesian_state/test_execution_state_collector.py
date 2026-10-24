"""Independent joint-state examples distinguish availability from execution."""
import numpy as np

from src.Bayesian_state.evaluation.execution_states import ExecutionStateCollector


def test_three_sources_and_choice_conditioning_preserve_joint_states():
    # First particle lacks target rule 0; second has it but executes rule 1;
    # third executes target rule 0 and can still make a classification error.
    weights = np.array([.2, .3, .5])
    active = np.array([[0, 1], [1, 1], [1, 1]])
    belief = np.array([[0., 1.], [.8, .2], [.7, .3]])
    execution = np.array([[0., 1.], [0., 1.], [1., 0.]])
    per_rule = np.tile(np.array([[.9, .1], [.2, .8]]), (3, 1, 1))
    per_rule[0, 0] = 0.
    prediction = np.einsum('ph,phc->pc', execution, per_rule)
    observer = ExecutionStateCollector(np.array([2]), np.array([1]), [0])
    observer(0, 'predictive', dict(weights=weights, active=active, belief=belief,
        choice_weights=execution, prediction=prediction, rule_choice_probabilities=per_rule))
    values = observer.arrays()
    evidence = np.array([.2*.8, .3*.8, .5*.1])
    np.testing.assert_allclose(values['conditional_decision_sources'][0], evidence / evidence.sum())
    np.testing.assert_allclose(values['predictive_error_sources'][0], evidence)
    np.testing.assert_allclose(values['predictive_target_present'], [.8])
    np.testing.assert_allclose(values['predictive_target_execution'], [.5])
    np.testing.assert_allclose(values['filtered_target_execution'], [evidence[2] / evidence.sum()])
    np.testing.assert_allclose(values['filtered_target_present'], [evidence[1:].sum() / evidence.sum()])


def test_marginal_execution_uncertainty_does_not_imply_pathwise_mixture():
    observer = ExecutionStateCollector(np.array([1]), np.array([1]), [0])
    per_rule = np.tile(np.array([[.8, .2], [.2, .8]]), (2, 1, 1))
    observer(0, 'predictive', dict(weights=np.array([.5, .5]), active=np.ones((2, 2)),
        belief=np.full((2, 2), .5), choice_weights=np.eye(2),
        prediction=np.array([[.8, .2], [.2, .8]]), rule_choice_probabilities=per_rule))
    values = observer.arrays()
    np.testing.assert_allclose(values['predictive_execution'], [[.5, .5]])
    np.testing.assert_allclose(values['conditional_decision_sources'], [[0., .2, .8]])
