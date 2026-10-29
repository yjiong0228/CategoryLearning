"""Validate the declared Exp5 discrete Model 1024 baseline before construction."""
from collections.abc import Mapping


def validate_exp5_config(config: Mapping) -> None:
    """Keep task-specific emissions and the shared cognitive mechanism aligned."""
    prefix = "src.Bayesian_state."
    if (config["provenance"].get("specification_version") != "model1024_exp5_parity_v1"
            or config.get("partition") != {
                "class": prefix + "hypothesis_space.observation_model.exp5_partition.Exp5ParityPartition",
                "kwargs": dict(n_dims=5, n_cats=2, include_intercept=False)}):
        raise ValueError("Exp5 requires the declared 62-rule discrete partition/version")
    likelihood = config.get("likelihood", {})
    if (likelihood.get("distance_mode") != "rule" or likelihood.get("beta_source") != "action"
            or likelihood.get("feedback_likelihood_mode") != "category_feedback"
            or likelihood.get("feedback_lapse", 0) != 0):
        raise ValueError("Exp5 requires rule/action binary feedback with zero lapse")
    modules = config.get("modules", {})
    classes = dict(perception_mod="discrete_perception.ExactBinaryPerceptionModule",
                   beta_mod="beta.BetaModule", memory_mod="memory.DualMemoryModule",
                   hypo_transitions_mod="hypothesis_transition.mechanistic_rule_search.MechanisticRuleSearchHypothesisTransitionModule")
    if set(modules) != set(classes) or any(modules[k].get("class") != prefix + "model.modules." + v
                                         for k, v in classes.items()):
        raise ValueError("Exp5 baseline requires shared beta, search and memory with exact perception")
    if config.get("agenda") != ["perception_mod", "hypo_transitions_mod", "memory_mod", "beta_mod"]:
        raise ValueError("Exp5 agenda must update memory before precision")
    beta = modules["beta_mod"].get("kwargs", {})
    expected = dict(beta_update_mode="feedback_gradient", update_scope="active_hypotheses",
                    beta_min=.1, beta_max=25., probabilistic_feedback_lapse=0.,
                    use_prior_scaling=False, prior_beta_scale=0.)
    if any(beta.get(k) != v for k, v in expected.items()) or "learning_rate" not in beta:
        raise ValueError("Exp5 requires shared active-rule feedback-gradient precision")
    if modules["memory_mod"].get("kwargs") != dict(forgetting_mode="exact_power", gamma=1., w0=0., feedback_gain=1.):
        raise ValueError("Exp5 belief retention uses the 1024 transition coordinate")
    transition = modules["hypo_transitions_mod"].get("kwargs", {})
    if (transition.get("mechanism_version") != "0923-full-v3"
            or transition.get("relation_samples") != 32 or transition.get("relation_seed") != 0):
        raise ValueError("Exp5 requires shared v3 transitions and exact 32-pattern relations")
    if (config.get("choice_readout") != dict(kwargs=dict(method="expectation", power=1., strategy_confidence_gain=0.))
            or config.get("output_noise") != dict(kwargs=dict(enabled=False))
            or config.get("oral_readout") is not None):
        raise ValueError("Exp5 baseline uses plain behavior readout with disabled output noise and oral readout")
