"""Passive pre-feedback rule-belief and execution summaries for recovery.

Predictive summaries condition on previous choices. Filtered summaries also
condition on the current choice; neither uses current feedback or future
choices. Correct-rule sets are specified from task geometry independently of
behavior. The three decision-source masses partition current-choice evidence.
"""
from __future__ import annotations

import numpy as np


class ExecutionStateCollector:
    """Aggregate joint particle states without changing the cognitive filter."""

    def __init__(self, choices: np.ndarray, categories: np.ndarray, target_rules: list[int]):
        self.choices = np.asarray(choices)
        self.categories = np.asarray(categories)
        self.targets = np.asarray(target_rules, dtype=int)
        if (self.choices.ndim != 1 or self.choices.shape != self.categories.shape
                or any(not np.isfinite(v).all() or np.any(v < 1) or np.any(v != np.floor(v))
                       for v in (self.choices, self.categories))
                or not len(self.targets) or len(np.unique(self.targets)) != len(self.targets)):
            raise ValueError('Require ordered choices/categories and a nonempty unique target set')
        self.choices = self.choices.astype(int)
        self.categories = self.categories.astype(int)
        self.rows = []

    def __call__(self, trial: int, phase: str, snapshot) -> None:
        if phase == 'filtered':
            return
        if phase != 'predictive' or trial != len(self.rows):
            raise ValueError('Require one pre-feedback predictive snapshot per trial')
        weights, active, belief, contribution, prediction, per_rule = [
            np.asarray(snapshot[k]) for k in ('weights', 'active', 'belief', 'choice_weights',
                                              'prediction', 'rule_choice_probabilities')]
        particles, rules = active.shape
        if (weights.shape != (particles,) or belief.shape != active.shape
                or contribution.shape != active.shape or per_rule.shape[:2] != active.shape
                or prediction.shape != (particles, per_rule.shape[2])
                or np.any(self.targets < 0) or np.any(self.targets >= rules)
                or any(not np.isfinite(v).all() for v in (weights, active, belief, contribution, prediction, per_rule))
                or not np.isin(active, [0, 1]).all()):
            raise ValueError('Invalid execution observer dimensions or probabilities')
        for value in (weights, belief, contribution, prediction, per_rule):
            if np.any(value < 0):
                raise ValueError('Negative observer probabilities')
        np.testing.assert_allclose(weights.sum(), 1., atol=1e-12)
        for value in (belief, contribution, prediction):
            np.testing.assert_allclose(value.sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(np.einsum('ph,phc->pc', contribution, per_rule), prediction, atol=1e-12)
        if np.any(contribution[active == 0]) or np.any(belief[active == 0]):
            raise ValueError('Inactive rule has belief or choice contribution')
        present = np.any(active[:, self.targets] > 0, axis=1)
        target_belief = belief[:, self.targets].sum(axis=1)
        target_execution = np.clip(contribution[:, self.targets].sum(axis=1), 0., 1.)
        chosen, correct = self.choices[trial] - 1, self.categories[trial] - 1
        if max(chosen, correct) >= prediction.shape[1]:
            raise ValueError('Choice/category outside the task response space')
        selected = prediction[:, chosen]
        denominator = float(weights @ selected)
        if denominator <= 0:
            raise ValueError('Cannot condition on impossible choice')
        conditional = weights * selected / denominator
        target_choice = (contribution[:, self.targets] * per_rule[:, self.targets, chosen]).sum(axis=1)
        decision_sources = np.column_stack((~present * selected,
            present * (selected - target_choice), target_choice))
        # Small subtraction roundoff is harmless; substantial negative mass
        # signals an incorrect decision snapshot.
        if decision_sources.min() < -1e-12:
            raise ValueError('Negative decision-source mass')
        decision_sources = np.maximum(decision_sources, 0.)
        target_error = (contribution[:, self.targets] * (1. - per_rule[:, self.targets, correct])).sum(axis=1)
        error = 1. - prediction[:, correct]
        error_sources = np.column_stack((~present * error, present * (error - target_error), target_error))
        if error_sources.min() < -1e-12:
            raise ValueError('Negative error-source mass')
        error_sources = np.maximum(error_sources, 0.)
        np.testing.assert_allclose(decision_sources.sum(axis=1), selected, atol=1e-12)
        np.testing.assert_allclose(error_sources.sum(axis=1), error, atol=1e-12)
        row = dict(prediction=weights @ prediction,
            conditional_decision_sources=weights @ decision_sources / denominator,
            predictive_error_sources=weights @ error_sources)
        for prefix, own in [('predictive', weights), ('filtered', conditional)]:
            for key, value in [('active', active), ('belief', belief), ('execution', contribution),
                               ('target_present', present), ('target_belief', target_belief),
                               ('target_high_belief', present & (target_belief >= .5)),
                               ('target_execution', target_execution),
                               ('available_other_execution', present * (1.-target_execution)),
                               ('high_belief_other_execution',
                                (present & (target_belief >= .5)) * (1.-target_execution))]:
                row[prefix + '_' + key] = own @ value
        self.rows.append(row)

    def arrays(self) -> dict[str, np.ndarray]:
        if len(self.rows) != len(self.choices) or not self.rows:
            raise ValueError('Incomplete execution-state capture')
        return dict({k: np.stack([row[k] for row in self.rows]) for k in self.rows[0]},
                    choices=self.choices, categories=self.categories, target_rules=self.targets)
