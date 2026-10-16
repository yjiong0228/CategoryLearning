"""Research adapters for precision-learning speed, range and coordinates.

All arms use the core's exact feedback derivative and cached pre-feedback
pairing weights. Configuration validation first checks the archived baseline;
the explicit experimental protocol then overrides the bound/update in process.
Default model files and their numerical references remain unchanged.
"""
from contextlib import contextmanager

import numpy as np
from .base_module import ModuleRole


def precision_step(beta, gradient, rate, *, mode, lower, upper):
    """Return a bounded additive or log-coordinate feedback-gradient step."""
    beta, gradient = np.asarray(beta, float), np.asarray(gradient, float)
    if mode == 'log':
        return np.exp(np.clip(np.log(beta) + rate * beta * gradient,
                              np.log(lower), np.log(upper)))
    if mode not in ('base', 'fast'):
        raise ValueError('Unknown precision coordinate')
    return np.clip(beta + rate * gradient * (10 if mode == 'fast' else 1), lower, upper)


@contextmanager
def precision_intervention(cls, *, mode='base', upper=25., onset=0):
    """Install a declared experimental update, retaining the core lifecycle."""
    if mode == 'base' and upper == 25.:
        yield
        return
    if mode not in ('base', 'fast', 'log') or upper not in (25., 100.) or onset < 0:
        raise ValueError('Unsupported precision experiment')
    initialize, update = cls.__init__, cls._update_beta_feedback_gradient

    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if onset == 0:
            self.beta_max = upper

    def step(self, stimulus, choice, feedback, indices):
        transition = self.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        if transition.trial_index - 1 < onset:
            return update(self, stimulus, choice, feedback, indices)
        self.beta_max = upper
        likelihood = self.engine.observation_likelihood
        derivative = likelihood.feedback_precision_gradient
        had_local = 'feedback_precision_gradient' in likelihood.__dict__
        def transformed(*args, **kwargs):
            gradient = derivative(*args, **kwargs)
            current = np.asarray(args[2], float)
            if self.learning_rate == 0:
                return gradient
            updated = precision_step(current, gradient, self.learning_rate,
                                     mode=mode, lower=self.beta_min, upper=self.beta_max)
            return (updated - current) / self.learning_rate
        likelihood.feedback_precision_gradient = transformed
        try:
            update(self, stimulus, choice, feedback, indices)
        finally:
            if had_local:
                likelihood.feedback_precision_gradient = derivative
            else:
                del likelihood.feedback_precision_gradient

    try:
        cls.__init__, cls._update_beta_feedback_gradient = init, step
        yield
    finally:
        cls.__init__, cls._update_beta_feedback_gradient = initialize, update
