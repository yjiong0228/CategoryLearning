"""Task selection for the existing discrete and continuous implementations."""
from __future__ import annotations

from typing import Literal

from .binary import FeatureMixtureStrict
from .continuous import ContinuousFeatureMixture
from .feedback import category_count


def create_model(task: Literal["binary", "exp123"], *, n_dimensions: int = 4,
                 condition: int | None = None, max_order: int | None = None,
                 beta: float = 5.0, eta: float = 0.05, seed: int | None = None,
                 update_rule: str | None = None, memory_window: int = 1,
                 quadrature_order: int | None = None
                 ) -> FeatureMixtureStrict | ContinuousFeatureMixture:
    """Build a task-specific model while retaining each model's learning interface.

    Binary inputs and feedback labels use +/-1. Exp123 stimuli lie in [0, 1],
    and learning uses choice/feedback with condition 1, 2 or 3. Both models expose
    predict_proba; columns are [+1, -1] for binary and [1, ..., K] for exp123.
    Defaults preserve the existing si_ratio / exact_gradient conventions.
    """
    common = dict(max_order=max_order, beta=beta, eta=eta, seed=seed)
    if task == "binary":
        if condition is not None or quadrature_order is not None:
            raise ValueError("condition and quadrature_order apply to task=exp123")
        return FeatureMixtureStrict(n_bits=n_dimensions, memory_window=memory_window,
                                    update_rule="si_ratio" if update_rule is None else update_rule,
                                    **common)
    if task == "exp123":
        if n_dimensions != 4 or memory_window != 1:
            raise ValueError("exp123 uses four dimensions and memory_window=1")
        return ContinuousFeatureMixture(n_dimensions=4, n_categories=category_count(condition),
            quadrature_order=8 if quadrature_order is None else quadrature_order,
            update_rule="exact_gradient" if update_rule is None else update_rule, **common)
    raise ValueError("task must be binary or exp123")
