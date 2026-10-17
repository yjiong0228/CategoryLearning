from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import numerical_constraints
from src.Bayesian_state.workflows.analysis.model_0923_oral_choice_audit import exact_applicable_branch


def tokens(text):
    return encode_report(text, ('neck', 'head', 'leg', 'tail')).tokens


def component(text):
    A, b = numerical_constraints(tokens(text))
    return SimpleNamespace(A=A, b=b)


def test_exact_branch_requires_complete_content_in_one_applicable_component():
    components = [component('脖子短，尾巴长'), component('脖子长，头短')]
    stimulus = np.array([.2, .2, .8, .8])
    assert exact_applicable_branch(tokens('脖子短，尾巴长'), 'complete', components, stimulus)
    for report, scope in [('脖子短', 'complete'), ('脖子短，尾巴长，腿长', 'complete'),
                          ('脖子长，头短', 'complete'), ('脖子短，尾巴长', 'parsed_fragment')]:
        assert not exact_applicable_branch(tokens(report), scope, components, stimulus)


def test_wrong_task_geometry_fails_before_an_exact_report_is_counted():
    with pytest.raises(ValueError, match='no component'):
        exact_applicable_branch(tokens('脖子长'), 'complete', [component('脖子长')], np.full(4, .2))
