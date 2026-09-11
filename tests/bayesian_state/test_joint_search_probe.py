"""A coupled-move audit must evaluate both component moves, not only the jump."""
from copy import deepcopy

from src.Bayesian_state.workflows.analysis.probe_model_0826_search import paired_square


def test_joint_probe_retains_both_single_moves_and_does_not_mutate_anchor():
    anchor = {"workspace": {"M": 5, "chi": False}, "gamma": .97, "beta": 5.}
    before = deepcopy(anchor)
    points = paired_square(anchor, "workspace", {"M": 4, "chi": True}, "gamma", .9)
    assert anchor == before
    assert points["single_a"] == {"workspace": {"M": 4, "chi": True}, "gamma": .97, "beta": 5.}
    assert points["single_b"] == {"workspace": {"M": 5, "chi": False}, "gamma": .9, "beta": 5.}
    assert points["joint"] == {"workspace": {"M": 4, "chi": True}, "gamma": .9, "beta": 5.}
    points["joint"]["workspace"]["M"] = 1
    assert points["single_a"]["workspace"]["M"] == 4
