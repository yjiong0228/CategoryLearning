"""Information-boundary checks for the bounded finalization precision adapter."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_finalization_precision import (
    INIT, RATE, build_cells, check_template, run, validate_settings,
)


def settings():
    return dict(
        scope="software_validation", subject=108, belief_mechanism="SEPARATE",
        reference_arm="SEARCH", cells=["SEARCH_ADD10", "SEARCH_LOG25", "POOL_LOG25"],
        parallel_budget=1, block_width=32, initial_values=[.5, 5.],
        families={
            "ADD10": dict(precision="fast", upper=25., rate_scale=10., rates=[.01, .1], anchor="reference"),
            "LOG25": dict(precision="log", upper=25., rate_scale=1., rates=[.01, .3], anchor=[.5, .3]),
        },
        screen=dict(particles=2, repeats=2),
        confirmation=dict(particles=2, repeats=2, finalists=2),
        evaluation=dict(particles=2, repeats=2, paths=4, path_chunk=1),
    )


def test_three_cells_preserve_same_h_and_memory():
    reference = dict(id="reference", hyperparams={INIT: 5., RATE: .1, "capacity": 4, "gamma": .85})
    original = deepcopy(reference)
    cells = build_cells(settings(), reference)
    assert reference == original
    for cell in cells.values():
        for point in cell["points"]:
            assert {k: v for k, v in point["hyperparams"].items() if k not in (INIT, RATE)} == {
                "capacity": 4, "gamma": .85,
            }
    assert cells["SEARCH_LOG25"]["points"] == cells["POOL_LOG25"]["points"]
    assert cells["SEARCH_LOG25"]["arm"] == "SEARCH"
    assert cells["POOL_LOG25"]["arm"] == "POOL"
    anchor = next(p for p in cells["SEARCH_ADD10"]["points"] if p["id"] == cells["SEARCH_ADD10"]["anchor"])
    assert anchor["hyperparams"] == original["hyperparams"]


@pytest.mark.parametrize("change", [
    {"cells": ["SEARCH_ADD10", "SEARCH_ADD10"]},
    {"cells": ["SELECT_LOG25"]},
    {"belief_mechanism": "unknown"},
    {"parallel_budget": 129},
    {"initial_values": list(range(1, 26))},
])
def test_rejects_unregistered_expansion(change):
    config = dict(settings(), **change)
    with pytest.raises(ValueError):
        validate_settings(config)


@pytest.mark.parametrize("mechanism,intervention,enabled", [
    ("ONLINE", "online_retention_evidence", True),
    ("SEPARATE", "separate_gamma_belief_evidence", False),
])
def test_source_identity_includes_mechanism_and_mixture(mechanism, intervention, enabled):
    config = dict(settings(), belief_mechanism=mechanism)
    point = dict(id="source", hyperparams={INIT: 5., RATE: .1})
    job = dict(phase="autonomous", candidate=point, smoke=False,
               probe_mechanism=mechanism, cohort_belief_mechanism=mechanism,
               belief_intervention=intervention, online_belief=enabled, retain_precision=True)
    check_template(job, config, point, "autonomous")
    for change in (dict(execution_policy="SELECT"), dict(online_belief=not enabled),
                   dict(candidate=dict(point, id="another")), dict(smoke=True)):
        with pytest.raises(ValueError):
            check_template(dict(job, **change), config, point, "autonomous")


def test_software_only_config_cannot_start_full_run(tmp_path):
    import yaml

    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(settings()))
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="require --smoke"):
        run(config, output)
    assert not output.exists()
