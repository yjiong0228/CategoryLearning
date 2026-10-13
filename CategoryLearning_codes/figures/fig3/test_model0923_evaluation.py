"""The fixed report retains trial identities and explicit missing-check status."""
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3 import model0923_evaluation as renderer
from CategoryLearning_codes.figures.fig3.model0923_evaluation import diagnostics_figures


def _trials() -> pd.DataFrame:
    return pd.DataFrame({
        "valid_trial": [1, 2, 3, 4],
        "predictive_ess_fraction": [1., .8, .9, .6],
        "filtered_ess_fraction": [.8, .6, .5, .4],
        "resampled_fraction": [0., .25, .5, 1.],
        "predictive_search": [.1, .4, .7, .3],
        "predictive_global_search": [.2, .3, .5, .2],
        "predictive_failure": [0., .5, 1.5, 1.2],
        "predictive_search_event": [0., .25, .5, .25],
        "predictive_admission_event": [0., .25, .25, .1],
    })


@pytest.mark.parametrize("paired_check", [False, True])
def test_diagnostics_has_fixed_outputs_and_full_source_data(tmp_path: Path, paired_check: bool):
    trials = _trials()
    comparison = pd.DataFrame({
        "valid_trial": trials.valid_trial,
        "mean_abs_probability_difference": [.0, .01, .02, .03],
        "max_abs_probability_difference": [.0, .02, .04, .06],
    }) if paired_check else None
    image, data = diagnostics_figures(335, 3, trials, tmp_path, budget_comparison=comparison)

    assert image.name == "diagnostics.png" and image.read_bytes().startswith(b"\x89PNG")
    assert data.name == "diagnostics_plot_source.csv"
    source = pd.read_csv(data)
    pd.testing.assert_frame_equal(source[list(trials.columns)], trials)
    assert source.budget_comparison_status.unique().tolist() == ["completed" if paired_check else "not_performed"]
    if paired_check:
        np.testing.assert_array_equal(source.mean_abs_probability_difference, comparison.mean_abs_probability_difference)
    else:
        assert source.mean_abs_probability_difference.isna().all()
    with pytest.raises(FileExistsError, match="already exists"):
        diagnostics_figures(335, 3, trials, tmp_path, budget_comparison=comparison)


def test_budget_comparison_requires_the_same_trial_order(tmp_path: Path):
    comparison = pd.DataFrame({
        "valid_trial": [1, 3, 2, 4],
        "mean_abs_probability_difference": [0.] * 4,
        "max_abs_probability_difference": [0.] * 4,
    })
    with pytest.raises(ValueError, match="same trial order"):
        diagnostics_figures(307, 3, _trials(), tmp_path, budget_comparison=comparison)
    assert not (tmp_path / "diagnostics.png").exists()


@pytest.mark.parametrize("column,value", [
    ("filtered_ess_fraction", 1.1),
    ("predictive_search", np.nan),
    ("predictive_failure", -1.),
])
def test_invalid_diagnostic_measurements_fail_before_rendering(tmp_path: Path, column: str, value: float):
    trials = _trials()
    trials.loc[0, column] = value
    with pytest.raises(ValueError):
        diagnostics_figures(307, 3, trials, tmp_path)
    assert not (tmp_path / "diagnostics.png").exists()


def test_canonical_cli_supplies_all_mandatory_renderers(tmp_path: Path, monkeypatch):
    workflow = ModuleType("src.Bayesian_state.workflows.analysis.evaluate_model_0923")
    called = {}

    def run(config, output, smoke, **options):
        called.update(config=config, output=output, smoke=smoke, **options)
        return {"status": "dry_run"}

    workflow.run = run
    monkeypatch.setitem(sys.modules, workflow.__name__, workflow)
    monkeypatch.setattr(sys, "argv", [
        "model0923_evaluation", "--config", str(tmp_path / "evaluation.yaml"),
        "--output-dir", str(tmp_path / "results"),
        "--figure-output-dir", str(tmp_path / "figures"), "--dry-run",
    ])
    renderer.main()
    assert called["config"] == tmp_path / "evaluation.yaml"
    assert called["output"] == tmp_path / "results"
    assert called["figure_output"] == tmp_path / "figures"
    assert called["dry_run"] is True and called["smoke"] is False
    assert called["plot_subject"] is renderer.subject_figures
    assert called["plot_cohort"] is renderer.cohort_figures
    assert called["plot_diagnostics"] is renderer.diagnostics_figures
