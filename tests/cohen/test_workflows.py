"""Small synthetic CSV/config checks for both command-line workflows."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.Cohen.data import prepare_exp123
from src.Cohen.exp123 import run_config
from src.Cohen.feedback import task_feedback


def sample_frame(condition=3, subject=301):
    choice = [1, 2, 3, 4] if condition != 1 else [1, 2, 1, 2]
    y = [2, 1, 1, 4] if condition != 1 else [2, 1, 1, 2]
    return pd.DataFrame({"iSub": subject, "condition": condition, "iSession": [1, 1, 2, 2],
                         "iTrial": [1, 2, 1, 2], "choice": choice, "category": y,
                         "feedback": [task_feedback(c, t, condition) for c, t in zip(choice, y)],
                         "ambiguous": [0, 1, 0, 0],
                         **{f"feature{i}": np.linspace(.1 * i, .5 + .1 * i, 4) for i in range(1, 5)}})


def test_data_orders_sessions_and_keeps_ambiguous_trials():
    frame, X, y, r, fb, condition = prepare_exp123(sample_frame().iloc[::-1])
    assert frame.iSession.tolist() == [1, 1, 2, 2]
    assert len(X) == 4
    assert frame.ambiguous.sum() == 1
    bad = sample_frame()
    bad.loc[0, "feedback"] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        prepare_exp123(bad)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_exp123(pd.concat([sample_frame(), sample_frame()]))
    with pytest.raises(ValueError, match="one nonempty"):
        prepare_exp123(pd.concat([sample_frame(), sample_frame(1, 101)]))


@pytest.mark.parametrize("mode", ["replay", "simulate", "fit"])
def test_exp123_outputs_and_no_overwrite(tmp_path, mode):
    data = tmp_path / "input.csv"
    pd.concat([sample_frame(1, 101), sample_frame(2, 201), sample_frame()]).to_csv(data, index=False)
    cfg = {"data": str(data), "output_dir": str(tmp_path / "output"), "session_policy": "carry",
           "mode": mode, "seed": 2, "max_trials": 4, "model": {"quadrature_order": 3},
           "fit": {"population_size": 2, "generations": 1, "anneal_steps": 0}}
    if mode == "fit":
        cfg["fit_first_n"] = 2
    result = run_config(cfg)
    assert result["status"] == "computationally_complete"
    assert len(result["runs"]) == 3
    for sid in (101, 201, 301):
        folder = tmp_path / "output" / f"subject_{sid}"
        p = pd.read_csv(folder / "trial_predictions.csv")
        assert len(p) == 4
        np.testing.assert_allclose(p.filter(regex="^p_category").sum(axis=1), 1.)
        assert "oral_A" not in p
        state = np.load(folder / "states.npz")
        assert len(state["alpha_before_feedback"]) == 4
        params = json.loads((folder / "parameters.json").read_text())
        assert params["initial"]["update_rule"] == "exact_gradient"
        if mode == "fit":
            assert p.scope.tolist() == ["train", "train", "conditional_test", "conditional_test"]
    with pytest.raises(FileExistsError):
        run_config(cfg)


def test_binary_cli_compatibility_and_custom_formula_filename(tmp_path):
    data = tmp_path / "binary.csv"
    pd.DataFrame({"iSub": [1] * 4, "iTrial": [1, 2, 3, 4], "stiID": [1, 2, 3, 4],
                  "category": [1, 2, 2, 1], "choice": [2, 2, 1, 1]}).to_csv(data, index=False)
    out = tmp_path / "binary"
    proc = subprocess.run([sys.executable, "-m", "src.Cohen.cohen_schneidman", "--csv", str(data),
                           "--outdir", str(out), "--n-bits", "2", "--max-order", "1",
                           "--formula-csv-name", "steps.csv"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (out / "steps.csv").exists()
    assert not (out / "model_oral_similarity.csv").exists()
    assert len(pd.read_csv(out / "trial_predictions.csv")) == 4


def test_multistart_full_replay_and_selection(tmp_path):
    from src.Cohen.continuous import ContinuousFeatureMixture
    from src.Cohen.fit_study import run_study

    data = tmp_path / "input.csv"
    sample_frame(1, 102).to_csv(data, index=False)
    cfg = {"data": str(data), "subjects": [102], "mode": "fit", "session_policy": "carry",
           "output_dir": str(tmp_path / "study"), "model": {"quadrature_order": 2},
           "fit": {"population_size": 2, "generations": 1, "anneal_steps": 0}}
    plan = run_study(cfg, [2, 3], [3, 4])
    assert plan["status"] == "computationally_complete"
    dest = tmp_path / "study" / "diagnostics"
    comparisons = pd.read_csv(dest / "seed_comparison.csv")
    best = plan["diagnostics"]
    assert best["nll"] == pytest.approx(comparisons.nll_order4.min())
    assert best["best_seed"] == int(comparisons.iloc[0].seed)
    params = json.loads((dest / "best_parameters.json").read_text())
    _, X, y, r, fb, condition = prepare_exp123(sample_frame(1, 102))
    replay = ContinuousFeatureMixture(**params["initial"]).run_session(X, r, fb, condition, y)
    predictions = pd.read_csv(dest / "best_trial_predictions.csv")
    np.testing.assert_allclose(predictions.p_correct, replay["p_correct"])
    assert best["nll"] == pytest.approx(-replay["log_likelihood"].sum())
    assert set(predictions.scope) == {"train"}
    assert (dest / "fit_diagnostics.png").exists()
    with pytest.raises(FileExistsError):
        run_study(cfg, [2, 3], [3, 4])
    with pytest.raises(ValueError, match="complete"):
        run_study({**cfg, "fit_first_n": 2}, [2, 3], [3, 4])
