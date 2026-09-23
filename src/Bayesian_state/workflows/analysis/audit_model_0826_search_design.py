"""Inspect saved fits for split versus unified feedback-search design clues.

No PF, behavioral refitting, or hypothesis test is run. Unified projections
compress a fitted controller curve, not observed human search events. Run from
the repository root; output must be a new directory. See workflows/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from scipy.special import expit, logit

from ...optimization.model_0826 import extract_model_0826_parameters


def failure_history(previous_error: np.ndarray, decay: float) -> np.ndarray:
    """Analytical exponential filter for controller diagnostics only."""
    return lfilter([1.0 - decay], [1.0, -decay], previous_error)


def project_unified(previous_error: np.ndarray, target: np.ndarray) -> tuple[dict, np.ndarray]:
    """Approximate saved logits by b + c*F(delta), with c >= 0.

    At each decay, intercept and slope have a constrained least-squares
    solution. Search a deterministic 101-point grid, then refine around its
    best point. This is a numerical curve approximation, not a model fit.
    """
    y = logit(target)

    def evaluate(decay: float) -> tuple[float, float, float, np.ndarray]:
        f = failure_history(previous_error, decay)[1:]
        centered = f - f.mean()
        denominator = float(centered @ centered)
        slope = max(0.0, float(centered @ (y - y.mean())) / denominator) if denominator else 0.0
        intercept = float(y.mean() - slope * f.mean())
        loss = float(np.mean((intercept + slope * f - y) ** 2))
        return loss, intercept, slope, f

    grid = np.linspace(0.0, 0.999, 101)
    index = int(np.argmin([evaluate(d)[0] for d in grid]))
    refined = minimize_scalar(
        lambda d: evaluate(d)[0], method="bounded",
        bounds=(grid[max(0, index - 1)], grid[min(100, index + 1)]),
        options={"xatol": 1e-10},
    )
    assert refined.success
    decay = float(min([grid[index], refined.x], key=lambda d: evaluate(d)[0]))
    loss, intercept, slope, f = evaluate(decay)
    predicted = expit(intercept + slope * f)
    difference = abs(predicted - target)
    constant = bool(np.ptp(target) < 1e-12)
    return {
        "projection_decay": np.nan if constant else decay,
        "projection_decay_identifiable": not constant,
        "projection_E0": float(expit(intercept)), "projection_gain": slope,
        "projection_logit_mse": loss,
        "event_mae_pp": float(100 * difference.mean()),
        "event_max_error_pp": float(100 * difference.max()),
        "event_first128_mae_pp": float(100 * difference[:128].mean()),
    }, f


def project_range(failure: np.ndarray, target: np.ndarray) -> dict:
    """Best linear range curve a+b*F with a,b>=0 and a+b<=1.

    Check the interior least-squares solution and all three simplex edges.
    This measures the cost of using the event-projection decay for range too.
    """
    x = np.column_stack([np.ones(len(failure)), failure])
    candidates = [np.linalg.lstsq(x, target, rcond=None)[0]]
    candidates.append(np.array([np.clip(target.mean(), 0, 1), 0.0]))
    b = np.clip(failure @ target / (failure @ failure), 0, 1) if failure @ failure else 0.0
    candidates.append(np.array([0.0, b]))
    z = failure - 1.0
    b = np.clip(z @ (target - 1.0) / (z @ z), 0, 1) if z @ z else 0.0
    candidates.append(np.array([1.0 - b, b]))
    feasible = [v for v in candidates if np.all(v >= 0) and v.sum() <= 1.0]
    a, b = min(feasible, key=lambda v: np.mean((x @ v - target) ** 2))
    difference = abs(a + b * failure - target)
    return {"range_projection_g0": float(a),
            "range_projection_cG": float(b / (1.0 - a)) if a < 1 else 0.0,
            "range_mae_pp_at_event_decay": float(100 * difference.mean()),
            "range_max_error_pp_at_event_decay": float(100 * difference.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-config", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="Existing base_analysis directory with trials.csv and subjects.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs: dict[str, str] = {}

    def record(path: Path) -> None:
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read_json(path: Path) -> dict:
        record(path)
        return json.loads(path.read_text())

    config = read_json(args.cohort_config)
    for name in ("trials.csv", "subjects.csv"):
        record(args.analysis / name)
    trials = pd.read_csv(args.analysis / "trials.csv")
    saved_subjects = pd.read_csv(args.analysis / "subjects.csv").set_index("subject")
    assert set(trials.iSub.unique()) == set(config["subjects"]) == set(saved_subjects.index)
    assert not trials.duplicated(["iSub", "trial"]).any()
    results, checks, candidates, selected = {}, [], [], []
    for source in config["source_runs"]:
        manifest = read_json(Path(source) / "manifest.json")
        fits = read_json(Path(source) / "fit_results.json")
        assert not fits["smoke_only"] and not manifest["smoke"]
        for spec in manifest["subjects"]:
            sid = spec["subject"]
            assert sid not in results
            engine_path = Path("configs") / spec["engine"].split("/configs/", 1)[1]
            record(engine_path)
            assert inputs[str(engine_path)] == manifest["input_sha256"][spec["engine"]]
            engine = yaml.safe_load(engine_path.read_text())
            kwargs = engine["modules"]["hypo_transitions_mod"]["kwargs"]
            controller = kwargs["nested_feedback_accumulator_controller"]
            assert controller["event_history_excludes_latest_error"]
            assert controller["initial_failure"] == 0
            if spec["condition"] == 3:
                assert kwargs["feedback_interpretation"] == "full_success"
            results[sid] = (fits["subjects"][str(sid)], float(controller["accumulator_decay"]), source)

    assert set(results) == set(config["subjects"])
    for sid in config["subjects"]:
        result, decay, source = results[sid]
        g = trials.loc[trials.iSub == sid]
        np.testing.assert_array_equal(g.trial, np.arange(1, len(g) + 1))
        assert len(g) == result["trial_count"]
        assert g.condition.nunique() == 1 and g.condition.iloc[0] == result["condition"]
        assert g.feedback.isin([0, 0.5, 1]).all()
        assert g.feedback.isin([0, 1]).all() or result["condition"] == 3
        previous_error = np.r_[0.0, 1.0 - g.feedback.eq(1).to_numpy(dtype=float)[:-1]]
        failure = failure_history(previous_error, decay)
        history = np.r_[0.0, failure[:-1]]
        assert np.all((failure >= 0) & (failure <= 1))
        audit = result["independent_audit"]
        best_score = min(audit["scores"].values())
        assert result["selected"] == saved_subjects.loc[sid, "selected"]
        for candidate in result["candidate_bank"]:
            p = extract_model_0826_parameters(candidate["hyperparams"])
            pid = candidate["id"]
            is_selected = pid == result["selected"]
            score = audit["scores"][pid]
            row = {"subject": sid, "condition": result["condition"], "candidate": pid,
                   "selected": is_selected, "saved_near": pid in result["near_candidates"],
                   "audit_mean_nll": score, "audit_nll_above_bank_best": score - best_score,
                   "audit_nll_minus_selected": score - audit["scores"][result["selected"]],
                   "fixed_decay": decay, **p,
                   "isolated_error_logit_lag1": p["delta_E"],
                   "isolated_error_logit_lag2": (1 - decay) * p["c_A"],
                   "source": source}
            interval = audit["selected_minus_candidates"].get(pid, {}).get("interval95")
            row["numerical_seed_delta_nll_lower"] = -interval[1] if interval else np.nan
            row["numerical_seed_delta_nll_upper"] = -interval[0] if interval else np.nan
            # Closed-form audit of the documented controller, not another model engine.
            event = expit(logit(p["E_C"]) + p["delta_E"] * previous_error + p["c_A"] * history)
            range_probability = p["g_0"] + (1 - p["g_0"]) * p["c_G"] * failure
            if is_selected:
                for key, value in p.items():
                    np.testing.assert_allclose(value, saved_subjects.loc[sid, key], atol=1e-12)
                # Saved replay has no replacement event at workspace initialization.
                assert abs(g.search.iloc[0]) < 1e-12
                np.testing.assert_allclose(event[1:], g.search.to_numpy()[1:], atol=1e-12, rtol=0)
                np.testing.assert_allclose(range_probability[1:], g.global_range.to_numpy()[1:], atol=1e-12, rtol=0)
                checks.append({"subject": sid, "n": len(g),
                               "event_max_reconstruction_error": float(max(abs(event[1:] - g.search.to_numpy()[1:]))),
                               "range_max_reconstruction_error": float(max(abs(range_probability[1:] - g.global_range.to_numpy()[1:])))})
            if row["saved_near"]:
                projection, projected_failure = project_unified(previous_error, event[1:])
                row.update(projection)
                row.update(project_range(projected_failure, range_probability[1:]))
            candidates.append(row)
            if is_selected:
                selected.append({**row, "n": len(g), "fit_status": result["status"],
                                 "audit_status": audit["status"], "issues": ";".join(result["issues"])})

    args.output.mkdir(parents=True)
    pd.DataFrame(candidates).to_csv(args.output / "candidates.csv", index=False)
    pd.DataFrame(selected).to_csv(args.output / "selected.csv", index=False)
    pd.DataFrame(checks).to_csv(args.output / "validation.csv", index=False)
    record(Path(__file__).relative_to(Path.cwd()))
    metadata = {"input_sha256": inputs, "subjects": config["subjects"],
                "trials": len(trials), "controller_trials_compared": len(trials) - len(results),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__, "PyYAML": yaml.__version__},
                "scope": "Saved candidates and deterministic controller-curve projections only; no PF or choice refit.",
                "projection": {"objective": "Unweighted logit MSE on t>=2, conditional on actual feedback",
                               "decay_bounds": [0, 0.999], "gain": "nonnegative, no upper bound",
                               "baseline": "free logit intercept, no original E_C lower bound",
                               "history": "full chronological sequence, no session reset",
                               "range": "Re-estimate g0,cG at event-optimal decay; not a joint optimum"},
                "limitations": ["Latent search curves are model outputs, not measured search events.",
                                "Small curve error does not bound choice likelihood or trajectory error.",
                                "Near banks are finite search products, not parameter confidence sets.",
                                "Saved bootstrap intervals describe PF seed noise, not participant uncertainty.",
                                "Trial 1 initializes the workspace and has saved search=0; it is excluded."]}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(selected)[["subject", "event_mae_pp", "event_max_error_pp", "range_mae_pp_at_event_decay"]].to_string(index=False))
    print(f"Validated {len(results)} subjects and {len(trials) - len(results)} controller trials.")


if __name__ == "__main__":
    main()
