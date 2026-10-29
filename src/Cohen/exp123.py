"""Run small or explicitly configured continuous exp123 fits/replays/simulations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
import yaml
from threadpoolctl import threadpool_limits

from .continuous import ContinuousFeatureMixture
from .data import prepare_exp123
from .feedback import category_count
from .fitting import FitConfig, fit_continuous


def quadrature_diagnostic(model: ContinuousFeatureMixture, X: np.ndarray,
                          higher_order: int) -> dict:
    """Compare the same fixed state under a finer integration grid."""
    if higher_order <= model.quadrature_order:
        raise ValueError("quadrature_check_order must exceed quadrature_order")
    refined = ContinuousFeatureMixture(
        n_dimensions=model.n_dimensions, n_categories=model.n_categories,
        max_order=model.max_order, beta=model.beta, eta=model.eta,
        quadrature_order=higher_order, alpha=model.alpha, bias=model.bias,
        update_rule=model.update_rule,
    )
    z, e = model._statistics()
    zr, er = refined._statistics()
    return {"base_order": model.quadrature_order, "check_order": higher_order,
            "max_abs_log_z_difference": float(np.max(np.abs(z - zr))),
            "max_abs_expected_feature_difference": float(np.max(np.abs(e - er))),
            "max_abs_choice_probability_difference": float(np.max(np.abs(
                model.predict_proba(X) - refined.predict_proba(X))))}


def parameter_record(model: ContinuousFeatureMixture) -> dict:
    record = asdict(model)
    record["alpha"], record["bias"] = model.alpha.tolist(), model.bias.tolist()
    return record


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run_config(config: dict) -> dict:
    """Run the declared scope into a new directory, recording parameters and timing."""
    allowed = {"data", "subjects", "max_trials", "session_policy", "mode", "model",
               "fit", "fit_first_n", "seed", "output_dir", "quadrature_check_order"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if config.get("session_policy") != "carry":
        raise ValueError("session_policy must explicitly be 'carry'; each subject starts fresh")
    mode = config.get("mode", "replay")
    if mode not in ("fit", "replay", "simulate"):
        raise ValueError("mode must be fit, replay or simulate")
    if mode != "fit" and config.get("fit_first_n") is not None:
        raise ValueError("fit_first_n applies only to mode=fit")
    source = Path(config["data"])
    outdir = Path(config["output_dir"])
    if outdir.exists():
        raise FileExistsError(f"Choose a new output directory: {outdir}")
    data = pd.read_csv(source)
    if "iSub" not in data or data.iSub.isna().any():
        raise ValueError("data requires nonmissing iSub values")
    requested = config.get("subjects")
    subjects = sorted(data.iSub.unique()) if requested is None else requested
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("subjects must be nonempty and unique")
    prepared = []
    for sid in subjects:
        if not isinstance(sid, (int, np.integer)) or sid < 1:
            raise ValueError("exp123 subject IDs must be positive integers")
        prepared.append((int(sid), prepare_exp123(data[data.iSub == sid], max_trials=config.get("max_trials"))))
    settings = dict(config.get("model", {}))
    if settings.get("n_dimensions", 4) != 4:
        raise ValueError("exp123 requires four continuous dimensions")
    if any(key in settings for key in ("n_categories", "alpha", "bias", "seed")):
        raise ValueError("model category count, initial state and seed are supplied by the workflow")
    seed = config.get("seed", 20260430)
    fit_cfg = FitConfig(**{**config.get("fit", {}), "seed": seed})
    fit_cfg.validate()
    # Check model configuration before creating artifacts.
    ContinuousFeatureMixture(**settings, n_categories=2, seed=seed)
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = {"status": "running", "config": config, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "code_commit": _git_output("rev-parse", "HEAD"),
               "worktree_status": _git_output("status", "--short"),
               "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))},
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__, "pyyaml": yaml.__version__},
               "numeric_threads": 1, "runs": []}
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        with threadpool_limits(limits=1):
            for sid, (frame, X, y, choices, feedback, condition) in prepared:
                subdir = outdir / f"subject_{sid}"
                subdir.mkdir()
                fit_info = None
                if mode == "fit":
                    initial, fit_info = fit_continuous(X, choices, feedback, condition,
                        model_config={**settings, "seed": seed}, fit_first_n=config.get("fit_first_n"), config=fit_cfg)
                else:
                    initial = ContinuousFeatureMixture(**settings, n_categories=category_count(condition), seed=seed)
                model = initial.copy()
                result = model.run_session(X, None if mode == "simulate" else choices,
                    None if mode == "simulate" else feedback, condition, y, simulation_seed=seed)
                P = result["probabilities"]
                if not np.all(np.isfinite(P)) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-12):
                    raise FloatingPointError("invalid choice probabilities")
                meta = [c for c in frame.columns if c in {"iSub", "condition", "iSession", "iBlock", "iTrial",
                        "category", "choice", "feedback", "ambiguous"} or c.startswith("feature") and c.endswith("_name")]
                predictions = frame[meta + [f"feature{i}" for i in range(1, 5)]].copy()
                predictions["sequence_trial"] = np.arange(1, len(frame) + 1)
                for j in range(model.n_categories):
                    predictions[f"p_category{j+1}"] = P[:, j]
                for key in ("predicted_category", "p_correct", "expected_feedback", "log_likelihood"):
                    predictions[key] = result[key]
                predictions["model_choice"] = result["choices"]
                predictions["model_feedback"] = result["feedback"]
                n_fit = fit_info["n_fit_trials"] if fit_info else 0
                predictions["scope"] = (["train"] * n_fit + ["conditional_test"] * (len(frame) - n_fit)
                                        if mode == "fit" else mode)
                predictions.to_csv(subdir / "trial_predictions.csv", index=False)
                np.savez_compressed(subdir / "states.npz", alpha_before_feedback=result["alpha_before_feedback"],
                                    bias_before_feedback=result["bias_before_feedback"],
                                    sequence_trial=np.arange(1, len(frame) + 1),
                                    iSession=frame.iSession.to_numpy(), iTrial=frame.iTrial.to_numpy())
                parameters = {"initial": parameter_record(initial), "final": parameter_record(model),
                              "feature_subsets_zero_based": model.subsets}
                (subdir / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
                row = {"subject": sid, "condition": condition, "n_trials": len(frame),
                       "n_categories": model.n_categories, "mode": mode,
                       "sequence_negative_log_likelihood": float(-result["log_likelihood"].sum()),
                       "mean_p_correct": float(result["p_correct"].mean()),
                       "mean_expected_feedback": float(result["expected_feedback"].mean()),
                       "fit": fit_info}
                if fit_info:
                    row["train_nll"] = float(-result["log_likelihood"][:n_fit].sum())
                    row["conditional_test_nll"] = (float(-result["log_likelihood"][n_fit:].sum())
                                                   if n_fit < len(frame) else None)
                check = config.get("quadrature_check_order")
                if check is not None:
                    row["quadrature_initial"] = quadrature_diagnostic(initial, X, check)
                    row["quadrature_final"] = quadrature_diagnostic(model, X, check)
                summary["runs"].append(row)
        summary["status"] = "computationally_complete"
        summary["evidence_scope"] = "Implementation validation; behavioral fit quality and parameter recovery require separate studies."
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--mode", choices=("fit", "replay", "simulate"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    for key in ("subjects", "max_trials", "mode"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.outdir is not None:
        config["output_dir"] = str(args.outdir)
    summary = run_config(config)
    print(json.dumps({"status": summary["status"], "output_dir": config["output_dir"],
                      "n_subjects": len(summary["runs"])}, indent=2))


if __name__ == "__main__":
    main()
