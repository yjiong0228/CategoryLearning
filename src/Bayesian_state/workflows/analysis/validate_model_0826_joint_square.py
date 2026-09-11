"""Independently rescore a preselected four-point S129 coordinate-trap example."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from src.Bayesian_state.optimization.search.cd_v2 import canonical_point_key
from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.utils.subjects import deep_update


def run(pipeline: Path, probe: Path, jobs: int) -> None:
    output = probe / "coordinate_trap_validation"
    output.mkdir(exist_ok=False)
    source = probe / "trapped_start4_square.json"
    square = json.loads(source.read_text())
    hyper = pipeline / "configs/PMH_hyper.yaml"
    cfg = yaml.safe_load(hyper.read_text())
    cfg["output_dir"] = str((output / "optimizer_context").resolve())
    optimizer = HyperCDOptimizer(cfg, hyper.resolve())
    stage_cfg = deep_update(optimizer.base_sim_config, cfg["final_rescore"]["simulation_overrides"])
    candidates = json.loads((probe / "validation_candidates.json").read_text())
    reuse = {canonical_point_key(c["hyperparams"]): probe / "validation" / f"candidate_{c['candidate']}.npz"
             for c in candidates}
    probabilities, rows = [], []
    observed, mask = None, None
    for i, point in enumerate(square["points"]):
        hp = point["hyperparams"]
        path = output / f"point_{i}.npz"
        previous = reuse.get(canonical_point_key(hp))
        if previous is not None and previous.is_file():
            shutil.copy2(previous, path)
        else:
            sub, eng, pm, sm, loss, delta, window, _ = optimizer._resolve_sim_components(stage_cfg, 129, [129])
            point_cfg, point_eng = optimizer._apply_hyperparams(hp, sub, eng)
            runner, paths = optimizer._build_runner(point_cfg, point_eng)
            runs, _, point_seed, _ = optimizer._simulate_runs_for_point(
                stage_name="final_rescore", runner=runner, dataset_paths=paths, subject_id=129,
                point=hp, simulation_repeats=16, window_size=window, stop_at=1., max_trials=None,
                keep_logs=False, prediction_mode=pm, selection_prediction_mode=sm, loss_metric=loss,
                loss_delta=delta, hyper_candidate_seed=20260910, n_jobs=jobs,
                evaluation_protocol=point_cfg.get("evaluation_protocol"), force_common_random_numbers=True,
                seed_family="S129_joint_probe_validation_20260910_v1")
            pp = []
            for result in runs:
                metrics = result.metrics_by_mode[sm]
                y = np.asarray(metrics["observed_choice_index"], dtype=int)
                m = np.asarray(metrics["valid_trial_mask"], dtype=bool)
                if observed is not None:
                    np.testing.assert_array_equal(y, observed)
                    np.testing.assert_array_equal(m, mask)
                observed, mask = y, m
                p = np.asarray(metrics["pred_category_probs"], dtype=float)
                np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-8)
                pp.append(p[np.arange(len(y)), y])
            np.savez_compressed(path, observed_choice_probability=np.array(pp), observed=observed,
                                score_mask=mask, simulation_point_seed=point_seed)
        with np.load(path) as data:
            if observed is not None:
                np.testing.assert_array_equal(observed, data["observed"])
                np.testing.assert_array_equal(mask, data["score_mask"])
            observed, mask = data["observed"].copy(), data["score_mask"].copy()
            p = data["observed_choice_probability"].copy()
            assert p.shape == (16, 256) and mask.sum() == 255
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            probabilities.append(p)
            rows.append({"role": point["role"], "original_index": point["combination_index"],
                         "fine_mean_nll": point["aggregated_error"],
                         "validation_mean_nll": float(-np.log(np.clip(p.mean(axis=0)[mask], 1e-12, 1)).mean()),
                         "simulation_point_seed": int(data["simulation_point_seed"]),
                         "reused_from": str(previous) if previous is not None else None,
                         "hyperparams": hp})
        (output / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "hyperparams"}), flush=True)
    assert len({r["simulation_point_seed"] for r in rows}) == 1
    p = np.array(probabilities)
    total = -np.log(np.clip(p.mean(axis=1)[:, mask], 1e-12, 1)).sum(axis=1)
    weights = np.random.default_rng(20260912).multinomial(16, np.full(16, 1/16), size=4000) / 16
    boot = -np.log(np.clip(np.einsum("kb,cbt->ckt", weights, p)[:, :, mask], 1e-12, 1)).sum(axis=2)
    gain = boot[0] - boot
    intervals = np.quantile(gain, [.025, .975], axis=1)
    summary = {"validation_total_nll": total.tolist(), "gain_from_anchor": (total[0] - total).tolist(),
               "paired_mc_gain95_lower": intervals[0].tolist(), "paired_mc_gain95_upper": intervals[1].tolist(),
               "numerical_escape_frequency": float(((gain[1] <= 0) & (gain[2] <= 0) & (gain[3] > 0)).mean()),
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "jobs": jobs, "particles": 128, "repeats": 16, "bootstrap_seed": 20260912,
               "interpretation": "A selected archived example, independently rescored. PF bootstrap conditional on the fitted sequence; not a parameter interval or global-optimum claim."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "bootstrap.npz", total_nll=boot, gain_from_anchor=gain)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    run(args.pipeline, args.probe, args.jobs)
