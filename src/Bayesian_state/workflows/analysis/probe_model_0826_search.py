"""Bounded S129 joint-move and dispersed-start probe using the shared evaluator.

This is an exploratory workflow, not a replacement production optimizer.
Search comparisons reuse the archived fine budget/seeds after anchor replay.
Final validation uses a fresh common seed family and fixed candidate parameters.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from src.Bayesian_state.optimization.artifacts import to_builtin
from src.Bayesian_state.optimization.diagnostics.search import flatten_hyperparams
from src.Bayesian_state.optimization.search.cd_v2 import canonical_point_key
from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.utils.subjects import deep_update


def paired_square(anchor: dict, key_a: str, value_a: object,
                  key_b: str, value_b: object) -> dict[str, dict]:
    """Keep a full two-block comparison, including both one-block controls."""
    if key_a == key_b:
        raise ValueError("The two blocks must differ")
    points = {name: deepcopy(anchor) for name in ("single_a", "single_b", "joint")}
    points["single_a"][key_a] = deepcopy(value_a)
    points["single_b"][key_b] = deepcopy(value_b)
    points["joint"][key_a] = deepcopy(value_a)
    points["joint"][key_b] = deepcopy(value_b)
    return points


def write_json(path: Path, payload: object) -> None:
    # Atomic replacement is only for this new probe's own checkpoint/summary.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")
    temporary.replace(path)


def run(pipeline: Path, output: Path, jobs: int, resume: bool, prepare_only: bool) -> None:
    hyper = pipeline / "configs/PMH_hyper.yaml"
    archive = pipeline / "models/PMH/optimization/subject_129"
    config = yaml.safe_load(hyper.read_text())
    assert config["subjects"] == [129]
    files = [hyper, pipeline / "configs/PMH_base.yaml", pipeline / "configs/PMH_engine.yaml",
             archive / "all_combinations.jsonl", archive / "final_rescore.jsonl", Path(__file__)]
    # Resume must also detect changes behind config paths, including shared code.
    provenance = json.loads((pipeline / "provenance.json").read_text())
    files += [Path(p) for p in provenance["files_sha256"]
              if (p.startswith("data/") or (p.startswith("src/") and p.endswith(".py")))
              and Path(p).is_file()]
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if resume:
        assert json.loads((output / "context.json").read_text())["input_sha256"] == fingerprints
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "context.json", {
            "subject": 129, "condition": 1, "model": "PMH Model0826",
            "jobs": jobs, "input_sha256": fingerprints,
            "search_budget": {"particles": 64, "repeats": 8, "seed_family": "archived fine"},
            "validation_budget": {"particles": 128, "repeats": 16,
                                  "seed_family": "S129_joint_probe_validation_20260910_v1"},
            "proposal_seed": 20260910, "bounds": "unchanged archived fine support",
            "purpose": "Finite counterexample search; absence of improvement cannot establish global optimality."})
    config["output_dir"] = str((output / "optimizer_context").resolve())
    optimizer = HyperCDOptimizer(config, hyper.resolve())
    optimizer.parallel_budget = jobs
    fine_cfg = deep_update(optimizer.base_sim_config, config["stages"]["fine"]["simulation_overrides"])
    records = [json.loads(s) for s in (archive / "all_combinations.jsonl").read_text().splitlines()]
    fine = sorted([r for r in records if r["stage"] == "fine"], key=lambda r: r["aggregated_error"])
    original_final = [json.loads(s) for s in (archive / "final_rescore.jsonl").read_text().splitlines()]
    c4 = min(original_final, key=lambda r: r["aggregated_error"])["hyperparams"]
    cache = {canonical_point_key(r["hyperparams"]): r for r in fine}
    original_keys = set(cache)
    next_index = max(r["combination_index"] for r in records) + 1
    result_path = output / "new_fine_evaluations.jsonl"
    if result_path.exists():
        for row in map(json.loads, result_path.read_text().splitlines()):
            cache[canonical_point_key(row["hyperparams"])] = row
            next_index = max(next_index, row["combination_index"] + 1)
    blocks = list(config["stages"]["fine"]["hyperparam_space"])
    space = {k: v["values"] for k, v in config["stages"]["fine"]["hyperparam_space"].items()}

    anchors = [{"name": "C4", "point": c4}]
    filters = [("chi_1", lambda p: p["persistent_execution"] == 1),
               ("low_memory", lambda p: p["gamma"] <= .8),
               ("low_capacity", lambda p: p["capacity"] <= 2),
               ("low_beta", lambda p: p["beta_init"] <= 2)]
    for name, predicate in filters:
        selected = next(r for r in fine if predicate(flatten_hyperparams(r["hyperparams"])))
        anchors.append({"name": name, "point": selected["hyperparams"],
                        "archived_combination_index": selected["combination_index"],
                        "archived_mean_nll": selected["aggregated_error"]})
    write_json(output / "anchors.json", anchors)

    def evaluate(points: list[dict], phase: str, force: bool = False) -> None:
        nonlocal next_index
        unique = {canonical_point_key(p): p for p in points}
        pending = [p for k, p in unique.items() if force or k not in cache]
        for start in range(0, len(pending), 4):
            chunk = pending[start:start + 4]
            entries = [{"position": i, "point": p, "combination_index": next_index + i}
                       for i, p in enumerate(chunk)]
            next_index += len(entries)
            began = time.time()
            results, _ = optimizer._evaluate_missing_entries_flat(
                stage_name="fine", stage_sim_cfg=fine_cfg, subjects=[129], restart_id=0,
                iter_id=0, coordinate=phase, missing_entries=entries, value_jobs=jobs, repeat_jobs=1)
            for result in results:
                row = to_builtin(asdict(result))
                key = canonical_point_key(row["hyperparams"])
                if force:
                    np.testing.assert_allclose(row["aggregated_error"], cache[key]["aggregated_error"], atol=1e-10, rtol=0)
                    with (output / "anchor_replays.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                else:
                    assert np.isfinite(row["aggregated_error"])
                    cache[key] = row
                    with result_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
            progress = {"phase": phase, "phase_completed": min(start + 4, len(pending)),
                        "phase_pending_total": len(pending), "batch_seconds": time.time() - began,
                        "new_points_total": len(set(cache) - original_keys),
                        "best_fine_mean_nll": min(r["aggregated_error"] for r in cache.values())}
            write_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)

    def alternative(anchor: dict, block: str, offset: int = 0) -> object:
        distinct = []
        seen = {canonical_point_key({block: anchor[block]})}
        for row in fine:
            value = row["hyperparams"][block]
            key = canonical_point_key({block: value})
            if key not in seen:
                distinct.append(value)
                seen.add(key)
        return distinct[min(offset, len(distinct) - 1)]

    def square(anchor: dict, label: str, ia: int, ib: int, offset: int = 0) -> dict:
        a, b = blocks[ia], blocks[ib]
        return {"anchor_label": label, "anchor": anchor, "block_a": a, "block_b": b,
                "points": paired_square(anchor, a, alternative(anchor, a, offset),
                                         b, alternative(anchor, b, offset))}

    squares = [square(c4, "C4", a, b) for a, b in
               [(0, 1), (0, 2), (1, 4), (2, 4), (2, 3), (3, 4), (5, 6), (5, 7)]]
    for anchor, pairs in zip(anchors[1:], [[(0, 2), (3, 4)], [(1, 2), (1, 5)],
                                         [(0, 1), (0, 5)], [(5, 6), (5, 7)]]):
        squares.extend(square(anchor["point"], anchor["name"], a, b) for a, b in pairs)
    rng = np.random.default_rng(20260910)
    dispersed = [{k: deepcopy(v[int(rng.integers(len(v)))]) for k, v in space.items()} for _ in range(8)]
    plan = {"squares": squares, "dispersed_grid_points": dispersed,
            "adaptive_rule": "Best two distinct new points each receive (event,c_A) and (global,c_A) paired squares with second-best archived alternative values.",
            "final_selection": "Always C4; best new point; best distant-origin point; best fine-stage synergy joint if any; fill to four with archived fine best. Deduplicate, maximum four."}
    write_json(output / "proposal_plan.json", plan)
    if prepare_only:
        print(json.dumps({"prepared": str(output), "initial_squares": len(squares), "dispersed_points": len(dispersed)}))
        return
    if not (output / "anchor_replay_complete.json").exists():
        evaluate([a["point"] for a in anchors], "anchor_replay", force=True)
        write_json(output / "anchor_replay_complete.json", {"matched": len(anchors)})
    evaluate([p for s in squares for p in s["points"].values()] + dispersed, "initial_joint_and_dispersed")
    adaptive_path = output / "adaptive_plan.json"
    if adaptive_path.exists():
        adaptive = json.loads(adaptive_path.read_text())
    else:
        new_ranked = sorted([r for k, r in cache.items() if k not in original_keys], key=lambda r: r["aggregated_error"])
        adaptive = [square(row["hyperparams"], f"adaptive_{i+1}", a, b, offset=1)
                    for i, row in enumerate(new_ranked[:2]) for a, b in [(2, 4), (3, 4)]]
        write_json(adaptive_path, adaptive)
    evaluate([p for s in adaptive for p in s["points"].values()], "adaptive_joint")
    squares += adaptive
    effects = []
    for s in squares:
        vals = {name: cache[canonical_point_key(p)]["aggregated_error"] for name, p in s["points"].items()}
        base = cache[canonical_point_key(s["anchor"])]["aggregated_error"]
        effects.append({**s, "anchor_mean_nll": base, "scores": vals,
                        "joint_gain_total_nll": (base - vals["joint"]) * 255,
                        "blocked_single_moves": vals["single_a"] >= base and vals["single_b"] >= base,
                        "synergy_escape": vals["single_a"] >= base and vals["single_b"] >= base and vals["joint"] < base - 1e-4})
    write_json(output / "joint_effects.json", effects)
    new_ranked = sorted([r for k, r in cache.items() if k not in original_keys], key=lambda r: r["aggregated_error"])
    distant = [p for s in squares if s["anchor_label"] not in ("C4", "adaptive_1", "adaptive_2")
               for p in s["points"].values()] + dispersed
    distant_best = min(distant, key=lambda p: cache[canonical_point_key(p)]["aggregated_error"])
    candidates = [c4, new_ranked[0]["hyperparams"], distant_best]
    escapes = sorted([s for s in effects if s["synergy_escape"]], key=lambda s: s["scores"]["joint"])
    candidates += [s["points"]["joint"] for s in escapes[:1]] + [fine[0]["hyperparams"]]
    selected = list({canonical_point_key(p): p for p in candidates}.values())[:4]
    write_json(output / "validation_candidates.json", [{"candidate": i + 1, "hyperparams": p,
               "fine_mean_nll": cache[canonical_point_key(p)]["aggregated_error"]} for i, p in enumerate(selected)])
    validation = output / "validation"
    validation.mkdir(exist_ok=True)
    validation_cfg = deep_update(optimizer.base_sim_config, config["final_rescore"]["simulation_overrides"])
    probabilities = []
    validation_observed = None
    validation_mask = None
    for i, point in enumerate(selected):
        path = validation / f"candidate_{i+1}.npz"
        if not path.exists():
            sub, eng, pm, sm, loss, delta, window, _ = optimizer._resolve_sim_components(validation_cfg, 129, [129])
            point_cfg, point_eng = optimizer._apply_hyperparams(point, sub, eng)
            runner, paths = optimizer._build_runner(point_cfg, point_eng)
            runs, _, point_seed, _ = optimizer._simulate_runs_for_point(
                stage_name="final_rescore", runner=runner, dataset_paths=paths, subject_id=129,
                point=point, simulation_repeats=16, window_size=window, stop_at=1., max_trials=None,
                keep_logs=False, prediction_mode=pm, selection_prediction_mode=sm, loss_metric=loss,
                loss_delta=delta, hyper_candidate_seed=20260910, n_jobs=jobs,
                evaluation_protocol=point_cfg.get("evaluation_protocol"), force_common_random_numbers=True,
                seed_family="S129_joint_probe_validation_20260910_v1")
            pp = []
            for run in runs:
                metrics = run.metrics_by_mode[sm]
                y = np.asarray(metrics["observed_choice_index"], dtype=int)
                mask = np.asarray(metrics["valid_trial_mask"], dtype=bool)
                if pp:
                    np.testing.assert_array_equal(y, observed)
                    np.testing.assert_array_equal(mask, score_mask)
                observed, score_mask = y, mask
                p = np.asarray(metrics["pred_category_probs"], dtype=float)
                np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-8)
                pp.append(p[np.arange(len(y)), y])
            np.savez_compressed(path, observed_choice_probability=np.array(pp), observed=observed,
                                score_mask=score_mask, simulation_point_seed=point_seed)
        with np.load(path) as data:
            pp = data["observed_choice_probability"]
            assert pp.shape == (16, 256) and np.isfinite(pp).all() and ((pp >= 0) & (pp <= 1)).all()
            if validation_observed is not None:
                np.testing.assert_array_equal(data["observed"], validation_observed)
                np.testing.assert_array_equal(data["score_mask"], validation_mask)
            observed = data["observed"].copy()
            score_mask = data["score_mask"].copy()
            validation_observed, validation_mask = observed, score_mask
            probabilities.append(pp.copy())
        assert score_mask.sum() == 255
        score = float(-np.log(np.clip(pp.mean(axis=0)[score_mask], 1e-12, 1)).mean())
        progress = {"phase": "independent_validation", "candidate": i + 1, "total": len(selected), "mean_nll": score}
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    pp = np.array(probabilities)
    losses = -np.log(np.clip(pp.mean(axis=1)[:, score_mask], 1e-12, 1)).sum(axis=1)
    weights = np.random.default_rng(20260911).multinomial(16, np.full(16, 1/16), size=4000) / 16
    boot = -np.log(np.clip(np.einsum("kb,cbt->ckt", weights, pp)[:, :, score_mask], 1e-12, 1)).sum(axis=2)
    # Positive gain means the alternative improves on C4, candidate 1.
    gain = boot[0] - boot
    intervals = np.quantile(gain, [.025, .975], axis=1)
    summary = {"new_fine_points": len(new_ranked), "squares_evaluated": len(effects),
               "fine_synergy_escapes": len(escapes), "C4_fine_mean_nll": cache[canonical_point_key(c4)]["aggregated_error"],
               "best_new_fine_mean_nll": new_ranked[0]["aggregated_error"],
               "validation_total_nll": losses.tolist(), "validation_gain_from_C4": (losses[0] - losses).tolist(),
               "paired_mc_gain95_lower": intervals[0].tolist(), "paired_mc_gain95_upper": intervals[1].tolist(),
               "bootstrap_selected_frequency": [(boot.argmin(axis=0) == i).mean() for i in range(len(selected))],
               "interpretation": "Fixed shortlist, independent paired PF seed family, same fitted sequence. Numerical bootstrap, not parameter confidence or global-optimum certification."}
    write_json(output / "summary.json", summary)
    np.savez_compressed(validation / "bootstrap.npz", total_nll=boot, gain_from_C4=gain)
    write_json(output / "progress.json", {"phase": "complete", **summary})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    run(args.pipeline, args.output, args.jobs, args.resume, args.prepare_only)
