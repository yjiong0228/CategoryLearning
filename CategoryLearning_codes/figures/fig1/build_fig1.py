"""Build a new, non-overwriting Fig. 1 evidence bundle from the full cohort.

Run from the repository root with python -m CategoryLearning_codes.figures.fig1.build_fig1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/categorylearning-mpl")

import matplotlib
import numpy as np
import pandas as pd

from .behavior import KEYS, measure, order_trials, reconcile_raw, representatives, sha256, validate_task_rule
from .render import atlases, main_figure, sensitivity, oral_atlases
from .reporting import readout
from .nonoral import analyze, plot_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/exp123/processed/Task2_processed.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/exp123/raw/Task2"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True, help="New directory; refuses existing paths")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new version directory")
    config = json.loads(args.config.read_text())
    source_hash = sha256(args.data)
    frame = pd.read_csv(args.data)
    from src.oral_coding import Recording_Processor_Center, FEATURE_NAME_TO_PART
    use_cols = [f"feature{i}_use" for i in range(1,5)]
    coder = Recording_Processor_Center()
    coded = coder.process_use(frame)
    expected = pd.DataFrame(index=frame.index, columns=use_cols, dtype=float)
    for i in range(1, 5):
        for name, positions in frame.groupby(f"feature{i}_name").groups.items():
            part_index = coder.parts.index(FEATURE_NAME_TO_PART[name]) + 1
            expected.loc[positions, f"feature{i}_use"] = coded.loc[
                positions, f"feature{part_index}_oraluse"].to_numpy()
    pd.testing.assert_frame_equal(frame[use_cols].astype(float), expected)
    raw_duplicate_n = int(frame.duplicated(KEYS).sum())
    ordered = order_trials(frame)
    validate_task_rule(ordered)
    data, flagged, raw_hashes = reconcile_raw(ordered, args.raw_dir)
    trials, summary, blocks = measure(data, config)
    selected = representatives(summary)
    for task in config['tasks']:
        assert selected.loc[selected.condition == task['condition'], 'iSub'].tolist() == task['example_subjects']
    assert len(trials) == len(frame)
    assert len(summary) == frame.groupby(["condition", "iSub"]).ngroups
    assert not trials.duplicated(KEYS).any()
    assert trials.correct.isin([0, 1]).all()
    if trials.stimulus_feedback_mismatch.any():
        raise ValueError("Raw stimulus labels disagree with feedback; resolve before plotting")
    audit = {"rows": len(trials), "subjects": len(summary), "duplicate_keys": raw_duplicate_n,
             "label_feedback_mismatch": int(trials.label_feedback_mismatch.sum()),
             "stimulus_feedback_mismatch": int(trials.stimulus_feedback_mismatch.sum()),
             "category_source_mismatch": int(trials.category_source_mismatch.sum()),
             "missing_text": int((~trials.report_present).sum()),
             "recognized_reports": int(trials.report_recognized.sum()),
             "ambiguous_trials": int(trials.ambiguous.sum()),
             "behavior_rows_excluded": 0, "subjects_excluded": 0, "task_rule_mismatch": 0}
    args.output.mkdir(parents=True, exist_ok=False)
    source_cols = KEYS + ["source_row", "trial", "category", "raw_stimulus_category", "choice", "feedback",
                         "correct", "ambiguous", "rolling_accuracy", "rolling_label_accuracy", "rolling_clear_n",
                         "rolling_clear_accuracy", "report_present", "report_recognized", "feature_count",
                         "feature_set", "report_set_change", "report_comparison_gap", "required_features",
                         "path_coverage", "irrelevant_fraction", "feature1_use", "feature2_use", "feature3_use", "feature4_use"]
    trials[source_cols].to_csv(args.output/"trial_source.csv", index=False)
    summary.to_csv(args.output/"subject_summary.csv", index=False)
    phase_rows = []
    for item in summary.itertuples():
        for phase in ['first', 'last']:
            for metric in ['feature_count', 'path_coverage', 'irrelevant_fraction']:
                phase_rows.append({'condition': item.condition, 'iSub': item.iSub, 'phase': phase,
                                   'metric': metric, 'value': getattr(item, f'{phase}_{metric}'),
                                   'recognized_n': getattr(item, f'{phase}_report_n'),
                                   'disjoint_periods_eligible': item.n_trials >= 2*config['criterion_window']})
    pd.DataFrame(phase_rows).to_csv(args.output/'oral_phase_summary.csv', index=False)
    blocks.to_csv(args.output/"block_summary.csv", index=False)
    selected.to_csv(args.output/"representatives.csv", index=False)
    flagged[KEYS + ["source_row", "category", "raw_behavior_category", "raw_stimulus_category", "choice", "feedback",
                   "label_feedback_mismatch", "category_source_mismatch"]].to_csv(args.output/"category_audit.csv", index=False)
    ordering = summary.sort_values(["condition", "n_trials", "iSub"])[["condition", "iSub", "n_trials"]].copy()
    ordering["heatmap_row"] = ordering.groupby("condition").cumcount()+1
    ordering.to_csv(args.output/"heatmap_order.csv", index=False)
    coverage = []
    for task in config["tasks"]:
        sub = summary[summary.condition == task["condition"]]
        for t in range(1, int(summary.n_trials.max())+1):
            coverage.append({"task": task["task"], "condition": task["condition"], "trial": t,
                             "recorded_n": int((sub.n_trials >= t).sum()),
                             "complete_window_n": int((sub.n_trials >= t).sum()) if t >= config["rolling_window"] else 0})
    pd.DataFrame(coverage).to_csv(args.output/"coverage.csv", index=False)
    print(json.dumps(audit, indent=2), flush=True)
    nonoral_trials, rt_summary, rt_periods, feedback_rt, boundary = analyze(trials, config['criterion_window'])
    rt_summary.to_csv(args.output/'nonoral_subject_summary.csv', index=False)
    rt_periods.to_csv(args.output/'nonoral_periods.csv', index=False)
    feedback_rt.to_csv(args.output/'nonoral_feedback_rt.csv', index=False)
    boundary.to_csv(args.output/'nonoral_boundary.csv', index=False)
    nonoral_trials[KEYS+['trial','choRT','valid_rt','previous_feedback','centered_log_rt',
                        'boundary_distance','distance_bin']].to_csv(args.output/'nonoral_trial_source.csv', index=False)
    main_figure(trials, summary, selected, config, args.output, rt_summary)
    plot_candidates(rt_summary, feedback_rt, boundary, config, args.output)
    atlases(trials, summary, config, args.output)
    oral_atlases(trials, summary, config, args.output)
    sensitivity(trials, summary, config, args.output)
    (args.output/"readout.md").write_text(readout(trials, summary, selected, config, audit, rt_summary))
    assert sha256(args.data) == source_hash, "Input changed during analysis"
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    for path in list(Path(__file__).parent.glob('*.py')) + [args.config, Path(__file__).parent/'REVISION_v10.md', Path('src/oral_coding.py'), Path('src/preprocess_b.py')]:
        shutil.copy2(path, snapshot/path.name)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "exploratory_figure_draft",
                "source": str(args.data), "source_sha256": source_hash, "raw_source_sha256": raw_hashes,
                "config": config, "config_sha256": sha256(args.config), "audit": audit,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "outputs": {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*') if p.is_file()}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
