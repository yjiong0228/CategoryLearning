"""Build the paired-seed S103 rule-commitment guardrail report inputs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.Bayesian_state.utils.streaming import StreamList


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "s103_rule_commitment_guardrail"
RESULT_ROOT = ROOT / "results" / "model_dynamic_continuous"
SUBJECTS = [103, 104, 105, 108, 111, 120, 124, 132]
BASELINE_RESULT = "0812_controller_v2j_guardrail_c0p0_selected8"
INITIAL_RESULT = "0812_controller_v2j_guardrail_c1p4_selected8"
REFINED_RESULT = "0812_controller_v2k_refined_guardrail_selected8"


def load_subject(result_name: str, subject_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = (
        RESULT_ROOT
        / result_name
        / "simulation"
        / "subjects"
        / f"subject_{subject_id}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    reference = payload["raw_runs_ref"]
    stream_path = (path.parent / reference["path"]).resolve()
    runs = list(StreamList(str(stream_path), int(reference["count"])))
    if len(runs) != int(reference["count"]):
        raise ValueError(f"Incomplete run stream: {stream_path}")
    metrics = [run["metrics_by_mode"]["prior_t"] for run in runs]
    return payload, metrics


def mean_array(metrics: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.mean(np.asarray([item[field] for item in metrics], dtype=float), axis=0)


def json_safe(value: Any) -> Any:
    """Replace non-finite numeric values before writing the portable artifact."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def observed_shape_labels() -> dict[int, dict[str, Any]]:
    data = pd.read_csv(ROOT / "data" / "processed" / "Task2_processed.csv")
    data = data.loc[data["condition"].eq(1)].copy()
    labels: dict[int, dict[str, Any]] = {}
    for subject_id in SUBJECTS:
        rows = data.loc[data["iSub"].eq(subject_id)].sort_values(
            ["iSession", "iBlock", "iTrial"]
        )
        feedback = rows["feedback"].to_numpy(dtype=float)
        rolling = np.asarray(
            [np.nanmean(feedback[start : start + 16]) for start in range(1, len(feedback) - 15)],
            dtype=float,
        )
        if rolling.size >= 81:
            candidates = [
                (
                    min(
                        rolling[index - 16] - rolling[index],
                        rolling[index + 16] - rolling[index],
                    ),
                    index,
                )
                for index in range(47, rolling.size - 16)
            ]
            valley_strength, valley_index = max(candidates)
            minimum_stable = float(np.min(rolling[47:-16]))
        else:
            valley_strength = float("nan")
            valley_index = None
            minimum_stable = float(np.min(rolling))

        if subject_id == 105:
            trajectory_type = "short high-accuracy"
        elif minimum_stable >= 0.50 and (
            not np.isfinite(valley_strength) or valley_strength <= 0.125
        ):
            trajectory_type = "smooth rising"
        elif (
            np.isfinite(valley_strength)
            and valley_strength >= 0.30
            and minimum_stable <= 0.50
        ):
            trajectory_type = "deep valley"
        else:
            trajectory_type = "mixed"
        labels[subject_id] = {
            "trial_count": int(feedback.size),
            "trajectory_type": trajectory_type,
            "observed_valley_strength": float(valley_strength),
            "observed_stable_minimum": minimum_stable,
            "observed_valley_end_trial": (
                None if valley_index is None else int(valley_index + 17)
            ),
        }
    return labels


def build_datasets() -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    shape_labels = observed_shape_labels()
    subject_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    s103_metrics: dict[str, list[dict[str, Any]]] = {}
    s103_payloads: dict[str, dict[str, Any]] = {}

    for subject_id in SUBJECTS:
        baseline_payload, baseline_metrics = load_subject(BASELINE_RESULT, subject_id)
        refined_payload, refined_metrics = load_subject(REFINED_RESULT, subject_id)
        if (
            baseline_payload["selection"]["simulation_point_seed"]
            != refined_payload["selection"]["simulation_point_seed"]
        ):
            raise ValueError(f"Unpaired seeds for subject {subject_id}")

        baseline_curve = mean_array(baseline_metrics, "sliding_pred_acc")
        refined_curve = mean_array(refined_metrics, "sliding_pred_acc")
        curve_delta = refined_curve - baseline_curve
        stable_start = min(47, curve_delta.size - 1)
        commitment = np.asarray(
            [
                item["particle_predictive_rule_commitment_probability"]
                for item in refined_metrics
            ],
            dtype=float,
        )
        entries = np.asarray(
            [
                item["particle_predictive_rule_commitment_entry_event_probability"]
                for item in refined_metrics
            ],
            dtype=float,
        )
        exits = np.asarray(
            [
                item["particle_predictive_rule_commitment_exit_event_probability"]
                for item in refined_metrics
            ],
            dtype=float,
        )
        baseline_summary = baseline_payload["statistics"]["marginal_prediction"]
        refined_summary = refined_payload["statistics"]["marginal_prediction"]
        row = {
            "subject_id": int(subject_id),
            **shape_labels[subject_id],
            "baseline_choice_nll": float(baseline_summary["choice_nll"]),
            "refined_choice_nll": float(refined_summary["choice_nll"]),
            "delta_choice_nll": float(
                refined_summary["choice_nll"] - baseline_summary["choice_nll"]
            ),
            "baseline_trajectory_mae": float(
                baseline_summary["trajectory_mean_mae"]
            ),
            "refined_trajectory_mae": float(
                refined_summary["trajectory_mean_mae"]
            ),
            "delta_trajectory_mae": float(
                refined_summary["trajectory_mean_mae"]
                - baseline_summary["trajectory_mean_mae"]
            ),
            "delta_trajectory_crps": float(
                refined_summary["trajectory_crps"]
                - baseline_summary["trajectory_crps"]
            ),
            "commitment_activation_rate": float(np.mean(commitment)),
            "mean_entry_events": float(np.mean(np.sum(entries, axis=1))),
            "mean_exit_events": float(np.mean(np.sum(exits, axis=1))),
            "mean_absolute_curve_change": float(np.mean(np.abs(curve_delta))),
            "largest_stable_downshift": float(np.min(curve_delta[stable_start:])),
            "baseline_stable_minimum": float(np.min(baseline_curve[stable_start:])),
            "refined_stable_minimum": float(np.min(refined_curve[stable_start:])),
        }
        subject_rows.append(row)
        delta_rows.append(
            {
                "subject_id": str(subject_id),
                "trajectory_type": row["trajectory_type"],
                "delta_choice_nll": row["delta_choice_nll"],
                "delta_trajectory_mae": row["delta_trajectory_mae"],
                "commitment_activation_rate": row["commitment_activation_rate"],
                "trial_count": row["trial_count"],
            }
        )
        if subject_id == 103:
            initial_payload, initial_metrics = load_subject(INITIAL_RESULT, subject_id)
            if (
                initial_payload["selection"]["simulation_point_seed"]
                != baseline_payload["selection"]["simulation_point_seed"]
            ):
                raise ValueError("Unpaired initial-gate S103 seed")
            s103_metrics = {
                "baseline": baseline_metrics,
                "initial_gate": initial_metrics,
                "refined_gate": refined_metrics,
            }
            s103_payloads = {
                "baseline": baseline_payload,
                "initial_gate": initial_payload,
                "refined_gate": refined_payload,
            }

    if not s103_metrics:
        raise RuntimeError("S103 was not loaded")

    observed = np.asarray(
        s103_metrics["baseline"][0]["sliding_true_acc"], dtype=float
    )
    trajectory_rows: list[dict[str, Any]] = []
    curves = {
        name: mean_array(metrics, "sliding_pred_acc")
        for name, metrics in s103_metrics.items()
    }
    for index in range(observed.size):
        trajectory_rows.append(
            {
                "window_end_trial": int(index + 17),
                "human_accuracy": float(observed[index]),
                "baseline_accuracy": float(curves["baseline"][index]),
                "initial_gate_accuracy": float(curves["initial_gate"][index]),
                "refined_gate_accuracy": float(curves["refined_gate"][index]),
            }
        )

    probe_rows: list[dict[str, Any]] = []
    for order, (name, label) in enumerate(
        [
            ("baseline", "Baseline: commitment off"),
            ("initial_gate", "Initial gate"),
            ("refined_gate", "Refined conservative gate"),
        ],
        start=1,
    ):
        payload = s103_payloads[name]
        metrics = s103_metrics[name]
        curve = curves[name]
        commitment = mean_array(
            metrics, "particle_predictive_rule_commitment_probability"
        )
        entries = mean_array(
            metrics, "particle_predictive_rule_commitment_entry_event_probability"
        )
        summary = payload["statistics"]["marginal_prediction"]
        probe_rows.append(
            {
                "order": order,
                "probe": label,
                "choice_nll": float(summary["choice_nll"]),
                "trajectory_mae": float(summary["trajectory_mean_mae"]),
                "trajectory_crps": float(summary["trajectory_crps"]),
                "deep_window_prediction": float(curve[82]),
                "predicted_minimum": float(np.min(curve)),
                "predicted_minimum_end_trial": int(np.argmin(curve) + 17),
                "late_prediction": float(np.mean(curve[175:])),
                "commitment_activation_rate": float(np.mean(commitment)),
                "expected_entry_events": float(np.sum(entries)),
            }
        )

    subject_frame = pd.DataFrame(subject_rows)
    s103_row = subject_frame.loc[subject_frame["subject_id"].eq(103)].iloc[0]
    protected = subject_frame.loc[
        subject_frame["trajectory_type"].isin(
            ["smooth rising", "short high-accuracy"]
        )
    ]
    exact_protection = int(
        np.sum(
            np.isclose(protected["delta_choice_nll"], 0.0)
            & np.isclose(protected["delta_trajectory_mae"], 0.0)
            & np.isclose(protected["mean_absolute_curve_change"], 0.0)
        )
    )
    summary = {
        "s103_delta_choice_nll": float(s103_row["delta_choice_nll"]),
        "s103_delta_trajectory_mae": float(s103_row["delta_trajectory_mae"]),
        "s103_delta_trajectory_crps": float(s103_row["delta_trajectory_crps"]),
        "protected_exact_count": exact_protection,
        "protected_total_count": int(len(protected)),
        "selected8_mean_delta_choice_nll": float(
            subject_frame["delta_choice_nll"].mean()
        ),
        "selected8_mean_delta_trajectory_mae": float(
            subject_frame["delta_trajectory_mae"].mean()
        ),
        "selected8_mean_delta_trajectory_crps": float(
            subject_frame["delta_trajectory_crps"].mean()
        ),
        "selected8_mean_activation_rate": float(
            subject_frame["commitment_activation_rate"].mean()
        ),
    }
    datasets = {
        "summary": [summary],
        "s103_trajectory": trajectory_rows,
        "s103_probes": probe_rows,
        "subject_guardrail": subject_rows,
        "subject_nll_delta": delta_rows,
    }
    return datasets, summary


def source_objects() -> list[dict[str, Any]]:
    return [
        {
            "id": "source_summary",
            "label": "Paired selected-eight guardrail summary",
            "path": "reports/s103_rule_commitment_guardrail/summary.csv",
            "query": {
                "engine": "sqlite3",
                "language": "sql",
                "description": "Derived paired-seed metrics loaded into an in-memory summary_input table for the metric cards.",
                "sql": "SELECT s103_delta_choice_nll, s103_delta_trajectory_mae, s103_delta_trajectory_crps, protected_exact_count, protected_total_count, selected8_mean_delta_choice_nll, selected8_mean_delta_trajectory_mae, selected8_mean_delta_trajectory_crps, selected8_mean_activation_rate FROM summary_input",
                "tables_used": [
                    f"results/model_dynamic_continuous/{BASELINE_RESULT}/simulation",
                    f"results/model_dynamic_continuous/{REFINED_RESULT}/simulation",
                ],
                "filters": [
                    "subjects = 103,104,105,108,111,120,124,132",
                    "four particle-filter repeats per condition",
                    "prediction_mode = prior_t",
                    "window_size = 16",
                ],
                "metric_definitions": [
                    "Delta choice NLL = refined marginal choice NLL minus paired baseline marginal choice NLL; negative is better.",
                    "Delta trajectory MAE = refined mean rolling-accuracy MAE minus paired baseline MAE; negative is better.",
                    "Activation rate = particle- and trial-averaged pre-choice rule-commitment probability.",
                ],
            },
        },
        {
            "id": "source_trajectory",
            "label": "S103 paired rolling trajectories",
            "path": "reports/s103_rule_commitment_guardrail/s103_trajectory.csv",
            "query": {
                "engine": "sqlite3",
                "language": "sql",
                "description": "CSV loaded into an in-memory trajectory_input table for the embedded chart.",
                "sql": "SELECT window_end_trial, human_accuracy, baseline_accuracy, initial_gate_accuracy, refined_gate_accuracy FROM trajectory_input ORDER BY window_end_trial",
                "tables_used": [
                    f"results/model_dynamic_continuous/{BASELINE_RESULT}/simulation/cache/subject_103_raw_runs.gz",
                    f"results/model_dynamic_continuous/{INITIAL_RESULT}/simulation/cache/subject_103_raw_runs.gz",
                    f"results/model_dynamic_continuous/{REFINED_RESULT}/simulation/cache/subject_103_raw_runs.gz",
                ],
                "filters": [
                    "subject_id = 103",
                    "common simulation-point seed across conditions",
                    "four repeats and 32 particles",
                    "rolling windows start at trial 2 and contain 16 trials",
                ],
                "metric_definitions": [
                    "Human accuracy is observed exact-category accuracy in each 16-trial window.",
                    "Model accuracy is the repeat-mean predicted probability of the correct category in each window.",
                ],
            },
        },
        {
            "id": "source_probes",
            "label": "S103 paired mechanism probes",
            "path": "reports/s103_rule_commitment_guardrail/s103_probe_metrics.csv",
            "query": {
                "engine": "sqlite3",
                "language": "sql",
                "description": "Exact paired S103 scores and activation diagnostics.",
                "sql": "SELECT * FROM s103_probe_input ORDER BY \"order\"",
                "tables_used": [
                    f"results/model_dynamic_continuous/{BASELINE_RESULT}/simulation/subjects/subject_103.json",
                    f"results/model_dynamic_continuous/{INITIAL_RESULT}/simulation/subjects/subject_103.json",
                    f"results/model_dynamic_continuous/{REFINED_RESULT}/simulation/subjects/subject_103.json",
                ],
                "filters": ["subject_id = 103", "four paired repeats"],
            },
        },
        {
            "id": "source_guardrail",
            "label": "Selected-eight subject guardrails",
            "path": "reports/s103_rule_commitment_guardrail/subject_guardrail.csv",
            "query": {
                "engine": "sqlite3",
                "language": "sql",
                "description": "Exact per-subject paired fit changes, behavioral trajectory labels, and activation diagnostics.",
                "sql": "SELECT subject_id, trajectory_type, trial_count, observed_valley_strength, delta_choice_nll, delta_trajectory_mae, commitment_activation_rate, mean_entry_events, mean_absolute_curve_change, largest_stable_downshift FROM guardrail_input ORDER BY subject_id",
                "tables_used": [
                    "data/processed/Task2_processed.csv",
                    f"results/model_dynamic_continuous/{BASELINE_RESULT}/simulation",
                    f"results/model_dynamic_continuous/{REFINED_RESULT}/simulation",
                ],
                "filters": [
                    "condition = 1",
                    "selected-eight structural pilot",
                    "paired simulation-point seed within subject",
                    "four repeats and 32 particles",
                ],
                "metric_definitions": [
                    "Observed valley strength is the smaller of the two 16-window flank-to-valley drops after the early acquisition phase.",
                    "Smooth rising requires stable-phase rolling accuracy >= 0.50 and localized valley strength <= 0.125.",
                    "Largest stable downshift is the minimum pointwise refined-minus-baseline predicted rolling accuracy after curve index 47.",
                ],
            },
        },
        {
            "id": "source_delta_chart",
            "label": "Selected-eight choice-NLL deltas",
            "path": "reports/s103_rule_commitment_guardrail/subject_nll_delta.csv",
            "query": {
                "engine": "sqlite3",
                "language": "sql",
                "description": "One paired choice-NLL delta per subject for the embedded comparison chart.",
                "sql": "SELECT subject_id, trajectory_type, delta_choice_nll, delta_trajectory_mae, commitment_activation_rate, trial_count FROM nll_delta_input ORDER BY subject_id",
                "tables_used": [
                    "reports/s103_rule_commitment_guardrail/subject_guardrail.csv"
                ],
                "filters": ["eight selected condition-1 subjects"],
            },
        },
    ]


def build_artifact(datasets: dict[str, list[dict[str, Any]]], generated_at: str) -> dict[str, Any]:
    sources = source_objects()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "S103 深谷机制：收益与跨阶段守门",
            "description": "19-rule history-only commitment 的 paired-seed S103 与 selected-eight 守门评估。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "s103_nll_card",
                    "description": "S103 refined commitment 相对同随机种子 baseline 的 marginal choice NLL 变化。",
                    "dataset": "summary",
                    "sourceId": "source_summary",
                    "metrics": [
                        {
                            "label": "S103 choice NLL Δ",
                            "field": "s103_delta_choice_nll",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "s103_mae_card",
                    "description": "S103 16-trial rolling-accuracy trajectory MAE 的 paired change。",
                    "dataset": "summary",
                    "sourceId": "source_summary",
                    "metrics": [
                        {
                            "label": "S103 trajectory MAE Δ",
                            "field": "s103_delta_trajectory_mae",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "protected_card",
                    "description": "严格平稳上升 S124 与短程高准确 S105 的 NLL、MAE 和预测曲线均逐点不变。",
                    "dataset": "summary",
                    "sourceId": "source_summary",
                    "metrics": [
                        {
                            "label": "Protected controls unchanged",
                            "field": "protected_exact_count",
                            "format": "number",
                        },
                        {
                            "label": "Controls evaluated",
                            "field": "protected_total_count",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "group_nll_card",
                    "description": "八名被试 refined-minus-baseline choice NLL 的未加权平均；正值表示整体恶化。",
                    "dataset": "summary",
                    "sourceId": "source_summary",
                    "metrics": [
                        {
                            "label": "Selected-eight mean NLL Δ",
                            "field": "selected8_mean_delta_choice_nll",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "s103_trajectory_chart",
                    "title": "S103 16-trial rolling accuracy",
                    "subtitle": "Common-random-number baseline, initial gate, and refined gate; four PF repeats per condition",
                    "type": "line",
                    "intent": "comparison",
                    "question": "Does conservative commitment deepen the S103 valley without the early false trigger of the initial gate?",
                    "rationale": "The 240 rolling windows reveal onset, minimum timing, recovery, and late-platform effects.",
                    "dataset": "s103_trajectory",
                    "sourceId": "source_trajectory",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "window_end_trial", "type": "quantitative"},
                        "y": {
                            "fields": [
                                "human_accuracy",
                                "baseline_accuracy",
                                "initial_gate_accuracy",
                                "refined_gate_accuracy",
                            ],
                            "type": "quantitative",
                        },
                    },
                    "xAxisTitle": "16-trial window end",
                    "yAxisTitle": "Accuracy / expected accuracy",
                    "layout": "full",
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 0.5,
                            "label": "Chance",
                            "color": "neutral",
                            "lineStyle": "dashed",
                        }
                    ],
                    "surface": {
                        "surface": "export",
                        "interactiveLegend": True,
                        "showControls": False,
                        "viewMode": "visualization",
                    },
                    "palette": {"kind": "categorical", "name": "commitment-comparison"},
                },
                {
                    "id": "subject_nll_chart",
                    "title": "Refined commitment choice-NLL change by subject",
                    "subtitle": "Refined minus paired baseline; negative values improve one-step choice prediction",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "Is the S103 improvement shared across the selected-eight pilot?",
                    "rationale": "A zero-referenced subject bar chart makes heterogeneous benefits and harms explicit.",
                    "dataset": "subject_nll_delta",
                    "sourceId": "source_delta_chart",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {
                            "field": "subject_id",
                            "type": "nominal",
                            "label": "Subject",
                        },
                        "y": {
                            "field": "delta_choice_nll",
                            "type": "quantitative",
                            "label": "Choice NLL Δ",
                            "format": "number",
                        },
                        "tooltip": [
                            {
                                "field": "trajectory_type",
                                "type": "nominal",
                                "label": "Observed trajectory",
                            },
                            {
                                "field": "commitment_activation_rate",
                                "type": "quantitative",
                                "label": "Activation rate",
                                "format": "percent",
                            },
                        ],
                    },
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 0.0,
                            "label": "No change",
                            "color": "neutral",
                            "lineStyle": "solid",
                        }
                    ],
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "s103_probe_table",
                    "title": "S103 paired mechanism probes",
                    "subtitle": "Four PF repeats per condition; lower NLL, MAE, and CRPS are better",
                    "dataset": "s103_probes",
                    "sourceId": "source_probes",
                    "defaultSort": {"field": "order", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "order", "label": "#", "format": "number"},
                        {"field": "probe", "label": "Probe", "type": "text"},
                        {"field": "choice_nll", "label": "Choice NLL", "format": "number"},
                        {"field": "trajectory_mae", "label": "Trajectory MAE", "format": "number"},
                        {"field": "trajectory_crps", "label": "Trajectory CRPS", "format": "number"},
                        {"field": "deep_window_prediction", "label": "At trial 99", "format": "percent"},
                        {"field": "predicted_minimum", "label": "Pred. minimum", "format": "percent"},
                        {"field": "predicted_minimum_end_trial", "label": "Min. end trial", "format": "number"},
                        {"field": "commitment_activation_rate", "label": "Activation", "format": "percent"},
                        {"field": "expected_entry_events", "label": "Expected entries", "format": "number"},
                    ],
                },
                {
                    "id": "guardrail_table",
                    "title": "Selected-eight paired guardrail detail",
                    "subtitle": "One row per subject; negative deltas improve fit and zero indicates exact baseline equivalence",
                    "dataset": "subject_guardrail",
                    "sourceId": "source_guardrail",
                    "defaultSort": {"field": "subject_id", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "subject_id", "label": "Subject", "format": "number"},
                        {"field": "trajectory_type", "label": "Observed type", "type": "text"},
                        {"field": "trial_count", "label": "Trials", "format": "number"},
                        {"field": "observed_valley_strength", "label": "Valley strength", "format": "number"},
                        {"field": "delta_choice_nll", "label": "NLL Δ", "format": "number"},
                        {"field": "delta_trajectory_mae", "label": "MAE Δ", "format": "number"},
                        {"field": "delta_trajectory_crps", "label": "CRPS Δ", "format": "number"},
                        {"field": "commitment_activation_rate", "label": "Activation", "format": "percent"},
                        {"field": "mean_entry_events", "label": "Mean entries", "format": "number"},
                        {"field": "mean_absolute_curve_change", "label": "Mean |curve Δ|", "format": "number"},
                        {"field": "largest_stable_downshift", "label": "Largest downshift", "format": "number"},
                    ],
                },
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# S103 深谷机制：收益与跨阶段守门",
                    "layout": "full",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "source_summary",
                    "body": "## 技术结论\n\n- **保留 refined rule commitment 作为默认关闭的实验模型，不建议全局默认开启。** 它在 S103 上把 paired choice NLL 改善 0.0246、trajectory MAE 改善 0.0108，并消除了 initial gate 在 trial 34 的早期误触发。\n- **对明确无深谷的守门被试没有可见副作用。** 平稳上升 S124 与短程高准确 S105 的 activation、NLL、MAE 和整条预测曲线均与 baseline 完全相同。\n- **跨被试证据仍未通过采用门槛。** selected-eight 的平均 trajectory MAE 小幅改善 0.0010，但平均 choice NLL 恶化 0.0074；收益集中在 S103，S108、S111、S120 和 S132 的 NLL 均恶化。\n- **因此当前结论是“机制有局部解释力，但尚无普适性”。** 下一步应预先声明候选模型并做 held-out/LOSO 比较，而不是继续围绕 S103 调阈值。",
                    "layout": "full",
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "s103_nll_card",
                        "s103_mae_card",
                        "protected_card",
                        "group_nll_card",
                    ],
                },
                {
                    "id": "s103_finding",
                    "type": "markdown",
                    "sourceId": "source_trajectory",
                    "body": "## 保守门控保留了 S103 深谷效应，并去除了早期误触发\n\nInitial gate 能把预测谷值压得更低，但会在真实深谷之前过早进入 commitment，并拖累恢复段。Refined gate 新增历史 peak mastery、16-trial 最小证据、0.80 候选相容度、0.10 runner-up margin，以及 compatibility collapse release。结果是 entry 集中到真实深谷内，S103 的 NLL/MAE/CRPS 仍优于 baseline，但预测谷底仍在 window end 109，晚于真实谷底 99。\n\n图中应把 refined 读作保守折中：它减少了假阳性和恢复拖尾，同时也牺牲了部分谷深，而不是已经完整复现真实 0.25 的深谷。",
                    "layout": "full",
                },
                {"id": "trajectory", "type": "chart", "chartId": "s103_trajectory_chart", "layout": "full"},
                {
                    "id": "s103_table_note",
                    "type": "markdown",
                    "sourceId": "source_probes",
                    "body": "## S103 的改善不是由随机种子差异造成\n\n三个条件共享同一 subject-specific simulation-point seed，每个条件使用相同四条 trajectory seeds。Commitment 关闭时，confidence gain 的变化曾经被验证为概率与曲线逐点完全相同；因此这里的差异来自 commitment 状态和其 gated confidence，而不是 Monte Carlo 抽样错位。",
                    "layout": "full",
                },
                {"id": "s103_probe_detail", "type": "table", "tableId": "s103_probe_table", "layout": "full"},
                {
                    "id": "guardrail_definition",
                    "type": "markdown",
                    "sourceId": "source_guardrail",
                    "body": "## 平稳阶段守门通过，但跨被试采用门槛没有通过\n\n守门不只检查总体分数，还检查 commitment activation、整条滚动曲线的平均绝对变化，以及早期学习之后的最大局部下拉。严格平稳上升 S124 和短程高准确 S105 均为零激活、零 NLL/MAE 变化、零点对点曲线变化，说明新 mastery gate 的确阻止了这两类不该进入的轨迹。\n\n但“有深谷”不等于“该深谷由同一错误规则 commitment 产生”。S108、S111 和 S132 仍有多次触发，其 NLL 均恶化；S120 也恶化。这表明当前 history-only coherence gate 对错误簇仍不够特异，不能把 S103 的成功外推为通用机制。",
                    "layout": "full",
                },
                {"id": "subject_nll", "type": "chart", "chartId": "subject_nll_chart", "layout": "full"},
                {
                    "id": "guardrail_detail_note",
                    "type": "markdown",
                    "sourceId": "source_guardrail",
                    "body": "## 精确逐被试结果支持“保留变体、拒绝默认开启”\n\n八人平均 choice NLL Δ 为 +0.0074，而 trajectory MAE Δ 为 −0.0010。两者方向不一致并不矛盾：机制能在某些局部窗口制造更像行为曲线的低谷，却同时降低其他 trial 的 one-step choice probability。对模型选择而言，应优先把 held-out choice NLL/Brier 作为主标准，trajectory CRPS/MAE 作为形状守门，而不能只凭曲线视觉改善采用机制。",
                    "layout": "full",
                },
                {"id": "guardrail_detail", "type": "table", "tableId": "guardrail_table", "layout": "full"},
                {
                    "id": "model_specification",
                    "type": "markdown",
                    "body": "## 模型规格：history-only、semi-Markov、默认关闭\n\n每个 particle 维护完整 19-rule 空间的指数衰减 choice compatibility、当前 overt executed rule、commitment age、disconfirmation、cooldown 和历史 peak mastery。Entry 要求：历史证据数、failure pressure、peak mastery、最佳候选绝对相容度及相对 runner-up margin 同时过阈值。Entry 后候选被保证安装到 workspace 并成为 executed rule；达到 minimum dwell 后，累计错误或候选相容度跌破 hold threshold 均可释放。\n\n所有 entry/hold/release 信号只使用 trial t−1 及更早的选择和反馈。`rule_commitment_confidence_gain` 只在 active commitment 时做对称 precision 变换，不读取正确答案。机制与旧 `misconception_capture` 互斥，配置默认关闭，因此现有模型在未显式启用时行为不变。",
                    "layout": "full",
                },
                {
                    "id": "methods",
                    "type": "markdown",
                    "sourceId": "source_guardrail",
                    "body": "## 范围、数据与比较方法\n\n分析使用 Cond1 selected-eight 结构探针（S103、104、105、108、111、120、124、132），每名被试保留既有 capacity、readout power 和 candidate seed。Baseline 与 refined 每人各运行 4 次、每次 32 particles，并显式配对 simulation-point seed；评价为完整序列 `prior_t` one-step predictions。\n\n行为类型只用于守门解释，不参与模型触发：16-trial rolling curve 去除早期 acquisition 后，localized valley strength 定义为谷点相对前后各16个窗口的较小落差。Smooth rising 要求 stable-phase minimum ≥0.50 且 valley strength ≤0.125。主指标是 marginal choice NLL；trajectory mean MAE 与 CRPS、activation 和点对点曲线变化是辅助守门指标。",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 局限、稳健性与不能声称的结论\n\n- Selected-eight 是既有结构探针，不是从全部 Cond1 被试随机抽取；只有一名严格 smooth-rising 长序列被试，因此“平稳被试无伤害”目前是强个案守门，不是群体等效性结论。\n- 每个条件只有 4 repeats × 32 particles；common random numbers 提高了条件差异精度，但不能替代高粒子或 exact-filter 复核。\n- 阈值是在观察 S103 后设计，S103 改善属于开发集结果；不能作为 held-out 证据。\n- 完整序列 NLL 不是 held-out generalization，且当前阈值未重新拟合；本报告只评价固定参数的结构性反事实。\n- Refined gate 仍把 S103 预测谷底放在 trial 109，而非真实 trial 99，也未把期望准确率压到 0.25。",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 建议下一步：停止 S103 调参，做预先声明的模型比较\n\n1. **保持默认关闭。** 将 refined commitment 作为与 baseline 并列的候选模型，而不是对所有 condition 或所有被试自动启用。\n2. **扩大无深谷守门样本。** 在全部 Cond1 被试上预先定义 smooth/no-valley cohort；首先要求 activation 接近零、NLL 非劣且不制造新的低于 chance 窗口。\n3. **做 LOSO 或 forward-held-out 比较。** 所有门槛仅在训练被试/前段选择，在留出被试/后段用 choice NLL、Brier、trajectory CRPS 和 false-valley rate 判定。\n4. **把机制身份变成待检验假设。** 若其他深谷被试无法共享固定门槛，应比较 attention/subspace state、rule commitment 和纯 history kernel，而不是继续放宽 commitment entry。\n5. **提高 inference ceiling。** 对少数代表被试用更多 particles 或 exact/Rao–Blackwellized executed-rule filtering，确认跨被试失败不是粒子贫化。",
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## 仍需回答的问题\n\n- 无深谷 cohort 中，零触发能否在更多长序列被试上复现？\n- S108/S111/S132 的触发为何恶化 NLL：候选规则身份不稳定、entry 太晚，还是 confidence 在错误窗口外过强？\n- 用 held-out choice sequence 估计的错误规则 episode precision/recall 是否优于简单的近期错误率？\n- Cond2/Cond3 的任务结构是否允许同一 commitment 定义，还是应在没有直接证据时保持机制关闭？",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": json_safe(datasets),
        },
        "sources": sources,
    }


def main() -> None:
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing report directory: {REPORT_DIR}"
        )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    datasets, _ = build_datasets()
    frames = {
        "summary.csv": pd.DataFrame(datasets["summary"]),
        "s103_trajectory.csv": pd.DataFrame(datasets["s103_trajectory"]),
        "s103_probe_metrics.csv": pd.DataFrame(datasets["s103_probes"]),
        "subject_guardrail.csv": pd.DataFrame(datasets["subject_guardrail"]),
        "subject_nll_delta.csv": pd.DataFrame(datasets["subject_nll_delta"]),
    }
    for filename, frame in frames.items():
        frame.to_csv(REPORT_DIR / filename, index=False)

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact = build_artifact(datasets, generated_at)
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "source_notes.md").write_text(
        """# Source and chart notes

Audience: technical. Delivery mode: portable HTML because an MCP report renderer and full Sites lifecycle are unavailable in this runtime.

Required structure mapping: title; technical conclusion; evidence on S103 and cross-subject guardrails; scope and metric definitions; model specification; methods; limitations/robustness; recommendations; further questions. No required section was omitted.

Chart map:

- S103 effect: highlighted multi-series line, 240 rolling windows, human/baseline/initial/refined, categorical palette plus line identity, supports onset/depth/timing/recovery comparison.
- Cross-subject guardrail: zero-referenced single-series bar, eight subject-level paired NLL deltas, supports heterogeneous benefit/harm and the non-adoption decision.

The report deliberately omits inferential confidence intervals: four common-random-number repeats estimate a structural paired contrast but are not an independent subject sample. Exact per-subject values are retained in the audit table.
""",
        encoding="utf-8",
    )
    print(REPORT_DIR)


if __name__ == "__main__":
    main()
