"""Inventory development reports for Model 0923 without fitting or relabeling people."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _boolean(value: str) -> bool:
    if value in {"True", "true", "1", "1.0"}:
        return True
    if value in {"False", "false", "0", "0.0", ""}:
        return False
    raise ValueError(f"invalid saved oral-valid flag: {value!r}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty inventory: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_inventory(
    *, cohort_config: Path, analysis_dir: Path, behavior_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Preserve original report text; representability remains a review decision."""
    cohort = json.loads(cohort_config.read_text())
    subjects = [int(value) for value in cohort["subjects"]]
    if len(set(subjects)) != len(subjects) or not subjects:
        raise ValueError("cohort subjects must be nonempty and unique")
    source_trials = analysis_dir / "trials.csv"
    source_subjects = analysis_dir / "subjects.csv"
    trials = _rows(source_trials)
    subjects_rows = _rows(source_subjects)
    behavior_rows = _rows(behavior_summary)
    summary = {int(row["subject"]): row for row in subjects_rows}
    behavior = {int(row["subject"]): row for row in behavior_rows}
    if len(summary) != len(subjects_rows) or len(behavior) != len(behavior_rows):
        raise ValueError("duplicate subject summary rows")
    if set(summary) != set(subjects) or set(behavior) != set(subjects):
        raise ValueError("cohort and summary subject IDs must agree exactly")
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    keys = set()
    inventory: dict[tuple[int, int, str], dict[str, Any]] = {}
    report_subjects: dict[tuple[int, int, str], set[int]] = defaultdict(set)
    for row in trials:
        sid = int(row["iSub"])
        condition = int(row["condition"])
        if sid not in summary or condition != int(summary[sid]["condition"]):
            raise ValueError("trial subject/condition does not match the cohort")
        key = (sid, row["iSession"], row["iBlock"], row["iTrial"])
        if key in keys:
            raise ValueError(f"duplicate trial key: {key}")
        keys.add(key)
        groups[sid].append(row)
        text = row["text"]
        if not text.strip() or text.strip().lower() == "nan":
            continue
        report_key = (condition, int(row["choice"]), text)
        if report_key not in inventory:
            encoded = json.dumps(report_key, ensure_ascii=False).encode()
            inventory[report_key] = {
                "report_id": sha256(encoded).hexdigest()[:16],
                "condition": condition, "choice": int(row["choice"]), "text": text,
                "occurrences": 0, "encoded_occurrences": 0,
                "subjects": "", "first_subject": sid, "first_trial": row["trial"],
                "catalogue_coverage": "unreviewed", "canonical_structure": "",
                "review_notes": "",
            }
        item = inventory[report_key]
        item["occurrences"] += 1
        item["encoded_occurrences"] += int(_boolean(row["oral_valid"]))
        report_subjects[report_key].add(sid)
    rows = []
    for sid in subjects:
        records = groups[sid]
        if len(records) != int(summary[sid]["n"]):
            raise ValueError(f"subject {sid}: trial count mismatch")
        if [int(row["trial"]) for row in records] != list(range(1, len(records) + 1)):
            raise ValueError(f"subject {sid}: trial order is not preserved")
        shape = float(behavior[sid]["delta_bic"])
        shape_evidence = "step" if shape >= 6 else "trend" if shape <= -6 else "unresolved"
        rows.append({
            "subject": sid, "condition": summary[sid]["condition"],
            "task": summary[sid]["task"], "n_trials": len(records),
            "encoded_reports": sum(_boolean(row["oral_valid"]) for row in records),
            "scored_encoded_reports": sum(_boolean(row["oral_valid"]) and _boolean(row["scored"])
                                          for row in records),
            "unique_report_choice_pairs": len({(row["choice"], row["text"]) for row in records
                                               if row["text"].strip() and row["text"].strip().lower() != "nan"}),
            "first_criterion_trial": behavior[sid]["criterion"],
            "shape_delta_bic": shape, "shape_evidence": shape_evidence,
            "learning_type": "not_assigned", "sample_role": "development",
        })
    reports = []
    for key in sorted(inventory):
        item = inventory[key]
        item["subjects"] = ";".join(map(str, sorted(report_subjects[key])))
        reports.append(item)
    if not reports:
        raise ValueError("no nonempty reports to review")
    if len({item["report_id"] for item in reports}) != len(reports):
        raise ValueError("report identifier collision")
    source_paths = [cohort_config, source_trials, source_subjects, behavior_summary, Path(__file__)]
    manifest = {
        "model_id": "model_0923", "stage": "R0_report_inventory",
        "n_subjects": len(subjects), "n_trials": len(trials),
        "encoded_reports": sum(row["encoded_reports"] for row in rows),
        "scored_encoded_reports": sum(row["scored_encoded_reports"] for row in rows),
        "unique_condition_choice_reports": len(reports),
        "conditions": dict(Counter(int(row["condition"]) for row in rows)),
        "sources": [{"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
                    for path in source_paths],
        "python": platform.python_version(),
        "interpretation": {
            "encoding_valid_is_not_catalogue_coverage": True,
            "learning_types_used_as_model_inputs": False,
            "oral_likelihood_available": False,
            "fit_performed": False,
            "report_timing": "choice_then_report_then_feedback",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "subjects.csv", rows)
    _write_csv(output_dir / "report_review.csv", reports)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "# Model 0923：口述准备清单\n\n"
        f"{len(subjects)} 人，{len(trials)} 试次，{manifest['encoded_reports']} 条现有有效编码；"
        f"其中沿用旧计分 mask 后有 {manifest['scored_encoded_reports']} 条；"
        f"共 {len(reports)} 种 condition × choice × 原话组合。\n\n"
        "`report_review.csv` 保留原话与所选类别，所有目录覆盖结论均为 unreviewed。"
        "现有编码有效不表示规则目录能表达原话；相同文字在不同类别/条件下不合并。\n\n"
        "`subjects.csv` 记录既有达标时间和行为形状证据，不给人强行分三类。"
        "这些人用于机制开发，不是新的独立验证样本；没有重拟合或计算 0923 口述似然。\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-config", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--behavior-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    manifest = prepare_inventory(**vars(build_parser().parse_args()))
    print(json.dumps({key: manifest[key] for key in (
        "n_subjects", "n_trials", "encoded_reports", "scored_encoded_reports", "unique_condition_choice_reports"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
