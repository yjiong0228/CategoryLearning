"""Export one saved evaluation bundle for paper review, without model imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np


DEFINITIONS = {
    "catalogue_prior": "Pre-current-choice particle-weighted pre-feedback belief mass.",
    "catalogue_filtered_prior": "Current-choice-filtered pre-feedback belief mass.",
    "catalogue_posterior": "Current-choice-filtered post-feedback belief mass.",
    "catalogue_predictive_execution_probability": "Pre-current-choice probability of realized rule execution.",
    "catalogue_execution_probability": "Current-choice-filtered executed-rule probability before feedback conditioning.",
    "catalogue_active_probability": "Pre-current-choice probability that the rule is present.",
}
BITS = (np.arange(16)[:, None] >> np.arange(4)) & 1
FEATURE_SETS = "conditional_feature_set_probability"
FEATURES = "conditional_feature_probability"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _probabilities(value: np.ndarray, shape: tuple[int, ...], valid: np.ndarray,
                   name: str, *, normalized: bool = False) -> None:
    if value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric with shape {shape}")
    rows = value[valid]
    if not np.isfinite(rows).all():
        raise ValueError(f"{name}: valid rows must be finite")
    if (rows < 0).any() or (rows > 1 + 1e-9).any():
        raise ValueError(f"{name}: probability outside [0,1]")
    if not np.isnan(value[~valid]).all():
        raise ValueError(f"{name}: invalid rows must be NaN")
    if normalized and not np.allclose(rows.sum(axis=1), 1, atol=1e-9, rtol=0):
        raise ValueError(f"{name}: row sum must equal one")


def load_bundle(source: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify exact input bytes, declared array shapes/dtypes, and export axes."""
    manifest = json.loads((source / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "gm1031-evaluation-inputs-v1":
        raise ValueError("Unsupported evaluation manifest schema_version")
    raw = (source / "evaluation_inputs.npz").read_bytes()
    if _sha(raw) != manifest.get("inputs_sha256"):
        raise ValueError("evaluation_inputs.npz SHA256 missing or mismatched")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    specs = manifest.get("arrays")
    if not isinstance(specs, dict) or set(specs) != set(arrays):
        raise ValueError("manifest arrays must describe every saved array")
    for key, value in arrays.items():
        spec = specs[key]
        if (not isinstance(spec, dict) or spec.get("shape") != list(value.shape)
                or spec.get("dtype") != str(value.dtype)):
            raise ValueError(f"manifest shape/dtype mismatch for {key}")
    required = {"valid", "trial_ids", "session_ids", "catalogue_rule_ids"}
    required.update("conditional_" + key for key in DEFINITIONS)
    if required - arrays.keys():
        raise ValueError(f"Missing catalogue/trial inputs: {sorted(required - arrays.keys())}; re-evaluate with diagnostics")
    valid = arrays["valid"]
    if (valid.ndim != 1 or not len(valid) or valid.dtype.kind not in "biu"
            or not np.isin(valid, [0, 1]).all()):
        raise ValueError("valid must be a nonempty binary mask [T]")
    n = len(valid)
    valid = valid.astype(bool)
    if type(manifest.get("trials")) is not int or manifest["trials"] != n:
        raise ValueError("manifest trials must equal T")
    if (arrays["trial_ids"].shape not in ((n,), (n, 3))
            or arrays["trial_ids"].dtype.kind not in "iu"
            or arrays["session_ids"].shape != (n,)
            or arrays["session_ids"].dtype.kind not in "iu"):
        raise ValueError("trial_ids must have shape [T] or [T,3]; session_ids must have shape [T], both integer")
    order = manifest.get("protocol", {}).get("feature_order")
    if (not isinstance(order, list) or len(order) != 4
            or not all(isinstance(x, str) and x.strip() for x in order)
            or len(set(order)) != 4):
        raise ValueError("protocol.feature_order must contain four unique bodypart names")
    ids = arrays["catalogue_rule_ids"]
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids)
            or (ids < 0).any() or len(np.unique(ids)) != len(ids)):
        raise ValueError("catalogue_rule_ids must be unique nonnegative integer IDs")
    metadata = manifest.get("catalogue_metadata")
    if not isinstance(metadata, list) or len(metadata) != len(ids):
        raise ValueError("catalogue_metadata must describe each rule column")
    for rule, item in zip(ids, metadata):
        if (not isinstance(item, dict) or item.get("index") != int(rule)
                or item.get("label") != f"H{rule}"):
            raise ValueError("catalogue_metadata does not match stable rule IDs")
    for key in DEFINITIONS:
        name = "conditional_" + key
        _probabilities(arrays[name], (n, len(ids)), valid, name)
        if key != "catalogue_active_probability" and (arrays[name][valid].sum(axis=1) > 1 + 1e-9).any():
            raise ValueError(f"{name}: catalogue row sum exceeds one")
    present = {FEATURE_SETS, FEATURES} & arrays.keys()
    if present and len(present) != 2:
        raise ValueError("Incomplete conditional feature extension")
    if present:
        _probabilities(arrays[FEATURE_SETS], (n, 16), valid, FEATURE_SETS, normalized=True)
        _probabilities(arrays[FEATURES], (n, 4), valid, FEATURES)
        if not np.allclose(arrays[FEATURE_SETS][valid] @ BITS, arrays[FEATURES][valid], atol=1e-9, rtol=0):
            raise ValueError("conditional feature marginal disagrees with dimension-set distribution")
    reports = manifest.get("oral_reports", [])
    if not isinstance(reports, list) or len(reports) not in (0, n):
        raise ValueError("oral_reports must be empty or trial-aligned with T entries")
    return arrays, manifest


def _oral_scores(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(arrays["valid"])
    reports = manifest.get("oral_reports", []) or [None] * n
    oral = np.full((n, 4), np.nan)
    exact, jaccard = np.full(n, np.nan), np.full(n, np.nan)
    for t, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        text = report.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            values = np.array([float(report[f"feature{i}_use"]) for i in range(1, 5)])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isin(values, [0, 1]).all():
            continue
        oral[t] = values
        if not arrays["valid"][t] or FEATURE_SETS not in arrays:
            continue
        code = int(values @ (1 << np.arange(4)))
        distribution = arrays[FEATURE_SETS][t]
        exact[t] = distribution[code]
        intersection = (BITS & values.astype(int)).sum(axis=1)
        union = (BITS | values.astype(int)).sum(axis=1)
        # Identical empty dimension sets have Jaccard 1 by convention.
        similarity = np.divide(intersection, union, out=np.ones(16), where=union != 0)
        jaccard[t] = distribution @ similarity
    return oral, exact, jaccard


def _csv(path: Path, columns: list[str], records: Any) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                             for k, v in row.items()})


def _trial(arrays: dict[str, np.ndarray], manifest: dict[str, Any], t: int) -> dict[str, Any]:
    return dict(subject_id=manifest.get("subject_id"), condition=manifest.get("condition"),
                trial_index=t, trial_id=json.dumps(arrays["trial_ids"][t].tolist()),
                session_id=int(arrays["session_ids"][t]), valid=int(arrays["valid"][t]))


def _plots(output: Path, arrays: dict[str, np.ndarray], order: list[str],
           oral: np.ndarray, selected: list[int], fourcat: bool) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(arrays["valid"]))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for ax, key in zip(axes.flat, DEFINITIONS):
        for rule in selected:
            column = int(np.flatnonzero(arrays["catalogue_rule_ids"] == rule)[0])
            label = f"H{rule}" + (" (conventional reference)" if fourcat and rule == 42 else "")
            ax.plot(x, arrays["conditional_" + key][:, column], label=label)
        ax.set(title=key.removeprefix("catalogue_").replace("_", " "), ylim=(0, 1), ylabel="Probability / mass")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Trial index (zero-based)")
    fig.suptitle("Observed-history conditional rule diagnostics")
    fig.tight_layout()
    try:
        fig.savefig(output / "rule_probabilities.png", dpi=160)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for d, ax in enumerate(axes):
        if FEATURES in arrays:
            ax.plot(x, arrays[FEATURES][:, d], color="tab:blue", label="Current-choice-filtered execution")
        else:
            ax.text(.5, .5, "Feature diagnostics not available in saved bundle", transform=ax.transAxes, ha="center")
        eligible = np.isfinite(oral[:, d]) & arrays["valid"].astype(bool)
        ax.scatter(x[eligible], oral[eligible, d], marker="x", color="tab:red", label="Oral dimension use", zorder=3)
        ax.set(ylabel=order[d], ylim=(-.08, 1.08))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Trial index (zero-based)")
    fig.suptitle("Executed-content dimension usage; oral reports provisional")
    fig.tight_layout()
    try:
        fig.savefig(output / "feature_usage.png", dpi=160)
    finally:
        plt.close(fig)
    return matplotlib.__version__


def export_bundle(source: Path, output: Path, *, window: int = 32) -> Path:
    """Write new CSV/PNG outputs in a separate directory after validation."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the input tree")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    arrays, manifest = load_bundle(source)
    oral, exact, jaccard = _oral_scores(arrays, manifest)
    order = manifest["protocol"]["feature_order"]
    n, ids = len(arrays["valid"]), arrays["catalogue_rule_ids"]
    selected = [rule for rule in [0, 4, 6] + ([42] if manifest.get("n_categories") == 4 else []) if rule in ids]
    output.mkdir(parents=True, exist_ok=False)
    base_columns = list(_trial(arrays, manifest, 0))
    rule_columns = ["rule_id", "rule_label", "family", "program", "feature_order"]

    def rule_info(h: int) -> dict[str, Any]:
        item = manifest["catalogue_metadata"][h]
        return dict(rule_id=int(ids[h]), rule_label=f"H{ids[h]}", family=item.get("family", ""),
                    program=json.dumps(item.get("program", {}), ensure_ascii=True), feature_order=json.dumps(order))

    def trial_records():
        for t in range(n):
            for h in range(len(ids)):
                yield {**_trial(arrays, manifest, t), **rule_info(h),
                       **{key: arrays["conditional_" + key][t, h] for key in DEFINITIONS}}

    _csv(output / "rule_trials.csv", base_columns + rule_columns + list(DEFINITIONS), trial_records())

    def window_records():
        windows = [("whole_run", 0, n)] + [("fixed_window", start, min(n, start + window)) for start in range(0, n, window)]
        for scope, start, stop in windows:
            live = arrays["valid"][start:stop].astype(bool)
            for h in range(len(ids)):
                yield dict(subject_id=manifest.get("subject_id"), condition=manifest.get("condition"),
                           scope=scope, start_index=start, stop_index=stop, n_trials=stop-start,
                           n_valid=int(live.sum()), **rule_info(h),
                           **{key: float(arrays["conditional_" + key][start:stop, h][live].mean()) if live.any() else np.nan
                              for key in DEFINITIONS})

    _csv(output / "rule_windows.csv", ["subject_id", "condition", "scope", "start_index", "stop_index", "n_trials", "n_valid"]
         + rule_columns + list(DEFINITIONS), window_records())

    def feature_records():
        for t in range(n):
            record = _trial(arrays, manifest, t)
            for d, name in enumerate(order):
                record[name + "_probability"] = arrays[FEATURES][t, d] if FEATURES in arrays else np.nan
                record[name + "_oral_use"] = oral[t, d]
            record.update({f"dimension_set_{code}_probability": arrays[FEATURE_SETS][t, code] if FEATURE_SETS in arrays else np.nan
                           for code in range(16)})
            record.update(oral_report_valid=int(np.isfinite(oral[t]).all()), oral_scored=int(np.isfinite(exact[t])),
                          oral_exact_set_probability=exact[t], oral_expected_discrete_dimension_set_jaccard=jaccard[t])
            yield record

    _csv(output / "feature_trials.csv", list(next(feature_records())), feature_records())
    matplotlib_version = _plots(output, arrays, order, oral, selected, manifest.get("n_categories") == 4)
    summary = dict(schema_version="gm1031-paper-tables-v1", subject_id=manifest.get("subject_id"),
                   condition=manifest.get("condition"), trials=n, window=window,
                   window_definition="Whole run plus nonoverlapping fixed trial-index windows [start,stop); final window may be shorter. Invalid trials excluded; gaps retained.",
                   feature_order=order, plotted_rule_ids=selected,
                   missing_requested_rule_ids=[i for i in [0, 4, 6] + ([42] if manifest.get("n_categories") == 4 else []) if i not in ids],
                   rule_definitions=DEFINITIONS, catalogue_axis=manifest.get("catalogue_axis"),
                   normalization="Original catalogue mass retained; templates/uncommitted have separate mass.",
                   feature_status="available" if FEATURES in arrays else "not_available",
                   feature_definition="Current-choice-filtered executed-content dimension-set distribution before feedback conditioning; bit d names protocol.feature_order[d]; marginals are set probabilities times bit indicators.",
                   oral=dict(status="provisional_external_comparison", used_for_fit=False,
                             scored_trials=int(np.isfinite(exact).sum()),
                             exact_definition="Probability mass on the reported binary dimension set; content compatibility, not probability of generating a report.",
                             jaccard_definition="Expected discrete dimension-set Jaccard |model intersect oral| / |model union oral|; not geometric region IoU. Empty/empty = 1.",
                             missing_definition="Report availability requires nonblank string text, including for all-zero flags. Missing/null/blank text and nonbinary flags are excluded from oral overlays and scores. Nonblank text with four zero flags defines an explicit empty coded set. Invalid trials are unscored (blank CSV cells)."),
                   provenance=dict(inputs_sha256=manifest["inputs_sha256"],
                                   manifest_sha256=_sha((source / "evaluation_manifest.json").read_bytes()),
                                   exporter_sha256=_sha(Path(__file__).read_bytes()),
                                   python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib_version))
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True, allow_nan=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One subject's saved evaluation bundle")
    parser.add_argument("--output", type=Path, required=True, help="New directory separate from the input tree")
    parser.add_argument("--window", type=int, default=32, help="Fixed nonoverlapping window length (default: 32)")
    args = parser.parse_args()
    try:
        result = export_bundle(args.input, args.output, window=args.window)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Export failed: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
