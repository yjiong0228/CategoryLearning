#!/usr/bin/env python3
"""Generate Task2 oral-report diagnostics and fidelity summaries."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.oral_coding import (
    FEATURE_NAME_TO_PART,
    PARTS,
    PART_TO_DIM,
    FidelityAnalyzer,
    Recording_Processor_Center,
    Recording_Processor_Region,
    normalize_text,
    parse_literal,
)


PART_TO_CANONICAL_DIM = PART_TO_DIM

NONSTANDARD_STYLE_TAGS = {
    "equality",
    "ranking",
    "body_ref",
    "group_sum",
    "count_abstract",
    "negation",
    "meta",
    "other",
}


def is_empty_region(value: Any) -> bool:
    parsed = parse_literal(value)
    return not isinstance(parsed, list) or len(parsed) == 0


def feature_order_indices(row: pd.Series) -> list[int]:
    return [
        PART_TO_CANONICAL_DIM[FEATURE_NAME_TO_PART[str(row[f"feature{idx}_name"]).strip()]]
        for idx in range(1, 5)
    ]


def clean_scalar(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def clean_vector(values: Any) -> list[float | None]:
    parsed = parse_literal(values)
    if not isinstance(parsed, list):
        return []
    return [clean_scalar(value) for value in parsed]


def reorder_center_to_feature_order(values: Any, row: pd.Series) -> list[float | None]:
    parsed = parse_literal(values)
    if not isinstance(parsed, list) or len(parsed) < len(PARTS):
        return []
    return [clean_scalar(parsed[idx]) for idx in feature_order_indices(row)]


def reorder_region_to_feature_order(values: Any, row: pd.Series) -> list[list[float | None]]:
    parsed = parse_literal(values)
    if not isinstance(parsed, list):
        return []
    order = feature_order_indices(row)
    out = []
    for constraint in parsed:
        if not isinstance(constraint, list) or len(constraint) < len(PARTS):
            continue
        out.append([clean_scalar(constraint[idx]) for idx in order])
    return out


def stringify_list(items: list[str], limit: int | None = None) -> str:
    if limit is not None:
        items = items[:limit]
    return "; ".join(items)


def build_center_region_intermediates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    center_processor = Recording_Processor_Center()
    region_processor = Recording_Processor_Region()
    center_frames = []
    region_frames = []

    for sub_id, subj in df.groupby("iSub", sort=True):
        rec = subj[["iSession", "iTrial", "text"]].copy().reset_index(drop=True)
        center_df = center_processor.process(rec)
        center_df.insert(0, "iSub", int(sub_id))
        center_frames.append(center_df)

        region_df = region_processor.process(rec)
        region_df.insert(0, "iSub", int(sub_id))
        region_frames.append(region_df)

    center_all = pd.concat(center_frames, ignore_index=True) if center_frames else pd.DataFrame()
    region_all = pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame()
    return center_all, region_all


def build_diagnostics(
    df: pd.DataFrame,
    center_mid: pd.DataFrame | None = None,
    region_mid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    analyzer = FidelityAnalyzer()
    records = []
    key_cols = ["iSub", "iSession", "iTrial"]
    center_lookup = center_mid.set_index(key_cols).to_dict("index") if center_mid is not None and not center_mid.empty else {}
    region_lookup = region_mid.set_index(key_cols).to_dict("index") if region_mid is not None and not region_mid.empty else {}
    for _, row in df.iterrows():
        result = analyzer.analyze_row(row)
        key = (int(row["iSub"]), int(row["iSession"]), int(row["iTrial"]))
        center_info = center_lookup.get(key, {})
        region_info = region_lookup.get(key, {})
        oral_center = reorder_center_to_feature_order(center_info.get("all", row.get("oral_center")), row)
        oral_A = reorder_region_to_feature_order(region_info.get("A", row.get("oral_A")), row)
        oral_b = clean_vector(region_info.get("b", row.get("oral_b")))
        records.append(
            {
                "iSub": int(row["iSub"]),
                "iSession": int(row["iSession"]),
                "iTrial": int(row["iTrial"]),
                "text": row.get("text"),
                "fidelity": result["fidelity"],
                "fidelity_status": result["status"],
                "n_fidelity_claims": result["n_claims"],
                "style_tags": result["style_tags"],
                "claim_labels": result["claim_labels"],
                "failed_claims": result["failed_claims"],
                "legacy_region_empty": is_empty_region(oral_A),
                "oral_center": oral_center,
                "oral_A": oral_A,
                "oral_b": oral_b,
            }
        )
    return pd.DataFrame(records)


def summarize_subjects(df: pd.DataFrame, diag: pd.DataFrame, region_mid: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    region_lookup = {}
    if not region_mid.empty:
        for sub_id, subj in region_mid.groupby("iSub"):
            unparsed = subj["un_pro"].astype(str).ne("[]").sum() if "un_pro" in subj else np.nan
            encoded = subj["n_constraints"].gt(0).sum() if "n_constraints" in subj else np.nan
            region_lookup[int(sub_id)] = {
                "legacy_region_encoded_rate": encoded / len(subj) if len(subj) else np.nan,
                "legacy_region_unparsed_rate": unparsed / len(subj) if len(subj) else np.nan,
            }

    for sub_id, subj in df.groupby("iSub", sort=True):
        sub_diag = diag[diag["iSub"] == sub_id]
        text_nonempty = subj["text"].notna() & subj["text"].astype(str).str.strip().ne("")
        parseable = sub_diag["fidelity"].notna()
        tags = Counter()
        for tag_list in sub_diag["style_tags"]:
            tags.update(tag_list)
        top_tags = ", ".join(f"{tag}:{count}" for tag, count in tags.most_common(5))
        summaries.append(
            {
                "iSub": int(sub_id),
                "n_trials": int(len(subj)),
                "n_text": int(text_nonempty.sum()),
                "fidelity_parseable_rate": float(parseable.mean()),
                "fidelity_mean": float(sub_diag["fidelity"].mean()) if parseable.any() else np.nan,
                "fidelity_full_rate": float((sub_diag["fidelity"] >= 0.999).mean()),
                "fidelity_low_rate": float((sub_diag["fidelity"] < 0.5).mean()),
                "unsupported_rate": float(sub_diag["fidelity_status"].isin(["unsupported", "meta", "empty"]).mean()),
                "dominant_style_tags": top_tags,
                **region_lookup.get(int(sub_id), {}),
            }
        )
    return pd.DataFrame(summaries)


def unusual_text_table(subj_diag: pd.DataFrame, region_sub: pd.DataFrame) -> pd.DataFrame:
    diag = subj_diag.copy()
    diag["text_clean"] = diag["text"].fillna("").astype(str).str.strip()
    diag = diag[diag["text_clean"].ne("")]

    def nonstandard_tags(row: pd.Series) -> list[str]:
        return [tag for tag in row["style_tags"] if tag in NONSTANDARD_STYLE_TAGS]

    diag["nonstandard_styles"] = diag.apply(nonstandard_tags, axis=1)
    diag = diag[diag["nonstandard_styles"].map(bool)]
    if diag.empty:
        return pd.DataFrame(columns=["text", "count", "nonstandard_styles", "sample_trials"])

    rows = []
    for text, group in diag.groupby("text_clean", sort=False):
        tag_counter = Counter()
        for tags in group["nonstandard_styles"]:
            tag_counter.update(tags)
        sample_trials = ", ".join(
            f"S{int(s)}T{int(t)}" for s, t in group[["iSession", "iTrial"]].head(8).to_numpy()
        )
        rows.append(
            {
                "text": text,
                "count": int(len(group)),
                "nonstandard_styles": ", ".join(name for name, _ in tag_counter.most_common(6)),
                "sample_trials": sample_trials,
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "text"], ascending=[False, True])


def low_fidelity_text_table(subj_diag: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    diag = subj_diag.copy()
    diag["text_clean"] = diag["text"].fillna("").astype(str).str.strip()
    diag = diag[diag["text_clean"].ne("")]
    diag = diag[diag["fidelity"].notna() & (diag["fidelity"] < threshold)]
    if diag.empty:
        return pd.DataFrame(columns=["text", "count", "mean_fidelity", "sample_failed_claims", "sample_trials"])

    rows = []
    for text, group in diag.groupby("text_clean", sort=False):
        failed = []
        for value in group["failed_claims"]:
            parsed = parse_literal(value)
            if isinstance(parsed, list):
                failed.extend(str(item) for item in parsed)
            elif isinstance(value, str) and value not in {"[]", ""}:
                failed.append(value)
        failed_counts = Counter(failed)
        sample_failed = "; ".join(name for name, _ in failed_counts.most_common(4))
        sample_trials = ", ".join(
            f"S{int(s)}T{int(t)}" for s, t in group[["iSession", "iTrial"]].head(8).to_numpy()
        )
        rows.append(
            {
                "text": text,
                "count": int(len(group)),
                "mean_fidelity": group["fidelity"].mean(),
                "sample_failed_claims": sample_failed,
                "sample_trials": sample_trials,
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "mean_fidelity", "text"], ascending=[False, True, True])


def parse_list_field(value: Any) -> list[Any]:
    parsed = parse_literal(value)
    if isinstance(parsed, list):
        return parsed
    return []


def classify_unprocessed_text(text: Any, un_pro_items: list[Any]) -> list[str]:
    joined = "，".join(str(item) for item in un_pro_items) if un_pro_items else normalize_text(text)
    checks = [
        ("meta_or_uncertain", r"选错|不确定|不知道|随便|假设|无关|没什么区别|看不出"),
        ("count_abstract", r"两长两短|三长|一短|两个部位|三个部位|一个部位|几个部位|奇数|偶数|有部位"),
        ("other_reference", r"其他|其余|另外|剩下|其它"),
        ("body_geometry", r"躯体|身体|躯干|上面|下面|下方|上方|之上|之下|高大|身材"),
        ("proportion_or_ratio", r"比例|协调|总和|之和|加起来|合起来|组合"),
        ("global_balance", r"均衡|均匀|匀称|均等|平均|差不多长|差不多$|相当"),
        ("disjoint_inequality", r"不一样|不相同|不同|各不相同|不等于"),
        ("extreme_endpoint", r"达到最长|达到最短|最大长度|最小长度|最长长度|最短长度|极限"),
        ("vague_size", r"比较小|很小|高大|挺高大|身材|体型"),
        ("ordinal_or_secondary", r"其次|第二|第三|第四|次之|第一|最后|然后|再是|接着|第[一二三四1234]"),
    ]
    labels = [label for label, pattern in checks if re.search(pattern, joined)]
    return labels or ["other_unparsed"]


def build_unprocessed_trials(intermediate: pd.DataFrame, diag: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    diag_cols = ["iSub", "iSession", "iTrial", "fidelity", "fidelity_status", "style_tags"]
    diag_small = diag[diag_cols].copy() if set(diag_cols).issubset(diag.columns) else pd.DataFrame()

    for _, row in intermediate.iterrows():
        un_items = parse_list_field(row.get("un_pro"))
        if not un_items:
            continue
        base = {
            "mode": mode,
            "iSub": int(row["iSub"]),
            "iSession": int(row["iSession"]),
            "iTrial": int(row["iTrial"]),
            "text": row.get("text"),
            "un_pro": un_items,
            "un_pro_items": "；".join(str(item) for item in un_items),
            "un_pro_count": len(un_items),
            "unprocessed_category": ", ".join(classify_unprocessed_text(row.get("text"), un_items)),
        }
        if mode == "center":
            base["encoded_all"] = row.get("all")
        else:
            base["n_constraints"] = row.get("n_constraints")
            base["matched_rules"] = row.get("matched_rules")
            base["A"] = row.get("A")
            base["b"] = row.get("b")
        rows.append(base)

    out = pd.DataFrame(rows)
    if out.empty or diag_small.empty:
        return out
    return out.merge(diag_small, on=["iSub", "iSession", "iTrial"], how="left")


def unprocessed_text_summary(unprocessed: pd.DataFrame) -> pd.DataFrame:
    if unprocessed.empty:
        return pd.DataFrame(columns=["text", "count", "un_pro_items", "categories", "sample_trials"])
    frame = unprocessed.copy()
    frame["text_clean"] = frame["text"].fillna("").astype(str).str.strip()
    rows = []
    for text, group in frame.groupby("text_clean", sort=False):
        items = Counter()
        categories = Counter()
        for value in group["un_pro"]:
            parsed = value if isinstance(value, list) else parse_list_field(value)
            items.update(str(item) for item in parsed)
        for value in group["unprocessed_category"]:
            categories.update(str(value).split(", "))
        sample_trials = ", ".join(
            f"S{int(s)}T{int(t)}" for s, t in group[["iSession", "iTrial"]].head(8).to_numpy()
        )
        rows.append(
            {
                "text": text,
                "count": int(len(group)),
                "un_pro_items": "; ".join(item for item, _ in items.most_common(5)),
                "categories": ", ".join(item for item, _ in categories.most_common(5)),
                "sample_trials": sample_trials,
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "text"], ascending=[False, True])


def unprocessed_subject_summary(unprocessed: pd.DataFrame) -> pd.DataFrame:
    if unprocessed.empty:
        return pd.DataFrame(columns=["iSub", "unprocessed_trials", "unique_texts", "top_categories", "top_texts"])
    rows = []
    for sub_id, group in unprocessed.groupby("iSub", sort=True):
        category_counter = Counter()
        for value in group["unprocessed_category"]:
            category_counter.update(str(value).split(", "))
        text_counter = group["text"].fillna("").astype(str).str.strip().value_counts()
        rows.append(
            {
                "iSub": int(sub_id),
                "unprocessed_trials": int(len(group)),
                "unique_texts": int(text_counter.size),
                "top_categories": "; ".join(f"{k}:{v}" for k, v in category_counter.most_common(5)),
                "top_texts": "; ".join(f"{k}:{v}" for k, v in text_counter.head(5).items()),
            }
        )
    return pd.DataFrame(rows)


def generate_unprocessed_report(
    center_unprocessed: pd.DataFrame,
    region_unprocessed: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Task2 未编码文本分析报告")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 本报告只分析 `Task2_oral_center_intermediate.csv` 与 `Task2_oral_region_intermediate.csv` 中 `un_pro` 非空的 trial。")
    lines.append("- `un_pro` 表示当前 parser/encoder 没有正式编码的文本片段；同一 trial 可能已有部分内容被编码。")
    lines.append("- 目前 center 与 region 共享同一个语义 parser，因此二者的 `un_pro` 理论上应高度一致；差异只应来自后续投影策略。")
    lines.append("")

    def overview_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "mode": name,
            "unprocessed_trials": len(frame),
            "subjects": frame["iSub"].nunique() if not frame.empty else 0,
            "unique_texts": frame["text"].fillna("").astype(str).str.strip().nunique() if not frame.empty else 0,
            "top_categories": "; ".join(
                f"{k}:{v}"
                for k, v in Counter(
                    cat
                    for value in frame.get("unprocessed_category", [])
                    for cat in str(value).split(", ")
                ).most_common(8)
            ),
        }

    overview = pd.DataFrame(
        [
            overview_row("center", center_unprocessed),
            overview_row("region", region_unprocessed),
        ]
    )
    lines.append("## 总览")
    lines.append("")
    lines.extend(markdown_table(overview, list(overview.columns)))
    lines.append("")

    key_cols = ["iSub", "iSession", "iTrial", "text", "un_pro_items"]
    identical = False
    if not center_unprocessed.empty and not region_unprocessed.empty:
        center_key = center_unprocessed[key_cols].sort_values(key_cols).reset_index(drop=True)
        region_key = region_unprocessed[key_cols].sort_values(key_cols).reset_index(drop=True)
        identical = center_key.equals(region_key)
    lines.append("## Center 与 Region 差异")
    lines.append("")
    if identical:
        lines.append("本轮 center 与 region 的 `un_pro` trial 清单完全一致，说明残留主要来自共享语义解析层，而不是投影到 center 或 region 的差异。")
    else:
        lines.append("本轮 center 与 region 的 `un_pro` 清单存在差异，需要分别检查投影层。")
    lines.append("")

    for mode, frame in [("center", center_unprocessed), ("region", region_unprocessed)]:
        lines.append(f"## {mode.capitalize()} 未编码分析")
        lines.append("")
        if frame.empty:
            lines.append("无未编码 trial。")
            lines.append("")
            continue

        subject_summary = unprocessed_subject_summary(frame)
        lines.append("### 被试摘要")
        lines.append("")
        lines.extend(markdown_table(subject_summary, ["iSub", "unprocessed_trials", "unique_texts", "top_categories", "top_texts"]))
        lines.append("")

        text_summary = unprocessed_text_summary(frame).head(80)
        lines.append("### 高频未编码文本 Top 80")
        lines.append("")
        lines.extend(markdown_table(text_summary, ["text", "count", "un_pro_items", "categories", "sample_trials"]))
        lines.append("")

        lines.append("### 逐被试未编码文本")
        lines.append("")
        for sub_id, group in frame.groupby("iSub", sort=True):
            lines.append(f"#### S{int(sub_id)}")
            lines.append("")
            sub_summary = unprocessed_text_summary(group)
            lines.extend(markdown_table(sub_summary, ["text", "count", "un_pro_items", "categories", "sample_trials"]))
            lines.append("")

    lines.append("## 编码规则改进建议")
    lines.append("")
    lines.append("1. 本轮已加入 `其他/其余/另外/剩下/剩余` 的补集指代：如 `腿短，其他长` 会把 `其他` 解析为未点名的三个部位；`只有腿很长` 会编码为腿长、其他短。后续需要重点检查残留的 `other_reference` 是否属于更复杂的比较句或计数句。")
    lines.append("2. 本轮已把 `达到最长/最短长度` 简化为绝对 `长/短`：center 分别为 0.75/0.25，region 分别为 `>0.5`/`<0.5`。若之后需要更严格端点，可再把这类规则单独改成 0.9/0.1。")
    lines.append("3. 本轮已加入跨分句排序和 `次之` 逻辑：如 `腿最长，脖子第二，尾巴第三，头最短` 编码为 strict ranking；如 `头最长，脖子次之，腿、尾巴极短` 编码为 `头 > 脖子 > 腿/尾巴`；如 `脖子和尾巴明显长，头最短，腿次之` 编码为 `脖子/尾巴 > 腿 > 头`。并列部位只作为同一层级参与组间比较，不额外加入组内相等或大小关系。")
    lines.append("4. 本轮已加入否定 direct 描述和限定范围最高级：`不长/不是很长/并非很长` 编码为短，`不短/不是很短/并非很短` 编码为长；`不是最短/并非最长` 暂不编码；`脖子是头、尾巴、脖子里最短` 只编码 `脖子 < 头`、`脖子 < 尾巴`，不涉及未出现在限定范围内的腿。")
    lines.append("5. 计数抽象先不要强行编码为单个 `A,b`：例如 `两长两短`、`三个部位很长`、`最长的两个/最短的两个`。这类语义通常是多个区域的并集，单个凸 region 表达不了；若要编码，需要扩展为 multi-region 或结合刺激上下文。")
    lines.append("6. 全局形态词需要人工定义语义：`比较均衡/匀称/协调/高大/体型中等`。其中 `均衡/匀称` 可考虑映射为四维 pairwise equality；`高大` 可考虑映射为多数或全部维度偏长，但这需要你确认。")
    lines.append("7. `比例` 句式要加强 group-sum/ratio parser：如 `头和脖子的比例大于腿和尾巴` 可近似编码为 `head + neck > leg + tail`，但如果被试真的在说比例而非总和，需要另定语义。")
    lines.append("8. 身体几何/方位描述需要任务图像约定：如 `头在躯干之下`、`躯体下方比上面高`。这可能涉及 `body_ori` 或视觉布局，不应只靠四个长度维度猜。")
    lines.append("9. `不一样/各不相同` 是非凸或补集语义：当前单个 `A,b` 难以表达，应标记为 unsupported 或引入 disjunctive region 表示。")
    lines.append("10. `假设X无关`、`选错了`、`不知道` 更像 meta 策略或信心报告，建议继续不编码为 center/region，但在诊断表中保留。")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["无。"]
    lines = []
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.extend([header, sep])
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if math.isnan(val) else f"{val:.3f}")
            else:
                vals.append(str(val).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def generate_markdown_report(
    df: pd.DataFrame,
    diag: pd.DataFrame,
    summary: pd.DataFrame,
    region_mid: pd.DataFrame,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Task2 口头汇报分析报告")
    lines.append("")
    lines.append("## 生成说明")
    lines.append("")
    lines.append("- `fidelity` 是文本语义与当前 trial 的 `feature1~feature4` 是否一致的自动评分，范围为 0 到 1。")
    lines.append("- 有明确可解析断言时，按断言通过比例计算；没有新语义断言但已有旧版 region 编码时，用旧版 region 约束的通过比例兜底。")
    lines.append("- `一样长/差不多/均匀` 按相对接近处理，默认容差为 0.10；严格的 `等于` 默认容差不超过 0.06。")
    lines.append("- `躯干/身体` 在 fidelity 中按 0.5 处理；`3/4躯干` 按 0.375 处理；单独的 `一半` 按 0.5 处理。")
    lines.append("- `两长两短/三个部位很长` 等未点名部位的计数抽象目前只做标记，不用 feature 自动反推部位。")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    overview = pd.DataFrame(
        [
            {
                "被试数": df["iSub"].nunique(),
                "trial数": len(df),
                "非空文本": int(df["text"].notna().sum()),
                "fidelity可评分率": diag["fidelity"].notna().mean(),
                "平均fidelity": diag["fidelity"].mean(),
                "完全忠实率": (diag["fidelity"] >= 0.999).mean(),
            }
        ]
    )
    lines.extend(markdown_table(overview, list(overview.columns)))
    lines.append("")
    lines.append("## 被试摘要表")
    lines.append("")
    display_summary = summary.copy()
    columns = [
        "iSub",
        "n_trials",
        "n_text",
        "fidelity_parseable_rate",
        "fidelity_mean",
        "fidelity_full_rate",
        "fidelity_low_rate",
        "legacy_region_encoded_rate",
        "legacy_region_unparsed_rate",
        "dominant_style_tags",
    ]
    columns = [col for col in columns if col in display_summary.columns]
    lines.extend(markdown_table(display_summary, columns))

    lines.append("")
    lines.append("## 逐被试报告")
    lines.append("")
    region_by_sub = {int(k): v for k, v in region_mid.groupby("iSub")} if not region_mid.empty else {}

    for sub_id, subj in df.groupby("iSub", sort=True):
        sub_id = int(sub_id)
        sub_diag = diag[diag["iSub"] == sub_id]
        sub_summary = summary[summary["iSub"] == sub_id].iloc[0]
        region_sub = region_by_sub.get(sub_id, pd.DataFrame())
        lines.append(f"### S{sub_id}")
        lines.append("")
        lines.append(
            f"- trial 数: {int(sub_summary['n_trials'])}; 非空文本: {int(sub_summary['n_text'])}; "
            f"fidelity 可评分率: {sub_summary['fidelity_parseable_rate']:.3f}; "
            f"平均 fidelity: {sub_summary['fidelity_mean']:.3f}; "
            f"完全忠实率: {sub_summary['fidelity_full_rate']:.3f}; "
            f"低 fidelity 率: {sub_summary['fidelity_low_rate']:.3f}."
        )
        if "legacy_region_encoded_rate" in sub_summary.index:
            lines.append(
                f"- 旧版 region 覆盖率: {sub_summary['legacy_region_encoded_rate']:.3f}; "
                f"旧版 region 有未处理片段率: {sub_summary['legacy_region_unparsed_rate']:.3f}."
            )

        tag_counter = Counter()
        for tags in sub_diag["style_tags"]:
            tag_counter.update(tags)
        tag_df = pd.DataFrame(
            [
                {"style": tag, "count": count, "rate": count / len(sub_diag)}
                for tag, count in tag_counter.most_common()
            ]
        )
        lines.append("")
        lines.append("汇报风格标签：")
        lines.extend(markdown_table(tag_df, ["style", "count", "rate"]))

        text_counts = (
            subj["text"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: s.ne("")]
            .value_counts()
            .head(20)
            .reset_index()
        )
        text_counts.columns = ["text", "count"]
        lines.append("")
        lines.append("典型说法 Top 20：")
        lines.extend(markdown_table(text_counts, ["text", "count"]))

        unusual = unusual_text_table(sub_diag, region_sub)
        lines.append("")
        lines.append("非常规风格对应试次：")
        lines.extend(markdown_table(unusual, ["text", "count", "nonstandard_styles", "sample_trials"]))

        low_fidelity = low_fidelity_text_table(sub_diag)
        lines.append("")
        lines.append("低忠实率对应试次（fidelity < 0.5）：")
        lines.extend(markdown_table(low_fidelity, ["text", "count", "mean_fidelity", "sample_failed_claims", "sample_trials"]))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    processed_out = Path(args.processed_out)
    results_dir = Path(args.results_dir)

    df = pd.read_csv(input_csv)
    results_dir.mkdir(parents=True, exist_ok=True)
    center_mid, region_mid = build_center_region_intermediates(df)
    center_mid.to_csv(results_dir / args.center_intermediate_name, index=False)
    region_mid.to_csv(results_dir / args.region_intermediate_name, index=False)

    diag = build_diagnostics(df, center_mid=center_mid, region_mid=region_mid)
    processed = df.copy()
    processed.insert(processed.columns.get_loc("text"), "fidelity", diag["fidelity"])
    for col in ["oral_center", "oral_A", "oral_b"]:
        if col in processed.columns:
            processed[col] = diag[col]
    processed_out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_out, index=False)

    summary = summarize_subjects(df, diag, region_mid)
    center_unprocessed = build_unprocessed_trials(center_mid, diag, mode="center")
    region_unprocessed = build_unprocessed_trials(region_mid, diag, mode="region")

    diag.to_csv(results_dir / "Task2_oral_trial_diagnostics.csv", index=False)
    summary.to_csv(results_dir / "Task2_oral_subject_summary.csv", index=False)
    center_unprocessed.to_csv(results_dir / "Task2_oral_center_unprocessed_trials.csv", index=False)
    region_unprocessed.to_csv(results_dir / "Task2_oral_region_unprocessed_trials.csv", index=False)
    generate_markdown_report(
        df=df,
        diag=diag,
        summary=summary,
        region_mid=region_mid,
        out_path=results_dir / "Task2_oral_subject_report.md",
    )
    generate_unprocessed_report(
        center_unprocessed=center_unprocessed,
        region_unprocessed=region_unprocessed,
        out_path=results_dir / "Task2_oral_unprocessed_report.md",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="data/processed/Task2_processed.csv")
    parser.add_argument("--processed-out", default="data/processed/Task2_processed_with_fidelity.csv")
    parser.add_argument("--results-dir", default="results/oral_analysis")
    parser.add_argument("--center-intermediate-name", default="Task2_oral_center_intermediate.csv")
    parser.add_argument("--region-intermediate-name", default="Task2_oral_region_intermediate.csv")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
