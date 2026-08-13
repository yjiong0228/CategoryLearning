#!/usr/bin/env python3
"""Audit the Task2 hypothesis space against refreshed oral reports.

This module is an analysis tool, not part of the model execution path.  It
separates three questions that are easy to conflate:

1. Which relational primitives do participants demonstrably verbalize?
2. Which of those primitives are expressible by the current continuous space
   library?
3. Which small additions improve empirical coverage without making the space
   cognitively or computationally unconstrained?

Run from the repository root:

    python -m src.Bayesian_state.hypothesis_space.analysis

The default output directory is ``results/hypothesis_analysis``.  The script
expects ``results/oral_analysis`` to have been regenerated from the current
``data/processed/Task2_processed.csv`` and stops on any key mismatch.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.oral_coding import FEATURE_NAME_TO_PART, SemanticParser, normalize_text

from ..observation_model import ContinuousPartition


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_DIAGNOSTICS = ROOT / "results/oral_analysis/Task2_oral_trial_diagnostics.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results/hypothesis_analysis"
KEY_COLUMNS = ["iSub", "iSession", "iTrial"]


@dataclass(frozen=True)
class PrimitiveSpec:
    """One cognitively interpretable relation used to audit coverage."""

    key: str
    label_en: str
    label_zh: str
    definition: str
    evidence_kind: str


PRIMITIVES: tuple[PrimitiveSpec, ...] = (
    PrimitiveSpec(
        "absolute_side",
        "Absolute side",
        "绝对长/短侧",
        "x_i is above or below the fixed reference 0.5",
        "direct",
    ),
    PrimitiveSpec(
        "center_band",
        "Near center",
        "接近中等/躯干",
        "|x_i - 0.5| <= delta_center",
        "direct",
    ),
    PrimitiveSpec(
        "pairwise_order",
        "Pairwise order",
        "两维大小关系",
        "x_i > x_j or x_i < x_j",
        "direct",
    ),
    PrimitiveSpec(
        "pairwise_near_equal",
        "Pairwise near-equality",
        "两维近似相等",
        "|x_i - x_j| <= delta_equal",
        "direct",
    ),
    PrimitiveSpec(
        "multiway_near_equal",
        "Global balance",
        "三/四维均衡",
        "max(x_S) - min(x_S) <= delta_equal for |S| >= 3",
        "direct",
    ),
    PrimitiveSpec(
        "group_sum_order",
        "Group-sum order",
        "两组和的大小关系",
        "sum(x_A) > sum(x_B) or the reverse",
        "direct",
    ),
    PrimitiveSpec(
        "group_sum_near_equal",
        "Group-sum near-equality",
        "两组和近似相等",
        "|sum(x_A) - sum(x_B)| <= delta_sum",
        "direct",
    ),
    PrimitiveSpec(
        "ranking_extreme",
        "Rank or extreme",
        "排序/最大最小维",
        "rank(x_i) or argmax/argmin over dimensions",
        "direct",
    ),
    PrimitiveSpec(
        "count_cardinality",
        "Count/cardinality",
        "长短维度计数",
        "sum_i 1[x_i > 0.5] satisfies a count or parity condition",
        "direct lexical",
    ),
    PrimitiveSpec(
        "ordinal_degree",
        "Graded intensity",
        "分级的很/较/略长短",
        "one dimension is described with more than a binary intensity level",
        "indirect lexical",
    ),
    PrimitiveSpec(
        "negated_relation",
        "Negated relation",
        "否定关系",
        "a relation is explicitly negated",
        "direct lexical",
    ),
)

PRIMITIVE_BY_KEY = {item.key: item for item in PRIMITIVES}

# These primitives add distinct partition geometry.  ``ordinal_degree`` is
# evidence for an ordinal extension but does not make a current binary-side
# report unrepresentable; ``negated_relation`` usually negates another listed
# primitive.  They are therefore excluded from the strict coverage denominator.
COVERAGE_PRIMITIVES = tuple(
    key
    for key in PRIMITIVE_BY_KEY
    if key not in {"ordinal_degree", "negated_relation"}
)

COUNT_PATTERN = re.compile(
    r"两长两短|三长(?:一短)?|三短(?:一长)?|一长三短|一短三长|"
    r"(?<!其他)(?<!其余)(?:有|只有|其中|共|总共)?(?:一|二|三|1|2|3)个部位"
    r"(?:都(?:很|较|偏)?(?:长|短)|(?:是|比较|很|较|偏)(?:长|短)|(?:长|短)(?!度))|"
    r"奇数|偶数"
)
ORDINAL_DEGREE_PATTERN = re.compile(
    r"(?:非常|特别|极其|很|较|比较|略|稍微|稍|偏|挺)[^，。；]{0,3}(?:长|短|大|小|高|低)|"
    r"(?:长|短|大|小|高|低)[^，。；]{0,2}(?:很多|一些|一点)"
)
GROUP_SUM_CUE_PATTERN = re.compile(
    r"总和|之和|加起来|合起来|组合|相加|加[^，。；]{0,8}加"
)
NEGATED_RELATION_PATTERN = re.compile(
    r"不一样|不相同|不同|各不相同|不相等|不等于|不等长|"
    r"不是|并非|不算|不太|不够|不怎么"
)
GLOBAL_BALANCE_PATTERN = re.compile(
    r"四个(?:部位)?(?:都|长度)?(?:差不多|一样|相近|接近|均匀|均衡|平均)|"
    r"所有(?:部位)?(?:都|长度)?(?:差不多|一样|相近|接近|均匀|均衡)|"
    r"整体(?:比较)?(?:均匀|均衡|匀称)|长度(?:比较)?(?:均匀|均衡|接近)"
)


@dataclass(frozen=True)
class CandidateFamily:
    """One proposed family with a bounded enumeration for four dimensions."""

    arity: str
    candidate_id: str
    label_zh: str
    formal_rule: str
    concrete_rules_d4: int
    primary_primitive: str
    cognitive_predicates: int
    max_decision_depth: int
    max_dimensions: int
    geometry_note: str
    recommendation: str
    rationale: str


CANDIDATE_FAMILIES: tuple[CandidateFamily, ...] = (
    CandidateFamily(
        "binary",
        "B1_pair_similarity_band",
        "成对近似相等带",
        "category 1 iff |x_i-x_j|<=delta; category 2 otherwise",
        6,
        "pairwise_near_equal",
        1,
        1,
        2,
        "The outside category is a union of two halfspaces.",
        "add_core",
        "Directly turns the frequent 'same/similar' relation into a positive-volume category.",
    ),
    CandidateFamily(
        "binary",
        "B2_center_band",
        "中等值带",
        "category 1 iff |x_i-0.5|<=delta_center; category 2 otherwise",
        4,
        "center_band",
        1,
        1,
        1,
        "The outside category has low and high components.",
        "add_core",
        "Represents 'middle/appropriate/about the body' rather than forcing it into low or high.",
    ),
    CandidateFamily(
        "binary",
        "B3_global_balance_band",
        "全局均衡带",
        "category 1 iff max_i(x_i)-min_i(x_i)<=delta_global",
        1,
        "multiway_near_equal",
        1,
        1,
        4,
        "One named relational predicate over all four dimensions.",
        "pilot",
        "Captures 'all four are balanced' without enumerating all six pairwise equalities.",
    ),
    CandidateFamily(
        "binary",
        "B4_group_sum_similarity_band",
        "两组和近似相等带",
        "category 1 iff |(x_i+x_j)-(x_k+x_l)|<=delta_sum",
        3,
        "group_sum_near_equal",
        1,
        1,
        4,
        "The complement has two directional components.",
        "defer",
        "Direct oral support for near-equal group sums is too weak for the bounded core grammar.",
    ),
    CandidateFamily(
        "binary",
        "B5_extreme_one_vs_rest",
        "指定维度是否为最大/最小",
        "category 1 iff argmax(x)=i (or argmin); category 2 otherwise",
        8,
        "ranking_extreme",
        1,
        1,
        4,
        "A named argmax/argmin operator; complement is a union.",
        "pilot",
        "Verbal extremes are common, but binary category-label use is not established by mention frequency.",
    ),
    CandidateFamily(
        "binary",
        "B6_count_of_long",
        "长维度计数",
        "category 1 iff sum_i 1[x_i>0.5]=k (or a fixed count set)",
        5,
        "count_cardinality",
        4,
        2,
        4,
        "Highly disconnected regions; exact count set must be fixed a priori.",
        "defer",
        "Do not add until choices show stable use beyond lexical mention.",
    ),
    CandidateFamily(
        "four_category",
        "F1_pair_similarity_quartet",
        "同一对维度的近似相等四分法",
        "near&low-mean; near&high-mean; x_i>x_j+delta; x_j>x_i+delta",
        6,
        "pairwise_near_equal",
        3,
        2,
        2,
        "Four convex regions; compatible with the current A@x<=b representation.",
        "add_core",
        "It gives equality positive volume while retaining direction and overall level on the same pair.",
    ),
    CandidateFamily(
        "four_category",
        "F2_similarity_x_axis",
        "近似相等带 × 另一维高低",
        "(|x_i-x_j|<=delta) x (x_k<=0.5)",
        12,
        "pairwise_near_equal",
        2,
        2,
        3,
        "One category can be disconnected unless stored as union components.",
        "restricted_core",
        "Use only combinations supported by subject/session evidence; do not enumerate arbitrary products.",
    ),
    CandidateFamily(
        "four_category",
        "F3_two_similarity_bands",
        "两组成对近似相等带",
        "(|x_i-x_j|<=delta) x (|x_k-x_l|<=delta)",
        3,
        "pairwise_near_equal",
        2,
        2,
        4,
        "Cartesian product of two binary relational predicates.",
        "pilot",
        "Parsimonious, but requires evidence that two equality predicates are maintained together.",
    ),
    CandidateFamily(
        "four_category",
        "F4_center_band_x_axis",
        "中等值带 × 另一维高低",
        "(|x_i-0.5|<=delta_center) x (x_j<=0.5)",
        12,
        "center_band",
        2,
        2,
        2,
        "Factorized two-predicate partition; some cells are disconnected.",
        "restricted_core",
        "Adds the directly verbalized middle state while staying within a two-predicate budget.",
    ),
    CandidateFamily(
        "four_category",
        "F5_single_axis_quartiles",
        "单维四级顺序分割",
        "x_i split at fixed q1<q2<q3 into four ordered bins",
        4,
        "ordinal_degree",
        3,
        3,
        1,
        "Four contiguous intervals on one dimension.",
        "pilot",
        "Very simple in dimensional attention, but graded adjectives only indirectly imply category bins.",
    ),
    CandidateFamily(
        "four_category",
        "F6_global_balance_x_axis",
        "全局均衡 × 指定维度高低",
        "(range(x)<=delta_global) x (x_i<=0.5)",
        4,
        "multiway_near_equal",
        2,
        2,
        4,
        "Named global predicate plus one axis predicate.",
        "pilot",
        "Adds global balance without an unconstrained multiway Boolean expansion.",
    ),
    CandidateFamily(
        "four_category",
        "F7_count_levels",
        "长维度计数四级化",
        "map count=sum_i 1[x_i>0.5] to four fixed labels",
        1,
        "count_cardinality",
        4,
        2,
        4,
        "Five possible counts require an arbitrary merge to produce four labels.",
        "defer",
        "The mapping is underdetermined and geometrically disconnected.",
    ),
)


def parse_list(value: Any) -> list[Any]:
    """Parse a CSV list cell without executing arbitrary code."""

    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable source fingerprint."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    """Return the current commit without requiring a clean worktree."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def unique_parts(parts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(part) for part in parts if part))


def classify_text(text: Any, parser: SemanticParser) -> dict[str, Any]:
    """Map one oral report to auditable primitive flags.

    Semantic claims control the direct relation codes. Narrow lexical rules
    add count, degree, and explicit-negation flags that are intentionally kept
    separate from geometric coverage. Equality groups are retained so the
    report can audit tolerance sensitivity at the claim level.
    """

    clean = normalize_text(text)
    parsed = parser.parse(clean)
    flags = {key: False for key in PRIMITIVE_BY_KEY}
    equality_pairs: set[tuple[str, ...]] = set()
    multiway_groups: set[tuple[str, ...]] = set()
    middle_parts: set[str] = set()
    has_group_sum_cue = bool(GROUP_SUM_CUE_PATTERN.search(clean))

    absolute_kinds = {
        "general_case",
        "universal_quantifier",
        "exclusive_case",
        "exclusion",
        "complement",
    }

    for claim in parsed.claims:
        parts = unique_parts(claim.parts)
        right_parts = unique_parts(claim.right_parts)
        if claim.kind in absolute_kinds:
            if claim.desc in {"long", "short"}:
                flags["absolute_side"] = True
            elif claim.desc == "middle":
                flags["center_band"] = True
                middle_parts.update(parts)

        if claim.kind == "body_ref":
            if claim.op in {">", "<"}:
                flags["absolute_side"] = True
            elif claim.op == "==":
                flags["center_band"] = True
                middle_parts.update(parts)

        if claim.kind in {"comparison", "chained_comparison"}:
            if claim.op in {">", "<"}:
                flags["pairwise_order"] = True
            elif claim.op == "==":
                relation_parts = unique_parts(parts + right_parts)
                relation_group = tuple(sorted(relation_parts))
                if len(relation_parts) == 2:
                    flags["pairwise_near_equal"] = True
                    equality_pairs.add(relation_group)
                elif len(relation_parts) >= 3 and has_group_sum_cue:
                    flags["group_sum_near_equal"] = True
                elif len(relation_parts) >= 3:
                    flags["multiway_near_equal"] = True
                    multiway_groups.add(relation_group)
            elif claim.op == "!=":
                flags["negated_relation"] = True

        if claim.kind == "equality":
            relation_group = tuple(sorted(parts))
            if len(parts) == 2:
                flags["pairwise_near_equal"] = True
                equality_pairs.add(relation_group)
            elif len(parts) >= 3 and has_group_sum_cue:
                flags["group_sum_near_equal"] = True
            elif len(parts) >= 3:
                flags["multiway_near_equal"] = True
                multiway_groups.add(relation_group)

        if claim.kind == "group_sum":
            if claim.op in {">", "<"}:
                flags["group_sum_order"] = True
            elif claim.op == "==":
                flags["group_sum_near_equal"] = True
            elif claim.op == "!=":
                flags["negated_relation"] = True

        if claim.kind in {"ranking", "superlative"}:
            flags["ranking_extreme"] = True

    if GLOBAL_BALANCE_PATTERN.search(clean):
        flags["multiway_near_equal"] = True
        multiway_groups.add(("__all__",))
    if COUNT_PATTERN.search(clean) and not has_group_sum_cue:
        flags["count_cardinality"] = True
    if ORDINAL_DEGREE_PATTERN.search(clean):
        flags["ordinal_degree"] = True
    if NEGATED_RELATION_PATTERN.search(clean):
        flags["negated_relation"] = True

    structural_count = sum(bool(flags[key]) for key in COVERAGE_PRIMITIVES)
    return {
        **flags,
        "n_structural_primitives": int(structural_count),
        "pair_equality_claim_count": int(len(equality_pairs)),
        "middle_part_count": int(len(middle_parts)),
        "unparsed_item_count": int(len(parsed.unparsed)),
        "pair_equality_groups": ["|".join(group) for group in sorted(equality_pairs)],
        "multiway_equality_groups": ["|".join(group) for group in sorted(multiway_groups)],
    }

def load_and_validate(data_path: Path, diagnostics_path: Path) -> pd.DataFrame:
    """Load the controlling sources and enforce exact trial-key agreement."""

    data_columns = [
        "iSub",
        "condition",
        "iSession",
        "iTrial",
        "text",
        "feature1_name",
        "feature2_name",
        "feature3_name",
        "feature4_name",
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "category",
        "choice",
        "feedback",
    ]
    data = pd.read_csv(data_path, usecols=data_columns)
    diagnostics = pd.read_csv(diagnostics_path)

    for name, frame in (("Task2 data", data), ("oral diagnostics", diagnostics)):
        duplicates = frame.duplicated(KEY_COLUMNS, keep=False)
        if duplicates.any():
            sample = frame.loc[duplicates, KEY_COLUMNS].head(5).to_dict("records")
            raise ValueError(f"{name} contains duplicate trial keys; examples: {sample}")

    data_keys = pd.MultiIndex.from_frame(data[KEY_COLUMNS])
    diagnostics_keys = pd.MultiIndex.from_frame(diagnostics[KEY_COLUMNS])
    missing_diagnostics = data_keys.difference(diagnostics_keys)
    extra_diagnostics = diagnostics_keys.difference(data_keys)
    if len(missing_diagnostics) or len(extra_diagnostics):
        raise ValueError(
            "Oral diagnostics are stale relative to Task2_processed.csv: "
            f"missing={len(missing_diagnostics)}, extra={len(extra_diagnostics)}. "
            "Regenerate results/oral_analysis before running this audit."
        )

    diagnostic_columns = [
        *KEY_COLUMNS,
        "fidelity",
        "fidelity_status",
        "n_fidelity_claims",
        "style_tags",
        "claim_labels",
        "failed_claims",
    ]
    merged = data.merge(
        diagnostics[diagnostic_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    merged["text"] = merged["text"].fillna("").astype(str).str.strip()
    merged["nonempty_text"] = merged["text"].ne("")
    merged["task_arity"] = np.where(
        merged["condition"].eq(1), "binary", "four_category"
    )
    return merged


def add_primitive_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse all texts once and append primitive columns."""

    parser = SemanticParser()
    records = [classify_text(text, parser) for text in frame["text"]]
    flags = pd.DataFrame.from_records(records, index=frame.index)
    return pd.concat([frame, flags], axis=1)


def subject_primitive_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return subject-level counts so long protocols do not dominate evidence."""

    rows: list[dict[str, Any]] = []
    for (task_arity, condition, subject), group in frame.groupby(
        ["task_arity", "condition", "iSub"], sort=True
    ):
        denominator = int(group["nonempty_text"].sum())
        for primitive in PRIMITIVES:
            count = int(group[primitive.key].sum())
            rows.append(
                {
                    "task_arity": task_arity,
                    "condition": int(condition),
                    "iSub": int(subject),
                    "primitive": primitive.key,
                    "mention_trials": count,
                    "nonempty_trials": denominator,
                    "within_subject_rate": count / denominator if denominator else np.nan,
                    "mentioned": count > 0,
                    "repeated_3plus": count >= 3,
                }
            )
    return pd.DataFrame(rows)


def summarize_primitives(
    frame: pd.DataFrame,
    subject_counts: pd.DataFrame,
    group_columns: Sequence[str],
    level_kind: str,
) -> pd.DataFrame:
    """Summarize trial, subject, repeated-subject, and session prevalence."""

    rows: list[dict[str, Any]] = []
    grouper: str | list[str]
    if len(group_columns) == 1:
        grouper = group_columns[0]
    else:
        grouper = list(group_columns)

    for group_values, group in frame.groupby(grouper, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_identity = dict(zip(group_columns, group_values))
        subject_filter = np.ones(len(subject_counts), dtype=bool)
        for column, value in group_identity.items():
            subject_filter &= subject_counts[column].eq(value).to_numpy()
        subject_group = subject_counts.loc[subject_filter]
        nonempty = int(group["nonempty_text"].sum())
        n_subjects = int(group["iSub"].nunique())
        n_subject_sessions = int(group[["iSub", "iSession"]].drop_duplicates().shape[0])

        for primitive in PRIMITIVES:
            mask = group[primitive.key].astype(bool)
            sub = subject_group[subject_group["primitive"].eq(primitive.key)]
            mentioned_sessions = int(
                group.loc[mask, ["iSub", "iSession"]].drop_duplicates().shape[0]
            )
            fidelity = pd.to_numeric(group.loc[mask, "fidelity"], errors="coerce")
            rows.append(
                {
                    "level_kind": level_kind,
                    **group_identity,
                    "primitive": primitive.key,
                    "primitive_label_en": primitive.label_en,
                    "primitive_label_zh": primitive.label_zh,
                    "definition": primitive.definition,
                    "evidence_kind": primitive.evidence_kind,
                    "trials": int(len(group)),
                    "nonempty_trials": nonempty,
                    "mention_trials": int(mask.sum()),
                    "mention_rate_nonempty": float(mask.sum() / nonempty) if nonempty else np.nan,
                    "subjects": n_subjects,
                    "subjects_ever": int(sub["mentioned"].sum()),
                    "subject_rate": float(sub["mentioned"].mean()),
                    "subjects_repeated_3plus": int(sub["repeated_3plus"].sum()),
                    "repeated_subject_rate": float(sub["repeated_3plus"].mean()),
                    "median_subject_rate": float(sub["within_subject_rate"].median()),
                    "subject_sessions": n_subject_sessions,
                    "subject_sessions_ever": mentioned_sessions,
                    "subject_session_rate": (
                        mentioned_sessions / n_subject_sessions if n_subject_sessions else np.nan
                    ),
                    "parseable_fidelity_rate": float(fidelity.notna().mean()) if mask.any() else np.nan,
                    "mean_fidelity": float(fidelity.mean()) if fidelity.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def evidence_examples(frame: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Save high-frequency wording examples for transparent lexical auditing."""

    rows: list[dict[str, Any]] = []
    for task_arity, group in frame.groupby("task_arity", sort=True):
        for primitive in PRIMITIVES:
            subset = group[group[primitive.key] & group["nonempty_text"]]
            counts = subset["text"].value_counts().head(top_n)
            for rank, (text, count) in enumerate(counts.items(), start=1):
                example_rows = subset[subset["text"].eq(text)]
                sample = example_rows.iloc[0]
                rows.append(
                    {
                        "task_arity": task_arity,
                        "primitive": primitive.key,
                        "rank": rank,
                        "text": text,
                        "count": int(count),
                        "subjects": int(example_rows["iSub"].nunique()),
                        "sample_subject": int(sample["iSub"]),
                        "sample_session": int(sample["iSession"]),
                        "sample_trial": int(sample["iTrial"]),
                        "mean_fidelity": float(example_rows["fidelity"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def partition_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe the current library without loading similarity matrices."""

    rows: list[dict[str, Any]] = []
    for n_cats, arity in ((2, "binary"), (4, "four_category")):
        partition = ContinuousPartition(4, n_cats, similarity_n_samples=1)
        for index, split in enumerate(partition.hypothesis_space):
            active_dims = sorted(
                {
                    dim
                    for coefficients, _ in split.hyperplanes
                    for dim, coefficient in enumerate(coefficients)
                    if not math.isclose(float(coefficient), 0.0)
                }
            )
            if split.family in {"dimension_max", "dimension_min"}:
                cognitive_predicates = 1
                decision_depth = 1
            else:
                cognitive_predicates = len(split.hyperplanes)
                decision_depth = 1 if n_cats == 2 else min(2, len(split.hyperplanes))
            rows.append(
                {
                    "task_arity": arity,
                    "n_categories": n_cats,
                    "hypothesis_index": index,
                    "family": split.family,
                    "geometric_constraints": len(split.hyperplanes),
                    "cognitive_predicates": cognitive_predicates,
                    "decision_depth": decision_depth,
                    "active_dimensions": len(active_dims),
                    "active_dimension_indices": ",".join(str(dim) for dim in active_dims),
                }
            )
    inventory = pd.DataFrame(rows)
    family_summary = (
        inventory.groupby(
            [
                "task_arity",
                "n_categories",
                "family",
                "cognitive_predicates",
                "decision_depth",
            ],
            as_index=False,
        )
        .agg(
            hypotheses=("hypothesis_index", "size"),
            min_active_dimensions=("active_dimensions", "min"),
            max_active_dimensions=("active_dimensions", "max"),
            geometric_constraints=("geometric_constraints", "max"),
        )
        .sort_values(["task_arity", "cognitive_predicates", "family"])
    )
    return inventory, family_summary


def equality_tolerance_sensitivity(
    frame: pd.DataFrame, deltas: Sequence[float] = (0.06, 0.10, 0.15)
) -> pd.DataFrame:
    """Evaluate whether verbal similarity claims match several fixed bands.

    Each explicit equality group contributes one claim. ``__all__`` denotes a
    global-balance phrase and is evaluated across all four observed features.
    This is a descriptive fidelity audit, not an estimate of a model boundary.
    """

    claims: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        values: dict[str, float] = {}
        all_values: list[float] = []
        for feature_index in range(1, 5):
            value = float(row[f"feature{feature_index}"])
            feature_name = str(row[f"feature{feature_index}_name"]).strip()
            part = FEATURE_NAME_TO_PART.get(feature_name, feature_name)
            values[part] = value
            all_values.append(value)

        for relation, column in (
            ("pairwise_near_equal", "pair_equality_groups"),
            ("multiway_near_equal", "multiway_equality_groups"),
        ):
            for encoded_group in row[column]:
                parts = encoded_group.split("|")
                claim_values = all_values if parts == ["__all__"] else [values[p] for p in parts if p in values]
                if len(claim_values) < 2:
                    continue
                claims.append(
                    {
                        "task_arity": row["task_arity"],
                        "relation": relation,
                        "iSub": int(row["iSub"]),
                        "spread": float(max(claim_values) - min(claim_values)),
                    }
                )

    claim_frame = pd.DataFrame(claims)
    rows: list[dict[str, Any]] = []
    if claim_frame.empty:
        return pd.DataFrame(rows)
    for (task_arity, relation), group in claim_frame.groupby(
        ["task_arity", "relation"], sort=True
    ):
        for delta in deltas:
            within = group["spread"].le(float(delta))
            rows.append(
                {
                    "task_arity": task_arity,
                    "relation": relation,
                    "delta": float(delta),
                    "claims": int(len(group)),
                    "subjects": int(group["iSub"].nunique()),
                    "median_spread": float(group["spread"].median()),
                    "p90_spread": float(group["spread"].quantile(0.90)),
                    "claims_within_delta": int(within.sum()),
                    "claim_fidelity": float(within.mean()),
                }
            )
    return pd.DataFrame(rows)


def candidate_catalog(
    prevalence: pd.DataFrame, cooccurrence: pd.DataFrame
) -> pd.DataFrame:
    """Attach primitive or candidate-specific co-occurrence support."""

    arity_prevalence = prevalence[prevalence["level_kind"].eq("task_arity")].copy()
    primitive_lookup = {
        (str(row["task_arity"]), str(row["primitive"])): row
        for _, row in arity_prevalence.iterrows()
    }
    cooccurrence_lookup = {
        (str(row["task_arity"]), str(row["cooccurrence"])): row
        for _, row in cooccurrence.iterrows()
    }
    composite_basis = {
        "F2_similarity_x_axis": "pair_equal_plus_absolute",
        "F3_two_similarity_bands": "two_pair_equality_claims",
        "F4_center_band_x_axis": "center_plus_absolute",
    }

    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_FAMILIES:
        co_name = composite_basis.get(candidate.candidate_id)
        if co_name is not None:
            support = cooccurrence_lookup.get((candidate.arity, co_name))
            support_basis = f"cooccurrence:{co_name}"
            support_subjects = int(support["subjects_ever"]) if support is not None else 0
            support_rate = float(support["subject_rate"]) if support is not None else np.nan
            repeated_rate = float(support["repeated_subject_rate"]) if support is not None else np.nan
            trial_rate = float(support["mention_rate_nonempty"]) if support is not None else np.nan
            mean_fidelity = float(support["mean_fidelity"]) if support is not None else np.nan
        else:
            support = primitive_lookup.get((candidate.arity, candidate.primary_primitive))
            support_basis = f"primitive:{candidate.primary_primitive}"
            support_subjects = int(support["subjects_ever"]) if support is not None else 0
            support_rate = float(support["subject_rate"]) if support is not None else np.nan
            repeated_rate = float(support["repeated_subject_rate"]) if support is not None else np.nan
            trial_rate = float(support["mention_rate_nonempty"]) if support is not None else np.nan
            mean_fidelity = float(support["mean_fidelity"]) if support is not None else np.nan

        row = asdict(candidate)
        row.update(
            {
                "support_basis": support_basis,
                "support_subjects_ever": support_subjects,
                "support_subject_rate": support_rate,
                "support_repeated_subject_rate": repeated_rate,
                "support_trial_rate_nonempty": trial_rate,
                "support_mean_fidelity": mean_fidelity,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

def coverage_scenarios(
    frame: pd.DataFrame, subject_counts: pd.DataFrame
) -> pd.DataFrame:
    """Quantify coverage under current, core, and extended grammars."""

    scenarios: dict[str, dict[str, set[str]]] = {
        "binary": {
            "current": {"absolute_side", "pairwise_order", "group_sum_order"},
            "core_additions": {
                "absolute_side",
                "center_band",
                "pairwise_order",
                "pairwise_near_equal",
                "group_sum_order",
            },
            "extended_pilot": {
                "absolute_side",
                "center_band",
                "pairwise_order",
                "pairwise_near_equal",
                "multiway_near_equal",
                "group_sum_order",
                "group_sum_near_equal",
                "ranking_extreme",
            },
        },
        "four_category": {
            "current": {"absolute_side", "pairwise_order", "ranking_extreme"},
            "core_additions": {
                "absolute_side",
                "center_band",
                "pairwise_order",
                "pairwise_near_equal",
                "ranking_extreme",
            },
            "extended_pilot": {
                "absolute_side",
                "center_band",
                "pairwise_order",
                "pairwise_near_equal",
                "multiway_near_equal",
                "group_sum_order",
                "group_sum_near_equal",
                "ranking_extreme",
            },
        },
    }

    rows: list[dict[str, Any]] = []
    for task_arity, group in frame.groupby("task_arity", sort=True):
        structured = group[group["n_structural_primitives"].gt(0)].copy()
        total_mentions = int(structured[list(COVERAGE_PRIMITIVES)].to_numpy(dtype=bool).sum())
        subject_group = subject_counts[subject_counts["task_arity"].eq(task_arity)]
        frequent = subject_group[
            subject_group["primitive"].isin(COVERAGE_PRIMITIVES)
            & subject_group["mention_trials"].ge(3)
        ]
        frequent_by_subject = {
            int(subject): set(values["primitive"])
            for subject, values in frequent.groupby("iSub")
        }
        n_subjects = int(group["iSub"].nunique())

        for scenario, covered in scenarios[task_arity].items():
            matrix = structured[list(COVERAGE_PRIMITIVES)].to_numpy(dtype=bool)
            uncovered_columns = [
                idx
                for idx, primitive in enumerate(COVERAGE_PRIMITIVES)
                if primitive not in covered
            ]
            if uncovered_columns:
                fully_covered = ~matrix[:, uncovered_columns].any(axis=1)
            else:
                fully_covered = np.ones(len(structured), dtype=bool)
            covered_mentions = int(
                structured[[key for key in COVERAGE_PRIMITIVES if key in covered]]
                .to_numpy(dtype=bool)
                .sum()
            )
            frequent_covered_subjects = sum(
                frequent_by_subject.get(int(subject), set()).issubset(covered)
                for subject in group["iSub"].unique()
            )
            rows.append(
                {
                    "task_arity": task_arity,
                    "scenario": scenario,
                    "structured_trials": int(len(structured)),
                    "fully_covered_trials": int(fully_covered.sum()),
                    "full_trial_coverage_rate": float(fully_covered.mean()),
                    "structural_primitive_mentions": total_mentions,
                    "covered_primitive_mentions": covered_mentions,
                    "primitive_mention_coverage_rate": (
                        covered_mentions / total_mentions if total_mentions else np.nan
                    ),
                    "subjects": n_subjects,
                    "subjects_with_all_repeated_primitives_covered": frequent_covered_subjects,
                    "repeated_primitive_subject_coverage_rate": (
                        frequent_covered_subjects / n_subjects if n_subjects else np.nan
                    ),
                    "covered_primitives": ",".join(sorted(covered)),
                }
            )
    return pd.DataFrame(rows)


def cooccurrence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit candidate-specific evidence for factorized partitions."""

    definitions = {
        "pair_equal_plus_absolute": frame["pairwise_near_equal"] & frame["absolute_side"],
        "pair_equal_plus_center": frame["pairwise_near_equal"] & frame["center_band"],
        "center_plus_absolute": frame["center_band"] & frame["absolute_side"],
        "two_pair_equality_claims": frame["pair_equality_claim_count"].ge(2),
        "two_middle_parts": frame["middle_part_count"].ge(2),
    }
    rows: list[dict[str, Any]] = []
    for task_arity, group in frame.groupby("task_arity", sort=True):
        denominator = int(group["nonempty_text"].sum())
        n_subjects = int(group["iSub"].nunique())
        for name, global_mask in definitions.items():
            mask = global_mask.loc[group.index]
            counts = group.loc[mask].groupby("iSub").size()
            fidelity = pd.to_numeric(group.loc[mask, "fidelity"], errors="coerce")
            subjects_ever = int(counts.size)
            repeated = int(counts.ge(3).sum())
            rows.append(
                {
                    "task_arity": task_arity,
                    "cooccurrence": name,
                    "mention_trials": int(mask.sum()),
                    "mention_rate_nonempty": float(mask.sum() / denominator) if denominator else np.nan,
                    "subjects_ever": subjects_ever,
                    "subjects": n_subjects,
                    "subject_rate": float(subjects_ever / n_subjects) if n_subjects else np.nan,
                    "subjects_repeated_3plus": repeated,
                    "repeated_subject_rate": float(repeated / n_subjects) if n_subjects else np.nan,
                    "mean_fidelity": float(fidelity.mean()) if fidelity.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)
