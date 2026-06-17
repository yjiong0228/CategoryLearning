"""Shared parser and encoders for Task2 oral reports.

The coding space is four-dimensional. Each experiment can provide its own
parallel feature vocabulary, such as body parts or colors.
"""

from __future__ import annotations

import ast
import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


PARTS = ("脖子", "头", "腿", "尾巴")
PART_TO_DIM = {"脖子": 0, "头": 1, "腿": 2, "尾巴": 3}
FEATURE_NAME_TO_PART = {
    "neck": "脖子",
    "head": "头",
    "leg": "腿",
    "tail": "尾巴",
    "green": "绿色",
    "yellow": "黄色",
    "pink": "粉色",
    "blue": "蓝色",
    "绿色": "绿色",
    "黄色": "黄色",
    "粉色": "粉色",
    "蓝色": "蓝色",
}
PART_ALIASES = {
    "脖子": ((r"脖(?!子)", "脖子"),),
    "尾巴": ((r"尾(?!巴)", "尾巴"),),
    "绿色": ((r"绿(?!色)", "绿色"),),
    "黄色": ((r"黄(?!色)", "黄色"),),
    "粉色": ((r"粉(?!色)", "粉色"),),
    "蓝色": ((r"蓝(?!色)", "蓝色"),),
}
PART_RE = r"(?:脖子|头|腿|尾巴)"
PART_GROUP_RE = r"(?:脖子|头|腿|尾巴)(?:[、和与及]*(?:脖子|头|腿|尾巴))*"
BODY_RE = r"(?:躯干|身体)"
HALF_RE = r"(?:1/2|一半|二分之一)"
REFERENCE_RE = rf"(?:{BODY_RE}|{HALF_RE})"
COMPLEMENT_RE = r"(?:其他|其余|另外|剩下|剩余)"
EQUALITY_WORD_RE = r"一样|差不多|相等|相当|相近|接近|相似|类似|一致|均匀|均衡|平均|(?<!中)等长"
INEQUALITY_WORD_RE = r"不一样|不同|不相等|各不相同|不等于|不等长"
NEGATION_RE = r"(?:不是|并非|不算|不太|不够|不怎么|没有|未|不)"
COMPARISON_MODIFIER_RE = r"(?:略|稍|稍微|更|较|比较|明显|显著|相对|有点|稍稍|稍许|更加)"
SPLIT_RE = re.compile(r"[，,。！？!？；;]+")
ORDINAL_SEQUENCE_RE = re.compile(
    r"最长|最短|最大|最小|最高|最低|第一|第二|第三|第四|第[一二三四1234]|其次|次之|最后|末尾|再是|然后|接着"
)
SEQUENCE_CONTINUE_RE = re.compile(r"再是|然后|接着|再者")


@dataclass
class SemanticClaim:
    kind: str
    item: str
    parts: list[str] = field(default_factory=list)
    op: str | None = None
    right_parts: list[str] = field(default_factory=list)
    desc: str | None = None
    threshold: float | None = None
    order: list[str] = field(default_factory=list)
    supported: bool = True
    note: str = ""


@dataclass
class SemanticResult:
    text: Any
    items: list[str]
    claims: list[SemanticClaim]
    unparsed: list[str]


@dataclass
class FidelityClaim:
    kind: str
    label: str
    value: float
    passed: bool


def part_regex(parts: tuple[str, ...] = PARTS) -> str:
    escaped = [re.escape(part) for part in sorted(parts, key=len, reverse=True)]
    return "(?:" + "|".join(escaped) + ")"


def part_group_regex(parts: tuple[str, ...] = PARTS) -> str:
    item_re = part_regex(parts)
    return rf"{item_re}(?:[、和与及]*{item_re})*"


def part_to_dim(parts: tuple[str, ...] = PARTS) -> dict[str, int]:
    return {part: idx for idx, part in enumerate(parts)}


def parts_from_feature_map(feature_map: dict[str, int] | None) -> tuple[str, ...]:
    if not feature_map:
        return PARTS
    ordered: list[str | None] = [None] * len(feature_map)
    for feature_name, dim in feature_map.items():
        idx = int(dim)
        if idx < 0 or idx >= len(ordered):
            return PARTS
        ordered[idx] = FEATURE_NAME_TO_PART.get(str(feature_name).strip(), str(feature_name).strip())
    if any(part is None for part in ordered):
        return PARTS
    return tuple(str(part) for part in ordered)


def normalize_text(text: Any, parts: tuple[str, ...] = PARTS) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    for part in parts:
        for pattern, replacement in PART_ALIASES.get(part, ()):
            text = re.sub(pattern, replacement, text)
    text = text.replace("身子", "身体")
    text = text.replace("其它", "其他")
    return text


def split_items(text: str) -> list[str]:
    return [item.strip(" 、\t\r\n") for item in SPLIT_RE.split(text) if item.strip(" 、\t\r\n")]


def parts_in_text(text: str, parts: tuple[str, ...] = PARTS) -> list[str]:
    found = []
    for part in parts:
        pos = text.find(part)
        if pos >= 0:
            found.append((pos, part))
    return [part for _, part in sorted(found)]


def all_parts_phrase(text: str) -> bool:
    return bool(
        re.search(
            r"四个(?:部位)?|所有(?:部位)?|全部(?:部位)?|各(?:个)?部位|每(?:个|一个)部位|四肢",
            text,
        )
    )


def parse_literal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    return value


def part_values_from_row(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx in range(1, 5):
        part = FEATURE_NAME_TO_PART[str(row[f"feature{idx}_name"]).strip()]
        out[part] = float(row[f"feature{idx}"])
    return out


def feature_vector_from_row(row: pd.Series) -> np.ndarray:
    return row[["feature1", "feature2", "feature3", "feature4"]].to_numpy(dtype=float)


def compare_values(left: float, op: str, right: float, tol: float, eq_eps: float) -> tuple[bool, float]:
    if op == ">":
        value = left - right
        return bool(value > -tol), float(value)
    if op == "<":
        value = right - left
        return bool(value > -tol), float(value)
    if op == "==":
        value = abs(left - right)
        return bool(value <= eq_eps), float(value)
    if op == "!=":
        value = abs(left - right)
        return bool(value > eq_eps), float(value)
    raise ValueError(f"Unsupported op: {op}")


def describe_op(op: str) -> str:
    return {">": ">", "<": "<", "==": "=", "!=": "!="}.get(op, op)


def desc_label(text: str, allow_middle: bool = True) -> str | None:
    if re.search(INEQUALITY_WORD_RE, text):
        return None
    negated = negated_direct_desc(text)
    if negated is not None:
        return negated
    if is_negated_superlative(text):
        return None
    if re.search(r"达到.{0,3}(?:最|极)?(?:短|小|低)|(?:最|极)(?:短|小|低)(?:的)?(?:长度|程度|值)?|最小(?:长度|值)?|最低(?:长度|值)?", text):
        return "short"
    if re.search(r"达到.{0,3}(?:最|极)?(?:长|大|高)|(?:最|极)(?:长|大|高)(?:的)?(?:长度|程度|值)?|最大(?:长度|值)?|最高(?:长度|值)?", text):
        return "long"
    if re.search(r"大于一半|超过一半|高于一半|超过(?:其|它|其自身|自身)?(?:最大|最长)长度的1/2", text):
        return "long"
    if re.search(r"小于一半|不到一半|低于一半|少于一半", text):
        return "short"
    if re.search(r"长一点|长一些|稍长|偏长一点", text):
        return "long"
    if re.search(r"短一点|短一些|稍短|偏短一点", text):
        return "short"
    if re.search(r"中等偏长|偏长|较长|略长|很长|非常长|特别长|超长|极长|挺长|比较长|均较长|均长|都长", text):
        return "long"
    if re.search(r"中等偏短|偏短|较短|略短|很短|非常短|特别短|极短|挺短|比较短|均较短|均短|都短", text):
        return "short"
    if allow_middle and re.search(r"中等(?:长度|长)?|适中|一般|平均|中间|还好|(?:其他|其余|另外|剩下|剩余)(?:都|部位|三个|两)?中$", text):
        return "middle"
    if "短" in text:
        return "short"
    if "长" in text and not re.search(r"一样长|差不多长|相等|相近|相似|相当|长度", text):
        return "long"
    return None


def is_negated_superlative(text: str) -> bool:
    return bool(re.search(rf"{NEGATION_RE}.{{0,8}}(?:最|最大|最小|最高|最低)", text))


def negated_direct_desc(text: str) -> str | None:
    if not re.search(NEGATION_RE, text):
        return None
    if is_negated_superlative(text) or re.search(INEQUALITY_WORD_RE, text):
        return None
    if re.search(rf"没有.*(?:{PART_RE}|{BODY_RE}).*?(?:长|短|高|低|大|小)", text):
        return None

    if re.search(rf"{NEGATION_RE}.{{0,8}}(?:长|高|大)", text):
        return "short"
    if re.search(rf"{NEGATION_RE}.{{0,8}}(?:短|低|小)", text):
        return "long"
    return None


def opposite_desc(desc: str) -> str:
    if desc == "long":
        return "short"
    if desc == "short":
        return "long"
    return desc


def relation_op(raw_op: str, desc: str | None = None) -> str:
    if raw_op in {"大于", "大于等于", "长于", "高于", "超过"}:
        return ">"
    if raw_op in {"小于", "小于等于", "短于", "低于", "不到"}:
        return "<"
    if raw_op in {"等于", "约等于", "相等", "持平"}:
        return "=="
    if raw_op in {"不等于", "不一样"}:
        return "!="
    if raw_op == "比":
        if desc in {"长", "高", "大"}:
            return ">"
        if desc in {"短", "低", "小"}:
            return "<"
    return ""


def body_reference_threshold(text: str, body_value: float = 0.5) -> float:
    has_body = bool(re.search(BODY_RE, text))
    if has_body and re.search(r"3/4|四分之三", text):
        return float(body_value) * 0.75
    if has_body and re.search(HALF_RE, text):
        return float(body_value) * 0.5
    if re.search(HALF_RE, text):
        return 0.5
    if re.search(r"3/4|四分之三", text):
        return 0.75
    return float(body_value)


def has_reference_threshold(text: str) -> bool:
    return bool(re.search(REFERENCE_RE, text))


def reference_relation_op(text: str) -> str:
    if re.search(r"比.*?(?:长|高|大)|长于|大于|高于|超过", text):
        return ">"
    if re.search(r"比.*?(?:短|低|小)|短于|小于|低于|不到|少于|没有.*?长", text):
        return "<"
    if not re.search(NEGATION_RE, text) and re.search(r"等于|约等于|一样|持平|差不多|相当|接近|是|为|=", text):
        return "=="
    return ""


def ordinal_sequence_groups(items: list[str], schema_parts: tuple[str, ...] = PARTS) -> list[list[str]]:
    """Parse cross-clause long-to-short descriptions into ordered part groups."""
    if not items:
        return []
    joined = "，".join(items)
    if not ORDINAL_SEQUENCE_RE.search(joined):
        return []
    if re.search(r"1/2|一半|二分之一|3/4|四分之三|大概|可能", joined):
        return []

    entries: list[tuple[float, list[str]]] = []
    last_rank: float | None = None
    last_direction: str | None = None
    has_top = False
    has_bottom = False
    has_inner_ordinal = False
    has_sequence_continue = False

    for idx, item in enumerate(items):
        item_parts = parts_in_text(item, schema_parts)
        has_entity = bool(item_parts or has_reference_threshold(item))
        if not has_entity:
            continue

        rank: float | None = None
        direction: str | None = None
        desc = desc_label(item, allow_middle=True)
        if re.search(r"最.{0,2}(?:长|大|高)|最大|最高|第一|第1|第一长|第一个", item):
            rank = 1.0
            direction = "down"
            has_top = True
        elif re.search(r"第四|第4|最.{0,2}(?:短|小|低)|最小|最低|最后|末尾|第四个", item):
            rank = 100.0
            direction = "up"
            has_bottom = True
        elif re.search(r"第二|第2|其次|次之|二长|第二长|第二个", item):
            if last_rank is None:
                rank = 2.0
                direction = "down"
            elif last_direction == "up":
                rank = last_rank - 1.0
                direction = "up"
            else:
                rank = last_rank + 1.0
                direction = "down"
            has_inner_ordinal = True
        elif re.search(r"第三|第3|三长|第三长|第三个", item):
            if last_rank is not None and last_direction == "up":
                rank = last_rank - 1.0
                direction = "up"
            else:
                rank = 3.0
                direction = "down"
            has_inner_ordinal = True
        elif item_parts:
            if SEQUENCE_CONTINUE_RE.search(item):
                if last_rank is None:
                    rank = 2.0
                    direction = "down"
                elif last_direction == "up":
                    rank = last_rank - 1.0
                    direction = "up"
                else:
                    rank = last_rank + 1.0
                    direction = "down"
                has_sequence_continue = True
            elif last_rank is not None and last_direction == "down" and (
                re.search(r"稍短|较短|略短|短一点|短一些|偏短|更短|居中|中等|还好", item)
                or (has_inner_ordinal and desc in {"short", "middle"})
                or _has_later_bottom(items, idx)
                or _has_later_sequence_continue(items, idx)
            ):
                rank = last_rank + 1.0
                direction = "down"
            elif last_rank is not None and last_direction == "up" and (
                re.search(r"稍长|较长|略长|长一点|长一些|偏长|更长|居中|中等|还好", item)
                or (has_inner_ordinal and desc in {"long", "middle"})
                or _has_later_sequence_continue(items, idx)
            ):
                rank = last_rank - 1.0
                direction = "up"
            elif last_rank is None and desc == "long" and (
                _has_later_bottom(items, idx) or _has_later_secondary(items, idx)
            ):
                rank = 1.0
                direction = "down"
            elif last_rank is None and desc == "short" and _has_later_secondary(items, idx):
                rank = 100.0
                direction = "up"

        if rank is None:
            continue
        if item_parts:
            entries.append((rank, item_parts))
        last_rank = rank
        last_direction = direction or last_direction

    seen_parts = {part for _, group_parts in entries for part in group_parts}
    if seen_parts != set(schema_parts):
        return []
    if not (has_top or has_bottom):
        return []
    if not (has_inner_ordinal or (has_top and has_bottom) or (has_top and has_sequence_continue)):
        return []

    grouped: dict[int, list[str]] = {}
    for rank, group_parts in entries:
        grouped.setdefault(rank, [])
        for part in group_parts:
            if part not in grouped[rank]:
                grouped[rank].append(part)

    ordered_groups: list[list[str]] = []
    used: set[str] = set()
    for rank in sorted(grouped):
        group = [part for part in grouped[rank] if part not in used]
        if group:
            ordered_groups.append(group)
            used.update(group)
    return ordered_groups if used == set(schema_parts) and len(ordered_groups) >= 2 else []


def _has_later_bottom(items: list[str], idx: int) -> bool:
    return any(
        re.search(r"第四|第4|最.{0,2}(?:短|小|低)|最小|最低|最后|末尾|第四个", item)
        for item in items[idx + 1 :]
    )


def _has_later_sequence_continue(items: list[str], idx: int) -> bool:
    return any(SEQUENCE_CONTINUE_RE.search(item) for item in items[idx + 1 :])


def _has_later_secondary(items: list[str], idx: int) -> bool:
    return any(re.search(r"第二|第2|其次|次之|二长|第二长|第二个", item) for item in items[idx + 1 :])


def parse_superlative_relation(item: str, parts: tuple[str, ...] = PARTS) -> tuple[str, str, list[str]] | None:
    if is_negated_superlative(item):
        return None
    if "最" not in item:
        return None
    part_re = part_regex(parts)

    scoped_is_match = re.search(
        rf"(?P<target>{part_re}).{{0,4}}(?:是|为)(?P<scope>.+?)(?:中|里|当中|之中|之间).{{0,4}}最.{{0,2}}(?P<desc>长|短|大|小|高|低)",
        item,
    )
    if scoped_is_match:
        target = scoped_is_match.group("target")
        desc = scoped_is_match.group("desc")
        right_parts = [part for part in parts_in_text(scoped_is_match.group("scope"), parts) if part != target]
        if right_parts:
            return target, desc, right_parts

    scoped_match = re.search(
        rf"(?P<target>{part_re}).{{0,4}}(?:在|于)(?P<scope>.+?)(?:中|里|当中|之中|之间).{{0,4}}最.{{0,2}}(?P<desc>长|短|大|小|高|低)",
        item,
    )
    if scoped_match:
        target = scoped_match.group("target")
        desc = scoped_match.group("desc")
        right_parts = [part for part in parts_in_text(scoped_match.group("scope"), parts) if part != target]
        if right_parts:
            return target, desc, right_parts

    super_match = re.search(r"最.{0,2}(长|短|大|小|高|低)", item)
    if not super_match:
        return None
    prefix = item[: super_match.start()]
    prefix_parts = parts_in_text(prefix, parts)
    if not prefix_parts:
        return None
    target = prefix_parts[-1]
    desc = super_match.group(1)

    scope_match = re.search(r"(?P<scope>.+?)(?:中|里|当中|之中|之间)\s*$", prefix)
    if scope_match:
        scope_parts = parts_in_text(scope_match.group("scope"), parts)
        right_parts = [part for part in scope_parts if part != target]
        if right_parts:
            return target, desc, right_parts

    return target, desc, [part for part in parts if part != target]


class SemanticParser:
    def __init__(self, body_value: float = 0.5, parts: tuple[str, ...] | None = None):
        self.body_value = float(body_value)
        self.parts = tuple(parts or PARTS)
        self.part_to_dim = part_to_dim(self.parts)
        self.part_re = part_regex(self.parts)
        self.part_group_re = part_group_regex(self.parts)
        self.meta_patterns = ("选错了", "不确定", "不知道", "随便选", "没什么区别", "看不出")

    def normalize_text(self, text: Any) -> str:
        return normalize_text(text, self.parts)

    def parts_in_text(self, text: str) -> list[str]:
        return parts_in_text(text, self.parts)

    def parse(self, text: Any) -> SemanticResult:
        normalized = self.normalize_text(text)
        if not normalized:
            return SemanticResult(text=text, items=[], claims=[], unparsed=[])

        items = split_items(normalized)
        claims: list[SemanticClaim] = []
        unparsed: list[str] = []
        context_parts: list[str] = []
        last_parts: list[str] = []
        last_desc: str | None = None
        last_was_superlative = False

        ordinal_claims = self._parse_ordinal_sequence(items, normalized)
        if ordinal_claims:
            return SemanticResult(text=text, items=items, claims=ordinal_claims, unparsed=[])

        i = 0
        while i < len(items):
            item = items[i]
            if self._is_meta_only(item):
                claims.append(SemanticClaim(kind="meta", item=item, supported=False, note="meta"))
                unparsed.append(item)
                i += 1
                continue

            same_item_exclusion = self._parse_same_item_exclusion(item)
            if same_item_exclusion:
                claims.extend(same_item_exclusion)
                context_parts = self._update_context_parts(context_parts, item)
                last_parts = self.parts_in_text(item)
                last_desc = desc_label(item, allow_middle=True)
                last_was_superlative = False
                i += 1
                continue

            if ("除" in item) and i + 1 < len(items):
                exclusion = self._parse_exclusion(item, items[i + 1])
                if exclusion:
                    claims.extend(exclusion)
                    context_parts = self._update_context_parts(context_parts, item)
                    last_parts = self.parts_in_text(item)
                    last_desc = desc_label(items[i + 1], allow_middle=True)
                    last_was_superlative = False
                    i += 2
                    continue

            contextual_endpoint = self._parse_contextual_endpoint(item, last_parts)
            if contextual_endpoint:
                claims.extend(contextual_endpoint)
                i += 1
                continue

            secondary_relation = self._parse_secondary_relation(item, last_parts, last_desc, last_was_superlative)
            if secondary_relation is not None:
                claims.extend(secondary_relation)
                context_parts = self._update_context_parts(context_parts, item)
                item_parts = self.parts_in_text(item)
                if item_parts:
                    last_parts = item_parts
                last_desc = None
                last_was_superlative = False
                i += 1
                continue

            other_comparison = self._parse_other_comparison(item, last_parts)
            if other_comparison:
                claims.extend(other_comparison)
                context_parts = self._update_context_parts(context_parts, item)
                item_parts = self.parts_in_text(item)
                if item_parts:
                    last_parts = item_parts
                last_desc = desc_label(item, allow_middle=True)
                last_was_superlative = False
                i += 1
                continue

            lookahead_parts = self.parts_in_text(items[i + 1]) if i + 1 < len(items) else []
            complement = self._parse_complement(item, context_parts, last_parts, lookahead_parts)
            if complement:
                claims.extend(complement)
                context_parts = self._update_context_parts(context_parts, item)
                item_parts = self.parts_in_text(item)
                if item_parts:
                    last_parts = item_parts
                last_desc = desc_label(item, allow_middle=True)
                last_was_superlative = False
                i += 1
                continue

            parsed = self._parse_item(item)
            if parsed:
                claims.extend(parsed)
                context_parts = self._update_context_parts(context_parts, item)
                item_parts = self.parts_in_text(item) or self._parts_from_claims(parsed)
                if item_parts:
                    last_parts = item_parts
                last_desc = desc_label(item, allow_middle=True)
                last_was_superlative = any(claim.kind == "superlative" for claim in parsed)
            else:
                unsupported = self._parse_unsupported(item)
                if unsupported:
                    claims.append(unsupported)
                unparsed.append(item)
            i += 1

        return SemanticResult(text=text, items=items, claims=claims, unparsed=unparsed)

    def _parse_secondary_relation(
        self,
        item: str,
        last_parts: list[str],
        last_desc: str | None,
        last_was_superlative: bool,
    ) -> list[SemanticClaim] | None:
        if not re.search(r"第二|第2|其次|次之|二长|第二长|第二个", item):
            return None
        parts = self.parts_in_text(item)
        if not parts:
            return None
        if last_was_superlative or not last_parts or last_desc is None:
            return []
        if last_desc == "short":
            return [SemanticClaim(kind="comparison", item=item, parts=parts, op=">", right_parts=last_parts)]
        return [SemanticClaim(kind="comparison", item=item, parts=last_parts, op=">", right_parts=parts)]

    def _parse_ordinal_sequence(self, items: list[str], normalized: str) -> list[SemanticClaim]:
        groups = ordinal_sequence_groups(items, self.parts)
        if not groups:
            return []
        if all(len(group) == 1 for group in groups):
            return [
                SemanticClaim(
                    kind="ranking",
                    item=normalized,
                    order=[group[0] for group in groups],
                    op="desc",
                )
            ]

        claims: list[SemanticClaim] = []
        for left, right in zip(groups[:-1], groups[1:]):
            claims.append(
                SemanticClaim(
                    kind="comparison",
                    item=normalized,
                    parts=left,
                    op=">",
                    right_parts=right,
                )
            )
        return claims

    def _is_meta_only(self, item: str) -> bool:
        cleaned = item
        for pattern in self.meta_patterns:
            cleaned = cleaned.replace(pattern, "")
        cleaned = re.sub(r"[，,。！？!？；;\s]", "", cleaned)
        return item != "" and cleaned == ""

    def _update_context_parts(self, context_parts: list[str], item: str) -> list[str]:
        out = list(context_parts)
        for part in self.parts_in_text(item):
            if part not in out:
                out.append(part)
        return out

    def _parts_from_claims(self, claims: list[SemanticClaim]) -> list[str]:
        out: list[str] = []
        for claim in claims:
            for part in [*claim.parts, *claim.right_parts, *claim.order]:
                if part in self.part_to_dim and part not in out:
                    out.append(part)
        return out

    def _parse_item(self, item: str) -> list[SemanticClaim]:
        parsers = (
            self._parse_superlative,
            self._parse_exclusive_case,
            self._parse_group_sum,
            self._parse_ranking,
            self._parse_body_reference,
            self._parse_comparison,
            self._parse_equality,
        )
        out: list[SemanticClaim] = []
        for parser in parsers:
            claims = parser(item)
            if claims:
                out.extend(claims)
        if not out:
            out.extend(self._parse_absolute(item))
        return out

    def _parse_unsupported(self, item: str) -> SemanticClaim | None:
        if re.search(r"两长两短|三长|一短|两个部位|三个部位|一个部位|奇数|偶数", item):
            return SemanticClaim(kind="count_abstract", item=item, supported=False, note="count abstract")
        if "不一样" in item or "不等于" in item or "各不相同" in item:
            return SemanticClaim(kind="inequality_disjoint", item=item, supported=False, note="disjoint region")
        return None

    def _parse_exclusion(self, item: str, next_item: str) -> list[SemanticClaim]:
        if is_negated_superlative(item) or is_negated_superlative(next_item):
            return []
        excluded = self.parts_in_text(item)
        if not excluded:
            return []
        desc = desc_label(next_item, allow_middle=True)
        is_equality = bool(re.search(EQUALITY_WORD_RE, next_item))
        if desc is None and not is_equality:
            return []
        remain = [part for part in self.parts if part not in excluded]
        if is_equality and desc is None:
            return [SemanticClaim(kind="equality", item=f"{item}，{next_item}", parts=remain, op="==")]
        claims = [
            SemanticClaim(kind="exclusion", item=f"{item}，{next_item}", parts=remain, desc=desc),
        ]
        if desc in {"long", "short"}:
            claims.append(SemanticClaim(kind="exclusion", item=f"{item}，{next_item}", parts=excluded, desc=opposite_desc(desc)))
        return claims

    def _parse_same_item_exclusion(self, item: str) -> list[SemanticClaim]:
        if is_negated_superlative(item):
            return []
        match = re.search(COMPLEMENT_RE, item)
        if "除" not in item or not match:
            return []
        excluded = self.parts_in_text(item)
        if not excluded:
            return []
        desc = desc_label(item[match.start() :], allow_middle=True)
        is_equality = bool(re.search(EQUALITY_WORD_RE, item[match.start() :]))
        if desc is None and not is_equality:
            return []
        remain = [part for part in self.parts if part not in excluded]
        if is_equality and desc is None:
            return [SemanticClaim(kind="equality", item=item, parts=remain, op="==")]
        claims = [SemanticClaim(kind="exclusion", item=item, parts=remain, desc=desc)]
        if desc in {"long", "short"}:
            claims.append(SemanticClaim(kind="exclusion", item=item, parts=excluded, desc=opposite_desc(desc)))
        return claims

    def _parse_contextual_endpoint(self, item: str, last_parts: list[str]) -> list[SemanticClaim]:
        if not last_parts or self.parts_in_text(item):
            return []
        if re.search(COMPLEMENT_RE, item):
            return []
        if has_reference_threshold(item):
            op = reference_relation_op(item)
            if not op:
                return []
            threshold = body_reference_threshold(item, self.body_value)
            return [
                SemanticClaim(kind="body_ref", item=item, parts=[part], op=op, threshold=threshold)
                for part in last_parts
            ]
        if re.search(r"没有|不是|未|不算|1/2|一半|二分之一|3/4|四分之三|可能|大概", item):
            return []
        if not self._is_absolute_endpoint(item):
            return []
        desc = desc_label(item, allow_middle=False)
        if desc is None:
            return []
        return [SemanticClaim(kind="general_case", item=item, parts=last_parts, desc=desc)]

    def _parse_other_comparison(self, item: str, last_parts: list[str]) -> list[SemanticClaim]:
        if not re.search(COMPLEMENT_RE, item):
            return []
        left_parts: list[str] = []
        op = ""

        match = re.search(rf"(?P<left>{self.part_group_re})?(?P<op>长于|短于|大于|小于|高于|低于|超过)(?:{COMPLEMENT_RE})(?:三个|部位)?", item)
        if match:
            left_parts = self.parts_in_text(match.group("left") or "") or last_parts
            op = relation_op(match.group("op"))
        else:
            match = re.search(rf"(?P<left>{self.part_group_re})?比(?:{COMPLEMENT_RE})(?:三个|部位)?.{{0,6}}?(?P<desc>长|短|高|低|大|小)", item)
            if match:
                left_parts = self.parts_in_text(match.group("left") or "") or last_parts
                op = relation_op("比", match.group("desc"))

        if not left_parts or op not in {">", "<"}:
            return []
        right_parts = [part for part in self.parts if part not in left_parts]
        if not right_parts:
            return []
        return [SemanticClaim(kind="comparison", item=item, parts=left_parts, op=op, right_parts=right_parts)]

    def _parse_complement(
        self,
        item: str,
        context_parts: list[str],
        last_parts: list[str],
        lookahead_parts: list[str],
    ) -> list[SemanticClaim]:
        if is_negated_superlative(item):
            return []
        match = re.search(COMPLEMENT_RE, item)
        if not match:
            return []
        if "除" in item:
            return []
        desc = desc_label(item[match.start() :], allow_middle=True)
        is_equality = bool(re.search(EQUALITY_WORD_RE, item[match.start() :]))
        has_reference = has_reference_threshold(item[match.start() :])
        if desc is None and not is_equality and not has_reference:
            return []

        explicit_parts = self.parts_in_text(item[: match.start()])
        excluded_options = [explicit_parts, context_parts, last_parts, lookahead_parts]
        excluded = next(
            (parts for parts in excluded_options if parts and [part for part in self.parts if part not in parts]),
            [],
        )
        if not excluded:
            return []
        complement_parts = [part for part in self.parts if part not in excluded]
        if not complement_parts:
            return []
        prefix_claims = self._parse_item(item[: match.start()].strip()) if explicit_parts else []
        if has_reference:
            op = reference_relation_op(item[match.start() :])
            if op:
                threshold = body_reference_threshold(item[match.start() :], self.body_value)
                return [
                    *prefix_claims,
                    *[
                        SemanticClaim(kind="body_ref", item=item, parts=[part], op=op, threshold=threshold)
                        for part in complement_parts
                    ],
                ]
        if is_equality and desc is None:
            return [
                *prefix_claims,
                SemanticClaim(kind="equality", item=item, parts=complement_parts, op="=="),
            ]
        return [
            *prefix_claims,
            SemanticClaim(kind="complement", item=item, parts=complement_parts, desc=desc),
        ]

    def _parse_superlative(self, item: str) -> list[SemanticClaim]:
        if self._is_absolute_endpoint(item):
            return []
        parsed = parse_superlative_relation(item, self.parts)
        if parsed is None:
            return []
        target, desc, right_parts = parsed
        op = ">" if desc in {"长", "大", "高"} else "<"
        return [
            SemanticClaim(kind="superlative", item=item, parts=[target], op=op, right_parts=[other])
            for other in right_parts
        ]

    def _is_absolute_endpoint(self, item: str) -> bool:
        return bool(
            re.search(
                r"达到.{0,4}(?:最|极)(?:长|短|大|小|高|低)|"
                r"(?:最大|最小|最高|最低|极长|极短)(?:的)?(?:长度|程度|值)|"
                r"最(?:长|短)(?:的)?长度",
                item,
            )
        )

    def _parse_exclusive_case(self, item: str) -> list[SemanticClaim]:
        if is_negated_superlative(item):
            return []
        if "只有" not in item:
            return []
        parts = self.parts_in_text(item)
        desc = desc_label(item, allow_middle=False)
        if not parts or desc is None:
            return []
        others = [part for part in self.parts if part not in parts]
        return [
            SemanticClaim(kind="exclusive_case", item=item, parts=parts, desc=desc),
            SemanticClaim(kind="exclusive_case", item=item, parts=others, desc=opposite_desc(desc)),
        ]

    def _parse_group_sum(self, item: str) -> list[SemanticClaim]:
        if not re.search(r"总和|之和|加起来|合起来|组合|比例|加", item):
            return []

        if "比" in item:
            left_text, right_text = item.split("比", 1)
            desc = desc_label(right_text, allow_middle=False)
            if desc is None:
                return []
            op = ">" if desc == "long" else "<"
            return self._group_claim_from_text(item, left_text, op, right_text)

        pattern = re.compile(
            rf"(?P<left>.+?)(?P<op>大于等于|小于等于|大于|小于|长于|短于|高于|低于|等于|约等于)(?P<right>.+)"
        )
        match = pattern.search(item)
        if not match:
            return []
        op = relation_op(match.group("op"))
        if not op:
            return []
        return self._group_claim_from_text(item, match.group("left"), op, match.group("right"))

    def _group_claim_from_text(self, item: str, left_text: str, op: str, right_text: str) -> list[SemanticClaim]:
        left = self.parts_in_text(left_text)
        right = self.parts_in_text(right_text)
        if not left or not right:
            return []
        return [SemanticClaim(kind="group_sum", item=item, parts=left, op=op, right_parts=right)]

    def _parse_ranking(self, item: str) -> list[SemanticClaim]:
        if not re.search(r"从小到大|从大到小|从短到长|从长到短|排序|长短顺序", item):
            return []
        parts = self.parts_in_text(item)
        if len(parts) < 2:
            return []
        if re.search(r"从小到大|从短到长", item):
            order = parts
        elif re.search(r"从大到小|从长到短|长短顺序", item):
            order = parts
        else:
            return []
        direction = "asc" if re.search(r"从小到大|从短到长", item) else "desc"
        return [SemanticClaim(kind="ranking", item=item, order=order, op=direction)]

    def _parse_body_reference(self, item: str) -> list[SemanticClaim]:
        if not has_reference_threshold(item):
            return []
        parts = list(self.parts) if all_parts_phrase(item) else self.parts_in_text(item)
        if not parts:
            return []
        threshold = body_reference_threshold(item, self.body_value)
        op = reference_relation_op(item)
        if not op:
            return []
        return [
            SemanticClaim(kind="body_ref", item=item, parts=[part], op=op, threshold=threshold)
            for part in parts
        ]

    def _parse_comparison(self, item: str) -> list[SemanticClaim]:
        claims: list[SemanticClaim] = []

        for match in re.finditer(
            rf"(?P<left>{self.part_group_re})比(?P<right>{self.part_group_re}|{BODY_RE}).{{0,8}}?(?P<desc>长|短|高|低|大|小)",
            item,
        ):
            if re.search(BODY_RE, match.group("right")):
                continue
            op = relation_op("比", match.group("desc"))
            if op:
                claims.append(
                    SemanticClaim(
                        kind="comparison",
                        item=item,
                        parts=self.parts_in_text(match.group("left")),
                        op=op,
                        right_parts=self.parts_in_text(match.group("right")),
                    )
                )

        relation_pattern = re.compile(
            rf"(?P<left>{self.part_group_re})(?:{COMPARISON_MODIFIER_RE})?(?P<op>大于等于|小于等于|大于|小于|长于|短于|高于|低于|超过|等于|不等于|约等于|相等|持平)(?P<right>{self.part_group_re}|{BODY_RE})"
        )
        first_left: list[str] | None = None
        for match in relation_pattern.finditer(item):
            if re.search(BODY_RE, match.group("right")):
                continue
            op = relation_op(match.group("op"))
            if not op:
                continue
            left = self.parts_in_text(match.group("left"))
            right = self.parts_in_text(match.group("right"))
            first_left = first_left or left
            claims.append(SemanticClaim(kind="comparison", item=item, parts=left, op=op, right_parts=right))

        if first_left:
            chain_pattern = re.compile(
                rf"[、，,](?P<op>大于等于|小于等于|大于|小于|长于|短于|高于|低于|超过|等于|不等于|约等于)(?P<right>{self.part_group_re})"
            )
            for match in chain_pattern.finditer(item):
                op = relation_op(match.group("op"))
                if op:
                    claims.append(
                        SemanticClaim(
                            kind="chained_comparison",
                            item=item,
                            parts=first_left,
                            op=op,
                            right_parts=self.parts_in_text(match.group("right")),
                        )
                    )

        omitted_left_pattern = re.compile(
            rf"(?:[、，,]|^)(?:{COMPARISON_MODIFIER_RE})?(?P<op>大于等于|小于等于|大于|小于|长于|短于|高于|低于|超过|等于|不等于|约等于|相等|持平)(?P<right>{self.part_group_re}|{BODY_RE})"
        )
        for match in omitted_left_pattern.finditer(item):
            if re.search(BODY_RE, match.group("right")):
                continue
            left = self.parts_in_text(item[: match.start()])
            right = self.parts_in_text(match.group("right"))
            if not left or not right:
                continue
            op = relation_op(match.group("op"))
            if not op:
                continue
            prefix_desc = desc_label(item[: match.start()], allow_middle=True)
            if prefix_desc is not None:
                claims.append(SemanticClaim(kind="general_case", item=item[: match.start()], parts=left, desc=prefix_desc))
            claims.append(SemanticClaim(kind="comparison", item=item, parts=left, op=op, right_parts=right))

        return claims

    def _parse_equality(self, item: str) -> list[SemanticClaim]:
        if not re.search(EQUALITY_WORD_RE, item):
            return []
        if re.search(INEQUALITY_WORD_RE, item):
            return [SemanticClaim(kind="inequality_disjoint", item=item, supported=False, note="disjoint region")]
        parts = list(self.parts) if all_parts_phrase(item) else self.parts_in_text(item)
        if len(parts) < 2:
            return []
        return [SemanticClaim(kind="equality", item=item, parts=parts, op="==")]

    def _parse_absolute(self, item: str) -> list[SemanticClaim]:
        if re.search(r"比(?!较)|大于|小于|等于|长于|短于|高于|低于|超过|排序", item):
            return []
        desc = desc_label(item, allow_middle=True)
        if desc is None:
            return []
        parts = list(self.parts) if all_parts_phrase(item) else self.parts_in_text(item)
        if not parts:
            return []
        kind = "universal_quantifier" if all_parts_phrase(item) else "general_case"
        return [SemanticClaim(kind=kind, item=item, parts=parts, desc=desc)]


class RegionEncoder:
    """Encode semantic claims as model-form region constraints ``A @ x <= b``."""

    def __init__(
        self,
        long_threshold: float = 0.5,
        short_threshold: float = 0.5,
        middle_lower: float = 0.25,
        middle_upper: float = 0.75,
        comparison_margin: float = 0.0,
        equality_epsilon: float = 0.10,
        parts: tuple[str, ...] | None = None,
    ):
        self.parts = tuple(parts or PARTS)
        self.part_to_dim = part_to_dim(self.parts)
        self.n_dims = len(self.parts)
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.middle_lower = float(middle_lower)
        self.middle_upper = float(middle_upper)
        self.comparison_margin = float(comparison_margin)
        self.equality_epsilon = float(equality_epsilon)

    def encode_text(self, text: Any, parser: SemanticParser | None = None) -> tuple[list[list[float]], list[float], list[str], list[str]]:
        parser = parser or SemanticParser(parts=self.parts)
        return self.encode(parser.parse(text))

    def encode(self, parsed: SemanticResult) -> tuple[list[list[float]], list[float], list[str], list[str]]:
        constraints: list[tuple[np.ndarray, float]] = []
        matched_rules: list[str] = []
        unparsed = list(parsed.unparsed)

        for claim in parsed.claims:
            cons = self._claim_constraints(claim)
            if cons:
                constraints.extend(cons)
                matched_rules.append(claim.kind)
            elif not claim.supported and claim.item not in unparsed:
                unparsed.append(claim.item)

        return self._merge_constraints(constraints) + (matched_rules, unparsed)

    def _claim_constraints(self, claim: SemanticClaim) -> list[tuple[np.ndarray, float]]:
        if not claim.supported:
            return []
        if claim.kind in {"general_case", "universal_quantifier", "exclusive_case", "exclusion", "complement"}:
            return self._desc_constraints(claim.parts, claim.desc)
        if claim.kind in {"comparison", "chained_comparison", "superlative"}:
            return self._comparison_constraints(claim.parts, claim.op, claim.right_parts)
        if claim.kind == "body_ref":
            return self._threshold_constraints(claim.parts, claim.op, claim.threshold)
        if claim.kind == "equality":
            return self._equality_constraints(claim.parts)
        if claim.kind == "ranking":
            return self._ranking_constraints(claim.order, claim.op)
        if claim.kind == "group_sum":
            return self._group_sum_constraints(claim.parts, claim.op, claim.right_parts)
        return []

    def _row(self, weights: dict[str, float]) -> np.ndarray:
        row = np.zeros(self.n_dims, dtype=float)
        for part, weight in weights.items():
            row[self.part_to_dim[part]] += float(weight)
        return row

    def _desc_constraints(self, parts: list[str], desc: str | None) -> list[tuple[np.ndarray, float]]:
        out: list[tuple[np.ndarray, float]] = []
        if desc is None:
            return out
        for part in parts:
            if desc == "long":
                out.append((self._row({part: -1.0}), -self.long_threshold))
            elif desc == "short":
                out.append((self._row({part: 1.0}), self.short_threshold))
            elif desc == "middle":
                out.append((self._row({part: -1.0}), -self.middle_lower))
                out.append((self._row({part: 1.0}), self.middle_upper))
        return out

    def _threshold_constraints(self, parts: list[str], op: str | None, threshold: float | None) -> list[tuple[np.ndarray, float]]:
        if op is None or threshold is None:
            return []
        out: list[tuple[np.ndarray, float]] = []
        for part in parts:
            if op == ">":
                out.append((self._row({part: -1.0}), -float(threshold)))
            elif op == "<":
                out.append((self._row({part: 1.0}), float(threshold)))
            elif op == "==":
                eps = self.equality_epsilon
                out.append((self._row({part: -1.0}), -float(threshold) + eps))
                out.append((self._row({part: 1.0}), float(threshold) + eps))
        return out

    def _comparison_constraints(self, left_parts: list[str], op: str | None, right_parts: list[str]) -> list[tuple[np.ndarray, float]]:
        if op not in {">", "<", "=="} or not left_parts or not right_parts:
            return []
        if op == "==":
            return self._equality_constraints(left_parts + right_parts)
        out: list[tuple[np.ndarray, float]] = []
        for left in left_parts:
            for right in right_parts:
                if op == ">":
                    out.append((self._row({right: 1.0, left: -1.0}), -self.comparison_margin))
                elif op == "<":
                    out.append((self._row({left: 1.0, right: -1.0}), -self.comparison_margin))
        return out

    def _equality_constraints(self, parts: list[str]) -> list[tuple[np.ndarray, float]]:
        unique_parts = list(dict.fromkeys(parts))
        if len(unique_parts) < 2:
            return []
        out: list[tuple[np.ndarray, float]] = []
        for left, right in itertools.combinations(unique_parts, 2):
            out.append((self._row({left: 1.0, right: -1.0}), self.equality_epsilon))
            out.append((self._row({right: 1.0, left: -1.0}), self.equality_epsilon))
        return out

    def _ranking_constraints(self, order: list[str], direction: str | None) -> list[tuple[np.ndarray, float]]:
        if len(order) < 2:
            return []
        out: list[tuple[np.ndarray, float]] = []
        for first, second in zip(order[:-1], order[1:]):
            if direction == "asc":
                out.append((self._row({first: 1.0, second: -1.0}), -self.comparison_margin))
            else:
                out.append((self._row({second: 1.0, first: -1.0}), -self.comparison_margin))
        return out

    def _group_sum_constraints(self, left_parts: list[str], op: str | None, right_parts: list[str]) -> list[tuple[np.ndarray, float]]:
        if op not in {">", "<", "=="} or not left_parts or not right_parts:
            return []
        weights = {part: 0.0 for part in self.parts}
        for part in left_parts:
            weights[part] += 1.0
        for part in right_parts:
            weights[part] -= 1.0
        row = self._row(weights)
        if op == ">":
            return [(-row, -self.comparison_margin)]
        if op == "<":
            return [(row, -self.comparison_margin)]
        return [(row, self.equality_epsilon), (-row, self.equality_epsilon)]

    def _merge_constraints(self, constraints: list[tuple[np.ndarray, float]]) -> tuple[list[list[float]], list[float]]:
        if not constraints:
            return [], []
        dedup = []
        seen = set()
        for row, rhs in constraints:
            key = (tuple(np.round(row.astype(float), 8)), round(float(rhs), 8))
            if key not in seen:
                seen.add(key)
                dedup.append((row.astype(float), float(rhs)))
        return [row.tolist() for row, _ in dedup], [rhs for _, rhs in dedup]


class CenterEncoder:
    rule_names = [
        "exclusion",
        "superlative",
        "universal_quantifier",
        "exclusive_case",
        "complement",
        "comparison",
        "general_case",
        "addition",
        "equality",
        "ranking",
        "body_ref",
        "group_sum",
        "meta",
        "unsupported",
    ]

    def __init__(self, body_value: float = 0.5, parts: tuple[str, ...] | None = None):
        self.body_value = float(body_value)
        self.parts = tuple(parts or PARTS)
        self.part_to_dim = part_to_dim(self.parts)
        self.n_dims = len(self.parts)

    def encode_text(self, text: Any, parser: SemanticParser | None = None) -> tuple[dict[str, list[float | None]], list[str], list[str]]:
        parser = parser or SemanticParser(body_value=self.body_value, parts=self.parts)
        return self.encode(parser.parse(text))

    def encode(self, parsed: SemanticResult) -> tuple[dict[str, list[float | None]], list[str], list[str]]:
        per_rule = {rule: [None] * self.n_dims for rule in self.rule_names}
        anchors: list[list[float | None]] = []
        equality_groups: list[list[str]] = []
        unparsed = list(parsed.unparsed)
        matched_rules: list[str] = []

        for claim in parsed.claims:
            if claim.supported and (
                claim.kind == "equality"
                or (claim.kind in {"comparison", "chained_comparison"} and claim.op == "==")
            ):
                equality_groups.append(claim.parts + claim.right_parts)
                matched_rules.append(claim.kind)
                continue

            vector = self._claim_vector(claim)
            if vector is not None:
                rule = self._center_rule_name(claim.kind)
                per_rule[rule] = self._merge_vectors(per_rule[rule], vector)
                anchors.append(vector)
                matched_rules.append(claim.kind)
            elif not claim.supported and claim.item not in unparsed:
                unparsed.append(claim.item)
                per_rule["unsupported"] = self._merge_vectors(per_rule["unsupported"], [None] * self.n_dims)

        for group in equality_groups:
            group_vector = self._equality_group_vector(group, anchors)
            per_rule["equality"] = self._merge_vectors(per_rule["equality"], group_vector)
            anchors.append(group_vector)

        all_vector = self._merge_vectors(*anchors) if anchors else [None] * self.n_dims
        if any(value is not None for value in all_vector):
            all_vector = [0.5 if value is None else value for value in all_vector]
        per_rule["all"] = all_vector
        return per_rule, matched_rules, unparsed

    def _center_rule_name(self, kind: str) -> str:
        if kind == "chained_comparison":
            return "comparison"
        if kind in self.rule_names:
            return kind
        return "unsupported"

    def _claim_vector(self, claim: SemanticClaim) -> list[float | None] | None:
        if not claim.supported:
            return None
        if claim.kind in {"general_case", "universal_quantifier", "exclusive_case", "exclusion", "complement"}:
            return self._desc_vector(claim.parts, claim.desc)
        if claim.kind in {"comparison", "chained_comparison"}:
            return self._comparison_vector(claim.parts, claim.op, claim.right_parts)
        if claim.kind == "superlative":
            return self._superlative_vector(claim.parts, claim.op, claim.right_parts)
        if claim.kind == "equality":
            return self._equality_group_vector(claim.parts, [])
        if claim.kind == "ranking":
            return self._ranking_vector(claim.order, claim.op)
        if claim.kind == "body_ref":
            return self._body_ref_vector(claim.parts, claim.op, claim.threshold)
        if claim.kind == "group_sum":
            return self._group_sum_vector(claim.parts, claim.op, claim.right_parts)
        return None

    def _empty(self) -> list[float | None]:
        return [None] * self.n_dims

    def _desc_vector(self, parts: list[str], desc: str | None) -> list[float | None] | None:
        if desc is None:
            return None
        value = {"long": 0.75, "short": 0.25, "middle": 0.5}.get(desc)
        if value is None:
            return None
        vec = self._empty()
        for part in parts:
            vec[self.part_to_dim[part]] = value
        return vec

    def _comparison_vector(self, left_parts: list[str], op: str | None, right_parts: list[str]) -> list[float | None] | None:
        if op not in {">", "<", "=="} or not left_parts or not right_parts:
            return None
        if op == "==":
            return self._equality_group_vector(left_parts + right_parts, [])
        vec = self._empty()
        left_value, right_value = (2 / 3, 1 / 3) if op == ">" else (1 / 3, 2 / 3)
        for part in left_parts:
            vec[self.part_to_dim[part]] = left_value
        for part in right_parts:
            vec[self.part_to_dim[part]] = right_value
        return vec

    def _superlative_vector(self, parts: list[str], op: str | None, right_parts: list[str]) -> list[float | None] | None:
        if not parts or not right_parts:
            return None
        vec = self._empty()
        target_value, other_value = (0.8, 0.4) if op == ">" else (0.4, 0.8)
        for part in parts:
            vec[self.part_to_dim[part]] = target_value
        for part in right_parts:
            vec[self.part_to_dim[part]] = other_value
        return vec

    def _equality_group_vector(self, parts: list[str], anchors: list[list[float | None]]) -> list[float | None]:
        vec = self._empty()
        values = []
        for anchor in anchors:
            for part in parts:
                value = anchor[self.part_to_dim[part]]
                if value is not None:
                    values.append(float(value))
        group_value = float(np.mean(values)) if values else 0.5
        for part in parts:
            vec[self.part_to_dim[part]] = group_value
        return vec

    def _ranking_vector(self, order: list[str], direction: str | None) -> list[float | None] | None:
        if len(order) < 2:
            return None
        values = [(idx + 1) / (len(order) + 1) for idx in range(len(order))]
        if direction != "asc":
            values = list(reversed(values))
        vec = self._empty()
        for part, value in zip(order, values):
            vec[self.part_to_dim[part]] = float(value)
        return vec

    def _body_ref_vector(self, parts: list[str], op: str | None, threshold: float | None) -> list[float | None] | None:
        if op not in {">", "<", "=="} or threshold is None:
            return None
        threshold = float(threshold)
        if op == ">":
            value = (threshold + 1.0) / 2.0
        elif op == "<":
            value = threshold / 2.0
        else:
            value = threshold
        vec = self._empty()
        for part in parts:
            vec[self.part_to_dim[part]] = float(value)
        return vec

    def _group_sum_vector(self, left_parts: list[str], op: str | None, right_parts: list[str]) -> list[float | None] | None:
        if op not in {">", "<", "=="} or not left_parts or not right_parts:
            return None
        vec = self._empty()
        if op == "==":
            for part in left_parts + right_parts:
                vec[self.part_to_dim[part]] = 0.5
            return vec
        left_value, right_value = (2 / 3, 1 / 3) if op == ">" else (1 / 3, 2 / 3)
        for part in left_parts:
            vec[self.part_to_dim[part]] = left_value
        for part in right_parts:
            vec[self.part_to_dim[part]] = right_value
        return vec

    def _merge_vectors(self, *vectors: list[float | None]) -> list[float | None]:
        if not vectors:
            return [None] * self.n_dims
        out: list[float | None] = [None] * self.n_dims
        for dim in range(self.n_dims):
            values = [float(vec[dim]) for vec in vectors if vec is not None and vec[dim] is not None]
            if values:
                out[dim] = float(np.mean(values))
        return out


class Recording_Processor_Region:
    """
    Encode oral reports as region constraints A @ x <= b.

    The dimension order follows ``parts``. Default: neck, head, leg, tail.
    """

    def __init__(
        self,
        long_threshold: float = 0.5,
        short_threshold: float = 0.5,
        middle_lower: float = 0.25,
        middle_upper: float = 0.75,
        comparison_margin: float = 0.0,
        use_average_in_addition: bool = False,
        parts: tuple[str, ...] | None = None,
    ):
        self.parts = tuple(parts or PARTS)
        self.body_parts = part_to_dim(self.parts)
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.middle_lower = float(middle_lower)
        self.middle_upper = float(middle_upper)
        self.comparison_margin = float(comparison_margin)
        self.use_average_in_addition = use_average_in_addition
        self.semantic_parser = SemanticParser(body_value=0.5, parts=self.parts)
        self.region_encoder = RegionEncoder(
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
            middle_lower=self.middle_lower,
            middle_upper=self.middle_upper,
            comparison_margin=self.comparison_margin,
            equality_epsilon=0.10,
            parts=self.parts,
        )

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        results = {
            "iSession": [],
            "iTrial": [],
            "text": [],
            "A": [],
            "b": [],
            "n_constraints": [],
            "matched_rules": [],
            "un_pro": [],
        }
        for _, row in df.iterrows():
            text = row["text"] if "text" in row else None
            A, b, matched_rules, un_pro = self.extract_region(text)
            results["iSession"].append(row["iSession"])
            results["iTrial"].append(row["iTrial"])
            results["text"].append(text)
            results["A"].append(A)
            results["b"].append(b)
            results["n_constraints"].append(len(b))
            results["matched_rules"].append(matched_rules)
            results["un_pro"].append(un_pro)
        return pd.DataFrame(results)

    def extract_region(self, text: Any) -> tuple[list[list[float]], list[float], list[str], list[str]]:
        parsed = self.semantic_parser.parse(text)
        return self.region_encoder.encode(parsed)


class Recording_Processor_Center:
    """Encode oral reports as four-dimensional centers. The dimension order follows ``parts``."""

    def __init__(self, parts: tuple[str, ...] | None = None):
        self.parts = tuple(parts or PARTS)
        self.body_parts = part_to_dim(self.parts)
        self.semantic_parser = SemanticParser(body_value=0.5, parts=self.parts)
        self.center_encoder = CenterEncoder(body_value=0.5, parts=self.parts)

    def process_use(self, df: pd.DataFrame) -> pd.DataFrame:
        results = {
            "iSession": df["iSession"],
            "iTrial": df["iTrial"],
        }
        use_columns = {part: f"feature{idx + 1}_oraluse" for idx, part in enumerate(self.parts)}
        for col in use_columns.values():
            results[col] = [0] * len(df)
        for idx, text in enumerate(df["text"]):
            if pd.isna(text) or not str(text).strip():
                continue
            text = self.semantic_parser.normalize_text(text)
            for part, col in use_columns.items():
                if part in text:
                    results[col][idx] = 1
        return pd.DataFrame(results)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        rule_columns = list(self.center_encoder.rule_names) + ["all"]
        results = {col: [] for col in rule_columns}
        results.update({"un_pro": [], "text": []})

        for text in df["text"]:
            encoded, un_pro = self.extract_values(text)
            for col in rule_columns:
                results[col].append(encoded.get(col, [None] * len(self.parts)))
            results["un_pro"].append(un_pro)
            results["text"].append(text)

        result_df = pd.DataFrame(results)
        result_df["iSession"] = df["iSession"].to_numpy()
        result_df["iTrial"] = df["iTrial"].to_numpy()
        columns = ["iSession", "iTrial"] + [col for col in result_df.columns if col not in ["iSession", "iTrial"]]
        return result_df[columns]

    def extract_values(self, text: Any) -> tuple[dict[str, list[float | None]], list[str]]:
        parsed = self.semantic_parser.parse(text)
        encoded, matched_rules, un_pro = self.center_encoder.encode(parsed)
        return encoded, un_pro


class FidelityAnalyzer:
    """Evaluate whether parsed oral-report claims match trial features."""

    def __init__(
        self,
        parser: SemanticParser | None = None,
        eq_eps: float = 0.10,
        tol: float = 1e-9,
        body_value: float = 0.5,
    ):
        self.parser = parser or SemanticParser(body_value=body_value)
        self.eq_eps = float(eq_eps)
        self.tight_eq_eps = min(float(eq_eps), 0.06)
        self.tol = float(tol)
        self.meta_patterns = ("选错了", "不确定", "不知道", "随便选", "没什么区别", "看不出")

    def analyze_row(self, row: pd.Series) -> dict[str, Any]:
        text = normalize_text(row.get("text"))
        values = part_values_from_row(row)
        parsed = self.parser.parse(text)
        claims = self.evaluate(parsed, values)

        if claims:
            score = float(np.mean([claim.passed for claim in claims]))
            status = "ok" if math.isclose(score, 1.0) else "mismatch"
            return {
                "fidelity": score,
                "status": status,
                "n_claims": len(claims),
                "claim_labels": [claim.label for claim in claims],
                "failed_claims": [claim.label for claim in claims if not claim.passed],
                "style_tags": self.style_tags(text),
            }

        legacy_score = self.legacy_region_score(row)
        if not math.isnan(legacy_score):
            return {
                "fidelity": legacy_score,
                "status": "legacy_region",
                "n_claims": 0,
                "claim_labels": [],
                "failed_claims": [],
                "style_tags": self.style_tags(text),
            }

        status = "empty"
        if text and self.is_meta_only(text):
            status = "meta"
        elif text:
            status = "unsupported"
        return {
            "fidelity": np.nan,
            "status": status,
            "n_claims": 0,
            "claim_labels": [],
            "failed_claims": [],
            "style_tags": self.style_tags(text),
        }

    def evaluate(self, parsed: SemanticResult, values: dict[str, float]) -> list[FidelityClaim]:
        claims: list[FidelityClaim] = []
        for claim in parsed.claims:
            if not claim.supported:
                continue
            if claim.kind in {"general_case", "universal_quantifier", "exclusive_case", "exclusion", "complement"}:
                self._add_desc_claims(claims, values, claim.parts, claim.desc, self._desc_source(claim))
            elif claim.kind in {"comparison", "chained_comparison", "superlative"}:
                self._add_binary_claim(claims, values, claim.parts, claim.op or "", claim.right_parts, source=claim.kind)
            elif claim.kind == "body_ref":
                self._add_threshold_claims(claims, values, claim)
            elif claim.kind == "equality":
                self._add_equality_claim(claims, values, claim.parts, source="equality")
            elif claim.kind == "ranking":
                self._add_ranking_claims(claims, values, claim)
            elif claim.kind == "group_sum":
                self._add_group_sum_claim(claims, values, claim)
        return claims

    def _desc_source(self, claim: SemanticClaim) -> str:
        if claim.kind in {"general_case", "universal_quantifier"}:
            if claim.desc in {"long", "short"}:
                return f"absolute_{claim.desc}"
            return "absolute"
        return claim.kind

    def _add_binary_claim(
        self,
        claims: list[FidelityClaim],
        values: dict[str, float],
        left_parts: list[str],
        op: str,
        right_parts: list[str] | None = None,
        right_value: float | None = None,
        source: str = "comparison",
        eq_eps: float | None = None,
    ) -> None:
        if not left_parts or op not in {">", "<", "==", "!="}:
            return
        eq_eps = self.eq_eps if eq_eps is None else float(eq_eps)
        left = float(np.mean([values[p] for p in left_parts]))
        if right_value is None:
            if not right_parts:
                return
            right = float(np.mean([values[p] for p in right_parts]))
            right_label = "+".join(right_parts) if len(right_parts) > 1 else right_parts[0]
        else:
            right = float(right_value)
            right_label = f"{right:.2f}"

        passed, distance = compare_values(left, op, right, self.tol, eq_eps)
        left_label = "+".join(left_parts) if len(left_parts) > 1 else left_parts[0]
        label = f"{source}:{left_label} {describe_op(op)} {right_label}"
        claims.append(FidelityClaim(source, label, distance, passed))

    def _add_desc_claims(
        self,
        claims: list[FidelityClaim],
        values: dict[str, float],
        parts: list[str],
        desc: str | None,
        source: str,
    ) -> None:
        if not parts or desc is None:
            return
        if desc == "long":
            for part in parts:
                self._add_binary_claim(claims, values, [part], ">", right_value=0.5, source=source)
        elif desc == "short":
            for part in parts:
                self._add_binary_claim(claims, values, [part], "<", right_value=0.5, source=source)
        elif desc == "middle":
            for part in parts:
                lower_passed, lower_dist = compare_values(values[part], ">", 0.25, self.tol, self.eq_eps)
                upper_passed, upper_dist = compare_values(values[part], "<", 0.75, self.tol, self.eq_eps)
                claims.append(FidelityClaim(source, f"{source}:{part} middle_lower", lower_dist, lower_passed))
                claims.append(FidelityClaim(source, f"{source}:{part} middle_upper", upper_dist, upper_passed))

    def _add_threshold_claims(
        self,
        claims: list[FidelityClaim],
        values: dict[str, float],
        claim: SemanticClaim,
    ) -> None:
        if claim.op is None or claim.threshold is None:
            return
        for part in claim.parts:
            self._add_binary_claim(
                claims,
                values,
                [part],
                claim.op,
                right_value=claim.threshold,
                source="body_ref",
                eq_eps=self.tight_eq_eps if claim.op == "==" else None,
            )

    def _add_equality_claim(
        self,
        claims: list[FidelityClaim],
        values: dict[str, float],
        parts: list[str],
        source: str,
    ) -> None:
        unique_parts = list(dict.fromkeys(parts))
        if len(unique_parts) < 2:
            return
        observed = [values[p] for p in unique_parts]
        spread = float(np.max(observed) - np.min(observed))
        passed = bool(spread <= self.eq_eps)
        label = f"{source}_range:{'+'.join(unique_parts)} ="
        claims.append(FidelityClaim(source, label, spread, passed))

    def _add_ranking_claims(
        self,
        claims: list[FidelityClaim],
        values: dict[str, float],
        claim: SemanticClaim,
    ) -> None:
        if len(claim.order) < 2:
            return
        op = "<" if claim.op == "asc" else ">"
        for left, right in zip(claim.order[:-1], claim.order[1:]):
            self._add_binary_claim(claims, values, [left], op, [right], source="ranking")

    def _add_group_sum_claim(
        self,
        claims: list[FidelityClaim],
        values: dict[str, float],
        claim: SemanticClaim,
    ) -> None:
        if claim.op not in {">", "<", "==", "!="} or not claim.parts or not claim.right_parts:
            return
        left = float(np.sum([values[p] for p in claim.parts]))
        right = float(np.sum([values[p] for p in claim.right_parts]))
        passed, distance = compare_values(left, claim.op, right, self.tol, self.eq_eps)
        label = f"group_sum:{'+'.join(claim.parts)} {describe_op(claim.op)} {'+'.join(claim.right_parts)}"
        claims.append(FidelityClaim("group_sum", label, distance, passed))

    def legacy_region_score(self, row: pd.Series) -> float:
        A = parse_literal(row.get("oral_A"))
        b = parse_literal(row.get("oral_b"))
        if not isinstance(A, list) or not isinstance(b, list) or len(A) == 0 or len(b) == 0:
            return float("nan")
        try:
            A_arr = np.asarray(A, dtype=float)
            b_arr = np.asarray(b, dtype=float).reshape(-1)
            x = feature_vector_from_row(row)
        except (TypeError, ValueError):
            return float("nan")
        if A_arr.ndim != 2 or b_arr.ndim != 1 or A_arr.shape[0] != b_arr.shape[0]:
            return float("nan")
        return float(np.mean((A_arr @ x) <= (b_arr + self.tol)))

    def is_meta_only(self, text: str) -> bool:
        if not text:
            return False
        cleaned = text
        for phrase in self.meta_patterns:
            cleaned = cleaned.replace(phrase, "")
        cleaned = re.sub(r"[，,。！？!？；;\s]", "", cleaned)
        return cleaned == ""

    def style_tags(self, text: str) -> list[str]:
        if not text:
            return ["empty"]
        checks = {
            "meta": r"选错|不确定|不知道|随便|看不出",
            "direct_absolute": r"长|短|中等|适中|一般",
            "superlative": r"最长|最短|最大|最小|最高|最低",
            "comparison": r"比|大于|小于|长于|短于|高于|低于|超过",
            "equality": r"一样|差不多|等于|相等|相近|接近|相似|相当|均匀|均衡|平均|一致|(?<!中)等长",
            "ranking": r"排序|从大到小|从小到大|从长到短|从短到长|第一|第二|第三|第四|第[一二三四1234]|其次|次之|最后|再是|然后|接着",
            "body_ref": r"躯干|身体|一半|3/4|四分之三",
            "group_sum": r"总和|之和|加起来|合起来|组合|比例",
            "count_abstract": r"两长两短|三长|一短|两个部位|三个部位|一个部位|奇数|偶数",
            "negation": r"不是|并非|没有|未|不算|不太|不够|不怎么|不一样|不长|不短",
        }
        tags = [name for name, pattern in checks.items() if re.search(pattern, text)]
        return tags or ["other"]
