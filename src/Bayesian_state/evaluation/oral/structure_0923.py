"""Conservative, auditable report coding for 0923; old encodings stay unchanged.

Only whole clauses accepted by the explicit grammar are scored. The shared
parser supplies review suggestions for all other text, never automatic labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
import re
from typing import Any

import numpy as np
from scipy.optimize import linprog

from src.oral_coding import (
    FEATURE_NAME_TO_PART, RegionEncoder, SemanticClaim, SemanticParser,
    SemanticResult, normalize_text, parts_in_text, split_items,
)
from ...model.oral_report_space import body_token, component_tokens, facet_token, Report

VERSION = "0923-oral-structure-v1"
BODY = r"(?:躯干|身体)"
OP = r"(?:长于|短于|大于|小于|高于|低于)"
MOD = r"(?:都|均|也|的长度|长度|很|比较|非常|特别|较|略|稍微|稍|偏|挺|更)*"


@dataclass(frozen=True)
class StructuredReport:
    status: str
    tokens: Report
    relations: tuple[dict[str, Any], ...]
    unparsed: tuple[str, ...]
    flags: tuple[str, ...]
    review_suggestions: tuple[dict[str, Any], ...]
    legacy_body_direction_disagreements: int


def _direction(operator: str) -> str:
    return "gt" if operator in {"长于", "大于", "高于", "长", "大", "高"} else "lt"


def _invert(direction: str) -> str:
    return "lt" if direction == "gt" else "gt"


def encode_report(text: Any, feature_names: tuple[str, ...]) -> StructuredReport:
    """Encode without choice, feedback, correctness or fitted model state.

    Bare long/short use the existing 0.5 coding convention. Body comparisons
    remain symbolic. Unequal weighted sums retain coefficients but require a
    verified physical affine transform before numerical geometry is assigned.
    """
    if len(feature_names) != 4 or len(set(feature_names)) != 4:
        raise ValueError("four distinct ordered feature names are required")
    try:
        parts = tuple(FEATURE_NAME_TO_PART[name] for name in feature_names)
    except KeyError as error:
        raise ValueError("unknown feature name") from error
    normalized = normalize_text(text, parts)
    if not normalized or normalized.lower() == "nan":
        return StructuredReport("missing", (), (), (), (), (), 0)
    parser = SemanticParser(parts=parts)
    encoder = RegionEncoder(parts=parts)
    part = "(?:" + "|".join(map(re.escape, parts)) + ")"
    group = rf"{part}(?:[、和与及跟]+{part})*"
    endpoint = rf"(?:{BODY}|{group})"
    # A comma-like delimiter introduces a repeated comparison, not a new
    # comparator. e.g. 躯干短于脖子、长于尾巴 has the same body subject twice.
    items = split_items(re.sub(rf"、(?={OP})", "，", normalized))
    tokens: list[str] = []
    relations: list[dict[str, Any]] = []
    unparsed: list[str] = []
    flags: set[str] = set()
    suggestions: list[dict[str, Any]] = []
    old_disagreements = 0
    previous_body_subject = False

    def record_claim(claim: SemanticClaim) -> None:
        relations.append(asdict(claim))
        A, b, _, rejected = encoder.encode(SemanticResult(claim.item, [claim.item], [claim], []))
        if not A or rejected:
            unparsed.append(claim.item)
        else:
            tokens.extend(component_tokens(np.asarray(A), np.asarray(b)))

    for original in items:
        item = re.sub(r"^(?:且|并且|而且)", "", original).strip()
        if previous_body_subject and re.fullmatch(rf"{OP}{group}", item):
            item = "躯干" + item
        previous_body_subject = False
        # Strict linear comparisons, including body on either side.
        match = re.fullmatch(rf"(?P<left>{endpoint})(?P<op>{OP})(?P<right>{endpoint})", item)
        if match is None:
            match = re.fullmatch(rf"(?P<left>{endpoint})比(?P<right>{endpoint}){MOD}(?P<op>长|短|大|小|高|低)(?:一点|一些)?", item)
        if match is not None:
            left, right = match['left'], match['right']
            direction = _direction(match['op'])
            left_body, right_body = bool(re.fullmatch(BODY, left)), bool(re.fullmatch(BODY, right))
            if left_body and right_body:
                unparsed.append(original)
                continue
            if left_body or right_body:
                feature_text = right if left_body else left
                feature_direction = _invert(direction) if left_body else direction
                mentioned = parts_in_text(feature_text, parts)
                for name in mentioned:
                    tokens.append(body_token(parts.index(name), feature_direction))
                    relations.append({"kind": "body_reference", "item": original,
                                      "feature": name, "dimension": parts.index(name),
                                      "direction": feature_direction,
                                      "physical_reference": 0.75, "psychological_reference": None})
                old = parser.parse(original)
                old_ops = {claim.parts[0]: claim.op for claim in old.claims
                           if claim.kind == 'body_ref' and len(claim.parts) == 1}
                expected = '>' if feature_direction == 'gt' else '<'
                old_disagreements += sum(name in old_ops and old_ops[name] != expected for name in mentioned)
                flags.add("body_reference_uncertain")
                previous_body_subject = left_body
            else:
                left_parts, right_parts = parts_in_text(left, parts), parts_in_text(right, parts)
                if set(left_parts) & set(right_parts):
                    unparsed.append(original)
                    flags.add('self_comparison_review')
                else:
                    record_claim(SemanticClaim("comparison", original, left_parts,
                                               '>' if direction == 'gt' else '<', right_parts))
            continue
        # Weighted group sums: keep multiplicity; never silently collapse four
        # legs into one. A common affine offset cancels only for balanced sums.
        sum_group = rf"(?:{group}|四条腿)(?:之和|总和)?"
        summed = re.fullmatch(rf"(?P<left>{sum_group})比(?P<right>{sum_group})(?P<op>长|短)", item)
        if summed and re.search(r"之和|总和|四条腿", item):
            def weights(value: str) -> dict[str, int]:
                return {name: 4 if name == '腿' and '四条腿' in value else 1
                        for name in parts_in_text(value, parts)}
            left_weights, right_weights = weights(summed['left']), weights(summed['right'])
            relations.append({"kind": "weighted_sum", "item": original, "left": left_weights,
                              "right": right_weights, "direction": _direction(summed['op'])})
            if sum(left_weights.values()) != sum(right_weights.values()):
                flags.add("physical_scale_required")
                if '四条腿' in item:
                    flags.add("explicit_leg_multiplicity_preserved")
                unparsed.append(original)
            else:
                row = np.array([left_weights.get(name, 0) - right_weights.get(name, 0) for name in parts], dtype=float)
                if not np.any(row):
                    unparsed.append(original)
                else:
                    if summed['op'] == '长':
                        row = -row
                    tokens.append(facet_token(row[None, :], np.array([0.0])))
            continue
        # Equality is a band as a single predicate; near equality cannot be
        # silently reduced to one direction of a comparison.
        equal = re.fullmatch(rf"(?P<parts>{group}|四个部位|所有部位){MOD}(?:差不多一样|差不多同样|差不多|一样|相等|等长|相近|均匀)(?:长|的)?", item)
        if equal:
            mentioned = list(parts) if equal['parts'] in {'四个部位', '所有部位'} else parts_in_text(equal['parts'], parts)
            if len(mentioned) < 2:
                unparsed.append(original)
            else:
                for first, second in combinations(mentioned, 2):
                    record_claim(SemanticClaim("equality", original, [first, second], "=="))
                flags.add("equality_tolerance_convention")
            continue
        absolute = re.fullmatch(rf"(?P<parts>{group}|四个部位|所有部位){MOD}(?P<desc>长|短)(?:的|一点|一些)?", item)
        if absolute:
            mentioned = list(parts) if absolute['parts'] in {'四个部位', '所有部位'} else parts_in_text(absolute['parts'], parts)
            record_claim(SemanticClaim("general_case", original, mentioned, desc='long' if absolute['desc'] == '长' else 'short'))
            flags.add("absolute_threshold_convention")
            if re.search(r"很|比较|非常|特别|较|略|稍|偏|挺|一点|一些", item):
                flags.add("intensity_coarsened")
            continue
        extreme = re.fullmatch(rf"(?P<part>{part})(?:是)?(?:最)(?P<desc>长|短)(?:的)?", item)
        if extreme:
            record_claim(SemanticClaim("superlative", original, [extreme['part']],
                                      '>' if extreme['desc'] == '长' else '<',
                                      [name for name in parts if name != extreme['part']]))
            continue
        unparsed.append(original)
        suggestions.extend(asdict(claim) for claim in parser.parse(original).claims)
        if re.search(r"之和|加|总和", item):
            flags.add("sum_reference_unresolved")
        if re.search(r"[一二三四1234]个部位|[两三四]长|[两三四]短", item):
            flags.add("count_expression_review")
    unique = tuple(sorted(set(tokens)))
    # Contradictory report directions must not become a confident code.
    for dimension in range(4):
        if body_token(dimension, 'gt') in unique and body_token(dimension, 'lt') in unique:
            flags.add('contradictory_body_relations')
            unparsed.append('opposing body directions')
    if unique and not unparsed:
        numeric_rows = [row for token in unique if token.startswith('facet:')
                        for row in json.loads(token[6:])]
        if numeric_rows:
            array = np.asarray(numeric_rows, dtype=float)
            A, b = array[:, :-1], array[:, -1]
            # Require a positive-volume intersection in the feature cube.
            # Opposing strict long/short statements must not pass merely at
            # their shared boundary, where the continuous region has zero volume.
            solution = linprog(np.array([0., 0., 0., 0., -1.]),
                               A_ub=np.column_stack((A, np.linalg.norm(A, axis=1))),
                               b_ub=b, bounds=[(0., 1.)] * 5, method='highs')
            if solution.status == 2 or (solution.success and solution.x[-1] <= 1e-9):
                flags.add('contradictory_numeric_relations')
                unparsed.append('no positive-volume numeric region')
            elif not solution.success:
                raise RuntimeError(f'report feasibility check failed: {solution.message}')
    status = 'coded' if unique and not unparsed else 'needs_review'
    return StructuredReport(status, unique, tuple(relations), tuple(unparsed), tuple(sorted(flags)),
                            tuple(suggestions), old_disagreements)
