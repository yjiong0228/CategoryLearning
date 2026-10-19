"""Keep contrast syntax separate from report content and psychological roles.

Offsets refer to Python character indices in the unchanged input. Each segment
is passed to the existing conservative encoder. Removing a leading concession
and contrast markers permits a content conjunction to be checked by the same
encoder, including its contradiction checks. Syntax does not assign a feature
positive or negative responsibility for the participant's response.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .structure_0923 import StructuredReport, encode_report

VERSION = '0923-oral-discourse-content-v1'
CONTRAST = re.compile(r'(?<!不)(?:但是|但|可是|然而)')
# These forms need a distinct grammatical decision; do not silently split them.
UNHANDLED = re.compile(r'不但|不过')
QUALIFIERS = re.compile(r'相对|比较|非常|特别|稍微|一点|一些|有点|很|较|略|稍|偏|挺|更|尤其')


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ContentSegment:
    span: TextSpan
    content: TextSpan
    concession: TextSpan | None
    qualifiers: tuple[TextSpan, ...]
    encoding: StructuredReport


@dataclass(frozen=True)
class DiscourseReport:
    version: str
    text: str
    original: StructuredReport
    segments: tuple[ContentSegment, ...]
    contrast_markers: tuple[TextSpan, ...]
    unhandled_connectives: tuple[TextSpan, ...]
    content_text: str
    content: StructuredReport
    content_complete: bool
    role_status: str


def encode_discourse(text: str, feature_names: tuple[str, ...]) -> DiscourseReport:
    """Encode text without stimuli, selected responses or model predictions.

    Explicit half references use the independently confirmed range midpoint.
    Bare long/short retain the old midpoint convention. Their reference remains
    ambiguous; relative sensitivity must be carried separately by the workflow.
    Elliptical subjects and unsupported modifiers remain in unparsed content.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a string; represent missing reports by an empty string')

    def encode(value: str) -> StructuredReport:
        return encode_report(value, feature_names, half_reference='range_midpoint')

    def span(start: int, end: int) -> TextSpan:
        return TextSpan(start, end, text[start:end])

    original = encode(text)
    markers = tuple(span(m.start(), m.end()) for m in CONTRAST.finditer(text))
    unhandled = tuple(span(m.start(), m.end()) for m in UNHANDLED.finditer(text))
    starts = (0, *(m.end for m in markers))
    ends = (*(m.start for m in markers), len(text))
    segments = []
    for start, end in zip(starts, ends):
        raw = span(start, end)
        prefix = re.match(r'\s*虽然', raw.text)
        concession = (span(start + prefix.end() - 2, start + prefix.end())
                      if prefix else None)
        body = span(start + prefix.end() if prefix else start, end)
        qualifiers = tuple(span(body.start + m.start(), body.start + m.end())
                           for m in QUALIFIERS.finditer(body.text))
        segments.append(ContentSegment(raw, body, concession, qualifiers, encode(body.text)))
    # Preserve the original text exactly in the no-transformation case.
    transformed = bool(markers or any(s.concession for s in segments))
    content_text = '，'.join(s.content.text for s in segments) if transformed else text
    content = encode(content_text) if transformed else original
    complete = (content.status == 'coded' and not unhandled
                and all(s.encoding.status == 'coded' for s in segments))
    return DiscourseReport(VERSION, text, original, tuple(segments), markers,
                           unhandled, content_text, content, bool(complete), 'unassigned')
