"""Helpers for normalizing provider field citation bboxes."""

from __future__ import annotations

import bisect
import logging
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from extract_bench.schemas.extract_output import FieldCitation

logger = logging.getLogger(__name__)


def extract_extend_field_citations(
    raw_output: Mapping[str, Any],
    *,
    review_agent: bool = False,
) -> list[FieldCitation]:
    """Extract citations and optional Review Agent scores from Extend metadata."""
    output = _as_mapping(_as_mapping(raw_output.get("processor_run")).get("output"))
    metadata = _as_mapping(output.get("metadata"))
    citations = _dedupe(_collect_field_map(metadata, source="extend"))
    if not review_agent:
        return citations

    signals = _collect_extend_review_agent_signals(metadata)
    normalized: list[FieldCitation] = []
    cited_paths: set[str] = set()
    for citation in citations:
        signal = signals.get(citation.field_path)
        cited_paths.add(citation.field_path)
        normalized.append(
            citation.model_copy(
                update={
                    # Review Agent is the sole trust signal for this arm. Do
                    # not silently fall back to citation/OCR/logprob values.
                    "confidence": signal[0] if signal is not None else None,
                    "metadata": {
                        **(citation.metadata if citation.metadata is not None else {}),
                        **(signal[1] if signal is not None else {}),
                    },
                }
            )
        )
    for field_path, (confidence, signal_metadata) in signals.items():
        if field_path in cited_paths:
            continue
        normalized.append(
            FieldCitation(
                field_path=field_path,
                page=None,
                bbox=None,
                source="extend",
                confidence=confidence,
                metadata=signal_metadata,
            )
        )
    return _dedupe(normalized)


def _collect_extend_review_agent_signals(
    metadata: Mapping[str, Any],
) -> dict[str, tuple[float, dict[str, Any]]]:
    signals: dict[str, tuple[float, dict[str, Any]]] = {}

    def walk(node: Any, path: Sequence[str]) -> None:
        node_map = _as_mapping(node)
        if len(node_map) > 0:
            score_value = _first_present(node_map, ("reviewAgentScore", "review_agent_score"))
            raw_score = _coerce_float(score_value)
            field_path = format_field_path(path)
            if len(field_path) > 0 and raw_score is not None and math.isfinite(raw_score):
                normalized = max(0.0, min(1.0, (raw_score - 1.0) / 4.0))
                signal_metadata: dict[str, Any] = {
                    "review_agent_score": raw_score,
                    "confidence_semantics": "ordinal_1_5",
                }
                insights = _first_present(node_map, ("insights", "reviewAgentInsights", "review_agent_insights"))
                if insights is not None:
                    signal_metadata["review_agent_insights"] = insights
                signals[field_path] = (normalized, signal_metadata)

            for key, value in node_map.items():
                if key in {
                    "citations",
                    "citation",
                    "reviewAgentScore",
                    "review_agent_score",
                    "insights",
                    "reviewAgentInsights",
                    "review_agent_insights",
                }:
                    continue
                next_path = (*path, key)
                walk(value, next_path)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, item in enumerate(node):
                walk(item, [*path, f"[{index}]"])

    walk(metadata, ())
    return signals


REDUCTO_CONFIDENCE_SIGNAL_TYPE = "confidence_signal_type"
REDUCTO_HYBRID_CONFIDENCE = "reducto_extract_or_high_low"
REDUCTO_CATEGORICAL_SOURCE = "categorical_high_or_zero"


def extract_reducto_field_citations(
    raw_output: Mapping[str, Any],
    *,
    confidence_signal_type: Literal["continuous", "ordinal", "categorical"] | None = None,
    extracted_data: Any | None = None,
) -> Sequence[FieldCitation]:
    """Extract Reducto grounding and value confidence independently."""
    result = raw_output.get("result")
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    citations = _collect_reducto_citations(result, path=())
    if confidence_signal_type not in {"categorical", "continuous"}:
        return _dedupe(citations)

    categorical = confidence_signal_type == "categorical"
    signals = _collect_reducto_extraction_signals(result, categorical=categorical)
    cited_default = (
        (
            0.0,
            {
                "citation_confidence_labels": [],
                "confidence_source": REDUCTO_CATEGORICAL_SOURCE,
                "confidence_semantics": REDUCTO_HYBRID_CONFIDENCE,
            },
        )
        if categorical
        else (None, {})
    )
    normalized: list[FieldCitation] = []
    cited_paths: set[str] = set()
    for citation in citations:
        cited_paths.add(citation.field_path)
        confidence, signal_metadata = signals.get(citation.field_path, cited_default)
        update: dict[str, Any] = {"confidence": confidence}
        if len(signal_metadata) > 0:
            # Keep the raw metadata of each surviving grounding citation.
            update["metadata"] = {**signal_metadata, **(citation.metadata if citation.metadata is not None else {})}
        normalized.append(citation.model_copy(update=update))

    if categorical:
        for field_path in _iter_emitted_leaf_paths(extracted_data):
            if field_path not in cited_paths:
                signals.setdefault(
                    field_path,
                    (
                        0.0,
                        {
                            "confidence_source": "missing_or_unavailable",
                            "confidence_semantics": REDUCTO_HYBRID_CONFIDENCE,
                        },
                    ),
                )
    for field_path, (confidence, signal_metadata) in signals.items():
        if field_path not in cited_paths:
            normalized.append(
                FieldCitation(
                    field_path=field_path,
                    page=None,
                    bbox=None,
                    source="reducto",
                    confidence=confidence,
                    metadata=signal_metadata,
                )
            )
    return _dedupe(normalized)


def _iter_emitted_leaf_paths(value: Any, path: Sequence[str] | None = None) -> Sequence[str]:
    path = () if path is None else path
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_iter_emitted_leaf_paths(item, [*path, str(key)]))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_iter_emitted_leaf_paths(item, [*path, f"[{index}]"]))
        return paths
    if value is None:
        return []
    field_path = format_field_path(path)
    return [field_path] if len(field_path) > 0 else []


def _collect_reducto_extraction_signals(
    result: Any,
    *,
    categorical: bool,
) -> dict[str, tuple[float, dict[str, Any]]]:
    """Read scores from field wrappers and raw citations, including rejected boxes.

    Use the minimum granular extraction score for a field. Categorical pipelines
    use citation labels when no granular score is available.
    """
    granular_scores: dict[str, tuple[float, dict[str, Any]]] = {}
    citation_labels: dict[str, set[str]] = {}

    def record_granular(field_path: str, entry: Mapping[str, Any]) -> None:
        granular = _as_mapping(entry.get("granular_confidence"))
        raw_confidence = _coerce_float(granular.get("extract_confidence"))
        if raw_confidence is None or not math.isfinite(raw_confidence):
            return
        confidence = max(0.0, min(1.0, raw_confidence))
        current = granular_scores.get(field_path)
        if current is not None and current[0] <= confidence:
            return
        metadata: dict[str, Any] = {
            "granular_confidence": dict(granular),
            "confidence_semantics": "continuous_0_1",
        }
        if isinstance(label := entry.get("confidence"), str):
            metadata["confidence_label"] = label
        granular_scores[field_path] = (confidence, metadata)

    def walk(node: Any, path: Sequence[str]) -> None:
        if isinstance(node, Mapping):
            wrapper = "value" in node and "citations" in node
            field_path = format_field_path(path)
            if len(field_path) > 0:
                record_granular(field_path, node)
            if wrapper:
                for raw_entry in _iter_citation_entries(node):
                    entry = _as_mapping(raw_entry)
                    record_granular(field_path, entry)
                    raw_label = entry.get("confidence")
                    if categorical and isinstance(raw_label, str) and len(label := raw_label.strip().lower()) > 0:
                        citation_labels.setdefault(field_path, set()).add(label)

            for key, value in node.items():
                if key in {"citations", "citation", "granular_confidence"}:
                    continue
                next_path = path if key == "value" and wrapper else (*path, key)
                walk(value, next_path)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, item in enumerate(node):
                walk(item, [*path, f"[{index}]"])

    walk(result, ())
    if not categorical:
        return granular_scores

    signals: dict[str, tuple[float, dict[str, Any]]] = {}
    for field_path in dict.fromkeys((*granular_scores, *citation_labels)):
        labels = sorted(citation_labels.get(field_path, ()), key=lambda label: (label != "low", label != "high", label))
        if field_path in granular_scores:
            confidence, metadata = granular_scores[field_path]
            source = "granular_extract_confidence"
        else:
            confidence = 1.0 if labels == ["high"] else 0.0
            metadata = {}
            source = REDUCTO_CATEGORICAL_SOURCE
        signals[field_path] = (
            confidence,
            {
                **metadata,
                "citation_confidence_labels": labels,
                "confidence_source": source,
                "confidence_semantics": REDUCTO_HYBRID_CONFIDENCE,
            },
        )
    return signals


def _collect_reducto_citations(node: Any, *, path: Sequence[str]) -> Sequence[FieldCitation]:
    citations: list[FieldCitation] = []
    if isinstance(node, Mapping) and "value" in node and "citations" in node:
        citations.extend(
            _collect_node_citations(
                field_path=format_field_path(path),
                node=node,
                source="reducto",
                default_page=None,
                default_dimensions=None,
            )
        )
        citations.extend(_collect_reducto_citations(node["value"], path=path))
    else:
        for token, child in _iter_field_children(node):
            citations.extend(_collect_reducto_citations(child, path=(*path, token)))
    return citations


def _iter_field_children(node: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(node, Mapping):
        yield from node.items()
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for index, item in enumerate(node):
            yield f"[{index}]", item


def extract_landingai_field_citations(
    raw_output: Mapping[str, Any],
    *,
    pdf_path: str | Path | None = None,
    extracted_data: Mapping[str, Any] | None = None,
) -> list[FieldCitation]:
    """Extract citations from LandingAI extraction references and parse grounding.

    LandingAI's API ships a single ``chunkText`` grounding box for whole header
    sections, then references the same box from every cover-page scalar field
    (e.g., ``filing_manager_name``, ``report_calendar_quarter_end`` all share
    one UUID under ``parse_response.grounding``). When ``pdf_path`` and
    ``extracted_data`` are supplied, those coarse chunk bboxes are refined to a
    tight per-value bbox by text-searching for the extracted value within the
    chunk's clip rectangle (PyMuPDF ``page.search_for``). Cell-level groundings
    (``tableCell``) are already tight and pass through unchanged.
    """
    extraction_metadata = _as_mapping(raw_output.get("extraction_metadata"))
    parse_response = _as_mapping(raw_output.get("parse_response"))
    grounding = _as_mapping(parse_response.get("grounding"))
    if not extraction_metadata or not grounding:
        return []
    citations = _dedupe(
        _collect_landingai_metadata_references(
            node=extraction_metadata,
            grounding=grounding,
            path=[],
        )
    )
    if pdf_path is not None and extracted_data is not None:
        citations = _refine_chunk_citations_via_text_search(
            citations,
            pdf_path=Path(pdf_path),
            extracted_data=extracted_data,
        )
    return citations


def extract_landingai_v2_field_citations(
    extraction_metadata: Any,
    structure: Any,
    markdown: str | None = None,
) -> list[FieldCitation]:
    """Map LandingAI ADE v2 extraction ranges to page boxes from the v2 parse structure.

    v2 Extract returns ``extraction_metadata``: a tree that mirrors the extraction, with
    each leaf replaced by ``{"value": ..., "ranges": [{"start", "end"}, ...]}``. The
    ranges are Unicode code-point offsets into the Markdown that was extracted from,
    which is the v2 Parse Markdown. The v2 Parse ``structure`` tree carries the same
    kind of ``grounding.range`` on every block, plus ``atomic_grounding`` (one entry
    per word on ``dpt-3-pro-20261111`` and later, per visual line on earlier DPT-3
    snapshots) on leaf blocks; boxes are normalized ``xmin/ymin/xmax/ymax`` and
    pages are 1-indexed.

    Each leaf range is matched to the overlapping atomic spans first (word boxes on
    current snapshots, the tightest available), then to the smallest overlapping blocks
    when no atomic span covers it. This is the same join as the public ``/v2/ground``
    endpoint, done locally at atomic granularity instead of block granularity.

    When the parse ``markdown`` is available, the atomic spans are refined so that each
    citation covers the value as printed:

    - A range often includes the field's printed label and neighbouring lines. Only the
      words that match the extracted value are kept (checkbox marks for booleans).
    - Word boxes that belong to one occurrence of the value (same block, adjacent in the
      Markdown) are merged into one box, since a value is one region on the page.
    - When Extract returns a value with no ranges (for example a value reformatted from
      several printed lines), the value's words are located in the Markdown directly.
    """
    metadata = _as_mapping(extraction_metadata)
    structure_map = _as_mapping(structure)
    if not metadata or not structure_map:
        return []

    atomic, blocks = _landingai_v2_spans(structure_map)
    if not atomic and not blocks:
        return []
    atomic.sort(key=lambda span: span["start"])
    blocks.sort(key=lambda span: span["start"])
    index = {
        "atomic": atomic,
        "atomic_starts": [span["start"] for span in atomic],
        "blocks": blocks,
        "block_starts": [span["start"] for span in blocks],
        "markdown": markdown if isinstance(markdown, str) and markdown else None,
    }
    return _dedupe(_collect_landingai_v2_citations(node=metadata, index=index, path=[]))


def _landingai_v2_spans(structure: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten the v2 parse structure into atomic spans (words or lines) and leaf-block spans."""
    atomic: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        node_map = _as_mapping(node)
        if not node_map:
            return
        children = _as_sequence(node_map.get("children"))
        node_type = node_map.get("type")
        if node_type not in ("document", "page"):
            grounding = _landingai_v2_span(node_map.get("grounding"))
            if grounding is not None and not children:
                grounding["type"] = node_type
                grounding["block_id"] = node_map.get("id")
                blocks.append(grounding)
            for entry in _as_sequence(node_map.get("atomic_grounding")):
                span = _landingai_v2_span(entry)
                if span is not None:
                    span["type"] = node_type
                    span["block_id"] = node_map.get("id")
                    atomic.append(span)
        for child in children:
            visit(child)

    visit(structure)
    return atomic, blocks


def _landingai_v2_span(grounding: Any) -> dict[str, Any] | None:
    """Return ``{start, end, page, bbox}`` for one v2 grounding entry, or None."""
    grounding_map = _as_mapping(grounding)
    text_range = _as_mapping(grounding_map.get("range"))
    box = _as_mapping(grounding_map.get("box"))
    start = _coerce_int(text_range.get("start"))
    end = _coerce_int(text_range.get("end"))
    page = _coerce_int(grounding_map.get("page"))
    xmin = _coerce_float(box.get("xmin"))
    ymin = _coerce_float(box.get("ymin"))
    xmax = _coerce_float(box.get("xmax"))
    ymax = _coerce_float(box.get("ymax"))
    if None in (start, end, xmin, ymin, xmax, ymax) or end <= start:  # type: ignore[operator]
        return None
    bbox = _normalize_bbox([xmin, ymin, xmax - xmin, ymax - ymin], None)  # type: ignore[operator]
    if bbox is None:
        return None
    return {"start": start, "end": end, "page": page if page and page >= 1 else None, "bbox": bbox}


def _landingai_v2_overlaps(
    spans: list[dict[str, Any]],
    starts: list[int],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """Spans (sorted by start) that overlap the half-open range [start, end)."""
    hits: list[dict[str, Any]] = []
    position = bisect.bisect_left(starts, end) - 1
    while position >= 0:
        span = spans[position]
        if span["end"] > start:
            hits.append(span)
        else:
            # Atomic spans and leaf blocks do not nest, so earlier spans end even earlier.
            break
        position -= 1
    hits.reverse()
    return hits


def _collect_landingai_v2_citations(
    *,
    node: Any,
    index: Mapping[str, Any],
    path: list[str],
) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if node_map:
        if "ranges" in node_map and "value" in node_map:
            return _landingai_v2_leaf_citations(node_map, index=index, field_path=format_field_path(path))
        citations: list[FieldCitation] = []
        for key, value in node_map.items():
            citations.extend(_collect_landingai_v2_citations(node=value, index=index, path=[*path, str(key)]))
        return citations

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        citations = []
        for item_index, item in enumerate(node):
            citations.extend(_collect_landingai_v2_citations(node=item, index=index, path=[*path, f"[{item_index}]"]))
        return citations

    return []


def _landingai_v2_leaf_citations(
    leaf: Mapping[str, Any],
    *,
    index: Mapping[str, Any],
    field_path: str,
) -> list[FieldCitation]:
    if not field_path:
        return []
    value = leaf.get("value")
    reference_text = None if value is None else str(value)
    markdown = index.get("markdown")
    ranges: list[tuple[int, int]] = []
    for raw_range in _as_sequence(leaf.get("ranges")):
        range_map = _as_mapping(raw_range)
        start = _coerce_int(range_map.get("start"))
        end = _coerce_int(range_map.get("end"))
        if start is not None and end is not None and end > start:
            ranges.append((start, end))
    range_source = "extract"
    if not ranges and markdown is not None and value is not None:
        located = _landingai_v2_locate_value(markdown, value)
        if located is not None:
            ranges = [located]
            range_source = "located"

    citations: list[FieldCitation] = []
    for start, end in ranges:
        granularity = "atomic"
        hits = _landingai_v2_overlaps(index["atomic"], index["atomic_starts"], start, end)
        if hits and markdown is not None:
            selected = _landingai_v2_value_spans(hits, markdown, value)
            if selected:
                groups = _landingai_v2_group_spans(selected)
                if isinstance(value, str):
                    groups = _landingai_v2_drop_partial_groups(groups, markdown)
                granularity = "value"
            else:
                groups = [[hit] for hit in hits]
        elif hits:
            groups = [[hit] for hit in hits]
        else:
            granularity = "block"
            groups = [[hit] for hit in _landingai_v2_overlaps(index["blocks"], index["block_starts"], start, end)]
        for group in groups:
            first = group[0]
            citations.append(
                FieldCitation(
                    field_path=field_path,
                    page=first["page"],
                    bbox=_landingai_v2_union_bbox([span["bbox"] for span in group]),
                    reference_text=reference_text,
                    source="landingai",
                    metadata={
                        "range": [start, end],
                        "range_source": range_source,
                        "granularity": granularity,
                        "span_count": len(group),
                        "block_type": first.get("type"),
                        "block_id": first.get("block_id"),
                    },
                )
            )
    return citations


_LANDINGAI_V2_CHECKED_MARKS = ("[x]", "[X]", "\u2611", "\u2612")
_LANDINGAI_V2_UNCHECKED_MARKS = ("[ ]", "\u2610")
_LANDINGAI_V2_HTML_TAG = re.compile(r"<[^>]*>")
_LANDINGAI_V2_WORD = re.compile(r"[^\W_]+")
# Largest Markdown gap (characters) between two value words that still belong to one
# printed occurrence: covers separators such as ", ", "<br>" and a newline.
_LANDINGAI_V2_MERGE_GAP = 16


def _landingai_v2_words(text: str) -> list[str]:
    """Lowercase word tokens of ``text`` with HTML tags removed."""
    return _LANDINGAI_V2_WORD.findall(_LANDINGAI_V2_HTML_TAG.sub(" ", text).casefold())


def _landingai_v2_number_digits(value: int | float) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        text = str(int(value)) if value.is_integer() else format(value, "f").rstrip("0")
    else:
        text = str(value)
    return re.sub(r"\D", "", text)


def _landingai_v2_value_spans(
    hits: list[dict[str, Any]],
    markdown: str,
    value: Any,
) -> list[dict[str, Any]]:
    """The atomic spans in ``hits`` whose text is part of ``value`` (empty if none match)."""
    if value is None:
        return []
    texts = [markdown[span["start"] : span["end"]] for span in hits]
    if isinstance(value, bool):
        # Prefer marks in the state the value reports (checked for True), else any mark.
        preferred = _LANDINGAI_V2_CHECKED_MARKS if value else _LANDINGAI_V2_UNCHECKED_MARKS
        for marks in (preferred, _LANDINGAI_V2_CHECKED_MARKS + _LANDINGAI_V2_UNCHECKED_MARKS):
            selected = [span for span, text in zip(hits, texts, strict=True) if any(mark in text for mark in marks)]
            if selected:
                return selected
        return []
    if isinstance(value, (int, float)):
        digits = _landingai_v2_number_digits(value)
        if not digits:
            return []
        selected = []
        for span, text in zip(hits, texts, strict=True):
            span_digits = re.sub(r"\D", "", _LANDINGAI_V2_HTML_TAG.sub(" ", text))
            if span_digits and span_digits in digits and len(span_digits) >= min(2, len(digits)):
                selected.append(span)
        return selected
    if not isinstance(value, str):
        return []
    value_words = {word for word in _landingai_v2_words(value) if len(word) > 1 or word.isdigit()}
    if not value_words:
        return []
    # A span matches when at least half its words are words of the value. A span that is
    # a fragment of a value word (the parser can split one printed word, e.g. "ENTI" +
    # "TY") joins only a kept neighbour whose letters, read together with it, are still
    # part of the value, so short label words ("is", "of") are not picked up.
    value_letters = "".join(_landingai_v2_words(value))
    letters = ["".join(_landingai_v2_words(text)) for text in texts]
    keep = []
    for text in texts:
        words = [word for word in _landingai_v2_words(text) if len(word) > 1 or word.isdigit()]
        keep.append(bool(words) and sum(word in value_words for word in words) * 2 >= len(words))
    changed = True
    while changed:
        changed = False
        for position in range(len(hits)):
            if keep[position] or not letters[position]:
                continue
            for neighbour in (position - 1, position + 1):
                if not 0 <= neighbour < len(hits) or not keep[neighbour]:
                    continue
                pair = (neighbour, position) if neighbour < position else (position, neighbour)
                joined = letters[pair[0]] + letters[pair[1]]
                if joined in value_letters and hits[pair[1]]["start"] - hits[pair[0]]["end"] <= 1:
                    keep[position] = changed = True
                    break
    return [span for span, kept in zip(hits, keep, strict=True) if kept]


def _landingai_v2_group_spans(spans: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split spans (sorted by start) into runs on one page and block with small Markdown gaps."""
    groups: list[list[dict[str, Any]]] = []
    for span in spans:
        if groups:
            last = groups[-1][-1]
            if (
                span["page"] == last["page"]
                and span.get("block_id") == last.get("block_id")
                and span["start"] - last["end"] <= _LANDINGAI_V2_MERGE_GAP
            ):
                groups[-1].append(span)
                continue
        groups.append([span])
    return groups


def _landingai_v2_drop_partial_groups(groups: list[list[dict[str, Any]]], markdown: str) -> list[list[dict[str, Any]]]:
    """Drop groups whose text is contained in a longer group's text.

    A printed label can share a word with the value ("What type of entity is this
    partner? DISREGARDED ENTITY"); that stray word forms a small group of its own. A
    value printed twice in full keeps both occurrences.
    """
    if len(groups) < 2:
        return groups
    letters = ["".join(_landingai_v2_words(markdown[group[0]["start"] : group[-1]["end"]])) for group in groups]
    return [
        group
        for group, own in zip(groups, letters, strict=True)
        if not any(len(own) < len(other) and own in other for other in letters)
    ]


def _landingai_v2_union_bbox(boxes: list[list[float]]) -> list[float]:
    """Smallest ``[x, y, w, h]`` box containing every box in ``boxes``."""
    if len(boxes) == 1:
        return list(boxes[0])
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def _landingai_v2_locate_value(markdown: str, value: Any) -> tuple[int, int] | None:
    """Range of the first occurrence of a text ``value`` in ``markdown``, allowing any punctuation,
    whitespace or ``<br>`` between its words (a value joined with ", " from printed lines)."""
    if not isinstance(value, str) or isinstance(value, bool):
        return None
    words = _LANDINGAI_V2_WORD.findall(value)
    if not words or len("".join(words)) < 3:
        return None
    separator = r"(?:[\W_]|<br\s*/?>){1,12}"
    pattern = r"(?<![^\W_])" + separator.join(re.escape(word) for word in words) + r"(?![^\W_])"
    match = re.search(pattern, markdown, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()


def extract_llamaextract_field_citations(metadata: Any, *, source: str) -> Sequence[FieldCitation]:
    """Unwrap LlamaExtract's document, page, and row metadata envelopes."""
    metadata_map = _as_mapping(metadata)
    field_metadata = _as_mapping(metadata_map.get("field_metadata"))
    sections = ("document_metadata", "page_metadata", "row_metadata")
    citations: list[FieldCitation] = []
    if any(key in field_metadata for key in sections):
        envelope = field_metadata
    else:
        envelope = metadata_map
        citations.extend(_collect_field_tree(node=field_metadata, source=source, path=()))

    for key in ("document_metadata", "fields"):
        citations.extend(_collect_field_tree(node=envelope.get(key), source=source, path=()))

    for key in ("page_metadata", "row_metadata"):
        for entry in _as_sequence(envelope.get(key)):
            entry_map = _as_mapping(entry)
            field_tree = entry_map
            for field_key in ("field_metadata", "document_metadata", "fields"):
                if field_key in entry_map:
                    field_tree = _as_mapping(entry_map[field_key])
                    break
            citations.extend(
                _collect_field_tree(
                    node=field_tree,
                    source=source,
                    path=(),
                    default_page=_extract_page(entry_map),
                    default_dimensions=_extract_dimensions(entry_map),
                )
            )
    return _dedupe(citations)


def _collect_field_map(
    field_map: Mapping[str, Any],
    *,
    source: str,
    default_page: int | None = None,
    default_dimensions: tuple[float, float] | None = None,
) -> list[FieldCitation]:
    citations: list[FieldCitation] = []
    for field_path, node in field_map.items():
        if field_path.startswith("_"):
            continue
        citations.extend(
            _collect_node_citations(
                field_path=field_path,
                node=node,
                source=source,
                default_page=default_page,
                default_dimensions=default_dimensions,
            )
        )
    return citations


def confidence_only_citation(field_path: str, *, source: str, confidence: float | None) -> FieldCitation | None:
    """Keep a score from a citation with unusable grounding."""
    if confidence is None or not math.isfinite(confidence):
        return None
    return FieldCitation(field_path=field_path, page=None, bbox=None, confidence=confidence, source=source)


def _collect_node_citations(
    *,
    field_path: str,
    node: Any,
    source: str,
    default_page: int | None,
    default_dimensions: tuple[float, float] | None,
) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if not node_map:
        return []

    page = _extract_page(node_map)
    if page is None:
        page = default_page
    dimensions = _extract_dimensions(node_map) or default_dimensions
    field_confidence = _extract_confidence(node_map)
    citations: list[FieldCitation] = []
    confidence_only: list[FieldCitation] = []
    for citation in _iter_citation_entries(node_map):
        normalized = _normalize_citation(
            field_path=field_path,
            citation=citation,
            source=source,
            default_page=page,
            default_dimensions=dimensions,
            default_confidence=field_confidence,
        )
        citations.extend(normalized)
        if len(normalized) == 0:
            signal = confidence_only_citation(
                field_path,
                source=source,
                confidence=_extract_confidence(_as_mapping(citation)),
            )
            if signal is not None:
                confidence_only.append(signal)
    if len(citations) == 0 and field_confidence is not None:
        citations.append(
            FieldCitation(
                field_path=field_path,
                page=page,
                bbox=None,
                source=source,
                confidence=field_confidence,
                metadata=_compact_metadata(node_map),
            )
        )
    return citations + confidence_only


def _iter_citation_entries(node: Mapping[str, Any]) -> list[Any]:
    """Iterate citation entries supporting both plural `citations` and singular `citation` keys."""
    entries: list[Any] = []
    for key in ("citations", "citation"):
        for entry in _as_sequence(node.get(key)):
            entries.append(entry)
    return entries


def _collect_field_tree(
    *,
    node: Any,
    source: str,
    path: Sequence[str],
    default_page: int | None = None,
    default_dimensions: tuple[float, float] | None = None,
) -> Sequence[FieldCitation]:
    citations: list[FieldCitation] = []
    if isinstance(node, Mapping):
        page = _extract_page(node)
        dimensions = _extract_dimensions(node)
        if page is not None:
            default_page = page
        if dimensions is not None:
            default_dimensions = dimensions
        if len(path) > 0:
            citations.extend(
                _collect_node_citations(
                    field_path=format_field_path(path),
                    node=node,
                    source=source,
                    default_page=default_page,
                    default_dimensions=default_dimensions,
                )
            )
    for token, child in _iter_field_children(node):
        if token in ("citation", "citations"):
            continue
        citations.extend(
            _collect_field_tree(
                node=child,
                source=source,
                path=(*path, token),
                default_page=default_page,
                default_dimensions=default_dimensions,
            )
        )
    return citations


def _collect_landingai_metadata_references(
    *,
    node: Any,
    grounding: Mapping[str, Any],
    path: list[str],
) -> list[FieldCitation]:
    node_map = _as_mapping(node)
    if node_map:
        field_path = format_field_path(path)
        references = _as_sequence(node_map.get("references"))
        if field_path and references:
            citations: list[FieldCitation] = []
            reference_text = _extract_reference_text(node_map)
            for reference in references:
                if not isinstance(reference, str) or not reference:
                    continue
                grounding_entry = _as_mapping(grounding.get(reference))
                citation = _landingai_reference_to_citation(
                    field_path=field_path,
                    reference=reference,
                    reference_text=reference_text,
                    grounding_entry=grounding_entry,
                )
                if citation is not None:
                    citations.append(citation)
            return citations

        citations = []
        for key, value in node_map.items():
            if key in {"references", "value"}:
                continue
            citations.extend(
                _collect_landingai_metadata_references(
                    node=value,
                    grounding=grounding,
                    path=[*path, key],
                )
            )
        return citations

    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        citations = []
        for index, item in enumerate(node):
            citations.extend(
                _collect_landingai_metadata_references(
                    node=item,
                    grounding=grounding,
                    path=[*path, f"[{index}]"],
                )
            )
        return citations

    return []


def _landingai_reference_to_citation(
    *,
    field_path: str,
    reference: str,
    reference_text: str | None,
    grounding_entry: Mapping[str, Any],
) -> FieldCitation | None:
    confidence = _extract_confidence(grounding_entry)
    box = _as_mapping(grounding_entry.get("box"))
    left = _coerce_float(box.get("left"))
    top = _coerce_float(box.get("top"))
    right = _coerce_float(box.get("right"))
    bottom = _coerce_float(box.get("bottom"))
    bbox = None
    if left is not None and top is not None and right is not None and bottom is not None:
        bbox = _normalize_bbox([left, top, right - left, bottom - top], None)
    if bbox is None:
        return confidence_only_citation(field_path, source="landingai", confidence=confidence)

    page_index = _coerce_int(grounding_entry.get("page"))
    page = page_index + 1 if page_index is not None and page_index >= 0 else None
    metadata = {"reference": reference}
    grounding_type = grounding_entry.get("type")
    if isinstance(grounding_type, str):
        metadata["type"] = grounding_type

    return FieldCitation(
        field_path=field_path,
        page=page,
        bbox=bbox,
        reference_text=reference_text,
        confidence=confidence,
        source="landingai",
        metadata=metadata,
    )


def format_field_path(path: Sequence[str]) -> str:
    """Render path tokens so list-index tokens (`[N]`) attach to the prior key without a dot.

    GT field paths use bracket notation (`employees[0].basic_salary`). We collect tokens during
    the recursive walk and convert any leading-bracket tokens into bracket-joined segments so
    predictions match GT field path scope.
    """
    rendered = ""
    for token in path:
        if token.startswith("[") and token.endswith("]"):
            rendered += token
        elif rendered:
            rendered += "." + token
        else:
            rendered = token
    return rendered


def _normalize_citation(
    *,
    field_path: str,
    citation: Any,
    source: str,
    default_page: int | None,
    default_dimensions: tuple[float, float] | None,
    default_confidence: float | None = None,
) -> list[FieldCitation]:
    citation_map = _as_mapping(citation)
    if not citation_map:
        return []

    page = _extract_page(citation_map)
    if page is None:
        page = _extract_nested_bbox_page(citation_map)
    if page is None:
        page = default_page
    dimensions = _extract_dimensions(citation_map) or default_dimensions
    polygon = _extract_polygon(citation_map)
    reference_text = _extract_reference_text(citation_map)
    confidence = _extract_confidence(citation_map)
    if confidence is None:
        confidence = default_confidence
    metadata = _compact_metadata(citation_map)

    plural_bboxes = _extract_bbox_list(citation_map)
    if plural_bboxes:
        normalized_polygon = _normalize_polygon(polygon, dimensions) if polygon is not None else None
        results: list[FieldCitation] = []
        for entry_bbox in plural_bboxes:
            normalized_bbox = _normalize_bbox(entry_bbox, dimensions)
            if normalized_bbox is None:
                continue
            results.append(
                FieldCitation(
                    field_path=field_path,
                    page=page,
                    bbox=normalized_bbox,
                    polygon=normalized_polygon,
                    reference_text=reference_text,
                    confidence=confidence,
                    source=source,
                    metadata=metadata,
                )
            )
        if results:
            return results
        # All entries failed to normalize — the bbox attempt was real but
        # garbage. Drop rather than emit a misleading page-only fallback.
        return []

    raw_bbox = _bbox_from_polygon(polygon) if polygon is not None else _extract_bbox(citation_map)
    normalized_bbox = _normalize_bbox(raw_bbox, dimensions)
    normalized_polygon = _normalize_polygon(polygon, dimensions) if polygon is not None else None
    if normalized_bbox is None and not _has_bbox_attempt(citation_map):
        # v0.2: page-only citation (e.g. Extend metadata with `{page: N}` and no
        # bbox/polygon keys at all). Distinct from a malformed citation where a
        # bbox key was present but failed to parse — those still get dropped.
        return [
            FieldCitation(
                field_path=field_path,
                page=page,
                bbox=None,
                polygon=normalized_polygon,
                reference_text=reference_text,
                confidence=confidence,
                source=source,
                metadata=metadata,
            )
        ]
    if normalized_bbox is None:
        return []
    return [
        FieldCitation(
            field_path=field_path,
            page=page,
            bbox=normalized_bbox,
            polygon=normalized_polygon,
            reference_text=reference_text,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
    ]


def _extract_bbox_list(node: Mapping[str, Any]) -> list[list[float]] | None:
    """Extract a plural list of bboxes if `bounding_boxes` is present.

    Each entry can be either a 4-element [x, y, w, h] sequence or a mapping with
    x/y/w/h or x1/y1/x2/y2 keys.
    """
    raw = node.get("bounding_boxes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    if not raw:
        return None
    bboxes: list[list[float]] = []
    for entry in raw:
        bbox: list[float] | None = None
        if isinstance(entry, Mapping):
            bbox = _bbox_from_mapping(entry)
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
            bbox = _bbox_from_sequence(entry)
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes or None


def _extract_page(node: Mapping[str, Any]) -> int | None:
    for key in ("page", "page_number", "pageNumber"):
        raw_value = node.get(key)
        if key == "page" and isinstance(raw_value, Mapping):
            raw_value = raw_value.get("number")
        value = _coerce_int(raw_value)
        if value is not None and value >= 1:
            return value
    for key in ("page_index", "pageIndex"):
        value = _coerce_int(node.get(key))
        if value is not None and value >= 0:
            return value + 1
    return None


def _extract_nested_bbox_page(node: Mapping[str, Any]) -> int | None:
    for key in ("bbox", "bounding_box", "boundingBox", "box"):
        page = _extract_page(_as_mapping(node.get(key)))
        if page is not None:
            return page
    return None


def _extract_dimensions(node: Mapping[str, Any]) -> tuple[float, float] | None:
    width = _coerce_float(_first_present(node, ("page_width", "pageWidth", "width", "image_width", "imageWidth")))
    height = _coerce_float(_first_present(node, ("page_height", "pageHeight", "height", "image_height", "imageHeight")))
    if width is not None and height is not None and width > 0 and height > 0:
        return width, height

    for key in (
        "page",
        "page_dimensions",
        "pageDimensions",
        "page_size",
        "pageSize",
        "dimensions",
        "image_size",
        "imageSize",
    ):
        size = _as_mapping(node.get(key))
        width = _coerce_float(_first_present(size, ("width", "w")))
        height = _coerce_float(_first_present(size, ("height", "h")))
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height
    return None


def _extract_bbox(node: Mapping[str, Any]) -> list[float] | None:
    for key in ("bbox", "bounding_box", "boundingBox", "box"):
        bbox = node.get(key)
        bbox_from_dict = _bbox_from_mapping(_as_mapping(bbox))
        if bbox_from_dict is not None:
            return bbox_from_dict
        bbox_from_sequence = _bbox_from_sequence(bbox)
        if bbox_from_sequence is not None:
            return bbox_from_sequence

    bbox_from_dict = _bbox_from_mapping(node)
    if bbox_from_dict is not None:
        return bbox_from_dict
    return None


def _has_bbox_attempt(node: Mapping[str, Any]) -> bool:
    """True when the citation map contains a non-null bbox/polygon-shaped key.

    Used to distinguish v0.2 page-only citations (no bbox metadata at all, e.g.
    Extend's `{page: 1}` shape) from malformed citations where a bbox key was
    provided but unparseable. Page-only citations are emitted; malformed are
    silently dropped, matching pre-v0.2 behavior on bad input.
    """
    bbox_keys = (
        "bbox",
        "bounding_box",
        "boundingBox",
        "box",
        "bounding_boxes",
        "polygon",
        "bounding_polygon",
        "boundingPolygon",
        "points",
        "vertices",
    )
    for key in bbox_keys:
        if key in node and node[key] is not None:
            return True
    # Inline x/y/w/h or x1/y1/x2/y2 also count as a bbox attempt.
    inline_keys = ("x", "y", "w", "h", "left", "top", "width", "height", "x1", "y1", "x2", "y2", "right", "bottom")
    return any(key in node and node[key] is not None for key in inline_keys)


def _bbox_from_mapping(node: Mapping[str, Any]) -> list[float] | None:
    if not node:
        return None

    x = _coerce_float(_first_present(node, ("x", "left")))
    y = _coerce_float(_first_present(node, ("y", "top")))
    width = _coerce_float(_first_present(node, ("w", "width")))
    height = _coerce_float(_first_present(node, ("h", "height")))
    if x is not None and y is not None and width is not None and height is not None:
        return [x, y, width, height]

    x1 = _coerce_float(_first_present(node, ("x1", "left")))
    y1 = _coerce_float(_first_present(node, ("y1", "top")))
    x2 = _coerce_float(_first_present(node, ("x2", "right")))
    y2 = _coerce_float(_first_present(node, ("y2", "bottom")))
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2 - x1, y2 - y1]
    return None


def _bbox_from_sequence(raw: Any) -> list[float] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 4:
        return None
    values = [_coerce_float(value) for value in raw]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _extract_polygon(node: Mapping[str, Any]) -> list[list[float]] | None:
    for key in ("polygon", "bounding_polygon", "boundingPolygon", "points", "vertices"):
        polygon = _polygon_from_raw(node.get(key))
        if polygon is not None:
            return polygon
    return None


def _polygon_from_raw(raw: Any) -> list[list[float]] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    if not raw:
        return None

    points: list[list[float]] = []
    if all(isinstance(point, Mapping) for point in raw):
        for point in raw:
            point_map = _as_mapping(point)
            x = _coerce_float(point_map.get("x"))
            y = _coerce_float(point_map.get("y"))
            if x is None or y is None:
                return None
            points.append([x, y])
    elif all(isinstance(point, Sequence) and not isinstance(point, (str, bytes, bytearray)) for point in raw):
        for point in raw:
            if len(point) < 2:
                return None
            x = _coerce_float(point[0])
            y = _coerce_float(point[1])
            if x is None or y is None:
                return None
            points.append([x, y])
    else:
        values = [_coerce_float(value) for value in raw]
        if len(values) % 2 != 0 or any(value is None for value in values):
            return None
        numeric_values = [float(value) for value in values if value is not None]
        points = [[numeric_values[index], numeric_values[index + 1]] for index in range(0, len(numeric_values), 2)]

    return points if len(points) >= 2 else None


def _bbox_from_polygon(polygon: list[list[float]] | None) -> list[float] | None:
    if not polygon:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left = min(xs)
    top = min(ys)
    return [left, top, max(xs) - left, max(ys) - top]


def _normalize_bbox(raw_bbox: list[float] | None, dimensions: tuple[float, float] | None) -> list[float] | None:
    if raw_bbox is None or len(raw_bbox) != 4:
        return None
    x, y, width, height = raw_bbox
    if width <= 0 or height <= 0:
        return None

    if _looks_normalized(raw_bbox):
        normalized = raw_bbox
    elif dimensions is not None:
        page_width, page_height = dimensions
        normalized = [x / page_width, y / page_height, width / page_width, height / page_height]
    else:
        return None

    if not _looks_normalized(normalized):
        return None
    return [round(value, 8) for value in normalized]


def _normalize_polygon(
    polygon: list[list[float]] | None,
    dimensions: tuple[float, float] | None,
) -> list[list[float]] | None:
    if polygon is None:
        return None
    flat = [coordinate for point in polygon for coordinate in point]
    if all(0 <= value <= 1 for value in flat):
        return [[round(point[0], 8), round(point[1], 8)] for point in polygon]
    if dimensions is None:
        return None
    page_width, page_height = dimensions
    normalized = [[point[0] / page_width, point[1] / page_height] for point in polygon]
    if not all(0 <= value <= 1 for point in normalized for value in point):
        return None
    return [[round(point[0], 8), round(point[1], 8)] for point in normalized]


def _looks_normalized(bbox: list[float]) -> bool:
    x, y, width, height = bbox
    return (
        0 <= x <= 1
        and 0 <= y <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x + width <= 1.000001
        and y + height <= 1.000001
    )


def _extract_reference_text(node: Mapping[str, Any]) -> str | None:
    value = _first_present(
        node, ("reference_text", "referenceText", "matching_text", "matchingText", "text", "content", "value")
    )
    if isinstance(value, str):
        return value
    return None


def _extract_confidence(node: Mapping[str, Any]) -> float | None:
    confidence = _coerce_float(_first_present(node, ("confidence", "extraction_confidence", "score", "probability")))
    if confidence is None or math.isnan(confidence) or math.isinf(confidence):
        return None
    return confidence


def _compact_metadata(node: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = {
        key: value
        for key, value in node.items()
        if key
        not in {
            "bbox",
            "bounding_box",
            "boundingBox",
            "box",
            "bounding_boxes",
            "polygon",
            "bounding_polygon",
            "boundingPolygon",
            "points",
            "vertices",
        }
    }
    return dict(metadata) if metadata else None


def _dedupe(citations: Sequence[FieldCitation]) -> list[FieldCitation]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[FieldCitation] = []
    for citation in citations:
        key = (
            citation.field_path,
            citation.page,
            tuple(citation.bbox) if citation.bbox is not None else None,
            citation.reference_text,
            citation.source,
            # Distinct scores with no location belong to the same field, but
            # must reach the confidence scorer for its per-field aggregation.
            citation.confidence if citation.page is None and citation.bbox is None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _first_present(node: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in node:
            return node[key]
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Chunk-bbox refinement (LandingAI cover-scalar fix)
#
# LandingAI's ``extraction_metadata.references`` for cover-page scalars all
# point to the same chunk-level grounding UUID — one rectangle covering the
# whole header section, reused by every leaf field. We refine those coarse
# bboxes by text-searching for the extracted value within the chunk
# rectangle, replacing the coarse bbox with a tight value-anchored bbox when
# found. Cell-level groundings (``tableCell``) are already tight; gating on
# ``metadata.type == "chunkText"`` keeps them untouched.
# ---------------------------------------------------------------------------

# Grounding types whose bboxes are eligible for value-text refinement. Add
# new entries here only after confirming the source bbox is genuinely coarse
# (a section/line block, not an already-tight cell).
_REFINABLE_GROUNDING_TYPES = frozenset({"chunkText"})


def _refine_chunk_citations_via_text_search(
    citations: list[FieldCitation],
    *,
    pdf_path: Path,
    extracted_data: Mapping[str, Any],
) -> list[FieldCitation]:
    """Refine coarse chunk-level citations to tight value-anchored bboxes.

    For each citation whose ``metadata.type`` is in
    :data:`_REFINABLE_GROUNDING_TYPES`, look up the extracted value at
    ``citation.field_path``, open the PDF page, clip to the citation's bbox,
    and run :func:`fitz.Page.search_for` for the value text. If found, replace
    the bbox with the search hit. Otherwise keep the original.

    Falls back gracefully (returns ``citations`` unchanged) when:

    * ``pdf_path`` does not exist or fails to open
    * The citation's grounding type is not refinable
    * The extracted value cannot be resolved or has no useful text form
    * No search hit lands inside the chunk rectangle
    """
    import fitz  # PyMuPDF; declared as a hard dep in pyproject.toml

    if not pdf_path.exists():
        logger.debug("PDF not found at %s; skipping chunk-citation refinement.", pdf_path)
        return citations

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to open PDF %s for chunk refinement: %s", pdf_path, exc)
        return citations

    refined: list[FieldCitation] = []
    try:
        for citation in citations:
            metadata = citation.metadata or {}
            grounding_type = metadata.get("type")
            if grounding_type not in _REFINABLE_GROUNDING_TYPES:
                refined.append(citation)
                continue
            value = _get_value_at_field_path(extracted_data, citation.field_path)
            queries = _value_search_candidates(value)
            if not queries:
                refined.append(citation)
                continue
            if citation.page is None:
                refined.append(citation)
                continue
            page_index = citation.page - 1
            if page_index < 0 or page_index >= document.page_count:
                refined.append(citation)
                continue
            if citation.bbox is None or len(citation.bbox) != 4:
                refined.append(citation)
                continue
            page = document[page_index]
            page_width, page_height = page.rect.width, page.rect.height
            x, y, w, h = citation.bbox
            clip = fitz.Rect(
                x * page_width,
                y * page_height,
                (x + w) * page_width,
                (y + h) * page_height,
            )
            new_bbox = _search_for_value_bbox(page, queries, clip, page_width, page_height)
            if new_bbox is None:
                refined.append(citation)
                continue
            refined_metadata = dict(metadata)
            refined_metadata["refined_from"] = grounding_type
            refined.append(citation.model_copy(update={"bbox": new_bbox, "metadata": refined_metadata}))
    finally:
        document.close()

    return refined


def _search_for_value_bbox(
    page: Any,
    queries: list[str],
    clip: Any,
    page_width: float,
    page_height: float,
) -> list[float] | None:
    """Return a normalized ``[x, y, w, h]`` bbox for a query hit inside ``clip``.

    Queries are tried in order; the first query producing any hit wins.
    Within a query, :func:`fitz.Page.search_for` returns rectangles in
    reading order (top-to-bottom, left-to-right), so we pick ``rects[0]`` —
    the first occurrence of the value text inside the chunk. This matches
    the typical SEC-form layout convention where label and value sit on the
    same line in reading order, so the first-hit value bbox is the one a
    human annotator would have picked.

    We empirically verified this against gold annotations on the SEC-form
    documents in the corpus; alternative tie-breakers like "closest to chunk center"
    regress the metric on real cover pages where the same value appears once
    at the cover header and once again later in the chunk (e.g. the filer
    name on an NPORT cover repeats in a series-info row below).

    Caveat: when a value is so short and ambiguous that it legitimately
    appears multiple times inside the chunk (a bare ``"52"``), the first
    occurrence may not be the one the schema field refers to. Callers that
    are sensitive to this should ship a tighter chunk bbox or use cell-level
    groundings — both of which the LandingAI provider already does for
    ``tableCell`` references, so the failure mode is bounded to coarse
    ``chunkText`` references.
    """
    if page_width <= 0 or page_height <= 0:
        return None
    for query in queries:
        try:
            rects = page.search_for(query, clip=clip)
        except Exception:  # pragma: no cover - defensive against fitz quirks
            continue
        if not rects:
            continue
        chosen = rects[0]
        return [
            chosen.x0 / page_width,
            chosen.y0 / page_height,
            (chosen.x1 - chosen.x0) / page_width,
            (chosen.y1 - chosen.y0) / page_height,
        ]
    return None


def _value_search_candidates(value: Any) -> list[str]:
    """Build a deduped list of plausible PDF text variants for an extracted value.

    Returns an empty list when the value is not searchable (None, empty, bool).
    For ints/floats: includes the bare numeric, comma-grouped, and ``$``-prefixed
    forms. For ISO dates ``YYYY-MM-DD``: also tries ``MM-DD-YYYY`` and
    ``MM/DD/YYYY`` (common SEC-form rendering). For strings: returns the
    stripped value. Order is preserved (first occurrence wins) so callers
    that try queries in order still get the most-specific form first.
    """
    if value is None or isinstance(value, bool):
        return []
    candidates: list[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        candidates.append(stripped)
        if len(stripped) == 10 and stripped[4] == "-" and stripped[7] == "-":
            year, month, day = stripped[:4], stripped[5:7], stripped[8:10]
            if year.isdigit() and month.isdigit() and day.isdigit():
                candidates.append(f"{month}-{day}-{year}")
                candidates.append(f"{month}/{day}/{year}")
    elif isinstance(value, int):
        formatted_int = f"{value:,}"
        candidates.extend([str(value), formatted_int, f"${formatted_int}"])
    elif isinstance(value, float):
        candidates.append(f"{value:g}")
        if value.is_integer():
            as_int = int(value)
            candidates.extend([str(as_int), f"{as_int:,}", f"${as_int:,}"])
    else:
        return []
    # Order-preserving dedupe: a small integer like ``52`` produces ``"52"``,
    # ``"52"`` again from comma-grouped, and ``"$52"`` — three searches collapse
    # to two distinct queries.
    return list(dict.fromkeys(candidates))


def _get_value_at_field_path(data: Mapping[str, Any] | Any, field_path: str) -> Any:
    """Resolve dotted/bracketed field paths like ``holdings[0].issuer_name``.

    Returns ``None`` when any segment cannot be resolved or the path is
    malformed (e.g. unclosed brackets — the tokenizer signals this by
    returning an empty list, which we must not silently treat as "root
    value" since the caller asked for a real path).
    """
    if not field_path:
        return None
    tokens = _tokenize_field_path(field_path)
    if not tokens:
        return None
    node: Any = data
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(node, list) or token < 0 or token >= len(node):
                return None
            node = node[token]
        else:
            if not isinstance(node, Mapping) or token not in node:
                return None
            node = node[token]
    return node


def _tokenize_field_path(field_path: str) -> list[Any]:
    """Split ``a.b[2].c`` into ``["a", "b", 2, "c"]``."""
    tokens: list[Any] = []
    current = ""
    i = 0
    while i < len(field_path):
        ch = field_path[i]
        if ch == ".":
            if current:
                tokens.append(current)
                current = ""
            i += 1
            continue
        if ch == "[":
            if current:
                tokens.append(current)
                current = ""
            close = field_path.find("]", i)
            if close == -1:
                return []
            try:
                tokens.append(int(field_path[i + 1 : close]))
            except ValueError:
                return []
            i = close + 1
            continue
        current += ch
        i += 1
    if current:
        tokens.append(current)
    return tokens


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None
