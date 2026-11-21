"""LandingAI ADE v2 provider: job flow, settings, cost and atomic (word-level) citations."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("landingai_ade")

from extract_bench.inference.cost_contract import COST_PER_PAGE_USD, COST_USD, RAW_COST_COMPLETE
from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.citations import extract_landingai_v2_field_citations
from extract_bench.inference.providers.extract.landingai_v2 import LandingAIADEV2ExtractProvider

MARKDOWN = "Invoice INV-7\nTotal: 42.00\n"


def _structure() -> dict:
    """One page: a text block with two atomic (word) spans, and a table cell holding the total."""
    return {
        "type": "document",
        "children": [
            {
                "type": "page",
                "page": 1,
                "children": [
                    {
                        "type": "text",
                        "id": "text-0",
                        "grounding": {
                            "page": 1,
                            "range": {"start": 0, "end": 13},
                            "box": {"xmin": 0.1, "ymin": 0.1, "xmax": 0.9, "ymax": 0.3},
                        },
                        "atomic_grounding": [
                            {
                                "page": 1,
                                "range": {"start": 0, "end": 7},
                                "box": {"xmin": 0.1, "ymin": 0.1, "xmax": 0.3, "ymax": 0.15},
                            },
                            {
                                "page": 1,
                                "range": {"start": 8, "end": 13},
                                "box": {"xmin": 0.32, "ymin": 0.1, "xmax": 0.45, "ymax": 0.15},
                            },
                        ],
                    },
                    {
                        "type": "table",
                        "id": "table-0",
                        "grounding": {
                            "page": 1,
                            "range": {"start": 14, "end": 26},
                            "box": {"xmin": 0.1, "ymin": 0.5, "xmax": 0.9, "ymax": 0.7},
                        },
                        "children": [
                            {
                                "type": "table_cell",
                                "id": "table_cell-0",
                                "grounding": {
                                    "page": 1,
                                    "range": {"start": 21, "end": 26},
                                    "box": {"xmin": 0.6, "ymin": 0.55, "xmax": 0.8, "ymax": 0.6},
                                },
                            }
                        ],
                    },
                ],
            }
        ],
    }


def _job(status: str, job_id: str, result: dict | None = None, error: dict | None = None) -> SimpleNamespace:
    raw = {"job_id": job_id, "status": status}
    if result is not None:
        raw["result"] = result
    return SimpleNamespace(
        job_id=job_id,
        status=SimpleNamespace(value=status),
        progress=None,
        error=SimpleNamespace(**error) if error else None,
        raw=raw,
    )


def _provider(config: dict | None = None) -> LandingAIADEV2ExtractProvider:
    provider = LandingAIADEV2ExtractProvider.__new__(LandingAIADEV2ExtractProvider)
    config = config or {}
    provider._base_config = config
    provider._model = config.get("model", "extract-20261111")
    provider._parse_model = config.get("parse_model", "dpt-3-pro-20261111")
    provider._grounding = config.get("grounding", True)
    provider._service_tier = config.get("service_tier", "standard")
    provider._parse_options = None
    provider._poll_interval_s = 0.0
    provider._client = Mock()
    return provider


def _wire_success(provider: LandingAIADEV2ExtractProvider) -> None:
    parse_result = {
        "markdown": MARKDOWN,
        "structure": _structure(),
        "metadata": {"job_id": "parse-1", "page_count": 2, "billing": {"service_tier": "standard", "total_credits": 3}},
    }
    extract_result = {
        "extraction": {"invoice_number": "INV-7", "total": 42.0},
        "extraction_metadata": {
            "invoice_number": {"value": "INV-7", "ranges": [{"start": 8, "end": 13}]},
            "total": {"value": 42.0, "ranges": [{"start": 21, "end": 26}]},
        },
        "markdown": MARKDOWN,
        "metadata": {"job_id": "extract-1", "duration_ms": 10, "billing": {"total_credits": 7}},
    }
    v2 = provider._client.v2
    v2.parse_jobs.create.return_value = SimpleNamespace(job_id="parse-1")
    v2.parse_jobs.get.side_effect = [_job("processing", "parse-1"), _job("completed", "parse-1", parse_result)]
    v2.extract_jobs.create.return_value = SimpleNamespace(job_id="extract-1")
    v2.extract_jobs.get.side_effect = [_job("completed", "extract-1", extract_result)]


def test_parse_then_extract_uses_pinned_models_and_passes_schema_unchanged() -> None:
    provider = _provider()
    _wire_success(provider)
    schema = {"type": "object", "properties": {"invoice_number": {"type": ["string", "null"]}}}

    result = provider._extract_from_document(Path("doc.pdf"), schema)

    parse_kwargs = provider._client.v2.parse_jobs.create.call_args.kwargs
    assert parse_kwargs["model"] == "dpt-3-pro-20261111"
    assert parse_kwargs["service_tier"] == "standard"
    assert provider._client.v2.extract_jobs.create.call_args.kwargs["service_tier"] == "standard"
    extract_kwargs = provider._client.v2.extract_jobs.create.call_args.kwargs
    assert extract_kwargs["schema"] is schema
    assert extract_kwargs["markdown"] == MARKDOWN
    assert extract_kwargs["model"] == "extract-20261111"
    assert extract_kwargs["grounding"] is True
    assert result["data"] == {"invoice_number": "INV-7", "total": 42.0}


def test_cost_comes_from_v2_billing_for_both_stages() -> None:
    provider = _provider()
    _wire_success(provider)

    result = provider._extract_from_document(Path("doc.pdf"), {})

    assert result["parse_credits"] == 3
    assert result["extract_credits"] == 7
    assert result[COST_USD] == pytest.approx(0.10)
    assert result[RAW_COST_COMPLETE] is True
    assert result[COST_PER_PAGE_USD] == pytest.approx(0.05)


def test_failed_job_is_permanent_and_timeout_is_transient() -> None:
    provider = _provider()
    v2 = provider._client.v2
    v2.parse_jobs.create.return_value = SimpleNamespace(job_id="parse-1")
    v2.parse_jobs.get.return_value = _job("failed", "parse-1", error={"code": "invalid_pdf", "message": "bad file"})
    with pytest.raises(ProviderPermanentError):
        provider._extract_from_document(Path("doc.pdf"), {})

    v2.parse_jobs.get.return_value = _job("failed", "parse-1", error={"code": "timeout", "message": "timed out"})
    with pytest.raises(ProviderTransientError):
        provider._extract_from_document(Path("doc.pdf"), {})


def test_extract_job_timeout_fails_the_document_without_retry() -> None:
    provider = _provider()
    _wire_success(provider)
    v2 = provider._client.v2
    v2.extract_jobs.get.side_effect = None
    v2.extract_jobs.get.return_value = _job(
        "failed", "extract-1", error={"code": "timeout", "message": "Request timed out"}
    )

    with pytest.raises(ProviderPermanentError) as caught:
        provider._extract_from_document(Path("doc.pdf"), {})

    # The parse credits already spent are still reported.
    assert caught.value.debug_payload["cost_accounting"]["known_cost_usd"] == pytest.approx(0.03)


def test_unknown_config_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANDING_AI_API_KEY", "test-key")
    with pytest.raises(ProviderConfigError):
        LandingAIADEV2ExtractProvider("landingai_ade_v2_extract", {"temperature": 0})


def test_citations_use_atomic_boxes_and_fall_back_to_leaf_blocks() -> None:
    metadata = {
        "invoice_number": {"value": "INV-7", "ranges": [{"start": 8, "end": 13}]},
        "total": {"value": 42.0, "ranges": [{"start": 21, "end": 26}]},
        "missing": {"value": None, "ranges": None},
    }
    citations = extract_landingai_v2_field_citations(metadata, _structure())
    by_path = {citation.field_path: citation for citation in citations}

    assert set(by_path) == {"invoice_number", "total"}
    word = by_path["invoice_number"]
    assert word.page == 1
    assert word.bbox == pytest.approx([0.32, 0.1, 0.13, 0.05])
    assert word.metadata["granularity"] == "atomic"
    cell = by_path["total"]
    assert cell.bbox == pytest.approx([0.6, 0.55, 0.2, 0.05])
    assert cell.metadata["granularity"] == "block"
    assert cell.metadata["block_type"] == "table_cell"


def test_citations_follow_array_paths() -> None:
    metadata = {"items": [{"amount": {"value": "42.00", "ranges": [{"start": 21, "end": 26}]}}]}
    citations = extract_landingai_v2_field_citations(metadata, _structure())
    assert [citation.field_path for citation in citations] == ["items[0].amount"]


def _word_structure(markdown: str, line_breaks: tuple[str, ...] = ("<br>", "\n")) -> dict:
    """One text block whose atomic spans are the whitespace-separated tokens of ``markdown``.

    Each token gets a 0.05-wide box; a token after a line break starts a new line 0.02 lower.
    """
    spans = []
    x, y, position = 0.1, 0.1, 0
    for token in markdown.split(" "):
        start = position
        position += len(token) + 1
        if not token:
            continue
        spans.append(
            {
                "page": 1,
                "range": {"start": start, "end": start + len(token)},
                "box": {"xmin": x, "ymin": y, "xmax": x + 0.05, "ymax": y + 0.01},
            }
        )
        x += 0.06
        if any(token.endswith(mark) for mark in line_breaks):
            x, y = 0.1, y + 0.02
    return {
        "type": "document",
        "children": [
            {
                "type": "page",
                "page": 1,
                "children": [
                    {
                        "type": "text",
                        "id": "text-0",
                        "grounding": {
                            "page": 1,
                            "range": {"start": 0, "end": len(markdown)},
                            "box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
                        },
                        "atomic_grounding": spans,
                    }
                ],
            }
        ],
    }


def test_value_words_are_kept_and_merged_without_the_label() -> None:
    markdown = "F Partner's name, address<br> ESK LLC<br> 237 MAYFAIR DRIVE<br> BROOKLYN, NY 11234"
    metadata = {
        "partner_address": {
            "value": "237 MAYFAIR DRIVE, BROOKLYN, NY 11234",
            "ranges": [{"start": 0, "end": len(markdown)}],
        }
    }

    citations = extract_landingai_v2_field_citations(metadata, _word_structure(markdown), markdown)

    assert len(citations) == 1
    citation = citations[0]
    # Two address lines: x from the first word to the end of "11234", y over both lines.
    assert citation.bbox == pytest.approx([0.1, 0.14, 0.17, 0.03])
    assert citation.metadata["granularity"] == "value"
    assert citation.metadata["span_count"] == 6


def test_value_without_ranges_is_located_in_the_markdown() -> None:
    markdown = "TELCO EXPERTS LLC<br> 38 PARK AVENUE<br> RUTHERFORD, NJ 07070"
    metadata = {"address": {"value": "38 PARK AVENUE, RUTHERFORD, NJ 07070", "ranges": None}}

    citations = extract_landingai_v2_field_citations(metadata, _word_structure(markdown), markdown)

    assert len(citations) == 1
    assert citations[0].metadata["range_source"] == "located"
    assert citations[0].metadata["range"] == [markdown.index("38"), len(markdown)]
    assert citations[0].bbox == pytest.approx([0.1, 0.12, 0.17, 0.03])


def test_numbers_without_ranges_are_located_as_printed() -> None:
    markdown = "Coupon 5.780 Par 1,000 Value 1,003,418 Shares 12"
    metadata = {
        "value_usd": {"value": 1003418, "ranges": None},
        "coupon": {"value": 5.78, "ranges": None},
        "shares": {"value": 12, "ranges": None},
        "flag": {"value": True, "ranges": None},
    }

    citations = extract_landingai_v2_field_citations(metadata, _word_structure(markdown), markdown)
    by_path = {citation.field_path: citation for citation in citations}

    assert by_path["value_usd"].metadata["range"] == [markdown.index("1,003,418"), markdown.index(" Shares")]
    assert by_path["coupon"].metadata["range"] == [markdown.index("5.780"), markdown.index(" Par")]
    # Under three digits is too ambiguous to search for; booleans are never searched.
    assert "shares" not in by_path
    assert "flag" not in by_path


def test_numbers_match_printed_digits_and_skip_label_numbers() -> None:
    markdown = "1 Ordinary income (loss) | -5,307."
    metadata = {"box_1": {"value": -5307, "ranges": [{"start": 0, "end": len(markdown)}]}}

    citations = extract_landingai_v2_field_citations(metadata, _word_structure(markdown), markdown)

    assert [c.bbox[0] for c in citations] == pytest.approx([0.1 + 5 * 0.06])


def test_booleans_cite_the_checkbox_mark() -> None:
    markdown = "H [x] Domestic partner"
    metadata = {"is_domestic": {"value": True, "ranges": [{"start": 0, "end": len(markdown)}]}}

    citations = extract_landingai_v2_field_citations(metadata, _word_structure(markdown), markdown)

    assert [c.bbox[0] for c in citations] == pytest.approx([0.16])


def test_word_fragments_join_only_next_to_matching_words() -> None:
    # The label shares the word "entity" with the value; it forms its own group and is dropped.
    markdown = "What type of entity is this partner? DISREGARDED ENTI TY"
    metadata = {"entity_type": {"value": "DISREGARDED ENTITY", "ranges": [{"start": 0, "end": len(markdown)}]}}

    citations = extract_landingai_v2_field_citations(metadata, _word_structure(markdown), markdown)

    assert len(citations) == 1
    assert citations[0].metadata["span_count"] == 3
    assert citations[0].bbox[0] == pytest.approx(0.1 + 7 * 0.06)


def test_unmatched_values_keep_the_word_boxes() -> None:
    markdown = "Total due 42.00"
    metadata = {"note": {"value": "paid in full", "ranges": [{"start": 0, "end": 9}]}}

    citations = extract_landingai_v2_field_citations(metadata, _word_structure(markdown), markdown)

    assert len(citations) == 2
    assert {c.metadata["granularity"] for c in citations} == {"atomic"}


def test_raw_output_keeps_the_parse_markdown() -> None:
    provider = _provider()
    _wire_success(provider)

    result = provider._extract_from_document(Path("doc.pdf"), {})

    assert result["parse_markdown"] == MARKDOWN


def test_booleans_prefer_marks_in_the_reported_state() -> None:
    markdown = "G [x] General partner [ ] Limited partner"
    structure = _word_structure(markdown)
    checked = {"flag": {"value": True, "ranges": [{"start": 0, "end": len(markdown)}]}}
    unchecked = {"flag": {"value": False, "ranges": [{"start": 0, "end": len(markdown)}]}}

    # Tokens: G, [x], General, partner, [, ], Limited, partner -> "[x]" is the 2nd token.
    assert [c.bbox[0] for c in extract_landingai_v2_field_citations(checked, structure, markdown)] == pytest.approx(
        [0.16]
    )
    # The unchecked mark is split into "[" and "]" here, so no token holds "[ ]": fall back to any mark.
    assert [c.bbox[0] for c in extract_landingai_v2_field_citations(unchecked, structure, markdown)] == pytest.approx(
        [0.16]
    )


def test_booleans_recognize_unicode_checkbox_symbols() -> None:
    markdown = "Form 4972 □ Other ☑ done"
    structure = _word_structure(markdown)
    unchecked = {"flag": {"value": False, "ranges": [{"start": 0, "end": len(markdown)}]}}
    checked = {"flag": {"value": True, "ranges": [{"start": 0, "end": len(markdown)}]}}

    assert [c.bbox[0] for c in extract_landingai_v2_field_citations(unchecked, structure, markdown)] == pytest.approx(
        [0.22]
    )
    assert [c.bbox[0] for c in extract_landingai_v2_field_citations(checked, structure, markdown)] == pytest.approx(
        [0.34]
    )


def test_attempt_cost_matches_the_sum_of_stage_charges() -> None:
    # 0.1 + 0.6 credits: (0.1 + 0.6) * 0.01 rounds to 0.006999…, below the two stage
    # charges summed (0.007), which the harness rejects as invalid billing.
    import math

    provider = _provider()
    rate = provider.CREDIT_RATE_USD
    snapshot = provider._cost_accounting({"parse_credits": 0.1, "extract_credits": 0.6})
    assert snapshot["known_cost_usd"] == math.fsum([0.1 * rate, 0.6 * rate])
