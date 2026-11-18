from __future__ import annotations

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")
pytest.importorskip("datalab_sdk", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import compute_unified_evidence_metrics
from extract_bench.inference.providers.extract.datalab import (
    _attach_datalab_extract_pricing,
    _build_block_lookup,
    _build_field_citations,
    _promote_api_cost_breakdown,
    _strip_citation_and_score_keys,
)
from extract_bench.test_cases.schema import ExtractFieldTestRule, FieldEvidence


def _meta() -> dict:
    return {
        "extraction_status": "EXTRACTED",
        "reasoning": None,
        "citations": ["/page/0/Table/7"],
        "verification": {"status": "PASS", "feedback": "ok"},
    }


def test_strip_removes_meta_alongside_citations_and_score() -> None:
    payload = {
        "invoice_number": "NF67652",
        "invoice_number_meta": _meta(),
        "invoice_number_citations": ["/page/0/Table/7"],
        "total": 1382.0,
        "total_meta": _meta(),
        "total_score": {"score": 5},
    }

    assert _strip_citation_and_score_keys(payload) == {
        "invoice_number": "NF67652",
        "total": 1382.0,
    }


def test_strip_removes_nested_meta_keys() -> None:
    payload = {
        "line_items": [
            {
                "description": "widget",
                "description_meta": _meta(),
                "amount": 10.0,
                "amount_meta": _meta(),
            }
        ],
        "vendor": {"name": "ACME", "name_meta": _meta()},
    }

    assert _strip_citation_and_score_keys(payload) == {
        "line_items": [{"description": "widget", "amount": 10.0}],
        "vendor": {"name": "ACME"},
    }


def test_meta_sidecar_recovers_citation_and_verification() -> None:
    payload = {"invoice_number": "NF67652", "invoice_number_meta": _meta()}

    citations = _build_field_citations(payload, block_lookup={})
    assert len(citations) == 1
    assert citations[0].metadata["block_id"] == "/page/0/Table/7"
    assert citations[0].confidence == 1.0


@pytest.mark.parametrize(
    ("parse_mode", "extraction_mode", "expected_cost_per_page"),
    [
        ("fast", "fast", 0.010),
        ("accurate", "balanced", 0.025),
        ("accurate", "accurate", 0.030),
    ],
)
def test_explicit_parse_pricing_includes_extraction_cost(
    parse_mode: str,
    extraction_mode: str,
    expected_cost_per_page: float,
) -> None:
    raw_output: dict[str, object] = {}
    _attach_datalab_extract_pricing(raw_output, page_count=10, parse_mode=parse_mode, extraction_mode=extraction_mode)

    assert raw_output["cost_per_page_usd"] == pytest.approx(expected_cost_per_page)
    assert raw_output["cost_usd"] == pytest.approx(expected_cost_per_page * 10)


def test_api_cost_takes_precedence_over_base_rate() -> None:
    raw_output: dict[str, object] = {
        "page_count": 7,
        "cost_breakdown": {"final_cost_cents": 48.0, "list_cost_cents": 50.0},
    }
    _attach_datalab_extract_pricing(raw_output, page_count=7, parse_mode="accurate", extraction_mode="accurate")
    _promote_api_cost_breakdown(raw_output)

    assert raw_output["cost_usd"] == pytest.approx(0.48)
    assert raw_output["cost_per_page_usd"] == pytest.approx(0.48 / 7)


def test_unresolved_block_id_does_not_claim_page_one() -> None:
    citations = _build_field_citations({"amount": 10, "amount_citations": ["missing-block"]}, block_lookup={})
    assert len(citations) == 1
    assert citations[0].page is None
    assert citations[0].bbox is None


@pytest.mark.parametrize(("interval", "expired"), [(None, False), (0.5, False), (2.0, False), (2.0, True)])
def test_sdk_poll_allowance_uses_remaining_time_and_keeps_fractional_interval(
    monkeypatch: pytest.MonkeyPatch,
    interval: float | None,
    expired: bool,
) -> None:
    import asyncio
    from dataclasses import dataclass
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from extract_bench.inference import limits
    from extract_bench.inference.providers.base import ProviderPermanentError
    from extract_bench.inference.providers.extract import datalab

    @dataclass
    class Result:
        extraction_schema_json: str = "{}"
        page_count: int = 1

    now = 0.0
    monkeypatch.setattr(limits, "time", SimpleNamespace(monotonic=lambda: now))
    budget = limits.ExecutionBudget(10.0)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.extract.return_value = Result()
    monkeypatch.setattr(datalab, "AsyncDatalabClient", lambda **kwargs: client)
    provider = datalab.DatalabExtractProvider("datalab_extract", {"api_key": "test", "poll_interval": interval})
    with budget.activate():
        now = 11.0 if expired else 6.75
        if expired:
            with pytest.raises(ProviderPermanentError, match="deadline exceeded"):
                asyncio.run(provider._extract_pdf_async("unused.pdf", {"type": "object"}))
            client.extract.assert_not_awaited()
            return
        result = asyncio.run(provider._extract_pdf_async("unused.pdf", {"type": "object"}))
    cadence = limits.DEFAULT_POLL_INTERVAL_S if interval is None else interval
    kwargs = client.extract.await_args.kwargs
    assert kwargs["poll_interval"] == cadence
    assert 3.25 <= kwargs["max_polls"] * cadence < 3.25 + cadence
    assert result["extraction_schema_json"] == "{}"


@pytest.mark.parametrize("multiline", [False, True])
def test_phrase_citations_use_tight_boxes_and_keep_verification_separate(multiline: bool) -> None:
    block_id = "/page/0/Text/3"
    phrase = {
        "block": block_id,
        "page": 0,
        "text": "Invoice number",
        "offset": 16,
        "length": 14,
        "score": 0.7,
        "bbox": [20, 40, 80, 120],
    }
    if multiline:
        phrase["bboxes"] = [[20, 40, 80, 60], [20, 100, 60, 120]]
    payload = {
        "items": [
            {
                "number": "Invoice number",
                "number_citations": [phrase],
                "number_meta": {"citations": [block_id], "verification": {"status": "FAIL_CITATIONS"}},
            }
        ],
    }
    lookup = {
        block_id: {
            "page_number": 1,
            "page_w": 200,
            "page_h": 400,
            "bbox": [0, 0, 200, 400],
            "html": "whole block",
            "block_type": "Text",
        }
    }

    citations = _build_field_citations(payload, lookup)

    envelope = [0.1, 0.1, 0.3, 0.2]
    data = {"items": [{"number": "Invoice number"}]}
    metrics = compute_unified_evidence_metrics(
        data,
        data,
        [ExtractFieldTestRule(field_path="items[0].number", evidence=[FieldEvidence(page=1, bbox=envelope)])],
        field_citations=citations,
        data_schema={
            "properties": {"items": {"type": "array", "items": {"properties": {"number": {"type": "string"}}}}}
        },
    )
    assert next(m.value for m in metrics if m.metric_name == "extract_unified_grounded_f1") == 1.0
    expected = [envelope]
    if multiline:
        expected.extend([[0.1, 0.1, 0.3, 0.05], [0.1, 0.25, 0.2, 0.05]])
    assert [citation.bbox for citation in citations] == expected
    for citation in citations:
        assert citation.field_path == "items[0].number"
        assert citation.page == 1
        assert citation.reference_text == "Invoice number"
        assert citation.confidence == 1.0
        assert citation.metadata["native_verification_status"] == "FAIL_CITATIONS"
        assert citation.metadata["datalab_phrase"] == phrase


@pytest.mark.parametrize("known_block", [False, True])
def test_phrase_without_word_box_keeps_available_block_or_page_evidence(known_block: bool) -> None:
    block_id = "/page/1/Text/2"
    phrase = {"block": block_id, "page": 1, "value_index": 0}
    payload = {"names": ["Alice"], "names_citations": [phrase]}
    lookup = (
        {
            block_id: {
                "page_number": 2,
                "page_w": 200,
                "page_h": 400,
                "bbox": [20, 40, 80, 120],
            }
        }
        if known_block
        else {}
    )

    citations = _build_field_citations(payload, lookup)

    assert len(citations) == 1
    assert citations[0].field_path == "names"
    assert citations[0].page == 2
    assert citations[0].bbox == ([0.1, 0.1, 0.3, 0.2] if known_block else None)
    assert citations[0].confidence is None
    assert citations[0].metadata["datalab_phrase"]["value_index"] == 0


@pytest.mark.parametrize("numbers", [[0, 1], [1, 2], [8, 9], [None, None]])
def test_legacy_block_citations_preserve_distinct_document_pages(numbers: list[int | None]) -> None:
    tree = {
        "children": [
            {"block_type": "Page", "page_number": number, "children": [{"id": f"block-{i}"}]}
            for i, number in enumerate(numbers)
        ]
    }
    lookup = _build_block_lookup(tree)
    citations = _build_field_citations({"name": "A", "name_citations": ["block-0", "block-1"]}, lookup)
    assert [c.page for c in citations] == ([8, 9] if numbers == [8, 9] else [1, 2])
