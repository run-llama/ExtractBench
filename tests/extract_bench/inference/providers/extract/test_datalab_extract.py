from __future__ import annotations

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")
pytest.importorskip("datalab_sdk", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.extract.datalab import (
    DatalabExtractProvider,
    _attach_datalab_extract_pricing,
    _build_field_citations,
    _promote_api_cost_breakdown,
    _strip_citation_and_score_keys,
)


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


def test_meta_sidecar_produces_no_citations() -> None:
    payload = {"invoice_number": "NF67652", "invoice_number_meta": _meta()}

    assert _build_field_citations(payload, block_lookup={}) == []


@pytest.mark.parametrize(
    ("pipeline_name", "parse_mode", "extraction_mode", "expected_cost_per_page"),
    [
        ("datalab_parse_fast_extract_fast", "fast", "fast", 0.010),
        ("datalab_parse_accurate_extract_balanced", "accurate", "balanced", 0.025),
        ("datalab_parse_accurate_extract_accurate", "accurate", "accurate", 0.030),
    ],
)
def test_explicit_datalab_extract_pipeline_pricing(
    pipeline_name: str, parse_mode: str, extraction_mode: str, expected_cost_per_page: float
) -> None:
    pipeline = get_pipeline(pipeline_name)
    assert pipeline.config["parse_mode"] == parse_mode
    assert pipeline.config["extraction_mode"] == extraction_mode
    assert pipeline.config["skip_cache"] is True
    DatalabExtractProvider("datalab_extract", {**pipeline.config, "api_key": "test-key"})

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
