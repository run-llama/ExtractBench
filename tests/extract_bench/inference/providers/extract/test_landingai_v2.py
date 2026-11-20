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
