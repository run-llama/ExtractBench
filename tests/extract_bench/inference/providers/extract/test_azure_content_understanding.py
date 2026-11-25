"""Azure Content Understanding result validation and citation geometry."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from extract_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from extract_bench.inference.providers.extract.azure_content_understanding import AzureContentUnderstandingProvider
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from extract_bench.schemas.product import ProductType

_PROVIDER = "azure_content_understanding"
_OPERATION_URL = "https://example.invalid/operations/1"


@pytest.fixture
def provider() -> AzureContentUnderstandingProvider:
    return AzureContentUnderstandingProvider(_PROVIDER, {"endpoint": "https://example.invalid", "api_key": "offline"})


def _pipeline() -> PipelineSpec:
    return PipelineSpec(pipeline_name="azure_extract", provider_name=_PROVIDER, product_type=ProductType.EXTRACT)


def _request(path: Path) -> InferenceRequest:
    return InferenceRequest(
        example_id="invoice",
        source_file_path=str(path),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"total": {"type": "number"}}},
    )


def _normalize(provider: AzureContentUnderstandingProvider, content: dict[str, Any]) -> ExtractOutput:
    pipeline = _pipeline()
    now = datetime.now(UTC)
    raw = RawInferenceResult(
        request=_request(Path("invoice.pdf")),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.EXTRACT,
        raw_output={"result": {"contents": [{"kind": "document", **content}]}},
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    output = provider.normalize(raw).output
    assert isinstance(output, ExtractOutput)
    return output


@pytest.mark.parametrize(
    ("source", "width", "expected"),
    [
        ("D(1,0.2,0.2,0.8,0.2,0.8,0.8,0.2,0.8)", 10, [0.02, 0.02, 0.06, 0.06]),
        ("D(1,2,3,1,2)", 10, [0.2, 0.3, 0.1, 0.2]),
        ("D(1,8,9,4,3)", 10, [0.8, 0.9, 0.2, 0.1]),
        ("D(1,0.2,0.2,0.8,0.2,0.8,0.8,0.2,0.8)", float("inf"), [0.2, 0.2, 0.6, 0.6]),
    ],
    ids=["sub-unit-polygon", "rectangle-extents", "clipped-rectangle", "invalid-page-width"],
)
def test_source_coordinates_normalize_to_page(
    provider: AzureContentUnderstandingProvider,
    source: str,
    width: float,
    expected: list[float],
) -> None:
    output = _normalize(
        provider,
        {
            "pages": [{"pageNumber": 1, "width": width, "height": 10, "unit": "inch"}],
            "fields": {"total": {"type": "number", "valueNumber": 42, "source": source}},
        },
    )

    assert output.extracted_data == {"total": 42}
    (citation,) = output.field_citations
    assert citation.page == 1
    assert citation.bbox == pytest.approx(expected)


def test_multi_page_citations_keep_page_local_boxes_and_reference_text(
    provider: AzureContentUnderstandingProvider,
) -> None:
    output = _normalize(
        provider,
        {
            "markdown": "Label Alpha Beta Gamma",
            "pages": [{"pageNumber": 1, "width": 10, "height": 10}, {"pageNumber": 2, "width": 20, "height": 20}],
            "fields": {
                "description": {
                    "type": "string",
                    "valueString": "Alpha Beta Gamma",
                    "confidence": 0.9,
                    "source": "D(1,8,8,1,1);D(1,1,2,1,1);D(1,2,2,1,1);D(2,4,6,2,2)",
                    "spans": [
                        {"offset": 0, "length": 5},
                        {"offset": 6, "length": 5},
                        {"offset": 12, "length": 4},
                        {"offset": 17, "length": 5},
                    ],
                }
            },
        },
    )

    first, second = output.field_citations
    assert [(citation.page, citation.reference_text) for citation in (first, second)] == [
        (1, "Alpha Beta"),
        (2, "Gamma"),
    ]
    assert first.bbox == pytest.approx([0.1, 0.2, 0.2, 0.1])
    assert second.bbox == pytest.approx([0.2, 0.3, 0.1, 0.1])
    assert all(citation.field_path == "description" and citation.confidence == 0.9 for citation in (first, second))


def _response(body: dict[str, Any], status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode()
    return response


def _mock_analysis(
    monkeypatch: pytest.MonkeyPatch,
    provider: AzureContentUnderstandingProvider,
    body: dict[str, Any],
    *,
    polled: bool,
) -> None:
    responses = [_response({"status": "ready"})]
    if polled:
        accepted = _response({"status": "running"}, status_code=202)
        accepted.headers["Operation-Location"] = _OPERATION_URL
        responses.append(accepted)
    responses.append(_response(body))
    monkeypatch.setattr(provider._session, "request", Mock(side_effect=responses))


@pytest.mark.parametrize("polled", [False, True], ids=["inline", "polled"])
@pytest.mark.parametrize(
    "contents",
    [[], [{"kind": "image", "fields": {}}], [{"kind": "document", "fields": None}]],
    ids=["no-content", "wrong-content-kind", "missing-field-object"],
)
def test_malformed_analysis_is_rejected_and_keeps_reported_usage(
    provider: AzureContentUnderstandingProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    polled: bool,
    contents: list[dict[str, Any]],
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"document fixture")
    usage = {"documentPagesStandard": 2}
    body = {"status": "succeeded", "usage": usage, "result": {"contents": contents}}
    _mock_analysis(monkeypatch, provider, body, polled=polled)

    with pytest.raises(ProviderTransientError, match="incomplete or malformed result") as caught:
        provider.run_inference(_pipeline(), _request(source))

    assert caught.value.job_id == (_OPERATION_URL if polled else None)
    assert caught.value.debug_payload["usage"] == usage
    assert caught.value.debug_payload["cost_accounting"]["known_cost_usd"] == pytest.approx(0.01)


@pytest.mark.parametrize("polled", [False, True], ids=["inline", "polled"])
def test_completed_document_can_have_no_fields(
    provider: AzureContentUnderstandingProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    polled: bool,
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"document fixture")
    result = {"contents": [{"kind": "document", "fields": {}}]}
    body = {"status": "succeeded", "result": result} if polled else result
    _mock_analysis(monkeypatch, provider, body, polled=polled)

    raw = provider.run_inference(_pipeline(), _request(source))

    output = provider.normalize(raw).output
    assert isinstance(output, ExtractOutput)
    assert output.extracted_data == {}


def test_failed_inline_analysis_is_permanent(
    provider: AzureContentUnderstandingProvider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"document fixture")
    body = {"status": "failed", "error": {"code": "InvalidDocument"}}
    _mock_analysis(monkeypatch, provider, body, polled=False)

    with pytest.raises(ProviderPermanentError, match="failed") as caught:
        provider.run_inference(_pipeline(), _request(source))

    assert caught.value.debug_payload["error"] == {"code": "InvalidDocument"}
