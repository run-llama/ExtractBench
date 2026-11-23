"""Offline tests for the PageIndex Page Extract provider: HTTP flow, billing, citations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from extract_bench.inference.limits import ExecutionBudget
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import (
    ProviderConfigError,
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.pageindex import (
    API_URL,
    MAX_POLL_ERRORS,
    UPLOAD_PRICE_PER_PAGE_USD,
    PageIndexExtractProvider,
    _citations,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType

SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "rows": {"type": "array", "items": {"type": "object", "properties": {"item": {"type": "string"}}}},
    },
}
RESULT = {"invoice_number": "A-1", "rows": [{"item": "x", "amount": 1}, {"item": "y", "amount": 2}]}
PDF = b"%PDF-1.4\n"

UPLOAD = ("POST", f"{API_URL}/doc")
DOCUMENT = ("GET", f"{API_URL}/doc/doc_1/metadata")
CREATE = ("POST", f"{API_URL}/extract")
POLL = ("GET", f"{API_URL}/extract/job_1")
CANCEL = ("POST", f"{API_URL}/extract/job_1/cancel")


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = "" if payload is None else str(payload)

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def _document(status: str) -> _FakeResponse:
    return _FakeResponse(200, {"id": "doc_1", "status": status, "pageNum": 12})


def _job(status: str) -> _FakeResponse:
    job: dict[str, Any] = {"id": "job_1", "status": status, "doc_id": "doc_1"}
    if status == "completed":
        job["extract_result"] = RESULT
        job["extract_metadata"] = {"field_metadata": {"/rows/1": {"page": 4}}}
        job["usage"] = {"pages": 12, "credits": 36}  # one credit is one US cent
    if status == "failed":
        job["error_code"] = "unfinished"
        job["error_message"] = "The document could not be finished."
    return _FakeResponse(200, job)


class _FakeSession:
    """Scripted API: upload (queued, then completed) -> create job -> poll (running, then completed).

    ``script[(method, url)]`` holds responses, or exceptions to raise, served first and in order.
    """

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.sent: list[dict[str, Any]] = []
        self.script: dict[tuple[str, str], list[Any]] = {}
        self.on_request: Any = None

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append((method, url))
        self.sent.append(kwargs)
        if self.on_request is not None:
            self.on_request(method, url)
        if self.script.get((method, url)):
            response = self.script[(method, url)].pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        if (method, url) == UPLOAD:
            return _FakeResponse(200, {"doc_id": "doc_1", "name": "doc.pdf", "status": "queued"})
        if (method, url) == DOCUMENT:
            return _document("completed")
        if (method, url) == CREATE:
            return _FakeResponse(202, {"id": "job_1", "status": "pending", "doc_id": "doc_1"})
        if (method, url) == POLL:
            return _job("completed")
        if (method, url) == CANCEL:
            return _job("cancelled")
        return _FakeResponse(404, {"detail": f"{method} {url}"})


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> tuple[PageIndexExtractProvider, _FakeSession]:
    monkeypatch.setenv("PAGEINDEX_API_KEY", "k")
    prov = PageIndexExtractProvider("pageindex", {"tier": "pro", "poll_interval": 0})
    fake = _FakeSession()
    prov._session = fake  # type: ignore[assignment]
    return prov, fake


def _request(tmp_path: Path) -> tuple[PipelineSpec, InferenceRequest]:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(PDF)
    pipeline = PipelineSpec(
        pipeline_name="pageindex_test", provider_name="pageindex", product_type=ProductType.EXTRACT, config={}
    )
    request = InferenceRequest(
        example_id="ex1", source_file_path=str(pdf), product_type=ProductType.EXTRACT, schema_override=SCHEMA
    )
    return pipeline, request


@pytest.mark.parametrize("tier", ["standard", "pro"])
def test_registered_pipelines(tier: str) -> None:
    pipeline = get_pipeline(f"pageindex_extract_{tier}")
    assert pipeline.provider_name == "pageindex"
    assert pipeline.config == {"tier": tier}


def test_unknown_tier_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAGEINDEX_API_KEY", "k")
    with pytest.raises(ProviderConfigError, match="tier must be one of"):
        PageIndexExtractProvider("pageindex", {"tier": "max"})


def test_missing_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAGEINDEX_API_KEY", raising=False)
    with pytest.raises(ProviderConfigError, match="PAGEINDEX_API_KEY"):
        PageIndexExtractProvider("pageindex", {"tier": "standard"})


def test_run_inference_uploads_extracts_and_bills_both(
    provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path
) -> None:
    prov, fake = provider
    fake.script[DOCUMENT] = [_document("processing")]
    fake.script[POLL] = [_job("running")]

    raw = prov.run_inference(*_request(tmp_path))

    assert fake.calls == [UPLOAD, DOCUMENT, DOCUMENT, CREATE, POLL, POLL]
    assert fake.sent[0]["data"] == {"algorithm": "none"}
    assert fake.sent[0]["files"]["file"][1] == PDF
    assert fake.sent[3]["data"]["doc_id"] == "doc_1"
    assert json.loads(fake.sent[3]["data"]["configuration"]) == {
        "data_schema": SCHEMA,
        "tier": "pro",
        "cite_sources": True,
    }
    upload_cost = 12 * UPLOAD_PRICE_PER_PAGE_USD
    assert raw.raw_output["upload_cost_usd"] == pytest.approx(upload_cost)
    assert raw.raw_output["extract_cost_usd"] == pytest.approx(0.36)
    assert raw.raw_output["cost_usd"] == pytest.approx(0.36 + upload_cost)
    assert raw.raw_output["cost_per_page_usd"] == pytest.approx(0.03 + UPLOAD_PRICE_PER_PAGE_USD)

    normalized = prov.normalize(type(raw).model_validate_json(raw.model_dump_json()))
    assert normalized.output.extracted_data == RESULT
    assert {(c.field_path, c.page, c.bbox) for c in normalized.output.field_citations} == {
        ("rows[1].item", 4, None),
        ("rows[1].amount", 4, None),
    }


def test_failed_job_is_retried_with_one_upload_charge_and_the_first_start(
    provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path
) -> None:
    prov, fake = provider
    fake.script[POLL] = [_job("failed")]
    with pytest.raises(ProviderTransientError, match="could not be finished") as failed:
        prov.run_inference(*_request(tmp_path))
    # The failed job is not charged: the attempt's whole bill is its upload.
    assert failed.value.debug_payload["cost_accounting"] == {
        "known_cost_usd": pytest.approx(12 * UPLOAD_PRICE_PER_PAGE_USD),
        "complete": True,
    }

    # The runner's retry uploads the same file again (free) and starts a new job.
    fake.script[UPLOAD] = [_FakeResponse(200, {"doc_id": "doc_1", "status": "completed", "duplicate": True})]
    first_start = prov._started["ex1"]
    raw = prov.run_inference(*_request(tmp_path))
    assert fake.calls.count(CREATE) == 2
    assert raw.started_at == first_start
    assert raw.raw_output["upload_cost_usd"] == 0
    assert raw.raw_output["cost_usd"] == pytest.approx(0.36)
    assert "ex1" not in prov._started


def test_cancelled_job_is_permanent(provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path) -> None:
    prov, fake = provider
    fake.script[POLL] = [_job("cancelled")]
    with pytest.raises(ProviderPermanentError, match="cancelled"):
        prov.run_inference(*_request(tmp_path))


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (_FakeResponse(503, {"detail": "The job could not be queued. Try again."}), ProviderTransientError),
        (_FakeResponse(502), ProviderTransientError),
        (_FakeResponse(429), ProviderRateLimitError),
        # The job may have started: a retry would bill a second one.
        (_FakeResponse(500), ProviderPermanentError),
        (requests.ReadTimeout("slow"), ProviderPermanentError),
        (_FakeResponse(402, {"detail": "insufficient_credits"}), ProviderPermanentError),
    ],
)
def test_job_creation_errors(
    provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path, response: Any, error: type
) -> None:
    prov, fake = provider
    fake.script[CREATE] = [response]
    with pytest.raises(error):
        prov.run_inference(*_request(tmp_path))
    assert POLL not in fake.calls


@pytest.mark.parametrize(
    ("script", "error"),
    [
        # Uploading the same file again is free, so the runner may retry the whole attempt.
        ({UPLOAD: [_FakeResponse(502)]}, ProviderTransientError),
        ({UPLOAD: [requests.ConnectionError("reset")]}, ProviderTransientError),
        ({UPLOAD: [_FakeResponse(429)]}, ProviderRateLimitError),
        ({UPLOAD: [_FakeResponse(400, {"detail": "The PDF is encrypted."})]}, ProviderPermanentError),
        ({DOCUMENT: [_document("failed")]}, ProviderPermanentError),
    ],
)
def test_upload_errors(
    provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path, script: dict[Any, Any], error: type
) -> None:
    prov, fake = provider
    fake.script.update(script)
    with pytest.raises(error):
        prov.run_inference(*_request(tmp_path))
    assert CREATE not in fake.calls


def test_polling_rides_out_transient_errors(
    provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path
) -> None:
    prov, fake = provider
    fake.script[POLL] = [requests.ConnectionError("reset"), _FakeResponse(502), _FakeResponse(429)]
    raw = prov.run_inference(*_request(tmp_path))
    assert raw.raw_output["job"]["status"] == "completed"
    assert fake.calls.count(CREATE) == 1


def test_polling_errors_in_a_row_cancel_the_job(
    provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path
) -> None:
    prov, fake = provider
    fake.script[POLL] = [_FakeResponse(503)] * MAX_POLL_ERRORS
    with pytest.raises(ProviderPermanentError, match="Polling job job_1 failed"):
        prov.run_inference(*_request(tmp_path))
    assert fake.calls[-1] == CANCEL


def test_deadline_cancels_the_job(provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path) -> None:
    prov, fake = provider
    budget = ExecutionBudget(3600)
    fake.script[POLL] = [_job("running")]
    # The runner's per-file timeout fires while the job is still running.
    fake.on_request = lambda method, url: budget.cancel() if (method, url) == POLL else None
    with budget.activate(), pytest.raises(TimeoutError):
        prov.run_inference(*_request(tmp_path))
    assert fake.calls[-1] == CANCEL


def test_missing_result_is_a_failure(provider: tuple[PageIndexExtractProvider, _FakeSession], tmp_path: Path) -> None:
    prov, fake = provider
    raw = prov.run_inference(*_request(tmp_path))
    raw.raw_output["job"]["extract_result"] = None
    with pytest.raises(ProviderPermanentError, match="no JSON object"):
        prov.normalize(raw)


def test_citations_follow_pointers_and_skip_what_they_cannot_place() -> None:
    data = {"a/b": [{"x": 1, "y": {"z": 2}}], "tags": ["t0", "t1"]}
    metadata = {
        "/a~1b/0": {"page": 2},
        "/tags/1": {"page": 5},
        "/tags/9": {"page": 1},  # out of range
        "/missing/0": {"page": 1},
        "/a~1b/0/x": {"page": 0},  # not a page
    }
    assert {(c.field_path, c.page) for c in _citations(data, metadata)} == {
        ("a/b[0].x", 2),
        ("a/b[0].y.z", 2),
        ("tags[1]", 5),
    }
    assert _citations(data, None) == []


def test_a_value_cited_on_its_own_keeps_its_page_over_its_items() -> None:
    data = {"rows": [{"name": "a", "total": 3}], "phone": None}
    metadata = {
        "/rows/0/name": {"page": 4},  # listed before its item: order does not matter
        "/rows/0": {"page": 2},
        "/phone": {"page": 1},  # a null is cited too
    }
    citations = _citations(data, metadata)
    assert sorted((c.field_path, c.page) for c in citations) == [
        ("phone", 1),
        ("rows[0].name", 4),
        ("rows[0].total", 2),
    ]
