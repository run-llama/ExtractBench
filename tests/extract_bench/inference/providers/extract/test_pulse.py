from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.pulse import (
    PulseExtractProvider,
    _apply_usage_cost_fields,
    _build_pulse_anchor_index,
    _extract_pulse_field_citations,
    _resolve_pulse_citation_anchors,
)
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType

_PROMPT = "Extract the document into the provided JSON schema. Use only information present in the document."


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


def _provider(config: dict[str, Any] | None = None) -> PulseExtractProvider:
    return PulseExtractProvider(
        "pulse_extract",
        {
            "api_key": "test-key",
            "api_base_url": "https://pulse.test",
            "request_timeout": 1,
            "job_timeout": 10,
            "poll_interval": 0.001,
            "poll_max_interval": 0.001,
            **(config or {}),
        },
    )


def _request(source: Path, schema: dict[str, Any]) -> InferenceRequest:
    return InferenceRequest(
        example_id="invoice/doc-1",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override=schema,
    )


def test_registered_pipelines_match_submitted_modes() -> None:
    non_effort = get_pipeline("pulse_schema_non_effort")
    effort = get_pipeline("pulse_schema_effort")

    for pipeline in (non_effort, effort):
        assert pipeline.provider_name == "pulse_extract"
        assert pipeline.product_type == ProductType.EXTRACT
        assert pipeline.config["model"] == "pulse-ultra-2"
        assert pipeline.config["extensions"] == {"altOutputs": {"wlbb": True}}
        assert pipeline.config["schema_prompt"] == _PROMPT
        assert pipeline.config["async_run"] is True
        assert set(pipeline.config) == {"model", "extensions", "schema_prompt", "async_run", "effort"}
        assert pipeline.per_file_timeout is None

    assert non_effort.config["effort"] is False
    assert effort.config["effort"] is True


def test_async_pipeline_pins_ultra_2_and_polls_accepted_jobs_without_resubmitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    schema = {
        "$defs": {"amount": {"type": "number"}},
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Keep / punctuation"},
        },
    }
    pipeline = get_pipeline("pulse_schema_non_effort")
    provider = _provider(
        {
            **pipeline.config,
            "poll_interval": 0.001,
            "poll_max_interval": 0.001,
        }
    )
    post_calls: list[dict[str, Any]] = []
    get_urls: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        if url.endswith("/extract"):
            files = kwargs["files"]
            fields = {name: value[1] for name, value in files if name != "file"}
            post_calls.append({"url": url, "fields": fields})
            return _FakeResponse({"job_id": "extract-job", "status": "pending"}, status_code=202)

        post_calls.append({"url": url, "json": kwargs["json"]})
        return _FakeResponse({"job_id": "schema-job", "status": "pending"}, status_code=202)

    poll_responses: list[_FakeResponse | requests.RequestException] = [
        _FakeResponse(
            {
                "status": "completed",
                "result": {
                    "extraction_id": "extract-1",
                    "page_count": 2,
                    "bounding_boxes": {
                        "Header": [
                            {
                                "id": "txt-header-1",
                                "page": 1,
                                "bounding_box": [0.1, 0.1, 0.4, 0.1, 0.4, 0.2, 0.1, 0.2],
                                "text": "Invoice",
                            }
                        ]
                    },
                },
            }
        ),
        requests.ConnectionError("temporary poll disconnect"),
        _FakeResponse({"error": "not ready"}, status_code=503),
        _FakeResponse({"status": "processing"}),
        _FakeResponse(
            {
                "status": "completed",
                "result": {
                    "schema_output": {
                        "values": {"title": "Invoice"},
                        "citations": {"title": "txt-header-1"},
                    }
                },
            }
        ),
    ]

    def fake_get(url: str, **_: Any) -> _FakeResponse:
        get_urls.append(url)
        response = poll_responses.pop(0)
        if isinstance(response, requests.RequestException):
            raise response
        return response

    monkeypatch.setattr("extract_bench.inference.providers.extract.pulse.requests.post", fake_post)
    monkeypatch.setattr("extract_bench.inference.providers.extract.pulse.requests.get", fake_get)
    raw = provider.run_inference(pipeline, _request(source, schema))
    normalized = provider.normalize(raw)

    assert len(post_calls) == 2
    assert post_calls[0]["url"] == "https://pulse.test/extract"
    assert post_calls[0]["fields"] == {
        "model": "pulse-ultra-2",
        "async": "true",
        "extensions": json.dumps({"altOutputs": {"wlbb": True}}),
    }
    assert post_calls[1]["url"] == "https://pulse.test/schema"
    assert post_calls[1]["json"] == {
        "extraction_id": "extract-1",
        "schema_config": {
            "input_schema": schema,
            "effort": False,
            "schema_prompt": _PROMPT,
        },
        "async": True,
    }
    assert get_urls == [
        "https://pulse.test/job/extract-job",
        "https://pulse.test/job/schema-job",
        "https://pulse.test/job/schema-job",
        "https://pulse.test/job/schema-job",
        "https://pulse.test/job/schema-job",
    ]
    assert raw.raw_output["job_id"] == "schema-job"
    assert raw.raw_output["num_pages"] == 2
    assert raw.raw_output["extract_credits_used"] == 2
    assert raw.raw_output["extract_cost_usd"] == pytest.approx(0.03)
    assert raw.raw_output["schema_credits_used"] == 2
    assert raw.raw_output["schema_cost_usd"] == pytest.approx(0.03)
    assert raw.raw_output["credits_used"] == 4
    assert raw.raw_output["cost_per_page_usd"] == pytest.approx(0.03)
    assert normalized.output.extracted_data == {"title": "Invoice"}
    assert len(normalized.output.field_citations) == 1
    assert normalized.output.field_citations[0].field_path == "title"
    assert normalized.output.field_citations[0].page == 1


def test_submission_transport_failure_is_not_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    provider = _provider({"async_run": True})
    calls = 0

    def fail_post(*_: Any, **__: Any) -> _FakeResponse:
        nonlocal calls
        calls += 1
        raise requests.Timeout("response lost")

    monkeypatch.setattr("extract_bench.inference.providers.extract.pulse.requests.post", fail_post)
    control = provider._register_request("invoice/doc-1")

    with pytest.raises(ProviderPermanentError, match="submission outcome is unknown") as exc_info:
        provider._extract_file(
            source,
            example_id="invoice/doc-1",
            control=control,
        )

    assert calls == 1
    assert exc_info.value.debug_payload == {
        "context": "extract submission",
        "exception_type": "Timeout",
        "exception": "response lost",
        "submission_outcome": "unknown",
    }


@pytest.mark.parametrize(
    "terminal_error",
    [
        "Temporary schema failure. Retry later.",
        "Temporary schema failure. Retry the request.",
        "Temporary schema failure. Please resubmit the request.",
    ],
)
def test_known_terminal_schema_failure_is_transient_for_the_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_error: str,
) -> None:
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    pipeline = get_pipeline("pulse_schema_effort")
    provider = _provider(pipeline.config)
    schema_calls = 0

    def fake_extract(*_: Any, **__: Any) -> dict[str, Any]:
        return {"extraction_id": "extract-1", "page_count": 1}

    def fake_schema(**_: Any) -> dict[str, Any]:
        nonlocal schema_calls
        schema_calls += 1
        raise ProviderPermanentError(
            "Pulse schema job schema-job-1 ended with status=failed",
            job_id="schema-job-1",
            debug_payload={
                "context": "schema",
                "job_id": "schema-job-1",
                "state": {"status": "failed", "error": terminal_error},
            },
        )

    monkeypatch.setattr(provider, "_extract_file", fake_extract)
    monkeypatch.setattr(provider, "_apply_schema", fake_schema)

    with pytest.raises(ProviderTransientError) as excinfo:
        provider.run_inference(
            pipeline,
            _request(source, {"type": "object", "properties": {"title": {"type": "string"}}}),
        )

    assert schema_calls == 1
    assert excinfo.value.job_id == "schema-job-1"


def test_rate_limited_submission_raises_without_waiting() -> None:
    provider = _provider({"async_run": True})
    control = provider._register_request("invoice/doc-1")
    calls = 0

    def submit(_: float) -> _FakeResponse:
        nonlocal calls
        calls += 1
        return _FakeResponse({"error": "capacity"}, status_code=429, headers={"Retry-After": "60"})

    with pytest.raises(ProviderRateLimitError, match="rate-limited"):
        provider._submit(submit, context="extract submission", control=control)

    assert calls == 1


def test_large_result_only_forwards_api_key_to_pulse_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    seen: list[tuple[str, dict[str, str]]] = []
    redirect_flags: list[bool] = []
    control = provider._register_request("invoice/doc-1")

    def fake_get(url: str, *, headers: dict[str, str], **kwargs: Any) -> _FakeResponse:
        seen.append((url, headers))
        redirect_flags.append(kwargs["allow_redirects"])
        if url == "https://pulse.test/results/redirected-result":
            return _FakeResponse(
                {},
                status_code=302,
                headers={"Location": "https://storage.example/redirected-result.json"},
            )
        return _FakeResponse({"schema_output": {"values": {}}})

    monkeypatch.setattr("extract_bench.inference.providers.extract.pulse.requests.get", fake_get)

    provider._fetch_large_result(
        {"is_url": True, "url": "https://storage.example/result.json"},
        context="schema",
        control=control,
    )
    provider._fetch_large_result(
        {"is_url": True, "url": "/results/result-id"},
        context="schema",
        control=control,
    )
    provider._fetch_large_result(
        {"is_url": True, "url": "/results/redirected-result"},
        context="schema",
        control=control,
    )

    assert seen == [
        ("https://storage.example/result.json", {}),
        ("https://pulse.test/results/result-id", {"x-api-key": "test-key"}),
        ("https://pulse.test/results/redirected-result", {"x-api-key": "test-key"}),
        ("https://storage.example/redirected-result.json", {}),
    ]
    assert redirect_flags == [False, False, False, False]


def test_cost_uses_document_page_count_and_official_credit_rates() -> None:
    actual = {
        "extract": {
            "page_count": 2,
            "plan_info": {"pages_used": 999},
            "credits_used": 4,
        },
        "schema": {"credits_used": 10},
        "_config": {"effort": True},
    }
    _apply_usage_cost_fields(actual)

    assert actual["num_pages"] == 2
    assert actual["extract_credits_used"] == 4
    assert actual["schema_credits_used"] == 10
    assert actual["cost_usd"] == pytest.approx(0.21)
    assert actual["cost_per_page_usd"] == pytest.approx(0.105)
    assert actual["extract_credits_estimated"] is False
    assert actual["schema_credits_estimated"] is False

    estimated = {
        "extract": {"page_count": 2},
        "schema": {},
        "_config": {"effort": True},
    }
    _apply_usage_cost_fields(estimated)

    assert estimated["extract_credits_used"] == 2
    assert estimated["schema_credits_used"] == 12
    assert estimated["cost_per_page_usd"] == pytest.approx(0.105)
    assert estimated["extract_credits_estimated"] is True
    assert estimated["schema_credits_estimated"] is True


def test_anchor_index_covers_all_block_categories_and_never_invents_page() -> None:
    boxes = {
        "Title": [
            {"id": "txt-title", "page": 1, "bounding_box": [0.1, 0.1, 0.4, 0.2]},
        ],
        "Footer": [
            {"id": "txt-footer", "page": 2, "bbox_normalized": [0.1, 0.8, 0.4, 0.9]},
        ],
        "Tables": [
            {
                "table_info": {
                    "id": "tbl-1",
                    "location": {"page": 2, "coordinates": [0.1, 0.1, 0.4, 0.4]},
                },
                "cell_data": [
                    {
                        "location": {"page": 2, "coordinates": [0.2, 0.2, 0.3, 0.3]},
                        "position": {"row": 0, "column": 0},
                    }
                ],
            }
        ],
        "FutureCategory": [
            {"id": "txt-future", "page": 3, "coordinates": [0.2, 0.2, 0.3, 0.3]},
        ],
        "Header": [
            {"id": "txt-no-page", "bounding_box": [0.1, 0.1, 0.2, 0.2]},
        ],
        "Words": [{"text": "Invoice", "page": 1, "bounding_box": [0.1, 0.1, 0.2, 0.2]}],
    }

    index = _build_pulse_anchor_index(boxes)
    resolved = _resolve_pulse_citation_anchors(
        {
            "title": "txt-title",
            "footer": "txt-footer",
            "amount": "tbl-1-r0c0",
            "future": "txt-future",
            "missing_page": "txt-no-page",
        },
        index,
    )
    citations = _extract_pulse_field_citations(resolved)

    assert set(index) == {
        "txt-title",
        "txt-footer",
        "tbl-1",
        "tbl-1-r0c0",
        "txt-future",
        "txt-no-page",
    }
    assert {(citation.field_path, citation.page) for citation in citations} == {
        ("title", 1),
        ("footer", 2),
        ("amount", 2),
        ("future", 3),
    }

    composite = _extract_pulse_field_citations(
        _resolve_pulse_citation_anchors(
            {"combined": "txt-title, tbl-1-r0c0, missing-anchor"},
            index,
        )
    )
    assert [(citation.field_path, citation.page) for citation in composite] == [
        ("combined", 1),
        ("combined", 2),
    ]


def test_cancel_targets_current_remote_job(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider({"async_run": True})
    control = provider._register_request("invoice/doc-1")
    assert provider._register_job("invoice/doc-1", control, "job-1") is True
    calls: list[str] = []

    def fake_delete(url: str, **_: Any) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse({"status": "canceled"})

    monkeypatch.setattr("extract_bench.inference.providers.extract.pulse.requests.delete", fake_delete)

    assert provider.cancel("invoice/doc-1") is True
    assert control.cancelled.is_set()
    assert calls == ["https://pulse.test/job/job-1"]
    assert provider.cancel("other") is False


def test_replacement_attempt_cannot_be_erased_by_old_attempt_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider({"async_run": True})
    canceled_jobs: list[str] = []
    monkeypatch.setattr(provider, "_cancel_remote_job", canceled_jobs.append)

    old_control = provider._register_request("invoice/doc-1")
    assert provider._register_job("invoice/doc-1", old_control, "old-job") is True

    new_control = provider._register_request("invoice/doc-1")
    assert old_control.cancelled.is_set()
    assert canceled_jobs == ["old-job"]
    assert provider._register_job("invoice/doc-1", new_control, "new-job") is True

    provider._clear_job("invoice/doc-1", old_control, "old-job")
    provider._clear_request("invoice/doc-1", old_control)
    assert provider._attempts["invoice/doc-1"] is new_control
    assert provider._inflight_jobs["invoice/doc-1"] == (new_control, "new-job")


def test_cancelled_attempt_never_submits() -> None:
    provider = _provider({"async_run": True})
    control = provider._register_request("invoice/doc-1")
    control.cancelled.set()
    calls = 0

    def submit(_: float) -> _FakeResponse:
        nonlocal calls
        calls += 1
        return _FakeResponse({"job_id": "should-not-exist"})

    with pytest.raises(ProviderPermanentError, match="cancelled by the benchmark runner"):
        provider._submit(
            submit,
            context="extract submission",
            control=control,
        )

    assert calls == 0


def test_job_accepted_after_runner_cancel_is_immediately_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider({"async_run": True})
    control = provider._register_request("invoice/doc-1")
    control.cancelled.set()
    canceled_jobs: list[str] = []
    monkeypatch.setattr(provider, "_cancel_remote_job", canceled_jobs.append)

    with pytest.raises(ProviderPermanentError, match="cancelled by the benchmark runner"):
        provider._resolve_submission(
            {"job_id": "late-job", "status": "pending"},
            context="extract",
            example_id="invoice/doc-1",
            control=control,
        )

    assert canceled_jobs == ["late-job"]


def test_async_terminal_failure_includes_job_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider({"async_run": True, "poll_interval": 0.001, "poll_max_interval": 0.001})
    monkeypatch.setattr(
        "extract_bench.inference.providers.extract.pulse.requests.get",
        lambda *_args, **_kwargs: _FakeResponse(
            {"status": "failed", "error": "dependency unavailable", "updated_at": "2026-09-10T00:00:00Z"}
        ),
    )

    control = provider._register_request("invoice/doc-1")
    with pytest.raises(ProviderPermanentError, match="dependency unavailable") as exc_info:
        provider._poll_job(
            "job-1",
            context="schema",
            example_id="invoice/doc-1",
            control=control,
        )

    assert exc_info.value.job_id == "job-1"
    assert exc_info.value.debug_payload["state"]["status"] == "failed"
    assert exc_info.value.debug_payload["poll_history"][0]["status"] == "failed"
