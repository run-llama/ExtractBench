"""Operator API serialization, billing, and retry boundaries."""

import concurrent.futures
import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("extend_ai", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

import httpx
from extend_ai import Extend

from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.inference.cost_contract import (
    COMPLETE,
    COST_ACCOUNTING,
    COST_COMPONENT_EVENT,
    COST_PER_PAGE_USD,
    COST_USD,
    KNOWN_COST_USD,
)
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.extract.extend_operator import (
    ExtendOperatorProvider,
    _attach_operator_cost_stats,
)
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


def test_sdk_poll_preserves_full_operator_cost_without_double_counting_parse(monkeypatch, tmp_path: Path):
    """Catch wrong API serialization, dropped agent charges, and parsing billed twice."""
    pipeline = get_pipeline("extend_extract_operator")
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"document")
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["paid"]},
                        "qty": {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]},
                    },
                },
            },
        },
    }
    response = {
        "id": "exr_test",
        "status": "PROCESSING",
        "reviewed": False,
        "edited": False,
        "config": {"baseProcessor": "extraction_operator", "schema": schema},
        "dashboardUrl": "https://app.extend.ai/runs/exr_test",
        "createdAt": "2026-10-07T00:00:00Z",
        "updatedAt": "2026-10-07T00:00:00Z",
    }
    methods = []

    def handle(request):
        methods.append(request.method)
        assert request.headers["x-extend-api-version"] == pipeline.config["api_version"]
        if request.method == "POST":
            body = json.loads(request.content)
            assert request.url.path == "/extract_runs"
            properties = body["config"]["schema"]["properties"]
            assert properties["invoice_number"]["type"] == ["string", "null"]
            item_status = properties["items"]["items"]["properties"]["status"]
            assert item_status["type"] == ["string", "null"]
            assert item_status["enum"] == ["paid", None]
            assert properties["items"]["items"]["properties"]["qty"]["type"] == ["string", "null"]
            assert body["config"]["baseProcessor"] == response["config"]["baseProcessor"]
            assert body["file"] == {"id": "file_test"}
            return httpx.Response(200, json=response)
        assert request.url.path == "/extract_runs/exr_test"
        # A transient polling failure must poll the same run, never POST again.
        if methods.count("GET") == 1:
            return httpx.Response(503, json={"message": "temporarily unavailable"})
        return httpx.Response(
            200,
            json={
                **response,
                "status": "PROCESSED",
                "output": {"value": {"invoice_number": "INV-42"}, "metadata": {}},
                "usage": {
                    "credits": 248,
                    "totalCredits": 248,
                    "breakdown": [
                        {
                            "object": "extract_run",
                            "id": "exr_test",
                            "credits": 248,
                            "charges": [
                                {"product": "parse_performance", "unit": "page", "quantity": 40, "credits": 80},
                                {"product": "extraction_agent", "unit": "page", "quantity": 40, "credits": 80},
                                {"product": "agent_usage", "unit": "usd", "quantity": 1.10, "credits": 88},
                            ],
                        }
                    ],
                },
                "file": {
                    "id": "file_test",
                    "name": "invoice.pdf",
                    "metadata": {"pageCount": 40},
                    "createdAt": response["createdAt"],
                    "updatedAt": response["updatedAt"],
                },
            },
        )

    monkeypatch.setenv("EXTEND_API_KEY", "test")
    provider = ExtendOperatorProvider("extend_operator", {**pipeline.config, "async_poll_interval": 0.0})
    provider._client = Extend(
        token="test",
        extend_api_version=pipeline.config["api_version"],
        httpx_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(provider, "_upload_file", lambda _: "file_test")
    result = provider.normalize(
        provider.run_inference(
            pipeline,
            InferenceRequest(
                example_id="invoice",
                source_file_path=str(source),
                product_type=ProductType.EXTRACT,
                schema_override=schema,
            ),
        )
    )
    assert methods == ["POST", "GET", "GET"]
    assert result.output.extracted_data == {"invoice_number": "INV-42"}
    assert result.raw_output["total_credits"] == 248
    assert result.raw_output["parse_credits"] == 80
    assert result.raw_output["extract_credits"] == 80
    assert result.raw_output["agent_usage_credits"] == 88
    assert result.raw_output["agent_usage_usd"] == 1.10
    assert result.raw_output[COST_USD] == 3.10
    assert result.raw_output[COST_PER_PAGE_USD] == 3.10 / 40
    stats = {stat.name: stat.value for stat in build_operational_stats(result)}
    assert stats["agent_usage_usd"] == 1.10
    assert stats[COST_USD] == 3.10


def test_missing_rollup_does_not_add_parse_estimate_or_invent_agent_cost():
    raw = {"processor_run": {"usage": {"credits": 4}, "file": {"metadata": {"pageCount": 1}}}}
    _attach_operator_cost_stats(raw, {"usd_per_credit": 0.01})
    assert raw[COST_USD] == 0.04
    assert "agent_usage_usd" not in raw
    assert "parse_credits" not in raw


def test_poll_exhaustion_is_permanent_to_prevent_duplicate_billed_run(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("EXTEND_API_KEY", "test")
    provider = ExtendOperatorProvider("extend_operator", {"async_max_poll_errors": 0, "async_poll_interval": 0.0})
    monkeypatch.setattr(provider, "_upload_file", lambda _: "file_test")
    from unittest.mock import Mock

    provider._client = Mock()
    provider._client.extract_runs.create.return_value.model_dump.return_value = {
        "id": "exr_existing",
        "status": "PROCESSING",
    }
    provider._client.extract_runs.retrieve.side_effect = httpx.ReadTimeout("poll failed")
    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"document")
    with pytest.raises(ProviderPermanentError, match="exr_existing"):
        provider.run_inference(
            get_pipeline("extend_extract_operator"),
            InferenceRequest(
                example_id="invoice",
                source_file_path=str(source),
                product_type=ProductType.EXTRACT,
                schema_override={"type": "object"},
            ),
        )
    assert provider._client.extract_runs.create.call_count == 1


def test_cancel_stops_only_the_requested_operator_worker(monkeypatch, tmp_path: Path):
    """A timed-out Operator request must release its worker without cancelling another document."""
    from unittest.mock import Mock

    provider = ExtendOperatorProvider("extend_operator", {"api_key": "test", "async_poll_interval": 0.01})
    started = {name: threading.Event() for name in ("first", "second")}
    finish = threading.Event()

    def response(run_id, status):
        result = Mock()
        result.model_dump.return_value = {"id": run_id, "status": status, "output": {"value": {}}}
        return result

    def create(*, file, **kwargs):
        started[file["id"]].set()
        return response(f"exr_{file['id']}", "PROCESSING")

    provider._client = Mock()
    provider._client.extract_runs.create.side_effect = create
    provider._client.extract_runs.retrieve.side_effect = lambda run_id, **kwargs: response(
        run_id, "PROCESSED" if finish.is_set() else "PROCESSING"
    )
    monkeypatch.setattr(provider, "_upload_file", lambda path: Path(path).stem)
    pipeline = get_pipeline("extend_extract_operator")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        try:
            for name in started:
                source = tmp_path / f"{name}.pdf"
                source.write_bytes(b"document")
                request = InferenceRequest(
                    example_id=name,
                    source_file_path=str(source),
                    product_type=ProductType.EXTRACT,
                    schema_override={"type": "object"},
                )
                futures[name] = executor.submit(provider.run_inference, pipeline, request)
            for event in started.values():
                assert event.wait(timeout=5)
            assert provider.cancel("first")
            with pytest.raises(ProviderPermanentError, match="exr_first") as error:
                futures["first"].result(timeout=5)
            assert error.value.job_id == "exr_first"
            assert error.value.debug_payload["operator_run_id"] == "exr_first"
            assert error.value.debug_payload["processor_run"]["status"] == "PROCESSING"
            assert not futures["second"].done()
            assert not provider.cancel("first")
        finally:
            finish.set()
            for name in started:
                provider.cancel(name)
    assert not provider.cancel("second")


@pytest.mark.parametrize("credits", [None, 12])
def test_failed_operator_retains_returned_billing(monkeypatch: pytest.MonkeyPatch, credits: int | None) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from extract_bench.inference.events import bind_provider_event_sink

    provider = ExtendOperatorProvider("extend_operator", {"api_key": "test"})
    monkeypatch.setattr(provider, "_upload_file", lambda path: "file_test")
    provider._cancel_event.set(threading.Event())
    run = {"id": "exr_failed", "status": "FAILED", "usage": {"totalCredits": credits}}
    provider._client = Mock()
    provider._client.extract_runs.create.return_value = SimpleNamespace(model_dump=lambda **kwargs: run)
    events = []
    with bind_provider_event_sink(lambda event, payload: events.append((event, payload))):
        with pytest.raises(ProviderPermanentError) as caught:
            provider._extract_document("unused.pdf", {"type": "object"}, {"usd_per_credit": 0.02})
    assert caught.value.job_id == "exr_failed"
    assert caught.value.debug_payload["processor_run"] == run
    accounting = caught.value.debug_payload[COST_ACCOUNTING]
    assert accounting[KNOWN_COST_USD] == (None if credits is None else 0.24)
    assert accounting[COMPLETE] is (credits is not None)
    charges = [payload for event, payload in events if event == COST_COMPONENT_EVENT]
    assert len(charges) == (0 if credits is None else 1)
