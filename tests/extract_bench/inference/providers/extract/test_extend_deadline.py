"""Exercise cancellation through the real Extend polling loop without vendor calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("extend_ai", reason="Extend SDK is optional")

import httpx

from extract_bench.inference.events import bind_provider_event_sink
from extract_bench.inference.limits import ExecutionBudget
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.extract.extend import ExtendProvider
from extract_bench.schemas.pipeline_io import InferenceRequest


@pytest.mark.parametrize("stop_at", ["create", "get", "poll_error", "deadline", "cancel_error"])
def test_cancelled_polling_cannot_return_late_success_or_start_another_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stop_at: str,
) -> None:
    pipeline = get_pipeline("extend_extract")
    provider = ExtendProvider("extend", {**pipeline.config, "api_key": "test", "async_poll_interval": 0})
    source = tmp_path / "document.pdf"
    source.write_bytes(b"%PDF-1.4")
    request = InferenceRequest.model_validate(
        {
            "example_id": "document",
            "source_file_path": str(source),
            "product_type": "extract",
            "schema_override": {"type": "object"},
        },
    )
    monkeypatch.setattr(provider, "_upload_file", lambda path: "file_test")
    monkeypatch.setattr(provider, "_get_or_create_processor", lambda config: "dp_test")
    client = Mock()
    provider._client = client
    budget = ExecutionBudget(30)

    def response(status: str) -> SimpleNamespace:
        return SimpleNamespace(
            processor_run=SimpleNamespace(id="dpr_test", status=status),
            model_dump=lambda: {"processor_run": {"id": "dpr_test", "status": status}},
        )

    def create(**kwargs: object) -> SimpleNamespace:
        if stop_at == "create":
            provider.cancel("document")
        return response("PROCESSING")

    def get(**kwargs: object) -> SimpleNamespace:
        if stop_at == "deadline":
            budget.deadline = 0
        else:
            provider.cancel("document")
        if stop_at == "poll_error" and client.processor_run.get.call_count == 1:
            raise httpx.ReadTimeout("poll interrupted")
        return response("PROCESSED")

    client.processor_run.create.side_effect = create
    client.processor_run.get.side_effect = get
    client.processor_run.cancel.return_value = SimpleNamespace(model_dump=lambda: {"success": True})
    if stop_at == "cancel_error":
        client.processor_run.cancel.side_effect = httpx.ReadTimeout("cancel unavailable")

    events = []
    with (
        bind_provider_event_sink(lambda event, payload: events.append((event, payload))),
        budget.activate(),
        pytest.raises(ProviderPermanentError, match="deadline") as caught,
    ):
        provider.run_inference(pipeline, request)
    assert events[0] == ("vendor.job_created", {"provider": "extend", "stage": "processor_run", "job_id": "dpr_test"})
    assert any(event == "vendor.cancellation_requested" for event, _ in events)
    assert events[-1][0] == (
        "vendor.cancellation_failed" if stop_at == "cancel_error" else "vendor.cancellation_completed"
    )
    assert caught.value.job_id == "dpr_test"
    assert caught.value.debug_payload is not None
    execution = caught.value.debug_payload["_extend_execution"]
    assert execution["create_response"]["processor_run"]["id"] == "dpr_test"
    assert "cancel_error" in execution if stop_at == "cancel_error" else execution["cancel_response"]["success"]
    assert client.processor_run.create.call_count == 1
    assert client.processor_run.get.call_count == (0 if stop_at == "create" else 1)
    assert client.processor_run.cancel.call_count == 1
    assert client.processor_run.cancel.call_args.kwargs["id"] == "dpr_test"
    options = client.processor_run.create.call_args.kwargs["request_options"]
    assert options["max_retries"] == 0
    assert 0 < options["timeout_in_seconds"] <= 30
    assert provider.cancel("document") is False
