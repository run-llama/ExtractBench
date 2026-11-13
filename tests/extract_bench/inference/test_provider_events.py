"""Provider progress belongs to its attempt even when a paid stage fails."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from extract_bench.inference.events import bind_provider_event_sink, emit_provider_event
from extract_bench.inference.limits import ExecutionBudget


def test_event_context_survives_worker_handoff_and_is_reset_after_failure() -> None:
    received = []

    def fail() -> None:
        emit_provider_event("vendor.job_created", job_id="accepted")
        raise RuntimeError("poll failed")

    async def run() -> None:
        with bind_provider_event_sink(lambda event, data: received.append((event, data))):
            await asyncio.to_thread(fail)

    with pytest.raises(RuntimeError, match="poll failed"):
        asyncio.run(run())
    emit_provider_event("unbound")
    assert received == [("vendor.job_created", {"job_id": "accepted"})]


def test_datalab_submission_is_recorded_before_sdk_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("datalab_sdk")
    from datalab_sdk import AsyncDatalabClient

    from extract_bench.inference.providers.extract.datalab_client import TracingDatalabClient

    submit = AsyncMock(return_value={"request_id": "paid", "request_check_url": "https://example.test/status"})
    monkeypatch.setattr(AsyncDatalabClient, "_submit_with_retry", submit)
    received = []
    client = TracingDatalabClient.__new__(TracingDatalabClient)
    with bind_provider_event_sink(lambda event, data: received.append((event, data))):
        asyncio.run(client._submit_with_retry("extract", data={"schema": "{}"}))
    assert received[0][0] == "vendor.job_created"
    assert received[0][1]["job_id"] == "paid"
    assert received[0][1]["request_check_url"] == submit.return_value["request_check_url"]


@pytest.mark.parametrize("cancel_after_parse", [False, True])
def test_landingai_records_parse_before_cancellation_and_rechecks_budget(cancel_after_parse: bool) -> None:
    pytest.importorskip("landingai_ade")
    from extract_bench.inference.providers.extract.landingai import LandingAIExtractProvider

    budget = ExecutionBudget(100)
    client = Mock()

    def parse(**kwargs: object) -> SimpleNamespace:
        budget.deadline -= 30
        if cancel_after_parse:
            budget.cancel()
        return SimpleNamespace(markdown="text", model_dump=lambda: {"markdown": "text", "credit_usage": 2})

    client.parse.side_effect = parse
    client.extract.return_value = SimpleNamespace(extraction={}, model_dump=lambda: {"extraction": {}})
    provider = LandingAIExtractProvider.__new__(LandingAIExtractProvider)
    provider._base_config = {}
    provider._model = None
    provider._client = client
    received = []
    with budget.activate(), bind_provider_event_sink(lambda event, data: received.append((event, data))):
        if cancel_after_parse:
            with pytest.raises(TimeoutError, match="deadline"):
                provider._extract_from_document(Path("doc.pdf"), {})
            client.extract.assert_not_called()
        else:
            provider._extract_from_document(Path("doc.pdf"), {})
            assert 0 < client.extract.call_args.kwargs["timeout"] < 70
        assert 70 < client.parse.call_args.kwargs["timeout"] <= 100
    assert received[0][0] == "stage.completed"
    assert received[0][1]["raw_output"]["credit_usage"] == 2
