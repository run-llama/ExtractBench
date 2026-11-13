"""Reducto must stop between paid stages when a document is cancelled."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("reducto", reason="dev and runners extras required")
pytest.importorskip("pypdf", reason="dev and runners extras required")

from extract_bench.inference.limits import ExecutionBudget
from extract_bench.inference.providers.extract import reducto


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("cancel_upload", [False, True])
def test_upload_consumes_budget_and_cancellation_prevents_extract(
    monkeypatch: pytest.MonkeyPatch,
    deep: bool,
    cancel_upload: bool,
) -> None:
    budget = ExecutionBudget(100)
    client = Mock()

    def upload(**kwargs: object) -> str:
        budget.deadline -= 30
        if cancel_upload:
            budget.cancel()
        return "uploaded"

    client.upload.side_effect = upload
    result = SimpleNamespace(model_dump=lambda: {"data": {}})
    client.extract.run.return_value = result
    client.extract.run_job.return_value = SimpleNamespace(job_id="job")
    client.job.get.return_value = SimpleNamespace(
        status="Completed",
        result=result,
        model_dump=lambda: {"status": "Completed", "result": result.model_dump()},
    )
    monkeypatch.setattr(reducto, "Reducto", lambda **kwargs: client)
    provider = reducto.ReductoExtractProvider("reducto_extract", {"api_key": "test"})
    with budget.activate():
        if cancel_upload:
            with pytest.raises(TimeoutError, match="deadline"):
                asyncio.run(provider._extract_pdf_async("unused.pdf", {}, deep_extract=deep))
            client.extract.run.assert_not_called()
            client.extract.run_job.assert_not_called()
        else:
            asyncio.run(provider._extract_pdf_async("unused.pdf", {}, deep_extract=deep))
            call = client.extract.run_job if deep else client.extract.run
            assert 0 < call.call_args.kwargs["timeout"] < 70
            assert client.upload.call_args.kwargs["timeout"] > 90


def test_cancelled_deep_poll_stops_polling_and_cancels_vendor_job(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = ExecutionBudget(100)
    client = Mock()
    client.extract.run_job.return_value = SimpleNamespace(job_id="job")

    def poll(*args: object, **kwargs: object) -> SimpleNamespace:
        budget.cancel()
        if client.job.get.call_count > 1:
            return SimpleNamespace(status="Completed", result=SimpleNamespace(data={}))
        return SimpleNamespace(status="Processing")

    client.job.get.side_effect = poll
    monkeypatch.setattr(reducto, "Reducto", lambda **kwargs: client)
    provider = reducto.ReductoExtractProvider("reducto_extract", {"api_key": "test"})
    with budget.activate(), pytest.raises(TimeoutError, match="deadline"):
        asyncio.run(provider._extract_pdf_async("unused.pdf", {}, deep_extract=True))
    assert client.job.get.call_count == 1
    client.job.cancel.assert_called_once_with("job", timeout=10)


def test_cancelled_awaiter_still_records_late_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK worker can accept a paid job after its async caller was cancelled."""
    import threading

    from extract_bench.inference.events import bind_provider_event_sink

    started = threading.Event()
    release = threading.Event()
    recorded = threading.Event()
    events = []
    client = Mock()

    def submit(**kwargs: object) -> SimpleNamespace:
        started.set()
        assert release.wait(5)
        return SimpleNamespace(job_id="late-paid-job")

    def record(event: str, payload: object) -> None:
        events.append((event, payload))
        recorded.set()

    client.extract.run_job.side_effect = submit
    monkeypatch.setattr(reducto, "Reducto", lambda **kwargs: client)
    provider = reducto.ReductoExtractProvider("reducto_extract", {"api_key": "test"})

    async def run() -> None:
        with ExecutionBudget(100).activate(), bind_provider_event_sink(record):
            task = asyncio.create_task(provider._extract_pdf_async("unused.pdf", {}, deep_extract=True))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
            assert await asyncio.to_thread(recorded.wait, 5)

    asyncio.run(run())
    assert events[0][0] == "vendor.job_created"
    assert events[0][1]["job_id"] == "late-paid-job"
    client.job.get.assert_not_called()
