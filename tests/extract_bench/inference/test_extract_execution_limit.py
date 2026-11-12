"""Extract deadlines must not be multiplied by retries or pipeline overrides."""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest

from extract_bench.inference.limits import EXTRACT_HARD_TIMEOUT_S
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.runner import InferenceRunner
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType


@pytest.mark.parametrize(
    "options",
    [
        {"per_file_timeout": EXTRACT_HARD_TIMEOUT_S + 1},
        {"per_file_timeout": float("inf")},
        {"per_file_timeout": float("nan")},
        {"per_file_timeout": 0},
        {"timeout_retries": 1},
    ],
)
def test_extract_cannot_extend_its_budget_with_overrides_or_timeout_retries(tmp_path: Path, options: dict) -> None:
    pipeline = PipelineSpec(pipeline_name="test", provider_name="test", product_type=ProductType.EXTRACT)
    with pytest.raises(ValueError):
        InferenceRunner(provider=Mock(), pipeline=pipeline, output_dir=tmp_path, use_rich=False, **options)


def test_cancelled_document_cannot_submit_another_billed_request(tmp_path: Path) -> None:
    provider = Mock()
    pipeline = PipelineSpec(pipeline_name="test", provider_name="test", product_type=ProductType.EXTRACT)
    runner = InferenceRunner(provider=provider, pipeline=pipeline, output_dir=tmp_path, use_rich=False)
    try:
        future: Future = Future()
        runner._signal_cancel_and_cancel_future("doc", future)
        request = InferenceRequest(example_id="doc", source_file_path="unused.pdf", product_type=ProductType.EXTRACT)
        with pytest.raises(ProviderPermanentError, match="deadline exceeded"):
            runner._run_provider_inference(request)
        provider.run_inference.assert_not_called()
        provider.cancel.assert_called_once_with("doc")
    finally:
        runner.shutdown()


def test_registered_extract_pipelines_do_not_shorten_the_shared_deadline() -> None:
    from extract_bench.inference.pipelines.extract import register_extract_pipelines

    pipelines: list[PipelineSpec] = []
    register_extract_pipelines(pipelines.append)
    offenders = {p.pipeline_name: p.per_file_timeout for p in pipelines if p.per_file_timeout != EXTRACT_HARD_TIMEOUT_S}
    assert offenders == {}


def test_healthy_pending_polls_do_not_back_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from extract_bench.inference.providers.extract import pulse

    provider = pulse.PulseExtractProvider("pulse_extract", {"api_key": "test", "poll_interval": 2.5})
    control = pulse._AttemptControl(deadline=time.monotonic() + 100)
    waits: list[float] = []

    def wait(control: object, seconds: float, **kwargs: object) -> None:
        waits.append(seconds)

    monkeypatch.setattr(provider, "_interruptible_wait", wait)
    states = iter([{"status": "pending"}] * 4 + [{"status": "completed", "result": {"ok": True}}])
    response = Mock(status_code=200)
    response.json.side_effect = lambda: next(states)
    monkeypatch.setattr(pulse.requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(provider, "_register_job", lambda *args: True)
    provider._poll_job("job", context="extract", example_id="doc", control=control)
    assert waits == [2.5] * 5


def test_extract_limit_does_not_change_parse_timeout_policy(tmp_path: Path) -> None:
    from extract_bench.inference.runner import DEFAULT_PER_FILE_TIMEOUT_S

    parse = PipelineSpec(pipeline_name="parse", provider_name="test", product_type=ProductType.PARSE)
    extract = PipelineSpec(pipeline_name="extract", provider_name="test", product_type=ProductType.EXTRACT)
    parse_runner = InferenceRunner(provider=Mock(), pipeline=parse, output_dir=tmp_path / "parse", use_rich=False)
    extract_runner = InferenceRunner(provider=Mock(), pipeline=extract, output_dir=tmp_path / "extract", use_rich=False)
    try:
        assert parse_runner.per_file_timeout == DEFAULT_PER_FILE_TIMEOUT_S
        assert extract_runner.per_file_timeout == EXTRACT_HARD_TIMEOUT_S
        assert extract_runner.per_file_timeout > parse_runner.per_file_timeout
    finally:
        parse_runner.shutdown()
        extract_runner.shutdown()


@pytest.mark.asyncio
async def test_cancellation_stops_async_vendor_polling_and_releases_client() -> None:
    import asyncio

    from extract_bench.inference.limits import ExecutionBudget

    budget = ExecutionBudget()
    started = asyncio.Event()
    released = asyncio.Event()

    async def poll() -> None:
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            released.set()

    task = asyncio.create_task(budget.run_async(poll()))
    await started.wait()
    budget.cancel()
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        await task
    assert released.is_set()


def test_provider_stages_and_retries_share_remaining_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from extract_bench.inference import limits

    now = 100.0
    monkeypatch.setattr(limits.time, "monotonic", lambda: now)
    pipeline = PipelineSpec(pipeline_name="test", provider_name="test", product_type=ProductType.EXTRACT)
    runner = InferenceRunner(provider=Mock(), pipeline=pipeline, output_dir=tmp_path, use_rich=False)
    request = InferenceRequest(example_id="doc", source_file_path="unused.pdf", product_type=ProductType.EXTRACT)
    try:
        with runner._execution_budget_context(request):
            first = limits.execution_budget()
        now += 17
        with runner._execution_budget_context(request):
            second = limits.execution_budget()
            assert second is first
            assert second.remaining() == runner.per_file_timeout - 17
        runner._signal_cancel_and_cancel_future("doc", Future())
        with pytest.raises(TimeoutError, match="deadline exceeded"):
            second.remaining()
    finally:
        runner.shutdown()


@pytest.mark.parametrize("key", ["timeout", "timeout_s"])
@pytest.mark.parametrize("seconds", [3600, 14400])
def test_pipeline_factory_rejects_provider_budget_that_conflicts_with_runner(key: str, seconds: int) -> None:
    from extract_bench.inference.pipelines.extract import _pipeline_spec

    with pytest.raises(ValueError, match=f"{key} must equal"):
        _pipeline_spec(pipeline_name="test", provider_name="test", config={key: seconds})
