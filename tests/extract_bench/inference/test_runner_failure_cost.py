"""Paid attempts remain visible even when extraction or a later retry fails."""

import concurrent.futures
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.evaluation.stats import build_cost_stats, build_operational_stats
from extract_bench.inference import runner as runner_module
from extract_bench.inference.events import emit_provider_event
from extract_bench.inference.providers.base import Provider, ProviderPermanentError, ProviderTransientError
from extract_bench.inference.runner import InferenceRunner, RunSummary
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.schema import ExtractTestCase


class _Provider(Provider):
    provider_name = "cost_test"

    def __init__(self, responses: list[Callable[[], dict[str, Any]]]) -> None:
        self.responses = iter(responses)

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        raw = next(self.responses)()
        now = datetime.now(UTC)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=pipeline.product_type,
            raw_output=raw,
            started_at=now,
            completed_at=now,
            latency_in_ms=1,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        return InferenceResult.model_validate(
            {
                **raw_result.model_dump(),
                "output": ExtractOutput(
                    example_id=raw_result.request.example_id, pipeline_name=raw_result.pipeline_name, extracted_data={}
                ),
            }
        )


def _runner(tmp_path: Path, responses: list[Callable[[], dict[str, Any]]]) -> InferenceRunner:
    return InferenceRunner(
        provider=_Provider(responses),
        pipeline=PipelineSpec(pipeline_name="cost_test", provider_name="cost_test", product_type=ProductType.EXTRACT),
        output_dir=tmp_path / "cost_test",
        use_rich=False,
    )


@pytest.mark.parametrize("failed_bill,complete", [(2.0, True), (2.0, False), (None, False)])
def test_retry_cost_includes_failed_attempt_without_assuming_unknown_bill_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_bill: float | None,
    complete: bool,
) -> None:
    monkeypatch.setattr(runner_module, "INITIAL_BACKOFF_S", 0)

    def fail() -> dict[str, Any]:
        if failed_bill is not None:
            emit_provider_event("cost.component", component_id="first", cost_usd=failed_bill)
        raise ProviderTransientError(
            "retry", debug_payload={"cost_accounting": {"known_cost_usd": failed_bill, "complete": complete}}
        )

    runner = _runner(tmp_path, [fail, lambda: {"cost_usd": 3.0}])
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"pdf")
    try:
        raw, normalized, error = runner._process_document(source, "doc", ProductType.EXTRACT)
        assert error is None
        assert raw is not None and normalized is not None
        known = 3.0 + (failed_bill or 0)
        stats = {stat.name: stat.value for stat in build_cost_stats(normalized.raw_output)}
        assert stats["known_cost_usd"] == known
        assert stats.get("cost_usd") == (known if complete else None)
        accounting = raw.raw_output["_attempt_accounting"]
        assert len(accounting["attempts"]) == 2
        assert accounting["successful_attempt_cost_usd"] == 3
        assert json.loads((runner.output_dir / "doc.cost.json").read_text(encoding="utf-8")) == accounting
    finally:
        runner.shutdown()


@pytest.mark.parametrize(
    "parse,extraction,snapshot,expected_total", [(2.0, 3.0, 5.0, 5.0), (2.0, 3.0, 4.0, None), (0.1, 0.2, 0.3, 0.3)]
)
def test_component_snapshots_and_final_bill_are_not_added_twice(
    tmp_path: Path,
    parse: float,
    extraction: float,
    snapshot: float,
    expected_total: float | None,
) -> None:
    def paid() -> dict[str, Any]:
        emit_provider_event("cost.component", component_id="parse", cost_usd=parse / 2)
        emit_provider_event("cost.component", component_id="parse", cost_usd=parse, usage={"pages": 4})
        emit_provider_event("cost.component", component_id="extract", cost_usd=extraction, usage={"tokens": 500})
        emit_provider_event("cost.attempt", known_cost_usd=snapshot, complete=True)
        return {"cost_usd": snapshot}

    runner = _runner(tmp_path, [paid])
    request = InferenceRequest(
        example_id="doc", source_file_path=str(tmp_path / "doc.pdf"), product_type=ProductType.EXTRACT
    )
    Path(request.source_file_path).write_bytes(b"pdf")
    try:
        raw = runner._run_provider_inference(request)
        stats = {stat.name: stat.value for stat in build_cost_stats(raw.raw_output)}
        assert stats["known_cost_usd"] == pytest.approx(parse + extraction)
        assert stats.get("cost_usd") == (pytest.approx(expected_total) if expected_total is not None else None)
        persisted = json.loads((runner.output_dir / "doc.cost.json").read_text(encoding="utf-8"))
        assert persisted["attempts"][0]["components"] == {
            "parse": {"cost_usd": parse, "usage": {"pages": 4}},
            "extract": {"cost_usd": extraction, "usage": {"tokens": 500}},
        }
    finally:
        runner.shutdown()


@pytest.mark.parametrize("bill", [None, 0.0, 2.0])
def test_failure_report_retains_known_cost_and_distinguishes_unbilled_from_free(
    tmp_path: Path, bill: float | None
) -> None:
    def fail() -> dict[str, Any]:
        raise ProviderPermanentError(
            "failed", debug_payload={"cost_accounting": {"known_cost_usd": bill, "complete": bill is not None}}
        )

    runner = _runner(tmp_path, [fail])
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"pdf")
    try:
        runner._process_document(source, "doc", ProductType.EXTRACT)
        summary = RunSummary(errors=[{"example_id": "doc", "error": "failed"}])
        runner._attach_error_costs(summary)
        (runner.output_dir / "_errors.json").write_text(json.dumps(summary.errors), encoding="utf-8")
        evaluator = EvaluationRunner.__new__(EvaluationRunner)
        entries = evaluator._load_inference_errors(runner.output_dir, pipeline_name=None)
        case = ExtractTestCase(
            test_id="doc", group="g", file_path=source, schema={"type": "object"}, expected_output={}
        )
        results = evaluator._synthesize_failure_results(
            failure_entries=entries,
            evaluated_result_keys=set(),
            test_cases_dict={"doc": case},
            successful_results=[],
            pipeline_name="cost_test",
            product_type="extract",
        )
        stats = {stat.name: stat.value for stat in results[0].stats}
        assert stats.get("known_cost_usd") == bill
        assert stats.get("cost_usd") == bill
        assert stats["cost_complete"] == float(bill is not None)
    finally:
        runner.shutdown()


def test_cancelled_worker_preserves_stage_charge_before_it_returns(tmp_path: Path) -> None:
    runner = _runner(tmp_path, [])

    def cancel_during_call() -> dict[str, Any]:
        emit_provider_event("cost.component", component_id="parse", cost_usd=2)
        runner._signal_cancel_and_cancel_future("doc", concurrent.futures.Future())
        saved = json.loads((runner.output_dir / "doc.cost.json").read_text(encoding="utf-8"))
        assert saved["known_cost_usd"] == 2
        assert saved["total_cost_usd"] is None
        raise ProviderPermanentError("cancelled")

    runner.provider.responses = iter([cancel_during_call])
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"pdf")
    try:
        _, _, error = runner._process_document(source, "doc", ProductType.EXTRACT)
        assert error is not None
        assert runner._cost_snapshot("doc")["attempts"][0]["outcome"] == "cancelled"
    finally:
        runner.shutdown()


def test_report_does_not_publish_billed_subset_as_complete_run_cost(tmp_path: Path) -> None:
    from extract_bench.schemas.evaluation import EvaluationResult

    evaluator = EvaluationRunner.__new__(EvaluationRunner)
    results = [
        EvaluationResult(
            test_id=str(index),
            example_id=str(index),
            pipeline_name="cost_test",
            product_type="extract",
            success=True,
            stats=build_cost_stats(raw),
        )
        for index, raw in enumerate([{"cost_usd": 3}, {"cost_usd": 2, "_cost_complete": False}])
    ]
    stats = evaluator._aggregate_stats(results)
    assert "cost_usd" not in stats
    assert stats["known_cost_usd"]["total"] == 5
    assert stats["cost_complete"]["avg"] == 0.5


@pytest.mark.parametrize("crashed", [False, True])
def test_restarted_runner_keeps_paid_attempts_and_does_not_complete_interrupted_work(
    tmp_path: Path,
    crashed: bool,
) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"pdf")
    checkpoint: str | None = None
    cost_path = tmp_path / "cost_test/doc.cost.json"

    def first_attempt() -> dict[str, Any]:
        nonlocal checkpoint
        emit_provider_event("cost.component", component_id="extract", cost_usd=2, usage={"tokens": 20})
        emit_provider_event("cost.attempt", known_cost_usd=2, complete=True)
        checkpoint = cost_path.read_text(encoding="utf-8")
        saved = json.loads(checkpoint)
        assert saved["cost_complete"] is False
        assert saved["attempts"][0]["cost_complete"] is False
        raise ProviderPermanentError("failed")

    runner = _runner(tmp_path, [first_attempt])
    try:
        runner._process_document(source, "doc", ProductType.EXTRACT)
    finally:
        runner.shutdown()
    if crashed:
        assert checkpoint is not None
        # Replay the last artifact available when a process dies before returning.
        cost_path.write_text(checkpoint, encoding="utf-8")
    restarted = _runner(tmp_path, [lambda: {"cost_usd": 3}])
    try:
        raw, _, error = restarted._process_document(source, "doc", ProductType.EXTRACT)
        assert error is None and raw is not None
        assert raw.raw_output["cost_usd"] == 5
        assert raw.raw_output["_cost_complete"] is (not crashed)
        previous = raw.raw_output["_attempt_accounting"]["attempts"][0]
        assert previous["components"]["extract"]["usage"] == {"tokens": 20}
        assert previous["outcome"] == ("interrupted" if crashed else "error")
        assert json.loads(cost_path.read_text(encoding="utf-8"))["known_cost_usd"] == 5
    finally:
        restarted.shutdown()


def test_cancellation_hook_costs_are_captured_from_controller_thread(tmp_path: Path) -> None:
    import threading

    started = threading.Event()
    release = threading.Event()

    def wait_for_cancel() -> dict[str, Any]:
        started.set()
        assert release.wait(timeout=10)
        raise ProviderPermanentError("cancelled")

    runner = _runner(tmp_path, [wait_for_cancel])

    def cancel(example_id: str) -> None:
        emit_provider_event("cost.component", component_id="cancelled_job", cost_usd=7)
        emit_provider_event("cost.attempt", known_cost_usd=7, complete=True)

    runner.provider.cancel = cancel
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"pdf")
    try:
        future = runner._submit_attempt(runner._process_document, source, "doc", ProductType.EXTRACT)
        assert started.wait(timeout=10)
        runner._signal_cancel_and_cancel_future("doc", future)
        saved = json.loads((runner.output_dir / "doc.cost.json").read_text(encoding="utf-8"))
        assert saved["known_cost_usd"] == 7
        assert saved["cost_complete"] is False
        release.set()
        future.result(timeout=10)
        assert runner._cost_snapshot("doc")["total_cost_usd"] == 7
    finally:
        release.set()
        runner.shutdown()


@pytest.mark.parametrize("content", ["{", "{}", '{"known_cost_usd": 2, "cost_complete": "yes"}'])
def test_malformed_cost_sidecar_preserves_prediction_and_failure_rows(tmp_path: Path, content: str) -> None:
    runner = _runner(tmp_path, [lambda: {"cost_usd": 2}])
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"pdf")
    try:
        runner._process_document(source, "doc", ProductType.EXTRACT)
        (runner.output_dir / "doc.cost.json").write_text(content, encoding="utf-8")
        evaluator = EvaluationRunner.__new__(EvaluationRunner)
        with pytest.warns(UserWarning, match="Unable to read cost accounting"):
            result = evaluator._load_inference_result(runner.output_dir / "doc.result.json")
        assert result is not None
        stats = {stat.name: stat.value for stat in build_operational_stats(result)}
        assert stats["known_cost_usd"] == 2
        assert "cost_usd" not in stats
        (runner.output_dir / "_errors.json").write_text('[{"example_id": "doc", "error": "failed"}]', encoding="utf-8")
        with pytest.warns(UserWarning, match="Unable to read cost accounting"):
            errors = evaluator._load_inference_errors(runner.output_dir, pipeline_name=None)
        assert errors[0]["error"] == "failed"
    finally:
        runner.shutdown()


def test_parse_provider_keeps_existing_raw_costs_and_does_not_create_extract_accounting(tmp_path: Path) -> None:
    from extract_bench.schemas.parse_output import ParseOutput

    runner = _runner(tmp_path, [lambda: {"cost_usd": 2, "cost_per_page_usd": 0.1}])
    runner.pipeline = runner.pipeline.model_copy(update={"product_type": ProductType.PARSE})
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"pdf")
    try:
        raw = runner._run_provider_inference(
            InferenceRequest(example_id="doc", source_file_path=str(source), product_type=ProductType.PARSE)
        )
        assert raw.raw_output == {"cost_usd": 2, "cost_per_page_usd": 0.1}
        result = InferenceResult.model_validate(
            {
                **raw.model_dump(),
                "output": ParseOutput(example_id="doc", pipeline_name="cost_test", markdown="", pages=[]),
            }
        )
        stats = {stat.name: stat.value for stat in build_operational_stats(result)}
        assert stats == {"latency_ms": 1, "cost_usd": 2, "cost_per_page_usd": 0.1}
        assert not (runner.output_dir / "doc.cost.json").exists()
    finally:
        runner.shutdown()
