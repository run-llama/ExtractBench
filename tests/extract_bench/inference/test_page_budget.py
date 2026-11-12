import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from extract_bench.inference.cli import InferenceCLI
from extract_bench.inference.page_budget import PageBudget
from extract_bench.inference.runner import InferenceRunner
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.product import ProductType

pymupdf = pytest.importorskip("pymupdf")


class RecordingRunner(InferenceRunner):
    observed = []

    def _process_test_case(self, test_case, product_type):
        self.observed.append(test_case.test_id)
        return None, None, None


def run_mock_cli(tmp_path, name, seed, cases, concurrent=1, **kwargs):
    spec = PipelineSpec(pipeline_name=name, provider_name="fake", product_type=ProductType.EXTRACT, config={})
    RecordingRunner.observed = []
    with (
        patch("extract_bench.inference.cli.get_pipeline", return_value=spec),
        patch("extract_bench.inference.cli.load_test_cases", return_value=cases),
        patch("extract_bench.inference.cli._detect_product_type", return_value=ProductType.EXTRACT),
        patch("extract_bench.inference.cli.create_provider", return_value=object()),
        patch("extract_bench.inference.cli.InferenceRunner", RecordingRunner),
    ):
        result = InferenceCLI().run(
            name,
            input_dir=tmp_path,
            output_dir=tmp_path,
            shuffle_seed=seed,
            max_concurrent=concurrent,
            no_rich=True,
            force=True,
            force_exit_on_completion=False,
            **kwargs,
        )
    assert result == 0
    return list(RecordingRunner.observed), json.loads((tmp_path / name / "_submission_order.json").read_text())


def pdf_cases(tmp_path, counts):
    cases = []
    for i, pages in enumerate(counts):
        path = tmp_path / f"doc-{i}.pdf"
        with pymupdf.open() as pdf:
            for _ in range(pages):
                pdf.new_page()
            pdf.save(path)
        cases.append(SimpleNamespace(test_id=f"doc-{i}", file_path=path))
    return cases


def events(path):
    return [json.loads(line) for line in (path / "_admission_events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    "counts,max_docs,max_pages",
    [
        ([60, 60, 10], 8, 100),
        ([123, 10, 20], 8, 100),
        ([60, 101, 10], 8, 100),
        ([10, 10, 10], 2, 100),
        ([50, 50, 1], 8, 100),
    ],
)
def test_fifo_limits_and_oversize(tmp_path, counts, max_docs, max_pages):
    cases = pdf_cases(tmp_path, counts)

    async def scenario():
        budget = PageBudget([(c.test_id, c.file_path) for c in cases], max_docs, max_pages, tmp_path)
        started = [asyncio.Event() for _ in cases]
        release = [asyncio.Event() for _ in cases]
        order = []

        async def worker(i):
            async with budget.slot(cases[i].test_id):
                order.append(i)
                started[i].set()
                await release[i].wait()

        tasks = [asyncio.create_task(worker(i)) for i in range(len(cases))]
        for i in range(len(cases)):
            await asyncio.wait_for(started[i].wait(), 1)
            assert order == sorted(order)
            release[i].set()
        await asyncio.gather(*tasks)
        assert budget.active_documents == budget.active_pages == 0

    asyncio.run(scenario())
    log = events(tmp_path)
    assert [e["example_id"] for e in log if e["event"] == "admitted"] == [c.test_id for c in cases]
    for event in log:
        assert event["active_documents"] <= max_docs
        assert event["active_pages"] <= max_pages or (event["active_documents"] == 1 and event["pages"] > max_pages)


def test_fifo_does_not_backfill_and_holds_full_lifetime(tmp_path):
    cases = pdf_cases(tmp_path, [60, 60, 10])

    async def scenario():
        budget = PageBudget([(c.test_id, c.file_path) for c in cases], 8, 100, tmp_path)
        releases = [asyncio.Event() for _ in cases]
        starts = [asyncio.Event() for _ in cases]

        async def worker(i):
            async with budget.slot(cases[i].test_id):
                starts[i].set()
                await releases[i].wait()

        tasks = [asyncio.create_task(worker(i)) for i in range(3)]
        await starts[0].wait()
        await asyncio.sleep(0)
        assert not starts[1].is_set() and not starts[2].is_set()
        assert (budget.active_documents, budget.active_pages) == (1, 60)
        releases[0].set()
        await starts[1].wait()
        await starts[2].wait()
        assert (budget.active_documents, budget.active_pages) == (2, 70)
        releases[1].set()
        releases[2].set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())


@pytest.mark.parametrize("rich", [False, True])
def test_real_runner_limits_and_manifest_order(tmp_path, rich):
    cases = pdf_cases(tmp_path, [60, 60, 10, 123, 5, 1, 50, 50, 1])
    spec = PipelineSpec(pipeline_name="mock", provider_name="fake", product_type=ProductType.EXTRACT, config={})

    class SlowRunner(RecordingRunner):
        def _process_test_case(self, case, product_type):
            time.sleep(0.005)
            return super()._process_test_case(case, product_type)

    runner = SlowRunner(
        provider=object(),
        pipeline=spec,
        output_dir=tmp_path,
        max_concurrent=3,
        max_concurrent_pages=100,
        use_rich=rich,
        force=True,
        timeout_retries=0,
    )
    try:
        summary = asyncio.run(runner.run_test_cases(cases, ProductType.EXTRACT))
        assert summary.successful == len(cases)
    finally:
        runner.shutdown()
    log = events(tmp_path)
    assert [e["example_id"] for e in log if e["event"] == "admitted"] == [c.test_id for c in cases]
    assert all(e["active_documents"] <= 3 for e in log)
    assert all(e["active_pages"] <= 100 or e["active_documents"] == 1 for e in log)
    assert log[-1]["active_pages"] == log[-1]["active_documents"] == 0


def test_processing_error_is_recorded_and_later_documents_complete(tmp_path):
    cases = pdf_cases(tmp_path, [1, 1, 1])
    spec = PipelineSpec(pipeline_name="mock", provider_name="fake", product_type=ProductType.EXTRACT, config={})

    class FailingRunner(RecordingRunner):
        def _process_test_case(self, case, product_type):
            if case.test_id == "doc-0":
                return None, None, ("Unknown remote outcome", "", "ProviderPermanentError")
            return None, None, None

    runner = FailingRunner(
        provider=object(),
        pipeline=spec,
        output_dir=tmp_path,
        max_concurrent=1,
        max_concurrent_pages=100,
        use_rich=False,
        force=True,
        timeout_retries=0,
    )
    try:
        summary = asyncio.run(runner.run_test_cases(cases, ProductType.EXTRACT))
    finally:
        runner.shutdown()
    assert summary.successful == 2 and summary.failed == 1
    assert summary.errors[0]["example_id"] == "doc-0"
    log = events(tmp_path)
    assert [e["example_id"] for e in log if e["event"] == "admitted"] == ["doc-0", "doc-1", "doc-2"]
    assert log[-1]["active_pages"] == 0


@pytest.mark.parametrize("pages", [0, -1, True])
def test_invalid_budget_rejected(tmp_path, pages):
    with pytest.raises(ValueError):
        PageBudget([], 8, pages, tmp_path)


def test_unreadable_pdf_fails_before_any_submission(tmp_path):
    cases = pdf_cases(tmp_path, [1])
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a PDF")
    with pytest.raises(pymupdf.FileDataError):
        PageBudget([(cases[0].test_id, cases[0].file_path), ("bad", bad)], 8, 100, tmp_path)
    assert not (tmp_path / "_admission_events.jsonl").exists()


def test_cli_forwards_budget_and_single_worker_uses_gate(tmp_path):
    cases = pdf_cases(tmp_path, [123, 20, 10])
    _, manifest = run_mock_cli(tmp_path, "bounded", 0, cases, concurrent=1, max_concurrent_pages=100)
    log = events(tmp_path / "bounded")
    assert [e["example_id"] for e in log if e["event"] == "admitted"] == manifest["example_ids"]
    assert all(e["active_documents"] <= 1 for e in log)


@pytest.mark.parametrize("max_documents", [1, 2])
@pytest.mark.parametrize("mode", ["test_cases", "files"])
def test_healthy_documents_execute_after_stuck_attempt(tmp_path, max_documents, mode):
    cases = pdf_cases(tmp_path, [1, 1, 1])
    spec = PipelineSpec(pipeline_name="mock", provider_name="fake", product_type=ProductType.EXTRACT, config={})
    release = threading.Event()
    stopped = threading.Event()
    started = []

    class TimeoutRunner(RecordingRunner):
        def process(self, example_id):
            started.append(example_id)
            if example_id == "doc-0" and started.count(example_id) == 1:
                try:
                    assert release.wait(5), "test did not release the stuck attempt"
                finally:
                    stopped.set()
            return None, None, None

        def _process_test_case(self, case, product_type):
            return self.process(case.test_id)

        def _process_document(self, path, example_id, product_type):
            return self.process(example_id)

        async def _cancel_inflight_and_drain_async(self, example_id, future, **kwargs):
            await super()._cancel_inflight_and_drain_async(example_id, future, drain_timeout_seconds=0.001)

    runner = TimeoutRunner(
        provider=object(),
        pipeline=spec,
        output_dir=tmp_path,
        max_concurrent=max_documents,
        max_concurrent_pages=100,
        use_rich=False,
        force=True,
        per_file_timeout=0.1,
        timeout_retries=0,
    )
    try:
        if mode == "test_cases":
            summary = asyncio.run(runner.run_test_cases(cases, ProductType.EXTRACT))
        else:
            summary = asyncio.run(runner.run_files([case.file_path for case in cases], ProductType.EXTRACT))
        assert not stopped.is_set()
        assert set(started) == {"doc-0", "doc-1", "doc-2"}
        assert started.count("doc-0") == 1
        assert summary.successful == 2
        assert summary.failed == 1
        assert len(summary.errors) == 1
        assert [e["example_id"] for e in events(tmp_path) if e["event"] == "admitted"] == ["doc-0", "doc-1", "doc-2"]
    finally:
        release.set()
        assert stopped.wait(1)
        runner.shutdown()


def test_cached_documents_do_not_take_capacity(tmp_path):
    cases = pdf_cases(tmp_path, [60, 123, 60])
    spec = PipelineSpec(pipeline_name="mock", provider_name="fake", product_type=ProductType.EXTRACT, config={})

    class CachedRunner(RecordingRunner):
        def _is_already_processed(self, example_id):
            return example_id == "doc-1"

    runner = CachedRunner(
        provider=object(),
        pipeline=spec,
        output_dir=tmp_path,
        max_concurrent=8,
        max_concurrent_pages=100,
        use_rich=False,
        timeout_retries=0,
    )
    try:
        summary = asyncio.run(runner.run_test_cases(cases, ProductType.EXTRACT))
    finally:
        runner.shutdown()
    assert summary.successful == 2 and summary.skipped == 1
    assert [e["example_id"] for e in events(tmp_path) if e["event"] == "admitted"] == ["doc-0", "doc-2"]


def test_cancelled_waiter_does_not_deadlock_fifo(tmp_path):
    cases = pdf_cases(tmp_path, [60, 60, 10])

    async def scenario():
        budget = PageBudget([(c.test_id, c.file_path) for c in cases], 8, 100, tmp_path)
        async with budget.slot("doc-0"):

            async def blocked():
                async with budget.slot("doc-1"):
                    raise AssertionError("must wait")

            task = asyncio.create_task(blocked())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            async with budget.slot("doc-2"):
                assert budget.active_pages == 70
        assert budget.active_pages == 0

    asyncio.run(scenario())


def test_page_budget_forwarded_by_benchmark_commands(tmp_path):
    from extract_bench.cli import BenchCLI
    from extract_bench.pipeline.cli import PipelineCLI

    with patch.object(PipelineCLI, "run", return_value=0) as run:
        assert BenchCLI().run("fake", input_dir=tmp_path, max_concurrent_pages=100) == 0
    assert run.call_args.kwargs["max_concurrent_pages"] == 100
    with patch.object(InferenceCLI, "run", return_value=1) as run:
        assert PipelineCLI().run("fake", input_dir=tmp_path, max_concurrent_pages=100) == 1
    assert run.call_args.kwargs["max_concurrent_pages"] == 100


def test_file_runner_obeys_page_budget(tmp_path):
    cases = pdf_cases(tmp_path, [60, 60, 123, 10])
    spec = PipelineSpec(pipeline_name="mock", provider_name="fake", product_type=ProductType.EXTRACT, config={})

    class FileRunner(InferenceRunner):
        def _process_document(self, path, example_id, product_type):
            time.sleep(0.005)
            return None, None, None

    runner = FileRunner(
        provider=object(),
        pipeline=spec,
        output_dir=tmp_path,
        max_concurrent=3,
        max_concurrent_pages=100,
        use_rich=False,
        force=True,
        timeout_retries=0,
    )
    try:
        summary = asyncio.run(runner.run_files([case.file_path for case in cases], ProductType.EXTRACT))
    finally:
        runner.shutdown()
    assert summary.successful == 4
    log = events(tmp_path)
    assert [e["example_id"] for e in log if e["event"] == "admitted"] == [case.test_id for case in cases]
    assert all(e["active_pages"] <= 100 or e["active_documents"] == 1 for e in log)


@pytest.mark.parametrize("failed_event", ["admitted", "released"])
def test_log_failure_releases_capacity_and_wakes_waiter(tmp_path, failed_event):
    cases = pdf_cases(tmp_path, [101, 10])

    async def scenario():
        budget = PageBudget([(c.test_id, c.file_path) for c in cases], 1, 100, tmp_path)
        original_record = budget._record
        first_started = asyncio.Event()
        release = asyncio.Event()

        def record(event, example_id=None):
            if event == failed_event and example_id == "doc-0":
                raise OSError("disk full")
            original_record(event, example_id)

        async def first():
            async with budget.slot("doc-0"):
                first_started.set()
                await release.wait()

        async def second():
            async with budget.slot("doc-1"):
                assert (budget.active_documents, budget.active_pages) == (1, 10)

        with patch.object(budget, "_record", side_effect=record):
            if failed_event == "admitted":
                with pytest.raises(OSError, match="disk full"):
                    await first()
                await asyncio.wait_for(second(), 1)
            else:
                task = asyncio.create_task(first())
                await asyncio.wait_for(first_started.wait(), 1)
                waiter = asyncio.create_task(second())
                await asyncio.sleep(0)
                assert not waiter.done()
                release.set()
                with pytest.raises(OSError, match="disk full"):
                    await task
                await asyncio.wait_for(waiter, 1)
        assert budget.active_documents == budget.active_pages == 0
        assert len(budget.waiting) == 0

    asyncio.run(scenario())


def test_resume_archives_previous_admission_sessions(tmp_path):
    cases = pdf_cases(tmp_path, [1, 1])
    spec = PipelineSpec(pipeline_name="mock", provider_name="fake", product_type=ProductType.EXTRACT, config={})

    class ResumingRunner(RecordingRunner):
        cached = set()

        def _is_already_processed(self, example_id):
            return example_id in self.cached

    runner = ResumingRunner(
        provider=object(),
        pipeline=spec,
        output_dir=tmp_path,
        max_concurrent=1,
        max_concurrent_pages=100,
        use_rich=False,
    )
    try:
        asyncio.run(runner.run_test_cases(cases, ProductType.EXTRACT))
        first_events = (tmp_path / "_admission_events.jsonl").read_bytes()
        first_plan = (tmp_path / "_admission_plan.json").read_bytes()
        runner.cached = {"doc-0"}
        resumed = asyncio.run(runner.run_test_cases(cases, ProductType.EXTRACT))
        assert resumed.successful == resumed.skipped == 1
        assert [e["example_id"] for e in events(tmp_path) if e["event"] == "admitted"] == ["doc-1"]
        archives = list((tmp_path / "_admission_history").iterdir())
        assert len(archives) == 1
        assert (archives[0] / "_admission_events.jsonl").read_bytes() == first_events
        assert (archives[0] / "_admission_plan.json").read_bytes() == first_plan
        # An interrupted append is evidence too; preserve it byte for byte.
        with (tmp_path / "_admission_events.jsonl").open("ab") as log:
            log.write(b'{"event":')
        interrupted = (tmp_path / "_admission_events.jsonl").read_bytes()
        runner.cached = {"doc-0", "doc-1"}
        complete = asyncio.run(runner.run_test_cases(cases, ProductType.EXTRACT))
        assert complete.skipped == 2
        assert events(tmp_path) == []
        archives = list((tmp_path / "_admission_history").iterdir())
        assert len(archives) == 2
        assert {(d / "_admission_events.jsonl").read_bytes() for d in archives} == {first_events, interrupted}
    finally:
        runner.shutdown()
