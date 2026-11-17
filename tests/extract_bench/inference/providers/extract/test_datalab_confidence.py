"""Fast extraction keeps its result when optional confidence polling fails."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datalab_sdk")
pytest.importorskip("pypdf")

from aiohttp import web
from datalab_sdk import AsyncDatalabClient

from extract_bench.inference.cost_contract import (
    COMPLETE,
    COMPONENT_ID,
    COST_ACCOUNTING,
    COST_COMPONENT_EVENT,
    COST_PER_PAGE_USD,
    COST_USD,
    KNOWN_COST_USD,
)
from extract_bench.inference.limits import ExecutionBudget
from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.extract import datalab


@pytest.mark.parametrize(
    "outcome",
    [
        "scores",
        "ready",
        "turbo",
        "balanced",
        "accurate",
        "override",
        "no_cost",
        "delayed",
        "pending_deadline",
        "http_error",
        "invalid",
        "failed",
        "cancelled",
        "deadline",
        "signed",
        "invalid_url",
    ],
)
def test_extraction_survives_optional_scoring_with_separate_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    async def run() -> None:
        calls = {"submit": 0, "poll": 0}
        clock = {"now": 0.0}
        monkeypatch.setattr(datalab, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))
        budget = ExecutionBudget(30)
        initial = {
            "success": True,
            "status": "complete",
            "page_count": 1,
            "cost_breakdown": None if outcome in ("turbo", "no_cost") else {"final_cost_cents": 1.7},
            "extraction_schema_json": '{"total":10}',
            "extraction_score_average": None,
            "extraction_scoring_status": "partial",
        }
        scored = {
            **initial,
            "extraction_schema_json": '{"total":10,"total_score":{"score":3}}',
            "extraction_score_average": 3,
        }

        async def submit(request: web.Request) -> web.Response:
            form = await request.post()
            assert form["extraction_mode"] == pipeline.config["extraction_mode"]
            if outcome == "override":
                assert form["mode"] == "fast"
            else:
                assert "mode" not in form
            calls["submit"] += 1
            return web.json_response({"success": True, "request_id": "job", "request_check_url": "/api/v1/extract/job"})

        async def poll(request: web.Request) -> web.Response:
            calls["poll"] += 1
            clock["now"] += 2.0 if calls["poll"] == 1 else 5.0
            if outcome == "ready":
                return web.json_response(scored)
            if calls["poll"] == 1 or (outcome == "delayed" and calls["poll"] < 66):
                return web.json_response(initial)
            if outcome == "pending_deadline":
                if calls["poll"] == 4:
                    budget.deadline = 0
                return web.json_response(initial)
            if outcome == "http_error":
                return web.json_response({"error": "unavailable"}, status=503)
            if outcome == "invalid":
                return web.json_response([])
            if outcome == "failed":
                return web.json_response(
                    {**initial, "extraction_schema_json": None, "extraction_scoring_status": "failed"},
                )
            if outcome == "cancelled":
                budget.cancel()
            if outcome == "deadline":
                budget.deadline = 0
            if outcome == "invalid_url":
                return web.json_response({"status": "complete", "result_url": {}})
            if outcome == "signed":
                return web.json_response({"status": "complete", "result_url": f"{base_url}/signed"})
            return web.json_response(scored)

        async def download(request: web.Request) -> web.Response:
            assert "X-API-Key" not in request.headers
            return web.json_response(scored)

        app = web.Application()
        app.router.add_post("/api/v1/extract", submit)
        app.router.add_get("/api/v1/extract/job", poll)
        app.router.add_get("/signed", download)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        monkeypatch.setattr(
            datalab,
            "AsyncDatalabClient",
            lambda **kwargs: AsyncDatalabClient(base_url=base_url, **kwargs),
        )
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 local test")
        extraction_mode = {
            "turbo": "turbo",
            "balanced": "balanced",
            "accurate": "accurate",
            "override": "balanced",
            "no_cost": "balanced",
        }.get(outcome, "fast")
        pipeline = get_pipeline(f"datalab_extract_{extraction_mode}")
        config = {**pipeline.config, "api_key": "local-test", "poll_interval": 0.001}
        if outcome == "override":
            config["parse_mode"] = "fast"
        provider = datalab.DatalabExtractProvider(
            "datalab_extract",
            config,
        )
        try:
            if outcome in ("invalid", "invalid_url"):
                with budget.activate(), pytest.raises(datalab.ProviderPermanentError) as caught:
                    await budget.run_async(provider._extract_pdf_async(str(source), {"type": "object"}))
                assert caught.value.debug_payload[COST_ACCOUNTING] == {KNOWN_COST_USD: 0.017, COMPLETE: True}
                return
            with budget.activate():
                result = await budget.run_async(provider._extract_pdf_async(str(source), {"type": "object"}))
            assert json.loads(result["extraction_schema_json"])["total"] == 10
            assert calls["submit"] == 1
            assert result["extraction_latency_ms"] == 2000
            expected_polls = (
                1 if extraction_mode != "fast" else {"ready": 1, "delayed": 66, "pending_deadline": 4}.get(outcome, 2)
            )
            assert calls["poll"] == expected_polls
            if outcome == "no_cost":
                assert COST_USD not in result
            elif outcome == "turbo":
                assert result["parse_cost_usd"] == 0
                assert result[COST_USD] == pytest.approx(0.006)
            else:
                assert result[COST_USD] == pytest.approx(0.017)
                assert result[COST_PER_PAGE_USD] == pytest.approx(0.017)
            if outcome not in ("turbo", "override"):
                assert "datalab_pricing" not in result
            if extraction_mode != "fast":
                assert "confidence_wait_ms" not in result
                assert datalab._build_field_citations(json.loads(result["extraction_schema_json"]), {}) == []
                return
            assert result["confidence_wait_ms"] == (expected_polls - 1) * 5000
            if outcome in ("scores", "signed", "ready", "delayed"):
                assert result["_confidence_polling"]["status"] == "scores_ready"
                citations = datalab._build_field_citations(json.loads(result["extraction_schema_json"]), {})
                assert [(citation.field_path, citation.confidence, citation.bbox) for citation in citations] == [
                    ("total", 0.5, None)
                ]
            else:
                assert result["_confidence_polling"]["status"] in {
                    "poll_error",
                    "scoring_failed",
                    "deadline_exceeded",
                }
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("score", [True, "5", float("nan"), float("inf")])
def test_invalid_scores_do_not_become_confidence(score: object) -> None:
    assert datalab._build_field_citations({"total": 10, "total_score": {"score": score}}, {}) == []


def test_score_survives_unknown_citation_and_parent_scores_are_ignored() -> None:
    citations = datalab._build_field_citations(
        {
            "items": [{"total": None, "total_score": {"score": 4}, "total_citations": ["unknown"]}],
            "items_score": {"score": 5},
        },
        {},
    )
    assert [(citation.field_path, citation.confidence, citation.page) for citation in citations] == [
        ("items.0.total", 0.75, None)
    ]


def test_missing_request_ids_do_not_merge_separate_extraction_charges(monkeypatch: pytest.MonkeyPatch) -> None:
    from datalab_sdk.models import ConversionResult

    from extract_bench.inference.events import bind_provider_event_sink

    class Client:
        base_url = "https://unused.invalid"

        async def __aenter__(self) -> "Client":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def extract(self, *args: object, **kwargs: object) -> ConversionResult:
            return ConversionResult(success=True, output_format="json", cost_breakdown={"final_cost_cents": 2})

    monkeypatch.setattr(datalab, "AsyncDatalabClient", lambda **kwargs: Client())
    monkeypatch.setattr(datalab, "_read_pdf_page_count", lambda path: 1)
    provider = datalab.DatalabExtractProvider("datalab_extract", {"api_key": "unused", "extraction_mode": "balanced"})
    charges = {}

    def record(event: str, payload: dict) -> None:
        if event == COST_COMPONENT_EVENT:
            charges[payload[COMPONENT_ID]] = payload[COST_USD]

    with bind_provider_event_sink(record):
        asyncio.run(provider._extract_pdf_async("unused.pdf", {}))
        asyncio.run(provider._extract_pdf_async("unused.pdf", {}))
    assert sum(charges.values()) == pytest.approx(0.04)
