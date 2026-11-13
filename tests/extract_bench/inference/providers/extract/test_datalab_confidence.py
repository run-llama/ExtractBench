"""Fast extraction keeps its result when optional confidence polling fails."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("datalab_sdk")
pytest.importorskip("pypdf")

from aiohttp import web
from datalab_sdk import AsyncDatalabClient

from extract_bench.inference.limits import ExecutionBudget
from extract_bench.inference.providers.extract import datalab


@pytest.mark.parametrize(
    "outcome",
    ["scores", "timeout", "http_error", "invalid", "failed", "cancelled", "deadline", "signed", "invalid_url"],
)
def test_fast_wait_preserves_extraction_and_never_resubmits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    async def run() -> None:
        calls = {"submit": 0, "poll": 0}
        budget = ExecutionBudget(30)
        initial = {
            "success": True,
            "status": "complete",
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
            await request.read()
            calls["submit"] += 1
            return web.json_response({"success": True, "request_id": "job", "request_check_url": "/api/v1/extract/job"})

        async def poll(request: web.Request) -> web.Response:
            calls["poll"] += 1
            if calls["poll"] == 1 or outcome == "timeout":
                return web.json_response(initial)
            if outcome == "http_error":
                return web.json_response({"error": "unavailable"}, status=503)
            if outcome == "invalid":
                return web.json_response([])
            if outcome == "failed":
                return web.json_response(
                    {**initial, "extraction_schema_json": None, "extraction_scoring_status": "failed"}
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
            datalab, "AsyncDatalabClient", lambda **kwargs: AsyncDatalabClient(base_url=base_url, **kwargs)
        )
        source = tmp_path / "doc.pdf"
        source.write_bytes(b"%PDF-1.4 local test")
        provider = datalab.DatalabExtractProvider(
            "datalab_extract",
            {"api_key": "local-test", "extraction_mode": "fast", "score_max_polls": 1, "score_poll_interval": 0.001},
        )
        try:
            with budget.activate():
                result = await budget.run_async(provider._extract_pdf_async(str(source), {"type": "object"}))
            assert json.loads(result["extraction_schema_json"])["total"] == 10
            assert calls["submit"] == 1
            assert calls["poll"] == 2
            if outcome in ("scores", "signed"):
                assert result["_confidence_polling"]["status"] == "scores_ready"
                citations = datalab._build_field_citations(json.loads(result["extraction_schema_json"]), {})
                assert [(citation.field_path, citation.confidence, citation.bbox) for citation in citations] == [
                    ("total", 0.5, None)
                ]
            else:
                assert result["_confidence_polling"]["status"] in {
                    "timeout",
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
